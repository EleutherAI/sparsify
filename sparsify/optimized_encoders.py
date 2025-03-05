import math
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn


@dataclass
class PKMConfig:
    pad: bool = False
    """Pad the PKM encoder to a power of 2."""

    softmax: bool = False
    """Apply softmax to PKM outputs."""

    heads: int = 1
    """Number of heads for PKM."""

    bias: bool = True
    """Non-decomposed bias for PKM."""

    init_scale: float = 1.0
    """Scale factor for PKM encoder initialization."""


class PKMLinear(nn.Module):
    """Product Key Memory Linear layer.

    Deviations from the original paper:
    * Learned per-latent bias
    * PKM grouping: learn multiple encoders with the same
        factorization and slightly lower k, combine their outputs
    """

    def __init__(
        self,
        d_in: int,
        num_latents: int,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
        *,
        cfg: PKMConfig,
    ):
        super().__init__()
        self.d_in = d_in
        self.num_latents = num_latents
        if cfg.pad:
            self.pkm_base = int(2 ** math.ceil(math.log2(num_latents) / 2))
        else:
            self.pkm_base = int(math.ceil(math.sqrt(num_latents)))
        self.cfg = cfg
        self.num_heads = cfg.heads
        self._weight = nn.Linear(
            d_in, self.num_heads * 2 * self.pkm_base, device=device, dtype=dtype
        )
        self._weight.weight.data *= cfg.init_scale / 4
        # Orthogonal matrices have the same FVU  as /4,
        # but produce more dead latents
        # torch.nn.init.orthogonal_(
        #     self._weight.weight, gain=0.5 / math.sqrt(self.d_in))
        self._scale = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
        if cfg.bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    self.num_heads * self.pkm_base**2, dtype=dtype, device=device
                )
            )
        if cfg.softmax:
            self.scaling = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))

    @torch.compile(mode="max-autotune")
    def forward(self, x):
        xs = self._weight(x)
        x1, x2 = xs[..., : self.pkm_base], xs[..., self.pkm_base :]
        y = (x1[..., :, None] + x2[..., None, :]).flatten(-2)
        if self.cfg.bias:
            y += self.bias
        y = y[..., : self.num_latents]
        return y

    @torch.compile(mode="max-autotune")
    def topk(self, x, k: int):
        orig_batch_size = x.shape[:-1]
        x1, x2 = torch.chunk(
            self._weight(x).unflatten(-1, (self.num_heads, self.pkm_base * 2)),
            2,
            dim=-1,
        )
        k_head = max(1, k // self.num_heads)
        k1, k2 = k_head, k_head
        w1, i1 = x1.topk(k1, dim=-1)
        w2, i2 = x2.topk(k2, dim=-1)
        w = torch.nn.functional.relu(w1[..., :, None] + w2[..., None, :]).clone()
        i = i1[..., :, None] * self.pkm_base + i2[..., None, :]
        mask = i >= self.num_latents
        if self.cfg.bias:
            bias_i = (
                i
                + torch.arange(self.num_heads, device=i.device, dtype=i.dtype)[
                    :, None, None
                ]
                * self.pkm_base**2
            )
            w = w + self.bias[bias_i] * mask
        w[mask] = -1
        w = w.view(-1, self.num_heads, k1 * k2)
        w, i = w.topk(k_head, dim=-1, sorted=True)
        i1 = torch.gather(i1, -1, i // k2)
        i2 = torch.gather(i2, -1, i % k2)
        i = i1 * self.pkm_base + i2
        w = w * (i < self.num_latents)
        i = i.clamp_max(self.num_latents - 1)
        if self.cfg.softmax:
            # w = torch.nn.functional.softmax(w, dim=-1)
            w = torch.nn.functional.sigmoid(w)
            w = w * torch.nn.functional.softplus(self.scaling)  # [i])
        else:
            w, i = w[..., :k_head], i[..., :k_head]
            w, i = w.flatten(-2), i.flatten(-2)
            w_, i_ = w.topk(k, dim=-1)
            w = torch.gather(w, -1, i_)
            i = torch.gather(i, -1, i_)
            w, i = w.contiguous(), i.contiguous()
        return w.view(*orig_batch_size, k), i.reshape(*orig_batch_size, k)

    @property
    def weight(self):
        w = self._weight.weight
        w = w.reshape(self.num_heads, self.pkm_base * 2, self.d_in).transpose(0, 1)
        w1, w2 = torch.chunk(w, 2, dim=0)
        pkm_trim = math.ceil(self.num_latents / self.pkm_base)
        w1 = w1[:pkm_trim]
        w1 = w1[:, None, ...]
        w2 = w2[None, :, ...]
        w1 = w1.expand(-1, w2.shape[1], -1, -1)
        w2 = w2.expand(w1.shape[0], -1, -1, -1)
        return (
            (w1 + w2)
            .reshape(self.pkm_base * pkm_trim, self.num_heads, self.d_in)[
                : self.num_latents
            ]
            .sum(1)
        )


@dataclass
class KroneckerConfig:
    in_group: int = 2
    """Kronecker factorization input group size."""
    out_group: int = 4
    """Kronecker factorization output group size."""
    u: int = 4
    """Number of matrices to mix for the Kronecker product."""
    lora_dim: float = 1.0
    """How much to reduce the dimensionality of the input to the Kronecker product."""


class KroneckerLinear(nn.Module):
    def __init__(
        self,
        d_in: int,
        num_latents: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        cfg: KroneckerConfig,
    ):
        assert d_in % cfg.in_group == 0
        assert num_latents % cfg.out_group == 0
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.lora_dim = int(d_in * cfg.lora_dim)
        self.pre = nn.Linear(d_in, self.lora_dim, device=device, dtype=dtype)
        self.inner = nn.Parameter(
            torch.randn(cfg.out_group, cfg.u, cfg.in_group, dtype=dtype, device=device)
        )
        self.outer = nn.Parameter(
            torch.randn(
                num_latents // cfg.out_group,
                cfg.u,
                self.lora_dim // cfg.in_group,
                dtype=dtype,
                device=device,
            )
        )
        self.bias = nn.Parameter(torch.zeros(num_latents, dtype=dtype, device=device))
        self.num_latents = num_latents

    @torch.compile
    def forward(self, x):
        x = self.pre(x)
        x = x.unflatten(-1, (x.shape[-1] // self.cfg.in_group, self.cfg.in_group))
        x = torch.einsum("...nd,cud->...nuc", x, self.inner)
        x = torch.einsum("...nuc,mun->...cm", x, self.outer)
        return x.reshape(*x.shape[:-2], -1) + self.bias

    @torch.compile(mode="max-autotune")
    def topk(self, x, k: int):
        return self.forward(x).topk(k, dim=-1)

    @property
    def weight(self):
        mat = torch.einsum("yux,mun->ymxn", self.inner, self.outer)
        mat = mat.reshape(self.num_latents, self.lora_dim)
        return nn.Parameter(mat @ self.pre.weight, requires_grad=False)


class OptimizedEncoderConfig(Enum):
    PKM = "pkm"
    Kronecker = "kron"
    None_ = None

    def build_encoder(
        self,
        d_in: int,
        num_latents: int,
        device: str | torch.device,
        dtype: torch.dtype | None,
        pkm_config=None,
        kron_config=None,
    ) -> nn.Module | None:
        if self is OptimizedEncoderConfig.PKM:
            return PKMLinear(d_in, num_latents, device, dtype, cfg=pkm_config)
        elif self is OptimizedEncoderConfig.Kronecker:
            return KroneckerLinear(d_in, num_latents, device, dtype, cfg=kron_config)
        elif self is OptimizedEncoderConfig.None_:
            return None
        else:
            raise ValueError(f"Unknown encoder type {self}")
