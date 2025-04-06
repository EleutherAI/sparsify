import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Literal

import torch
import torch.nn as nn
from simple_parsing import Serializable

from .fused_encoder import binary_fused_encoder


@dataclass
class PKMConfig(Serializable):
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


@dataclass
class KroneckerConfig(Serializable):
    in_group: int = 2
    """Kronecker factorization input group size."""
    out_group: int = 4
    """Kronecker factorization output group size."""
    u: int = 4
    """Number of matrices to mix for the Kronecker product."""
    lora_dim: float = 1.0
    """How much to reduce the dimensionality of the input to the Kronecker product."""


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
        # xs = self._weight(x)
        # x1, x2 = xs[..., : self.pkm_base], xs[..., self.pkm_base :]
        # y = (x1[..., :, None] + x2[..., None, :]).flatten(-2)
        # if self.cfg.bias:
        #     y += self.bias
        # y = y[..., : self.num_latents]
        # return y
        raise NotImplementedError

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


@dataclass
class FFFConfig(Serializable):
    fff_k: Optional[int] = None
    gradients: Literal["ste", "reinmax", "gumbel", "almost_gumbel"] = "ste"
    n_samples: int = 1
    binarize: bool = False
    full_fff: bool = False
    cheating: bool = False
    temperature: float = 0.3
    sigmoid_approx: bool = False
    expand_sigmoid: bool = False
    ste: bool = False


# Thresholded
# STE

class DenseBinaryEncode(torch.autograd.Function):
    @staticmethod
    @torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
    )
    def forward(ctx, x, W_enc, b_enc, log_threshold):
        preacts = x @ W_enc.T + b_enc

        threshold = torch.exp(log_threshold)

        ctx.save_for_backward(x, W_enc, b_enc, threshold)

        return (preacts > threshold).float()

    @staticmethod
    @torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
    )
    def backward(ctx, grad_output):
        x, W_enc, b_enc, threshold = ctx.saved_tensors

        # Use sigmoid STE to approximate the binary activations
        preacts = x @ W_enc.T + b_enc
        ste_acts = torch.sigmoid(preacts - threshold)

        # Gradient of a sigmoid wrt its input = sigmoid(x) * (1 - sigmoid(x))
        grad_sigmoid_wrt_thresholded_preacts = ste_acts * (1 - ste_acts)

        # dL/d(thresholded_preacts) = dL/d(ste_acts) * d(ste_acts)/d(thresholded_preacts)
        grad_thresholded_preacts = grad_output * grad_sigmoid_wrt_thresholded_preacts       

        # The gradient of the loss wrt the preacts is the same as the gradient of the loss 
        # wrt the thresholded preacts because the difference is a constant (threshold)
        grad_preacts = grad_thresholded_preacts

        # Gradient of the loss wrt the input
        grad_input = grad_preacts @ W_enc

        # Gradient of the loss wrt the encoder weights
        grad_W_enc = grad_preacts.T @ x

        # Gradient of the loss wrt the encoder bias
        grad_b_enc = grad_preacts.sum(0)

        # Gradient of the loss wrt the threshold
        grad_log_threshold = -grad_thresholded_preacts.sum(0)

        # print("encode backwards")
        # print("input grad", grad_input)
        # print("W_enc grad", grad_W_enc)
        # print("b_enc grad", grad_b_enc)
        # print("thresholds", threshold)
        # print("log threshold grad", grad_log_threshold)

        return grad_input, grad_W_enc, grad_b_enc, grad_log_threshold

def dense_binary_encode(x, W_enc, b_enc, threshold):
    """Convenience function for dense binary encoding."""
    return DenseBinaryEncode.apply(x, W_enc, b_enc, threshold) # type: ignore


class FFFLinear(nn.Module):
    def __init__(
        self,
        d_in: int,
        num_latents: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        cfg: FFFConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = num_latents
        self._weight = nn.Linear(d_in, num_latents, device=device, dtype=dtype)
        if cfg.full_fff:
            if cfg.cheating:
                self._decisions = nn.Linear(d_in, num_latents, device=device, dtype=dtype)
            else:
                self.n_levels = int(math.log2(num_latents // cfg.fff_k)) - 1
                self.n_decisions = 2 ** self.n_levels - 1
                self._decisions = nn.Linear(d_in, cfg.fff_k * self.n_decisions, device=device, dtype=dtype)
                assert num_latents // cfg.fff_k == 2 ** (self.n_levels + 1), \
                    f"Number of latents per k must be a power of 2 ({2 ** (self.n_levels+1)}), got {num_latents}"
        else:
            self.n_decisions = int(math.log2(num_latents // cfg.fff_k))
            self._decisions = nn.Linear(d_in, self.n_decisions * cfg.fff_k, device=device, dtype=dtype)
        if cfg.ste:
            self.threshold = nn.Parameter((torch.full((self.num_latents,), 0.004, device=device, dtype=dtype)).log())

    def forward(self, x):
        raise NotImplementedError

    # @torch.compile(mode="reduce-overhead")
    def topk(self, x, k: int):
        if self.cfg.ste:
            return binary_fused_encoder(
                x, self._weight.weight, self._weight.bias, k, self.threshold.exp()
            )
            values = dense_binary_encode(
                x, self._weight.weight, self._weight.bias, self.threshold
            )
            return values, torch.arange(0, self.num_latents, device=x.device, dtype=torch.int32)[None, :].expand(x.shape[:-1] + (self.num_latents,))
        
        if not self.cfg.binarize:
            pre_acts = self._weight(x)
        # pre_acts = torch.nn.functional.relu(pre_acts)
        
        decisions = self._decisions(x)
        if self.cfg.sigmoid_approx:
            assert self.cfg.cheating and self.cfg.full_fff and self.cfg.binarize
            if self.cfg.expand_sigmoid:
                values, indices = decisions.topk(k * 2, dim=-1, sorted=True)
                threshold = values[..., k:k+1]
                sigmoided = torch.nn.functional.sigmoid(values)
                values = (values > threshold) + (sigmoided - sigmoided.detach())
                return values, indices
            else:
                values, indices = decisions.topk(k, dim=-1)
                sigmoided = torch.nn.functional.sigmoid(values)
                values = torch.ones_like(sigmoided) + (sigmoided - sigmoided.detach())
                return values, indices
        if self.cfg.full_fff:
            assert self.cfg.n_samples == 1, "Full FFF does not support multiple samples."
            if not self.cfg.cheating:
                decisions = decisions.unflatten(-1, (self.cfg.fff_k, self.n_decisions))
                decisions = torch.stack([log_sigmoid(decisions), log_sigmoid(-decisions)], dim=-1)
                probs = torch.zeros(x.shape[:-1] + (self.cfg.fff_k, self.num_latents // self.cfg.fff_k), device=x.device, dtype=x.dtype)
                for level_width in range(self.n_levels):
                    probs = probs.unflatten(-1, (-1, 2, 2 ** level_width))
                    probs = probs + decisions[..., level_width, None, :, None]
                    probs = probs.flatten(-3, -1)
            else:
                probs = decisions.unflatten(-1, (self.cfg.fff_k, self.num_latents // self.cfg.fff_k))
            if self.cfg.gradients in ("reinmax", "gumbel"):
                if self.cfg.gradients == "reinmax":
                    hard_decisions, soft_decisions = reinmax(probs, 1.0 + self.cfg.temperature)
                elif self.cfg.gradients == "gumbel":
                    hard_decisions = torch.nn.functional.gumbel_softmax(probs, tau=self.cfg.temperature, hard=True)
                else:
                    raise ValueError(f"Unknown gradient estimator {self.cfg.gradients}")
                hard_decisions = hard_decisions.flatten(-2, -1)
                if not self.cfg.binarize:
                    values = pre_acts * hard_decisions
                else:
                    values = hard_decisions
                return values, \
                    torch.arange(0, self.num_latents, device=x.device, dtype=torch.int32)[None, :].expand(x.shape[:-1] + (self.num_latents,))
            elif self.cfg.gradients == "almost_gumbel":
                hard_decisions = torch.nn.functional.gumbel_softmax(probs, tau=self.cfg.temperature, hard=True)
                probify = lambda probs: probs.flatten(-2, -1).topk(k * 2, dim=-1)[1]
                indices = torch.cat([probify(probs), probify(-probs)], dim=-1)
                # indices = torch.randn_like(probs).flatten(-2, -1).topk(k * 4, dim=-1)[1]
                values = torch.gather(pre_acts, -1, indices) * torch.gather(hard_decisions.flatten(-2, -1), -1, indices)
                return values, indices
                # decvision_values, decision_indices = hard_decisions.max(-1)
                # decision_indices = decision_indices + \
                #     torch.arange(0, self.cfg.fff_k, device=decision_indices.device, dtype=decision_indices.dtype) * probs.shape[-1]
                # return torch.gather(pre_acts, -1, decision_indices) * decvision_values, decision_indices
            elif self.cfg.gradients == "ste":
                probs = torch.nn.functional.softmax(probs, dim=-1)
                values, indices = groupmax_random(probs, k)
                return torch.gather(pre_acts, -1, indices) * values, indices
            else:
                raise ValueError(f"Unknown gradient estimator {self.cfg.gradients}")
        else:
            decisions = decisions.unflatten(-1, (self.cfg.fff_k, self.n_decisions))
            decisions = expand_probabilities(torch.nn.LogSigmoid()(decisions))
            if self.cfg.reinmax:
                if self.cfg.n_samples > 1:
                    raise ValueError("Reinmax does not support multiple samples.")
                aux_loss = 0.0  # -(decisions.exp() * decisions).sum(-1).mean()
                hard_decisions, soft_decisions = reinmax(decisions, 1.0)
                indices_per_group = hard_decisions.argmax(-1)
                decision_values = torch.gather(hard_decisions * soft_decisions, -1, indices_per_group.unsqueeze(-1))[..., 0]
                indices = (
                    indices_per_group
                    + torch.arange(0, k,
                                device=indices_per_group.device,
                                dtype=indices_per_group.dtype)
                    * decisions.shape[-2]
                ).flatten(0, -2)
                return torch.gather(pre_acts, -1, indices) * decision_values, indices, aux_loss
            else:
                all_values, all_indices = [], []
                for _ in range(self.cfg.n_samples):
                    decision_values, indices = groupmax_random(decisions.flatten(-2), k)
                    values = torch.gather(pre_acts, -1, indices) * decision_values.exp()
                    all_values.append(values)
                    all_indices.append(indices)
                aux_loss = 0.0
                values = torch.cat(all_values, dim=-1)
                indices = torch.cat(all_indices, dim=-1)
                return values, indices, aux_loss

    @property
    def weight(self):
        return self._weight.weight


def log_sigmoid(x):
    return torch.nn.LogSigmoid()(x)


def expand_probabilities(probs):
    """
    Expands a tensor of log-probabilities of shape (*batch_dims, log2(N)) into a tensor
    of shape (*batch_dims, N) by considering all paths through a binary tree.

    Args:
        probs: Tensor of shape (*batch_dims, log2(N)) where log2(N) is the depth of the binary tree.

    Returns:
        Tensor of shape (*batch_dims, N) containing the log probabilities of all leaf nodes.
    """
    # Get the shape of the input tensor
    *batch_dims, depth = probs.shape
    N = 2 ** depth  # Number of leaf nodes

    # Create a binary mask for all possible paths
    binary_mask = torch.arange(N, device=probs.device).reshape(*([1] * len(batch_dims)), N)
    binary_mask = (binary_mask >> torch.arange(depth - 1, -1, -1, device=probs.device).unsqueeze(-1)) & 1

    # Convert binary mask to probabilities (0 -> 1 - p, 1 -> p)
    path_probs = torch.where(binary_mask == 1, probs.unsqueeze(-1), -probs.unsqueeze(-1))

    # Compute the product of probabilities along each path
    result = torch.sum(path_probs, dim=-2)

    return result

def groupmax(x, k):
    x = x.unflatten(-1, (k, -1))
    v, i = x.max(dim=-1)
    i = i + torch.arange(0, k, device=i.device, dtype=i.dtype) * x.shape[-1]
    return v, i

def groupmax_random(x, k):
    x = x.unflatten(-1, (k, -1))
    x_shape = x.shape
    i = torch.multinomial(x.flatten(0, -2).softmax(dim=-1), 1).squeeze(-1)
    i = i.view(*x_shape[:-1])
    v = torch.gather(x, -1, i.unsqueeze(-1)).squeeze(-1)
    i = i + torch.arange(0, k, device=i.device, dtype=i.dtype) * x.shape[-1]
    return v, i


class ReinMaxCore(torch.autograd.Function):
    """
    `torch.autograd.Function` implementation of the ReinMax gradient estimator.
    """
    
    @staticmethod
    # @torch.compile(mode="max-autotune")
    def forward(
        ctx, 
        logits: torch.Tensor, 
        tau: torch.Tensor,
    ):
        y_soft = logits.softmax(dim=-1)
        sample = torch.multinomial(
            y_soft,
            num_samples=1,
            replacement=True,
        )
        one_hot_sample = torch.zeros_like(
            y_soft, 
            memory_format=torch.legacy_contiguous_format
        ).scatter_(-1, sample, 1.0)
        ctx.save_for_backward(one_hot_sample, logits, y_soft, tau)
        return one_hot_sample, y_soft

    @staticmethod
    @torch.compile(mode="max-autotune")
    def backward(
        ctx, 
        grad_at_sample: torch.Tensor, 
        grad_at_p: torch.Tensor,
    ):
        one_hot_sample, logits, y_soft, tau = ctx.saved_tensors
        
        shifted_y_soft = .5 * ((logits / tau).softmax(dim=-1) + one_hot_sample)
        grad_at_input_1 = (2 * grad_at_sample) * shifted_y_soft
        grad_at_input_1 = grad_at_input_1 - shifted_y_soft * grad_at_input_1.sum(dim=-1, keepdim=True)
        
        grad_at_input_0 = (-0.5 * grad_at_sample + grad_at_p) * y_soft
        grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)
        
        grad_at_input = grad_at_input_0 + grad_at_input_1
        return grad_at_input - grad_at_input.mean(dim=-1, keepdim=True), None

def reinmax(
        logits: torch.Tensor, 
        tau: float, 
    ):
    r"""
    Parameters
    ---------- 
    
    logits: ``torch.Tensor``, required
        The input Tensor for the softmax. Note that the softmax operation would be conducted along the 
        last dimension. 
    tau: ``float``, required
        The temperature hyper-parameter. Note note that reinmax prefers to set tau >= 1, while 
        gumbel-softmax prefers to set tau < 1.  For more details, please refer to our paper. 

    Returns
    -------
    y_hard: ``torch.Tensor``
        The one-hot sample generated from ``multinomial(softmax(logits))``. 
    y_soft: ``torch.Tensor``
        The output of the softmax function, i.e., ``softmax(logits)``. 
    
    Example
    -------
    Below is an example replacing Straight-Through Gumbel-Softmax with ReinMax
    
    .. code-block:: python
        :linenos:
        :emphasize-added: 2
        :emphasize-removed: 1
        
        y_hard = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True)
        y_hard, _ = reinmax.reinmax(logits, tau)
        
    Below is an example replacing Straight-Through with ReinMax
    
    .. code-block:: python
        :linenos:
        :emphasize-added: 4
        :emphasize-removed: 1,2,3
        
        y_hard = one_hot_multinomial(logits.softmax()) 
        y_soft_tau = (logits/tau).softmax()
        y_hard = y_soft_tau - y_soft_tau.detach() + y_hard 
        y_hard, y_soft = reinmax.reinmax(logits, tau)
    """
    if tau < 1:
        raise ValueError("ReinMax prefers to set the temperature (tau) larger or equal to 1.")
    shape = logits.size()
    logits = logits.view(-1, shape[-1])
    grad_sample, y_soft = ReinMaxCore.apply(logits, logits.new_empty(1).fill_(tau))
    return grad_sample.view(shape), y_soft.view(shape)

class OptimizedEncoderConfig(Enum):
    PKM = "pkm"
    Kronecker = "kron"
    FFF = "fff"
    None_ = None

    def build_encoder(
        self,
        d_in: int,
        k: int,
        num_latents: int,
        device: str | torch.device,
        dtype: torch.dtype | None,
        pkm_config=None,
        kron_config=None,
        fff_config=None,
    ) -> nn.Module | None:
        if self is OptimizedEncoderConfig.PKM:
            assert pkm_config is not None
            return PKMLinear(d_in, num_latents, device, dtype, cfg=pkm_config)
        elif self is OptimizedEncoderConfig.Kronecker:
            assert kron_config is not None
            return KroneckerLinear(d_in, num_latents, device, dtype, cfg=kron_config)
        elif self is OptimizedEncoderConfig.FFF:
            assert fff_config is not None
            if fff_config.fff_k is None:
                fff_config = replace(fff_config, fff_k=k)
            return FFFLinear(d_in, num_latents, device, dtype, cfg=fff_config)
        elif self is OptimizedEncoderConfig.None_:
            raise ValueError("No encoder specified.")
        else:
            raise ValueError(f"Unknown encoder type {self}")

    @property
    def is_not_none(self):
        return self is not OptimizedEncoderConfig.None_
