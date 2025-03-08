# PYTHONPATH=hamm/src/python CUDA_VISIBLE_DEVICES=5 python -m sae EleutherAI/pythia-160m togethercomputer/RedPajama-Data-1T-Sample
import json
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple
import math
from einops import rearrange

import einops
import torch
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_model, save_model
from torch import Tensor, nn

from .config import SaeConfig
from .utils import decoder_impl
from .xformers_decoder import xformers_embedding_bag


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""

    top_indices: Tensor
    """Indices of the top-k features."""


class ForwardOutput(NamedTuple):
    sae_out: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    fvu: Tensor
    """Fraction of variance unexplained."""

    auxk_loss: Tensor
    """AuxK loss, if applicable."""

    multi_topk_fvu: Tensor
    """Multi-TopK FVU, if applicable."""


class PKMLinear(nn.Module):
    def __init__(self,
                 d_in: int, num_latents: int,
                 device: str | torch.device,
                 dtype: torch.dtype | None = None,
                 *,
                 cfg: SaeConfig
                 ):
        super().__init__()
        self.d_in = d_in
        self.num_latents = num_latents
        if cfg.pkm_pad:
            self.pkm_base = int(2 ** math.ceil(math.log2(num_latents) / 2))
        else:
            self.pkm_base = int(math.ceil(math.sqrt(num_latents)))
        self.cfg = cfg
        self.num_heads = cfg.pkm_heads
        self._weight = nn.Linear(d_in, cfg.pkm_heads * 2 * self.pkm_base, device=device, dtype=dtype)
        self._weight.weight.data *= cfg.pkm_init_scale / 4
        # Orthogonal matrices have the same FVU  as /4, but produce more dead latents
        # torch.nn.init.orthogonal_(self._weight.weight, gain=0.5 / math.sqrt(self.d_in))
        self._scale = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
        if cfg.pkm_bias:
            self.bias = nn.Parameter(torch.zeros(self.num_heads * self.pkm_base**2, dtype=dtype, device=device))
        if cfg.pkm_softmax:
            self.scaling = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
        if cfg.pkm_rand:
            self.ordering = torch.stack([
                torch.randperm(self.pkm_base ** 2, device=device, dtype=torch.int64)
                for _ in range(self.num_heads)
            ], dim=0)

    @torch.compile(mode="max-autotune")
    def forward(self, x):
        xs = self._weight(x)
        x1, x2 = xs[..., :self.pkm_base], xs[..., self.pkm_base:]
        y = (x1[..., :, None] + x2[..., None, :]).flatten(-2)
        if self.cfg.pkm_bias:
            y += self.bias
        y = y[..., : self.num_latents]
        return y

    @torch.compile(mode="max-autotune")
    def topk(self, x, k: int):
        if not self.cfg.topk_separate:
            x = self.forward(x)
            return x.topk(k, dim=-1)
        orig_batch_size = x.shape[:-1]
        x1, x2 = torch.chunk(
            self._weight(x).unflatten(
                -1, (self.num_heads, self.pkm_base * 2)
            ), 2, dim=-1)
        k_head = max(1, k // self.num_heads)
        k1, k2 = k_head, k_head
        w1, i1 = x1.topk(k1, dim=-1)
        w2, i2 = x2.topk(k2, dim=-1)
        w = torch.nn.functional.relu(w1[..., :, None] + w2[..., None, :]).clone()
        i = i1[..., :, None] * self.pkm_base + i2[..., None, :]
        mask = i >= self.num_latents
        if self.cfg.pkm_bias:
            bias_i = i + torch.arange(self.num_heads, device=i.device, dtype=i.dtype)[:, None, None] * self.pkm_base**2
            w = w + self.bias[bias_i] * mask
        if self.cfg.pkm_rand:
            i_i = self.ordering[:, None, :] * 0 + torch.arange(self.num_heads, device=i.device, dtype=i.dtype)[:, None, None]
            i = i[i_i, :, self.ordering[:, None, :]]
        w[mask] = -1
        w = w.view(-1, self.num_heads, k1 * k2)
        w, i = w.topk(k_head, dim=-1, sorted=True)
        i1 = torch.gather(i1, -1, i // k2)
        i2 = torch.gather(i2, -1, i % k2)
        i = i1 * self.pkm_base + i2
        w = w * (i < self.num_latents)
        i = i.clamp_max(self.num_latents - 1)
        if self.cfg.pkm_softmax:
            # w = torch.nn.functional.softmax(w, dim=-1)
            w = torch.nn.functional.sigmoid(w)
            w = w * torch.nn.functional.softplus(self.scaling)#[i])
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
        return (w1 + w2).reshape(self.pkm_base * pkm_trim, self.num_heads, self.d_in)[:self.num_latents].sum(1)


class KroneckerLinear(nn.Module):
    def __init__(self,
            d_in: int, num_latents: int,
            in_group: int = 2, out_group: int = 4,
            u: int = 4, lora_dim: float = 0.25,
            device: str | torch.device = "cpu",
            dtype: torch.dtype | None = None,
            ):
        assert d_in % in_group == 0
        assert num_latents % out_group == 0
        super().__init__()
        self.in_group = in_group
        self.out_group = out_group
        self.u = u
        self.d_in = d_in
        self.lora_dim = int(d_in * lora_dim)
        self.pre = nn.Linear(d_in, self.lora_dim, device=device, dtype=dtype)
        self.inner = nn.Parameter(torch.randn(out_group, u, in_group, dtype=dtype, device=device))
        self.outer = nn.Parameter(torch.randn(
            num_latents // out_group, u, self.lora_dim // in_group,
            dtype=dtype, device=device))
        self.num_latents = num_latents        
    
    @torch.compile
    def forward(self, x):
        x = self.pre(x)
        x = x.unflatten(-1, (x.shape[-1] // self.in_group, self.in_group))
        x = torch.einsum("...nd,cud->...nuc", x, self.inner)
        x = torch.einsum("...nuc,mun->...cm", x, self.outer)
        return x.reshape(*x.shape[:-2], -1)

    @torch.compile(mode="max-autotune")
    def topk(self, x, k: int):
        return self.forward(x).topk(k, dim=-1)
    
    @property
    def weight(self):
        mat = torch.einsum("yux,mun->ymxn", self.inner, self.outer)
        mat = mat.reshape(self.num_latents, self.lora_dim)
        return mat @ self.pre.weight


class MonarchLinear(nn.Module):
    def __init__(self,
                 d_in: int, num_latents: int,
                 group_dim: int, group_mul: int,
                 device: str | torch.device,
                 dtype: torch.dtype | None = None,
                 ):
        assert d_in % group_dim == 0
        in_groups = d_in // group_dim
        assert num_latents % group_dim == 0
        out_groups = num_latents // group_dim
        super().__init__()
        self.d_in = d_in
        self.num_latents = num_latents
        self.group_dim = group_dim
        self.in_groups = in_groups
        self.group_mul = group_mul
        self._pre = nn.Linear(d_in, d_in, device=device, dtype=dtype)
        self._inner = nn.Parameter(torch.randn(group_mul, in_groups, group_dim, group_dim, dtype=dtype, device=device))
        self._outer = nn.Parameter(torch.randn(out_groups, group_mul, in_groups, dtype=dtype, device=device))
        self._bias = nn.Parameter(torch.zeros(num_latents, dtype=dtype, device=device))

    @property
    def weight(self):
        weight = torch.einsum(
            "mixy,omi->omyix",
            self._inner, self._outer
        )
        weight = rearrange(
            weight,
            "o m y i x -> (o m y) (i x)"
        )
        return weight @ self._pre.weight
    
    @torch.compile
    def forward(self, x):
        x = self._pre(x)
        x = x.unflatten(-1, (self.in_groups, self.group_dim))
        x = torch.einsum("...ix,mixy->...miy", x, self._inner)
        x = torch.einsum("...miy,omi->...oy", x, self._outer)
        return x.flatten(-2)

    @torch.compile(mode="max-autotune")
    def topk(self, x, k: int):
        return self.forward(x).topk(k, dim=-1)

class Sae(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SaeConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in

        self.device = device
        self.dtype = dtype

        self.W_skip = nn.Parameter(
            torch.zeros(d_in, d_in, device=device, dtype=dtype)
        ) if cfg.skip_connection else None
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

        if cfg.monet:
            from .monet import Monet
            self.monet = Monet(cfg.monet_config).to(device=device, dtype=dtype)
            self.num_latents = cfg.monet_config.moe_experts ** 2
            return

        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor
        if cfg.encoder_halut:
            from halutmatmul.modules import HalutLinear
            self.encoder = HalutLinear(d_in, self.num_latents, device=device, dtype=dtype)
            self.encoder.halut_active[:] = 1
            # TODO properly set parameters. We should have the data at this point
            # Reference:
            # https://github.com/joennlae/halutmatmul/blob/master/src/python/halutmatmul/model.py#L92
            self.encoder.bias.data.zero_()
        elif cfg.encoder_pkm:
            self.encoder = PKMLinear(d_in, self.num_latents, device=device, dtype=dtype, cfg=cfg)
            self.encoder._weight.bias.data.zero_()
        elif cfg.encoder_kron:
            self.encoder = KroneckerLinear(
                d_in, self.num_latents,
                in_group=cfg.kron_in_group, out_group=cfg.kron_out_group,
                u=cfg.kron_u, lora_dim=cfg.kron_lora,
                device=device, dtype=dtype)
        elif cfg.encoder_monarch:
            self.encoder = MonarchLinear(
                d_in, self.num_latents,
                group_dim=cfg.monarch_inner_dim, group_mul=cfg.monarch_mul,
                device=device, dtype=dtype
            )
        else:
            self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
            self.encoder.bias.data.zero_()

        if False:
            self.W_dec = nn.Parameter(torch.randn(self.num_latents, d_in, device=device, dtype=dtype)) if decoder else None
        else:
            self.W_dec = nn.Parameter(self.encoder.weight.clone()) if decoder else None
        if decoder and self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

    @staticmethod
    def load_many(
        name: str,
        local: bool = False,
        layers: list[str] | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
        pattern: str | None = None,
    ) -> dict[str, "Sae"]:
        """Load SAEs for multiple hookpoints on a single model and dataset."""
        pattern = pattern + "/*" if pattern is not None else None
        if local:
            repo_path = Path(name)
        else:
            repo_path = Path(snapshot_download(name, allow_patterns=pattern))

        if layers is not None:
            return {
                layer: Sae.load_from_disk(
                    repo_path / layer, device=device, decoder=decoder
                )
                for layer in natsorted(layers)
            }
        files = [
            f
            for f in repo_path.iterdir()
            if f.is_dir() and (pattern is None or fnmatch(f.name, pattern))
        ]
        return {
            f.name: Sae.load_from_disk(f, device=device, decoder=decoder)
            for f in natsorted(files, key=lambda f: f.name)
        }

    @staticmethod
    def load_from_hub(
        name: str,
        hookpoint: str | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "Sae":
        # Download from the HuggingFace Hub
        repo_path = Path(
            snapshot_download(
                name,
                allow_patterns=f"{hookpoint}/*" if hookpoint is not None else None,
            )
        )
        if hookpoint is not None:
            repo_path = repo_path / hookpoint

        # No layer specified, and there are multiple layers
        elif not repo_path.joinpath("cfg.json").exists():
            raise FileNotFoundError("No config file found; try specifying a layer.")

        return Sae.load_from_disk(repo_path, device=device, decoder=decoder)

    @staticmethod
    def load_from_disk(
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "Sae":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SaeConfig.from_dict(cfg_dict, drop_extra_fields=True)

        sae = Sae(d_in, cfg, device=device, decoder=decoder)
        load_model(
            model=sae,
            filename=str(path / "sae.safetensors"),
            device=str(device),
            # TODO: Maybe be more fine-grained about this in the future?
            strict=decoder,
        )
        return sae

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    **self.cfg.to_dict(),
                    "d_in": self.d_in,
                },
                f,
            )

    def pre_acts(self, x: Tensor) -> Tensor:
        # Remove decoder bias as per Anthropic
        sae_in = x.to(self.dtype)  # - self.b_dec
        if self.cfg.monet:
            og_shape = sae_in.shape
            sae_in = sae_in.flatten(0, -2)
            codes = self.monet.encode(sae_in).reshape(*og_shape[:-1], -1)
            return codes
        out = self.encoder(sae_in)

        return nn.functional.relu(out)

    def select_topk(self, latents: Tensor) -> EncoderOutput:
        """Select the top-k latents."""
        return EncoderOutput(*latents.topk(self.cfg.k, sorted=False))

    @torch.compile
    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        if self.cfg.monet:
            return EncoderOutput(self.pre_acts(x), torch.arange(self.num_latents, device=x.device))
        if self.cfg.encoder_weird:
            return self.encoder.topk(x, self.cfg.k)
        return self.select_topk(self.pre_acts(x))

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."

        if self.cfg.decoder_xformers:
            og_batch_size = top_acts.shape[:-1]
            top_acts = top_acts.flatten(0, -2)
            top_indices = top_indices.flatten(0, -2)
            y = xformers_embedding_bag(top_indices, self.W_dec, top_acts.to(torch.bfloat16))
            y = y.view(*og_batch_size, -1)
        else:
            y = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    # Wrapping the forward in bf16 autocast improves performance by almost 2x
    @torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
    )
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:

        # If we aren't given a distinct target, we're autoencoding
        if y is None:
            y = x

        if self.cfg.monet:
            pre_acts = self.pre_acts(x)
            top_acts = pre_acts
            top_indices = torch.arange(self.num_latents, device=x.device).broadcast_to(*top_acts.shape)
            sae_out = self.monet(x.to(self.dtype))  # - self.b_dec)
        else:
            if self.cfg.topk_separate:
                top_acts, top_indices = self.encode(x)  # - self.b_dec)
            else:
                pre_acts = self.pre_acts(x)
                # Decode
                top_acts, top_indices = self.select_topk(pre_acts)
            # if self.cfg.encoder_pkm and self.cfg.pkm_softmax:
                # top_acts = torch.einsum("...d,...kd,...k->...k", x, self.W_dec[top_indices], top_acts)
            sae_out = self.decode(top_acts, top_indices)
            
        if self.W_skip is not None:
            sae_out += x.to(self.dtype) @ self.W_skip.mT
        # del pre_acts

        # Compute the residual
        e = sae_out - y

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (y - y.mean(0)).pow(2).sum()

        # Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = y.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        if self.cfg.multi_topk:
            top_acts, top_indices = pre_acts.topk(4 * self.cfg.k, sorted=False)
            sae_out = self.decode(top_acts, top_indices)

            multi_topk_fvu = (sae_out - y).pow(2).sum() / total_variance
        else:
            multi_topk_fvu = sae_out.new_tensor(0.0)

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
            multi_topk_fvu,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        if self.cfg.monet:
            return

        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        if self.cfg.monet:
            return

        assert self.W_dec is not None, "Decoder weight was not initialized."
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
