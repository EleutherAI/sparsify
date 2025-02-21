import json
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple
from sympy import divisors

import einops
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_model, save_model

from .config import SparseCoderConfig
from .utils import decoder_impl


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


class SparseCoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        if decoder:
            # Transcoder initialization: use zeros
            if cfg.transcode:
                self.W_dec = nn.Parameter(torch.zeros_like(self.encoder.weight.data))

            # Sparse autoencoder initialization: use the transpose of encoder weights
            else:
                self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
                if self.cfg.normalize_decoder:
                    self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))
        self.W_skip = (
            nn.Parameter(torch.zeros(d_in, d_in, device=device, dtype=dtype))
            if cfg.skip_connection
            else None
        )

        self.divisors = divisors(self.num_latents)

        def build_k_lookup(possible_k_values):
            lookup = {}
            for cfg_k in possible_k_values:
                lookup[cfg_k] = min(self.divisors, key=lambda x: abs(x - cfg_k))
            return lookup
        
        self.k_lookup = build_k_lookup(list(range(self.cfg.k, 10_000)))
        

    @staticmethod
    def load_many(
        name: str,
        local: bool = False,
        layers: list[str] | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
        pattern: str | None = None,
    ) -> dict[str, "SparseCoder"]:
        """Load sparse coders for multiple hookpoints on a single model and dataset."""
        pattern = pattern + "/*" if pattern is not None else None
        if local:
            repo_path = Path(name)
        else:
            repo_path = Path(snapshot_download(name, allow_patterns=pattern))

        if layers is not None:
            return {
                layer: SparseCoder.load_from_disk(
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
            f.name: SparseCoder.load_from_disk(f, device=device, decoder=decoder)
            for f in natsorted(files, key=lambda f: f.name)
        }

    @staticmethod
    def load_from_hub(
        name: str,
        hookpoint: str | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "SparseCoder":
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

        return SparseCoder.load_from_disk(repo_path, device=device, decoder=decoder)

    @staticmethod
    def load_from_disk(
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "SparseCoder":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SparseCoderConfig.from_dict(cfg_dict, drop_extra_fields=True)

        sae = SparseCoder(d_in, cfg, device=device, decoder=decoder)
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

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def pre_acts(self, x: Tensor) -> Tensor:
        sae_in = x.to(self.dtype)

        # Remove decoder bias as per Anthropic if we're autoencoding. This doesn't
        # really make sense for transcoders because the input and output spaces are
        # different.
        if not self.cfg.transcode:
            sae_in -= self.b_dec

        out = self.encoder(sae_in)
        return nn.functional.relu(out)

    def groupmax_simple(self, z: Tensor) -> EncoderOutput:
        k = self.k_lookup[self.cfg.k]
        print("rounding k from ", self.cfg.k, "to ", k, self.num_latents, self.num_latents % k)
        # if self.num_latents % self.cfg.k != 0:
            
            # k = min(divisors(self.num_latents), key=lambda x: abs(x - self.cfg.k))
        # else:
            # k = self.cfg.k

        values, indices = z.unflatten(-1, (k, -1)).max(dim=-1)

        # torch.max gives us indices into each group, but we want indices into the
        # flattened tensor. Add the offsets to get the correct indices.
        offsets = torch.arange(
            0, self.num_latents, self.num_latents // k, device=z.device
        )
        indices = offsets + indices
        return EncoderOutput(values, indices)


    # def groupmax(self, z: Tensor) -> EncoderOutput:
    #     # If num latents is not a multiple of k we can view there as being either too many or too few latents.
    #     # 1. If too few, we can pad the undersized groups.
    #     # 2. If too many, we can select from a divisible subset of num_latents which means some latents don't get picked.

    #     # This method is option 1.

    #     padding = -self.num_latents % self.cfg.k

    #     if padding == 0:
    #         return EncoderOutput(*z.unflatten(-1, (self.cfg.k, -1)).max(dim=-1))

    #     group_size = (self.num_latents + padding) // self.cfg.k
    #     will_have_empty_groups = padding >= group_size

    #     if will_have_empty_groups:
    #         return self._groupmax_with_empty_groups(z, padding, group_size)
    #     else:
    #         return self._groupmax_simple_padding(z, padding)

    # def _groupmax_simple_padding(self, z: Tensor, padding: int) -> EncoderOutput:
    #     """Handle case where padding is small enough that all groups have real values"""
    #     # Use all padding on left side for simplicity since no empty groups
    #     z_padded = F.pad(z, (padding, 0), value=-1e9)
    #     values, indices = z_padded.unflatten(-1, (self.cfg.k, -1)).max(dim=-1)
        
    #     # Simple offset calculation since all groups have same size
    #     offsets = torch.arange(0, self.num_latents + padding, (self.num_latents + padding) // self.cfg.k, 
    #                         device=z.device)
    #     indices = indices + offsets - padding
        
    #     assert indices.max() < self.num_latents and indices.min() >= 0
    #     return EncoderOutput(values, indices)

    #     # We pad with -inf to produce a latent count that is a multiple of k. 
    #     # For small k all groups will have some non-padding latents

    #     # left_pad = int((padding * torch.rand(1, device=z.device)).item())
    #     left_pad = padding
    #     z = F.pad(z, (left_pad, padding - left_pad), value=-1e9)

    #     values, indices = z.unflatten(-1, (self.cfg.k, -1)).max(dim=-1)

        # For large k, some groups at the start and end will have no non-padding latents

        # Convert relative indices within groups to global indices

        # The first non-empty group will have an offset of 0, and the next will have an offset of 
        # however many non-zero latents there were in the previous group

        # empty_offsets = torch.zeros(self.cfg.k, device=z.device)

        # # offsets = torch.arange(0, self.num_latents + padding, group_size, device=z.device)
        # indices = indices + offsets - left_pad


        # # Mask out empty groups
        # group_size = (self.num_latents + padding) // self.cfg.k
        # num_left_empty = left_pad // group_size
        # num_right_empty = (padding - left_pad) // group_size

        # if num_left_empty > 0:
        #     indices = indices[..., num_left_empty:]
        #     values = values[..., num_left_empty:]
        # if num_right_empty > 0:
        #     indices = indices[..., :-num_right_empty]
        #     values = values[..., :-num_right_empty]

        

        # left pad in units of latents divided by group size in units of latents gives
        

        

        # assert indices.max() < self.num_latents and indices.min() >= 0, "Indices are out of bounds"
        # return EncoderOutput(values, indices)

    # def groupmax_subset(self, z: Tensor) -> EncoderOutput:
    #     # If num latents is not a multiple of k we can view there as being either too many or too few latents.
    #     # 1. If too few, we can pad the undersized groups.
    #     # 2. If too many, we can select from a divisible subset of num_latents which means some latents don't get picked.

    #     # This method is option 2.

    #     # We use randomness to select how many latents are removed from left vs right.
    #     excess_latents = self.num_latents % self.cfg.k
    #     if excess_latents > 0:
    #         # uniform random sample from 0 to excess_latents
    #         left_pad = int((excess_latents * torch.rand(1, device=z.device)).item())
    #         z = z[:, left_pad:-(excess_latents - left_pad)]

    #     # Now we can just do a normal groupmax
    #     return EncoderOutput(*z.unflatten(-1, (self.cfg.k, -1)).max(dim=-1))

    def select_topk(self, z: Tensor) -> EncoderOutput:
        """Select the top-k latents."""

        # Use GroupMax activation to get the k "top" latents
        if self.cfg.activation == "groupmax":
            return self.groupmax_simple(z)

        # Use TopK activation
        return EncoderOutput(*z.topk(self.cfg.k, sorted=False))

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        return self.select_topk(self.pre_acts(x))

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."

        y = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    # Wrapping the forward in bf16 autocast improves performance by almost 2x
    @torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
    )
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        pre_acts = self.pre_acts(x)

        # If we aren't given a distinct target, we're autoencoding
        if y is None:
            y = x

        # Decode
        top_acts, top_indices = self.select_topk(pre_acts)
        sae_out = self.decode(top_acts, top_indices)
        if self.W_skip is not None:
            sae_out += x.to(self.dtype) @ self.W_skip.mT

        # Compute the residual
        e = y - sae_out

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
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
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


# Allow for alternate naming conventions
Sae = SparseCoder
