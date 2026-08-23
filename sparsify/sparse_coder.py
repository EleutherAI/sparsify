import json
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple

import einops
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors import safe_open
from safetensors.torch import load_model, save_model
from torch import Tensor, nn

from .config import SparseCoderConfig
from .fused_encoder import EncoderOutput, fused_encoder
from .latent_parallel import (
    Winners,
    all_reduce_sum,
    compact_winners,
    global_sum,
    global_topk_mask,
    ragged_decode,
    shard_size,
    sparse_encoder_grad,
)
from .utils import decoder_impl


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
        latent_shard: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.global_num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        # `latent_shard` is (rank, world_size): this coder holds only its slice of
        # the latent dimension and cooperates with its peers in `forward`.
        if latent_shard is None:
            self.num_latents = self.global_num_latents
            self.latent_offset = 0
            self.latent_world_size = 1
        else:
            shard_rank, world_size = latent_shard
            if cfg.activation != "topk":
                raise ValueError(
                    "Latent sharding currently supports activation='topk' only; "
                    f"'{cfg.activation}' partitions the latent dimension itself."
                )
            self.num_latents = shard_size(self.global_num_latents, world_size)
            self.latent_offset = shard_rank * self.num_latents
            self.latent_world_size = world_size

        if latent_shard is None:
            self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        else:
            # Draw the *global* initialization and keep only this rank's rows.
            # Every rank runs the same seed, so drawing at the sharded shape
            # directly would give all of them identical weights, collapsing the
            # dictionary to num_latents // world_size distinct latents. Drawing at
            # the full shape also makes a sharded run reproduce an unsharded run
            # with the same seed, row for row. The full draw is transient.
            full = nn.Linear(d_in, self.global_num_latents, device=device, dtype=dtype)
            self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
            self.encoder.weight.data.copy_(
                full.weight.data[
                    self.latent_offset : self.latent_offset + self.num_latents
                ]
            )
            del full

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

        safetensors_path = str(path / "sae.safetensors")

        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            first_key = next(iter(f.keys()))
            reference_dtype = f.get_tensor(first_key).dtype

        sae = SparseCoder(
            d_in, cfg, device=device, decoder=decoder, dtype=reference_dtype
        )

        load_model(
            model=sae,
            filename=safetensors_path,
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

    @property
    def is_latent_sharded(self) -> bool:
        return self.latent_world_size > 1

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        if not self.cfg.transcode:
            x = x - self.b_dec

        return fused_encoder(
            x, self.encoder.weight, self.encoder.bias, self.cfg.k, self.cfg.activation
        )

    def _count_dead(self, dead_mask: Tensor) -> int:
        """Number of dead latents, counted over the whole dictionary."""
        if not self.is_latent_sharded:
            return int(dead_mask.sum())

        return int(global_sum(dead_mask.sum()))

    def _sharded_encode(self, x: Tensor) -> tuple[Tensor, Winners, Tensor]:
        """Select globally, then rejoin the graph on the surviving pairs only.

        The dense pre-activation exists solely to run the top-k, so it is built
        outside the graph; `sparse_encoder_grad` reattaches the encoder gradient
        to the winners, which keeps the backward proportional to how many latents
        this rank actually owns rather than to how many it offered.
        """
        shifted = x if self.cfg.transcode else x - self.b_dec

        with torch.no_grad():
            pre_acts = F.relu(F.linear(shifted, self.encoder.weight, self.encoder.bias))
            cand_acts, cand_idx = pre_acts.topk(self.cfg.k, dim=-1, sorted=False)
            winners = compact_winners(
                global_topk_mask(cand_acts, self.cfg.k), cand_acts, cand_idx
            )

        acts = sparse_encoder_grad(
            winners.values, shifted, self.encoder.weight, self.encoder.bias, winners
        )
        return acts, winners, pre_acts

    def _sharded_decode(self, acts: Tensor, winners: Winners, x: Tensor) -> Tensor:
        """Decode this rank's winners and sum the partial reconstructions.

        `b_dec` and `W_skip` are replicated rather than sharded, so each rank
        contributes 1/W of them: the sum reproduces each exactly once, and each
        rank ends up holding 1/W of their gradient, which
        `latent_parallel.sync_replicated_grads` then adds back up.
        """
        assert self.W_dec is not None, "Decoder weight was not initialized."
        world_size = self.latent_world_size

        partial = ragged_decode(winners, acts, self.W_dec)
        partial = partial + self.b_dec / world_size
        if self.W_skip is not None:
            partial = partial + (x.to(self.dtype) @ self.W_skip.mT) / world_size
        return all_reduce_sum(partial)

    def _sharded_global_winners(self, scores: Tensor, k: int) -> Winners:
        """Global top-k over a shard-local score matrix, as a flat winner list."""
        candidates = min(k, self.num_latents)
        vals, idx = scores.topk(candidates, sorted=False)
        return compact_winners(global_topk_mask(vals, k), vals, idx)

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."

        y = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y + self.b_dec

    # Wrapping the forward in bf16 autocast improves performance by almost 2x
    @torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_bf16_supported(),
    )
    def forward(
        self,
        x: Tensor,
        y: Tensor | None = None,
        *,
        dead_mask: Tensor | None = None,
        total_variance: Tensor | None = None,
    ) -> ForwardOutput:
        if self.is_latent_sharded:
            top_acts, winners, pre_acts = self._sharded_encode(x)
            top_indices = winners.latents
        else:
            top_acts, top_indices, pre_acts = self.encode(x)

        # If we aren't given a distinct target, we're autoencoding
        if y is None:
            y = x

        # Decode
        if self.is_latent_sharded:
            sae_out = self._sharded_decode(top_acts, winners, x)
        else:
            sae_out = self.decode(top_acts, top_indices)
            if self.W_skip is not None:
                sae_out = sae_out + x.to(self.dtype) @ self.W_skip.mT

        # Compute the residual
        e = y - sae_out

        # Denominator for scale; chunked callers pass the unchunked batch's variance
        if total_variance is None:
            total_variance = (y - y.mean(0)).pow(2).sum()

        # Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := self._count_dead(dead_mask)) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = y.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents. We call decoder_impl directly rather than
            # self.decode because the residual target e already accounts for b_dec
            # (sae_out includes it), so adding b_dec again here would double-count it.
            assert self.W_dec is not None, "Decoder weight was not initialized."

            if self.is_latent_sharded:
                aux = self._sharded_global_winners(auxk_latents, k_aux)
                e_hat = all_reduce_sum(ragged_decode(aux, aux.values, self.W_dec))
            else:
                auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)
                e_hat = decoder_impl(
                    auxk_indices, auxk_acts.to(self.dtype), self.W_dec.mT
                )
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        if self.cfg.multi_topk:
            wide_k = 4 * self.cfg.k
            if self.is_latent_sharded:
                wide = self._sharded_global_winners(pre_acts, wide_k)
                sae_out = self._sharded_decode(wide.values, wide, x)
            else:
                top_acts, top_indices = pre_acts.topk(wide_k, sorted=False)
                sae_out = self.decode(top_acts, top_indices)
                if self.W_skip is not None:
                    sae_out = sae_out + x.to(self.dtype) @ self.W_skip.mT

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
