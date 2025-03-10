import json
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple

import einops
import torch
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_model, save_model
from torch import Tensor, nn

from .config import SparseCoderConfig
from .utils import decoder_impl


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents from the final layer."""

    top_indices: Tensor
    """Indices of the top-k features."""


class ForwardOutput(NamedTuple):
    sae_out: Tensor
    """Output of the sparse autoencoder after decoding."""

    latent_acts: Tensor
    """Activations of the top-k latents from the final layer."""

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
        
        # Support multiple layers
        if cfg.num_layers <= 0:
            raise ValueError("num_layers must be at least 1")
        
        # Create a ModuleList for the encoders
        self.encoders = nn.ModuleList()
        for _ in range(cfg.num_layers):
            layer = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
            layer.bias.data.zero_()
            self.encoders.append(layer)

        if decoder:
            # Create decoders corresponding to each encoder layer
            self.W_decs = nn.ParameterList()
            self.b_decs = nn.ParameterList()
            
            if cfg.residual_mode == "densenet":
                self.skip_layers = nn.ModuleList()
                self.out_projections = nn.ModuleList()
            
            # Initialize each decoder
            for encoder_idx, encoder in enumerate(self.encoders):
                # Transcoder initialization: use zeros
                if cfg.transcode:
                    self.W_decs.append(nn.Parameter(torch.zeros_like(encoder.weight.data)))
                # SAE initialization: use the transpose of encoder weights
                else:
                    self.W_decs.append(nn.Parameter(encoder.weight.data.clone()))
                self.b_decs.append(nn.Parameter(torch.zeros_like(encoder.bias.data)))
                if cfg.residual_mode == "densenet":
                    if encoder_idx > 0:
                        self.out_projections.append(nn.Linear(d_in, d_in, device=device, dtype=dtype))
                    for i in range(encoder_idx - 1):
                        self.skip_layers.append(nn.Linear(d_in, d_in, device=device, dtype=dtype))
            # Normalize the final decoder if needed
            if self.cfg.normalize_decoder and not cfg.transcode:
                self.set_decoder_norm_to_unit_norm()
        else:
            self.W_decs = None
            self.b_decs = None
        
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
        return self.encoders[0].weight.device

    @property
    def dtype(self):
        return self.encoders[0].weight.dtype

    # def pre_acts(self, x: Tensor) -> Tensor:
    #     sae_in = x.to(self.dtype)

    #     # Remove decoder bias as per Anthropic if we're autoencoding. This doesn't
    #     # really make sense for transcoders because the input and output spaces are
    #     # different.
    #     if not self.cfg.transcode:
    #         sae_in -= self.b_dec

    #     # Process through all encoder layers
    #     out = sae_in
    #     for encoder in self.encoders:
    #         out = encoder(out)
    #         out = nn.functional.relu(out)
        
    #     return out

    def select_topk(self, z: Tensor) -> EncoderOutput:
        """Select the top-k latents."""

        # Use GroupMax activation to get the k "top" latents
        if self.cfg.activation == "groupmax":
            values, indices = z.unflatten(-1, (self.cfg.k, -1)).max(dim=-1)

            # torch.max gives us indices into each group, but we want indices into the
            # flattened tensor. Add the offsets to get the correct indices.
            offsets = torch.arange(
                0, self.num_latents, self.num_latents // self.cfg.k, device=z.device
            )
            indices = offsets + indices
            return EncoderOutput(values, indices)

        # Use TopK activation
        return EncoderOutput(*z.topk(self.cfg.k, sorted=False))

    # def encode(self, x: Tensor) -> EncoderOutput:
    #     """Encode the input and select the top-k latents."""
    #     return self.select_topk(self.pre_acts(x))

    # def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
    #     assert self.W_decs is not None, "Decoder weights were not initialized."

    #     # Start with the output from the final encoder layer
    #     y = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_decs[-1].mT)
        
    #     # If we have multiple layers, we need to process backward through the decoders
    #     # Starting from the second-to-last decoder and going backward
    #     for i in range(len(self.W_decs) - 2, -1, -1):
    #         y = decoder_impl(top_indices, y, self.W_decs[i].mT)
            
    #     return y + self.b_dec

    # Wrapping the forward in bf16 autocast improves performance by almost 2x
    @torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
    )
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        sae_in = x.to(self.dtype)
        if not self.cfg.transcode:
            sae_in -= self.b_decs[0]

        pre_acts = nn.functional.relu(self.encoders[0](sae_in))
        top_acts, top_indices = self.select_topk(pre_acts)
        sae_out = decoder_impl(top_indices, top_acts, self.W_decs[0].mT)
        sae_out += self.b_decs[0]
        
        sae_out_original = sae_out.clone()
        if self.cfg.residual_mode != "residual":
            sae_in.zero_()
        sae_in += sae_out
        previous_outputs, densenet_index = [], 0
        for encoder_idx, encoder in enumerate(self.encoders[1:], start=1):
            if self.cfg.residual_mode == "densenet":
                sae_in[:] = sae_out_original
                for i in range(encoder_idx - 1):
                    sae_in += self.skip_layers[densenet_index](previous_outputs[i])
                    densenet_index += 1
            if not self.cfg.transcode:
                sae_in -= self.b_decs[encoder_idx]

            encoded = nn.functional.relu(encoder(sae_in))
            top_acts, top_indices = self.select_topk(encoded)
            most_recent_sae_out = decoder_impl(top_indices, top_acts, self.W_decs[encoder_idx].mT)
            most_recent_sae_out += self.b_decs[encoder_idx]
            if self.cfg.residual_mode == "densenet":
                previous_outputs.append(most_recent_sae_out)
                sae_out += self.out_projections[encoder_idx - 1](most_recent_sae_out)
            else:
                sae_out += most_recent_sae_out
                sae_in += most_recent_sae_out

        # If we aren't given a distinct target, we're autoencoding
        if y is None:
            y = x

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
            top_acts, top_indices = acts.topk(4 * self.cfg.k, sorted=False)
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
        assert self.W_decs is not None, "Decoder weight was not initialized."

        # Normalize each decoder layer
        for decoder in self.W_decs:
            eps = torch.finfo(decoder.dtype).eps
            norm = torch.norm(decoder.data, dim=1, keepdim=True)
            decoder.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_decs is not None, "Decoder weight was not initialized."
        
        # Process each decoder layer
        for decoder in self.W_decs:
            if decoder.grad is not None:
                parallel_component = einops.einsum(
                    decoder.grad,
                    decoder.data,
                    "d_sae d_in, d_sae d_in -> d_sae",
                )
                decoder.grad -= einops.einsum(
                    parallel_component,
                    decoder.data,
                    "d_sae, d_sae d_in -> d_sae d_in",
                )


# Allow for alternate naming conventions
Sae = SparseCoder
