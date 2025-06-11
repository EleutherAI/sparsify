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
from .fused_encoder import EncoderOutput, fused_encoder
from .utils import decoder_impl, ito_step
from .kernels import DenseDenseSparseOutMatmul


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

    aux_loss: Tensor
    """Auxiliary loss, if applicable."""

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

        encoder_num_latents, decoder_num_latents = self.num_latents, self.num_latents
        if cfg.slice_encoder and not cfg.slice_decoder:
            encoder_num_latents = encoder_num_latents * cfg.k
        elif not cfg.slice_encoder and cfg.slice_decoder:
            decoder_num_latents = decoder_num_latents * cfg.k
        self.encoder_slice_size = encoder_num_latents // cfg.k
        self.decoder_slice_size = decoder_num_latents // cfg.k

        self.encoder = nn.Linear(d_in, encoder_num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()
        if cfg.normalize_encoder:
            self.set_encoder_norm_to_unit_norm()

        if decoder:
            # Transcoder initialization: use zeros
            if cfg.transcode:
                self.W_dec = nn.Parameter(torch.zeros((decoder_num_latents, d_in), device=device, dtype=dtype))

            # Sparse autoencoder initialization: use the transpose of encoder weights
            else:
                self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
                if self.cfg.normalize_decoder:
                    self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None
        
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

        if cfg.transcode and cfg.matching_pursuit and cfg.mp_untie:
            if cfg.mp_encoder:
                self.W_dec_2 = self.encoder.weight
            else:
                self.W_dec_2 = nn.Parameter(torch.zeros_like(self.encoder.weight.data))
            self.b_dec_2 = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))
        else:
            self.W_dec_2 = self.W_dec
            self.b_dec_2 = self.b_dec

        self.W_skip = (
            nn.Parameter(torch.zeros(d_in, d_in, device=device, dtype=dtype))
            if cfg.skip_connection
            else None
        )
        
        self.in_proj = None
        # self.in_proj = (
        #     nn.Parameter(torch.eye(d_in, device=device, dtype=dtype))
        #     if cfg.matching_pursuit
        #     else None
        # )

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

    def encode_single(self, x: Tensor, k: int = None, banned: Tensor | None = None, start: int | None = None, end: int | None = None) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        
        if not self.cfg.transcode:
            x = x - self.b_dec
        
        if self.in_proj is not None:
            x = x @ self.in_proj.mT
            
        if start is None:
            start = 0
        if end is None:
            end = self.encoder.weight.shape[0]
        weight = self.encoder.weight[start:end]
        bias = self.encoder.bias[start:end]
        return fused_encoder(
            x, weight, bias, k, self.cfg.activation, banned=banned
        )
    
    @torch.compile
    def encode(self, x: Tensor, k: int | None = None) -> EncoderOutput:
        if k is None:
            k = self.cfg.k
        if not self.cfg.matching_pursuit:
            return self.encode_single(x, k)
        else:
            # x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            sae_out = torch.zeros_like(x)
            sae_out += self.b_dec_2
            banned_indices = []
            for i in range(self.cfg.k):
                if i > 0:
                    sae_out = self.decode(top_acts, top_indices, secondary=True, use_bias=True)
                residual = x - sae_out
                if self.cfg.mp_detach:
                    residual = residual.detach()
                # Encode
                encoding = self.encode_single(
                    residual,
                    k=1,
                    banned=torch.cat(banned_indices, dim=-1) if banned_indices else None,
                    start=None if not self.cfg.slice_encoder else i * self.encoder_slice_size,
                    end=None if not self.cfg.slice_encoder else (i + 1) * self.encoder_slice_size,
                )
                top_act, top_index = encoding.top_acts, encoding.top_indices
                if not self.cfg.slice_encoder:
                    banned_indices.append(top_index)
                else:
                    if self.cfg.slice_decoder:
                        top_index = top_index + i * self.decoder_slice_size
                if i == 0:
                    top_acts, top_indices = top_act, top_index
                else:
                    top_acts = torch.cat([top_acts, top_act], dim=-1)
                    top_indices = torch.cat([top_indices, top_index], dim=-1)
                    if self.cfg.ito:
                        assert self.cfg.mp_encoder, "Inference-time optimization requires mp_encoder to be enabled."
                        top_acts = ito_step(top_acts, top_indices, self.encoder.weight, x - sae_out)
            return EncoderOutput(top_acts, top_indices, None)

    def decode(self, top_acts: Tensor, top_indices: Tensor, secondary: bool = False, use_bias: bool = True) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        if not secondary:
            W_dec, b_dec = self.W_dec, self.b_dec
        else:
            W_dec, b_dec = self.W_dec_2, self.b_dec_2
        y = decoder_impl(top_indices, top_acts.to(self.dtype), W_dec.mT)
        if not self.cfg.matching_pursuit and use_bias:
            y += b_dec
        return y 

    # Wrapping the forward in bf16 autocast improves performance by almost 2x
    @torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_bf16_supported(),
    )
    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        # If we aren't given a distinct target, we're autoencoding
        if y is None:
            y = x

        top_acts, top_indices, pre_acts = self.encode(x)

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
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = top_acts.new_tensor(0.0)

        # Decode
        aux_loss = top_acts.new_tensor(0.0)
        for is_encoder in ((True, False) if self.cfg.mp_aux else (False,)):
            sae_out = self.decode(top_acts, top_indices, secondary=is_encoder)
            if self.W_skip is not None and not is_encoder: 
                sae_out += x.to(self.dtype) @ self.W_skip.mT
            
            # Compute the residual
            e = y - sae_out
            
            l2_loss = e.pow(2).sum()
            fvu = l2_loss / total_variance
            if is_encoder:
                aux_loss = fvu

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
            aux_loss,
            multi_topk_fvu,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps
    
    @torch.no_grad()
    def set_encoder_norm_to_unit_norm(self):
        assert self.encoder.weight is not None, "Encoder weight was not initialized."
        eps = torch.finfo(self.encoder.weight.dtype).eps
        norm = torch.norm(self.encoder.weight.data, dim=1, keepdim=True)
        self.encoder.weight.data /= norm + eps

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
    
    @torch.no_grad()
    def remove_gradient_parallel_to_encoder_directions(self):
        assert self.encoder.weight is not None, "Encoder weight was not initialized."
        assert self.encoder.weight.grad is not None  # keep pyright happy
        
        parallel_component = einops.einsum(
            self.encoder.weight.grad,
            self.encoder.weight.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.encoder.weight.grad -= einops.einsum(
            parallel_component,
            self.encoder.weight.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

# Allow for alternate naming conventions
Sae = SparseCoder
