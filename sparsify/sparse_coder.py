import json
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple, Optional

import einops
import torch
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_model, save_model
from torch import Tensor, nn

from .config import SparseCoderConfig
from .fused_encoder import EncoderOutput, binary_fused_encoder, fused_encoder
from .utils import decoder_impl, eager_decode

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

        # dL/d(thresholded_preacts) =
        #   dL/d(ste_acts) * d(ste_acts)/d(thresholded_preacts)
        grad_thresholded_preacts = grad_output * grad_sigmoid_wrt_thresholded_preacts

        # The gradient of the loss wrt the preacts is the same as the gradient of the
        # loss wrt the thresholded preacts because the difference is a constant
        # (the threshold)
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
    return DenseBinaryEncode.apply(x, W_enc, b_enc, threshold)  # type: ignore


class BinaryDecode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, top_indices, top_acts, W_dec):
        ctx.save_for_backward(top_indices, top_acts, W_dec)
        return eager_decode(top_indices, top_acts, W_dec)

    @staticmethod
    def backward(ctx, grad_output):
        top_indices, W_dec = ctx.saved_tensors
        grad_top_indices = grad_top_acts = grad_W_dec = None

        return grad_top_indices, grad_top_acts, grad_W_dec


def custom_gumbel_softmax(
    logits: Tensor,
    tau: float = 1,
    eps: float = 1e-10,
    dim: int = -1,
    k: Optional[int] = None,
) -> Tensor:
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(
        logits, memory_format=torch.legacy_contiguous_format
    ).scatter_(dim, index, 1.0)
    if k is not None:
        soft_values, soft_indices = y_soft.topk(k, dim=dim)
        hard_values = torch.gather(y_hard, dim, soft_indices)
        return hard_values + (soft_values - soft_values.detach()), soft_indices
    return y_hard - y_soft.detach() + y_soft


@torch.compile(mode="max-autotune-no-cudagraphs")
def gumbel_encoder(x: Tensor, W_enc: Tensor, b_enc: Tensor, k: int) -> EncoderOutput:
    preacts = torch.einsum("...i,oi->...o", x, W_enc) + b_enc
    grouped = einops.rearrange(preacts, "... (k h) -> ... k h", k=k)
    values, indices = custom_gumbel_softmax(grouped, tau=0.5, dim=-1, k=4)
    values = values.flatten(-2, -1)
    indices = indices + torch.arange(
        indices.shape[-2], device=indices.device
    ).unsqueeze(-1) * (grouped.shape[-1] // indices.shape[-2])
    indices = indices.flatten(-2, -1)
    return EncoderOutput(values, indices, preacts)


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
        # if self.cfg.transcode:
        # self.encoder.weight.data *= 1e-4
        # else:
        # self.encoder.weight.data *= 1e-1

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

        if self.cfg.activation == "binary" or self.cfg.activation == "topk_binary":
            self.log_threshold = nn.Parameter(
                (
                    torch.full((self.num_latents,), 0.004, dtype=dtype, device=device)
                ).log()
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
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def encode(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        if not self.cfg.transcode:
            x = x - self.b_dec

        if self.cfg.activation == "binary":
            # Dense encode with binary activations
            acts = dense_binary_encode(
                x, self.encoder.weight, self.encoder.bias, self.log_threshold
            )
            return EncoderOutput(acts, None, None)
        elif self.cfg.activation == "topk_binary":
            return binary_fused_encoder(
                x,
                self.encoder.weight,
                self.encoder.bias,
                self.cfg.k,
                self.log_threshold.exp(),
            )
        elif self.cfg.activation == "gumbel_binary":
            return gumbel_encoder(x, self.encoder.weight, self.encoder.bias, self.cfg.k)
        else:
            return fused_encoder(
                x,
                self.encoder.weight,
                self.encoder.bias,
                self.cfg.k,
                self.cfg.activation,
            )

    def binary_decode(self, top_indices: Tensor, top_acts: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."
        y = eager_decode(top_indices, top_acts, self.W_dec.mT)
        # y = BinaryDecode.apply(top_indices, top_acts, self.W_dec.mT)
        return y + self.b_dec

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
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        if self.cfg.activation == "binary":
            # sae_out = binary_topk_encode_decode(
            #   x, self.encoder.weight, self.encoder.bias,
            #   self.W_dec.mT, self.b_dec, self.cfg.k)
            # top_acts, top_indices, pre_acts = None, None, None
            # top_acts = dense_binary_encode(x, self.encoder.weight, self.encoder.bias)
            # top_indices = None
            # pre_acts = None

            top_acts = dense_binary_encode(
                x, self.encoder.weight, self.encoder.bias, self.log_threshold
            )
            top_indices, pre_acts = None, None
            sae_out = top_acts @ self.W_dec + self.b_dec
        else:
            top_acts, top_indices, pre_acts = self.encode(x)

        # If we aren't given a distinct target, we're autoencoding
        if y is None:
            y = x

        # Decode
        if self.cfg.activation in "topk_binary":
            sae_out = self.binary_decode(top_indices, top_acts)
        elif self.cfg.activation == "binary":
            pass
            # sae_out = top_acts @ self.W_dec + self.b_dec
        else:
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
            auxk_loss = (e_hat - e.detach()).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        if self.cfg.multi_topk:
            assert (
                not self.cfg.activation == "topk_binary"
            ), "Multi-TopK is not supported for binary activation."
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
