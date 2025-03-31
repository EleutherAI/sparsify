import json
from fnmatch import fnmatch
from pathlib import Path
from turtle import right
from typing import NamedTuple

import einops
import torch
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_model, save_model
from torch import Tensor, nn

from .config import SparseCoderConfig
from .fused_encoder import EncoderOutput, fused_encoder, binary_fused_encoder
from .utils import decoder_impl, eager_decode

import torch.nn.functional as F


# class BinaryTopKEncodeDecode(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, W_enc, b_enc, W_dec, b_dec, k):
#         preacts = x @ W_enc.T + b_enc

#         top_acts, top_indices = preacts.topk(k, sorted=False)

#         # Implicitly binarize top_acts to 1 by using the default weights. 
#         # The gradient of top_acts is lost. 
#         # We use a STE in the backwards pass to approximate its binarization.
#         sae_out = nn.functional.embedding_bag(
#             top_indices, W_dec.mT, mode="sum"
#         ) + b_dec

#         ctx.save_for_backward(x, W_enc, b_enc, W_dec, b_dec, top_indices, top_acts)
#         return sae_out
    
#     @staticmethod
#     @torch.autocast(
#         "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
#     )
#     def backward(ctx, grad_output):
#         x, W_enc, b_enc, W_dec, b_dec, top_indices, top_acts = ctx.saved_tensors

#         # STE for binarized top_acts
#         top_acts_sigmoid = torch.sigmoid(top_acts)
#         grad_top_acts_sigmoid = top_acts_sigmoid * (1 - top_acts_sigmoid)

#         # Gradient of decoder weights wrt decoder output
#         # W_dec is [d_in, d_sae] but only the top k columns are non-zero for each batch of activations,
#         # so only the corresponding columns of grad_output in the batch are non-zero
#         # grad_W_dec = torch.zeros_like(W_dec)
#         # # for each act scale the grad by its magnitude. 
#         # # we may want to come back later and use a different STE than identity for top_acts
#         # # grad_output is [batch, d_resid] and top_acts is [batch, k]
#         # # TODO there may be an error, do we need the gradient of the sigmoid here?
#         # grad_contributions = grad_output.unsqueeze(1) * top_acts_sigmoid.unsqueeze(2)
#         # # grad_contributions is [batch, k, d_in]
#         # _, _, D = grad_contributions.shape
#         # grad_contributions = grad_contributions.reshape(-1, D)
#         # grad_W_dec.index_add_(1, top_indices.flatten(), grad_contributions.T) 

#         # Gradient wrt decoder bias
#         # The derivative of the output wrt the bias is 1 because it gets added (because changing the bias by x changes the output by x)
#         grad_b_dec = grad_output.sum(0)

#         # Gradient of the loss wrt top acts (using sigmoid STE) - how does changing the top acts affect the sae_out?
#         from xformers import embedding_bag_bw_rev_indices
#         # top acts is [batch, k] so the grad is too.
#         # grad_output is [batch, d_resid]
#         # The derivative of sigmoid(top acts) wrt top acts is sigmoid(top acts) * (1 - sigmoid(top acts))
#         grad_top_acts = embedding_bag_bw_rev_indices(top_indices, W_dec, grad_output, grad_output * top_acts_sigmoid * (1 - top_acts_sigmoid))

#         grad_top_acts, grad_W_dec = embedding_bag_bw_rev_indices(
#             top_indices,
#             W_dec,
#             grad_top_acts_sigmoid,
#             grad_output,
#         )

#         # --- Grad w.r.t. input ---
#         if ctx.needs_input_grad[0]:
#             grad_input = F.embedding_bag(
#                 indices,
#                 right,
#                 mode="sum",
#                 per_sample_weights=grad_top_acts.type_as(W_enc),
#             )

#         # --- Grad w.r.t. weight ---
#         if ctx.needs_input_grad[1]:
#             grad_W_enc = torch.zeros_like(W_enc)
#             # Compute contributions from each top-k element:
#             # computed as grad_values * input for each top-k location.
#             contributions = grad_top_acts.unsqueeze(2) * input.unsqueeze(1)
#             _, _, D = contributions.shape
#             # Flatten contributions to shape (N*k, D)
#             contributions = contributions.reshape(-1, D)

#             # Accumulate contributions into the correct rows of grad_W_enc.
#             grad_W_enc.index_add_(0, indices.flatten(), contributions.type_as(W_enc))
#             print("grad weight", grad_W_enc)
#         # --- Grad w.r.t. bias ---
#         if b_enc is not None and ctx.needs_input_grad[2]:
#             grad_b_enc = torch.zeros_like(b_enc)
#             grad_b_enc.index_add_(
#                 0, indices.flatten(), grad_top_acts.flatten().type_as(b_enc)
#             )
#             print("grad bias", grad_b_enc)

#         return grad_input, grad_W_enc, grad_b_enc, grad_W_dec, grad_b_dec
        

# def binary_topk_encode_decode(x, W_enc, b_enc, W_dec, b_dec, k) -> Tensor:
#     return BinaryTopKEncodeDecode.apply(x, W_enc, b_enc, W_dec, b_dec, k) # type: ignore

# Try a different STE eg rectangle or tanh


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
            self.log_threshold = nn.Parameter((torch.full((self.num_latents,), 0.004, dtype=dtype, device=device)).log())

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
            acts = dense_binary_encode(x, self.encoder.weight, self.encoder.bias, self.log_threshold)
            return EncoderOutput(acts, None, None)
        elif self.cfg.activation == "topk_binary":
            return binary_fused_encoder(
                x, self.encoder.weight, self.encoder.bias, self.cfg.k, self.log_threshold.exp()
            )
        else:
            return fused_encoder(
                x, self.encoder.weight, self.encoder.bias, self.cfg.k, self.cfg.activation
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
            # sae_out = binary_topk_encode_decode(x, self.encoder.weight, self.encoder.bias, self.W_dec.mT, self.b_dec, self.cfg.k)
            # top_acts, top_indices, pre_acts = None, None, None
            # top_acts = dense_binary_encode(x, self.encoder.weight, self.encoder.bias)
            # top_indices = None
            # pre_acts = None

            top_acts = dense_binary_encode(x, self.encoder.weight, self.encoder.bias, self.log_threshold)
            top_indices, pre_acts = None, None
            sae_out = top_acts @ self.W_dec + self.b_dec
        else:
            top_acts, top_indices, pre_acts = self.encode(x)

        # If we aren't given a distinct target, we're autoencoding
        if y is None:
            y = x

        # Decode
        if self.cfg.activation == "topk_binary":
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
            assert not self.cfg.activation == "topk_binary", "Multi-TopK is not supported for binary activation."
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
