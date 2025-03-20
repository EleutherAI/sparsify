import json
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple
import math

import einops
import torch
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_model, save_model
from torch import Tensor, nn

from .config import SparseCoderConfig
# from .step import Step, JumpReLU

import torch

import lovely_tensors as lt
lt.monkey_patch()


def rectangle(x):
    """Kernel function for straight-through estimator"""
    return ((x > -0.5) & (x < 0.5)).type_as(x)


class Step(torch.autograd.Function):
    """
    Heaviside step function with custom backwards pass for L0 loss
    """

    bandwidth = 2. # 0.001 in original

    @staticmethod
    def forward(ctx, pre_acts, threshold):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(pre_acts, threshold)

        return (pre_acts > threshold).type_as(pre_acts)

    @staticmethod
    def backward(ctx, output_grad):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        pre_acts, threshold = ctx.saved_tensors

        # Pseudo-derivative of the Dirac delta component of the Heaviside function
        rectangle_vals = rectangle((pre_acts - threshold) / Step.bandwidth)
        threshold_grad = torch.sum(
            -(1.0 / Step.bandwidth)
            * rectangle_vals
            * output_grad,
            dim=0
        )

        # print("threshold_grad in step", threshold_grad)
        return None, threshold_grad


class JumpReLU(torch.autograd.Function):
    """
    JumpReLU function with custom backwards pass
    """

    bandwidth = 2. # 0.001 in original

    @staticmethod
    def forward(ctx, pre_acts, threshold):
        mask = (pre_acts > threshold).type_as(pre_acts)
        out = pre_acts * mask

        ctx.save_for_backward(pre_acts, threshold)
        
        return out, mask.detach().sum(0) > 0

    @staticmethod
    def backward(ctx, output_grad, did_fire_grad):

        pre_acts, threshold = ctx.saved_tensors

        # We don’t apply STE to x input
        pre_acts_grad = (pre_acts > threshold).type_as(output_grad) * output_grad

        # Pseudo-derivative of the Dirac delta component of the JumpRelU function
        rectangle_vals = rectangle((pre_acts - threshold) / JumpReLU.bandwidth)
        threshold_grad = torch.sum(
            -(threshold / JumpReLU.bandwidth)
            * rectangle_vals
            * output_grad,
            dim=0
        )
        # print("threshold_grad in jump", threshold_grad)

        return pre_acts_grad, threshold_grad
                 


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""

    top_indices: Tensor
    """Indices of the top-k features."""


class ForwardOutput(NamedTuple):
    sae_out: Tensor

    did_fire: Tensor

    l0: Tensor
    """Number of latents fired."""

    per_example_l0: Tensor
    """Per-example L0 loss."""

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
        # import torch.nn.init as init
        # init.kaiming_normal_(self.encoder.weight)

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

        if self.cfg.activation == "jumprelu":
            self.log_threshold = nn.Parameter(
                # 0.05, and lambda from 1-10 is decent
                torch.full((self.num_latents,), math.log(0.04), device=device), 
                requires_grad=True
            )
            self.jump_relu = JumpReLU.apply
            self.step = Step.apply

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
    
    def jump_decode(self, acts: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."        
        return acts @ self.W_dec + self.b_dec

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

        pre_acts = self.pre_acts(x)
        # print("log threshold", self.log_threshold)
        # print("threshold", self.log_threshold.exp())

        acts, did_fire = self.jump_relu(pre_acts, self.log_threshold.exp()) # type: ignore
        
        per_example_l0 = self.step(pre_acts, self.log_threshold.exp()).sum(dim=-1) # type: ignore
        l0_loss = (per_example_l0 / self.cfg.k - 1).pow(2).sum()

        sae_out = self.jump_decode(acts)

        # Compute the residual
        e = y - sae_out

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (y - y.mean(0)).pow(2).sum()

        # Second decoder pass for AuxK loss
        # if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
        #     # Heuristic from Appendix B.1 in the paper
        #     k_aux = y.shape[-1] // 2

        #     # Reduce the scale of the loss if there are a small number of dead latents
        #     scale = min(num_dead / k_aux, 1.0)
        #     k_aux = min(k_aux, num_dead)

        #     # Don't include living latents in this loss
        #     auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)

        #     # Top-k dead latents
        #     auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

        #     # Encourage the top ~50% of dead latents to predict the residual of the
        #     # top k living latents
        #     e_hat = self.decode(auxk_acts, auxk_indices)
        #     auxk_loss = (e_hat - e).pow(2).sum()
        #     auxk_loss = scale * auxk_loss / total_variance
        # else:
        auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        # if self.cfg.multi_topk:
        #     raise NotImplementedError("Not implemented")
            # assert not self.cfg.activation == "jumprelu", "Multi-TopK is not supported for JumpReLU."
            # top_acts, top_indices = pre_acts.topk(4 * self.cfg.k, sorted=False)
            # sae_out = self.decode(top_acts, top_indices)

            # multi_topk_fvu = (sae_out - y).pow(2).sum() / total_variance
        # else:
        multi_topk_fvu = sae_out.new_tensor(0.0)

        return ForwardOutput(
            sae_out,
            did_fire,
            l0_loss, 
            per_example_l0.float().mean().item(),
            fvu,
            auxk_loss,
            multi_topk_fvu
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
