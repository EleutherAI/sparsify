import json
import os
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple, Optional

import einops
import torch
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_model
from torch import Tensor, nn
from torch.distributed import tensor as dtensor
from torch.distributed.tensor.device_mesh import DeviceMesh

from .config import SparseCoderConfig
from .fused_encoder import EncoderOutput, fused_encoder
from .utils import decoder_impl, load_sharded, save_sharded


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

    is_last: bool = False
    """Whether this is the last target in a multi-target setup."""


class MidDecoder:
    def __init__(
        self,
        sparse_coder: "SparseCoder",
        x: Tensor,
        activations: Tensor,
        indices: Tensor,
        pre_acts: Optional[Tensor] = None,
        dead_mask: Optional[Tensor] = None,
    ):
        self.sparse_coder = sparse_coder
        self.x = x
        self.latent_acts = activations
        self.latent_indices = indices
        self.pre_acts = pre_acts
        self.dead_mask = dead_mask
        self._index = 0

    def detach(self):
        if not hasattr(self, "original_activations"):
            self.original_activations = self.latent_acts
            self.latent_acts = self.latent_acts.detach()
            self.latent_acts.requires_grad = True

    def restore(self, is_last: bool = False):
        grad = self.latent_acts.grad
        assert grad is not None, "Activations have no gradient."
        self.latent_acts = self.original_activations
        self.latent_acts.backward(grad, retain_graph=not is_last)
        del self.original_activations

    def next(self):
        self._index += 1

    @torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=torch.cuda.is_bf16_supported(),
    )
    def __call__(
        self,
        y: Tensor | None,
        index: Optional[int] = None,
        addition: float | Tensor = 0,
        no_extras: bool = False,
    ) -> ForwardOutput:
        # If we aren't given a distinct target, we're autoencoding
        if y is None:
            y = self.x
            if isinstance(y, dtensor.DTensor):
                y = y.redistribute(
                    y.device_mesh, (dtensor.Replicate(), dtensor.Shard(1))
                )

        assert isinstance(y, Tensor), "y must be a tensor."
        if index is None:
            index = self._index
            self.next()
        is_last = self._index >= self.sparse_coder.cfg.n_targets

        # Decode
        sae_out = self.sparse_coder.decode(self.latent_acts, self.latent_indices, index)
        W_skip = (
            self.sparse_coder.W_skips[index]
            if self.sparse_coder.multi_target
            else self.sparse_coder.W_skip
        )
        if W_skip is not None:
            sae_out += self.x.to(self.sparse_coder.dtype) @ W_skip.mT
        sae_out += addition

        if no_extras:
            return ForwardOutput(
                sae_out,
                self.latent_acts,
                self.latent_indices,
                sae_out.new_tensor(0.0),
                sae_out.new_tensor(0.0),
                sae_out.new_tensor(0.0),
                is_last,
            )
        else:
            # Compute the residual
            e = y - sae_out

            # Used as a denominator for putting everything on a reasonable scale
            total_variance = (y - y.mean(0)).pow(2).sum()

            l2_loss = e.pow(2).sum()
            fvu = l2_loss / total_variance

            # Second decoder pass for AuxK loss
            if (
                self.dead_mask is not None
                and self.pre_acts is not None
                and (num_dead := int(self.dead_mask.sum())) > 0
            ):
                # Heuristic from Appendix B.1 in the paper
                k_aux = y.shape[-1] // 2

                # Reduce the scale of the loss
                # if there are a small number of dead latents
                scale = min(num_dead / k_aux, 1.0)
                k_aux = min(k_aux, num_dead)

                # Don't include living latents in this loss
                auxk_latents = torch.where(
                    self.dead_mask[None], self.pre_acts, -torch.inf
                )

                # Top-k dead latents
                auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

                # Encourage the top ~50% of dead latents to
                # predict the residual of the top k living latents
                e_hat = self.sparse_coder.decode(auxk_acts, auxk_indices, index)
                auxk_loss = (e_hat - e.detach()).pow(2).sum()
                auxk_loss = scale * auxk_loss / total_variance
            else:
                auxk_loss = sae_out.new_tensor(0.0)

            if self.sparse_coder.cfg.multi_topk and self.pre_acts is not None:
                top_acts, top_indices = self.pre_acts.topk(
                    4 * self.sparse_coder.cfg.k, sorted=False
                )
                sae_out = self.sparse_coder.decode(top_acts, top_indices, index)

                multi_topk_fvu = (sae_out - y).pow(2).sum() / total_variance
            else:
                multi_topk_fvu = sae_out.new_tensor(0.0)

        return ForwardOutput(
            sae_out,
            self.latent_acts,
            self.latent_indices,
            fvu,
            auxk_loss,
            multi_topk_fvu,
            is_last,
        )


class SparseCoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: SparseCoderConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
        mesh: Optional[DeviceMesh] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor
        self.multi_target = cfg.n_targets > 0 and cfg.transcode
        self.mesh = mesh

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()
        if mesh is None:
            self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
            self.encoder.bias.data.zero_()
        else:
            self.encoder = nn.Linear(
                d_in,
                self.num_latents // mesh.shape[1],
                device=device,
                dtype=dtype,
            )
            self.encoder.bias.data.zero_()
            scaling = 1 / self.encoder.weight.shape[1] ** 0.5
            self.encoder.register_parameter(
                "weight",
                nn.Parameter(
                    # default torch initialization
                    dtensor.rand(
                        (self.num_latents, d_in),
                        dtype=dtype,
                        device_mesh=mesh,
                        placements=[dtensor.Replicate(), dtensor.Shard(0)],
                    )
                    * (2.0 * scaling)
                    - scaling
                ),
            )
            self.encoder.register_parameter(
                "bias",
                nn.Parameter(
                    dtensor.DTensor.from_local(
                        self.encoder.bias.data,
                        mesh,
                        placements=[
                            dtensor.Replicate(),
                            dtensor.Shard(0),
                        ],
                    )
                ),
            )

        if decoder:
            # Transcoder initialization: use zeros
            if cfg.transcode:

                def create_W_dec():
                    if mesh is not None:
                        result = dtensor.zeros(
                            (self.num_latents, d_in),
                            dtype=dtype,
                            device_mesh=mesh,
                            placements=[dtensor.Replicate(), dtensor.Shard(1)],
                        )
                    else:
                        result = torch.zeros_like(self.encoder.weight.data)
                    return nn.Parameter(result)

                if self.multi_target:
                    self.W_decs = nn.ParameterList()
                    for _ in range(cfg.n_targets):
                        self.W_decs.append(create_W_dec())
                else:
                    self.W_dec = create_W_dec()

            # Sparse autoencoder initialization: use the transpose of encoder weights
            else:
                self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
                if self.cfg.normalize_decoder:
                    self.set_decoder_norm_to_unit_norm()
        else:
            self.W_dec = None

        def create_bias():
            if mesh is not None:
                result = dtensor.zeros(
                    (self.d_in,),
                    dtype=dtype,
                    device_mesh=mesh,
                    placements=[dtensor.Replicate(), dtensor.Shard(0)],
                )
            else:
                result = torch.zeros(self.d_in, device=device, dtype=dtype)
            return nn.Parameter(result)

        def create_W_skip():
            if not cfg.skip_connection:
                return None
            if mesh is not None:
                result = dtensor.zeros(
                    (self.d_in, self.d_in),
                    dtype=dtype,
                    device_mesh=mesh,
                    placements=[dtensor.Replicate(), dtensor.Shard(0)],
                )
            else:
                result = torch.zeros(self.d_in, self.d_in, device=device, dtype=dtype)
            return nn.Parameter(result)

        if self.multi_target:
            self.b_decs = nn.ParameterList()
            self.W_skips = nn.ParameterList()
            for _ in range(cfg.n_targets):
                self.b_decs.append(create_bias())
                self.W_skips.append(create_W_skip())
        else:
            self.b_dec = create_bias()
            self.W_skip = create_W_skip()

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
        mesh: Optional[DeviceMesh] = None,
    ) -> "SparseCoder":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = SparseCoderConfig.from_dict(cfg_dict, drop_extra_fields=True)

        sae = SparseCoder(d_in, cfg, device=device, decoder=decoder, mesh=mesh)
        sae.load_state(path, strict=decoder)
        return sae

    def load_state(self, path: os.PathLike, strict: bool = True):
        filename = str(Path(path) / "sae.safetensors")
        if self.mesh is None:
            load_model(
                model=self,
                filename=filename,
                device=str(self.device),
                # TODO: Maybe be more fine-grained about this in the future?
                strict=strict,
            )
        else:
            torch.distributed.barrier()
            state_dict = load_sharded(
                filename,
                self.state_dict(),
                self.mesh,
            )
            torch.distributed.barrier()
            self.load_state_dict(state_dict, strict=strict)
            torch.distributed.barrier()

    def save_to_disk(self, path: Path | str):
        torch.distributed.barrier()
        path = Path(path)
        if (
            not torch.distributed.is_initialized()
        ) or torch.distributed.get_rank() == 0:
            path.mkdir(parents=True, exist_ok=True)

        filename = str(path / "sae.safetensors")
        if save_sharded(self.state_dict(), filename, mesh=self.mesh):
            with open(path / "cfg.json", "w") as f:
                json.dump(
                    {
                        **self.cfg.to_dict(),
                        "d_in": self.d_in,
                    },
                    f,
                )
        torch.distributed.barrier()

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

        return fused_encoder(
            x, self.encoder.weight, self.encoder.bias, self.cfg.k, self.cfg.activation
        )

    def decode(
        self,
        top_acts: Tensor | dtensor.DTensor,
        top_indices: Tensor | dtensor.DTensor,
        index: int = 0,
    ) -> Tensor:
        W_dec = self.W_decs[index] if self.multi_target else self.W_dec
        b_dec = self.b_decs[index] if self.multi_target else self.b_dec

        assert W_dec is not None, "Decoder weight was not initialized."

        y = decoder_impl(top_indices, top_acts.to(self.dtype), W_dec)
        return y + b_dec

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
        return_mid_decoder: bool = False,
    ) -> ForwardOutput | MidDecoder:
        top_acts, top_indices, pre_acts = self.encode(x)
        if self.multi_target:
            pre_acts = None

        mid_decoder = MidDecoder(self, x, top_acts, top_indices, dead_mask, pre_acts)
        if self.multi_target or return_mid_decoder:
            return mid_decoder
        else:
            return mid_decoder(0, y)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        for W_dec in self.W_decs if self.multi_target else (self.W_dec,):
            assert W_dec is not None, "Decoder weight was not initialized."

            eps = torch.finfo(W_dec.dtype).eps
            norm = torch.norm(W_dec.data, dim=1, keepdim=True)
            W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        for W_dec in self.W_decs if self.multi_target else (self.W_dec,):
            assert W_dec is not None, "Decoder weight was not initialized."
            assert W_dec.grad is not None  # keep pyright happy

            parallel_component = einops.einsum(
                W_dec.grad,
                W_dec.data,
                "d_sae d_in, d_sae d_in -> d_sae",
            )
            W_dec.grad -= einops.einsum(
                parallel_component,
                W_dec.data,
                "d_sae, d_sae d_in -> d_sae d_in",
            )


# Allow for alternate naming conventions
Sae = SparseCoder
