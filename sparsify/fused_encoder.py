from typing import Literal, NamedTuple

import torch
import torch.distributed.tensor as dtensor
import torch.nn.functional as F
from torch.distributed._functional_collectives import permute_tensor

from .kernels import triton_sparse_transpose_dense_matmul
from .utils import decoder_impl


class EncoderOutput(NamedTuple):
    top_acts: torch.Tensor
    """Activations of the top-k latents."""

    top_indices: torch.Tensor
    """Indices of the top-k features."""

    pre_acts: torch.Tensor
    """Activations before the top-k selection."""


CONTRIB_BATCH_SIZE = 4096


class FusedEncoder(torch.autograd.Function):
    @staticmethod
    @torch.compile
    def forward(
        ctx,
        input,
        weight,
        bias,
        post_enc,
        k: int,
        activation: Literal["groupmax", "topk"],
    ):
        """
        input:  (N, D)
        weight: (M, D)
        bias:   (M,)
        k:      int (number of top elements to select along dim=1)
        """
        preacts = F.relu(F.linear(input, weight, bias))

        # Get top-k values and indices for each row
        if activation == "topk":
            if (
                isinstance(preacts, dtensor.DTensor)
                and preacts.device_mesh.shape[1] == 1
            ):
                mesh = preacts.device_mesh
                local_acts = preacts.to_local()
                local_values, local_indices = local_acts.topk(k, dim=1, sorted=False)
                values = dtensor.DTensor.from_local(
                    local_values,
                    mesh,
                    (dtensor.Shard(0), dtensor.Replicate()),
                )
                indices = dtensor.DTensor.from_local(
                    local_indices,
                    mesh,
                    (dtensor.Shard(0), dtensor.Replicate()),
                )
            elif isinstance(preacts, dtensor.DTensor):
                # TODO leo gao's ring topk
                mesh = preacts.device_mesh
                local_acts = preacts.to_local()
                local_values, local_indices = local_acts.topk(k, dim=1, sorted=False)
                local_indices += mesh.get_local_rank(1) * local_acts.shape[1]
                tp_size = mesh.shape[1]
                tp_group = mesh.get_group(1)
                src_dst = [(i + 1) % tp_size for i in range(tp_size)]
                result_values, result_indices = local_values, local_indices
                local_values, local_indices = (
                    local_values.T.flatten(),
                    local_indices.T.flatten(),
                )
                for _ in range(tp_size):
                    # TODO non-permute op
                    next_local_values = permute_tensor(
                        local_values,
                        src_dst,
                        tp_group,
                    )
                    next_local_indices = permute_tensor(
                        local_indices, src_dst, tp_group
                    )
                    rotated_values = next_local_values.view(result_values.shape[::-1]).T
                    rotated_indices = next_local_indices.view(
                        result_indices.shape[::-1]
                    ).T
                    # TODO faster merge
                    combined_values = torch.cat((result_values, rotated_values), dim=1)
                    combined_indices = torch.cat(
                        (result_indices, rotated_indices), dim=1
                    )
                    result_values, result_indices_ = combined_values.topk(
                        k, dim=1, sorted=False
                    )
                    result_indices = torch.gather(combined_indices, 1, result_indices_)
                    local_values = next_local_values
                    local_indices = next_local_indices

                values = dtensor.DTensor.from_local(
                    result_values,
                    mesh,
                    (dtensor.Shard(0), dtensor.Replicate()),
                )
                indices = dtensor.DTensor.from_local(
                    result_indices,
                    mesh,
                    (dtensor.Shard(0), dtensor.Replicate()),
                )
            else:
                values, indices = torch.topk(preacts, k, dim=-1, sorted=False)
        elif activation == "groupmax":
            if isinstance(preacts, dtensor.DTensor):
                mesh = preacts.device_mesh
                local_acts = preacts.to_local()
                assert k % mesh.shape[1] == 0
                local_k = k // mesh.shape[1]
                local_values, local_indices = local_acts.unflatten(
                    -1, (local_k, -1)
                ).max(dim=-1)
                offsets = torch.arange(
                    0,
                    local_acts.shape[1],
                    local_acts.shape[1] // local_k,
                    device=preacts.device,
                )
                mesh_offset = mesh.get_local_rank(1) * local_k
                indices = mesh_offset + offsets + local_indices
                values = local_values
                values = dtensor.DTensor.from_local(
                    values,
                    mesh,
                    (dtensor.Shard(0), dtensor.Shard(1)),
                )
                indices = dtensor.DTensor.from_local(
                    indices,
                    mesh,
                    (dtensor.Shard(0), dtensor.Shard(1)),
                )
                values = values.redistribute(
                    mesh, (dtensor.Shard(0), dtensor.Replicate())
                )
                indices = indices.redistribute(
                    mesh, (dtensor.Shard(0), dtensor.Replicate())
                )
            else:
                num_latents = preacts.shape[1]
                values, indices = preacts.unflatten(-1, (k, -1)).max(dim=-1)
                offsets = torch.arange(
                    0, num_latents, num_latents // k, device=preacts.device
                )
                indices = offsets + indices
        else:
            raise ValueError(f"Unknown activation: {activation}")

        values = values + post_enc[indices]

        # Save tensors needed for the backward pass
        ctx.save_for_backward(input, weight, bias, indices)
        ctx.k = k
        ctx.activation = activation
        return values, indices, preacts

    @staticmethod
    @torch.compile
    @torch.no_grad()
    def backward(ctx, grad_values, grad_indices, grad_preacts):
        input, weight, bias, indices = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        activation = ctx.activation

        # --- Grad w.r.t. input ---
        if ctx.needs_input_grad[0]:
            grad_input = decoder_impl(
                indices,
                grad_values.type_as(weight),
                weight,
            )

        if isinstance(grad_values, dtensor.DTensor):
            mesh = grad_values.device_mesh
            local_size = weight.to_local().shape[0]
            start_feature = mesh.get_local_rank(1) * local_size
            end_feature = start_feature + local_size

        # --- Grad w.r.t. bias ---
        if bias is not None and ctx.needs_input_grad[2]:
            if isinstance(bias, dtensor.DTensor):
                mesh = bias.device_mesh
                grad_bias = torch.zeros_like(bias.to_local())
                all_indices = indices.flatten()
                all_indices = all_indices.redistribute(
                    mesh, (dtensor.Replicate(), dtensor.Replicate())
                ).to_local()
                all_values = grad_values.flatten()
                all_values = all_values.redistribute(
                    mesh, (dtensor.Replicate(), dtensor.Replicate())
                ).to_local()

                # TODO bespoke all-to-all gradient communication
                # likely won't be necessary, the encoder backward pass is fast
                mask = (all_indices >= start_feature) & (all_indices < end_feature)
                all_indices = all_indices[mask] - start_feature
                all_values = all_values[mask]

                grad_bias.index_add_(
                    0, all_indices, all_values.type_as(bias.to_local())
                )
                grad_bias = dtensor.DTensor.from_local(
                    grad_bias, mesh, (dtensor.Replicate(), dtensor.Shard(0))
                )
            else:
                grad_bias = torch.zeros_like(bias)
                grad_bias.index_add_(
                    0, indices.flatten(), grad_values.flatten().type_as(bias)
                )

        # --- Grad w.r.t. weight ---
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight)

            # Accumulate contributions into the correct rows of grad_weight.
            _, D = input.shape
            if not isinstance(grad_weight, dtensor.DTensor):
                # Compute contributions from each top-k element:
                # computed as grad_values * input for each top-k location.
                contributions = grad_values.unsqueeze(2) * input.unsqueeze(1)
                # Flatten contributions to shape (N*k, D)
                contributions = contributions.reshape(-1, D)
                # print(grad_weight)  # (M, D); TP sharded along dim=0
                # print(indices.flatten())  # (N*k); DP sharded along dim=0
                # print(contributions)  # (N*k, D); DP sharded along dim=0
                grad_weight.index_add_(
                    0, indices.flatten(), contributions.type_as(weight)
                )
            else:
                mesh = grad_weight.device_mesh
                local_grad_weight = grad_weight.to_local()
                gathered_input = input.redistribute(
                    mesh, (dtensor.Replicate(), dtensor.Replicate())
                ).to_local()
                if activation == "groupmax":
                    indices = indices.redistribute(
                        mesh, (dtensor.Replicate(), dtensor.Shard(1))
                    ).to_local()
                    values = grad_values.redistribute(
                        mesh, (dtensor.Replicate(), dtensor.Shard(1))
                    ).to_local()
                    local_k = ctx.k // mesh.shape[1]
                    start_f = mesh.get_local_rank(1) * local_k
                    indices = indices - start_f
                else:
                    gathered_indices = indices.redistribute(
                        mesh, (dtensor.Replicate(), dtensor.Replicate())
                    ).to_local()
                    gathered_values = grad_values.redistribute(
                        mesh, (dtensor.Replicate(), dtensor.Replicate())
                    ).to_local()

                    indices = gathered_indices.view(-1, ctx.k)
                    values = gathered_values.view(-1, ctx.k)

                    mask = (indices >= start_feature) & (indices < end_feature)
                    values *= mask.type_as(values)
                    indices = (indices - start_feature).clamp(
                        0, local_grad_weight.shape[0] - 1
                    )
                local_grad_weight += triton_sparse_transpose_dense_matmul(
                    indices,
                    values.float(),
                    gathered_input,
                    N=local_grad_weight.shape[0],
                )

        # The k parameter is an int, so return None for its gradient.
        return grad_input, grad_weight, grad_bias, None, None, None


def fused_encoder(
    input,
    weight,
    bias,
    post_enc,
    k: int,
    activation: Literal["groupmax", "topk"],
) -> EncoderOutput:
    """
    Convenience wrapper that performs an nn.Linear followed by `activation` with
    a backward pass optimized using index_add.
    """
    return EncoderOutput(
        *FusedEncoder.apply(input, weight, bias, post_enc, k, activation)  # type: ignore
    )
