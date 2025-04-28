from typing import Literal, NamedTuple

import torch
import torch.distributed.tensor as dtensor
import torch.nn.functional as F
from torch.distributed._functional_collectives import permute_tensor

# from .xformers import embedding_bag_bw_rev_indices


class EncoderOutput(NamedTuple):
    top_acts: torch.Tensor
    """Activations of the top-k latents."""

    top_indices: torch.Tensor
    """Indices of the top-k features."""

    pre_acts: torch.Tensor
    """Activations before the top-k selection."""


class FusedEncoder(torch.autograd.Function):
    @staticmethod
    @torch.compile(disable=True)
    def forward(
        ctx, input, weight, bias, k: int, activation: Literal["groupmax", "topk"]
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
            if isinstance(preacts, dtensor.DTensor):
                # TODO leo gao's ring topk
                mesh = preacts.device_mesh
                local_values, local_indices = preacts.to_local().topk(
                    k, dim=1, sorted=False
                )
                tp_size = mesh.shape[1]
                src_dst = [(i + 1) % tp_size for i in range(tp_size)]
                result_values, result_indices = local_values, local_indices
                local_values, local_indices = (
                    local_values.T.contiguous(),
                    local_indices.T.contiguous(),
                )
                # for some reason, this is necessary
                # (sigterm without)
                torch.distributed.barrier()
                for _ in range(tp_size):
                    # TODO non-permute op
                    next_local_values = permute_tensor(
                        local_values.ravel(),
                        src_dst,
                        mesh["tp"],
                    )
                    next_local_indices = permute_tensor(
                        local_indices.ravel(), src_dst, mesh["tp"]
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
            values, indices = preacts.unflatten(-1, (k, -1)).max(dim=-1)

            # torch.max gives us indices into each group, but we want indices into the
            # flattened tensor. Add the offsets to get the correct indices.
            num_latents = preacts.shape[1]
            offsets = torch.arange(
                0, num_latents, num_latents // k, device=preacts.device
            )
            indices = offsets + indices
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Save tensors needed for the backward pass
        ctx.save_for_backward(input, weight, bias, indices)
        ctx.k = k
        return values, indices, preacts

    @staticmethod
    @torch.compile(disable=True)
    @torch.no_grad()
    def backward(ctx, grad_values, grad_indices, grad_preacts):
        input, weight, bias, indices = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # --- Grad w.r.t. input ---
        if ctx.needs_input_grad[0]:
            grad_input = F.embedding_bag(
                indices,
                weight,
                mode="sum",
                per_sample_weights=grad_values.type_as(weight),
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
                dp_size, tp_size = mesh.shape
                local_grad_weight = grad_weight.to_local()
                # local_weight = weight.to_local()
                local_indices = indices.flatten().to_local()
                local_values = grad_values.flatten().to_local()
                local_input = input.flatten().to_local()
                for _ in range(dp_size):
                    # TODO filtering indices
                    # TODO use COO backward kernel?

                    local_contributions = local_values.view(
                        -1, ctx.k, 1
                    ) * local_input.view(-1, 1, D)
                    # perform local update
                    # TODO filtering indices
                    local_grad_weight.index_add_(
                        0,
                        local_indices,
                        local_contributions.reshape(-1, D).type_as(weight),
                    )

                    # example_indices = torch.arange(len(local_indices),
                    # device=local_indices.device) // ctx.k
                    # result = triton_coo_sparse_dense_matmul(
                    #     torch.stack([example_indices, local_indices]),
                    #     local_values.float(),
                    #     local_input.view(-1, D).float(),
                    #     N=local_weight.shape[0],
                    #     # out=local_grad_weight,
                    # )
                    # local_grad_weight += result * 0

                    # embedding_bag_bw_rev_indices(
                    #     local_indices.view(-1, ctx.k),
                    #     local_weight,
                    #     local_values.view(-1, ctx.k),
                    #     local_input,
                    #     local_grad_weight,
                    # )

                    # rotate indices/inputs/values
                    src_dst = [(i + 1) % dp_size for i in range(dp_size)]
                    local_indices = permute_tensor(local_indices, src_dst, mesh["dp"])
                    local_values = permute_tensor(local_values, src_dst, mesh["dp"])
                    local_input = permute_tensor(local_input, src_dst, mesh["dp"])

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

        # The k parameter is an int, so return None for its gradient.
        return grad_input, grad_weight, grad_bias, None, None


def fused_encoder(
    input,
    weight,
    bias,
    k: int,
    activation: Literal["groupmax", "topk"],
) -> EncoderOutput:
    """
    Convenience wrapper that performs an nn.Linear followed by `activation` with
    a backward pass optimized using index_add.
    """
    return EncoderOutput(
        *FusedEncoder.apply(input, weight, bias, k, activation)  # type: ignore
    )
