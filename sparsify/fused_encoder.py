from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F

# Upper bound on the temporary built inside the weight-gradient accumulation. The
# unchunked form allocates N * k * D elements at once, which for a wide model and
# a large k is the single biggest tensor in the backward pass -- at N=8192, k=32,
# D=1024 it is 1 GiB. Consuming it in row-blocks bounds that without changing the
# result.
BACKWARD_CHUNK_BYTES = 64 * 1024 * 1024


class EncoderOutput(NamedTuple):
    top_acts: torch.Tensor
    """Activations of the top-k latents."""

    top_indices: torch.Tensor
    """Indices of the top-k features."""

    pre_acts: torch.Tensor
    """Activations before the top-k selection."""


class FusedEncoder(torch.autograd.Function):
    @staticmethod
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
            # Each top-k location contributes `grad_values * input` to its row of
            # grad_weight. Materialising all of them at once costs N * k * D
            # elements, so walk the batch in blocks sized to BACKWARD_CHUNK_BYTES
            # and fold each block in as it is built. The arithmetic is unchanged;
            # only the accumulation order inside index_add_ differs, and that was
            # already unspecified on CUDA.
            N, k = grad_values.shape
            D = input.shape[-1]
            itemsize = torch.promote_types(grad_values.dtype, input.dtype).itemsize
            rows = max(1, BACKWARD_CHUNK_BYTES // max(1, k * D * itemsize))

            for start in range(0, N, rows):
                stop = min(start + rows, N)
                block = grad_values[start:stop].unsqueeze(2) * input[
                    start:stop
                ].unsqueeze(1)
                grad_weight.index_add_(
                    0,
                    indices[start:stop].flatten(),
                    block.reshape(-1, D).type_as(weight),
                )

        # --- Grad w.r.t. bias ---
        if bias is not None and ctx.needs_input_grad[2]:
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
