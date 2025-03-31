from optparse import Values
from typing import Literal, NamedTuple

from tokenizers import Encoding
import torch
import torch.nn.functional as F
from torch import Tensor
import lovely_tensors as lt
lt.monkey_patch()

class EncoderOutput(NamedTuple):
    top_acts: torch.Tensor
    """Activations of the top-k latents."""

    top_indices: torch.Tensor
    """Indices of the top-k features."""

    pre_acts: torch.Tensor
    """Activations before the top-k selection."""


# class SigmoidSTE(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         return (input > 0).float()

#     @staticmethod
#     def backward(ctx, grad_output):
#         (input,) = ctx.saved_tensors
#         sig = torch.sigmoid(input)
#         return grad_output * sig * (1 - sig)
    

# def sigmoid_ste(input) -> Tensor:
#     return SigmoidSTE.apply(input)  # type: ignore


class FusedEncoder(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input, weight, bias, k: int, activation: Literal["groupmax", "topk", "topk_binary"]
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
            values, indices = torch.topk(preacts, k, dim=1, sorted=False)
        elif activation == "groupmax":
            values, indices = preacts.unflatten(-1, (k, -1)).max(dim=-1)

            # torch.max gives us indices into each group, but we want indices into the
            # flattened tensor. Add the offsets to get the correct indices.
            num_latents = preacts.shape[1]
            offsets = torch.arange(
                0, num_latents, num_latents // k, device=preacts.device
            )
            indices = offsets + indices
        elif activation == "topk_binary":
            topk_values, indices = torch.topk(preacts, k, dim=1, sorted=False)
            values = (topk_values > 1).float()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Save tensors needed for the backward pass
        ctx.save_for_backward(input, weight, bias, indices, torch.sigmoid(topk_values))
        ctx.k = k
        ctx.activation = activation
        return values, indices, preacts

    @staticmethod
    def backward(ctx, grad_values, grad_indices, grad_preacts):
        input, weight, bias, indices, sigmoid_topk_values = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # Use STE to update grad_values for binary
        if ctx.activation == "topk_binary":
            grad_values = grad_values * sigmoid_topk_values * (1 - sigmoid_topk_values)

        # grad_values, the only non-zero gradient, is of shape (N, k)
        # Input is of shape (N, D)
        # We want to propagate the gradient to the preacts, which is of shape (N, D)
        # And the weight of the pre acts
        # And the bias of the preacts

        # When the preacts are binary, we want to use a sigmoid STE to avoid having the gradients be too large
        # when the activations before the binary threshold are much larger than 1 or much smaller than -1.

        # ie, if we have a preact of 10, and the naive gradient is negative, we want to use a STE to set the gradient to 0.

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
            # Compute contributions from each top-k element:
            # computed as grad_values * input for each top-k location.
            contributions = grad_values.unsqueeze(2) * input.unsqueeze(1)
            _, _, D = contributions.shape
            # Flatten contributions to shape (N*k, D)
            contributions = contributions.reshape(-1, D)

            # Accumulate contributions into the correct rows of grad_weight.
            grad_weight.index_add_(0, indices.flatten(), contributions.type_as(weight))
            print("grad weight", grad_weight)
        # --- Grad w.r.t. bias ---
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.zeros_like(bias)
            grad_bias.index_add_(
                0, indices.flatten(), grad_values.flatten().type_as(bias)
            )
            print("grad bias", grad_bias) 

        # The k parameter is an int, so return None for its gradient.
        return grad_input, grad_weight, grad_bias, None, None


def fused_encoder(
    input,
    weight,
    bias,
    k: int,
    activation: Literal["groupmax", "topk", "topk_binary"],
) -> EncoderOutput:
    """
    Convenience wrapper that performs an nn.Linear followed by `activation` with
    a backward pass optimized using index_add.
    """
    return EncoderOutput(
        *FusedEncoder.apply(input, weight, bias, k, activation)  # type: ignore
    )
