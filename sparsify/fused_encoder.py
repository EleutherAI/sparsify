from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

class EncoderOutput(NamedTuple):
    top_acts: torch.Tensor
    """Activations of the top-k latents."""

    top_indices: torch.Tensor
    """Indices of the top-k features."""

    pre_acts: torch.Tensor
    """Activations before the top-k selection."""


def rectangle(x):
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)

class BinaryFusedEncoder(torch.autograd.Function):
    @staticmethod
    @torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
    )
    def forward(
        ctx, input, weight, bias, k: int, threshold: Tensor, ste_activation: Literal["sigmoid", "rectangle"] = "sigmoid"
    ):
        """
        input:  (N, D)
        weight: (M, D)
        bias:   (M,)
        k:      int (number of top elements to select along dim=1)
        """
        preacts = F.relu(F.linear(input, weight, bias))

        # Get top-k values and indices for each row
        
        topk_values, indices = torch.topk(preacts, k, dim=1, sorted=False)
        values = (topk_values > threshold[indices]).float()
       
        # Save tensors needed for the backward pass
        ctx.save_for_backward(input, weight, bias, values, indices, threshold, topk_values)
        ctx.k = k
        ctx.ste_activation = ste_activation
        return values, indices, preacts

    @staticmethod
    @torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
    )
    def backward(ctx, grad_values, grad_indices, grad_preacts): # bandwidth: float = 2.
        input, weight, bias, values, indices, threshold, topk_values = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.ste_activation == "sigmoid":
            ste_vals = torch.sigmoid(topk_values)
            sigmoid_grad_wrt_vals = ste_vals * (1 - ste_vals) 
            
            # dL/dvals = dL/dsigmoid * dsigmoid/dvals
            grad_values = grad_values * sigmoid_grad_wrt_vals

            # dL/dthreshold = dL/dvals * dvals/dthreshold
            grad_threshold = torch.zeros_like(threshold)
            grad_threshold.index_add_(0, indices.flatten(), -grad_values.flatten())
        elif ctx.ste_activation == "rectangle":
            ste_vals = rectangle(topk_values)
            rect_grad_wrt_vals = ste_vals
            grad_values = grad_values * rect_grad_wrt_vals

            grad_threshold = (
                -(threshold / 2.)
                * rectangle((topk_values) / 2.)
                * grad_values
            )

        # import time

        # start = time.time()
        # for _ in range(1000):
        #     grad_threshold = torch.zeros_like(threshold)
        
        # print(time.time() - start)

        # start = time.time()
        # for _ in range(1000):
        #     flat_indices = indices.flatten()
        #     grad_values_flat = grad_values.flatten()

        #     grad_threshold = torch.zeros_like(threshold).scatter_add_(0, flat_indices, threshold_grad_contributions)

        #     grad_threshold = torch.zeros_like(threshold)
        #     grad_threshold.index_add_(0, flat_indices, grad_values_flat)

        # print(time.time() - start)
        # breakpoint()

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

        # --- Grad w.r.t. bias ---
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.zeros_like(bias)
            grad_bias.index_add_(
                0, indices.flatten(), grad_values.flatten().type_as(bias)
            )

        # The k parameter is an int, so return None for its gradient.
        return grad_input, grad_weight, grad_bias, None, grad_threshold



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
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Save tensors needed for the backward pass
        ctx.save_for_backward(input, weight, bias, values, indices)
        ctx.k = k
        ctx.activation = activation
        return values, indices, preacts

    @staticmethod
    def backward(ctx, grad_values, grad_indices, grad_preacts):
        input, weight, bias, values, indices = ctx.saved_tensors
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
            # Compute contributions from each top-k element:
            # computed as grad_values * input for each top-k location.
            contributions = grad_values.unsqueeze(2) * input.unsqueeze(1)
            _, _, D = contributions.shape
            # Flatten contributions to shape (N*k, D)
            contributions = contributions.reshape(-1, D)

            # Accumulate contributions into the correct rows of grad_weight.
            grad_weight.index_add_(0, indices.flatten(), contributions.type_as(weight))

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
    activation: Literal["groupmax", "topk", "topk_binary"]
) -> EncoderOutput:
    """
    Convenience wrapper that performs an nn.Linear followed by `activation` with
    a backward pass optimized using index_add.
    """
    return EncoderOutput(
        *FusedEncoder.apply(input, weight, bias, k, activation)  # type: ignore
    )

def binary_fused_encoder(
    input,
    weight,
    bias,
    k: int,
    threshold: Tensor,
) -> EncoderOutput:
    return EncoderOutput(
        *BinaryFusedEncoder.apply(input, weight, bias, k, threshold)  # type: ignore
    )
