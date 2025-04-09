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


@torch.inference_mode()
@torch.compile
def find_sigmoid_threshold(logits, target_sum, max_iter=7, tol=1e-6):
    """
    Find threshold x such that when added to logits, the sum of sigmoid
    probabilities equals target_sum Uses Newton's method for fast convergence

    Args:
        logits: tensor of shape (batch_size, num_classes) - raw logits before sigmoid
        target_sum: desired sum of probabilities
            (scalar or tensor of shape [batch_size])
        max_iter: maximum number of Newton iterations
        tol: tolerance for convergence

    Returns:
        Tensor of shape [batch_size] containing threshold value for each sample
    """
    batch_size = logits.shape[0]

    # If target_sum is a scalar, expand it to match batch_size
    if isinstance(target_sum, (int, float)) or (
        isinstance(target_sum, torch.Tensor) and target_sum.dim() == 0
    ):
        target_sum = torch.ones(batch_size, device=logits.device) * target_sum

    # Initialize x with zeros
    x = torch.zeros(batch_size, device=logits.device)

    for _ in range(max_iter):
        # Calculate sigmoid values for current x
        sig = torch.sigmoid(logits + x.unsqueeze(1))

        # Calculate function value: sum(sigmoid(logits + x)) - target_sum
        f_x = torch.sum(sig, dim=1) - target_sum

        # Check for convergence
        if torch.max(torch.abs(f_x)) < tol:
            break

        # Calculate derivative: sum(sigmoid(logits + x) * (1 - sigmoid(logits + x)))
        df_x = torch.sum(sig * (1 - sig), dim=1)

        # Newton update: x = x - f(x)/f'(x)
        x = x - f_x / df_x

    return x


class BinaryFusedEncoder(torch.autograd.Function):
    @staticmethod
    @torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
    )
    def forward(
        ctx,
        input,
        weight,
        bias,
        k: int,
        threshold: Tensor,
        ste_activation: Literal["sigmoid", "rectangle"] = "sigmoid",
        ste_thresh_est: bool = False,
        ste_temperature: float = 1.0,
    ):
        """
        input:  (N, D)
        weight: (M, D)
        bias:   (M,)
        k:      int (number of top elements to select along dim=1)
        """
        preacts = F.linear(input, weight, bias)  #  F.relu(
        # Get top-k values and indices for each row

        threshold_ = threshold
        if ste_thresh_est:
            threshold_ = torch.zeros_like(preacts)
            threshold_ += -find_sigmoid_threshold(preacts, k).unsqueeze(1).detach()

        topk_values, indices = torch.topk(preacts - threshold_, k, dim=1, sorted=False)
        # values = (topk_values > 0).float()
        values = torch.ones_like(topk_values).float()

        # Save tensors needed for the backward pass
        ctx.save_for_backward(
            input, weight, bias, values, indices, threshold, topk_values
        )
        ctx.k = k
        ctx.ste_activation = ste_activation
        ctx.ste_temperature = ste_temperature
        return values, indices, preacts

    @staticmethod
    @torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
    )
    def backward(ctx, grad_values, grad_indices, grad_preacts):  # bandwidth: float = 2.
        input, weight, bias, values, indices, threshold, topk_values = ctx.saved_tensors
        ste_temperature = ctx.ste_temperature
        grad_input = grad_weight = grad_bias = None

        grad_threshold = torch.zeros_like(threshold)
        if ctx.ste_activation == "sigmoid":
            ste_grad_wrt_vals = torch.sigmoid(topk_values / ste_temperature)
            sigmoid_grad_wrt_vals = ste_grad_wrt_vals * (1 - ste_grad_wrt_vals)

            # dL/dvals = dL/dsigmoid * dsigmoid/dvals
            grad_values = grad_values * sigmoid_grad_wrt_vals

            # dL/dthreshold = dL/dvals * dvals/dthreshold
            grad_threshold.index_add_(0, indices.flatten(), -grad_values.flatten())
        elif ctx.ste_activation == "rectangle":
            # This is essentially always true for the first condition in rectangle
            ste_grad_wrt_vals = rectangle(topk_values / ste_temperature)
            grad_values = grad_values * ste_grad_wrt_vals

            grad_threshold_vals = (
                -(threshold[indices].flatten() / ste_temperature)
                * grad_values.flatten()
            )
            grad_threshold.index_add_(0, indices.flatten(), grad_threshold_vals)
        elif ctx.ste_activation == "re":
            o = ste_temperature
            ste_grad_wrt_vals = 1 / o * torch.abs(topk_values).pow((1 - o) / o)
            grad_values = grad_values * ste_grad_wrt_vals
            grad_threshold_vals = (
                -(threshold[indices].flatten() / o) * grad_values.flatten()
            )
            grad_threshold.index_add_(0, indices.flatten(), grad_threshold_vals)
        elif ctx.ste_activation == "identity":
            grad_threshold_vals = (
                -grad_values * threshold[indices]
            ).flatten() / ste_temperature
            grad_threshold.index_add_(0, indices.flatten(), grad_threshold_vals)

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
        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            grad_threshold,
            None,
            None,
            None,
        )


class BinaryGroupMaxFusedEncoder(torch.autograd.Function):
    @staticmethod
    @torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
    )
    def forward(
        ctx,
        input,
        weight,
        bias,
        k: int,
        threshold: Tensor,
        ste_activation: Literal["sigmoid", "rectangle"] = "sigmoid",
        ste_temperature: float = 1.0,
    ):
        """
        input:  (N, D)
        weight: (M, D)
        bias:   (M,)
        k:      int (number of top elements to select along dim=1)
        """
        preacts = F.linear(input, weight, bias)  #  F.relu(
        # Get top-k values and indices for each row

        # Group max
        topk_values, indices = (preacts - threshold).unflatten(-1, (k, -1)).max(dim=-1)
        num_latents = preacts.shape[1]
        offsets = torch.arange(0, num_latents, num_latents // k, device=preacts.device)
        indices = offsets + indices

        values = (topk_values > 0).float()

        # Save tensors needed for the backward pass
        ctx.save_for_backward(
            input, weight, bias, values, indices, threshold, topk_values
        )
        ctx.k = k
        ctx.ste_activation = ste_activation
        ctx.ste_temperature = ste_temperature
        return values, indices, preacts

    @staticmethod
    @torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
    )
    def backward(ctx, grad_values, grad_indices, grad_preacts):  # bandwidth: float = 2.
        input, weight, bias, values, indices, threshold, topk_values = ctx.saved_tensors
        ste_temperature = ctx.ste_temperature
        grad_input = grad_weight = grad_bias = None

        grad_threshold = torch.zeros_like(threshold)
        if ctx.ste_activation == "sigmoid":
            ste_grad_wrt_vals = torch.sigmoid(topk_values / ste_temperature)
            sigmoid_grad_wrt_vals = ste_grad_wrt_vals * (1 - ste_grad_wrt_vals)

            # dL/dvals = dL/dsigmoid * dsigmoid/dvals
            grad_values = grad_values * sigmoid_grad_wrt_vals

            # dL/dthreshold = dL/dvals * dvals/dthreshold
            grad_threshold.index_add_(0, indices.flatten(), -grad_values.flatten())
        elif ctx.ste_activation == "rectangle":
            # This is essentially always true for the first condition in rectangle
            ste_grad_wrt_vals = rectangle(topk_values / ste_temperature)
            grad_values = grad_values * ste_grad_wrt_vals

            grad_threshold_vals = (
                -(threshold[indices].flatten() / ste_temperature)
                * grad_values.flatten()
            )
            grad_threshold.index_add_(0, indices.flatten(), grad_threshold_vals)

        elif ctx.ste_activation == "re":
            o = ste_temperature
            ste_grad_wrt_vals = 1 / o * torch.abs(topk_values).pow((1 - o) / o)
            grad_values = grad_values * ste_grad_wrt_vals
            grad_threshold_vals = (
                -(threshold[indices].flatten() / o) * grad_values.flatten()
            )
            grad_threshold.index_add_(0, indices.flatten(), grad_threshold_vals)

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
        return grad_input, grad_weight, grad_bias, None, grad_threshold, None, None


class FusedEncoder(torch.autograd.Function):
    @staticmethod
    @torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
    )
    def forward(
        ctx,
        input,
        weight,
        bias,
        k: int,
        activation: Literal["groupmax", "topk", "topk_binary"],
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

    @torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()
    )
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
    input, weight, bias, k: int, activation: Literal["groupmax", "topk", "topk_binary"]
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
    ste_activation: Literal["sigmoid", "rectangle"] = "sigmoid",
    ste_temperature: float = 1.0,
    ste_thresh_est: bool = False,
) -> EncoderOutput:
    return EncoderOutput(
        *BinaryFusedEncoder.apply(
            input,
            weight,
            bias,
            k,
            threshold,
            ste_activation,
            ste_temperature,
            ste_thresh_est,
        )  # type: ignore
    )
