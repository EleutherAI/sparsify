from contextlib import contextmanager
from functools import partial
from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel

from sparsify import SparseCoder


def sparsify_forward(sparse_model: SparseCoder, input: Tensor, device: str) -> Tensor:
    """Helper method to process an input through a sparse model.
    Handles flattening and unflattening."""
    unflattened = [input.shape[0], input.shape[1]]

    return (
        sparse_model.forward(input.flatten(0, 1))
        .sae_out.unflatten(dim=0, sizes=unflattened)
        .to(device)
    )


@contextmanager
def edit_for_generation(
    model: PreTrainedModel,
    hookpoints: list[str],
    sparse_models: dict[str, SparseCoder],
    device="cuda",
):
    """
    Context manager that splices a sparse model
    into a base model.

    Args:
        model: The transformer model to hook
        hookpoints: List of hookpoints to edit
        sparse_models: Dictionary of sparse models to use for editing
        device: Device to use for editing

    Yields:
        None
    """
    handles = []

    def create_edit_hook(hookpoint: str):
        def hook_fn(
            module: nn.Module, input: Any, output: Tensor
        ) -> Tensor | tuple[Tensor, ...]:
            tensor_input = input[0] if isinstance(input, tuple) else input

            if isinstance(output, tuple):
                sparse_forward_input = (
                    tensor_input
                    if sparse_models[hookpoint].cfg.transcode
                    else output[0]
                )
                edited_tensor = sparsify_forward(
                    sparse_models[hookpoint], sparse_forward_input, device
                )
                return (edited_tensor, *output[1:])
            else:
                sparse_forward_input = (
                    tensor_input if sparse_models[hookpoint].cfg.transcode else output
                )
                return sparsify_forward(
                    sparse_models[hookpoint], sparse_forward_input, device
                )

        return hook_fn

    for name, module in model.named_modules():
        if name in hookpoints:
            handle = module.register_forward_hook(create_edit_hook(name))
            handles.append(handle)
    try:
        yield None
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def edit_with_mse(
    model: PreTrainedModel,
    hookpoints: list[str],
    sparse_models: dict[str, SparseCoder],
    device="cuda",
):
    """
    Context manager that temporarily hooks a model, edits the forward pass,
    and computes reconstruction MSE loss.

    Args:
        model: The transformer model to hook
        hookpoints: List of hookpoints to edit
        sparse_models: Dictionary of sparse models to use for editing
        device: Device to use for editing

    Yields:
        Dictionary mapping hookpoints to their reconstruction MSE loss
    """
    handles = []
    mses = {}

    def create_edit_hook(hookpoint: str):
        def hook_fn(
            module: nn.Module, input: Any, output: Tensor
        ) -> Tensor | tuple[Tensor, ...]:
            tensor_input = input[0] if isinstance(input, tuple) else input

            if isinstance(output, tuple):
                sparse_forward_input = (
                    tensor_input
                    if sparse_models[hookpoint].cfg.transcode
                    else output[0]
                )
                encoding = sparsify_forward(
                    sparse_models[hookpoint], sparse_forward_input, device
                )
                mses[hookpoint] = F.mse_loss(output[0], encoding)
                return (encoding, *output[1:])
            else:
                sparse_forward_input = (
                    tensor_input if sparse_models[hookpoint].cfg.transcode else output
                )
                encoding = sparsify_forward(
                    sparse_models[hookpoint], sparse_forward_input, device
                )
                mses[hookpoint] = F.mse_loss(output, encoding)
                return encoding

        return hook_fn

    for name, module in model.named_modules():
        if name in hookpoints:
            handle = module.register_forward_hook(create_edit_hook(name))
            handles.append(handle)
    try:
        yield mses
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def collect_activations(
    model: PreTrainedModel, hookpoints: list[str], input_acts: bool = False
):
    """
    Context manager that hooks a model and collects activations.
    An activation tensor is produced for each batch processed and stored
    in added to a list for that hookpoint in the activations dictionary.

    Args:
        model: The transformer model to hook
        hookpoints: List of hookpoints to collect activations from
        input_acts: Whether to collect input activations or output activations

    Yields:
        Dictionary mapping hookpoints to their collected activations
    """
    activations = {}
    handles = []

    def create_input_hook(hookpoint: str):
        def input_hook(module: nn.Module, input: Any, output: Any) -> None:
            if isinstance(input, tuple):
                activations[hookpoint] = input[0]
            else:
                activations[hookpoint] = input

        return input_hook

    def create_output_hook(hookpoint: str):
        def output_hook(module: nn.Module, input: Any, output: Any) -> None:
            if isinstance(output, tuple):
                activations[hookpoint] = output[0]
            else:
                activations[hookpoint] = output

        return output_hook

    for name, module in model.named_modules():
        if name in hookpoints:
            hook = create_input_hook(name) if input_acts else create_output_hook(name)
            handle = module.register_forward_hook(hook)
            handles.append(handle)

    try:
        yield activations
    finally:
        for handle in handles:
            handle.remove()


collect_input_activations = partial(collect_activations, input_acts=True)
collect_output_activations = partial(collect_activations, input_acts=False)
