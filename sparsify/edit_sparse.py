from contextlib import contextmanager
from typing import Any

from matplotlib.pyplot import bone
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel
from sparsify import SparseCoder


@contextmanager
def edit(
    model: PreTrainedModel, hookpoints: list[str], sparse_models: dict[str, SparseCoder], transcode: bool = False
):
    """
    Context manager that temporarily hooks models and edits their forward pass.

    Args:
        model: The transformer model to hook
        hookpoints: List of hookpoints to edit

    Yields:
        None
    """
    handles = []

    def create_edit_hook(hookpoint: str, transcode: bool = False, device = "cuda"):
        def hook_fn(module: nn.Module, input: Any, output: Tensor) -> Tensor | None:
            if transcode:
                # Transcoder - maps input to output
                if isinstance(output, tuple):
                    output[0] = sparse_models[hookpoint].forward(input[0].flatten(0, 1)).sae_out.unflatten(
                            dim=0,
                            sizes=[input[0].shape[0], input[0].shape[1]]
                        ).to(device)
                else:
                    output = sparse_models[hookpoint].forward(input[0].flatten(0, 1)).sae_out.unflatten(
                            dim=0, sizes=[input[0].shape[0], input[0].shape[1]]
                        ).to(device)
            else:
                # SAE - maps output to output
                if isinstance(output, tuple):
                    output[0] = sparse_models[hookpoint].forward(output[0].flatten(0, 1)).sae_out.unflatten(
                        dim=0,
                        sizes=[output[0].shape[0], output[0].shape[1]]
                    ).to(device)
                else:
                    output = sparse_models[hookpoint].forward(output.flatten(0, 1)).sae_out.unflatten(
                        dim=0,
                        sizes=[output.shape[0], output.shape[1]]
                    ).to(device)
            return output
        return hook_fn

    for name, module in model.named_modules():
        if name in hookpoints:
            handle = module.register_forward_hook(create_edit_hook(name, transcode))
            handles.append(handle)
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def edit_with_mse(
    model: PreTrainedModel, hookpoints: list[str], sparse_models: dict[str, SparseCoder], transcode: bool = False
):
    """
    Context manager that temporarily hooks models and edits their forward pass.

    Args:
        model: The transformer model to hook
        hookpoints: List of hookpoints to edit

    Yields:
        None
    """
    handles = []
    mses = {}

    def create_edit_hook(hookpoint: str, transcode: bool = False, device = "cuda"):
        def hook_fn(module: nn.Module, input: Any, output: Tensor) -> Tensor | None:
            if transcode:
                if isinstance(output, tuple):
                    encoding = sparse_models[hookpoint].forward(input[0].flatten(0, 1)).sae_out.unflatten(
                            dim=0,
                            sizes=[input[0].shape[0], input[0].shape[1]]
                        ).to(device)
                    mses[hookpoint] = F.mse_loss(output[0], encoding)
                    output[0] = encoding
                else:
                    encoding = sparse_models[hookpoint].forward(input[0].flatten(0, 1)).sae_out.unflatten(
                            dim=0, sizes=[input[0].shape[0], input[0].shape[1]]
                        ).to(device)
                    mses[hookpoint] = F.mse_loss(output, encoding)
                    output = encoding
            else:
                if isinstance(output, tuple):
                    # SAE - maps output to output
                    encoding = sparse_models[hookpoint].forward(output[0].flatten(0, 1)).sae_out.unflatten(
                        dim=0,
                        sizes=[output[0].shape[0], output[0].shape[1]]
                    ).to(device)
                    mses[hookpoint] = F.mse_loss(output[0], encoding)
                    output[0] = encoding
                else:
                    # SAE - maps output to output
                    encoding = sparse_models[hookpoint].forward(output.flatten(0, 1)).sae_out.unflatten(
                        dim=0,
                        sizes=[output.shape[0], output.shape[1]]
                    ).to(device)
                    mses[hookpoint] = F.mse_loss(output, encoding)
                    output = encoding
            return output
        return hook_fn

    for name, module in model.named_modules():
        if name in hookpoints:
            handle = module.register_forward_hook(create_edit_hook(name, transcode))
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
    Context manager that temporarily hooks models and collects their activations.
    An activation tensor is produced for each batch processed and stored in a list
    for that hookpoint in the activations dictionary.

    Args:
        model: The transformer model to hook
        hookpoints: List of hookpoints to collect activations from

    Yields:
        Dictionary mapping hookpoints to their collected activations
    """
    activations = {}
    handles = []

    def create_hook(hookpoint: str, input_acts: bool = False):
        def hook_fn(module: nn.Module, input: Any, output: Tensor) -> Tensor | None:
            # If output is a tuple (like in some transformer layers), take first element
            if input_acts:
                if isinstance(input, tuple):
                    activations[hookpoint] = input[0]
                else:
                    activations[hookpoint] = input
            else:
                if isinstance(output, tuple):
                    activations[hookpoint] = output[0]
                else:
                    activations[hookpoint] = output

        return hook_fn

    for name, module in model.named_modules():
        if name in hookpoints:
            handle = module.register_forward_hook(create_hook(name, input_acts))
            handles.append(handle)

    try:
        yield activations
    finally:
        for handle in handles:
            handle.remove()

