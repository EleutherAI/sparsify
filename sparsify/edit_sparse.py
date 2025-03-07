from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel
from sparsify import SparseCoder
import lovely_tensors as lt
lt.monkey_patch()


@contextmanager
def collect_activations(
    model: PreTrainedModel, hookpoints: list[str], transcode: bool = False
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

    def create_hook(hookpoint: str, transcode: bool = False):
        def hook_fn(module: nn.Module, input: Any, output: Tensor) -> Tensor | None:
            # If output is a tuple (like in some transformer layers), take first element
            if transcode:
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
            handle = module.register_forward_hook(create_hook(name, transcode))
            handles.append(handle)

    try:
        yield activations
    finally:
        for handle in handles:
            handle.remove()

# Run input activations through a transcoder
# Save somewhere
# Edit the output activations to be the saved transcoder outputs

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
    def create_edit_hook(hookpoint: str):
        def hook_fn(module: nn.Module, input: Any, output: Tensor) -> Tensor | None:
            print("old output:")
            print(output)
            # assert not isinstance(output, tuple)
            # if isinstance(output, tuple):
            #     if isinstance(input, tuple):
            #         output[0] = sparse_models[hookpoint].forward(input[0]).sae_out
            #     else:
            #         output[0] = sparse_models[hookpoint].forward(input).sae_out
            # else:
            # if isinstance(input, tuple):
            sae_output = sparse_models[hookpoint].cuda()(output.flatten(0, 1).cuda())
            fvu = sae_output.fvu
            new_output = sae_output.sae_out.unflatten(
                dim=0,
                sizes=[output.shape[0], output.shape[1]]
            ).cpu()
            # else:
                # raise ValueError("Input is not a tuple")

            print("new output with fvu", fvu.item())
            
            print(new_output)

            return new_output
        return hook_fn

    for name, module in model.named_modules():
        if name in hookpoints:
            handle = module.register_forward_hook(create_edit_hook(name))
            handles.append(handle)

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
