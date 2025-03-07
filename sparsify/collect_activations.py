from contextlib import contextmanager
from typing import Any

from torch import Tensor, nn
from transformers import PreTrainedModel


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
