import os
from typing import Any, Type, TypeVar, cast

import torch
from accelerate.utils import send_to_device
from torch import Tensor, nn
from transformers import PreTrainedModel

T = TypeVar("T")


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


def get_layer_list(model: PreTrainedModel) -> tuple[str, nn.ModuleList]:
    """Get the list of layers to train SAEs on."""
    N = assert_type(int, model.config.num_hidden_layers)
    candidates = [
        (name, mod)
        for (name, mod) in model.named_modules()
        if isinstance(mod, nn.ModuleList) and len(mod) == N
    ]
    assert len(candidates) == 1, "Could not find the list of layers."

    return candidates[0]


@torch.inference_mode()
def resolve_widths(
    model: PreTrainedModel,
    module_names: list[str],
    dim: int = -1,
) -> dict[str, int]:
    """Find number of output dimensions for the specified modules."""
    module_to_name = {model.get_submodule(name): name for name in module_names}
    shapes: dict[str, int] = {}

    def hook(module, _, output):
        # Unpack tuples if needed
        if isinstance(output, tuple):
            output, *_ = output

        name = module_to_name[module]
        shapes[name] = output.shape[dim]

    handles = [mod.register_forward_hook(hook) for mod in module_to_name]
    dummy = send_to_device(model.dummy_inputs, model.device)
    try:
        model(**dummy)
    finally:
        for handle in handles:
            handle.remove()

    return shapes


# Fallback implementation of SAE decoder
def eager_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    buf = top_acts.new_zeros(top_acts.shape[:-1] + (W_dec.shape[-1],))
    acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
    return acts @ W_dec.mT


# Triton implementation of SAE decoder
def triton_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    return TritonDecoder.apply(top_indices, top_acts, W_dec)


try:
    from .kernels import TritonDecoder
except ImportError:
    decoder_impl = eager_decode
    print("Triton not installed, using eager implementation of SAE decoder.")
else:
    if os.environ.get("SPARSIFY_DISABLE_TRITON") == "1":
        print("Triton disabled, using eager implementation of SAE decoder.")
        decoder_impl = eager_decode
    else:
        decoder_impl = triton_decode
