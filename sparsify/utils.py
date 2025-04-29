import os
from typing import Any, Optional, Type, TypeVar, cast

import torch
from torch import Tensor, nn
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental import local_map
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
        for (name, mod) in model.base_model.named_modules()
        if isinstance(mod, nn.ModuleList) and len(mod) == N
    ]
    assert len(candidates) == 1, "Could not find the list of layers."

    return candidates[0]


def resolve_widths(
    model: PreTrainedModel,
    module_names: list[str],
    dim: int = -1,
    mesh: Optional[DeviceMesh] = None,
) -> dict[str, int]:
    """Find number of output dimensions for the specified modules."""
    module_to_name = {
        model.base_model.get_submodule(name): name for name in module_names
    }
    shapes: dict[str, int] = {}

    def hook(module, _, output):
        # Unpack tuples if needed
        if isinstance(output, tuple):
            output, *_ = output

        name = module_to_name[module]
        shapes[name] = output.shape[dim]

    handles = [mod.register_forward_hook(hook) for mod in module_to_name]
    with torch.inference_mode() if mesh is None else torch.no_grad():
        dummy = {
            k: v.to(model.device) if mesh is None else distribute_tensor(v, mesh)
            for k, v in model.dummy_inputs.items()
        }
        try:
            model(**dummy)
        finally:
            for handle in handles:
                handle.remove()

    return shapes


def set_submodule(model: nn.Module, submodule_path: str, new_submodule: nn.Module):
    """
    Replaces a submodule in a PyTorch model dynamically.

    Args:
        model (nn.Module): The root model containing the submodule.
        submodule_path (str): Dotted path to the submodule.
        new_submodule (nn.Module): The new module to replace the existing one.

    Example:
        set_submodule(model, "encoder.layer.0.attention.self", nn.Identity())
    """
    parent_path, _, last_name = submodule_path.rpartition(".")
    parent_module = model.get_submodule(parent_path) if parent_path else model
    setattr(parent_module, last_name, new_submodule)


def sharded_axis(
    state_dict: dict[str, DTensor],
) -> dict[str, Optional[int]]:
    """
    Checks which axis each DTensor is sharded on.
    Returns a dictionary mapping each key to the axis it is sharded on,
    or None if it is not sharded.

    Args:
        state_dict (dict[str, DTensor]): The state dictionary containing DTensors.

    Returns:
        dict[str, Optional[int]]: A dictionary mapping each key to the
        axis it is sharded on.
    """
    sharded_axes = {}
    for key, tensor in state_dict.items():
        sharding = tensor.placements
        assert isinstance(sharding[0], Replicate)
        if not isinstance(sharding[1], Shard):
            sharded_axes[key] = None
        else:
            sharded_axes[key] = sharding[1].dim
    return sharded_axes


# Fallback implementation of SAE decoder
def eager_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    return nn.functional.embedding_bag(
        top_indices, W_dec, per_sample_weights=top_acts, mode="sum"
    )


# Triton implementation of SAE decoder
def triton_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    return xformers_embedding_bag(top_indices, W_dec, top_acts)


def parallelize_decoder(decoder):
    """
    Decorator to make the decoder function work on torch.DTensor.
    """

    def wrapper(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
        # Check if the input is a DTensor
        if (
            isinstance(top_indices, DTensor)
            and isinstance(top_acts, DTensor)
            and isinstance(W_dec, DTensor)
        ):
            assert top_indices.device_mesh == top_acts.device_mesh == W_dec.device_mesh
            assert top_indices.placements == top_acts.placements
            placement = {}
            for i, p in enumerate(W_dec.placements):
                if isinstance(p, Shard) and p.dim == 1:
                    placement[i] = Shard(1)
            for i, p in enumerate(top_indices.placements):
                if isinstance(p, Shard) and p.dim == 0:
                    placement[i] = Shard(0)
            placement = [
                placement.get(i, Replicate()) for i in range(len(W_dec.placements))
            ]
            # Use local_map to apply the decoder function on each local tensor
            return local_map(
                decoder,
                placement,
            )(top_indices, top_acts, W_dec)
        else:
            # If not a DTensor, call the decoder function directly
            return decoder(top_indices, top_acts, W_dec)

    return wrapper


try:
    from .xformers import xformers_embedding_bag
except ImportError:
    decoder_impl = eager_decode
    print("Triton not installed, using eager implementation of sparse decoder.")
else:
    if os.environ.get("SPARSIFY_DISABLE_TRITON") == "1":
        print("Triton disabled, using eager implementation of sparse decoder.")
        decoder_impl = eager_decode
    else:
        decoder_impl = triton_decode
decoder_impl = parallelize_decoder(decoder_impl)

DISTRIBUTE_MODEL: bool = False
