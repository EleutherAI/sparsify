"""Tensor parallelism over the latent dimension of a sparse coder.

`--distribute_modules` shards whole hookpoints across ranks: each rank owns a few
SAEs outright. That removes the replicated parameter/optimizer state, but it pays
for it twice: every rank must all-gather the world-sized activation for *every*
hookpoint, including the ones it does not own, and the number of hookpoints has to
be divisible by the world size.

Sharding the latent dimension instead keeps every hookpoint on every rank and
splits each sparse coder's rows. Rank `r` owns latents
`[r * M/W, (r+1) * M/W)` of every coder, so parameters, gradients and optimizer
state all shrink by `W` exactly as they do under `--distribute_modules`, while the
`[N, M]` pre-activation, the dominant transient, shrinks to `[N, M/W]` instead of
growing.

Two collectives per coder per forward, and none in the backward:

1. `global_topk`: each rank offers its local top-k as candidates and all ranks
   agree on the same global winners. The global top-k of size k can draw at most k
   entries from any one rank, so local top-k candidates are sufficient.
2. `all_reduce_sum`: each rank decodes only the winners it owns and the partial
   reconstructions are summed. The backward of that sum is the identity, because
   every rank computes the same loss from the same summed output.
"""

from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from .fused_encoder import BACKWARD_CHUNK_BYTES


def shard_size(num_latents: int, world_size: int) -> int:
    """Number of latents owned by each rank."""
    if num_latents % world_size:
        raise ValueError(
            f"num_latents ({num_latents}) must be divisible by the world size "
            f"({world_size}) to shard the latent dimension"
        )
    return num_latents // world_size


class AllReduceSum(torch.autograd.Function):
    """Sum a tensor across ranks; the backward is the identity.

    Valid only when every rank goes on to compute the same scalar loss from the
    summed result, which is the case here: all ranks see the same input batch and
    the same reconstruction, so `dL/d(partial_r)` is `dL/d(sum)` on every rank and
    no gradient communication is needed.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:  # type: ignore[override]
        out = x.contiguous().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    @staticmethod
    def backward(ctx, grad: Tensor):  # type: ignore[override]
        return grad


def all_reduce_sum(x: Tensor) -> Tensor:
    return AllReduceSum.apply(x)  # type: ignore[return-value]


def global_sum(x: Tensor) -> Tensor:
    """Sum a tensor across ranks in place, outside the autograd graph."""
    total = x.detach().clone()
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return total


def global_topk_mask(local_acts: Tensor, k: int) -> Tensor:
    """Which of this rank's local top-k entries survive the *global* top-k.

    `local_acts` is `[N, k]`, the activations this rank offers as candidates.
    Returns a `[N, k]` bool mask. Ranks agree because they all run the same
    deterministic reduction over the same gathered tensor.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    n, k_local = local_acts.shape

    # all_gather_into_tensor concatenates along dim 0, so gather to [W, N, k] and
    # transpose to put the candidate axis last: position p means rank p // k.
    gathered = local_acts.new_empty(world_size * n, k_local)
    dist.all_gather_into_tensor(gathered, local_acts.detach().contiguous())
    candidates = gathered.view(world_size, n, k_local).permute(1, 0, 2).reshape(n, -1)

    winners = candidates.topk(k, dim=-1, sorted=False).indices  # [N, k]
    mine = winners.div(k_local, rounding_mode="floor") == rank
    local_pos = winners % k_local

    mask = torch.zeros(n * k_local, dtype=torch.bool, device=local_acts.device)
    rows = torch.arange(n, device=local_acts.device).unsqueeze(1).expand_as(local_pos)
    mask[(rows * k_local + local_pos)[mine]] = True
    return mask.view(n, k_local)


# Parameters split along dim 0; everything else in a coder is replicated.
SHARDED_KEYS = ("encoder.weight", "encoder.bias", "W_dec")


def gather_full_state(sae) -> dict[str, Tensor]:
    """Reassemble the unsharded state dict of a latent-sharded coder.

    Collective: every rank must call it. Only rank 0's result is meaningful to
    write, but all ranks get the same tensors. Checkpoints written from this load
    with the ordinary `SparseCoder.load_from_disk`, so a sharded run and a
    single-GPU run produce interchangeable artifacts.
    """
    world_size = dist.get_world_size()
    full: dict[str, Tensor] = {}
    for name, tensor in sae.state_dict().items():
        if name in SHARDED_KEYS:
            buf = tensor.new_empty(tensor.shape[0] * world_size, *tensor.shape[1:])
            dist.all_gather_into_tensor(buf, tensor.contiguous())
            full[name] = buf
        else:
            full[name] = tensor
    return full


def sync_replicated_grads(saes) -> None:
    """Sum the gradients of the parameters that are *not* sharded.

    The sharded rows see every token on every rank, so their gradients are already
    complete and need no communication. Only `b_dec` (and `W_skip`) are replicated.
    """
    for sae in saes:
        for param in (sae.b_dec, sae.W_skip):
            if param is not None and param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)


# --------------------------------------------------------------------------------
# Ragged winners.
#
# After the global reduction a rank owns only some of the k candidates it offered
# -- k / world_size of them on average. Keeping the [N, k] layout and zeroing the
# losers is correct but wasteful: the decode and the encoder weight-gradient are
# both O(N * k * d) regardless of how few entries survive, so per-rank work stays
# flat as the world grows even though each rank owns fewer latents.
#
# Compacting the survivors into a flat (token, latent, value) list makes both
# O(nnz * d) with nnz = N * k / world_size, which is the scaling the sharding was
# supposed to buy. Padding to the busiest token instead would only recover a
# factor of about k / max_per_token, roughly 2.5x at k=32 and 8 ranks, against
# world_size here.
# --------------------------------------------------------------------------------


class Winners(NamedTuple):
    values: Tensor
    """Pre-activation of each surviving (token, latent) pair, `[nnz]`."""

    rows: Tensor
    """Token index of each pair, ascending, `[nnz]`."""

    latents: Tensor
    """Shard-local latent index of each pair, `[nnz]`."""

    offsets: Tensor
    """Start of each token's run in the flat list, `[N]`, for `embedding_bag`."""


def compact_winners(mask: Tensor, cand_acts: Tensor, cand_idx: Tensor) -> Winners:
    """Flatten the surviving candidates, dropping the losers entirely."""
    rows, cols = mask.nonzero(as_tuple=True)  # row-major, so rows is ascending
    counts = mask.sum(1)
    offsets = torch.zeros_like(counts)
    offsets[1:] = counts.cumsum(0)[:-1]
    return Winners(cand_acts[rows, cols], rows, cand_idx[rows, cols], offsets)


def ragged_decode(winners: Winners, acts: Tensor, w_dec: Tensor) -> Tensor:
    """Sum `acts * w_dec[latent]` per token, over the flat winner list."""
    return F.embedding_bag(
        winners.latents,
        w_dec,
        offsets=winners.offsets,
        mode="sum",
        per_sample_weights=acts.to(w_dec.dtype),
    )


class SparseEncoderGrad(torch.autograd.Function):
    """Carry the encoder's gradient for a ragged set of selected latents.

    The dense pre-activation is computed outside the graph, because it exists
    only to run the top-k. This hands the selected values back to autograd with
    the gradient the encoder would have produced, so the backward touches the
    surviving pairs rather than every candidate the rank offered. The arithmetic
    matches `FusedEncoder.backward`, restricted to those pairs -- which is exact,
    since the losers were multiplied by zero and carry no gradient anyway.
    """

    @staticmethod
    def forward(ctx, values, x, weight, bias, rows, latents):  # type: ignore[override]
        ctx.save_for_backward(x, weight, bias, rows, latents)
        return values.clone()

    @staticmethod
    def backward(ctx, grad_values):  # type: ignore[override]
        x, weight, bias, rows, latents = ctx.saved_tensors
        _, needs_x, needs_w, needs_b, _, _ = ctx.needs_input_grad

        grad_x = torch.zeros_like(x) if needs_x else None
        grad_w = torch.zeros_like(weight) if needs_w else None

        d = x.shape[-1]
        itemsize = torch.promote_types(grad_values.dtype, x.dtype).itemsize
        block = max(1, BACKWARD_CHUNK_BYTES // max(1, d * itemsize))

        for start in range(0, latents.numel(), block):
            stop = start + block
            gv = grad_values[start:stop].unsqueeze(1)
            row = rows[start:stop]
            latent = latents[start:stop]
            if grad_x is not None:
                grad_x.index_add_(0, row, (weight[latent] * gv).type_as(x))
            if grad_w is not None:
                grad_w.index_add_(0, latent, (x[row] * gv).type_as(weight))

        grad_b = None
        if bias is not None and needs_b:
            grad_b = torch.zeros_like(bias)
            grad_b.index_add_(0, latents, grad_values.type_as(bias))

        return None, grad_x, grad_w, grad_b, None, None


def sparse_encoder_grad(
    values: Tensor, x: Tensor, weight: Tensor, bias: Tensor, winners: Winners
) -> Tensor:
    return SparseEncoderGrad.apply(  # type: ignore[return-value]
        values, x, weight, bias, winners.rows, winners.latents
    )
