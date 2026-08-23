"""A latent-sharded coder must be indistinguishable from the unsharded one.

Runs on CPU over gloo, so it needs no GPU and no launcher: the equivalence being
checked is algebraic, not hardware-dependent.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import sparsify.sparse_coder as sparse_coder_module
from sparsify.config import SparseCoderConfig
from sparsify.latent_parallel import gather_full_state, sync_replicated_grads
from sparsify.sparse_coder import SparseCoder
from sparsify.utils import eager_decode


def _use_eager_decoder() -> None:
    """These workers run on CPU over gloo whether or not the host has a GPU.

    `decoder_impl` is bound at import time and prefers the Triton kernel, which
    rejects CPU tensors, so pin the eager path in the worker rather than relying
    on the host having no CUDA.
    """
    sparse_coder_module.decoder_impl = eager_decode


D_IN = 32
WORLD_SIZE = 2
SHARDED = ("encoder.weight", "encoder.bias", "W_dec")


def _build_pair(rank: int, cfg: SparseCoderConfig):
    """A full coder and this rank's shard of it, holding identical weights."""
    torch.manual_seed(0)
    full = SparseCoder(D_IN, cfg, dtype=torch.float32)

    shard = SparseCoder(D_IN, cfg, dtype=torch.float32, latent_shard=(rank, WORLD_SIZE))
    lo = rank * shard.num_latents
    hi = lo + shard.num_latents
    shard.encoder.weight.data.copy_(full.encoder.weight.data[lo:hi])
    shard.encoder.bias.data.copy_(full.encoder.bias.data[lo:hi])
    assert full.W_dec is not None and shard.W_dec is not None
    shard.W_dec.data.copy_(full.W_dec.data[lo:hi])
    shard.b_dec.data.copy_(full.b_dec.data)
    return full, shard, lo, hi


def _worker(rank: int, port: int, case: dict, errors):
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(WORLD_SIZE),
        )
        dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
        _use_eager_decoder()

        case = dict(case)
        wants_dead = case.pop("dead", False)
        cfg = SparseCoderConfig(expansion_factor=8, **case)
        full, shard, lo, hi = _build_pair(rank, cfg)

        # The same latents must be dead in both, so derive one global mask and
        # hand each coder its slice.
        dead_full = dead_shard = None
        if wants_dead:
            torch.manual_seed(99)
            dead_full = torch.rand(full.num_latents) < 0.5
            dead_shard = dead_full[lo:hi]

        torch.manual_seed(1234)
        x = torch.randn(64, D_IN)

        out_full = full(x, dead_mask=dead_full)
        out_shard = shard(x, dead_mask=dead_shard)

        # 1. Same reconstruction, hence the same losses.
        tol = dict(atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(out_shard.sae_out, out_full.sae_out, **tol)
        torch.testing.assert_close(out_shard.fvu, out_full.fvu, **tol)
        torch.testing.assert_close(out_shard.auxk_loss, out_full.auxk_loss, **tol)
        torch.testing.assert_close(
            out_shard.multi_topk_fvu, out_full.multi_topk_fvu, **tol
        )
        if wants_dead:
            assert out_full.auxk_loss > 0, "auxk case is vacuous"
        if cfg.multi_topk:
            assert out_full.multi_topk_fvu > 0, "multi_topk case is vacuous"

        # 2. Same gradients on the rows this rank owns.
        loss = out_full.fvu + out_full.auxk_loss + out_full.multi_topk_fvu
        loss.backward()
        (out_shard.fvu + out_shard.auxk_loss + out_shard.multi_topk_fvu).backward()
        sync_replicated_grads([shard])

        for got, want in (
            (shard.encoder.weight, full.encoder.weight),
            (shard.encoder.bias, full.encoder.bias),
            (shard.W_dec, full.W_dec),
        ):
            assert got.grad is not None and want.grad is not None
            torch.testing.assert_close(got.grad, want.grad[lo:hi], **tol)

        # Replicated parameters: correct only after the reduction.
        for got, want in ((shard.b_dec, full.b_dec), (shard.W_skip, full.W_skip)):
            if got is None:
                continue
            assert got.grad is not None and want.grad is not None
            torch.testing.assert_close(got.grad, want.grad, **tol)

        # 3. The gathered checkpoint is the unsharded coder.
        gathered = gather_full_state(shard)
        reference = full.state_dict()
        assert set(gathered) == set(reference)
        for name in SHARDED:
            torch.testing.assert_close(gathered[name], reference[name])
    except Exception:
        import traceback

        errors.put((rank, traceback.format_exc()))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn(target, port: int, *extra):
    ctx = mp.get_context("spawn")
    errors = ctx.Queue()
    procs = [
        ctx.Process(target=target, args=(r, port, *extra, errors))
        for r in range(WORLD_SIZE)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=180)

    failures = []
    while not errors.empty():
        failures.append(errors.get())
    assert not failures, "\n\n".join(f"rank {r}:\n{tb}" for r, tb in failures)
    assert all(proc.exitcode == 0 for proc in procs), [p.exitcode for p in procs]


def _init_worker(rank: int, port: int, errors):
    """A shard built from a seed must equal that slice of the unsharded draw."""
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK=str(rank),
            WORLD_SIZE=str(WORLD_SIZE),
        )
        dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
        _use_eager_decoder()
        cfg = SparseCoderConfig(expansion_factor=8, k=4)

        torch.manual_seed(7)
        full = SparseCoder(D_IN, cfg, dtype=torch.float32)
        torch.manual_seed(7)
        shard = SparseCoder(
            D_IN, cfg, dtype=torch.float32, latent_shard=(rank, WORLD_SIZE)
        )

        lo = rank * shard.num_latents
        hi = lo + shard.num_latents
        torch.testing.assert_close(shard.encoder.weight, full.encoder.weight[lo:hi])
        assert full.W_dec is not None and shard.W_dec is not None
        torch.testing.assert_close(shard.W_dec, full.W_dec[lo:hi])

        # The real failure this guards against: every rank runs the same seed, so
        # drawing at the sharded shape hands them all identical rows and the
        # global dictionary silently collapses to num_latents // world_size.
        peer = shard.encoder.weight.detach().clone()
        dist.broadcast(peer, src=0)
        if rank != 0:
            assert not torch.allclose(
                peer, shard.encoder.weight
            ), "ranks hold identical latents; the global dictionary collapsed"
    except Exception:
        import traceback

        errors.put((rank, traceback.format_exc()))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_shard_initialization_matches_global_draw():
    _spawn(_init_worker, 29600)


CASES = {
    "k1": dict(k=1),
    "k4": dict(k=4),
    "k32": dict(k=32),
    "multi_topk": dict(k=4, multi_topk=True),
    "skip_connection": dict(k=4, skip_connection=True),
    "auxk": dict(k=4, dead=True),
    "auxk_and_skip": dict(k=4, dead=True, skip_connection=True),
}


@pytest.mark.parametrize("name", list(CASES))
def test_latent_shard_matches_unsharded(name: str):
    """Equivalence must hold across the loss terms, not just the plain top-k."""
    _spawn(_worker, 29500 + list(CASES).index(name), CASES[name])
