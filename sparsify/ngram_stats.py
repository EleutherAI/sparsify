"""Build a frozen n-gram conditional-mean activation table for a model + corpus.

This is the statistics pass of the frozen-lookup baseline (spec Stages 2-4, minus the
raw activation cache). One forward pass over the corpus accumulates, per hookpoint,
the fp32 sum / sum-of-squares / count of activations grouped by each position's
longest-suffix gram id. We then aggregate bottom-up over the backoff tree, shrink
each mean toward its parent, and report the headline R^2(n) ANOVA table (the fraction
of activation variance explained by conditioning on the suffix n-gram, per order n).

Run e.g.::

    python -m sparsify.ngram_stats HuggingFaceTB/SmolLM2-135M \\
        EleutherAI/SmolLM2-135M-10B --hookpoints layers.9 \\
        --max_tokens 10_000_000 --tau 30 --output_dir ngram_tables/smoke

The output directory gets one ``GramTable`` per hookpoint plus ``manifest.json`` and
``rsquared.json``. Point ``sparsify`` training at it with ``--gram_table_path``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import psutil
import torch
from datasets import Dataset, load_dataset
from simple_parsing import Serializable, field as sp_field, list_field, parse
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from .data import MemmapDataset, chunk_and_tokenize
from .ngram import (
    GLOBAL_ID,
    GramTable,
    GramVocab,
    ancestor_at_order,
    annotate,
    select_vocab,
)
from .utils import resolve_widths, simple_parse_args_string

# Unbuffered prints so progress shows up promptly when stdout is redirected to a file.
print = functools.partial(print, flush=True)  # noqa: A001


@dataclass
class NgramStatsConfig(Serializable):
    model: str = sp_field(default="HuggingFaceTB/SmolLM2-135M", positional=True)
    """Model whose activations are averaged."""

    dataset: str = sp_field(default="EleutherAI/SmolLM2-135M-10B", positional=True)
    """Pre-tokenized corpus (``input_ids``) or a raw text dataset / ``.bin`` memmap."""

    hookpoints: list[str] = list_field("layers.9")
    """Residual-stream hookpoints to build tables for (sparsify module names)."""

    output_dir: str = "ngram_tables/run"

    split: str = "train"
    ctx_len: int = 2048

    max_tokens: int | None = None
    """Token budget for the statistics pass. ``None`` uses the whole corpus (minus the
    held-out tail). The vocabulary is selected over the same budget."""

    max_docs: int | None = None
    """Limit the number of raw text documents tokenized (before chunking). Use for fast
    shakeouts on a raw-text corpus; ``None`` tokenizes the whole dataset (cached)."""

    tau: int = 30
    """Minimum corpus occurrence count for an n-gram (n>=2) to be kept."""

    lam: float = 64.0
    """Shrinkage strength toward the backoff parent mean."""

    orders: list[int] = list_field(1, 2, 3, 4)

    batch_size: int = 16
    """Sequences per forward-pass batch."""

    accum_device: str = "auto"
    """Where the fp32 accumulators live. 'auto' keeps them on the GPU when they fit in
    a fraction of free VRAM (fastest, no per-batch host transfer), else on CPU."""

    eod_id: int | None = None
    """End-of-document token id. Defaults to the tokenizer's eos id."""

    vocab_size: int | None = None
    """Radix base for key packing. Defaults to ``len(tokenizer)``."""

    revision: str | None = None
    hf_token: str | None = sp_field(default=None, encoding_fn=lambda _: None)
    text_column: str = "text"
    data_args: str = ""
    shuffle_seed: int = 42
    data_preprocessing_num_proc: int = field(default_factory=lambda: cpu_count() // 2)

    select_row_chunk: int = 20_000
    """Rows per block during vocabulary selection."""

    checkpoint_every_tokens: int = 20_000_000
    holdout_tokens: int = 1_000_000
    """Held-out tail used for the FVU sanity check (disjoint from the stats budget)."""

    num_sanity_grams: int = 100
    resume: bool = True

    mode: str = "all"
    """Pipeline stage: 'all' (single process, default), or for multi-GPU sharding
    'vocab' (build+save the vocabulary), 'accum' (accumulate one row shard against a
    saved vocabulary), or 'merge' (sum shard accumulators and finalize)."""

    shard_id: int = 0
    num_shards: int = 1
    """Contiguous row shard handled by this 'accum' process."""

    vocab_dir: str | None = None
    """Directory holding the shared vocabulary (defaults to output_dir)."""


# --------------------------------------------------------------------------- IO


class _RowView:
    """Lazy ``[N, S]`` numpy view over an HF dataset's ``input_ids`` column."""

    def __init__(self, ds: Dataset):
        self.ds = ds.with_format("numpy")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, sl):
        return np.asarray(self.ds[sl]["input_ids"])


def load_model_and_rows(cfg: NgramStatsConfig):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModel.from_pretrained(
        cfg.model,
        device_map={"": "cuda:0"},
        torch_dtype=dtype,
        revision=cfg.revision,
        token=cfg.hf_token,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, token=cfg.hf_token)

    if cfg.dataset.endswith(".bin"):
        ds = MemmapDataset(cfg.dataset, cfg.ctx_len)
        rows = ds.mmap  # already [N, ctx_len] uint16
    else:
        try:
            kwargs = simple_parse_args_string(cfg.data_args)
            ds = load_dataset(cfg.dataset, split=cfg.split, **kwargs)
        except ValueError as e:
            if "load_from_disk" in str(e):
                ds = Dataset.load_from_disk(cfg.dataset, keep_in_memory=False)
            else:
                raise
        assert isinstance(ds, Dataset)
        if cfg.max_docs is not None and "input_ids" not in ds.column_names:
            ds = ds.select(range(min(cfg.max_docs, len(ds))))
            print(f"Limited to {len(ds):,} raw documents (max_docs).")
        if "input_ids" not in ds.column_names:
            ds = chunk_and_tokenize(
                ds, tokenizer, max_seq_len=cfg.ctx_len,
                num_proc=cfg.data_preprocessing_num_proc, text_key=cfg.text_column,
            )
        else:
            print("Dataset already tokenized; skipping tokenization.")
        # Deterministic order shared with training (same seed) so arms see one corpus.
        ds = ds.shuffle(cfg.shuffle_seed)
        rows = _RowView(ds)

    return model, tokenizer, rows


# ---------------------------------------------------------------- finalize math


def aggregate_bottom_up(raw: np.ndarray, vocab: GramVocab, orders) -> np.ndarray:
    """Sum an accumulator indexed by *finest* gram id up the backoff tree so that
    ``agg[g]`` covers every position whose length-``order[g]`` suffix is ``g``.
    Works for a ``[G]`` or ``[G, d]`` array. Not in place.
    """
    agg = raw.astype(np.float64 if raw.dtype != np.int64 else np.int64).copy()
    parent = vocab.parent
    order = vocab.order
    # Highest order first so contributions cascade all the way to GLOBAL.
    for n in sorted(orders, reverse=True):
        idx = np.nonzero(order == n)[0]
        if len(idx) == 0:
            continue
        np.add.at(agg, parent[idx], agg[idx])
    return agg


def compute_within_variance(
    raw_sums: np.ndarray,
    raw_sq: np.ndarray,
    raw_counts: np.ndarray,
    bucket: np.ndarray,
    num_buckets: int,
) -> np.ndarray:
    """Per-dimension within-group sum of squares for a given grouping.

    ``bucket[g]`` maps each finest gram id to its group id. Returns a ``[d]`` vector
    ``sum_group (Q_group - S_group^2 / C_group)`` where S/Q are per-dim sum / sum-sq.
    """
    d = raw_sums.shape[1]
    s = np.zeros((num_buckets, d), np.float64)
    q = np.zeros((num_buckets, d), np.float64)
    c = np.zeros(num_buckets, np.float64)
    np.add.at(s, bucket, raw_sums)
    np.add.at(q, bucket, raw_sq)
    np.add.at(c, bucket, raw_counts)
    nz = c > 0
    within = q[nz] - (s[nz] ** 2) / c[nz][:, None]
    return within.sum(0)  # [d]


def shrink_means(
    agg_sums: np.ndarray, agg_counts: np.ndarray, vocab: GramVocab, orders, lam: float
) -> np.ndarray:
    """Bottom-up James-Stein-style shrinkage of each gram mean toward its backoff
    parent mean: ``mu[g] = (c[g] mu_raw[g] + lam mu[parent]) / (c[g] + lam)``.
    Parents (lower order) are finalized first. GLOBAL shrinks toward itself (no-op).
    """
    counts = np.maximum(agg_counts, 1)[:, None]
    mu_raw = agg_sums / counts
    mu = np.zeros_like(mu_raw)
    mu[GLOBAL_ID] = mu_raw[GLOBAL_ID]

    parent = vocab.parent
    order = vocab.order
    for n in sorted(orders):  # ascending: parent order n-1 already done
        idx = np.nonzero(order == n)[0]
        if len(idx) == 0:
            continue
        c = agg_counts[idx][:, None].astype(np.float64)
        mu[idx] = (c * mu_raw[idx] + lam * mu[parent[idx]]) / (c + lam)
    return mu


# --------------------------------------------------------------------- the pass


def _register_hooks(model, hookpoints):
    captured: dict[str, Tensor] = {}
    handles = []
    for hp in hookpoints:
        mod = model.base_model.get_submodule(hp)

        def make(hp):
            def hook(module, inputs, outputs):
                captured[hp] = (outputs[0] if isinstance(outputs, tuple) else outputs)
            return hook

        handles.append(mod.register_forward_hook(make(hp)))
    return captured, handles


def _prepare(cfg: NgramStatsConfig, need_model: bool = True):
    """Load model/rows and compute the shared row-budget geometry."""
    device = "cuda:0"
    model, tokenizer, rows = load_model_and_rows(cfg)
    eod_id = cfg.eod_id if cfg.eod_id is not None else int(tokenizer.eos_token_id)
    vocab_size = cfg.vocab_size or len(tokenizer)
    widths = resolve_widths(model, cfg.hookpoints)
    d_model = next(iter(widths.values()))
    assert len(set(widths.values())) == 1, f"hookpoints differ in width: {widths}"

    n_total = len(rows)
    holdout_rows = max(1, (cfg.holdout_tokens + cfg.ctx_len - 1) // cfg.ctx_len)
    holdout_rows = min(holdout_rows, n_total // 10)
    avail_stats = n_total - holdout_rows
    if cfg.max_tokens is None:
        stats_rows = avail_stats
    else:
        stats_rows = min(avail_stats, (cfg.max_tokens + cfg.ctx_len - 1) // cfg.ctx_len)
    print(f"corpus rows={n_total:,} ctx_len={cfg.ctx_len} d_model={d_model} "
          f"eod_id={eod_id} vocab_size={vocab_size}")
    print(f"stats rows={stats_rows:,} (~{stats_rows*cfg.ctx_len/1e6:.0f}M tokens), "
          f"holdout rows={holdout_rows:,}")
    return dict(device=device, model=model, tokenizer=tokenizer, rows=rows,
                eod_id=eod_id, vocab_size=vocab_size, d_model=d_model,
                n_total=n_total, holdout_rows=holdout_rows, stats_rows=stats_rows)


def _accumulate(cfg, ctx, vocab, r_start, r_end, ckpt=None):
    """Accumulate fp32 sums/sq/counts over rows [r_start, r_end). Returns numpy dicts."""
    device, model, rows = ctx["device"], ctx["model"], ctx["rows"]
    d_model, G = ctx["d_model"], vocab.num_grams
    n_hp = len(cfg.hookpoints)
    adev = _choose_accum_device(cfg, G, d_model, n_hp, device)
    print(f"accumulating rows [{r_start:,},{r_end:,}) on {adev}")
    sums = {hp: torch.zeros(G, d_model, dtype=torch.float32, device=adev)
            for hp in cfg.hookpoints}
    sq = {hp: torch.zeros(G, d_model, dtype=torch.float32, device=adev)
          for hp in cfg.hookpoints}
    counts_t = torch.zeros(G, dtype=torch.int64, device=adev)
    start_row = r_start
    if ckpt is not None and cfg.resume:
        resumed = _maybe_resume(cfg, ckpt, sums, sq, counts_t)
        start_row = max(r_start, resumed) if resumed else r_start

    captured, handles = _register_hooks(model, cfg.hookpoints)
    vocab_t = vocab.to_torch(device)
    t0 = time.time()
    tokens_since_ckpt = 0
    try:
        for r0 in range(start_row, r_end, cfg.batch_size):
            r1 = min(r0 + cfg.batch_size, r_end)
            x = torch.from_numpy(np.asarray(rows[r0:r1]).astype(np.int64)).to(device)
            with torch.no_grad():
                model(x)
            gram_ids = annotate(x, vocab_t).flatten().to(adev)
            counts_t.index_add_(0, gram_ids, torch.ones_like(gram_ids))
            for hp in cfg.hookpoints:
                acts = captured[hp].flatten(0, 1).float()
                if acts.device != torch.device(adev):
                    acts = acts.to(adev)
                sums[hp].index_add_(0, gram_ids, acts)
                sq[hp].index_add_(0, gram_ids, acts * acts)

            tokens_since_ckpt += (r1 - r0) * cfg.ctx_len
            if ckpt is not None and tokens_since_ckpt >= cfg.checkpoint_every_tokens:
                _save_ckpt(ckpt, r1, sums, sq, counts_t)
                tokens_since_ckpt = 0
                done = (r1 - start_row) * cfg.ctx_len
                print(f"  rows {r1:,}/{r_end:,}  ({done/1e6:.0f}M tok, "
                      f"{time.time()-t0:.0f}s, {done/max(time.time()-t0,1)/1e3:.0f}k "
                      f"tok/s)", flush=True)
    finally:
        for h in handles:
            h.remove()
    return ({hp: sums[hp].cpu().numpy() for hp in cfg.hookpoints},
            {hp: sq[hp].cpu().numpy() for hp in cfg.hookpoints},
            counts_t.cpu().numpy())


def _finalize(cfg, ctx, vocab, sums, sq, counts, run_sanity=True):
    out = Path(cfg.output_dir)
    print("\n[Finalize] aggregating bottom-up, shrinking, computing R^2 ...")
    agg_counts = aggregate_bottom_up(counts, vocab, cfg.orders).astype(np.int64)
    _sanity_counts(agg_counts, vocab)
    manifest = _build_manifest(cfg, vocab, ctx["eod_id"], ctx["vocab_size"],
                               ctx["d_model"])
    rsquared_all = {}
    captured, handles = ({}, [])
    if run_sanity:
        captured, handles = _register_hooks(ctx["model"], cfg.hookpoints)
    vocab_t = vocab.to_torch(ctx["device"])
    try:
        for hp in cfg.hookpoints:
            agg_sums = aggregate_bottom_up(sums[hp], vocab, cfg.orders)
            mu = shrink_means(agg_sums, agg_counts, vocab, cfg.orders, cfg.lam)
            rsq = _rsquared_report(sums[hp], sq[hp], counts, vocab, cfg.orders, hp)
            rsquared_all[hp] = rsq
            GramTable(vocab=vocab, d_model=ctx["d_model"], hookpoint=hp,
                      mu=mu.astype(np.float32), counts=agg_counts,
                      manifest=manifest).save(out / hp.replace("/", "_"))
            if run_sanity:
                try:
                    _sanity_brute_force(cfg, ctx["rows"], ctx["model"], vocab, vocab_t,
                                        mu, hp, captured, ctx["device"],
                                        ctx["stats_rows"])
                    _sanity_holdout_fvu(cfg, ctx["rows"], ctx["model"], vocab, vocab_t,
                                        mu, hp, captured, ctx["device"],
                                        ctx["n_total"], ctx["holdout_rows"], rsq)
                except Exception as e:  # noqa: BLE001
                    print(f"  WARNING: sanity for {hp} failed ({e}); table saved.")
    finally:
        for h in handles:
            h.remove()
    with open(out / "rsquared.json", "w") as f:
        json.dump(rsquared_all, f, indent=2)
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _build_and_report_vocab(cfg, ctx):
    print("\n[Stage 2] selecting n-gram vocabulary ...")
    vocab = select_vocab(
        ctx["rows"], vocab_size=ctx["vocab_size"], orders=tuple(cfg.orders),
        tau=cfg.tau, eod_id=ctx["eod_id"], max_rows=ctx["stats_rows"],
        row_chunk=cfg.select_row_chunk,
    )
    G, n_hp = vocab.num_grams, len(cfg.hookpoints)
    accum_bytes = 2 * G * ctx["d_model"] * 4 * n_hp
    avail_ram = psutil.virtual_memory().available
    print(f"G={G:,}  accumulator RAM = {accum_bytes/1e9:.1f} GB "
          f"({n_hp} hookpoint(s)); available = {avail_ram/1e9:.0f} GB")
    print(f"shipped table (bf16) ~= {G*ctx['d_model']*2/1e9:.1f} GB per hookpoint")
    if accum_bytes > 0.9 * avail_ram:
        raise MemoryError(
            f"fp32 accumulators need {accum_bytes/1e9:.1f} GB but only "
            f"{avail_ram/1e9:.0f} GB available. Raise --tau or reduce --hookpoints."
        )
    return vocab


def _shard_bounds(cfg, stats_rows):
    per = (stats_rows + cfg.num_shards - 1) // cfg.num_shards
    r0 = cfg.shard_id * per
    r1 = min(stats_rows, r0 + per)
    return r0, r1


def run_stats(cfg: NgramStatsConfig):
    t0 = time.time()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    vocab_dir = Path(cfg.vocab_dir) if cfg.vocab_dir else out

    if cfg.mode == "vocab":
        ctx = _prepare(cfg)
        vocab = _build_and_report_vocab(cfg, ctx)
        vocab.save(vocab_dir)
        print(f"\nDONE vocab in {time.time()-t0:.0f}s -> {vocab_dir}")
        return

    if cfg.mode == "accum":
        ctx = _prepare(cfg)
        vocab = GramVocab.load(vocab_dir)
        r0, r1 = _shard_bounds(cfg, ctx["stats_rows"])
        ckpt = out / "_ckpt" / f"shard{cfg.shard_id}"
        sums, sq, counts = _accumulate(cfg, ctx, vocab, r0, r1, ckpt=ckpt)
        shard_dir = out / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        arrays = {"counts": counts}
        for hp in cfg.hookpoints:
            safe = hp.replace("/", "_")
            arrays[f"sums_{safe}"] = sums[hp]
            arrays[f"sq_{safe}"] = sq[hp]
        np.savez(shard_dir / f"shard{cfg.shard_id}.npz", **arrays)  # type: ignore[arg-type]
        print(f"\nDONE shard {cfg.shard_id} in {time.time()-t0:.0f}s")
        return

    if cfg.mode == "merge":
        ctx = _prepare(cfg)
        vocab = GramVocab.load(vocab_dir)
        sums = {hp: None for hp in cfg.hookpoints}
        sq = {hp: None for hp in cfg.hookpoints}
        counts = None
        for f in sorted((out / "shards").glob("shard*.npz")):
            z = np.load(f)
            counts = z["counts"] if counts is None else counts + z["counts"]
            for hp in cfg.hookpoints:
                safe = hp.replace("/", "_")
                sums[hp] = (z[f"sums_{safe}"] if sums[hp] is None
                            else sums[hp] + z[f"sums_{safe}"])
                sq[hp] = (z[f"sq_{safe}"] if sq[hp] is None
                          else sq[hp] + z[f"sq_{safe}"])
            print(f"  merged {f.name}")
        _finalize(cfg, ctx, vocab, sums, sq, counts, run_sanity=True)
        # Reclaim the large per-shard accumulator files now that the table is saved.
        for f in (out / "shards").glob("shard*.npz"):
            f.unlink()
        print(f"\nDONE merge in {time.time()-t0:.0f}s. Tables + manifest in {out}")
        return

    # mode == "all": single process end to end
    ctx = _prepare(cfg)
    vocab = _build_and_report_vocab(cfg, ctx)
    ckpt = out / "_ckpt"
    sums, sq, counts = _accumulate(cfg, ctx, vocab, 0, ctx["stats_rows"], ckpt=ckpt)
    _finalize(cfg, ctx, vocab, sums, sq, counts, run_sanity=True)
    if ckpt.exists():
        for p in ckpt.rglob("*"):
            if p.is_file():
                p.unlink()
    print(f"\nDONE in {time.time()-t0:.0f}s. Tables + manifest in {out}")


# ------------------------------------------------------------------ checkpoints


def _choose_accum_device(cfg, G, d_model, n_hp, device) -> str:
    """Keep accumulators on GPU when they comfortably fit free VRAM (no per-batch host
    transfer), else CPU. fp32 sums + sq per hookpoint, plus int64 counts."""
    if cfg.accum_device != "auto":
        return cfg.accum_device
    if not torch.cuda.is_available():
        return "cpu"
    need = 2 * G * d_model * 4 * n_hp + G * 8
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    return device if need < 0.5 * free else "cpu"


def _save_ckpt(ckpt: Path, row: int, sums, sq, counts):
    ckpt.mkdir(parents=True, exist_ok=True)
    tmp = ckpt / "tmp"
    tmp.mkdir(exist_ok=True)
    np.save(tmp / "counts.npy", counts.cpu().numpy())
    for hp in sums:
        safe = hp.replace("/", "_")
        np.save(tmp / f"sums_{safe}.npy", sums[hp].cpu().numpy())
        np.save(tmp / f"sq_{safe}.npy", sq[hp].cpu().numpy())
    with open(tmp / "row.json", "w") as f:
        json.dump({"row": int(row)}, f)
    # Atomic-ish swap.
    for p in tmp.iterdir():
        p.rename(ckpt / p.name)
    tmp.rmdir()


def _maybe_resume(cfg, ckpt: Path, sums, sq, counts) -> int:
    row_file = ckpt / "row.json"
    if not row_file.exists():
        return 0
    with open(row_file) as f:
        row = json.load(f)["row"]
    counts.copy_(torch.from_numpy(np.load(ckpt / "counts.npy")))
    for hp in sums:
        safe = hp.replace("/", "_")
        sums[hp].copy_(torch.from_numpy(np.load(ckpt / f"sums_{safe}.npy")))
        sq[hp].copy_(torch.from_numpy(np.load(ckpt / f"sq_{safe}.npy")))
    print(f"Resumed from checkpoint at row {row:,}")
    return int(row)


# ---------------------------------------------------------------- sanity checks


def _sanity_counts(agg_counts, vocab: GramVocab):
    ok = np.array_equal(agg_counts, vocab.vocab_counts)
    print(f"  sanity (a) aggregated counts == vocab counts: "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        diff = np.nonzero(agg_counts != vocab.vocab_counts)[0][:5]
        raise AssertionError(
            f"count mismatch at ids {diff}: agg={agg_counts[diff]} "
            f"vocab={vocab.vocab_counts[diff]}"
        )


def _rsquared_report(raw_sums, raw_sq, raw_counts, vocab, orders, hp):
    total_count = float(raw_counts.sum())
    global_sum = raw_sums.sum(0, dtype=np.float64)
    global_sq = raw_sq.sum(0, dtype=np.float64)
    total_within = global_sq - (global_sum ** 2) / total_count  # [d], grouping order 0
    total_ss = float(total_within.sum())

    print(f"\n  === R^2(n) ANOVA @ {hp} (total variance SS={total_ss:.4g}) ===")
    report = {"total_ss": total_ss, "total_count": total_count, "by_order": {}}
    max_order = max(orders)
    for n in range(0, max_order + 1):
        bucket = ancestor_at_order(vocab.parent, vocab.order, n)
        within = compute_within_variance(
            raw_sums.astype(np.float64), raw_sq.astype(np.float64),
            raw_counts.astype(np.float64), bucket, vocab.num_grams,
        )
        within_ss = float(within.sum())
        r2 = 1.0 - within_ss / total_ss if total_ss > 0 else 0.0
        n_groups = int(len(np.unique(bucket)))
        report["by_order"][n] = {"within_ss": within_ss, "r2": r2, "groups": n_groups}
        print(f"    n<={n}: R^2 = {r2:.4f}   (within SS={within_ss:.4g}, "
              f"{n_groups:,} groups)")
    return report


@torch.no_grad()
def _forward_acts_batched(cfg, rows, model, captured, hp, device, r0, r1):
    """Run the model over rows [r0, r1) in mini-batches; return CPU acts + gram ids."""
    acts_parts, block_parts = [], []
    for b0 in range(r0, r1, cfg.batch_size):
        b1 = min(b0 + cfg.batch_size, r1)
        block = np.asarray(rows[b0:b1]).astype(np.int64)
        x = torch.from_numpy(block).to(device)
        model(x)
        acts_parts.append(captured[hp].flatten(0, 1).float().cpu())
        block_parts.append(block)
    return torch.cat(acts_parts).numpy(), np.concatenate(block_parts)


def _sanity_brute_force(cfg, rows, model, vocab, vocab_t, mu, hp, captured,
                        device, stats_rows):
    """(b) Recompute mu for a few random grams by brute force over a slice."""
    rng = np.random.default_rng(0)
    n_probe_rows = min(cfg.num_sanity_grams * 8, 512, stats_rows)
    r = rng.integers(0, stats_rows - n_probe_rows) if stats_rows > n_probe_rows else 0
    acts, block = _forward_acts_batched(
        cfg, rows, model, captured, hp, device, r, r + n_probe_rows
    )
    gids = annotate(torch.from_numpy(block), vocab_t).flatten().cpu().numpy()

    # Pick grams that actually occur in this slice, prefer higher order.
    present, cnt = np.unique(gids, return_counts=True)
    present = present[cnt >= 20]
    if len(present) == 0:
        print("  sanity (b) brute-force: no sufficiently frequent grams in slice; skip")
        return
    probe = present[np.argsort(vocab.order[present])[-min(20, len(present)):]]

    # index_add accumulation must reproduce the brute-force per-gram mean exactly
    # (this is the fp32-accumulation pitfall guard, at the finest grouping).
    s = np.zeros((vocab.num_grams, acts.shape[1]), np.float64)
    c = np.zeros(vocab.num_grams, np.float64)
    np.add.at(s, gids, acts)
    np.add.at(c, gids, 1.0)
    err = max(
        float(np.abs(acts[gids == g].mean(0) - s[g] / c[g]).max()) for g in probe
    )
    print(f"  sanity (b) index_add mean vs brute-force (fp32): max abs err={err:.2e} "
          f"{'PASS' if err < 1e-2 else 'FAIL'}")


def _sanity_holdout_fvu(cfg, rows, model, vocab, vocab_t, mu, hp, captured,
                        device, n_total, holdout_rows, rsq):
    """(c) FVU of the mu-predictor on the held-out tail ~= 1 - R^2(max order)."""
    h0 = n_total - holdout_rows
    mu_t = torch.from_numpy(mu)
    global_mu = torch.from_numpy(mu[GLOBAL_ID])
    num = 0.0
    den = 0.0
    for r0 in range(h0, n_total, cfg.batch_size):
        r1 = min(r0 + cfg.batch_size, n_total)
        acts_np, block = _forward_acts_batched(
            cfg, rows, model, captured, hp, device, r0, r1
        )
        acts = torch.from_numpy(acts_np)
        gids = annotate(torch.from_numpy(block), vocab_t).flatten().cpu()
        pred = mu_t[gids]
        num += (acts - pred).pow(2).sum().item()
        den += (acts - global_mu).pow(2).sum().item()
    fvu = num / den if den > 0 else float("nan")
    max_order = max(cfg.orders)
    expected = 1.0 - rsq["by_order"][max_order]["r2"]
    print(f"  sanity (c) holdout FVU={fvu:.4f}  vs  1-R^2({max_order})={expected:.4f} "
          f"(gap {abs(fvu-expected):.4f})")


# ------------------------------------------------------------------- manifest


def _build_manifest(cfg, vocab, eod_id, vocab_size, d_model):
    key_hashes = {
        n: hashlib.sha1(k.tobytes()).hexdigest()[:16]
        for n, k in vocab.sorted_keys.items()
    }
    payload = {
        "model": cfg.model,
        "revision": cfg.revision,
        "dataset": cfg.dataset,
        "split": cfg.split,
        "hookpoints": list(cfg.hookpoints),
        "ctx_len": cfg.ctx_len,
        "max_tokens": cfg.max_tokens,
        "tau": cfg.tau,
        "lam": cfg.lam,
        "orders": list(cfg.orders),
        "eod_id": eod_id,
        "vocab_size": vocab_size,
        "d_model": d_model,
        "shuffle_seed": cfg.shuffle_seed,
        "num_grams": int(vocab.num_grams),
        "key_hashes": key_hashes,
        "dtypes": {"accum": "fp32", "shipped": "bf16", "counts": "int64"},
    }
    payload["hash"] = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    return payload


def main():
    cfg = parse(NgramStatsConfig)
    torch.manual_seed(0)
    run_stats(cfg)


if __name__ == "__main__":
    main()
