"""Frozen n-gram conditional-mean lookup tables for SAE training.

This module is the single source of truth for:

* **Key packing** - an n-gram of token ids is packed into one ``int64`` using a
  radix-``V`` mixed-base encoding (``V`` = vocabulary size). For ``n <= 4`` and
  ``V = 49152`` the largest key is ``V**4 - 1 ~= 5.8e18 < 2**63``, so everything
  fits in signed ``int64`` and we never need ``uint64`` (which torch lacks).
* **Document-boundary masking** - a gram suffix must never cross an end-of-document
  token or the start of the stream. ``document_boundary_suffix_len`` computes, per
  position, the longest suffix length that stays inside the current document.
* **Vocabulary selection** - ``select_vocab`` streams the token corpus and keeps,
  for each order ``n``, the grams whose corpus occurrence count is at least ``tau``
  (unigrams are always kept so every position matches at least its unigram).
* **Annotation** - ``annotate`` maps every position to exactly one gram id: the id
  of its longest suffix present in the vocabulary (longest-match-with-backoff).
* **GramTable** - carries the selected vocabulary plus the finalized conditional
  mean table ``mu`` and knows how to remap gram ids to a coarser max order.

Orientation note (matches the spec): vocabulary selection enumerates the *suffix*
of every position, and annotation looks up the *suffix* of every position. Counts
are monotonic under suffix truncation (``count(suffix) >= count(gram)``), so the
kept vocabulary is suffix-closed and every gram's backoff parent is guaranteed to
be present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

# Order 0 is the global mean; it is the backoff parent of every unigram.
GLOBAL_ID = 0


def _as_long_tensor(tokens) -> Tensor:
    if isinstance(tokens, np.ndarray):
        tokens = torch.from_numpy(tokens.astype(np.int64))
    elif not torch.is_tensor(tokens):
        tokens = torch.as_tensor(tokens)
    return tokens.long()


def document_boundary_suffix_len(tokens: Tensor, max_n: int, eod_id: int) -> Tensor:
    """Longest suffix length (1..max_n) ending at each position that does not cross
    an end-of-document token or the start of its sequence.

    Operates along the last axis, so ``tokens`` may be 1-D ``[S]`` (a token stream)
    or 2-D ``[B, S]`` (independent sequences, each starting at a fresh boundary).

    Documents are delimited by EOD tokens that *start* a document (the SmolLM2
    ``chunk_and_tokenize`` convention prefixes every doc with ``<eos>``). Each EOD
    therefore begins a new segment, and a suffix may reach back to *and include* the
    most recent EOD (a legitimate "document start" context token) but never past it,
    and never before the start of the sequence. Equivalently
    ``L[i] = min(max_n, i - seg_start + 1)`` where ``seg_start`` is the most recent
    EOD position at or before ``i`` (or ``0`` if there is none).
    """
    tokens = _as_long_tensor(tokens)
    s = tokens.shape[-1]
    idx = torch.arange(s, device=tokens.device)
    idx_b = idx.expand_as(tokens)

    is_eod = tokens == eod_id
    # EOD positions carry their index; everything else carries 0 (the stream start,
    # which is also a boundary). Running max gives the most recent segment start.
    seg = torch.where(is_eod, idx_b, torch.zeros_like(tokens))
    seg_start = torch.cummax(seg, dim=-1).values

    return torch.clamp(idx_b - seg_start + 1, max=max_n)


def pack_suffix_keys(tokens: Tensor, n: int, vocab_size: int) -> Tensor:
    """Radix-``vocab_size`` packed key of the length-``n`` suffix ending at each
    position, along the last axis.

    ``key[i] = sum_{k=0}^{n-1} tokens[i-k] * vocab_size**k`` (current token is the
    lowest-order digit). Positions with fewer than ``n`` valid predecessors receive
    a value that mixes in out-of-document tokens; callers must mask those out using
    :func:`document_boundary_suffix_len` (``L >= n``).
    """
    tokens = _as_long_tensor(tokens)
    assert int(tokens.max()) < vocab_size, "token id >= vocab_size; radix packing broken"
    key = tokens.clone()
    weight = 1
    for k in range(1, n):
        weight *= vocab_size
        # tokens[i-k]: shift the array right by k along the last axis.
        key[..., k:] += tokens[..., :-k] * weight
    return key


@dataclass
class GramVocab:
    """The selected n-gram vocabulary and the contiguous gram-id space.

    Gram ids are laid out as ``[GLOBAL, unigrams..., bigrams..., ...]``. For order
    ``n`` the id of a gram is ``offsets[n] + searchsorted(sorted_keys[n], key)``.
    """

    vocab_size: int
    """Radix base for key packing (the model vocabulary size)."""

    max_order: int
    tau: int
    eod_id: int

    sorted_keys: dict[int, np.ndarray]
    """order n -> sorted int64 array of packed keys kept at that order."""

    offsets: dict[int, int]
    """order n -> global id of the first gram of that order."""

    order: np.ndarray  # int8 [G]
    parent: np.ndarray  # int64 [G]  backoff parent (length n-1 suffix; GLOBAL for n=1)
    vocab_counts: np.ndarray  # int64 [G]  full corpus occurrence counts (any-suffix)

    @property
    def num_grams(self) -> int:
        return len(self.order)

    def save(self, path: str | Path):
        """Persist just the vocabulary (no means), for sharded accumulation."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {
            "order": self.order, "parent": self.parent,
            "vocab_counts": self.vocab_counts,
            **{f"keys_{n}": k for n, k in self.sorted_keys.items()},
        }
        np.savez(path / "vocab.npz", **arrays)  # type: ignore[arg-type]
        with open(path / "vocab_meta.json", "w") as f:
            json.dump({
                "vocab_size": self.vocab_size, "max_order": self.max_order,
                "tau": self.tau, "eod_id": self.eod_id, "offsets": self.offsets,
            }, f, indent=2)

    @staticmethod
    def load(path: str | Path) -> "GramVocab":
        path = Path(path)
        with open(path / "vocab_meta.json") as f:
            meta = json.load(f)
        npz = np.load(path / "vocab.npz")
        offsets = {int(k): int(v) for k, v in meta["offsets"].items()}
        return GramVocab(
            vocab_size=meta["vocab_size"], max_order=meta["max_order"],
            tau=meta["tau"], eod_id=meta["eod_id"],
            sorted_keys={n: npz[f"keys_{n}"] for n in offsets},
            offsets=offsets, order=npz["order"], parent=npz["parent"],
            vocab_counts=npz["vocab_counts"],
        )

    # -- torch-side caches for fast annotation -------------------------------
    def to_torch(self, device: str | torch.device = "cpu") -> "GramVocabTorch":
        return GramVocabTorch(
            vocab_size=self.vocab_size,
            max_order=self.max_order,
            eod_id=self.eod_id,
            sorted_keys={
                n: torch.from_numpy(k.astype(np.int64)).to(device)
                for n, k in self.sorted_keys.items()
            },
            offsets=dict(self.offsets),
        )


@dataclass
class GramVocabTorch:
    """Device-resident view used by :func:`annotate` in the hot loop."""

    vocab_size: int
    max_order: int
    eod_id: int
    sorted_keys: dict[int, Tensor]
    offsets: dict[int, int]


def annotate(tokens: Tensor, vocab: GramVocabTorch) -> Tensor:
    """Map every position to its longest-suffix gram id (backoff to shorter orders,
    finally to ``GLOBAL_ID``). Works on CPU or CUDA; ``tokens`` is ``[..., S]``.
    """
    tokens = _as_long_tensor(tokens).to(
        next(iter(vocab.sorted_keys.values())).device
    )
    lengths = document_boundary_suffix_len(tokens, vocab.max_order, vocab.eod_id)

    gram_ids = torch.full_like(tokens, -1)
    # Ascending order so the longest present suffix wins (later writes override).
    for n in range(1, vocab.max_order + 1):
        keys = vocab.sorted_keys.get(n)
        if keys is None or keys.numel() == 0:
            continue
        packed = pack_suffix_keys(tokens, n, vocab.vocab_size)
        pos = torch.searchsorted(keys, packed)
        pos_clamped = pos.clamp(max=keys.numel() - 1)
        hit = (lengths >= n) & (keys[pos_clamped] == packed)
        ids_n = pos + vocab.offsets[n]
        gram_ids = torch.where(hit, ids_n, gram_ids)

    # Any position that matched nothing (a token unseen during selection) backs off
    # all the way to the global mean.
    gram_ids = torch.where(gram_ids < 0, torch.zeros_like(gram_ids), gram_ids)
    return gram_ids


def _merge_sorted_counts(
    run_keys: np.ndarray,
    run_counts: np.ndarray,
    new_keys: np.ndarray,
    new_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum-by-key merge of two (keys, counts) pairs into a sorted unique pair."""
    all_keys = np.concatenate([run_keys, new_keys])
    all_counts = np.concatenate([run_counts, new_counts])
    uniq, inv = np.unique(all_keys, return_inverse=True)
    # counts stay exact in float64 (< 2**53) and are cast back to int64.
    summed = np.bincount(inv, weights=all_counts, minlength=len(uniq))
    return uniq, summed.astype(np.int64)


def select_vocab(
    token_rows,
    *,
    vocab_size: int,
    orders: Sequence[int] = (1, 2, 3, 4),
    tau: int = 30,
    eod_id: int = 0,
    max_rows: int | None = None,
    row_chunk: int = 20_000,
    verbose: bool = True,
) -> GramVocab:
    """Select, per order, the grams with corpus occurrence count >= ``tau`` by
    streaming ``token_rows`` (an ``[N, S]`` int array / memmap of independent
    sequences). Unigrams ignore ``tau`` (all observed tokens are kept) so every
    position is guaranteed a match.

    Each row is treated as a self-contained sequence (its start is a boundary and
    grams never cross an EOD inside it), matching how the trainer annotates batches,
    so nothing leaks across the arbitrary row cuts of a shuffled dataset.

    Pure numpy, deterministic memory: each row block is uniqued and merged into a
    running (keys, counts) pair per order.
    """
    max_order = max(orders)
    orders = sorted(orders)
    n_rows = len(token_rows)
    if max_rows is not None:
        n_rows = min(max_rows, n_rows)

    running: dict[int, tuple[np.ndarray, np.ndarray]] = {
        n: (np.empty(0, np.int64), np.empty(0, np.int64)) for n in orders
    }

    for r0 in range(0, n_rows, row_chunk):
        r1 = min(r0 + row_chunk, n_rows)
        block = np.asarray(token_rows[r0:r1])
        block_t = torch.from_numpy(np.ascontiguousarray(block).astype(np.int64))

        lengths = document_boundary_suffix_len(block_t, max_order, eod_id).numpy()
        for n in orders:
            keys = pack_suffix_keys(block_t, n, vocab_size).numpy()
            sel = keys[lengths >= n]
            if sel.size == 0:
                continue
            uk, uc = np.unique(sel, return_counts=True)
            running[n] = _merge_sorted_counts(
                running[n][0], running[n][1], uk, uc.astype(np.int64)
            )

        if verbose:
            pct = 100.0 * r1 / n_rows
            sizes = {n: len(running[n][0]) for n in orders}
            print(f"  select_vocab rows {r1:,}/{n_rows:,} ({pct:.1f}%) "
                  f"distinct-so-far={sizes}", flush=True)

    # Apply the threshold (unigrams keep everything so every position matches).
    kept: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n in orders:
        uk, uc = running[n]
        thresh = 1 if n == 1 else tau
        mask = uc >= thresh
        kept[n] = (uk[mask], uc[mask])

    return _assemble_vocab(kept, vocab_size, max_order, tau, eod_id, orders, verbose)


def _assemble_vocab(kept, vocab_size, max_order, tau, eod_id, orders, verbose):
    # Global-id layout: [GLOBAL, order-1 grams, order-2 grams, ...].
    offsets: dict[int, int] = {}
    sorted_keys: dict[int, np.ndarray] = {}
    order_list = [np.array([0], np.int8)]  # GLOBAL has order 0
    parent_list = [np.array([GLOBAL_ID], np.int64)]  # GLOBAL is its own parent
    counts_list = [np.array([0], np.int64)]  # filled after (sum of unigram counts)

    next_id = 1
    for n in orders:
        keys, counts = kept[n]
        offsets[n] = next_id
        sorted_keys[n] = keys
        order_list.append(np.full(len(keys), n, np.int8))
        counts_list.append(counts.astype(np.int64))
        next_id += len(keys)

    # Backoff parents: parent key = key % vocab_size**(n-1) (drop earliest token).
    for n in orders:
        keys = sorted_keys[n]
        if n == 1:
            parents = np.zeros(len(keys), np.int64)  # -> GLOBAL
        else:
            parent_keys = keys % (vocab_size ** (n - 1))
            pk = sorted_keys[n - 1]
            pos = np.searchsorted(pk, parent_keys)
            # Suffix-closure guarantees the parent is present.
            assert np.all(pos < len(pk)) and np.all(pk[pos] == parent_keys), (
                f"backoff parent missing for order {n}; vocabulary not suffix-closed"
            )
            parents = (offsets[n - 1] + pos).astype(np.int64)
        parent_list.append(parents)

    order = np.concatenate(order_list)
    parent = np.concatenate(parent_list)
    vocab_counts = np.concatenate(counts_list)
    # GLOBAL count = total observed unigram occurrences.
    vocab_counts[GLOBAL_ID] = int(vocab_counts[order == 1].sum())

    vocab = GramVocab(
        vocab_size=vocab_size,
        max_order=max_order,
        tau=tau,
        eod_id=eod_id,
        sorted_keys=sorted_keys,
        offsets=offsets,
        order=order,
        parent=parent,
        vocab_counts=vocab_counts,
    )
    if verbose:
        _print_vocab_report(vocab, orders)
    return vocab


def _print_vocab_report(vocab: GramVocab, orders):
    print("\n=== n-gram vocabulary ===")
    total = int(vocab.vocab_counts[GLOBAL_ID])
    for n in orders:
        g = int((vocab.order == n).sum())
        occ = int(vocab.vocab_counts[vocab.order == n].sum())
        cov = 100.0 * occ / total if total else 0.0
        print(f"  order {n}: G={g:,}  total occurrences={occ:,} "
              f"(any-suffix coverage {cov:.1f}%)")
    print(f"  total grams G={vocab.num_grams:,} (incl. GLOBAL)")


def ancestor_at_order(parent: np.ndarray, order: np.ndarray, max_order: int) -> np.ndarray:
    """For every gram id, the id of its ancestor of order exactly ``max_order``
    (or itself if its own order is already <= ``max_order``). Used both to build the
    ``lookup_max_order`` remap and the R^2(n) restricted grouping.
    """
    remap = np.arange(len(parent), dtype=np.int64)
    # Walk parents up while order is too high. Depth is bounded by max n (<=4).
    changed = True
    while changed:
        too_deep = order[remap] > max_order
        if not np.any(too_deep):
            break
        remap = np.where(too_deep, parent[remap], remap)
    return remap


@dataclass
class GramTable:
    """A selected vocabulary plus the finalized conditional-mean table.

    ``mu[g]`` is ``E[activation | longest-suffix gram is an order-<=order[g]
    ancestor]`` after shrinkage, one d_model vector per gram id. Look-ups return a
    d_model vector; ``remap_to_order`` collapses the id space to a coarser max order
    for the ``lookup_max_order`` ablation.
    """

    vocab: GramVocab
    d_model: int
    hookpoint: str
    mu: np.ndarray  # fp32 [G, d_model] master (shrunk)
    counts: np.ndarray  # int64 [G] aggregated (== vocab_counts after finalize)
    manifest: dict

    def remap_to_order(self, max_order: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(remap, mu_eff)`` where ``remap[g]`` is the order-<=max_order
        ancestor id and ``mu_eff = mu[remap]`` so that ``mu_eff[g]`` is the mean of
        the coarser bucket. ``max_order == 0`` yields the global mean everywhere.
        """
        remap = ancestor_at_order(self.vocab.parent, self.vocab.order, max_order)
        return remap, self.mu[remap]

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {
            "order": self.vocab.order,
            "parent": self.vocab.parent,
            "vocab_counts": self.vocab.vocab_counts,
            "counts": self.counts,
            **{f"keys_{n}": k for n, k in self.vocab.sorted_keys.items()},
        }
        np.savez(path / "vocab.npz", **arrays)  # type: ignore[arg-type]
        np.save(path / "mu_fp32.npy", self.mu.astype(np.float32))
        np.save(path / "mu_bf16.npy",
                torch.from_numpy(self.mu).bfloat16().view(torch.int16).numpy())
        meta = {
            "vocab_size": self.vocab.vocab_size,
            "max_order": self.vocab.max_order,
            "tau": self.vocab.tau,
            "eod_id": self.vocab.eod_id,
            "offsets": self.vocab.offsets,
            "d_model": self.d_model,
            "hookpoint": self.hookpoint,
            "manifest": self.manifest,
        }
        with open(path / "table.json", "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(path: str | Path) -> "GramTable":
        path = Path(path)
        with open(path / "table.json") as f:
            meta = json.load(f)
        npz = np.load(path / "vocab.npz")
        offsets = {int(k): int(v) for k, v in meta["offsets"].items()}
        sorted_keys = {n: npz[f"keys_{n}"] for n in offsets}
        vocab = GramVocab(
            vocab_size=meta["vocab_size"],
            max_order=meta["max_order"],
            tau=meta["tau"],
            eod_id=meta["eod_id"],
            sorted_keys=sorted_keys,
            offsets=offsets,
            order=npz["order"],
            parent=npz["parent"],
            vocab_counts=npz["vocab_counts"],
        )
        mu = np.load(path / "mu_fp32.npy")
        return GramTable(
            vocab=vocab,
            d_model=meta["d_model"],
            hookpoint=meta["hookpoint"],
            mu=mu,
            counts=npz["counts"],
            manifest=meta["manifest"],
        )
