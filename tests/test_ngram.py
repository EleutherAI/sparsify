"""Tests for the frozen n-gram lookup library and its SAE integration."""

import numpy as np
import pytest
import torch

from sparsify import SaeConfig
from sparsify.ngram import (
    GLOBAL_ID,
    ancestor_at_order,
    annotate,
    document_boundary_suffix_len,
    pack_suffix_keys,
    select_vocab,
)
from sparsify.sparse_coder import SparseCoder


def test_document_boundary_suffix_len():
    # eod=0 begins each segment; a suffix may include the eod but not cross it.
    toks = torch.tensor([0, 1, 2, 3, 0, 4, 5])
    L = document_boundary_suffix_len(toks, max_n=4, eod_id=0)
    assert L.tolist() == [1, 2, 3, 4, 1, 2, 3]

    # 2-D: rows are independent, each starting at a fresh boundary.
    toks2d = torch.tensor([[1, 2, 3, 4], [1, 0, 2, 3]])
    L2 = document_boundary_suffix_len(toks2d, 4, 0)
    assert L2.tolist() == [[1, 2, 3, 4], [1, 1, 2, 3]]

    # max_n caps the length.
    long = torch.arange(1, 8)
    assert document_boundary_suffix_len(long, 3, 0).tolist() == [1, 2, 3, 3, 3, 3, 3]


def test_pack_suffix_keys_and_dtype():
    V = 10
    t = torch.tensor([1, 2, 3, 4])
    # key[i] = sum_k tokens[i-k] * V**k  (current token is the lowest digit)
    assert pack_suffix_keys(t, 1, V).tolist() == [1, 2, 3, 4]
    assert pack_suffix_keys(t, 2, V).tolist() == [1, 12, 23, 34]
    assert pack_suffix_keys(t, 3, V)[3].item() == 234
    # keys stay int64
    assert pack_suffix_keys(t, 4, V).dtype == torch.int64
    # 4-grams over a realistic vocab fit signed int64 (radix packing invariant).
    big = torch.tensor([49151, 49151, 49151, 49151])
    k = pack_suffix_keys(big, 4, 49152)[3].item()
    assert 0 < k < 2**63


def _repeating_corpus():
    # docs of "1 2 3" repeated, each prefixed by eod=0.
    seq = np.tile(np.array([1, 2, 3], np.int64), 300)
    stream = np.concatenate([[0], seq, [0], seq])
    return stream.reshape(1, -1)


def test_select_vocab_suffix_closed_and_counts():
    rows = _repeating_corpus()
    vocab = select_vocab(rows, vocab_size=10, orders=(1, 2, 3, 4), tau=30,
                         eod_id=0, row_chunk=1, verbose=False)
    # every unigram observed is kept (tau ignored for n=1): tokens {0,1,2,3}
    assert int((vocab.order == 1).sum()) == 4
    # parent counts dominate child counts (suffix closure / monotonicity)
    for g in range(1, vocab.num_grams):
        assert vocab.vocab_counts[vocab.parent[g]] >= vocab.vocab_counts[g]
    # GLOBAL count == total positions
    assert vocab.vocab_counts[GLOBAL_ID] == rows.size


def test_annotate_longest_match_backoff():
    rows = _repeating_corpus()
    vocab = select_vocab(rows, vocab_size=10, orders=(1, 2, 3, 4), tau=30,
                         eod_id=0, row_chunk=1, verbose=False)
    vt = vocab.to_torch("cpu")
    ids = annotate(torch.from_numpy(rows), vt)
    assert (ids >= 0).all()
    order = vocab.order[ids.numpy().reshape(-1)]
    # First position (the eod) can only be a unigram; deep inside the repetition we
    # reach the maximum order.
    assert order[0] == 1
    assert order.max() == 4
    # A token never seen during selection backs off to GLOBAL.
    unseen = torch.tensor([[9, 9, 9]])
    assert (annotate(unseen, vt) == GLOBAL_ID).all()


def test_ancestor_remap():
    rows = _repeating_corpus()
    vocab = select_vocab(rows, vocab_size=10, orders=(1, 2, 3, 4), tau=30,
                         eod_id=0, row_chunk=1, verbose=False)
    r0 = ancestor_at_order(vocab.parent, vocab.order, 0)
    assert np.all(r0 == GLOBAL_ID)
    for n in (1, 2, 3, 4):
        rn = ancestor_at_order(vocab.parent, vocab.order, n)
        assert np.all(vocab.order[rn] <= n)
        # grams already at or below n map to themselves
        keep = vocab.order <= n
        assert np.all(rn[keep] == np.arange(vocab.num_grams)[keep])


def test_fp32_accumulation_pitfall():
    """A high-count group's mean must survive accumulation; bf16 sums corrupt it."""
    torch.manual_seed(0)
    d = 32
    n = 200_000
    acts = torch.randn(n, d) + 7.0  # large offset stresses summation
    true_mean = acts.mean(0)

    fp32_sum = torch.zeros(d, dtype=torch.float32)
    for chunk in acts.split(1024):
        fp32_sum += chunk.float().sum(0)
    assert torch.allclose(fp32_sum / n, true_mean, atol=1e-3)

    bf16_sum = torch.zeros(d, dtype=torch.bfloat16)
    for chunk in acts.split(1024):
        bf16_sum += chunk.to(torch.bfloat16).sum(0).to(torch.bfloat16)
    bf16_err = (bf16_sum.float() / n - true_mean).abs().max()
    assert bf16_err > 1e-2, "bf16 accumulation should visibly corrupt the mean"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sparse_coder_gram_lookup_subtracts_and_addsback():
    torch.manual_seed(0)
    d = 16
    n = 64
    cfg = SaeConfig(gram_lookup=True, expansion_factor=4, k=4)
    sae = SparseCoder(d, cfg, device="cuda", dtype=torch.float32)

    y = torch.randn(n, d, device="cuda")
    means = torch.randn(n, d, device="cuda")
    out = sae(y, gram_means=means)
    # The residual reconstruction plus the frozen mean is the returned full recon,
    # so subtracting the mean back must equal an SAE forward on the residual target.
    resid_out = sae(y - means, gram_means=torch.zeros_like(means))
    assert torch.allclose(out.sae_out - means, resid_out.sae_out, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gram_lookup_requires_means():
    cfg = SaeConfig(gram_lookup=True, expansion_factor=4, k=4)
    sae = SparseCoder(16, cfg, device="cuda", dtype=torch.float32)
    with pytest.raises(AssertionError):
        sae(torch.randn(8, 16, device="cuda"))
