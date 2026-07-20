# Frozen n-gram lookup baselines for SAE training

A ladder of frozen (not jointly trained) null models — unigram → bigram → trigram →
4-gram, longest-suffix-with-backoff — whose **conditional-mean activations** are
subtracted from the SAE target. Whatever the frozen table explains is provably *not*
SAE computation but local-context/token geometry. Generalizes the `embed_skip` arm and
the Tokenized-SAE per-token bias (Dooms & Wilhelm, arXiv:2502.17332) into a tunable
`lookup_max_order ∈ {0,1,2,3,4}`.

Code: `sparsify/ngram.py` (library), `sparsify/ngram_stats.py` (build the table),
`sparsify/ngram_eval.py` (Stage-6 eval); `gram_lookup` wired into
`SparseCoder.forward` and `Trainer` alongside `embed_skip` (kept untouched).
Tests: `tests/test_ngram.py`. Helper scripts: `scripts/run_ngram_stats_sharded.sh`,
`scripts/run_ladder.sh`. Use `/opt/conda/bin/python` (GPU); run from the repo root.

## 1. Build the conditional-mean table (statistics pass)

One forward pass accumulates fp32 sum/sum-of-squares/count of activations grouped by
each position's longest-suffix gram id, then aggregates bottom-up over the backoff tree,
shrinks each mean toward its parent (λ), and reports the headline **R²(n) ANOVA** — the
fraction of activation variance explained by conditioning on the suffix n-gram, per order
n, *before any SAE is trained*.

Single process:

```bash
/opt/conda/bin/python -m sparsify.ngram_stats HuggingFaceTB/SmolLM2-1.7B \
    EleutherAI/SmolLM2-135M-10B --hookpoints layers.17 \
    --max_tokens 200_000_000 --tau 30 --output_dir ngram_tables/smollm17_l17
```

Sharded across N GPUs (vocab once → N accum shards in parallel → merge):

```bash
bash scripts/run_ngram_stats_sharded.sh HuggingFaceTB/SmolLM2-1.7B \
    <tokenized_corpus> layers.17 ngram_tables/smollm17_l17 200000000 8 \
    --accum_device cuda:0 --batch_size 8
```

Sizing note: fp32 accumulators are `2·G·d_model·4` bytes and the shipped bf16 table is
`G·d_model·2`. `G` grows with the token budget (τ=30, SmolLM2 corpus): ~0.1M @10M tok,
~0.76M @100M, ~1.5M @200M, ~2.2M @300M. At `d_model=2048` a full-10B build would exceed
both a 48 GB GPU and 754 GB RAM, so pick a budget whose accumulators fit
(200M → 24 GB, GPU-resident 8-way sharding). Sanity checks run at the end:
(a) aggregated counts == vocab counts (exact), (b) index_add mean == brute force,
(c) held-out FVU ≈ 1 − R²(max order).

## 2. Train the ablation ladder

`gram_lookup` subtracts `μ[gram_id]` from the target (and, for autoencoders, from the
encoder input); reconstruction adds it back. `lookup_max_order=0` subtracts only the
global mean, `1` a per-token bias, `2–4` the longest-suffix table restricted to that order.

```bash
# one arm:
/opt/conda/bin/python -m sparsify HuggingFaceTB/SmolLM2-1.7B <corpus> \
    --hookpoints layers.17 -k 32 --expansion_factor 32 \
    --gram_lookup True --lookup_max_order 4 \
    --gram_table_path ngram_tables/smollm17_l17/layers.17 --run_name l17_lookup4

# full ladder (base, embed_skip, lookup 0..4) one arm per GPU:
bash scripts/run_ladder.sh HuggingFaceTB/SmolLM2-1.7B <corpus> layers.17 \
    ngram_tables/smollm17_l17/layers.17 32 32 50000 l17
```

## 3. Evaluate (Stage 6)

```bash
/opt/conda/bin/python -m sparsify.ngram_eval HuggingFaceTB/SmolLM2-1.7B <corpus> \
    --hookpoint layers.17 --gram_table_path ngram_tables/smollm17_l17/layers.17 \
    --sae_paths checkpoints/l17_base/layers.17 checkpoints/l17_lookup4/layers.17 ...
```

Reports per arm: FVU on the raw activation (μ added back) and on the residual scale,
CE-added (reconstruction patched into the model), and token-triviality (per-latent MI
between firing and the order-1 gram; fraction of latents >90% explained by the token),
alongside the R²(n) table.

## Headline R²(n): SmolLM2-1.7B, layers.17 (200M-token table, τ=30)

n≤1 (per-token) **0.757**, n≤2 0.767, n≤3 0.769, n≤4 0.769. A frozen per-token mean
already explains ~76% of the layer-17 activation variance; bigrams and beyond add ~1%.
G=1,489,603. Built GPU-resident 8-way sharded in ~20 min. (Ladder training + eval at
this scale run via `scripts/run_1p7b_ladder_and_eval.sh`.)

## Shakeout result (SmolLM2-135M, layers.9, 10M-token table, 500-step arms)

R²(n): n≤1 **0.66**, n≤2 0.73, n≤3 0.74, n≤4 0.74 (a per-token mean already explains 66%
of layer-9 variance; diminishing returns past bigrams).

| arm | FVU_raw | FVU_res | CE-added | dead% | trivial% | MI |
|---|---|---|---|---|---|---|
| base | 0.164 | 0.165 | 0.201 | 2.4 | 3.1 | 0.018 |
| lookup0 (global) | 0.173 | 0.174 | 0.233 | 8.7 | 3.9 | 0.017 |
| lookup1 | 0.146 | 0.363 | 0.187 | 17.1 | 1.4 | 0.008 |
| lookup2 | 0.139 | 0.402 | 0.176 | 17.7 | 1.8 | 0.007 |
| lookup3 | 0.138 | 0.409 | 0.178 | 17.5 | 1.7 | 0.007 |
| lookup4 | 0.138 | 0.408 | 0.176 | 17.3 | 2.0 | 0.007 |
| embed_skip | 0.156 | 0.156 | 0.178 | 2.4 | 3.1 | 0.017 |

Reading: lookup arms reconstruct the **raw** activation better than baseline/embed_skip
(FVU_raw 0.138 vs 0.164) and lower CE-added, while their residual target is harder
(FVU_res up). Crucially they carry **fewer token-trivial latents** (trivial% 1.4–2.0 vs
3.1, MI 0.007 vs 0.018) — the lookup removes token-geometry latents, which is the point.
(Arms are 500-step undertrained shakeouts; absolute numbers are rough, the ordering is
the signal.)
