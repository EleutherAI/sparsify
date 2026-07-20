"""Stage-6 evaluation harness for the frozen n-gram lookup ablation ladder.

For one or more trained sparse coders (baseline, embed_skip, or ``gram_lookup`` at
various ``lookup_max_order``), all on the same model + hookpoint, this reports on a
held-out slice, per arm:

* **FVU** on the raw activation (n-gram mean added back) and on the residual scale, so
  arms trained on different targets are comparable on the same yardstick.
* **CE-added** - the increase in cross-entropy when the reconstruction is patched into
  the model at the hookpoint (this needs live forwards).
* **Token-triviality** - per-latent mutual information between a latent firing and the
  order-1 gram (token identity), and the fraction of live latents whose firing is >90%
  explained by the token alone. This is the metric that says whether the lookup removed
  dataset-geometry latents rather than merely improving reconstruction.

The Stage-4 R^2(n) table (from ``rsquared.json`` beside the table) is printed alongside.

    python -m sparsify.ngram_eval HuggingFaceTB/SmolLM2-135M EleutherAI/SmolLM2-135M-10B \\
        --hookpoint layers.9 --gram_table_path ngram_tables/smoke_135m/layers.9 \\
        --sae_paths checkpoints/base/layers.9 checkpoints/lookup4/layers.9
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from simple_parsing import Serializable, field as sp_field, list_field, parse
from transformers import AutoModelForCausalLM

from .ngram import GLOBAL_ID, GramTable, annotate
from .ngram_stats import load_model_and_rows
from .sparse_coder import SparseCoder

print = functools.partial(print, flush=True)  # noqa: A001


@dataclass
class NgramEvalConfig(Serializable):
    model: str = sp_field(default="HuggingFaceTB/SmolLM2-135M", positional=True)
    dataset: str = sp_field(default="EleutherAI/SmolLM2-135M-10B", positional=True)

    hookpoint: str = "layers.9"
    sae_paths: list[str] = list_field()
    """One directory per arm (the hookpoint subdir of a training run)."""

    gram_table_path: str | None = None
    """n-gram table dir (needed for gram_lookup arms and the token-triviality metric)."""

    output_dir: str | None = None

    split: str = "train"
    ctx_len: int = 2048
    eval_tokens: int = 2_000_000
    batch_size: int = 8

    triviality_max_tokens: int = 500_000
    """Cap tokens used for the per-latent x token fire-count table (the [L,V] scatter is
    the eval's slowest step). FVU/CE still use the full eval_tokens. 0 disables it."""

    eod_id: int | None = None
    revision: str | None = None
    hf_token: str | None = sp_field(default=None, encoding_fn=lambda _: None)
    text_column: str = "text"
    data_args: str = ""
    shuffle_seed: int = 42
    max_docs: int | None = None
    data_preprocessing_num_proc: int = 8
    vocab_size: int | None = None


@torch.no_grad()
def evaluate(cfg: NgramEvalConfig):
    device = "cuda:0"
    # Reuse the stats loader for the tokenized rows + tokenizer, but we need an LM head
    # for CE, so load a causal LM (its base_model carries the same hookpoints).
    _, tokenizer, rows = load_model_and_rows(  # type: ignore[arg-type]
        _EvalAsStatsCfg(cfg)
    )
    lm = AutoModelForCausalLM.from_pretrained(
        cfg.model, device_map={"": device},
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        revision=cfg.revision, token=cfg.hf_token,
    )
    lm.eval()

    vocab_size = cfg.vocab_size or len(tokenizer)
    n_total = len(rows)
    eval_rows = max(1, min(n_total, (cfg.eval_tokens + cfg.ctx_len - 1) // cfg.ctx_len))
    h0 = n_total - eval_rows  # tail slice (disjoint from a front-loaded fit)
    print(f"Evaluating {len(cfg.sae_paths)} arm(s) on rows [{h0:,},{n_total:,}) "
          f"(~{eval_rows*cfg.ctx_len/1e6:.1f}M tok) @ {cfg.hookpoint}")

    table = GramTable.load(cfg.gram_table_path) if cfg.gram_table_path else None
    vocab_t = table.vocab.to_torch(device) if table is not None else None
    global_mu = (torch.from_numpy(table.mu[GLOBAL_ID]).to(device).float()
                 if table is not None else None)

    arms = _load_arms(cfg, table, device)
    stats = {name: _Accum(sae.num_latents, vocab_size, cfg.triviality_max_tokens)
             for name, (sae, _) in arms.items()}

    layer = lm.base_model.get_submodule(cfg.hookpoint)
    captured: dict = {}
    handles = [layer.register_forward_hook(
        lambda m, i, o: captured.__setitem__("y", o[0] if isinstance(o, tuple) else o)
    )]
    needs_embed = any(sae.cfg.embed_skip for sae, _ in arms.values())
    if needs_embed:
        handles.append(lm.get_input_embeddings().register_forward_hook(
            lambda m, i, o: captured.__setitem__("embed", o)))
    try:
        for r0 in range(h0, n_total, cfg.batch_size):
            r1 = min(r0 + cfg.batch_size, n_total)
            x = torch.from_numpy(np.asarray(rows[r0:r1]).astype(np.int64)).to(device)
            clean_loss = lm(x, labels=x).loss.item()
            y = captured["y"].flatten(0, 1).float()
            embed = captured["embed"].flatten(0, 1) if needs_embed else None

            gram_ids = annotate(x, vocab_t).flatten() if vocab_t is not None else None
            tokens_flat = x.flatten()

            for name, (sae, mu) in arms.items():
                gmeans = (mu[gram_ids.to(mu.device)].to(device).float()
                          if sae.cfg.gram_lookup else None)
                sae_embed = embed if sae.cfg.embed_skip else None
                out = sae(y, embed=sae_embed, gram_means=gmeans)
                recon = out.sae_out.float()
                ce_dirty = _ce_with_patch(lm, x, cfg.hookpoint, recon)
                stats[name].update(
                    y, recon, gmeans, global_mu, out.latent_indices,
                    tokens_flat, clean_loss, ce_dirty, x.numel(),
                )
    finally:
        for h in handles:
            h.remove()

    return _report(cfg, arms, stats, table)


class _EvalAsStatsCfg:
    """Adapter so ``load_model_and_rows`` (typed for NgramStatsConfig) can consume the
    eval config. Only the attributes that loader touches are forwarded."""

    def __init__(self, cfg: NgramEvalConfig):
        self._c = cfg

    def __getattr__(self, name):
        return getattr(self._c, name)


def _load_arms(cfg, table, device):
    arms = {}
    for p in cfg.sae_paths:
        sae = SparseCoder.load_from_disk(p, device=device)
        name = Path(p).parent.name if Path(p).name == cfg.hookpoint else Path(p).name
        mu = None
        if sae.cfg.gram_lookup:
            assert table is not None, f"arm '{name}' is gram_lookup but no --gram_table_path"
            _, mu_eff = table.remap_to_order(sae.cfg.lookup_max_order)
            mu = torch.from_numpy(mu_eff).to(torch.bfloat16)
        arms[name] = (sae, mu)
        print(f"  arm '{name}': gram_lookup={sae.cfg.gram_lookup} "
              f"order={sae.cfg.lookup_max_order if sae.cfg.gram_lookup else '-'} "
              f"embed_skip={sae.cfg.embed_skip} k={sae.cfg.k} L={sae.num_latents}")
    return arms


@torch.no_grad()
def _ce_with_patch(lm, x, hookpoint, recon_flat) -> float:
    layer = lm.base_model.get_submodule(hookpoint)

    def patch(module, inputs, outputs):
        is_tuple = isinstance(outputs, tuple)
        base = outputs[0] if is_tuple else outputs
        new = recon_flat.view(base.shape).to(base.dtype)
        return (new, *outputs[1:]) if is_tuple else new

    h = layer.register_forward_hook(patch)
    try:
        return lm(x, labels=x).loss.item()
    finally:
        h.remove()


class _Accum:
    """Running eval metrics for one arm, including per-latent x token fire counts."""

    def __init__(self, num_latents: int, vocab_size: int, tri_cap: int = 500_000):
        self.raw_num = self.raw_den = 0.0
        self.res_num = self.res_den = 0.0
        self.ce_dirty_sum = self.ce_clean_sum = 0.0
        self.numel_tot = 0
        self.n_tok = 0
        self.n_tri_tok = 0
        self.tri_cap = tri_cap
        self.L = num_latents
        self.V = vocab_size
        # fire[j, v] = #tokens with token id v where latent j is in the top-k
        self.fire_lt = np.zeros((num_latents, vocab_size), np.int32)
        self.token_tot = np.zeros(vocab_size, np.int64)
        self.fire_tot = np.zeros(num_latents, np.int64)

    def update(self, y, recon, gmeans, global_mu, latent_idx, tokens, ce_clean,
               ce_dirty, numel):
        self.raw_num += (y - recon).pow(2).sum().item()
        gmean = global_mu if global_mu is not None else y.mean(0)
        self.raw_den += (y - gmean).pow(2).sum().item()
        if gmeans is not None:
            resid, resid_rec = y - gmeans, recon - gmeans
            self.res_num += (resid - resid_rec).pow(2).sum().item()
            self.res_den += (resid - resid.mean(0)).pow(2).sum().item()
        else:
            self.res_num += (y - recon).pow(2).sum().item()
            self.res_den += (y - y.mean(0)).pow(2).sum().item()
        self.ce_dirty_sum += ce_dirty * numel
        self.ce_clean_sum += ce_clean * numel
        self.numel_tot += numel
        self.n_tok += y.shape[0]

        # Token-triviality counts (vectorized): for each fired (token, latent) pair,
        # bump fire_lt[latent, token]. Capped in token count since this [L,V] scatter
        # is the eval's slowest step.
        if self.tri_cap and self.n_tri_tok < self.tri_cap:
            tok = tokens.cpu().numpy()
            self.token_tot += np.bincount(tok, minlength=self.V)
            idx = latent_idx.reshape(latent_idx.shape[0], -1).cpu().numpy()  # [N, k]
            lat = idx.reshape(-1)
            tok_rep = np.repeat(tok, idx.shape[1])
            np.add.at(self.fire_lt, (lat, tok_rep), 1)
            self.fire_tot += np.bincount(lat, minlength=self.L)
            self.n_tri_tok += len(tok)

    @property
    def ce_added(self) -> float:
        if self.numel_tot == 0:
            return 0.0
        return (self.ce_dirty_sum - self.ce_clean_sum) / self.numel_tot


def _token_triviality(acc: _Accum) -> dict:
    n = acc.n_tri_tok
    if n == 0:
        return {}
    p_fire = np.clip(acc.fire_tot / n, 1e-12, 1 - 1e-12)
    h_fire = -(p_fire * np.log2(p_fire) + (1 - p_fire) * np.log2(1 - p_fire))  # [L]
    # H(fire_j | token) = sum_v p(v) H(fire_j | v)
    pv = acc.token_tot / n  # [V]
    p_fire_given = acc.fire_lt / np.clip(acc.token_tot, 1, None)  # [L, V]
    p_fire_given = np.clip(p_fire_given, 1e-12, 1 - 1e-12)
    h_cond_per = -(p_fire_given * np.log2(p_fire_given)
                   + (1 - p_fire_given) * np.log2(1 - p_fire_given))  # [L, V]
    h_cond = (h_cond_per * pv[None, :]).sum(1)  # [L]
    mi = np.clip(h_fire - h_cond, 0, None)
    frac = mi / np.clip(h_fire, 1e-12, None)
    alive = acc.fire_tot > 0
    return {
        "mean_mi_order1": float(mi[alive].mean()) if alive.any() else 0.0,
        "frac_trivial_order1": float((frac[alive] > 0.9).mean()) if alive.any() else 0.0,
        "num_alive": int(alive.sum()),
        "dead_frac": float((~alive).mean()),
    }


def _report(cfg, arms, stats, table):
    rsq = None
    if cfg.gram_table_path:
        rq = Path(cfg.gram_table_path).parent / "rsquared.json"
        if rq.exists():
            rsq = json.load(open(rq))

    print("\n================ EVAL SUMMARY ================")
    if rsq is not None:
        by = rsq.get(cfg.hookpoint) or next(iter(rsq.values()))
        r2s = {int(k): v["r2"] for k, v in by["by_order"].items()}
        print("R^2(n) null model:  " +
              "  ".join(f"n<={n}:{r2s[n]:.3f}" for n in sorted(r2s)))

    report = {}
    print(f"{'arm':<20}{'FVU_raw':>9}{'FVU_res':>9}{'CEadd':>9}"
          f"{'dead%':>7}{'triv%':>7}{'MI':>7}")
    for name, (sae, _) in arms.items():
        acc = stats[name]
        fvu_raw = acc.raw_num / acc.raw_den
        fvu_res = acc.res_num / acc.res_den
        tri = _token_triviality(acc) if table is not None else {}
        report[name] = {
            "fvu_raw": fvu_raw, "fvu_resid": fvu_res, "ce_added": acc.ce_added,
            "gram_lookup": sae.cfg.gram_lookup,
            "lookup_max_order": sae.cfg.lookup_max_order,
            "embed_skip": sae.cfg.embed_skip, **tri,
        }
        print(f"{name:<20}{fvu_raw:>9.4f}{fvu_res:>9.4f}{acc.ce_added:>9.4f}"
              f"{tri.get('dead_frac', 0)*100:>7.1f}"
              f"{tri.get('frac_trivial_order1', 0)*100:>7.1f}"
              f"{tri.get('mean_mi_order1', 0):>7.3f}")

    if cfg.output_dir:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(cfg.output_dir) / "eval.json", "w") as f:
            json.dump(report, f, indent=2)
    return report


def main():
    cfg = parse(NgramEvalConfig)
    evaluate(cfg)


if __name__ == "__main__":
    main()
