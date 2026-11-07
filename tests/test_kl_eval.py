from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import GPT2Config, GPT2LMHeadModel

from sparsify import SaeConfig, TrainConfig, Trainer

VOCAB_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_model(n_layer: int = 2) -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=n_layer, n_embd=16, n_head=2, vocab_size=VOCAB_SIZE, n_positions=32
    )
    return GPT2LMHeadModel(config).to(DEVICE).eval()


def make_dataset(num_seqs: int, seed: int) -> Dataset:
    torch.manual_seed(seed)
    tokens = torch.randint(0, VOCAB_SIZE, (num_seqs, 32))
    return Dataset.from_dict({"input_ids": tokens.tolist()}).with_format("torch")


def make_trainer(tmp_path, model, layers=(0, 1), **kwargs) -> Trainer:
    cfg = TrainConfig(
        SaeConfig(expansion_factor=2, k=4),
        batch_size=4,
        log_to_wandb=False,
        save_dir=str(tmp_path),
        layers=list(layers),
        **kwargs,
    )
    return Trainer(cfg, make_dataset(8, seed=0), model, make_dataset(6, seed=1))


def spliced_logits(model: GPT2LMHeadModel, saes: dict, x: torch.Tensor):
    """Run GPT-2 block by block, replacing the output of each hooked block with its
    sparse coder reconstruction."""
    gpt = model.transformer
    positions = torch.arange(x.shape[1], device=x.device)
    h = gpt.wte(x) + gpt.wpe(positions)
    for i, block in enumerate(gpt.h):
        h = block(h)
        h = h[0] if isinstance(h, tuple) else h
        if f"h.{i}" in saes:
            h = saes[f"h.{i}"](h.flatten(0, 1)).sae_out.reshape(h.shape).type_as(h)
    return model.lm_head(gpt.ln_f(h))


def test_evaluate_matches_manual_splicing(tmp_path):
    model = make_model()
    trainer = make_trainer(tmp_path, model)

    kl = trainer.evaluate()

    x = trainer.eval_dataset[:]["input_ids"].to(DEVICE)
    with torch.no_grad():
        clean = model(x).logits.float().log_softmax(-1)
        spliced = spliced_logits(model, trainer.saes, x).float().log_softmax(-1)
    expected = F.kl_div(spliced, clean, log_target=True, reduction="sum") / x.numel()

    assert kl > 1e-3
    assert kl == pytest.approx(expected.item(), rel=1e-4)


def test_evaluate_is_zero_without_spliced_tokens(tmp_path):
    trainer = make_trainer(
        tmp_path, make_model(), exclude_tokens=list(range(VOCAB_SIZE))
    )
    assert trainer.evaluate() == pytest.approx(0.0, abs=1e-6)


def test_evaluate_is_zero_for_perfect_reconstruction(tmp_path):
    trainer = make_trainer(tmp_path, make_model())
    for name in trainer.saes:
        trainer.saes[name] = lambda x: SimpleNamespace(sae_out=x)
    assert trainer.evaluate() == pytest.approx(0.0, abs=1e-6)


def test_fit_evaluates_every_save(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path, make_model(), save_every=1)
    calls = []
    monkeypatch.setattr(trainer, "evaluate", lambda: calls.append(1) or 0.0)
    trainer.fit()
    assert len(calls) == 2


@pytest.mark.parametrize(
    "kwargs", [dict(init_seeds=[0, 1]), dict(distribute_modules=True)]
)
def test_unsupported_configs(tmp_path, kwargs):
    with pytest.raises(ValueError, match="KL evaluation"):
        make_trainer(tmp_path, make_model(), **kwargs)
