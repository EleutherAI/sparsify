import sys
import types

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import GPT2Config, GPT2LMHeadModel

from sparsify import SaeConfig, TrainConfig, Trainer

VOCAB_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_model(n_layer: int = 1, n_positions: int = 128) -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=n_layer,
        n_embd=16,
        n_head=2,
        vocab_size=VOCAB_SIZE,
        n_positions=n_positions,
    )
    return GPT2LMHeadModel(config).to(DEVICE)


def make_dataset(num_seqs: int = 64, seq_len: int = 128) -> Dataset:
    torch.manual_seed(0)
    tokens = torch.randint(0, VOCAB_SIZE, (num_seqs, seq_len))
    return Dataset.from_dict({"input_ids": tokens.tolist()}).with_format("torch")


def make_config(tmp_path, multi_topk: bool = False, **kwargs) -> TrainConfig:
    kwargs.setdefault("log_to_wandb", False)
    return TrainConfig(
        SaeConfig(expansion_factor=2, k=4, multi_topk=multi_topk),
        batch_size=16,
        save_dir=str(tmp_path),
        layers=[0],
        lr=1e-3,
        **kwargs,
    )


def train_deltas(tmp_path, micro_acc_steps: int) -> dict[str, torch.Tensor]:
    """Train one run and return each SAE's weight update."""
    trainer = Trainer(
        make_config(tmp_path, micro_acc_steps=micro_acc_steps),
        make_dataset(),
        make_model(),
    )
    initial = {
        name: sae.encoder.weight.detach().clone() for name, sae in trainer.saes.items()
    }
    trainer.fit()
    return {
        name: (sae.encoder.weight.detach() - initial[name]).flatten()
        for name, sae in trainer.saes.items()
    }


def first_step_grads(tmp_path, *, grad_acc_steps: int = 1, **kwargs):
    """Train up to the first optimizer step and return every SAE parameter's grad."""
    # One batch per accumulation step, so there is exactly one optimizer step.
    trainer = Trainer(
        make_config(tmp_path, grad_acc_steps=grad_acc_steps, **kwargs),
        make_dataset(num_seqs=16 * grad_acc_steps),
        make_model(),
    )

    if kwargs.get("auxk_alpha", 0.0) > 0:
        # Nothing is dead on step 0, so mark half the latents dead by hand to make
        # sure the AuxK branch actually runs.
        for counts in trainer.num_tokens_since_fired.values():
            counts[: counts.numel() // 2] = trainer.cfg.dead_feature_threshold + 1

    grads: dict[str, torch.Tensor] = {}

    def capture(optimizer, args, kw):
        if grads:  # Only the first step
            return

        for name, sae in trainer.saes.items():
            for pname, p in sae.named_parameters():
                if p.grad is not None:
                    grads[f"{name}.{pname}"] = p.grad.detach().clone().flatten()

    for optimizer in trainer.optimizers:
        optimizer.register_step_pre_hook(capture)

    trainer.fit()
    return grads


class FakeWandb(types.ModuleType):
    """Minimal stand-in for `wandb` that records what the trainer logs."""

    def __init__(self):
        super().__init__("wandb")
        self.logs: list[dict] = []

    def init(self, *args, **kwargs):
        pass

    def log(self, info, step=None):
        self.logs.append(dict(info))


def logged_fvu(tmp_path, monkeypatch, micro_acc_steps: int) -> dict[str, float]:
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)

    trainer = Trainer(
        make_config(tmp_path, micro_acc_steps=micro_acc_steps, log_to_wandb=True),
        make_dataset(num_seqs=16),
        make_model(),
    )
    trainer.fit()

    assert len(fake.logs) == 1
    return {k: v for k, v in fake.logs[0].items() if k.startswith("fvu/")}


def test_micro_acc_steps_matches_unchunked_update(tmp_path):
    # Chunking must not change the optimizer step. Every chunk is normalized by the
    # variance of the full batch, so the chunked update matches the unchunked one up
    # to float noise.
    #
    # NOTE: Adam's first step is roughly lr * sign(grad), so this test cannot see a
    # wrong gradient *scale*. test_micro_acc_steps_gradients_match_unchunked does.
    unchunked = train_deltas(tmp_path / "a", micro_acc_steps=1)
    chunked = train_deltas(tmp_path / "b", micro_acc_steps=2)

    assert set(unchunked) == set(chunked)
    for name, d1 in unchunked.items():
        d2 = chunked[name]
        assert d1.norm() > 1e-6, f"{name} did not move; test would be vacuous"

        # Compare the updates relative to the size of the update itself.
        rel_gap = (d2 - d1).norm() / d1.norm()
        cos = F.cosine_similarity(d1, d2, dim=0)
        assert rel_gap < 0.01, f"{name} update differs by {rel_gap:.3f} of its norm"
        assert cos > 0.999, f"{name} update direction diverged (cos={cos:.4f})"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(micro_acc_steps=2),
        dict(micro_acc_steps=3),  # Uneven chunks
        dict(micro_acc_steps=8),
        dict(micro_acc_steps=2, grad_acc_steps=2),
        dict(micro_acc_steps=3, multi_topk=True),
        dict(micro_acc_steps=3, auxk_alpha=1 / 32),
    ],
    ids=lambda kw: ",".join(f"{k}={v}" for k, v in kw.items()),
)
def test_micro_acc_steps_gradients_match_unchunked(tmp_path, kwargs):
    # The gradients of a chunked step must equal those of an unchunked step. Compared
    # before the optimizer runs, because Adam, Muon and Signum all hide a constant
    # gradient scale (e.g. dividing the loss by `micro_acc_steps` twice).
    unchunked_kwargs = {k: v for k, v in kwargs.items() if k != "micro_acc_steps"}
    unchunked = first_step_grads(tmp_path / "a", **unchunked_kwargs)
    chunked = first_step_grads(tmp_path / "b", **kwargs)

    assert unchunked and set(unchunked) == set(chunked)
    for name, g1 in unchunked.items():
        g2 = chunked[name]
        assert g1.norm() > 1e-8, f"{name} has no gradient; test would be vacuous"

        ratio = g2.norm() / g1.norm()
        cos = F.cosine_similarity(g1, g2, dim=0)
        assert abs(ratio - 1) < 0.05, f"{name} gradient scaled by {ratio:.3f}"
        assert cos > 0.99, f"{name} gradient direction diverged (cos={cos:.4f})"


def test_micro_acc_steps_logged_fvu_matches_unchunked(tmp_path, monkeypatch):
    # The FVU sent to wandb should not depend on how the batch was chunked.
    unchunked = logged_fvu(tmp_path / "a", monkeypatch, micro_acc_steps=1)
    chunked = logged_fvu(tmp_path / "b", monkeypatch, micro_acc_steps=4)

    assert unchunked and set(unchunked) == set(chunked)
    for name, fvu in unchunked.items():
        assert chunked[name] == pytest.approx(fvu, rel=0.05), name


def test_micro_acc_steps_actually_trains(tmp_path):
    # Guards the regression this file exists for: micro_acc_steps was silently
    # ignored (it only scaled the loss denominator), so chunking was a no-op.
    trainer = Trainer(
        make_config(tmp_path, micro_acc_steps=2), make_dataset(), make_model()
    )
    initial = {
        name: sae.encoder.weight.detach().clone() for name, sae in trainer.saes.items()
    }
    trainer.fit()

    for name, sae in trainer.saes.items():
        assert not torch.equal(
            sae.encoder.weight, initial[name]
        ), f"SAE {name} was never updated with micro_acc_steps=2"


@pytest.mark.parametrize("loss_fn", ["ce", "kl"])
def test_micro_acc_steps_rejects_e2e(tmp_path, loss_fn):
    # e2e losses need the full reconstruction, so chunking is not wired up for them.
    # Reject loudly rather than silently ignoring the flag (the original bug).
    cfg = make_config(tmp_path, micro_acc_steps=2, loss_fn=loss_fn)

    with pytest.raises(ValueError, match="micro_acc_steps"):
        Trainer(cfg, make_dataset(), make_model())
