import pytest
import torch
from datasets import Dataset
from transformers import GPT2Config, GPT2LMHeadModel

from sparsify import SaeConfig, TrainConfig, Trainer

VOCAB_SIZE = 64


def make_model(n_layer: int = 1, n_positions: int = 128) -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=n_layer,
        n_embd=16,
        n_head=2,
        vocab_size=VOCAB_SIZE,
        n_positions=n_positions,
    )
    return GPT2LMHeadModel(config).to("cuda")


def make_dataset(num_seqs: int = 64, seq_len: int = 128) -> Dataset:
    torch.manual_seed(0)
    tokens = torch.randint(0, VOCAB_SIZE, (num_seqs, seq_len))
    return Dataset.from_dict({"input_ids": tokens.tolist()}).with_format("torch")


def make_config(tmp_path, **kwargs) -> TrainConfig:
    return TrainConfig(
        SaeConfig(expansion_factor=2, k=4),
        batch_size=16,
        log_to_wandb=False,
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


def test_micro_acc_steps_matches_unchunked_update(tmp_path):
    # Chunking must not change the optimizer step. `acc_steps` scales the loss and
    # `total_variance` is a sum (not a mean), so the two updates coincide.
    #
    # This is exact only in the limit of large chunks: each chunk normalizes FVU by
    # its own y.mean(0), so tiny chunks diverge (~22% at 32 tokens/chunk, ~9% at
    # 256). At the 8x128=1024 tokens/chunk used here it is already float-noise
    # exact, and realistic training chunks are far larger still.
    unchunked = train_deltas(tmp_path / "a", micro_acc_steps=1)
    chunked = train_deltas(tmp_path / "b", micro_acc_steps=2)

    assert set(unchunked) == set(chunked)
    for name, d1 in unchunked.items():
        d2 = chunked[name]
        assert d1.norm() > 1e-6, f"{name} did not move; test would be vacuous"

        # Compare the updates relative to the size of the update itself.
        rel_gap = (d2 - d1).norm() / d1.norm()
        cos = torch.nn.functional.cosine_similarity(d1, d2, dim=0)
        assert rel_gap < 0.01, f"{name} update differs by {rel_gap:.3f} of its norm"
        assert cos > 0.999, f"{name} update direction diverged (cos={cos:.4f})"


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
