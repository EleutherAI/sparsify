import pytest
import torch
from datasets import Dataset
from transformers import GPT2Config, GPT2LMHeadModel

from sparsify import SaeConfig, TrainConfig, Trainer

VOCAB_SIZE = 64


def make_model(n_layer: int = 2, n_positions: int = 32) -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=n_layer,
        n_embd=16,
        n_head=2,
        vocab_size=VOCAB_SIZE,
        n_positions=n_positions,
    )
    return GPT2LMHeadModel(config).to("cuda")


def make_dataset(num_seqs: int = 8, seq_len: int = 16) -> Dataset:
    torch.manual_seed(0)
    tokens = torch.randint(0, VOCAB_SIZE, (num_seqs, seq_len))
    return Dataset.from_dict({"input_ids": tokens.tolist()}).with_format("torch")


def make_config(tmp_path, **kwargs) -> TrainConfig:
    return TrainConfig(
        SaeConfig(expansion_factor=2, k=4),
        batch_size=4,
        log_to_wandb=False,
        save_dir=str(tmp_path),
        **kwargs,
    )


def train_deltas(tmp_path, micro_acc_steps: int) -> dict[str, torch.Tensor]:
    """Train one run and return each SAE's weight update."""
    model = make_model(n_layer=1, n_positions=128)
    dataset = make_dataset(num_seqs=64, seq_len=128)
    cfg = TrainConfig(
        SaeConfig(expansion_factor=2, k=4),
        batch_size=16,
        log_to_wandb=False,
        save_dir=str(tmp_path),
        layers=[0],
        micro_acc_steps=micro_acc_steps,
        lr=1e-3,
    )
    trainer = Trainer(cfg, dataset, model)
    initial = {
        name: sae.encoder.weight.detach().clone() for name, sae in trainer.saes.items()
    }
    trainer.fit()
    return {
        name: (sae.encoder.weight.detach() - initial[name]).flatten()
        for name, sae in trainer.saes.items()
    }


def test_micro_acc_steps_matches_unchunked_update(tmp_path):
    # Chunking must not change the optimizer step. Every chunk is normalized by the
    # full batch's total_variance scaled to the chunk's share of rows, so the summed
    # chunk losses telescope back to the unchunked loss exactly, at any chunk size.
    unchunked = train_deltas(tmp_path / "a", micro_acc_steps=1)
    chunked = train_deltas(tmp_path / "b", micro_acc_steps=2)

    assert set(unchunked) == set(chunked)
    for name, d1 in unchunked.items():
        d2 = chunked[name]
        assert d1.norm() > 1e-6, f"{name} did not move; test would be vacuous"

        # Compare the updates relative to the size of the update itself.
        rel_gap = (d2 - d1).norm() / d1.norm()
        cos = torch.nn.functional.cosine_similarity(d1, d2, dim=0)
        assert rel_gap < 1e-3, f"{name} update differs by {rel_gap:.2e} of its norm"
        assert cos > 0.99999, f"{name} update direction diverged (cos={cos:.6f})"


def test_micro_acc_steps_actually_trains(tmp_path):
    # Guards the regression this test file exists for: micro_acc_steps was silently
    # ignored (it only scaled the loss denominator), so chunking was a no-op.
    model = make_model()
    dataset = make_dataset()
    cfg = make_config(tmp_path, layers=[0], micro_acc_steps=2)
    trainer = Trainer(cfg, dataset, model)

    initial = {
        name: sae.encoder.weight.detach().clone() for name, sae in trainer.saes.items()
    }
    trainer.fit()

    for name, sae in trainer.saes.items():
        assert not torch.equal(
            sae.encoder.weight, initial[name]
        ), f"SAE {name} was never updated with micro_acc_steps=2"


def test_micro_acc_steps_with_embed_skip(tmp_path):
    # The embedding activations must be chunked in lockstep with x/y, otherwise
    # the skip connection sees a batch of the wrong length.
    model = make_model()
    dataset = make_dataset()
    cfg = TrainConfig(
        SaeConfig(expansion_factor=2, k=4, embed_skip=True),
        batch_size=4,
        log_to_wandb=False,
        save_dir=str(tmp_path),
        layers=[0],
        micro_acc_steps=2,
    )
    trainer = Trainer(cfg, dataset, model)
    trainer.fit()

    for name, sae in trainer.saes.items():
        assert sae.W_embed_skip is not None
        assert torch.isfinite(sae.W_embed_skip).all(), f"{name} W_embed_skip has NaNs"


@pytest.mark.parametrize("loss_fn", ["ce", "kl"])
def test_micro_acc_steps_rejects_e2e(tmp_path, loss_fn):
    # e2e losses need the full reconstruction, so chunking is not wired up for them.
    # Reject loudly rather than silently ignoring the flag (the original bug).
    model = make_model()
    dataset = make_dataset()
    cfg = make_config(tmp_path, layers=[0], micro_acc_steps=2, loss_fn=loss_fn)

    with pytest.raises(ValueError, match="micro_acc_steps"):
        Trainer(cfg, dataset, model)


def test_chunked_fvu_matches_unchunked():
    # The trainer normalizes each chunk by the full batch's total_variance, scaled
    # to the chunk's share of rows. The mean of the chunk FVUs must then equal the
    # unchunked FVU; passing the unscaled full-batch variance halves every chunk's
    # FVU (and the loss) at micro_acc_steps=2.
    from sparsify.sparse_coder import SparseCoder

    torch.manual_seed(0)
    sae = SparseCoder(64, SaeConfig(expansion_factor=2, k=4), device="cuda")
    y = torch.randn(256, 64, device="cuda")

    full = sae(y, y).fvu
    total_variance = (y - y.mean(0)).pow(2).sum()
    chunk_fvus = [
        sae(c, c, total_variance=total_variance * (c.shape[0] / y.shape[0])).fvu
        for c in y.chunk(2)
    ]
    chunked = sum(chunk_fvus) / len(chunk_fvus)

    torch.testing.assert_close(chunked, full, rtol=2e-2, atol=1e-3)
