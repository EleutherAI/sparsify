import torch
from datasets import Dataset
from transformers import GPT2Config, GPT2LMHeadModel

from sparsify import SaeConfig, TrainConfig, Trainer

VOCAB_SIZE = 64


def make_model(n_layer: int = 2) -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=n_layer,
        n_embd=16,
        n_head=2,
        vocab_size=VOCAB_SIZE,
        n_positions=32,
    )
    return GPT2LMHeadModel(config).to("cuda")


def make_dataset(num_seqs: int = 8, seq_len: int = 16) -> Dataset:
    torch.manual_seed(0)
    tokens = torch.randint(0, VOCAB_SIZE, (num_seqs, seq_len))
    return Dataset.from_dict({"input_ids": tokens.tolist()}).with_format("torch")


def test_embed_skip_training(tmp_path):
    model = make_model()
    dataset = make_dataset()
    cfg = TrainConfig(
        SaeConfig(expansion_factor=2, k=4, embed_skip=True),
        batch_size=4,
        layers=[1],
        log_to_wandb=False,
        save_dir=str(tmp_path),
    )
    trainer = Trainer(cfg, dataset, model)
    sae = trainer.saes["h.1"]

    assert sae.W_embed_skip is not None
    assert sae.b_embed_skip is not None
    assert sae.W_embed_skip.shape == (sae.d_in, sae.d_embed)

    initial_w_embed = sae.W_embed_skip.detach().clone()
    initial_b_embed = sae.b_embed_skip.detach().clone()
    initial_encoder = sae.encoder.weight.detach().clone()

    trainer.fit()

    assert not torch.equal(sae.encoder.weight, initial_encoder)
    assert not torch.equal(sae.W_embed_skip, initial_w_embed)
    assert not torch.equal(sae.b_embed_skip, initial_b_embed)

    # Round-trip through disk should preserve the embed-skip parameters
    sae.save_to_disk(tmp_path / "sae_dump")
    from sparsify.sparse_coder import SparseCoder

    reloaded = SparseCoder.load_from_disk(tmp_path / "sae_dump", device="cuda")
    assert reloaded.cfg.embed_skip
    assert torch.equal(reloaded.W_embed_skip, sae.W_embed_skip)
    assert torch.equal(reloaded.b_embed_skip, sae.b_embed_skip)


def test_embed_skip_forward_requires_embed():
    from sparsify.config import SaeConfig
    from sparsify.sparse_coder import SparseCoder

    sae = SparseCoder(16, SaeConfig(expansion_factor=2, k=4, embed_skip=True), device="cuda")
    x = torch.randn(8, 16, device="cuda")
    try:
        sae(x)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected forward() to require `embed` when embed_skip=True")
