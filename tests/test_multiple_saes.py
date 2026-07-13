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


def make_config(tmp_path, **kwargs) -> TrainConfig:
    return TrainConfig(
        SaeConfig(expansion_factor=2, k=4),
        batch_size=4,
        log_to_wandb=False,
        save_dir=str(tmp_path),
        **kwargs,
    )


def test_multi_seed_training(tmp_path):
    model = make_model()
    dataset = make_dataset()
    cfg = make_config(tmp_path, init_seeds=[0, 42], layers=[0, 1])
    trainer = Trainer(cfg, dataset, model)

    expected_names = {f"h.{i}/seed{s}" for i in (0, 1) for s in (0, 42)}
    assert set(trainer.saes) == expected_names

    initial_weights = {
        name: sae.encoder.weight.detach().clone() for name, sae in trainer.saes.items()
    }

    # Used to raise KeyError: 'h.0' on the first batch
    trainer.fit()

    # Every SAE should have been trained, not just the one whose name happens
    # to match the hookpoint
    for name, sae in trainer.saes.items():
        assert not torch.equal(
            sae.encoder.weight, initial_weights[name]
        ), f"SAE {name} was never updated during training"

    # SAEs initialized from different seeds should stay distinct
    assert not torch.equal(
        trainer.saes["h.0/seed0"].encoder.weight,
        trainer.saes["h.0/seed42"].encoder.weight,
    )

    # Checkpoints should be saved under the seed-suffixed names
    for name in expected_names:
        assert (tmp_path / "unnamed" / name / "sae.safetensors").exists()


def test_single_seed_names_unchanged(tmp_path):
    model = make_model()
    dataset = make_dataset(num_seqs=4)
    cfg = make_config(tmp_path, init_seeds=[0], layers=[0, 1])
    trainer = Trainer(cfg, dataset, model)

    # With a single seed the names should not get a /seed suffix
    assert set(trainer.saes) == {"h.0", "h.1"}

    trainer.fit()


def test_hookpoint_names_do_not_collide(tmp_path):
    # "h.1" is a prefix of "h.10", so substring matching of hookpoint names
    # against SAE names would run h.10's SAEs on h.1's activations too
    model = make_model(n_layer=11)
    dataset = make_dataset(num_seqs=4)
    cfg = make_config(tmp_path, init_seeds=[0, 1], hookpoints=["h.1", "h.10"])
    trainer = Trainer(cfg, dataset, model)

    forward_counts = {name: 0 for name in trainer.saes}

    def make_counter(name):
        def count(module, args):
            forward_counts[name] += 1

        return count

    for name, sae in trainer.saes.items():
        sae.register_forward_pre_hook(make_counter(name))

    # 4 sequences / batch_size 4 = a single batch, so each SAE should do
    # exactly one forward pass on its own hookpoint's activations
    trainer.fit()

    assert forward_counts == {name: 1 for name in trainer.saes}
