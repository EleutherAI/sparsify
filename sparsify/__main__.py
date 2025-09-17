import os
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import cpu_count

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from safetensors.torch import load_model
from simple_parsing import field, parse
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)

from .data import MemmapDataset, chunk_and_tokenize
from .trainer import TrainConfig, Trainer


@dataclass
class RunConfig(TrainConfig):
    model: str = field(
        default="HuggingFaceTB/SmolLM2-135M",
        positional=True,
    )
    """Name of the model to train."""

    dataset: str = field(
        default="EleutherAI/SmolLM2-135M-10B",
        positional=True,
    )
    """Path to the dataset to use for training."""

    split: str = "train"
    """Dataset split to use for training."""

    ctx_len: int = 2048
    """Context length to use for training."""

    hf_token: str | None = field(default=None, encoding_fn=lambda _: None)
    """Huggingface API token for downloading models."""

    eval_split: str | None = None
    """Optional dataset split to use for evaluation."""

    revision: str | None = None
    """Model revision to use for training."""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    max_examples: int | None = None
    """Maximum number of examples to use for training."""

    resume: bool = False
    """Whether to try resuming from checkpoint at `checkpoints/run_name`."""

    text_column: str = "text"
    """Column name to use for text data."""

    shuffle_seed: int = 42
    """Random seed for shuffling the dataset."""

    data_preprocessing_num_proc: int = field(
        default_factory=lambda: cpu_count() // 2,
    )
    """Number of processes to use for preprocessing data"""


def load_artifacts(
    args: RunConfig, rank: int
) -> tuple[PreTrainedModel, Dataset | MemmapDataset, Dataset | None]:
    if args.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model_cls = AutoModel if args.loss_fn == "fvu" else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        args.model,
        device_map={"": f"cuda:{rank}"},
        quantization_config=(
            BitsAndBytesConfig(load_in_8bit=args.load_in_8bit)
            if args.load_in_8bit
            else None
        ),
        revision=args.revision,
        dtype=dtype,  # ✅ replaces deprecated torch_dtype
        token=args.hf_token,
    )

    def prepare_dataset(split: str) -> Dataset | MemmapDataset:
        if args.dataset.endswith(".bin"):
            return MemmapDataset(args.dataset, args.ctx_len, args.max_examples)

        ds = load_dataset(
            args.dataset,
            split=split,
            trust_remote_code=True,
        )

        assert isinstance(ds, Dataset)
        if "input_ids" not in ds.column_names:
            tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
            ds = chunk_and_tokenize(
                ds,
                tokenizer,
                max_seq_len=args.ctx_len,
                num_proc=args.data_preprocessing_num_proc,
                text_key=args.text_column,
            )
        else:
            print(f"Split '{split}' already tokenized; skipping tokenization.")

        print(f"Shuffling split '{split}' with seed {args.shuffle_seed}")
        ds = ds.shuffle(args.shuffle_seed).with_format("torch")
        if limit := args.max_examples:
            ds = ds.select(range(limit))
        return ds

    # Train dataset
    dataset = prepare_dataset(args.split)
    print(f"Train dataset size: {len(dataset)}")

    # Optional eval dataset
    eval_dataset = None
    if args.eval_split:
        eval_dataset = prepare_dataset(args.eval_split)
        print(f"Eval dataset size: {len(eval_dataset)}")

    return model, dataset, eval_dataset


def run():
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group(
            "nccl", device_id=torch.device(rank), timeout=timedelta(weeks=1)
        )
        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    args = parse(RunConfig)

    with nullcontext() if rank == 0 else redirect_stdout(None):
        if not ddp or rank == 0:
            model, dataset, eval_dataset = load_artifacts(args, rank)
        if ddp:
            dist.barrier()
            if rank != 0:
                model, dataset, eval_dataset = load_artifacts(args, rank)

            remainder_examples = len(dataset) % dist.get_world_size()
            dataset = dataset.select(range(len(dataset) - remainder_examples))
            dataset = dataset.shard(dist.get_world_size(), rank)

        print(f"Training on '{args.dataset}' (split '{args.split}')")
        if args.eval_split:
            print(f"Evaluating on split '{args.eval_split}'")
        print(f"Storing model weights in {model.dtype}")

        # ✅ Ensure training won’t silently skip
        if args.epochs == 0 and args.max_steps == 0:
            args.epochs = 1
            print("⚠️  No training length specified, defaulting to epochs=1")

        trainer = Trainer(args, dataset, model, eval_dataset=eval_dataset)

        if args.resume:
            trainer.load_state(f"checkpoints/{args.run_name}" or "checkpoints/unnamed")
        elif args.finetune:
            for name, sae in trainer.saes.items():
                load_model(
                    sae,
                    f"{args.finetune}/{name}/sae.safetensors",
                    device=str(model.device),
                )

        trainer.fit()


if __name__ == "__main__":
    run()
