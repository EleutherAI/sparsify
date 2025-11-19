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
from .distributed import handle_distribute
from .trainer import TrainConfig, Trainer
from .utils import simple_parse_args_string


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

    tokenizer: str = ""
    """Name of the tokenizer to use for training."""

    # Use a dummy encoding function to prevent the token from being saved
    # to disk in plain text
    hf_token: str | None = field(default=None, encoding_fn=lambda _: None)
    """Huggingface API token for downloading models."""

    revision: str | None = None
    """Model revision to use for training."""

    load_in_8bit: bool = False
    """Load the model in 8-bit mode."""

    max_examples: int | None = None
    """Maximum number of sequences to use for training."""

    resume: bool = False
    """Whether to try resuming from the checkpoint present at `checkpoints/run_name`."""

    text_column: str = "text"
    """Column name to use for text data."""

    shuffle_seed: int = 42
    """Random seed for shuffling the dataset."""

    data_preprocessing_num_proc: int = field(
        default_factory=lambda: cpu_count() // 2,
    )
    """Number of processes to use for preprocessing data"""

    data_args: str = field(
        default="",
    )
    """Arguments to pass to the HuggingFace dataset constructor in the
    format 'arg1=val1,arg2=val2'."""

    model_args: str = field(
        default="",
    )
    """Arguments to pass to the model constructor in the
    format 'arg1=val1,arg2=val2'."""


def load_data(args: RunConfig):
    # For memmap-style datasets
    if args.dataset.endswith(".bin"):
        dataset = MemmapDataset(args.dataset, args.ctx_len, args.max_examples)
    else:
        # For Huggingface datasets
        try:
            kwargs = simple_parse_args_string(args.data_args)
            dataset = load_dataset(args.dataset, split=args.split, **kwargs)
        except Exception as e:
            # Automatically use load_from_disk if appropriate
            if "load_from_disk" in str(e) or "doesn't exist" in str(e):
                dataset = Dataset.load_from_disk(args.dataset, keep_in_memory=False)
            else:
                raise e

        assert isinstance(dataset, Dataset)
        if "input_ids" not in dataset.column_names:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer or args.model, token=args.hf_token
            )
            dataset = chunk_and_tokenize(
                dataset,
                tokenizer,
                max_seq_len=args.ctx_len,
                num_proc=args.data_preprocessing_num_proc,
                text_key=args.text_column,
            )
        else:
            print("Dataset already tokenized; skipping tokenization.")

        print(f"Shuffling dataset with seed {args.shuffle_seed}")
        dataset = dataset.shuffle(args.shuffle_seed)

        dataset = dataset.with_format("torch")
        if limit := args.max_examples:
            dataset = dataset.select(range(limit))

    return dataset


def load_model_artifact(args: RunConfig, rank: int) -> PreTrainedModel:
    if args.load_in_8bit:
        dtype = torch.float16
    elif torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    # End-to-end training requires a model with a causal LM head
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
        torch_dtype=dtype,
        token=args.hf_token,
        **simple_parse_args_string(args.model_args),
    )

    return model


def train_worker(
    rank: int, world_size: int, args: RunConfig, dataset: Dataset | MemmapDataset
):
    torch.cuda.set_device(rank)

    # These should be set by the main process
    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{rank}"),
            rank=rank,
            timeout=timedelta(minutes=3),
            world_size=world_size,
        )
        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    # Prevent ranks other than 0 from printing
    with nullcontext() if rank == 0 else redirect_stdout(None):
        model = load_model_artifact(args, rank)
        if world_size > 1:
            # Drop examples that are indivisible across processes to prevent deadlock
            remainder_examples = len(dataset) % dist.get_world_size()
            dataset = dataset.select(range(len(dataset) - remainder_examples))
            dataset = dataset.shard(dist.get_world_size(), rank)

        print(f"Training on '{args.dataset}' (split '{args.split}')")
        print(f"Storing model weights in {model.dtype}")

        trainer = Trainer(args, dataset, model)
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


def run():
    args = parse(RunConfig)

    dataset = load_data(args)

    handle_distribute(
        process_name="sparsify",
        worker=train_worker,
        const_worker_args=[args, dataset],
    )
