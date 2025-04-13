import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify import SparseCoder
from sparsify.data import chunk_and_tokenize
from sparsify.plot_optimizers import extract_learning_rate, extract_optimizer
from sparsify.sparsify_hooks import collect_activations, edit_with_mse

mp.set_start_method("spawn", force=True)


def chunked_torch_variance(data_loader, device):
    # Initialize running values
    n_total = 0
    mean_total = 0
    M2_total = 0

    # Process data in batches
    for batch in data_loader:
        acts = batch["activations"].to(device).flatten(0, 1)

        n_acts = acts.shape[0]
        mean_acts = torch.mean(acts, dim=0)
        var_acts = torch.var(acts, dim=0, unbiased=True)

        # Update combined statistics using Welford's online algorithm
        delta = mean_acts - mean_total
        mean_total = (n_total * mean_total + n_acts * mean_acts) / (n_total + n_acts)
        M2_total = (
            M2_total
            + n_acts * var_acts
            + (delta**2) * n_total * n_acts / (n_total + n_acts)
        )
        n_total += n_acts

    variance = M2_total / (n_total - 1)
    return variance.cpu()


def get_acts(batch, model, hookpoint, device, input_acts=True):
    # input_acts is a boolean that indicates whether the activations are input or output

    with collect_activations(
        model,
        hookpoints=[hookpoint],
        input_acts=input_acts,
    ) as activations:
        model(batch["input_ids"].to(device))

    return {
        "input_ids": batch["input_ids"].cpu(),
        "activations": activations[hookpoint].cpu(),
    }


@torch.inference_mode()
def process_hookpoint_variance(args):
    hookpoint, gpu_id, batch_size, model_name, dataset, input = args
    device = f"cuda:{gpu_id}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = chunk_and_tokenize(
        load_dataset(dataset, split="train"),  # type: ignore
        tokenizer,
        max_seq_len=1024,
    ).select(range(4096))

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    return hookpoint, chunked_torch_variance(
        DataLoader(
            dataset.map(
                partial(
                    get_acts,
                    input_acts=input,
                    model=model,
                    hookpoint=f"model.{hookpoint}",
                    device=device,
                ),
                batched=True,
                batch_size=batch_size,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=24,
        ),
        device=device,
    )


@torch.inference_mode()
def populate_variances(
    model_name: str,
    cache_dir: str,
    hookpoints: list[str],
    dataset: str,
    batch_size: int,
):
    num_gpus = min(len(hookpoints), torch.cuda.device_count())

    args_list = [
        (hookpoint, i, batch_size, model_name, dataset, False)
        for i, hookpoint in enumerate(hookpoints)
    ]

    ctx = mp.get_context("spawn")
    output_variances = {}
    with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
        for hookpoint, variance in executor.map(process_hookpoint_variance, args_list):
            output_variances[hookpoint] = variance
    # Save variance to cache
    return pd.DataFrame(
        {
            "output_variance": [
                output_variances[hookpoint].mean().item() for hookpoint in hookpoints
            ],
            "model": [model_name] * len(hookpoints),
            "hookpoint": hookpoints,
        }
    )


@torch.inference_mode()
def process_hookpoint_losses(args):
    hookpoint, gpu_id, batch_size, model_name, sparse_path, variance, dataset = args

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = chunk_and_tokenize(
        load_dataset(dataset, split="train"),  # type: ignore
        tokenizer,
        max_seq_len=1024,
    ).select(range(4096))
    del tokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    try:
        sparse_model = SparseCoder.load_from_hub(
            sparse_path, hookpoint=hookpoint, device=device
        )
    except Exception as e:
        print(f"Error loading sparse model from hub: {e}")
        print("Attempting to load from disk...")
        sparse_model = SparseCoder.load_from_disk(
            f"{sparse_path}/{hookpoint}", device=device
        )
    sparse_model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sparse_losses = []
    fvus = []

    for batch in dataloader:
        tokens = batch["input_ids"].to(device)
        with edit_with_mse(
            model,
            hookpoints=[f"model.{hookpoint}"],
            sparse_models={f"model.{hookpoint}": sparse_model},
        ) as mses:
            output = model(tokens.long())

        loss = F.cross_entropy(
            output.logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()
        )
        sparse_losses.append(loss.item())
        fvus.append((mses[f"model.{hookpoint}"] / variance).cpu())

    return hookpoint, np.mean(sparse_losses), np.mean(fvus)


@torch.inference_mode()
def populate_losses_fvus(
    model_name, sparse_path, cache_dir, variance_df, dataset, batch_size
):
    hookpoints = variance_df["hookpoint"].tolist()
    num_gpus = torch.cuda.device_count()

    ctx = mp.get_context("spawn")
    args_list = [
        (
            hookpoint,
            i,
            batch_size,
            model_name,
            sparse_path,
            variance_df[variance_df["hookpoint"] == hookpoint][
                "output_variance"
            ].values[0],
            dataset,
        )
        for i, hookpoint in enumerate(hookpoints)
    ]

    results = {}
    with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
        for hookpoint, loss, fvu in executor.map(process_hookpoint_losses, args_list):
            results[hookpoint] = {"loss": loss, "fvu": fvu}

    return pd.DataFrame(
        {
            "hookpoint": hookpoints,
            "loss": [results[hp]["loss"] for hp in hookpoints],
            "fvu": [results[hp]["fvu"] for hp in hookpoints],
            "model": [model_name] * len(hookpoints),
            "sparse_model_name": [sparse_path] * len(hookpoints),
            "lr": [extract_learning_rate(sparse_path.split("/")[-1])] * len(hookpoints),
            "optimizer": [extract_optimizer(sparse_path.split("/")[-1])]
            * len(hookpoints),
            "skip": ["skip" in sparse_path] * len(hookpoints),
        }
    )


@torch.inference_mode()
def evaluate(
    model_name: str,
    sparse_path: str,
    hookpoints: list[str],
    batch_size: int,
    dataset: str,
):
    cache_dir = f"images/ce_loss_increases/{model_name.replace('/', '-')}"
    os.makedirs(cache_dir, exist_ok=True)

    # Get variance
    if os.path.exists(f"{cache_dir}/variance.csv"):
        df = pd.read_csv(f"{cache_dir}/variance.csv")
    else:
        print(f"Populating variance for {model_name}...")
        df = populate_variances(model_name, cache_dir, hookpoints, dataset, batch_size)
        df.to_csv(f"{cache_dir}/variance.csv", index=False)

    print(f"Populating loss increase for {sparse_path}...")
    df = populate_losses_fvus(
        model_name, sparse_path, cache_dir, df, dataset, batch_size
    )
    df.to_csv(f"{cache_dir}/{sparse_path.split('/')[-1]}_loss_fvu.csv", index=False)

    print(f"Done! Results: {df.head()}")
    print(f"Saved to {cache_dir}/{sparse_path.split('/')[-1]}_loss_fvu.csv")


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--sparse_model", type=str, required=True)
    parser.add_argument("--hookpoints", nargs="+", required=True)
    parser.add_argument("--dataset", type=str, default="EleutherAI/SmolLM2-135M-10B")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    evaluate(
        args.base_model,
        args.sparse_model,
        args.hookpoints,
        args.batch_size,
        args.dataset,
    )


if __name__ == "__main__":
    main()
