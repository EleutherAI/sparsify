import os
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from datasets import DownloadConfig, load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify import SparseCoder
from sparsify.data import chunk_and_tokenize
from sparsify.sparsify_hooks import ablate_with_mse, collect_activations, edit_with_mse

mp.set_start_method("spawn", force=True)


def extract_k(run_name):
    """Extract the k from the run name."""
    k_str = [item for item in run_name.split("-") if "k=" in item]
    if len(k_str) == 0:
        return 32

    return int(k_str[0].split("=")[1])


def extract_optimizer(run_name):
    """Extract the optimizer type from the run name."""
    if "adam" in run_name.lower():
        return "adam"
    elif "signum" in run_name.lower():
        return "signum"
    else:
        return "unknown"


def extract_learning_rate(run_name):
    """Extract the learning rate from the run name."""
    lr_patterns = ["1e-3", "1e-4", "5e-3", "5e-4"]

    # Check for specific pattern like "adam-5e-4" or "signum-1e-3"
    for pattern in ["adam-", "signum-"]:
        for lr in lr_patterns:
            if f"{pattern}{lr}" in run_name:
                return lr

    # Check for patterns like "(adam 5e-4)" or "(1e-3)"
    parenthesis_match = re.search(r"\((.*?)\)", run_name)
    if parenthesis_match:
        content = parenthesis_match.group(1)
        for lr in lr_patterns:
            if lr in content:
                return lr

    # Just check for the lr pattern anywhere in the name
    for lr in lr_patterns:
        if lr in run_name:
            return lr

    return "unknown"


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
        load_dataset(
            dataset,
            split="train",
            download_config=DownloadConfig(disable_tqdm=True),
        ),  # type: ignore
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
    (
        hookpoint,
        gpu_id,
        batch_size,
        model_name,
        sparse_path,
        variance,
        dataset,
        (ablate_indices, random_indices),
    ) = args

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
    except Exception:
        sparse_model = SparseCoder.load_from_disk(
            f"{sparse_path}/{hookpoint}", device=device
        )
    sparse_model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sparse_losses = []
    sparse_accuracies = []
    uninterpretable_ablation_losses = []
    uninterpretable_ablation_accuracies = []
    random_ablation_losses = []
    random_ablation_accuracies = []
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

        accuracy = (
            (output.logits[:, :-1].argmax(dim=-1) == tokens[:, 1:]).float().mean()
        )
        sparse_accuracies.append(accuracy.item())

        fvus.append((mses[f"model.{hookpoint}"] / variance).cpu())

        if ablate_indices is not None:
            with ablate_with_mse(
                model,
                hookpoints=[f"model.{hookpoint}"],
                sparse_models={f"model.{hookpoint}": sparse_model},
                ablate_features={f"model.{hookpoint}": ablate_indices},
            ) as mses:
                output = model(tokens.long())

                loss = F.cross_entropy(
                    output.logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()
                )
                uninterpretable_ablation_losses.append(loss.item())

                accuracy = (
                    (output.logits[:, :-1].argmax(dim=-1) == tokens[:, 1:])
                    .float()
                    .mean()
                )
                uninterpretable_ablation_accuracies.append(accuracy.item())

        if random_indices is not None:
            with ablate_with_mse(
                model,
                hookpoints=[f"model.{hookpoint}"],
                sparse_models={f"model.{hookpoint}": sparse_model},
                ablate_features={f"model.{hookpoint}": random_indices},
            ) as mses:
                output = model(tokens.long())

                loss = F.cross_entropy(
                    output.logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()
                )
                random_ablation_losses.append(loss.item())

                accuracy = (
                    (output.logits[:, :-1].argmax(dim=-1) == tokens[:, 1:])
                    .float()
                    .mean()
                )
                random_ablation_accuracies.append(accuracy.item())

    data = {
        "hookpoint": hookpoint,
        "sparse_loss": np.mean(sparse_losses),
        "sparse_fvu": np.mean(fvus),
        "sparse_acc": np.mean(sparse_accuracies),
        "uninterpretable_ablation_loss": np.mean(uninterpretable_ablation_losses),
        "uninterpretable_ablation_acc": np.mean(uninterpretable_ablation_accuracies),
        "random_ablation_loss": np.mean(random_ablation_losses),
        "random_ablation_acc": np.mean(random_ablation_accuracies),
    }
    return data


@torch.inference_mode()
def process_baseline_losses(args):
    gpu_id, batch_size, model_name, dataset = args

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

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    losses = []
    accuracies = []

    for batch in dataloader:
        tokens = batch["input_ids"].to(device)
        output = model(tokens.long())

        loss = F.cross_entropy(
            output.logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()
        )
        losses.append(loss.item())

        accuracy = (
            (output.logits[:, :-1].argmax(dim=-1) == tokens[:, 1:]).float().mean()
        )
        accuracies.append(accuracy.item())

    return np.mean(losses), np.mean(accuracies)


@torch.inference_mode()
def populate_performance_metrics(
    model_name: str,
    sparse_path: str,
    cache_dir: str,
    variance_df: pd.DataFrame,
    dataset: str,
    batch_size: int,
    ablate_scores_path: str | None,
    threshold: float = 0.7,
):
    results = {}
    hookpoints = variance_df["hookpoint"].tolist()
    num_gpus = torch.cuda.device_count()

    baseline_loss, baseline_acc = process_baseline_losses(
        (0, batch_size, model_name, dataset)
    )
    for hookpoint in hookpoints:
        results[hookpoint] = {
            "baseline_loss": baseline_loss,
            "baseline_acc": baseline_acc,
        }

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
            (
                get_ablation_indices(Path(ablate_scores_path), [hookpoint], threshold)
                if ablate_scores_path is not None
                else (None, None)
            ),
        )
        for i, hookpoint in enumerate(hookpoints)
    ]

    with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
        for data in executor.map(process_hookpoint_losses, args_list):
            results[data["hookpoint"]].update(data)

    return pd.DataFrame(
        {
            "hookpoint": hookpoints,
            "baseline_loss": [results[hp]["baseline_loss"] for hp in hookpoints],
            "sparse_loss_increase": [
                results[hp]["sparse_loss"] - results[hp]["baseline_loss"]
                for hp in hookpoints
            ],
            "uninterpretable_ablation_loss_increase": [
                results[hp]["uninterpretable_ablation_loss"]
                - results[hp]["baseline_loss"]
                for hp in hookpoints
            ],
            "random_ablation_loss_increase": [
                results[hp]["random_ablation_loss"] - results[hp]["baseline_loss"]
                for hp in hookpoints
            ],
            "baseline_acc": [results[hp]["baseline_acc"] for hp in hookpoints],
            "sparse_acc_decrease": [
                results[hp]["baseline_acc"] - results[hp]["sparse_acc"]
                for hp in hookpoints
            ],
            "uninterpretable_ablation_acc_decrease": [
                results[hp]["baseline_acc"]
                - results[hp]["uninterpretable_ablation_acc"]
                for hp in hookpoints
            ],
            "random_ablation_acc_decrease": [
                results[hp]["baseline_acc"] - results[hp]["random_ablation_acc"]
                for hp in hookpoints
            ],
            # "loss": [results[hp]["loss"] for hp in hookpoints],
            # "loss_increase": [results[hp]["loss"] - baseline_loss
            # for hp in hookpoints],
            # "loss_increase_pct": [
            #     (results[hp]["loss"] - baseline_loss) / baseline_loss * 100
            #     for hp in hookpoints
            # ],
            # "acc": [results[hp]["acc"] for hp in hookpoints],
            # "acc_decrease": [baseline_acc - results[hp]["acc"] for hp in hookpoints],
            # "acc_decrease_pct": [
            #     (baseline_acc - results[hp]["acc"]) / baseline_acc * 100
            #     for hp in hookpoints
            # ],
            "sparse_fvu": [results[hp]["sparse_fvu"] for hp in hookpoints],
            "model": [model_name] * len(hookpoints),
            "sparse_model_name": [sparse_path] * len(hookpoints),
            # "lr": [extract_learning_rate(sparse_path.split("/")[-1])]
            # * len(hookpoints),
            # "optimizer": [extract_optimizer(sparse_path.split("/")[-1])]
            # * len(hookpoints),
            # "skip": ["skip" in sparse_path] * len(hookpoints),
            "k": [extract_k(sparse_path.split("/")[-1])] * len(hookpoints),
            # "random_loss": [results[hp]["random_loss"] for hp in hookpoints],
            # "random_acc": [results[hp]["random_acc"] for hp in hookpoints],
            # "random_loss_increase":
            # [results[hp]["random_loss"] - baseline_loss for hp in hookpoints],
            # "random_loss_increase_pct": [
            #     (results[hp]["random_loss"] - baseline_loss) / baseline_loss * 100
            #     for hp in hookpoints
            # ],
            # "random_acc_decrease":
            # [baseline_acc - results[hp]["random_acc"] for hp in hookpoints],
            # "random_acc_decrease_pct": [
            #     (baseline_acc - results[hp]["random_acc"]) / baseline_acc * 100
            #     for hp in hookpoints
            # ],
        }
    )


def get_ablation_indices(
    scores_path: Path, target_modules: list[str], threshold: float = 0.7
):
    from delphi.log.result_analysis import build_scores_df

    log_path = scores_path.parent / "log" / "hookpoint_firing_counts.pt"
    hookpoint_firing_counts: dict[str, Tensor] = torch.load(log_path, weights_only=True)
    df = build_scores_df(scores_path, target_modules, hookpoint_firing_counts)

    width = len(hookpoint_firing_counts[target_modules[0]])

    # df.to_csv(f"images/uninterpretable_indices_{scores_path.stem}.csv", index=False)

    # Get the features with an interpretability accuracy below the threshold
    features = df.query(f"accuracy < {threshold}")
    # perfect_indices = df.query("accuracy == 1")["latent_idx"].unique()

    print(f"Number of features: {len(df)}")
    print(
        "Number of features with an interpretability accuracy"
        f" below {threshold}: {len(features)}"
    )

    uninterpretable_indices = features["latent_idx"].unique()
    random_indices = torch.randint(0, width, (len(uninterpretable_indices),)).tolist()

    return uninterpretable_indices, random_indices


@torch.inference_mode()
def evaluate(
    model_name: str,
    sparse_path: str,
    hookpoints: list[str],
    batch_size: int,
    dataset: str,
    ablate_scores_path: str | None,
    threshold: float = 0.7,
):
    torch.manual_seed(42)

    cache_dir = f"images/ce_loss_increases/{model_name.replace('/', '-')}"
    os.makedirs(cache_dir, exist_ok=True)

    # Get variance
    if os.path.exists(f"{cache_dir}/variance.csv"):
        df = pd.read_csv(f"{cache_dir}/variance.csv")
    else:
        print(f"Populating variance for {model_name}...")
        df = populate_variances(model_name, cache_dir, hookpoints, dataset, batch_size)
        df.to_csv(f"{cache_dir}/variance.csv", index=False)

    sparse_name = sparse_path.split("/")[-1]
    ablate_tag = f"ablate={ablate_scores_path is not None}"
    loss_fvu_path = f"{cache_dir}/{sparse_name}_loss_fvu_{ablate_tag}.csv"
    # if os.path.exists(loss_fvu_path):
    # df = pd.read_csv(loss_fvu_path)
    # else:
    print(f"Populating loss increase for {sparse_path}...")
    df = df[df["hookpoint"].isin(hookpoints)]
    df = populate_performance_metrics(
        model_name,
        sparse_path,
        cache_dir,
        df,
        dataset,
        batch_size,
        ablate_scores_path,
        threshold,
    )
    df.to_csv(loss_fvu_path, index=False)

    print("Done! Results:")
    print(df.head())
    print(f"Saved to {loss_fvu_path}")


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--sparse_model", type=str, required=True)
    parser.add_argument("--hookpoints", nargs="+", required=True)
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--dataset", type=str, default="EleutherAI/SmolLM2-135M-10B")
    parser.add_argument("--ablate_scores_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    evaluate(
        args.base_model,
        args.sparse_model,
        args.hookpoints,
        args.batch_size,
        args.dataset,
        args.ablate_scores_path,
    )


if __name__ == "__main__":
    main()
