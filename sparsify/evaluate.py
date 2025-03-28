from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp
from functools import partial
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sparsify import SparseCoder
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from sparsify.data import chunk_and_tokenize
import os
import pandas as pd
from torch.utils.data import DataLoader

from sparsify.edit_sparse import edit_with_mse, collect_activations
from sparsify.plot_optimizers import extract_learning_rate, extract_optimizer

mp.set_start_method('spawn', force=True)

def chunked_torch_variance(data_loader, device):
    # Initialize running values
    n_total = 0
    mean_total = 0
    M2_total = 0
    
    # Process data in batches
    for batch in data_loader:
        acts = batch['activations'].to(device).flatten(0, 1)

        n_acts = acts.shape[0]
        mean_acts = torch.mean(acts, dim=0)
        var_acts = torch.var(acts, dim=0, unbiased=True)
        
        # Update combined statistics using Welford's online algorithm
        delta = mean_acts - mean_total
        mean_total = (n_total * mean_total + n_acts * mean_acts) / (n_total + n_acts)
        M2_total = M2_total + n_acts * var_acts + (delta**2) * n_total * n_acts / (n_total + n_acts)
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
        model(batch['input_ids'].to(device))

    return {
        "input_ids": batch['input_ids'].cpu(),
        "activations": activations[hookpoint].cpu()
    }

@torch.inference_mode()
def process_hookpoint_variance(args):
    hookpoint, gpu_id, batch_size, model_name, input = args
    device = f"cuda:{gpu_id}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = chunk_and_tokenize(
        load_dataset("EleutherAI/SmolLM2-135M-10B", split="train"),
        tokenizer,
        max_seq_len=1024,
    ).select(range(4096))

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    return hookpoint, chunked_torch_variance(
        DataLoader(
            dataset.map(partial(get_acts, 
                               input_acts=input, 
                               model=model, 
                               hookpoint=f"model.{hookpoint}", 
                               device=device),
                               batched=True,
                               batch_size=batch_size),
            batch_size=batch_size, 
            shuffle=False,
            num_workers=24
        ), 
        device=device
    )

@torch.inference_mode()
def populate_variances(
    model_name, cache_dir
):
    if model_name == "HuggingFaceTB/SmolLM2-135M":
        batch_size = 64
        hookpoints = [
            "layers.0.mlp",
            "layers.9.mlp",
            "layers.18.mlp",
            "layers.27.mlp",
        ]
    elif model_name == "HuggingFaceTB/SmolLM2-1.7B":
        batch_size = 16
        hookpoints = [
            "layers.0.mlp",
            "layers.7.mlp",
            "layers.14.mlp",
            "layers.21.mlp",
        ]
    else:
        raise ValueError(f"Model {model_name} not supported")

    num_gpus = len(hookpoints)

    args_list = [(hookpoint, i, batch_size // 2, model_name, False) for i, hookpoint in enumerate(hookpoints)]

    ctx = mp.get_context('spawn')
    output_variances = {}
    with ProcessPoolExecutor(max_workers=min(len(hookpoints), num_gpus), mp_context=ctx) as executor:
        for hookpoint, variance in executor.map(process_hookpoint_variance, args_list):
            output_variances[hookpoint] = variance
    # Save variance to cache
    df = pd.DataFrame({
        "output_variance": [output_variances[hookpoint].mean().item() for hookpoint in hookpoints],
        "model": [model_name] * len(hookpoints),
        "hookpoint": hookpoints,
    })
    df.to_csv(f"{cache_dir}/variance.csv", index=False)

@torch.inference_mode()
def process_hookpoint_losses(args):
    hookpoint, gpu_id, batch_size, model_name, sparse_path, variance = args
    
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = chunk_and_tokenize(
        load_dataset("EleutherAI/SmolLM2-135M-10B", split="train"),
        tokenizer,
        max_seq_len=1024,
    ).select(range(4096))
    del tokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    sparse_model = SparseCoder.load_from_disk(sparse_path, device=device)
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
            transcode=sparse_model.cfg.transcode,
        ) as mses:
            output = model(tokens.long())

        loss = F.cross_entropy(output.logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten())
        sparse_losses.append(loss.item())
        fvus.append((mses[f"model.{hookpoint}"] / variance).cpu())

    return hookpoint, np.mean(sparse_losses), np.mean(fvus)

@torch.inference_mode()
def populate_losses_fvus(model_name, sparse_path, cache_dir, variance_df):
    batch_size = 16 if "135M" in model_name else 1
    hookpoints = variance_df["hookpoint"].tolist()
    num_gpus = torch.cuda.device_count()

    ctx = mp.get_context('spawn')
    args_list = [
        (
            hookpoint,
            i,
            batch_size,
            model_name,
            f"{sparse_path}/{hookpoint}",
            variance_df[variance_df["hookpoint"] == hookpoint]["output_variance"].values[0],
        )
        for i, hookpoint in enumerate(hookpoints)
    ]

    results = {}
    with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
        for hookpoint, loss, fvu in executor.map(process_hookpoint_losses, args_list):
            results[hookpoint] = {"loss": loss, "fvu": fvu}

    df = pd.DataFrame({
        "hookpoint": hookpoints,
        "loss": [results[hp]["loss"] for hp in hookpoints],
        "fvu": [results[hp]["fvu"] for hp in hookpoints],
        "model": model_name,
        "sparse_model_name": sparse_path.split("/")[-1],
        "lr": [extract_learning_rate(sparse_path.split("/")[-1])] * len(hookpoints),
        "optimizer": [extract_optimizer(sparse_path.split("/")[-1])] * len(hookpoints),
        "skip": ["skip" in sparse_path] * len(hookpoints),
    })
    df.to_csv(f"{cache_dir}/{sparse_path.split('/')[-1]}_loss_fvu.csv", index=False)

    return df

@torch.inference_mode()
def evaluate(model_name, sparse_paths):
    cache_dir = f"images/ce_loss_increases/{model_name.replace('/', '-')}"
    os.makedirs(cache_dir, exist_ok=True)

    # Get variance
    if os.path.exists(f"{cache_dir}/variance.csv"):
        df = pd.read_csv(f"{cache_dir}/variance.csv")
    else:
        print(f"Populating variance for {model_name}")
        populate_variances(model_name, cache_dir)
        df = pd.read_csv(f"{cache_dir}/variance.csv")

    for sparse_path in sparse_paths:
        try:
            print(f"Populating loss increases for {sparse_path}")
            populate_losses_fvus(
                model_name, sparse_path, cache_dir, df
            )
        except Exception as e:
            print(f"Error populating loss increases for {sparse_path}: {e}")


def main():
    torch.manual_seed(42)

    model_name = "HuggingFaceTB/SmolLM2-135M"

    unverified_sparse_paths = [
        "SmolLM2-135M-signum-5e-3-m0.85",
        "SmolLM2-135M-signum-5e-4-m0.85",
        "SmolLM2-135M-signum-1e-4-m0.85",
        "SmolLM2-135M-signum-1e-3-m0.85",

        "SmolLM2-135M-skip-signum-5e-3-m0.85",
        "SmolLM2-135M-skip-signum-5e-4-m0.85",
        "SmolLM2-135M-skip-signum-1e-3-m0.85",
        "SmolLM2-135M-skip-signum-1e-4-m0.85",
        "SmolLM2-135M-skip-adam-5e-4",
        "SmolLM2-135M-skip-adam-1e-4",
        
        "SmolLM2-135M-skip-adam-5e-3",
    ]
    sparse_paths = [f"checkpoints/{sparse_path}" for sparse_path in unverified_sparse_paths]
    verified = [
        "SmolLM2-135M-topk-adam-1e-4",
        "SmolLM2-135M-topk-adam-5e-4",
        "SmolLM135M2-topk-adam-5e-3",
        "SmolLM135M2-topk-adam-1e-3",
        "SmolLM2-135M-skip-adam-1e-3",
    ]
    sparse_paths.extend([f"checkpoints/verified-paper/{sparse_path}" for sparse_path in verified])

    evaluate(model_name, sparse_paths)
    
    model_name = "HuggingFaceTB/SmolLM2-1.7B"
    

    sparse_paths = [
        "SmolLM2-1.7B-topk-adam",
        "SmolLM2-1.7B-topk-signum",
        "SmolLM2-1.7B-gm-signum",
        "SmolLM2-1.7B-skip-gm",
        "SmolLM2-1.7B-skip-topk-signum",
    ]
    sparse_paths = [f"checkpoints/verified-paper/{sparse_path}" for sparse_path in sparse_paths]

    evaluate(model_name, sparse_paths)

if __name__ == "__main__":
    main()
