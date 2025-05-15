import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import plotly.graph_objects as go
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from sparsify import SparseCoder
from sparsify.data import chunk_and_tokenize
from sparsify.sparsify_hooks import ablate_with_mse

mp.set_start_method("spawn", force=True)

def import_plotly():
    try:
        import plotly.express as px
        import plotly.io as pio
    except ImportError:
        raise ImportError(
            "Plotly is not installed.\n"
            "Please install it using `pip install plotly`, "
            "or install the `[visualize]` extra."
        )
    pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469
    return px

def make_equal_fire_bins(df, n_bins=100):
    df = df.sort_values('f1_score', ascending=False).reset_index(drop=True)
    edges = np.linspace(0, df.firing_count.sum(), n_bins + 1)
    df['bin'] = np.searchsorted(edges, df.firing_count.cumsum(), side='right') - 1

    return df


@torch.inference_mode()
def get_ablation_metrics(args):
    (
        hookpoint,
        gpu_id,
        batch_size,
        model_name,
        sparse_path,
        dataset,
        interpretability_df,
        n_seqs,
        num_bins
    ) = args


    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = chunk_and_tokenize(
        load_dataset(dataset, split="train"),  # type: ignore
        tokenizer,
        max_seq_len=1024,
    ).select(range(n_seqs))
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
    
    # bins = True
    bin_data = []
    # if bins:
    

    interp_df = make_equal_fire_bins(interpretability_df, num_bins)

    print("bin with highest f1 score", interp_df[interp_df.bin == 0].f1_score.mean())
    print("bin with lowest f1 score", interp_df[interp_df.bin == 19].f1_score.mean())

    # original_outlier_bins = [0, 1]
    # interp_df["is_outlier"] = interp_df["bin"].isin(original_outlier_bins)
    # outlier = interp_df.loc[interp_df.is_outlier]
    # rest    = interp_df.loc[~interp_df.is_outlier]

    
    # n_new_outlier_bins = 15
    # outlier = make_equal_fire_bins(outlier.drop(columns="bin"), n_new_outlier_bins)          # ensure no collision
    # outlier["is_outlier"] = True                  
    # rest["bin"] += (n_new_outlier_bins - len(original_outlier_bins))

    # interp_df = pd.concat([outlier, rest]).reset_index(drop=True)

    print("bins: ", interp_df.bin.unique())

    binned_idxs = []
    bin_mean_autointerp_scores = []
    for bin in interp_df.bin.unique():
        binned_idxs.append(interp_df[interp_df.bin == bin].index.tolist())
        bin_mean_autointerp_scores.append(np.average(interp_df[interp_df.bin == bin].f1_score, weights=interp_df[interp_df.bin == bin].firing_count))
    
    # binned_idxs = [list(idxs) for _, idxs in interp_df.groupby("bin").groups.items()]
    # bin_mean_autointerp_scores = (
    #     interp_df.groupby("bin")
    #             .apply(lambda g: np.average(g.f1_score, weights=g.firing_count))
    #             .tolist()
    # )
    # print(len(binned_idxs), "bins")

    # for i in range(len(binned_idxs)):
    #     bin_data.append({
    #         "bin_idx": i,
    #         "autointerp_score": np.average(interp_df.iloc[binned_idxs[i]]["f1_score"], weights=interp_df.iloc[binned_idxs[i]]["firing_count"]),
    #         "bin_idxs": binned_idxs[i]
    #     })

    # else:
    #     interp_idxs = interpretability_df[interpretability_df["f1_score"] > 0.75].index.tolist()
    #     uninterp_idxs = interpretability_df[interpretability_df["f1_score"] <= 0.75].index.tolist()
    #     binned_idxs = [interp_idxs, uninterp_idxs]
    #     # Select relevant rows
    #     bin_mean_autointerp_scores = [interpretability_df.iloc[interp_idxs]["f1_score"].mean(), interpretability_df.iloc[uninterp_idxs]["f1_score"].mean()]


    print(len(binned_idxs))
    print(len(binned_idxs[0]))
    print(len(binned_idxs[5]))
    print(len(binned_idxs[10]))
    print(len(binned_idxs[15]))
    # print(len(binned_idxs[20]))
    # print(len(binned_idxs[21]))
    # print(len(binned_idxs[22]))
    # print(len(binned_idxs[23]))


    feature_data = {
        i: {
            "loss": [],
            "acc": [],
            "fvu": [],
            "original_loss": [],
            "original_acc": [],
        }
        for i in range(len(binned_idxs))
    }

     # Scale by outlier bin
    # per_bin = (interp_df.groupby("bin")
    #     .agg(total_fire=("firing_count", "sum"),
    #         is_outlier=("is_outlier", "any"))
    #     .reset_index())

    # mean_fire_non_out = per_bin.loc[~per_bin.is_outlier, "total_fire"].mean()
    # per_bin["scale"]  = np.where(per_bin.is_outlier,
    #     n_new_outlier_bins / len(original_outlier_bins),
    #     1.0
    # )

    # scale_map      = per_bin.set_index("bin")["scale"].to_dict()
    # outlier_map    = per_bin.set_index("bin")["is_outlier"].to_dict()
    # bin_fire_total = per_bin.set_index("bin")["total_fire"].to_dict()
    # print(list(scale_map.keys()))

    for batch in tqdm(dataloader):
        tokens = batch["input_ids"].to(device)

        original_loss = F.cross_entropy(
            model(tokens.long()).logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()
        )
        original_acc = (
            (model(tokens.long()).logits[:, :-1].argmax(dim=-1) == tokens[:, 1:])
            .float()
            .mean()
        )
        for i in range(len(binned_idxs)):
            feature_data[i]["original_loss"].append(original_loss.item())
            feature_data[i]["original_acc"].append(original_acc.item())

        for i in range(len(binned_idxs)):
            with ablate_with_mse(
                model,
                hookpoints=[f"model.{hookpoint}"],
                sparse_models={f"model.{hookpoint}": sparse_model},
                ablate_features={
                    f"model.{hookpoint}": binned_idxs[i]
                },
            ) as mses:
                
                output = model(tokens.long())

                loss = F.cross_entropy(
                    output.logits[:, :-1].flatten(0, 1), tokens[:, 1:].flatten()
                )
                feature_data[i]["loss"].append(loss.item())

                accuracy = (
                    (output.logits[:, :-1].argmax(dim=-1) == tokens[:, 1:])
                    .float()
                    .mean()
                )
                feature_data[i]["acc"].append(accuracy.item())

                feature_data[i]["fvu"].append(mses[f"model.{hookpoint}"])

   

    data = []
    for bin_idx, autointerp_score in zip(range(len(binned_idxs)), bin_mean_autointerp_scores):
        # raw_inc = np.mean(feature_data[bin_idx]["loss"]) - np.mean(feature_data[bin_idx]["original_loss"])
        # scaled  = raw_inc * scale_map[bin_idx]

        # orig = np.mean(feature_data[bin_idx]["original_loss"])
        # raw_inc  = np.mean(feature_data[bin_idx]["loss"]) - orig
        # raw_rat  = raw_inc / orig                       # loss increase / original loss
        # scaled   = raw_inc * scale_map[bin_idx]
        # scaled_r = raw_rat * scale_map[bin_idx]

        data.append({
            "hookpoint": hookpoint,
            "bin_idx": bin_idx,
            # "feature_idx": feature_idx,
            "autointerp_score": autointerp_score,
            "ablation_loss": np.mean(feature_data[bin_idx]["loss"]),
            "ablation_acc": np.mean(feature_data[bin_idx]["acc"]),
            "ablation_fvu": np.mean(feature_data[bin_idx]["fvu"]),
            "original_loss": np.mean(feature_data[bin_idx]["original_loss"]),
            "original_acc": np.mean(feature_data[bin_idx]["original_acc"]),
            "loss_increase": np.mean(feature_data[bin_idx]["loss"]) - np.mean(feature_data[bin_idx]["original_loss"]),
            "acc_increase": np.mean(feature_data[bin_idx]["acc"]) - np.mean(feature_data[bin_idx]["original_acc"]),
            # "scaled_loss_increase": scaled,
            # "scaled_loss_increase_ratio": scaled_r,
            # "bin_fire_total": bin_fire_total[bin_idx],
        })


    return pd.DataFrame(data)


@torch.inference_mode()
def get_ablation_df(
    model_name: str,
    sparse_path: str,
    hookpoints: list[str],
    dataset: str,
    batch_size: int,
    interpretability_dfs: dict[str, pd.DataFrame],
    n_seqs: int,
    num_bins: int
):
    results = []
    num_gpus = torch.cuda.device_count()

    ctx = mp.get_context("spawn")
    args_list = [
        (
            hookpoint,
            i,
            batch_size,
            model_name,
            sparse_path,
            dataset,
            interpretability_dfs[hookpoint],
            n_seqs,
            num_bins
        )
        for i, hookpoint in enumerate(hookpoints)
    ]

    with ProcessPoolExecutor(max_workers=num_gpus, mp_context=ctx) as executor:
        for df in executor.map(get_ablation_metrics, args_list):
            results.append(df)

    return pd.concat(results)



def get_autointerp_df(
    scores_path: Path, target_modules: list[str]
) -> pd.DataFrame:
    df_path = f"images/uninterpretable_indices_{str(scores_path).replace('/', '-')}.csv"

    log_path = scores_path.parent / "log" / "hookpoint_firing_counts.pt"
    hookpoint_firing_counts: dict[str, Tensor] = torch.load(log_path, weights_only=True)

    if os.path.exists(df_path):
        df = pd.read_csv(df_path)
    else:
        from delphi.log.result_analysis import build_scores_df
        df = build_scores_df(scores_path, target_modules, hookpoint_firing_counts)
        df.to_csv(df_path, index=False)
    
    
    return df, hookpoint_firing_counts


@torch.inference_mode()
def evaluate_ablation(
    model_name: str,
    sparse_path: str,
    hookpoints: list[str],
    batch_size: int,
    dataset: str,
    ablate_scores_path: str,
    score_type="fuzz",
    n_seqs = 1024,
    num_bins = 200
):
    """
    This method uses caching logic that assumes
    score_type will remain consistent for the same model.
    """

    px = import_plotly() # plotly bug workaround
    torch.manual_seed(42)

    cache_dir = f"images/ce_loss_increases/{model_name.replace('/', '-')}"
    sparse_name = sparse_path.split("/")[-1]
    
    ablation_df_path = f"{cache_dir}/{sparse_name}_grouped_ablation_n={n_seqs}.csv"

    interpretability_dfs = {}
    for hookpoint in hookpoints:
        interp_df, hookpoint_firing_counts = get_autointerp_df(
            Path(ablate_scores_path), 
            [hookpoint], 
        )
        interp_df = interp_df[interp_df["score_type"] == score_type]
        interp_df = interp_df[interp_df["module"] == hookpoint]

        aligned_firing_counts = []
        firing_counts = hookpoint_firing_counts[hookpoint]
        for idx, row in interp_df.iterrows():
            latent_idx = row["latent_idx"]
            aligned_firing_counts.append(firing_counts[latent_idx].item())

        interp_df["firing_count"] = aligned_firing_counts

        # TODO remove this
        # top_indices = interp_df["firing_count"].nlargest(500).index
        # interp_df = interp_df.drop(top_100_indices)
        
        # # 533
        # max_firing_row = interp_df["firing_count"].idxmax()
        # interp_df.loc[max_firing_row]
        # # f1 of 1.
        # breakpoint()

        # Filter dead neurons
        # interp_df = interp_df[interp_df["firing_count"] > 0]

        # Calculate normalized weights that sum to n_features
        interp_df["normalized_weight"] = interp_df["firing_count"] / interp_df["firing_count"].mean()
        weight_sum = interp_df["normalized_weight"].sum()
        print(f"Sum of weights: {weight_sum:.2f}, Number of features: {len(interp_df)}")

        interp_df["normalized_f1"] = interp_df["f1_score"] * interp_df["normalized_weight"]
        interp_df["f1_score_weighted"] = interp_df["f1_score"] * interp_df["normalized_weight"]

        # interp_df["garbage_score"] = (1 - interp_df["f1_score"]) * (
        #     interp_df["normalized_f1"] / interp_df["normalized_f1"].sum() * (1 - interp_df["f1_score"]).sum()
        # )

        # firing_count_sum = float(interp_df["firing_count"].sum())
        # interp_df["firing_frequency"] = interp_df["firing_count"] / firing_count_sum

        # Get firing count-weighted f1 score
        # interp_df["f1_score_weighted"] = interp_df["f1_score"] * interp_df["firing_frequency"]
        print("weighted f1", interp_df["f1_score_weighted"].mean())
        print("unweighted f1", interp_df["f1_score"].mean())

        # Re-calculate with high-frequency features excluded
        # subset_interp_df = interp_df[interp_df["firing_count"] < 100_000]
        # subset_interp_df["normalized_weight"] = subset_interp_df["firing_count"] / subset_interp_df["firing_count"].mean()
        # subset_interp_df["f1_score_weighted"] = subset_interp_df["f1_score"] * subset_interp_df["normalized_weight"]
        # print("subset weighted f1", subset_interp_df["f1_score_weighted"].mean())
        # print("subset unweighted f1", subset_interp_df["f1_score"].mean())

        # Plot firing count vs. interp f1
        num_tokens = 10_000_000
        interp_df["firing_rate"] = interp_df["firing_count"] / num_tokens
        fig = px.scatter(interp_df, x="firing_rate", y="f1_score", log_x=True)
        fig.update_layout(
            xaxis_title="Firing rate",
            yaxis_title="F1 score",
            title="",
            xaxis_range=[-5.4, 0]
        )
        fig.write_image(f"images/{sparse_name}_{hookpoint}_firing_rates.pdf", format="pdf")

        # filter to rows for latent idxs in feature_idx_range
        # interp_df = interp_df[interp_df["latent_idx"].isin(
        #     range(latent_idx_range[0], latent_idx_range[1])
        # )]

        interpretability_dfs[hookpoint] = interp_df

    # ablation_df = pd.read_csv(ablation_df_path)
    ablation_df = get_ablation_df(
        model_name,
        sparse_path,
        hookpoints,
        dataset,
        batch_size,
        interpretability_dfs,
        n_seqs,
        num_bins
    )
     
    print("Done! Results:")
    print(ablation_df.head())
    print(f"Saved to {ablation_df_path}")
    ablation_df.to_csv(ablation_df_path, index=False)

    ablation_df["loss_increase"] = ablation_df["ablation_loss"] - ablation_df["original_loss"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ablation_df["autointerp_score"], 
            y=(
                ablation_df["loss_increase"] / ablation_df["original_loss"]
            ), 
            mode="markers",
        )
    )

    name = f"images/{sparse_name}_binned_ablations_n={n_seqs}_bins={num_bins}.pdf"

    if "bins" in name:
        fig.update_layout(
            yaxis_range=[0.0, 0.02]
        )
    fig.update_layout(
        yaxis_title="Relative loss increase",
        xaxis_title="F1 score",
    )
    fig.write_image(
        name, format="pdf"
    )
    print(f"Saved to {name}")

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

    evaluate_ablation(
        args.base_model,
        args.sparse_model,
        args.hookpoints,
        args.batch_size,
        args.dataset,
        args.ablate_scores_path,
    )


if __name__ == "__main__":
    main()
