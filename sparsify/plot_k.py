import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
from plotly.subplots import make_subplots

import wandb

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469

from delphi.log.result_analysis import build_scores_df


def load_wandb_data(
    run_names: list[str], metrics: list[str], project: str = "eleutherai/sparsify"
) -> pd.DataFrame:
    """
    Load specific metrics data from WandB runs.

    Args:
        run_names: List of run names to fetch data for
        metrics: List of metrics to extract (e.g., "fvu/layers.0.mlp")
        project: WandB project path

    Returns:
        DataFrame with columns for run_name, step, metric, and value
    """
    api = wandb.Api(timeout=1000)
    data = []

    for run_name in run_names:

        # Find the run by name
        try:
            # Try exact match first
            runs = api.runs(project, {"display_name": run_name})

            if not runs:
                # Try partial match if exact match fails
                all_runs = api.runs(project)
                runs = [run for run in all_runs if run_name in run.name]

            if not runs:
                print(f"Warning: No runs found for {run_name}")
                continue

            # Take the most recent run if multiple matches
            matching_runs = sorted(runs, key=lambda x: x.created_at, reverse=True)
            print(f"Found {len(matching_runs)} runs for {run_name}")
            # run = matching_runs[0]

            # history = run.scan_history(keys=["_step"] + metrics)

            run_data = []
            for run in matching_runs:
                history = run.scan_history(keys=["_step"] + metrics)

                for row in history:
                    if "_step" not in row:
                        continue

                    step = row["_step"]

                    for metric in metrics:
                        if metric in row and not pd.isna(row[metric]):
                            run_data.append(
                                {
                                    "run_name": run_name,
                                    "step": step,
                                    "metric": metric,
                                    "value": row[metric],
                                }
                            )

            # for row in history:
            #     if "_step" not in row:
            #         continue

            #     step = row["_step"]

            #     for metric in metrics:
            #         if metric in row and not pd.isna(row[metric]):
            #             data.append({
            #                 "run_name": run_name,
            #                 "step": step,
            #                 "metric": metric,
            #                 "value": row[metric]
            #             })

        except Exception as e:
            print(f"Error processing run {run_name}: {e}")

    return pd.DataFrame(data)


def process_data(df):
    """
    Args:
        df: DataFrame with the raw data

    Returns:
        Step filtered df
    """

    df["skip"] = df["run_name"].str.contains("skip", case=False)
    df["layer"] = df["metric"].str.extract(r"layers\.([0-9]+)\.mlp")

    df["activation"] = "Continuous"  # Default value
    df.loc[df["run_name"].str.contains("binary", case=False), "activation"] = "Binary"

    def filter_log2_intervals(df):
        df = (
            df.sort_values("step", ascending=False)
            .groupby(["run_name", "step", "metric"], as_index=False)
            .first()  # or .mean() if appropriate
        )

        filtered_groups = []
        for _, group in df.groupby(["run_name", "metric"]):
            steps = group["step"].unique()
            max_step = steps.max()
            log2_steps = [
                min(steps, key=lambda x: abs(x - 2**p))
                for p in range(int(np.log2(max_step)) + 1)
            ]

            # group = group.sort_values('step', ascending=False)
            # group = group.drop_duplicates(subset=['run_name', 'step'])

            filtered_groups.append(group[group["step"].isin(log2_steps)])

            # If there are duplicates for the same run name, keep the latest one
            # group by run_name and step and keep the latest one

            # deduplicated = run_df.sort_values('value').groupby(['metric', 'step']).first().reset_index()
            # filtered_groups.append(group)

        print("len df = ", len(pd.concat(filtered_groups, ignore_index=True)))
        return pd.concat(filtered_groups, ignore_index=True)

    df = filter_log2_intervals(df)

    return df.sort_values("step")


def hex_to_rgba(hex_color, opacity=0.5):
    """Convert hex color to rgba with opacity."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r}, {g}, {b}, {opacity})"


def plot_metrics(autoencoder_df, skip_df, metric_type, layer_nums):
    """
    Create subplot figures for a given metric type.

    Args:
        regular_df: DataFrame with non-skip runs
        skip_df: DataFrame with skip runs
        metric_type: 'fvu' or 'dead_pct'
        layer_nums: List of layer numbers to plot

    Returns:
        Plotly figure
    """
    skip_df = skip_df.sort_values("step")
    autoencoder_df = autoencoder_df.sort_values("step")

    activation_colors = {
        "Binary": px.colors.qualitative.Plotly[0],
        "Continuous": px.colors.qualitative.Plotly[1],
    }

    n_rows = len(layer_nums)

    subplot_titles = []
    for layer in layer_nums:
        if metric_type == "fvu":
            subplot_titles.extend(
                [
                    f"FVU (Transcoder Layer {layer} MLP)",
                    f"FVU (SAE Layer {layer} MLP)",
                ]
            )
        else:
            subplot_titles.extend(
                [
                    f"{metric_type.upper()} (Transcoder Layer {layer} MLP)",
                    f"{metric_type.upper()} (SAE Layer {layer} MLP)",
                ]
            )

    fig = make_subplots(
        rows=n_rows,
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    marker_symbols = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "triangle-down",
    ]

    # Plot skip runs (left column)
    for row_idx, layer in enumerate(layer_nums, start=1):
        metric_name = f"{metric_type}/layers.{layer}.mlp"
        layer_data = skip_df[skip_df["metric"] == metric_name]

        # Mean run as line
        for i, (name, group) in enumerate(layer_data.groupby(["activation"])):
            activation = name[0]
            display_name = activation

            # Use consistent color for each activation type
            color = activation_colors[activation]

            # Calculate mean values per step
            mean_data = group.groupby("step")["value"].mean().reset_index()

            fig.add_trace(
                go.Scatter(
                    x=mean_data["step"],
                    y=mean_data["value"],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=display_name,
                    legendgroup=display_name,
                    showlegend=(row_idx == 1),  # Only show in legend for first row
                ),
                row=row_idx,
                col=1,
            )

        # Then plot individual runs (seeds) as semi-transparent dots with matching colors
        for i, run_name in enumerate(layer_data["run_name"].unique()):
            run_data = layer_data[layer_data["run_name"] == run_name]
            # Get the activation type for this run
            activation = run_data["activation"].iloc[0]

            # Use the same color as the activation line, but with transparency
            color = activation_colors[activation]
            rgba_color = hex_to_rgba(color, 0.3)

            fig.add_trace(
                go.Scatter(
                    x=run_data["step"],
                    y=run_data["value"],
                    mode="markers",
                    marker=dict(
                        symbol=marker_symbols[i % len(marker_symbols)],
                        size=6,
                        color=rgba_color,
                    ),
                    name=run_name,
                    legendgroup=activation,  # Group with the activation type
                    showlegend=False,  # Don't show individual runs in legend
                ),
                row=row_idx,
                col=1,
            )

    # Plot SAE runs (right column)
    for row_idx, layer in enumerate(layer_nums, start=1):
        metric_name = f"{metric_type}/layers.{layer}.mlp"
        layer_data = autoencoder_df[autoencoder_df["metric"] == metric_name]

        # Mean run as line
        for i, (name, group) in enumerate(layer_data.groupby(["activation"])):
            activation = name[0]
            display_name = activation

            color = activation_colors[activation]

            mean_data = group.groupby("step")["value"].mean().reset_index()

            fig.add_trace(
                go.Scatter(
                    x=mean_data["step"],
                    y=mean_data["value"],
                    mode="lines",
                    line=dict(color=color),
                    name=display_name,
                    legendgroup=display_name,
                    showlegend=False,  # Don't duplicate in legend
                ),
                row=row_idx,
                col=2,
            )

        for i, run_name in enumerate(layer_data["run_name"].unique()):
            run_data = layer_data[layer_data["run_name"] == run_name]
            activation = run_data["activation"].iloc[0]

            color = activation_colors[activation]
            rgba_color = hex_to_rgba(color, 0.3)

            fig.add_trace(
                go.Scatter(
                    x=run_data["step"],
                    y=run_data["value"],
                    mode="markers",
                    marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=6),
                    line=dict(color=rgba_color),
                    name=run_name,
                    legendgroup=activation,
                    showlegend=False,  # Don't duplicate in legend
                ),
                row=row_idx,
                col=2,
            )

    fig.update_layout(
        height=300 * n_rows,
        width=1000,
        title=f"{metric_type.upper()} Across Layers",
        title_x=0.5,
        legend=dict(
            y=1.02,
            x=0.5,
            xanchor="center",
            yanchor="bottom",
            orientation="h",
            font=dict(size=10),
        ),
        margin=dict(l=50, r=20, t=80, b=50),
    )

    # Update axes
    fig.update_xaxes(
        title_text="Steps",
        type="log",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
    )

    for i in range(1, n_rows + 1):
        for j in range(1, 3):
            # Calculate appropriate log2 tick values
            max_step = max(
                max(autoencoder_df["step"], default=2**10),
                max(skip_df["step"], default=2**10),
            )

            log2_ticks = []
            power = 0
            while 2**power <= max_step:
                log2_ticks.append(2**power)
                power += 1

            fig.update_xaxes(
                row=i,
                col=j,
                type="log",
                title_text="Steps" if i == n_rows else None,
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
                tickvals=log2_ticks,
                ticktext=[f"2<sup>{int(np.log2(val))}</sup>" for val in log2_ticks],
            )

            # Range starts at 0 for both axes
            y_title = "FVU" if metric_type == "fvu" else "Dead Neurons (%)"
            fig.update_yaxes(
                row=i,
                col=j,
                title_text=y_title if j == 1 else None,
                range=[0, None] if metric_type == "fvu" else [0, 1],
                showgrid=True,
                gridwidth=1,
                gridcolor="lightgray",
            )

    return fig


def agg_stats(df):
    g = (
        df.groupby("k")["acc_decrease"]
        .agg(["mean", "std", "count"])  # count == num seeds
        .reset_index()
    )
    g["sem"] = g["std"] / np.sqrt(g["count"])  # 1 σ or use 1.96*sem for 95 % CI
    return g


def main():
    images_path = Path("images")
    images_path.mkdir(exist_ok=True)

    data_path = Path("images/ce_loss_increases/HuggingFaceTB-SmolLM2-135M")
    skip_run_names = [
        "SmolLM2-135M-skip-topk-s=1-k=16",
        "SmolLM2-135M-skip-topk-s=2-k=16",
        "SmolLM2-135M-skip-topk-s=3-k=16",
        "SmolLM2-135M-skip-binary-s=1-k=16",
        "SmolLM2-135M-skip-binary-s=2-k=16",
        "SmolLM2-135M-skip-binary-s=3-k=16",
        "SmolLM2-135M-topk-s=1",
        "SmolLM2-135M-topk-s=2",
        "SmolLM2-135M-topk-s=3",
        "SmolLM2-135M-topk-binary-s=2",
        "SmolLM2-135M-topk-binary-s=3",
        "SmolLM2-135M-skip-binary",
        "SmolLM2-135M-skip-topk-s=1-k=64-5e-3",
        "SmolLM2-135M-skip-topk-s=2-k=64-5e-3",
        "SmolLM2-135M-skip-topk-s=3-k=64-5e-3",
        "SmolLM2-135M-skip-binary-topk-k=64-s=1" "SmolLM2-135M-topk-binary-s=2-k=64",
        "SmolLM2-135M-topk-binary-s=3-k=64",
        "SmolLM2-135M-skip-topk-binary-s=1-k=128",
        "SmolLM2-135M-skip-topk-binary-s=2-k=128",
        "SmolLM2-135M-skip-topk-binary-s=3-k=128",
        "SmolLM2-135M-skip-topk-s=1-k=128",
        "SmolLM2-135M-skip-topk-s=2-k=128",
        "SmolLM2-135M-skip-topk-s=3-k=128",
    ]
    sae_run_names = [
        "SmolLM2-135M-sae-topk-s=1-k=16",
        "SmolLM2-135M-sae-topk-s=2-k=16",
        "SmolLM2-135M-sae-topk-s=3-k=16",
        "SmolLM2-135M-sae-topk-s=1-k=32",
        "SmolLM2-135M-sae-topk-s=2-k=32",
        "SmolLM2-135M-sae-topk-s=3-k=32",
        "SmolLM2-135M-sae-topk-s=1-k=64",
        "SmolLM2-135M-sae-topk-s=2-k=64",
        "SmolLM2-135M-sae-topk-s=3-k=64",
        "SmolLM2-135M-sae-topk-s=1-k=128",
        "SmolLM2-135M-sae-topk-s=2-k=128",
        "SmolLM2-135M-sae-topk-s=3-k=128",
        # Seed 1 done, needs eval + autointerp
        "SmolLM2-135M-sae-topk-binary-s=2-k=16-sig",
        "SmolLM2-135M-sae-topk-binary-s=3-k=16-sig",
        # Seed 1 done, needs eval + autointerp
        "SmolLM2-135M-sae-topk-binary-s=2-k=32-sig",
        "SmolLM2-135M-sae-topk-binary-s=3-k=32-sig",
        "SmolLM2-135M-sae-topk-binary-s=1-k=64-sig"
        "SmolLM2-135M-sae-topk-binary-s=2-k=64-sig"
        "SmolLM2-135M-sae-topk-binary-s=3-k=64-sig",
        # s=1 running on BYU (also some alternate lrs in a separate run)
        "SmolLM2-135M-sae-binary-s=2-k=128-sig",
        "SmolLM2-135M-sae-binary-s=3-k=128-sig",
    ]

    # "/mnt/ssd-1/lucia/sparsify/checkpoints/SmolLM2-135M-sae-binary-s=2-k=128-sig"
    # "/mnt/ssd-1/lucia/sparsify/checkpoints/SmolLM2-135M-sae-topk-binary-s=1-k=64-sig"
    # "/mnt/ssd-1/lucia/sparsify/checkpoints/SmolLM2-135M-sae-topk-binary-s=2-k=64-sig"
    # "/mnt/ssd-1/lucia/sparsify/checkpoints/SmolLM2-135M-sae-topk-binary-s=2-k=32-sig"
    # "/mnt/ssd-1/lucia/sparsify/checkpoints/SmolLM2-135M-sae-topk-binary-s=2-k=16-sig"

    layer_nums = [0, 9, 18, 27]
    metrics = []
    for num in layer_nums:
        metrics.append(f"fvu/layers.{num}.mlp")
        metrics.append(f"dead_pct/layers.{num}.mlp")

    if os.path.exists("135M_bae_data.csv"):
        df = pd.read_csv("135M_bae_data.csv")
        skip_df = df[df["skip"]]
        sae_df = df[~df["skip"]]
    else:
        skip_df = load_wandb_data(skip_run_names, metrics)
        skip_df["skip"] = True
        sae_df = load_wandb_data(sae_run_names, metrics)
        sae_df["skip"] = False

        pd.concat([skip_df, sae_df]).to_csv("135M_bae_data.csv", index=False)

    skip_df = process_data(skip_df)
    sae_df = process_data(sae_df)

    for df in [skip_df, sae_df]:
        if "k" not in df.columns:
            df["k"] = (
                df["run_name"]
                .str.split("-k=")
                .str[1]
                .str.split("_")
                .str[0]
                .str.split("-")
                .str[0]
            )
            df.loc[df["k"] == "", "k"] = "32"
            df["k"] = df["k"].fillna(32)
            df["k"] = df["k"].astype(int)

    for k in [16, 32, 64, 128]:
        sae_df_k = sae_df[sae_df["k"] == k]
        skip_df_k = skip_df[skip_df["k"] == k]

        fvu_fig = plot_metrics(sae_df_k, skip_df_k, "fvu", layer_nums)
        dead_pct_fig = plot_metrics(sae_df_k, skip_df_k, "dead_pct", layer_nums)

        fvu_fig.write_image(images_path / f"btc_fvu_metrics_k={k}.pdf", format="pdf")
        dead_pct_fig.write_image(
            images_path / f"btc_dead_neurons_metrics_k={k}.pdf", format="pdf"
        )

    # Print final loss, acc, and fvu for each run
    sae_binary_evaluate_names = [
        "SmolLM2-135M-sae-topk-binary-s=2-k=16-sig_loss_fvu",
        "SmolLM2-135M-sae-topk-binary-s=3-k=16-sig_loss_fvu",
        # Seed 1 in progress on BYU and locally
        "SmolLM2-135M-sae-topk-binary-s=1-k=32-sig_loss_fvu",  # Might have copied an old checkpoint
        "SmolLM2-135M-sae-topk-binary-s=2-k=32-sig_loss_fvu",
        "SmolLM2-135M-sae-topk-binary-s=3-k=32-sig_loss_fvu",
        "SmolLM2-135M-sae-topk-binary-s=1-k=64-sig_loss_fvu",
        "SmolLM2-135M-sae-topk-binary-s=2-k=64-sig_loss_fvu",
        "SmolLM2-135M-sae-topk-binary-s=3-k=64-sig_loss_fvu",
        # s1 still running
        "SmolLM2-135M-sae-binary-s=2-k=128-sig_loss_fvu",
        "SmolLM2-135M-sae-binary-s=3-k=128-sig_loss_fvu",
    ]
    skip_binary_evaluate_names = [
        "SmolLM2-135M-skip-binary-s=1-k=16_loss_fvu",
        "SmolLM2-135M-skip-binary-s=2-k=16_loss_fvu",
        "SmolLM2-135M-skip-binary-s=3-k=16_loss_fvu",
        # k=32
        "SmolLM2-135M-topk-binary-s=2_loss_fvu",
        "SmolLM2-135M-topk-binary-s=3_loss_fvu",
        "best_loss_fvu",
        "SmolLM2-135M-skip-topk-binary-k=64-s=1_loss_fvu",
        "SmolLM2-135M-topk-binary-s=2-k=64_loss_fvu",
        "SmolLM2-135M-topk-binary-s=3-k=64_loss_fvu",
        "SmolLM2-135M-skip-topk-binary-s=1-k=128_loss_fvu",
        "SmolLM2-135M-skip-topk-binary-s=2-k=128_loss_fvu",
        "SmolLM2-135M-skip-topk-binary-s=3-k=128_loss_fvu",
    ]
    # From BYU
    sae_topk_evaluate_names = [
        "SmolLM2-135M-sae-topk-s=1-k=16_loss_fvu",
        "SmolLM2-135M-sae-topk-s=2-k=16_loss_fvu",
        "SmolLM2-135M-sae-topk-s=3-k=16_loss_fvu",
        "SmolLM2-135M-sae-topk-s=1-k=32_loss_fvu",
        "SmolLM2-135M-sae-topk-s=2-k=32_loss_fvu",
        "SmolLM2-135M-sae-topk-s=3-k=32_loss_fvu",
        "SmolLM2-135M-sae-topk-s=1-k=64_loss_fvu",
        "SmolLM2-135M-sae-topk-s=2-k=64_loss_fvu",
        "SmolLM2-135M-sae-topk-s=3-k=64_loss_fvu",
        "SmolLM2-135M-sae-topk-s=1-k=128_loss_fvu",
        "SmolLM2-135M-sae-topk-s=2-k=128_loss_fvu",
        "SmolLM2-135M-sae-topk-s=3-k=128_loss_fvu",
    ]
    skip_topk_evaluate_names = [
        "SmolLM2-135M-skip-topk-s=1-k=16_loss_fvu",
        "SmolLM2-135M-skip-topk-s=2-k=16_loss_fvu",
        "SmolLM2-135M-skip-topk-s=3-k=16_loss_fvu",
        "SmolLM2-135M-topk-s=1_loss_fvu",
        "SmolLM2-135M-topk-s=2_loss_fvu",
        "SmolLM2-135M-topk-s=3_loss_fvu",
        "SmolLM2-135M-topk-s=3-k=64_loss_fvu",
        "SmolLM2-135M-topk-s=1-k=64_loss_fvu",
        "SmolLM2-135M-skip-topk-s=2-k=128_loss_fvu",
        "SmolLM2-135M-skip-topk-s=3-k=128_loss_fvu",
    ]

    def concat_dfs(names, skip=False, activation="Binary"):
        dfs = []
        for i, name in enumerate(names):
            df = pd.read_csv(data_path / f"{name}.csv")

            if "k" not in df.columns:
                if "k=" in name:
                    k_str = name.split("-k=")[1].split("_")[0].split("-")[0]
                    df["k"] = k_str
                else:
                    df["k"] = 32

            df["seed"] = i

            df.rename(
                columns={
                    "sparse_loss_increase": "loss_increase",
                    "sparse_acc_decrease": "acc_decrease",
                    "sparse_loss": "loss",
                    "sparse_fvu": "fvu",
                    "sparse_acc": "acc",
                },
                inplace=True,
            )

            df["skip"] = skip
            df["activation"] = activation

            # Todo add autointerp score_df to dfs from
            short_name = name.replace("_loss_fvu", "")
            scores_df_path = (
                f"/mnt/ssd-1/lucia/delphi/results/{short_name}/scores_df.csv"
            )
            if os.path.exists(scores_df_path):
                scores_df = pd.read_csv(scores_df_path)
            else:
                base_interp_path = f"/mnt/ssd-1/lucia/delphi/results/{short_name}"
                if not os.path.exists(base_interp_path):
                    print(f"Missing interp results for {short_name}")
                    continue

                hookpoint_firing_counts = torch.load(
                    f"{base_interp_path}/log/hookpoint_firing_counts.pt"
                )
                scores_df = build_scores_df(
                    scores_df_path,
                    [f"layers.{layer}.mlp" for layer in layer_nums],
                    hookpoint_firing_counts,
                )
                scores_df.to_csv(scores_df_path, index=False)

            # Set the average F1 over the layer to the F1 in the corresponding row
            df[df["hookpoint"] == f"layers.{layer_nums[0]}.mlp"]["f1_score"] = (
                scores_df[scores_df["module"] == f"layers.{layer_nums[0]}.mlp"][
                    "f1_score"
                ].mean()
            )
            dfs.append(df)

        return pd.concat(dfs)

    layer_nums = [9, 18, 27]
    binary_df = concat_dfs(skip_binary_evaluate_names, skip=True, activation="Binary")
    topk_df = concat_dfs(skip_topk_evaluate_names, skip=True, activation="TopK")
    sae_topk_df = concat_dfs(sae_topk_evaluate_names, skip=False, activation="TopK")
    sae_binary_df = concat_dfs(
        sae_binary_evaluate_names, skip=False, activation="Binary"
    )

    colors = {
        "TopK": px.colors.qualitative.Plotly[0],
        "Binary": px.colors.qualitative.Plotly[1],
    }

    def stats(df, metric):
        g = df.groupby("k")[metric]
        return g.mean().to_frame("mean").assign(sem=g.sem()).reset_index()

    metrics = {
        "acc_decrease": "Accuracy decrease",
        "loss_increase": "Relative loss increase",
        "f1_score": "F1 fuzzing score",
    }

    y_ranges = {
        ("btc", "acc_decrease"): [0, 0.012],
        ("bae", "acc_decrease"): [0, 0.03],
        ("btc", "loss_increase"): [0, 0.07],  # placeholder
        ("bae", "loss_increase"): [0, 0.3],  # placeholder
        ("btc", "f1_score"): [0, 1.0],
        ("bae", "f1_score"): [0, 1.0],
    }

    runs = [
        ("btc", [binary_df, topk_df]),
        ("bae", [sae_binary_df, sae_topk_df]),
    ]

    for tag, sources in runs:
        for metric, ylab in metrics.items():
            fig = make_subplots(
                rows=1,
                cols=len(layer_nums),
                shared_yaxes=True,
                subplot_titles=[f"Layer {layer}" for layer in layer_nums],
                horizontal_spacing=0.02,
                vertical_spacing=0.08,
            )

            for col, layer in enumerate(layer_nums, 1):
                for src in sources:
                    d = src[src["hookpoint"] == f"layers.{layer}.mlp"]
                    act = d["activation"].iloc[0]
                    s = stats(d, metric)

                    fig.add_trace(
                        go.Scatter(
                            x=s["k"],
                            y=s["mean"],
                            mode="lines+markers",
                            name=act,
                            showlegend=col == 1,
                            line=dict(color=colors[act]),
                            error_y=dict(
                                type="data", array=s["sem"], arrayminus=s["sem"]
                            ),
                        ),
                        row=1,
                        col=col,
                    )
                fig.update_xaxes(title_text="k", row=1, col=col)

            yr = y_ranges.get((tag, metric))
            fig.update_layout(
                yaxis_title=ylab,
                **({"yaxis_range": yr} if yr else {}),
                legend=dict(yanchor="top", y=0.95, xanchor="right", x=0.98),
                width=800,
                height=500,
            )
            fig.write_image(images_path / f"{tag}_{metric}_by_k.pdf")


if __name__ == "__main__":
    main()
