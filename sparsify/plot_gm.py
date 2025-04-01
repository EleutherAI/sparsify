from pathlib import Path
import pandas as pd
import numpy as np

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


def prepare_dataframe(df: pd.DataFrame):
    df = df.rename(columns={
        "topk_mean": "TopK",
        "groupmax_mean": "GroupMax",
        "topk_std": "TopK Std",
        "groupmax_std": "GroupMax Std",
    })
    
    for column in ["TopK", "GroupMax", "TopK Std", "GroupMax Std"]:
        # Convert to milliseconds
        df[column] = df[column] * 1000

    return df


def create_plots(df, output_dir="images/benchmark_plots"):
    px = import_plotly()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print("plotting")
    
    # Filter for standard conditions
    standard_k = 64
    standard_expansion = 64
    standard_batch = 16384
    standard_d_model = 768

    num_latents = standard_d_model * standard_expansion
    
    # # 1. Plot varying k values (fixed batch_size and expansion_factor)
    df_k = df[(df["batch_size"] == standard_batch) & 
              (df["expansion_factor"] == standard_expansion) &
              (df["d_model"] == standard_d_model)]
    
    fig_k = px.line(
        df_k.sort_values("k"), 
        x="k", 
        y=["TopK", "GroupMax"],
        title=f"Performance vs. k (batch size of {standard_batch}, {num_latents} latents)",
        labels={
            "k": "k",
            "value": "Runtime (ms)",
            "variable": "Activation"
        },
        color_discrete_map={
            "TopK": "#1f77b4",
            "GroupMax": "#ff7f0e"
        },
        markers=True
    )
    
    # Find the appropriate log range for y-axis
    min_time_k = min(df_k["TopK"].min(), df_k["GroupMax"].min())
    max_time_k = max(df_k["TopK"].max(), df_k["GroupMax"].max())
    
    min_exp = int(np.floor(np.log10(min_time_k)))
    max_exp = int(np.ceil(np.log10(max_time_k)))
    
    tick_vals = [10**i for i in range(min_exp, max_exp+1)]
    tick_text = [f"10<sup>{i}</sup>" for i in range(min_exp, max_exp+1)]

    fig_k.update_layout(
        legend=dict(x=0.05, y=0.95, xanchor="left", yanchor="top"),
        # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(
            title="Runtime (ms)",
            type="log",
            tickvals=tick_vals,
            ticktext=tick_text
        )
    )
    
    fig_k.write_image(f"{output_dir}/varying_k_performance.pdf", format="pdf")
    fig_k.write_image(f"{output_dir}/varying_k_performance.png", format="png")
    
    # 2. Plot varying batch sizes (fixed k and expansion_factor)
    df_batch = df[(df["k"] == standard_k) & 
                  (df["expansion_factor"] == standard_expansion) &
                  (df["d_model"] == standard_d_model)]
    
    fig_batch = px.line(
        df_batch.sort_values("batch_size"), 
        x="batch_size", 
        y=["TopK", "GroupMax"],
        title=f"Performance vs. Batch size (k of {standard_k}, {num_latents} latents)",
        labels={
            "batch_size": "Batch size",
            "value": "Runtime (ms)",
            "variable": "Activation"
        },
        log_x=True,
        color_discrete_map={
            "TopK": "#1f77b4",
            "GroupMax": "#ff7f0e"
        },
        markers=True
    )
    
    min_time_batch = min(df_batch["TopK"].min(), df_batch["GroupMax"].min())
    max_time_batch = max(df_batch["TopK"].max(), df_batch["GroupMax"].max())
    
    # Find the appropriate log range
    min_exp = int(np.floor(np.log10(min_time_batch)))
    max_exp = int(np.ceil(np.log10(max_time_batch)))
    
    tick_vals = [10**i for i in range(min_exp, max_exp+1)]
    tick_text = [f"10<sup>{i}</sup>" for i in range(min_exp, max_exp+1)]

    fig_batch.update_layout(
        legend=dict(x=0.05, y=0.95, xanchor="left", yanchor="top"),
        # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(
            title="Runtime (ms)",
            type="log",
            tickvals=tick_vals,
            ticktext=tick_text
        )
    )
    
    print("writing to ", f"{output_dir}/varying_batch_performance.pdf")
    fig_batch.write_image(f"{output_dir}/varying_batch_performance.pdf", format="pdf")
    fig_batch.write_image(f"{output_dir}/varying_batch_performance.png", format="png")
    # # 3. Plot varying number of latents (fixed k and batch_size)
    df_latents = df[(df["k"] == standard_k) & 
                    (df["batch_size"] == standard_batch) &
                    (df["d_model"] == standard_d_model)]
    
    fig_latents = px.line(
        df_latents.sort_values("num_latents"), 
        x="num_latents", 
        y=["TopK", "GroupMax"],
        title=f"Performance vs. Number of Latents (k of {standard_k}, batch size of {standard_batch})",
        labels={
            "num_latents": "Number of Latents",
            "value": "Runtime (ms)",
            "variable": "Activation"
        },
        color_discrete_map={
            "TopK": "#1f77b4",
            "GroupMax": "#ff7f0e"
        },
        markers=True
    )
    
    # Find the appropriate log range for y-axis
    min_time_latents = min(df_latents["TopK"].min(), df_latents["GroupMax"].min())
    max_time_latents = max(df_latents["TopK"].max(), df_latents["GroupMax"].max())
    
    min_exp = int(np.floor(np.log10(min_time_latents)))
    max_exp = int(np.ceil(np.log10(max_time_latents)))
    
    tick_vals = [10**i for i in range(min_exp, max_exp+1)]
    tick_text = [f"10<sup>{i}</sup>" for i in range(min_exp, max_exp+1)]

    fig_latents.update_layout(
        legend=dict(x=0.05, y=0.95, xanchor="left", yanchor="top"),
        # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(
            title="Runtime (ms)",
            type="log",
            tickvals=tick_vals,
            ticktext=tick_text
        )
    )
    
    fig_latents.write_image(f"{output_dir}/varying_latents_performance.pdf", format="pdf")
    fig_latents.write_image(f"{output_dir}/varying_latents_performance.png", format="png")


def main():
    results = pd.read_csv("images/benchmark_results/benchmark_results.csv")
    # results = load_results()
    df = prepare_dataframe(results)
    create_plots(df)
    
    # Display summary statistics
    print(f"Total experiments: {len(df)}")
    print(f"Average speedup across all experiments: {df['groupmax_speedup'].mean():.2f}x")
    print(f"Max speedup: {df['groupmax_speedup'].max():.2f}x at {df.loc[df['groupmax_speedup'].idxmax()][['batch_size', 'k', 'num_latents']].to_dict()}")

if __name__ == "__main__":
    main()