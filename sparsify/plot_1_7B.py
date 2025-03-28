import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import wandb
import plotly.io as pio
import re
from pathlib import Path
import os

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469

def load_wandb_data(run_names: list[str], metrics: list[str], project: str = "eleutherai/sae") -> pd.DataFrame:
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
                            run_data.append({
                                "run_name": run_name,
                                "step": step,
                                "metric": metric,
                                "value": row[metric]
                            })
            
            if run_data:
                run_df = pd.DataFrame(run_data)
                
                deduplicated = []
                for metric_name, group in run_df.groupby('metric'):
                    unique_steps = group.drop_duplicates(subset=['step'], keep='first')
                    deduplicated.append(unique_steps)
                
                if deduplicated:
                    combined_data = pd.concat(deduplicated)
                    data.extend(combined_data.to_dict('records'))
            
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


def plot_metrics_by_lr(regular_df, skip_df, metric_type, layer_nums, lr_value):
    """
    Create subplot figures for a given metric type and learning rate.
    
    Args:
        regular_df: DataFrame with non-skip runs
        skip_df: DataFrame with skip runs
        metric_type: 'fvu' or 'dead_pct'
        layer_nums: List of layer numbers to plot
        lr_value: Learning rate value to filter for
        
    Returns:
        Plotly figure
    """
    # Filter dataframes by learning rate
    skip_df_lr = skip_df[skip_df['lr'] == lr_value]
    regular_df_lr = regular_df[regular_df['lr'] == lr_value]
    
    
    n_rows = len(layer_nums)

    subplot_titles = []
    for layer in layer_nums:
        if metric_type == "fvu":
            subplot_titles.extend([
                f"FVU (SAE Layer {layer} MLP)",
                f"FVU (Transcoder Layer {layer} MLP)"
            ])
        else:
            subplot_titles.extend([
                f"{metric_type.upper()} (SAE Layer {layer} MLP)",
                f"{metric_type.upper()} (Transcoder Layer {layer} MLP)"
            ])

    fig = make_subplots(
        rows=n_rows, 
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )
    
    marker_symbols = ["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down"]
    
    # Plot regular runs (left column)
    for row_idx, layer in enumerate(layer_nums, start=1):
        metric_name = f"{metric_type}/layers.{layer}.mlp"
        layer_data = regular_df_lr[regular_df_lr['metric'] == metric_name]
        
        for i, run_name in enumerate(layer_data['run_name'].unique()):
            run_data = layer_data[layer_data['run_name'] == run_name]
            optimizer = run_data['optimizer'].unique()[0]
            color_idx = i % len(px.colors.qualitative.Plotly)
            color = px.colors.qualitative.Plotly[color_idx]
            
            fig.add_trace(
                go.Scatter(
                    x=run_data['step'],
                    y=run_data['value'],
                    mode='lines+markers',
                    marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=6),
                    line=dict(color=color),
                    name=optimizer.title(),
                    legendgroup=optimizer.title(),
                    showlegend=(row_idx == 1),  # Only show in legend for first row
                ),
                row=row_idx,
                col=1,
            )
    
    # Plot skip runs (right column)
    for row_idx, layer in enumerate(layer_nums, start=1):
        metric_name = f"{metric_type}/layers.{layer}.mlp"
        layer_data = skip_df_lr[skip_df_lr['metric'] == metric_name]
        
        for i, run_name in enumerate(layer_data['run_name'].unique()):
            run_data = layer_data[layer_data['run_name'] == run_name]
            optimizer = run_data['optimizer'].unique()[0]
            color_idx = i % len(px.colors.qualitative.Plotly)
            color = px.colors.qualitative.Plotly[color_idx]
            
            fig.add_trace(
                go.Scatter(
                    x=run_data['step'],
                    y=run_data['value'],
                    mode='lines+markers',
                    marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=6),
                    line=dict(color=color),
                    name=optimizer.title(),
                    legendgroup=optimizer.title(),
                    showlegend=False,  # Don't duplicate in legend
                ),
                row=row_idx,
                col=2,
            )

    fig.update_layout(
        height=300 * n_rows,
        width=1000,
        title=f"{metric_type.upper()} Across Layers (LR: {lr_value})",
        title_x=0.5,
        legend=dict(
            y=0.85,  # Position at 80% of the height (near bottom of first row)
            x=0.52,  # Position at 52% of width (just inside second column)
            xanchor='left',
            yanchor='bottom',
            orientation='v',  # Vertical orientation
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
        gridcolor='lightgray',
    )

    for i in range(1, n_rows + 1):
        for j in range(1, 3):
            # Calculate appropriate log2 tick values
            max_step = max(
                max(regular_df['step'], default=2**10),
                max(skip_df['step'], default=2**10)
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
                gridcolor='lightgray',
                tickvals=log2_ticks,
                ticktext=[f"2<sup>{int(np.log2(val))}</sup>" for val in log2_ticks]
            )

            # Range starts at 0 for both axes
            y_title = "FVU" if metric_type == "fvu" else "Dead Neurons (%)"
            fig.update_yaxes(
                row=i, 
                col=j,
                title_text=y_title if j == 1 else None,
                range=[0, None],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
            )
    
    return fig

def extract_optimizer(run_name):
    """Extract the optimizer type from the run name."""
    if "adam" in run_name.lower():
        return "adam"

    return "signum"


def extract_learning_rate(run_name):
    """Extract the learning rate from the run name."""
    lr_patterns = ['1e-3', '1e-4', '5e-3', '5e-4']
    
    # Check for specific pattern like "adam-5e-4" or "signum-1e-3"
    for pattern in ['adam-', 'signum-']:
        for lr in lr_patterns:
            if f"{pattern}{lr}" in run_name:
                return lr
    
    # Check for patterns like "(adam 5e-4)" or "(1e-3)"
    parenthesis_match = re.search(r'\((.*?)\)', run_name)
    if parenthesis_match:
        content = parenthesis_match.group(1)
        for lr in lr_patterns:
            if lr in content:
                return lr
    
    # Just check for the lr pattern anywhere in the name
    for lr in lr_patterns:
        if lr in run_name:
            return lr
    
    return 'unknown'

def process_data(df, run_names):
    """
    Process the data into two dataframes: one for runs with 'skip' in the name
    and one for runs without 'skip'.
    
    Args:
        df: DataFrame with the raw data
        run_names: List of run names
        
    Returns:
        Two DataFrames (regular_df, skip_df)
    """

    df['lr'] = df['run_name'].apply(extract_learning_rate)
    df['optimizer'] = df['run_name'].apply(extract_optimizer)    
    df['skip'] = df['run_name'].str.contains('skip', case=False)
    df['layer'] = df['metric'].str.extract(r'layers\.([0-9]+)\.mlp')
    
    df['activation'] = 'JumpReLU'  # Default value
    df.loc[df['run_name'].str.contains('gm', case=False), 'activation'] = 'GroupMax'
    df.loc[df['run_name'].str.contains('topk', case=False), 'activation'] = 'TopK'

    skip_df = df[df['skip']].sort_values('step')
    regular_df = df[~df['skip']].sort_values('step')
    
    def filter_log2_intervals(df):
        filtered_groups = []
        for _, group in df.groupby(['run_name', 'metric']):
            steps = group['step'].unique()
            max_step = steps.max()
            log2_steps = [min(steps, key=lambda x: abs(x - 2**p))
                          for p in range(int(np.log2(max_step)) + 1)]
            filtered_groups.append(group[group['step'].isin(log2_steps)])
        return pd.concat(filtered_groups, ignore_index=True)

    skip_df = filter_log2_intervals(skip_df)
    regular_df = filter_log2_intervals(regular_df)

    return df, regular_df, skip_df
    
def aggregate_optimizer_performance(df):
    fvu_df = df[df['metric'].str.contains('fvu', case=False)]

    min_fvu_summary = (
        fvu_df.groupby(['optimizer', 'activation', 'skip', 'layer'])['value']
        .min()
        .reset_index()
        .pivot(index=['optimizer', 'activation', 'layer'], columns='skip', values='value')
        .rename(columns={False: 'regular_min_fvu', True: 'skip_min_fvu'})
        .reset_index()
    )

    return min_fvu_summary.sort_values(['layer', 'activation'])


def hex_to_rgba(hex_color, opacity=0.5):
    """Convert hex color to rgba with opacity."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r}, {g}, {b}, {opacity})"

def plot_metrics(regular_df, skip_df, metric_type, layer_nums):
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
    n_rows = len(layer_nums)

    subplot_titles = []
    for layer in layer_nums:
        if metric_type == "fvu":
            subplot_titles.extend([
                f"FVU (Transcoder Layer {layer} MLP)",
                f"FVU (SAE Layer {layer} MLP)",
            ])
        else:
            subplot_titles.extend([
                f"{metric_type.upper()} (Transcoder Layer {layer} MLP)",
                f"{metric_type.upper()} (SAE Layer {layer} MLP)",
            ])

    fig = make_subplots(
        rows=n_rows, 
        cols=2,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )
    
    marker_symbols = ["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down"]
    
    # Plot regular runs (left column)
    for row_idx, layer in enumerate(layer_nums, start=1):
        metric_name = f"{metric_type}/layers.{layer}.mlp"
        layer_data = skip_df[skip_df['metric'] == metric_name]
        
        for i, (name, group) in enumerate(layer_data.groupby(['activation', 'optimizer'])):
            activation, optimizer = name
            display_name = f"{activation}-{optimizer}"
            
            color_idx = i % len(px.colors.qualitative.Plotly)
            color = px.colors.qualitative.Plotly[color_idx]
            
            fig.add_trace(
                go.Scatter(
                    x=group['step'],
                    y=group['value'],
                    mode='lines+markers',
                    marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=6),
                    line=dict(color=color),
                    name=display_name,
                    legendgroup=display_name,
                    showlegend=(row_idx == 1),  # Only show in legend for first row
                ),
                row=row_idx,
                col=1,
            )

        # for i, run_name in enumerate(layer_data['run_name'].unique()):
        #     run_data = layer_data[layer_data['run_name'] == run_name]
        #     color_idx = i % len(px.colors.qualitative.Plotly)
        #     color = px.colors.qualitative.Plotly[color_idx]
            
        #     fig.add_trace(
        #         go.Scatter(
        #             x=run_data['step'],
        #             y=run_data['value'],
        #             mode='lines+markers',
        #             marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=6),
        #             line=dict(color=color),
        #             name=run_name,
        #             legendgroup=run_name,
        #             showlegend=(row_idx == 1),  # Only show in legend for first row
        #         ),
        #         row=row_idx,
        #         col=1,
        #     )
    
    # Plot skip runs (right column)
    for row_idx, layer in enumerate(layer_nums, start=1):
        metric_name = f"{metric_type}/layers.{layer}.mlp"
        layer_data = regular_df[regular_df['metric'] == metric_name]

        for i, (name, group) in enumerate(layer_data.groupby(['activation', 'optimizer'])):
            activation, optimizer = name
            display_name = f"{activation}-{optimizer}"
            
            color_idx = i % len(px.colors.qualitative.Plotly)
            color = px.colors.qualitative.Plotly[color_idx]
            
            fig.add_trace(
                go.Scatter(
                    x=group['step'],
                    y=group['value'],
                    mode='lines+markers',
                    marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=6),
                    line=dict(color=color),
                    name=display_name,
                    legendgroup=display_name,
                    showlegend=False,  # Don't duplicate in legend
                ),
                row=row_idx,
                col=2,
            )
        
        # for i, run_name in enumerate(layer_data['run_name'].unique()):
        #     run_data = layer_data[layer_data['run_name'] == run_name]
        #     color_idx = i % len(px.colors.qualitative.Plotly)
        #     color = px.colors.qualitative.Plotly[color_idx]
            
        #     fig.add_trace(
        #         go.Scatter(
        #             x=run_data['step'],
        #             y=run_data['value'],
        #             mode='lines+markers',
        #             marker=dict(symbol=marker_symbols[i % len(marker_symbols)], size=6),
        #             line=dict(color=color),
        #             name=run_name,
        #             legendgroup=run_name,
        #             showlegend=False,  # Don't duplicate in legend
        #         ),
        #         row=row_idx,
        #         col=2,
        #     )

    fig.update_layout(
        height=300 * n_rows,
        width=1000,
        title=f"{metric_type.upper()} Across Layers",
        title_x=0.5,
        legend=dict(
            y=1.02,
            x=0.5,
            xanchor='center',
            yanchor='bottom',
            orientation='h',
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
        gridcolor='lightgray',
    )

    for i in range(1, n_rows + 1):
        for j in range(1, 3):
            # Calculate appropriate log2 tick values
            max_step = max(
                max(regular_df['step'], default=2**10),
                max(skip_df['step'], default=2**10)
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
                gridcolor='lightgray',
                tickvals=log2_ticks,
                ticktext=[f"2<sup>{int(np.log2(val))}</sup>" for val in log2_ticks]
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
                gridcolor='lightgray',

            )
    
    return fig

def main(): 
    images_path = Path("images")
    images_path.mkdir(exist_ok=True)

    run_names = [
        "SmolLM2-1.7B-topk-adam",
        "SmolLM2-1.7B-topk-signum",
        "SmolLM2-1.7B-gm-signum",
        "SmolLM2-1.7B-skip-gm",
        "SmolLM2-1.7B-skip-topk-signum",
    ]
    if not run_names:
        raise NotImplementedError("This script is not yet implemented")
    
    layer_nums = [0, 7, 14, 21]
    metrics = []
    for num in layer_nums:
        metrics.append(f"fvu/layers.{num}.mlp")
        metrics.append(f"dead_pct/layers.{num}.mlp")
    
    if os.path.exists("1_7B_optimizer_data.csv"):
        df = pd.read_csv("1_7B_optimizer_data.csv")
    else:
        df = load_wandb_data(run_names, metrics)
        df.to_csv("1_7B_optimizer_data.csv", index=False)

    df, regular_df, skip_df = process_data(df, run_names)
    optimizer_performance = aggregate_optimizer_performance(df)
    print(optimizer_performance)
    
    fvu_fig = plot_metrics(regular_df, skip_df, "fvu", layer_nums)
    dead_pct_fig = plot_metrics(regular_df, skip_df, "dead_pct", layer_nums)
    
    fvu_fig.write_image(images_path / "1.7B_fvu_metrics.pdf", format="pdf")
    dead_pct_fig.write_image(images_path / "1.7B_dead_neurons_metrics.pdf", format="pdf")

if __name__ == "__main__":
    main()