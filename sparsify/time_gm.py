import argparse
import timeit
import torch
import numpy as np
import json
import pandas as pd

from sparsify.sparse_coder import SparseCoder
from sparsify.config import SparseCoderConfig

device = "cuda"

def run_benchmark(batch_sizes: list[int], d_models: list[int], num_latents_factors: list[float], 
                  k_values: list[int], num_runs: int = 10):
    """
    Benchmark the performance of groupmax vs topk activation functions in SparseCoder.
    
    Args:
        batch_sizes: List of batch sizes to test
        d_models: List of model dimensions to test
        num_latents_factors: List of expansion factors for num_latents
        k_values: List of k values to test
        num_runs: Number of runs to average over
        use_cuda: Whether to use CUDA
    
    Returns:
        Dictionary of benchmark results
    """
    results = []
    
    for batch_size in batch_sizes:
        for d_model in d_models:
            for factor in num_latents_factors:
                num_latents = int(d_model * factor)
                
                for k in k_values:
                    print(k, d_model, num_latents, batch_size, factor)
                    # # TODO should be able to multiply everything together by the dtype 
                    # # and filter all OOMs for an A10 with 49 GB
                    if factor == 256 and batch_size >= 16384:
                        # prevent OOM 
                        continue

                    if k > num_latents:
                        continue
                        
                    # Create configs for both activation functions
                    topk_cfg = SparseCoderConfig(
                        num_latents=num_latents,
                        k=k,
                        activation="topk"
                    )
                    
                    groupmax_cfg = SparseCoderConfig(
                        num_latents=num_latents,
                        k=k,
                        activation="groupmax"
                    )
                    
                    # Initialize the models 
                    topk_model = SparseCoder(d_model, topk_cfg, device=device)
                    groupmax_model = SparseCoder(d_model, groupmax_cfg, device=device)
                    
                    # Create random input
                    x = torch.randn(batch_size, d_model, device=device)
                    
                    def run_topk_fwd():
                        return topk_model.encode(x)
                    
                    def run_groupmax_fwd():
                        return groupmax_model.encode(x)

                    def topk_fwd_bwd():
                        x.grad = None
                        out = topk_model.encode(x)
                        # Simple loss function
                        loss = out.top_acts.sum()
                        loss.backward() 
                        return out.top_acts

                    def gm_fwd_bwd():
                        x.grad = None
                        out = groupmax_model.encode(x)
                        # Simple loss function
                        loss = out.top_acts.sum()
                        loss.backward()
                        return out.top_acts
                    
                    topk_times = timeit.repeat(
                        lambda: topk_fwd_bwd(),
                        repeat=num_runs,
                        number=10,
                        globals=globals()
                    )

                    topk_fwd_times = timeit.repeat(
                        lambda: run_topk_fwd(),
                        repeat=num_runs,
                        number=10,
                        globals=globals()
                    )
                    
                    groupmax_times = timeit.repeat(
                        lambda: gm_fwd_bwd(),
                        repeat=num_runs,
                        number=10,
                        globals=globals()
                    )

                    groupmax_fwd_times = timeit.repeat(
                        lambda: run_groupmax_fwd(),
                        repeat=num_runs,
                        number=10,
                        globals=globals()
                    )
                    
                    # Record results
                    results.append({
                        "batch_size": batch_size,
                        "d_model": d_model,
                        "num_latents": num_latents,
                        "k": k,
                        "expansion_factor": factor,
                        "topk_mean": np.mean(topk_times),
                        "topk_std": np.std(topk_times),
                        "groupmax_mean": np.mean(groupmax_times),
                        "groupmax_std": np.std(groupmax_times),
                        "groupmax_speedup": np.mean(topk_times) / np.mean(groupmax_times),
                        "topk_fwd_mean": np.mean(topk_fwd_times),
                        "topk_fwd_std": np.std(topk_fwd_times),
                        "groupmax_fwd_mean": np.mean(groupmax_fwd_times),
                        "groupmax_fwd_std": np.std(groupmax_fwd_times),
                        "groupmax_fwd_speedup": np.mean(topk_fwd_times) / np.mean(groupmax_fwd_times),
                    })
                    
                    # Print progress
                    # print(f"Batch: {batch_size}, d_model: {d_model}, latents: {num_latents}, k: {k}")
                    # print(f"  TopK: {results[key]['topk_mean']:.6f}s ± {results[key]['topk_std']:.6f}")
                    # print(f"  GroupMax: {results[key]['groupmax_mean']:.6f}s ± {results[key]['groupmax_std']:.6f}")
                    # print(f"  Speedup: {results[key]['groupmax_speedup']:.2f}x")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark SAE activation functions')
    # Probably don't need to show larger due to using gradient accumulation at larger batches
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[
        1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096, 16384,
    ]) # 65536, 262144, 1048576,  1048576 * 4
    parser.add_argument('--d-models', type=int, nargs='+', default=[768]) # , 2048
    parser.add_argument('--expansion-factors', type=float, nargs='+', default=[16, 32, 64, 128]) 
    parser.add_argument('--k-values', type=int, nargs='+', default=[16, 32, 64, 128, 256, ]) # 512, 1024
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='benchmark_results')
    
    args = parser.parse_args()
    
    results = run_benchmark(
        batch_sizes=args.batch_sizes,
        d_models=args.d_models,
        num_latents_factors=args.expansion_factors,
        k_values=args.k_values,
        num_runs=args.num_runs,
    )
    
    # plot_results(results, args.output_dir)

    # Save df
    df = pd.DataFrame(results)
    df.to_csv(f'{args.output_dir}/benchmark_results.csv', index=False)
    
    # Convert tuple keys to strings for JSON serialization
    # with open(f'{args.output_dir}/benchmark_results.json', 'w') as f:
    #     json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    # Print mean over all experiments
    # mean_topk_time = np.mean([topk_mean for topk_mean in [v["topk_mean"] for v in results.values()]])
    # mean_groupmax_time = np.mean([groupmax_mean for groupmax_mean in [v["groupmax_mean"] for v in results.values()]])
    # print(f"Mean TopK time: {mean_topk_time:.6f}s")
    # print(f"Mean GroupMax time: {mean_groupmax_time:.6f}s")
    
    
    
if __name__ == "__main__":
    main()