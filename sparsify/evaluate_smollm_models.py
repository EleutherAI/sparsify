import torch
import torch.multiprocessing as mp

from sparsify.evaluate import evaluate

mp.set_start_method("spawn", force=True)


def main():
    torch.manual_seed(42)

    model_name = "HuggingFaceTB/SmolLM2-135M"
    dataset = "EleutherAI/SmolLM2-135M-10B"

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
    sparse_paths = [
        f"checkpoints/{sparse_path}" for sparse_path in unverified_sparse_paths
    ]
    verified = [
        "SmolLM2-135M-topk-adam-1e-4",
        "SmolLM2-135M-topk-adam-5e-4",
        "SmolLM135M2-topk-adam-5e-3",
        "SmolLM135M2-topk-adam-1e-3",
        "SmolLM2-135M-skip-adam-1e-3",
    ]
    sparse_paths.extend(
        [f"checkpoints/verified-paper/{sparse_path}" for sparse_path in verified]
    )
    hookpoints = [
        "layers.0.mlp",
        "layers.9.mlp",
        "layers.18.mlp",
        "layers.27.mlp",
    ]
    batch_size = 64

    for path in sparse_paths:
        evaluate(model_name, path, hookpoints, batch_size, dataset)

    model_name = "HuggingFaceTB/SmolLM2-1.7B"
    sparse_paths = [
        "SmolLM2-1.7B-topk-adam",
        "SmolLM2-1.7B-topk-signum",
        "SmolLM2-1.7B-gm-signum",
        "SmolLM2-1.7B-skip-gm",
        "SmolLM2-1.7B-skip-topk-signum",
    ]
    sparse_paths = [
        f"checkpoints/verified-paper/{sparse_path}" for sparse_path in sparse_paths
    ]
    batch_size = 16
    hookpoints = [
        "layers.0.mlp",
        "layers.7.mlp",
        "layers.14.mlp",
        "layers.21.mlp",
    ]

    for path in sparse_paths:
        try:
            evaluate(model_name, path, hookpoints, batch_size, dataset)
        except Exception as e:
            print(f"Error evaluating {path}: {e}")


if __name__ == "__main__":
    main()
