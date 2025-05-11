#%%
%env CUDA_VISIBLE_DEVICES=7
import torch
from math import log2

# batch size
B = 2**15
# input dimension
D = 512
# number of experts
E = D * 128
# number of top-k latents
K = 32

input_data = torch.randn(B, D, dtype=torch.bfloat16, device='cuda')
encoder_weight = torch.randn(E, D, dtype=torch.bfloat16, device='cuda')
encoder_weight = torch.nn.Parameter(encoder_weight.requires_grad_(True), requires_grad=True)
#%%
import torch.utils.benchmark

def timing(name, fn, *args):
    fn(*args)
    timer = torch.utils.benchmark.Timer(
        stmt="fn(*args)",
        globals={"fn": fn, "args": args},
        setup="fn(*args)",
    ).blocked_autorange()
    print(name, timer.mean)

@torch.compile
def full_gumbel(x, W_enc):
    preacts = x @ W_enc.T
    grouped = preacts.unflatten(-1, (K, -1))
    gumbeled = torch.nn.functional.gumbel_softmax(grouped, tau=0.5, hard=True, dim=-1)
    values, indices = grouped.max(dim=-1)
    gradient = torch.randn_like(values)
    return torch.autograd.grad(values, W_enc, gradient)

timing("full_gumbel", full_gumbel, input_data, encoder_weight)

#%%
from torch import Tensor
from typing import Literal
import torch.nn.functional as F
import rtopk
from sparsify.kernels import triton_sparse_transpose_dense_matmul
from sparsify.xformers import embedding_bag_bw_rev_indices

@torch.compile
def naive_sparse_gumbel(x, W_enc):
    preacts = x @ W_enc.T
    gumbel_noise = -torch.empty_like(preacts).exponential_().log()
    gumbels = (preacts + gumbel_noise) / 0.5
    grouped = gumbels.unflatten(-1, (K, -1))
    K_ = 4
    _, top_k_indices = rtopk.ops.rtopk(grouped, K_)
    top_k_indices = grouped.argmax(dim=-1)[..., None]

    # grouped_grouped = grouped.unflatten(-1, (K_, -1))
    # indices_ = grouped_grouped.argmax(dim=-1)
    # top_k_indices = indices_ + torch.arange(0, E // K, E // K // K_, device=x.device)

    top_k_indices = top_k_indices + torch.arange(0, E, E // K, device=x.device)[:, None]
    top_k_indices = top_k_indices.flatten(-2, -1)

    grad_values = torch.randn(top_k_indices.shape, device=x.device, dtype=torch.bfloat16)
    indices = top_k_indices

    return embedding_bag_bw_rev_indices(
        indices,
        W_enc,
        grad_values.to(torch.float32),
        x.to(torch.float32),
    )
    # return triton_sparse_transpose_dense_matmul(
    #     indices,
    #     grad_values.to(torch.float32),
    #     x.to(torch.float32),
    #     N=E,
    # )

    # grad_weight = torch.zeros_like(W_enc)
    # # Compute contributions from each top-k element:
    # # computed as grad_values * input for each top-k location.
    # contributions = grad_values.unsqueeze(2) * x.unsqueeze(1)
    # _, _, D = contributions.shape
    # # Flatten contributions to shape (N*k, D)
    # contributions = contributions.reshape(-1, D)

    # # Accumulate contributions into the correct rows of grad_weight.
    # grad_weight.index_add_(0, indices.flatten(), contributions.type_as(encoder_weight))
    # return grad_weight


timing("naive_sparse_gumbel", naive_sparse_gumbel, input_data, encoder_weight)
# %%
