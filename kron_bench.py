#%%
%env CUDA_VISIBLE_DEVICES=7
import torch
from math import log2
import torch
torch.set_grad_enabled(False)

B = 2**15
N = 1024
M = N * 64
U = 4
N_b = 4
M_b = 2
N_n = N // N_b
M_n = M // M_b

vals = torch.randn(B, N, dtype=torch.bfloat16, device='cuda')
matrix_inner = torch.randn(M_b, U, N_b, dtype=torch.bfloat16, device='cuda')
matrix_outer = torch.randn(M_n, U, N_n, dtype=torch.bfloat16, device='cuda')
#%%
flops_estimate_matmul = 2 * B * N * M * U
flops_estimate_kron = 2 * B * M_b * (N // N_b) * (M // M_b) * U
flops_estimate_kron / flops_estimate_matmul
# %%
import torch.utils.benchmark
from opt_einsum import contract

def timing(name, fn, *args):
    # fn = torch.compile(fn)
    fn(*args)
    timer = torch.utils.benchmark.Timer(
        stmt="fn(*args)",
        globals={"fn": fn, "args": args},
        setup="fn(*args)",
    ).blocked_autorange()
    print(name, timer.mean)


# @torch.compile(mode="max-autotune")
# def naive(vals, matrix_inner, matrix_outer):
#     return torch.einsum('bn,mn->bm', vals, torch.kron(matrix_outer, matrix_inner))

# outside_kron = torch.kron(matrix_outer, matrix_inner)
# @torch.compile(mode="max-autotune")
# def cached(vals, matrix_inner, matrix_outer):
#     return torch.einsum('bn,mn->bm', vals, outside_kron)

# @torch.compile(mode="max-autotune")
# def sequential(vals, matrix_inner, matrix_outer):
#     vals_inner = vals.reshape(B, N_n, N_b)
#     vals_inner = vals_inner @ matrix_inner.T
#     vals_inner = vals_inner.reshape(B, N_n, M_b)
#     vals_outer = torch.einsum('bnd,mn->bmd', vals_inner, matrix_outer)
#     return vals_outer.reshape(B, M)

# @torch.compile(mode="max-autotune")
# def simulate_lora(vals, matrix_inner, matrix_outer):
#     vals_inner = vals.reshape(B, N_n, N_b)
#     vals_inner = vals_inner @ matrix_inner.T
#     vals_outer = vals_inner.reshape(B * M_b, N_n)
#     vals_outer = vals_outer @ matrix_outer.T
#     return vals_outer

# @torch.compile(mode="max-autotune")
# def real_lora(vals, matrix_inner, matrix_outer):
#     vals_inner = vals.reshape(B, N_n, N_b)
#     vals_inner = vals_inner @ matrix_inner.T
#     vals_outer = vals_inner.reshape(B * M_b, N_n)
#     vals_outer = torch.einsum('bnd,mn->bdm', vals_inner, matrix_outer)
#     vals_outer = vals_outer.reshape(B, M)
#     return vals_outer

@torch.compile(mode="max-autotune")
def unified(vals, matrix_inner, matrix_outer):
    vals_inner = vals.reshape(B, N_n, N_b)
    a = torch.einsum("bnd,cud->bnuc", vals_inner, matrix_inner)
    b = torch.einsum("bnuc,mun->bcm", a, matrix_outer)
    return b.reshape(B, M)
    # vals_inner = torch.einsum("bnd,umd->bum", vals_inner, matrix_inner)
    # vals_outer = torch.einsum('bun,mun->bdm', vals_inner, matrix_outer)
    # return torch.einsum(
    #     "bnx,yx,mn->bym",
    #     vals_inner, matrix_inner.view(M_b, N_b), matrix_outer.view(M_n, N_n),
    # ).reshape(-1, M)
    # vals_outer = vals_outer.reshape(B, M)
    # return vals_outer
    # return vals_inner

def gigamatmul(vals, matrix_inner, matrix_outer):
    a = vals
    b = torch.einsum("yux,mun->umnyx", matrix_inner, matrix_outer)
    b = b.reshape(U, M, N)
    return torch.einsum("bx,uyx->buy", a, b)

# PB: 13ms

# timing("naive", naive, vals, matrix_inner, matrix_outer)
# timing("cached", cached, vals, matrix_inner, matrix_outer)
# # timing("sequential", sequential, vals, matrix_inner, matrix_outer)
# # timing("simulate_lora", simulate_lora, vals, matrix_inner, matrix_outer)
# timing("real_lora", real_lora, vals, matrix_inner, matrix_outer)
timing("unified", unified, vals, matrix_inner, matrix_outer)
timing("gigamatmul", gigamatmul, vals, matrix_inner, matrix_outer)
# %%
# real_lora(vals, matrix_inner, matrix_outer).dim_order()
torch.compiler.cudagraph_mark_step_begin()
real_lora(vals, matrix_inner, matrix_outer) - cached(vals, matrix_inner, matrix_outer)