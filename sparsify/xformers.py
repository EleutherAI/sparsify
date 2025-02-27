# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications by Stepan Shabalin and Nora Belrose
import torch
import triton
from torch import Tensor
from triton import language as tl


@triton.jit
def embedding_bag_k(
    out_ptr,  # [B, dim]
    indices_ptr,  # [B, bag_size]
    weight_ptr,  # [n_keys**2, dim]
    per_sample_weights,  # [B, bag_size]
    dim: tl.constexpr,
    dim_padded: tl.constexpr,
    bag_size: tl.constexpr,
):
    out_idx = tl.program_id(axis=0).to(tl.int64)
    out_value = tl.zeros([dim_padded], dtype=tl.float32)
    dim_mask = tl.arange(0, dim_padded) < dim
    for bag in range(0, bag_size):
        my_index = tl.load(indices_ptr + out_idx * bag_size + bag).to(tl.int64)
        my_scaling = tl.load(per_sample_weights + out_idx * bag_size + bag)
        my_weight = tl.load(
            weight_ptr + tl.arange(0, dim_padded) + my_index * dim, mask=dim_mask
        )
        out_value = out_value + my_weight.to(tl.float32) * my_scaling
    tl.store(
        out_ptr + out_idx * dim + tl.arange(0, dim_padded), out_value, mask=dim_mask
    )


def embedding_bag_triton(
    indices: Tensor, weight: Tensor, per_sample_weights: Tensor
) -> Tensor:
    trt_out = torch.empty(
        [indices.shape[0], weight.shape[1]], dtype=weight.dtype, device=weight.device
    )
    grid = (indices.shape[0],)

    embedding_bag_k[grid](
        trt_out,
        indices,
        weight,
        per_sample_weights,
        dim=weight.shape[-1],
        dim_padded=triton.next_power_of_2(weight.shape[-1]),
        bag_size=indices.shape[1],
        num_warps=1,
        num_stages=1,
    )
    return trt_out


@triton.jit
def count_per_embedding_k(
    count_per_emb_ptr,  # [K+1] (out)
    indices_ptr,  # [B, bag_size]
    bag_size: tl.constexpr,
):
    batch_id = tl.program_id(axis=0).to(tl.int64)
    for i in range(bag_size):
        embedding_id = tl.load(indices_ptr + batch_id * bag_size + i)
        tl.atomic_add(
            count_per_emb_ptr + embedding_id + 1,
            1,
            sem="relaxed",
        )


@triton.jit
def map_embeddings_and_outputs_k(
    reverse_mapping_ptr,  # [B*bag_size] (out)
    mapping_write_pos_ptr,  # [K] (tmp)
    indices_ptr,  # [B, bag_size]
    bag_size: tl.constexpr,
):
    batch_id = tl.program_id(axis=0).to(tl.int64)
    for bag_id in range(bag_size):
        embedding_id = tl.load(indices_ptr + batch_id * bag_size + bag_id)
        write_pos = tl.atomic_add(
            mapping_write_pos_ptr + embedding_id, 1, sem="relaxed"
        )
        tl.store(reverse_mapping_ptr + write_pos, batch_id * bag_size + bag_id)


@triton.jit
def aggregate_gradient_for_embedding_k(
    weight_grad_ptr,  # [K, dim] (out)
    per_sample_weights_grad_ptr,  # [B, bag_size] (out)
    emb_argsorted_ptr,  # [K+1]
    weight_ptr,  # [K, dim] (out)
    emb_begin_pos_ptr,  # [K+1]
    reverse_mapping_ptr,  # [B*bag_size]
    per_sample_weights_ptr,  # [B, bag_size]
    gradient_ptr,  # [B, dim]
    dim: tl.constexpr,
    dim_padded: tl.constexpr,
    bag_size: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    first_embedding_id = tl.program_id(axis=0).to(tl.int64)
    for k in range(0, BLOCK_SIZE):
        embedding_id = first_embedding_id + (K // BLOCK_SIZE) * k
        # embedding_id = first_embedding_id * BLOCK_SIZE + k
        embedding_id = tl.load(emb_argsorted_ptr + embedding_id).to(tl.int64)
        weight_grad = tl.zeros([dim_padded], dtype=tl.float32)
        begin = tl.load(emb_begin_pos_ptr + embedding_id)
        end = tl.load(emb_begin_pos_ptr + embedding_id + 1)
        dim_mask = tl.arange(0, dim_padded) < dim
        weight = tl.load(
            weight_ptr + embedding_id * dim + tl.arange(0, dim_padded),
            mask=dim_mask,
        ).to(tl.float32)
        for idx in range(begin, end):
            output_indice_id = tl.load(reverse_mapping_ptr + idx).to(tl.int64)
            batch_id = output_indice_id // bag_size
            output_indice_id % bag_size
            per_sample_w = tl.load(per_sample_weights_ptr + output_indice_id)
            gradient = tl.load(
                gradient_ptr + batch_id * dim + tl.arange(0, dim_padded), mask=dim_mask
            ).to(tl.float32)
            weight_grad = weight_grad + per_sample_w * gradient
            per_sample_weights_grad = gradient * weight
            per_sample_weights_grad = tl.sum(per_sample_weights_grad)
            tl.store(
                per_sample_weights_grad_ptr + output_indice_id, per_sample_weights_grad
            )
        tl.store(
            weight_grad_ptr + embedding_id * dim + tl.arange(0, dim_padded),
            weight_grad,
            mask=dim_mask,
        )


def embedding_bag_bw_rev_indices(
    indices: Tensor,
    weight: Tensor,
    per_sample_weights: Tensor,
    gradient: Tensor,
) -> tuple[Tensor, Tensor]:
    # Returns: [weight.grad, per_sample_weights.grad]

    K, dim = weight.shape
    B, bag_size = indices.shape
    count_per_emb = torch.zeros((K + 1,), dtype=torch.uint32, device=indices.device)
    count_per_embedding_k[(B,)](count_per_emb, indices, bag_size=bag_size, num_warps=1)
    emb_argsorted = count_per_emb[1:].int().argsort(descending=True)
    emb_begin_pos = count_per_emb.cumsum(0)
    reverse_mapping = torch.empty(
        [B * bag_size], dtype=torch.uint32, device=indices.device
    )
    assert B * bag_size < 2 ** (reverse_mapping.dtype.itemsize * 8 - 1)
    map_embeddings_and_outputs_k[(B,)](
        reverse_mapping_ptr=reverse_mapping,
        mapping_write_pos_ptr=emb_begin_pos.clone(),
        indices_ptr=indices,
        bag_size=bag_size,
        num_warps=1,
    )
    weight_grad = torch.empty_like(weight)
    per_sample_weights_grad = torch.empty_like(per_sample_weights)
    BLOCK_SIZE = 8
    assert (K % BLOCK_SIZE) == 0
    aggregate_gradient_for_embedding_k[(K // BLOCK_SIZE,)](
        weight_grad_ptr=weight_grad,
        emb_begin_pos_ptr=emb_begin_pos,
        emb_argsorted_ptr=emb_argsorted,
        per_sample_weights_grad_ptr=per_sample_weights_grad,
        weight_ptr=weight,
        reverse_mapping_ptr=reverse_mapping,
        per_sample_weights_ptr=per_sample_weights,
        gradient_ptr=gradient,
        dim=dim,
        dim_padded=triton.next_power_of_2(dim),
        bag_size=bag_size,
        B=B,
        K=K,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )
    return weight_grad, per_sample_weights_grad


class xFormersEmbeddingBag(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        indices: Tensor,
        weight: Tensor,
        per_sample_weights: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(indices, weight, per_sample_weights)
        return embedding_bag_triton(indices, weight, per_sample_weights)

    @staticmethod
    def backward(ctx, gradient):
        indices, weight, per_sample_weights = ctx.saved_tensors

        weight_g, per_sample_weights_g = embedding_bag_bw_rev_indices(
            indices,
            weight,
            per_sample_weights,
            gradient,
        )
        return None, weight_g, per_sample_weights_g, None


def xformers_embedding_bag(
    indices: Tensor,
    weight: Tensor,
    per_sample_weights: Tensor,
) -> Tensor:
    return xFormersEmbeddingBag.apply(indices, weight, per_sample_weights)
