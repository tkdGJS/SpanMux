# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton partial-attention kernel for tiered blockwise context execution.

This module is a narrow ZoneAtten 0.5.x prototype.  It computes the partial
online-softmax state for one query tile and one materialized K/V group:

* partial_max: [Q, H]
* partial_exp_sum: [Q, H]
* partial_num: [Q, H, D]

The caller still performs the outer context-ring traversal, DRAM/HBM staging,
and cross-group online-softmax merge.  Keeping this kernel scoped to a single
partial state lets the serving path fall back to the existing PyTorch reference
implementation whenever a shape or platform is unsupported.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _partial_attention_kernel(
    Q,
    K,
    V,
    QUERY_POSITIONS,
    PARTIAL_MAX,
    PARTIAL_EXP_SUM,
    PARTIAL_NUM,
    q_stride_q,
    q_stride_h,
    q_stride_d,
    k_stride_k,
    k_stride_h,
    k_stride_d,
    v_stride_k,
    v_stride_h,
    v_stride_d,
    out_max_stride_q,
    out_max_stride_h,
    out_num_stride_q,
    out_num_stride_h,
    out_num_stride_d,
    key_start: tl.constexpr,
    query_tokens: tl.constexpr,
    key_tokens: tl.constexpr,
    query_heads: tl.constexpr,
    kv_group_num: tl.constexpr,
    head_dim: tl.constexpr,
    padded_head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    query_head = tl.program_id(1)
    kv_head = query_head // kv_group_num

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, padded_head_dim)

    mask_m = offs_m < query_tokens
    mask_d = offs_d < head_dim

    q = tl.load(
        Q
        + offs_m[:, None] * q_stride_q
        + query_head * q_stride_h
        + offs_d[None, :] * q_stride_d,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )

    q_pos = tl.load(
        QUERY_POSITIONS + offs_m,
        mask=mask_m,
        other=-1,
    )

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, padded_head_dim], dtype=tl.float32)

    for start_n in range(0, key_tokens, BLOCK_N):
        cur_n = start_n + offs_n
        mask_n = cur_n < key_tokens
        k = tl.load(
            K
            + cur_n[:, None] * k_stride_k
            + kv_head * k_stride_h
            + offs_d[None, :] * k_stride_d,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        v = tl.load(
            V
            + cur_n[:, None] * v_stride_k
            + kv_head * v_stride_h
            + offs_d[None, :] * v_stride_d,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        mask = mask_m[:, None] & mask_n[None, :]
        if causal:
            key_pos = key_start + cur_n
            mask = mask & (key_pos[None, :] <= q_pos[:, None])
        qk = tl.where(mask, qk, -float("inf"))

        m_block = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_block)
        valid_new = m_new > -float("inf")
        alpha = tl.where(valid_new, tl.exp(m_i - m_new), 0.0)
        p = tl.exp(qk - m_new[:, None])
        p = tl.where(mask & valid_new[:, None], p, 0.0)

        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    tl.store(
        PARTIAL_MAX + offs_m * out_max_stride_q + query_head * out_max_stride_h,
        m_i,
        mask=mask_m,
    )
    tl.store(
        PARTIAL_EXP_SUM + offs_m * out_max_stride_q
        + query_head * out_max_stride_h,
        l_i,
        mask=mask_m,
    )
    tl.store(
        PARTIAL_NUM
        + offs_m[:, None] * out_num_stride_q
        + query_head * out_num_stride_h
        + offs_d[None, :] * out_num_stride_d,
        acc,
        mask=mask_m[:, None] & mask_d[None, :],
    )


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_positions: torch.Tensor,
) -> None:
    if query.device.type != "cuda":
        raise ValueError("Triton partial attention requires CUDA query tensors")
    if key.device.type != "cuda" or value.device.type != "cuda":
        raise ValueError("Triton partial attention requires CUDA K/V tensors")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query/key/value must have shape [tokens, heads, dim]")
    if tuple(key.shape) != tuple(value.shape):
        raise ValueError("key and value shapes must match")
    if int(query.shape[-1]) != int(key.shape[-1]):
        raise ValueError("query and K/V head dimensions must match")
    if int(query.shape[1]) % int(key.shape[1]) != 0:
        raise ValueError("query heads must be divisible by KV heads")
    if int(query.shape[-1]) <= 0 or int(query.shape[-1]) > 256:
        raise ValueError("head_dim must be in the supported range 1..256")
    if int(query.shape[0]) <= 0 or int(key.shape[0]) <= 0:
        raise ValueError("query and key token counts must be positive")
    if query_positions.numel() != int(query.shape[0]):
        raise ValueError("query_positions length must match query tokens")


def run_triton_partial_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_positions: torch.Tensor,
    key_start: int,
    causal: bool,
    softmax_scale: float,
    block_m: int = 16,
    block_n: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a partial online-softmax state with Triton.

    The output contract intentionally matches
    ``TieredBlockwiseAttentionExecutor._partial_attention``.
    """

    _validate_inputs(query, key, value, query_positions)
    if block_m <= 0 or block_n <= 0:
        raise ValueError("block_m and block_n must be positive")

    if not query.is_contiguous():
        query = query.contiguous()
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    if query_positions.device != query.device:
        query_positions = query_positions.to(device=query.device)
    if not query_positions.is_contiguous():
        query_positions = query_positions.contiguous()

    query_tokens = int(query.shape[0])
    key_tokens = int(key.shape[0])
    query_heads = int(query.shape[1])
    kv_heads = int(key.shape[1])
    head_dim = int(query.shape[2])
    padded_head_dim = triton.next_power_of_2(head_dim)
    kv_group_num = query_heads // kv_heads

    partial_max = torch.empty(
        (query_tokens, query_heads),
        device=query.device,
        dtype=torch.float32,
    )
    partial_exp_sum = torch.empty_like(partial_max)
    partial_num = torch.empty(
        (query_tokens, query_heads, head_dim),
        device=query.device,
        dtype=torch.float32,
    )

    grid = (triton.cdiv(query_tokens, block_m), query_heads)
    _partial_attention_kernel[grid](
        query,
        key,
        value,
        query_positions,
        partial_max,
        partial_exp_sum,
        partial_num,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_num.stride(0),
        partial_num.stride(1),
        partial_num.stride(2),
        key_start,
        query_tokens,
        key_tokens,
        query_heads,
        kv_group_num,
        head_dim,
        padded_head_dim,
        float(softmax_scale),
        bool(causal),
        triton.next_power_of_2(min(block_m, query_tokens)),
        triton.next_power_of_2(min(block_n, key_tokens)),
        num_warps=4,
    )
    return partial_max, partial_exp_sum, partial_num
