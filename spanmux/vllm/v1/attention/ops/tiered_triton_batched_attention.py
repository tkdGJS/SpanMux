# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batched partial attention and device-side online-softmax reduction.

The bounded tiered runtime materializes one or more K/V groups into a padded
GPU tensor. Stage A computes every group's independent online-softmax state in
one launch. Stage B reduces the group dimension and writes the exact attention
output without a host-side merge loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


@dataclass(frozen=True)
class BatchedTieredPartialState:
    max_score: torch.Tensor
    exp_sum: torch.Tensor
    numerator: torch.Tensor


@triton.jit
def _batched_partial_attention_kernel(
    Q,
    K_GROUPS,
    V_GROUPS,
    GROUP_KEY_STARTS,
    GROUP_TOKEN_COUNTS,
    QUERY_POSITIONS,
    PARTIAL_MAX,
    PARTIAL_EXP_SUM,
    PARTIAL_NUM,
    q_stride_q,
    q_stride_h,
    q_stride_d,
    k_stride_g,
    k_stride_k,
    k_stride_h,
    k_stride_d,
    v_stride_g,
    v_stride_k,
    v_stride_h,
    v_stride_d,
    max_stride_g,
    max_stride_q,
    max_stride_h,
    num_stride_g,
    num_stride_q,
    num_stride_h,
    num_stride_d,
    query_tokens: tl.constexpr,
    query_heads: tl.constexpr,
    kv_group_num: tl.constexpr,
    head_dim: tl.constexpr,
    padded_head_dim: tl.constexpr,
    max_group_tokens: tl.constexpr,
    softmax_scale: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    group_id = tl.program_id(0)
    start_m = tl.program_id(1)
    query_head = tl.program_id(2)
    kv_head = query_head // kv_group_num

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, padded_head_dim)
    mask_m = offs_m < query_tokens
    mask_d = offs_d < head_dim
    key_tokens = tl.load(GROUP_TOKEN_COUNTS + group_id)
    key_start = tl.load(GROUP_KEY_STARTS + group_id)

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

    for start_n in range(0, max_group_tokens, BLOCK_N):
        cur_n = start_n + offs_n
        mask_n = cur_n < key_tokens
        k = tl.load(
            K_GROUPS
            + group_id * k_stride_g
            + cur_n[:, None] * k_stride_k
            + kv_head * k_stride_h
            + offs_d[None, :] * k_stride_d,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        v = tl.load(
            V_GROUPS
            + group_id * v_stride_g
            + cur_n[:, None] * v_stride_k
            + kv_head * v_stride_h
            + offs_d[None, :] * v_stride_d,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        legal = mask_m[:, None] & mask_n[None, :]
        if causal:
            legal = legal & (
                key_start + cur_n[None, :] <= q_pos[:, None]
            )
        qk = tl.where(legal, qk, -float("inf"))

        block_max = tl.max(qk, 1)
        merged_max = tl.maximum(m_i, block_max)
        has_mass = merged_max > -float("inf")
        old_scale = tl.where(has_mass, tl.exp(m_i - merged_max), 0.0)
        p = tl.exp(qk - merged_max[:, None])
        p = tl.where(legal & has_mass[:, None], p, 0.0)
        l_i = l_i * old_scale + tl.sum(p, 1)
        acc = acc * old_scale[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = merged_max

    max_offset = (
        group_id * max_stride_g
        + offs_m * max_stride_q
        + query_head * max_stride_h
    )
    tl.store(PARTIAL_MAX + max_offset, m_i, mask=mask_m)
    tl.store(PARTIAL_EXP_SUM + max_offset, l_i, mask=mask_m)
    tl.store(
        PARTIAL_NUM
        + group_id * num_stride_g
        + offs_m[:, None] * num_stride_q
        + query_head * num_stride_h
        + offs_d[None, :] * num_stride_d,
        acc,
        mask=mask_m[:, None] & mask_d[None, :],
    )


@triton.jit
def _ragged_batched_partial_attention_kernel(
    Q,
    K_GROUPS,
    V_GROUPS,
    GROUP_KEY_STARTS,
    GROUP_TOKEN_COUNTS,
    GROUP_QUERY_STARTS,
    GROUP_QUERY_COUNTS,
    QUERY_POSITIONS,
    PARTIAL_MAX,
    PARTIAL_EXP_SUM,
    PARTIAL_NUM,
    q_stride_q,
    q_stride_h,
    q_stride_d,
    k_stride_g,
    k_stride_k,
    k_stride_h,
    k_stride_d,
    v_stride_g,
    v_stride_k,
    v_stride_h,
    v_stride_d,
    max_stride_g,
    max_stride_q,
    max_stride_h,
    num_stride_g,
    num_stride_q,
    num_stride_h,
    num_stride_d,
    query_heads: tl.constexpr,
    kv_group_num: tl.constexpr,
    head_dim: tl.constexpr,
    padded_head_dim: tl.constexpr,
    max_query_tokens: tl.constexpr,
    max_group_tokens: tl.constexpr,
    softmax_scale: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    group_id = tl.program_id(0)
    start_m = tl.program_id(1)
    query_head = tl.program_id(2)
    kv_head = query_head // kv_group_num

    local_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    query_start = tl.load(GROUP_QUERY_STARTS + group_id)
    query_count = tl.load(GROUP_QUERY_COUNTS + group_id)
    global_m = query_start + local_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, padded_head_dim)
    mask_m = local_m < query_count
    mask_d = offs_d < head_dim
    key_tokens = tl.load(GROUP_TOKEN_COUNTS + group_id)
    key_start = tl.load(GROUP_KEY_STARTS + group_id)

    q = tl.load(
        Q
        + global_m[:, None] * q_stride_q
        + query_head * q_stride_h
        + offs_d[None, :] * q_stride_d,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )
    q_pos = tl.load(
        QUERY_POSITIONS + global_m,
        mask=mask_m,
        other=-1,
    )
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, padded_head_dim], dtype=tl.float32)

    for start_n in range(0, max_group_tokens, BLOCK_N):
        cur_n = start_n + offs_n
        mask_n = cur_n < key_tokens
        k = tl.load(
            K_GROUPS
            + group_id * k_stride_g
            + cur_n[:, None] * k_stride_k
            + kv_head * k_stride_h
            + offs_d[None, :] * k_stride_d,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        v = tl.load(
            V_GROUPS
            + group_id * v_stride_g
            + cur_n[:, None] * v_stride_k
            + kv_head * v_stride_h
            + offs_d[None, :] * v_stride_d,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        legal = mask_m[:, None] & mask_n[None, :]
        if causal:
            legal = legal & (
                key_start + cur_n[None, :] <= q_pos[:, None]
            )
        qk = tl.where(legal, qk, -float("inf"))
        block_max = tl.max(qk, 1)
        merged_max = tl.maximum(m_i, block_max)
        has_mass = merged_max > -float("inf")
        old_scale = tl.where(has_mass, tl.exp(m_i - merged_max), 0.0)
        p = tl.exp(qk - merged_max[:, None])
        p = tl.where(legal & has_mass[:, None], p, 0.0)
        l_i = l_i * old_scale + tl.sum(p, 1)
        acc = acc * old_scale[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = merged_max

    max_offset = (
        group_id * max_stride_g
        + local_m * max_stride_q
        + query_head * max_stride_h
    )
    tl.store(PARTIAL_MAX + max_offset, m_i, mask=mask_m)
    tl.store(PARTIAL_EXP_SUM + max_offset, l_i, mask=mask_m)
    tl.store(
        PARTIAL_NUM
        + group_id * num_stride_g
        + local_m[:, None] * num_stride_q
        + query_head * num_stride_h
        + offs_d[None, :] * num_stride_d,
        acc,
        mask=mask_m[:, None] & mask_d[None, :],
    )


@triton.jit
def _ragged_hierarchical_online_softmax_merge_kernel(
    PARTIAL_MAX,
    PARTIAL_EXP_SUM,
    PARTIAL_NUM,
    REQUEST_QUERY_STARTS,
    REQUEST_QUERY_COUNTS,
    REQUEST_GROUP_STARTS,
    REQUEST_GROUP_COUNTS,
    OUTPUT,
    max_stride_g,
    max_stride_q,
    max_stride_h,
    num_stride_g,
    num_stride_q,
    num_stride_h,
    num_stride_d,
    output_stride_q,
    output_stride_h,
    output_stride_d,
    head_dim: tl.constexpr,
    padded_head_dim: tl.constexpr,
    max_groups_per_request: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    request_index = tl.program_id(0)
    local_query_index = tl.program_id(1)
    query_head = tl.program_id(2)
    query_start = tl.load(REQUEST_QUERY_STARTS + request_index)
    query_count = tl.load(REQUEST_QUERY_COUNTS + request_index)
    group_start = tl.load(REQUEST_GROUP_STARTS + request_index)
    group_count = tl.load(REQUEST_GROUP_COUNTS + request_index)
    valid_query = local_query_index < query_count
    offs_g = tl.arange(0, BLOCK_GROUPS)
    offs_d = tl.arange(0, padded_head_dim)
    mask_g = (offs_g < group_count) & (offs_g < max_groups_per_request)
    mask_d = offs_d < head_dim
    group_ids = group_start + offs_g
    max_offsets = (
        group_ids * max_stride_g
        + local_query_index * max_stride_q
        + query_head * max_stride_h
    )
    group_max = tl.load(
        PARTIAL_MAX + max_offsets,
        mask=mask_g & valid_query,
        other=-float("inf"),
    ).to(tl.float32)
    merged_max = tl.max(group_max, 0)
    has_mass = merged_max > -float("inf")
    group_scale = tl.where(
        mask_g & valid_query & has_mass,
        tl.exp(group_max - merged_max),
        0.0,
    )
    group_sum = tl.load(
        PARTIAL_EXP_SUM + max_offsets,
        mask=mask_g & valid_query,
        other=0.0,
    ).to(tl.float32)
    merged_sum = tl.sum(group_sum * group_scale, 0)
    partial_num = tl.load(
        PARTIAL_NUM
        + group_ids[:, None] * num_stride_g
        + local_query_index * num_stride_q
        + query_head * num_stride_h
        + offs_d[None, :] * num_stride_d,
        mask=(
            mask_g[:, None]
            & mask_d[None, :]
            & valid_query
        ),
        other=0.0,
    ).to(tl.float32)
    merged_num = tl.sum(partial_num * group_scale[:, None], 0)
    denominator = tl.where(merged_sum > 0.0, merged_sum, 1.0)
    result = tl.where(
        (merged_sum > 0.0) & mask_d & valid_query,
        merged_num / denominator,
        0.0,
    )
    output_query = query_start + local_query_index
    tl.store(
        OUTPUT
        + output_query * output_stride_q
        + query_head * output_stride_h
        + offs_d * output_stride_d,
        result,
        mask=mask_d & valid_query,
    )


@triton.jit
def _hierarchical_online_softmax_merge_kernel(
    PARTIAL_MAX,
    PARTIAL_EXP_SUM,
    PARTIAL_NUM,
    OUTPUT,
    LSE,
    max_stride_g,
    max_stride_q,
    max_stride_h,
    num_stride_g,
    num_stride_q,
    num_stride_h,
    num_stride_d,
    output_stride_q,
    output_stride_h,
    output_stride_d,
    lse_stride_h,
    lse_stride_q,
    group_count: tl.constexpr,
    query_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    padded_head_dim: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    query_index = tl.program_id(0)
    query_head = tl.program_id(1)
    offs_g = tl.arange(0, BLOCK_GROUPS)
    offs_d = tl.arange(0, padded_head_dim)
    mask_g = offs_g < group_count
    mask_d = offs_d < head_dim

    max_offsets = (
        offs_g * max_stride_g
        + query_index * max_stride_q
        + query_head * max_stride_h
    )
    group_max = tl.load(
        PARTIAL_MAX + max_offsets,
        mask=mask_g,
        other=-float("inf"),
    ).to(tl.float32)
    merged_max = tl.max(group_max, 0)
    has_mass = merged_max > -float("inf")
    group_scale = tl.where(
        mask_g & has_mass,
        tl.exp(group_max - merged_max),
        0.0,
    )
    group_sum = tl.load(
        PARTIAL_EXP_SUM + max_offsets,
        mask=mask_g,
        other=0.0,
    ).to(tl.float32)
    merged_sum = tl.sum(group_sum * group_scale, 0)

    partial_num = tl.load(
        PARTIAL_NUM
        + offs_g[:, None] * num_stride_g
        + query_index * num_stride_q
        + query_head * num_stride_h
        + offs_d[None, :] * num_stride_d,
        mask=mask_g[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    merged_num = tl.sum(partial_num * group_scale[:, None], 0)
    denominator = tl.where(merged_sum > 0.0, merged_sum, 1.0)
    result = tl.where(
        (merged_sum > 0.0) & mask_d,
        merged_num / denominator,
        0.0,
    )
    tl.store(
        OUTPUT
        + query_index * output_stride_q
        + query_head * output_stride_h
        + offs_d * output_stride_d,
        result,
        mask=mask_d,
    )
    if STORE_LSE:
        merged_lse = tl.where(
            merged_sum > 0.0,
            tl.log(denominator) + merged_max,
            -float("inf"),
        )
        tl.store(
            LSE + query_head * lse_stride_h + query_index * lse_stride_q,
            merged_lse,
        )


def _validate_batched_inputs(
    query: torch.Tensor,
    key_groups: torch.Tensor,
    value_groups: torch.Tensor,
    group_key_starts: torch.Tensor,
    group_token_counts: torch.Tensor,
    query_positions: torch.Tensor,
) -> None:
    if query.device.type != "cuda":
        raise ValueError("batched tiered attention requires CUDA tensors")
    if key_groups.device != query.device or value_groups.device != query.device:
        raise ValueError("staged K/V groups must share the query CUDA device")
    if query.ndim != 3 or key_groups.ndim != 4 or value_groups.ndim != 4:
        raise ValueError("expected Q[Q,H,D] and K/V[G,K,H,D]")
    if tuple(key_groups.shape) != tuple(value_groups.shape):
        raise ValueError("batched K/V shapes must match")
    if int(query.shape[-1]) != int(key_groups.shape[-1]):
        raise ValueError("query and K/V head dimensions must match")
    if int(query.shape[1]) % int(key_groups.shape[2]):
        raise ValueError("query heads must be divisible by KV heads")
    group_count = int(key_groups.shape[0])
    if group_count <= 0 or group_count > 512:
        raise ValueError("batched group count must be in the range 1..512")
    if group_key_starts.numel() != group_count:
        raise ValueError("group_key_starts length mismatch")
    if group_token_counts.numel() != group_count:
        raise ValueError("group_token_counts length mismatch")
    if query_positions.numel() != int(query.shape[0]):
        raise ValueError("query_positions length mismatch")


def run_batched_tiered_partial_attention(
    query: torch.Tensor,
    key_groups: torch.Tensor,
    value_groups: torch.Tensor,
    *,
    group_key_starts: torch.Tensor,
    group_token_counts: torch.Tensor,
    query_positions: torch.Tensor,
    causal: bool,
    softmax_scale: float,
    block_m: int = 16,
    block_n: int = 64,
) -> BatchedTieredPartialState:
    _validate_batched_inputs(
        query,
        key_groups,
        value_groups,
        group_key_starts,
        group_token_counts,
        query_positions,
    )
    if int(key_groups.shape[0]) > 128:
        raise ValueError("single-request group count exceeds merge guard")
    if int(key_groups.shape[0]) * int(query.shape[-1]) > 16384:
        raise ValueError("group x head_dim exceeds merge-kernel register guard")
    query = query.contiguous()
    key_groups = key_groups.contiguous()
    value_groups = value_groups.contiguous()
    query_positions = query_positions.to(
        device=query.device, dtype=torch.int32
    ).contiguous()
    group_key_starts = group_key_starts.to(
        device=query.device, dtype=torch.int32
    ).contiguous()
    group_token_counts = group_token_counts.to(
        device=query.device, dtype=torch.int32
    ).contiguous()

    group_count, max_group_tokens, kv_heads, head_dim = map(
        int, key_groups.shape
    )
    query_tokens, query_heads, _ = map(int, query.shape)
    partial_max = torch.empty(
        (group_count, query_tokens, query_heads),
        dtype=torch.float32,
        device=query.device,
    )
    partial_exp_sum = torch.empty_like(partial_max)
    partial_num = torch.empty(
        (group_count, query_tokens, query_heads, head_dim),
        dtype=torch.float32,
        device=query.device,
    )
    padded_head_dim = triton.next_power_of_2(head_dim)
    grid = (
        group_count,
        triton.cdiv(query_tokens, block_m),
        query_heads,
    )
    _batched_partial_attention_kernel[grid](
        query,
        key_groups,
        value_groups,
        group_key_starts,
        group_token_counts,
        query_positions,
        partial_max,
        partial_exp_sum,
        partial_num,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_groups.stride(0),
        key_groups.stride(1),
        key_groups.stride(2),
        key_groups.stride(3),
        value_groups.stride(0),
        value_groups.stride(1),
        value_groups.stride(2),
        value_groups.stride(3),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        partial_num.stride(0),
        partial_num.stride(1),
        partial_num.stride(2),
        partial_num.stride(3),
        query_tokens,
        query_heads,
        query_heads // kv_heads,
        head_dim,
        padded_head_dim,
        max_group_tokens,
        float(softmax_scale),
        bool(causal),
        triton.next_power_of_2(min(block_m, query_tokens)),
        triton.next_power_of_2(min(block_n, max_group_tokens)),
        num_warps=4,
    )
    return BatchedTieredPartialState(
        max_score=partial_max,
        exp_sum=partial_exp_sum,
        numerator=partial_num,
    )


def run_ragged_batched_tiered_partial_attention(
    query: torch.Tensor,
    key_groups: torch.Tensor,
    value_groups: torch.Tensor,
    *,
    group_key_starts: torch.Tensor,
    group_token_counts: torch.Tensor,
    group_query_starts: torch.Tensor,
    group_query_counts: torch.Tensor,
    max_query_tokens: int,
    query_positions: torch.Tensor,
    causal: bool,
    softmax_scale: float,
    block_m: int = 16,
    block_n: int = 64,
) -> BatchedTieredPartialState:
    _validate_batched_inputs(
        query,
        key_groups,
        value_groups,
        group_key_starts,
        group_token_counts,
        query_positions,
    )
    group_count, max_group_tokens, kv_heads, head_dim = map(
        int, key_groups.shape
    )
    if max_query_tokens <= 0 or max_query_tokens > int(query.shape[0]):
        raise ValueError("invalid ragged max_query_tokens")
    if group_query_starts.numel() != group_count:
        raise ValueError("group_query_starts length mismatch")
    if group_query_counts.numel() != group_count:
        raise ValueError("group_query_counts length mismatch")

    query = query.contiguous()
    key_groups = key_groups.contiguous()
    value_groups = value_groups.contiguous()
    query_positions = query_positions.to(
        device=query.device, dtype=torch.int32
    ).contiguous()
    group_key_starts = group_key_starts.to(
        device=query.device, dtype=torch.int32
    ).contiguous()
    group_token_counts = group_token_counts.to(
        device=query.device, dtype=torch.int32
    ).contiguous()
    group_query_starts = group_query_starts.to(
        device=query.device, dtype=torch.int32
    ).contiguous()
    group_query_counts = group_query_counts.to(
        device=query.device, dtype=torch.int32
    ).contiguous()

    query_heads = int(query.shape[1])
    partial_max = torch.empty(
        (group_count, max_query_tokens, query_heads),
        dtype=torch.float32,
        device=query.device,
    )
    partial_exp_sum = torch.empty_like(partial_max)
    partial_num = torch.empty(
        (group_count, max_query_tokens, query_heads, head_dim),
        dtype=torch.float32,
        device=query.device,
    )
    padded_head_dim = triton.next_power_of_2(head_dim)
    grid = (
        group_count,
        triton.cdiv(max_query_tokens, block_m),
        query_heads,
    )
    _ragged_batched_partial_attention_kernel[grid](
        query,
        key_groups,
        value_groups,
        group_key_starts,
        group_token_counts,
        group_query_starts,
        group_query_counts,
        query_positions,
        partial_max,
        partial_exp_sum,
        partial_num,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_groups.stride(0),
        key_groups.stride(1),
        key_groups.stride(2),
        key_groups.stride(3),
        value_groups.stride(0),
        value_groups.stride(1),
        value_groups.stride(2),
        value_groups.stride(3),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        partial_num.stride(0),
        partial_num.stride(1),
        partial_num.stride(2),
        partial_num.stride(3),
        query_heads,
        query_heads // kv_heads,
        head_dim,
        padded_head_dim,
        max_query_tokens,
        max_group_tokens,
        float(softmax_scale),
        bool(causal),
        triton.next_power_of_2(min(block_m, max_query_tokens)),
        triton.next_power_of_2(min(block_n, max_group_tokens)),
        num_warps=4,
    )
    return BatchedTieredPartialState(
        max_score=partial_max,
        exp_sum=partial_exp_sum,
        numerator=partial_num,
    )


def merge_ragged_batched_tiered_partial_states(
    state: BatchedTieredPartialState,
    *,
    request_query_starts: torch.Tensor,
    request_query_counts: torch.Tensor,
    request_group_starts: torch.Tensor,
    request_group_counts: torch.Tensor,
    max_query_tokens: int,
    output: torch.Tensor,
) -> None:
    request_count = int(request_query_starts.numel())
    if request_count <= 0:
        raise ValueError("ragged merge requires at least one request")
    if request_query_counts.numel() != request_count:
        raise ValueError("request_query_counts length mismatch")
    if request_group_starts.numel() != request_count:
        raise ValueError("request_group_starts length mismatch")
    if request_group_counts.numel() != request_count:
        raise ValueError("request_group_counts length mismatch")
    group_count = int(state.max_score.shape[0])
    head_dim = int(state.numerator.shape[-1])
    query_heads = int(state.max_score.shape[-1])
    max_groups_per_request = min(128, group_count)
    block_groups = triton.next_power_of_2(max_groups_per_request)
    if block_groups * head_dim > 16384:
        raise ValueError("group x head_dim exceeds ragged merge guard")
    request_query_starts = request_query_starts.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    request_query_counts = request_query_counts.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    request_group_starts = request_group_starts.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    request_group_counts = request_group_counts.to(
        device=output.device, dtype=torch.int32
    ).contiguous()
    padded_head_dim = triton.next_power_of_2(head_dim)
    _ragged_hierarchical_online_softmax_merge_kernel[
        (request_count, max_query_tokens, query_heads)
    ](
        state.max_score,
        state.exp_sum,
        state.numerator,
        request_query_starts,
        request_query_counts,
        request_group_starts,
        request_group_counts,
        output,
        state.max_score.stride(0),
        state.max_score.stride(1),
        state.max_score.stride(2),
        state.numerator.stride(0),
        state.numerator.stride(1),
        state.numerator.stride(2),
        state.numerator.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        head_dim,
        padded_head_dim,
        max_groups_per_request,
        block_groups,
        num_warps=4,
    )


def merge_batched_tiered_partial_states(
    state: BatchedTieredPartialState,
    *,
    output: torch.Tensor,
    lse: torch.Tensor | None = None,
) -> None:
    group_count, query_tokens, query_heads = map(
        int, state.max_score.shape
    )
    head_dim = int(state.numerator.shape[-1])
    if tuple(state.exp_sum.shape) != tuple(state.max_score.shape):
        raise ValueError("partial max/sum shapes must match")
    if tuple(state.numerator.shape[:3]) != tuple(state.max_score.shape):
        raise ValueError("partial numerator prefix shape mismatch")
    if tuple(output.shape) != (query_tokens, query_heads, head_dim):
        raise ValueError("output shape mismatch")
    block_groups = triton.next_power_of_2(group_count)
    if block_groups * head_dim > 16384:
        raise ValueError("group x head_dim exceeds merge-kernel register guard")
    padded_head_dim = triton.next_power_of_2(head_dim)
    lse_kernel_arg = lse if lse is not None else output
    _hierarchical_online_softmax_merge_kernel[(query_tokens, query_heads)](
        state.max_score,
        state.exp_sum,
        state.numerator,
        output,
        lse_kernel_arg,
        state.max_score.stride(0),
        state.max_score.stride(1),
        state.max_score.stride(2),
        state.numerator.stride(0),
        state.numerator.stride(1),
        state.numerator.stride(2),
        state.numerator.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        0 if lse is None else lse.stride(0),
        0 if lse is None else lse.stride(1),
        group_count,
        query_tokens,
        head_dim,
        padded_head_dim,
        block_groups,
        lse is not None,
        num_warps=4,
    )
