# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton online-softmax update for tiered context attention.

The reference tiered executor used to materialize three partial tensors for
every ``Q tile x KV group`` pair and merge them with several PyTorch kernels.
This module keeps the running online-softmax state on the GPU and updates it in
the same kernel that consumes one materialized K/V group.  The outer tier
traversal is still owned by the caller; eliminating that Python traversal is a
separate tensor-plan integration step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm.isolation_memory import isolation_memory_enabled

_ISOLATION_DYNAMIC_KERNEL = isolation_memory_enabled("RING_DYNAMIC_KERNEL")


@dataclass(frozen=True)
class TritonOnlineSoftmaxState:
    max_score: torch.Tensor
    exp_sum: torch.Tensor
    numerator: torch.Tensor


@triton.jit(do_not_specialize=[
    "runtime_key_start", "runtime_query_tokens", "runtime_key_tokens",
])
def _fused_online_softmax_update_kernel(
    Q,
    K,
    V,
    QUERY_POSITIONS,
    STATE_MAX,
    STATE_EXP_SUM,
    STATE_NUM,
    q_stride_q,
    q_stride_h,
    q_stride_d,
    k_stride_k,
    k_stride_h,
    k_stride_d,
    v_stride_k,
    v_stride_h,
    v_stride_d,
    state_max_stride_q,
    state_max_stride_h,
    state_num_stride_q,
    state_num_stride_h,
    state_num_stride_d,
    key_start: tl.constexpr,
    query_tokens: tl.constexpr,
    key_tokens: tl.constexpr,
    kv_group_num: tl.constexpr,
    head_dim: tl.constexpr,
    padded_head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    causal: tl.constexpr,
    native_reduction: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    runtime_key_start=0,
    runtime_query_tokens=0,
    runtime_key_tokens=0,
    ISOLATION_DYNAMIC: tl.constexpr = False,
):
    if ISOLATION_DYNAMIC:
        key_start = runtime_key_start
        query_tokens = runtime_query_tokens
        key_tokens = runtime_key_tokens
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

    state_offset = (
        offs_m * state_max_stride_q + query_head * state_max_stride_h
    )
    m_i = tl.load(
        STATE_MAX + state_offset,
        mask=mask_m,
        other=-float("inf"),
    ).to(tl.float32)
    l_i = tl.load(
        STATE_EXP_SUM + state_offset,
        mask=mask_m,
        other=0.0,
    ).to(tl.float32)
    acc = tl.load(
        STATE_NUM
        + offs_m[:, None] * state_num_stride_q
        + query_head * state_num_stride_h
        + offs_d[None, :] * state_num_stride_d,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)

    for start_n in range(0, key_tokens, BLOCK_N):
        cur_n = start_n + offs_n
        mask_n = cur_n < key_tokens
        if native_reduction:
            # Match the native Triton prefill operand shape: Q is [M, D]
            # and K is presented to tl.dot as [D, N].  This flag does not
            # remove context-ring group boundaries; it only aligns the
            # reduction form within each group.
            k = tl.load(
                K
                + cur_n[None, :] * k_stride_k
                + kv_head * k_stride_h
                + offs_d[:, None] * k_stride_d,
                mask=mask_n[None, :] & mask_d[:, None],
                other=0.0,
            )
        else:
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

        if native_reduction:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk = tl.dot(q, k, acc=qk)
            qk *= softmax_scale
        else:
            qk = tl.dot(q, tl.trans(k)) * softmax_scale
        legal = mask_m[:, None] & mask_n[None, :]
        if causal:
            key_pos = key_start + cur_n
            legal = legal & (key_pos[None, :] <= q_pos[:, None])
        qk = tl.where(legal, qk, -float("inf"))

        block_max = tl.max(qk, 1)
        merged_max = tl.maximum(m_i, block_max)
        has_mass = merged_max > -float("inf")
        if native_reduction:
            old_scale = tl.where(
                has_mass, tl.math.exp(m_i - merged_max), 0.0
            )
            p = tl.math.exp(qk - merged_max[:, None])
        else:
            old_scale = tl.where(
                has_mass, tl.exp(m_i - merged_max), 0.0
            )
            p = tl.exp(qk - merged_max[:, None])
        p = tl.where(legal & has_mass[:, None], p, 0.0)

        if native_reduction:
            l_ij = tl.sum(p, 1)
            l_i_new = old_scale * l_i + l_ij
            acc = acc * old_scale[:, None]
            p = p.to(v.dtype)
            acc = tl.dot(p, v, acc=acc, input_precision="ieee")
            l_i = l_i_new
        else:
            l_i = l_i * old_scale + tl.sum(p, 1)
            acc = acc * old_scale[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = merged_max

    tl.store(STATE_MAX + state_offset, m_i, mask=mask_m)
    tl.store(STATE_EXP_SUM + state_offset, l_i, mask=mask_m)
    tl.store(
        STATE_NUM
        + offs_m[:, None] * state_num_stride_q
        + query_head * state_num_stride_h
        + offs_d[None, :] * state_num_stride_d,
        acc,
        mask=mask_m[:, None] & mask_d[None, :],
    )


@triton.jit(do_not_specialize=[
    "runtime_key_start", "runtime_query_tokens", "runtime_key_tokens",
])
def _fused_paged_online_softmax_update_kernel(
    Q,
    KEY_CACHE,
    VALUE_CACHE,
    BLOCK_IDS,
    QUERY_POSITIONS,
    STATE_MAX,
    STATE_EXP_SUM,
    STATE_NUM,
    q_stride_q,
    q_stride_h,
    q_stride_d,
    k_stride_b,
    k_stride_t,
    k_stride_h,
    k_stride_d,
    v_stride_b,
    v_stride_t,
    v_stride_h,
    v_stride_d,
    state_max_stride_q,
    state_max_stride_h,
    state_num_stride_q,
    state_num_stride_h,
    state_num_stride_d,
    key_start: tl.constexpr,
    query_tokens: tl.constexpr,
    key_tokens: tl.constexpr,
    page_size: tl.constexpr,
    kv_group_num: tl.constexpr,
    head_dim: tl.constexpr,
    padded_head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    causal: tl.constexpr,
    native_reduction: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    runtime_key_start=0,
    runtime_query_tokens=0,
    runtime_key_tokens=0,
    ISOLATION_DYNAMIC: tl.constexpr = False,
):
    if ISOLATION_DYNAMIC:
        key_start = runtime_key_start
        query_tokens = runtime_query_tokens
        key_tokens = runtime_key_tokens
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
    state_offset = (
        offs_m * state_max_stride_q + query_head * state_max_stride_h
    )
    m_i = tl.load(
        STATE_MAX + state_offset,
        mask=mask_m,
        other=-float("inf"),
    ).to(tl.float32)
    l_i = tl.load(
        STATE_EXP_SUM + state_offset,
        mask=mask_m,
        other=0.0,
    ).to(tl.float32)
    acc = tl.load(
        STATE_NUM
        + offs_m[:, None] * state_num_stride_q
        + query_head * state_num_stride_h
        + offs_d[None, :] * state_num_stride_d,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)

    for start_n in range(0, key_tokens, BLOCK_N):
        cur_n = start_n + offs_n
        mask_n = cur_n < key_tokens
        logical_page = cur_n // page_size
        page_offset = cur_n % page_size
        physical_block = tl.load(
            BLOCK_IDS + logical_page,
            mask=mask_n,
            other=0,
        )
        if native_reduction:
            k = tl.load(
                KEY_CACHE
                + physical_block[None, :] * k_stride_b
                + page_offset[None, :] * k_stride_t
                + kv_head * k_stride_h
                + offs_d[:, None] * k_stride_d,
                mask=mask_n[None, :] & mask_d[:, None],
                other=0.0,
            )
        else:
            k = tl.load(
                KEY_CACHE
                + physical_block[:, None] * k_stride_b
                + page_offset[:, None] * k_stride_t
                + kv_head * k_stride_h
                + offs_d[None, :] * k_stride_d,
                mask=mask_n[:, None] & mask_d[None, :],
                other=0.0,
            )
        v = tl.load(
            VALUE_CACHE
            + physical_block[:, None] * v_stride_b
            + page_offset[:, None] * v_stride_t
            + kv_head * v_stride_h
            + offs_d[None, :] * v_stride_d,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        )

        if native_reduction:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk = tl.dot(q, k, acc=qk)
            qk *= softmax_scale
        else:
            qk = tl.dot(q, tl.trans(k)) * softmax_scale
        legal = mask_m[:, None] & mask_n[None, :]
        if causal:
            key_pos = key_start + cur_n
            legal = legal & (key_pos[None, :] <= q_pos[:, None])
        qk = tl.where(legal, qk, -float("inf"))

        block_max = tl.max(qk, 1)
        merged_max = tl.maximum(m_i, block_max)
        has_mass = merged_max > -float("inf")
        if native_reduction:
            old_scale = tl.where(
                has_mass, tl.math.exp(m_i - merged_max), 0.0
            )
            p = tl.math.exp(qk - merged_max[:, None])
        else:
            old_scale = tl.where(
                has_mass, tl.exp(m_i - merged_max), 0.0
            )
            p = tl.exp(qk - merged_max[:, None])
        p = tl.where(legal & has_mass[:, None], p, 0.0)
        if native_reduction:
            l_ij = tl.sum(p, 1)
            l_i_new = old_scale * l_i + l_ij
            acc = acc * old_scale[:, None]
            p = p.to(v.dtype)
            acc = tl.dot(p, v, acc=acc, input_precision="ieee")
            l_i = l_i_new
        else:
            l_i = l_i * old_scale + tl.sum(p, 1)
            acc = acc * old_scale[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = merged_max

    tl.store(STATE_MAX + state_offset, m_i, mask=mask_m)
    tl.store(STATE_EXP_SUM + state_offset, l_i, mask=mask_m)
    tl.store(
        STATE_NUM
        + offs_m[:, None] * state_num_stride_q
        + query_head * state_num_stride_h
        + offs_d[None, :] * state_num_stride_d,
        acc,
        mask=mask_m[:, None] & mask_d[None, :],
    )


@triton.jit(do_not_specialize=["runtime_query_tokens"])
def _finalize_online_softmax_kernel(
    STATE_MAX,
    STATE_EXP_SUM,
    STATE_NUM,
    OUTPUT,
    LSE,
    state_max_stride_q,
    state_max_stride_h,
    state_num_stride_q,
    state_num_stride_h,
    state_num_stride_d,
    output_stride_q,
    output_stride_h,
    output_stride_d,
    lse_stride_h,
    lse_stride_q,
    query_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    padded_head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    STORE_LSE: tl.constexpr,
    runtime_query_tokens=0,
    ISOLATION_DYNAMIC: tl.constexpr = False,
):
    if ISOLATION_DYNAMIC:
        query_tokens = runtime_query_tokens
    start_m = tl.program_id(0)
    query_head = tl.program_id(1)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, padded_head_dim)
    mask_m = offs_m < query_tokens
    mask_d = offs_d < head_dim

    state_offset = (
        offs_m * state_max_stride_q + query_head * state_max_stride_h
    )
    max_score = tl.load(
        STATE_MAX + state_offset,
        mask=mask_m,
        other=-float("inf"),
    ).to(tl.float32)
    exp_sum = tl.load(
        STATE_EXP_SUM + state_offset,
        mask=mask_m,
        other=0.0,
    ).to(tl.float32)
    numerator = tl.load(
        STATE_NUM
        + offs_m[:, None] * state_num_stride_q
        + query_head * state_num_stride_h
        + offs_d[None, :] * state_num_stride_d,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)

    has_mass = exp_sum > 0.0
    denominator = tl.where(has_mass, exp_sum, 1.0)
    output = tl.where(
        has_mass[:, None], numerator / denominator[:, None], 0.0
    )
    tl.store(
        OUTPUT
        + offs_m[:, None] * output_stride_q
        + query_head * output_stride_h
        + offs_d[None, :] * output_stride_d,
        output,
        mask=mask_m[:, None] & mask_d[None, :],
    )
    if STORE_LSE:
        lse = tl.where(
            has_mass,
            tl.log(denominator) + max_score,
            -float("inf"),
        )
        tl.store(
            LSE + query_head * lse_stride_h + offs_m * lse_stride_q,
            lse,
            mask=mask_m,
        )


def _validate_query_kv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_positions: torch.Tensor,
) -> None:
    if query.device.type != "cuda":
        raise ValueError("fused tiered attention requires CUDA tensors")
    if key.device != query.device or value.device != query.device:
        raise ValueError("materialized K/V must be on the query CUDA device")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("query/key/value must have shape [tokens, heads, dim]")
    if tuple(key.shape) != tuple(value.shape):
        raise ValueError("key and value shapes must match")
    if int(query.shape[-1]) != int(key.shape[-1]):
        raise ValueError("query and K/V head dimensions must match")
    if int(key.shape[1]) <= 0 or int(query.shape[1]) % int(key.shape[1]):
        raise ValueError("query heads must be divisible by KV heads")
    if not 1 <= int(query.shape[-1]) <= 256:
        raise ValueError("head_dim must be in the supported range 1..256")
    if int(query.shape[0]) <= 0 or int(key.shape[0]) <= 0:
        raise ValueError("query and key token counts must be positive")
    if query_positions.numel() != int(query.shape[0]):
        raise ValueError("query_positions length must match query tokens")


def create_triton_online_softmax_state(
    query: torch.Tensor,
    reusable_state: TritonOnlineSoftmaxState | None = None,
) -> TritonOnlineSoftmaxState:
    if query.device.type != "cuda" or query.ndim != 3:
        raise ValueError("query must be a rank-3 CUDA tensor")
    query_tokens, query_heads, head_dim = map(int, query.shape)
    if reusable_state is not None:
        _validate_state(query, reusable_state)
        reusable_state.max_score.fill_(-float("inf"))
        reusable_state.exp_sum.zero_()
        reusable_state.numerator.zero_()
        return reusable_state
    return TritonOnlineSoftmaxState(
        max_score=torch.full(
            (query_tokens, query_heads),
            -float("inf"),
            dtype=torch.float32,
            device=query.device,
        ),
        exp_sum=torch.zeros(
            (query_tokens, query_heads),
            dtype=torch.float32,
            device=query.device,
        ),
        numerator=torch.zeros(
            (query_tokens, query_heads, head_dim),
            dtype=torch.float32,
            device=query.device,
        ),
    )


def _validate_state(
    query: torch.Tensor,
    state: TritonOnlineSoftmaxState,
) -> None:
    query_tokens, query_heads, head_dim = map(int, query.shape)
    if tuple(state.max_score.shape) != (query_tokens, query_heads):
        raise ValueError("state max_score shape does not match query")
    if tuple(state.exp_sum.shape) != (query_tokens, query_heads):
        raise ValueError("state exp_sum shape does not match query")
    if tuple(state.numerator.shape) != (
        query_tokens,
        query_heads,
        head_dim,
    ):
        raise ValueError("state numerator shape does not match query")
    for tensor in (state.max_score, state.exp_sum, state.numerator):
        if tensor.device != query.device or tensor.dtype != torch.float32:
            raise ValueError("online-softmax state must be FP32 on query device")


def update_triton_online_softmax_state(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: TritonOnlineSoftmaxState,
    *,
    query_positions: torch.Tensor,
    key_start: int,
    causal: bool,
    softmax_scale: float,
    block_m: int = 16,
    block_n: int = 64,
    native_reduction: bool = False,
) -> None:
    """Consume one contiguous K/V group and update ``state`` in place."""

    _validate_query_kv(query, key, value, query_positions)
    _validate_state(query, state)
    if block_m <= 0 or block_n <= 0:
        raise ValueError("block_m and block_n must be positive")

    query = query if query.is_contiguous() else query.contiguous()
    key = key if key.is_contiguous() else key.contiguous()
    value = value if value.is_contiguous() else value.contiguous()
    if query_positions.device != query.device:
        query_positions = query_positions.to(device=query.device)
    if not query_positions.is_contiguous():
        query_positions = query_positions.contiguous()

    query_tokens = int(query.shape[0])
    key_tokens = int(key.shape[0])
    query_heads = int(query.shape[1])
    kv_heads = int(key.shape[1])
    head_dim = int(query.shape[2])
    padded_head_dim = max(16, triton.next_power_of_2(head_dim))
    kernel_block_m = max(
        16, triton.next_power_of_2(min(block_m, query_tokens))
    )
    kernel_block_n = max(
        16, triton.next_power_of_2(min(block_n, key_tokens))
    )

    grid = (triton.cdiv(query_tokens, kernel_block_m), query_heads)
    _fused_online_softmax_update_kernel[grid](
        query,
        key,
        value,
        query_positions,
        state.max_score,
        state.exp_sum,
        state.numerator,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        state.max_score.stride(0),
        state.max_score.stride(1),
        state.numerator.stride(0),
        state.numerator.stride(1),
        state.numerator.stride(2),
        0 if _ISOLATION_DYNAMIC_KERNEL else int(key_start),
        0 if _ISOLATION_DYNAMIC_KERNEL else query_tokens,
        0 if _ISOLATION_DYNAMIC_KERNEL else key_tokens,
        query_heads // kv_heads,
        head_dim,
        padded_head_dim,
        float(softmax_scale),
        bool(causal),
        bool(native_reduction),
        kernel_block_m,
        kernel_block_n,
        runtime_key_start=int(key_start) if _ISOLATION_DYNAMIC_KERNEL else 0,
        runtime_query_tokens=query_tokens if _ISOLATION_DYNAMIC_KERNEL else 0,
        runtime_key_tokens=key_tokens if _ISOLATION_DYNAMIC_KERNEL else 0,
        ISOLATION_DYNAMIC=_ISOLATION_DYNAMIC_KERNEL,
        num_warps=4,
    )


def update_triton_online_softmax_state_paged(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_ids: torch.Tensor,
    state: TritonOnlineSoftmaxState,
    *,
    query_positions: torch.Tensor,
    key_start: int,
    key_tokens: int,
    causal: bool,
    softmax_scale: float,
    block_m: int = 16,
    block_n: int = 64,
    native_reduction: bool = False,
) -> None:
    """Update online state from a paged HBM KV-cache segment."""

    if query.device.type != "cuda":
        raise ValueError("paged fused tiered attention requires CUDA")
    if key_cache.device != query.device or value_cache.device != query.device:
        raise ValueError("paged HBM K/V cache must share the query device")
    if key_cache.ndim != 4 or tuple(key_cache.shape) != tuple(value_cache.shape):
        raise ValueError("paged K/V cache must have matching [B, P, H, D] shape")
    if block_ids.ndim != 1 or block_ids.device != query.device:
        raise ValueError("block_ids must be a rank-1 tensor on the query device")
    if int(query.shape[-1]) != int(key_cache.shape[-1]):
        raise ValueError("query and paged K/V head dimensions must match")
    if int(query.shape[1]) % int(key_cache.shape[2]):
        raise ValueError("query heads must be divisible by paged KV heads")
    page_size = int(key_cache.shape[1])
    if key_tokens <= 0 or key_tokens > int(block_ids.numel()) * page_size:
        raise ValueError("key_tokens is outside the paged HBM segment")
    _validate_state(query, state)
    if query_positions.numel() != int(query.shape[0]):
        raise ValueError("query_positions length must match query tokens")

    query = query if query.is_contiguous() else query.contiguous()
    block_ids = block_ids if block_ids.is_contiguous() else block_ids.contiguous()
    if query_positions.device != query.device:
        query_positions = query_positions.to(device=query.device)
    if not query_positions.is_contiguous():
        query_positions = query_positions.contiguous()

    query_tokens = int(query.shape[0])
    query_heads = int(query.shape[1])
    kv_heads = int(key_cache.shape[2])
    head_dim = int(query.shape[2])
    padded_head_dim = max(16, triton.next_power_of_2(head_dim))
    kernel_block_m = max(
        16, triton.next_power_of_2(min(block_m, query_tokens))
    )
    kernel_block_n = max(
        16, triton.next_power_of_2(min(block_n, key_tokens))
    )
    grid = (triton.cdiv(query_tokens, kernel_block_m), query_heads)
    _fused_paged_online_softmax_update_kernel[grid](
        query,
        key_cache,
        value_cache,
        block_ids,
        query_positions,
        state.max_score,
        state.exp_sum,
        state.numerator,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        state.max_score.stride(0),
        state.max_score.stride(1),
        state.numerator.stride(0),
        state.numerator.stride(1),
        state.numerator.stride(2),
        0 if _ISOLATION_DYNAMIC_KERNEL else int(key_start),
        0 if _ISOLATION_DYNAMIC_KERNEL else query_tokens,
        0 if _ISOLATION_DYNAMIC_KERNEL else int(key_tokens),
        page_size,
        query_heads // kv_heads,
        head_dim,
        padded_head_dim,
        float(softmax_scale),
        bool(causal),
        bool(native_reduction),
        kernel_block_m,
        kernel_block_n,
        runtime_key_start=int(key_start) if _ISOLATION_DYNAMIC_KERNEL else 0,
        runtime_query_tokens=query_tokens if _ISOLATION_DYNAMIC_KERNEL else 0,
        runtime_key_tokens=int(key_tokens) if _ISOLATION_DYNAMIC_KERNEL else 0,
        ISOLATION_DYNAMIC=_ISOLATION_DYNAMIC_KERNEL,
        num_warps=4,
    )


def finalize_triton_online_softmax_state(
    query: torch.Tensor,
    state: TritonOnlineSoftmaxState,
    *,
    block_m: int = 16,
    output: torch.Tensor | None = None,
    return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Finalize the in-place state to vLLM output and ``[H, Q]`` LSE."""

    _validate_state(query, state)
    query_tokens, query_heads, head_dim = map(int, query.shape)
    if output is None:
        output = torch.empty_like(query)
    elif (
        tuple(output.shape) != tuple(query.shape)
        or output.device != query.device
        or output.dtype != query.dtype
    ):
        raise ValueError("output buffer must match the query shape/device/dtype")
    lse = (
        torch.empty(
            (query_heads, query_tokens),
            dtype=torch.float32,
            device=query.device,
        )
        if return_lse
        else None
    )
    lse_kernel_arg = lse if lse is not None else output
    lse_stride_h = lse.stride(0) if lse is not None else 0
    lse_stride_q = lse.stride(1) if lse is not None else 0
    padded_head_dim = max(16, triton.next_power_of_2(head_dim))
    kernel_block_m = max(
        16, triton.next_power_of_2(min(block_m, query_tokens))
    )
    grid = (triton.cdiv(query_tokens, kernel_block_m), query_heads)
    _finalize_online_softmax_kernel[grid](
        state.max_score,
        state.exp_sum,
        state.numerator,
        output,
        lse_kernel_arg,
        state.max_score.stride(0),
        state.max_score.stride(1),
        state.numerator.stride(0),
        state.numerator.stride(1),
        state.numerator.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse_stride_h,
        lse_stride_q,
        0 if _ISOLATION_DYNAMIC_KERNEL else query_tokens,
        head_dim,
        padded_head_dim,
        kernel_block_m,
        bool(return_lse),
        runtime_query_tokens=query_tokens if _ISOLATION_DYNAMIC_KERNEL else 0,
        ISOLATION_DYNAMIC=_ISOLATION_DYNAMIC_KERNEL,
        num_warps=4,
    )
    return output, lse
