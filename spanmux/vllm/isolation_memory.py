# SPDX-License-Identifier: Apache-2.0
"""Default-off, independently reversible serving memory optimizations.

Read these switches at owner/model construction, not by mutating them while a
Future or CUDA Graph is live. This module deliberately has no torch dependency.
"""

import os


def isolation_execution_enabled(feature: str) -> bool:
    """New opt-ins deliberately do not inherit the older memory bundle."""
    value = os.getenv(
        "VLLM_ISOLATION_" + feature,
        os.getenv("VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS", "0"),
    )
    if value not in {"0", "1"}:
        raise ValueError(f"VLLM_ISOLATION_{feature} must be 0 or 1, got {value!r}")
    return value == "1"


def isolation_unused_embed_staging(*, supports_mm_inputs: bool,
                                   prompt_embeds: bool, speculative: bool,
                                   encoder_decoder: bool, pooling: bool,
                                   tensor_parallel_size: int,
                                   pipeline_parallel_size: int) -> bool:
    """A configuration-lifetime decision, not a per-request memory policy."""
    return (isolation_execution_enabled("TEXT_INPUT_BUFFERS")
            and not any((supports_mm_inputs, prompt_embeds, speculative,
                         encoder_decoder, pooling))
            and tensor_parallel_size == pipeline_parallel_size == 1)


def isolation_allocator_snapshot(cuda, device, *, iteration: int,
                                  sample_every: int) -> dict[str, int]:
    """Host counters only, process/device scope, not an exclusive phase peak."""
    if iteration < 0 or iteration % max(1, sample_every):
        return {}
    stats = cuda.memory_stats(device)
    return {
        "isolation_gpu_allocated_bytes": int(stats.get("allocated_bytes.all.current", -1)),
        "isolation_gpu_reserved_bytes": int(stats.get("reserved_bytes.all.current", -1)),
        "isolation_gpu_peak_allocated_bytes": int(stats.get("allocated_bytes.all.peak", -1)),
        "isolation_gpu_peak_reserved_bytes": int(stats.get("reserved_bytes.all.peak", -1)),
        "isolation_gpu_active_bytes": int(stats.get("active_bytes.all.current", -1)),
        "isolation_gpu_inactive_split_bytes": int(stats.get("inactive_split_bytes.all.current", -1)),
    }


def isolation_ring_query_bytes(tokens: int, heads: int, head_size: int,
                               element_bytes: int) -> int:
    """Conservative per-executor query scratch, not total GPU/KV memory.

    Includes rounded FP32 softmax state, packed Q, separate output/LSE and
    int64 positions. Historical staging and model workspace are independent.
    """
    if min(tokens, heads, head_size, element_bytes) <= 0:
        raise ValueError("Ring scratch geometry must be positive")
    capacity = 1 << (tokens - 1).bit_length()
    return (capacity * heads * (head_size + 2) * 4
            + tokens * heads * head_size * element_bytes * 2
            + tokens * heads * 4 + tokens * 8)


def isolation_ring_query_token_limit(*, max_tokens: int, heads: int,
                                     head_size: int, element_bytes: int,
                                     budget_bytes: int, block_size: int) -> int:
    """Largest block-aligned query fitting the declared scratch ceiling."""
    if min(max_tokens, heads, head_size, element_bytes, budget_bytes, block_size) <= 0:
        raise ValueError("Ring query limits must be positive")
    low, high = 0, max_tokens // block_size
    while low < high:
        middle = (low + high + 1) // 2
        if isolation_ring_query_bytes(
            middle * block_size, heads, head_size, element_bytes
        ) <= budget_bytes:
            low = middle
        else:
            high = middle - 1
    return low * block_size


def isolation_memory_enabled(feature: str) -> bool:
    value = os.getenv(
        "VLLM_ISOLATION_" + feature,
        os.getenv("VLLM_ISOLATION_MEMORY_OPTIMIZATIONS", "0"),
    )
    if value not in {"0", "1"}:
        raise ValueError(f"VLLM_ISOLATION_{feature} must be 0 or 1, got {value!r}")
    return value == "1"


def isolation_decode_token_capacity(
    *, phase: str, max_tokens: int, max_requests: int,
    uniform_query_len: int, supported: bool,
) -> int:
    """Allocation size only; never an admission or batch-size policy."""
    if (not supported or phase != "decode" or uniform_query_len != 1
            or not isolation_memory_enabled("COMPACT_DECODE_BUFFERS")):
        return max_tokens
    if max_requests <= 0 or max_tokens <= 0:
        raise ValueError("Decode buffer budgets must be positive")
    return min(max_tokens, 1 << (max_requests - 1).bit_length())
