#!/usr/bin/env python3
"""CUDA correctness smoke for the fused native-reload K/V scatter."""

from __future__ import annotations

import sys

try:
    import torch
except ModuleNotFoundError:
    torch = None


def main() -> int:
    if torch is None:
        print("[SKIP] fused K/V scatter smoke requires PyTorch")
        return 0
    if not torch.cuda.is_available():
        print("[SKIP] fused K/V scatter smoke requires CUDA")
        return 0

    from vllm.v1.attention.ops.tiered_triton_kv_scatter import (
        clear_fused_kv_scatter_state,
        fused_scatter_staged_kv_pages,
        get_fused_kv_scatter_stats,
    )

    clear_fused_kv_scatter_state()
    device = torch.device("cuda")
    num_pages = 3
    block_size = 16
    kv_heads = 2
    head_dim = 8
    destination_blocks = (5, 1, 7)

    staged_key = torch.randn(
        num_pages * block_size,
        kv_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    staged_value = torch.randn_like(staged_key)
    key_cache = torch.zeros(
        8,
        block_size,
        kv_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    value_cache = torch.zeros_like(key_cache)

    for _ in range(2):
        if not fused_scatter_staged_kv_pages(
            staged_key=staged_key,
            staged_value=staged_value,
            key_cache=key_cache,
            value_cache=value_cache,
            destination_block_ids=destination_blocks,
        ):
            raise AssertionError("fused K/V scatter rejected a valid input")
    torch.cuda.synchronize(device)

    expected_key = staged_key.view(
        num_pages,
        block_size,
        kv_heads,
        head_dim,
    )
    expected_value = staged_value.view_as(expected_key)
    for page_index, block_id in enumerate(destination_blocks):
        torch.testing.assert_close(key_cache[block_id], expected_key[page_index])
        torch.testing.assert_close(
            value_cache[block_id], expected_value[page_index]
        )

    stats = get_fused_kv_scatter_stats()
    if stats["dram_provider_reload_fused_scatter_launches"] != 2:
        raise AssertionError(f"unexpected launch stats: {stats}")
    if stats["dram_provider_reload_scatter_descriptor_cache_hits"] < 1:
        raise AssertionError(f"descriptor cache was not reused: {stats}")
    print("[OK] fused K/V scatter matches index-copy semantics")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise
