#!/usr/bin/env python3
"""CPU-only state-machine smoke for the D2H control dispatcher."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = Path(
    os.environ.get("VLLM_VENDOR_DIR", ROOT / "vendor_v4")
).resolve()
sys.path.insert(0, str(VENDOR_DIR))

os.environ["VLLM_VIRTUAL_CONTEXT_RING_BACKEND_DRAM_PROVIDER"] = "1"
os.environ["VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_FROM_HBM"] = "1"
os.environ["VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_ASYNC"] = "1"
os.environ["VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_ASYNC_WORKERS"] = "1"
os.environ[
    "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_CONTROL_DISPATCHER"
] = "1"
os.environ[
    "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_CONTROL_DISPATCHER_WORKERS"
] = "1"

try:
    import torch
except ModuleNotFoundError:
    print("[SKIP] D2H control dispatcher smoke requires torch")
    raise SystemExit(0)

from vllm.v1.attention.ops.tiered_residency_provider import (
    clear_dram_kv_pages,
    get_dram_kv_page_registry_stats,
    lookup_dram_kv_page,
    register_hbm_kv_cache_pages,
)


def main() -> int:
    request_id = "d2h-control-smoke"
    layer_name = "smoke.layer.0.attn"
    block_tokens = 16
    key_cache = torch.arange(
        block_tokens * 2 * 4,
        dtype=torch.float32,
    ).reshape(1, block_tokens, 2, 4)
    value_cache = key_cache + 1000

    clear_dram_kv_pages()
    copied = register_hbm_kv_cache_pages(
        layer_name=layer_name,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=torch.tensor([[0]], dtype=torch.int64),
        seq_lens=torch.tensor([block_tokens], dtype=torch.int64),
        query_start_loc=torch.tensor(
            [0, block_tokens],
            dtype=torch.int64,
        ),
        request_ids=[request_id],
        page_versions_by_request={request_id: {0: 7}},
    )
    if copied != 1:
        raise AssertionError(f"expected one submitted page, got {copied}")

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        stats = get_dram_kv_page_registry_stats()
        if (
            stats["dram_provider_d2h_control_dispatch_pending_jobs"] == 0
            and stats["dram_provider_register_from_hbm_async_pending_pages"]
            == 0
        ):
            break
        time.sleep(0.01)
    else:
        raise AssertionError("dispatcher/completion futures did not drain")

    lookup = lookup_dram_kv_page(
        request_id=request_id,
        layer_name=layer_name,
        logical_page=0,
        expected_token_count=block_tokens,
        version=7,
    )
    if lookup is None:
        raise AssertionError("copied page was not published")
    torch.testing.assert_close(lookup.page.key, key_cache[0])
    torch.testing.assert_close(lookup.page.value, value_cache[0])

    stats = get_dram_kv_page_registry_stats()
    expected = {
        "dram_provider_d2h_control_dispatch_enabled": 1,
        "dram_provider_d2h_control_dispatch_submit_count": 1,
        "dram_provider_d2h_control_dispatch_complete_count": 1,
        "dram_provider_d2h_control_dispatch_fail_count": 0,
        "dram_provider_register_from_hbm_async_complete_count": 1,
        "dram_provider_register_from_hbm_async_fail_count": 0,
    }
    for field, value in expected.items():
        if stats[field] != value:
            raise AssertionError(
                f"{field}: expected={value} actual={stats[field]}"
            )
    clear_dram_kv_pages()
    print("[OK] D2H control dispatcher CPU state machine validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
