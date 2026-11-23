#!/usr/bin/env python3
"""CPU-only smoke tests for the native-decode copy completion barrier."""

from __future__ import annotations

import ast
import time
from concurrent.futures import Future
from pathlib import Path
from threading import Lock


ROOT = Path(__file__).resolve().parents[2]
VENDORS = ("vendor_v4", "vendor_v5", "vendor_v6")
PROVIDER_REL = Path(
    "vllm/v1/attention/ops/tiered_residency_provider.py"
)
TRITON_REL = Path("vllm/v1/attention/backends/triton_attn.py")


class _BlockingFuture:
    def __init__(self) -> None:
        self.result_called = False

    def done(self) -> bool:
        return False

    def result(self) -> None:
        self.result_called = True


def _load_barrier_functions(provider_path: Path) -> dict[str, object]:
    tree = ast.parse(provider_path.read_text())
    selected_names = {
        "wait_for_pending_hbm_d2h_request_reads",
        "record_decode_reload_completion_barrier",
    }
    selected_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in selected_names
    ]
    assert {node.name for node in selected_functions} == selected_names

    namespace: dict[str, object] = {
        "time": time,
        "_PENDING_HBM_D2H_FENCES_LOCK": Lock(),
        "_PENDING_HBM_D2H_FENCES": {},
        "_DECODE_RELOAD_BARRIER_CHECK_COUNT": 0,
        "_DECODE_RELOAD_BARRIER_WAIT_COUNT": 0,
        "_DECODE_RELOAD_BARRIER_WAIT_PAGE_COUNT": 0,
        "_DECODE_RELOAD_BARRIER_WAIT_US_TOTAL": 0,
        "_DECODE_RELOAD_BARRIER_WAIT_US_LAST": 0,
        "_DECODE_RELOAD_BARRIER_COMPLETE_COUNT": 0,
        "_DECODE_RELOAD_BARRIER_H2D_RETRY_COUNT": 0,
        "_DECODE_RELOAD_BARRIER_H2D_RETRY_PAGE_COUNT": 0,
        "_DECODE_RELOAD_BARRIER_INCOMPLETE_COUNT": 0,
        "_DECODE_RELOAD_BARRIER_INCOMPLETE_PAGE_COUNT": 0,
    }
    module = ast.Module(body=selected_functions, type_ignores=[])
    exec(compile(module, str(provider_path), "exec"), namespace)
    return namespace


def _exercise_provider_barrier(provider_path: Path) -> None:
    namespace = _load_barrier_functions(provider_path)
    request_future = _BlockingFuture()
    unrelated_future: Future[None] = Future()
    unrelated_future.set_result(None)
    namespace["_PENDING_HBM_D2H_FENCES"] = {
        ("req-a", 0, 1): request_future,
        ("req-a", 0, 2): request_future,
        ("req-b", 0, 3): unrelated_future,
    }

    waited_futures = namespace[
        "wait_for_pending_hbm_d2h_request_reads"
    ](["req-a"])

    assert waited_futures == 1
    assert request_future.result_called
    assert namespace["_DECODE_RELOAD_BARRIER_CHECK_COUNT"] == 1
    assert namespace["_DECODE_RELOAD_BARRIER_WAIT_COUNT"] == 1
    assert namespace["_DECODE_RELOAD_BARRIER_WAIT_PAGE_COUNT"] == 2

    record = namespace["record_decode_reload_completion_barrier"]
    record(
        h2d_retry_count=1,
        h2d_retry_page_count=7,
        remaining_page_count=0,
    )
    record(
        h2d_retry_count=2,
        h2d_retry_page_count=5,
        remaining_page_count=3,
    )
    assert namespace["_DECODE_RELOAD_BARRIER_COMPLETE_COUNT"] == 1
    assert namespace["_DECODE_RELOAD_BARRIER_H2D_RETRY_COUNT"] == 3
    assert namespace["_DECODE_RELOAD_BARRIER_H2D_RETRY_PAGE_COUNT"] == 12
    assert namespace["_DECODE_RELOAD_BARRIER_INCOMPLETE_COUNT"] == 1
    assert namespace["_DECODE_RELOAD_BARRIER_INCOMPLETE_PAGE_COUNT"] == 3


def _check_triton_ordering(triton_path: Path) -> None:
    source = triton_path.read_text()
    drain_pos = source.index("wait_for_pending_hbm_d2h_request_reads(")
    consume_pos = source.index("consume_tiered_layer_reload_prefetch(")
    state_pos = source.index(
        "_layer_specific_tiered_page_state(attn_metadata, layer_name)"
    )
    assert drain_pos < consume_pos < state_pos
    assert (
        "max_reload_passes = 3 if decode_reload_barrier_requested else 1"
        in source
    )
    assert "remaining_reload_pages == 0" in source
    assert "record_decode_reload_completion_barrier(" in source


def main() -> None:
    for vendor in VENDORS:
        provider_path = ROOT / vendor / PROVIDER_REL
        triton_path = ROOT / vendor / TRITON_REL
        _exercise_provider_barrier(provider_path)
        _check_triton_ordering(triton_path)
        print(f"[PASS] {vendor}: decode reload completion barrier")


if __name__ == "__main__":
    main()
