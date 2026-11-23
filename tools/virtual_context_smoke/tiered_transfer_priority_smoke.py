#!/usr/bin/env python3
"""CPU-only ordering smoke for native-reload-first staging admission."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from threading import Event, Lock, Thread


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / (
    "vendor_v4/vllm/v1/attention/ops/tiered_transfer_priority.py"
)
SPEC = importlib.util.spec_from_file_location(
    "tiered_transfer_priority_smoke_module",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> int:
    first_context_entered = Event()
    release_first_context = Event()
    order: list[str] = []
    order_lock = Lock()

    def append(value: str) -> None:
        with order_lock:
            order.append(value)

    def first_context() -> None:
        with MODULE.acquire_context_ring_priority():
            append("context-1")
            first_context_entered.set()
            assert release_first_context.wait(timeout=5)

    def native_reload() -> None:
        assert first_context_entered.wait(timeout=5)
        with MODULE.acquire_native_reload_priority():
            append("native")

    def second_context() -> None:
        with MODULE.acquire_context_ring_priority():
            append("context-2")

    first_thread = Thread(target=first_context)
    native_thread = Thread(target=native_reload)
    second_thread = Thread(target=second_context)
    first_thread.start()
    assert first_context_entered.wait(timeout=5)
    native_thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if MODULE.get_tiered_transfer_priority_stats()[
            "tiered_transfer_priority_waiting_native"
        ] == 1:
            break
        time.sleep(0.001)
    else:
        raise AssertionError("native reload did not enter the wait queue")
    second_thread.start()
    release_first_context.set()
    for thread in (first_thread, native_thread, second_thread):
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert order == ["context-1", "native", "context-2"], order
    stats = MODULE.get_tiered_transfer_priority_stats()
    assert stats["tiered_transfer_priority_native_acquires"] == 1
    assert stats["tiered_transfer_priority_context_acquires"] == 2
    assert stats["tiered_transfer_priority_context_wait_count"] >= 1
    print("[OK] native reload is admitted before a later context-ring user")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
