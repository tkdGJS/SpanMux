#!/usr/bin/env python3
"""CPU-only state-machine smoke for asynchronous D2H source-block leases."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / (
    "vendor_v4/vllm/v1/core/tiered_hbm_source_block_lease.py"
)
SPEC = importlib.util.spec_from_file_location(
    "tiered_hbm_source_block_lease",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
LEASES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LEASES
SPEC.loader.exec_module(LEASES)


def main() -> int:
    os.environ["VLLM_VIRTUAL_CONTEXT_D2H_SOURCE_BLOCK_LEASE"] = "1"
    LEASES.clear_tiered_hbm_source_block_leases_for_test()

    first = LEASES.acquire_tiered_hbm_source_block_lease([5, 2, 5])
    assert first.block_ids == (2, 5)
    assert LEASES.get_tiered_hbm_source_leased_block_ids() == {2, 5}

    second = LEASES.acquire_tiered_hbm_source_block_lease([5, 7])
    stats = LEASES.get_tiered_hbm_source_block_lease_stats()
    assert stats.active_blocks == 3
    assert stats.active_block_refs == 4
    assert stats.peak_active_blocks == 3
    assert stats.peak_active_block_refs == 4

    first.release()
    first.release()
    assert LEASES.get_tiered_hbm_source_leased_block_ids() == {5, 7}
    stats = LEASES.get_tiered_hbm_source_block_lease_stats()
    assert stats.release_count == 1
    assert stats.active_block_refs == 2

    second.release()
    assert not LEASES.get_tiered_hbm_source_leased_block_ids()
    stats = LEASES.get_tiered_hbm_source_block_lease_stats()
    assert stats.acquire_count == 2
    assert stats.release_count == 2
    assert stats.acquired_block_refs == stats.released_block_refs == 4

    try:
        LEASES.acquire_tiered_hbm_source_block_lease([-1])
    except ValueError:
        pass
    else:
        raise AssertionError("negative block id did not fail")

    print("[OK] HBM D2H source-block lease state machine validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
