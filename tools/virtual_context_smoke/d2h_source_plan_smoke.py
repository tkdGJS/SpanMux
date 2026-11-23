#!/usr/bin/env python3
"""CPU-only contract test for the layer-independent D2H source plan."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = (
    ROOT
    / "vendor_v4/vllm/v1/attention/ops/tiered_d2h_source_plan.py"
)
SPEC = importlib.util.spec_from_file_location("tiered_d2h_source_plan", MODULE)
assert SPEC is not None and SPEC.loader is not None
SOURCE_PLAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SOURCE_PLAN
SPEC.loader.exec_module(SOURCE_PLAN)


class FakeBlockTable:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, index: tuple[int, int]) -> int:
        row, column = index
        return self.rows[row][column]


def main() -> int:
    table = FakeBlockTable([[10, 11, 12, -1], [20, 21, 22, -1]])
    scheduled = SOURCE_PLAN.build_tiered_d2h_source_plan(
        request_ids=["req0", "req1"],
        block_table=table,
        seq_lens=[40, 33],
        query_start_loc=[0, 8, 24],
        block_size=16,
        copy_policy="scheduled",
        allowed_request_ids={"req0", "req1"},
        force_all_request_ids=set(),
        max_pages=0,
    )
    assert [
        (page.request_id, page.logical_page, page.block_id, page.token_count)
        for page in scheduled.pages
    ] == [
        ("req0", 2, 12, 8),
        ("req1", 1, 21, 16),
        ("req1", 2, 22, 1),
    ]

    forced = SOURCE_PLAN.build_tiered_d2h_source_plan(
        request_ids=["req0", "req1"],
        block_table=table,
        seq_lens=[40, 33],
        query_start_loc=[0, 8, 24],
        block_size=16,
        copy_policy="scheduled",
        allowed_request_ids={"req0"},
        force_all_request_ids={"req0"},
        max_pages=0,
    )
    assert [page.logical_page for page in forced.pages] == [0, 1, 2]
    assert all(page.force_all for page in forced.pages)
    assert forced.for_request_ids(["req1"]).pages == ()
    assert forced.matches(["req0", "req1"], 16, {"req0"}, {"req0"})
    assert not forced.matches(["req0"], 16, {"req0"}, {"req0"})
    assert not forced.matches(["req0", "req1"], 16, {"req1"}, {"req0"})

    capped = SOURCE_PLAN.build_tiered_d2h_source_plan(
        request_ids=["req0", "req1"],
        block_table=table,
        seq_lens=[40, 33],
        query_start_loc=[0, 8, 24],
        block_size=16,
        copy_policy="all",
        allowed_request_ids=None,
        force_all_request_ids=set(),
        max_pages=2,
    )
    assert len(capped.pages) == 2

    print("[OK] D2H source-page plan is exact, filterable, and CPU-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
