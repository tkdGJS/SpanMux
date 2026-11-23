#!/usr/bin/env python3
"""CPU-only plan tests for the Dynamic-PD-HBM versus vendor_v8 workload."""

from measure_vendor_v8_short_validation import build_validation_plan


def test_completion_gated_default_is_preserved() -> None:
    plan = build_validation_plan(
        long_count=3,
        short_per_long=2,
        long_context_tokens=100_000,
        short_context_tokens=10_000,
        short_offsets_sec=[5.0, 10.0],
    )

    assert len(plan) == 9
    assert [item.role for item in plan[:3]] == ["long", "short", "short"]


def test_concurrent_longs_use_global_short_offsets() -> None:
    offsets = [2.0, 4.0, 6.0, 8.0, 10.0]
    plan = build_validation_plan(
        submission_mode="concurrent_longs",
        long_count=3,
        short_per_long=2,
        long_context_tokens=100_000,
        short_context_tokens=10_000,
        short_offsets_sec=offsets,
    )

    assert len(plan) == 8
    assert [item.role for item in plan] == ["long"] * 3 + ["short"] * 5
    assert [item.request_index for item in plan[:3]] == list(range(3))
    assert [item.offset_from_long_submit_s for item in plan[3:]] == offsets


def test_workload_a_preserves_current_three_long_five_short_pattern() -> None:
    offsets = [60.0, 120.0, 180.0, 240.0, 300.0]
    plan = build_validation_plan(
        workload="A",
        long_count=3,
        short_per_long=2,
        short_count=5,
        long_context_tokens=50_000,
        short_context_tokens=10_000,
        short_offsets_sec=offsets,
    )

    assert [item.role for item in plan] == ["long"] * 3 + ["short"] * 5
    assert [item.offset_from_workload_start_s for item in plan] == (
        [0.0] * 3 + offsets
    )
    assert {item.workload for item in plan} == {"A"}


def test_workload_b_starts_with_short_burst_then_periodic_longs() -> None:
    plan = build_validation_plan(
        workload="B",
        long_count=3,
        short_per_long=2,
        short_count=5,
        long_context_tokens=50_000,
        short_context_tokens=10_000,
        short_offsets_sec=[60.0, 120.0, 180.0, 240.0, 300.0],
        long_offsets_sec=[300.0, 600.0, 900.0],
    )

    assert [item.role for item in plan] == ["short"] * 5 + ["long"] * 3
    assert [item.offset_from_workload_start_s for item in plan] == (
        [0.0] * 5 + [300.0, 600.0, 900.0]
    )
    assert [item.request_index for item in plan[:5]] == list(range(5))


def test_workload_c_interleaves_then_drains_remaining_shorts() -> None:
    plan = build_validation_plan(
        workload="C",
        long_count=3,
        short_per_long=2,
        short_count=5,
        long_context_tokens=50_000,
        short_context_tokens=10_000,
        short_offsets_sec=[60.0, 120.0, 180.0, 240.0, 300.0],
        interleaved_interval_sec=300.0,
    )

    assert [item.role for item in plan] == [
        "short",
        "long",
        "short",
        "long",
        "short",
        "long",
        "short",
        "short",
    ]
    assert [item.offset_from_workload_start_s for item in plan] == [
        0.0,
        300.0,
        600.0,
        900.0,
        1200.0,
        1500.0,
        1800.0,
        2100.0,
    ]
