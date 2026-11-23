#!/usr/bin/env python3
"""CPU-only checks for the vendor_v8 JIT-Min-Copy residency policy."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "vendor_v8/vllm/v1/core/sched/residency_policy.py"
SCHEDULER_PATH = ROOT / "vendor_v8/vllm/v1/core/sched/scheduler.py"

SPEC = importlib.util.spec_from_file_location(
    "vendor_v8_jit_min_copy_residency_test",
    POLICY_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

Candidate = MODULE.ResidencyVictimCandidate
Policy = MODULE.ResidencyVictimPolicy


def candidate(
    request_id: str,
    *,
    reclaimable: int,
    d2h: int,
    ring_h2d: int,
    reload_h2d: int,
    arrival: float = 1.0,
) -> object:
    return Candidate(
        request_id=request_id,
        arrival_time=arrival,
        prompt_tokens=10_000,
        logical_context_tokens=4_000,
        estimated_reclaimable_blocks=reclaimable,
        estimated_d2h_eviction_tokens=d2h,
        estimated_ring_staging_h2d_tokens=ring_h2d,
        estimated_decode_reload_h2d_tokens=reload_h2d,
    )


class JitMinCopyPolicyTest(unittest.TestCase):
    def test_alias_and_copy_component_estimate(self) -> None:
        self.assertEqual(Policy.parse("jit-min-copy"), Policy.JIT_MIN_COPY)
        self.assertEqual(
            MODULE.estimate_jit_min_copy_components(4096, 8192, 2048),
            (8192, 4096 + 6144, 8192),
        )

    def test_minimizes_total_copy_per_useful_reclaim_block(self) -> None:
        covering = candidate(
            "covering",
            reclaimable=100,
            d2h=400,
            ring_h2d=400,
            reload_h2d=200,
        )
        efficient_partial = candidate(
            "efficient_partial",
            reclaimable=50,
            d2h=100,
            ring_h2d=200,
            reload_h2d=100,
        )
        ordered = MODULE.order_residency_victims(
            [covering, efficient_partial],
            policy=Policy.JIT_MIN_COPY,
            required_blocks=100,
            ring_chunk_tokens=2048,
        )
        self.assertEqual(ordered[0].request_id, "efficient_partial")

    def test_reclaim_credit_is_capped_at_current_deficit(self) -> None:
        oversized = candidate(
            "oversized",
            reclaimable=1000,
            d2h=400,
            ring_h2d=400,
            reload_h2d=200,
        )
        fitted = candidate(
            "fitted",
            reclaimable=100,
            d2h=300,
            ring_h2d=400,
            reload_h2d=200,
        )
        ordered = MODULE.order_residency_victims(
            [oversized, fitted],
            policy=Policy.JIT_MIN_COPY,
            required_blocks=100,
            ring_chunk_tokens=2048,
        )
        self.assertEqual(ordered[0].request_id, "fitted")

    def test_zero_reclaim_candidate_is_last(self) -> None:
        no_progress = candidate(
            "no_progress",
            reclaimable=0,
            d2h=1,
            ring_h2d=1,
            reload_h2d=1,
        )
        useful = candidate(
            "useful",
            reclaimable=1,
            d2h=100,
            ring_h2d=100,
            reload_h2d=100,
        )
        ordered = MODULE.order_residency_victims(
            [no_progress, useful],
            policy=Policy.JIT_MIN_COPY,
            required_blocks=1,
            ring_chunk_tokens=2048,
        )
        self.assertEqual([row.request_id for row in ordered], ["useful", "no_progress"])


class JitMinCopySchedulerContractTest(unittest.TestCase):
    def test_scheduler_keeps_jit_shortfall_and_safety_guards(self) -> None:
        source = SCHEDULER_PATH.read_text(encoding="utf-8")
        for required in (
            "jit_min_copy_active and required_blocks <= 0",
            "request.request_id in self._pd_pending_request_ids",
            "plan_virtual_context_hbm_release_candidates(",
            "plan_virtual_context_hbm_reload_pages(",
            "if any(page not in covered_pages for page in range(required_pages))",
            "_reclaim_prefill_victims_until_hbm_target(",
        ):
            self.assertIn(required, source)

    def test_scheduler_exports_copy_breakdown_metrics(self) -> None:
        source = SCHEDULER_PATH.read_text(encoding="utf-8")
        for metric in (
            "virtual_context_capacity_floor_victim_jit_triggered",
            "virtual_context_capacity_floor_victim_jit_safe_candidate_count",
            "virtual_context_capacity_floor_victim_candidate_copy_tokens",
            "virtual_context_capacity_floor_victim_candidate_copy_scores",
            "virtual_context_capacity_floor_victim_selected_total_copy_tokens",
        ):
            self.assertIn(f'"{metric}"', source)

    def test_reclaim_returns_the_snapshot_used_to_select_each_victim(
        self,
    ) -> None:
        source = SCHEDULER_PATH.read_text(encoding="utf-8")
        for required in (
            "class ReleasedPrefillVictim:",
            "for candidate, candidate_snapshot in candidates:",
            "candidate_snapshot=candidate_snapshot",
            "selected_candidate = released_victim.candidate_snapshot",
        ):
            self.assertIn(required, source)
        self.assertNotIn(
            "selected_candidate = candidate_snapshots[\n"
            "                    victim_request_id\n"
            "                ]",
            source,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
