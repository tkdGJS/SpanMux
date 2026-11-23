#!/usr/bin/env python3
"""CPU-only checks for capacity-floor release and context-ring liveness."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VIRTUAL_CONTEXT_PATH = (
    ROOT / "vendor_v6/vllm/v1/core/virtual_context.py"
)
SCHEDULER_PATH = ROOT / "vendor_v6/vllm/v1/core/sched/scheduler.py"

SPEC = importlib.util.spec_from_file_location(
    "vendor_v6_virtual_context_capacity_floor_test",
    VIRTUAL_CONTEXT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

needs_snapshot = (
    MODULE.virtual_context_prefill_needs_capacity_floor_snapshot
)
native_reload_min_free = (
    MODULE.virtual_context_pressure_native_reload_min_free_blocks
)
active_decode_progress_reserve_blocks = (
    MODULE.virtual_context_active_decode_ring_progress_reserve_blocks
)
select_active_decode_progress_tokens = (
    MODULE.virtual_context_select_active_decode_ring_progress_tokens
)
victim_context_tokens = (
    MODULE.virtual_context_residency_victim_context_tokens
)
victim_sort_key = MODULE.virtual_context_residency_victim_sort_key


class CapacityFloorLivenessTest(unittest.TestCase):
    def test_snapshot_starts_one_progress_window_before_floor(self) -> None:
        self.assertFalse(needs_snapshot(4976, 2048, 1024))
        self.assertTrue(needs_snapshot(2928, 2048, 1024))
        self.assertTrue(needs_snapshot(880, 880, 1024))

    def test_pressure_runway_allows_repeated_bounded_snapshots(self) -> None:
        self.assertTrue(needs_snapshot(32768, 2048, 32768))
        self.assertFalse(needs_snapshot(65536, 2048, 32768))

    def test_snapshot_requires_actual_schedulable_work(self) -> None:
        self.assertFalse(needs_snapshot(0, 2048, 1024))
        self.assertFalse(needs_snapshot(2928, 0, 1024))

    def test_pressure_reload_preserves_reserve_and_next_chunk(self) -> None:
        # 4,000 historical blocks + 576 reserved blocks + 1,024 tokens /
        # 16 tokens per block = 4,640 blocks.
        self.assertEqual(
            native_reload_min_free(4000, 576, 1024, 16),
            4640,
        )
        self.assertLess(4576, native_reload_min_free(4000, 576, 1024, 16))

    def test_pressure_reload_rejects_invalid_block_size(self) -> None:
        with self.assertRaises(ValueError):
            native_reload_min_free(1, 1, 1, 0)

    def test_active_decode_reserves_one_2k_prefill_quantum(self) -> None:
        # One 2,048-token prefill quantum uses 128 blocks. Preserve another
        # 64 blocks for decode growth; no HBM staging blocks are configured.
        self.assertEqual(
            active_decode_progress_reserve_blocks(2048, 16, 0, 64),
            192,
        )

    def test_active_decode_progress_uses_soft_safety_as_liveness_fallback(
        self,
    ) -> None:
        # The normal policy sees 378 - (512 safety + 64 growth) == 0 safe
        # blocks. The protected floor may still admit 2,048 tokens because
        # physical capacity covers 128 progress + 64 decode-growth blocks.
        self.assertEqual(
            select_active_decode_progress_tokens(
                378,
                2048,
                1024,
                16,
                0,
                64,
            ),
            2048,
        )

    def test_active_decode_progress_never_spends_decode_growth(self) -> None:
        self.assertEqual(
            select_active_decode_progress_tokens(
                191,
                2048,
                2048,
                16,
                0,
                64,
            ),
            0,
        )

    def test_active_decode_progress_rejects_invalid_block_size(self) -> None:
        with self.assertRaises(ValueError):
            active_decode_progress_reserve_blocks(2048, 0, 0, 64)
        with self.assertRaises(ValueError):
            select_active_decode_progress_tokens(
                378,
                2048,
                1024,
                0,
                0,
                64,
            )

    def test_victim_ranking_uses_longest_durable_logical_context(self) -> None:
        candidates = [
            ("short", victim_context_tokens(25_000, 0), 1.0),
            ("long", victim_context_tokens(75_000, 0), 3.0),
            ("deferred", victim_context_tokens(0, 50_000), 2.0),
        ]
        candidates.sort(
            key=lambda item: victim_sort_key(item[1], item[2], item[0])
        )
        self.assertEqual(
            [request_id for request_id, _, _ in candidates],
            ["long", "deferred", "short"],
        )

    def test_victim_ranking_keeps_fcfs_tie_break(self) -> None:
        candidates = [("new", 50_000, 2.0), ("old", 50_000, 1.0)]
        candidates.sort(
            key=lambda item: victim_sort_key(item[1], item[2], item[0])
        )
        self.assertEqual(candidates[0][0], "old")


class CapacityFloorSchedulerContractTest(unittest.TestCase):
    def test_scheduler_retries_release_and_tags_snapshot(self) -> None:
        source = SCHEDULER_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "virtual_context_capacity_floor_release_attempt_count += 1",
            source,
        )
        self.assertLess(
            source.index("capacity_floor_candidates = ["),
            source.index(
                "ttft_prefill_policy_candidate_order = (\n"
                "            self._ttft_begin_prefill_chunk_iteration()"
            ),
        )
        self.assertIn(
            "virtual_context_residency_victim_sort_key(",
            source,
        )
        self.assertIn(
            '"virtual_context_capacity_floor_victim_selected_request_id"',
            source,
        )
        self.assertIn(
            "self._try_defer_running_prefill_for_virtual_context(\n"
            "                        request\n"
            "                    )",
            source,
        )
        self.assertEqual(
            source.count("self._prefill_needs_capacity_floor_snapshot("),
            2,
        )
        self.assertIn(
            '"virtual_context_capacity_floor_release_success_count"',
            source,
        )

    def test_pressure_resume_uses_progress_preserving_reload_threshold(
        self,
    ) -> None:
        source = SCHEDULER_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "virtual_context_pressure_native_reload_min_free_blocks(",
            source,
        )
        self.assertIn(
            'self.virtual_context_prefill_ring_required_mode == "pressure"',
            source,
        )

    def test_active_decode_progress_reserve_wires_admission_and_handoff(
        self,
    ) -> None:
        source = SCHEDULER_PATH.read_text(encoding="utf-8")
        for required in (
            "virtual_context_select_active_decode_ring_progress_tokens(",
            "scheduled_with_ring_progress_reserve",
            "insufficient_hbm_blocks_with_ring_progress_reserve",
            '"context_ring_bw_progress_reserve_applied"',
            '"pd_no_mix_ring_progress_lane_allowed"',
        ):
            self.assertIn(required, source)

    def test_decode_handoff_reclaims_multiple_prefill_victims(self) -> None:
        source = SCHEDULER_PATH.read_text(encoding="utf-8")
        for required in (
            "def _reclaim_prefill_victims_until_hbm_target(",
            "while True:",
            "target_free_blocks=free_blocks + required_blocks",
            "target_free_blocks=decode_handoff_min_free_blocks",
            "excluded_request_ids={request.request_id}",
            "virtual_context_capacity_floor_victim_selected_request_ids",
            "virtual_context_decode_handoff_reclaim_retry_success_count",
            '"virtual_context_decode_handoff_reclaim_victim_ids"',
        ):
            self.assertIn(required, source)

    def test_cold_decode_prefill_inherits_progress_quantum(self) -> None:
        source = SCHEDULER_PATH.read_text(encoding="utf-8")
        self.assertIn(
            '"VLLM_VIRTUAL_CONTEXT_PREFILL_RING_TOKENS_WITH_DECODE_COLD",\n'
            "                self.pd_ring_min_prefill_tokens_with_decode,",
            source,
        )
        self.assertIn(
            '"virtual_context_prefill_ring_with_decode_cold_allowed_tokens"',
            source,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
