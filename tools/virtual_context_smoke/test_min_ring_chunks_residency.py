#!/usr/bin/env python3
"""CPU-only checks for the Min-Ring-Chunks residency policy."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENDOR = Path(os.environ.get("VLLM_VENDOR_DIR", ROOT / "vendor_v6")).resolve()
POLICY_PATH = (
    VENDOR / "vllm/v1/core/sched/residency_policy.py"
)
SCHEDULER_PATH = VENDOR / "vllm/v1/core/sched/scheduler.py"

SPEC = importlib.util.spec_from_file_location(
    "vendor_v6_min_ring_chunks_residency_test",
    POLICY_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

Candidate = MODULE.ResidencyVictimCandidate
Policy = MODULE.ResidencyVictimPolicy


class MinRingChunksPolicyTest(unittest.TestCase):
    def test_fixed_quantum_predicts_last_partial_chunk(self) -> None:
        self.assertEqual(MODULE.fixed_ring_chunk_count(0, 2048), 0)
        self.assertEqual(MODULE.fixed_ring_chunk_count(2048, 2048), 1)
        self.assertEqual(MODULE.fixed_ring_chunk_count(2049, 2048), 2)

    def test_fixed_quantum_rejects_non_positive_size(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.fixed_ring_chunk_count(1024, 0)

    def test_required_reclaim_preserves_decode_reserve_and_chunk(self) -> None:
        # 576 free blocks are all reserved. A 2,048-token chunk needs 128
        # additional blocks with a 16-token KV block size.
        self.assertEqual(
            MODULE.required_reclaim_blocks(576, 576, 2048, 16),
            128,
        )
        self.assertEqual(
            MODULE.required_reclaim_blocks(800, 576, 2048, 16),
            0,
        )

    def test_policy_alias_and_invalid_value(self) -> None:
        self.assertEqual(Policy.parse("min-ring-chunks"), Policy.MIN_RING_CHUNKS)
        self.assertEqual(Policy.parse("min_ring_chunk"), Policy.MIN_RING_CHUNKS)
        self.assertEqual(
            Policy.parse(""),
            Policy.LONGEST_LOGICAL_CONTEXT_FIRST,
        )
        with self.assertRaises(ValueError):
            Policy.parse("unknown")

    def test_min_ring_chunks_prefers_smallest_covering_invocation_count(
        self,
    ) -> None:
        candidates = [
            Candidate("long", 1.0, 100_000, 90_000, 4000),
            Candidate("medium", 2.0, 50_000, 46_000, 500),
            Candidate("short", 3.0, 25_000, 24_000, 200),
        ]
        ordered = MODULE.order_residency_victims(
            candidates,
            policy=Policy.MIN_RING_CHUNKS,
            required_blocks=300,
            ring_chunk_tokens=2000,
        )
        # "short" needs one ring chunk but cannot cover 300 blocks alone.
        # "medium" covers the deficit in two chunks and beats "long"'s five.
        self.assertEqual(
            [candidate.request_id for candidate in ordered],
            ["medium", "long", "short"],
        )

    def test_min_ring_chunks_uses_overshoot_before_work_tie_break(
        self,
    ) -> None:
        candidates = [
            Candidate("large", 1.0, 80_000, 78_000, 1000),
            Candidate("fitted", 2.0, 40_000, 38_000, 350),
        ]
        ordered = MODULE.order_residency_victims(
            candidates,
            policy=Policy.MIN_RING_CHUNKS,
            required_blocks=300,
            ring_chunk_tokens=2048,
        )
        self.assertEqual(ordered[0].request_id, "fitted")

    def test_insufficient_candidates_still_make_reclaim_progress(self) -> None:
        candidates = [
            Candidate("tiny", 1.0, 10_000, 8_000, 50),
            Candidate("progress", 2.0, 10_000, 8_000, 200),
        ]
        ordered = MODULE.order_residency_victims(
            candidates,
            policy=Policy.MIN_RING_CHUNKS,
            required_blocks=300,
            ring_chunk_tokens=2048,
        )
        self.assertEqual(ordered[0].request_id, "progress")

    def test_longest_policy_preserves_previous_order(self) -> None:
        candidates = [
            Candidate("short", 1.0, 25_000, 20_000, 1000),
            Candidate("long", 2.0, 75_000, 70_000, 2000),
        ]
        ordered = MODULE.order_residency_victims(
            candidates,
            policy=Policy.LONGEST_LOGICAL_CONTEXT_FIRST,
            required_blocks=100,
            ring_chunk_tokens=2048,
        )
        self.assertEqual(
            [candidate.request_id for candidate in ordered],
            ["long", "short"],
        )


class MinRingChunksSchedulerContractTest(unittest.TestCase):
    def test_scheduler_wires_policy_and_metrics(self) -> None:
        source = SCHEDULER_PATH.read_text(encoding="utf-8")
        for required in (
            "VLLM_VIRTUAL_CONTEXT_RESIDENCY_VICTIM_POLICY",
            "VLLM_VIRTUAL_CONTEXT_RESIDENCY_RING_CHUNK_TOKENS",
            "order_residency_victims(",
            "required_reclaim_blocks(",
            '"virtual_context_capacity_floor_victim_candidate_ring_chunks"',
            '"virtual_context_capacity_floor_victim_selected_ring_chunks"',
            '"virtual_context_capacity_floor_victim_selected_ring_work"',
            '"virtual_context_capacity_floor_victim_selected_request_ids"',
            "def _ordered_virtual_context_prefill_reclaim_candidates(",
        ):
            self.assertIn(required, source)

    def test_previous_default_sort_path_remains_present(self) -> None:
        source = SCHEDULER_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "ResidencyVictimPolicy.LONGEST_LOGICAL_CONTEXT_FIRST",
            source,
        )
        self.assertIn(
            "virtual_context_residency_victim_sort_key(",
            source,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
