#!/usr/bin/env python3
"""CPU-only tests for vendor_v8 Prefill chunk scheduling policies."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = Path(os.environ.get("VLLM_VENDOR_DIR", ROOT / "vendor_v8")).resolve()
MODULE_PATH = VENDOR / "vllm/v1/core/sched/prefill_chunk_scheduler.py"
SPEC = importlib.util.spec_from_file_location(
    "vendor_v8_prefill_chunk_scheduler", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PrefillChunkScheduler = MODULE.PrefillChunkScheduler
PrefillAdmissionCandidate = MODULE.PrefillAdmissionCandidate
build_native_prefill_admission_frontier = (
    MODULE.build_native_prefill_admission_frontier
)


@dataclass
class FakeRequest:
    request_id: str
    arrival_time: float
    num_prompt_tokens: int
    num_computed_tokens: int = 0
    virtual_context_deferred_logical_computed_tokens: int = 0
    ttft_prefill_deadline_ms: float | None = None


class PrefillChunkSchedulerTest(unittest.TestCase):
    def test_fcfs_and_srpt_fill_identical_native_budgets(self) -> None:
        requests = [
            FakeRequest("a", 1.0, 3000),
            FakeRequest("b", 2.0, 1000),
            FakeRequest("c", 3.0, 2000),
        ]
        by_id = {r.request_id: r for r in requests}
        for policy, expected in (("fcfs", ("a", "b", "c")), ("srpt", ("b", "c", "a"))):
            with self.subTest(policy=policy):
                scheduler = PrefillChunkScheduler(policy, quantum_tokens=0)
                order = scheduler.begin_iteration(requests)
                frontier = build_native_prefill_admission_frontier(
                    [PrefillAdmissionCandidate(rid, by_id[rid].num_prompt_tokens,
                                               needs_new_sequence=True) for rid in order],
                    token_budget_before=16384, new_sequence_slots=10,
                )
                self.assertEqual(frontier.request_ids, expected)
                self.assertEqual(frontier.token_budget_after, 16384 - 6000)

    def test_fcfs_budget_path_does_not_remove_pending_owner_guard(self) -> None:
        source = (VENDOR / "vllm/v1/core/sched/scheduler.py").read_text()
        begin = source.split("def _ttft_begin_prefill_chunk_iteration", 1)[1].split(
            "def _ttft_decode_admission_reservation", 1)[0]
        self.assertIn('not self.blocks_next_schedule(request, "prefill")', begin)
        self.assertIn(
            "self.pd_request_fcfs\n                    and not ttft_prefill_policy_active_this_iteration",
            source,
        )

    def test_native_frontier_fills_token_budget_without_hard_width(self) -> None:
        frontier = build_native_prefill_admission_frontier(
            [
                PrefillAdmissionCandidate(f"request-{index}", 2048)
                for index in range(10)
            ],
            token_budget_before=16384,
            new_sequence_slots=10,
        )
        self.assertEqual(
            frontier.request_ids,
            tuple(f"request-{index}" for index in range(8)),
        )
        self.assertEqual(frontier.token_budget_after, 0)
        self.assertEqual(frontier.stop_reason, "max_num_batched_tokens")

    def test_native_frontier_reserves_decode_and_sequence_slots(self) -> None:
        frontier = build_native_prefill_admission_frontier(
            [
                PrefillAdmissionCandidate(
                    "waiting-short",
                    2048,
                    needs_new_sequence=True,
                ),
                PrefillAdmissionCandidate("running-long", 2048),
            ],
            token_budget_before=4096,
            decode_reserved_tokens=1024,
            new_sequence_slots=0,
        )
        self.assertEqual(frontier.request_ids, ("running-long",))
        self.assertEqual(
            frontier.sequence_blocked_request_ids,
            ("waiting-short",),
        )
        self.assertEqual(frontier.decode_reserved_tokens, 1024)
        self.assertEqual(frontier.token_budget_after, 1024)

    def test_native_frontier_keeps_tiered_work_single_target(self) -> None:
        frontier = build_native_prefill_admission_frontier(
            [
                PrefillAdmissionCandidate(
                    "ring-target",
                    2048,
                    requires_single_target=True,
                ),
                PrefillAdmissionCandidate("native-peer", 2048),
            ],
            token_budget_before=16384,
            new_sequence_slots=10,
        )
        self.assertEqual(frontier.request_ids, ("ring-target",))
        self.assertEqual(
            frontier.stop_reason,
            "single_target_tiered_prefill",
        )

    def test_fcfs_orders_by_arrival_then_request_id(self) -> None:
        scheduler = PrefillChunkScheduler("fcfs")
        order = scheduler.begin_iteration(
            [
                FakeRequest("req-c", 3.0, 100),
                FakeRequest("req-b", 1.0, 100),
                FakeRequest("req-a", 1.0, 100),
            ]
        )
        self.assertEqual(order, ("req-a", "req-b", "req-c"))

    def test_edf_prefers_finite_absolute_deadline(self) -> None:
        scheduler = PrefillChunkScheduler("edf")
        order = scheduler.begin_iteration(
            [
                FakeRequest("no-deadline", 1.0, 100),
                FakeRequest(
                    "later-arrival-earlier-deadline",
                    2.0,
                    100,
                    ttft_prefill_deadline_ms=100.0,
                ),
                FakeRequest(
                    "earlier-arrival-later-deadline",
                    1.0,
                    100,
                    ttft_prefill_deadline_ms=5000.0,
                ),
            ]
        )
        self.assertEqual(
            order,
            (
                "later-arrival-earlier-deadline",
                "earlier-arrival-later-deadline",
                "no-deadline",
            ),
        )

    def test_edf_global_relative_deadline_falls_back_to_fcfs(self) -> None:
        scheduler = PrefillChunkScheduler(
            "edf", default_deadline_ms=1000.0
        )
        order = scheduler.begin_iteration(
            [
                FakeRequest("second", 2.0, 100),
                FakeRequest("first", 1.0, 100),
            ]
        )
        self.assertEqual(order, ("first", "second"))

    def test_srpt_uses_durable_virtual_context_progress(self) -> None:
        scheduler = PrefillChunkScheduler("srpt")
        order = scheduler.begin_iteration(
            [
                FakeRequest("remaining-80", 1.0, 100, 20),
                FakeRequest(
                    "remaining-10-after-offload",
                    2.0,
                    100,
                    0,
                    virtual_context_deferred_logical_computed_tokens=90,
                ),
            ]
        )
        self.assertEqual(
            order,
            ("remaining-10-after-offload", "remaining-80"),
        )

    def test_round_robin_moves_only_after_successful_chunk(self) -> None:
        scheduler = PrefillChunkScheduler("round_robin")
        requests = [
            FakeRequest("a", 1.0, 100),
            FakeRequest("b", 2.0, 100),
            FakeRequest("c", 3.0, 100),
        ]
        self.assertEqual(scheduler.begin_iteration(requests)[0], "a")
        self.assertEqual(scheduler.begin_iteration(requests)[0], "a")
        scheduler.observe_scheduled_chunks(["a"])
        self.assertEqual(scheduler.begin_iteration(requests)[0], "b")
        scheduler.observe_scheduled_chunks(["b"])
        self.assertEqual(scheduler.begin_iteration(requests)[0], "c")
        self.assertEqual(
            scheduler.iteration_metrics()[
                "ttft_prefill_chunk_switch_count_total"
            ],
            1,
        )

    def test_prioritization_preserves_non_prefill_queue_slots(self) -> None:
        scheduler = PrefillChunkScheduler("srpt")
        longer = FakeRequest("longer-prefill", 1.0, 100, 10)
        shorter = FakeRequest("shorter-prefill", 2.0, 100, 90)
        decode_ready = FakeRequest("decode-ready", 0.5, 100, 100)
        scheduler.begin_iteration([longer, shorter])
        reordered = scheduler.prioritize_preferred_candidate_slot(
            [decode_ready, longer, shorter]
        )
        self.assertEqual(
            [request.request_id for request in reordered],
            ["decode-ready", "shorter-prefill", "longer-prefill"],
        )

    def test_srpt_reorders_an_explicit_native_admission_frontier(self) -> None:
        scheduler = PrefillChunkScheduler("srpt")
        longest = FakeRequest("longest", 1.0, 300)
        middle = FakeRequest("middle", 2.0, 200)
        shortest = FakeRequest("shortest", 3.0, 100)
        decode_ready = FakeRequest("decode-ready", 0.5, 100, 100)
        scheduler.begin_iteration([longest, middle, shortest])

        reordered = scheduler.prioritize_selected_candidate_slots(
            [decode_ready, longest, middle, shortest],
            selected_request_ids=("shortest", "middle"),
        )
        self.assertEqual(
            [request.request_id for request in reordered],
            ["decode-ready", "shortest", "middle", "longest"],
        )

    def test_quantum_caps_prefill_but_not_decode(self) -> None:
        scheduler = PrefillChunkScheduler(
            "round_robin", quantum_tokens=16
        )
        prefill = FakeRequest("prefill", 1.0, 100, 32)
        decode = FakeRequest("decode", 2.0, 100, 100)
        self.assertEqual(scheduler.cap_prefill_tokens(prefill, 64), 16)
        self.assertEqual(scheduler.cap_prefill_tokens(decode, 64), 64)

    def test_forget_removes_round_robin_and_metric_state(self) -> None:
        scheduler = PrefillChunkScheduler("round_robin")
        request = FakeRequest("done", 1.0, 100)
        scheduler.begin_iteration([request])
        scheduler.observe_scheduled_chunks(["done"])
        scheduler.forget_request("done")
        self.assertEqual(scheduler.begin_iteration([]), ())
        self.assertEqual(
            scheduler.request_metrics(request)[
                "ttft_prefill_policy_chunk_count"
            ],
            0,
        )


class VendorV8IntegrationContractTest(unittest.TestCase):
    def test_default_is_disabled_and_all_policies_are_cli_selectable(self) -> None:
        config = (VENDOR / "vllm/config/scheduler.py").read_text()
        args = (VENDOR / "vllm/engine/arg_utils.py").read_text()
        request = (VENDOR / "vllm/v1/request.py").read_text()
        self.assertIn(
            "enable_ttft_prefill_chunk_scheduling: bool = False", config
        )
        for policy in ("fcfs", "edf", "srpt", "round_robin"):
            self.assertIn(f'"{policy}"', config)
        self.assertIn(
            '"--enable-ttft-prefill-chunk-scheduling"', args
        )
        self.assertIn(
            '"--ttft-prefill-chunk-scheduling-policy"', args
        )
        self.assertIn(
            "enable_ttft_prefill_chunk_scheduling=(", args
        )
        self.assertIn(
            'sampling_params.extra_args.get(\n'
            '                    "ttft_prefill_deadline_ms"',
            request,
        )

    def test_vendor_v4_baseline_has_no_new_selector(self) -> None:
        for relative_path in (
            "vllm/config/scheduler.py",
            "vllm/engine/arg_utils.py",
            "vllm/v1/request.py",
            "vllm/v1/core/sched/scheduler.py",
        ):
            source = (ROOT / "vendor_v4" / relative_path).read_text()
            self.assertNotIn("ttft_prefill_chunk_scheduling", source)

    def test_scheduler_uses_native_budget_admission_frontier(self) -> None:
        scheduler = (VENDOR / "vllm/v1/core/sched/scheduler.py").read_text()
        self.assertIn("_ttft_begin_prefill_chunk_iteration", scheduler)
        self.assertIn("_ttft_plan_prefill_admission_frontier", scheduler)
        self.assertIn("_ttft_decode_admission_reservation", scheduler)
        self.assertIn("_ttft_prefill_request_is_selected", scheduler)
        self.assertIn("_ttft_cap_selected_prefill_chunk", scheduler)
        self.assertIn(
            "ttft_prefill_policy_skipped_waiting_reqs += 1", scheduler
        )
        self.assertIn(
            "ttft_prefill_chunk_admission_frontier_tokens",
            scheduler,
        )
        self.assertNotIn("_ttft_configured_prefill_batch_limit", scheduler)
        self.assertNotIn("_ttft_select_prefill_batch", scheduler)
        self.assertNotIn("max_num_partial_prefills", scheduler)
        self.assertNotIn("max_long_partial_prefills", scheduler)


if __name__ == "__main__":
    unittest.main(verbosity=2)
