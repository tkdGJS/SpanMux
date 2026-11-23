#!/usr/bin/env python3
"""CPU-only tests for vendor_v6-derived decode-ready scheduling."""

from __future__ import annotations

import importlib.util
import os
import unittest
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = Path(os.environ.get("VLLM_VENDOR_DIR", ROOT / "vendor_v6")).resolve()
MODULE_PATH = VENDOR / "vllm/v1/core/sched/decode_scheduler.py"
SPEC = importlib.util.spec_from_file_location(
    "vendor_v6_decode_scheduler", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TBTDecodeScheduler = MODULE.TBTDecodeScheduler


@dataclass
class FakeRequest:
    request_id: str
    arrival_time: float
    prefill_completed_at: float | None = None
    prefill_completed_iteration_id: int = -1


class TBTDecodeSchedulerTest(unittest.TestCase):
    def test_fcfs_uses_prefill_completion_then_arrival_and_id(self) -> None:
        scheduler = TBTDecodeScheduler("fcfs")
        order = scheduler.begin_iteration(
            [
                FakeRequest("later", 1.0, 30.0, 3),
                FakeRequest("same-b", 2.0, 20.0, 2),
                FakeRequest("same-a", 1.0, 20.0, 2),
                FakeRequest("same-a2", 1.0, 20.0, 2),
            ]
        )
        self.assertEqual(
            order,
            ("same-a", "same-a2", "same-b", "later"),
        )

    def test_unconfirmed_prefill_is_not_decode_candidate(self) -> None:
        scheduler = TBTDecodeScheduler()
        order = scheduler.begin_iteration(
            [
                FakeRequest("prefill", 1.0),
                FakeRequest("decode", 2.0, 10.0, 4),
            ]
        )
        self.assertEqual(order, ("decode",))

    def test_prefill_completion_is_recorded_once(self) -> None:
        request = FakeRequest("req", 1.0)
        self.assertTrue(
            TBTDecodeScheduler.mark_prefill_completed(
                request,
                completed_at=10.0,
                iteration_id=7,
            )
        )
        self.assertFalse(
            TBTDecodeScheduler.mark_prefill_completed(
                request,
                completed_at=20.0,
                iteration_id=8,
            )
        )
        self.assertEqual(request.prefill_completed_at, 10.0)
        self.assertEqual(request.prefill_completed_iteration_id, 7)

    def test_prioritization_preserves_prefill_slots(self) -> None:
        scheduler = TBTDecodeScheduler()
        decode_later = FakeRequest("decode-later", 1.0, 30.0, 3)
        prefill_a = FakeRequest("prefill-a", 2.0)
        decode_first = FakeRequest("decode-first", 3.0, 20.0, 2)
        prefill_b = FakeRequest("prefill-b", 4.0)
        scheduler.begin_iteration([decode_later, decode_first])
        reordered = scheduler.prioritize_candidate_slots(
            [decode_later, prefill_a, decode_first, prefill_b]
        )
        self.assertEqual(
            [request.request_id for request in reordered],
            ["decode-first", "prefill-a", "decode-later", "prefill-b"],
        )

    def test_observe_and_forget_update_metrics(self) -> None:
        scheduler = TBTDecodeScheduler()
        request = FakeRequest("decode", 1.0, 10.0, 2)
        scheduler.begin_iteration([request])
        scheduler.observe_scheduled_tokens(["decode"])
        self.assertEqual(
            scheduler.request_metrics(request)["tbt_decode_scheduled_turns"],
            1,
        )
        scheduler.forget_request("decode")
        self.assertEqual(
            scheduler.request_metrics(request)["tbt_decode_scheduled_turns"],
            0,
        )


class VendorV6DecodeIntegrationContractTest(unittest.TestCase):
    def test_default_off_config_and_cli_wiring(self) -> None:
        config = (VENDOR / "vllm/config/scheduler.py").read_text()
        args = (VENDOR / "vllm/engine/arg_utils.py").read_text()
        self.assertIn("enable_tbt_decode_scheduling: bool = False", config)
        self.assertIn('TBTDecodeSchedulingPolicy = Literal["fcfs"]', config)
        self.assertIn('"--enable-tbt-decode-scheduling"', args)
        self.assertIn('"--tbt-decode-scheduling-policy"', args)
        self.assertIn("enable_tbt_decode_scheduling=(", args)

    def test_request_and_unified_scheduler_hooks_exist(self) -> None:
        request = (VENDOR / "vllm/v1/request.py").read_text()
        scheduler = (VENDOR / "vllm/v1/core/sched/scheduler.py").read_text()
        self.assertIn("self.prefill_completed_at: float | None = None", request)
        self.assertIn("_tbt_begin_decode_iteration", scheduler)
        self.assertIn("_tbt_prioritize_decode_candidates", scheduler)
        self.assertIn("mark_prefill_completed(", scheduler)
        self.assertIn("tbt_decode_scheduled_request_ids", scheduler)

    def test_vendor_v4_baseline_has_no_decode_selector(self) -> None:
        for relative_path in (
            "vllm/config/scheduler.py",
            "vllm/engine/arg_utils.py",
            "vllm/v1/request.py",
            "vllm/v1/core/sched/scheduler.py",
        ):
            source = (ROOT / "vendor_v4" / relative_path).read_text()
            self.assertNotIn("tbt_decode_scheduling", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
