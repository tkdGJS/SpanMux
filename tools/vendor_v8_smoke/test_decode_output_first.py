#!/usr/bin/env python3
"""Run the real Engine recovery method with CPU Futures, not a copied policy."""
from __future__ import annotations

import ast
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "vendor_v8/vllm/v1/engine/core.py"
tree = ast.parse(PATH.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCore")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == "step_with_pd_phase_queues")
submit_method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                     and n.name == "_submit_pd_scheduler_lanes")
ns = {"time": time, "pd_engine_progress_generation": lambda: 1,
      "TieredBackendEligibilityError": type("TieredError", (Exception,), {}),
      "TTFTLayerwisePrefillBoundary": type("Boundary", (), {})}
exec(compile(ast.fix_missing_locations(ast.Module(body=[
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    method, submit_method], type_ignores=[])), str(PATH), "exec"), ns)


class AdmissionBoundaryTests(unittest.TestCase):
    def make_engine(self, *, output_first=True, bounded=False, ready=False,
                    independent=True, becomes_ready=False, failed=False):
        calls = []
        d = Future()
        if ready:
            d.set_exception(RuntimeError("worker error")) if failed else d.set_result(None)
        q = {"decode": deque([(d, NS(), d)]), "prefill": deque()}
        engine = NS(batch_queue_size=2, isolation_decode_output_first=output_first)
        engine.scheduler = NS(isolation_bounded_prefill_planning=bounded,
                              has_requests=lambda: True,
                              prepare_pd_prefill_admission=lambda: True)
        engine._pd_phase_queue_total_depth = lambda qs: sum(map(len, qs.values()))
        engine._select_pd_phase_queue_recovery_item = lambda qs, **kw: (
            "decode" if independent else "prefill", 0, {})
        engine._try_submit_pd_paired_phase_launch = lambda qs: (False, None, None, False)
        def can_submit(qs, phase):
            # Completion after the entry and first lane checks, before Prefill.
            if phase == "decode" and becomes_ready and not d.done():
                d.set_result(None)
            return phase == "prefill"
        engine._pd_phase_queue_can_submit_phase = can_submit
        engine._pd_capacity_retry_can_submit = lambda phase: True
        engine._pd_phase_queue_pending_request_ids = lambda qs: {"decode-owner"}
        engine._pd_phase_queue_pending_request_ids_by_phase = lambda qs: {"decode": {"decode-owner"}}
        engine._set_scheduler_pd_pending_request_ids = lambda ids: calls.append(("pending", ids))
        engine._set_scheduler_pd_pending_phase_request_ids = lambda ids: calls.append(("phases", ids))
        engine._scheduler_has_pd_phase_work = lambda s, phase: True
        def schedule(s, phase):
            calls.append(("schedule", phase))
            return NS(total_num_scheduled_tokens=0, finished_req_ids=set())
        engine._scheduler_schedule_pd_phase = schedule
        engine._record_pd_capacity_retry_block = lambda *a: None
        engine._write_pd_phase_queue_trace = lambda event, *a: calls.append(("trace", event))
        return engine, q, calls

    def run_submit(self, **kwargs):
        e, q, calls = self.make_engine(**kwargs)
        result = ns["_submit_pd_scheduler_lanes"](
            e, q, ("decode", "prefill"), allow_multiple_rounds=True)
        self.assertEqual(result, (False, None, None, False))
        self.assertEqual(calls[-2:], [("pending", set()), ("phases", {})])
        self.assertEqual(len(q["decode"]), 1)  # no early pop/output commit
        return calls

    def test_output_first_does_not_require_bounded_prewarming(self):
        calls = self.run_submit(ready=True)
        self.assertNotIn(("schedule", "prefill"), calls)

    def test_completion_during_lane_scan_yields_before_prefill(self):
        calls = self.run_submit(becomes_ready=True)
        self.assertNotIn(("schedule", "prefill"), calls)

    def test_error_future_goes_to_normal_recovery_not_next_admission(self):
        calls = self.run_submit(ready=True, failed=True)
        self.assertNotIn(("schedule", "prefill"), calls)

    def test_incomplete_decode_does_not_starve_prefill(self):
        self.assertIn(("schedule", "prefill"), self.run_submit())

    def test_dependency_unsafe_decode_does_not_bypass_selector(self):
        self.assertIn(("schedule", "prefill"),
                      self.run_submit(ready=True, independent=False))

    def test_both_flags_disabled_preserve_existing_order(self):
        self.assertIn(("schedule", "prefill"),
                      self.run_submit(output_first=False, ready=True))

    def test_bounded_only_still_yields(self):
        self.assertNotIn(("schedule", "prefill"),
                         self.run_submit(output_first=False, bounded=True, ready=True))


class RecoveryTests(unittest.TestCase):
    def make_engine(self, enabled=True, ready=True, error=None, aborted=False):
        calls = []
        d, p = Future(), Future()
        output = NS(ttft_layerwise_prefill_preemption=False,
                    kv_retention_layerwise_transfer=None, total_num_scheduled_tokens=1)
        model = NS(model_runner_debug_timings={})
        if ready:
            d.set_exception(error) if error else d.set_result(model)
        q = {"decode": deque([(d, output, d)]),
             "prefill": deque([(p, output, p)])}
        e = NS(pd_phase_queues=q, isolation_decode_output_first=enabled,
               pd_dual_continuous_batching_enabled=True,
               pd_separate_scheduler_lanes_enabled=True,
               pd_nonblocking_phase_recovery_enabled=True,
               pd_phase_queue_trace_jsonl="", full_iteration_debug_enabled=False,
               batch_queue_size=2, _ttft_prefill_suspended={},
               _iteration_debug_timings={id(output): {}}, calls=calls)
        e._pd_phase_queue_total_depth = lambda queues: sum(map(len, queues.values()))
        e._select_pd_phase_queue_recovery_item = lambda queues, **kw: (
            ("decode", 0, {}) if queues["decode"] else ("prefill", 0, {}))
        e._oldest_pd_phase_queue_item = lambda queues: {"future": next(
            item[0] for queue in queues.values() for item in queue)}
        e._pop_pd_phase_queue_at = lambda queues, phase, index: queues[phase].popleft()
        for name in ("_advance_pd_sm_transition", "_poll_kv_retention_parked_offload",
                     "_poll_pd_layerwise_handoff", "_prune_ttft_prefill_suspended",
                     "_request_ttft_prefill_preemption_if_needed"):
            setattr(e, name, lambda *a, name=name: calls.append(name))
        e._write_pd_phase_queue_trace = lambda *a: None
        e._pd_capacity_retry_snapshot = lambda: {}
        e._kv_retention_future_binding = lambda *a: None
        e.log_error_detail = e.log_iteration_details = lambda *a: nullcontext()
        e._process_aborts_queue = lambda: calls.append("abort")
        e._pd_capacity_retry_free_blocks = lambda: 100
        e._notify_pd_capacity_retry_after_output = lambda *a: None
        e._pd_ttft_handoff_clear_from_outputs = lambda *a: None
        e._observe_pd_sm_plan = lambda *a: calls.append("observe")
        e._write_iteration_debug_jsonl = lambda *a: calls.append("metrics")
        e._wait_for_pd_progress = lambda *a, **kw: calls.append("wait")
        e._set_scheduler_pd_pending_request_ids = lambda *a: None
        e._set_scheduler_pd_pending_phase_request_ids = lambda *a: None
        def update(*args):
            calls.append("update")
            return {} if aborted else {0: "token"}
        e.scheduler = NS(has_requests=lambda: True, update_from_output=update)
        def submit(*a, **kw):
            calls.append("admission")
            return False, None, None, False
        e._submit_pd_scheduler_lanes = submit
        return e, d, p

    def test_ready_decode_delivered_before_control_and_next_admission(self):
        e, _, p = self.make_engine()
        outputs, _ = ns["step_with_pd_phase_queues"](e)
        self.assertEqual(outputs, {0: "token"})
        self.assertEqual(e.calls, ["abort", "update", "observe", "metrics"])
        self.assertFalse(p.done())
        self.assertEqual(len(e.pd_phase_queues["prefill"]), 1)

    def test_deferred_control_runs_on_next_step_no_starvation(self):
        e, _, _ = self.make_engine()
        ns["step_with_pd_phase_queues"](e)
        e.calls.clear()
        ns["step_with_pd_phase_queues"](e)
        self.assertIn("_advance_pd_sm_transition", e.calls)
        self.assertIn("_poll_kv_retention_parked_offload", e.calls)
        self.assertIn("admission", e.calls)
        self.assertEqual(e.calls[-1], "wait")

    def test_flag_off_preserves_control_and_post_recovery_submit(self):
        e, _, _ = self.make_engine(False)
        ns["step_with_pd_phase_queues"](e)
        self.assertEqual(e.calls[0], "_advance_pd_sm_transition")
        self.assertGreater(e.calls.index("admission"), e.calls.index("update"))

    def test_not_ready_waits_without_fabricated_output(self):
        e, _, _ = self.make_engine(ready=False)
        outputs, _ = ns["step_with_pd_phase_queues"](e)
        self.assertIsNone(outputs)
        self.assertNotIn("update", e.calls)
        self.assertIn("_advance_pd_sm_transition", e.calls)
        self.assertEqual(e.calls[-1], "wait")

    def test_error_propagates_without_success_update(self):
        e, _, p = self.make_engine(error=RuntimeError("worker failed"))
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            ns["step_with_pd_phase_queues"](e)
        self.assertNotIn("update", e.calls)
        self.assertFalse(p.done())

    def test_cancel_still_precedes_update_without_new_admission(self):
        e, _, _ = self.make_engine(aborted=True)
        outputs, _ = ns["step_with_pd_phase_queues"](e)
        self.assertEqual(outputs, {})
        self.assertLess(e.calls.index("abort"), e.calls.index("update"))
        self.assertNotIn("admission", e.calls)


if __name__ == "__main__":
    unittest.main()
