#!/usr/bin/env python3
"""CPU contracts for real phase quota methods/frontier and pre-mutation hooks."""
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from test_slo_prefill_scheduler import MODULE as P


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vendor_v8/vllm/v1/core/sched/scheduler.py"
TREE = ast.parse(SOURCE.read_text())
SCHEDULER = next(n for n in TREE.body if isinstance(n, ast.ClassDef)
                 and n.name == "Scheduler")
METHODS = {n.name: n for n in SCHEDULER.body if isinstance(n, ast.FunctionDef)}
ENV_P = "VLLM_ISOLATION_PREFILL_BATCH_MAX_REQUESTS"
ENV_D = "VLLM_ISOLATION_DECODE_BATCH_MAX_REQUESTS"


def harness():
    names = ("_configure_isolation_phase_batch_limits", "_isolation_phase_request_limit",
             "_ttft_plan_prefill_admission_frontier", "can_resume_kv_retention_prefill",
             "resume_kv_retention_prefill")
    cls = ast.ClassDef(name="Harness", bases=[], keywords=[], decorator_list=[],
                      body=[copy.deepcopy(METHODS[n]) for n in names])
    ns = dict(os=os, logger=NS(info=lambda *args: None), Request=object,
              SchedulerOutput=object, PrefillAdmissionFrontier=P.PrefillAdmissionFrontier,
              PrefillAdmissionCandidate=P.PrefillAdmissionCandidate,
              PrefillChunkSchedulingPolicy=P.PrefillChunkSchedulingPolicy,
              build_native_prefill_admission_frontier=P.build_native_prefill_admission_frontier)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(SOURCE), "exec"), ns)
    h = ns["Harness"]()
    h.scheduler_config = NS(max_num_seqs=2, long_prefill_token_threshold=2048)
    h.pd_decoupled_no_mix = True
    return h


def quota_guard(loop):
    guard = loop.body[0]
    assert isinstance(guard, ast.If)
    assert "isolation_phase_request_limit" in ast.unparse(guard.test)
    return compile(ast.Expression(copy.deepcopy(guard.test)), str(SOURCE), "eval")


class PhaseBatchLimitTests(unittest.TestCase):
    def test_default_off_allows_mixed(self):
        h = harness()
        h.pd_decoupled_no_mix = False
        with patch.dict(os.environ, {ENV_P: "0", ENV_D: "0"}):
            h._configure_isolation_phase_batch_limits()
        self.assertEqual(h._isolation_phase_request_limit("prefill"), 0)
        self.assertEqual(h._isolation_phase_request_limit("decode"), 0)

    def test_separate_limits_do_not_change_sequence_or_future_budgets(self):
        h = harness()
        h.pd_prefill_inflight_depth = 2
        h.pd_decode_inflight_depth = 1
        with patch.dict(os.environ, {ENV_P: "1", ENV_D: "2"}):
            h._configure_isolation_phase_batch_limits()
        self.assertEqual(h._isolation_phase_request_limit("prefill"), 1)
        self.assertEqual(h._isolation_phase_request_limit("decode"), 2)
        self.assertEqual(h._isolation_phase_request_limit("idle"), 0)
        self.assertEqual(h._isolation_phase_request_limit("mixed"), 0)
        self.assertEqual(h.scheduler_config.max_num_seqs, 2)
        self.assertEqual(h.pd_prefill_inflight_depth, 2)

    def test_invalid_limits_fail_closed(self):
        for raw in ("-1", "1.0", "x", "", " 1", "３", "3"):
            with self.subTest(raw=raw), patch.dict(os.environ, {ENV_P: raw, ENV_D: "0"}):
                with self.assertRaises(ValueError):
                    harness()._configure_isolation_phase_batch_limits()

    def test_phase_limit_cannot_contaminate_native_mixed(self):
        h = harness()
        h.pd_decoupled_no_mix = False
        with patch.dict(os.environ, {ENV_P: "0", ENV_D: "1"}):
            with self.assertRaisesRegex(ValueError, "phase-pure"):
                h._configure_isolation_phase_batch_limits()

    def test_frontier_short_tail_still_consumes_one_request(self):
        rows = [P.PrefillAdmissionCandidate("short", 1, True, False),
                P.PrefillAdmissionCandidate("long", 2048, True, False)]
        capped = P.build_native_prefill_admission_frontier(
            rows, token_budget_before=16384, new_sequence_slots=2, request_limit=1)
        self.assertEqual(capped.request_ids, ("short",))
        self.assertEqual(capped.token_budget_after, 16383)
        self.assertEqual(capped.stop_reason, "isolation_phase_request_limit")
        default = P.build_native_prefill_admission_frontier(
            rows, token_budget_before=16384, new_sequence_slots=2)
        self.assertEqual(default.request_ids, ("short", "long"))

    def test_frontier_negative_limit_rejected(self):
        with self.assertRaises(ValueError):
            P.build_native_prefill_admission_frontier(
                [], token_budget_before=2048, new_sequence_slots=2, request_limit=-1)

    def test_scheduler_frontier_keeps_waiting_short_ahead_of_running_long(self):
        h = harness()
        h.enable_ttft_prefill_chunk_scheduling = True
        h.max_num_scheduled_tokens = 16384
        h.max_num_running_reqs = 2
        h.max_model_len = 65536
        h.isolation_prefill_batch_max_requests = 1
        h.ttft_prefill_chunk_scheduler = NS(policy=P.PrefillChunkSchedulingPolicy.SRPT)
        short = NS(request_id="S", num_computed_tokens=0,
                   virtual_context_deferred_logical_computed_tokens=0)
        long = NS(request_id="L", num_computed_tokens=20000,
                  virtual_context_deferred_logical_computed_tokens=0)
        h.running, h.waiting, h.requests = [long], [short], {"S": short, "L": long}
        h._pd_prefill_lane_remaining_tokens = lambda r, **kw: 100 if r is short else 30000
        h._ttft_prefill_execution_mode = lambda r: "native"
        h._ttft_cap_selected_prefill_chunk = lambda r, n, **kw: min(n, 2048)
        frontier = h._ttft_plan_prefill_admission_frontier(
            ("S", "L"), reserve_decode_work=False)
        self.assertEqual(frontier.request_ids, ("S",))
        # The mixed Baseline+SRPT has no phase-only request restriction.
        h.pd_decoupled_no_mix = False
        frontier = h._ttft_plan_prefill_admission_frontier(
            ("S", "L"), reserve_decode_work=False)
        self.assertEqual(frontier.request_ids, ("S", "L"))

    def test_real_running_and_waiting_guards_precede_all_candidate_mutations(self):
        schedule = METHODS["schedule"]
        loops = [n for n in ast.walk(schedule) if isinstance(n, ast.While)]
        running = next(n for n in loops if "req_index < len(self.running)" in ast.unparse(n.test))
        waiting = next(n for n in loops if ast.unparse(n.test) == "self.waiting and token_budget > 0")
        for loop in (running, waiting):
            with self.subTest(loop=ast.unparse(loop.test)):
                condition = quota_guard(loop)
                for limit, admitted, expected in ((0, {"L": 2048}, False),
                        (1, {}, False), (1, {"L": 2048}, True),
                        (1, {"D": 1}, True), (2, {"D": 1}, False)):
                    self.assertEqual(eval(condition, dict(
                        isolation_phase_request_limit=limit,
                        num_scheduled_tokens=admitted)), expected)
                self.assertIsInstance(loop.body[0].body[-1], ast.Break)
                # Evaluation cannot allocate, inspect/modify requests, or publish output.
                self.assertFalse(any(isinstance(n, (ast.Attribute, ast.Subscript))
                                     for n in ast.walk(loop.body[0].test)))

    def test_failed_allocation_and_preemption_refund_row_budget(self):
        loop = next(n for n in ast.walk(METHODS["schedule"]) if isinstance(n, ast.While)
                    and "req_index < len(self.running)" in ast.unparse(n.test))
        condition = quota_guard(loop)
        admitted = {}
        env = dict(isolation_phase_request_limit=1, num_scheduled_tokens=admitted)
        self.assertFalse(eval(condition, env))  # failed allocation never enters map
        admitted["L"] = 2048
        self.assertTrue(eval(condition, env))
        admitted.pop("L")  # ordinary preemption rollback removes actual row
        self.assertFalse(eval(condition, env))

    def test_whole_parked_resume_does_not_trim_or_reschedule(self):
        h = harness()
        h.max_num_running_reqs = 2
        h.isolation_prefill_batch_max_requests = 1
        row = NS(is_finished=lambda: False)
        h._kv_retention_parked_requests = {"L": row}
        h.requests = {"L": row}
        h.running = [object(), object()]
        output = NS(num_scheduled_tokens={"L": 2048})
        self.assertFalse(h.can_resume_kv_retention_prefill(output))
        h.running.pop()  # actual active completion, not a quota bypass
        self.assertTrue(h.can_resume_kv_retention_prefill(output))
        h.resume_kv_retention_prefill(output)
        self.assertIs(h.running[-1], row)
        self.assertEqual(output.num_scheduled_tokens, {"L": 2048})
        self.assertEqual(h._kv_retention_parked_requests, {})


if __name__ == "__main__":
    unittest.main()
