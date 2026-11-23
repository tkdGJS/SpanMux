#!/usr/bin/env python3
"""CPU-only predictive SM policy plus real Engine/Scheduler method contracts.

Synthetic latency values test decisions; they are NOT deployment calibration.
AST loading avoids importing CUDA/vLLM dependencies, not the methods under test.
"""
from __future__ import annotations

import ast
from collections import deque
from concurrent.futures import Future
from dataclasses import replace
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
LEGACY = runpy.run_path(str(Path(__file__).with_name("test_tbt_profiled_sm_allocator.py")))
Controller = LEGACY["PDComputeAllocationController"]
Config = LEGACY["PDSMAllocatorConfig"]
Profile = LEGACY["TBTOverlapProfile"]
from vllm.pd_sm_profiles import parse_pd_sm_profiles


def method(path, cls, name, namespace=None):
    tree = ast.parse((ROOT / path).read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    ns = {"time": time, **(namespace or {})}
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), path, "exec"), ns)
    return ns[name]


ENGINE = "vendor_v8/vllm/v1/engine/core.py"
SCHEDULER = "vendor_v8/vllm/v1/core/sched/scheduler.py"


class PredictiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "profile.json"
        self.payload = {"schema_version": 2, "measurement_contract": LEGACY["MEASUREMENT_CONTRACT"], "points": []}
        for profile_id, low, high in (("d30_p10", 24, 33), ("d32_p8", 22, 28),
                                      ("d34_p6", 20, 24), ("d38_p2", 18, 21), ("d40_p0", 17, 20)):
            for mode in (("none",) if profile_id == "d40_p0" else ("native", "ring")):
                for length, cost in ((1000, low), (10000, high)):
                    self.payload["points"].append({"profile_id": profile_id,
                        "max_decode_context_tokens": length, "max_decode_requests": 4,
                        "prefill_mode": mode, "p95_per_token_tbt_ms": cost + (8 if mode == "ring" else 0)})
        self.path.write_text(json.dumps(self.payload))
        profiles = parse_pd_sm_profiles("0/40,30/10,32/8,34/6,38/2,40/0", total_sms=40, alignment=2)
        self.config = LEGACY["_config"](str(self.path), profiles=profiles, initial_profile=profiles[1],
            policy="tbt_slo_predictive", tbt_profile_margin_ms=1.0, tbt_default_slo_ms=30.0,
            tbt_predictive_lead_ms=40.0, tbt_predictive_cutover_ms=2.0, tbt_predictive_min_step_ms=10.0)
        self.c = Controller(self.config)

    def plan(self, **kw):
        args = dict(decode_ready=True, prefill_ready=True, decode_inflight=False,
                    prefill_inflight=True, decode_context_tokens=900, decode_request_count=2,
                    decode_tbt_slo_ms=30.0, decode_min_tbt_slo_ms=30.0, prefill_mode="native")
        args.update(kw)
        return self.c.plan(**args)

    def test_smallest_sm_and_preemptive_upshift_before_observed_violation(self):
        self.assertEqual(self.plan().target_profile.profile_id, "d30_p10")
        p = self.plan(decode_context_tokens=996)
        self.assertEqual(p.target_profile.profile_id, "d34_p6")
        self.assertEqual(p.profile.profile_id, "d30_p10")
        self.assertEqual(p.tbt_predictive["pd_tbt_predictive_context_end"], 1001)
        self.assertTrue(p.deferred)
        self.assertTrue(p.tbt_slo_emergency)  # bypass cooldown before violation
        self.assertTrue(p.tbt_slo_feasible)
        self.assertEqual(self.c.epoch, 0)  # prediction grants no execution authority

    def test_lead_is_not_decode_pause(self):
        p = self.plan(decode_context_tokens=996)
        self.assertEqual(p.tbt_predicted_next_ms, 27.0)  # 24 + margin1 + cutover2, NOT +40
        self.assertEqual(p.tbt_predictive["pd_tbt_predictive_lead_ms"], 40.0)

    def test_no_source_decode_really_waits_for_preparation(self):
        self.c.current_profile = self.config.profiles[0]
        p = self.plan()
        self.assertTrue(p.tbt_slo_overload)
        self.assertGreater(p.tbt_predicted_next_ms, 30)
        self.assertEqual(p.profile.decode_sms, 0)

    def test_running_decode_residual_is_not_fabricated(self):
        p = self.plan(decode_inflight=True, decode_tbt_elapsed_ms=25)
        self.assertEqual(p.tbt_predicted_next_ms, -1.0)
        self.assertFalse(p.tbt_slo_feasible)
        self.assertEqual(p.tbt_predictive["pd_tbt_predictive_forecast_feasible"], 1.0)
        self.assertEqual(p.target_profile.decode_sms, 30)

    def test_strictest_slo_constrains_future_after_critical_request_changes(self):
        p = self.plan(decode_context_tokens=996, decode_inflight=True,
                      decode_tbt_slo_ms=100, decode_tbt_elapsed_ms=90, decode_min_tbt_slo_ms=30)
        self.assertEqual(p.target_profile.decode_sms, 34)

    def test_ring_interference_changes_minimum_feasible_sm(self):
        self.assertEqual(self.plan(prefill_mode="ring").target_profile.decode_sms, 38)

    def test_no_extrapolation_or_feasibility_claim_under_overload(self):
        p = self.plan(decode_context_tokens=9999)
        self.assertTrue(p.tbt_slo_overload)
        self.assertFalse(p.tbt_profile_state_covered)
        self.assertEqual(p.target_profile.decode_sms, 38)
        p = self.plan(decode_request_count=5)
        self.assertTrue(p.tbt_slo_overload)
        self.assertEqual(p.tbt_predictive["pd_tbt_predictive_covered_profiles"], 0)

    def test_next_unsubmitted_token_honors_remaining_not_full_slo(self):
        p = self.plan(decode_tbt_elapsed_ms=20)
        self.assertTrue(p.tbt_slo_overload)
        self.assertFalse(p.tbt_slo_feasible)

    def test_solo_decode_and_prefill_keep_work_conserving_endpoints(self):
        p = self.plan(prefill_ready=False, prefill_inflight=False)
        self.assertEqual(p.profile.decode_sms, 40)
        p = self.plan(decode_ready=False, prefill_inflight=False)
        self.assertEqual(p.profile.prefill_sms, 40)

    def test_envelope_checks_intermediate_nonmonotonic_bucket(self):
        base = self.payload["points"][0]
        payload = dict(self.payload, points=[dict(base, max_decode_context_tokens=c, p95_per_token_tbt_ms=t)
                                          for c, t in ((1000, 20), (1002, 60), (1010, 10))])
        result = Profile.from_dict(payload).predict_context_envelope(profile_id="d30_p10",
                    context_start=999, context_end=1005, decode_requests=2, prefill_mode="native")
        self.assertEqual(result.p95_per_token_tbt_ms, 60)

    def test_calibration_and_profile_required_even_when_strict_false(self):
        for name, value in (("tbt_predictive_lead_ms", 0), ("tbt_predictive_cutover_ms", -1),
                            ("tbt_predictive_min_step_ms", 0), ("tbt_predictive_lead_ms", float("nan"))):
            with self.subTest(name=name), self.assertRaises(ValueError):
                Controller(replace(self.config, **{name: value}))
        with self.assertRaises(ValueError):
            Controller(replace(self.config, tbt_profile_path="missing", tbt_profile_strict=False))

    def test_environment_and_old_policy_rollback(self):
        with patch.dict(os.environ, {"VLLM_PD_DYNAMIC_SM_ENABLED": "1", "VLLM_PD_DYNAMIC_SM_TOTAL_SMS": "40",
                "VLLM_PD_DYNAMIC_SM_POLICY": "tbt_slo_predictive", "VLLM_PD_TBT_PROFILE_PATH": str(self.path),
                "VLLM_PD_TBT_PREDICTIVE_LEAD_MS": "40", "VLLM_PD_TBT_PREDICTIVE_CUTOVER_MS": "2",
                "VLLM_PD_TBT_PREDICTIVE_MIN_STEP_MS": "10"}, clear=True):
            self.assertTrue(Controller(Config.from_env()).predictive_enabled)
        c = Controller(replace(self.config, policy="tbt_slo_feedback", tbt_profile_path="missing",
                               tbt_predictive_lead_ms=0, tbt_predictive_min_step_ms=0))
        self.assertFalse(c.predictive_enabled)

    def test_deadline_wake_once_per_token_and_abort_disarms(self):
        self.c.arm_tbt_deadline(request_id="a", deadline_at=11)
        self.assertAlmostEqual(self.c.tbt_deadline_wait_s(10), .999)
        self.assertTrue(self.c.consume_tbt_deadline(11))
        self.c.arm_tbt_deadline(request_id="a", deadline_at=11)
        self.assertIsNone(self.c.tbt_deadline_wait_s(11))
        self.c.arm_tbt_deadline(request_id="a", deadline_at=12)
        self.assertIsNotNone(self.c.tbt_deadline_wait_s(11))
        self.c.arm_tbt_deadline(request_id="", deadline_at=0)
        self.assertIsNone(self.c.tbt_deadline_wait_s(11))

    def test_transition_measurement_excludes_useful_decode_service_and_cancellation(self):
        d = {"pd_sm_transition_source_profile": "d30_p10", "pd_sm_transition_target_profile": "d34_p6",
             "pd_sm_transition_requested_ts": 10, "pd_sm_transition_target_decode_sms": 34,
             "pd_sm_transition_committed_epoch": 1}
        self.c.note_tbt_transition_request(requested_at=10)
        self.c.note_tbt_decode_completion(profile_id="d30_p10", now=10.039)
        self.c.note_tbt_transition_commit(d)
        self.c.note_tbt_decode_submission(profile_id="d34_p6", epoch=1, now=10.045)
        lead, cutover = self.c._tbt_predictive_transition_highwater["native"]
        self.assertAlmostEqual(lead, 45)
        self.assertAlmostEqual(cutover, 6)
        self.c.note_tbt_transition_commit(dict(d, pd_sm_transition_cancelled=1))
        self.c.note_tbt_decode_submission(profile_id="d34_p6", epoch=1, now=100)
        self.assertEqual(self.c._tbt_predictive_transition_highwater["native"], (lead, cutover))

    def test_transition_sample_binds_epoch_and_original_mode(self):
        self.c._tbt_predictive_mode = "ring"
        self.c.note_tbt_transition_request(requested_at=10)
        self.c._tbt_predictive_mode = "native"
        self.c.note_tbt_transition_request(requested_at=10)  # same transition retarget
        self.c.note_tbt_transition_commit({"pd_sm_transition_source_profile": "d30_p10",
            "pd_sm_transition_target_profile": "d34_p6", "pd_sm_transition_requested_ts": 10,
            "pd_sm_transition_target_decode_sms": 34, "pd_sm_transition_committed_epoch": 3})
        self.c.note_tbt_decode_submission(profile_id="d34_p6", epoch=2, now=10.1)
        self.assertEqual(self.c._tbt_predictive_transition_highwater, {})
        self.c.note_tbt_decode_submission(profile_id="d34_p6", epoch=3, now=10.2)
        self.assertIn("ring", self.c._tbt_predictive_transition_highwater)
        self.assertNotIn("native", self.c._tbt_predictive_transition_highwater)

    def test_new_transition_discards_unconsumed_old_sample(self):
        self.c.note_tbt_transition_request(requested_at=10)
        self.c.note_tbt_transition_commit({"pd_sm_transition_source_profile": "d30_p10",
            "pd_sm_transition_target_profile": "d34_p6", "pd_sm_transition_requested_ts": 10,
            "pd_sm_transition_target_decode_sms": 34, "pd_sm_transition_committed_epoch": 3})
        self.c.note_tbt_transition_request(requested_at=11)
        self.c.note_tbt_decode_submission(profile_id="d34_p6", epoch=3, now=11.1)
        self.assertEqual(self.c._tbt_predictive_transition_highwater, {})

    def test_actual_scheduler_subset_and_stable_deadline(self):
        state = method(SCHEDULER, "Scheduler", "get_tbt_allocation_state")
        def req(rid, slo, ts, context):
            return NS(request_id=rid, tbt_slo_ms=slo, tbt_last_token_at=ts,
                      tbt_last_observed_ms=20, num_computed_tokens=context,
                      num_prompt_tokens=1, is_finished=lambda: False)
        requests = {r.request_id: r for r in (req("long", 100, 10, 900), req("short", 30, 10.08, 100), req("waiting", 1, 1, 9000))}
        s = NS(requests=requests, _tbt_decode_request_is_candidate=lambda r: True,
               tbt_decode_scheduler=NS(feedback_snapshot=lambda *a, **kw: {}))
        with patch.object(time, "monotonic", return_value=10.09):
            a = state(s, decode_request_ids={"long", "short"})
        with patch.object(time, "monotonic", return_value=10.091):
            b = state(s, decode_request_ids={"long", "short"})
        self.assertEqual(a["critical_request_id"], "long")
        self.assertEqual(a["decode_min_tbt_slo_ms"], 30)
        self.assertEqual(a["decode_request_count"], 2)
        self.assertEqual(a["decode_tbt_deadline_at"], b["decode_tbt_deadline_at"])

    def test_actual_scheduler_enables_token_clock_without_decode_ordering(self):
        tree = ast.parse((ROOT / SCHEDULER).read_text())
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
        init = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        assignment = next(n for n in ast.walk(init) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Attribute) and t.attr == "tbt_slo_observability_enabled"
                                  for t in n.targets))
        expression = compile(ast.Expression(body=assignment.value), SCHEDULER, "eval")
        scheduler = NS(enable_tbt_decode_scheduling=False)
        for enabled, policy, expected in (("1", "tbt_slo_predictive", True),
                                           ("0", "tbt_slo_predictive", False),
                                           ("1", "ewma_best_fit", False),
                                           ("1", "tbt_slo_feedback", True)):
            with self.subTest(enabled=enabled, policy=policy), patch.dict(os.environ, {
                    "VLLM_PD_DYNAMIC_SM_ENABLED": enabled,
                    "VLLM_PD_DYNAMIC_SM_POLICY": policy}, clear=True):
                self.assertEqual(eval(expression, {"self": scheduler, "os": os}), expected)

    def engine(self):
        output_state = method(ENGINE, "EngineCore", "_pd_sm_output_tbt_state", {"EngineCore": NS()})
        q = {"decode": deque(), "prefill": deque()}
        s = NS(running=[NS(request_id="d")])
        s.get_tbt_allocation_state = lambda **kw: {
            "decode_request_count": len(kw["decode_request_ids"]), "decode_context_tokens": 900,
            "decode_tbt_slo_ms": 30, "decode_min_tbt_slo_ms": 30,
            "decode_tbt_elapsed_ms": 0, "critical_request_id": "d", "decode_tbt_deadline_at": 11}
        e = NS(pd_sm_allocator=self.c, scheduler=s, pd_dynamic_sm_enforced=True,
               pd_layerwise_handoff_enabled=False, _pd_layerwise_handoff_fence_active=False,
               _tbt_controller_state={}, _pd_sm_transition_last_commit_debug={})
        e.pd_sm_transition = NS(active=False, target_profile_id="", stage="idle", requested_at=0)
        e._scheduler_output_phase = lambda output: output.phase
        e._scheduler_has_pd_phase_work = lambda scheduler, phase: True
        e._pd_capacity_retry_free_blocks = lambda: 100
        e._pd_capacity_retry_gate = NS(is_blocked=lambda *a: False, blocking_cause=lambda *a: "")
        e._pd_phase_queue_total_depth = lambda queues: sum(map(len, queues.values()))
        e._pd_sm_output_tbt_state = output_state
        e._pd_sm_output_tokens = lambda o: (1, 0) if o.phase == "decode" else (0, 2048)
        e._request_pd_sm_transition = lambda *a, **kw: None
        e._maybe_request_pd_layerwise_handoff = lambda **kw: None
        return e, q

    def test_actual_engine_no_new_prefill_output_sees_queued_ring(self):
        e, q = self.engine()
        ring = NS(pd_prefill_mode="ring", request_debug={}, num_scheduled_tokens={"p": 2048})
        q["prefill"].append((Future(), ring, Future()))
        assign = method(ENGINE, "EngineCore", "_assign_pd_sm_plan")
        p = assign(e, q)
        self.assertEqual(p.target_profile.decode_sms, 38)
        self.assertEqual(p.tbt_profile_prefill_mode, "ring")
        self.assertEqual(p.profile.decode_sms, 30)

    def test_private_ring_authority_overrides_native_label(self):
        e, q = self.engine()
        output = NS(pd_prefill_mode="native", request_debug={},
                    kv_retention_layerwise_transfer=NS(state="released"))
        q["prefill"].append((Future(), output, Future()))
        p = method(ENGINE, "EngineCore", "_assign_pd_sm_plan")(e, q)
        self.assertEqual(p.tbt_profile_prefill_mode, "ring")

    def test_native_reload_is_not_ring_just_because_a_dram_mirror_exists(self):
        e, q = self.engine()
        output = NS(pd_prefill_mode="native", request_debug={"p": {
            "phase": "prefill", "prefill_tokens": 2048, "virtual_context_request_dram_pages": 100}})
        q["prefill"].append((Future(), output, Future()))
        p = method(ENGINE, "EngineCore", "_assign_pd_sm_plan")(e, q)
        self.assertEqual(p.tbt_profile_prefill_mode, "native")
        self.assertEqual(p.target_profile.decode_sms, 30)

    def test_actual_control_hook_does_not_require_prefill_submission_or_handoff_flag(self):
        calls = []
        e = NS(pd_sm_allocator=self.c, scheduler=NS(),
               _scheduler_has_pd_phase_work=lambda *a: True,
               _assign_pd_sm_plan=lambda *a, **kw: calls.append(kw))
        method(ENGINE, "EngineCore", "_poll_pd_layerwise_handoff")(e, {"decode": [Future()], "prefill": []})
        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["force_prefill_ready"])

    def test_actual_wait_consumes_timer_without_progress_or_audit(self):
        self.c.arm_tbt_deadline(request_id="d", deadline_at=11)
        calls = []
        def wait(event, ready, **kw):
            calls.append(kw["timeout_s"])
            return False
        fn = method(ENGINE, "EngineCore", "_wait_for_pd_progress", {
            "wait_for_pd_engine_progress": wait, "PD_EXTERNAL_PROGRESS_FALLBACK_WAIT_S": .01})
        e = NS(pd_sm_allocator=self.c, _pd_input_pending=lambda: False,
               _pd_recoverable_future_ready=lambda q: False, pd_future_liveness_audit_enabled=False,
               _pd_engine_progress_event=None, _pd_progress_generation_at_step_start=1,
               _write_pd_phase_queue_trace=lambda *a: calls.append("deadline"))
        future = Future()
        with patch.object(time, "monotonic", return_value=11):
            self.assertFalse(fn(e, {}, selected_future=future))
            self.assertFalse(fn(e, {}, selected_future=future))
        self.assertEqual(calls, [0.0, "deadline", None])
        self.assertFalse(future.done())

    def test_predictive_recommendation_uses_existing_drain_authority(self):
        transition_cls = runpy.run_path(str(ROOT / "vendor_v8/vllm/v1/engine/pd_sm_transition.py"))["PDSMDrainTransition"]
        transition = transition_cls()
        plan = self.plan(decode_context_tokens=996, decode_inflight=True)
        engine = NS(pd_dynamic_sm_enforced=True, pd_sm_allocator=self.c,
                    pd_sm_transition=transition, _pd_layerwise_handoff_fence_active=False)
        method(ENGINE, "EngineCore", "_request_pd_sm_transition")(engine, plan)
        self.assertTrue(transition.active)
        self.assertTrue(transition.urgent)
        self.assertTrue(transition.allows_phase("decode"))
        self.assertFalse(transition.allows_phase("prefill"))
        self.assertEqual(self.c.epoch, 0)
        self.assertFalse(transition.advance(old_decode_futures=1, old_prefill_futures=1, global_queue_depth=2))
        self.assertFalse(transition.advance(old_decode_futures=1, old_prefill_futures=0, global_queue_depth=1))
        self.assertFalse(transition.allows_phase("decode"))
        self.assertTrue(transition.advance(old_decode_futures=0, old_prefill_futures=0, global_queue_depth=0))
        self.c.commit_profile_at_boundary(transition.target_profile_id, expected_epoch=0)
        self.assertEqual(self.c.current_profile.decode_sms, 34)
        self.assertEqual(self.c.epoch, 1)

    def test_exact_launcher_preflight_validates_without_cuda_import(self):
        script = (ROOT / "run_vendor_v8_short_validation.sh").read_text()
        probe = script.split('"${VLLM_VENDOR_DIR}/vllm/v1/engine/pd_tbt_profile.py" <<\'PY\'\n', 1)[1].split("\nPY", 1)[0]
        args = [sys.executable, "-c", probe, "40", "2", "10", str(self.path),
                str(ROOT / "vendor_v8/vllm/v1/engine/pd_tbt_profile.py")]
        result = subprocess.run(args, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        args[3] = "0"
        result = subprocess.run(args, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("uncalibrated/invalid", result.stderr)


if __name__ == "__main__":
    unittest.main()
