#!/usr/bin/env python3
"""Real Engine/allocator/parked-Scheduler adapters; CPU, not CUDA E2E."""
from __future__ import annotations

import time
import unittest
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace as NS
from typing import cast

import test_exact_phase_cleanup as C
import test_p1_parked_prefill as P
import test_tbt_feedback_sm_allocator as A


PLAN = C.adapters("vendor_v8/vllm/v1/engine/core.py", "EngineCore", {
    "_assign_pd_sm_plan", "_request_pd_sm_transition",
    "_advance_pd_sm_transition", "_pd_layerwise_transition_boundary_ready",
    "_pd_sm_output_tbt_state", "_submit_pd_scheduler_lanes",
    "_kv_retention_parked_has_precedence_candidate",
}, {"time": time, "cast": cast, "ModelRunnerOutput": object})
PLAN._pd_sm_output_tbt_state = staticmethod(PLAN._pd_sm_output_tbt_state)


class Engine(PLAN, P.ENGINE):
    pass


def scenario(*, initial="d0_p40", width=1, cap=10):
    e = Engine()
    e.__dict__.update(P.engine().__dict__)
    ids = tuple(f"long-{i}" for i in range(width))
    output = P.out(ids)
    output.pd_sm_profile_id, output.pd_sm_profile_epoch = "d0_p40", 0
    output, registry, binding, boundary, _ = P.work(output)
    output.phase = "prefill"
    output.iteration_debug = {}
    output.request_debug = {}
    e.scheduler = P.scheduler(output, cap=cap)
    e.scheduler.park_kv_retention_prefill(output)
    registry.suspend(binding, boundary.next_layer)
    registry.park(binding)
    e._ttft_prefill_suspended = {"parked:1": (boundary, output)}
    output.kv_retention_p1_parked = True
    e._ttft_prefill_suspended_request_ids = lambda: set(ids)
    e._kv_retention_parked_has_precedence_candidate = lambda *args: False
    e.scheduler.has_requests = lambda: True
    e._scheduler_has_pd_phase_work = lambda scheduler, phase: phase == "decode"
    drow = P.row("short")
    e.scheduler.requests["short"] = drow
    e.scheduler.running = [drow]
    profiles = A.dense_pd_sm_profiles(40, 2)
    e.pd_sm_allocator = A.PDComputeAllocationController(A._config(
        initial_profile=next(p for p in profiles if p.profile_id == initial)))
    # A non-startup source was committed after the original Prefill paused.
    if initial != "d0_p40":
        e.pd_sm_allocator.epoch = 1
    e.pd_dynamic_sm_enabled = e.pd_dynamic_sm_enforced = True
    e.pd_layerwise_handoff_enabled = False
    e._pd_layerwise_handoff_fence_active = False
    e.pd_sm_transition = A.PDSMDrainTransition()
    e._pd_capacity_retry_free_blocks = lambda: 0
    e._pd_capacity_retry_gate = NS(
        is_blocked=lambda *args: False, blocking_cause=lambda *args: "",
        notify_state_change=lambda *args: None)
    e._scheduler_output_phase = lambda out: out.phase
    e._pd_sm_output_tokens = lambda out: (
        (out.total_num_scheduled_tokens, 0) if out.phase == "decode"
        else (0, out.total_num_scheduled_tokens))
    e._maybe_request_pd_layerwise_handoff = lambda **kw: None
    e._tbt_controller_state = {}
    e._pd_sm_transition_last_commit_debug = {}
    for name in (
        "_pd_sm_transition_commit_count", "_pd_sm_transition_wait_ms_total",
        "_pd_sm_transition_prefill_drain_ms_total", "_pd_sm_transition_all_fence_ms_total",
        "_pd_layerwise_handoff_fence_blocked_decode_checks",
        "_pd_layerwise_handoff_fence_blocked_prefill_checks",
        "_pd_layerwise_handoff_source_decode_admission_checks",
        "_pd_layerwise_handoff_source_decode_admission_allowed",
        "_pd_layerwise_handoff_decode_inflight_defer_checks",
        "_pd_layerwise_handoff_decode_backlog_defer_checks",
        "_pd_layerwise_handoff_prefill_only_retarget_count",
        "_pd_layerwise_handoff_prefill_only_retarget_miss_count",
        "_ttft_prefill_resume_submit_count",
    ):
        setattr(e, name, 0)
    e._cancel_ttft_prefill_yield_for_output = lambda out: None
    e._kv_retention_bind_prefill = lambda out, resume: registry.resume(out, P.W.epochs(out))
    e._register_pd_model_future_completion = lambda *a, **kw: None
    e.full_iteration_debug_enabled = False
    executed = []
    e.model_executor = NS(execute_model=lambda out, **kw: executed.append((out, kw)) or Future())
    queues = {"prefill": deque(), "decode": deque()}
    e.pd_phase_queues = queues
    return e, queues, output, executed


def decode_output(seq=13):
    return NS(phase="decode", total_num_scheduled_tokens=1,
              num_scheduled_tokens={"short": 1}, pd_prefill_mode="native",
              iteration_debug={}, request_debug={}, pd_phase_submit_seq=seq)


class DemandRegressionTests(unittest.TestCase):
    def test_actual_decode_first_lane_loop_resumes_original_without_extra_schedule(self):
        e, q, original, executed = scenario()
        scheduled = []
        e._try_submit_pd_paired_phase_launch = lambda queues: (False, None, None, False)
        e._pd_phase_queue_can_submit_phase = lambda queues, phase: not queues[phase]
        e._pd_capacity_retry_can_submit = lambda phase: True
        e._pd_phase_queue_pending_request_ids = lambda queues: set(original.num_scheduled_tokens)
        e._pd_phase_queue_pending_request_ids_by_phase = lambda queues: {}
        e._set_scheduler_pd_pending_request_ids = lambda ids: None
        e._set_scheduler_pd_pending_phase_request_ids = lambda ids: None
        e._pd_capacity_retry_gate.clear = lambda phase: None
        def schedule(scheduler, phase):
            scheduled.append(phase)
            self.assertEqual(phase, "decode", "Continuation must not be scheduled again")
            return decode_output()
        def submit(queues, out, *args):
            e._assign_pd_sm_plan(queues, out)
            f = Future()
            queues[out.phase].append((f, out, f))
            return True, None, f
        e._scheduler_schedule_pd_phase = schedule
        e._submit_to_pd_phase_queue = submit  # Keep real SM assignment, no CUDA RPC.
        result = e._submit_pd_scheduler_lanes(q, ("decode", "prefill"), allow_multiple_rounds=False)
        self.assertTrue(result[3])
        self.assertEqual(scheduled, ["decode"])
        self.assertEqual((len(q["decode"]), len(q["prefill"])), (1, 1))
        self.assertIs(executed[0][0], original)
        self.assertEqual(q["decode"][0][1].pd_sm_profile_id, original.pd_sm_profile_id)

    def test_decode_first_assignment_keeps_split_for_eligible_parked_long(self):
        e, q, original, executed = scenario()
        original_claims = dict(original.pd_output_placeholder_owners)
        d = decode_output()
        plan = e._assign_pd_sm_plan(q, d)
        self.assertEqual(plan.profile.profile_id, "d30_p10")
        self.assertEqual(plan.epoch, 1)
        # Resource planning must not mutate/rebind the original parked owner.
        self.assertEqual(original.pd_phase_submit_seq, 3)
        self.assertTrue(original.kv_retention_p1_parked)
        self.assertEqual(executed, [])
        f = Future()
        q["decode"].append((f, d, f))
        self.assertTrue(e._try_resume_ttft_prefill(q, respect_srpt=True))
        self.assertIs(q["prefill"][0][1], original)
        self.assertIs(executed[0][0], original)
        self.assertEqual(original.pd_output_placeholder_owners, original_claims)
        self.assertEqual(original.pd_sm_profile_id, "d30_p10")
        self.assertEqual(original.pd_sm_profile_epoch, d.pd_sm_profile_epoch)
        self.assertFalse(e._try_resume_ttft_prefill(q))
        self.assertEqual(len(executed), 1)

    def test_committed_split_not_undone_by_next_decode_first_submit(self):
        e, q, original, executed = scenario(initial="d40_p0")
        d = decode_output()
        d.pd_sm_profile_id, d.pd_sm_profile_epoch = "d40_p0", 1
        f = Future()
        q["decode"].append((f, d, f))
        self.assertFalse(e._try_resume_ttft_prefill(q, respect_srpt=True))
        self.assertTrue(e.pd_sm_transition.active)
        self.assertEqual(e.pd_sm_transition.target_profile_id, "d30_p10")
        self.assertEqual(e._pd_sm_transition_future_counts(q), (1, 0, 0))
        self.assertFalse(e._advance_pd_sm_transition(q))
        f.set_result(None)
        q["decode"].popleft()  # Simulate normal final source Decode recovery.
        self.assertTrue(e._advance_pd_sm_transition(q))
        d2 = decode_output(14)
        plan = e._assign_pd_sm_plan(q, d2)
        self.assertEqual(plan.profile.profile_id, "d30_p10")
        self.assertEqual(plan.epoch, 2, "Decode must not switch back to d40 epoch3")
        f2 = Future()
        q["decode"].append((f2, d2, f2))
        self.assertTrue(e._try_resume_ttft_prefill(q, respect_srpt=True))
        self.assertFalse(e.pd_sm_transition.active)
        self.assertEqual(len(executed), 1)

    def test_no_suspended_work_keeps_stable_decode_only_epoch(self):
        e, q, _, executed = scenario()
        e._ttft_prefill_suspended.clear()
        for seq in range(16):
            d = decode_output(seq)
            plan = e._assign_pd_sm_plan(q, d)
            self.assertEqual((plan.profile.profile_id, plan.epoch), ("d40_p0", 1))
            self.assertEqual(d.iteration_debug["pd_dynamic_sm_suspended_prefill_ready"], 0)
        self.assertFalse(e._try_resume_ttft_prefill(q))
        self.assertEqual(executed, [])

    def test_sequence_budget_blocks_demand_and_resume_then_completion_releases_it(self):
        e, q, original, executed = scenario(width=2, cap=2)
        self.assertEqual(e._ttft_prefill_resume_candidate(q), (None, "sequence_budget"))
        plan = e._assign_pd_sm_plan(q, decode_output())
        self.assertEqual(plan.profile.profile_id, "d40_p0")
        self.assertFalse(e._try_resume_ttft_prefill(q))
        self.assertEqual(executed, [])
        e.scheduler.running.clear()  # Short terminates and releases one active row.
        self.assertEqual(e._ttft_prefill_resume_candidate(q), ("parked:1", "ready"))
        self.assertTrue(e._try_resume_ttft_prefill(q))
        self.assertIs(executed[0][0], original)

    def test_external_first_token_fence_blocks_without_planner_counter_mutation(self):
        e, q, _, executed = scenario()
        e._pd_ttft_handoff_active = lambda: True
        e._pd_ttft_handoff_request_ids = {"short"}
        d = decode_output()
        plan = e._assign_pd_sm_plan(q, d)
        self.assertEqual(plan.profile.profile_id, "d40_p0")
        self.assertEqual(d.iteration_debug["pd_dynamic_sm_suspended_prefill_reason"],
                         "first_token_handoff")
        self.assertEqual(e._pd_ttft_handoff_blocked_prefill_checks, 0)
        self.assertFalse(e._try_resume_ttft_prefill(q))
        self.assertEqual(e._pd_ttft_handoff_blocked_prefill_checks, 1)
        self.assertEqual(executed, [])

    def test_self_owned_first_token_fence_does_not_deadlock_continuation(self):
        e, q, original, _ = scenario()
        e._pd_ttft_handoff_active = lambda: True
        e._pd_ttft_handoff_request_ids = set(original.num_scheduled_tokens)
        self.assertEqual(e._assign_pd_sm_plan(q, decode_output()).profile.profile_id, "d30_p10")
        self.assertTrue(e._try_resume_ttft_prefill(q))

    def test_owned_global_budget_remains_a_blocker(self):
        e, q, _, _ = scenario()
        q["decode"].extend([object(), object()])
        self.assertEqual(e._ttft_prefill_resume_candidate(q), (None, "global_slots_full"))
        self.assertFalse(e._try_resume_ttft_prefill(q))

    def test_actual_prefill_queue_excludes_parked_prediction_but_retains_inflight_demand(self):
        e, q, _, executed = scenario(initial="d30_p10")
        p = P.out(("other",))
        p.phase, p.request_debug, p.iteration_debug = "prefill", {}, {}
        p.pd_sm_profile_epoch = 1
        f = Future()
        q["prefill"].append((f, p, f))
        d = decode_output()
        plan = e._assign_pd_sm_plan(q, d)
        self.assertEqual(plan.profile.profile_id, "d30_p10")
        self.assertEqual(d.iteration_debug["pd_dynamic_sm_suspended_prefill_reason"], "prefill_inflight")
        self.assertEqual(d.iteration_debug["pd_dynamic_sm_prefill_inflight"], 1)
        self.assertFalse(e._try_resume_ttft_prefill(q))
        self.assertEqual(executed, [])

    def test_pending_copy_holds_epoch_even_without_a_queued_model_future(self):
        e, q, original, executed = scenario()
        copy = Future()
        boundary = e._ttft_prefill_suspended["parked:1"][0]
        e._kv_retention_parked_offload_future = (copy, boundary, original)
        for done in (False, True):
            if done:
                copy.set_result(None)  # Not an ACK until Engine consumes it.
            self.assertEqual(e._ttft_prefill_resume_candidate(q), (None, "offload_pending"))
            plan = e._assign_pd_sm_plan(q, decode_output())
            self.assertEqual((plan.profile.profile_id, plan.epoch), ("d0_p40", 0))
            self.assertTrue(plan.deferred)
            self.assertEqual(e._pd_sm_transition_future_counts(q), (0, 1, 0))
            self.assertFalse(e._advance_pd_sm_transition(q))
            self.assertFalse(e._try_resume_ttft_prefill(q))
        self.assertEqual(executed, [])
        e._kv_retention_parked_offload_future = None  # Normal receipt recovery.
        self.assertTrue(e._advance_pd_sm_transition(q))
        self.assertTrue(e._try_resume_ttft_prefill(q))

    def test_new_admission_capacity_gate_cannot_hide_already_committed_work(self):
        e, q, _, _ = scenario()
        e._pd_capacity_retry_gate.is_blocked = lambda *args: True
        e._pd_capacity_retry_gate.blocking_cause = lambda *args: "prefill_kv_capacity"
        plan = e._assign_pd_sm_plan(q, decode_output())
        self.assertEqual(plan.profile.profile_id, "d30_p10")
        self.assertTrue(e._try_resume_ttft_prefill(q))
        # A genuinely capacity-gated NEW Prefill with no continuation stays gated.
        q["prefill"].clear()
        e._scheduler_has_pd_phase_work = lambda *args: True
        self.assertEqual(e._assign_pd_sm_plan(q, decode_output()).profile.profile_id, "d40_p0")

    def test_normal_srpt_precedence_and_post_admission_fallback_unchanged(self):
        e, q, original, executed = scenario()
        e._kv_retention_parked_has_precedence_candidate = lambda *args: True
        self.assertEqual(e._ttft_prefill_resume_candidate(q, respect_srpt=True),
                         (None, "srpt_precedence"))
        self.assertFalse(e._try_resume_ttft_prefill(q, respect_srpt=True))
        self.assertEqual(executed, [])
        # A shorter newcomer cannot be admitted; existing fallback can run A.
        e._pd_capacity_retry_gate.is_blocked = lambda *args: True
        self.assertEqual(e._assign_pd_sm_plan(q, decode_output()).profile.profile_id, "d30_p10")
        self.assertTrue(e._try_resume_ttft_prefill(q))
        self.assertIs(executed[0][0], original)

    def test_real_srpt_rank_compares_outstanding_chunk_and_excludes_pending_rows(self):
        e, q, original, _ = scenario()
        e._kv_retention_parked_has_precedence_candidate = (
            PLAN._kv_retention_parked_has_precedence_candidate.__get__(e))
        e.scheduler.ttft_prefill_chunk_scheduler = NS(
            remaining_prefill_tokens=lambda req: req.remaining)
        for rid, request in e.scheduler.requests.items():
            request.remaining = 100 if rid.startswith("long") else 0
            request.arrival_time = 0
        candidate = P.row("new-short")
        candidate.remaining, candidate.arrival_time = 102, 1
        e.scheduler.requests[candidate.request_id] = candidate
        pending = set(original.num_scheduled_tokens)
        e._pd_phase_queue_pending_request_ids = lambda queues: pending
        # 102 < (100 + outstanding4), even though 102 > optimistic remainder100.
        self.assertEqual(e._ttft_prefill_resume_candidate(q, respect_srpt=True),
                         (None, "srpt_precedence"))
        pending.add(candidate.request_id)
        self.assertEqual(e._ttft_prefill_resume_candidate(q, respect_srpt=True),
                         ("parked:1", "ready"))

    def test_whole_cancel_is_not_runnable_and_planner_does_no_cleanup(self):
        e, q, original, executed = scenario()
        e._ttft_prefill_deferred_abort_ids = set(original.num_scheduled_tokens)
        plan = e._assign_pd_sm_plan(q, decode_output())
        self.assertEqual(plan.profile.profile_id, "d40_p0")
        self.assertEqual(e._ttft_prefill_resume_candidate(q), (None, "cancel_cleanup"))
        self.assertIn("parked:1", e._ttft_prefill_suspended)
        self.assertEqual(executed, [])

    def test_partial_abort_retains_original_whole_batch(self):
        e, q, original, executed = scenario(width=2)
        e._ttft_prefill_deferred_abort_ids = {"long-0"}
        self.assertEqual(e._assign_pd_sm_plan(q, decode_output()).profile.profile_id, "d30_p10")
        self.assertTrue(e._try_resume_ttft_prefill(q))
        self.assertEqual(set(executed[0][0].num_scheduled_tokens), {"long-0", "long-1"})
        self.assertIs(executed[0][0], original)

    def test_planning_defers_missing_owner_cleanup_and_resume_fails_on_partial_loss(self):
        e, q, original, executed = scenario(width=2)
        del e.scheduler.requests["long-0"]
        self.assertEqual(e._ttft_prefill_resume_candidate(q), (None, "owner_cleanup"))
        self.assertEqual(e._assign_pd_sm_plan(q, decode_output()).profile.profile_id, "d40_p0")
        self.assertIn("parked:1", e._ttft_prefill_suspended)
        e._cancel_ttft_prefill_worker_slot = lambda *args: None
        e._discard_ttft_prefill_deferred_aborts = lambda *args: None
        with self.assertRaisesRegex(RuntimeError, "lost only part"):
            e._try_resume_ttft_prefill(q)
        self.assertEqual(executed, [])

    def test_reused_request_id_keeps_existing_incarnation_assertion(self):
        e, q, _, executed = scenario()
        e.scheduler.requests["long-0"] = P.row("long-0")
        with self.assertRaisesRegex(RuntimeError, "incarnation"):
            e._assign_pd_sm_plan(q, decode_output())
        self.assertEqual(executed, [])
        self.assertEqual(e.pd_sm_allocator.epoch, 0)

    def test_retained_nonparked_boundary_is_ready_not_physical_sm_drain(self):
        e, q, original, _ = scenario()
        b = e._ttft_prefill_suspended["parked:1"][0]
        b.kv_retention_parked_ticket = None
        self.assertEqual(e._pd_owned_total_depth(q), 1)
        d = decode_output()
        plan = e._assign_pd_sm_plan(q, d)
        self.assertEqual(plan.profile.profile_id, "d30_p10")
        self.assertEqual(d.iteration_debug["pd_dynamic_sm_prefill_inflight"], 0)
        self.assertEqual(d.iteration_debug["pd_dynamic_sm_suspended_prefill_ready"], 1)

    def test_explicit_new_prefill_output_not_combined_with_parked_work(self):
        e, q, _, _ = scenario()
        new = P.out(("new",))
        new.phase, new.iteration_debug, new.request_debug = "prefill", {}, {}
        e._assign_pd_sm_plan(q, new)
        self.assertEqual(new.iteration_debug["pd_dynamic_sm_prefill_work_tokens"], 4)
        self.assertEqual(new.iteration_debug["pd_dynamic_sm_suspended_prefill_reason"], "prefill_selected")
        self.assertEqual(new.iteration_debug["pd_dynamic_sm_suspended_prefill_ready"], 0)

    def test_dispatch_failure_does_not_discard_original_logical_owner(self):
        e, q, original, _ = scenario()
        def fail(*args, **kwargs):
            raise RuntimeError("injected worker dispatch failure")
        e.model_executor.execute_model = fail
        claims = dict(original.pd_output_placeholder_owners)
        with self.assertRaisesRegex(RuntimeError, "dispatch failure"):
            e._try_resume_ttft_prefill(q)
        self.assertIs(e._ttft_prefill_suspended["parked:1"][1], original)
        self.assertEqual(original.pd_output_placeholder_owners, claims)
        self.assertEqual(len(q["prefill"]), 0)

    def test_predictive_policy_sees_private_ring_before_resume_without_changing_owner(self):
        import test_tbt_predictive_sm as T
        fixture = T.PredictiveTests("test_ring_interference_changes_minimum_feasible_sm")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        e, q, original, executed = scenario()
        e.pd_sm_allocator = fixture.c
        e.scheduler.get_tbt_allocation_state = lambda **kw: {
            "decode_request_count": 1, "decode_context_tokens": 900,
            "decode_tbt_slo_ms": 30, "decode_min_tbt_slo_ms": 30,
            "decode_tbt_elapsed_ms": 0}
        original.kv_retention_layerwise_transfer = NS(state="released")
        plan = e._assign_pd_sm_plan(q, decode_output())
        self.assertEqual(plan.tbt_profile_prefill_mode, "ring")
        self.assertEqual(plan.target_profile.decode_sms, 38)
        self.assertEqual(original.pd_prefill_mode, "native")
        self.assertEqual(original.pd_sm_profile_epoch, 0)
        self.assertEqual(executed, [])
        # Capacity-blocked parked work must not create hypothetical Ring load.
        e.scheduler.max_num_running_reqs = 1
        plan = e._assign_pd_sm_plan(q, decode_output())
        self.assertEqual(plan.target_profile.decode_sms, 40)


if __name__ == "__main__":
    unittest.main()
