#!/usr/bin/env python3
"""Execute real Scheduler methods with CPU allocator/provider doubles.

Tests the WAITING reclaim protocol, not CUDA numeric correctness/performance.
"""
from __future__ import annotations

import ast
import itertools
import unittest
from collections import defaultdict
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_exact_phase_cleanup as C
import test_native_reload_admission as N
import test_decode_reload_physical_budget as D
import test_phase_waiting_progress as P
import test_scheduled_prefill_fit_residency as R


POLICY = R.MODULE
ENABLED = False
METHODS = {
    "_configure_isolation_waiting_decode_reclaim",
    "_isolation_waiting_decode_capacity", "_progress_isolation_waiting_decode_reclaim",
    "_isolation_waiting_decode_snapshot_plan", "_virtual_context_waiting_decode_service_candidate",
    "_commit_isolation_waiting_decode_retirements",
    "_virtual_context_waiting_decode_service_reserve", "_virtual_context_waiting_decode_ready",
    "_reclaim_prefill_victims_until_hbm_target", "_maximum_immediate_prefill_reclaim_blocks",
    "_immediate_prefill_reclaim_state", "_restrict_reclaim_candidates_to_preparation",
    "_scheduled_prefill_fit_preparation_status", "_consume_scheduled_prefill_fit_preparation",
    "_active_virtual_context_prefill_requests",
    "_debug_has_prefill_remaining", "_pd_prefill_lane_remaining_tokens",
    "_pd_prefill_lane_boundary_tokens",
}
S = C.adapters(N.PATH, "Scheduler", METHODS, dict(
    RequestStatus=N.STATUS, _VIRTUAL_CONTEXT_TIERED_PROVIDER_STATES=N.STATES,
    ResidencyVictimPolicy=POLICY.ResidencyVictimPolicy,
    ResidencyVictimCandidate=POLICY.ResidencyVictimCandidate,
    select_best_fit_whole_request_victims=POLICY.select_best_fit_whole_request_victims,
    virtual_context_residency_victim_context_tokens=max, itertools=itertools,
    DecodeHandoffReclaimInfeasibleObservation=NS, ReleasedPrefillVictim=NS,
    TieredAllocatorInvariantError=RuntimeError,
    _read_bool_env=lambda name: ENABLED, logger=NS(info=lambda *args: None),
))
S._virtual_context_waiting_decode_ready = staticmethod(S._virtual_context_waiting_decode_ready)
S._restrict_reclaim_candidates_to_preparation = staticmethod(S._restrict_reclaim_candidates_to_preparation)


def row(rid, tokens=32, prompt=128, *, decode=False):
    r = P.row(rid, decode=decode)
    r.num_prompt_tokens = prompt
    r.num_computed_tokens = 0 if decode else tokens
    r.virtual_context_deferred_logical_computed_tokens = tokens if decode else 0
    r.finished = False
    r.is_finished = lambda: r.finished
    return r


def harness(*, free=1, counts=(2, 2), ready=True):
    s = S()
    cold = row("D", 64, 64, decode=True)
    prefills = [row(f"P{i}", count * 16) for i, count in enumerate(counts)]
    s.waiting, s.running = P.Queue([cold]), list(prefills)
    s.requests = {r.request_id: r for r in [cold, *prefills]}
    s.free = free
    s.resident = {r.request_id: c for r, c in zip(prefills, counts)}
    s.copy_ready = set(s.resident) if ready else set()
    s._pd_pending_request_ids = set()
    s.blocks_next_schedule = lambda r, phase: r.request_id in s._pd_pending_request_ids
    s.virtual_context_enabled = s.pd_decoupled_no_mix = True
    s.virtual_context_reload_to_hbm = s.isolation_prefill_batch_kv_budget = True
    s.isolation_required_d2h_progress = s.isolation_waiting_decode_reclaim = True
    s.isolation_phase_waiting_scan = s.isolation_waiting_decode_fcfs_scope = True
    s.virtual_context_residency_victim_policy = POLICY.ResidencyVictimPolicy.SCHEDULED_PREFILL_FIT
    s.virtual_context_reload_to_hbm_min_free_blocks = 0
    s.virtual_context_staging_reserved_blocks = 0
    s.virtual_context_ring_min_free_blocks = 0
    s.virtual_context_decode_hotset_reserve_blocks = 0
    s.virtual_context_scheduled_prefill_fit_headroom_blocks = 82
    s._isolation_prefill_batch_budget_debug = {}
    s._isolation_waiting_decode_reclaim_debug = {}
    s._isolation_waiting_decode_retirements = set()
    s.is_encoder_decoder = False
    s.block_size = 16
    s.isolation_logical_release_readiness = False
    s.virtual_context_pending_victim_intent_enabled = True
    s._virtual_context_pending_victim_intent = None
    s._native_decode_handoff_reclaim_infeasible_cache = {}
    s._scheduled_prefill_fit_preparation = None
    s._scheduled_prefill_fit_last_release_blocks = 0
    for suffix in ("feasibility_check_count", "infeasible_count", "avoided_partial_blocks",
                   "infeasible_cache_hit_count"):
        setattr(s, "_native_decode_handoff_reclaim_" + suffix, 0)
    for suffix in ("cancelled_count", "target_mismatch_count", "reclaim_consumed_count",
                   "consumed_victim_count"):
        setattr(s, "_scheduled_prefill_fit_preparation_" + suffix, 0)
    s._scheduled_prefill_fit_unprepared_reclaim_count = 0
    s.scans = s.release_calls = 0
    s.snapshot_calls = []

    def snapshot_pages(**kw):
        s.snapshot_calls.append(kw)
        return tuple(range(min(s.resident.get(kw["request_id"], 0), kw["max_pages"])))

    s.kv_cache_manager = NS(
        block_pool=NS(get_num_free_blocks=lambda: s.free),
        estimate_num_blocks_to_allocate=lambda r, n, **kw: (64 + n + 15) // 16,
        get_request_resident_block_count=lambda rid: s.resident.get(rid, 0),
        get_block_ids=lambda rid: [tuple(range(1, s.resident.get(rid, 0) + 1))],
        plan_virtual_context_hbm_snapshot_pages=snapshot_pages,
        plan_virtual_context_hbm_release_candidates=lambda **kw: [
            NS(request_id=rid, kv_cache_group_id=0, logical_token_page=p)
            for rid in kw["request_ids"] if rid in s.copy_ready
            for p in range(s.resident.get(rid, 0))],
    )

    def candidates(*, required_blocks, excluded_request_ids):
        s.scans += 1
        return [(r, POLICY.ResidencyVictimCandidate(
            request_id=r.request_id, arrival_time=r.arrival_time,
            prompt_tokens=r.num_prompt_tokens, logical_context_tokens=r.num_computed_tokens,
            estimated_reclaimable_blocks=s.resident[r.request_id]))
            for r in s._active_virtual_context_prefill_requests()
            if r.request_id not in excluded_request_ids and r.request_id in s.copy_ready
            and s.resident.get(r.request_id, 0) > 0]

    def defer(r, **kw):
        assert r.request_id not in s._pd_pending_request_ids
        assert r.request_id in s.copy_ready
        s.release_calls += 1
        logical = max(r.num_computed_tokens, r.virtual_context_deferred_logical_computed_tokens)
        s.free += s.resident.pop(r.request_id)
        r.virtual_context_deferred_logical_computed_tokens = logical
        r.num_computed_tokens, r.status = 0, N.STATUS.PREEMPTED
        if r not in s.waiting:
            s.waiting.prepend_request(r)
        s._consume_scheduled_prefill_fit_preparation({r.request_id})
        return True, "tiered_provider_deferred", logical

    s._ordered_virtual_context_prefill_reclaim_candidates = candidates
    s._try_defer_running_prefill_for_virtual_context = defer
    return s, cold, prefills


def install_preparation(s, plan):
    target, victims, logical, chunk, raw, reserve, free, deficit, pages = plan
    s._scheduled_prefill_fit_preparation = NS(
        target_request_id=target.request_id, victim_request_ids=tuple(r.request_id for r in victims),
        projected_logical_tokens=logical, next_chunk_tokens=chunk,
        predicted_deficit_blocks=deficit, snapshot_pages=pages, target_self_victim=False)


class WaitingReclaimTests(unittest.TestCase):
    def test_retirement_survives_empty_output_and_transfers_exactly_once(self):
        s, _, _ = harness()
        s._progress_isolation_waiting_decode_reclaim()
        ids = s._isolation_waiting_decode_retirements
        self.assertEqual(ids, {"P0", "P1"})
        output = NS(total_num_scheduled_tokens=0, finished_req_ids=set(),
                    residency_deferred_req_ids={"P0", "P1"}, num_scheduled_tokens={})
        s._commit_isolation_waiting_decode_retirements(output)
        self.assertEqual(ids, {"P0", "P1"})
        # Later Decode output carries retirement after an empty Prefill pass.
        output.total_num_scheduled_tokens = 1
        output.num_scheduled_tokens = {"D": 1}
        s._commit_isolation_waiting_decode_retirements(output)
        self.assertEqual(ids, set())
        self.assertEqual(output.residency_deferred_req_ids, {"P0", "P1"})
        s._commit_isolation_waiting_decode_retirements(output)
        self.assertFalse(ids)

    def test_retirement_abort_and_same_step_rehydration(self):
        s, _, _ = harness()
        ids = s._isolation_waiting_decode_retirements
        ids.update(("P0", "P1"))
        output = NS(total_num_scheduled_tokens=1, finished_req_ids={"P0"},
                    residency_deferred_req_ids=set(), num_scheduled_tokens={"P1": 1})
        s._commit_isolation_waiting_decode_retirements(output)
        self.assertFalse(ids)
        ids.add("P0")
        output.total_num_scheduled_tokens = 0
        output.num_scheduled_tokens = {}
        s._commit_isolation_waiting_decode_retirements(output)  # finished-only accepted
        self.assertFalse(ids)

    def test_cold_decode_preparation_survives_prefill_target_selection(self):
        s, cold, _ = harness(ready=False)
        self.assertTrue(s._debug_has_prefill_remaining(cold))  # physical reset!
        self.assertTrue(s._virtual_context_waiting_decode_ready(cold))
        assignment = next(n for n in N.METHODS["schedule"].body
                          if isinstance(n, ast.Assign) and ast.unparse(n.targets[0])
                          == "scheduled_prefill_fit_decode_handoff_prepared")
        code = compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
                       N.PATH, "exec")
        for enabled in (False, True):
            s.isolation_waiting_decode_reclaim = enabled
            scope = dict(self=s, existing_fit_target=cold)
            exec(code, scope)
            self.assertEqual(scope["scheduled_prefill_fit_decode_handoff_prepared"], enabled)

    def test_complete_transaction_before_queue_scan_and_receipts(self):
        s, cold, victims = harness()
        wait, released = s._progress_isolation_waiting_decode_reclaim()
        self.assertFalse(wait)
        self.assertEqual([r.request_id for r in released], ["P0", "P1"])
        self.assertEqual(s.free, 5)
        self.assertEqual(s.running, [])
        self.assertIsNot(s.waiting.peek_request(), cold)  # reclaim prepends!
        scan = P.setup(list(s.waiting))
        seen, _ = P.scan(scan, "decode", deferred=[v.request_id for v in released])
        self.assertEqual(seen, ["D"])

    def test_impossible_aggregate_is_cached_without_copy_scan_or_mutation(self):
        s, _, _ = harness(counts=(2,))
        for _ in range(3):
            self.assertEqual(s._progress_isolation_waiting_decode_reclaim(), (frozenset(), []))
        self.assertEqual((s.free, s.scans, s.release_calls), (1, 0, 0))
        self.assertEqual(s._native_decode_handoff_reclaim_infeasible_cache_hit_count, 2)

    def test_missing_copy_cannot_commit_a_partial_victim_set(self):
        s, _, _ = harness()
        s.copy_ready.remove("P1")
        self.assertFalse(s._progress_isolation_waiting_decode_reclaim()[1])
        self.assertEqual((s.free, s.release_calls), (1, 0))
        self.assertEqual(s._isolation_waiting_decode_reclaim_debug[
            "isolation_waiting_decode_reclaim_reason"], "aggregate_copy_ready_reclaim_insufficient")

    def test_future_owned_victim_excluded_even_with_pending_intent_enabled(self):
        s, _, _ = harness()
        s._pd_pending_request_ids.add("P0")
        self.assertFalse(s._progress_isolation_waiting_decode_reclaim()[1])
        self.assertEqual((s.free, s.release_calls), (1, 0))
        s._pd_pending_request_ids.clear()
        self.assertEqual(len(s._progress_isolation_waiting_decode_reclaim()[1]), 2)

    def test_prefill_producer_prepares_projected_current_rows(self):
        s, _, prefills = harness(ready=False)
        # Slot allocation occurs before proactive prediction. The new third
        # page is written by this Prefill Future, not ready for release yet.
        s.resident["P0"] = 3
        plan = s._isolation_waiting_decode_snapshot_plan({"P0": 16})
        self.assertIsNotNone(plan)
        self.assertEqual({v.request_id for v in plan[1]}, {"P0", "P1"})
        self.assertEqual(len(plan[-1]), 5)
        p0 = next(c for c in s.snapshot_calls if c["request_id"] == "P0")
        self.assertEqual(p0["logical_tokens"], 48)
        self.assertEqual((s.free, s.release_calls), (1, 0))

    def test_copy_and_model_completion_both_required_then_consume_once(self):
        s, _, _ = harness(ready=False)
        plan = s._isolation_waiting_decode_snapshot_plan({"P0": 1})
        install_preparation(s, plan)
        s._pd_pending_request_ids.add("P0")
        for copy_done in (False, True):
            if copy_done:
                s.copy_ready.update(("P0", "P1"))
            wait, released = s._progress_isolation_waiting_decode_reclaim()
            self.assertEqual(wait, frozenset({"P0", "P1"}))
            self.assertFalse(released)
        s._pd_pending_request_ids.clear()
        self.assertEqual(len(s._progress_isolation_waiting_decode_reclaim()[1]), 2)
        self.assertIsNone(s._scheduled_prefill_fit_preparation)
        self.assertEqual(s._scheduled_prefill_fit_preparation_reclaim_consumed_count, 1)
        self.assertFalse(s._progress_isolation_waiting_decode_reclaim()[1])

    def test_copy_completion_without_free_block_change_authorizes_retry(self):
        s, _, _ = harness(ready=False)
        install_preparation(s, s._isolation_waiting_decode_snapshot_plan({"P0": 1}))
        self.assertTrue(s._progress_isolation_waiting_decode_reclaim()[0])
        self.assertEqual(s.free, 1)
        s.copy_ready.update(("P0", "P1"))  # no allocator free yet
        self.assertEqual(len(s._progress_isolation_waiting_decode_reclaim()[1]), 2)

    def test_pending_decode_or_abort_cannot_retain_a_claim(self):
        s, cold, _ = harness(ready=False)
        install_preparation(s, s._isolation_waiting_decode_snapshot_plan({"P0": 1}))
        s._pd_pending_request_ids.add("D")
        self.assertEqual(s._progress_isolation_waiting_decode_reclaim(), (frozenset(), []))
        s._pd_pending_request_ids.clear()
        cold.finished = True
        s.waiting.remove_request(cold)
        s.requests.pop("D")
        self.assertEqual(s._progress_isolation_waiting_decode_reclaim(), (frozenset(), []))
        self.assertIsNone(s._scheduled_prefill_fit_preparation)
        self.assertEqual(s.resident, {"P0": 2, "P1": 2})

    def test_fitting_capacity_cancels_cpu_barrier_not_storage(self):
        s, _, _ = harness(ready=False)
        install_preparation(s, s._isolation_waiting_decode_snapshot_plan({"P0": 1}))
        s.free = 5
        self.assertEqual(s._progress_isolation_waiting_decode_reclaim(), (frozenset(), []))
        self.assertIsNone(s._scheduled_prefill_fit_preparation)
        self.assertEqual((s.release_calls, s.resident), (0, {"P0": 2, "P1": 2}))

    def test_finishing_row_and_pending_peers_cannot_be_future_victims(self):
        s, _, prefills = harness(ready=False)
        prefills[0].num_prompt_tokens = 48
        self.assertIsNone(s._isolation_waiting_decode_snapshot_plan({"P0": 16}))
        prefills[0].num_prompt_tokens = 128
        s._pd_pending_request_ids.add("P1")
        self.assertIsNone(s._isolation_waiting_decode_snapshot_plan({"P0": 16}))
        self.assertEqual(s.snapshot_calls, [])

    def test_no_copy_plan_without_real_prefill_carrier(self):
        s, _, _ = harness(ready=False)
        for tokens in ({}, {"P0": 0}, {"other-decode": 1}):
            self.assertIsNone(s._isolation_waiting_decode_snapshot_plan(tokens))
        self.assertIsNone(s._scheduled_prefill_fit_preparation)

    def test_existing_barrier_not_replaced_on_other_work(self):
        s, _, _ = harness(ready=False)
        install_preparation(s, s._isolation_waiting_decode_snapshot_plan({"P0": 1}))
        prior = s._scheduled_prefill_fit_preparation
        count = len(s.snapshot_calls)
        plan = s._isolation_waiting_decode_snapshot_plan({"other-prefill": 16})
        self.assertFalse(plan[1])
        self.assertFalse(plan[-1])
        self.assertIs(prior, s._scheduled_prefill_fit_preparation)
        self.assertEqual(len(s.snapshot_calls), count)

    def test_waiting_reserve_not_running_headroom_after_reclaim(self):
        s, _, _ = harness(free=5)
        self.assertEqual(s._virtual_context_waiting_decode_service_reserve(), 5)
        s.isolation_waiting_decode_reclaim = False
        self.assertEqual(s._virtual_context_waiting_decode_service_reserve(), 0)

    def test_actual_allocator_counts_holes_not_layers_and_includes_next_token(self):
        for holes, expected in ((None, 3), ((False, True), 2), ((False, False), 1)):
            s, r, _, _ = D.harness(holes=holes, waiting=True)
            s._isolation_waiting_decode_capacity = S._isolation_waiting_decode_capacity.__get__(s)
            self.assertEqual(s._isolation_waiting_decode_capacity(r), (expected, expected))

    def test_waiting_scan_pauses_only_victim_and_keeps_order(self):
        rows = [P.row("P"), P.row("D", decode=True), P.row("Q")]
        s = P.setup(rows)
        s._pd_forced_schedule_phase = "prefill"
        seen, state = P.scan(s, "prefill", reclaim_wait={"P"})
        self.assertEqual(seen, ["Q"])
        self.assertEqual([r.request_id for r in s.waiting], ["P", "D"])
        self.assertEqual(state["prefill_defer_reason_counts"]["scheduled_prefill_fit_d2h_wait"], 1)

    def test_partial_transaction_failure_remains_fatal(self):
        s, _, _ = harness()
        defer = s._try_defer_running_prefill_for_virtual_context
        def fail_second(r, **kw):
            if r.request_id == "P1":
                return False, "copy_not_ready", 0
            return defer(r, **kw)
        s._try_defer_running_prefill_for_virtual_context = fail_second
        with self.assertRaisesRegex(RuntimeError, "partially committed"):
            s._progress_isolation_waiting_decode_reclaim()

    def test_startup_default_off_and_incomplete_copy_producer_rejected(self):
        s, _, _ = harness()
        with patch.dict(S._configure_isolation_waiting_decode_reclaim.__globals__,
                        _read_bool_env=lambda _: False):
            s._configure_isolation_waiting_decode_reclaim()
        self.assertFalse(s.isolation_waiting_decode_reclaim)
        for field in ("isolation_required_d2h_progress", "isolation_prefill_batch_kv_budget",
                      "virtual_context_reload_to_hbm", "isolation_phase_waiting_scan",
                      "isolation_waiting_decode_fcfs_scope"):
            s, _, _ = harness()
            setattr(s, field, False)
            with patch.dict(S._configure_isolation_waiting_decode_reclaim.__globals__,
                            _read_bool_env=lambda _: True):
                with self.assertRaises(ValueError):
                    s._configure_isolation_waiting_decode_reclaim()

    def test_native_or_hbm_only_control_ignores_proposal_flag(self):
        for field in ("pd_decoupled_no_mix", "virtual_context_enabled"):
            s, _, _ = harness()
            setattr(s, field, False)
            with patch.dict(S._configure_isolation_waiting_decode_reclaim.__globals__,
                            _read_bool_env=lambda _: True):
                s._configure_isolation_waiting_decode_reclaim()
            self.assertFalse(s.isolation_waiting_decode_reclaim)

    def test_reclaim_hook_precedes_scans_and_output_retirement_not_just_debug(self):
        source = ast.unparse(N.METHODS["schedule"])
        self.assertLess(source.index("self._progress_isolation_waiting_decode_reclaim()"),
                        source.index("while req_index < len(self.running)"))
        receipt_loop = next(n for n in N.METHODS["schedule"].body if isinstance(n, ast.For)
                            and ast.unparse(n.iter) == "isolation_waiting_decode_released")
        namespace = dict(isolation_waiting_decode_released=[NS(
            request_id="P", deferred_tokens=32, reclaimed_blocks=2)],
            virtual_context_deferred_req_ids=set(), virtual_context_decode_handoff_reclaim_victim_ids=[])
        for name in ("prefill_deferred_by_residency_budget", "preempt_reset_avoided_tokens",
                     "prefill_dram_resident_tokens", "virtual_context_residency_pressure_release_success_count",
                     "virtual_context_residency_pressure_release_tokens", "scheduled_prefill_fit_whole_request_victim_count",
                     "scheduled_prefill_fit_whole_request_reclaimed_blocks", "virtual_context_decode_handoff_reclaim_success_count",
                     "virtual_context_decode_handoff_reclaimed_blocks"):
            namespace[name] = 0
        exec(compile(ast.fix_missing_locations(ast.Module(body=[receipt_loop], type_ignores=[])),
                     N.PATH, "exec"), namespace)
        self.assertEqual(namespace["virtual_context_deferred_req_ids"], {"P"})
        self.assertIn("residency_deferred_req_ids=residency_deferred_req_ids_for_output", source)
        self.assertIn("getattr(self, '_isolation_waiting_decode_retirements', ())", source)
        self.assertIn("self._commit_isolation_waiting_decode_retirements(scheduler_output)", source)


if __name__ == "__main__":
    unittest.main()
