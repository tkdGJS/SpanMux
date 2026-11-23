#!/usr/bin/env python3
"""Exercise the actual WAITING traversal prefix and native reload admission.

Only allocation/transport outside the tested prefix is doubled. CPU proof of
queue reachability is not proof of the scale2 GPU stall's unique root cause.
"""
from __future__ import annotations

import ast
import copy
import os
import unittest
from collections import defaultdict
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_exact_phase_cleanup as C
import test_native_reload_admission as N
import test_waiting_decode_service_protection as W


# Import the real queue implementation without importing torch/vllm.__init__.
queue_path = C.ROOT / "vendor_v8/vllm/v1/core/sched/request_queue.py"
tree = ast.parse(queue_path.read_text())
tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + [
    n for n in tree.body if not (isinstance(n, ast.ImportFrom)
                                and n.module == "vllm.v1.request")]
queue_scope = {"Request": object}
exec(compile(ast.fix_missing_locations(tree), str(queue_path), "exec"), queue_scope)
Queue = queue_scope["FCFSRequestQueue"]

S = C.adapters(N.PATH, "Scheduler", {
    "has_pd_phase_work", "_debug_running_phase_counts", "_debug_needs_prefill_lane",
    "_debug_has_prefill_remaining", "_pd_prefill_lane_remaining_tokens",
    "_pd_prefill_lane_boundary_tokens", "_virtual_context_waiting_decode_ready",
    "_has_older_active_prefill_request",
}, {"RequestStatus": N.STATUS, "_VIRTUAL_CONTEXT_TIERED_PROVIDER_STATES": N.STATES})
S._virtual_context_waiting_decode_ready = staticmethod(S._virtual_context_waiting_decode_ready)


def row(rid, *, decode=False):
    r = N.request(decode)
    r.request_id = rid
    r.pd_output_placeholder_ledger = NS(active=None)
    r.arrival_time, r.arrival_iteration_id = 2.0, 2
    return r


def setup(rows, *, enabled=True):
    s = S()
    s.waiting, s.running = Queue(rows), []
    s.requests = {r.request_id: r for r in rows}
    s.max_num_running_reqs = 100
    s.pd_decoupled_no_mix = True
    s.isolation_phase_waiting_scan = enabled
    s.isolation_waiting_decode_fcfs_scope = enabled
    s._isolation_waiting_decode_fcfs_scope_logged = False
    s._pd_forced_schedule_phase = "decode"
    s.pd_request_fcfs = False
    s.virtual_context_enabled = True
    s.prev_step_scheduled_req_ids = set()
    s._pd_pending_request_ids = set()
    s._pd_pending_phase_request_ids = {}
    s._ttft_prefill_request_is_selected = lambda r, ids: r.request_id in ids
    s._debug_record_request_defer = lambda *args: None
    return s


def scan(s, phase, *, limit=0, selected=None, admit=None, deferred=(), reclaim_wait=(),
         token_budget=16384):
    loop = next(n for n in ast.walk(N.METHODS["schedule"]) if isinstance(n, ast.While)
                and ast.unparse(n.test) == "self.waiting and token_budget > 0")
    # Preserve all phase/policy/ownership guards through the deferred resume
    # guard; connector/allocator work is exercised separately below.
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.If)
               and "RequestStatus.WAITING_FOR_REMOTE_KVS" in ast.unparse(n.test))
    prefix = copy.deepcopy(loop.body[:end])
    seen = []
    def accept(r):
        if admit is not None and not admit(r):
            return False
        seen.append(r.request_id)
        s.waiting.pop_request()
        return True
    suffix = ast.parse("if not accept(request):\n    break\n"
                       "num_scheduled_tokens[request.request_id] = 1\n"
                       "token_budget -= 1\n").body
    actual_loop = ast.While(test=copy.deepcopy(loop.test), body=prefix + suffix, orelse=[])
    restore = next(n for n in N.METHODS["schedule"].body if isinstance(n, ast.If)
                   and ast.unparse(n.test) == "skipped_waiting_requests")
    scope = dict(self=s, token_budget=token_budget, isolation_phase_request_limit=limit,
                 num_scheduled_tokens={}, isolation_phase_request_limit_hit=False,
                 skipped_waiting_requests=Queue(), pd_decoupled_selected_phase=phase,
                 ttft_prefill_policy_active_this_iteration=selected is not None,
                 ttft_prefill_policy_selected_request_id_set=selected or set(),
                 ttft_prefill_policy_skipped_waiting_reqs=0,
                 scheduled_prefill_fit_preparation_waiting=False,
                 active_decode_during_prefill_phase=False,
                 waiting_break_defer_reason=None, prefill_defer_reason_counts=defaultdict(int),
                 deferred_request_ids=set(), virtual_context_deferred_req_ids=set(deferred),
                 virtual_context_resume_guard_count=0, virtual_context_resume_guard_request_ids=set(),
                 isolation_phase_waiting_skip_count=0, isolation_phase_waiting_skipped_ids=[],
                 isolation_waiting_decode_prefill_fcfs_bypass_count=0,
                 isolation_waiting_decode_prefill_fcfs_first_id="",
                 isolation_waiting_decode_resume_owner_block_count=0,
                 isolation_waiting_decode_reclaim_wait_ids=frozenset(reclaim_wait),
                 accept=accept, RequestStatus=N.STATUS)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[actual_loop, copy.deepcopy(restore)],
                                                      type_ignores=[])), N.PATH, "exec"), scope)
    return seen, scope


class PhaseWaitingTests(unittest.TestCase):
    def test_decode_demand_behind_prefill_is_reachable(self):
        s = setup([row("P1"), row("P2"), row("D", decode=True)])
        self.assertTrue(s.has_pd_phase_work("decode"))
        admitted, state = scan(s, "decode")
        self.assertEqual(admitted, ["D"])
        self.assertEqual([r.request_id for r in s.waiting], ["P1", "P2"])
        self.assertEqual(state["isolation_phase_waiting_skip_count"], 2)

    def test_prefill_progress_when_first_decode_cannot_fit(self):
        s = setup([row("D", decode=True), row("P1"), row("P2")])
        s._pd_forced_schedule_phase = "prefill"
        # The existing Ring/native capacity admission is still authoritative.
        admitted, _ = scan(s, "prefill", admit=lambda r: r.request_id != "P2")
        self.assertEqual(admitted, ["P1"])
        self.assertEqual([r.request_id for r in s.waiting], ["D", "P2"])

    def test_unserviceable_decode_keeps_fcfs_and_real_capacity_guard(self):
        s = setup([row("P"), row("D1", decode=True), row("D2", decode=True)])
        admitted, _ = scan(s, "decode", admit=lambda r: r.request_id != "D1")
        self.assertEqual(admitted, [])
        self.assertEqual([r.request_id for r in s.waiting], ["P", "D1", "D2"])

    def test_no_opposite_phase_requests_lost_or_reversed_across_retries(self):
        rows = [row("P1"), row("P2"), row("D1", decode=True),
                row("P3"), row("D2", decode=True)]
        s = setup(rows)
        for _ in range(3):
            self.assertEqual(scan(s, "decode", admit=lambda r: False)[0], [])
            self.assertEqual(list(s.waiting), rows)
        self.assertEqual(scan(s, "decode")[0], ["D1", "D2"])
        self.assertEqual([r.request_id for r in s.waiting], ["P1", "P2", "P3"])

    def test_same_phase_pending_owner_is_never_readmitted(self):
        s = setup([row("P"), row("D1", decode=True), row("D2", decode=True)])
        s._pd_pending_request_ids = {"D1"}
        self.assertEqual(scan(s, "decode")[0], ["D2"])
        self.assertEqual([r.request_id for r in s.waiting], ["P", "D1"])

    def test_srpt_selection_and_shared_row_budget_still_apply(self):
        s = setup([row("D", decode=True), row("P1"), row("P2")])
        s._pd_forced_schedule_phase = "prefill"
        self.assertEqual(scan(s, "prefill", selected={"P2"}, limit=1)[0], ["P2"])
        self.assertEqual([r.request_id for r in s.waiting], ["D", "P1"])
        s = setup([row("P"), row("D1", decode=True), row("D2", decode=True)])
        self.assertEqual(scan(s, "decode", limit=1)[0], ["D1"])
        s.running = [object()] * 100
        self.assertEqual(scan(s, "decode")[0], [])

    def test_rollback_and_nonforced_lane_preserve_legacy_break(self):
        s = setup([row("P"), row("D", decode=True)], enabled=False)
        self.assertEqual(scan(s, "decode")[0], [])
        s.isolation_phase_waiting_scan = True
        s._pd_forced_schedule_phase = None
        self.assertEqual(scan(s, "decode")[0], [])

    def test_mixed_baseline_path_does_not_filter_either_phase(self):
        s = setup([row("P"), row("D", decode=True)])
        s.pd_decoupled_no_mix = False
        self.assertEqual(scan(s, "mixed")[0], ["P", "D"])

    def test_reachable_decode_uses_real_reload_admission_without_recompute(self):
        d = row("D", decode=True)
        s = setup([row("P"), d])
        self.assertEqual(scan(s, "decode")[0], ["D"])
        reload = N.scheduler(d)
        state = N.run_waiting_candidate(reload, d)
        self.assertGreater(state["num_scheduled_tokens"]["D"], 0)
        self.assertEqual(d.virtual_context_deferred_state, "decode_hbm_reload_pending")
        plans = reload._build_tiered_request_execution_plans(
            state["num_scheduled_tokens"], state["tiered_explicit_execution_plans"])
        self.assertEqual(plans, (("D", 1, "native_reload", 4),))
        self.assertEqual(d.kv_retention_execution_plan_mode, "native_reload")
        self.assertEqual([r.request_id for r in s.waiting], ["P"])

    def test_protected_capacity_cycle_has_a_producer_without_timeout(self):
        # Real reserve: three blocks for D plus 82 headroom fill all 85 free
        # blocks. P correctly waits. Previously D also waited on P's position.
        s, d = W.harness(85)
        p = W.F.request("P", 100)
        p.status = N.STATUS.WAITING
        s.waiting = Queue([p, d])
        s.max_num_running_reqs = 100
        s.pd_decoupled_no_mix = True
        s._pd_forced_schedule_phase = "decode"
        s.pd_request_fcfs = False
        s.prev_step_scheduled_req_ids = set()
        s._pd_pending_request_ids = set()
        self.assertEqual(s._virtual_context_waiting_decode_service_reserve(), 3)
        self.assertFalse(s._check_virtual_context_prefill_residency_admission(p, 1, "prefill")[0])
        s.isolation_phase_waiting_scan = False
        self.assertEqual(scan(s, "decode")[0], [])
        s.isolation_phase_waiting_scan = True
        self.assertEqual(scan(s, "decode")[0], [d.request_id])
        self.assertEqual(list(s.waiting), [p])
        self.assertEqual(s.kv_cache_manager.block_pool.get_num_free_blocks(), 85)

    def test_cancel_after_rejected_scan_removes_same_restored_object(self):
        p, d = row("P"), row("D", decode=True)
        s = setup([p, d])
        self.assertEqual(scan(s, "decode", admit=lambda r: False)[0], [])
        self.assertIs(s.waiting.peek_request(), p)
        s.waiting.remove_request(d)
        s.requests.pop("D")
        self.assertEqual(scan(s, "decode")[0], [])
        self.assertEqual(list(s.waiting), [p])

    def test_unset_default_is_proposal_only_and_explicit_zero_rolls_back(self):
        assignment = next(n for n in N.METHODS["__init__"].body if isinstance(n, ast.Assign)
                          and ast.unparse(n.targets[0]) == "self.isolation_phase_waiting_scan")
        true_values = next(n.value for n in N.TREE.body if isinstance(n, ast.Assign)
                           and ast.unparse(n.targets[0]) == "_TRUE_VALUES")
        scope = {"os": os, "_TRUE_VALUES": ast.literal_eval(true_values)}
        read_bool = next(n for n in N.TREE.body if isinstance(n, ast.FunctionDef)
                         and n.name == "_read_bool_env")
        code = compile(ast.fix_missing_locations(ast.Module(
            body=[copy.deepcopy(read_bool), copy.deepcopy(assignment)], type_ignores=[])), N.PATH, "exec")
        for virtual in (False, True):
            for separated in (False, True):
                for parked in (False, True):
                    for option in (None, "0", "1"):
                        env = {"VLLM_KV_RETENTION_P1_PARKED_PREFILL": str(int(parked))}
                        if option is not None:
                            env["VLLM_ISOLATION_PHASE_WAITING_SCAN"] = option
                        scope["self"] = s = NS(virtual_context_enabled=virtual, pd_decoupled_no_mix=separated)
                        with patch.dict(os.environ, env, clear=True):
                            exec(code, scope)
                        self.assertEqual(s.isolation_phase_waiting_scan,
                                         virtual and separated and parked and option != "0")


class WaitingDecodeFCFSScopeTests(unittest.TestCase):
    def harness(self, rows=None, *, pending_prefill=False, enabled=True):
        s = setup(rows if rows is not None else [row("D", decode=True)])
        s.pd_request_fcfs = True  # The actual launcher default, unlike old tests.
        s.isolation_waiting_decode_fcfs_scope = enabled
        older = row("older-P")
        older.arrival_time, older.arrival_iteration_id = 1.0, 1
        older.status = N.STATUS.RUNNING
        s.requests[older.request_id] = older
        if pending_prefill:
            s._pd_pending_request_ids = {older.request_id}
            s._pd_pending_phase_request_ids = {"prefill": {older.request_id}}
        else:
            s.running = [older]
        return s

    def test_running_prefill_does_not_block_sampled_decode(self):
        d = row("D", decode=True)
        s = self.harness([row("queued-P"), d])
        self.assertTrue(s.has_pd_phase_work("decode"))
        self.assertTrue(s._has_older_active_prefill_request(d))
        self.assertEqual(scan(s, "decode")[0], ["D"])
        self.assertEqual([r.request_id for r in s.waiting], ["queued-P"])

    def test_pending_prefill_does_not_block_sampled_decode(self):
        s = self.harness(pending_prefill=True)
        self.assertEqual(scan(s, "decode")[0], ["D"])
        self.assertEqual(s._pd_pending_request_ids, {"older-P"})
        self.assertEqual(s._pd_pending_phase_request_ids, {"prefill": {"older-P"}})

    def test_rollback_keeps_scan_but_restores_older_prefill_guard(self):
        s = self.harness([row("P"), row("D", decode=True)], enabled=False)
        admitted, state = scan(s, "decode")
        self.assertEqual(admitted, [])
        self.assertEqual(state["isolation_phase_waiting_skip_count"], 1)
        self.assertEqual(state["waiting_break_defer_reason"], "fcfs_request_order")
        self.assertEqual([r.request_id for r in s.waiting], ["P", "D"])

    def test_prefill_mixed_and_nonforced_paths_keep_guard(self):
        for phase, separated, forced, decode in (
            ("prefill", True, "prefill", False),
            ("mixed", False, None, True),
            ("decode", True, None, True),
            ("decode", True, "prefill", True),
        ):
            with self.subTest(phase=phase, forced=forced):
                s = self.harness([row("candidate", decode=decode)])
                s.pd_decoupled_no_mix = separated
                s._pd_forced_schedule_phase = forced
                admitted, state = scan(s, phase)
                self.assertEqual(admitted, [])
                self.assertEqual(state["waiting_break_defer_reason"], "fcfs_request_order")

    def test_decode_fcfs_order_and_row_limit_are_preserved(self):
        s = self.harness([row("P"), row("D1", decode=True), row("D2", decode=True)])
        self.assertEqual(scan(s, "decode", limit=1)[0], ["D1"])
        self.assertEqual([r.request_id for r in s.waiting], ["P", "D2"])
        self.assertEqual(scan(s, "decode")[0], ["D2"])

    def test_pending_decode_remains_blocked_after_ordering_exemption(self):
        s = self.harness([row("P"), row("D1", decode=True), row("D2", decode=True)])
        s._pd_pending_request_ids = {"D1"}
        s._pd_pending_phase_request_ids = {"decode": {"D1"}}
        admitted, state = scan(s, "decode")
        self.assertEqual(admitted, [])
        self.assertEqual(state["waiting_break_defer_reason"], "fcfs_request_order")
        self.assertEqual([r.request_id for r in s.waiting], ["P", "D1", "D2"])

    def test_all_resume_owner_guards_remain_authoritative(self):
        for owner in ("pending", "previous_step", "deferred_this_pass"):
            with self.subTest(owner=owner):
                s = self.harness([row("D1", decode=True), row("D2", decode=True)])
                if owner == "pending":
                    s._pd_pending_request_ids = {"D1"}
                elif owner == "previous_step":
                    s.prev_step_scheduled_req_ids = {"D1"}
                admitted, state = scan(
                    s, "decode", deferred={"D1"} if owner == "deferred_this_pass" else ())
                self.assertEqual(admitted, [])
                self.assertEqual(state["isolation_waiting_decode_prefill_fcfs_bypass_count"], 1)
                self.assertEqual(state["isolation_waiting_decode_prefill_fcfs_first_id"], "D1")
                self.assertEqual(state["isolation_waiting_decode_resume_owner_block_count"], 1)
                self.assertEqual(state["waiting_break_defer_reason"], "fcfs_request_order")
                self.assertEqual([r.request_id for r in s.waiting], ["D1", "D2"])

    def test_no_gate_bypass_metric_without_an_actual_older_prefill(self):
        s = setup([row("D", decode=True)])
        s.pd_request_fcfs = True
        admitted, state = scan(s, "decode")
        self.assertEqual(admitted, ["D"])
        self.assertEqual(state["isolation_waiting_decode_prefill_fcfs_bypass_count"], 0)
        self.assertEqual(state["isolation_waiting_decode_prefill_fcfs_first_id"], "")

    def test_first_token_and_durable_prefill_boundary_still_required(self):
        for missing in ("sampled_token", "logical_prefix", "dram_state"):
            with self.subTest(missing=missing):
                d = row("D", decode=True)
                if missing == "sampled_token":
                    d.num_output_tokens = 0
                elif missing == "logical_prefix":
                    d.virtual_context_deferred_logical_computed_tokens = 16
                else:
                    d.virtual_context_deferred_state = ""
                s = self.harness([d])
                self.assertFalse(s._virtual_context_waiting_decode_ready(d))
                admitted, state = scan(s, "decode")
                self.assertEqual(admitted, [])
                self.assertEqual(list(s.waiting), [d])
                self.assertEqual(state["isolation_waiting_decode_prefill_fcfs_bypass_count"], 0)

    def test_nonfitting_first_decode_stays_first_then_can_be_cancelled(self):
        p, d1, d2 = row("P"), row("D1", decode=True), row("D2", decode=True)
        s = self.harness([p, d1, d2])
        checked = []
        def reject(r):
            checked.append(r.request_id)
            return False
        for _ in range(2):
            admitted, state = scan(s, "decode", admit=reject)
            self.assertEqual(admitted, [])
            self.assertEqual(state["isolation_waiting_decode_prefill_fcfs_bypass_count"], 1)
            self.assertEqual(list(s.waiting), [p, d1, d2])
            self.assertEqual(d1.kv_retention_execution_plan_generation, 0)
        self.assertEqual(checked, ["D1", "D1"])  # D2 never jumps a nonfitting D1.
        s.waiting.remove_request(d1)
        s.requests.pop(d1.request_id)
        self.assertEqual(scan(s, "decode")[0], ["D2"])
        self.assertEqual(list(s.waiting), [p])

    def test_real_native_reload_rejection_and_commit_after_fcfs_exemption(self):
        for failure in ("phase", "residency", "allocate", ""):
            with self.subTest(failure=failure):
                d = row("D", decode=True)
                s = self.harness([row("P"), d])
                reload = N.scheduler(d, fail=failure)
                results = []
                def admit(r):
                    results.append(N.run_waiting_candidate(reload, r))
                    return bool(results[-1]["num_scheduled_tokens"])
                admitted, state = scan(s, "decode", admit=admit)
                self.assertEqual(len(results), 1)  # The real admission was reached.
                self.assertEqual(state["isolation_waiting_decode_prefill_fcfs_bypass_count"], 1)
                plans = reload._build_tiered_request_execution_plans(
                    results[0]["num_scheduled_tokens"], results[0]["tiered_explicit_execution_plans"])
                if failure:
                    self.assertEqual(admitted, [])
                    self.assertEqual(plans, ())
                    self.assertEqual(d.kv_retention_execution_plan_generation, 0)
                    self.assertEqual(d.virtual_context_deferred_state, "deferred_tiered_provider")
                    self.assertEqual([r.request_id for r in s.waiting], ["P", "D"])
                else:
                    self.assertEqual(admitted, ["D"])
                    self.assertEqual(plans, (("D", 1, "native_reload", 4),))
                    self.assertEqual(d.virtual_context_deferred_state, "decode_hbm_reload_pending")
                    self.assertEqual(d.kv_retention_execution_plan_acked_generation, 0)
                    self.assertEqual([r.request_id for r in s.waiting], ["P"])

    def test_sequence_and_token_budgets_still_bound_admission(self):
        s = self.harness([row("D1", decode=True), row("D2", decode=True)])
        s.max_num_running_reqs = 1  # Already occupied by the older Prefill.
        admitted, state = scan(s, "decode")
        self.assertEqual(admitted, [])
        self.assertEqual(state["waiting_break_defer_reason"], "max_num_seqs")
        s.max_num_running_reqs = 100
        self.assertEqual(scan(s, "decode", token_budget=0)[0], [])
        self.assertEqual(scan(s, "decode", token_budget=1)[0], ["D1"])
        self.assertEqual([r.request_id for r in s.waiting], ["D2"])

    def test_first_use_log_is_bounded_and_reports_no_admission(self):
        s = self.harness()
        s._pd_pending_request_ids = {"D"}
        _, state = scan(s, "decode")
        node = next(n for n in N.METHODS["schedule"].body if isinstance(n, ast.If)
                    and ast.unparse(n.test).startswith(
                        "isolation_waiting_decode_prefill_fcfs_bypass_count > 0"))
        code = compile(ast.fix_missing_locations(ast.Module(
            body=[copy.deepcopy(node)], type_ignores=[])), N.PATH, "exec")
        logs = []
        state.update(logger=NS(info=lambda *args: logs.append(args)),
                     total_num_scheduled_tokens=sum(state["num_scheduled_tokens"].values()))
        exec(code, state)
        exec(code, state)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0][1:], (1, "D", 1, 0, "fcfs_request_order"))
        self.assertTrue(s._isolation_waiting_decode_fcfs_scope_logged)

    def test_flag_defaults_proposal_only_and_independent_rollback(self):
        names = {"self.isolation_phase_waiting_scan", "self.isolation_waiting_decode_fcfs_scope"}
        assignments = [copy.deepcopy(n) for n in N.METHODS["__init__"].body
                       if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in names]
        self.assertEqual(len(assignments), 2)
        true_values = next(n.value for n in N.TREE.body if isinstance(n, ast.Assign)
                           and ast.unparse(n.targets[0]) == "_TRUE_VALUES")
        read_bool = next(n for n in N.TREE.body if isinstance(n, ast.FunctionDef)
                         and n.name == "_read_bool_env")
        code = compile(ast.fix_missing_locations(ast.Module(
            body=[copy.deepcopy(read_bool), *assignments], type_ignores=[])), N.PATH, "exec")
        scope = {"os": os, "_TRUE_VALUES": ast.literal_eval(true_values)}
        for virtual in (False, True):
            for separated in (False, True):
                for parked in (False, True):
                    for scan_option in (None, "0", "1"):
                        for scope_option in (None, "0", "1"):
                            env = {"VLLM_KV_RETENTION_P1_PARKED_PREFILL": str(int(parked))}
                            if scan_option is not None:
                                env["VLLM_ISOLATION_PHASE_WAITING_SCAN"] = scan_option
                            if scope_option is not None:
                                env["VLLM_ISOLATION_WAITING_DECODE_FCFS_SCOPE"] = scope_option
                            scope["self"] = s = NS(virtual_context_enabled=virtual,
                                                   pd_decoupled_no_mix=separated)
                            with patch.dict(os.environ, env, clear=True):
                                exec(code, scope)
                            scan_enabled = virtual and separated and parked and scan_option != "0"
                            self.assertEqual(s.isolation_phase_waiting_scan, scan_enabled)
                            self.assertEqual(s.isolation_waiting_decode_fcfs_scope,
                                             scan_enabled and scope_option != "0")


if __name__ == "__main__":
    unittest.main()
