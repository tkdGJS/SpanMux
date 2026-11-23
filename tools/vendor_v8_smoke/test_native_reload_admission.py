#!/usr/bin/env python3
"""CPU execution of real reload admission/continuation code, not GPU proof."""
from __future__ import annotations

import ast
import copy
import time
import unittest
from collections import defaultdict, deque
from concurrent.futures import Future
from types import SimpleNamespace as NS
from unittest.mock import patch

import test_exact_phase_cleanup as C
import test_kv_retention_layerwise_checkpoint as K
import test_p1_parked_prefill as P


PATH = "vendor_v8/vllm/v1/core/sched/scheduler.py"
SOURCE = (C.ROOT / PATH).read_text()
TREE = ast.parse(SOURCE)
CLS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
METHODS = {n.name: n for n in CLS.body if isinstance(n, ast.FunctionDef)}
STATES = {"deferred_tiered_provider", "decode_hbm_reload_wait_dram", "tiered_provider_resumed"}
STATUS = NS(PREEMPTED="preempted", WAITING="waiting", RUNNING="running")
SCHED = C.adapters(PATH, "Scheduler", {
    "_commit_virtual_context_native_reload_admissions",
    "_build_tiered_request_execution_plans", "_ttft_prefill_execution_mode",
    "_pd_prefill_lane_remaining_tokens", "_pd_prefill_lane_boundary_tokens",
    "_debug_has_prefill_remaining", "_virtual_context_waiting_decode_ready",
    "_prepare_virtual_context_native_prefill_promotion",
    "_prepare_virtual_context_native_decode_handoff", "layerwise_offload_needed",
    "_native_reload_destinations_compatible",
    "_virtual_context_waiting_decode_service_candidate",
    "commit_layerwise_kv_offload",
}, dict(TieredAllocatorInvariantError=RuntimeError, RequestStatus=STATUS,
        _VIRTUAL_CONTEXT_TIERED_PROVIDER_STATES=STATES,
        _VIRTUAL_CONTEXT_REHYDRATE_STATES=STATES | {"hbm_reload_pending", "decode_hbm_reload_pending"},
        TIERED_EXECUTION_NATIVE_RELOAD="native_reload", TIERED_EXECUTION_AUTO="auto",
        TIERED_EXECUTION_CONTEXT_RING="context_ring", TIERED_EXECUTION_RING_DECODE="ring_decode",
        defaultdict=defaultdict, virtual_context_residency_victim_context_tokens=max))
SCHED._virtual_context_waiting_decode_ready = staticmethod(SCHED._virtual_context_waiting_decode_ready)


def request(decode=False):
    r = NS(request_id="A", status=STATUS.PREEMPTED, num_computed_tokens=0,
           num_prompt_tokens=32 if decode else 64, num_tokens=33 if decode else 64,
           num_output_tokens=int(decode), num_external_computed_tokens=0,
           virtual_context_deferred_logical_computed_tokens=32,
           virtual_context_deferred_state="deferred_tiered_provider",
           virtual_context_hbm_reload_planned_tokens=0,
           virtual_context_tiered_provider_virtual_external_tokens=0,
           kv_retention_execution_plan_generation=0,
           kv_retention_execution_plan_acked_generation=0,
           kv_retention_execution_plan_mode="", kv_retention_execution_plan_expected_reload_pages=0,
           virtual_context_resume_without_recompute_count=0,
           virtual_context_tiered_provider_resume_count=0,
           num_cached_tokens=-1, has_encoder_inputs=False, lora_request=None,
           pd_recompute_frontier_tokens=0)
    r.resume_from_external_residency = lambda **kw: setattr(r, "num_computed_tokens", kw["logical_computed_tokens"])
    return r


class Queue(list):
    def pop_request(self):
        return self.pop(0)

    def prepend_request(self, request):
        self.insert(0, request)


def scheduler(r, *, fail="", retained=False):
    s = SCHED()
    s.virtual_context_enabled = True
    s.virtual_context_reload_to_hbm = True
    s.pd_decoupled_no_mix = True
    s._tiered_execution_plan_next_generation = 0
    s._tiered_execution_plan_overwrite_reject_count = 0
    s.requests = {r.request_id: r}
    s.block_size = 16
    s.waiting, s.running = Queue([r]), []
    s.log_stats, s.lora_config, s.connector = False, None, None
    s.is_encoder_decoder, s.use_eagle, s.num_lookahead_tokens = False, False, 0
    s.scheduler_config = NS(long_prefill_token_threshold=16, enable_chunked_prefill=True)
    s.ttft_prefill_native_chunk_max_tokens = 16
    s.virtual_context_ring_bw_last_reason = ""
    blocks = NS(blocks=([NS(block_id=i, is_null=False, ref_cnt=1) for i in (2, 3)],))
    s.kv_cache_manager = NS(
        virtual_context=object(), empty_kv_cache_blocks=NS(blocks=([],)),
        block_pool=NS(get_num_free_blocks=lambda: 100),
        has_request_block_table=lambda rid: retained,
        is_request_partial_residency=lambda rid: False,
        get_retained_residency_counts=lambda *args: (2, 0),
        get_computed_blocks=lambda r: (NS(blocks=([],)), 0),
        get_blocks=lambda rid: blocks,
        get_virtual_context_hbm_reload_required_page_count=lambda ids: 4,
        plan_virtual_context_hbm_reload_pages=lambda **kw: [
            NS(kv_cache_group_id=0, logical_token_page=p) for p in (0, 1)],
        allocate_virtual_context_hbm_reload_blocks=lambda *args: blocks,
        allocate_slots=lambda *args, **kw: None if fail == "allocate" else blocks,
    )
    if retained:
        s.kv_cache_manager.get_computed_blocks = lambda r: (_ for _ in ()).throw(
            AssertionError("prepared reload destinations are not prefix-cache hits"))
    s._can_resume_with_tiered_virtual_context = lambda *args: True
    s._should_reload_virtual_context_to_hbm = lambda *args: (True, 4, 2, 4)
    s._check_virtual_context_prefill_ring_bandwidth_admission = lambda r, n: (True, n, 0, 0)
    s._cap_virtual_context_ring_prefill_admission_tokens = lambda r, n, **kw: n
    s._cap_pd_phase_tokens = lambda r, n, **kw: 0 if fail == "phase" else n
    s._check_virtual_context_prefill_residency_admission = lambda *args, **kw: (
        fail != "residency", 4, 100, 0, 0, 4, 0, False)
    s._prefill_needs_capacity_floor_snapshot = lambda *args: False
    s._needs_virtual_context_worker_rehydrate = lambda r: True
    s._update_connector_prefix_cache_stats = lambda *args: None
    s._debug_record_request_defer = lambda *args: None
    s._debug_running_decode_resident_blocks = lambda: 0
    s._pd_pending_decode_active = lambda: False
    s._scheduled_prefill_fit_capacity_claim = lambda *args, **kw: NS(
        reclaim_blocks=0, required_blocks_with_reserve=4)
    return s


def run_waiting_candidate(s, r, *, stale_intent=False):
    """Execute the unmodified WAITING admission body with fake allocator/queues.

    The earlier queue ranking and later transport are not under test here.
    All token/admission/allocation gates between reload intent and commit are.
    """
    loop = next(n for n in ast.walk(METHODS["schedule"]) if isinstance(n, ast.While)
                and ast.unparse(n.test) == "self.waiting and token_budget > 0")
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == "num_external_computed_tokens")
    body = copy.deepcopy(loop.body[start:])
    namespace = {}
    # Actual schedule-local initializers (counters, maps and sets), without
    # executing unrelated control-plane policy at the top of schedule().
    for node in METHODS["schedule"].body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if not isinstance(target, ast.Name) or node.value is None:
            continue
        try:
            value = eval(compile(ast.Expression(node.value), PATH, "eval"),
                         {"set": set, "dict": dict, "defaultdict": defaultdict, "int": int})
        except (NameError, AttributeError):
            continue
        namespace[target.id] = value
    namespace.update(self=s, request=r, token_budget=16, encoder_compute_budget=0,
                     time=time, RequestStatus=STATUS, defaultdict=defaultdict,
                     TieredAllocatorInvariantError=RuntimeError,
                     _VIRTUAL_CONTEXT_TIERED_PROVIDER_STATES=STATES,
                     TIERED_EXECUTION_NATIVE_RELOAD="native_reload",
                     skipped_waiting_requests=Queue(), ttft_prefill_policy_active_this_iteration=False,
                     pd_decoupled_selected_phase="decode" if r.num_output_tokens else "prefill",
                     active_decode_during_prefill_phase=False,
                     waiting_decode_ready=bool(r.num_output_tokens))
    if stale_intent:
        namespace["tiered_explicit_execution_plans"] = {"A": ("native_reload", 4)}
        namespace["tiered_native_reload_planned_tokens"] = {"A": 32}
        namespace["virtual_context_native_prefill_promotion_request_ids"] = {"A"}
    # Use a one-candidate loop so the real continue/break statements retain
    # their meaning without rerunning the queue-ranking stage.
    one = ast.For(target=ast.Name(id="_candidate", ctx=ast.Store()),
                  iter=ast.Tuple(elts=[ast.Constant(1)], ctx=ast.Load()), body=body, orelse=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[one], type_ignores=[])), PATH, "exec"),
         namespace)
    s._commit_virtual_context_native_reload_admissions(
        namespace["num_scheduled_tokens"], namespace["tiered_explicit_execution_plans"],
        namespace["tiered_native_reload_planned_tokens"])
    return namespace


class NativeReloadAdmissionTests(unittest.TestCase):
    def test_waiting_native_reload_commits_authority_and_mode_together(self):
        r = request()
        s = scheduler(r)
        env = run_waiting_candidate(s, r)
        self.assertEqual(env["num_scheduled_tokens"], {"A": 16})
        self.assertEqual(r.virtual_context_deferred_state, "hbm_reload_pending")
        self.assertEqual(s._ttft_prefill_execution_mode(
            r, env["virtual_context_native_prefill_promotion_request_ids"]), "native")
        plan = s._build_tiered_request_execution_plans(
            env["num_scheduled_tokens"], env["tiered_explicit_execution_plans"])
        self.assertEqual(plan, (("A", 1, "native_reload", 4),))
        self.assertEqual(r.virtual_context_hbm_reload_planned_tokens, 32)

    def test_admission_rejections_leave_no_pending_generation_and_retry(self):
        for reason in ("phase", "residency", "allocate"):
            for decode in (False, True):
                with self.subTest(reason=reason, decode=decode):
                    r = request(decode)
                    s = scheduler(r, fail=reason)
                    env = run_waiting_candidate(s, r)
                    self.assertEqual(env["num_scheduled_tokens"], {})
                    self.assertEqual(r.virtual_context_deferred_state, "deferred_tiered_provider")
                    self.assertEqual(r.virtual_context_hbm_reload_planned_tokens, 0)
                    self.assertEqual(s._build_tiered_request_execution_plans(
                        {}, env["tiered_explicit_execution_plans"]), ())
                    self.assertEqual(r.kv_retention_execution_plan_generation, 0)
                    self.assertEqual(SCHED._virtual_context_waiting_decode_ready(r), decode)
                    retry = scheduler(r)
                    run_waiting_candidate(retry, r)
                    self.assertEqual(r.virtual_context_deferred_state,
                                     "decode_hbm_reload_pending" if decode else "hbm_reload_pending")

    def test_retained_prepared_destinations_bypass_prefix_lookup_on_retry(self):
        r = request()
        s = scheduler(r, retained=True, fail="allocate")
        run_waiting_candidate(s, r)
        self.assertEqual(r.virtual_context_deferred_state, "deferred_tiered_provider")
        s.kv_cache_manager.allocate_slots = lambda *args, **kw: NS()
        run_waiting_candidate(s, r)
        self.assertEqual(r.virtual_context_deferred_state, "hbm_reload_pending")

    def test_same_round_reentry_does_not_keep_abandoned_native_intent(self):
        r = request()
        s = scheduler(r)
        s._should_reload_virtual_context_to_hbm = lambda *args: (False, 4, 2, 4)
        env = run_waiting_candidate(s, r, stale_intent=True)
        self.assertEqual(r.virtual_context_deferred_state, "tiered_provider_resumed")
        self.assertEqual(env["tiered_explicit_execution_plans"], {})
        self.assertEqual(env["virtual_context_native_prefill_promotion_request_ids"], set())
        self.assertEqual(s._build_tiered_request_execution_plans(
            env["num_scheduled_tokens"], env["tiered_explicit_execution_plans"])[0][2], "context_ring")

    def test_running_prepare_reuses_destinations_without_publishing_pending(self):
        r = request()
        r.num_computed_tokens = 32
        s = scheduler(r, retained=True)
        for _ in range(2):
            ok, pages, blocks, _ = s._prepare_virtual_context_native_prefill_promotion(r, ring_token_cap=16)
            self.assertTrue(ok)
            self.assertEqual((pages, blocks), (2, 2))
            self.assertEqual(r.virtual_context_deferred_state, "deferred_tiered_provider")
        s._commit_virtual_context_native_reload_admissions({}, {"A": ("native_reload", 4)}, {"A": 32})
        self.assertEqual(r.virtual_context_hbm_reload_planned_tokens, 0)

    def test_retry_cannot_overwrite_shared_reload_destination(self):
        r = request()
        r.num_computed_tokens = 32
        s = scheduler(r, retained=True)
        s.kv_cache_manager.get_blocks = lambda _: NS(blocks=([
            NS(block_id=1, is_null=False, ref_cnt=2),
            NS(block_id=2, is_null=False, ref_cnt=1),
        ],))
        self.assertFalse(s._prepare_virtual_context_native_prefill_promotion(r, ring_token_cap=16)[0])

    def test_destination_preflight_requires_exact_holes_and_no_aliases(self):
        r, blocks = request(), [NS(block_id=1, is_null=False, ref_cnt=1),
                               NS(block_id=0, is_null=True, ref_cnt=0)]
        s = scheduler(r, retained=True)
        s.kv_cache_manager.get_blocks = lambda _: NS(blocks=(blocks,))
        self.assertFalse(s._native_reload_destinations_compatible("A", {0: {0}}))
        self.assertTrue(s._native_reload_destinations_compatible("A", {0: {0, 1}}))
        blocks[1] = NS(block_id=1, is_null=False, ref_cnt=1)
        self.assertFalse(s._native_reload_destinations_compatible("A", {0: {0, 1}}))

    def test_waiting_shared_destination_fails_before_copy_or_future_publication(self):
        r = request()
        s = scheduler(r, retained=True)
        s.kv_cache_manager.get_blocks("A").blocks[0][0].ref_cnt = 2
        s.kv_cache_manager.allocate_virtual_context_hbm_reload_blocks = lambda *args: (
            self.fail("unsafe reload must not reach allocator mutation"))
        with self.assertRaisesRegex(RuntimeError, "unsafe destination ownership"):
            run_waiting_candidate(s, r)
        self.assertEqual(r.virtual_context_deferred_state, "deferred_tiered_provider")
        self.assertEqual(r.kv_retention_execution_plan_generation, 0)

    def test_decode_preparation_does_not_publish_a_future(self):
        r = request(True)
        r.num_computed_tokens = 32
        s = scheduler(r, retained=True)
        s._virtual_context_native_decode_handoff_requirement = lambda r: (4, {0: {0, 1}}, 2, 4, False)
        for _ in range(2):
            self.assertTrue(s._prepare_virtual_context_native_decode_handoff(r)[0])
            self.assertEqual(r.virtual_context_deferred_state, "deferred_tiered_provider")
            self.assertEqual(r.virtual_context_hbm_reload_planned_tokens, 0)

    def test_pending_without_explicit_plan_fails_before_generation_mutation(self):
        r = request()
        r.virtual_context_deferred_state = "hbm_reload_pending"
        s = scheduler(r)
        with self.assertRaisesRegex(RuntimeError, "explicit native_reload"):
            s._build_tiered_request_execution_plans({"A": 16}, {})
        self.assertEqual(r.kv_retention_execution_plan_generation, 0)

    def test_active_generation_cannot_be_replaced(self):
        r = request()
        r.kv_retention_execution_plan_generation = 2
        s = scheduler(r)
        with self.assertRaisesRegex(RuntimeError, "unacknowledged"):
            s._commit_virtual_context_native_reload_admissions(
                {"A": 16}, {"A": ("native_reload", 4)}, {"A": 32})
        self.assertEqual(r.virtual_context_deferred_state, "deferred_tiered_provider")

    def test_batch_commit_preflights_all_rows_before_changing_any(self):
        a, b = request(), request()
        b.request_id = "B"
        b.kv_retention_execution_plan_generation = 7
        s = scheduler(a)
        s.requests["B"] = b
        with self.assertRaisesRegex(RuntimeError, "unacknowledged"):
            s._commit_virtual_context_native_reload_admissions(
                {"A": 16, "B": 16}, {"A": ("native_reload", 4), "B": ("native_reload", 4)},
                {"A": 32, "B": 32})
        self.assertEqual(a.virtual_context_deferred_state, "deferred_tiered_provider")

    def test_non_reload_ring_and_hbm_default_keep_existing_behavior(self):
        r = request()
        s = scheduler(r)
        self.assertEqual(s._commit_virtual_context_native_reload_admissions({"A": 16}, {}, {}), set())
        self.assertEqual(s._build_tiered_request_execution_plans({"A": 16}, {})[0][2], "context_ring")
        s.virtual_context_enabled = False
        self.assertEqual(s._build_tiered_request_execution_plans({"A": 16}, {}), ())

    def test_backend_resolver_keeps_native_authority_for_each_unfinished_layer(self):
        path = C.ROOT / "vendor_v8/vllm/v1/core/tiered_execution.py"
        module = ast.parse(path.read_text())
        module.body = [n for n in module.body if not (
            isinstance(n, ast.ImportFrom) and (n.module or "").startswith("vllm"))]
        scope = dict(TieredBackendEligibilityError=RuntimeError)
        exec(compile(module, str(path), "exec"), scope)
        plans = scope["parse_tiered_execution_plan_entries"]((("A", 1, "native_reload", 4),))
        resolve = scope["resolve_tiered_reload_authority"]
        for phase in ("prefill", "decode"):
            for pages in ({0, 1}, {1}):
                self.assertEqual(resolve(phase=phase, reload_required_pages={"A": pages},
                                         execution_plans=plans), "native")
        plans["B"] = (2, "context_ring", 4)
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            resolve(phase="prefill", reload_required_pages={"A": {0}, "B": {0}},
                    execution_plans=plans)

    def test_final_publication_is_after_waiting_and_before_cached_payload(self):
        schedule = ast.unparse(METHODS["schedule"])
        self.assertLess(schedule.index("while self.waiting and token_budget > 0"),
                        schedule.index("self._commit_virtual_context_native_reload_admissions("))
        self.assertLess(schedule.index("self._commit_virtual_context_native_reload_admissions("),
                        schedule.index("self._make_cached_request_data("))
        # Keep the all-layer ACK failure path. No fake readiness was introduced.
        recovery = ast.unparse(METHODS["update_from_output"])
        self.assertIn("remaining_reload_pages", recovery)
        self.assertIn("virtual_context_reload_ack_incomplete_failure", recovery)

    def test_real_recovery_requires_all_layer_ack_and_clears_only_on_success(self):
        recovery = METHODS["update_from_output"]
        gate = next(n for n in ast.walk(recovery) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "request.virtual_context_deferred_state in {'hbm_reload_pending', 'decode_hbm_reload_pending'}")
        for remaining in ([], [object()]):
            with self.subTest(remaining=len(remaining)):
                r = request()
                r.virtual_context_deferred_state = "hbm_reload_pending"
                r.virtual_context_hbm_reload_planned_tokens = 32
                r.virtual_context_hbm_reloaded_tokens = 0
                r.client_index = 0
                r.trace_headers = None
                r.get_finished_reason = lambda: "error"
                r.take_events = lambda: []
                s = scheduler(r)
                s.kv_cache_manager.plan_virtual_context_hbm_reload_pages = lambda **kw: remaining
                finishes = []
                s.finish_requests = lambda rid, status: finishes.append(rid)
                result = NS(iteration_debug={})
                scope = dict(self=s, request=r, req_id="A", scheduler_output=result,
                             outputs=defaultdict(list), EngineCoreOutput=NS,
                             RequestStatus=NS(FINISHED_ERROR="error"), reached=False)
                body = [copy.deepcopy(gate), ast.Assign(
                    targets=[ast.Name(id="reached", ctx=ast.Store())], value=ast.Constant(True))]
                loop = ast.For(target=ast.Name(id="_", ctx=ast.Store()),
                               iter=ast.Tuple(elts=[ast.Constant(1)], ctx=ast.Load()),
                               body=body, orelse=[])
                exec(compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])),
                             PATH, "exec"), scope)
                self.assertEqual(scope["reached"], not remaining)
                if remaining:
                    self.assertEqual(finishes, ["A"])
                    self.assertEqual(r.virtual_context_hbm_reloaded_tokens, 0)
                    self.assertEqual(result.iteration_debug["virtual_context_reload_ack_incomplete_failure"], 1)
                else:
                    self.assertFalse(finishes)
                    self.assertEqual(r.virtual_context_deferred_state, "hbm_reload_resumed")
                    self.assertEqual(r.virtual_context_hbm_reloaded_tokens, 32)
                    self.assertEqual(r.virtual_context_hbm_reload_planned_tokens, 0)


class ReloadContinuationTests(unittest.TestCase):
    def output(self):
        out = P.out()
        out.tiered_request_execution_plans = (("A", 5, "native_reload", 4),)
        out.tiered_native_hbm_reload_request_ids = ()
        out.kv_retention_layerwise_offload_requested = False
        out.kv_retention_layerwise_pressure_checked = False
        out.ttft_layerwise_prefill_preemption = True
        return out

    def test_native_reload_guard_uses_immutable_future_not_residency_probe(self):
        out = self.output()
        self.assertTrue(C.HAS_NATIVE_RELOAD_AUTHORITY(out))
        out.tiered_request_execution_plans = (("A", 5, "native_resident", 0),)
        self.assertFalse(C.HAS_NATIVE_RELOAD_AUTHORITY(out))
        out.tiered_native_hbm_reload_request_ids = ("A",)
        self.assertTrue(C.HAS_NATIVE_RELOAD_AUTHORITY(out))

    def test_engine_allows_sm_yield_but_does_not_request_swap(self):
        e, out, f = P.engine(), self.output(), Future()
        e.ttft_layerwise_prefill_preemption_enabled = True
        e.pd_sm_transition = NS(active=False)
        queues = {"prefill": deque([(f, out, f)]), "decode": deque()}
        self.assertFalse(e._request_ttft_prefill_preemption_if_needed(queues))
        out.kv_retention_boundary_reason = ""
        e.pd_sm_transition.active = True
        with patch.dict(e._request_ttft_prefill_preemption_if_needed.__func__.__globals__,
                        request_ttft_layerwise_prefill_yield=lambda **kw: True):
            self.assertTrue(e._request_ttft_prefill_preemption_if_needed(queues))
        self.assertEqual(out.kv_retention_boundary_reason, "sm")
        self.assertFalse(out.kv_retention_layerwise_offload_requested)

    def test_scheduler_rejects_reload_victim_before_allocator_access(self):
        s = SCHED()
        out = self.output()
        self.assertFalse(s.layerwise_offload_needed(out, NS()))
        with self.assertRaisesRegex(RuntimeError, "all-layer reload ACK"):
            s.commit_layerwise_kv_offload(out, NS())

    def test_worker_sm_boundary_keeps_original_reload_owner(self):
        runner, state = K.make_runner_state()
        state.scheduler_output = self.output()
        state.scheduler_output.kv_retention_boundary_reason = "sm"
        state.kv_resume_frame = None
        runner._capture_ttft_kv_checkpoint(state)
        runner._capture_kv_retention_resume_frame(state)
        self.assertIsNone(state.kv_checkpoint)
        self.assertIsNone(state.kv_offload)
        self.assertIsNone(state.kv_resume_frame)
        self.assertEqual(state.model_runner_debug_timings["pd_layerwise_native_reload_sm_boundary"], 1)

    def test_worker_rejects_request_swap_or_snapshot_before_copy(self):
        for reason, offload in (("request", False), ("sm", True)):
            runner, state = K.make_runner_state()
            state.scheduler_output = self.output()
            state.scheduler_output.kv_retention_boundary_reason = reason
            state.scheduler_output.kv_retention_layerwise_offload_requested = offload
            state.kv_resume_frame = None
            with self.assertRaisesRegex(K.M.TTFTLayerwisePrefillError, "cannot checkpoint"):
                runner._capture_ttft_kv_checkpoint(state)
            self.assertIsNone(state.kv_offload)

    def test_late_pressure_worker_cannot_copy_reload_destinations(self):
        runner_type = C.adapters(C.RUNNER_PATH, "GPUModelRunner", {"offload_parked_prefill"},
                                {"TTFTLayerwisePrefillError": K.M.TTFTLayerwisePrefillError})
        runner = runner_type()
        runner._kv_retention_parked_store = NS(get=lambda _: (None, NS(scheduler_output=self.output())))
        owner = NS(validate_key=lambda _: None, binding=object())
        with self.assertRaisesRegex(K.M.TTFTLayerwisePrefillError, "parked offload"):
            runner.offload_parked_prefill(owner)

    def test_reload_abort_cleanup_still_uses_exact_future_owner(self):
        runner = C.RUNNER()
        out = C.output()
        out.tiered_request_execution_plans = (("A", 5, "native_reload", 4),)
        events = []
        ctx = C.context("prefill", out, events)
        runner._pd_phase_contexts = {("prefill", 0): ctx}
        self.assertTrue(runner.clear_pd_phase_execute_state("prefill", ctx.kv_retention_execute_owner))
        self.assertEqual(events, ["generator"])
        self.assertIsNone(ctx.ttft_prefill_continuation)

    def test_actual_reload_continuation_survives_two_sm_changes_without_frame(self):
        import test_prefill_resume_frame as F
        import test_prefill_work_binding as W
        fixture = F.ActualRunnerTest()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        runner, state, out, cont = fixture.runner, fixture.state, fixture.out, fixture.cont
        out.tiered_request_execution_plans = (
            (next(iter(out.num_scheduled_tokens)), 5, "native_reload", 4),
        )
        out.kv_retention_boundary_reason = "sm"
        out.kv_retention_layerwise_offload_requested = False
        out.kv_retention_layerwise_transfer = None
        state.kv_checkpoint = None
        runner._capture_ttft_kv_checkpoint = K.RUNNER._capture_ttft_kv_checkpoint.__get__(runner)
        runner._kv_retention_parked_store = P.P.KVRetentionParkedStore(1)
        scope = runner._import_kv_retention_resume_frame.__func__.__globals__
        finished = []
        with patch.dict(scope, {
            "kv_retention_record_retained_streams": F.F.kv_retention_record_retained_streams,
            "TTFTLayerwisePrefillError": K.M.TTFTLayerwisePrefillError,
        }), patch.object(F.F.KVRetentionResumeFrame, "capture", side_effect=AssertionError("reload frame clone")):
            runner._capture_ttft_kv_checkpoint(state)
            runner._capture_kv_retention_resume_frame(state)
            runner._finish_execute_model_forward = lambda **kw: finished.append(kw)
            fixture.registry.suspend(fixture.first, 2)
            out.pd_sm_profile_id, out.pd_sm_profile_epoch = "d32_p8", 2
            second = fixture.registry.resume(out, W.epochs(out))
            fixture.ctx.kv_retention_execute_owner = C.OWNER.from_output("prefill", out, second)
            cont.execution.note_resume()
            hidden = cont.state.hidden_states
            boundary = runner._resume_ttft_prefill_continuation(out, state)
            self.assertEqual(boundary.next_layer, 4)
            self.assertEqual(hidden.streams, ["target"])
            self.assertEqual(finished, [])
            fixture.registry.suspend(second, 4)
            out.pd_sm_profile_id, out.pd_sm_profile_epoch = "d0_p40", 3
            third = fixture.registry.resume(out, W.epochs(out))
            fixture.ctx.kv_retention_execute_owner = C.OWNER.from_output("prefill", out, third)
            cont.execution.note_resume()
            runner._resume_ttft_prefill_continuation(out, state)
        self.assertEqual(len(finished), 1)
        self.assertIs(finished[0]["scheduler_output"], out)
        reference, *_ = F.model_and_continuation(())
        with self.assertRaises(StopIteration) as complete:
            next(reference)
        F.np.testing.assert_equal(finished[0]["model_output"].data, complete.exception.value.data)
        self.assertIsNone(state.kv_checkpoint)
        self.assertIsNone(state.kv_resume_frame)
        self.assertEqual(runner._kv_retention_parked_store.entries, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
