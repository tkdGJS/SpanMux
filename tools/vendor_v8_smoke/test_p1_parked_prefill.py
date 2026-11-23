#!/usr/bin/env python3
"""Actual P1 coordinator/worker adapters with CPU doubles, not GPU E2E."""
from __future__ import annotations

import importlib.util
import os
import sys
import time
import unittest
from collections import deque
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace as NS
from typing import cast
from unittest.mock import patch

import numpy as np
import test_exact_phase_cleanup as C
import test_prefill_work_binding as W
import test_prefill_resume_frame as F
import test_hbm_completion_guard as H

SPEC = importlib.util.spec_from_file_location("p1_parked_contract",
    C.ROOT / "vendor_v8/vllm/model_executor/layers/kv_retention_parked_prefill.py")
P = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = P
SPEC.loader.exec_module(P)

STATUS = NS(RUNNING="running", FINISHED_ABORTED="aborted")
SCHED = C.adapters("vendor_v8/vllm/v1/core/sched/scheduler.py", "Scheduler", {
    "park_kv_retention_prefill", "can_resume_kv_retention_prefill",
    "resume_kv_retention_prefill", "discard_kv_retention_parked",
    "get_num_unfinished_requests"}, {"RequestStatus": STATUS})
ENGINE = C.adapters("vendor_v8/vllm/v1/engine/core.py", "EngineCore", {
    "_pd_owned_total_depth", "_pd_owned_phase_depth",
    "_pd_sm_transition_future_counts", "_request_ttft_prefill_preemption_if_needed",
    "_poll_kv_retention_parked_offload", "_try_resume_ttft_prefill",
    "_ttft_prefill_resume_candidate",
    "_pd_recoverable_future_ready",
    "_prune_ttft_prefill_suspended", "_suspend_ttft_prefill_boundary",
}, {"time": time, "cast": cast, "ModelRunnerOutput": object,
    "RequestStatus": STATUS, "KVRetentionLayerwiseTransfer": type("Transfer", (), {})})
EXEC = C.adapters("vendor_v8/vllm/v1/executor/uniproc_executor.py", "UniProcExecutor", {
    "offload_parked_prefill", "release_parked_execute_dependency"})


def out(ids=("A", "B")):
    o = W.output(ids=ids)
    o.pd_prefill_mode = "native"
    o.kv_retention_layerwise_transfer = None
    o.kv_retention_layerwise_offload_requested = False
    o.kv_retention_boundary_reason = "request"
    o.kv_retention_p1_park_declined = False
    o.kv_retention_p1_parked = False
    return o


def row(req_id):
    return NS(request_id=req_id, status="running", is_finished=lambda: False)


def scheduler(output, cap=2):
    s = SCHED()
    s.requests = {req_id: row(req_id) for req_id in output.num_scheduled_tokens}
    s.running = list(s.requests.values())
    s.waiting = []
    s.max_num_running_reqs = cap
    return s


def work(output=None):
    output = output or out()
    registry = W.REGISTRY(1, parked_capacity=1)
    first = registry.start(output, W.epochs(output))
    boundary = W.boundary(output, first)
    ticket = P.KVRetentionParkedTicket(first, 2, 64, tuple(output.num_scheduled_tokens))
    boundary.kv_retention_parked_ticket = ticket
    return output, registry, first, boundary, ticket


def engine():
    e = ENGINE()
    e._ttft_prefill_suspended = {}
    e._kv_retention_parked_offload_future = None
    e._kv_retention_parked_offload_attempt = None
    e.kv_retention_p1_parked_enabled = True
    e.batch_queue_size = 2
    e._pd_phase_queue_total_depth = lambda queues: sum(map(len, queues.values()))
    e._ttft_prefill_execution_key = lambda output: "prefill"
    e._scheduler_output_request_ids = lambda output: set(output.num_scheduled_tokens)
    e._iteration_debug_timings = {}
    e._ttft_prefill_deferred_abort_ids = set()
    e._notify_pd_engine_progress = lambda: None
    e._write_pd_phase_queue_trace = lambda *args, **kwargs: None
    e._kv_retention_binding_debug = lambda binding: {}
    e._pd_ttft_handoff_active = lambda: False
    e._pd_ttft_handoff_blocked_prefill_checks = 0
    return e


class RegistrySchedulerTests(unittest.TestCase):
    def test_p1_park_lends_slot_without_releasing_logical_claims(self):
        a, reg, first, boundary, _ = work()
        original_claims = dict(a.pd_output_placeholder_owners)
        reg.suspend(first, 2)
        reg.park(first)
        c = out(("C",))
        cb = reg.start(c, W.epochs(c))
        self.assertEqual((cb.slot, cb.slot_generation, len(reg)), (0, 2, 2))
        with self.assertRaises(RuntimeError):
            reg.resume(a, W.epochs(a))
        with self.assertRaises(RuntimeError):
            reg.start(out(("D",)), {"D": 11})
        reg.sampling(cb)
        reg.finish(cb)
        ab = reg.resume(a, W.epochs(a))
        self.assertEqual((ab.slot_generation, ab.segment_id, ab.start_layer), (3, 2, 2))
        self.assertIs(ab.work, first.work)
        reg.sampling(ab)
        reg.finish(ab)
        self.assertEqual(len(reg), 0)
        self.assertEqual(a.pd_output_placeholder_owners, original_claims)

    def test_full_width_batch_lends_sequence_budget_and_counts_unfinished(self):
        a = out()
        s = scheduler(a)
        s.park_kv_retention_prefill(a)
        self.assertEqual(s.running, [])
        self.assertEqual(s.get_num_unfinished_requests(), 2)
        s.running = [row("C"), row("D")]
        self.assertFalse(s.can_resume_kv_retention_prefill(a))
        with self.assertRaises(RuntimeError):
            s.resume_kv_retention_prefill(a)
        s.running.clear()
        s.resume_kv_retention_prefill(a)
        self.assertEqual([r.request_id for r in s.running], ["A", "B"])
        self.assertEqual(s.get_num_unfinished_requests(), 2)

    def test_changed_incarnation_and_partial_discard_rejected(self):
        a = out()
        s = scheduler(a)
        s.park_kv_retention_prefill(a)
        with self.assertRaises(RuntimeError):
            s.discard_kv_retention_parked(out(("A",)))
        s.requests["A"] = row("A")
        with self.assertRaisesRegex(RuntimeError, "incarnation"):
            s.can_resume_kv_retention_prefill(a)

    def test_parked_work_not_a_second_future_but_retained_boundary_and_copy_are(self):
        a, _, _, boundary, _ = work()
        e = engine()
        e._ttft_prefill_suspended["parked:1"] = (boundary, a)
        q = {"prefill": deque(), "decode": deque([object()])}
        self.assertEqual(e._pd_owned_total_depth(q), 1)
        self.assertEqual(e._pd_owned_phase_depth(q, "prefill"), 0)
        e._kv_retention_parked_offload_future = (Future(), boundary, a)
        self.assertEqual(e._pd_owned_total_depth(q), 2)
        self.assertEqual(e._pd_owned_phase_depth(q, "prefill"), 1)
        e._kv_retention_parked_offload_future = None
        c = out(("C",))
        e._ttft_prefill_suspended["prefill"] = (W.boundary(c, None), c)
        self.assertEqual(e._pd_owned_total_depth(q), 2)
        self.assertEqual(e._pd_owned_phase_depth(q, "prefill"), 1)

    def test_flag_and_bounded_store(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(P.kv_retention_p1_parked_requested())
            self.assertEqual(P.kv_retention_parked_limits(), (1, 256 << 20))
        with patch.dict(os.environ, {"VLLM_KV_RETENTION_PARKED_MAX_BATCHES": "2"}):
            with self.assertRaises(ValueError):
                P.kv_retention_parked_limits()
        _, _, _, _, ticket = work()
        store = P.KVRetentionParkedStore(32)
        with self.assertRaises(RuntimeError):
            store.put(ticket, object())
        self.assertEqual(store.entries, {})


class WorkerTests(unittest.TestCase):
    def test_exact_parked_cleanup_cannot_clear_c_in_reused_slot(self):
        a, reg, ab, boundary, ticket = work()
        r = W.RUNNER()
        ctx = W.context()
        r._pd_phase_contexts = {("prefill", 0): ctx}
        r._kv_retention_parked_store = store = P.KVRetentionParkedStore(100)
        closed = []
        state = NS(kv_offload=NS(close=lambda: closed.append("copy")),
                   continuation=NS(close=lambda: closed.append("frame")))
        store.put(ticket, state)
        reg.suspend(ab, 2)
        reg.park(ab)
        c = out(("C",))
        cb = reg.start(c, W.epochs(c))
        r._claim_pd_execute_state("prefill", c, ctx, cb)
        co = ctx.kv_retention_execute_owner
        self.assertTrue(r.clear_pd_phase_execute_state("prefill", C.OWNER.from_output("prefill", a, ab)))
        self.assertEqual(closed, ["copy", "frame"])
        self.assertEqual(ctx.kv_retention_execute_owner, co)
        self.assertEqual(ctx.kv_retention_last_slot_generation, 2)
        self.assertEqual(store.entries, {})

    def test_resume_claim_imports_exact_frame_and_raises_generation(self):
        a, reg, ab, boundary, ticket = work()
        r = W.RUNNER()
        ctx = W.context()
        r._kv_retention_parked_store = store = P.KVRetentionParkedStore(100)
        state = NS(scheduler_output=a, boundary=boundary, kv_sampling_restore_pending=False)
        store.put(ticket, state)
        reg.suspend(ab, 2)
        reg.park(ab)
        c = out(("C",))
        cb = reg.start(c, W.epochs(c))
        reg.finish(cb)
        ctx.kv_retention_last_slot_generation = cb.slot_generation
        new = reg.resume(a, W.epochs(a))
        r._claim_pd_execute_state("prefill", a, ctx, new)
        self.assertIs(ctx.ttft_prefill_continuation, state)
        self.assertTrue(state.kv_sampling_restore_pending)
        self.assertEqual(ctx.kv_retention_execute_owner.binding, new)
        self.assertEqual(store.entries, {})

    def test_cleanup_failure_retains_store_and_stale_ticket_rejected(self):
        a, _, ab, _, ticket = work()
        store = P.KVRetentionParkedStore(100)
        def fail():
            raise RuntimeError("copy failed")
        store.put(ticket, NS(kv_offload=NS(close=fail), continuation=None))
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            store.close(ab)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            store.close(replace(ab, segment_id=99))
        self.assertEqual(len(store.entries), 1)

    def test_executor_park_ack_requires_exact_done_dependency(self):
        a, _, ab, _, _ = work()
        owner = C.OWNER.from_output("prefill", a, ab)
        x = EXEC()
        f = Future()
        x._pd_async_phase_execute_owners = {"prefill": owner}
        x._pd_async_phase_execute_futures = {"prefill": f}
        with self.assertRaises(RuntimeError):
            x.release_parked_execute_dependency(owner)
        f.set_result(None)
        x.release_parked_execute_dependency(owner)
        self.assertEqual(x._pd_async_phase_execute_futures, {})
        self.assertEqual(x._pd_async_phase_execute_owners, {})
        calls = []
        x.collective_rpc = lambda *args, **kwargs: calls.append((args, kwargs)) or Future()
        self.assertIsInstance(x.offload_parked_prefill(owner), Future)
        self.assertTrue(calls[0][1]["non_block"])
        self.assertEqual(calls[0][1]["pd_green_context_phase"], "prefill")


class Batch:
    def __init__(self, rows):
        self.max_num_reqs = 2
        self.req_ids = []
        self.num_computed_tokens_cpu = np.zeros(2, dtype=int)
        self.num_tokens_no_spec = np.zeros(2, dtype=int)
        self.operations = []
        self.in_progress_prompt_logprobs_cpu = {}
        for row in rows:
            self.add_request(row)
    def add_request(self, row):
        i = len(self.req_ids)
        self.req_ids.append(row.req_id)
        self.num_computed_tokens_cpu[i] = row.num_computed_tokens
        self.num_tokens_no_spec[i] = len(row.prompt_token_ids) + len(row.output_token_ids)
        self.operations.append(("add", row.req_id))
        return i
    def remove_request(self, req_id):
        self.req_ids.remove(req_id)
        self.in_progress_prompt_logprobs_cpu.pop(req_id, None)
        self.operations.append(("remove", req_id))
    def condense(self):
        pass
    def refresh_metadata(self):
        self.operations.append(("metadata", tuple(self.req_ids)))


def sampling_runner():
    rows = [NS(req_id=req_id, num_computed_tokens=2, prompt_token_ids=[1, 2, 3, 4],
               output_token_ids=[5], block_ids=([1, 2],), generator=object(),
               sampling_params=object()) for req_id in ("A", "B")]
    r = NS(requests={row.req_id: row for row in rows}, input_batch=Batch(rows))
    for name in ("input_ids", "positions", "query_start_loc", "seq_lens", "discard_request_mask"):
        b = NS(np=np.arange(8), uploads=[])
        b.copy_to_gpu = lambda n, b=b: b.uploads.append(b.np[:n].copy())
        setattr(r, name, b)
    return r


class SamplingTests(unittest.TestCase):
    def test_per_request_prompt_logprob_accumulation_survives_batch_removal(self):
        r = sampling_runner()
        history = object()
        r.input_batch.in_progress_prompt_logprobs_cpu["A"] = history
        frame = P.KVRetentionSamplingFrame.capture(r, NS(
            num_reqs=2, num_scheduled_tokens=4, scheduler_output=out()))
        r.input_batch.remove_request("A")
        r.input_batch.remove_request("B")
        r.input_batch.req_ids = ["C"]
        r.input_batch.in_progress_prompt_logprobs_cpu["C"] = object()
        frame.restore(r)
        self.assertEqual(r.input_batch.in_progress_prompt_logprobs_cpu, {"A": history})
        self.assertIsNone(r.input_batch.async_copy_ready_event)

    def test_restore_rows_masks_tokens_rng_after_c_overwrites_every_buffer(self):
        r = sampling_runner()
        state = NS(num_reqs=2, num_scheduled_tokens=4, scheduler_output=out())
        frame = P.KVRetentionSamplingFrame.capture(r, state)
        generators = [row.generator for row in frame.rows]
        r.input_batch.req_ids = ["C"]
        for name in frame.buffers:
            getattr(r, name).np.fill(999)
        frame.restore(r)
        self.assertEqual(r.input_batch.req_ids, ["A", "B"])
        self.assertEqual([r.requests[i].generator for i in r.input_batch.req_ids], generators)
        self.assertIn(("metadata", ()), r.input_batch.operations)
        for name, values in frame.buffers.items():
            np.testing.assert_equal(getattr(r, name).uploads[-1], values)
        np.testing.assert_equal(r.input_batch.num_computed_tokens_cpu, [2, 2])
        self.assertIsNone(r.input_batch.prev_sampled_token_ids)

    def test_changed_tokens_or_blocks_rejected_before_reusing_input_batch(self):
        for attr, value in (("output_token_ids", [99]), ("block_ids", ([9],))):
            r = sampling_runner()
            frame = P.KVRetentionSamplingFrame.capture(r, NS(
                num_reqs=2, num_scheduled_tokens=4, scheduler_output=out()))
            setattr(r.requests["A"], attr, value)
            r.input_batch.req_ids = ["C"]
            with self.assertRaisesRegex(RuntimeError, "changed"):
                frame.restore(r)
            self.assertEqual(r.input_batch.req_ids, ["C"])

    def test_custom_processors_and_multimodal_rejected(self):
        for attr, value in (("prompt_embeds", object()), ("mm_features", [1]),
                            ("lora_request", object()), ("pooling_params", object())):
            r = sampling_runner()
            setattr(r.requests["A"], attr, value)
            with self.assertRaises(RuntimeError):
                P.KVRetentionSamplingFrame.capture(r, NS(
                    num_reqs=2, num_scheduled_tokens=4, scheduler_output=out()))


class ProgressTests(unittest.TestCase):
    def test_resume_cannot_bypass_short_first_token_fence(self):
        a, _, _, b, _ = work()
        e = engine()
        e.scheduler = scheduler(a)
        e.scheduler.park_kv_retention_prefill(a)
        e._ttft_prefill_suspended = {"parked:1": (b, a)}
        e._pd_ttft_handoff_active = lambda: True
        e._pd_ttft_handoff_request_ids = {"C"}
        e._ttft_prefill_suspended_request_ids = lambda: {"A", "B"}
        self.assertFalse(e._try_resume_ttft_prefill({"prefill": deque(), "decode": deque()}))
        self.assertEqual(e._pd_ttft_handoff_blocked_prefill_checks, 1)
        self.assertEqual(e.scheduler.running, [])

    def test_bounded_submission_keeps_only_one_unfinished_old_event(self):
        import threading
        events, seen = [], []
        main_thread = threading.get_ident()
        class Event:
            def synchronize(self):
                seen.append(threading.get_ident())
        for _ in range(12):
            P.kv_retention_bound_prefill_submission(events, record_event=Event)
            self.assertEqual(len(events), 1)
        self.assertEqual(len(seen), 11)
        self.assertTrue(all(thread != main_thread for thread in seen))

    def test_bounded_submission_cuda_failure_preserves_event_proof(self):
        class ErrorEvent:
            def synchronize(self):
                raise RuntimeError("CUDA failed")
        events = []
        P.kv_retention_bound_prefill_submission(events, record_event=ErrorEvent)
        with self.assertRaisesRegex(RuntimeError, "CUDA failed"):
            P.kv_retention_bound_prefill_submission(events, record_event=ErrorEvent)
        self.assertEqual(len(events), 2)

    def test_engine_park_and_resume_preserve_original_output_and_claims(self):
        a, reg, first, b, ticket = work()
        e = engine()
        e.scheduler = scheduler(a)
        e._ttft_prefill_boundary_count = 0
        e._ttft_prefill_resume_submit_count = 0
        e._cancel_ttft_prefill_yield_for_output = lambda _: None
        e._pd_capacity_retry_gate = NS(clear=lambda: None)
        events = []
        f = Future()
        e.model_executor = NS(
            release_parked_execute_dependency=lambda owner: events.append(("release", owner)),
            execute_model=lambda output, **kw: events.append(("execute", output, kw)) or f)
        self.assertTrue(e._suspend_ttft_prefill_boundary(b, a))
        reg.suspend(first, 2)
        reg.park(first)
        self.assertTrue(a.kv_retention_p1_parked)
        self.assertEqual(e.scheduler.running, [])
        self.assertEqual(e._pd_owned_total_depth({"prefill": [], "decode": []}), 0)
        e.pd_dynamic_sm_enabled = e.pd_dynamic_sm_enforced = False
        e._kv_retention_parked_has_precedence_candidate = lambda *args: False
        e._kv_retention_bind_prefill = lambda output, resume: reg.resume(output, W.epochs(output))
        e._register_pd_model_future_completion = lambda *args, **kwargs: None
        e.full_iteration_debug_enabled = False
        q = {"prefill": deque(), "decode": deque()}
        self.assertTrue(e._try_resume_ttft_prefill(q, respect_srpt=True))
        self.assertIs(q["prefill"][0][1], a)
        self.assertEqual(events[1][2]["kv_retention_binding"].segment_id, 2)
        self.assertFalse(a.kv_retention_p1_parked)
        self.assertEqual(a.kv_retention_boundary_reason, "")
        self.assertEqual(e._ttft_prefill_suspended, {})
        self.assertEqual([r.request_id for r in e.scheduler.running], ["A", "B"])

    def test_sm_request_is_one_shot_even_after_worker_consumes_signal(self):
        e = engine()
        e.ttft_layerwise_prefill_preemption_enabled = True
        e.pd_sm_transition = NS(active=True)
        a = out()
        a.kv_retention_boundary_reason = ""
        f = Future()
        q = {"prefill": deque([(f, a, f)]), "decode": deque()}
        calls = []
        method = e._request_ttft_prefill_preemption_if_needed
        with patch.dict(method.__func__.__globals__, {
                "request_ttft_layerwise_prefill_yield": lambda **kw: calls.append(kw) or True}):
            self.assertTrue(method(q))
            self.assertFalse(method(q))
        self.assertEqual(len(calls), 1)
        self.assertEqual(a.kv_retention_boundary_reason, "sm")

    def test_late_copy_counts_in_sm_drain_until_receipt_consumed(self):
        a, _, _, b, _ = work()
        e = engine()
        e.pd_sm_transition = NS(active=True, source_epoch=1)
        f = Future()
        e._kv_retention_parked_offload_future = (f, b, a)
        q = {"prefill": deque(), "decode": deque()}
        self.assertEqual(e._pd_sm_transition_future_counts(q), (0, 1, 0))
        self.assertFalse(e._pd_recoverable_future_ready(q))
        f.set_result(None)
        self.assertEqual(e._pd_sm_transition_future_counts(q), (0, 1, 0))
        self.assertTrue(e._pd_recoverable_future_ready(q))

    def test_late_pressure_copy_is_async_and_attempted_once(self):
        a, _, _, b, _ = work()
        e = engine()
        e._ttft_prefill_suspended = {"parked:1": (b, a)}
        e._pd_capacity_retry_free_blocks = lambda: 0
        events = []
        e._pd_capacity_retry_gate = NS(blocking_cause=lambda *a: "prefill_kv_capacity",
                                      notify_state_change=lambda why: events.append(why))
        f = Future()
        calls = []
        e.model_executor = NS(offload_parked_prefill=lambda owner: calls.append(owner) or f)
        q = {"prefill": deque(), "decode": deque()}
        e._poll_kv_retention_parked_offload(q)
        self.assertEqual(e._pd_owned_total_depth(q), 1)
        e._poll_kv_retention_parked_offload(q)
        self.assertEqual(len(calls), 1)
        f.set_result(None)  # bounded host budget refusal; source KV retained
        e._poll_kv_retention_parked_offload(q)
        self.assertEqual(e._pd_owned_total_depth(q), 0)
        e._poll_kv_retention_parked_offload(q)
        self.assertEqual(len(calls), 1)
        self.assertEqual(events, ["parked_kv_copy_completed"])

    def test_whole_parked_abort_waits_for_c_then_cleans_before_scheduler_free(self):
        a, _, ab, b, _ = work()
        e = engine()
        e._ttft_prefill_suspended = {"parked:1": (b, a)}
        e.scheduler = scheduler(a)
        e.scheduler.park_kv_retention_prefill(a)
        e._ttft_prefill_deferred_abort_ids = {"A"}
        e._prune_ttft_prefill_suspended()
        self.assertEqual(len(e._ttft_prefill_suspended), 1)
        e._ttft_prefill_deferred_abort_ids.add("B")
        e.pd_phase_queues = {"prefill": deque([object()])}
        e._prune_ttft_prefill_suspended()
        self.assertEqual(len(e._ttft_prefill_suspended), 1)
        e.pd_phase_queues["prefill"].clear()
        events = []
        e._cancel_ttft_prefill_yield_for_output = lambda _: None
        e._cancel_ttft_prefill_worker_slot = lambda *args: events.append("worker_ack")
        e._discard_ttft_prefill_deferred_aborts = lambda _: e._ttft_prefill_deferred_abort_ids.clear()
        e._pd_capacity_retry_gate = NS(clear=lambda: None)
        e.scheduler.finish_requests = lambda *args: events.append("free")
        e._prune_ttft_prefill_suspended()
        self.assertEqual(events, ["worker_ack", "free"])
        self.assertEqual(e._ttft_prefill_suspended, {})
        self.assertEqual(e.scheduler._kv_retention_parked_requests, {})


class HBMOnlyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = H.GuardTests()
        self.guard = self.fixture.setup_guard()
        self.e = engine()
        self.e.kv_retention_p1_hbm_only = True
        self.e.kv_retention_layerwise_offload_enabled = False
        self.e._kv_retention_hbm_preemption_blocker = None
        self.e.ttft_layerwise_prefill_preemption_enabled = True
        self.e.pd_sm_transition = NS(active=False)
        self.e._ttft_prefill_preemption_signal_count = 0
        self.a = out(("A",))
        self.a.kv_retention_boundary_reason = ""
        f = Future()
        self.q = {"prefill": deque([(f, self.a, f)]), "decode": deque()}
        self.e._pd_phase_queue_pending_request_ids = lambda q: {"A"}
        self.e.scheduler = self.fixture.scheduler
        self.e.scheduler.isolation_hbm_guard = self.guard
        for req, remaining in ((H.request("A", 40, 20), 38),
                               (H.request("C", 20, 10), 20)):
            req.remaining, req.arrival_time = remaining, 0.0
            self.e.scheduler.requests[req.request_id] = req
        self.e.scheduler.ttft_prefill_chunk_scheduler = NS(
            policy="srpt", remaining_prefill_tokens=lambda r: r.remaining)
        self.fixture.tables["A"] = [1, 2]
        self.fixture.free = 8
        self.signals, self.traces = [], []
        self.e._write_pd_phase_queue_trace = lambda *args: self.traces.append(args)
        self.patch = patch.dict(
            self.e._request_ttft_prefill_preemption_if_needed.__func__.__globals__, {
                "should_preempt_ttft_layerwise_prefill_batch": C.M.should_preempt_ttft_layerwise_prefill_batch,
                "request_ttft_layerwise_prefill_yield": lambda **kw: self.signals.append(kw) or True})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_mode_default_off_and_requires_retained_hbm_guard(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(P.kv_retention_p1_hbm_only_requested())
        with patch.dict(os.environ, {"VLLM_KV_RETENTION_P1_HBM_ONLY": "typo"}):
            with self.assertRaises(ValueError):
                P.kv_retention_p1_hbm_only_requested()
        valid = dict(enabled=True, parked=True, offload=False,
                     completion_guard=self.guard, virtual_context=None)
        P.validate_kv_retention_p1_hbm_only(**valid)
        for bad in (dict(parked=False), dict(offload=True),
                    dict(completion_guard=None), dict(virtual_context=object())):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                P.validate_kv_retention_p1_hbm_only(**{**valid, **bad})
        P.validate_kv_retention_p1_hbm_only(**{**valid, "enabled": False, "offload": True})

    def test_short_candidate_with_room_yields_without_copy_or_free(self):
        tables = {key: list(value) for key, value in self.fixture.tables.items()}
        self.assertTrue(self.e._request_ttft_prefill_preemption_if_needed(self.q))
        self.assertEqual(len(self.signals), 1)
        self.assertEqual(self.a.kv_retention_boundary_reason, "request")
        self.assertFalse(self.a.kv_retention_layerwise_offload_requested)
        self.assertEqual(self.fixture.tables, tables)
        self.assertEqual(self.fixture.free, 8)

    def test_insufficient_capacity_keeps_active_work_and_logs_changed_blocker(self):
        # C fits alone, but A's retained allocation + future growth must count.
        self.fixture.free = 5
        for _ in range(3):
            self.assertFalse(self.e._request_ttft_prefill_preemption_if_needed(self.q))
        self.assertEqual(self.signals, [])
        self.assertEqual(self.a.kv_retention_boundary_reason, "")
        self.assertEqual(len(self.traces), 1)
        self.assertEqual(self.traces[0][0], "kv_retention_p1_hbm_preemption_capacity_wait")
        self.fixture.free = 8  # another owner finishes; candidate can now fit
        self.assertTrue(self.e._request_ttft_prefill_preemption_if_needed(self.q))

    def test_hbm_request_yield_is_one_shot_until_boundary_recovery(self):
        key = dict(scheduler_iteration_id=self.a.scheduler_iteration_id,
                   phase_submit_seq=self.a.pd_phase_submit_seq,
                   phase_slot=self.a.pd_phase_slot,
                   scheduled_tokens=tuple(self.a.num_scheduled_tokens.items()))
        C.M.cancel_ttft_layerwise_prefill_yield(**key)
        self.addCleanup(lambda: C.M.cancel_ttft_layerwise_prefill_yield(**key))
        method = self.e._request_ttft_prefill_preemption_if_needed
        with patch.dict(method.__func__.__globals__, {
                "request_ttft_layerwise_prefill_yield": C.M.request_ttft_layerwise_prefill_yield}):
            self.assertTrue(method(self.q))
            # Actual controller removal simulates the worker consuming the
            # signal before export finishes and the Future is recovered.
            self.assertTrue(C.M.cancel_ttft_layerwise_prefill_yield(**key))
            self.assertFalse(method(self.q))
            self.assertFalse(C.M.cancel_ttft_layerwise_prefill_yield(**key))

    def test_refused_signal_does_not_latch_or_lose_future_yield(self):
        method = self.e._request_ttft_prefill_preemption_if_needed
        with patch.dict(method.__func__.__globals__, {
                "request_ttft_layerwise_prefill_yield": lambda **kw: False}):
            self.assertFalse(method(self.q))
        self.assertFalse(self.a.kv_retention_layerwise_pressure_checked)
        self.assertTrue(method(self.q))

    def test_parked_rows_still_consume_full_growth_budget(self):
        # The guard deliberately counts physical tables, regardless of active
        # sequence-slot membership. Simulate parking by removing running rows.
        self.e.scheduler.running = []
        self.e.scheduler._kv_retention_parked_requests = {"A": object()}
        self.fixture.free = 5
        self.assertFalse(self.guard.can_admit(self.e.scheduler.requests["C"]))
        self.assertEqual(self.guard.last_required_blocks, 7)  # A:4 + C:3
        self.assertEqual(self.fixture.tables["A"], [1, 2])

    def test_hbm_mode_never_attempts_late_pressure_offload(self):
        self.e._pd_capacity_retry_free_blocks = lambda: self.fail("must not plan a copy")
        self.e._poll_kv_retention_parked_offload(self.q)
        self.e._kv_retention_parked_offload_future = (Future(), object(), self.a)
        with self.assertRaisesRegex(RuntimeError, "cannot own a KV copy"):
            self.e._poll_kv_retention_parked_offload(self.q)

    def test_worker_rejects_direct_offload_before_touching_owner_or_cuda(self):
        runner = C.adapters(C.RUNNER_PATH, "GPUModelRunner", {"offload_parked_prefill"})()
        runner._kv_retention_p1_hbm_only = True
        with self.assertRaisesRegex(RuntimeError, "cannot offload KV"):
            runner.offload_parked_prefill(None)

    def test_changed_admission_can_fall_back_to_original_parked_batch(self):
        a, _, _, b, _ = work()
        self.e.scheduler = scheduler(a)
        self.e.scheduler.park_kv_retention_prefill(a)
        self.e._ttft_prefill_suspended = {"parked:1": (b, a)}
        self.e._kv_retention_parked_has_precedence_candidate = lambda *args: True
        self.q["prefill"].clear()
        # Normal SRPT defers to the new candidate, but failed admission must
        # not hold the original work hostage. This is the Engine fallback.
        key, reason = self.e._ttft_prefill_resume_candidate(self.q, respect_srpt=True)
        self.assertIsNone(key)
        self.assertEqual(reason, "srpt_precedence")
        key, reason = self.e._ttft_prefill_resume_candidate(self.q, respect_srpt=False)
        self.assertEqual((key, reason), ("parked:1", "ready"))


class ActualForwardParkingTests(unittest.TestCase):
    """Actual frame export/import + claim/cleanup + synthetic Llama layers."""
    def setUp(self):
        F.ActualRunnerTest.setUp(self)
        self.out.kv_retention_prefill_work = None
        self.out.kv_retention_boundary_reason = "request"
        self.registry = W.REGISTRY(1, parked_capacity=1)
        self.first = self.registry.start(self.out, W.epochs(self.out))
        self.ctx.kv_retention_execute_owner = C.OWNER.from_output("prefill", self.out, self.first)
        self.ctx.kv_retention_last_slot_generation = 1
        self.ctx.execute_model_state = None
        self.runner._kv_retention_parked_store = P.KVRetentionParkedStore(4096)
        self.runner._pd_phase_contexts = {("prefill", 0): self.ctx}
        rows = sampling_runner()
        self.runner.requests, self.runner.input_batch = rows.requests, rows.input_batch
        for name in ("input_ids", "positions", "query_start_loc", "seq_lens", "discard_request_mask"):
            setattr(self.runner, name, getattr(rows, name))
        self.globals_patch = patch.dict(self.runner._capture_kv_retention_resume_frame.__func__.__globals__, {
            "KVRetentionSamplingFrame": P.KVRetentionSamplingFrame,
            "KVRetentionParkedTicket": P.KVRetentionParkedTicket})
        self.globals_patch.start()

    def tearDown(self):
        self.globals_patch.stop()
        F.ActualRunnerTest.tearDown(self)

    def test_a_park_c_overwrite_a_remaining_layers_finalize_once(self):
        self.runner._capture_kv_retention_resume_frame(self.state)
        store = self.runner._kv_retention_parked_store
        self.assertIsNone(self.ctx.ttft_prefill_continuation)
        self.assertIsNone(self.ctx.kv_retention_execute_owner)
        self.assertEqual(len(store.entries), 1)
        ticket = self.state.boundary.kv_retention_parked_ticket
        ticket.validate(self.state.boundary, self.out)
        self.registry.suspend(self.first, 2)
        self.registry.park(self.first)
        c = out(("C",))
        cb = self.registry.start(c, W.epochs(c))
        W.RUNNER._claim_pd_execute_state(self.runner, "prefill", c, self.ctx, cb)
        # C can overwrite every slot-owned input and workspace after ACK.
        self.runner.input_batch.req_ids = ["C"]
        for name in self.state.kv_sampling_frame.buffers:
            getattr(self.runner, name).np.fill(999)
        self.registry.finish(cb)
        self.ctx.kv_retention_execute_owner = None
        new = self.registry.resume(self.out, W.epochs(self.out))
        W.RUNNER._claim_pd_execute_state(self.runner, "prefill", self.out, self.ctx, new)
        self.assertEqual(store.entries, {})
        self.assertTrue(self.state.kv_sampling_restore_pending)
        self.state.kv_sampling_frame.restore(self.runner)
        self.state.kv_sampling_restore_pending = False
        self.state.kv_sampling_frame = None
        self.out.kv_retention_boundary_reason = ""
        self.cont.execution.note_resume()
        # No further interrupt is pending: finish remaining layers in one call.
        self.cont.execution.consume_yield_request = lambda: False
        finals = []
        self.runner._finish_execute_model_forward = lambda **kwargs: finals.append(kwargs)
        self.runner._resume_ttft_prefill_continuation(self.out, self.state)
        self.assertEqual(len(finals), 1)
        self.assertIs(finals[0]["scheduler_output"], self.out)
        self.assertEqual(self.runner.input_batch.req_ids, ["A", "B"])
        self.assertTrue(self.cont.closed)
        self.registry.sampling(new)
        self.registry.finish(new)
        self.assertEqual(len(self.registry), 0)

    def test_sm_only_yield_never_clones_or_lends_slot(self):
        self.out.kv_retention_boundary_reason = "sm"
        with patch.object(F.F.KVRetentionResumeFrame, "capture", side_effect=AssertionError("copy")):
            self.runner._capture_kv_retention_resume_frame(self.state)
        self.assertIs(self.ctx.ttft_prefill_continuation, self.state)
        self.assertEqual(self.runner._kv_retention_parked_store.entries, {})
        self.assertIsNone(self.state.boundary.kv_retention_parked_ticket)

    def test_budget_decline_does_not_publish_park_ack(self):
        self.runner._kv_retention_parked_store.max_gpu_bytes = 1
        self.runner._capture_kv_retention_resume_frame(self.state)
        self.assertTrue(self.out.kv_retention_p1_park_declined)
        self.assertIs(self.ctx.ttft_prefill_continuation, self.state)
        self.assertIsNone(self.state.boundary.kv_retention_parked_ticket)
        self.assertEqual(self.runner._kv_retention_parked_store.entries, {})


if __name__ == "__main__":
    unittest.main()
