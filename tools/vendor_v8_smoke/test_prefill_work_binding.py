#!/usr/bin/env python3
"""CPU state/real-adapter tests; no claims about GPU execution or P1 parking."""
from __future__ import annotations

import ast
import contextlib
import os
import time
import unittest
from collections import deque
from concurrent.futures import Future
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace as NS
from typing import cast
from unittest.mock import patch

import test_exact_phase_cleanup as C

M = C.M
FLAG = "VLLM_KV_RETENTION_PREFILL_WORK_BINDING"
REGISTRY = M.KVRetentionPrefillWorkRegistry
RUNNER = C.adapters(C.RUNNER_PATH, "GPUModelRunner", {
    "_claim_pd_execute_state", "clear_pd_phase_execute_state",
    "_validate_kv_retention_sampling"})
ENGINE = C.adapters("vendor_v8/vllm/v1/engine/core.py", "EngineCore", {
    "_kv_retention_bind_prefill", "_kv_retention_future_binding",
    "_kv_retention_binding_debug", "_cancel_ttft_prefill_worker_slot",
    "_try_resume_ttft_prefill", "_enqueue_ttft_prefill_final_sampling",
    "_ttft_prefill_resume_candidate",
}, {"time": time, "cast": cast, "ModelRunnerOutput": object})
WRAPPER = C.adapters("vendor_v8/vllm/v1/worker/worker_base.py", "WorkerWrapperBase",
                     {"execute_model"})


class TieredError(RuntimeError):
    pass


class ModelOutput:
    pass


WORKER = C.adapters("vendor_v8/vllm/v1/worker/gpu_worker.py", "Worker",
    {"execute_model", "sample_tokens"}, {
        "get_pp_group": lambda: NS(is_first_rank=True),
        "TieredBackendEligibilityError": TieredError,
        "TTFTLayerwisePrefillBoundary": M.TTFTLayerwisePrefillBoundary,
        "ModelRunnerOutput": ModelOutput, "AsyncModelRunnerOutput": ModelOutput,
        "NoneType": type(None), "torch": NS(cuda=NS(synchronize=lambda _: None)),
    })


def output(slot=0, ids=("A", "B")):
    return NS(
        pd_phase_slot=slot, scheduler_iteration_id=2, pd_phase_submit_seq=3,
        pd_sm_profile_id="d30_p10", pd_sm_profile_epoch=1,
        pd_prefill_mode="native",
        scheduled_new_reqs=[NS(req_id=ids[0], num_computed_tokens=8)],
        scheduled_cached_reqs=NS(req_ids=list(ids[1:]),
                                num_computed_tokens=[16] * (len(ids) - 1)),
        num_scheduled_tokens=dict.fromkeys(ids, 4),
        total_num_scheduled_tokens=len(ids) * 4,
        pd_output_ownership_enabled=True,
        pd_output_placeholder_owners={ids[-1]: (11, 7, "prefill", 1)},
        kv_retention_prefill_work=None, ttft_layerwise_prefill_preemption=True,
        kv_retention_layerwise_pressure_checked=False,
    )


def epochs(out):
    return dict.fromkeys(out.num_scheduled_tokens, 11)


def boundary(out, binding, layer=2):
    return M.TTFTLayerwisePrefillBoundary(
        out.scheduler_iteration_id, out.pd_phase_submit_seq, out.pd_phase_slot,
        tuple(out.num_scheduled_tokens), tuple(out.num_scheduled_tokens.items()),
        out.pd_sm_profile_id, out.pd_sm_profile_epoch, "native", layer, 16, 1,
        kv_retention_binding=binding)


def context():
    return NS(kv_retention_execute_owner=None, kv_retention_last_slot_generation=0,
              execute_model_state=None, ttft_prefill_continuation=None)


def executor():
    obj = C.EXECUTOR()
    obj.pd_async_uniproc_enabled = True
    obj._pd_async_phase_execute_futures = {}
    obj._pd_async_phase_execute_owners = {}
    return obj


class WorkRegistryTest(unittest.TestCase):
    def setUp(self):
        self.registry = REGISTRY(2)
        self.out = output()
        self.first = self.registry.start(self.out, epochs(self.out))

    def test_frozen_logical_owner_uses_prechunk_positions_and_claims(self):
        self.assertEqual(self.first.work.rows, (("A", 11, 8, 4), ("B", 11, 16, 4)))
        self.assertEqual(self.first.work.claims, (("B", 11, 7, "prefill", 1),))
        with self.assertRaises(FrozenInstanceError):
            self.first.work.work_id = 100
        with self.assertRaises(FrozenInstanceError):
            self.first.segment_id = 7

    def test_a_suspend_c_complete_a_resume_preserves_transaction(self):
        original = self.first.work
        self.registry.suspend(self.first, 4)
        c = output(1, ("C",))
        c_binding = self.registry.start(c, epochs(c))
        self.registry.sampling(c_binding)
        self.registry.finish(c_binding)
        self.out.pd_sm_profile_id, self.out.pd_sm_profile_epoch = "d0_p40", 2
        second = self.registry.resume(self.out, epochs(self.out))
        self.assertIs(second.work, original)
        self.assertEqual((second.segment_id, second.slot_generation, second.start_layer), (2, 2, 4))
        self.registry.sampling(second)
        self.registry.finish(second)
        self.assertEqual(len(self.registry), 0)
        self.assertEqual(self.out.pd_output_placeholder_owners, {"B": (11, 7, "prefill", 1)})

    def test_stale_segment_cannot_finish_suspend_or_sample_newer_segment(self):
        self.registry.suspend(self.first, 2)
        second = self.registry.resume(self.out, epochs(self.out))
        for action in (lambda: self.registry.finish(self.first),
                       lambda: self.registry.suspend(self.first, 4),
                       lambda: self.registry.sampling(self.first)):
            with self.assertRaisesRegex(RuntimeError, "stale"):
                action()
            self.assertEqual(self.registry.require(second).state, "active")

    def test_duplicate_completion_and_registration_fail(self):
        with self.assertRaises(RuntimeError):
            self.registry.start(self.out, epochs(self.out))
        self.registry.finish(self.first)
        with self.assertRaises(RuntimeError):
            self.registry.finish(self.first)

    def test_work_budget_includes_suspended_slots(self):
        self.registry.suspend(self.first, 2)
        for other in (output(0, ("C",)), output(1, ("A",))):
            with self.assertRaises(RuntimeError):
                self.registry.start(other, epochs(other))
            self.assertIsNone(other.kv_retention_prefill_work)
        other = output(1, ("C",))
        self.registry.start(other, epochs(other))
        with self.assertRaises(RuntimeError):
            more = output(2, ("D",))
            self.registry.start(more, epochs(more))
        self.assertEqual(len(self.registry), 2)

    def test_changed_order_range_claim_or_epoch_rejects_resume(self):
        edits = (
            lambda o: setattr(o, "num_scheduled_tokens", {"B": 4, "A": 4}),
            lambda o: setattr(o.scheduled_new_reqs[0], "num_computed_tokens", 9),
            lambda o: o.pd_output_placeholder_owners.update({"B": (11, 8, "prefill", 1)}),
            lambda o: setattr(o, "pd_phase_submit_seq", 99),
        )
        for edit in edits:
            with self.subTest(edit=edit):
                registry, out = REGISTRY(2), output()
                b = registry.start(out, epochs(out))
                registry.suspend(b, 2)
                edit(out)
                with self.assertRaises(RuntimeError):
                    registry.resume(out, epochs(out))
                self.assertEqual(registry.require(b).state, "suspended")
        self.registry.suspend(self.first, 2)
        with self.assertRaisesRegex(RuntimeError, "epoch"):
            self.registry.resume(self.out, {"A": 12, "B": 11})

    def test_malformed_input_never_publishes_work(self):
        for edit in (
                lambda o: setattr(o, "total_num_scheduled_tokens", 9),
                lambda o: setattr(o, "pd_output_ownership_enabled", False),
                lambda o: setattr(o.scheduled_new_reqs[0], "num_computed_tokens", -1),
                lambda o: setattr(o.scheduled_cached_reqs, "num_computed_tokens", []),
                lambda o: o.pd_output_placeholder_owners.update({"B": (12, 7, "prefill", 1)})):
            registry, out = REGISTRY(2), output()
            edit(out)
            with self.assertRaises(RuntimeError):
                registry.start(out, epochs(out))
            self.assertEqual(len(registry), 0)
            self.assertIsNone(out.kv_retention_prefill_work)

    def test_sampling_and_cancel_terminal_are_distinct(self):
        self.registry.suspend(self.first, 2)
        with self.assertRaises(RuntimeError):
            self.registry.sampling(self.first)
        with self.assertRaises(RuntimeError):
            self.registry.finish(self.first)
        self.assertEqual(len(self.registry), 1)  # Waiting/timeout has no release API.
        self.registry.finish(self.first, cancelled=True)
        self.assertEqual(len(self.registry), 0)

    def test_profile_epoch_slot_and_generation_checked(self):
        self.registry.suspend(self.first, 2)
        self.out.pd_sm_profile_id = "d0_p40"
        with self.assertRaises(RuntimeError):
            self.registry.resume(self.out, epochs(self.out))
        self.out.pd_sm_profile_epoch = 2
        self.out.pd_phase_slot = 1
        with self.assertRaises(RuntimeError):
            self.registry.resume(self.out, epochs(self.out))
        for b in (replace(self.first, slot_generation=0), replace(self.first, segment_id=0)):
            with self.assertRaises(RuntimeError):
                b.validate_output("prefill", self.out)


class RunnerBindingTest(unittest.TestCase):
    def setUp(self):
        self.runner, self.ctx = RUNNER(), context()
        self.runner._pd_phase_contexts = {("prefill", 0): self.ctx}
        self.registry, self.out = REGISTRY(2), output()
        self.binding = self.registry.start(self.out, epochs(self.out))
        self.runner._claim_pd_execute_state("prefill", self.out, self.ctx, self.binding)

    def suspend(self):
        self.ctx.ttft_prefill_continuation = NS(
            scheduler_output=self.out, boundary=boundary(self.out, self.binding),
            continuation=NS(close=lambda: None), kv_offload=None)
        self.registry.suspend(self.binding, 2)

    def test_same_work_successor_accepts_new_sm_and_old_cleanup_is_noop(self):
        self.suspend()
        old_owner = self.ctx.kv_retention_execute_owner
        self.out.pd_sm_profile_id, self.out.pd_sm_profile_epoch = "d0_p40", 2
        new = self.registry.resume(self.out, epochs(self.out))
        self.runner._claim_pd_execute_state("prefill", self.out, self.ctx, new)
        self.assertFalse(self.runner.clear_pd_phase_execute_state("prefill", old_owner))
        self.assertEqual(self.ctx.kv_retention_execute_owner.binding, new)

    def test_cleanup_keeps_generation_highwater_and_replay_is_rejected(self):
        self.suspend()
        owner = self.ctx.kv_retention_execute_owner
        self.assertTrue(self.runner.clear_pd_phase_execute_state("prefill", owner))
        self.assertEqual(self.ctx.kv_retention_last_slot_generation, 1)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.runner._claim_pd_execute_state("prefill", self.out, self.ctx, self.binding)

    def test_missing_binding_never_enters_preparation(self):
        with self.assertRaisesRegex(RuntimeError, "explicit"):
            self.runner._claim_pd_execute_state("prefill", self.out, self.ctx)

    def test_resume_without_frame_and_pending_sampler_are_rejected(self):
        self.registry.suspend(self.binding, 2)
        new = self.registry.resume(self.out, epochs(self.out))
        with self.assertRaisesRegex(RuntimeError, "no retained"):
            self.runner._claim_pd_execute_state("prefill", self.out, self.ctx, new)
        self.ctx.execute_model_state = NS(scheduler_output=self.out)
        with self.assertRaisesRegex(RuntimeError, "pending sampling"):
            self.runner._claim_pd_execute_state("prefill", self.out, self.ctx, new)

    def test_sampling_requires_exact_final_segment(self):
        with self.assertRaises(RuntimeError):
            self.runner._validate_kv_retention_sampling("prefill", self.ctx, self.binding)
        self.ctx.execute_model_state = NS(scheduler_output=self.out)
        self.runner._validate_kv_retention_sampling("prefill", self.ctx, self.binding)
        with self.assertRaises(RuntimeError):
            self.runner._validate_kv_retention_sampling("prefill", self.ctx, None)
        with self.assertRaises(RuntimeError):
            self.runner._validate_kv_retention_sampling(
                "prefill", self.ctx, replace(self.binding, segment_id=2))

    def test_cleanup_failure_does_not_remove_exact_owner(self):
        self.suspend()
        def fail():
            raise RuntimeError("copy failed")
        self.ctx.ttft_prefill_continuation.kv_offload = NS(close=fail)
        owner = self.ctx.kv_retention_execute_owner
        with self.assertRaises(RuntimeError):
            self.runner.clear_pd_phase_execute_state("prefill", owner)
        self.assertEqual(self.ctx.kv_retention_execute_owner, owner)


class TransportTest(unittest.TestCase):
    def setUp(self):
        self.out, self.registry = output(), REGISTRY(2)
        self.binding = self.registry.start(self.out, epochs(self.out))
        self.executor = executor()

    def test_execute_rpc_captures_binding_on_exact_future(self):
        calls = []
        def rpc(method, **kwargs):
            calls.append((method, kwargs))
            return Future()
        self.executor.collective_rpc = rpc
        f = self.executor.execute_model(self.out, True, "prefill", self.binding)
        self.assertIs(f.kv_retention_binding, self.binding)
        self.assertEqual(calls[0][1]["kwargs"], {"kv_retention_binding": self.binding})
        self.assertEqual(self.executor._pd_async_phase_execute_owners["prefill"].binding,
                         self.binding)
        sampled = self.executor.sample_tokens(None, True, "prefill", self.binding)
        self.assertIs(sampled.kv_retention_binding, self.binding)
        self.assertIs(calls[1][1]["dependency"], f)
        self.assertEqual(calls[1][1]["kwargs"]["kv_retention_binding"], self.binding)

    def test_stale_sampler_does_not_pop_execute_dependency(self):
        self.executor.collective_rpc = lambda *a, **k: Future()
        f = self.executor.execute_model(self.out, True, "prefill", self.binding)
        for b in (None, replace(self.binding, segment_id=2)):
            with self.assertRaises(RuntimeError):
                self.executor.sample_tokens(None, True, "prefill", b)
            self.assertIs(self.executor._pd_async_phase_execute_futures["prefill"], f)

    def test_actual_wrapper_and_gpu_worker_forward_explicit_binding(self):
        seen = []
        runner = NS(execute_model=lambda *a, **kw: seen.append(kw),
                    sample_tokens=lambda *a, **kw: seen.append(kw))
        worker = WORKER()
        worker.model_runner = runner
        worker.vllm_config = NS(compilation_config=NS(),
                                parallel_config=NS(pipeline_parallel_size=1))
        worker.annotate_profile = lambda _: contextlib.nullcontext()
        wrapper = WRAPPER()
        wrapper.worker, wrapper._apply_mm_cache = worker, lambda _: None
        wrapper.execute_model(self.out, "prefill", self.binding)
        worker.sample_tokens(None, "prefill", self.binding)
        self.assertTrue(all(kw["kv_retention_binding"] is self.binding for kw in seen))
        seen.clear()
        wrapper.execute_model(self.out, "prefill")
        worker.sample_tokens(None, "prefill")
        self.assertTrue(all("kv_retention_binding" not in kw for kw in seen))

    def test_worker_typed_failure_cleans_captured_segment(self):
        seen = []
        def fail(*a, **kw):
            raise TieredError("tiered")
        worker = WORKER()
        worker.model_runner = NS(execute_model=fail,
                                 clear_pd_phase_execute_state=lambda *a: seen.append(a))
        worker.vllm_config = NS(compilation_config=NS(),
                                parallel_config=NS(pipeline_parallel_size=1))
        worker.annotate_profile = lambda _: contextlib.nullcontext()
        worker.device = "fake"
        with self.assertRaises(TieredError):
            worker.execute_model(self.out, "prefill", self.binding)
        self.assertEqual(seen[0][1].binding, self.binding)

    def test_profile_change_before_rpc_rejected_without_submit(self):
        self.executor.collective_rpc = lambda *a, **kw: self.fail("unexpected submit")
        self.out.pd_sm_profile_epoch += 1
        with self.assertRaises(RuntimeError):
            self.executor.execute_model(self.out, True, "prefill", self.binding)


class EngineBindingTest(unittest.TestCase):
    def setUp(self):
        self.engine, self.out = ENGINE(), output()
        self.registry = REGISTRY(2)
        self.engine._kv_retention_prefill_work_registry = self.registry
        self.engine.scheduler = NS(requests={k: NS(pd_output_placeholder_ledger=NS(
            request_epoch=11)) for k in self.out.num_scheduled_tokens},
            get_grammar_bitmask=lambda _: None)
        self.engine.model_executor = executor()
        self.engine.model_executor.collective_rpc = lambda *a, **kw: Future()
        self.binding = self.engine._kv_retention_bind_prefill(self.out)
        self.future = self.engine.model_executor.execute_model(
            self.out, True, "prefill", self.binding)

    def test_recovery_rejects_missing_old_binding_and_new_request_epoch(self):
        self.assertIs(self.engine._kv_retention_future_binding(self.future, self.out), self.binding)
        with self.assertRaises(RuntimeError):
            self.engine._kv_retention_future_binding(Future(), self.out)
        self.registry.suspend(self.binding, 2)
        self.registry.resume(self.out, epochs(self.out))
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.engine._kv_retention_future_binding(self.future, self.out)

    def test_result_checks_epoch_before_output_commit(self):
        self.engine.scheduler.requests["A"].pd_output_placeholder_ledger.request_epoch = 12
        with self.assertRaisesRegex(RuntimeError, "old request epoch"):
            self.engine._kv_retention_future_binding(self.future, self.out)

    def test_cancel_removes_registry_only_after_exact_ack(self):
        self.registry.suspend(self.binding, 2)
        def clear(key, owner):
            self.assertEqual(len(self.registry), 1)
            self.assertEqual(owner.binding, self.binding)
            return True
        self.engine.model_executor.clear_pd_phase_execute_state = clear
        self.engine._cancel_ttft_prefill_worker_slot("prefill", self.out, self.binding)
        self.assertEqual(len(self.registry), 0)

    def test_false_or_failed_ack_retains_registry(self):
        for result in (False, RuntimeError("cleanup")):
            def clear(*a):
                if isinstance(result, Exception):
                    raise result
                return result
            self.engine.model_executor.clear_pd_phase_execute_state = clear
            with self.assertRaises(RuntimeError):
                self.engine._cancel_ttft_prefill_worker_slot("prefill", self.out, self.binding)
            self.assertEqual(len(self.registry), 1)

    def setup_queue_fields(self):
        self.engine._ttft_prefill_suspended = {}
        self.engine._iteration_debug_timings = {}
        self.engine.full_iteration_debug_enabled = False
        self.engine._ttft_prefill_execution_key = lambda _: "prefill"
        self.engine._ttft_prefill_final_execute_count = 0
        self.engine._register_pd_model_future_completion = lambda future, *a, **kw: (
            self.assertIsNotNone(future.kv_retention_binding))

    def test_sampling_keeps_logical_work_and_captured_segment(self):
        self.setup_queue_fields()
        queues = {"prefill": deque(), "decode": deque()}
        self.engine._enqueue_ttft_prefill_final_sampling(queues, self.out, self.future)
        self.assertEqual(len(self.registry), 1)
        self.assertEqual(self.registry.require(self.binding).state, "sampling")
        sampled, out, original = queues["prefill"][0]
        self.assertEqual(sampled.kv_retention_binding, self.binding)
        self.assertIs(original, self.future)
        self.assertIs(out, self.out)

    def test_actual_engine_resume_submits_new_segment_and_preserves_output(self):
        self.setup_queue_fields()
        self.registry.suspend(self.binding, 2)
        self.engine._ttft_prefill_suspended["prefill"] = (
            boundary(self.out, self.binding), self.out)
        self.engine._prune_ttft_prefill_suspended = lambda: None
        self.engine.pd_dynamic_sm_enabled = False
        self.engine.pd_dynamic_sm_enforced = False
        self.engine._ttft_prefill_resume_submit_count = 0
        self.engine._write_pd_phase_queue_trace = lambda *a: None
        queues = {"prefill": deque(), "decode": deque()}
        self.assertTrue(self.engine._try_resume_ttft_prefill(queues))
        future, out, _ = queues["prefill"][0]
        self.assertIs(out, self.out)
        self.assertEqual(future.kv_retention_binding.segment_id, 2)
        self.assertEqual(future.kv_retention_binding.work, self.binding.work)

    def test_flag_off_does_not_register_or_forward_metadata(self):
        engine = ENGINE()
        engine._kv_retention_prefill_work_registry = None
        out = output()
        self.assertIsNone(engine._kv_retention_bind_prefill(out))
        self.assertIsNone(engine._kv_retention_future_binding(Future(), out))
        self.assertIsNone(out.kv_retention_prefill_work)


class WiringContractTest(unittest.TestCase):
    def test_actual_adapters_a_suspend_c_sample_a_resume_and_sample(self):
        """P2 integration with CPU model doubles, not a same-slot P1 test."""
        registry, runner = REGISTRY(2), RUNNER()
        contexts = {("prefill", 0): context(), ("prefill", 1): context()}
        runner._pd_phase_contexts = contexts
        retained = []

        def model_execute(out, _intermediate, *, pd_green_context_phase,
                          kv_retention_binding):
            binding = kv_retention_binding
            ctx = contexts[("prefill", binding.slot)]
            runner._claim_pd_execute_state(pd_green_context_phase, out, ctx, binding)
            if "A" in out.num_scheduled_tokens and binding.segment_id == 1:
                b = boundary(out, binding)
                ctx.ttft_prefill_continuation = NS(
                    scheduler_output=out, boundary=b, kv_offload=None,
                    continuation=NS(close=lambda: retained.append("closed")))
                return b
            ctx.ttft_prefill_continuation = None
            ctx.execute_model_state = NS(scheduler_output=out)
            return None

        def model_sample(grammar, *, pd_green_context_phase, kv_retention_binding):
            ctx = contexts[("prefill", kv_retention_binding.slot)]
            runner._validate_kv_retention_sampling(
                pd_green_context_phase, ctx, kv_retention_binding)
            ctx.execute_model_state = None
            return ModelOutput()

        worker, wrapper, ex = WORKER(), WRAPPER(), executor()
        worker.model_runner = NS(execute_model=model_execute, sample_tokens=model_sample)
        worker.vllm_config = NS(compilation_config=NS(),
                                parallel_config=NS(pipeline_parallel_size=1))
        worker.annotate_profile = lambda _: contextlib.nullcontext()
        wrapper.worker, wrapper._apply_mm_cache = worker, lambda _: None

        def rpc(method, *, args, kwargs=None, **_):
            f = Future()
            target = wrapper if method == "execute_model" else worker
            try:
                f.set_result(getattr(target, method)(*args, **(kwargs or {})))
            except Exception as exc:
                f.set_exception(exc)
            return f

        ex.collective_rpc = rpc
        a, c = output(), output(1, ("C",))
        b1 = registry.start(a, epochs(a))
        fa = ex.execute_model(a, True, "prefill", b1)
        self.assertEqual(fa.result().kv_retention_binding, b1)
        registry.suspend(b1, fa.result().next_layer)
        bc = registry.start(c, epochs(c))
        fc = ex.execute_model(c, True, "prefill:1", bc)
        self.assertIsNone(fc.result())
        registry.sampling(bc)
        sc = ex.sample_tokens(None, True, "prefill:1", bc)
        self.assertIsInstance(sc.result(), ModelOutput)
        registry.finish(sc.kv_retention_binding)
        self.assertIsNotNone(contexts[("prefill", 0)].ttft_prefill_continuation)
        a.pd_sm_profile_id, a.pd_sm_profile_epoch = "d0_p40", 2
        b2 = registry.resume(a, epochs(a))
        f2 = ex.execute_model(a, True, "prefill", b2)
        self.assertIsNone(f2.result())
        self.assertFalse(runner.clear_pd_phase_execute_state(
            "prefill", M.KVRetentionPhaseOwner.from_output("prefill", a, b1)))
        registry.sampling(b2)
        sa = ex.sample_tokens(None, True, "prefill", b2)
        self.assertIsInstance(sa.result(), ModelOutput)
        registry.finish(sa.kv_retention_binding)
        self.assertEqual(len(registry), 0)
        self.assertEqual(ex._pd_async_phase_execute_futures, {})
        self.assertEqual(ex._pd_async_phase_execute_owners, {})
        self.assertEqual(retained, [])  # Stale cleanup did not close the new owner.

    def test_default_off_and_stage2_required(self):
        for value in ("", "0", "off", "false"):
            with patch.dict(os.environ, {FLAG: value}):
                self.assertFalse(M.kv_retention_prefill_work_binding_requested())
        with patch.dict(os.environ, {FLAG: "1"}):
            self.assertTrue(M.kv_retention_prefill_work_binding_requested())
        source = (C.ROOT / "vendor_v8/vllm/v1/engine/core.py").read_text()
        start = source.index("        requested_work_binding =")
        end = source.index("        self.kv_retention_layerwise_offload_enabled", start)
        import textwrap
        code = compile(textwrap.dedent(source[start:end]), "engine_init_binding", "exec")
        with self.assertRaisesRegex(RuntimeError, "Stage-2"):
            exec(code, {"self": NS(ttft_layerwise_prefill_preemption_enabled=False,
                                  kv_retention_p1_parked_enabled=False),
                        "kv_retention_prefill_work_binding_requested": lambda: True})
        self.assertIn("self.pd_prefill_inflight_depth >= 2", source)
        self.assertIn("self.kv_retention_p1_parked_enabled and self.pd_prefill_inflight_depth == 1", source)

    def test_binding_recovered_before_output_commit_and_terminal_after_commit(self):
        source = (C.ROOT / "vendor_v8/vllm/v1/engine/core.py").read_text()
        source = source.split("    def step_with_pd_phase_queues(", 1)[1].split("\n    def ", 1)[0]
        self.assertLess(source.index("binding = self._kv_retention_future_binding"),
                        source.index("model_output = future.result()"))
        success = source.split("assert model_output is not None", 1)[1]
        self.assertLess(success.index("self.scheduler.update_from_output("),
                        success.index("self._kv_retention_prefill_work_registry.finish(binding)"))
        failure = source.split("if tiered_eligibility_error is not None:", 1)[1]
        self.assertLess(failure.index("self._fail_tiered_backend_scheduler_output("),
                        failure.index("self._kv_retention_prefill_work_registry.finish(binding)"))

    def test_worker_echoes_binding_and_validates_before_sample_mutation(self):
        source = (C.ROOT / C.RUNNER_PATH).read_text()
        self.assertIn("output.kv_retention_binding = kv_retention_binding", source)
        body = source.split("    def sample_tokens(\n", 1)[1].split("\n    def ", 1)[0]
        self.assertLess(body.index("self._validate_kv_retention_sampling("),
                        body.index("self._pd_green_context_sample_phase = None"))
        output_source = (C.ROOT / "vendor_v8/vllm/v1/core/sched/output.py").read_text()
        self.assertIn("kv_retention_prefill_work:", output_source)
        self.assertNotIn("kv_retention_binding:", output_source)


if __name__ == "__main__":
    unittest.main()
