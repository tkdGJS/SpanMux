#!/usr/bin/env python3
"""CPU tests of real Ring SM-only Engine/Runner paths; not CUDA E2E proof."""
from __future__ import annotations

import ast
import os
import unittest
from collections import deque
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import test_exact_phase_cleanup as C
import test_p1_parked_prefill as P
import test_prefill_resume_frame as F
import test_prefill_work_binding as W


ALLOWED = C.M.ttft_prefill_continuation_mode_allowed
RUNNER = C.adapters(C.RUNNER_PATH, "GPUModelRunner", {
    "_ttft_prefill_preemption_allowed", "_capture_ttft_kv_checkpoint",
}, {"ttft_layerwise_prefill_preemption_requested": lambda: True,
    "ttft_prefill_continuation_mode_allowed": ALLOWED,
    "prefill_workspace_slot_count": lambda: 1,
    "has_kv_transfer_group": lambda: False, "has_ec_transfer": lambda: False,
    "TTFTLayerwisePrefillError": C.M.TTFTLayerwisePrefillError, "os": os})


def engine_eligible(output, enabled, phase="prefill"):
    """Evaluate the actual submit assignment, not a copied mode condition."""
    tree = ast.parse((C.ROOT / "vendor_v8/vllm/v1/engine/core.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCore")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == "_submit_to_pd_phase_queue")
    assignment = next(n for n in fn.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Attribute)
                              and t.attr == "ttft_layerwise_prefill_preemption"
                              for t in n.targets))
    return eval(compile(ast.Expression(assignment.value), "<actual-submit-eligibility>", "eval"), {
        "self": NS(ttft_layerwise_prefill_preemption_enabled=True,
                   isolation_p1_ring_sm_handoff_enabled=enabled),
        "scheduler_output": output, "phase": phase, "model_executed": True,
        "ttft_prefill_continuation_mode_allowed": ALLOWED})


class RingSMTest(unittest.TestCase):
    def output(self):
        o = P.out()
        o.pd_prefill_mode = "ring"
        o.kv_retention_boundary_reason = "sm"
        o.kv_retention_layerwise_offload_requested = False
        o.pending_structured_output_tokens = False
        return o

    def test_real_engine_worker_eligibility_and_rollback_match(self):
        runner = RUNNER()
        runner._pd_phase_local_model_runner_enabled = True
        runner._pd_prefill_workspace_enabled = True
        runner.vllm_config = NS(parallel_config=NS(pipeline_parallel_size=1))
        runner.use_aux_hidden_state_outputs = False
        runner._pd_decode_gpu_component_trace_jsonl = ""
        with patch.dict(os.environ, {"VLLM_ITERATION_GPU_TIMING": "0"}):
            for enabled in (False, True):
                runner._isolation_p1_ring_sm_handoff = enabled
                for mode in ("native", "ring", "auto", "reload"):
                    for phase in ("prefill", "decode"):
                        with self.subTest(enabled=enabled, mode=mode, phase=phase):
                            o = self.output()
                            o.pd_prefill_mode = mode
                            expected = phase == "prefill" and (mode == "native" or (enabled and mode == "ring"))
                            self.assertEqual(engine_eligible(o, enabled, phase), expected)
                            self.assertEqual(runner._ttft_prefill_preemption_allowed(
                                o, phase=phase, phase_context=NS(slot=0), green_context_active=True), expected)
            o = self.output()
            o.ttft_layerwise_prefill_preemption = False
            self.assertFalse(runner._ttft_prefill_preemption_allowed(
                o, phase="prefill", phase_context=NS(slot=0), green_context_active=True))

    def test_ring_yield_is_sm_only_one_shot_without_srpt_scan(self):
        e, o, f = P.engine(), self.output(), Future()
        e.ttft_layerwise_prefill_preemption_enabled = True
        e.pd_sm_transition = NS(active=False)
        o.kv_retention_boundary_reason = ""
        q = {"prefill": deque([(f, o, f)]), "decode": deque()}
        # No scheduler is installed: Ring must not scan SRPT or request offload.
        self.assertFalse(e._request_ttft_prefill_preemption_if_needed(q))
        e.pd_sm_transition.active = True
        signals = []
        with patch.dict(e._request_ttft_prefill_preemption_if_needed.__func__.__globals__, {
                "request_ttft_layerwise_prefill_yield": lambda **kw: signals.append(kw) or True}):
            self.assertTrue(e._request_ttft_prefill_preemption_if_needed(q))
            self.assertFalse(e._request_ttft_prefill_preemption_if_needed(q))
        self.assertEqual(len(signals), 1)
        self.assertEqual(o.kv_retention_boundary_reason, "sm")
        self.assertFalse(o.kv_retention_layerwise_offload_requested)

    def test_ring_boundary_keeps_slot_and_counts_toward_admission_not_sm_drain(self):
        o = self.output()
        reg = W.REGISTRY(1, parked_capacity=1)
        binding = reg.start(o, W.epochs(o))
        b = replace(W.boundary(o, binding), prefill_mode="ring")
        reg.suspend(binding, 2)
        e = P.engine()
        e.scheduler = P.scheduler(o)
        e._ttft_prefill_suspended = {"prefill": (b, o)}
        e.pd_sm_transition = NS(active=True, source_epoch=1)
        q = {"prefill": deque(), "decode": deque()}
        self.assertEqual(e._pd_owned_total_depth(q), 1)
        self.assertEqual(e._pd_owned_phase_depth(q, "prefill"), 1)
        self.assertEqual(e._pd_sm_transition_future_counts(q), (0, 0, 0))
        with self.assertRaises(RuntimeError):
            reg.start(W.output(ids=("C",)), {"C": 11})
        # Target cannot resume before the existing transition has committed.
        e._prune_ttft_prefill_suspended = lambda: None
        e.pd_dynamic_sm_enabled = False
        e.pd_dynamic_sm_enforced = True
        self.assertFalse(e._try_resume_ttft_prefill(q))
        self.assertEqual(len(e._ttft_prefill_suspended), 1)

    def test_native_checkpoint_never_reads_ring_null_block_table(self):
        runner = RUNNER()
        runner._isolation_p1_ring_sm_handoff = True
        state = NS(scheduler_output=self.output(), kv_checkpoint=None,
                   kv_offload=None, kv_resume_frame=None,
                   boundary=NS(kv_retention_parked_ticket=None), model_runner_debug_timings={})
        runner._capture_ttft_kv_checkpoint(state)  # No InputBatch/KV table provided.
        self.assertIsNone(state.kv_checkpoint)
        self.assertEqual(state.model_runner_debug_timings["pd_layerwise_ring_sm_boundary"], 1)
        for attr, value in (("kv_retention_boundary_reason", "request"),
                            ("kv_retention_layerwise_offload_requested", True),
                            ("kv_retention_layerwise_transfer", object())):
            with self.subTest(attr=attr), patch.object(state.scheduler_output, attr, value):
                with self.assertRaisesRegex(C.M.TTFTLayerwisePrefillError, "cannot checkpoint"):
                    runner._capture_ttft_kv_checkpoint(state)
        runner._isolation_p1_ring_sm_handoff = False
        with self.assertRaises(C.M.TTFTLayerwisePrefillError):
            runner._capture_ttft_kv_checkpoint(state)

    def test_retained_stream_registration_does_not_allocate_or_clone(self):
        state, metadata = F.inputs()
        backend = F.Backend()
        touched = []
        count = F.F.kv_retention_record_retained_streams(
            model_state=state, attn_metadata=metadata, logits_indices=F.Tensor([1]),
            record_stream=lambda t: touched.append(t), backend=backend)
        self.assertEqual(backend.allocations, 0)
        self.assertEqual(count, len({id(t) for t in touched}))
        self.assertIn(state.hidden_states, touched)
        self.assertIn(state.residual, touched)

    def test_record_only_keeps_provider_lease_opaque_export_remains_strict(self):
        lease_type = type("TieredDRAMKVLease", (), {
            "__module__": "vllm.v1.attention.ops.tiered_residency_provider"})
        lease = lease_type()
        state, metadata = F.inputs()
        tensor = F.Tensor([7])
        metadata["ring_plan"] = {"pool_leases_by_page": {0: lease}, "tensor_plan": tensor}
        touched = []
        F.F.kv_retention_record_retained_streams(
            model_state=state, attn_metadata=metadata, logits_indices=None,
            record_stream=touched.append, backend=F.Backend())
        self.assertIn(tensor, touched)
        self.assertIs(metadata["ring_plan"]["pool_leases_by_page"][0], lease)
        with self.assertRaisesRegex(RuntimeError, "unsupported Prefill frame object"):
            F.F._SnapshotTree(F.Backend()).inspect(metadata)

    def test_ring_exact_cleanup_rejects_stale_owner_after_resume(self):
        o = self.output()
        reg = W.REGISTRY(1)
        first = reg.start(o, W.epochs(o))
        old_owner = C.OWNER.from_output("prefill", o, first)
        reg.suspend(first, 2)
        o.pd_sm_profile_id, o.pd_sm_profile_epoch = "d0_p40", 2
        second = reg.resume(o, W.epochs(o))
        current_owner = C.OWNER.from_output("prefill", o, second)
        closed = []
        ctx = W.context()
        ctx.kv_retention_execute_owner = current_owner
        ctx.ttft_prefill_continuation = NS(scheduler_output=o, kv_offload=None,
            continuation=NS(close=lambda: closed.append("ring")))
        runner = C.RUNNER()
        runner._pd_phase_contexts = {("prefill", 0): ctx}
        self.assertFalse(runner.clear_pd_phase_execute_state("prefill", old_owner))
        self.assertEqual(closed, [])
        # Caller has already proved GPU completion; exact abort may dispose.
        self.assertTrue(runner.clear_pd_phase_execute_state("prefill", current_owner))
        self.assertEqual(closed, ["ring"])
        self.assertFalse(runner.clear_pd_phase_execute_state("prefill", current_owner))

    def test_actual_two_row_runner_repeated_sm_resume_finalizes_once_no_park(self):
        fixture = F.ActualRunnerTest()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        r, s, o, cont = fixture.runner, fixture.state, fixture.out, fixture.cont
        o.pd_prefill_mode = cont.execution.prefill_mode = "ring"
        o.kv_retention_boundary_reason = "sm"
        o.kv_retention_layerwise_offload_requested = False
        o.kv_retention_layerwise_transfer = None
        s.kv_checkpoint = None
        s.boundary = replace(s.boundary, prefill_mode="ring")
        r._isolation_p1_ring_sm_handoff = True
        r._capture_ttft_kv_checkpoint = RUNNER._capture_ttft_kv_checkpoint.__get__(r)
        r._kv_retention_parked_store = P.P.KVRetentionParkedStore(1)
        globals_ = r._import_kv_retention_resume_frame.__func__.__globals__
        with patch.dict(globals_, {
                "kv_retention_record_retained_streams": F.F.kv_retention_record_retained_streams,
                "TTFTLayerwisePrefillError": C.M.TTFTLayerwisePrefillError}), patch.object(
                    F.F.KVRetentionResumeFrame, "capture", side_effect=AssertionError("Ring frame clone")):
            r._capture_ttft_kv_checkpoint(s)
            r._capture_kv_retention_resume_frame(s)
            finished = []
            r._finish_execute_model_forward = lambda **kw: finished.append(kw)
            fixture.registry.suspend(fixture.first, 2)
            o.pd_sm_profile_id, o.pd_sm_profile_epoch = "d32_p8", 2
            second = fixture.registry.resume(o, W.epochs(o))
            fixture.ctx.kv_retention_execute_owner = C.OWNER.from_output("prefill", o, second)
            cont.execution.note_resume()
            original_hidden = cont.state.hidden_states
            b = r._resume_ttft_prefill_continuation(o, s)
            self.assertEqual(b.next_layer, 4)
            self.assertEqual(original_hidden.streams, ["target"])
            self.assertEqual(finished, [])
            self.assertIsNone(s.kv_resume_frame)
            self.assertIsNone(b.kv_retention_parked_ticket)
            fixture.registry.suspend(second, 4)
            o.pd_sm_profile_id, o.pd_sm_profile_epoch = "d0_p40", 3
            third = fixture.registry.resume(o, W.epochs(o))
            fixture.ctx.kv_retention_execute_owner = C.OWNER.from_output("prefill", o, third)
            cont.execution.note_resume()
            r._resume_ttft_prefill_continuation(o, s)
        self.assertEqual(len(finished), 1)
        self.assertIs(finished[0]["scheduler_output"], o)
        reference, *_ = F.model_and_continuation(())
        with self.assertRaises(StopIteration) as complete:
            next(reference)
        np.testing.assert_equal(finished[0]["model_output"].data, complete.exception.value.data)
        self.assertEqual(r._kv_retention_parked_store.entries, {})
        fixture.registry.sampling(third)
        fixture.registry.finish(third)
        self.assertEqual(len(fixture.registry), 0)


if __name__ == "__main__":
    unittest.main()
