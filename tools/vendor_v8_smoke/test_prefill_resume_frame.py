#!/usr/bin/env python3
"""CPU/fault-injection tests of real forward-frame adapters; not GPU E2E."""
from __future__ import annotations

import ast
import contextvars
import contextlib
import importlib.util
import os
import sys
import time
import unittest
from dataclasses import dataclass, replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import test_exact_phase_cleanup as C
import test_prefill_work_binding as W

PATH = C.ROOT / "vendor_v8/vllm/model_executor/layers/kv_retention_resume_frame.py"
SPEC = importlib.util.spec_from_file_location("prefill_resume_frame_target", PATH)
F = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = F
SPEC.loader.exec_module(F)


class Tensor:
    def __init__(self, values, root=None, gpu=True):
        self.data = np.asarray(values)
        self.root = self.data if root is None else root
        self.gpu = gpu
        self.streams = []

    @property
    def shape(self):
        return self.data.shape

    def record_stream(self, stream):
        self.streams.append(stream)


class Backend:
    def __init__(self):
        self.allocations = 0

    @staticmethod
    def tensor(value):
        return isinstance(value, Tensor)

    @staticmethod
    def array(value):
        return isinstance(value, np.ndarray)

    @staticmethod
    def gpu(value):
        return value.gpu

    @staticmethod
    def compact_bytes(value):
        return value.data.nbytes

    @staticmethod
    def storage(value):
        return ("cuda" if value.gpu else "cpu", id(value.root)), value.root.nbytes

    def clone(self, value):
        self.allocations += 1
        return Tensor(value.data.copy(), gpu=value.gpu)

    def empty(self, value):
        self.allocations += 1
        return Tensor(np.zeros(value.data.shape, dtype=value.data.dtype), gpu=value.gpu)


@dataclass
class Metadata:
    positions: Tensor
    block_table: Tensor
    req_ids: list[str]
    page_versions: dict[str, dict[int, int]]


@dataclass
class TritonAttentionMetadata:
    query_start_loc: Tensor
    softmax_segm_output: Tensor
    softmax_segm_max: Tensor
    softmax_segm_expsum: Tensor


TritonAttentionMetadata.__module__ = "vllm.v1.attention.backends.triton_attn"


def inputs():
    state = F.KVRetentionModelState(
        2, 6, Tensor([8, 9, 16, 17]), Tensor(np.ones((4, 3))),
        Tensor(np.ones((4, 3)) * 2), [], True)
    metadata = Metadata(state.positions, Tensor([[1, 2], [3, 4]]),
                        ["A", "B"], {"A": {0: 9}})
    return state, {"layer0": metadata, "layer1": metadata}


def bindings():
    out = W.output()
    out.num_scheduled_tokens = {"A": 2, "B": 2}
    out.total_num_scheduled_tokens = 4
    registry = W.REGISTRY(2)
    first = registry.start(out, W.epochs(out))
    return out, registry, first


class FrameTest(unittest.TestCase):
    def setUp(self):
        self.out, self.registry, self.first = bindings()
        self.state, self.metadata = inputs()
        self.backend = Backend()
        self.calls = []

    def capture(self, **kwargs):
        args = dict(binding=self.first, model_state=self.state,
                    attn_metadata=self.metadata, logits_indices=Tensor([1, 3]),
                    num_scheduled_tokens_np=np.array([2, 2]), max_gpu_bytes=4096,
                    gpu_boundary_complete=True, copy_complete=lambda: self.calls.append("ready"),
                    backend=self.backend)
        args.update(kwargs)
        return F.KVRetentionResumeFrame.capture(**args)

    def test_default_off_and_budget_validation(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(F.kv_retention_resume_frame_requested())
            self.assertEqual(F.kv_retention_resume_frame_gpu_bytes(), 268435456)
        with patch.dict(os.environ, {"VLLM_KV_RETENTION_PREFILL_RESUME_FRAME": " 1 "}):
            self.assertTrue(F.kv_retention_resume_frame_requested())
        for value in ("0", "-1", "nope"):
            with patch.dict(os.environ, {"VLLM_KV_RETENTION_RESUME_FRAME_GPU_BYTES": value}):
                with self.assertRaises(ValueError):
                    F.kv_retention_resume_frame_gpu_bytes()

    def test_source_overwrite_cannot_change_snapshot(self):
        frame = self.capture()
        self.state.positions.data.fill(999)
        self.state.hidden_states.data.fill(999)
        self.state.residual.data.fill(999)
        self.metadata["layer0"].block_table.data.fill(999)
        self.metadata["layer0"].req_ids[:] = ["C"]
        self.metadata["layer0"].page_versions["A"][0] = -1
        state = frame.payload["model_state"]
        np.testing.assert_equal(state.positions.data, [8, 9, 16, 17])
        np.testing.assert_equal(state.hidden_states.data, np.ones((4, 3)))
        self.assertEqual(frame.payload["attn_metadata"]["layer0"].req_ids, ["A", "B"])
        self.assertEqual(frame.payload["attn_metadata"]["layer0"].page_versions, {"A": {0: 9}})
        self.assertEqual(self.calls, ["ready"])

    def test_same_object_copied_once_and_aliases_internal_only(self):
        frame = self.capture()
        a, b = frame.payload["attn_metadata"].values()
        self.assertIs(a, b)
        self.assertIs(a.positions, frame.payload["model_state"].positions)
        self.assertIsNot(a.positions, self.state.positions)
        self.assertEqual(self.backend.allocations, 5)

    def test_compact_views_and_entire_unique_storage_accounting(self):
        root = np.zeros(2048)
        view = Tensor(root[:12].reshape(4, 3), root)
        self.state.hidden_states = self.state.residual = view
        frame = self.capture(max_gpu_bytes=1024)
        self.assertGreater(frame.source_gpu_storage_bytes, 16384)
        self.assertLess(frame.gpu_storage_bytes, 1024)
        a = frame.payload["model_state"]
        self.assertIs(a.hidden_states, a.residual)
        self.assertEqual(a.hidden_states.root.nbytes, 96)

    def test_shape_and_logical_tokens_must_match_before_copy(self):
        for state in (replace(self.state, positions=Tensor([1, 2])),
                      replace(self.state, hidden_states=Tensor(np.zeros((2, 3)))),
                      replace(self.state, residual=Tensor(np.zeros((4, 4))))):
            with self.subTest(state=state), self.assertRaisesRegex(RuntimeError, "shape/token"):
                self.capture(model_state=state)
        with self.assertRaisesRegex(RuntimeError, "shape/token"):
            self.capture(num_scheduled_tokens_np=np.array([4, 4]))
        self.assertEqual(self.backend.allocations, 0)

    def test_budget_decline_allocates_nothing(self):
        with self.assertRaises(F.KVRetentionFrameBudgetExceeded):
            self.capture(max_gpu_bytes=1)
        self.assertEqual(self.backend.allocations, 0)
        self.assertEqual(self.calls, [])

    def test_backend_scratch_allocated_without_copying_decode_contents(self):
        scratch = Tensor(np.full((2, 3), float("nan")))
        metadata = TritonAttentionMetadata(Tensor([0, 2, 4]), scratch, scratch, scratch)
        frame = self.capture(attn_metadata=metadata)
        snapshot = frame.payload["attn_metadata"]
        self.assertFalse(np.isnan(snapshot.softmax_segm_output.data).any())
        self.assertIs(snapshot.softmax_segm_max, snapshot.softmax_segm_output)
        self.assertEqual(frame.gpu_storage_bytes - frame.copied_tensor_bytes, scratch.data.nbytes)

    def test_backend_scratch_may_not_alias_input(self):
        tensor = Tensor([1, 2, 3])
        metadata = TritonAttentionMetadata(tensor, tensor, tensor, tensor)
        with self.assertRaisesRegex(RuntimeError, "scratch aliases"):
            self.capture(attn_metadata=metadata)
        self.assertEqual(self.backend.allocations, 0)

    def test_missing_completion_or_binding_refuses_before_copy(self):
        for kwargs in ({"gpu_boundary_complete": False}, {"binding": None}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.capture(**kwargs)
        self.assertEqual(self.backend.allocations, 0)

    def test_unknown_mutable_metadata_refused_before_copy(self):
        for value in (NS(owner="phase"), np.array([object()], dtype=object),
                      {object(): "owner"}):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.capture(attn_metadata=value)
        self.assertEqual(self.backend.allocations, 0)

    def test_cyclic_metadata_refused(self):
        value = []
        value.append(value)
        with self.assertRaisesRegex(RuntimeError, "cyclic"):
            self.capture(attn_metadata=value)

    def test_copy_error_preserves_candidate_and_source(self):
        def failed():
            raise RuntimeError("event failed")
        with self.assertRaises(F.KVRetentionFrameCopyError) as raised:
            self.capture(copy_complete=failed)
        self.assertTrue(raised.exception.candidate)
        self.assertIsNotNone(self.state.hidden_states)
        self.assertIn("event failed", str(raised.exception.__cause__))

    def test_clone_alias_is_rejected(self):
        self.backend.clone = lambda value: value
        with self.assertRaises(F.KVRetentionFrameCopyError) as raised:
            self.capture()
        self.assertIn("aliases", str(raised.exception.__cause__))

    def test_actual_large_storage_is_rejected(self):
        def oversize(value):
            return Tensor(value.data.copy(), root=np.empty(8192))
        self.backend.clone = oversize
        with self.assertRaises(F.KVRetentionFrameCopyError) as raised:
            self.capture()
        self.assertIn("actual storage", str(raised.exception.__cause__))

    def test_pending_frame_not_exportable(self):
        frame = self.capture()
        frame.ready = False
        continuation = F.KVRetentionLayerContinuation(None, NS(), self.state)
        with self.assertRaises(RuntimeError):
            continuation.export(frame)
        self.assertIs(continuation.state, self.state)

    def test_consume_requires_target_stream_lifetime(self):
        frame = self.capture()
        with self.assertRaisesRegex(RuntimeError, "target stream"):
            frame.consume()
        self.assertFalse(frame.consumed)

    def test_failed_target_record_preserves_exported_owner(self):
        frame = self.capture()
        continuation = F.KVRetentionLayerContinuation(None, NS(), self.state)
        continuation.export(frame)
        self.registry.suspend(self.first, 2)
        second = self.registry.resume(self.out, W.epochs(self.out))
        def fail(tensor):
            raise RuntimeError("target stream unavailable")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            continuation.import_frame(frame, second, record_stream=fail)
        self.assertIs(continuation.exported, frame)
        self.assertFalse(frame.consumed)

    def test_import_requires_exact_successor_and_consumes_once(self):
        frame = self.capture()
        continuation = F.KVRetentionLayerContinuation(None, NS(), self.state)
        continuation.export(frame)
        with self.assertRaisesRegex(RuntimeError, "exported"):
            next(continuation)
        with self.assertRaises(RuntimeError):
            continuation.import_frame(frame, self.first, record_stream=lambda t: None)
        self.assertIs(continuation.exported, frame)
        self.registry.suspend(self.first, 2)
        second = self.registry.resume(self.out, W.epochs(self.out))
        recorded = []
        continuation.import_frame(frame, second, record_stream=recorded.append)
        self.assertEqual(len(recorded), len(frame.gpu_tensors))
        self.assertEqual(continuation.state.next_layer, 2)
        with self.assertRaises(RuntimeError):
            continuation.import_frame(frame, second, record_stream=recorded.append)
        frame.close()
        self.assertIsNotNone(continuation.state.positions)

    def test_cleanup_exact_frame_and_idempotent_close(self):
        frame = self.capture()
        continuation = F.KVRetentionLayerContinuation(
            None, NS(cancel_yield_request=lambda: self.calls.append("cancel")), self.state)
        continuation.export(frame)
        continuation.close()
        continuation.close()
        self.assertTrue(frame.closed)
        self.assertFalse(frame.payload)
        with self.assertRaises(StopIteration):
            next(continuation)

    def test_no_future_or_output_or_model_in_frame(self):
        frame = self.capture()
        self.assertEqual(set(frame.payload), {"model_state", "attn_metadata",
                                              "logits_indices", "num_scheduled_tokens_np"})
        self.assertNotIn("model", vars(frame.payload["model_state"]))


LLAMA = C.adapters("vendor_v8/vllm/model_executor/models/llama.py", "LlamaModel",
                   {"_kv_retention_advance_prefill", "_forward_decoder_layer_range"}, {
                       "time": time, "get_pp_group": lambda: NS(is_last_rank=True),
                   })


def model_and_continuation(yield_layers=(2, 4)):
    out, registry, binding = bindings()
    execution = C.M.TTFTLayerwisePrefillExecution(
        requested=True, active=True, phase="prefill", group_layers=2,
        preemption_requested=True, preemption_active=True,
        request_ids=("A", "B"), scheduled_tokens=(("A", 2), ("B", 2)),
        scheduler_iteration_id=out.scheduler_iteration_id,
        phase_submit_seq=out.pd_phase_submit_seq, phase_slot=0)
    execution.consume_yield_request = lambda: execution.last_layer_exclusive in yield_layers
    model = LLAMA()
    model.start_layer, model.end_layer = 0, 6
    model.aux_hidden_state_layers = []
    model.pd_layerwise_profile_handoff_enabled = False
    trace = []

    def layer(index):
        def forward(positions, hidden, residual):
            trace.append(index)
            residual = Tensor(hidden.data.copy()) if residual is None else residual
            residual.data[:] += positions.data[:, None] / (index + 2)
            return Tensor(hidden.data * 1.1 + residual.data), residual
        return NS(forward_with_pd_prefill_workspace=forward)
    model.layers = [layer(i) for i in range(6)]
    model.norm = lambda hidden, residual: (Tensor(hidden.data + residual.data), None)
    state, metadata = inputs()
    state.next_layer = 0
    continuation = F.KVRetentionLayerContinuation(model, execution, state)
    return continuation, metadata, trace, out, registry, binding


class ActualLlamaTest(unittest.TestCase):
    def test_a_c_a_reentry_matches_uninterrupted_remaining_layers(self):
        reference, _, reference_trace, *_ = model_and_continuation(())
        with self.assertRaises(StopIteration) as finished:
            next(reference)
        expected = finished.exception.value.data
        cont, metadata, trace, out, registry, binding = model_and_continuation()
        for layer in (2, 4):
            boundary = next(cont)
            self.assertEqual(boundary.next_layer, layer)
            source = cont.state
            frame = F.KVRetentionResumeFrame.capture(
                binding=binding, model_state=source, attn_metadata=metadata,
                logits_indices=Tensor([1, 3]), num_scheduled_tokens_np=np.array([2, 2]),
                max_gpu_bytes=4096, gpu_boundary_complete=True,
                copy_complete=lambda: None, backend=Backend())
            cont.export(frame)
            source.positions.data.fill(1000)
            source.hidden_states.data.fill(1000)
            source.residual.data.fill(1000)
            metadata["layer0"].block_table.data.fill(1000)
            registry.suspend(binding, layer)
            binding = registry.resume(out, W.epochs(out))
            cont.execution.note_resume()
            cont.import_frame(frame, binding, record_stream=lambda t: None)
            metadata = frame.payload["attn_metadata"]
            frame.close()
        with self.assertRaises(StopIteration) as finished:
            next(cont)
        np.testing.assert_equal(finished.exception.value.data, expected)
        self.assertEqual(trace, reference_trace)
        self.assertEqual(trace, list(range(6)))

    def test_flagoff_generator_remains_present(self):
        source = (C.ROOT / "vendor_v8/vllm/model_executor/models/llama.py").read_text()
        tree = ast.parse(source)
        model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LlamaModel")
        legacy = next(n for n in model.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_ttft_resumable_prefill_forward")
        self.assertTrue(any(isinstance(n, ast.Yield) for n in ast.walk(legacy)))
        self.assertIn("if kv_retention_resume_frame_requested():", source)
        engine = (C.ROOT / "vendor_v8/vllm/v1/engine/core.py").read_text()
        self.assertIn("self.pd_prefill_inflight_depth >= 2", engine)
        self.assertIn("self.kv_retention_p1_parked_enabled and self.pd_prefill_inflight_depth == 1", engine)


ACTIVE = contextvars.ContextVar("frame_test_active", default=None)
RUNNER = C.adapters(C.RUNNER_PATH, "GPUModelRunner", {
    "_capture_kv_retention_resume_frame", "_import_kv_retention_resume_frame",
    "_resume_ttft_prefill_continuation",
}, {"KVRetentionLayerContinuation": F.KVRetentionLayerContinuation,
    "KVRetentionResumeFrame": F.KVRetentionResumeFrame,
    "KVRetentionFrameBudgetExceeded": F.KVRetentionFrameBudgetExceeded,
    "kv_retention_resume_frame_gpu_bytes": lambda: 4096,
    "_PD_ACTIVE_MODEL_RUNNER_CONTEXT": ACTIVE, "CUDAGraphMode": NS(NONE="none"),
    "torch": NS(cuda=NS(current_stream=lambda: "target")),
    "active_ttft_layerwise_prefill_execution": lambda: ACTIVE.get().ttft_prefill_continuation.execution,
    "reset_operator_selective_pd_metrics": lambda: None,
    "reset_opmux_metrics": lambda: None,
    "reset_tiered_context_ring_runtime_metrics": lambda: None,
    "kv_retention_ring_scope": lambda *a, **k: contextlib.nullcontext(),
    "set_forward_context": lambda *a, **k: contextlib.nullcontext(),
    "record_function_or_nullcontext": lambda *a, **k: contextlib.nullcontext(),
    "TTFTLayerwisePrefillBoundary": C.M.TTFTLayerwisePrefillBoundary,
    "time": time,
})


class ActualRunnerTest(unittest.TestCase):
    def setUp(self):
        self.cont, self.meta, _, self.out, self.registry, self.first = model_and_continuation()
        boundary = next(self.cont)
        self.state = NS(
            continuation=self.cont, attn_metadata=self.meta, logits_indices=Tensor([1, 3]),
            num_scheduled_tokens_np=np.array([2, 2]), kv_resume_frame=None,
            kv_checkpoint=object(), num_tokens_across_dp=None, ubatch_slices_padded=None,
            spec_decode_metadata=None, spec_decode_common_attn_metadata=None,
            ec_connector_output=None, cudagraph_mode="none", scheduler_output=self.out,
            boundary=boundary, model_runner_debug_timings={}, execution=self.cont.execution,
            kv_offload=None, num_tokens_padded=4, num_reqs=2, num_scheduled_tokens=4,
            batch_desc=None, cudagraph_stats=None)
        self.ctx = NS(ttft_prefill_continuation=self.state,
                      kv_retention_execute_owner=C.OWNER.from_output("prefill", self.out, self.first))
        self.token = ACTIVE.set(self.ctx)
        self.runner = RUNNER()
        self.runner._synchronize_ttft_prefill_boundary = lambda: 0.001
        self.runner._validate_ttft_kv_checkpoint = lambda state: None
        self.runner._capture_ttft_kv_checkpoint = lambda state: None
        self.runner._accumulate_ttft_group_debug = lambda *a, **k: None
        self.runner._pd_cuda_phase_trace = lambda *a, **k: contextlib.nullcontext()
        self.runner._pd_green_context_execute_phase = "prefill"
        self.runner.vllm_config = object()
        self.backend_patch = patch.object(F, "_TorchSnapshotBackend", Backend)
        self.backend_patch.start()

    def tearDown(self):
        ACTIVE.reset(self.token)
        self.backend_patch.stop()

    def test_capture_and_import_real_runner_adapters(self):
        source = self.cont.state.positions
        self.runner._capture_kv_retention_resume_frame(self.state)
        frame = self.state.kv_resume_frame
        self.assertIs(self.cont.exported, frame)
        source.data.fill(-1)
        self.registry.suspend(self.first, 2)
        second = self.registry.resume(self.out, W.epochs(self.out))
        self.ctx.kv_retention_execute_owner = C.OWNER.from_output("prefill", self.out, second)
        self.runner._import_kv_retention_resume_frame(self.state)
        self.assertIsNone(self.state.kv_resume_frame)
        np.testing.assert_equal(self.cont.state.positions.data, [8, 9, 16, 17])
        self.assertTrue(frame.closed)
        self.assertEqual(self.cont.state.positions.streams, ["target"])

    def test_wrong_owner_cannot_import(self):
        self.runner._capture_kv_retention_resume_frame(self.state)
        frame = self.state.kv_resume_frame
        with self.assertRaises(RuntimeError):
            self.runner._import_kv_retention_resume_frame(self.state)
        self.assertIs(self.state.kv_resume_frame, frame)

    def test_actual_runner_two_resumes_only_finalizes_once(self):
        self.runner._capture_kv_retention_resume_frame(self.state)
        self.registry.suspend(self.first, 2)
        second = self.registry.resume(self.out, W.epochs(self.out))
        self.ctx.kv_retention_execute_owner = C.OWNER.from_output("prefill", self.out, second)
        self.cont.execution.note_resume()
        finalized = []
        self.runner._finish_execute_model_forward = lambda **kw: finalized.append(kw)
        boundary = self.runner._resume_ttft_prefill_continuation(self.out, self.state)
        self.assertEqual(boundary.next_layer, 4)
        self.assertIsNotNone(self.state.kv_resume_frame)
        self.assertEqual(finalized, [])
        self.registry.suspend(second, 4)
        third = self.registry.resume(self.out, W.epochs(self.out))
        self.ctx.kv_retention_execute_owner = C.OWNER.from_output("prefill", self.out, third)
        self.cont.execution.note_resume()
        self.runner._resume_ttft_prefill_continuation(self.out, self.state)
        self.assertEqual(len(finalized), 1)
        self.assertIs(finalized[0]["scheduler_output"], self.out)
        self.assertIsNone(self.ctx.ttft_prefill_continuation)
        self.assertIsNone(self.state.kv_resume_frame)
        self.assertTrue(self.cont.closed)

    def test_wrong_checkpoint_cannot_consume_frame(self):
        self.runner._capture_kv_retention_resume_frame(self.state)
        self.registry.suspend(self.first, 2)
        second = self.registry.resume(self.out, W.epochs(self.out))
        self.ctx.kv_retention_execute_owner = C.OWNER.from_output("prefill", self.out, second)
        def fail(state):
            raise RuntimeError("KV mapping changed")
        self.runner._validate_ttft_kv_checkpoint = fail
        frame = self.state.kv_resume_frame
        with self.assertRaisesRegex(RuntimeError, "KV mapping"):
            self.runner._resume_ttft_prefill_continuation(self.out, self.state)
        self.assertFalse(frame.consumed)
        self.assertIs(self.cont.exported, frame)

    def test_budget_decline_keeps_original_state_and_slot(self):
        original = self.cont.state
        with patch.object(F.KVRetentionResumeFrame, "capture",
                          side_effect=F.KVRetentionFrameBudgetExceeded()):
            self.runner._capture_kv_retention_resume_frame(self.state)
        self.assertIs(self.cont.state, original)
        self.assertIs(self.ctx.ttft_prefill_continuation, self.state)
        self.assertIsNone(self.state.kv_resume_frame)
        self.assertEqual(self.state.model_runner_debug_timings[
            "kv_retention_resume_frame_budget_declined"], 1)

    def test_unsupported_or_unowned_state_refused(self):
        for attr, value in (("cudagraph_mode", "full"), ("num_tokens_across_dp", Tensor([4])),
                            ("spec_decode_metadata", object()), ("kv_checkpoint", None)):
            before = getattr(self.state, attr)
            setattr(self.state, attr, value)
            with self.subTest(attr=attr), self.assertRaises(RuntimeError):
                self.runner._capture_kv_retention_resume_frame(self.state)
            setattr(self.state, attr, before)
        self.assertIsNone(self.state.kv_resume_frame)

    def test_generator_and_no_frame_are_noop(self):
        self.state.continuation = iter(())
        self.runner._capture_kv_retention_resume_frame(self.state)
        self.runner._import_kv_retention_resume_frame(self.state)
        self.assertEqual(self.state.model_runner_debug_timings, {})

    def test_lightweight_retains_only_bounded_frame_metrics(self):
        self.runner._capture_kv_retention_resume_frame(self.state)
        path = C.ROOT / "vendor_v8/vllm/v1/core/sched/observability.py"
        spec = importlib.util.spec_from_file_location("frame_observability", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        debug = dict(self.state.model_runner_debug_timings, unrelated_large_payload=[1, 2, 3])
        compact = module.compact_worker_debug_timings(debug)
        self.assertNotIn("unrelated_large_payload", compact)
        self.assertEqual(compact, self.state.model_runner_debug_timings)

    def test_exact_abort_closes_real_exported_frame_after_copy_owner(self):
        self.runner._capture_kv_retention_resume_frame(self.state)
        frame = self.state.kv_resume_frame
        events = []
        self.state.kv_offload = NS(close=lambda: events.append("copy"))
        self.cont.execution.cancel_yield_request = lambda: events.append("frame")
        self.ctx.execute_model_state = None
        cleanup = C.RUNNER()
        cleanup._pd_phase_contexts = {("prefill", 0): self.ctx}
        self.assertTrue(cleanup.clear_pd_phase_execute_state(
            "prefill", self.ctx.kv_retention_execute_owner))
        self.assertEqual(events, ["copy", "frame"])
        self.assertTrue(frame.closed)
        self.assertIsNone(self.ctx.ttft_prefill_continuation)

    def test_failed_copy_cleanup_keeps_frame_owned(self):
        self.runner._capture_kv_retention_resume_frame(self.state)
        frame = self.state.kv_resume_frame
        def fail():
            raise RuntimeError("DMA still pending")
        self.state.kv_offload = NS(close=fail)
        self.ctx.execute_model_state = None
        cleanup = C.RUNNER()
        cleanup._pd_phase_contexts = {("prefill", 0): self.ctx}
        with self.assertRaisesRegex(RuntimeError, "DMA still pending"):
            cleanup.clear_pd_phase_execute_state("prefill", self.ctx.kv_retention_execute_owner)
        self.assertFalse(frame.closed)
        self.assertIs(self.cont.exported, frame)
        self.assertIs(self.ctx.ttft_prefill_continuation, self.state)


if __name__ == "__main__":
    unittest.main()
