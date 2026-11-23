#!/usr/bin/env python3
"""CPU ownership/layout tests; GPU numerical tests live in the CUDA companion."""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / "vendor_v8/vllm"


def load_symbols(path, names, namespace=None):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in names]
    ns = dict(namespace or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[
            ast.alias(name="annotations")], level=0), *selected
    ], type_ignores=[])), str(path), "exec"), ns)
    return ns


spec = importlib.util.spec_from_file_location("isolation_flags", V / "isolation_memory.py")
flags = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flags)


class EagerLayoutTests(unittest.TestCase):
    def layout(self, extended=True):
        ns = load_symbols(
            V / "model_executor/layers/pd_prefill_workspace.py",
            {"LlamaPrefillWorkspaceLayout", "PrefillWorkspaceError"},
            {"dataclass": dataclasses.dataclass, "__name__": __name__},
        )
        layout = ns["LlamaPrefillWorkspaceLayout"](
            3072, 2048, 16384, 8192, 2048 if extended else 0
        )
        layout.validate()
        return layout

    def test_no_extra_arena_for_llama_1b(self):
        for tokens in (1, 721, 2048, 16384):
            regions = self.layout().regions(tokens)
            self.assertEqual(max(o + n for o, n in regions.values()),
                             tokens * 24576)

    def test_each_live_producer_consumer_disjoint(self):
        regions = self.layout().regions(16384)
        for a, b in (("norm", "qkv"), ("qkv", "attention"),
                     ("attention", "norm"), ("norm", "gate_up"),
                     ("gate_up", "activation"), ("activation", "hidden"),
                     ("hidden", "norm")):
            a0, an = regions[a]
            b0, bn = regions[b]
            self.assertTrue(a0 + an <= b0 or b0 + bn <= a0, (a, b))

    def test_legacy_layout_unchanged(self):
        self.assertEqual(self.layout(False).regions(2), {
            "qkv": (0, 6144), "attention": (6144, 4096),
            "gate_up": (0, 32768), "activation": (32768, 16384),
        })

    def test_bad_down_output_rejected(self):
        layout = dataclasses.replace(self.layout(), hidden_width=32768)
        with self.assertRaises(RuntimeError):
            layout.validate()

    def test_master_and_independent_rollback(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(flags.isolation_memory_enabled("FUSED_RMS_NORM"))
        with patch.dict(os.environ, {
            "VLLM_ISOLATION_MEMORY_OPTIMIZATIONS": "1",
            "VLLM_ISOLATION_FUSED_RMS_NORM": "0",
        }, clear=True):
            self.assertFalse(flags.isolation_memory_enabled("FUSED_RMS_NORM"))
            self.assertTrue(flags.isolation_memory_enabled("PREFILL_EXTENDED_WORKSPACE"))
        with patch.dict(os.environ, {"VLLM_ISOLATION_MEMORY_OPTIMIZATIONS": "maybe"}):
            with self.assertRaises(ValueError):
                flags.isolation_memory_enabled("FUSED_RMS_NORM")

    def test_norm_enforcement_only_eager_cuda(self):
        from types import SimpleNamespace as NS
        ns = load_symbols(V / "model_executor/models/llama.py",
                          {"_isolation_fused_norm_requested"}, {
            "isolation_memory_enabled": lambda _: True,
            "current_platform": NS(is_cuda=lambda: True),
            "prefill_workspace_requested": lambda: False,
            "layerwise_profile_handoff_requested": lambda: False,
            "ttft_layerwise_prefill_execution_requested": lambda: False,
        })
        self.assertFalse(ns["_isolation_fused_norm_requested"]())
        ns["prefill_workspace_requested"] = lambda: True
        self.assertTrue(ns["_isolation_fused_norm_requested"]())
        ns["current_platform"] = NS(is_cuda=lambda: False)
        self.assertFalse(ns["_isolation_fused_norm_requested"]())


class RingStateTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace as NS
        import numpy as np
        self.np = np
        self.stream = NS(waits=[])
        self.stream.wait_event = self.stream.waits.append
        self.events = []
        self.allocations = []
        owner = self

        class Tensor:
            def __init__(self, shape, data=None):
                self.data = np.empty(shape, np.float32) if data is None else data
                self.shape = self.data.shape
                self.streams = []
            def __getitem__(self, index):
                return Tensor((), self.data[index])
            def numel(self):
                return self.data.size
            def element_size(self):
                return 4
            def record_stream(self, stream):
                self.streams.append(stream)

        class Event:
            def record(self, stream):
                self.stream = stream
                owner.events.append(self)
        def empty(shape, **kwargs):
            tensor = Tensor(shape)
            self.allocations.append(tensor)
            return tensor
        torch = NS(empty=empty, float32="f32", cuda=NS(
            current_stream=lambda device: self.stream, Event=Event))
        ns = load_symbols(
            V / "v1/attention/ops/isolation_ring_state.py",
            {"IsolationRingStateArena"},
        )
        self.arena = ns["IsolationRingStateArena"](torch, lambda *args: args)
        self.query = lambda q, h=4, d=8: NS(shape=(q, h, d), device="cuda:0")

    def test_many_shapes_reuse_one_high_water_arena(self):
        first, reused = self.arena.acquire(self.query(7))
        self.assertFalse(reused)
        self.arena.release()
        for q in (1, 3, 5, 8, 7, 2):
            state, reused = self.arena.acquire(self.query(q))
            self.assertTrue(reused)
            self.assertTrue(self.np.shares_memory(state[0].data, first[0].data))
            self.arena.release()
        self.assertEqual(len(self.allocations), 3)
        self.assertEqual(self.arena.bytes, 8 * 4 * (8 + 2) * 4)

    def test_growth_and_geometry_replace_not_accumulate(self):
        self.arena.acquire(self.query(7))
        self.arena.release()
        self.arena.acquire(self.query(9))
        self.arena.release()
        self.assertEqual(self.arena.capacity, 16)
        self.assertEqual(len(self.arena.storage), 3)
        self.arena.acquire(self.query(2, h=2))
        self.assertEqual(self.arena.capacity, 2)
        self.assertEqual(self.arena.geometry, ("cuda:0", 2, 8))

    def test_cross_stream_wait_before_reuse(self):
        from types import SimpleNamespace as NS
        self.arena.acquire(self.query(7))
        self.arena.release()
        event = self.events[-1]
        self.stream = NS(waits=[])
        self.stream.wait_event = self.stream.waits.append
        self.arena.acquire(self.query(4))
        self.assertEqual(self.stream.waits, [event])
        self.assertTrue(all(self.stream in t.streams for t in self.arena.storage))

    def test_failed_replacement_does_not_publish_stale_capacity(self):
        self.arena.acquire(self.query(7))
        self.arena.release()
        with patch.object(self.arena.torch, "empty",
                          side_effect=RuntimeError("allocation failure")):
            with self.assertRaises(RuntimeError):
                self.arena.acquire(self.query(19))
        self.assertIsNone(self.arena.geometry)
        self.assertEqual(self.arena.capacity, 0)
        state, reused = self.arena.acquire(self.query(7))
        self.assertFalse(reused)
        self.assertEqual(state[0].shape, (7, 4))

    def test_reentrant_and_failed_release_fail_closed(self):
        self.arena.acquire(self.query(4))
        with self.assertRaises(RuntimeError):
            self.arena.acquire(self.query(4))
        def fail():
            raise RuntimeError("event failure")
        self.arena.torch.cuda.Event = fail
        with self.assertRaises(RuntimeError):
            self.arena.release()
        self.assertTrue(self.arena.active)


class RingKernelRoutingTests(unittest.TestCase):
    def test_exact_shapes_become_runtime_scalars_only_when_opted_in(self):
        from types import SimpleNamespace as NS
        import math
        device = NS(type="cuda")
        class Tensor:
            def __init__(self, shape):
                self.shape, self.ndim = shape, len(shape)
                self.device, self.dtype = device, "f32"
            def stride(self, axis):
                return math.prod(self.shape[axis + 1:])
            def numel(self):
                return math.prod(self.shape)
            def is_contiguous(self):
                return True
        calls = []
        class Kernel:
            def __getitem__(self, grid):
                return lambda *a, **k: calls.append((a, k))
        names = {"_validate_query_kv", "_validate_state",
                 "update_triton_online_softmax_state",
                 "update_triton_online_softmax_state_paged",
                 "finalize_triton_online_softmax_state"}
        ns = load_symbols(
            V / "v1/attention/ops/tiered_triton_fused_attention.py", names, {
                "torch": NS(float32="f32", empty=lambda shape, **k: Tensor(shape)),
                "triton": NS(next_power_of_2=lambda x: 1 << (x - 1).bit_length(),
                             cdiv=lambda a, b: (a + b - 1) // b),
                "_fused_online_softmax_update_kernel": Kernel(),
                "_fused_paged_online_softmax_update_kernel": Kernel(),
                "_finalize_online_softmax_kernel": Kernel(),
            })
        q, k = Tensor((19, 4, 32)), Tensor((29, 2, 32))
        state = NS(max_score=Tensor((19, 4)), exp_sum=Tensor((19, 4)),
                   numerator=Tensor((19, 4, 32)))
        common = dict(query_positions=Tensor((19,)), key_start=2031,
                      causal=True, softmax_scale=32 ** -0.5)
        for dynamic in (False, True):
            ns["_ISOLATION_DYNAMIC_KERNEL"] = dynamic
            ns["update_triton_online_softmax_state"](q, k, k, state, **common)
            args, kw = calls[-1]
            self.assertEqual(args[21:24], (0, 0, 0) if dynamic else (2031, 19, 29))
            self.assertEqual(kw["runtime_query_tokens"], 19 if dynamic else 0)
            kc = Tensor((2, 16, 2, 32))
            ns["update_triton_online_softmax_state_paged"](
                q, kc, kc, Tensor((2,)), state, key_tokens=29, **common)
            args, kw = calls[-1]
            self.assertEqual(args[24:27], (0, 0, 0) if dynamic else (2031, 19, 29))
            ns["finalize_triton_online_softmax_state"](q, state, output=q)
            args, kw = calls[-1]
            self.assertEqual(args[15], 0 if dynamic else 19)
            self.assertEqual(kw["runtime_query_tokens"], 19 if dynamic else 0)

    def test_all_kernels_disable_runtime_scalar_specialization(self):
        tree = ast.parse((V / "v1/attention/ops/tiered_triton_fused_attention.py").read_text())
        kernels = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name.endswith("_kernel")]
        self.assertEqual(len(kernels), 3)
        for kernel in kernels:
            runtime_args = {a.arg for a in kernel.args.args if a.arg.startswith("runtime_")}
            values = ast.literal_eval(kernel.decorator_list[0].keywords[0].value)
            self.assertEqual(runtime_args, set(values))


class DecodeBufferTests(unittest.TestCase):
    def test_unsupported_and_legacy_capacity_unchanged(self):
        opts = dict(phase="decode", max_tokens=16384, max_requests=10,
                    uniform_query_len=1, supported=True)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(flags.isolation_decode_token_capacity(**opts), 16384)
        with patch.dict(os.environ, {"VLLM_ISOLATION_MEMORY_OPTIMIZATIONS": "1"}):
            self.assertEqual(flags.isolation_decode_token_capacity(**opts), 16)
            for override in (dict(supported=False), dict(phase="prefill"),
                             dict(uniform_query_len=5)):
                self.assertEqual(flags.isolation_decode_token_capacity(
                    **(opts | override)), 16384)

    def test_actual_phase_factory_keeps_full_request_metadata(self):
        import sys
        from types import SimpleNamespace as NS
        path = V / "v1/worker/gpu_model_runner.py"
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_make_pd_phase_context")
        ns = dict(torch=NS(bool="bool", int32="int32", int64="int64"),
                  PDPhaseModelRunnerContext=lambda **k: NS(**k),
                  logger=NS(info=lambda *a: None))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[
                ast.alias(name="annotations")], level=0), method],
            type_ignores=[])), str(path), "exec"), ns)
        owner = NS(max_num_tokens=16384, max_num_reqs=10,
                   uniform_decode_query_len=1, supports_mm_inputs=False,
                   enable_prompt_embeds=False,
                   vllm_config=NS(speculative_config=None),
                   model_config=NS(is_encoder_decoder=False),
                   parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1),
                   inputs_embeds_size=2048, dtype="f16", dcp_world_size=1,
                   uses_mrope=False, uses_xdrope_dim=0, use_async_scheduling=False,
                   _make_buffer=lambda *shape, **kw: NS(shape=shape),
                   _make_pd_phase_input_batch=lambda: "unchanged-full-history")
        with patch.dict(sys.modules, {"vllm.isolation_memory": flags}), patch.dict(
            os.environ, {"VLLM_ISOLATION_MEMORY_OPTIMIZATIONS": "1"}
        ):
            decode = ns["_make_pd_phase_context"](owner, "decode", 0)
            prefill = ns["_make_pd_phase_context"](owner, "prefill", 0)
            owner._isolation_unused_embed_staging = True
            no_embed_decode = ns["_make_pd_phase_context"](owner, "decode", 0)
            no_embed_prefill = ns["_make_pd_phase_context"](owner, "prefill", 0)
        self.assertEqual(decode.inputs_embeds.shape, (16, 2048))
        self.assertEqual(prefill.inputs_embeds.shape, (16384, 2048))
        self.assertEqual(decode.seq_lens.shape, (10,))
        self.assertEqual(decode.query_start_loc.shape, (11,))
        self.assertEqual(decode.input_batch, "unchanged-full-history")
        self.assertEqual(no_embed_prefill.inputs_embeds.shape, (0, 2048))
        self.assertEqual(no_embed_decode.inputs_embeds.shape, (0, 2048))
        self.assertEqual(no_embed_prefill.input_ids.shape, (16384,))
        self.assertEqual(no_embed_decode.input_ids.shape, (16,))
        self.assertEqual(no_embed_decode.input_batch, "unchanged-full-history")


class PrefetchDependencyTests(unittest.TestCase):
    def test_only_gpu_source_groups_wait_current_qkv(self):
        from types import SimpleNamespace as NS
        ns = load_symbols(
            V / "v1/attention/ops/tiered_blockwise_attention.py",
            {"_ContextRingPrefetchPipeline"}, {
                "_torch": lambda: NS(),
            })
        copy = NS(waits=[])
        copy.wait_event = copy.waits.append
        event = object()
        pipeline = ns["_ContextRingPrefetchPipeline"](
            None, "cuda", non_blocking=True, cuda_event_timing_enabled=False,
            copy_stream=copy, source_ready_event=event)
        cpu = NS(device=NS(type="cpu"))
        gpu = NS(device=NS(type="cuda"))
        historical = NS(key=cpu, value=cpu)
        current = NS(key=gpu, value=gpu)
        pipeline._wait_for_gpu_sources([historical])
        self.assertEqual(copy.waits, [])
        self.assertEqual(pipeline.historical_early_groups, 1)
        pipeline._wait_for_gpu_sources([historical, current])
        self.assertEqual(copy.waits, [event])
        pipeline._wait_for_gpu_sources([current])
        self.assertEqual(copy.waits, [event])

    def test_numerical_module_flags_validate_same_as_common_helper(self):
        ns = load_symbols(
            V / "v1/attention/ops/tiered_blockwise_attention.py",
            {"_isolation_flag"}, {"os": os})
        for value in ("0", "1", "invalid"):
            with patch.dict(os.environ, {"VLLM_ISOLATION_MEMORY_OPTIMIZATIONS": value},
                            clear=True):
                if value == "invalid":
                    with self.assertRaises(ValueError):
                        ns["_isolation_flag"]("RING_STATE_REUSE")
                else:
                    self.assertEqual(ns["_isolation_flag"]("RING_STATE_REUSE"),
                                     flags.isolation_memory_enabled("RING_STATE_REUSE"))

    def test_private_ring_keeps_current_tail_and_slot_fences(self):
        text = (V / "v1/attention/ops/kv_retention_layerwise.py").read_text()
        self.assertIn("self.copy_stream.wait_event(ready)", text)
        self.assertIn("stream.wait_event(self.copy_event)", text)
        self.assertIn('getattr(self.executor, "isolation_early_prefetch", False)', text)
        text = (V / "v1/attention/ops/tiered_blockwise_attention.py").read_text()
        self.assertIn("self.staging_buffer.mark_slot_read_done(", text)
        self.assertIn("stream.wait_event(prefetched.event)", text)


if __name__ == "__main__":
    unittest.main()
