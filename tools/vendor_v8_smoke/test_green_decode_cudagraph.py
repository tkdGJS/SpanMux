#!/usr/bin/env python3
"""CPU-only source contracts for GreenContext Decode CUDA graphs."""

from __future__ import annotations

import ast
import enum
import importlib.util
from itertools import islice, product
import time
from types import SimpleNamespace
from typing import NamedTuple
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def load_dispatcher_without_cuda():
    """Execute the checkout's real classes, excluding only GPU-heavy imports."""
    namespace = {
        "__name__": __name__, "enum": enum, "NamedTuple": NamedTuple,
        "product": product, "VllmConfig": object,
    }
    for path, name in (
        ("vendor_v8/vllm/config/compilation.py", "CUDAGraphMode"),
        ("vendor_v8/vllm/forward_context.py", "BatchDescriptor"),
        ("vendor_v8/vllm/v1/cudagraph_dispatcher.py", "CudagraphDispatcher"),
    ):
        tree = ast.parse(read(path), filename=path)
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace["CUDAGraphMode"], namespace["CudagraphDispatcher"]


def load_llama_forward_without_cuda(active_execution, handoff=lambda index: None):
    """Run the real forward control flow with small layer/PP test doubles."""
    path = "vendor_v8/vllm/model_executor/models/llama.py"
    tree = ast.parse(read(path), filename=path)
    model = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name == "LlamaModel")
    methods = [n for n in model.body if isinstance(n, ast.FunctionDef)
               and n.name in ("forward", "_forward_decoder_layer_range")]
    pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    namespace = {
        "islice": islice, "time": time, "get_pp_group": lambda: pp,
        "active_ttft_layerwise_prefill_execution": active_execution,
        "pd_layerwise_handoff_layer_boundary": handoff,
        "kv_retention_resume_frame_requested": lambda: False,
        "TTFTLayerwisePrefillModelYield": SimpleNamespace,
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), path, "exec"), namespace)
    return namespace


def minimal_llama(namespace, *, enabled=False, handoff=False):
    model_type = type("MinimalLlama", (), {
        "forward": namespace["forward"],
        "_forward_decoder_layer_range": namespace["_forward_decoder_layer_range"],
    })
    model = model_type()
    model.pd_prefill_workspace_enabled = False
    model.pd_layerwise_profile_handoff_enabled = handoff
    model.ttft_layerwise_prefill_execution_enabled = enabled
    model.aux_hidden_state_layers = ()
    model.start_layer, model.end_layer = 0, 3
    model.layers = [lambda positions, hidden, residual: (hidden + 1, residual)] * 3
    model.norm = lambda hidden, residual: (hidden, residual)
    return model


class GreenDecodeCUDAGraphContractTest(unittest.TestCase):
    def test_disabled_layer_execution_never_reads_context_even_with_sm_handoff(self):
        def forbidden_context_read():
            raise AssertionError("disabled feature read ContextVar")

        for handoff_enabled in (False, True):
            with self.subTest(handoff=handoff_enabled):
                boundaries = []
                ns = load_llama_forward_without_cuda(
                    forbidden_context_read, boundaries.append)
                model = minimal_llama(ns, handoff=handoff_enabled)
                self.assertEqual(model.forward(None, None, None, inputs_embeds=5), 8)
                self.assertEqual(boundaries, [0, 1, 2] if handoff_enabled else [])

    def test_enabled_layer_execution_still_observes_context_and_group_boundaries(self):
        reads, groups = [], []
        execution = SimpleNamespace(
            preemption_active=False, group_layers=2,
            record_group=lambda **group: groups.append(group))

        def active():
            reads.append(True)
            return execution

        ns = load_llama_forward_without_cuda(active)
        model = minimal_llama(ns, enabled=True)
        self.assertEqual(model.forward(None, None, None, inputs_embeds=5), 8)
        self.assertEqual(len(reads), 1)
        self.assertEqual([(g["start_layer"], g["end_layer"]) for g in groups],
                         [(0, 2), (2, 3)])
        execution = None
        self.assertEqual(model.forward(None, None, None, inputs_embeds=5), 8)
        self.assertEqual(len(reads), 2)

    def test_enabled_preemption_keeps_yield_and_continuation_identity(self):
        execution = SimpleNamespace(preemption_active=True)
        ns = load_llama_forward_without_cuda(lambda: execution)
        model = minimal_llama(ns, enabled=True)
        boundary = object()

        def resume(**kwargs):
            self.assertIs(kwargs["execution"], execution)
            yield boundary
            return 8

        model._ttft_resumable_prefill_forward = resume
        result = model.forward(None, None, None, inputs_embeds=5)
        self.assertIs(result.boundary, boundary)
        with self.assertRaises(StopIteration) as done:
            next(result.continuation)
        self.assertEqual(done.exception.value, 8)

    def test_constructor_and_compile_use_same_static_execution_predicate(self):
        path = "vendor_v8/vllm/model_executor/models/llama.py"
        tree = ast.parse(read(path))
        model = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                     and n.name == "LlamaModel")
        init = next(n for n in model.body if isinstance(n, ast.FunctionDef)
                    and n.name == "__init__")
        assignment = next(n for n in ast.walk(init) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Attribute) and
                                  t.attr == "ttft_layerwise_prefill_execution_enabled"
                                  for t in n.targets))
        self.assertEqual(ast.unparse(assignment.value),
                         "ttft_layerwise_prefill_execution_requested()")
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "llama_model_torch_compile_enabled")
        for execution, handoff, workspace in product((False, True), repeat=3):
            ns = {
                "ttft_layerwise_prefill_execution_requested": lambda: execution,
                "layerwise_profile_handoff_requested": lambda: handoff,
                "prefill_workspace_requested": lambda: workspace,
                "logger": SimpleNamespace(info_once=lambda *args: None),
            }
            exec(compile(ast.Module(body=[helper], type_ignores=[]), path, "exec"), ns)
            self.assertEqual(ns[helper.name](None),
                             not (execution or handoff or workspace))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
    def test_disabled_real_forward_compiles_fullgraph_on_cpu(self):
        import contextvars
        import torch

        context = contextvars.ContextVar("test_layer_execution", default=None)
        ns = load_llama_forward_without_cuda(lambda: context.get())

        class Layer(torch.nn.Module):
            def forward(self, positions, hidden, residual):
                return hidden + 1, residual

        class Norm(torch.nn.Module):
            def forward(self, hidden, residual):
                return hidden, residual

        class Model(torch.nn.Module):
            forward = ns["forward"]

            def __init__(self):
                super().__init__()
                self.pd_prefill_workspace_enabled = False
                self.pd_layerwise_profile_handoff_enabled = False
                self.ttft_layerwise_prefill_execution_enabled = False
                self.aux_hidden_state_layers = ()
                self.start_layer, self.end_layer = 0, 2
                self.layers = torch.nn.ModuleList([Layer(), Layer()])
                self.norm = Norm()

        model = Model()
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        x = torch.ones(2, 4)
        torch.testing.assert_close(compiled(None, x, None, inputs_embeds=x), x + 2)

    def test_decode16_ceiling_has_full_keys_for_every_admitted_width(self) -> None:
        mode, dispatcher_type = load_dispatcher_without_cuda()
        sizes = [1, 2, 4, 8, 16, 24, 32]
        config = SimpleNamespace(
            compilation_config=SimpleNamespace(
                cudagraph_mode=mode.FULL_AND_PIECEWISE,
                cudagraph_capture_sizes=sizes,
                max_cudagraph_capture_size=max(sizes),
                is_attention_compiled_piecewise=lambda: True,
            ),
            speculative_config=None, lora_config=None,
            scheduler_config=SimpleNamespace(max_num_seqs=16),
            pad_for_cudagraph=lambda n: next(s for s in sizes if s >= n),
        )
        dispatcher = dispatcher_type(config)
        dispatcher.initialize_cudagraph_keys(mode.FULL_AND_PIECEWISE, 1)
        for batch in range(1, 17):
            with self.subTest(batch=batch):
                selected, descriptor = dispatcher.dispatch(batch, True, False)
                self.assertEqual(selected, mode.FULL)
                self.assertEqual(descriptor.num_tokens, config.pad_for_cudagraph(batch))
                self.assertLessEqual(descriptor.num_reqs, 16)
                self.assertIn(descriptor.num_tokens, [s for s in sizes if s <= 16])
                if batch > 8:
                    self.assertEqual(descriptor.num_tokens, 16)

    def test_decode_padding_has_full_keys_through_admission_ceiling(self) -> None:
        mode, dispatcher_type = load_dispatcher_without_cuda()
        for sizes, expected_mode in (
            ([1, 2, 4, 8, 16], mode.PIECEWISE),
            ([1, 2, 4, 8, 10, 16], mode.FULL),
        ):
            with self.subTest(sizes=sizes):
                config = SimpleNamespace(
                    compilation_config=SimpleNamespace(
                        cudagraph_mode=mode.FULL_AND_PIECEWISE,
                        cudagraph_capture_sizes=sizes,
                        max_cudagraph_capture_size=max(sizes),
                        is_attention_compiled_piecewise=lambda: True,
                    ),
                    speculative_config=None, lora_config=None,
                    scheduler_config=SimpleNamespace(max_num_seqs=10),
                    pad_for_cudagraph=lambda n: next(s for s in sizes if s >= n),
                )
                dispatcher = dispatcher_type(config)
                dispatcher.initialize_cudagraph_keys(mode.FULL_AND_PIECEWISE, 1)
                for batch in range(1, 11):
                    selected, descriptor = dispatcher.dispatch(batch, True, False)
                    self.assertEqual(selected, mode.FULL if batch <= 8 else expected_mode)
                    if selected == mode.FULL:
                        self.assertLessEqual(descriptor.num_reqs, 10)
                        # Same filtering rule as Green profile/slot capture.
                        self.assertIn(descriptor.num_tokens, [s for s in sizes if s <= 10])

    def test_graph_entries_are_namespaced_by_phase_slot_buffers(self) -> None:
        source = read("vendor_v8/vllm/compilation/cuda_graph.py")
        self.assertIn("def cudagraph_namespace(namespace: str)", source)
        self.assertIn("entry_key = (namespace, batch_descriptor)", source)
        self.assertIn("namespaced_graph_pools", source)
        self.assertIn("def has_cudagraph_entry(", source)

    def test_decode_capture_uses_phase_context_and_green_stream(self) -> None:
        source = read("vendor_v8/vllm/v1/worker/gpu_model_runner.py")
        body = source.split(
            "def _capture_pd_green_decode_cudagraphs(", 1
        )[1].split("def capture_model(", 1)[0]
        self.assertIn("_use_pd_phase_model_runner_context(execution_key)", body)
        self.assertIn("green_context_phase_stream(", body)
        self.assertIn("cudagraph_namespace(namespace)", body)
        self.assertIn("uniform_decode=True", body)
        self.assertIn("decode_mode != CUDAGraphMode.FULL", body)

    def test_prefill_remains_eager_and_strict_decode_fails_closed(self) -> None:
        source = read("vendor_v8/vllm/v1/worker/gpu_model_runner.py")
        determine = source.split(
            "def _determine_batch_execution_and_padding(", 1
        )[1].split("def _register_layerwise_nvtx_hooks(", 1)[0]
        self.assertIn(
            'self._pd_green_context_execute_phase != "decode"', determine
        )
        self.assertIn("force_eager = True", determine)
        self.assertIn(
            "strict GreenContext Decode CUDA graph execution fell", source
        )
        self.assertIn(
            "GreenContext Decode received a non-uniform live-token ",
            source,
        )
        self.assertIn("pd_green_decode_uniform_batch", source)
        self.assertIn("pd_green_decode_cudagraph_active", source)
        self.assertIn("pd_green_decode_cudagraph_entry_ready", source)

    def test_layerwise_prefill_bypasses_fullgraph_compile(self) -> None:
        source = read("vendor_v8/vllm/model_executor/models/llama.py")
        helper = source.split(
            "def llama_model_torch_compile_enabled(", 1
        )[1].split("@support_torch_compile(", 1)[0]
        decorator = source.split("@support_torch_compile(", 1)[1].split(
            "class LlamaModel", 1
        )[0]
        self.assertIn(
            "layerwise_requested = layerwise_profile_handoff_requested()",
            helper,
        )
        self.assertIn(
            "ttft_layerwise_prefill_execution_requested()",
            helper,
        )
        self.assertIn("not ttft_layer_groups_requested", helper)
        self.assertIn(
            "Layerwise Prefill execution/handoff or fixed Prefill workspace",
            helper,
        )
        self.assertIn(
            "enable_if=llama_model_torch_compile_enabled", decorator
        )

    def test_engine_accepts_selective_eager_layerwise_model(self) -> None:
        source = read("vendor_v8/vllm/v1/engine/core.py")
        validation = source.split(
            "layerwise_global_eager = (", 1
        )[1].split("self.pd_layerwise_handoff_enabled", 1)[0]
        self.assertIn("layerwise_selective_eager", validation)
        self.assertIn(
            "or ttft_layerwise_prefill_execution_requested()",
            validation,
        )
        self.assertIn("layerwise_model_eager", validation)
        self.assertIn("and layerwise_model_eager", validation)
        self.assertIn("global_eager=%s selective_eager=%s", validation)

    def test_runner_does_not_force_eager_for_compatible_path(self) -> None:
        starter = read("start_vllm_causality_microbench.sh")
        quick = read("run_vendor_v8_quick_tbt_sm_scaling.sh")
        self.assertIn('PD_GREEN_DECODE_CUDAGRAPH="${', starter)
        self.assertIn(
            '"${OPERATOR_SELECTIVE_GREEN_CONTEXT}" == "1" && '
            '"${PD_GREEN_DECODE_CUDAGRAPH}" != "1"',
            starter,
        )
        self.assertIn("cannot run with module/component hooks", starter)
        self.assertIn("VLLM_PD_GREEN_DECODE_CUDAGRAPH", quick)
        self.assertIn("VLLM_PD_GREEN_DECODE_CUDAGRAPH_STRICT", quick)


if __name__ == "__main__":
    unittest.main(verbosity=2)
