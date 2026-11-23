#!/usr/bin/env python3
"""CPU-only checkpoint contracts and real runner adapters with fake input rows."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
import sys
import time
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_exact_phase_cleanup import HAS_NATIVE_RELOAD_AUTHORITY

ROOT = Path(__file__).resolve().parents[2]
LAYER_PATH = ROOT / "vendor_v8/vllm/model_executor/layers/ttft_layerwise_prefill.py"
RUNNER_PATH = ROOT / "vendor_v8/vllm/v1/worker/gpu_model_runner.py"
SPEC = importlib.util.spec_from_file_location("kv_checkpoint_test_target", LAYER_PATH)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
FRAME_SPEC = importlib.util.spec_from_file_location(
    "kv_checkpoint_frame_target",
    ROOT / "vendor_v8/vllm/model_executor/layers/kv_retention_resume_frame.py")
FRAME = importlib.util.module_from_spec(FRAME_SPEC)
sys.modules[FRAME_SPEC.name] = FRAME
FRAME_SPEC.loader.exec_module(FRAME)
FLAG = "VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN"
NAMES = tuple(f"model.layers.{i}.self_attn.attn" for i in range(4))


class FakeFullAttentionSpec:
    block_size = 16
    sliding_window = None
    attention_chunk_size = None


class FakeUniformSpecs:
    pass


class FakeArray:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), max(map(len, rows)))

    def __getitem__(self, index):
        row, columns = index
        return self.rows[row][columns]


def load_runner_adapters():
    """Execute the actual methods without importing torch/CUDA/vLLM."""
    tree = ast.parse(RUNNER_PATH.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "GPUModelRunner")
    wanted = {
        "_ttft_kv_checkpoint_inputs", "_capture_ttft_kv_checkpoint",
        "_validate_ttft_kv_checkpoint", "_resume_ttft_prefill_continuation",
        "clear_pd_phase_execute_state",
        "_capture_kv_retention_resume_frame", "_import_kv_retention_resume_frame",
    }
    methods = [node for node in cls.body
               if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert len(methods) == len(wanted)
    code = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")],
                       level=0),
        ast.ClassDef(name="Runner", bases=[], keywords=[], body=methods,
                     decorator_list=[]),
    ], type_ignores=[])
    globals_ = {name: getattr(M, name) for name in (
        "KVRetentionCheckpointRow", "TTFTLayerwisePrefillError",
        "build_kv_retention_layerwise_checkpoint",
        "kv_retention_layerwise_checkpoint_requested",
        "TTFTLayerwisePrefillBoundary",
        "KVRetentionPhaseOwner",
    )}
    globals_.update(FullAttentionSpec=FakeFullAttentionSpec,
                    has_native_reload_authority=HAS_NATIVE_RELOAD_AUTHORITY,
                    KVRetentionLayerContinuation=FRAME.KVRetentionLayerContinuation,
                    UniformTypeKVCacheSpecs=FakeUniformSpecs,
                    kv_retention_layerwise_offload_enabled=lambda: False,
                    kv_retention_ring_scope=lambda owner: contextlib.nullcontext())
    exec(compile(ast.fix_missing_locations(code), str(RUNNER_PATH), "exec"), globals_)
    return globals_["Runner"], globals_, cls


RUNNER, RUNNER_GLOBALS, RUNNER_AST = load_runner_adapters()


def boundary(**changes):
    values = dict(
        scheduler_iteration_id=7, phase_submit_seq=9, phase_slot=1,
        request_ids=("long", "short"), scheduled_tokens=(("long", 5), ("short", 2)),
        profile_id="d30_p10", profile_epoch=1, prefill_mode="native",
        next_layer=2, end_layer=4, model_yield_count=1,
    )
    values.update(changes)
    return M.TTFTLayerwisePrefillBoundary(**values)


def make_runner_state():
    runner = RUNNER()
    blocks = FakeArray([[4], [1, 2]])
    table = NS(block_size=16, use_hybrid_blocks=False, blocks_per_kv_block=1,
               pcp_world_size=1, dcp_world_size=1, num_blocks_per_row=[1, 2],
               get_numpy_array=lambda: blocks)
    runner.input_batch = NS(
        block_table=[table], req_id_to_index={"short": 0, "long": 1}, num_reqs=2,
        req_ids=["short", "long"], num_computed_tokens_cpu=[0, 19],
    )
    # Deliberately reverse the cache group's list. Decoder order is authority.
    runner.kv_cache_config = NS(num_blocks=100, kv_cache_groups=[
        NS(kv_cache_spec=FakeFullAttentionSpec(), layer_names=tuple(reversed(NAMES)))
    ])
    model = NS(model=NS(start_layer=0, end_layer=4, layers=[
        NS(self_attn=NS(attn=NS(layer_name=name))) for name in NAMES
    ]))
    runner.get_model = lambda: model
    state = NS(
        kv_checkpoint=None, kv_offload=None, kv_resume_frame=None,
        boundary=boundary(), execution=NS(first_layer=0),
        scheduler_output=NS(num_scheduled_tokens={"long": 5, "short": 2},
                            pd_prefill_mode="native"),
        model_runner_debug_timings={},
    )
    return runner, state


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {FLAG: "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.runner, self.state = make_runner_state()

    def capture(self):
        self.runner._capture_ttft_kv_checkpoint(self.state)
        return self.state.kv_checkpoint

    def test_exact_per_layer_coverage_partial_tail_and_empty_prefix(self):
        plan = self.capture()
        self.assertEqual(plan.layer_names, NAMES)
        completed = list(plan.iter_layer_pages(1))
        pending = list(plan.iter_layer_pages(2))
        self.assertEqual([(p.request_id, p.physical_block, p.valid_tokens)
                          for p in completed],
                         [("long", 1, 16), ("long", 2, 8), ("short", 4, 2)])
        self.assertEqual([(p.request_id, p.valid_tokens) for p in pending],
                         [("long", 16), ("long", 3)])
        self.assertEqual(plan.rows[0].input_row, 1)
        self.assertEqual(plan.rows[1].input_row, 0)
        self.assertEqual(plan.as_debug_dict()[
            "kv_retention_checkpoint_layer_token_coverage"], 90)
        self.assertFalse(plan.release_authorized)
        self.assertTrue(all(isinstance(v, float) for v in plan.as_debug_dict().values()))

    def test_capture_is_tensor_free_and_frozen(self):
        plan = self.capture()
        with self.assertRaises(FrozenInstanceError):
            plan.next_layer = 3
        with self.assertRaises(FrozenInstanceError):
            plan.rows[0].block_ids = (80, 81)
        self.assertFalse(hasattr(plan, "attn_metadata"))
        self.assertFalse(hasattr(plan, "continuation"))

    def test_lightweight_trace_keeps_only_bounded_checkpoint_scalars(self):
        path = ROOT / "vendor_v8/vllm/v1/core/sched/observability.py"
        spec = importlib.util.spec_from_file_location("kv_checkpoint_observability", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metrics = self.capture().as_debug_dict()
        source = dict(metrics, kv_retention_checkpoint_block_ids=[1, 2, 4],
                      kv_retention_checkpoint_plan=self.state.kv_checkpoint)
        self.assertEqual(module.compact_worker_debug_timings(source), metrics)
        self.assertEqual(len(metrics), 8)
        self.assertIn("kv_retention_checkpoint_plan", source)

    def test_default_off_does_not_read_runner_inputs(self):
        with patch.dict(os.environ, {FLAG: "0"}):
            state = NS(kv_checkpoint=None, kv_offload=None,
                       scheduler_output=NS(pd_prefill_mode="native"))
            RUNNER()._capture_ttft_kv_checkpoint(state)
            RUNNER()._validate_ttft_kv_checkpoint(state)
            self.assertIsNone(state.kv_checkpoint)

    def test_flag_requires_existing_stage2(self):
        for enabled, preemption in (("0", "0"), ("1", "0"), ("0", "1")):
            with self.subTest(enabled=enabled, preemption=preemption), patch.dict(
                os.environ, {"VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION": enabled,
                             "VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION": preemption}
            ), self.assertRaisesRegex(M.TTFTLayerwisePrefillError, "requires"):
                with M.ttft_layerwise_prefill_execution("decode"):
                    pass

    def test_legal_profile_change_keeps_kv_checkpoint_valid(self):
        self.capture()
        self.state.boundary.profile_id = "d0_p40"
        self.state.boundary.profile_epoch = 2
        self.runner._validate_ttft_kv_checkpoint(self.state)

    def test_owner_and_boundary_changes_are_rejected(self):
        plan = self.capture()
        for field, value in dict(
            phase_slot=2, phase_submit_seq=10, scheduler_iteration_id=8,
            request_ids=("short", "long"), next_layer=3, end_layer=5,
            scheduled_tokens=(("long", 6), ("short", 2)),
            model_yield_count=2, prefill_mode="context_ring",
        ).items():
            with self.subTest(field=field), self.assertRaises(M.TTFTLayerwisePrefillError):
                plan.validate_resume(boundary=boundary(**{field: value}),
                                     rows=plan.rows, layer_names=NAMES)

    def test_rows_and_layer_namespace_changes_are_rejected(self):
        plan = self.capture()
        for row in (replace(plan.rows[0], start_token=18),
                    replace(plan.rows[0], scheduled_tokens=4),
                    replace(plan.rows[0], input_row=0),
                    replace(plan.rows[0], block_ids=(7, 8))):
            with self.subTest(row=row), self.assertRaises(M.TTFTLayerwisePrefillError):
                plan.validate_resume(boundary=boundary(), rows=(row, plan.rows[1]),
                                     layer_names=NAMES)
        with self.assertRaises(M.TTFTLayerwisePrefillError):
            plan.validate_resume(boundary=boundary(), rows=plan.rows,
                                 layer_names=tuple(reversed(NAMES)))

    def test_existing_plan_still_validates_when_flag_is_disabled(self):
        self.capture()
        self.runner.input_batch.block_table[0].get_numpy_array().rows[1][0] = 7
        with patch.dict(os.environ, {FLAG: "0"}), self.assertRaisesRegex(
            M.TTFTLayerwisePrefillError, "mapping"
        ):
            self.runner._validate_ttft_kv_checkpoint(self.state)

    def test_repeated_yield_advances_only_completed_layer_coverage(self):
        first = self.capture()
        self.runner._validate_ttft_kv_checkpoint(self.state)
        self.state.boundary = boundary(next_layer=3, model_yield_count=2)
        second = self.capture()
        self.assertEqual(first.rows, second.rows)
        self.assertEqual(second.next_layer, 3)
        self.assertEqual(sum(p.valid_tokens for p in first.iter_layer_pages(2)), 19)
        self.assertEqual(sum(p.valid_tokens for p in second.iter_layer_pages(2)), 26)
        self.assertFalse(second.release_authorized)

    def test_recapture_cannot_hide_a_row_mapping_change(self):
        first = self.capture()
        self.state.boundary = boundary(next_layer=3, model_yield_count=2)
        self.runner.input_batch.block_table[0].get_numpy_array().rows[1][0] = 7
        with self.assertRaises(M.TTFTLayerwisePrefillError):
            self.capture()
        self.assertIs(self.state.kv_checkpoint, first)

    def test_recapture_requires_same_owner_and_exact_next_yield_sequence(self):
        for changes in (dict(phase_submit_seq=10), dict(model_yield_count=3),
                        dict(next_layer=1), dict(next_layer=2)):
            with self.subTest(changes=changes):
                self.runner, self.state = make_runner_state()
                first = self.capture()
                values = dict(next_layer=3, model_yield_count=2)
                values.update(changes)
                self.state.boundary = boundary(**values)
                with self.assertRaises(M.TTFTLayerwisePrefillError):
                    self.capture()
                self.assertIs(self.state.kv_checkpoint, first)

    def test_shared_prefix_referenced_blocks_never_grant_free_permission(self):
        names, rows = self.runner._ttft_kv_checkpoint_inputs(self.state)
        rows = (rows[0], replace(rows[1], start_token=16, block_ids=(1, 4)))
        plan = M.build_kv_retention_layerwise_checkpoint(
            boundary=boundary(), first_layer=0, layer_names=names, rows=rows,
            gpu_boundary_complete=True,
        )
        self.assertEqual(plan.as_debug_dict()[
            "kv_retention_checkpoint_unique_referenced_blocks"], 3)
        self.assertFalse(plan.release_authorized)

    def test_uncompleted_boundary_never_creates_plan(self):
        names, rows = self.runner._ttft_kv_checkpoint_inputs(self.state)
        with self.assertRaisesRegex(M.TTFTLayerwisePrefillError, "completed"):
            M.build_kv_retention_layerwise_checkpoint(
                boundary=boundary(), first_layer=0, layer_names=names, rows=rows,
                gpu_boundary_complete=False,
            )

    def test_invalid_rows_and_ranges_are_rejected(self):
        _, rows = self.runner._ttft_kv_checkpoint_inputs(self.state)
        for changes in (dict(block_ids=(0, 1)), dict(block_ids=(-1, 2)),
                        dict(block_ids=(1, 1)), dict(block_ids=(1,)),
                        dict(block_ids=[1, 2]), dict(input_row=-1),
                        dict(scheduled_tokens=0), dict(start_token=-1)):
            with self.subTest(changes=changes), self.assertRaises(M.TTFTLayerwisePrefillError):
                replace(rows[0], **changes)

    def test_unsupported_cache_layouts_fail_closed(self):
        cases = (
            ("table", "use_hybrid_blocks", True),
            ("table", "blocks_per_kv_block", 2),
            ("table", "pcp_world_size", 2), ("table", "dcp_world_size", 2),
            ("spec", "sliding_window", 128),
            ("spec", "attention_chunk_size", 16), ("spec", "block_size", 32),
        )
        for target, name, value in cases:
            with self.subTest(name=name):
                runner, state = make_runner_state()
                obj = (runner.input_batch.block_table[0] if target == "table"
                       else runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec)
                setattr(obj, name, value)
                with self.assertRaises(M.TTFTLayerwisePrefillError):
                    runner._capture_ttft_kv_checkpoint(state)
                self.assertIsNone(state.kv_checkpoint)

    def test_lost_row_pool_bounds_or_model_mapping_fail_closed(self):
        mutations = (
            lambda r: r.input_batch.req_id_to_index.pop("long"),
            lambda r: r.input_batch.req_ids.__setitem__(1, "replacement"),
            lambda r: r.input_batch.block_table[0].get_numpy_array().rows[1].__setitem__(0, 100),
            lambda r: r.kv_cache_config.kv_cache_groups.append(r.kv_cache_config.kv_cache_groups[0]),
            lambda r: setattr(r.get_model().model, "start_layer", 1),
            lambda r: setattr(r.kv_cache_config.kv_cache_groups[0], "layer_names", ("bad",)),
        )
        for mutate in mutations:
            runner, state = make_runner_state()
            mutate(runner)
            with self.assertRaises(M.TTFTLayerwisePrefillError):
                runner._capture_ttft_kv_checkpoint(state)
            self.assertIsNone(state.kv_checkpoint)

    def test_real_resume_rejects_stale_table_before_advancing_generator(self):
        self.capture()
        advanced = []

        def continuation():
            advanced.append(True)
            yield boundary(next_layer=3, model_yield_count=2)

        self.state.continuation = continuation()
        self.runner.input_batch.num_computed_tokens_cpu[1] = 18
        context = NS(ttft_prefill_continuation=self.state)
        with patch.dict(RUNNER_GLOBALS, {
            "_PD_ACTIVE_MODEL_RUNNER_CONTEXT": NS(get=lambda: context),
            "active_ttft_layerwise_prefill_execution": lambda: self.state.execution,
        }), self.assertRaisesRegex(M.TTFTLayerwisePrefillError, "mapping"):
            self.runner._resume_ttft_prefill_continuation(
                self.state.scheduler_output, self.state
            )
        self.assertEqual(advanced, [])
        self.assertIs(context.ttft_prefill_continuation, self.state)

    def test_initial_and_repeat_capture_follow_existing_boundary_wait(self):
        for method_name in ("_execute_model_impl", "_resume_ttft_prefill_continuation"):
            method = next(node for node in RUNNER_AST.body
                          if isinstance(node, ast.FunctionDef) and node.name == method_name)
            for node in ast.walk(method):
                if not isinstance(node, ast.If):
                    continue
                statements = [ast.unparse(statement) for statement in node.body]
                captures = [i for i, text in enumerate(statements)
                            if text.startswith("self._capture_ttft_kv_checkpoint(")]
                if captures:
                    waits = [i for i, text in enumerate(statements)
                             if "self._synchronize_ttft_prefill_boundary()" in text]
                    self.assertEqual(len(captures), 1)
                    self.assertEqual(len(waits), 1)
                    self.assertLess(waits[0], captures[0])
                    break
            else:
                self.fail(f"missing capture/wait hook in {method_name}")

    def test_real_resume_recaptures_then_finishes_and_releases_slot(self):
        self.capture()
        output = object()

        def continuation():
            yield boundary(next_layer=3, model_yield_count=2)
            return output

        state = self.state
        state.continuation = continuation()
        for name in ("attn_metadata", "num_tokens_across_dp", "cudagraph_mode",
                     "batch_desc", "ubatch_slices_padded", "logits_indices",
                     "spec_decode_metadata", "spec_decode_common_attn_metadata",
                     "ec_connector_output", "cudagraph_stats", "num_scheduled_tokens_np"):
            setattr(state, name, None)
        state.num_tokens_padded = state.num_scheduled_tokens = 7
        state.num_reqs = 2
        self.runner.vllm_config = None
        self.runner._pd_green_context_execute_phase = "prefill"
        self.runner._pd_cuda_phase_trace = lambda *args: contextlib.nullcontext()
        self.runner._accumulate_ttft_group_debug = lambda *args, **kwargs: None
        waits = []
        self.runner._synchronize_ttft_prefill_boundary = lambda: waits.append(True) or 0.0
        finishes = []
        self.runner._finish_execute_model_forward = lambda **kwargs: finishes.append(kwargs) or output
        context = NS(ttft_prefill_continuation=state)
        overrides = {
            "_PD_ACTIVE_MODEL_RUNNER_CONTEXT": NS(get=lambda: context),
            "active_ttft_layerwise_prefill_execution": lambda: state.execution,
            "time": time,
            "set_forward_context": lambda *a, **kw: contextlib.nullcontext(),
            "record_function_or_nullcontext": lambda *a: contextlib.nullcontext(),
        }
        for name in ("reset_operator_selective_pd_metrics", "reset_opmux_metrics",
                     "reset_tiered_context_ring_runtime_metrics"):
            overrides[name] = lambda: None
        with patch.dict(RUNNER_GLOBALS, overrides):
            result = self.runner._resume_ttft_prefill_continuation(state.scheduler_output, state)
            self.assertEqual(result.next_layer, 3)
            self.assertEqual(state.kv_checkpoint.next_layer, 3)
            self.assertIs(context.ttft_prefill_continuation, state)
            result = self.runner._resume_ttft_prefill_continuation(state.scheduler_output, state)
        self.assertIs(result, output)
        self.assertEqual(waits, [True])
        self.assertEqual(len(finishes), 1)
        self.assertIsNone(context.ttft_prefill_continuation)

    def test_existing_error_cancel_cleanup_drops_plan_with_original_slot(self):
        plan = self.capture()
        self.state.scheduler_output.pd_phase_slot = 1
        identity = M.KVRetentionPhaseOwner.from_output(
            "prefill:1", self.state.scheduler_output
        )
        closed = []
        self.state.continuation = NS(close=lambda: closed.append(True))
        owner = NS(ttft_prefill_continuation=self.state, execute_model_state=None,
                   kv_retention_execute_owner=identity)
        unrelated = NS(ttft_prefill_continuation=object())
        self.runner._pd_phase_contexts = {("prefill", 1): owner, ("prefill", 0): unrelated}
        self.assertTrue(self.runner.clear_pd_phase_execute_state("prefill:1", identity))
        self.assertEqual(closed, [True])
        self.assertIsNone(owner.ttft_prefill_continuation)
        self.assertIsNotNone(unrelated.ttft_prefill_continuation)
        self.assertFalse(plan.release_authorized)


if __name__ == "__main__":
    unittest.main()
