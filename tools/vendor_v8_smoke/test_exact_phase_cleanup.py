#!/usr/bin/env python3
"""Run real cleanup adapters with CPU doubles; no torch/CUDA imports required."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import sys
import unittest
from concurrent.futures import Future
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import SimpleNamespace as NS


ROOT = Path(__file__).resolve().parents[2]
LAYER = ROOT / "vendor_v8/vllm/model_executor/layers/ttft_layerwise_prefill.py"
SPEC = importlib.util.spec_from_file_location("exact_phase_cleanup_contract", LAYER)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
OWNER = M.KVRetentionPhaseOwner


def load_native_reload_guard():
    path = ROOT / "vendor_v8/vllm/v1/core/tiered_execution.py"
    tree = ast.parse(path.read_text())
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "has_native_reload_authority")
    scope = {"TIERED_EXECUTION_NATIVE_RELOAD": "native_reload"}
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[definition], type_ignores=[])), str(path), "exec"), scope)
    return scope["has_native_reload_authority"]


HAS_NATIVE_RELOAD_AUTHORITY = load_native_reload_guard()


def adapters(relative, class_name, names, extra=None, functions=()):
    tree = ast.parse((ROOT / relative).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == class_name)
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    for method in methods:
        method.decorator_list = []
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")],
                           level=0)]
    body.extend(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in functions)
    body.append(ast.ClassDef(name=class_name, bases=[], keywords=[], body=methods,
                             decorator_list=[]))
    scope = {"KVRetentionPhaseOwner": OWNER, "Future": Future,
             "has_native_reload_authority": HAS_NATIVE_RELOAD_AUTHORITY}
    scope.update(extra or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 str(ROOT / relative), "exec"), scope)
    return scope[class_name]


RUNNER_PATH = "vendor_v8/vllm/v1/worker/gpu_model_runner.py"
RUNNER = adapters(RUNNER_PATH, "GPUModelRunner",
                  {"_claim_pd_execute_state", "clear_pd_phase_execute_state"})
EXECUTOR = adapters("vendor_v8/vllm/v1/executor/uniproc_executor.py", "UniProcExecutor",
                    {"clear_pd_phase_execute_state", "execute_model",
                     "execute_model_pd_pair", "sample_tokens"},
                    functions=("_pd_parse_execution_key", "_pd_execution_key"))
ENGINE = adapters("vendor_v8/vllm/v1/engine/core.py", "EngineCore",
                  {"_cancel_ttft_prefill_worker_slot"})


def output(slot=0, sequence=7):
    return NS(pd_phase_slot=slot, scheduler_iteration_id=3, pd_phase_submit_seq=sequence,
              pd_sm_profile_id="d30_p10", pd_sm_profile_epoch=1,
              total_num_scheduled_tokens=2, num_scheduled_tokens={"A": 1, "B": 1})


def context(key, scheduler_output, events=None, offload=False):
    events = [] if events is None else events
    continuation = NS(
        scheduler_output=scheduler_output,
        continuation=NS(close=lambda: events.append("generator")),
        kv_offload=NS(close=lambda: events.append("copy")) if offload else None,
    )
    return NS(kv_retention_execute_owner=OWNER.from_output(key, scheduler_output),
              ttft_prefill_continuation=continuation, execute_model_state=None,
              kv_connector_output=object(), _pd_green_context_sample_phase="prefill",
              _pd_green_context_sample_profile_id="d30_p10",
              _pd_green_context_sample_profile_epoch=1,
              _pd_decode_gpu_component_capture=object())


class ExactCleanupTest(unittest.TestCase):
    def setUp(self):
        self.runner = RUNNER()
        self.out = output()
        self.events = []
        self.ctx = context("prefill", self.out, self.events)
        self.other = context("prefill:1", output(1))
        self.decode = context("decode", output())
        self.runner._pd_phase_contexts = {
            ("prefill", 0): self.ctx, ("prefill", 1): self.other,
            ("decode", 0): self.decode,
        }
        self.owner = self.ctx.kv_retention_execute_owner

    def test_bare_slot_zero_does_not_clear_other_prefill_or_decode(self):
        other, decode = vars(self.other).copy(), vars(self.decode).copy()
        self.assertTrue(self.runner.clear_pd_phase_execute_state("prefill", self.owner))
        self.assertEqual(vars(self.other), other)
        self.assertEqual(vars(self.decode), decode)
        self.assertEqual(self.events, ["generator"])
        for attr in ("execute_model_state", "ttft_prefill_continuation",
                     "kv_connector_output", "kv_retention_execute_owner",
                     "_pd_decode_gpu_component_capture", "_pd_green_context_sample_phase"):
            self.assertIsNone(getattr(self.ctx, attr))

    def test_explicit_zero_alias_and_duplicate_cleanup(self):
        self.assertTrue(self.runner.clear_pd_phase_execute_state("prefill:0", self.owner))
        self.assertFalse(self.runner.clear_pd_phase_execute_state("prefill", self.owner))
        self.assertEqual(self.events, ["generator"])

    def test_slot_one_preserves_zero_and_decode(self):
        before = vars(self.ctx).copy()
        self.assertTrue(self.runner.clear_pd_phase_execute_state(
            "prefill:1", self.other.kv_retention_execute_owner))
        self.assertEqual(vars(self.ctx), before)
        self.assertIsNotNone(self.decode.ttft_prefill_continuation)

    def test_decode_cleanup_preserves_prefill(self):
        before = vars(self.ctx).copy()
        self.assertTrue(self.runner.clear_pd_phase_execute_state(
            "decode", self.decode.kv_retention_execute_owner))
        self.assertEqual(vars(self.ctx), before)

    def test_stale_output_with_identical_counters_is_nondestructive(self):
        replacement = output()
        self.ctx = context("prefill", replacement, self.events)
        self.runner._pd_phase_contexts[("prefill", 0)] = self.ctx
        before = vars(self.ctx).copy()
        self.assertFalse(self.runner.clear_pd_phase_execute_state("prefill", self.owner))
        self.assertEqual(vars(self.ctx), before)
        self.assertEqual(self.events, [])

    def test_absent_slot_is_nondestructive(self):
        self.assertFalse(self.runner.clear_pd_phase_execute_state(
            "prefill:2", OWNER.from_output("prefill:2", output(2))))

    def test_invalid_or_mismatched_key_does_not_mutate(self):
        before = vars(self.ctx).copy()
        for key in ("", "pd_pair", "prefill:", "prefill:-1", "prefill:bad",
                    "prefill:1:0", "prefill:１", "decode", "prefill:1"):
            with self.subTest(key=key), self.assertRaises((ValueError, RuntimeError)):
                self.runner.clear_pd_phase_execute_state(key, self.owner)
        self.assertEqual(vars(self.ctx), before)

    def test_retained_output_mismatch_fails_before_close(self):
        self.ctx.execute_model_state = NS(scheduler_output=output(sequence=8))
        before = vars(self.ctx).copy()
        with self.assertRaisesRegex(RuntimeError, "retained-state"):
            self.runner.clear_pd_phase_execute_state("prefill", self.owner)
        self.assertEqual(vars(self.ctx), before)
        self.assertEqual(self.events, [])

    def test_copy_drains_before_generator(self):
        self.ctx.ttft_prefill_continuation.kv_offload = NS(
            close=lambda: self.events.append("copy"))
        self.runner.clear_pd_phase_execute_state("prefill", self.owner)
        self.assertEqual(self.events, ["copy", "generator"])

    def test_close_failure_preserves_ownership(self):
        for which in ("copy", "generator"):
            with self.subTest(which=which):
                def fail():
                    raise RuntimeError("close failed")
                ctx = context("prefill", self.out, offload=True)
                target = (ctx.ttft_prefill_continuation.kv_offload if which == "copy"
                          else ctx.ttft_prefill_continuation.continuation)
                target.close = fail
                self.runner._pd_phase_contexts[("prefill", 0)] = ctx
                before = vars(ctx).copy()
                with self.assertRaisesRegex(RuntimeError, "close failed"):
                    self.runner.clear_pd_phase_execute_state("prefill", self.owner)
                self.assertEqual(vars(ctx), before)

    def test_early_failure_owner_is_clearable_without_handoff(self):
        self.ctx.ttft_prefill_continuation = None
        self.ctx.kv_retention_execute_owner = None
        self.runner._claim_pd_execute_state("prefill", self.out, self.ctx)
        self.assertTrue(self.runner.clear_pd_phase_execute_state("prefill", self.owner))

    def test_live_owner_cannot_be_replaced(self):
        for attr in ("execute_model_state", "ttft_prefill_continuation"):
            with self.subTest(attr=attr):
                setattr(self.ctx, attr, NS(scheduler_output=self.out))
                before = vars(self.ctx).copy()
                with self.assertRaisesRegex(RuntimeError, "live owner"):
                    self.runner._claim_pd_execute_state("prefill", output(), self.ctx)
                self.assertEqual(vars(self.ctx), before)

    def test_same_continuation_can_resume_after_sm_retarget(self):
        self.out.pd_sm_profile_id = "d0_p40"
        self.out.pd_sm_profile_epoch += 1
        self.runner._claim_pd_execute_state("prefill:0", self.out, self.ctx)
        self.assertEqual(self.ctx.kv_retention_execute_owner, self.owner)

    def test_scalar_immutable_identity_and_legacy_counters(self):
        legacy = NS()
        first = OWNER.from_output("prefill", legacy)
        self.assertEqual(first.scheduler_iteration_id, -1)
        self.assertNotEqual(first, OWNER.from_output("prefill", NS()))
        self.assertTrue(all(isinstance(getattr(first, f.name), (int, str, type(None)))
                            for f in fields(first)))
        with self.assertRaises(FrozenInstanceError):
            first.execution_key = "decode"
        with self.assertRaisesRegex(RuntimeError, "slot"):
            OWNER.from_output("prefill:1", legacy)

    def test_real_model_entry_claims_before_mutable_prepare(self):
        source = (ROOT / RUNNER_PATH).read_text()
        body = source.split("    def execute_model(\n", 1)[1].split(
            "    def ", 1)[0]
        self.assertLess(body.index("self._claim_pd_execute_state("),
                        body.index("self._pd_green_context_execute_phase = phase"))


class ExecutorCleanupTest(unittest.TestCase):
    def setUp(self):
        self.executor = EXECUTOR()
        self.executor.pd_async_uniproc_enabled = True
        self.out = output()
        self.owner = OWNER.from_output("prefill", self.out)
        self.dep = Future()
        self.executor._pd_async_phase_execute_futures = {"prefill": self.dep}
        self.executor._pd_async_phase_execute_owners = {"prefill": self.owner}

    def test_ack_precedes_dependency_removal_and_zero_alias_is_canonical(self):
        calls = []
        def rpc(method, **kwargs):
            calls.append((method, kwargs))
            self.assertIs(self.executor._pd_async_phase_execute_futures["prefill"], self.dep)
            ack = Future()
            ack.set_result(True)
            return ack
        self.executor.collective_rpc = rpc
        self.assertTrue(self.executor.clear_pd_phase_execute_state("prefill:0", self.owner))
        self.assertEqual(self.executor._pd_async_phase_execute_futures, {})
        self.assertEqual(self.executor._pd_async_phase_execute_owners, {})
        self.assertEqual(calls[0][1]["args"], ("prefill", self.owner))
        self.assertFalse(self.dep.cancelled())

    def test_failed_stale_or_invalid_ack_retains_dependency(self):
        for result in (RuntimeError("cleanup failed"), False, None):
            with self.subTest(result=result):
                def rpc(*args, **kwargs):
                    ack = Future()
                    if isinstance(result, Exception):
                        ack.set_exception(result)
                    else:
                        ack.set_result(result)
                    return ack
                self.executor.collective_rpc = rpc
                if result is False:
                    self.assertFalse(self.executor.clear_pd_phase_execute_state(
                        "prefill", self.owner))
                else:
                    with self.assertRaises(RuntimeError):
                        self.executor.clear_pd_phase_execute_state("prefill", self.owner)
                self.assertIs(self.executor._pd_async_phase_execute_futures["prefill"], self.dep)
                self.assertEqual(self.executor._pd_async_phase_execute_owners["prefill"], self.owner)

    def test_replaced_or_consumed_dependency_during_ack_is_not_removed(self):
        for change in ("future", "owner", "both", "consumed"):
            with self.subTest(change=change):
                self.setUp()
                def rpc(*args, **kwargs):
                    if change in ("future", "both"):
                        self.executor._pd_async_phase_execute_futures["prefill"] = Future()
                    if change in ("owner", "both"):
                        self.executor._pd_async_phase_execute_owners["prefill"] = (
                            OWNER.from_output("prefill", output(sequence=8)))
                    if change == "consumed":
                        self.executor._pd_async_phase_execute_futures.clear()
                        self.executor._pd_async_phase_execute_owners.clear()
                    return True
                self.executor.collective_rpc = rpc
                self.executor.clear_pd_phase_execute_state("prefill", self.owner)
                if change == "consumed":
                    self.assertEqual(self.executor._pd_async_phase_execute_futures, {})
                else:
                    self.assertIn("prefill", self.executor._pd_async_phase_execute_futures)
                    self.assertIn("prefill", self.executor._pd_async_phase_execute_owners)

    def test_newer_owner_present_before_cleanup_is_preserved(self):
        newer = OWNER.from_output("prefill", output(sequence=8))
        self.executor._pd_async_phase_execute_owners["prefill"] = newer
        self.executor.collective_rpc = lambda *a, **kw: True
        self.executor.clear_pd_phase_execute_state("prefill", self.owner)
        self.assertEqual(self.executor._pd_async_phase_execute_owners["prefill"], newer)

    def test_execute_tracks_owner_and_sample_consumes_both_maps(self):
        dep = Future()
        self.executor.collective_rpc = lambda *args, **kwargs: dep
        self.executor.execute_model(self.out, non_block=True, pd_green_context_phase="prefill")
        self.assertIs(self.executor._pd_async_phase_execute_futures["prefill"], dep)
        self.assertEqual(self.executor._pd_async_phase_execute_owners["prefill"], self.owner)
        calls = []
        self.executor.collective_rpc = lambda *a, **kw: calls.append(kw)
        self.executor.sample_tokens(None, non_block=True, pd_green_context_phase="prefill:0")
        self.assertIs(calls[0]["dependency"], dep)
        self.assertEqual(self.executor._pd_async_phase_execute_owners, {})

    def test_pair_tracks_independent_phase_owners(self):
        self.executor.collective_rpc = lambda *a, **kw: Future()
        self.executor._pd_pair_phase_dependency = lambda future, phase: Future()
        decode, prefill = output(), output(1)
        self.executor.execute_model_pd_pair(decode, prefill, non_block=True)
        for key, out in (("decode", decode), ("prefill:1", prefill)):
            self.assertEqual(self.executor._pd_async_phase_execute_owners[key],
                             OWNER.from_output(key, out))

    def test_non_phase_upstream_execution_does_not_add_owner(self):
        self.executor.pd_async_uniproc_enabled = False
        self.executor._pd_async_phase_execute_owners.clear()
        self.executor.collective_rpc = lambda *a, **kw: Future()
        self.executor.execute_model(NS(), non_block=True)
        self.assertEqual(self.executor._pd_async_phase_execute_owners, {})


class EligibilityError(RuntimeError):
    pass


class RoutingTest(unittest.TestCase):
    def test_engine_passes_original_output_identity(self):
        engine = ENGINE()
        calls = []
        engine.model_executor = NS(clear_pd_phase_execute_state=lambda *a: calls.append(a))
        out = output(1)
        engine._cancel_ttft_prefill_worker_slot("prefill:1", out)
        self.assertEqual(calls, [("prefill:1", OWNER.from_output("prefill:1", out))])

    def test_worker_single_and_pair_errors_route_matching_owner_after_sync(self):
        for paired in (False, True):
            with self.subTest(paired=paired):
                events = []
                worker_cls = adapters("vendor_v8/vllm/v1/worker/gpu_worker.py", "Worker",
                    {"execute_model", "execute_model_pd_pair", "clear_pd_phase_execute_state"},
                    extra={"TieredBackendEligibilityError": EligibilityError,
                           "torch": NS(cuda=NS(synchronize=lambda _: events.append("sync"))),
                           "get_pp_group": lambda: NS(is_first_rank=True)})
                worker = worker_cls()
                worker.vllm_config = NS(parallel_config=NS(pipeline_parallel_size=1),
                                        compilation_config=NS())
                worker.use_v2_model_runner = False
                worker.device = 0
                worker.annotate_profile = lambda _: contextlib.nullcontext()
                def fail(*args, **kwargs):
                    raise EligibilityError("test failure")
                worker.model_runner = NS(execute_model=fail,
                    clear_pd_phase_execute_state=lambda *a: events.append(a) or True)
                prefill, decode = output(1), output()
                if paired:
                    result = worker.execute_model_pd_pair(decode, prefill)
                    self.assertEqual(result["pd_paired_decode_tiered_error"], "test failure")
                    self.assertEqual(result["pd_paired_prefill_tiered_error"], "test failure")
                    expected = ["sync", ("decode", OWNER.from_output("decode", decode))]
                else:
                    with self.assertRaises(EligibilityError):
                        worker.execute_model(prefill, pd_green_context_phase="prefill:1")
                    expected = []
                expected += ["sync", ("prefill:1", OWNER.from_output("prefill:1", prefill))]
                self.assertEqual(events, expected)
                self.assertTrue(worker.clear_pd_phase_execute_state("prefill:1", expected[-1][1]))


if __name__ == "__main__":
    unittest.main()
