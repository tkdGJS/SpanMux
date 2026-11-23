#!/usr/bin/env python3
"""CPU-only startup ownership, ordering, deployment and rollback contracts."""

import ast
import contextlib
import json
import os
import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor_v8/vllm"
NATIVE = ROOT / "src/vllm-0.14-src/vllm"
MODULE = runpy.run_path(str(VENDOR / "isolation_startup_memory.py"))
Audit = MODULE["StartupMemoryAudit"]


class FakeCuda:
    def __init__(self):
        self.events = []
        self.reserved = 800
        self.capturing = False
        self.fail_sync = False
        self.fail_empty = False

    @contextlib.contextmanager
    def device(self, device):
        self.events.append(("enter_device", device))
        try:
            yield
        finally:
            self.events.append(("exit_device", device))

    def memory_stats(self, device):
        self.events.append(("stats", device))
        return {"allocated_bytes.all.current": 500,
                "active_bytes.all.current": 550,
                "reserved_bytes.all.current": self.reserved,
                "inactive_split_bytes.all.current": 50,
                "allocated_bytes.all.peak": 600,
                "reserved_bytes.all.peak": 800}

    def is_current_stream_capturing(self):
        self.events.append(("capturing",))
        return self.capturing

    def synchronize(self, device):
        self.events.append(("sync", device))
        if self.fail_sync:
            raise RuntimeError("completion failed")

    def empty_cache(self):
        self.events.append(("empty",))
        if self.fail_empty:
            raise RuntimeError("allocator failed")
        self.reserved = 650  # Not all inactive or reserved space is reclaimable.


class StartupMemoryTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.cuda = FakeCuda()
        self.logs = []
        self.collect = patch("gc.collect", side_effect=lambda: self.cuda.events.append(("gc",)))
        self.collect.start()
        self.addCleanup(self.collect.stop)

    def make(self):
        return Audit(cuda=self.cuda, device="cuda:1",
                     log=lambda fmt, row: self.logs.append(json.loads(row)))

    def finish(self, owner):
        owner.after_kernel_warmup()
        owner.after_capture()
        owner.finish()

    def test_both_disabled_no_cuda_no_logging_no_bundle_inheritance(self):
        os.environ["VLLM_ISOLATION_MEMORY_OPTIMIZATIONS"] = "1"
        os.environ["VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS"] = "1"
        owner = self.make()
        owner.before_capture()
        self.finish(owner)
        self.assertEqual(self.cuda.events, [])
        self.assertEqual(self.logs, [])

    def test_audit_only_no_barrier_or_collection(self):
        os.environ["VLLM_ISOLATION_STARTUP_MEMORY_AUDIT"] = "1"
        owner = self.make()
        owner.before_capture()
        self.finish(owner)
        self.assertEqual([x[0] for x in self.cuda.events], ["stats"] * 5)
        self.assertEqual([x["stage"] for x in self.logs], [
            "before_compile_warmup", "after_compile_warmup",
            "after_kernel_warmup", "after_capture", "after_sampler_warmup"])

    def test_trim_once_on_selected_device_and_keep_live_owners(self):
        os.environ["VLLM_ISOLATION_STARTUP_CACHE_TRIM"] = "1"
        owner = self.make()
        owner.before_capture()
        self.finish(owner)
        events = [x[0] for x in self.cuda.events]
        self.assertLess(events.index("gc"), events.index("sync"))
        self.assertLess(events.index("sync"), events.index("empty"))
        self.assertLess(events.index("empty"), events.index("exit_device"))
        self.assertEqual(events.count("empty"), 1)
        self.assertIn(("sync", "cuda:1"), self.cuda.events)
        after = next(x for x in self.logs if x["stage"] == "after_cache_trim")
        self.assertEqual(after["reserved_before_trim_bytes"], 800)
        self.assertEqual(after["reserved_bytes"], 650)
        self.assertEqual(after["allocated_bytes"], 500)
        self.assertEqual(after["active_bytes"], 550)
        self.assertEqual(after["peak_reserved_bytes"], 800)
        self.assertIn("trim_elapsed_ms", after)

    def test_late_duplicate_or_failed_trim_never_repeats(self):
        os.environ["VLLM_ISOLATION_STARTUP_CACHE_TRIM"] = "1"
        for complete in (False, True):
            with self.subTest(complete=complete):
                owner = self.make()
                owner.before_capture()
                if complete:
                    self.finish(owner)
                events = list(self.cuda.events)
                with self.assertRaisesRegex(RuntimeError, "one-shot"):
                    owner.before_capture()
                self.assertEqual(events, self.cuda.events)

    def test_capture_rejected_before_collect_sync_free(self):
        os.environ["VLLM_ISOLATION_STARTUP_CACHE_TRIM"] = "1"
        self.cuda.capturing = True
        owner = self.make()
        with self.assertRaisesRegex(RuntimeError, "during capture"):
            owner.before_capture()
        self.assertFalse({"empty", "sync", "gc"} & {x[0] for x in self.cuda.events})
        with self.assertRaisesRegex(RuntimeError, "out of order"):
            owner.after_kernel_warmup()

    def test_completion_error_never_frees_and_propagates(self):
        os.environ["VLLM_ISOLATION_STARTUP_CACHE_TRIM"] = "1"
        self.cuda.fail_sync = True
        owner = self.make()
        with self.assertRaisesRegex(RuntimeError, "completion failed"):
            owner.before_capture()
        self.assertNotIn(("empty",), self.cuda.events)
        with self.assertRaises(RuntimeError):
            owner.after_capture()

    def test_allocator_error_cannot_continue_or_publish_completion(self):
        os.environ["VLLM_ISOLATION_STARTUP_CACHE_TRIM"] = "1"
        self.cuda.fail_empty = True
        owner = self.make()
        with self.assertRaisesRegex(RuntimeError, "allocator failed"):
            owner.before_capture()
        self.assertNotIn("after_cache_trim", [r["stage"] for r in self.logs])
        with self.assertRaises(RuntimeError):
            owner.finish()

    def test_invalid_flags_rejected_without_cuda(self):
        for flag in ("STARTUP_CACHE_TRIM", "STARTUP_MEMORY_AUDIT"):
            with patch.dict(os.environ, {"VLLM_ISOLATION_" + flag: "yes"}):
                with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                    self.make()
        self.assertEqual(self.cuda.events, [])

    def test_native_vendor_mirror_and_worker_order(self):
        self.assertEqual((VENDOR / "isolation_startup_memory.py").read_bytes(),
                         (NATIVE / "isolation_startup_memory.py").read_bytes())
        for root in (NATIVE, VENDOR):
            with self.subTest(root=root):
                result = MODULE["validate_startup_memory_deployment"](root)
                self.assertEqual(result["contract_version"], 1)
                self.assertEqual(len(result["helper_sha256"]), 64)
                source = (root / "v1/worker/gpu_worker.py").read_text()
                tree = ast.parse(source)
                worker = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                              and n.name == "Worker")
                for method in worker.body:
                    if not isinstance(method, ast.FunctionDef):
                        continue
                    names = [n.id for n in ast.walk(method) if isinstance(n, ast.Name)]
                    if "startup_memory" in names:
                        self.assertEqual(method.name, "compile_or_warm_up_model")

    def test_stale_worker_or_bad_order_rejected(self):
        original = (VENDOR / "v1/worker/gpu_worker.py").read_text()
        for modified in (
            original.replace("startup_memory.before_capture()", "pass"),
            original.replace("startup_memory.before_capture()", "startup_memory.after_capture()"),
            original.replace("self._isolation_startup_memory_seen = True", "pass"),
        ):
            with patch.object(Path, "read_text", return_value=modified):
                with self.assertRaisesRegex(RuntimeError, "not wired|out of order|lifetime guard"):
                    MODULE["validate_startup_memory_deployment"](VENDOR)

    def test_worker_reentry_and_failed_first_warmup_never_trim_again(self):
        os.environ["VLLM_ISOLATION_STARTUP_CACHE_TRIM"] = "1"
        helper = ModuleType("vllm.isolation_startup_memory")
        helper.StartupMemoryAudit = Audit
        for root in (VENDOR, NATIVE):
            for fail_first in (False, True):
                with self.subTest(root=root, fail_first=fail_first):
                    self.cuda = FakeCuda()
                    tree = ast.parse((root / "v1/worker/gpu_worker.py").read_text())
                    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                               and n.name == "Worker")
                    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                                  and n.name == "compile_or_warm_up_model")
                    ns = {
                        "torch": SimpleNamespace(cuda=self.cuda),
                        "logger": SimpleNamespace(info=lambda *a: None),
                        "CompilationMode": SimpleNamespace(VLLM_COMPILE="compile"),
                        "kernel_warmup": lambda _: None,
                        "get_pp_group": lambda: SimpleNamespace(is_last_rank=False),
                        "set_random_seed": lambda _: None,
                    }
                    exec(compile(ast.Module(body=[method], type_ignores=[]),
                                 "<worker-warmup>", "exec"), ns)
                    worker = SimpleNamespace(
                        device="cuda:1",
                        vllm_config=SimpleNamespace(compilation_config=SimpleNamespace(mode="none")),
                        model_config=SimpleNamespace(enforce_eager=True, seed=0),
                        cache_config=SimpleNamespace(kv_cache_memory_bytes=2**31),
                        model_runner=SimpleNamespace(lora_config=None,
                            maybe_remove_all_loras=lambda _: None),
                    )
                    with patch.dict(sys.modules, {
                        "vllm": ModuleType("vllm"),
                        "vllm.isolation_startup_memory": helper,
                    }):
                        self.cuda.fail_sync = fail_first
                        if fail_first:
                            with self.assertRaisesRegex(RuntimeError, "completion failed"):
                                ns["compile_or_warm_up_model"](worker)
                        else:
                            ns["compile_or_warm_up_model"](worker)
                        self.cuda.fail_sync = False
                        ns["compile_or_warm_up_model"](worker)
                    events = [x[0] for x in self.cuda.events]
                    self.assertEqual(events.count("sync"), 1)
                    self.assertEqual(events.count("empty"), 0 if fail_first else 1)


if __name__ == "__main__":
    unittest.main()
