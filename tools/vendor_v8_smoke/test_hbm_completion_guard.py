#!/usr/bin/env python3
"""CPU capacity/progress checks; not a GPU benchmark or numerical test."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest

ROOT = Path(__file__).resolve().parents[2]
REL = "vllm/v1/core/sched/isolation_hbm_guard.py"
spec = importlib.util.spec_from_file_location("guard", ROOT / "vendor_v8" / REL)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def request(name, prompt=40, output=20):
    return NS(request_id=name, num_prompt_tokens=prompt, max_tokens=output,
              has_encoder_inputs=False, sampling_params=object())


class GuardTests(unittest.TestCase):
    def setup_guard(self):
        cache_spec = type("FullAttentionSpec", (), {"block_size": 10})()
        self.tables, self.messages = {}, []
        self.free = 10
        self.scheduler = NS(kv_cache_config=NS(kv_cache_groups=[NS(kv_cache_spec=cache_spec)]),
            parallel_config=NS(), vllm_config=NS(speculative_config=None),
            connector=None, is_encoder_decoder=False, max_model_len=1000, requests={})
        self.scheduler.kv_cache_manager = NS(
            coordinator=NS(single_type_managers=[NS(req_to_blocks=self.tables)]),
            block_pool=NS(num_gpu_blocks=11, get_num_free_blocks=lambda: self.free))
        self.log = lambda *a: self.messages.append(a)
        return mod.IsolationHBMCompletionGuard(self.scheduler, self.log)

    def test_two_partial_prompts_cannot_deadlock_each_other(self):
        guard = self.setup_guard()
        a, b = request("A"), request("B")
        self.assertTrue(guard.can_admit(a))
        self.scheduler.requests["A"] = a
        # A moves Prefill->Decode without changing its full growth budget.
        for blocks in (1, 2, 4, 5, 6):
            self.tables["A"] = [object()] * blocks
            self.free = 10 - blocks
            self.assertFalse(guard.can_admit(b))
            self.assertLessEqual(6 - blocks, self.free)
        self.assertEqual(guard.blocked_transitions, 1)  # not per token
        self.tables.clear()
        self.scheduler.requests.clear()  # EOS/cancel normal free
        self.free = 10
        self.assertTrue(guard.can_admit(b))

    def test_multiple_requests_admitted_when_complete_budget_fits(self):
        guard = self.setup_guard()
        a, b = request("A", 20, 10), request("B", 30, 10)
        self.scheduler.requests["A"] = a
        self.tables["A"] = [1, 2]
        self.free = 8
        self.assertTrue(guard.can_admit(b))
        self.assertEqual(guard.last_required_blocks, 5)

    def test_waiting_future_owner_counted_and_resume_not_double_counted(self):
        guard = self.setup_guard()
        a, b = request("waiting-A"), request("B")
        self.scheduler.requests[a.request_id] = a
        self.tables[a.request_id] = [1, 2, 3, 4]
        self.free = 6
        self.assertFalse(guard.can_admit(b))
        self.assertTrue(guard.can_admit(a))
        self.assertEqual(guard.last_required_blocks, 2)

    def test_exact_fit_and_one_block_growth_failure(self):
        guard = self.setup_guard()
        self.assertTrue(guard.can_admit(request("exact", 80, 20)))
        with self.assertRaisesRegex(ValueError, "empty pool"):
            guard.can_admit(request("impossible", 81, 20))

    def test_cached_prefix_not_acquired_is_not_assumed_free(self):
        guard = self.setup_guard()
        self.free = 4  # other outstanding owner/temporary block lease
        self.assertFalse(guard.can_admit(request("new")))

    def test_default_off_and_bad_flag(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(mod.make_isolation_hbm_completion_guard(None, None))
        with patch.dict(os.environ, {"VLLM_ISOLATION_HBM_COMPLETION_GUARD": "typo"}):
            with self.assertRaises(ValueError):
                mod.make_isolation_hbm_completion_guard(None, None)

    def test_unsupported_layout_and_offload_fail_closed(self):
        self.setup_guard()
        for name in ("virtual_context_enabled", "is_encoder_decoder"):
            setattr(self.scheduler, name, True)
            with self.assertRaises(ValueError):
                mod.IsolationHBMCompletionGuard(self.scheduler, self.log)
            setattr(self.scheduler, name, False)
        self.scheduler.kv_cache_config.kv_cache_groups *= 2
        with self.assertRaises(ValueError):
            mod.IsolationHBMCompletionGuard(self.scheduler, self.log)

    def test_source_vendor_identical_and_guard_before_waiting_allocation(self):
        self.assertEqual((ROOT / "vendor_v8" / REL).read_bytes(),
                         (ROOT / "src/vllm-0.14-src" / REL).read_bytes())
        for root in ("vendor_v8", "src/vllm-0.14-src"):
            source = (ROOT / root / "vllm/v1/core/sched/scheduler.py").read_text()
            tree = ast.parse(source)
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
            fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "schedule")
            guard = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                         and "self.isolation_hbm_guard.can_admit" in ast.unparse(n.test))
            loops = [n for n in ast.walk(fn) if isinstance(n, ast.While)
                     and "self.waiting" in ast.unparse(n.test)]
            self.assertTrue(any(guard in list(ast.walk(loop)) for loop in loops))
            self.assertTrue(any(isinstance(n, ast.Break) for n in guard.body))
            self.assertGreater(source.index("new_blocks = self.kv_cache_manager.allocate_slots(",
                                            source.index("self.isolation_hbm_guard.can_admit")),
                               source.index("self.isolation_hbm_guard.can_admit"))
        self.assertIn('pd_native_active_decode_prefill_capacity_reason = "hbm_completion_guard"', source if root == "vendor_v8" else (ROOT / "vendor_v8/vllm/v1/core/sched/scheduler.py").read_text())


if __name__ == "__main__":
    unittest.main()
