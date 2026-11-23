#!/usr/bin/env python3
"""Execute real provider validation and authority expressions without CUDA."""
from __future__ import annotations

import ast
import unittest
from types import SimpleNamespace as NS

import test_exact_phase_cleanup as C


def tree(path):
    return ast.parse((C.ROOT / path).read_text())


PROVIDER = tree("vendor_v8/vllm/v1/attention/ops/tiered_residency_provider.py")
BACKEND = tree("vendor_v8/vllm/v1/attention/backends/triton_attn.py")
PREFETCH = tree("vendor_v8/vllm/v1/attention/ops/tiered_layer_reload_prefetch.py")
AUTH = tree("vendor_v8/vllm/v1/core/tiered_execution.py")
AUTH.body = [n for n in AUTH.body if not (isinstance(n, ast.ImportFrom)
             and (n.module or "").startswith("vllm"))]
SCOPE = {"TieredBackendEligibilityError": RuntimeError}
exec(compile(AUTH, "authority", "exec"), SCOPE)
RESOLVE = SCOPE["resolve_tiered_reload_authority"]


def validate(ids, pages, concrete, valid=2, capacity=16):
    fn = next(n for n in PROVIDER.body if isinstance(n, ast.FunctionDef)
              and n.name == "reload_dram_kv_pages_to_hbm")
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == "null_block_id")
    end = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.If)
               and ast.unparse(n.test) == "duplicate_or_invalid")
    scope = dict(row_block_ids=ids, num_valid_pages=valid,
                 block_table_is_fully_concrete=concrete,
                 requested_pages=pages, request_id="A", key_cache=NS(shape=(capacity,)),
                 first_invalid_block_table_detail="")
    exec(compile(ast.Module(body=loop.body[start:end], type_ignores=[]),
                 "real_provider_validation", "exec"), scope)
    return scope["duplicate_or_invalid"], scope["page_to_block"], scope["first_invalid_block_table_detail"]


class DestinationAuthorityTests(unittest.TestCase):
    def test_padding_collision_ignored_only_with_concrete_proof(self):
        self.assertTrue(validate([7, 8, 7], {0}, False)[0])
        self.assertEqual(validate([7, 8, 7], {0}, True)[:2], (False, {0: 7}))

    def test_bad_native_tables_still_fail_closed(self):
        for ids, pages in [([7, 7, 0], {0}), ([-1, 8], {0}), ([7, 16], {0}),
                           ([7, 8], {-1}), ([7, 8], {2})]:
            with self.subTest(ids=ids, pages=pages):
                self.assertTrue(validate(ids, pages, True)[0])

    def test_native_proof_before_partial_tail_and_both_copy_calls(self):
        assignment = next(n for n in ast.walk(BACKEND) if isinstance(n, ast.Assign)
            and ast.unparse(n.targets[0]) == "native_reload_destinations_concrete")
        for phase, mode, expected in [("prefill", "native_reload", True),
                ("decode", "native_reload", True), ("prefill", "context_ring", False),
                ("prefill", "auto", False)]:
            scope = dict(resolve_tiered_reload_authority=RESOLVE,
                tiered_control_required=True, layer_reload_required_pages={"A": {0}},
                attn_metadata=NS(attention_phase=phase,
                    tiered_execution_plans={"A": (1, mode, 2)}))
            value = eval(compile(ast.Expression(assignment.value), "backend", "eval"), scope)
            self.assertEqual(value, expected)
        copies = [n for n in ast.walk(BACKEND) if isinstance(n, ast.Call)
                  and ast.unparse(n.func) == "reload_dram_kv_pages_to_hbm"]
        self.assertEqual(len(copies), 2)
        for call in copies:
            self.assertLess(assignment.lineno, call.lineno)
            kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
            self.assertEqual(kw["block_table_is_fully_concrete"], "native_reload_destinations_concrete")
            self.assertIn("diagnostics", kw)

    def test_next_layer_resolves_its_own_rows_before_async_dispatch(self):
        assignment = next(n for n in ast.walk(PREFETCH) if isinstance(n, ast.Assign)
            and ast.unparse(n.targets[0]) == "block_table_is_fully_concrete")
        scope = dict(resolve_tiered_reload_authority=RESOLVE, attention_phase="prefill",
                     execution_plans={"A": (1, "native_reload", 2),
                                      "B": (1, "context_ring", 2)})
        for rows, expected in [({"A": {0}}, True), ({"B": {1}}, False), ({}, False)]:
            scope["reload_pages_by_request"] = rows
            self.assertEqual(eval(compile(ast.Expression(assignment.value), "prefetch", "eval"), scope), expected)
        scope["reload_pages_by_request"] = {"A": {0}, "B": {1}}
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            eval(compile(ast.Expression(assignment.value), "prefetch", "eval"), scope)

    def test_real_partial_prefix_and_all_layer_ack_contracts_unchanged(self):
        calls = [n for n in ast.walk(BACKEND) if isinstance(n, ast.Call)
                 and ast.unparse(n.func) == "reload_dram_kv_pages_to_hbm"]
        partial = min(calls, key=lambda n: n.lineno)
        kw = {k.arg: ast.unparse(k.value) for k in partial.keywords}
        self.assertEqual(kw["expected_token_counts_by_request"], "partial_tail_provider_token_counts")
        self.assertEqual(kw["hbm_token_counts_by_request"], "partial_tail_hbm_token_counts")
        text = ast.unparse(BACKEND)
        self.assertIn("reload_diagnostics=", text)
        self.assertIn("phase=", text)


if __name__ == "__main__":
    unittest.main()
