#!/usr/bin/env python3
"""CPU contracts for mutable vendor-v8 direct-D2H destinations."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # Source-only checkouts need not install PyTorch.
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
PROVIDER_PATH = (
    ROOT
    / "vendor_v8/vllm/v1/attention/ops/tiered_residency_provider.py"
)
PROVIDER_SOURCE = PROVIDER_PATH.read_text(encoding="utf-8")
PROVIDER_TREE = ast.parse(PROVIDER_SOURCE)


def _top_level_function(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in PROVIDER_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _class_method(class_name: str, method_name: str) -> ast.FunctionDef:
    class_node = next(
        node
        for node in PROVIDER_TREE.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _source(node: ast.AST) -> str:
    source = ast.get_source_segment(PROVIDER_SOURCE, node)
    assert source is not None
    return source


def _direct_call_count(node: ast.AST, function_name: str) -> int:
    return sum(
        1
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == function_name
    )


def _provider_function(function_name: str):
    function = _top_level_function(function_name)
    module = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    )
    namespace: dict[str, Any] = {"Any": Any, "torch": torch}
    exec(compile(module, str(PROVIDER_PATH), "exec"), namespace)
    return namespace[function_name]


class TieredMutableDRAMTest(unittest.TestCase):
    def test_helpers_disable_ambient_inference_mode(self) -> None:
        allocate_source = _source(
            _top_level_function("_allocate_mutable_cpu_tensor")
        )
        view_source = _source(
            _top_level_function("_mutable_cpu_tensor_view")
        )
        for helper_source in (allocate_source, view_source):
            self.assertIn("with torch.inference_mode(False):", helper_source)
            self.assertIn("torch.is_inference", helper_source)

    def test_every_contiguous_slab_destination_uses_mutable_helpers(
        self,
    ) -> None:
        ensure = _class_method(
            "PinnedDRAMKVPool", "_ensure_contiguous_slab_locked"
        )
        store = _class_method(
            "PinnedDRAMKVPool", "_store_in_contiguous_slab_locked"
        )
        reserve = _class_method("PinnedDRAMKVPool", "reserve_from_hbm")

        self.assertEqual(
            _direct_call_count(ensure, "_allocate_mutable_cpu_tensor"), 2
        )
        self.assertEqual(
            _direct_call_count(store, "_mutable_cpu_tensor_view"), 2
        )
        self.assertEqual(
            _direct_call_count(reserve, "_mutable_cpu_tensor_view"), 2
        )

    def test_inference_tensor_fallback_has_a_dedicated_counter(self) -> None:
        init_source = _source(_class_method("DRAMKVPageRegistry", "__init__"))
        enqueue_source = _source(
            _class_method("DRAMKVPageRegistry", "enqueue_from_hbm_direct")
        )
        clear_source = _source(_class_method("DRAMKVPageRegistry", "clear"))
        stats_source = _source(_class_method("DRAMKVPageRegistry", "stats"))
        field_name = (
            "_register_from_hbm_direct_inference_tensor_error_count"
        )

        self.assertIn(f"self.{field_name} = 0", init_source)
        self.assertIn('if "inference tensor" in str(exc).lower():', enqueue_source)
        self.assertIn(f"self.{field_name} += 1", enqueue_source)
        self.assertIn(f"self.{field_name} = 0", clear_source)
        self.assertIn(
            "dram_provider_register_from_hbm_direct_"
            "inference_tensor_error_count",
            stats_source,
        )

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_slab_and_view_are_mutable_after_caller_inference_mode(
        self,
    ) -> None:
        assert torch is not None
        allocate = _provider_function("_allocate_mutable_cpu_tensor")
        mutable_view = _provider_function("_mutable_cpu_tensor_view")

        with torch.inference_mode():
            tensor = allocate(
                (2, 4),
                dtype=torch.float32,
                pin_memory=False,
            )
            view = mutable_view(tensor, (0, slice(None)))

        self.assertFalse(torch.is_inference(tensor))
        self.assertFalse(torch.is_inference(view))
        view.copy_(torch.ones_like(view))
        tensor[1].copy_(torch.ones_like(tensor[1]))
        self.assertEqual(float(tensor.sum()), 8.0)


if __name__ == "__main__":
    unittest.main()
