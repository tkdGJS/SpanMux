"""CPU-only tests for semantic dependency graph tooling."""

from __future__ import annotations

import copy
import unittest

from tools.dependency_graph.graph_lib import (
    DEFAULT_GRAPH_PATH,
    REPO_ROOT,
    build_impact_slice,
    load_graph,
    render_graph_markdown,
    validate_graph,
)


class DependencyGraphToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = load_graph(DEFAULT_GRAPH_PATH)

    def test_repository_graph_is_valid(self) -> None:
        errors, warnings = validate_graph(self.graph, repo_root=REPO_ROOT)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_two_hop_slice_includes_allocator_and_retry_dependencies(self) -> None:
        impact = build_impact_slice(
            self.graph,
            node_seeds=["scheduled_prefill_fit"],
            depth=2,
        )
        self.assertEqual(impact.seeds, ("scheduled_prefill_fit",))
        self.assertEqual(impact.distances["scheduled_prefill_fit"], 0)
        self.assertIn("kv_allocator_ownership", impact.distances)
        self.assertIn("tiered_provider_snapshot", impact.distances)
        self.assertIn("capacity_retry_gate", impact.distances)
        self.assertIn("engine_progress_event", impact.distances)

    def test_file_seed_maps_to_semantic_nodes(self) -> None:
        impact = build_impact_slice(
            self.graph,
            file_seeds=["vendor_v8/vllm/v1/output_placeholder.py"],
            depth=0,
        )
        self.assertEqual(impact.seeds, ("output_placeholder_ownership",))
        self.assertEqual(impact.unresolved_files, ())

    def test_unmapped_file_fails_closed_in_result(self) -> None:
        impact = build_impact_slice(
            self.graph,
            file_seeds=["vendor_v8/vllm/unknown_runtime_path.py"],
        )
        self.assertEqual(impact.seeds, ())
        self.assertEqual(
            impact.unresolved_files,
            ("vendor_v8/vllm/unknown_runtime_path.py",),
        )

    def test_missing_symbol_is_detected(self) -> None:
        broken = copy.deepcopy(self.graph)
        broken["nodes"][0]["symbols"][0] = (
            "tools/dependency_graph/validate_graph.py::missing_symbol"
        )
        errors, _warnings = validate_graph(broken, repo_root=REPO_ROOT)
        self.assertTrue(
            any("Python symbol does not exist" in error for error in errors),
            errors,
        )

    def test_render_is_deterministic(self) -> None:
        first = render_graph_markdown(self.graph)
        second = render_graph_markdown(self.graph)
        self.assertEqual(first, second)
        self.assertIn("```mermaid", first)
        self.assertIn("scheduled_prefill_fit", first)


if __name__ == "__main__":
    unittest.main()
