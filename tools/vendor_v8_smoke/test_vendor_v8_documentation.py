"""CPU-only documentation drift checks; no vLLM/torch import or GPU proof.

Run: python -B -m unittest -v tools.vendor_v8_smoke.test_vendor_v8_documentation
When a source hash changes, review its contract before refreshing the manifest.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import runpy
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/vendor_v8_documentation_source_snapshot.json"


def qualified_symbols(tree: ast.AST, prefix: str = "") -> set[str]:
    """Find real definitions without executing runtime imports."""
    found: set[str] = set()
    for node in getattr(tree, "body", ()):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f"{prefix}.{node.name}" if prefix else node.name
            found.add(name)
            if isinstance(node, ast.ClassDef):
                found.update(qualified_symbols(node, name))
    return found


class VendorV8DocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.guide_path = ROOT / cls.manifest["current_guide"]
        cls.guide = cls.guide_path.read_text(encoding="utf-8")
        cls.anchors = set(re.findall(r'<a id="([^"]+)"></a>', cls.guide))
        cls.source_texts = {
            path: (ROOT / path).read_text(encoding="utf-8")
            for path in cls.manifest["reviewed_source_files"]
        }

    def test_reviewed_source_fingerprints(self) -> None:
        self.assertEqual(self.manifest["schema_version"], 1)
        self.assertIn("not all repository", self.manifest["source_scope"])
        for path, expected in self.manifest["reviewed_source_files"].items():
            with self.subTest(path=path):
                actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                self.assertEqual(actual, expected, "Review docs before updating hash")

    def test_named_source_symbols_exist(self) -> None:
        for path, names in self.manifest["symbols"].items():
            definitions = qualified_symbols(ast.parse(self.source_texts[path]))
            for name in names:
                with self.subTest(path=path, symbol=name):
                    self.assertIn(name, definitions)

    def test_current_inventory_links_and_anchors(self) -> None:
        links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", self.guide)
        self.assertGreater(len(links), 50)
        for link in links:
            parts = urlsplit(link)
            if parts.scheme:
                continue
            target = (self.guide_path.parent / unquote(parts.path)).resolve()
            if not parts.path:
                target = self.guide_path
            with self.subTest(link=link):
                self.assertTrue(target.is_file(), f"Missing local target: {target}")
                if target == self.guide_path and parts.fragment:
                    self.assertIn(unquote(parts.fragment), self.anchors)

    def test_feature_documents_mark_historical_scope(self) -> None:
        for entry in self.manifest["historical_documents"]:
            path = ROOT / entry["path"]
            header = "\n".join(path.read_text(encoding="utf-8").splitlines()[:18])
            with self.subTest(path=entry["path"]):
                self.assertIn("2026-09-18", header)
                self.assertIn("역사 스냅샷", header)
                self.assertIn(
                    f"60_VENDOR_V8_CURRENT_FEATURE_GUIDE.md#{entry['section']}",
                    header,
                )
                self.assertIn(entry["section"], self.anchors)

    def test_current_guides_remove_same_request_chain_exception(self) -> None:
        for name in ("57_VENDOR_V8_0813.md", "58_VENDOR_V8_ALGORITHM_AND_PIPELINE_GUIDE.md"):
            text = (ROOT / "docs" / name).read_text(encoding="utf-8")
            with self.subTest(path=name):
                self.assertIn(self.manifest["reviewed_at"], text[:500])
                self.assertNotIn("if chainable: return false", text)
                self.assertNotIn("Prefill depth 2는 vendor_v9", text)
                if name.startswith("57"):
                    self.assertIn("request-disjoint", text)
                else:
                    self.assertIn("서로 다른 요청", text)
                self.assertIn("VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION", text)
        source = self.source_texts["vendor_v8/vllm/v1/core/sched/scheduler.py"]
        tree = ast.parse(source)
        method = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "blocks_next_schedule"
        )
        body = ast.get_source_segment(source, method)
        self.assertIn("pending_prefill_future_owner", body)
        self.assertIn("prefill_suspended", body)

    def test_layerwise_offload_scope_is_current_and_explicit(self) -> None:
        self.assertIn(self.manifest["reviewed_at"], self.guide[:500])
        self.assertIn("offload", self.anchors)
        for name in (
            "57_VENDOR_V8_0813.md",
            "58_VENDOR_V8_ALGORITHM_AND_PIPELINE_GUIDE.md",
            "60_VENDOR_V8_CURRENT_FEATURE_GUIDE.md",
        ):
            text = (ROOT / "docs" / name).read_text(encoding="utf-8")
            with self.subTest(path=name):
                self.assertIn("private Ring", text)
                self.assertIn("KVRetentionRingContinuation", text)
                self.assertIn("원격 GPU", text)
                self.assertNotIn("Ring layerwise 선점과 Future-owned KV victim은 미완성", text)
                self.assertNotIn("Ring은 이 Stage-2 continuation 대상이 아니다", text)
        for flag in (
            "VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN",
            "VLLM_KV_RETENTION_LAYERWISE_OFFLOAD",
            "VLLM_KV_RETENTION_LAYERWISE_HOST_BYTES",
            "VLLM_KV_RETENTION_RUN_CUDA_TESTS",
        ):
            self.assertIn(flag, self.guide)
        transfer = self.source_texts[
            "vendor_v8/vllm/model_executor/layers/kv_retention_layerwise.py"
        ]
        for state in ("copy_ready", "released", "published", "consumed", "retained"):
            self.assertIn(state, transfer)
            self.assertIn(state, self.guide)
        self.assertIn("수정 후 GPU 재검증 결과는 아직 확인되지 않았다", self.guide)

    def test_comparison_variants_distinguish_hbm_only_srpt(self) -> None:
        runner = self.source_texts["run_vendor_v8_dynamic_hbm_vs_full.sh"]
        for variant in (
            "native_40sm_chunked",
            "baseline_plus",
            "dynamic_hbm_serialized_fullsm",
            "dynamic_hbm_srpt",
            "full_vendor_v8",
        ):
            with self.subTest(variant=variant):
                self.assertIn(variant, runner)
                self.assertIn(f"`{variant}`", self.guide)
        self.assertIn("P/D + SRPT 비교군에 제안 기법의 KV offload/Ring이 섞이지 않는다", self.guide)
        self.assertIn("Stage-2의 P2/D1/G3과 기존 P1/D1/G2", self.guide)

    def test_native_chunk_and_workspace_contract_names(self) -> None:
        source = "\n".join(self.source_texts.values())
        self.assertIn("--ttft-prefill-native-chunk-max-tokens", self.guide)
        for flag in re.findall(r"\b(?:VLLM|FULL_VENDOR_V8|LOONGSERVE)_[A-Z0-9_]+\b", self.guide):
            with self.subTest(flag=flag):
                self.assertIn(flag, source)
        for name in ("57_VENDOR_V8_0813.md", "58_VENDOR_V8_ALGORITHM_AND_PIPELINE_GUIDE.md"):
            text = (ROOT / "docs" / name).read_text(encoding="utf-8")
            self.assertIn("VLLM_PD_PREFILL_WORKSPACE_SLOTS", text)
            self.assertNotIn("VLLM_PD_NATIVE_PREFILL_MAX_TOKENS", text)
            self.assertNotIn("VLLM_PD_PREFILL_WORKSPACE_PREFILL_SLOTS", text)

    def test_removed_controls_are_not_advertised_as_defaults(self) -> None:
        config = (ROOT / "docs/13_CONFIG_FLAGS.md").read_text(encoding="utf-8")
        metrics = (ROOT / "docs/10_METRICS_LOGGING.md").read_text(encoding="utf-8")
        self.assertIn("Removed Host-lease controls — historical only", config)
        self.assertNotIn("| `VLLM_PD_PREFILL_MAX_HOST_LEASE_MS`", config)
        self.assertNotIn("| `pd_prefill_host_lease_cap_count`", metrics)
        self.assertNotIn("| `phase_local_output_token_repair_invalid`", metrics)
        source = "\n".join(self.source_texts.values())
        self.assertNotIn("VLLM_PD_PREFILL_MAX_HOST_LEASE_MS", source)

    def test_rendered_guides_match_markdown(self) -> None:
        if importlib.util.find_spec("markdown_it") is None:
            self.skipTest("HTML renderer requires markdown-it-py; render check unavailable")
        renderer = runpy.run_path(str(ROOT / "tools/render_vendor_v8_study_guide.py"))["render"]
        for entry in self.manifest["rendered_guides"]:
            with self.subTest(path=entry["output"]):
                expected = renderer(ROOT / entry["source"], snapshot=entry["snapshot"])
                actual = (ROOT / entry["output"]).read_text(encoding="utf-8")
                self.assertEqual(actual, expected, "Regenerate HTML from its Markdown")


if __name__ == "__main__":
    unittest.main()
