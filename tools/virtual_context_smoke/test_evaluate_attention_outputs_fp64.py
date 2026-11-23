"""CPU contracts for the captured-output/own-input FP64 report."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "numpy", "matplotlib"))
if AVAILABLE:
    import numpy as np
    import torch
    import evaluate_attention_outputs_fp64 as audit


@unittest.skipUnless(AVAILABLE, "PyTorch, NumPy and matplotlib are required")
class AttentionOutputAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)

    def fixture(self, base: Path, layers: int = 2) -> tuple[Path, Path]:
        torch.manual_seed(71)
        for label in ("native", "ring"):
            root = base / label
            root.mkdir()
            manifest = []
            for layer in range(layers):
                torch.manual_seed(71 + layer)
                query = torch.randn(8, 4, 4).half()
                key = torch.randn(8, 2, 4).half()
                value = torch.randn(8, 2, 4).half()
                if label == "ring" and layer == 1:
                    query = (query.float() + .01).half()
                output = torch.stack([audit.dense_causal_attention_fp64(query[pos], key, value, query_position=pos, softmax_scale=.5).half() for pos in range(8)])
                chunks = [(0, 4), (4, 8)] if label == "native" else [(0, 2), (2, 5), (5, 8)]
                for start, end in chunks:
                    filename = f"l{layer}_{start}.pt"
                    torch.save({"query": query[start:end], "key": key[start:end], "value": value[start:end], "output": output[start:end], "softmax_scale": .5}, root / filename)
                    manifest.append({
                        "row": {"request_id": f"cmpl-case_{label}-foreground-loogle-1-1-0-abcdef", "request_ordinal": 1, "phase": "prefill"},
                        "layer_name": f"model.layers.{layer}.self_attn.attn", "execution_path": "context_ring" if label == "ring" and start >= 2 else "native_hbm",
                        "logical_query_start": start, "query_length": end - start,
                        "occurrence": 0, "file": filename, "dtype": "torch.float16", "shape": list(output[start:end].shape),
                    })
            (root / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in manifest))
        return base / "native", base / "ring"

    def test_metric_normalization_and_zero_reference(self) -> None:
        value = audit.error_metrics(torch.tensor([[2., 2.]]), torch.tensor([[3., 3.]]))
        self.assertEqual(value["rmse"], 1)
        self.assertEqual(value["nrmse_percent"], 50)
        zero = audit.error_metrics(torch.zeros(1, 2), torch.ones(1, 2))
        self.assertIsNone(zero["nrmse_percent"])
        self.assertIsNone(zero["cosine"])
        both = audit.error_metrics(torch.zeros(1, 2), torch.zeros(1, 2))
        self.assertEqual(both["nrmse_percent"], 0)
        self.assertEqual(both["cosine"], 1)
        json.dumps(zero, allow_nan=False)

    def test_metric_rejects_nonfinite(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            audit.error_metrics(torch.ones(1), torch.tensor([float("nan")]))

    def test_bfloat16_export_and_undefined_metric_display(self) -> None:
        value = torch.tensor([.1, -3., 1e-5], dtype=torch.bfloat16)
        stored = audit.numpy_output(value)
        self.assertEqual(stored.dtype, np.float32)
        np.testing.assert_array_equal(stored.astype(np.float64), value.double().numpy())
        self.assertIn("undefined", audit.display_number(None))
        self.assertEqual(audit.display_number(1., ".2f"), "1.00")

    def test_same_input_requires_full_causal_history(self) -> None:
        native = {key: torch.ones(8, 2, 4).half() for key in ("query", "key", "value", "output")}
        ring = {key: value.clone() for key, value in native.items()}
        native["softmax_scale"] = ring["softmax_scale"] = .5
        ring["key"][3, 0, 0] = 2
        actual = audit.same_input_mask(native, ring, np.array([2, 3, 7]))
        self.assertEqual(actual.tolist(), [True, False, False])

    def test_layer_parser(self) -> None:
        self.assertEqual(audit.parse_layers("0-2,7,15"), [0, 1, 2, 7, 15])
        with self.assertRaises(ValueError):
            audit.parse_layers("3-1")

    def test_history_gap_and_overlap_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            native, _ = self.fixture(Path(directory), layers=1)
            groups = audit.groups_for_request(native, 2, [0])
            rows = next(iter(groups.values()))
            with self.assertRaisesRegex(ValueError, "gap/overlap"):
                audit.load_stream(native, rows[1:])
            with self.assertRaisesRegex(ValueError, "overlapping"):
                audit.validate_stream_records(next(iter(groups)), rows + [rows[0]])

    def test_wrong_semantic_request_not_matched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            native, ring = self.fixture(Path(directory), layers=1)
            path = ring / "manifest.jsonl"
            path.write_text(path.read_text().replace("foreground-loogle-1-1-0", "foreground-other-1-1-0"))
            args = audit.parse_args(["--native-trace-root", str(native), "--ring-trace-root", str(ring), "--output-dir", str(Path(directory) / "output"), "--layers", "0"])
            with self.assertRaisesRegex(ValueError, "semantic request/layer"):
                audit.run(args)

    def test_missing_qkv_and_path_escape_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            native, _ = self.fixture(Path(directory), layers=1)
            rows = next(iter(audit.groups_for_request(native, 2, [0]).values()))
            torch.save({"output": torch.ones(4, 4, 4)}, native / rows[0].file)
            with self.assertRaisesRegex(ValueError, "missing QKV"):
                audit.load_stream(native, rows)
            from dataclasses import replace
            outside = replace(rows[0], file="../outside.pt")
            with self.assertRaisesRegex(ValueError, "escapes"):
                audit.load_stream(native, [outside, rows[1]])

    def test_complete_artifacts_and_overwrite_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native, ring = self.fixture(root)
            output = root / "output"
            original_hash = audit.sha256(native / "l0_0.pt")
            args = audit.parse_args(["--native-trace-root", str(native), "--ring-trace-root", str(ring), "--output-dir", str(output), "--layers", "0,1", "--samples-per-layer", "4", "--threads", "2", "--source-label", "CPU synthetic test"])
            result = audit.run(args)
            self.assertEqual(result["status"], "complete")
            self.assertFalse(result["actual_gpu_kernel_reexecution"])
            self.assertEqual([row["full_output"]["compared_tokens"] for row in result["layers"]], [6, 6])
            self.assertEqual([row["sampled_fp64"]["same_input_samples"] for row in result["layers"]], [4, 0])
            with np.load(output / "sampled_outputs/layer_00.npz") as stored:
                np.testing.assert_array_equal(stored["native_fp64"], stored["ring_fp64"])
                self.assertEqual(stored["native_output"].shape, (4, 4, 4))
                self.assertEqual(stored["native_fp64"].dtype, np.float64)
            self.assertEqual(len(result["figures"]), 8)
            for name in result["figures"]:
                self.assertGreater((output / name).stat().st_size, 100)
                self.assertTrue((output / name.replace(".png", ".pdf")).exists())
            self.assertTrue((output / "report.html").is_file())
            self.assertTrue((output / "same_input_output_example.csv").is_file())
            self.assertEqual(json.loads((output / "summary.json").read_text())["status"], "complete")
            self.assertEqual(original_hash, audit.sha256(native / "l0_0.pt"))
            with self.assertRaises(FileExistsError):
                audit.run(args)


if __name__ == "__main__":
    unittest.main()
