#!/usr/bin/env python3
"""CPU smoke tests for direct dense/context-ring token RMSE comparison."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parent / "compare_dense_ring_token_rmse.py"
)
SPEC = importlib.util.spec_from_file_location(
    "compare_dense_ring_token_rmse_under_test", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load direct comparator: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None


class DenseRingTokenRmseSmoke(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_token_error_reduces_every_non_token_dimension(self) -> None:
        import torch

        dense = torch.zeros((2, 2), dtype=torch.float16)
        ring = torch.full((2, 2), 2.0, dtype=torch.float16)
        metrics = MODULE.token_error_metrics(dense, ring)

        self.assertAlmostEqual(metrics["rmse"], 2.0)
        self.assertAlmostEqual(metrics["mean_abs_error"], 2.0)
        self.assertAlmostEqual(metrics["max_abs_error"], 2.0)
        self.assertAlmostEqual(metrics["dense_rms"], 0.0)
        self.assertAlmostEqual(metrics["ring_rms"], 2.0)
        self.assertAlmostEqual(metrics["cosine_similarity"], 0.0)
        self.assertEqual(metrics["element_count"], 4.0)

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
    def test_cosine_similarity_is_reduced_per_token(self) -> None:
        import torch

        dense = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        ring = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        metrics = MODULE.token_error_metrics_batch(dense, ring)

        self.assertEqual(metrics["cosine_similarity"].tolist(), [1.0, 0.0])

    def test_select_positions_is_even_and_preserves_interval_gaps(self) -> None:
        self.assertEqual(
            MODULE.select_positions([(10, 13), (20, 23)], 3),
            [10, 12, 22],
        )
        self.assertEqual(
            MODULE.select_positions([(10, 13), (20, 23)], 0),
            [10, 11, 12, 20, 21, 22],
        )

    def test_common_intervals_and_quantiles(self) -> None:
        record = MODULE.TraceRecord
        dense = [
            record(
                "d", 0, "layer", "native_hbm", "prefill",
                0, 4, 0, "d0", "f16", (4, 2),
            ),
            record(
                "d", 0, "layer", "native_hbm", "prefill",
                6, 2, 0, "d1", "f16", (2, 2),
            ),
        ]
        ring = [
            record(
                "r", 0, "layer", "context_ring", "prefill",
                2, 5, 0, "r0", "f16", (5, 2),
            ),
        ]
        self.assertEqual(MODULE.common_intervals(dense, ring), [(2, 4), (6, 7)])
        rows = [
            {
                "element_count": 1,
                "rmse": value,
                "mean_abs_error": value,
                "max_abs_error": value,
                "dense_rms": 1.0,
                "ring_rms": 1.0,
                "nrmse": value,
                "cosine_similarity": 1.0,
            }
            for value in (1.0, 2.0, 3.0)
        ]
        summary = MODULE.summarize_values(rows, [1.5])
        self.assertEqual(summary["rmse_p50"], 2.0)
        self.assertAlmostEqual(summary["rmse_p95"], 2.9)
        self.assertEqual(summary["cosine_similarity_global"], 1.0)
        self.assertEqual(summary["thresholds"][0]["exceed_count"], 2)

    def test_auto_request_identity_ignores_policy_and_nonce(self) -> None:
        dense = (
            "cmpl-case_dense-foreground-loogle-1-1-0-deadbeef"
        )
        ring = (
            "cmpl-case_ring-foreground-loogle-1-1-0-cafebabe"
        )
        self.assertEqual(
            MODULE.benchmark_request_identity(dense),
            ("benchmark:foreground:loogle:1:1:0", 2),
        )
        self.assertEqual(
            MODULE.benchmark_request_identity(dense),
            MODULE.benchmark_request_identity(ring),
        )

    @unittest.skipUnless(
        TORCH_AVAILABLE and MATPLOTLIB_AVAILABLE,
        "PyTorch and matplotlib are required",
    )
    def test_cli_aligns_tokens_and_writes_cdf_artifacts(self) -> None:
        import torch

        layer_name = "model.layers.15.self_attn.attn"

        def write_trace(
            root: Path,
            *,
            execution_path: str,
            logical_start: int,
            output: torch.Tensor,
            request_id: str,
        ) -> None:
            rank_dir = root / "rank_000"
            rank_dir.mkdir(parents=True)
            payload_name = "output.pt"
            torch.save({"output": output}, rank_dir / payload_name)
            row = {
                "row": {
                    "request_id": request_id,
                    "request_ordinal": 0,
                    "phase": "prefill",
                },
                "layer_name": layer_name,
                "execution_path": execution_path,
                "logical_query_start": logical_start,
                "query_length": int(output.shape[0]),
                "occurrence": 0,
                "file": payload_name,
                "dtype": str(output.dtype),
                "shape": list(output.shape),
            }
            (rank_dir / "manifest.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dense_root = root / "dense"
            ring_root = root / "ring"
            output_dir = root / "result"
            dense = torch.zeros((4, 2, 2), dtype=torch.float16)
            ring = torch.stack([
                torch.full((2, 2), value, dtype=torch.float16)
                for value in (1.0, 2.0, 3.0)
            ])
            write_trace(
                dense_root,
                execution_path="native_hbm",
                logical_start=0,
                output=dense,
                request_id="dense-request",
            )
            write_trace(
                ring_root,
                execution_path="context_ring",
                logical_start=1,
                output=ring,
                request_id="ring-request",
            )

            return_code = MODULE.main([
                "--dense-trace-root",
                str(dense_root),
                "--ring-trace-root",
                str(ring_root),
                "--output-dir",
                str(output_dir),
                "--layer",
                layer_name,
                "--request-index",
                "1",
            ])
            self.assertEqual(return_code, 0)
            with (output_dir / "token_rmse_values.csv").open(
                encoding="utf-8"
            ) as values_file:
                rows = list(csv.DictReader(values_file))
            self.assertEqual(
                [int(row["logical_position"]) for row in rows], [1, 2, 3]
            )
            self.assertEqual(
                [float(row["rmse"]) for row in rows], [1.0, 2.0, 3.0]
            )
            summary = json.loads(
                (output_dir / "token_rmse_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            stream = summary["streams"][0]
            self.assertEqual(summary["error_accumulation_dtype"], "torch.float32")
            self.assertFalse(summary["attention_reexecution"])
            self.assertEqual(stream["dense_token_count"], 4)
            self.assertEqual(stream["ring_token_count"], 3)
            self.assertEqual(stream["common_token_count"], 3)
            self.assertEqual(stream["summary"]["rmse_p50"], 2.0)
            self.assertAlmostEqual(stream["summary"]["rmse_p95"], 2.9)
            self.assertEqual(
                stream["summary"]["cosine_similarity_global"], 0.0
            )
            self.assertTrue((output_dir / "token_rmse_summary.csv").is_file())
            self.assertGreater(
                (output_dir / "token_rmse_cdf.png").stat().st_size, 0
            )


if __name__ == "__main__":
    unittest.main()
