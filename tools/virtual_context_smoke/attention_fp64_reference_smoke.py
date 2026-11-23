#!/usr/bin/env python3
"""CPU smoke test for the FP64 attention reference helper."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parent / "compare_attention_fp64_reference.py"
)
SPEC = importlib.util.spec_from_file_location(
    "compare_attention_fp64_reference_under_test", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load FP64 comparator: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AttentionFP64ReferenceSmoke(unittest.TestCase):
    def test_gqa_dense_reference_matches_expanded_dense_attention(self) -> None:
        import torch

        torch.manual_seed(7)
        query_heads = 4
        kv_heads = 2
        head_dim = 8
        query_position = 10
        query = torch.randn(query_heads, head_dim, dtype=torch.float16)
        key = torch.randn(12, kv_heads, head_dim, dtype=torch.float16)
        value = torch.randn(12, kv_heads, head_dim, dtype=torch.float16)
        scale = head_dim**-0.5

        actual = MODULE.dense_causal_attention_fp64(
            query,
            key,
            value,
            query_position=query_position,
            softmax_scale=scale,
        )
        repeat = query_heads // kv_heads
        expanded_key = key[: query_position + 1].double().repeat_interleave(
            repeat, dim=1
        )
        expanded_value = value[: query_position + 1].double().repeat_interleave(
            repeat, dim=1
        )
        scores = torch.einsum(
            "hd,khd->hk", query.double(), expanded_key
        ) * scale
        expected = torch.einsum(
            "hk,khd->hd", torch.softmax(scores, dim=-1), expanded_value
        )

        self.assertTrue(torch.allclose(actual, expected, rtol=1e-12, atol=1e-12))
        self.assertTrue(math.isfinite(float(actual.sum())))

    def test_trace_comparison_reconstructs_kv_and_flattened_output(self) -> None:
        import torch

        torch.manual_seed(11)
        token_count = 6
        query_heads = 4
        kv_heads = 2
        head_dim = 8
        scale = head_dim**-0.5
        query = torch.randn(
            token_count, query_heads, head_dim, dtype=torch.float16
        )
        key = torch.randn(
            token_count, kv_heads, head_dim, dtype=torch.float16
        )
        value = torch.randn(
            token_count, kv_heads, head_dim, dtype=torch.float16
        )
        output = torch.stack([
            MODULE.dense_causal_attention_fp64(
                query[position],
                key,
                value,
                query_position=position,
                softmax_scale=scale,
            ).half().flatten()
            for position in range(token_count)
        ])
        record = MODULE.TraceRecord(
            request_id="request-0",
            request_ordinal=0,
            layer_name="model.layers.0.self_attn.attn",
            execution_path="native_hbm",
            phase="prefill",
            logical_start=0,
            token_count=token_count,
            occurrence=0,
            file="trace.pt",
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            rank_dir = Path(temporary_dir)
            torch.save(
                {
                    "output": output,
                    "query": query,
                    "key": key,
                    "value": value,
                    "softmax_scale": scale,
                },
                rank_dir / record.file,
            )
            result = MODULE.compare_stream(
                rank_dir,
                [record],
                [record],
                max_samples=3,
                softmax_scale_override=None,
            )

        self.assertEqual(result["sample_count"], 3)
        self.assertLess(result["rmse"], 5e-4)

    def test_paired_cli_uses_shared_logical_positions(self) -> None:
        import torch

        torch.manual_seed(17)
        token_count = 6
        query_heads = 4
        kv_heads = 2
        head_dim = 8
        scale = head_dim**-0.5
        query = torch.randn(
            token_count, query_heads, head_dim, dtype=torch.float16
        )
        key = torch.randn(
            token_count, kv_heads, head_dim, dtype=torch.float16
        )
        value = torch.randn(
            token_count, kv_heads, head_dim, dtype=torch.float16
        )
        output = torch.stack([
            MODULE.dense_causal_attention_fp64(
                query[position],
                key,
                value,
                query_position=position,
                softmax_scale=scale,
            ).half().flatten()
            for position in range(token_count)
        ])

        def write_trace(
            root: Path,
            chunks: list[tuple[str, int, int]],
            request_id: str,
        ) -> None:
            rank_dir = root / "rank_000"
            rank_dir.mkdir(parents=True)
            manifest_rows = []
            for capture_index, (path, start, count) in enumerate(chunks):
                filename = f"chunk-{capture_index}.pt"
                end = start + count
                torch.save(
                    {
                        "output": output[start:end],
                        "query": query[start:end],
                        "key": key[start:end],
                        "value": value[start:end],
                        "softmax_scale": scale,
                    },
                    rank_dir / filename,
                )
                manifest_rows.append({
                    "row": {
                        "request_id": request_id,
                        "request_ordinal": 0,
                        "phase": "prefill",
                    },
                    "layer_name": "model.layers.0.self_attn.attn",
                    "execution_path": path,
                    "logical_query_start": start,
                    "query_length": count,
                    "occurrence": 0,
                    "file": filename,
                })
            (rank_dir / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in manifest_rows),
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            native_root = root / "native"
            ring_root = root / "ring"
            write_trace(
                native_root,
                [("native_hbm", 0, token_count)],
                "native-request",
            )
            write_trace(
                ring_root,
                [("native_hbm", 0, 2), ("context_ring", 2, 4)],
                "ring-request",
            )
            output_json = root / "pair.json"
            subprocess.run(
                [
                    sys.executable,
                    str(
                        Path(__file__).resolve().parent
                        / "compare_attention_fp64_pair.py"
                    ),
                    "--native-trace-root",
                    str(native_root),
                    "--ring-trace-root",
                    str(ring_root),
                    "--max-samples-per-stream",
                    "2",
                    "--output-json",
                    str(output_json),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output_json.read_text(encoding="utf-8"))

        self.assertEqual(result["streams"][0]["sample_positions"], [2, 5])
        self.assertAlmostEqual(
            result["native"]["rmse"], result["ring"]["rmse"], places=15
        )


if __name__ == "__main__":
    unittest.main()
