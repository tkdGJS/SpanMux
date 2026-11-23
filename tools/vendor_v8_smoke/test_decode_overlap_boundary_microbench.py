#!/usr/bin/env python3
"""CPU-only tests for sampled P/D CUDA-boundary overlap analysis."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from analyze_vendor_v8_decode_overlap_microbench import (  # noqa: E402
    classify_decode_records,
    load_cuda_forward_records,
    summarize_decode_records,
)


class DecodeOverlapBoundaryTest(unittest.TestCase):
    def test_partition_profiler_owns_exact_token_helper(self) -> None:
        source = (
            ROOT / "measure_vendor_v8_partition_profile.py"
        ).read_text(encoding="utf-8")
        import_block = source.split(
            "from measure_real_workload_motivation import (", 1
        )[1].split(")", 1)[0]
        self.assertNotIn("exact_completion_token_ids", import_block)
        self.assertNotIn("validate_completion_contract", import_block)
        self.assertIn("def exact_completion_token_ids(", source)
        self.assertIn("def validate_completion_contract(", source)
        self.assertIn('choice.get("token_ids")', source)
        exact_helper = source.split(
            "def exact_completion_token_ids(", 1
        )[1].split("def validate_completion_contract(", 1)[0]
        self.assertNotIn("tokenize", exact_helper)

    def test_partition_profiler_imports_with_legacy_workload_helpers(self) -> None:
        fake = types.ModuleType("measure_real_workload_motivation")
        fake.PreparedRequest = type("PreparedRequest", (), {})
        fake.get_model_id_from_server = lambda *args: "model"
        fake.get_requests_module = lambda: None
        fake.load_prepared_jsonl = lambda path: []
        fake.parse_sse_json = lambda line: {}
        fake.percentile = lambda values, quantile: 0.0
        previous = sys.modules.get("measure_real_workload_motivation")
        sys.modules["measure_real_workload_motivation"] = fake
        try:
            path = ROOT / "measure_vendor_v8_partition_profile.py"
            spec = importlib.util.spec_from_file_location(
                "partition_profile_legacy_import_test", path
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.assertEqual(
                module.exact_completion_token_ids({"token_ids": [1, 2]}),
                [1, 2],
            )
        finally:
            sys.modules.pop("partition_profile_legacy_import_test", None)
            if previous is None:
                sys.modules.pop("measure_real_workload_motivation", None)
            else:
                sys.modules["measure_real_workload_motivation"] = previous

    def test_shared_epoch_intersection_classifies_decode(self) -> None:
        records = [
            {
                "phase": "decode",
                "kind": "forward",
                "device_epoch_id": "epoch",
                "device_start_ms_from_epoch": 0.0,
                "device_end_ms_from_epoch": 10.0,
                "device_elapsed_ms": 10.0,
                "host_enqueue_start_ts": 1.0,
                "host_enqueue_end_ts": 1.001,
            },
            {
                "phase": "prefill",
                "kind": "forward",
                "device_epoch_id": "epoch",
                "device_start_ms_from_epoch": 12.0,
                "device_end_ms_from_epoch": 20.0,
                "device_elapsed_ms": 8.0,
            },
            {
                "phase": "decode",
                "kind": "forward",
                "device_epoch_id": "epoch",
                "device_start_ms_from_epoch": 15.0,
                "device_end_ms_from_epoch": 25.0,
                "device_elapsed_ms": 10.0,
                "host_enqueue_start_ts": 2.0,
                "host_enqueue_end_ts": 2.002,
            },
        ]
        classified = classify_decode_records(records)
        self.assertEqual(
            [row["prefill_device_overlap"] for row in classified], [0, 1]
        )
        self.assertEqual(classified[1]["prefill_device_overlap_ms"], 5.0)
        summary = summarize_decode_records(classified)
        self.assertEqual(summary[0]["sample_count"], 1)
        self.assertEqual(summary[1]["sample_count"], 1)

    def test_loader_rejects_host_only_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "kind": "forward",
                        "phase": "decode",
                        "device_elapsed_ms": 10.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "shared device epoch"):
                load_cuda_forward_records(path)

    def test_worker_trace_is_sampled_without_forcing_eager(self) -> None:
        source = (ROOT / "vendor_v8/vllm/v1/worker/gpu_model_runner.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def _get_pd_cuda_trace_epoch(", source)
        self.assertIn("device_start_ms_from_epoch", source)
        self.assertIn("VLLM_PD_CUDA_EVENT_TRACE_DECODE_EVERY_N", source)
        self.assertIn("VLLM_PD_CUDA_EVENT_TRACE_PREFILL_EVERY_N", source)
        trace_body = source.split("def _pd_cuda_phase_trace(", 1)[1].split(
            "def _warn_pd_decode_gpu_component_trace_once", 1
        )[0]
        self.assertNotIn("synchronize", trace_body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
