#!/usr/bin/env python3
"""CPU-only contracts for the BurstGPT hybrid workload path."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from measure_burstgpt_mixed_workload import load_workload, summarize
from prepare_ctx100k_dataset_examples import PromptSource
from prepare_burstgpt_loogle_sharegpt_workload import (
    BurstTraceRow,
    load_burst_trace_rows,
    select_burst_window,
    select_distinct_long_sources,
)


class BurstGPTHybridWorkloadTest(unittest.TestCase):
    def test_long_source_pack_is_explicit_and_source_disjoint(self) -> None:
        class WhitespaceTokenizer:
            @staticmethod
            def encode(text: str, add_special_tokens: bool = True) -> list[str]:
                del add_special_tokens
                return text.split()

            @staticmethod
            def decode(
                token_ids: list[str],
                skip_special_tokens: bool = False,
                clean_up_tokenization_spaces: bool = False,
            ) -> str:
                del skip_special_tokens, clean_up_tokenization_spaces
                return " ".join(token_ids)

        def source(source_id: str, tokens: int) -> PromptSource:
            return PromptSource(
                workload="loogle",
                prefix="",
                body=" ".join([source_id] * tokens),
                suffix="",
                source_ids=(source_id,),
                source_path=Path("loogle.jsonl"),
                role_default="background",
                stream_default=False,
                construction="single_document_truncated",
                source_faithful=True,
            )

        selected = select_distinct_long_sources(
            WhitespaceTokenizer(),
            [
                source("single", 35),
                source("pack-a", 6),
                source("pack-b", 5),
            ],
            count=2,
            target_tokens=35,
            underlength_policy="pack",
        )
        self.assertEqual(len(selected), 2)
        self.assertTrue(selected[0].source_faithful)
        self.assertFalse(selected[1].source_faithful)
        self.assertEqual(
            selected[1].construction,
            "multi_document_packed_capacity_stress",
        )
        self.assertEqual(set(selected[1].source_ids), {"pack-a", "pack-b"})
        self.assertFalse(
            set(selected[0].source_ids).intersection(selected[1].source_ids)
        )

    def test_official_column_aliases_and_invalid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "Timestamp",
                        "Session ID",
                        "Elapsed time",
                        "Model",
                        "Request tokens",
                        "Response tokens",
                        "Log Type",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Timestamp": "1.5",
                        "Session ID": "session-a",
                        "Elapsed time": "3.25",
                        "Model": "gpt-x",
                        "Request tokens": "1024",
                        "Response tokens": "64",
                        "Log Type": "chat",
                    }
                )
                writer.writerow(
                    {
                        "Timestamp": "2.0",
                        "Request tokens": "1024",
                        "Response tokens": "0",
                    }
                )
            rows = load_burst_trace_rows(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].source_row_index, 2)
        self.assertEqual(rows[0].request_tokens, 1024)
        self.assertEqual(rows[0].response_tokens, 64)
        self.assertEqual(rows[0].session_id, "session-a")
        self.assertEqual(rows[0].elapsed_time_s, 3.25)

    def test_long_position_replaces_only_request_length(self) -> None:
        rows = [
            BurstTraceRow(2, 1.0, 64, 32, "", "", ""),
            BurstTraceRow(3, 2.0, 128, 32, "", "", ""),
            BurstTraceRow(4, 3.0, 2048, 64, "", "", ""),
        ]
        selected = select_burst_window(
            rows,
            total_requests=2,
            long_event_indices={0},
            start_valid_index=0,
            short_min_tokens=512,
            short_max_tokens=4096,
            output_min_tokens=16,
            output_max_tokens=128,
        )
        self.assertEqual([row.source_row_index for row in selected], [2, 4])

    def test_selected_timestamps_must_be_ordered(self) -> None:
        rows = [
            BurstTraceRow(2, 2.0, 1024, 32, "", "", ""),
            BurstTraceRow(3, 1.0, 1024, 32, "", "", ""),
        ]
        with self.assertRaisesRegex(RuntimeError, "nondecreasing"):
            select_burst_window(
                rows,
                total_requests=2,
                long_event_indices=set(),
                start_valid_index=0,
                short_min_tokens=512,
                short_max_tokens=4096,
                output_min_tokens=16,
                output_max_tokens=128,
            )

    def test_replay_loader_requires_exact_output_and_ordered_offsets(
        self,
    ) -> None:
        base = {
            "request_id": "req-0",
            "tenant_id": "tenant-0",
            "role": "short",
            "workload": "sharegpt",
            "prompt": "hello",
            "input_tokens": 10,
            "max_tokens": 20,
            "min_tokens": 20,
            "scheduled_offset_s": 0.0,
            "source_id": "source-0",
            "burstgpt_source_row_index": 2,
            "burstgpt_timestamp_s": 1.0,
            "burstgpt_original_request_tokens": 10,
            "burstgpt_request_length_replaced": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.jsonl"
            second = dict(base)
            second.update(
                request_id="req-1",
                scheduled_offset_s=1.0,
                burstgpt_source_row_index=3,
                burstgpt_timestamp_s=2.0,
            )
            path.write_text(
                json.dumps(base) + "\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )
            requests = load_workload(path)
            self.assertEqual(len(requests), 2)
            second["min_tokens"] = 19
            path.write_text(json.dumps(second) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "min_tokens"):
                load_workload(path)

    def test_summary_reports_roles_and_all(self) -> None:
        raw = [
            {
                "request_id": "long",
                "role": "long",
                "status": "success",
                "input_tokens": 125000,
                "output_tokens": 2,
                "ttft_ms": 100.0,
                "p50_tbt_ms": 10.0,
                "p90_tbt_ms": 12.0,
                "p99_tbt_ms": 14.0,
                "completion_latency_ms": 120.0,
                "ttft_slo_met": 1,
            },
            {
                "request_id": "short",
                "role": "short",
                "status": "failed",
                "input_tokens": 1000,
            },
        ]
        tokens = [
            {
                "request_id": "long",
                "chunk_tokens": 2,
                "estimated_per_token_gap_ms": 10.0,
                "tbt_slo_met": 1,
            }
        ]
        by_role = {row["role"]: row for row in summarize(raw, tokens)}
        self.assertEqual(by_role["all"]["request_count"], 2)
        self.assertEqual(by_role["long"]["success_count"], 1)
        self.assertEqual(by_role["short"]["failure_count"], 1)

    def test_runner_dry_run_has_no_input_dependency(self) -> None:
        result = subprocess.run(
            [str(ROOT / "run_vendor_v8_burstgpt_mixed_125k.sh"), "--dry-run"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("replay=open_loop", result.stdout)
        self.assertIn("residency=jit_min_copy", result.stdout)
        self.assertIn("no files", result.stdout)


if __name__ == "__main__":
    unittest.main()
