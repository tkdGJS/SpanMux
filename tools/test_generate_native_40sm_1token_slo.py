#!/usr/bin/env python3
"""CPU-only tests for native 40-SM one-token SLO reconstruction."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.generate_native_40sm_1token_slo import (
    LEGACY_PROVENANCE,
    build_result_rows,
    build_runtime_policy,
    load_context_samples,
)


class Native40SMOneTokenSLOTest(unittest.TestCase):
    def write_csv(
        self,
        path: Path,
        fields: list[str],
        rows: list[dict[str, object]],
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_legacy_bos_fragments_become_one_token_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "run_metadata.json").write_text(
                json.dumps({"status": "completed", "case_id": "legacy"}),
                encoding="utf-8",
            )
            request_fields = [
                "phase",
                "status",
                "context_target_tokens",
                "requested_max_tokens",
                "request_id",
                "repeat_index",
                "output_tokens",
                "ttft_ms",
                "completion_latency_ms",
            ]
            self.write_csv(
                root / "request_runs.csv",
                request_fields,
                [
                    {
                        "phase": "measurement",
                        "status": "success",
                        "context_target_tokens": 100,
                        "requested_max_tokens": 4,
                        "request_id": "req",
                        "repeat_index": 1,
                        "output_tokens": 5,
                        "ttft_ms": 10,
                        "completion_latency_ms": 100,
                    }
                ],
            )
            token_fields = [
                "phase",
                "request_id",
                "chunk_index",
                "chunk_tokens",
                "chunk_gap_ms",
            ]
            self.write_csv(
                root / "token_timeline.csv",
                token_fields,
                [
                    {
                        "phase": "measurement",
                        "request_id": "req",
                        "chunk_index": 0,
                        "chunk_tokens": 2,
                        "chunk_gap_ms": "",
                    },
                    *[
                        {
                            "phase": "measurement",
                            "request_id": "req",
                            "chunk_index": index,
                            "chunk_tokens": 2,
                            "chunk_gap_ms": gap,
                        }
                        for index, gap in enumerate((20, 30, 40), start=1)
                    ],
                ],
            )

            samples = load_context_samples(root)
            self.assertEqual(samples.provenance, LEGACY_PROVENANCE)
            self.assertEqual(samples.one_token_tbt_ms, (20.0, 30.0, 40.0))
            rows = build_result_rows([samples])
            p50 = rows[0]
            self.assertEqual(p50["profiling_ttft_ms"], 10.0)
            self.assertEqual(p50["ttft_slo_ms"], 20.0)
            self.assertEqual(p50["profiling_one_token_tbt_ms"], 30.0)
            self.assertEqual(p50["tbt_slo_ms"], 37.5)
            self.assertEqual(p50["profiling_e2e_ms"], 100.0)
            self.assertEqual(p50["e2e_slo_ms"], 125.0)
            policy, policy_rows = build_runtime_policy(rows, [samples])
            self.assertEqual(policy["buckets"][0]["tbt_slo_ms"], 37.5)
            self.assertEqual(
                policy["buckets"][0]["ttft_prefill_deadline_ms"], 20.0
            )
            self.assertEqual(
                policy["slo_contract"]["tbt_measurement_source"],
                "choice.token_ids",
            )
            self.assertFalse(
                policy["calibration_provenance"]["direct_choice_token_ids"]
            )
            self.assertEqual(
                policy_rows[0]["calibration_provenance"],
                LEGACY_PROVENANCE,
            )


if __name__ == "__main__":
    unittest.main()
