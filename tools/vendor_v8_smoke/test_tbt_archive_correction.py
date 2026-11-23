from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from measure_real_workload_motivation import (
    IncompleteCompletionError,
    PreparedRequest,
    ReplayEvent,
    exact_completion_token_ids,
    request_payload,
)

SCRIPT = ROOT / "tools" / "recalculate_archived_native_tbt.py"


def test_live_measurement_requires_exact_vllm_delta_token_ids() -> None:
    assert exact_completion_token_ids({"token_ids": [7, 8]}) == [7, 8]
    with pytest.raises(IncompleteCompletionError):
        exact_completion_token_ids({"text": "retokenization is unsafe"})

    request = PreparedRequest("mock", "prompt", 1, 4, 4, "source")
    event = ReplayEvent("foreground", "mock", request, 0, 0.0, True, "t0")
    payload = request_payload("model", event, "request", 4, 4, True, True)
    assert payload["return_token_ids"] is True


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_archived_bos_correction_reconstructs_model_token_gaps(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "run_metadata.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    write_csv(
        input_dir / "request_runs.csv",
        [
            "timestamp_utc",
            "case_id",
            "phase",
            "bucket_index",
            "context_target_tokens",
            "repeat_index",
            "request_id",
            "tenant_id",
            "workload",
            "source_id",
            "status",
            "error_message",
            "input_tokens",
            "requested_min_tokens",
            "requested_max_tokens",
            "output_tokens",
            "ttft_ms",
            "mean_tbt_ms",
            "p50_tbt_ms",
            "p90_tbt_ms",
            "p99_tbt_ms",
            "completion_latency_ms",
            "time_to_last_token_ms",
            "wall_elapsed_ms",
            "cooldown_after_sec",
        ],
        [
            {
                "timestamp_utc": "",
                "case_id": "case",
                "phase": "measurement",
                "bucket_index": 0,
                "context_target_tokens": 100,
                "repeat_index": 1,
                "request_id": "req",
                "tenant_id": "tenant",
                "workload": "mock",
                "source_id": "source",
                "status": "success",
                "error_message": "",
                "input_tokens": 100,
                "requested_min_tokens": 4,
                "requested_max_tokens": 4,
                "output_tokens": 5,
                "ttft_ms": 100,
                "mean_tbt_ms": 5,
                "p50_tbt_ms": 5,
                "p90_tbt_ms": 5,
                "p99_tbt_ms": 5,
                "completion_latency_ms": 140,
                "time_to_last_token_ms": 140,
                "wall_elapsed_ms": 140,
                "cooldown_after_sec": 0,
            }
        ],
    )
    token_fields = [
        "case_id",
        "phase",
        "bucket_index",
        "context_target_tokens",
        "repeat_index",
        "request_id",
        "tenant_id",
        "workload",
        "chunk_index",
        "chunk_tokens",
        "chunk_gap_ms",
        "estimated_per_token_gap_ms",
        "chunk_arrival_offset_s",
    ]
    token_rows = []
    for index, tokens, gap in ((0, 2, ""), (1, 2, 10), (2, 3, 20)):
        token_rows.append(
            {
                "case_id": "case",
                "phase": "measurement",
                "bucket_index": 0,
                "context_target_tokens": 100,
                "repeat_index": 1,
                "request_id": "req",
                "tenant_id": "tenant",
                "workload": "mock",
                "chunk_index": index,
                "chunk_tokens": tokens,
                "chunk_gap_ms": gap,
                "estimated_per_token_gap_ms": "" if gap == "" else 5,
                "chunk_arrival_offset_s": index * 0.01,
            }
        )
    write_csv(input_dir / "token_timeline.csv", token_fields, token_rows)
    write_csv(
        input_dir / "bucket_summary.csv",
        [
            "timestamp_utc",
            "case_id",
            "bucket_index",
            "context_target_tokens",
            "actual_input_tokens",
            "warmup_count",
            "warmup_successes",
            "measured_repeat_count",
            "measurement_successes",
            "measurement_failures",
            "request_max_tokens",
            "request_min_tokens",
            "cooldown_sec",
            "mean_ttft_ms",
            "stdev_ttft_ms",
            "mean_request_tbt_ms",
            "stdev_request_tbt_ms",
            "pooled_p50_tbt_ms",
            "pooled_p90_tbt_ms",
            "pooled_p99_tbt_ms",
            "mean_completion_latency_ms",
            "stdev_completion_latency_ms",
            "mean_output_tokens",
            "mean_output_tokens_per_sec",
        ],
        [
            {
                "timestamp_utc": "",
                "case_id": "case",
                "bucket_index": 0,
                "context_target_tokens": 100,
                "actual_input_tokens": 100,
                "warmup_count": 0,
                "warmup_successes": 0,
                "measured_repeat_count": 1,
                "measurement_successes": 1,
                "measurement_failures": 0,
                "request_max_tokens": 4,
                "request_min_tokens": 4,
                "cooldown_sec": 0,
                "mean_ttft_ms": 100,
                "stdev_ttft_ms": "",
                "mean_request_tbt_ms": 5,
                "stdev_request_tbt_ms": "",
                "pooled_p50_tbt_ms": 5,
                "pooled_p90_tbt_ms": 5,
                "pooled_p99_tbt_ms": 5,
                "mean_completion_latency_ms": 140,
                "stdev_completion_latency_ms": "",
                "mean_output_tokens": 5,
                "mean_output_tokens_per_sec": 35.7143,
            }
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
    )

    with (output_dir / "request_runs.csv").open(
        "r", encoding="utf-8", newline=""
    ) as input_file:
        request = next(csv.DictReader(input_file))
    assert request["output_tokens"] == "4"
    assert request["mean_tbt_ms"] == "10.0"
    assert request["p50_tbt_ms"] == "10.0"

    with (output_dir / "token_timeline.csv").open(
        "r", encoding="utf-8", newline=""
    ) as input_file:
        tokens = list(csv.DictReader(input_file))
    assert [row["chunk_tokens"] for row in tokens] == ["1", "1", "2"]
    assert tokens[-1]["estimated_per_token_gap_ms"] == "10.0"
