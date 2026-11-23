# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

from graph_script.generate_dynamic_hbm_vendor_native_comparison import generate
from graph_script.report_pd_experiment import (
    collect_gpu_utilization,
    infer_artifact_dir,
    infer_graph_dir,
    render_report,
    summarize_experiment,
    write_native_comparison_inputs,
)


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_completed_result(tmp_path: Path) -> tuple[Path, Path]:
    results_dir = tmp_path / "case_full_vendor_v8_results"
    artifact_dir = tmp_path / "case_full_vendor_v8_artifacts"
    results_dir.mkdir()
    artifact_dir.mkdir()
    (results_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "case_id": "case_full_vendor_v8",
                "long_count": 1,
                "short_injection_offsets_sec": [1.0],
            }
        ),
        encoding="utf-8",
    )
    request_fields = [
        "role",
        "request_index",
        "request_id",
        "context_tokens",
        "status",
        "error_message",
        "submit_monotonic_ts",
        "first_token_monotonic_ts",
        "completion_monotonic_ts",
        "output_tokens",
        "ttft_ms",
        "mean_tbt_ms",
        "p95_tbt_ms",
        "p99_tbt_ms",
        "completion_latency_ms",
        "ttft_slo_ms",
        "tbt_slo_ms",
        "ttft_slo_met",
        "tbt_token_slo_attainment",
    ]
    write_csv(
        results_dir / "request_results.csv",
        request_fields,
        [
            {
                "role": "long",
                "request_index": 0,
                "request_id": "case-long-1",
                "context_tokens": 100,
                "status": "success",
                "error_message": "",
                "submit_monotonic_ts": 10,
                "first_token_monotonic_ts": 11,
                "completion_monotonic_ts": 20,
                "output_tokens": 20,
                "ttft_ms": 1000,
                "mean_tbt_ms": 3,
                "p95_tbt_ms": 4,
                "p99_tbt_ms": 4,
                "completion_latency_ms": 10000,
                "ttft_slo_ms": 2000,
                "tbt_slo_ms": 5,
                "ttft_slo_met": 1,
                "tbt_token_slo_attainment": 0.5,
            },
            {
                "role": "short",
                "request_index": 0,
                "request_id": "case-short-1",
                "context_tokens": 10,
                "status": "success",
                "error_message": "",
                "submit_monotonic_ts": 11,
                "first_token_monotonic_ts": 14,
                "completion_monotonic_ts": 18,
                "output_tokens": 10,
                "ttft_ms": 3000,
                "mean_tbt_ms": 7,
                "p95_tbt_ms": 8,
                "p99_tbt_ms": 9,
                "completion_latency_ms": 7000,
                "ttft_slo_ms": 2000,
                "tbt_slo_ms": 5,
                "ttft_slo_met": 0,
                "tbt_token_slo_attainment": 0,
            },
        ],
    )
    write_csv(
        results_dir / "client_token_timeline.csv",
        ["role", "request_id", "estimated_per_token_gap_ms", "tbt_slo_met"],
        [
            {
                "role": "long",
                "request_id": "case-long-1",
                "estimated_per_token_gap_ms": 4,
                "tbt_slo_met": 1,
            },
            {
                "role": "long",
                "request_id": "case-long-1",
                "estimated_per_token_gap_ms": 6,
                "tbt_slo_met": 0,
            },
            {
                "role": "short",
                "request_id": "case-short-1",
                "estimated_per_token_gap_ms": 7,
                "tbt_slo_met": 0,
            },
            {
                "role": "short",
                "request_id": "case-short-1",
                "estimated_per_token_gap_ms": 8,
                "tbt_slo_met": 0,
            },
        ],
    )
    (artifact_dir / "nvidia_smi_dmon.log").write_text(
        "#Date Time gpu pwr gtemp mtemp sm\n"
        "20260812 00:00:00 0 1 1 - 0\n"
        "20260812 00:00:01 0 1 1 - 100\n"
        "20260812 00:00:02 0 1 1 - 0\n"
        "20260812 00:00:03 0 1 1 - 5\n"
        "20260812 00:00:04 0 1 1 - 50\n"
        "20260812 00:00:05 0 1 1 - 0\n",
        encoding="utf-8",
    )
    return results_dir, artifact_dir


def test_completed_summary_uses_token_level_slo_and_workload_e2e(
    tmp_path: Path,
) -> None:
    results_dir, artifact_dir = make_completed_result(tmp_path)
    summary = summarize_experiment(results_dir, artifact_dir)

    assert summary["health"] == "PASS"
    assert summary["expected_requests"] == 2
    assert summary["e2e_s"] == 10
    assert summary["output_tokens_per_s"] == 3
    assert summary["ttft_slo_attainment"] == 0.5
    assert summary["tbt_token_slo_attainment"] == 0.25
    assert summary["p99_tbt_request_attainment"] == 0.5
    assert summary["request_token_attainment"] == {
        "case-long-1": 0.5,
        "case-short-1": 0.0,
    }
    report = render_report(summary)
    assert "normal operation: PASS" in report
    assert "Official TBT token observations: 4" in report
    assert "Req p99 TBT<=SLO is a diagnostic" in report


def test_interrupted_result_is_not_reported_as_complete(tmp_path: Path) -> None:
    results_dir, artifact_dir = make_completed_result(tmp_path)
    metadata = json.loads((results_dir / "run_metadata.json").read_text())
    metadata["status"] = "running"
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata))
    rows = list(csv.DictReader((results_dir / "request_results.csv").open()))
    write_csv(
        results_dir / "request_results.csv",
        list(rows[0]),
        [rows[0]],
    )

    summary = summarize_experiment(results_dir, artifact_dir)

    assert summary["health"] == "INCOMPLETE"
    assert summary["missing_count"] == 1
    assert not summary["e2e_is_complete"]
    assert "observed span" in render_report(summary)


def test_path_inference_and_active_gpu_envelope(tmp_path: Path) -> None:
    results_dir, artifact_dir = make_completed_result(tmp_path)

    assert infer_artifact_dir(results_dir) == artifact_dir
    assert infer_graph_dir(results_dir).name == "case_full_vendor_v8_abcde"
    gpu = collect_gpu_utilization(artifact_dir / "nvidia_smi_dmon.log")
    assert gpu["capture_samples"] == 6
    assert gpu["active_samples"] == 4
    assert gpu["mean"] == 38.75
    assert gpu["under_10_ratio"] == 0.5
    assert gpu["longest_under_10_samples"] == 2


def test_native_comparison_inputs_match_existing_plotter_contract(
    tmp_path: Path,
) -> None:
    results_dir, artifact_dir = make_completed_result(tmp_path)
    current = summarize_experiment(results_dir, artifact_dir)
    baseline = summarize_experiment(results_dir, artifact_dir)
    current["request_rows"][1]["status"] = "failed"
    comparison_dir = tmp_path / "comparison"

    request_path, run_path = write_native_comparison_inputs(
        current=current,
        baseline=baseline,
        output_dir=comparison_dir,
    )

    rows = list(csv.DictReader(request_path.open()))
    assert len(rows) == 6
    long_ttft = next(
        row
        for row in rows
        if row["role"] == "long" and row["metric"] == "ttft_ms"
    )
    assert long_ttft["native_40sm"] == "1.000000000"
    assert long_ttft["full_vendor_v8"] == "1.000000000"
    assert long_ttft["slo_value"] == "2.000000000"
    short_ttft = next(
        row
        for row in rows
        if row["role"] == "short" and row["metric"] == "ttft_ms"
    )
    assert short_ttft["full_vendor_v8"] == ""
    assert short_ttft["full_vendor_status"] == "failed"
    run_rows = list(csv.DictReader(run_path.open()))
    assert [row["experiment"] for row in run_rows] == [
        "Native 40SM chunked",
        "case_full_vendor_v8",
    ]
    generate(comparison_dir)
    assert {
        "01_whole_experiment_e2e_bar.png",
        "02_long_ttft_bar.png",
        "03_long_tbt_mean_p99_bar.png",
        "04_short_ttft_bar_log.png",
        "05_short_tbt_mean_p99_bar.png",
    } <= {path.name for path in comparison_dir.iterdir()}
