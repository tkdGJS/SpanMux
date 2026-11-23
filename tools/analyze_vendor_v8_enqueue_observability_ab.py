#!/usr/bin/env python3
"""Summarize the four-way host-enqueue/observability ablation."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def _time_metrics(path: Path) -> dict[str, float]:
    result = {
        "cpu_user_s": 0.0,
        "cpu_system_s": 0.0,
        "max_rss_kb": 0.0,
    }
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.strip().partition(":")
        if not separator:
            continue
        try:
            if key == "User time (seconds)":
                result["cpu_user_s"] = float(value.strip())
            elif key == "System time (seconds)":
                result["cpu_system_s"] = float(value.strip())
            elif key == "Maximum resident set size (kbytes)":
                result["max_rss_kb"] = float(value.strip())
        except ValueError:
            continue
    return result


def _host_order_metrics(path: Path) -> dict[str, int]:
    applied = 0
    violations = 0
    if not path.is_file():
        return {"host_order_samples": 0, "host_order_violations": 0}
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if float(record.get("pd_decode_first_host_enqueue_applied", 0) or 0) <= 0:
                continue
            phase = str(record.get("pd_async_uniproc_rpc_phase", "")).split(":", 1)[0]
            if phase not in {"decode", "prefill"}:
                continue
            records.append(
                (
                    float(record.get("pd_async_uniproc_submit_attempt_ts", 0) or 0),
                    float(record.get("pd_async_uniproc_worker_call_start_ts", 0) or 0),
                    float(record.get("pd_async_uniproc_worker_call_end_ts", 0) or 0),
                    phase,
                )
            )
    records.sort()
    pending_decode_end = 0.0
    for _, start, end, phase in records:
        applied += 1
        if phase == "decode":
            pending_decode_end = max(pending_decode_end, end)
        elif start < pending_decode_end:
            violations += 1
    return {
        "host_order_samples": applied,
        "host_order_violations": violations,
    }


def summarize_case(root: Path, experiment: str, cpu_dir: Path) -> dict[str, object]:
    result_dir = root / f"{experiment}_results"
    artifact_dir = root / f"{experiment}_artifacts"
    rows = []
    request_csv = result_dir / "request_results.csv"
    if request_csv.is_file():
        with request_csv.open(newline="", encoding="utf-8") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row.get("status") == "success"
            ]
    output_tokens = sum(int(_float(row, "output_tokens")) for row in rows)
    starts = [_float(row, "submit_monotonic_ts") for row in rows]
    ends = [_float(row, "completion_monotonic_ts") for row in rows]
    makespan_s = max(ends) - min(starts) if starts and ends else 0.0
    mean_tbts = [_float(row, "mean_tbt_ms") for row in rows]
    p95_tbts = [_float(row, "p95_tbt_ms") for row in rows]
    summary: dict[str, object] = {
        "experiment": experiment,
        "successes": len(rows),
        "output_tokens": output_tokens,
        "makespan_s": makespan_s,
        "throughput_tok_s": output_tokens / makespan_s if makespan_s > 0 else 0.0,
        "request_median_mean_tbt_ms": (
            statistics.median(mean_tbts) if mean_tbts else 0.0
        ),
        "request_median_p95_tbt_ms": statistics.median(p95_tbts) if p95_tbts else 0.0,
    }
    summary.update(_time_metrics(cpu_dir / f"{experiment}.time.txt"))
    summary.update(
        _host_order_metrics(artifact_dir / "iteration_metrics.jsonl")
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--base-experiment", required=True)
    parser.add_argument("--cpu-dir", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    cpu_dir = args.cpu_dir or args.root / f"{args.base_experiment}_cpu"
    cases = (
        "legacy_contention_debug",
        "enqueue_ordered_debug",
        "contention_minimal",
        "optimized_minimal",
    )
    summaries = [
        summarize_case(
            args.root,
            f"{args.base_experiment}_{case}",
            cpu_dir,
        )
        for case in cases
    ]
    output_csv = args.output_csv or args.root / f"{args.base_experiment}_summary.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print(
        "case\tmedian mean TBT(ms)\tmedian P95 TBT(ms)\t"
        "throughput(tok/s)\tCPU(s)\thost-order violations/samples"
    )
    for row in summaries:
        cpu_s = float(row["cpu_user_s"]) + float(row["cpu_system_s"])
        print(
            f"{row['experiment']}\t"
            f"{float(row['request_median_mean_tbt_ms']):.3f}\t"
            f"{float(row['request_median_p95_tbt_ms']):.3f}\t"
            f"{float(row['throughput_tok_s']):.3f}\t{cpu_s:.3f}\t"
            f"{row['host_order_violations']}/{row['host_order_samples']}"
        )
    print(f"[DONE] {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
