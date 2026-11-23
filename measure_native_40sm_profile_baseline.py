#!/usr/bin/env python3
"""Measure a no-contention native-vLLM profile baseline.

Requests are deliberately issued synchronously.  A trial returns only after
the streaming completion has produced its final token and the OpenAI ``[DONE]``
marker, and the next trial is not issued until the configured cooldown ends.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from measure_real_workload_motivation import (
    PreparedRequest,
    ReplayEvent,
    fmt,
    get_model_id_from_server,
    get_requests_module,
    load_prepared_jsonl,
    percentile,
    run_one_event,
)


RUN_FIELDNAMES = [
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
]

TOKEN_FIELDNAMES = [
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

SUMMARY_FIELDNAMES = [
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
]


@dataclass(frozen=True)
class Trial:
    bucket_index: int
    request: PreparedRequest
    phase: str
    repeat_index: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames, extrasaction="ignore"
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def mean_or_none(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def stdev_or_none(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def numeric_values(
    rows: Sequence[dict[str, Any]], key: str
) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if row.get(key) not in ("", None)
    ]


def build_trial_plan(
    requests: Sequence[PreparedRequest],
    warmup_count: int,
    repeat_count: int,
) -> list[Trial]:
    """Place every bucket's unmeasured warmups immediately before its trials."""

    trials: list[Trial] = []
    for bucket_index, request in enumerate(requests):
        trials.extend(
            Trial(bucket_index, request, "warmup", index)
            for index in range(1, warmup_count + 1)
        )
        trials.extend(
            Trial(bucket_index, request, "measurement", index)
            for index in range(1, repeat_count + 1)
        )
    return trials


def summarize_bucket(
    *,
    case_id: str,
    bucket_index: int,
    target_tokens: int,
    actual_input_tokens: int,
    warmup_count: int,
    measured_repeat_count: int,
    request_max_tokens: int,
    request_min_tokens: int,
    cooldown_sec: float,
    run_rows: Sequence[dict[str, Any]],
    pooled_token_gaps: Sequence[float],
) -> dict[str, Any]:
    warmup_rows = [
        row for row in run_rows if row.get("phase") == "warmup"
    ]
    measured_rows = [
        row for row in run_rows if row.get("phase") == "measurement"
    ]
    successful = [
        row for row in measured_rows if row.get("status") == "success"
    ]
    ttft = numeric_values(successful, "ttft_ms")
    request_tbt = numeric_values(successful, "mean_tbt_ms")
    completion = numeric_values(successful, "completion_latency_ms")
    output_tokens = numeric_values(successful, "output_tokens")
    output_rates = [
        float(row["output_tokens"])
        / (float(row["completion_latency_ms"]) / 1000.0)
        for row in successful
        if row.get("output_tokens") not in ("", None)
        and row.get("completion_latency_ms") not in ("", None)
        and float(row["completion_latency_ms"]) > 0
    ]
    return {
        "timestamp_utc": utc_now(),
        "case_id": case_id,
        "bucket_index": bucket_index,
        "context_target_tokens": target_tokens,
        "actual_input_tokens": actual_input_tokens,
        "warmup_count": warmup_count,
        "warmup_successes": sum(
            row.get("status") == "success" for row in warmup_rows
        ),
        "measured_repeat_count": measured_repeat_count,
        "measurement_successes": len(successful),
        "measurement_failures": len(measured_rows) - len(successful),
        "request_max_tokens": request_max_tokens,
        "request_min_tokens": request_min_tokens,
        "cooldown_sec": cooldown_sec,
        "mean_ttft_ms": fmt(mean_or_none(ttft)),
        "stdev_ttft_ms": fmt(stdev_or_none(ttft)),
        "mean_request_tbt_ms": fmt(mean_or_none(request_tbt)),
        "stdev_request_tbt_ms": fmt(stdev_or_none(request_tbt)),
        "pooled_p50_tbt_ms": fmt(percentile(pooled_token_gaps, 50)),
        "pooled_p90_tbt_ms": fmt(percentile(pooled_token_gaps, 90)),
        "pooled_p99_tbt_ms": fmt(percentile(pooled_token_gaps, 99)),
        "mean_completion_latency_ms": fmt(mean_or_none(completion)),
        "stdev_completion_latency_ms": fmt(stdev_or_none(completion)),
        "mean_output_tokens": fmt(mean_or_none(output_tokens)),
        "mean_output_tokens_per_sec": fmt(mean_or_none(output_rates)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential warmup and repeated native-vLLM profile baseline"
        )
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--token-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument(
        "--expected-input-tokens", type=int, nargs="+", required=True
    )
    parser.add_argument("--repeat-count", type=int, default=5)
    parser.add_argument("--warmup-count-per-bucket", type=int, default=1)
    parser.add_argument("--cooldown-sec", type=float, default=10.0)
    parser.add_argument("--request-max-tokens", type=int, default=1024)
    parser.add_argument("--request-min-tokens", type=int, default=1024)
    parser.add_argument("--request-timeout-sec", type=int, default=14400)
    parser.add_argument("--ignore-eos", type=int, choices=[0, 1], default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.repeat_count <= 0:
        raise ValueError("--repeat-count must be positive")
    if args.warmup_count_per_bucket < 0:
        raise ValueError("--warmup-count-per-bucket must be non-negative")
    if args.cooldown_sec < 0:
        raise ValueError("--cooldown-sec must be non-negative")
    if args.request_max_tokens <= 0:
        raise ValueError("--request-max-tokens must be positive")
    if not 0 <= args.request_min_tokens <= args.request_max_tokens:
        raise ValueError(
            "--request-min-tokens must be between zero and max tokens"
        )
    for path in (args.raw_csv, args.token_csv, args.summary_csv):
        if path.exists():
            raise FileExistsError(
                f"refusing to append to an existing result: {path}"
            )


def main() -> int:
    args = parse_args()
    validate_args(args)
    requests = load_prepared_jsonl(args.requests_jsonl)
    actual_tokens = [request.input_tokens for request in requests]
    if actual_tokens != args.expected_input_tokens:
        raise ValueError(
            "prepared input-token sequence mismatch: "
            f"expected={args.expected_input_tokens}, actual={actual_tokens}"
        )

    trials = build_trial_plan(
        requests,
        args.warmup_count_per_bucket,
        args.repeat_count,
    )
    if args.dry_run:
        for trial in trials:
            print(
                "[DRY-RUN] "
                f"bucket={trial.bucket_index} "
                f"tokens={trial.request.input_tokens} "
                f"phase={trial.phase} repeat={trial.repeat_index}"
            )
        print(
            f"[DRY-RUN] total_requests={len(trials)} "
            f"cooldown_sec={args.cooldown_sec}"
        )
        return 0

    base_url = args.base_url.rstrip("/")
    http = get_requests_module()
    with http.Session() as session:
        model = args.model or get_model_id_from_server(session, base_url)

    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "status": "running",
        "case_id": args.case_id,
        "base_url": base_url,
        "model": model,
        "requests_jsonl": str(args.requests_jsonl),
        "prompt_sha256_by_context": {
            str(request.input_tokens): hashlib.sha256(
                request.prompt.encode("utf-8")
            ).hexdigest()
            for request in requests
        },
        "expected_input_tokens": args.expected_input_tokens,
        "repeat_count": args.repeat_count,
        "warmup_count_per_bucket": args.warmup_count_per_bucket,
        "cooldown_sec": args.cooldown_sec,
        "request_max_tokens": args.request_max_tokens,
        "request_min_tokens": args.request_min_tokens,
        "request_timeout_sec": args.request_timeout_sec,
        "ignore_eos": bool(args.ignore_eos),
        "submission_mode": "strict_synchronous_closed_loop",
        "completion_boundary": "stream_tokens_then_openai_done_marker",
        "tbt_token_count_source": "choice.token_ids",
        "tbt_requires_return_token_ids": True,
        "trials": [
            {
                "bucket_index": trial.bucket_index,
                "input_tokens": trial.request.input_tokens,
                "phase": trial.phase,
                "repeat_index": trial.repeat_index,
                "source_id": trial.request.source_id,
            }
            for trial in trials
        ],
    }
    args.metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    bucket_rows: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(len(requests))
    }
    bucket_token_gaps: dict[int, list[float]] = {
        index: [] for index in range(len(requests))
    }
    failed = False

    for plan_index, trial in enumerate(trials):
        target_tokens = args.expected_input_tokens[trial.bucket_index]
        trial_case_id = (
            f"{args.case_id}_ctx{target_tokens}_{trial.phase}"
        )
        event = ReplayEvent(
            role="foreground",
            workload=trial.request.workload or "sharegpt",
            request=trial.request,
            request_index=trial.bucket_index,
            offset_s=0.0,
            measurement_window=trial.phase == "measurement",
            tenant_id=f"baseline_tenant_ctx{target_tokens}",
        )
        print(
            "[RUN] "
            f"{plan_index + 1}/{len(trials)} "
            f"ctx={target_tokens} phase={trial.phase} "
            f"repeat={trial.repeat_index}",
            flush=True,
        )
        run_start = time.monotonic()
        row, token_rows, token_gaps = run_one_event(
            base_url=base_url,
            model=model,
            case_id=trial_case_id,
            repeat_index=trial.repeat_index,
            event=event,
            run_start_ts=run_start,
            foreground_max_tokens=args.request_max_tokens,
            foreground_min_tokens=args.request_min_tokens,
            background_max_tokens=args.request_max_tokens,
            background_min_tokens=args.request_min_tokens,
            ignore_eos=bool(args.ignore_eos),
            request_timeout_sec=args.request_timeout_sec,
        )
        wall_elapsed_ms = (time.monotonic() - run_start) * 1000.0
        has_next = plan_index + 1 < len(trials)
        cooldown_after = args.cooldown_sec if has_next else 0.0
        row.update(
            {
                "case_id": args.case_id,
                "phase": trial.phase,
                "bucket_index": trial.bucket_index,
                "context_target_tokens": target_tokens,
                "repeat_index": trial.repeat_index,
                "workload": event.workload,
                "source_id": trial.request.source_id,
                "input_tokens": trial.request.input_tokens,
                "mean_tbt_ms": fmt(mean_or_none(token_gaps)),
                "wall_elapsed_ms": fmt(wall_elapsed_ms),
                "cooldown_after_sec": cooldown_after,
            }
        )
        for token_row in token_rows:
            token_row.update(
                {
                    "case_id": args.case_id,
                    "phase": trial.phase,
                    "bucket_index": trial.bucket_index,
                    "context_target_tokens": target_tokens,
                    "repeat_index": trial.repeat_index,
                }
            )
        write_csv(args.raw_csv, RUN_FIELDNAMES, [row])
        if token_rows:
            write_csv(args.token_csv, TOKEN_FIELDNAMES, token_rows)
        bucket_rows[trial.bucket_index].append(row)
        if trial.phase == "measurement" and row["status"] == "success":
            bucket_token_gaps[trial.bucket_index].extend(token_gaps)

        print(
            "[RESULT] "
            f"ctx={target_tokens} phase={trial.phase} "
            f"repeat={trial.repeat_index} status={row['status']} "
            f"ttft_ms={row.get('ttft_ms', '')} "
            f"mean_tbt_ms={row.get('mean_tbt_ms', '')} "
            f"completion_ms={row.get('completion_latency_ms', '')}",
            flush=True,
        )
        if row["status"] != "success":
            failed = True
            print(f"[ERROR] {row['error_message']}", flush=True)
            if args.fail_fast:
                metadata["status"] = "failed"
                metadata["failed_trial"] = {
                    "bucket_index": trial.bucket_index,
                    "phase": trial.phase,
                    "repeat_index": trial.repeat_index,
                    "error_message": row["error_message"],
                }
                metadata["completed_at_utc"] = utc_now()
                args.metadata_json.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                return 1
        if cooldown_after > 0:
            print(
                f"[COOLDOWN] sleeping {cooldown_after:g}s before next request",
                flush=True,
            )
            time.sleep(cooldown_after)

    summaries = [
        summarize_bucket(
            case_id=args.case_id,
            bucket_index=bucket_index,
            target_tokens=args.expected_input_tokens[bucket_index],
            actual_input_tokens=request.input_tokens,
            warmup_count=args.warmup_count_per_bucket,
            measured_repeat_count=args.repeat_count,
            request_max_tokens=args.request_max_tokens,
            request_min_tokens=args.request_min_tokens,
            cooldown_sec=args.cooldown_sec,
            run_rows=bucket_rows[bucket_index],
            pooled_token_gaps=bucket_token_gaps[bucket_index],
        )
        for bucket_index, request in enumerate(requests)
    ]
    write_csv(args.summary_csv, SUMMARY_FIELDNAMES, summaries)
    metadata["status"] = "failed" if failed else "completed"
    metadata["completed_at_utc"] = utc_now()
    metadata["summary_csv"] = str(args.summary_csv)
    args.metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for summary in summaries:
        print(
            "[SUMMARY] "
            f"ctx={summary['context_target_tokens']} "
            f"success={summary['measurement_successes']}/"
            f"{summary['measured_repeat_count']} "
            f"mean_ttft_ms={summary['mean_ttft_ms']} "
            f"mean_tbt_ms={summary['mean_request_tbt_ms']} "
            f"p99_tbt_ms={summary['pooled_p99_tbt_ms']}",
            flush=True,
        )
    print(f"[DONE] raw_csv={args.raw_csv}")
    print(f"[DONE] token_csv={args.token_csv}")
    print(f"[DONE] summary_csv={args.summary_csv}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
