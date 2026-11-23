#!/usr/bin/env python3
"""Sequential long-context request replay for P/D VRAM-contention runs.

Unlike measure_pd_vram_contention.py, this driver does not create a special
decode-vs-prefill pair. It submits N long-context streaming requests at fixed
offsets. Every request performs both prefill and decode, so the resulting
timeline can show how resident decode KV from earlier requests interacts with
prefill/decode progress from later requests.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from measure_pd_vram_contention import fit_request_to_token_window
from measure_real_workload_motivation import (
    RAW_FIELDNAMES,
    TOKEN_FIELDNAMES,
    PreparedRequest,
    ReplayEvent,
    fmt,
    get_model_id_from_server,
    get_requests_module,
    load_prepared_jsonl,
    percentile,
    run_one_event,
)


SUMMARY_FIELDNAMES = [
    "timestamp_utc",
    "case_id",
    "dataset_kind",
    "repeat_index",
    "context_target_tokens",
    "request_count",
    "request_gap_sec",
    "request_start_index",
    "request_input_tokens_min",
    "request_input_tokens_max",
    "request_input_tokens_total",
    "request_successes",
    "request_failures",
    "request_output_tokens_total",
    "request_p50_tbt_ms",
    "request_p90_tbt_ms",
    "request_p99_tbt_ms",
    "request_p99_ttft_ms",
    "request_p99_completion_latency_ms",
    "requested_max_tokens",
    "requested_min_tokens",
    "first_request_offset_s",
    "last_request_offset_s",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, fieldnames: Sequence[str],
              rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-offset N-request replay for P/D VRAM contention.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--dataset-kind", default="prepared")
    parser.add_argument("--repeat-index", type=int, default=1)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--token-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--context-target-tokens", type=int, default=0)
    parser.add_argument("--context-token-window", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--request-count", type=int, default=5)
    parser.add_argument("--request-start-index", type=int, default=0)
    parser.add_argument("--request-gap-sec", type=float, default=1.0)
    parser.add_argument("--request-max-tokens", type=int, default=512)
    parser.add_argument("--request-min-tokens", type=int, default=512)
    parser.add_argument("--request-timeout-sec", type=int, default=3600)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--ignore-eos", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def fit_requests(
    *,
    base_url: str,
    model: str,
    requests: List[PreparedRequest],
    request_count: int,
    request_start_index: int,
    context_target_tokens: int,
    context_token_window: int,
    max_model_len: int,
    request_max_tokens: int,
) -> List[PreparedRequest]:
    selected: List[PreparedRequest] = []
    http = get_requests_module()
    with http.Session() as session:
        for local_index in range(request_count):
            dataset_index = request_start_index + local_index
            request = requests[dataset_index % len(requests)]
            selected.append(
                fit_request_to_token_window(
                    session=session,
                    base_url=base_url,
                    model=model,
                    request=request,
                    target_tokens=context_target_tokens,
                    context_window=context_token_window,
                    max_model_len=max_model_len,
                    max_output_tokens=request_max_tokens,
                    label=f"request_{local_index}",
                )
            )
    return selected


def build_events(
    *,
    requests: List[PreparedRequest],
    request_gap_sec: float,
) -> List[ReplayEvent]:
    events: List[ReplayEvent] = []
    for index, request in enumerate(requests):
        events.append(
            ReplayEvent(
                role="foreground",
                workload=request.workload or "loogle_sequence",
                request=request,
                request_index=index,
                offset_s=max(0.0, request_gap_sec) * index,
                measurement_window=True,
                tenant_id=f"seq_tenant_{index}",
            )
        )
    return events


def summarize(
    *,
    args: argparse.Namespace,
    raw_rows: List[Dict[str, Any]],
    token_gaps: List[float],
    fitted_requests: List[PreparedRequest],
) -> Dict[str, Any]:
    success_rows = [row for row in raw_rows if row.get("status") == "success"]
    failed_rows = [row for row in raw_rows if row.get("status") != "success"]
    ttft_values = [
        float(row["ttft_ms"])
        for row in success_rows
        if row.get("ttft_ms") not in ("", None)
    ]
    latency_values = [
        float(row["completion_latency_ms"])
        for row in success_rows
        if row.get("completion_latency_ms") not in ("", None)
    ]
    output_tokens = sum(
        int(float(row["output_tokens"]))
        for row in success_rows
        if row.get("output_tokens") not in ("", None)
    )
    input_tokens = [request.input_tokens for request in fitted_requests]
    return {
        "timestamp_utc": utc_now(),
        "case_id": args.case_id,
        "dataset_kind": args.dataset_kind,
        "repeat_index": args.repeat_index,
        "context_target_tokens": args.context_target_tokens,
        "request_count": args.request_count,
        "request_gap_sec": args.request_gap_sec,
        "request_start_index": args.request_start_index,
        "request_input_tokens_min": min(input_tokens) if input_tokens else "",
        "request_input_tokens_max": max(input_tokens) if input_tokens else "",
        "request_input_tokens_total": sum(input_tokens),
        "request_successes": len(success_rows),
        "request_failures": len(failed_rows),
        "request_output_tokens_total": output_tokens,
        "request_p50_tbt_ms": fmt(percentile(token_gaps, 50)),
        "request_p90_tbt_ms": fmt(percentile(token_gaps, 90)),
        "request_p99_tbt_ms": fmt(percentile(token_gaps, 99)),
        "request_p99_ttft_ms": fmt(percentile(ttft_values, 99)),
        "request_p99_completion_latency_ms": fmt(
            percentile(latency_values, 99)),
        "requested_max_tokens": args.request_max_tokens,
        "requested_min_tokens": args.request_min_tokens,
        "first_request_offset_s": 0.0,
        "last_request_offset_s": max(0.0, args.request_gap_sec) *
        max(0, args.request_count - 1),
    }


def main() -> int:
    args = parse_args()
    if args.request_count <= 0:
        raise ValueError("--request-count must be positive")

    base_url = args.base_url.rstrip("/")
    http = get_requests_module()
    with http.Session() as session:
        model = args.model or get_model_id_from_server(session, base_url)

    prepared = load_prepared_jsonl(args.requests_jsonl)
    fitted = fit_requests(
        base_url=base_url,
        model=model,
        requests=prepared,
        request_count=args.request_count,
        request_start_index=args.request_start_index,
        context_target_tokens=args.context_target_tokens,
        context_token_window=args.context_token_window,
        max_model_len=args.max_model_len,
        request_max_tokens=args.request_max_tokens,
    )
    events = build_events(requests=fitted, request_gap_sec=args.request_gap_sec)

    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(
        json.dumps(
            {
                "created_at": utc_now(),
                "base_url": base_url,
                "model": model,
                "case_id": args.case_id,
                "dataset_kind": args.dataset_kind,
                "repeat_index": args.repeat_index,
                "requests_jsonl": str(args.requests_jsonl),
                "context_target_tokens": args.context_target_tokens,
                "context_token_window": args.context_token_window,
                "request_count": args.request_count,
                "request_start_index": args.request_start_index,
                "request_gap_sec": args.request_gap_sec,
                "request_max_tokens": args.request_max_tokens,
                "request_min_tokens": args.request_min_tokens,
                "events": [
                    {
                        "request_index": event.request_index,
                        "offset_s": event.offset_s,
                        "tenant_id": event.tenant_id,
                        "input_tokens": event.request.input_tokens,
                        "source_id": event.request.source_id,
                    }
                    for event in events
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    run_start_ts = time.monotonic() + 2.0
    raw_rows: List[Dict[str, Any]] = []
    all_token_rows: List[Dict[str, Any]] = []
    all_token_gaps: List[float] = []
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        for event in events:
            target_ts = run_start_ts + event.offset_s
            delay = target_ts - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            futures.append(
                executor.submit(
                    run_one_event,
                    base_url=base_url,
                    model=model,
                    case_id=args.case_id,
                    repeat_index=args.repeat_index,
                    event=event,
                    run_start_ts=run_start_ts,
                    foreground_max_tokens=args.request_max_tokens,
                    foreground_min_tokens=args.request_min_tokens,
                    background_max_tokens=args.request_max_tokens,
                    background_min_tokens=args.request_min_tokens,
                    ignore_eos=bool(args.ignore_eos),
                    request_timeout_sec=args.request_timeout_sec,
                )
            )

        for future in as_completed(futures):
            row, token_rows, token_gaps = future.result()
            row["background_workload"] = "none"
            for token_row in token_rows:
                token_row["case_id"] = args.case_id
                token_row["repeat_index"] = args.repeat_index
            raw_rows.append(row)
            all_token_rows.extend(token_rows)
            if row.get("status") == "success":
                all_token_gaps.extend(token_gaps)

    raw_rows.sort(
        key=lambda row: (
            float(row["scheduled_offset_s"]),
            int(row["request_index"]),
        )
    )
    all_token_rows.sort(
        key=lambda row: (
            str(row["request_id"]),
            int(row["chunk_index"]),
        )
    )
    write_csv(args.raw_csv, RAW_FIELDNAMES, raw_rows)
    write_csv(args.token_csv, TOKEN_FIELDNAMES, all_token_rows)
    summary = summarize(
        args=args,
        raw_rows=raw_rows,
        token_gaps=all_token_gaps,
        fitted_requests=fitted,
    )
    write_csv(args.summary_csv, SUMMARY_FIELDNAMES, [summary])

    print(
        "[SUMMARY] "
        f"case={args.case_id} repeat={args.repeat_index} "
        f"request_successes={summary['request_successes']} "
        f"request_failures={summary['request_failures']} "
        f"request_p99_tbt_ms={summary['request_p99_tbt_ms']}",
        flush=True,
    )
    print(f"[DONE] raw_csv={args.raw_csv}")
    print(f"[DONE] token_csv={args.token_csv}")
    print(f"[DONE] summary_csv={args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
