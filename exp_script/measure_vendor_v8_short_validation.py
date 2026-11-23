#!/usr/bin/env python3
"""Run a compact vendor_v8 SLO scheduling validation workload.

Legacy default sequence:

* submit one 100K request;
* while it is incomplete, inject two 10K requests at configured offsets;
* submit the next 100K immediately after the previous 100K completes;
* repeat for three long requests.

The six short requests may remain active when the next long request starts.
This preserves the requested long-request completion gate while retaining
cross-cycle pressure from delayed short requests.

The optional named workload matrix keeps the request population fixed while
changing only open-loop arrival order:

* ``A`` submits every long at the workload origin and injects shorts later;
* ``B`` submits every short at the origin and injects longs periodically;
* ``C`` interleaves ``S1,L1,S2,L2,S3,L3,S4,S5`` at a fixed interval.

The legacy ``completion_gated`` and ``concurrent_longs`` modes remain
available so existing commands keep their original behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from measure_real_workload_motivation import (
    PreparedRequest,
    get_model_id_from_server,
    get_requests_module,
    load_prepared_jsonl,
    percentile,
)
from measure_vendor_v8_partition_profile import (
    StreamResult,
    load_slo_policy,
    lookup_request_slo,
    stream_request,
)


PLAN_FIELDNAMES = [
    "cycle_index",
    "role",
    "workload",
    "request_index",
    "context_tokens",
    "planned_offset_from_long_submit_s",
    "planned_offset_from_workload_start_s",
    "actual_submit_monotonic_ts",
    "long_completion_gate_monotonic_ts",
    "request_id",
]

RAW_FIELDNAMES = [
    "cycle_index",
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

TOKEN_FIELDNAMES = [
    "cycle_index",
    "role",
    "request_index",
    "request_id",
    "context_tokens",
    "chunk_index",
    "chunk_tokens",
    "gap_start_monotonic_ts",
    "arrival_monotonic_ts",
    "estimated_per_token_gap_ms",
    "tbt_slo_ms",
    "tbt_slo_met",
]

SUMMARY_FIELDNAMES = [
    "role",
    "context_tokens",
    "request_count",
    "success_count",
    "failure_count",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "request_p99_tbt_p50_ms",
    "request_p99_tbt_p95_ms",
    "ttft_slo_attainment",
    "tbt_token_slo_attainment",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ValidationPlanItem:
    cycle_index: int
    role: str
    request_index: int
    context_tokens: int
    offset_from_long_submit_s: float
    workload: str = "legacy"
    offset_from_workload_start_s: float | None = None


def effective_submission_mode(*, workload: str, submission_mode: str) -> str:
    normalized = workload.upper()
    if normalized == "LEGACY":
        return submission_mode
    modes = {
        "A": "concurrent_longs",
        "B": "short_burst_periodic_longs",
        "C": "interleaved_open_loop",
    }
    try:
        return modes[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported workload: {workload}") from exc


@dataclass
class SubmittedRequest:
    plan: ValidationPlanItem
    request_id: str
    request: PreparedRequest
    slo: dict[str, float]
    future: Future[StreamResult]


def build_validation_plan(
    *,
    workload: str = "legacy",
    submission_mode: str = "completion_gated",
    long_count: int,
    short_per_long: int,
    short_count: int | None = None,
    long_context_tokens: int,
    short_context_tokens: int,
    short_offsets_sec: Sequence[float],
    long_offsets_sec: Sequence[float] = (),
    interleaved_interval_sec: float = 300.0,
) -> list[ValidationPlanItem]:
    if long_count <= 0:
        raise ValueError("long request count must be positive")
    normalized_workload = workload.upper()
    if normalized_workload not in {"LEGACY", "A", "B", "C"}:
        raise ValueError(f"unsupported workload: {workload}")
    if submission_mode not in {"completion_gated", "concurrent_longs"}:
        raise ValueError(f"unsupported submission mode: {submission_mode}")
    effective_mode = effective_submission_mode(
        workload=normalized_workload,
        submission_mode=submission_mode,
    )
    if (
        effective_mode in {"completion_gated", "concurrent_longs"}
        and not short_offsets_sec
    ):
        raise ValueError("at least one short request offset is required")
    if effective_mode == "completion_gated" and short_per_long <= 0:
        raise ValueError("short requests per long must be positive")
    if (
        effective_mode == "completion_gated"
        and len(short_offsets_sec) != short_per_long
    ):
        raise ValueError(
            "short offset count must equal short requests per long"
        )
    if short_offsets_sec and any(value <= 0 for value in short_offsets_sec):
        raise ValueError("short injection offsets must be positive")
    if list(short_offsets_sec) != sorted(short_offsets_sec):
        raise ValueError("short injection offsets must be sorted")
    if short_count is None:
        short_count = (
            long_count * short_per_long
            if effective_mode == "completion_gated"
            else len(short_offsets_sec)
        )
    if short_count <= 0:
        raise ValueError("short request count must be positive")
    plan: list[ValidationPlanItem] = []
    if effective_mode == "concurrent_longs":
        if len(short_offsets_sec) != short_count:
            raise ValueError(
                "workload A/concurrent_longs requires one offset per short"
            )
        for request_index in range(long_count):
            plan.append(
                ValidationPlanItem(
                    cycle_index=request_index,
                    role="long",
                    request_index=request_index,
                    context_tokens=long_context_tokens,
                    offset_from_long_submit_s=0.0,
                    workload=(
                        "A" if normalized_workload == "A" else "legacy"
                    ),
                    offset_from_workload_start_s=0.0,
                )
            )
        for request_index, offset in enumerate(short_offsets_sec):
            plan.append(
                ValidationPlanItem(
                    cycle_index=-1,
                    role="short",
                    request_index=request_index,
                    context_tokens=short_context_tokens,
                    offset_from_long_submit_s=float(offset),
                    workload=(
                        "A" if normalized_workload == "A" else "legacy"
                    ),
                    offset_from_workload_start_s=float(offset),
                )
            )
        return plan

    if effective_mode == "short_burst_periodic_longs":
        if len(long_offsets_sec) != long_count:
            raise ValueError(
                "workload B requires one periodic offset per long request"
            )
        if any(value <= 0 for value in long_offsets_sec):
            raise ValueError("long injection offsets must be positive")
        if list(long_offsets_sec) != sorted(long_offsets_sec):
            raise ValueError("long injection offsets must be sorted")
        for request_index in range(short_count):
            plan.append(
                ValidationPlanItem(
                    cycle_index=-1,
                    role="short",
                    request_index=request_index,
                    context_tokens=short_context_tokens,
                    offset_from_long_submit_s=0.0,
                    workload="B",
                    offset_from_workload_start_s=0.0,
                )
            )
        for request_index, offset in enumerate(long_offsets_sec):
            plan.append(
                ValidationPlanItem(
                    cycle_index=-1,
                    role="long",
                    request_index=request_index,
                    context_tokens=long_context_tokens,
                    offset_from_long_submit_s=float(offset),
                    workload="B",
                    offset_from_workload_start_s=float(offset),
                )
            )
        return plan

    if effective_mode == "interleaved_open_loop":
        if interleaved_interval_sec <= 0:
            raise ValueError("interleaved interval must be positive")
        long_index = 0
        short_index = 0
        event_index = 0
        while long_index < long_count or short_index < short_count:
            if short_index < short_count:
                offset = event_index * interleaved_interval_sec
                plan.append(
                    ValidationPlanItem(
                        cycle_index=-1,
                        role="short",
                        request_index=short_index,
                        context_tokens=short_context_tokens,
                        offset_from_long_submit_s=float(offset),
                        workload="C",
                        offset_from_workload_start_s=float(offset),
                    )
                )
                short_index += 1
                event_index += 1
            if long_index < long_count:
                offset = event_index * interleaved_interval_sec
                plan.append(
                    ValidationPlanItem(
                        cycle_index=-1,
                        role="long",
                        request_index=long_index,
                        context_tokens=long_context_tokens,
                        offset_from_long_submit_s=float(offset),
                        workload="C",
                        offset_from_workload_start_s=float(offset),
                    )
                )
                long_index += 1
                event_index += 1
        return plan

    long_request_index = 0
    short_request_index = 0
    for cycle in range(long_count):
        plan.append(
            ValidationPlanItem(
                cycle_index=cycle,
                role="long",
                request_index=long_request_index,
                context_tokens=long_context_tokens,
                offset_from_long_submit_s=0.0,
                workload="legacy",
            )
        )
        long_request_index += 1
        for offset in short_offsets_sec:
            plan.append(
                ValidationPlanItem(
                    cycle_index=cycle,
                    role="short",
                    request_index=short_request_index,
                    context_tokens=short_context_tokens,
                    offset_from_long_submit_s=float(offset),
                    workload="legacy",
                )
            )
            short_request_index += 1
    return plan


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def wait_until_offset_or_fail(
    *,
    target_ts: float,
    long_future: Future[StreamResult],
) -> None:
    while True:
        if long_future.done():
            result = long_future.result()
            raise RuntimeError(
                "100K request completed before its short-request injection "
                f"window; request_id={result.row['request_id']}"
            )
        remaining = target_ts - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.1, remaining))


def wait_until_global_offset_or_fail(
    *,
    target_ts: float,
    long_futures: Sequence[Future[StreamResult]],
) -> None:
    """Wait for a global injection offset while at least one long is active."""

    while True:
        if all(future.done() for future in long_futures):
            raise RuntimeError(
                "all 100K requests completed before a short-request "
                "injection window"
            )
        remaining = target_ts - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.1, remaining))


def wait_until_global_offset(*, target_ts: float) -> None:
    """Wait for an open-loop event without coupling it to request completion."""

    while True:
        remaining = target_ts - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.1, remaining))


def request_summary(
    rows: Sequence[dict[str, Any]],
    token_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    keys = sorted({(row["role"], int(row["context_tokens"])) for row in rows})
    for role, context in keys:
        selected = [
            row
            for row in rows
            if row["role"] == role
            and int(row["context_tokens"]) == context
        ]
        successful = [row for row in selected if row["status"] == "success"]
        ttft = [float(row["ttft_ms"]) for row in successful]
        p99 = [
            float(row["p99_tbt_ms"])
            for row in successful
            if row.get("p99_tbt_ms") not in ("", None)
        ]
        selected_tokens = [
            row
            for row in token_rows
            if row["role"] == role
            and int(row["context_tokens"]) == context
            and row.get("estimated_per_token_gap_ms") not in ("", None)
        ]
        summaries.append(
            {
                "role": role,
                "context_tokens": context,
                "request_count": len(selected),
                "success_count": len(successful),
                "failure_count": len(selected) - len(successful),
                "ttft_p50_ms": percentile(ttft, 50) or "",
                "ttft_p95_ms": percentile(ttft, 95) or "",
                "request_p99_tbt_p50_ms": percentile(p99, 50) or "",
                "request_p99_tbt_p95_ms": percentile(p99, 95) or "",
                "ttft_slo_attainment": (
                    statistics.fmean(
                        float(row["ttft_slo_met"]) for row in successful
                    )
                    if successful
                    else ""
                ),
                "tbt_token_slo_attainment": (
                    statistics.fmean(
                        float(row["tbt_slo_met"])
                        for row in selected_tokens
                    )
                    if selected_tokens
                    else ""
                ),
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the vendor_v8 3-long/6-short validation workload"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--slo-policy-json", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--submission-mode",
        choices=["completion_gated", "concurrent_longs"],
        default="completion_gated",
    )
    parser.add_argument(
        "--workload",
        choices=["legacy", "A", "B", "C"],
        default="legacy",
        help=(
            "named open-loop arrival pattern; legacy preserves "
            "--submission-mode"
        ),
    )
    parser.add_argument("--long-context-tokens", type=int, default=100_000)
    parser.add_argument("--short-context-tokens", type=int, default=10_000)
    parser.add_argument("--long-count", type=int, default=3)
    parser.add_argument("--short-per-long", type=int, default=2)
    parser.add_argument("--short-count", type=int, default=None)
    parser.add_argument(
        "--short-injection-offsets-sec",
        type=float,
        nargs="+",
        default=[5.0, 10.0],
    )
    parser.add_argument(
        "--long-injection-offsets-sec",
        type=float,
        nargs="+",
        default=[300.0, 600.0, 900.0],
    )
    parser.add_argument(
        "--interleaved-injection-interval-sec",
        type=float,
        default=300.0,
    )
    parser.add_argument("--long-output-tokens", type=int, default=32)
    parser.add_argument("--short-output-tokens", type=int, default=32)
    parser.add_argument("--warmup-count", type=int, default=0)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--warmup-cooldown-sec", type=float, default=3.0)
    parser.add_argument("--request-timeout-sec", type=int, default=14400)
    parser.add_argument("--ignore-eos", type=int, choices=[0, 1], default=1)
    parser.add_argument("--plan-csv", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--token-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    named_short_count = args.short_count
    if named_short_count is None and args.workload.upper() in {"B", "C"}:
        named_short_count = 5
    plan = build_validation_plan(
        workload=args.workload,
        submission_mode=args.submission_mode,
        long_count=args.long_count,
        short_per_long=args.short_per_long,
        short_count=named_short_count,
        long_context_tokens=args.long_context_tokens,
        short_context_tokens=args.short_context_tokens,
        short_offsets_sec=args.short_injection_offsets_sec,
        long_offsets_sec=args.long_injection_offsets_sec,
        interleaved_interval_sec=args.interleaved_injection_interval_sec,
    )
    effective_mode = effective_submission_mode(
        workload=args.workload,
        submission_mode=args.submission_mode,
    )
    if args.long_output_tokens <= 0 or args.short_output_tokens <= 0:
        raise ValueError("output token counts must be positive")
    if (
        args.warmup_count < 0
        or args.warmup_output_tokens <= 0
        or args.warmup_cooldown_sec < 0
    ):
        raise ValueError("invalid warmup configuration")
    for path in (
        args.plan_csv,
        args.raw_csv,
        args.token_csv,
        args.summary_csv,
        args.metadata_json,
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if args.dry_run:
        for item in plan:
            long_relative_offset: float | str = (
                item.offset_from_long_submit_s
                if effective_mode in {
                    "completion_gated",
                    "concurrent_longs",
                }
                else "n/a"
            )
            print(
                "[DRY-RUN] "
                f"workload={item.workload} role={item.role} "
                f"request={item.request_index + 1} "
                f"context={item.context_tokens} "
                "offset_from_workload_start_s="
                f"{item.offset_from_workload_start_s} "
                "offset_from_long_submit_s="
                f"{long_relative_offset}"
            )
        return 0

    requests = load_prepared_jsonl(args.requests_jsonl)
    long_requests = [
        request
        for request in requests
        if request.input_tokens == args.long_context_tokens
    ]
    short_requests = [
        request
        for request in requests
        if request.input_tokens == args.short_context_tokens
    ]
    if len(long_requests) != args.long_count:
        raise ValueError(
            f"expected {args.long_count} long requests, got "
            f"{len(long_requests)}"
        )
    expected_shorts = (
        args.long_count * args.short_per_long
        if effective_mode == "completion_gated"
        else (
            named_short_count
            if named_short_count is not None
            else len(args.short_injection_offsets_sec)
        )
    )
    if len(short_requests) != expected_shorts:
        raise ValueError(
            f"expected {expected_shorts} short requests, got "
            f"{len(short_requests)}"
        )
    slo_policy = load_slo_policy(args.slo_policy_json)
    http = get_requests_module()
    with http.Session() as session:
        model = args.model or get_model_id_from_server(
            session, args.base_url.rstrip("/")
        )

    warmup_slo = lookup_request_slo(slo_policy, args.long_context_tokens)
    for warmup_index in range(args.warmup_count):
        stream_request(
            base_url=args.base_url,
            model=model,
            request=long_requests[warmup_index % len(long_requests)],
            request_id=f"{args.case_id}-warmup-{warmup_index + 1}",
            role="warmup",
            tenant_id="validation_warmup",
            max_tokens=args.warmup_output_tokens,
            min_tokens=args.warmup_output_tokens,
            ignore_eos=bool(args.ignore_eos),
            request_timeout_sec=args.request_timeout_sec,
            xargs=warmup_slo,
        )
        if args.warmup_cooldown_sec > 0:
            time.sleep(args.warmup_cooldown_sec)

    metadata: dict[str, Any] = {
        "schema_version": 2,
        "created_at_utc": utc_now(),
        "status": "running",
        "case_id": args.case_id,
        "model": model,
        "workload": args.workload,
        "submission_mode": effective_mode,
        "short_injection_contract": (
            "per-long offsets while associated long is incomplete"
            if effective_mode == "completion_gated"
            else (
                "global offsets while at least one concurrent long is active"
                if effective_mode == "concurrent_longs"
                else (
                    "all shorts at origin; longs at periodic global offsets"
                    if effective_mode == "short_burst_periodic_longs"
                    else "S/L interleaving at fixed global intervals"
                )
            )
        ),
        "long_context_tokens": args.long_context_tokens,
        "short_context_tokens": args.short_context_tokens,
        "long_count": args.long_count,
        "short_per_long": args.short_per_long,
        "short_count": expected_shorts,
        "short_injection_offsets_sec": args.short_injection_offsets_sec,
        "long_injection_offsets_sec": args.long_injection_offsets_sec,
        "interleaved_injection_interval_sec": (
            args.interleaved_injection_interval_sec
        ),
        "planned_arrivals": [
            {
                "role": item.role,
                "request_index": item.request_index,
                "offset_from_workload_start_s": (
                    item.offset_from_workload_start_s
                ),
            }
            for item in plan
        ],
        "warmup_count": args.warmup_count,
        "warmup_output_tokens": args.warmup_output_tokens,
        "warmup_cooldown_sec": args.warmup_cooldown_sec,
        "slo_policy_json": str(args.slo_policy_json),
    }
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    submitted: list[SubmittedRequest] = []
    plan_rows: list[dict[str, Any]] = []
    workers = args.long_count + expected_shorts
    with ThreadPoolExecutor(max_workers=workers) as executor:
        if effective_mode != "completion_gated":
            workload_start_ts = time.monotonic()
            long_futures: list[Future[StreamResult]] = []
            for plan_item in plan:
                offset = plan_item.offset_from_workload_start_s
                if offset is None:
                    raise RuntimeError(
                        "open-loop plan item has no workload-origin offset"
                    )
                target_ts = workload_start_ts + offset
                if (
                    effective_mode == "concurrent_longs"
                    and plan_item.role == "short"
                ):
                    wait_until_global_offset_or_fail(
                        target_ts=target_ts,
                        long_futures=long_futures,
                    )
                else:
                    wait_until_global_offset(target_ts=target_ts)

                if plan_item.role == "long":
                    request = long_requests[plan_item.request_index]
                    context_tokens = args.long_context_tokens
                    output_tokens = args.long_output_tokens
                else:
                    request = short_requests[plan_item.request_index]
                    context_tokens = args.short_context_tokens
                    output_tokens = args.short_output_tokens
                slo = lookup_request_slo(slo_policy, context_tokens)
                request_id = (
                    f"{args.case_id}-{plan_item.role}-"
                    f"{plan_item.request_index + 1}"
                )
                submit_ts = time.monotonic()
                future = executor.submit(
                    stream_request,
                    base_url=args.base_url,
                    model=model,
                    request=request,
                    request_id=request_id,
                    role=plan_item.role,
                    tenant_id=(
                        f"validation_{plan_item.role}_"
                        f"{plan_item.request_index}"
                    ),
                    max_tokens=output_tokens,
                    min_tokens=output_tokens,
                    ignore_eos=bool(args.ignore_eos),
                    request_timeout_sec=args.request_timeout_sec,
                    xargs=slo,
                )
                if plan_item.role == "long":
                    long_futures.append(future)
                submitted.append(
                    SubmittedRequest(
                        plan=plan_item,
                        request_id=request_id,
                        request=request,
                        slo=slo,
                        future=future,
                    )
                )
                plan_rows.append(
                    {
                        **plan_item.__dict__,
                        "planned_offset_from_long_submit_s": (
                            plan_item.offset_from_long_submit_s
                            if effective_mode == "concurrent_longs"
                            else ""
                        ),
                        "planned_offset_from_workload_start_s": offset,
                        "actual_submit_monotonic_ts": submit_ts,
                        "long_completion_gate_monotonic_ts": "",
                        "request_id": request_id,
                    }
                )

        else:
            for cycle in range(args.long_count):
                long_item = next(
                    item
                    for item in plan
                    if item.cycle_index == cycle and item.role == "long"
                )
                long_request = long_requests[long_item.request_index]
                long_slo = lookup_request_slo(
                    slo_policy, args.long_context_tokens
                )
                long_request_id = f"{args.case_id}-long-{cycle + 1}"
                long_submit_ts = time.monotonic()
                long_future = executor.submit(
                    stream_request,
                    base_url=args.base_url,
                    model=model,
                    request=long_request,
                    request_id=long_request_id,
                    role="long",
                    tenant_id=f"validation_long_{cycle}",
                    max_tokens=args.long_output_tokens,
                    min_tokens=args.long_output_tokens,
                    ignore_eos=bool(args.ignore_eos),
                    request_timeout_sec=args.request_timeout_sec,
                    xargs=long_slo,
                )
                submitted.append(
                    SubmittedRequest(
                        plan=long_item,
                        request_id=long_request_id,
                        request=long_request,
                        slo=long_slo,
                        future=long_future,
                    )
                )
                plan_rows.append(
                    {
                        **long_item.__dict__,
                        "planned_offset_from_long_submit_s": 0.0,
                        "planned_offset_from_workload_start_s": "",
                        "actual_submit_monotonic_ts": long_submit_ts,
                        "long_completion_gate_monotonic_ts": "",
                        "request_id": long_request_id,
                    }
                )

                short_items = [
                    item
                    for item in plan
                    if item.cycle_index == cycle and item.role == "short"
                ]
                for short_item in short_items:
                    target_ts = (
                        long_submit_ts + short_item.offset_from_long_submit_s
                    )
                    wait_until_offset_or_fail(
                        target_ts=target_ts, long_future=long_future
                    )
                    short_request = short_requests[short_item.request_index]
                    short_slo = lookup_request_slo(
                        slo_policy, args.short_context_tokens
                    )
                    short_request_id = (
                        f"{args.case_id}-short-"
                        f"{short_item.request_index + 1}"
                    )
                    short_submit_ts = time.monotonic()
                    short_future = executor.submit(
                        stream_request,
                        base_url=args.base_url,
                        model=model,
                        request=short_request,
                        request_id=short_request_id,
                        role="short",
                        tenant_id=(
                            f"validation_short_{short_item.request_index}"
                        ),
                        max_tokens=args.short_output_tokens,
                        min_tokens=args.short_output_tokens,
                        ignore_eos=bool(args.ignore_eos),
                        request_timeout_sec=args.request_timeout_sec,
                        xargs=short_slo,
                    )
                    submitted.append(
                        SubmittedRequest(
                            plan=short_item,
                            request_id=short_request_id,
                            request=short_request,
                            slo=short_slo,
                            future=short_future,
                        )
                    )
                    plan_rows.append(
                        {
                            **short_item.__dict__,
                            "planned_offset_from_long_submit_s": (
                                short_item.offset_from_long_submit_s
                            ),
                            "planned_offset_from_workload_start_s": "",
                            "actual_submit_monotonic_ts": short_submit_ts,
                            "long_completion_gate_monotonic_ts": "",
                            "request_id": short_request_id,
                        }
                    )

                long_result = long_future.result()
                gate_ts = float(long_result.row["completion_monotonic_ts"])
                for row in plan_rows:
                    if int(row["cycle_index"]) == cycle:
                        row["long_completion_gate_monotonic_ts"] = gate_ts
                print(
                    "[GATE] "
                    f"long={cycle + 1}/{args.long_count} completed; "
                    "next long may be submitted",
                    flush=True,
                )

        raw_rows: list[dict[str, Any]] = []
        token_rows: list[dict[str, Any]] = []
        for item in submitted:
            common = {
                "cycle_index": item.plan.cycle_index,
                "role": item.plan.role,
                "request_index": item.plan.request_index,
                "request_id": item.request_id,
                "context_tokens": item.request.input_tokens,
                "ttft_slo_ms": item.slo["ttft_prefill_deadline_ms"],
                "tbt_slo_ms": item.slo["tbt_slo_ms"],
            }
            try:
                result = item.future.result()
                row = {**common, **result.row}
                row["ttft_slo_met"] = int(
                    float(row["ttft_ms"]) <= float(row["ttft_slo_ms"])
                )
                gaps = [
                    float(token["estimated_per_token_gap_ms"])
                    for token in result.token_rows
                    if token.get("estimated_per_token_gap_ms")
                    not in ("", None)
                ]
                row["tbt_token_slo_attainment"] = (
                    sum(value <= float(row["tbt_slo_ms"]) for value in gaps)
                    / len(gaps)
                    if gaps
                    else ""
                )
                raw_rows.append(row)
                for token in result.token_rows:
                    gap = token.get("estimated_per_token_gap_ms")
                    token_rows.append(
                        {
                            **common,
                            **token,
                            "tbt_slo_met": (
                                ""
                                if gap in ("", None)
                                else int(
                                    float(gap)
                                    <= float(item.slo["tbt_slo_ms"])
                                )
                            ),
                        }
                    )
            except Exception as exc:
                raw_rows.append(
                    {
                        **common,
                        "status": "failed",
                        "error_message": f"{type(exc).__name__}: {exc}",
                    }
                )

    raw_rows.sort(key=lambda row: float(row.get("submit_monotonic_ts", math.inf)))
    token_rows.sort(
        key=lambda row: float(row.get("arrival_monotonic_ts", math.inf))
    )
    summaries = request_summary(raw_rows, token_rows)
    write_csv(args.plan_csv, PLAN_FIELDNAMES, plan_rows)
    write_csv(args.raw_csv, RAW_FIELDNAMES, raw_rows)
    write_csv(args.token_csv, TOKEN_FIELDNAMES, token_rows)
    write_csv(args.summary_csv, SUMMARY_FIELDNAMES, summaries)

    failed = any(row["status"] != "success" for row in raw_rows)
    metadata["status"] = "failed" if failed else "completed"
    metadata["completed_at_utc"] = utc_now()
    metadata["request_successes"] = sum(
        row["status"] == "success" for row in raw_rows
    )
    metadata["request_failures"] = sum(
        row["status"] != "success" for row in raw_rows
    )
    args.metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] plan={args.plan_csv}")
    print(f"[DONE] raw={args.raw_csv}")
    print(f"[DONE] tokens={args.token_csv}")
    print(f"[DONE] summary={args.summary_csv}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
