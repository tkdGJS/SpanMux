#!/usr/bin/env python3
"""Replay a prepared BurstGPT/LooGLE/ShareGPT workload open-loop.

The prepared JSONL owns the arrival offsets and requested output lengths.
Every request is dispatched at its trace-derived offset without waiting for
earlier requests to finish. Exact TBT uses vLLM's returned ``choice.token_ids``
rather than retokenized text deltas.
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
    percentile,
)
from measure_vendor_v8_partition_profile import (
    StreamResult,
    load_slo_policy,
    lookup_request_slo,
    stream_request,
)


PLAN_FIELDS = [
    "event_index",
    "request_id",
    "role",
    "workload",
    "tenant_id",
    "scheduled_offset_s",
    "scheduled_monotonic_ts",
    "dispatch_monotonic_ts",
    "input_tokens",
    "requested_output_tokens",
    "burstgpt_source_row_index",
    "burstgpt_timestamp_s",
    "burstgpt_original_request_tokens",
    "burstgpt_request_length_replaced",
    "source_id",
]

RAW_FIELDS = PLAN_FIELDS + [
    "status",
    "error_message",
    "submit_monotonic_ts",
    "first_token_monotonic_ts",
    "completion_monotonic_ts",
    "arrival_lag_ms",
    "output_tokens",
    "ttft_ms",
    "mean_tbt_ms",
    "p50_tbt_ms",
    "p90_tbt_ms",
    "p95_tbt_ms",
    "p99_tbt_ms",
    "completion_latency_ms",
    "ttft_slo_ms",
    "tbt_slo_ms",
    "ttft_slo_met",
    "tbt_token_slo_attainment",
]

TOKEN_FIELDS = [
    "event_index",
    "request_id",
    "role",
    "workload",
    "input_tokens",
    "chunk_index",
    "chunk_tokens",
    "gap_start_monotonic_ts",
    "arrival_monotonic_ts",
    "estimated_per_token_gap_ms",
    "arrival_offset_from_trace_start_s",
    "tbt_slo_ms",
    "tbt_slo_met",
]

SUMMARY_FIELDS = [
    "role",
    "request_count",
    "success_count",
    "failure_count",
    "input_tokens_min",
    "input_tokens_max",
    "output_tokens_total",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p99_ms",
    "request_p50_tbt_p50_ms",
    "request_p90_tbt_p50_ms",
    "request_p99_tbt_p50_ms",
    "request_p50_tbt_p99_ms",
    "request_p90_tbt_p99_ms",
    "request_p99_tbt_p99_ms",
    "completion_p50_ms",
    "completion_p90_ms",
    "completion_p99_ms",
    "ttft_slo_attainment",
    "tbt_token_slo_attainment",
]


@dataclass(frozen=True)
class BurstReplayRequest:
    event_index: int
    request_id: str
    tenant_id: str
    role: str
    workload: str
    prompt: str
    input_tokens: int
    output_tokens: int
    scheduled_offset_s: float
    source_id: str
    burstgpt_source_row_index: int
    burstgpt_timestamp_s: float
    burstgpt_original_request_tokens: int
    burstgpt_request_length_replaced: int

    def prepared(self) -> PreparedRequest:
        return PreparedRequest(
            workload=self.workload,
            prompt=self.prompt,
            input_tokens=self.input_tokens,
            max_tokens=self.output_tokens,
            min_tokens=self.output_tokens,
            source_id=self.source_id,
        )


@dataclass
class ReplaySubmission:
    request: BurstReplayRequest
    tenant_id: str
    slo: dict[str, float]
    scheduled_monotonic_ts: float
    dispatch_monotonic_ts: float
    future: Future[StreamResult]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_workload(path: Path) -> list[BurstReplayRequest]:
    requests: list[BurstReplayRequest] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_index, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            max_tokens = int(payload["max_tokens"])
            min_tokens = int(payload["min_tokens"])
            if max_tokens <= 0 or min_tokens != max_tokens:
                raise ValueError(
                    f"line {line_index}: exact output replay requires "
                    "positive min_tokens == max_tokens"
                )
            requests.append(
                BurstReplayRequest(
                    event_index=len(requests),
                    request_id=str(payload["request_id"]),
                    tenant_id=str(
                        payload.get("tenant_id", payload["request_id"])
                    ),
                    role=str(payload["role"]),
                    workload=str(payload["workload"]),
                    prompt=str(payload["prompt"]),
                    input_tokens=int(payload["input_tokens"]),
                    output_tokens=max_tokens,
                    scheduled_offset_s=float(payload["scheduled_offset_s"]),
                    source_id=str(payload["source_id"]),
                    burstgpt_source_row_index=int(
                        payload["burstgpt_source_row_index"]
                    ),
                    burstgpt_timestamp_s=float(
                        payload["burstgpt_timestamp_s"]
                    ),
                    burstgpt_original_request_tokens=int(
                        payload["burstgpt_original_request_tokens"]
                    ),
                    burstgpt_request_length_replaced=int(
                        payload["burstgpt_request_length_replaced"]
                    ),
                )
            )
    if not requests:
        raise RuntimeError(f"No requests were found in {path}")
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("workload request_id values must be unique")
    if any(
        request.input_tokens <= 0
        or request.scheduled_offset_s < 0
        or not math.isfinite(request.scheduled_offset_s)
        for request in requests
    ):
        raise ValueError("workload contains an invalid length or offset")
    if any(
        current.scheduled_offset_s < previous.scheduled_offset_s
        for previous, current in zip(requests, requests[1:])
    ):
        raise ValueError("workload offsets must be nondecreasing")
    return requests


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def common_fields(
    submission: ReplaySubmission,
) -> dict[str, Any]:
    request = submission.request
    return {
        "event_index": request.event_index,
        "request_id": request.request_id,
        "role": request.role,
        "workload": request.workload,
        "tenant_id": submission.tenant_id,
        "scheduled_offset_s": request.scheduled_offset_s,
        "scheduled_monotonic_ts": submission.scheduled_monotonic_ts,
        "dispatch_monotonic_ts": submission.dispatch_monotonic_ts,
        "input_tokens": request.input_tokens,
        "requested_output_tokens": request.output_tokens,
        "burstgpt_source_row_index": request.burstgpt_source_row_index,
        "burstgpt_timestamp_s": request.burstgpt_timestamp_s,
        "burstgpt_original_request_tokens": (
            request.burstgpt_original_request_tokens
        ),
        "burstgpt_request_length_replaced": (
            request.burstgpt_request_length_replaced
        ),
        "source_id": request.source_id,
    }


def summarize(
    raw_rows: Sequence[dict[str, Any]],
    token_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    roles = ["all"] + sorted({str(row["role"]) for row in raw_rows})
    for role in roles:
        selected = [
            row for row in raw_rows if role == "all" or row["role"] == role
        ]
        successful = [row for row in selected if row["status"] == "success"]
        selected_ids = {str(row["request_id"]) for row in selected}
        selected_tokens = [
            row
            for row in token_rows
            if str(row["request_id"]) in selected_ids
            and row.get("estimated_per_token_gap_ms") not in ("", None)
        ]

        def values(name: str) -> list[float]:
            return [
                float(row[name])
                for row in successful
                if row.get(name) not in ("", None)
            ]

        token_attainment = [
            float(row["tbt_slo_met"])
            for row in selected_tokens
            for _ in range(int(row["chunk_tokens"]))
        ]
        contexts = [int(row["input_tokens"]) for row in selected]
        summaries.append(
            {
                "role": role,
                "request_count": len(selected),
                "success_count": len(successful),
                "failure_count": len(selected) - len(successful),
                "input_tokens_min": min(contexts) if contexts else "",
                "input_tokens_max": max(contexts) if contexts else "",
                "output_tokens_total": sum(
                    int(row["output_tokens"]) for row in successful
                ),
                "ttft_p50_ms": percentile(values("ttft_ms"), 50) or "",
                "ttft_p90_ms": percentile(values("ttft_ms"), 90) or "",
                "ttft_p99_ms": percentile(values("ttft_ms"), 99) or "",
                "request_p50_tbt_p50_ms": percentile(
                    values("p50_tbt_ms"), 50
                )
                or "",
                "request_p90_tbt_p50_ms": percentile(
                    values("p90_tbt_ms"), 50
                )
                or "",
                "request_p99_tbt_p50_ms": percentile(
                    values("p99_tbt_ms"), 50
                )
                or "",
                "request_p50_tbt_p99_ms": percentile(
                    values("p50_tbt_ms"), 99
                )
                or "",
                "request_p90_tbt_p99_ms": percentile(
                    values("p90_tbt_ms"), 99
                )
                or "",
                "request_p99_tbt_p99_ms": percentile(
                    values("p99_tbt_ms"), 99
                )
                or "",
                "completion_p50_ms": percentile(
                    values("completion_latency_ms"), 50
                )
                or "",
                "completion_p90_ms": percentile(
                    values("completion_latency_ms"), 90
                )
                or "",
                "completion_p99_ms": percentile(
                    values("completion_latency_ms"), 99
                )
                or "",
                "ttft_slo_attainment": (
                    statistics.fmean(
                        float(row["ttft_slo_met"]) for row in successful
                    )
                    if successful
                    else ""
                ),
                "tbt_token_slo_attainment": (
                    statistics.fmean(token_attainment)
                    if token_attainment
                    else ""
                ),
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open-loop replay of a prepared BurstGPT hybrid trace"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--workload-jsonl", type=Path, required=True)
    parser.add_argument("--slo-policy-json", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--request-timeout-sec", type=int, default=14400)
    parser.add_argument("--ignore-eos", type=int, choices=[0, 1], default=1)
    parser.add_argument("--warmup-count", type=int, default=1)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--warmup-cooldown-sec", type=float, default=5.0)
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--plan-csv", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--token-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requests = load_workload(args.workload_jsonl)
    if args.warmup_count < 0 or args.warmup_output_tokens <= 0:
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
        for request in requests:
            print(
                "[DRY-RUN] "
                f"event={request.event_index} request={request.request_id} "
                f"role={request.role} offset={request.scheduled_offset_s:g}s "
                f"input={request.input_tokens} output={request.output_tokens}"
            )
        return 0

    slo_policy = load_slo_policy(args.slo_policy_json)
    request_slos = {
        request.request_id: lookup_request_slo(
            slo_policy, request.input_tokens
        )
        for request in requests
    }
    http = get_requests_module()
    with http.Session() as session:
        model = args.model or get_model_id_from_server(
            session, args.base_url.rstrip("/")
        )

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "status": "running",
        "case_id": args.case_id,
        "model": model,
        "workload_kind": "trace_driven_hybrid_not_raw_production_replay",
        "submission_mode": "open_loop_trace_offsets",
        "tbt_measurement": "vllm_choice_token_ids",
        "workload_jsonl": str(args.workload_jsonl),
        "slo_policy_json": str(args.slo_policy_json),
        "request_count": len(requests),
        "long_request_count": sum(r.role == "long" for r in requests),
        "short_request_count": sum(r.role == "short" for r in requests),
        "warmup_count": args.warmup_count,
    }
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    warmup_request = min(requests, key=lambda request: request.input_tokens)
    for warmup_index in range(args.warmup_count):
        warmup_slo = request_slos[warmup_request.request_id]
        stream_request(
            base_url=args.base_url,
            model=model,
            request=warmup_request.prepared(),
            request_id=f"{args.case_id}-warmup-{warmup_index + 1}",
            role="warmup",
            tenant_id="burstgpt_warmup",
            max_tokens=args.warmup_output_tokens,
            min_tokens=args.warmup_output_tokens,
            ignore_eos=True,
            request_timeout_sec=args.request_timeout_sec,
            xargs=warmup_slo,
        )
    if args.warmup_count and args.warmup_cooldown_sec > 0:
        time.sleep(args.warmup_cooldown_sec)

    workers = args.max_workers or len(requests)
    if workers < len(requests):
        raise ValueError(
            "open-loop replay requires max-workers >= request count to "
            "prevent client-side queueing"
        )
    submissions: list[ReplaySubmission] = []
    trace_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for request in requests:
            scheduled_ts = trace_start + request.scheduled_offset_s
            remaining = scheduled_ts - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            dispatch_ts = time.monotonic()
            tenant_id = request.tenant_id
            slo = request_slos[request.request_id]
            future = executor.submit(
                stream_request,
                base_url=args.base_url,
                model=model,
                request=request.prepared(),
                request_id=request.request_id,
                role=request.role,
                tenant_id=tenant_id,
                max_tokens=request.output_tokens,
                min_tokens=request.output_tokens,
                ignore_eos=bool(args.ignore_eos),
                request_timeout_sec=args.request_timeout_sec,
                xargs=slo,
            )
            submissions.append(
                ReplaySubmission(
                    request=request,
                    tenant_id=tenant_id,
                    slo=slo,
                    scheduled_monotonic_ts=scheduled_ts,
                    dispatch_monotonic_ts=dispatch_ts,
                    future=future,
                )
            )

        raw_rows: list[dict[str, Any]] = []
        token_rows: list[dict[str, Any]] = []
        for submission in submissions:
            common = common_fields(submission)
            try:
                result = submission.future.result()
                row = {**common, **result.row}
                row["arrival_lag_ms"] = (
                    float(row["submit_monotonic_ts"])
                    - submission.scheduled_monotonic_ts
                ) * 1000.0
                gaps = [
                    float(token["estimated_per_token_gap_ms"])
                    for token in result.token_rows
                    if token.get("estimated_per_token_gap_ms")
                    not in ("", None)
                    for _ in range(int(token["chunk_tokens"]))
                ]
                row["p50_tbt_ms"] = percentile(gaps, 50) or ""
                row["p90_tbt_ms"] = percentile(gaps, 90) or ""
                row["ttft_slo_ms"] = submission.slo[
                    "ttft_prefill_deadline_ms"
                ]
                row["ttft_slo_met"] = int(
                    float(row["ttft_ms"]) <= float(row["ttft_slo_ms"])
                )
                row["tbt_token_slo_attainment"] = (
                    sum(
                        gap <= float(submission.slo["tbt_slo_ms"])
                        for gap in gaps
                    )
                    / len(gaps)
                    if gaps
                    else ""
                )
                raw_rows.append(row)
                for token in result.token_rows:
                    gap = token.get("estimated_per_token_gap_ms")
                    token_rows.append(
                        {
                            "event_index": submission.request.event_index,
                            "request_id": submission.request.request_id,
                            "role": submission.request.role,
                            "workload": submission.request.workload,
                            "input_tokens": submission.request.input_tokens,
                            **token,
                            "arrival_offset_from_trace_start_s": (
                                float(token["arrival_monotonic_ts"])
                                - trace_start
                            ),
                            "tbt_slo_ms": submission.slo["tbt_slo_ms"],
                            "tbt_slo_met": (
                                ""
                                if gap in ("", None)
                                else int(
                                    float(gap)
                                    <= float(submission.slo["tbt_slo_ms"])
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
                        "ttft_slo_ms": submission.slo[
                            "ttft_prefill_deadline_ms"
                        ],
                        "tbt_slo_ms": submission.slo["tbt_slo_ms"],
                    }
                )

    raw_rows.sort(key=lambda row: int(row["event_index"]))
    token_rows.sort(
        key=lambda row: float(row.get("arrival_monotonic_ts", math.inf))
    )
    plan_rows = [common_fields(submission) for submission in submissions]
    summary_rows = summarize(raw_rows, token_rows)
    write_csv(args.plan_csv, PLAN_FIELDS, plan_rows)
    write_csv(args.raw_csv, RAW_FIELDS, raw_rows)
    write_csv(args.token_csv, TOKEN_FIELDS, token_rows)
    write_csv(args.summary_csv, SUMMARY_FIELDS, summary_rows)

    failed = any(row["status"] != "success" for row in raw_rows)
    metadata["status"] = "failed" if failed else "completed"
    metadata["completed_at_utc"] = utc_now()
    metadata["request_successes"] = sum(
        row["status"] == "success" for row in raw_rows
    )
    metadata["request_failures"] = sum(
        row["status"] != "success" for row in raw_rows
    )
    metadata["trace_start_monotonic_ts"] = trace_start
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
