#!/usr/bin/env python3
"""Replay prepared real workloads against a vLLM OpenAI-compatible server."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class IncompleteCompletionError(RuntimeError):
    """The HTTP response ended without satisfying the generation contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_sse_json(line: str) -> Optional[Dict[str, Any]]:
    if not line.startswith("data: "):
        return None
    payload = line[6:].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


def validate_completion_contract(
    *,
    output_tokens: int,
    requested_min_tokens: int,
    finish_reason: Optional[str],
    saw_done: bool,
    streaming: bool,
) -> None:
    if streaming and not saw_done:
        raise IncompleteCompletionError(
            "stream ended before the OpenAI [DONE] marker"
        )
    if not finish_reason:
        raise IncompleteCompletionError(
            "completion ended without a finish_reason"
        )
    if requested_min_tokens > 0 and output_tokens < requested_min_tokens:
        raise IncompleteCompletionError(
            "completion returned fewer tokens than requested: "
            f"output_tokens={output_tokens}, "
            f"requested_min_tokens={requested_min_tokens}, "
            f"finish_reason={finish_reason}"
        )


def get_requests_module() -> Any:
    import requests

    return requests


def get_model_id_from_server(session: Any, base_url: str) -> str:
    from measure_vllm_run import get_model_id

    return get_model_id(session, base_url)


def tokenize_count(session: Any, base_url: str, model: str, text: str) -> int:
    from measure_vllm_run import tokenize_text

    count, _ = tokenize_text(session, base_url, model, text)
    return int(count)


def exact_completion_token_ids(choice: Dict[str, Any]) -> List[int]:
    """Return vLLM-provided generated token IDs or fail closed.

    Retokenizing an SSE text delta is not token-count preserving and, with the
    tokenize endpoint's default settings, also adds a BOS token per delta.
    Benchmark TBT must therefore use the response's exact delta token IDs.
    """

    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list):
        raise IncompleteCompletionError(
            "completion response omitted token_ids; exact TBT measurement "
            "requires return_token_ids=true"
        )
    return [int(token_id) for token_id in token_ids]


RAW_FIELDNAMES = [
    "timestamp_utc",
    "case_id",
    "repeat_index",
    "role",
    "workload",
    "background_workload",
    "request_index",
    "request_id",
    "tenant_id",
    "status",
    "error_message",
    "measurement_window",
    "scheduled_offset_s",
    "arrival_time_s",
    "input_tokens",
    "requested_min_tokens",
    "requested_max_tokens",
    "output_tokens",
    "ttft_ms",
    "p50_tbt_ms",
    "p90_tbt_ms",
    "p99_tbt_ms",
    "completion_latency_ms",
    "time_to_last_token_ms",
]

TOKEN_FIELDNAMES = [
    "case_id",
    "repeat_index",
    "request_id",
    "tenant_id",
    "role",
    "workload",
    "measurement_window",
    "chunk_index",
    "chunk_tokens",
    "chunk_gap_ms",
    "estimated_per_token_gap_ms",
    "chunk_arrival_offset_s",
]

SUMMARY_FIELDNAMES = [
    "timestamp_utc",
    "case_id",
    "repeat_index",
    "foreground_workload",
    "background_workload",
    "arrival_mode",
    "burst_size",
    "burst_count",
    "burst_interval_sec",
    "fg_rps",
    "bg_rps",
    "duration_sec",
    "warmup_sec",
    "foreground_successes",
    "foreground_failures",
    "background_successes",
    "background_failures",
    "foreground_output_tokens",
    "foreground_p50_tbt_ms",
    "foreground_p90_tbt_ms",
    "foreground_p99_tbt_ms",
    "foreground_p99_ttft_ms",
    "foreground_p99_completion_latency_ms",
    "background_p99_completion_latency_ms",
]


@dataclass(frozen=True)
class PreparedRequest:
    workload: str
    prompt: str
    input_tokens: int
    max_tokens: int
    min_tokens: int
    source_id: str


@dataclass(frozen=True)
class ReplayEvent:
    role: str
    workload: str
    request: PreparedRequest
    request_index: int
    offset_s: float
    measurement_window: bool
    tenant_id: str


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    frac = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * frac


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return ""
    return str(round(float(value), digits))


def load_prepared_jsonl(path: Path) -> List[PreparedRequest]:
    rows: List[PreparedRequest] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows.append(
                PreparedRequest(
                    workload=str(payload.get("workload", path.stem)),
                    prompt=str(payload["prompt"]),
                    input_tokens=int(payload.get("input_tokens", 0)),
                    max_tokens=int(payload.get("max_tokens", 1)),
                    min_tokens=int(payload.get("min_tokens", 0)),
                    source_id=str(payload.get("source_id", len(rows))),
                )
            )
    if not rows:
        raise RuntimeError(f"No prepared requests found in {path}")
    return rows


def poisson_offsets(rate: float, horizon_s: float, rng: random.Random) -> List[float]:
    if rate <= 0:
        return []
    offsets: List[float] = []
    current = 0.0
    while True:
        current += rng.expovariate(rate)
        if current > horizon_s:
            break
        offsets.append(current)
    return offsets


def burst_offsets(size: int, count: int, interval_s: float,
                  start_offset_s: float) -> List[float]:
    if size <= 0 or count <= 0:
        return []
    offsets: List[float] = []
    for burst_index in range(count):
        offset = start_offset_s + burst_index * interval_s
        offsets.extend([offset] * size)
    return offsets


def build_events(
    *,
    foreground: List[PreparedRequest],
    background: List[PreparedRequest],
    fg_rps: float,
    bg_rps: float,
    warmup_sec: float,
    duration_sec: float,
    foreground_tenant_count: int,
    background_tenant_count: int,
    seed: int,
    arrival_mode: str,
    burst_size: int,
    burst_count: int,
    burst_interval_sec: float,
    burst_start_offset_sec: float,
) -> List[ReplayEvent]:
    rng = random.Random(seed)
    horizon = warmup_sec + duration_sec
    events: List[ReplayEvent] = []

    if arrival_mode == "poisson":
        foreground_offsets = poisson_offsets(fg_rps, horizon, rng)
        background_offsets = poisson_offsets(bg_rps, horizon, rng)
    elif arrival_mode == "burst":
        foreground_offsets = burst_offsets(
            burst_size,
            burst_count,
            burst_interval_sec,
            burst_start_offset_sec,
        )
        background_offsets = []
        if background and bg_rps > 0:
            background_offsets = burst_offsets(
                burst_size,
                burst_count,
                burst_interval_sec,
                burst_start_offset_sec,
            )
    else:
        raise ValueError(f"Unsupported arrival mode: {arrival_mode}")

    for index, offset in enumerate(foreground_offsets):
        req = foreground[index % len(foreground)]
        tenant_id = f"fg_tenant_{index % max(1, foreground_tenant_count)}"
        events.append(
            ReplayEvent(
                role="foreground",
                workload=req.workload,
                request=req,
                request_index=index,
                offset_s=offset,
                measurement_window=offset >= warmup_sec,
                tenant_id=tenant_id,
            )
        )

    for index, offset in enumerate(background_offsets):
        req = background[index % len(background)]
        tenant_id = f"bg_tenant_{index % max(1, background_tenant_count)}"
        events.append(
            ReplayEvent(
                role="background",
                workload=req.workload,
                request=req,
                request_index=index,
                offset_s=offset,
                measurement_window=offset >= warmup_sec,
                tenant_id=tenant_id,
            )
        )

    events.sort(key=lambda event: (event.offset_s, event.role, event.request_index))
    return events


def request_payload(
    model: str,
    event: ReplayEvent,
    request_id: str,
    max_tokens_override: int,
    min_tokens_override: int,
    ignore_eos: bool,
    stream: bool,
) -> Dict[str, Any]:
    max_tokens = max_tokens_override if max_tokens_override > 0 else event.request.max_tokens
    min_tokens = min_tokens_override if min_tokens_override >= 0 else event.request.min_tokens
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": event.request.prompt,
        "max_tokens": max_tokens,
        "stream": stream,
        "return_token_ids": True,
        "request_id": request_id,
        "vllm_xargs": {"tenant_id": event.tenant_id},
    }
    if min_tokens > 0:
        payload["min_tokens"] = min_tokens
    if ignore_eos:
        payload["ignore_eos"] = True
    return payload


def stream_completion(
    *,
    session: Any,
    base_url: str,
    model: str,
    event: ReplayEvent,
    request_id: str,
    request_timeout_sec: int,
    max_tokens_override: int,
    min_tokens_override: int,
    ignore_eos: bool,
    run_start_ts: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[float]]:
    payload = request_payload(
        model,
        event,
        request_id,
        max_tokens_override,
        min_tokens_override,
        ignore_eos,
        True,
    )
    start_ts = time.monotonic()
    response = session.post(
        base_url + "/v1/completions",
        json=payload,
        headers={"X-Request-Id": request_id},
        stream=True,
        timeout=(10, request_timeout_sec),
    )
    response.raise_for_status()

    first_token_ts: Optional[float] = None
    prev_chunk_ts: Optional[float] = None
    token_rows: List[Dict[str, Any]] = []
    per_token_gaps: List[float] = []
    chunk_index = 0
    saw_done = False
    finish_reason: Optional[str] = None
    output_tokens = 0

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if raw_line.startswith("data: "):
            payload_text = raw_line[6:].strip()
            if payload_text == "[DONE]":
                saw_done = True
                break
        message = parse_sse_json(raw_line)
        if not message:
            continue
        choices = message.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason") is not None:
            finish_reason = str(choice["finish_reason"])
        delta_token_ids = exact_completion_token_ids(choice)
        if not delta_token_ids:
            continue
        now = time.monotonic()
        if first_token_ts is None:
            first_token_ts = now
        chunk_tokens = len(delta_token_ids)
        output_tokens += chunk_tokens
        chunk_gap_ms = None
        per_token_gap = None
        if prev_chunk_ts is not None:
            chunk_gap_ms = (now - prev_chunk_ts) * 1000.0
            if chunk_tokens > 0:
                per_token_gap = chunk_gap_ms / float(chunk_tokens)
                per_token_gaps.extend([per_token_gap] * chunk_tokens)
        prev_chunk_ts = now

        token_rows.append(
            {
                "case_id": "",
                "repeat_index": "",
                "request_id": request_id,
                "tenant_id": event.tenant_id,
                "role": event.role,
                "workload": event.workload,
                "measurement_window": int(event.measurement_window),
                "chunk_index": chunk_index,
                "chunk_tokens": chunk_tokens,
                "chunk_gap_ms": "" if chunk_gap_ms is None else round(chunk_gap_ms, 4),
                "estimated_per_token_gap_ms": "" if per_token_gap is None else round(per_token_gap, 4),
                "chunk_arrival_offset_s": round(now - run_start_ts, 6),
            }
        )
        chunk_index += 1

    end_ts = time.monotonic()
    if first_token_ts is None:
        raise RuntimeError("No streamed completion tokens received")

    requested_min_tokens = int(payload.get("min_tokens", 0) or 0)
    validate_completion_contract(
        output_tokens=output_tokens,
        requested_min_tokens=requested_min_tokens,
        finish_reason=finish_reason,
        saw_done=saw_done,
        streaming=True,
    )
    row = {
        "output_tokens": output_tokens,
        "ttft_ms": (first_token_ts - start_ts) * 1000.0,
        "completion_latency_ms": (end_ts - start_ts) * 1000.0,
        "time_to_last_token_ms": (end_ts - start_ts) * 1000.0,
        "p50_tbt_ms": percentile(per_token_gaps, 50),
        "p90_tbt_ms": percentile(per_token_gaps, 90),
        "p99_tbt_ms": percentile(per_token_gaps, 99),
    }
    return row, token_rows, per_token_gaps


def non_stream_completion(
    *,
    session: Any,
    base_url: str,
    model: str,
    event: ReplayEvent,
    request_id: str,
    request_timeout_sec: int,
    max_tokens_override: int,
    min_tokens_override: int,
    ignore_eos: bool,
) -> Dict[str, Any]:
    payload = request_payload(
        model,
        event,
        request_id,
        max_tokens_override,
        min_tokens_override,
        ignore_eos,
        False,
    )
    start_ts = time.monotonic()
    response = session.post(
        base_url + "/v1/completions",
        json=payload,
        headers={"X-Request-Id": request_id},
        timeout=(10, request_timeout_sec),
    )
    response.raise_for_status()
    end_ts = time.monotonic()
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise IncompleteCompletionError("completion response has no choices")
    choice = choices[0]
    output_tokens = len(exact_completion_token_ids(choice))
    requested_min_tokens = int(
        request_payload(
            model,
            event,
            request_id,
            max_tokens_override,
            min_tokens_override,
            ignore_eos,
            False,
        ).get("min_tokens", 0)
        or 0
    )
    finish_reason = choice.get("finish_reason")
    validate_completion_contract(
        output_tokens=output_tokens,
        requested_min_tokens=requested_min_tokens,
        finish_reason=(
            str(finish_reason) if finish_reason is not None else None
        ),
        saw_done=True,
        streaming=False,
    )
    return {
        "output_tokens": output_tokens,
        "ttft_ms": "",
        "completion_latency_ms": (end_ts - start_ts) * 1000.0,
        "time_to_last_token_ms": (end_ts - start_ts) * 1000.0,
        "p50_tbt_ms": "",
        "p90_tbt_ms": "",
        "p99_tbt_ms": "",
    }


def run_one_event(
    *,
    base_url: str,
    model: str,
    case_id: str,
    repeat_index: int,
    event: ReplayEvent,
    run_start_ts: float,
    foreground_max_tokens: int,
    foreground_min_tokens: int,
    background_max_tokens: int,
    background_min_tokens: int,
    ignore_eos: bool,
    request_timeout_sec: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[float]]:
    request_id = f"{case_id}-{event.role}-{event.workload}-{repeat_index}-{event.request_index}"
    row: Dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "case_id": case_id,
        "repeat_index": repeat_index,
        "role": event.role,
        "workload": event.workload,
        "background_workload": "",
        "request_index": event.request_index,
        "request_id": request_id,
        "tenant_id": event.tenant_id,
        "status": "failed",
        "error_message": "",
        "measurement_window": int(event.measurement_window),
        "scheduled_offset_s": round(event.offset_s, 6),
        "arrival_time_s": round(time.monotonic() - run_start_ts, 6),
        "input_tokens": event.request.input_tokens,
        "requested_min_tokens": "",
        "requested_max_tokens": "",
        "output_tokens": "",
        "ttft_ms": "",
        "p50_tbt_ms": "",
        "p90_tbt_ms": "",
        "p99_tbt_ms": "",
        "completion_latency_ms": "",
        "time_to_last_token_ms": "",
    }
    token_rows: List[Dict[str, Any]] = []
    token_gaps: List[float] = []
    try:
        requests = get_requests_module()
        with requests.Session() as session:
            if event.role == "foreground":
                max_override = foreground_max_tokens
                min_override = foreground_min_tokens
                row["requested_max_tokens"] = max_override if max_override > 0 else event.request.max_tokens
                row["requested_min_tokens"] = min_override if min_override >= 0 else event.request.min_tokens
                result, token_rows, token_gaps = stream_completion(
                    session=session,
                    base_url=base_url,
                    model=model,
                    event=event,
                    request_id=request_id,
                    request_timeout_sec=request_timeout_sec,
                    max_tokens_override=max_override,
                    min_tokens_override=min_override,
                    ignore_eos=ignore_eos,
                    run_start_ts=run_start_ts,
                )
            else:
                max_override = background_max_tokens
                min_override = background_min_tokens
                row["requested_max_tokens"] = max_override if max_override > 0 else event.request.max_tokens
                row["requested_min_tokens"] = min_override if min_override >= 0 else event.request.min_tokens
                result = non_stream_completion(
                    session=session,
                    base_url=base_url,
                    model=model,
                    event=event,
                    request_id=request_id,
                    request_timeout_sec=request_timeout_sec,
                    max_tokens_override=max_override,
                    min_tokens_override=min_override,
                    ignore_eos=ignore_eos,
                )
        row.update(
            {
                "status": "success",
                "output_tokens": result["output_tokens"],
                "ttft_ms": fmt(result.get("ttft_ms") if result.get("ttft_ms") != "" else None),
                "p50_tbt_ms": fmt(result.get("p50_tbt_ms") if result.get("p50_tbt_ms") != "" else None),
                "p90_tbt_ms": fmt(result.get("p90_tbt_ms") if result.get("p90_tbt_ms") != "" else None),
                "p99_tbt_ms": fmt(result.get("p99_tbt_ms") if result.get("p99_tbt_ms") != "" else None),
                "completion_latency_ms": fmt(float(result["completion_latency_ms"])),
                "time_to_last_token_ms": fmt(float(result["time_to_last_token_ms"])),
            }
        )
    except Exception as exc:
        row["error_message"] = repr(exc)
    return row, token_rows, token_gaps


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(
    raw_rows: List[Dict[str, Any]],
    foreground_token_gaps: List[float],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    fg_rows = [
        row for row in raw_rows
        if row["role"] == "foreground"
        and row["measurement_window"] == 1
        and row["status"] == "success"
    ]
    fg_failed = [
        row for row in raw_rows
        if row["role"] == "foreground"
        and row["measurement_window"] == 1
        and row["status"] != "success"
    ]
    bg_rows = [
        row for row in raw_rows
        if row["role"] == "background"
        and row["measurement_window"] == 1
        and row["status"] == "success"
    ]
    bg_failed = [
        row for row in raw_rows
        if row["role"] == "background"
        and row["measurement_window"] == 1
        and row["status"] != "success"
    ]
    ttft_values = [float(row["ttft_ms"]) for row in fg_rows if row.get("ttft_ms") not in ("", None)]
    fg_latency_values = [
        float(row["completion_latency_ms"]) for row in fg_rows
        if row.get("completion_latency_ms") not in ("", None)
    ]
    bg_latency_values = [
        float(row["completion_latency_ms"]) for row in bg_rows
        if row.get("completion_latency_ms") not in ("", None)
    ]
    fg_output_tokens = sum(
        int(float(row["output_tokens"])) for row in fg_rows
        if row.get("output_tokens") not in ("", None)
    )
    foreground_workloads = sorted(
        {
            str(row.get("workload", ""))
            for row in raw_rows
            if row.get("role") == "foreground" and row.get("workload")
        }
    )
    return {
        "timestamp_utc": utc_now(),
        "case_id": args.case_id,
        "repeat_index": args.repeat_index,
        "foreground_workload": "+".join(foreground_workloads) or "unknown",
        "background_workload": args.background_workload,
        "arrival_mode": args.arrival_mode,
        "burst_size": args.burst_size,
        "burst_count": args.burst_count,
        "burst_interval_sec": args.burst_interval_sec,
        "fg_rps": args.fg_rps,
        "bg_rps": args.bg_rps,
        "duration_sec": args.duration_sec,
        "warmup_sec": args.warmup_sec,
        "foreground_successes": len(fg_rows),
        "foreground_failures": len(fg_failed),
        "background_successes": len(bg_rows),
        "background_failures": len(bg_failed),
        "foreground_output_tokens": fg_output_tokens,
        "foreground_p50_tbt_ms": fmt(percentile(foreground_token_gaps, 50)),
        "foreground_p90_tbt_ms": fmt(percentile(foreground_token_gaps, 90)),
        "foreground_p99_tbt_ms": fmt(percentile(foreground_token_gaps, 99)),
        "foreground_p99_ttft_ms": fmt(percentile(ttft_values, 99)),
        "foreground_p99_completion_latency_ms": fmt(percentile(fg_latency_values, 99)),
        "background_p99_completion_latency_ms": fmt(percentile(bg_latency_values, 99)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poisson replay for real-workload motivation runs.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--foreground-jsonl", type=Path, required=True)
    parser.add_argument("--background-jsonl", type=Path)
    parser.add_argument("--background-workload", default="none")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--repeat-index", type=int, default=1)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--token-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--fg-rps", type=float, default=0.05)
    parser.add_argument("--bg-rps", type=float, default=0.05)
    parser.add_argument("--duration-sec", type=float, default=180)
    parser.add_argument("--warmup-sec", type=float, default=30)
    parser.add_argument("--arrival-mode", choices=["poisson", "burst"], default="poisson")
    parser.add_argument("--burst-size", type=int, default=2)
    parser.add_argument("--burst-count", type=int, default=3)
    parser.add_argument("--burst-interval-sec", type=float, default=30.0)
    parser.add_argument("--burst-start-offset-sec", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--request-timeout-sec", type=int, default=1800)
    parser.add_argument("--foreground-tenant-count", type=int, default=1)
    parser.add_argument("--background-tenant-count", type=int, default=1)
    parser.add_argument("--foreground-max-tokens", type=int, default=-1)
    parser.add_argument("--foreground-min-tokens", type=int, default=-1)
    parser.add_argument("--background-max-tokens", type=int, default=-1)
    parser.add_argument("--background-min-tokens", type=int, default=-1)
    parser.add_argument("--ignore-eos", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    requests = get_requests_module()
    with requests.Session() as session:
        model = args.model or get_model_id_from_server(session, base_url)

    foreground = load_prepared_jsonl(args.foreground_jsonl)
    background = load_prepared_jsonl(args.background_jsonl) if args.background_jsonl else []
    events = build_events(
        foreground=foreground,
        background=background,
        fg_rps=args.fg_rps,
        bg_rps=args.bg_rps if background else 0.0,
        warmup_sec=args.warmup_sec,
        duration_sec=args.duration_sec,
        foreground_tenant_count=args.foreground_tenant_count,
        background_tenant_count=args.background_tenant_count,
        seed=args.seed + args.repeat_index,
        arrival_mode=args.arrival_mode,
        burst_size=args.burst_size,
        burst_count=args.burst_count,
        burst_interval_sec=args.burst_interval_sec,
        burst_start_offset_sec=args.burst_start_offset_sec,
    )
    if not events:
        raise RuntimeError("No replay events generated; increase duration or request rates.")

    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(
        json.dumps(
            {
                "created_at": utc_now(),
                "base_url": base_url,
                "model": model,
                "case_id": args.case_id,
                "repeat_index": args.repeat_index,
                "foreground_jsonl": str(args.foreground_jsonl),
                "background_jsonl": str(args.background_jsonl) if args.background_jsonl else "",
                "fg_rps": args.fg_rps,
                "bg_rps": args.bg_rps if background else 0.0,
                "arrival_mode": args.arrival_mode,
                "burst_size": args.burst_size,
                "burst_count": args.burst_count,
                "burst_interval_sec": args.burst_interval_sec,
                "burst_start_offset_sec": args.burst_start_offset_sec,
                "duration_sec": args.duration_sec,
                "warmup_sec": args.warmup_sec,
                "events": len(events),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    raw_rows: List[Dict[str, Any]] = []
    all_token_rows: List[Dict[str, Any]] = []
    measured_fg_token_gaps: List[float] = []
    run_start_ts = time.monotonic() + 2.0

    futures = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
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
                    foreground_max_tokens=args.foreground_max_tokens,
                    foreground_min_tokens=args.foreground_min_tokens,
                    background_max_tokens=args.background_max_tokens,
                    background_min_tokens=args.background_min_tokens,
                    ignore_eos=bool(args.ignore_eos),
                    request_timeout_sec=args.request_timeout_sec,
                )
            )

        for future in as_completed(futures):
            row, token_rows, token_gaps = future.result()
            row["background_workload"] = args.background_workload
            for token_row in token_rows:
                token_row["case_id"] = args.case_id
                token_row["repeat_index"] = args.repeat_index
            raw_rows.append(row)
            all_token_rows.extend(token_rows)
            if (
                row["role"] == "foreground"
                and row["measurement_window"] == 1
                and row["status"] == "success"
            ):
                measured_fg_token_gaps.extend(token_gaps)

    raw_rows.sort(key=lambda row: (float(row["scheduled_offset_s"]), row["role"], int(row["request_index"])))
    all_token_rows.sort(
        key=lambda row: (
            str(row["request_id"]),
            int(row["chunk_index"]),
        )
    )
    write_csv(args.raw_csv, RAW_FIELDNAMES, raw_rows)
    write_csv(args.token_csv, TOKEN_FIELDNAMES, all_token_rows)
    summary = summarize(raw_rows, measured_fg_token_gaps, args)
    write_csv(args.summary_csv, SUMMARY_FIELDNAMES, [summary])

    print(
        "[SUMMARY] "
        f"case={args.case_id} repeat={args.repeat_index} "
        f"fg_success={summary['foreground_successes']} "
        f"bg_success={summary['background_successes']} "
        f"fg_p99_tbt_ms={summary['foreground_p99_tbt_ms']}",
        flush=True,
    )
    print(f"[DONE] raw_csv={args.raw_csv}")
    print(f"[DONE] token_csv={args.token_csv}")
    print(f"[DONE] summary_csv={args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
