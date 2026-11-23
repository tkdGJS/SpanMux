#!/usr/bin/env python3
"""Staged P/D VRAM-contention replay for GreenContext experiments.

The driver starts one long-context decode request, waits until its first token
arrives so the request has resident decode KV, and then injects one long-prefill
request. This creates the condition:

    resident decode KV + incoming prefill KV demand > available KV pool

when the selected context target is close to the model/KV context limit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from measure_real_workload_motivation import (
    RAW_FIELDNAMES,
    TOKEN_FIELDNAMES,
    PreparedRequest,
    ReplayEvent,
    exact_completion_token_ids,
    get_model_id_from_server,
    get_requests_module,
    load_prepared_jsonl,
    non_stream_completion,
    percentile,
    request_payload,
    tokenize_count,
    validate_completion_contract,
)


SUMMARY_FIELDNAMES = [
    "timestamp_utc",
    "case_id",
    "repeat_index",
    "decode_context_target_tokens",
    "prefill_context_target_tokens",
    "decode_input_tokens",
    "prefill_input_tokens",
    "decode_success",
    "prefill_success",
    "decode_output_tokens",
    "decode_ttft_ms",
    "decode_p50_tbt_ms",
    "decode_p90_tbt_ms",
    "decode_p99_tbt_ms",
    "decode_completion_latency_ms",
    "prefill_completion_latency_ms",
    "first_token_to_prefill_submit_ms",
    "decode_first_token_offset_s",
    "prefill_submit_offset_s",
    "decode_requested_max_tokens",
    "decode_requested_min_tokens",
    "prefill_requested_max_tokens",
    "prefill_requested_min_tokens",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return ""
    return str(round(float(value), digits))


def write_csv(path: Path, fieldnames: Sequence[str],
              rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


@dataclass
class StreamState:
    first_token_event: threading.Event
    done_event: threading.Event
    first_token_ts: Optional[float] = None
    error: str = ""
    row: Optional[Dict[str, Any]] = None
    token_rows: Optional[List[Dict[str, Any]]] = None
    token_gaps: Optional[List[float]] = None


def make_event(
    *,
    role: str,
    workload: str,
    request: PreparedRequest,
    request_index: int,
    tenant_id: str,
) -> ReplayEvent:
    return ReplayEvent(
        role=role,
        workload=workload,
        request=request,
        request_index=request_index,
        offset_s=0.0,
        measurement_window=True,
        tenant_id=tenant_id,
    )


def fit_request_to_token_window(
    *,
    session: Any,
    base_url: str,
    model: str,
    request: PreparedRequest,
    target_tokens: int,
    context_window: int,
    max_model_len: int,
    max_output_tokens: int,
    label: str,
) -> PreparedRequest:
    """Return a request whose prompt fits the exact vLLM tokenizer budget."""

    if target_tokens <= 0:
        target_tokens = request.input_tokens
    max_input_tokens = target_tokens
    if max_model_len > 0:
        max_input_tokens = min(max_input_tokens, max_model_len - max_output_tokens)
    if max_input_tokens <= 0:
        raise ValueError(
            f"{label}: max_input_tokens={max_input_tokens} is invalid; "
            f"max_model_len={max_model_len}, max_output_tokens={max_output_tokens}"
        )
    min_input_tokens = max(1, max_input_tokens - max(0, context_window))

    exact_tokens = tokenize_count(session, base_url, model, request.prompt)
    if min_input_tokens <= exact_tokens <= max_input_tokens:
        return PreparedRequest(
            workload=request.workload,
            prompt=request.prompt,
            input_tokens=exact_tokens,
            max_tokens=request.max_tokens,
            min_tokens=request.min_tokens,
            source_id=request.source_id,
        )
    if exact_tokens < min_input_tokens:
        print(
            f"[WARN] {label}: exact_tokens={exact_tokens} below target window "
            f"{min_input_tokens}-{max_input_tokens}; using unmodified prompt",
            file=sys.stderr,
            flush=True,
        )
        return PreparedRequest(
            workload=request.workload,
            prompt=request.prompt,
            input_tokens=exact_tokens,
            max_tokens=request.max_tokens,
            min_tokens=request.min_tokens,
            source_id=request.source_id,
        )

    low = 1
    high = len(request.prompt)
    best_prompt = ""
    best_tokens = 0
    while low <= high:
        mid = (low + high) // 2
        candidate = request.prompt[:mid]
        count = tokenize_count(session, base_url, model, candidate)
        if count <= max_input_tokens:
            best_prompt = candidate
            best_tokens = count
            low = mid + 1
        else:
            high = mid - 1

    if not best_prompt:
        raise RuntimeError(
            f"{label}: could not truncate prompt under {max_input_tokens} tokens"
        )
    if best_tokens < min_input_tokens:
        print(
            f"[WARN] {label}: truncated exact_tokens={best_tokens} below "
            f"target window {min_input_tokens}-{max_input_tokens}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"[FIT] {label}: exact_tokens {exact_tokens} -> {best_tokens} "
            f"(target_window={min_input_tokens}-{max_input_tokens})",
            file=sys.stderr,
            flush=True,
        )

    return PreparedRequest(
        workload=request.workload,
        prompt=best_prompt,
        input_tokens=best_tokens,
        max_tokens=request.max_tokens,
        min_tokens=request.min_tokens,
        source_id=request.source_id,
    )


def stream_decode_request(
    *,
    base_url: str,
    model: str,
    case_id: str,
    repeat_index: int,
    event: ReplayEvent,
    request_id: str,
    run_start_ts: float,
    max_tokens: int,
    min_tokens: int,
    ignore_eos: bool,
    request_timeout_sec: int,
    state: StreamState,
) -> None:
    row: Dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "case_id": case_id,
        "repeat_index": repeat_index,
        "role": event.role,
        "workload": event.workload,
        "background_workload": "loogle_prefill",
        "request_index": event.request_index,
        "request_id": request_id,
        "tenant_id": event.tenant_id,
        "status": "failed",
        "error_message": "",
        "measurement_window": 1,
        "scheduled_offset_s": 0.0,
        "arrival_time_s": round(time.monotonic() - run_start_ts, 6),
        "input_tokens": event.request.input_tokens,
        "requested_min_tokens": min_tokens,
        "requested_max_tokens": max_tokens,
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
            payload = request_payload(
                model,
                event,
                request_id,
                max_tokens,
                min_tokens,
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
            chunk_index = 0
            saw_done = False
            finish_reason: Optional[str] = None
            output_tokens = 0

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                if not raw_line.startswith("data: "):
                    continue
                payload_text = raw_line[6:].strip()
                if payload_text == "[DONE]":
                    saw_done = True
                    break
                if not payload_text:
                    continue
                message = json.loads(payload_text)
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
                    state.first_token_ts = now
                    state.first_token_event.set()
                chunk_tokens = len(delta_token_ids)
                output_tokens += chunk_tokens
                chunk_gap_ms = None
                per_token_gap = None
                if prev_chunk_ts is not None:
                    chunk_gap_ms = (now - prev_chunk_ts) * 1000.0
                    if chunk_tokens > 0:
                        per_token_gap = chunk_gap_ms / float(chunk_tokens)
                        token_gaps.extend([per_token_gap] * chunk_tokens)
                prev_chunk_ts = now
                token_rows.append(
                    {
                        "case_id": case_id,
                        "repeat_index": repeat_index,
                        "request_id": request_id,
                        "tenant_id": event.tenant_id,
                        "role": event.role,
                        "workload": event.workload,
                        "measurement_window": 1,
                        "chunk_index": chunk_index,
                        "chunk_tokens": chunk_tokens,
                        "chunk_gap_ms": ""
                        if chunk_gap_ms is None else round(chunk_gap_ms, 4),
                        "estimated_per_token_gap_ms": ""
                        if per_token_gap is None else round(per_token_gap, 4),
                        "chunk_arrival_offset_s": round(now - run_start_ts, 6),
                    }
                )
                chunk_index += 1

            end_ts = time.monotonic()
            if first_token_ts is None:
                raise RuntimeError("No streamed completion tokens received")
            validate_completion_contract(
                output_tokens=output_tokens,
                requested_min_tokens=min_tokens,
                finish_reason=finish_reason,
                saw_done=saw_done,
                streaming=True,
            )
            row.update(
                {
                    "status": "success",
                    "output_tokens": output_tokens,
                    "ttft_ms": fmt((first_token_ts - start_ts) * 1000.0),
                    "p50_tbt_ms": fmt(percentile(token_gaps, 50)),
                    "p90_tbt_ms": fmt(percentile(token_gaps, 90)),
                    "p99_tbt_ms": fmt(percentile(token_gaps, 99)),
                    "completion_latency_ms": fmt((end_ts - start_ts) * 1000.0),
                    "time_to_last_token_ms": fmt((end_ts - start_ts) * 1000.0),
                }
            )
    except Exception as exc:  # noqa: BLE001 - write exception into experiment CSV
        row["error_message"] = repr(exc)
        state.error = repr(exc)
    finally:
        state.row = row
        state.token_rows = token_rows
        state.token_gaps = token_gaps
        state.done_event.set()


def run_prefill_request(
    *,
    base_url: str,
    model: str,
    case_id: str,
    repeat_index: int,
    event: ReplayEvent,
    request_id: str,
    run_start_ts: float,
    max_tokens: int,
    min_tokens: int,
    ignore_eos: bool,
    request_timeout_sec: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "case_id": case_id,
        "repeat_index": repeat_index,
        "role": event.role,
        "workload": event.workload,
        "background_workload": "loogle_prefill",
        "request_index": event.request_index,
        "request_id": request_id,
        "tenant_id": event.tenant_id,
        "status": "failed",
        "error_message": "",
        "measurement_window": 1,
        "scheduled_offset_s": "",
        "arrival_time_s": round(time.monotonic() - run_start_ts, 6),
        "input_tokens": event.request.input_tokens,
        "requested_min_tokens": min_tokens,
        "requested_max_tokens": max_tokens,
        "output_tokens": "",
        "ttft_ms": "",
        "p50_tbt_ms": "",
        "p90_tbt_ms": "",
        "p99_tbt_ms": "",
        "completion_latency_ms": "",
        "time_to_last_token_ms": "",
    }
    try:
        requests = get_requests_module()
        with requests.Session() as session:
            result = non_stream_completion(
                session=session,
                base_url=base_url,
                model=model,
                event=event,
                request_id=request_id,
                request_timeout_sec=request_timeout_sec,
                max_tokens_override=max_tokens,
                min_tokens_override=min_tokens,
                ignore_eos=ignore_eos,
            )
        row.update(
            {
                "status": "success",
                "output_tokens": result["output_tokens"],
                "completion_latency_ms": fmt(
                    float(result["completion_latency_ms"])),
                "time_to_last_token_ms": fmt(
                    float(result["time_to_last_token_ms"])),
            }
        )
    except Exception as exc:  # noqa: BLE001 - write exception into experiment CSV
        row["error_message"] = repr(exc)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Staged decode-then-prefill VRAM-contention replay.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--decode-jsonl", type=Path, required=True)
    parser.add_argument("--prefill-jsonl", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--repeat-index", type=int, default=1)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--token-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--decode-max-tokens", type=int, default=512)
    parser.add_argument("--decode-min-tokens", type=int, default=512)
    parser.add_argument("--prefill-max-tokens", type=int, default=1)
    parser.add_argument("--prefill-min-tokens", type=int, default=1)
    parser.add_argument("--decode-index", type=int, default=0)
    parser.add_argument("--prefill-index", type=int, default=1)
    parser.add_argument("--decode-context-target-tokens", type=int, default=0)
    parser.add_argument("--prefill-context-target-tokens", type=int, default=0)
    parser.add_argument("--first-token-timeout-sec", type=int, default=1800)
    parser.add_argument("--prefill-delay-after-first-token-sec",
                        type=float,
                        default=0.0)
    parser.add_argument("--context-token-window", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--request-timeout-sec", type=int, default=3600)
    parser.add_argument("--ignore-eos", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    requests = get_requests_module()
    with requests.Session() as session:
        model = args.model or get_model_id_from_server(session, base_url)

    decode_requests = load_prepared_jsonl(args.decode_jsonl)
    prefill_requests = load_prepared_jsonl(args.prefill_jsonl)
    decode_req = decode_requests[args.decode_index % len(decode_requests)]
    prefill_req = prefill_requests[args.prefill_index % len(prefill_requests)]
    with requests.Session() as session:
        decode_req = fit_request_to_token_window(
            session=session,
            base_url=base_url,
            model=model,
            request=decode_req,
            target_tokens=args.decode_context_target_tokens,
            context_window=args.context_token_window,
            max_model_len=args.max_model_len,
            max_output_tokens=args.decode_max_tokens,
            label="decode",
        )
        prefill_req = fit_request_to_token_window(
            session=session,
            base_url=base_url,
            model=model,
            request=prefill_req,
            target_tokens=args.prefill_context_target_tokens,
            context_window=args.context_token_window,
            max_model_len=args.max_model_len,
            max_output_tokens=args.prefill_max_tokens,
            label="prefill",
        )

    run_start_ts = time.monotonic() + 2.0
    delay = run_start_ts - time.monotonic()
    if delay > 0:
        time.sleep(delay)

    decode_event = make_event(
        role="foreground",
        workload="loogle_decode",
        request=decode_req,
        request_index=args.decode_index,
        tenant_id="fg_decode_tenant",
    )
    prefill_event = make_event(
        role="background",
        workload="loogle_prefill",
        request=prefill_req,
        request_index=args.prefill_index,
        tenant_id="bg_prefill_tenant",
    )

    decode_request_id = (
        f"{args.case_id}-decode-{args.repeat_index}-{args.decode_index}")
    prefill_request_id = (
        f"{args.case_id}-prefill-{args.repeat_index}-{args.prefill_index}")
    state = StreamState(threading.Event(), threading.Event())
    decode_thread = threading.Thread(
        target=stream_decode_request,
        kwargs={
            "base_url": base_url,
            "model": model,
            "case_id": args.case_id,
            "repeat_index": args.repeat_index,
            "event": decode_event,
            "request_id": decode_request_id,
            "run_start_ts": run_start_ts,
            "max_tokens": args.decode_max_tokens,
            "min_tokens": args.decode_min_tokens,
            "ignore_eos": bool(args.ignore_eos),
            "request_timeout_sec": args.request_timeout_sec,
            "state": state,
        },
        daemon=True,
    )
    decode_thread.start()

    first_token_deadline = time.monotonic() + args.first_token_timeout_sec
    while (
        not state.first_token_event.is_set()
        and not state.done_event.is_set()
        and time.monotonic() < first_token_deadline
    ):
        time.sleep(0.5)
    if not state.first_token_event.is_set():
        state.done_event.wait(5)
        raw_rows = [state.row] if state.row else []
        write_csv(args.raw_csv, RAW_FIELDNAMES, raw_rows)
        write_csv(args.token_csv, TOKEN_FIELDNAMES, state.token_rows or [])
        raise RuntimeError(
            "Decode request did not produce a first token before timeout; "
            "cannot inject staged prefill contention request. "
            f"decode_error={state.error!r}")

    first_token_ts = state.first_token_ts or time.monotonic()
    if args.prefill_delay_after_first_token_sec > 0:
        time.sleep(args.prefill_delay_after_first_token_sec)
    prefill_submit_ts = time.monotonic()
    prefill_row = run_prefill_request(
        base_url=base_url,
        model=model,
        case_id=args.case_id,
        repeat_index=args.repeat_index,
        event=prefill_event,
        request_id=prefill_request_id,
        run_start_ts=run_start_ts,
        max_tokens=args.prefill_max_tokens,
        min_tokens=args.prefill_min_tokens,
        ignore_eos=bool(args.ignore_eos),
        request_timeout_sec=args.request_timeout_sec,
    )

    state.done_event.wait(args.request_timeout_sec + 30)
    decode_thread.join(timeout=1)
    decode_row = state.row or {}
    token_rows = state.token_rows or []
    token_gaps = state.token_gaps or []

    raw_rows = [row for row in [decode_row, prefill_row] if row]
    write_csv(args.raw_csv, RAW_FIELDNAMES, raw_rows)
    write_csv(args.token_csv, TOKEN_FIELDNAMES, token_rows)
    summary = {
        "timestamp_utc": utc_now(),
        "case_id": args.case_id,
        "repeat_index": args.repeat_index,
        "decode_context_target_tokens": args.decode_context_target_tokens,
        "prefill_context_target_tokens": args.prefill_context_target_tokens,
        "decode_input_tokens": decode_req.input_tokens,
        "prefill_input_tokens": prefill_req.input_tokens,
        "decode_success": int(decode_row.get("status") == "success"),
        "prefill_success": int(prefill_row.get("status") == "success"),
        "decode_output_tokens": decode_row.get("output_tokens", ""),
        "decode_ttft_ms": decode_row.get("ttft_ms", ""),
        "decode_p50_tbt_ms": fmt(percentile(token_gaps, 50)),
        "decode_p90_tbt_ms": fmt(percentile(token_gaps, 90)),
        "decode_p99_tbt_ms": fmt(percentile(token_gaps, 99)),
        "decode_completion_latency_ms": decode_row.get(
            "completion_latency_ms", ""),
        "prefill_completion_latency_ms": prefill_row.get(
            "completion_latency_ms", ""),
        "first_token_to_prefill_submit_ms": fmt(
            (prefill_submit_ts - first_token_ts) * 1000.0),
        "decode_first_token_offset_s": fmt(first_token_ts - run_start_ts, 6),
        "prefill_submit_offset_s": fmt(prefill_submit_ts - run_start_ts, 6),
        "decode_requested_max_tokens": args.decode_max_tokens,
        "decode_requested_min_tokens": args.decode_min_tokens,
        "prefill_requested_max_tokens": args.prefill_max_tokens,
        "prefill_requested_min_tokens": args.prefill_min_tokens,
    }
    write_csv(args.summary_csv, SUMMARY_FIELDNAMES, [summary])
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(
        json.dumps(
            {
                "created_at": utc_now(),
                "base_url": base_url,
                "model": model,
                "case_id": args.case_id,
                "repeat_index": args.repeat_index,
                "decode_jsonl": str(args.decode_jsonl),
                "prefill_jsonl": str(args.prefill_jsonl),
                "decode_request_id": decode_request_id,
                "prefill_request_id": prefill_request_id,
                "decode_input_tokens": decode_req.input_tokens,
                "prefill_input_tokens": prefill_req.input_tokens,
                "decode_context_target_tokens":
                args.decode_context_target_tokens,
                "prefill_context_target_tokens":
                args.prefill_context_target_tokens,
                "first_token_to_prefill_submit_ms":
                summary["first_token_to_prefill_submit_ms"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "[SUMMARY] "
        f"case={args.case_id} repeat={args.repeat_index} "
        f"decode_success={summary['decode_success']} "
        f"prefill_success={summary['prefill_success']} "
        f"decode_p99_tbt_ms={summary['decode_p99_tbt_ms']} "
        f"first_token_to_prefill_submit_ms="
        f"{summary['first_token_to_prefill_submit_ms']}",
        flush=True,
    )
    print(f"[DONE] raw_csv={args.raw_csv}")
    print(f"[DONE] token_csv={args.token_csv}")
    print(f"[DONE] summary_csv={args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
