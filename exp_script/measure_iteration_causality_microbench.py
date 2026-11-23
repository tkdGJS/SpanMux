#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from measure_vllm_run import (
    append_csv_row,
    derive_tbt_p95_ms,
    detokenize_ids,
    exact_completion_token_ids,
    get_model_id,
    parse_sse_json,
    tokenize_text,
    utc_now,
)


@dataclass(frozen=True)
class RequestSpec:
    label: str
    role: str
    prompt_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    launch_mode: str
    requests: Tuple[RequestSpec, ...]
    background: Optional[RequestSpec] = None
    after_first_token: Tuple[RequestSpec, ...] = ()


REQUEST_FIELDNAMES = [
    "timestamp_utc",
    "case_id",
    "repeat_index",
    "launch_mode",
    "role",
    "request_label",
    "client_request_id",
    "server_response_id",
    "metrics_request_id",
    "tenant_id",
    "status",
    "error_message",
    "prompt_target_tokens",
    "prompt_tokens",
    "requested_min_tokens",
    "requested_max_tokens",
    "output_tokens",
    "ttft_ms_client",
    "p95_tbt_ms_client",
    "ttlt_ms_client",
    "first_token_offset_ms_client",
    "blocking_time_ms",
    "prefill_time_ms",
    "decode_time_ms",
    "ttft_ms_server",
    "ttlt_ms_server",
    "prefix_hit_tokens",
]

TOKEN_FIELDNAMES = [
    "case_id",
    "repeat_index",
    "role",
    "request_label",
    "client_request_id",
    "metrics_request_id",
    "chunk_index",
    "chunk_tokens",
    "chunk_gap_ms",
]

VALIDATION_FIELDNAMES = [
    "case_id",
    "repeat_index",
    "role",
    "request_label",
    "client_request_id",
    "metrics_request_id",
    "matched_timeline_request_ids",
    "first_prefill_iteration_id",
    "first_decode_iteration_id",
    "prefill_rows",
    "decode_rows",
    "prefill_rows_with_decode_context",
    "decode_rows_with_prefill_tokens",
    "max_token_batch_decode_context",
    "max_token_batch_prefill_tokens",
    "mean_iteration_model_wait_ms",
    "max_iteration_model_wait_ms",
    "first_prefill_model_wait_ms",
    "first_decode_model_wait_ms",
]

SUMMARY_FIELDNAMES = [
    "case_id",
    "role",
    "request_label",
    "successful_requests",
    "failed_requests",
    "avg_prompt_tokens",
    "avg_output_tokens",
    "avg_blocking_time_ms",
    "avg_prefill_time_ms",
    "avg_decode_time_ms",
    "avg_ttft_ms_server",
    "avg_ttlt_ms_server",
    "avg_p95_tbt_ms_client",
]


def causal_cases(short_tokens: int, mid_tokens: int, long_tokens: int,
                 prefill_output_tokens: int, decode_output_tokens: int,
                 simultaneous_prefill_launch_mode: str) -> Tuple[CaseSpec, ...]:
    return (
        CaseSpec(
            case_id="prefill_same_short3_control",
            launch_mode=simultaneous_prefill_launch_mode,
            requests=(
                RequestSpec("short_a", "victim", short_tokens,
                            prefill_output_tokens),
                RequestSpec("short_b", "peer", short_tokens,
                            prefill_output_tokens),
                RequestSpec("short_c", "peer", short_tokens,
                            prefill_output_tokens),
            ),
        ),
        CaseSpec(
            case_id="prefill_same_long3_control",
            launch_mode=simultaneous_prefill_launch_mode,
            requests=(
                RequestSpec("long_a", "victim", long_tokens,
                            prefill_output_tokens),
                RequestSpec("long_b", "peer", long_tokens,
                            prefill_output_tokens),
                RequestSpec("long_c", "peer", long_tokens,
                            prefill_output_tokens),
            ),
        ),
        CaseSpec(
            case_id="prefill_hetero3_short_victim",
            launch_mode=simultaneous_prefill_launch_mode,
            requests=(
                RequestSpec("short_victim", "victim", short_tokens,
                            prefill_output_tokens),
                RequestSpec("mid_peer", "peer", mid_tokens,
                            prefill_output_tokens),
                RequestSpec("long_peer", "peer", long_tokens,
                            prefill_output_tokens),
            ),
        ),
        CaseSpec(
            case_id="prefill_short_victim_alone_control",
            launch_mode="simultaneous",
            requests=(
                RequestSpec("short_prefill", "victim", short_tokens,
                            prefill_output_tokens),
            ),
        ),
        CaseSpec(
            case_id="prefill_short_victim_with_short_decode_control",
            launch_mode="staged_after_background_first_token",
            requests=(),
            background=RequestSpec("short_decode_bg", "peer", short_tokens,
                                   decode_output_tokens),
            after_first_token=(
                RequestSpec("short_prefill_victim", "victim", short_tokens,
                            prefill_output_tokens),
            ),
        ),
        CaseSpec(
            case_id="prefill_short_victim_with_long_decode",
            launch_mode="staged_after_background_first_token",
            requests=(),
            background=RequestSpec("long_decode_bg", "peer", long_tokens,
                                   decode_output_tokens),
            after_first_token=(
                RequestSpec("short_prefill_victim", "victim", short_tokens,
                            prefill_output_tokens),
            ),
        ),
        CaseSpec(
            case_id="decode_short_victim_alone_control",
            launch_mode="simultaneous",
            requests=(
                RequestSpec("short_decode", "victim", short_tokens,
                            decode_output_tokens),
            ),
        ),
        CaseSpec(
            case_id="decode_short_victim_with_short_prefill_control",
            launch_mode="staged_after_background_first_token",
            requests=(),
            background=RequestSpec("short_decode_victim", "victim", short_tokens,
                                   decode_output_tokens),
            after_first_token=(
                RequestSpec("short_prefill_peer", "peer", short_tokens,
                            prefill_output_tokens),
            ),
        ),
        CaseSpec(
            case_id="decode_short_victim_with_long_prefill",
            launch_mode="staged_after_background_first_token",
            requests=(),
            background=RequestSpec("short_decode_victim", "victim", short_tokens,
                                   decode_output_tokens),
            after_first_token=(
                RequestSpec("long_prefill_peer", "peer", long_tokens,
                            prefill_output_tokens),
            ),
        ),
    )


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def wait_for_request_metrics(metrics_jsonl_path: str, request_id: str,
                             timeout_sec: int = 180) -> Dict[str, Any]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if os.path.exists(metrics_jsonl_path):
            with open(metrics_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("request_id") == request_id:
                        return payload
        time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for metrics: {request_id}")


def build_exact_prompt(session: requests.Session, base_url: str, model: str,
                       target_tokens: int) -> str:
    seed = (
        " causal microbenchmark synthetic workload token pattern"
        " with deterministic repeated text"
    )
    repeat_count = max(512, target_tokens // 4)
    text = (seed * repeat_count).strip()
    count, token_ids = tokenize_text(session, base_url, model, text)
    while token_ids is None or count < target_tokens:
        repeat_count *= 2
        text = (seed * repeat_count).strip()
        count, token_ids = tokenize_text(session, base_url, model, text)
    assert token_ids is not None

    candidate = detokenize_ids(session, base_url, model,
                               token_ids[:target_tokens])
    if candidate is None:
        raise RuntimeError("Detokenize endpoint did not return text")
    actual_tokens, _ = tokenize_text(session, base_url, model, candidate)
    if actual_tokens == target_tokens:
        return candidate

    lower = max(1, target_tokens - 128)
    upper = min(len(token_ids), target_tokens + 128)
    for length in range(target_tokens, lower - 1, -1):
        candidate = detokenize_ids(session, base_url, model, token_ids[:length])
        if candidate is None:
            continue
        actual_tokens, _ = tokenize_text(session, base_url, model, candidate)
        if actual_tokens == target_tokens:
            return candidate
    for length in range(target_tokens + 1, upper + 1):
        candidate = detokenize_ids(session, base_url, model, token_ids[:length])
        if candidate is None:
            continue
        actual_tokens, _ = tokenize_text(session, base_url, model, candidate)
        if actual_tokens == target_tokens:
            return candidate

    raise RuntimeError(
        f"Failed to build exact prompt with {target_tokens} tokens; "
        f"initial detokenized prompt had {actual_tokens} tokens")


def stream_completion(
    session: requests.Session,
    base_url: str,
    model: str,
    prompt: str,
    tenant_id: str,
    min_tokens: int,
    max_tokens: int,
    request_id: str,
    request_timeout_sec: int,
    first_token_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    url = base_url + "/v1/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "stream": True,
        "return_token_ids": True,
        "request_id": request_id,
        "vllm_xargs": {"tenant_id": tenant_id},
    }
    headers = {"X-Request-Id": request_id}
    start_ts = time.monotonic()
    response = session.post(
        url,
        json=payload,
        headers=headers,
        stream=True,
        timeout=(10, max(5, request_timeout_sec)),
    )
    response.raise_for_status()

    first_token_ts = None
    prev_text_ts = None
    chunk_timings: List[Tuple[int, float]] = []
    completion_parts: List[str] = []
    completion_tokens = 0
    response_id = None

    for raw_line in response.iter_lines(decode_unicode=True):
        if time.monotonic() - start_ts > request_timeout_sec:
            response.close()
            raise TimeoutError(
                f"Request exceeded timeout_sec={request_timeout_sec}")
        if not raw_line:
            continue
        message = parse_sse_json(raw_line)
        if not message:
            continue
        if response_id is None and isinstance(message.get("id"), str):
            response_id = message["id"]
        choices = message.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta_token_ids = exact_completion_token_ids(choice)
        text = choice.get("text") or ""
        if delta_token_ids:
            now = time.monotonic()
            if first_token_ts is None:
                first_token_ts = now
                if first_token_event is not None:
                    first_token_event.set()
            if prev_text_ts is not None:
                chunk_timings.append(
                    (len(delta_token_ids), (now - prev_text_ts) * 1000.0)
                )
            prev_text_ts = now
            completion_tokens += len(delta_token_ids)
        if text:
            completion_parts.append(text)

    end_ts = time.monotonic()
    if first_token_ts is None:
        raise RuntimeError("No completion tokens received")
    return {
        "completion_text": "".join(completion_parts),
        "completion_tokens": completion_tokens,
        "ttft_ms": (first_token_ts - start_ts) * 1000.0,
        "ttlt_ms": (end_ts - start_ts) * 1000.0,
        "first_token_offset_ms": (first_token_ts - start_ts) * 1000.0,
        "chunk_timings": chunk_timings,
        "response_id": response_id or f"cmpl-{request_id}",
    }


def append_token_rows(
    token_csv_path: str,
    csv_lock: threading.Lock,
    session: requests.Session,
    base_url: str,
    model: str,
    row_base: Dict[str, Any],
    metrics_request_id: str,
    chunk_timings: List[Tuple[int, float]],
) -> None:
    del session, base_url, model
    for chunk_index, (chunk_tokens, gap_ms) in enumerate(chunk_timings, start=1):
        row = {
            **row_base,
            "metrics_request_id": metrics_request_id,
            "chunk_index": chunk_index,
            "chunk_tokens": chunk_tokens,
            "chunk_gap_ms": round(gap_ms, 6),
        }
        with csv_lock:
            append_csv_row(token_csv_path, TOKEN_FIELDNAMES, row)


def execute_request(
    spec: RequestSpec,
    *,
    case_id: str,
    repeat_index: int,
    launch_mode: str,
    base_url: str,
    model: str,
    prompts: Dict[int, str],
    metrics_jsonl_path: str,
    raw_csv_path: str,
    token_csv_path: str,
    csv_lock: threading.Lock,
    request_timeout_sec: int,
    start_event: Optional[threading.Event] = None,
    first_token_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    if start_event is not None:
        start_event.wait()

    request_session = requests.Session()
    tenant_id = f"causal_{case_id}_{repeat_index}_{spec.label}"
    client_request_id = (
        f"causal-{case_id}-r{repeat_index}-{spec.label}")
    row: Dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "case_id": case_id,
        "repeat_index": repeat_index,
        "launch_mode": launch_mode,
        "role": spec.role,
        "request_label": spec.label,
        "client_request_id": client_request_id,
        "server_response_id": "",
        "metrics_request_id": "",
        "tenant_id": tenant_id,
        "status": "failed",
        "error_message": "",
        "prompt_target_tokens": spec.prompt_tokens,
        "prompt_tokens": "",
        "requested_min_tokens": spec.output_tokens,
        "requested_max_tokens": spec.output_tokens,
        "output_tokens": "",
        "ttft_ms_client": "",
        "p95_tbt_ms_client": "",
        "ttlt_ms_client": "",
        "first_token_offset_ms_client": "",
        "blocking_time_ms": "",
        "prefill_time_ms": "",
        "decode_time_ms": "",
        "ttft_ms_server": "",
        "ttlt_ms_server": "",
        "prefix_hit_tokens": "",
    }
    try:
        prompt = prompts[spec.prompt_tokens]
        prompt_tokens, _ = tokenize_text(request_session, base_url, model, prompt)
        stream_result = stream_completion(
            request_session,
            base_url,
            model,
            prompt,
            tenant_id,
            spec.output_tokens,
            spec.output_tokens,
            client_request_id,
            request_timeout_sec,
            first_token_event,
        )
        metrics_request_id = f"{stream_result['response_id']}-0"
        p95_tbt_ms, output_tokens = derive_tbt_p95_ms(
            request_session,
            base_url,
            model,
            stream_result["completion_text"],
            stream_result["chunk_timings"],
            int(stream_result["completion_tokens"]),
        )
        request_metrics = wait_for_request_metrics(
            metrics_jsonl_path, metrics_request_id)
        row.update(
            {
                "metrics_request_id": metrics_request_id,
                "server_response_id": stream_result["response_id"],
                "status": "success",
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "ttft_ms_client": round(float(stream_result["ttft_ms"]), 4),
                "p95_tbt_ms_client": (
                    "" if p95_tbt_ms is None else round(float(p95_tbt_ms), 4)
                ),
                "ttlt_ms_client": round(float(stream_result["ttlt_ms"]), 4),
                "first_token_offset_ms_client": round(
                    float(stream_result["first_token_offset_ms"]), 4
                ),
                "blocking_time_ms": round(
                    float(request_metrics.get("queued_time_s", 0.0)) * 1000.0,
                    4,
                ),
                "prefill_time_ms": round(
                    float(request_metrics.get("prefill_time_s", 0.0)) * 1000.0,
                    4,
                ),
                "decode_time_ms": round(
                    float(request_metrics.get("decode_time_s", 0.0)) * 1000.0,
                    4,
                ),
                "ttft_ms_server": round(
                    float(request_metrics.get("ttft_s", 0.0)) * 1000.0, 4
                ),
                "ttlt_ms_server": round(
                    float(request_metrics.get("ttlt_s", 0.0)) * 1000.0, 4
                ),
                "prefix_hit_tokens": int(
                    request_metrics.get("prefix_hit_tokens", 0)
                ),
            }
        )
        token_row_base = {
            "case_id": case_id,
            "repeat_index": repeat_index,
            "role": spec.role,
            "request_label": spec.label,
            "client_request_id": client_request_id,
        }
        append_token_rows(
            token_csv_path,
            csv_lock,
            request_session,
            base_url,
            model,
            token_row_base,
            metrics_request_id,
            stream_result["chunk_timings"],
        )
        print(
            f"[RESULT] case={case_id} repeat={repeat_index} "
            f"role={spec.role} label={spec.label} prompt={prompt_tokens} "
            f"out={output_tokens} blocking_ms={row['blocking_time_ms']} "
            f"prefill_ms={row['prefill_time_ms']} "
            f"ttft_ms={row['ttft_ms_server']} "
            f"p95_tbt_ms={row['p95_tbt_ms_client']}",
            flush=True,
        )
    except Exception as exc:
        row["error_message"] = str(exc)
        print(
            f"[FAIL] case={case_id} repeat={repeat_index} "
            f"role={spec.role} label={spec.label}: {exc}",
            flush=True,
        )
    finally:
        with csv_lock:
            append_csv_row(raw_csv_path, REQUEST_FIELDNAMES, row)
    return row


def run_simultaneous_case(
    case: CaseSpec,
    repeat_index: int,
    args: argparse.Namespace,
    model: str,
    prompts: Dict[int, str],
    csv_lock: threading.Lock,
) -> List[Dict[str, Any]]:
    start_event = threading.Event()
    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(case.requests)) as executor:
        futures = [
            executor.submit(
                execute_request,
                spec,
                case_id=case.case_id,
                repeat_index=repeat_index,
                launch_mode=case.launch_mode,
                base_url=args.base_url,
                model=model,
                prompts=prompts,
                metrics_jsonl_path=args.metrics_jsonl,
                raw_csv_path=args.raw_csv,
                token_csv_path=args.token_csv,
                csv_lock=csv_lock,
                request_timeout_sec=args.request_timeout_sec,
                start_event=start_event,
            )
            for spec in case.requests
        ]
        time.sleep(args.launch_warmup_sec)
        start_event.set()
        for future in as_completed(futures):
            rows.append(future.result())
    return rows


def run_coalesced_after_prefill_barrier_case(
    case: CaseSpec,
    repeat_index: int,
    args: argparse.Namespace,
    model: str,
    prompts: Dict[int, str],
    csv_lock: threading.Lock,
) -> List[Dict[str, Any]]:
    barrier_tokens = args.coalesce_barrier_prompt_tokens
    barrier_prompt = prompts[barrier_tokens]
    barrier_request_id = f"causal-{case.case_id}-r{repeat_index}-coalesce-barrier"
    barrier_tenant_id = f"causal_{case.case_id}_{repeat_index}_coalesce_barrier"

    def run_barrier() -> Dict[str, Any]:
        session = requests.Session()
        prompt_tokens, _ = tokenize_text(session, args.base_url, model,
                                         barrier_prompt)
        print(
            f"[COALESCE] case={case.case_id} repeat={repeat_index} "
            f"barrier_start prompt={prompt_tokens} "
            f"out={args.coalesce_barrier_output_tokens}",
            flush=True,
        )
        result = stream_completion(
            session,
            args.base_url,
            model,
            barrier_prompt,
            barrier_tenant_id,
            args.coalesce_barrier_output_tokens,
            args.coalesce_barrier_output_tokens,
            barrier_request_id,
            args.request_timeout_sec,
        )
        print(
            f"[COALESCE] case={case.case_id} repeat={repeat_index} "
            f"barrier_done response_id={result['response_id']} "
            f"ttft_ms={round(float(result['ttft_ms']), 4)}",
            flush=True,
        )
        return result

    with ThreadPoolExecutor(max_workers=1) as barrier_executor:
        barrier_future = barrier_executor.submit(run_barrier)
        time.sleep(args.coalesce_barrier_settle_ms / 1000.0)
        rows = run_simultaneous_case(
            case, repeat_index, args, model, prompts, csv_lock
        )
        barrier_future.result()
    return rows


def run_staged_case(
    case: CaseSpec,
    repeat_index: int,
    args: argparse.Namespace,
    model: str,
    prompts: Dict[int, str],
    csv_lock: threading.Lock,
) -> List[Dict[str, Any]]:
    assert case.background is not None
    first_token_event = threading.Event()
    rows: List[Dict[str, Any]] = []
    max_workers = 1 + len(case.after_first_token)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        background_future = executor.submit(
            execute_request,
            case.background,
            case_id=case.case_id,
            repeat_index=repeat_index,
            launch_mode=case.launch_mode,
            base_url=args.base_url,
            model=model,
            prompts=prompts,
            metrics_jsonl_path=args.metrics_jsonl,
            raw_csv_path=args.raw_csv,
            token_csv_path=args.token_csv,
            csv_lock=csv_lock,
            request_timeout_sec=args.request_timeout_sec,
            first_token_event=first_token_event,
        )
        if not first_token_event.wait(args.background_first_token_timeout_sec):
            rows.append(background_future.result())
            raise RuntimeError(
                f"Background request did not reach first token in case={case.case_id}"
            )
        if args.post_first_token_delay_ms > 0:
            time.sleep(args.post_first_token_delay_ms / 1000.0)
        futures = [background_future]
        for spec in case.after_first_token:
            futures.append(
                executor.submit(
                    execute_request,
                    spec,
                    case_id=case.case_id,
                    repeat_index=repeat_index,
                    launch_mode=case.launch_mode,
                    base_url=args.base_url,
                    model=model,
                    prompts=prompts,
                    metrics_jsonl_path=args.metrics_jsonl,
                    raw_csv_path=args.raw_csv,
                    token_csv_path=args.token_csv,
                    csv_lock=csv_lock,
                    request_timeout_sec=args.request_timeout_sec,
                )
            )
        for future in as_completed(futures):
            rows.append(future.result())
    return rows


def numeric_mean(rows: List[Dict[str, Any]], key: str) -> str:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            continue
        values.append(float(value))
    if not values:
        return ""
    return str(round(statistics.mean(values), 4))


def write_summary(raw_csv: str, summary_csv: str) -> None:
    if not os.path.exists(raw_csv):
        return
    with open(raw_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (row["case_id"], row["role"], row["request_label"]), []
        ).append(row)
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for key in sorted(grouped):
            group_rows = grouped[key]
            success_rows = [row for row in group_rows if row["status"] == "success"]
            writer.writerow(
                {
                    "case_id": key[0],
                    "role": key[1],
                    "request_label": key[2],
                    "successful_requests": len(success_rows),
                    "failed_requests": len(group_rows) - len(success_rows),
                    "avg_prompt_tokens": numeric_mean(success_rows,
                                                       "prompt_tokens"),
                    "avg_output_tokens": numeric_mean(success_rows,
                                                       "output_tokens"),
                    "avg_blocking_time_ms": numeric_mean(
                        success_rows, "blocking_time_ms"
                    ),
                    "avg_prefill_time_ms": numeric_mean(
                        success_rows, "prefill_time_ms"
                    ),
                    "avg_decode_time_ms": numeric_mean(
                        success_rows, "decode_time_ms"
                    ),
                    "avg_ttft_ms_server": numeric_mean(
                        success_rows, "ttft_ms_server"
                    ),
                    "avg_ttlt_ms_server": numeric_mean(
                        success_rows, "ttlt_ms_server"
                    ),
                    "avg_p95_tbt_ms_client": numeric_mean(
                        success_rows, "p95_tbt_ms_client"
                    ),
                }
            )


def write_validation_csv(raw_csv: str, iteration_jsonl: str,
                         decode_timeline_jsonl: str,
                         validation_csv: str) -> None:
    if not (
        os.path.exists(raw_csv)
        and os.path.exists(iteration_jsonl)
        and os.path.exists(decode_timeline_jsonl)
    ):
        return
    with open(raw_csv, "r", encoding="utf-8", newline="") as f:
        raw_rows = [row for row in csv.DictReader(f) if row["status"] == "success"]
    iteration_rows = {
        int(row.get("iteration_id", -1)): row
        for row in iter_jsonl(Path(iteration_jsonl))
    }
    timeline_by_request: Dict[str, List[Dict[str, Any]]] = {}
    for row in iter_jsonl(Path(decode_timeline_jsonl)):
        request_id = str(row.get("request_id", ""))
        if request_id:
            timeline_by_request.setdefault(request_id, []).append(row)

    def matching_timeline_rows(raw: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[str]]:
        metrics_request_id = raw.get("metrics_request_id", "")
        client_request_id = raw.get("client_request_id", "")
        server_response_id = raw.get("server_response_id", "")
        candidates = {
            metrics_request_id,
            client_request_id,
            server_response_id,
            f"{server_response_id}-0" if server_response_id else "",
            f"cmpl-{client_request_id}" if client_request_id else "",
            f"cmpl-{client_request_id}-0" if client_request_id else "",
        }
        if metrics_request_id.endswith("-0"):
            candidates.add(metrics_request_id[:-2])
        candidates = {value for value in candidates if value}

        matched_ids: List[str] = []
        matched_rows: List[Dict[str, Any]] = []
        for timeline_request_id, timeline_rows in timeline_by_request.items():
            for candidate in candidates:
                if (
                    timeline_request_id == candidate
                    or timeline_request_id.startswith(f"{candidate}-")
                    or candidate.startswith(f"{timeline_request_id}-")
                    or candidate in timeline_request_id
                ):
                    matched_ids.append(timeline_request_id)
                    matched_rows.extend(timeline_rows)
                    break
        matched_ids.sort()
        matched_rows.sort(
            key=lambda row: (
                int(row.get("iteration_id", -1)),
                str(row.get("request_id", "")),
            )
        )
        return matched_rows, matched_ids

    out_rows = []
    for raw in raw_rows:
        request_id = raw["metrics_request_id"]
        rows, matched_ids = matching_timeline_rows(raw)
        prefill_rows = [row for row in rows if row.get("phase") == "prefill"]
        decode_rows = [row for row in rows if row.get("phase") == "decode"]
        iteration_ids = sorted({
            int(row.get("iteration_id", -1))
            for row in rows
            if int(row.get("iteration_id", -1)) >= 0
        })
        waits_ms = [
            float(iteration_rows[it].get("model_wait_duration_s", 0.0)) * 1000.0
            for it in iteration_ids
            if it in iteration_rows
        ]
        first_prefill = min(
            [int(row.get("iteration_id", -1)) for row in prefill_rows],
            default=-1,
        )
        first_decode = min(
            [int(row.get("iteration_id", -1)) for row in decode_rows],
            default=-1,
        )
        out_rows.append(
            {
                "case_id": raw["case_id"],
                "repeat_index": raw["repeat_index"],
                "role": raw["role"],
                "request_label": raw["request_label"],
                "client_request_id": raw["client_request_id"],
                "metrics_request_id": request_id,
                "matched_timeline_request_ids": ";".join(matched_ids),
                "first_prefill_iteration_id": first_prefill,
                "first_decode_iteration_id": first_decode,
                "prefill_rows": len(prefill_rows),
                "decode_rows": len(decode_rows),
                "prefill_rows_with_decode_context": sum(
                    1
                    for row in prefill_rows
                    if float(row.get("token_batch_decode_context_max", 0) or 0) > 0
                ),
                "decode_rows_with_prefill_tokens": sum(
                    1
                    for row in decode_rows
                    if float(row.get("token_batch_prefill_tokens", 0) or 0) > 0
                ),
                "max_token_batch_decode_context": max(
                    [
                        float(row.get("token_batch_decode_context_max", 0) or 0)
                        for row in rows
                    ],
                    default=0,
                ),
                "max_token_batch_prefill_tokens": max(
                    [
                        float(row.get("token_batch_prefill_tokens", 0) or 0)
                        for row in rows
                    ],
                    default=0,
                ),
                "mean_iteration_model_wait_ms": (
                    "" if not waits_ms else round(statistics.mean(waits_ms), 4)
                ),
                "max_iteration_model_wait_ms": (
                    "" if not waits_ms else round(max(waits_ms), 4)
                ),
                "first_prefill_model_wait_ms": (
                    ""
                    if first_prefill not in iteration_rows
                    else round(
                        float(
                            iteration_rows[first_prefill].get(
                                "model_wait_duration_s", 0.0
                            )
                        )
                        * 1000.0,
                        4,
                    )
                ),
                "first_decode_model_wait_ms": (
                    ""
                    if first_decode not in iteration_rows
                    else round(
                        float(
                            iteration_rows[first_decode].get(
                                "model_wait_duration_s", 0.0
                            )
                        )
                        * 1000.0,
                        4,
                    )
                ),
            }
        )

    Path(validation_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(validation_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=VALIDATION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--token-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--metrics-jsonl", required=True)
    parser.add_argument("--iteration-jsonl", required=True)
    parser.add_argument("--decode-timeline-jsonl", required=True)
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument("--short-prompt-tokens", type=int, default=512)
    parser.add_argument("--mid-prompt-tokens", type=int, default=12288)
    parser.add_argument("--long-prompt-tokens", type=int, default=24576)
    parser.add_argument("--prefill-output-tokens", type=int, default=1)
    parser.add_argument("--decode-output-tokens", type=int, default=512)
    parser.add_argument("--request-timeout-sec", type=int, default=1800)
    parser.add_argument("--background-first-token-timeout-sec", type=int,
                        default=900)
    parser.add_argument("--launch-warmup-sec", type=float, default=0.2)
    parser.add_argument("--post-first-token-delay-ms", type=int, default=0)
    parser.add_argument("--case-sleep-sec", type=float, default=30.0)
    parser.add_argument("--coalesce-barrier-prompt-tokens", type=int, default=0)
    parser.add_argument("--coalesce-barrier-output-tokens", type=int, default=1)
    parser.add_argument("--coalesce-barrier-settle-ms", type=int, default=1000)
    parser.add_argument(
        "--simultaneous-prefill-launch-mode",
        choices=("simultaneous", "coalesced_after_prefill_barrier"),
        default="simultaneous",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.coalesce_barrier_prompt_tokens <= 0:
        args.coalesce_barrier_prompt_tokens = args.mid_prompt_tokens
    session = requests.Session()
    model = get_model_id(session, args.base_url)
    cases = causal_cases(
        args.short_prompt_tokens,
        args.mid_prompt_tokens,
        args.long_prompt_tokens,
        args.prefill_output_tokens,
        args.decode_output_tokens,
        args.simultaneous_prefill_launch_mode,
    )
    prompt_lengths = sorted({
        req.prompt_tokens
        for case in cases
        for req in (*case.requests, *((case.background,) if case.background else ()),
                    *case.after_first_token)
    })
    prompt_lengths = sorted({*prompt_lengths, args.coalesce_barrier_prompt_tokens})
    prompts: Dict[int, str] = {}
    for length in prompt_lengths:
        print(f"[PROMPT] building exact prompt tokens={length}", flush=True)
        prompt = build_exact_prompt(session, args.base_url, model, length)
        actual, _ = tokenize_text(session, args.base_url, model, prompt)
        print(f"[PROMPT] target={length} actual={actual}", flush=True)
        prompts[length] = prompt

    for path in [
        args.raw_csv,
        args.summary_csv,
        args.token_csv,
        args.validation_csv,
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    csv_lock = threading.Lock()
    for repeat_index in range(1, args.repeat_count + 1):
        for case in cases:
            print(
                f"[CASE] repeat={repeat_index}/{args.repeat_count} "
                f"case={case.case_id} mode={case.launch_mode}",
                flush=True,
            )
            if case.launch_mode == "simultaneous":
                run_simultaneous_case(
                    case, repeat_index, args, model, prompts, csv_lock
                )
            elif case.launch_mode == "coalesced_after_prefill_barrier":
                run_coalesced_after_prefill_barrier_case(
                    case, repeat_index, args, model, prompts, csv_lock
                )
            elif case.launch_mode == "staged_after_background_first_token":
                run_staged_case(
                    case, repeat_index, args, model, prompts, csv_lock
                )
            else:
                raise RuntimeError(f"Unknown launch mode: {case.launch_mode}")
            if args.case_sleep_sec > 0:
                time.sleep(args.case_sleep_sec)

    write_summary(args.raw_csv, args.summary_csv)
    write_validation_csv(
        args.raw_csv,
        args.iteration_jsonl,
        args.decode_timeline_jsonl,
        args.validation_csv,
    )
    print(f"[DONE] raw_csv={args.raw_csv}", flush=True)
    print(f"[DONE] summary_csv={args.summary_csv}", flush=True)
    print(f"[DONE] token_csv={args.token_csv}", flush=True)
    print(f"[DONE] validation_csv={args.validation_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
