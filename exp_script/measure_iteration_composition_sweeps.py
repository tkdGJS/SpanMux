#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests

from measure_iteration_causality_microbench import (
    RequestSpec,
    build_exact_prompt,
    execute_request,
    iter_jsonl,
    write_summary,
    write_validation_csv,
)
from measure_vllm_run import get_model_id, tokenize_text


COMPOSITION_FIELDNAMES = [
    "axis",
    "case_id",
    "repeat_index",
    "ratio_percent",
    "expected_prefill_scheduled_tokens",
    "expected_decode_scheduled_tokens",
    "expected_total_scheduled_tokens",
    "expected_prefill_proxy",
    "expected_decode_proxy",
    "expected_total_proxy",
    "prefill_request_count",
    "decode_request_count",
    "mixed_valid",
    "mixed_iteration_id",
    "observed_scheduled_total_tokens",
    "observed_scheduled_prefill_tokens",
    "observed_scheduled_decode_tokens",
    "observed_prefill_attention_work_proxy",
    "observed_decode_kv_read_token_proxy",
    "observed_decode_context_sum",
    "observed_decode_context_max",
    "observed_num_prefill_reqs",
    "observed_num_decode_reqs",
    "mixed_iteration_model_wait_ms",
    "mixed_iteration_duration_ms",
    "request_material",
]


@dataclass(frozen=True)
class CompositionCase:
    axis: str
    case_id: str
    ratio_percent: int
    requests: Tuple[RequestSpec, ...]
    expected_prefill_scheduled_tokens: int
    expected_decode_scheduled_tokens: int
    expected_total_scheduled_tokens: int
    expected_prefill_proxy: float
    expected_decode_proxy: float
    expected_total_proxy: float
    prefill_request_count: int
    decode_request_count: int


def parse_int_list(value: str) -> List[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def prefill_proxy(tokens: int) -> float:
    return float(tokens * (tokens + 1) // 2)


def inverse_prefill_proxy(target_proxy: float) -> int:
    if target_proxy <= 0:
        return 0
    return max(1, int((math.sqrt(1.0 + 8.0 * target_proxy) - 1.0) // 2))


def split_decode_contexts(total_context: int, max_context: int,
                          max_reqs: int) -> List[int]:
    if total_context <= 0:
        return []
    if max_context <= 0:
        raise ValueError("max decode context must be positive")
    contexts = []
    remaining = total_context
    while remaining > 0 and len(contexts) < max_reqs:
        current = min(max_context, remaining)
        contexts.append(current)
        remaining -= current
    if remaining > 0:
        raise ValueError(
            "decode proxy target requires more decode requests than allowed: "
            f"remaining={remaining}, max_context={max_context}, "
            f"max_reqs={max_reqs}")
    return contexts


def build_scheduled_total_cases(args: argparse.Namespace) -> List[CompositionCase]:
    cases: List[CompositionCase] = []
    for ratio in args.ratio_percents:
        prefill_tokens = int(round(args.scheduled_total_tokens * ratio / 100.0))
        decode_tokens = args.scheduled_total_tokens - prefill_tokens
        requests: List[RequestSpec] = [
            RequestSpec(
                f"prefill_{ratio:02d}",
                "prefill",
                prefill_tokens,
                args.prefill_output_tokens,
            )
        ]
        for index in range(1, decode_tokens + 1):
            requests.append(
                RequestSpec(
                    f"decode_{ratio:02d}_{index:05d}",
                    "decode",
                    args.decode_context_tokens,
                    args.decode_output_tokens,
                )
            )
        cases.append(
            CompositionCase(
                axis="scheduled_total_tokens_ratio",
                case_id=f"scheduled_total_ratio_p{ratio:02d}_d{100-ratio:02d}",
                ratio_percent=ratio,
                requests=tuple(requests),
                expected_prefill_scheduled_tokens=prefill_tokens,
                expected_decode_scheduled_tokens=decode_tokens,
                expected_total_scheduled_tokens=args.scheduled_total_tokens,
                expected_prefill_proxy=prefill_proxy(prefill_tokens),
                expected_decode_proxy=float(decode_tokens * args.decode_context_tokens),
                expected_total_proxy=(
                    prefill_proxy(prefill_tokens)
                    + float(decode_tokens * args.decode_context_tokens)
                ),
                prefill_request_count=1,
                decode_request_count=decode_tokens,
            )
        )
    return cases


def build_work_proxy_cases(args: argparse.Namespace) -> List[CompositionCase]:
    cases: List[CompositionCase] = []
    max_decode_reqs = max(1, args.max_num_seqs - 1)
    for ratio in args.ratio_percents:
        prefill_proxy_target = args.work_proxy_total * ratio / 100.0
        prefill_tokens = inverse_prefill_proxy(prefill_proxy_target)
        actual_prefill_proxy = prefill_proxy(prefill_tokens)
        decode_proxy_target = max(
            1,
            int(round(args.work_proxy_total - actual_prefill_proxy)),
        )
        decode_contexts = split_decode_contexts(
            decode_proxy_target,
            args.work_proxy_max_decode_context_tokens,
            max_decode_reqs,
        )
        requests: List[RequestSpec] = [
            RequestSpec(
                f"prefill_{ratio:02d}",
                "prefill",
                prefill_tokens,
                args.prefill_output_tokens,
            )
        ]
        for index, context_tokens in enumerate(decode_contexts, start=1):
            requests.append(
                RequestSpec(
                    f"decode_{ratio:02d}_{index:02d}",
                    "decode",
                    context_tokens,
                    args.decode_output_tokens,
                )
            )
        cases.append(
            CompositionCase(
                axis="work_proxy_composition",
                case_id=f"work_proxy_ratio_p{ratio:02d}_d{100-ratio:02d}",
                ratio_percent=ratio,
                requests=tuple(requests),
                expected_prefill_scheduled_tokens=prefill_tokens,
                expected_decode_scheduled_tokens=len(decode_contexts),
                expected_total_scheduled_tokens=prefill_tokens + len(decode_contexts),
                expected_prefill_proxy=actual_prefill_proxy,
                expected_decode_proxy=float(sum(decode_contexts)),
                expected_total_proxy=actual_prefill_proxy + float(sum(decode_contexts)),
                prefill_request_count=1,
                decode_request_count=len(decode_contexts),
            )
        )
    return cases


def validate_case_feasibility(case: CompositionCase, args: argparse.Namespace) -> None:
    request_count = len(case.requests)
    if request_count > args.max_num_seqs:
        raise ValueError(
            f"case={case.case_id} needs {request_count} concurrent seqs "
            f"({case.prefill_request_count} prefill + "
            f"{case.decode_request_count} decode), but max_num_seqs="
            f"{args.max_num_seqs}. For exact scheduled_total_tokens ratio "
            "sweep, decode share requires one decode request per decode token."
        )
    if request_count > args.max_concurrent_requests:
        raise ValueError(
            f"case={case.case_id} needs {request_count} concurrent client "
            f"requests, but max_concurrent_requests="
            f"{args.max_concurrent_requests}. Increase the override only if "
            "the server and OS can handle it.")
    for req in case.requests:
        if req.prompt_tokens > args.max_model_len:
            raise ValueError(
                f"case={case.case_id} request={req.label} prompt_tokens="
                f"{req.prompt_tokens} exceeds max_model_len={args.max_model_len}")


def run_composition_case(
    case: CompositionCase,
    repeat_index: int,
    args: argparse.Namespace,
    model: str,
    prompts: Dict[int, str],
    csv_lock: threading.Lock,
) -> List[Dict[str, Any]]:
    decode_requests = [req for req in case.requests if req.role == "decode"]
    prefill_requests = [req for req in case.requests if req.role == "prefill"]
    rows: List[Dict[str, Any]] = []
    decode_ready_events = [threading.Event() for _ in decode_requests]
    max_workers = len(decode_requests) + len(prefill_requests)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for req, first_token_event in zip(decode_requests, decode_ready_events):
            futures.append(
                executor.submit(
                    execute_request,
                    req,
                    case_id=case.case_id,
                    repeat_index=repeat_index,
                    launch_mode="decode_ready_then_prefill",
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
            )
        deadline = time.time() + args.background_first_token_timeout_sec
        for event in decode_ready_events:
            remaining = max(0.0, deadline - time.time())
            if not event.wait(remaining):
                raise RuntimeError(
                    f"Timed out waiting for decode backgrounds in {case.case_id}")
        if args.post_decode_ready_delay_ms > 0:
            time.sleep(args.post_decode_ready_delay_ms / 1000.0)
        start_event = threading.Event()
        for req in prefill_requests:
            futures.append(
                executor.submit(
                    execute_request,
                    req,
                    case_id=case.case_id,
                    repeat_index=repeat_index,
                    launch_mode="decode_ready_then_prefill",
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
            )
        time.sleep(args.launch_warmup_sec)
        start_event.set()
        for future in as_completed(futures):
            rows.append(future.result())
    return rows


def request_id_matches(request_id: str, case_id: str, repeat_index: int) -> bool:
    return f"cmpl-causal-{case_id}-r{repeat_index}-" in request_id


def material_label(row: Dict[str, Any]) -> str:
    phase = row.get("phase", "")
    request_id = str(row.get("request_id", ""))
    label = request_id.split(f"-r{row.get('repeat_index', '')}-")[-1]
    return (
        f"{phase}:scheduled={int(float(row.get('scheduled_tokens', 0) or 0))}:"
        f"context={int(float(row.get('token_request_context_len', 0) or 0))}:"
        f"{label}"
    )


def select_mixed_iteration(
    case: CompositionCase,
    repeat_index: int,
    iteration_rows: Dict[int, Dict[str, Any]],
    timeline_rows: List[Dict[str, Any]],
) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]]]:
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for row in timeline_rows:
        request_id = str(row.get("request_id", ""))
        if not request_id_matches(request_id, case.case_id, repeat_index):
            continue
        iteration_id = int(float(row.get("iteration_id", -1)))
        if iteration_id >= 0:
            groups.setdefault(iteration_id, []).append(row)

    candidates = []
    for iteration_id, rows in groups.items():
        prefill_tokens = sum(float(row.get("prefill_tokens", 0) or 0)
                             for row in rows)
        decode_tokens = sum(float(row.get("decode_tokens", 0) or 0)
                            for row in rows)
        if prefill_tokens <= 0 or decode_tokens <= 0:
            continue
        iteration = iteration_rows.get(iteration_id, {})
        scheduled_prefill = float(
            iteration.get("scheduled_prefill_tokens", prefill_tokens) or 0)
        scheduled_decode = float(
            iteration.get("scheduled_decode_tokens", decode_tokens) or 0)
        prefill_proxy_value = float(
            iteration.get("prefill_attention_work_proxy", 0.0) or 0.0)
        decode_proxy_value = float(
            iteration.get("decode_kv_read_token_proxy", 0.0) or 0.0)
        if case.axis == "scheduled_total_tokens_ratio":
            score = (
                abs(scheduled_prefill - case.expected_prefill_scheduled_tokens)
                + abs(scheduled_decode - case.expected_decode_scheduled_tokens)
            )
        else:
            score = (
                abs(prefill_proxy_value - case.expected_prefill_proxy)
                / max(1.0, case.expected_prefill_proxy)
                + abs(decode_proxy_value - case.expected_decode_proxy)
                / max(1.0, case.expected_decode_proxy)
            )
        candidates.append((score, iteration_id, iteration, rows))
    if not candidates:
        return -1, {}, []
    _, iteration_id, iteration, rows = min(candidates, key=lambda item: item[0])
    return iteration_id, iteration, rows


def write_composition_summary(
    cases: List[CompositionCase],
    repeat_count: int,
    iteration_jsonl: str,
    decode_timeline_jsonl: str,
    output_csv: str,
    validation_tolerance_ratio: float,
) -> None:
    iteration_rows = {
        int(row.get("iteration_id", -1)): row
        for row in iter_jsonl(Path(iteration_jsonl))
    }
    timeline_rows = list(iter_jsonl(Path(decode_timeline_jsonl)))
    out_rows = []
    for repeat_index in range(1, repeat_count + 1):
        for case in cases:
            iteration_id, iteration, rows = select_mixed_iteration(
                case, repeat_index, iteration_rows, timeline_rows)
            observed_prefill = float(
                iteration.get("scheduled_prefill_tokens", 0.0) or 0.0)
            observed_decode = float(
                iteration.get("scheduled_decode_tokens", 0.0) or 0.0)
            observed_prefill_proxy = float(
                iteration.get("prefill_attention_work_proxy", 0.0) or 0.0)
            observed_decode_proxy = float(
                iteration.get("decode_kv_read_token_proxy", 0.0) or 0.0)
            if iteration_id < 0:
                mixed_valid = 0
            elif case.axis == "scheduled_total_tokens_ratio":
                mixed_valid = int(
                    abs(observed_prefill - case.expected_prefill_scheduled_tokens)
                    <= max(1.0, case.expected_prefill_scheduled_tokens
                           * validation_tolerance_ratio)
                    and abs(observed_decode - case.expected_decode_scheduled_tokens)
                    <= max(1.0, case.expected_decode_scheduled_tokens
                           * validation_tolerance_ratio)
                )
            else:
                mixed_valid = int(
                    abs(observed_prefill_proxy - case.expected_prefill_proxy)
                    <= max(1.0, case.expected_prefill_proxy
                           * validation_tolerance_ratio)
                    and abs(observed_decode_proxy - case.expected_decode_proxy)
                    <= max(1.0, case.expected_decode_proxy
                           * validation_tolerance_ratio)
                )
            out_rows.append({
                "axis": case.axis,
                "case_id": case.case_id,
                "repeat_index": repeat_index,
                "ratio_percent": case.ratio_percent,
                "expected_prefill_scheduled_tokens": (
                    case.expected_prefill_scheduled_tokens
                ),
                "expected_decode_scheduled_tokens": (
                    case.expected_decode_scheduled_tokens
                ),
                "expected_total_scheduled_tokens": (
                    case.expected_total_scheduled_tokens
                ),
                "expected_prefill_proxy": round(case.expected_prefill_proxy, 4),
                "expected_decode_proxy": round(case.expected_decode_proxy, 4),
                "expected_total_proxy": round(case.expected_total_proxy, 4),
                "prefill_request_count": case.prefill_request_count,
                "decode_request_count": case.decode_request_count,
                "mixed_valid": mixed_valid,
                "mixed_iteration_id": iteration_id,
                "observed_scheduled_total_tokens": float(
                    iteration.get("scheduled_total_tokens", 0.0) or 0.0),
                "observed_scheduled_prefill_tokens": observed_prefill,
                "observed_scheduled_decode_tokens": observed_decode,
                "observed_prefill_attention_work_proxy": (
                    observed_prefill_proxy
                ),
                "observed_decode_kv_read_token_proxy": observed_decode_proxy,
                "observed_decode_context_sum": float(
                    iteration.get("decode_context_sum", 0.0) or 0.0),
                "observed_decode_context_max": float(
                    iteration.get("decode_context_max", 0.0) or 0.0),
                "observed_num_prefill_reqs": float(
                    iteration.get("num_prefill_reqs", 0.0) or 0.0),
                "observed_num_decode_reqs": float(
                    iteration.get("num_decode_reqs", 0.0) or 0.0),
                "mixed_iteration_model_wait_ms": round(
                    float(iteration.get("model_wait_duration_s", 0.0) or 0.0)
                    * 1000.0,
                    4,
                ),
                "mixed_iteration_duration_ms": round(
                    float(iteration.get("iteration_duration_s", 0.0) or 0.0)
                    * 1000.0,
                    4,
                ),
                "request_material": " | ".join(material_label(row)
                                               for row in rows),
            })
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPOSITION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--token-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--composition-summary-csv", required=True)
    parser.add_argument("--metrics-jsonl", required=True)
    parser.add_argument("--iteration-jsonl", required=True)
    parser.add_argument("--decode-timeline-jsonl", required=True)
    parser.add_argument(
        "--axis",
        choices=("scheduled_total_tokens_ratio", "work_proxy_composition"),
        required=True,
    )
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument("--ratio-percents", default="10,20,30,40,50,60,70,80,90")
    parser.add_argument("--scheduled-total-tokens", type=int, default=16384)
    parser.add_argument("--work-proxy-total", type=int, default=131072)
    parser.add_argument("--decode-context-tokens", type=int, default=512)
    parser.add_argument("--work-proxy-max-decode-context-tokens", type=int,
                        default=24576)
    parser.add_argument("--prefill-output-tokens", type=int, default=1)
    parser.add_argument("--decode-output-tokens", type=int, default=128)
    parser.add_argument("--request-timeout-sec", type=int, default=2400)
    parser.add_argument("--background-first-token-timeout-sec", type=int,
                        default=1200)
    parser.add_argument("--launch-warmup-sec", type=float, default=0.2)
    parser.add_argument("--post-decode-ready-delay-ms", type=int, default=0)
    parser.add_argument("--case-sleep-sec", type=float, default=30.0)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-concurrent-requests", type=int, default=64)
    parser.add_argument("--validation-tolerance-ratio", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.ratio_percents = parse_int_list(args.ratio_percents)
    session = requests.Session()
    model = get_model_id(session, args.base_url)

    if args.axis == "scheduled_total_tokens_ratio":
        cases = build_scheduled_total_cases(args)
    else:
        cases = build_work_proxy_cases(args)

    for case in cases:
        validate_case_feasibility(case, args)

    prompt_lengths = sorted({
        request.prompt_tokens for case in cases for request in case.requests
    })
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
        args.composition_summary_csv,
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    csv_lock = threading.Lock()
    for repeat_index in range(1, args.repeat_count + 1):
        for case in cases:
            print(
                f"[CASE] repeat={repeat_index}/{args.repeat_count} "
                f"axis={case.axis} ratio={case.ratio_percent} "
                f"prefill_reqs={case.prefill_request_count} "
                f"decode_reqs={case.decode_request_count} "
                f"expected_scheduled=({case.expected_prefill_scheduled_tokens},"
                f"{case.expected_decode_scheduled_tokens}) "
                f"expected_proxy=({round(case.expected_prefill_proxy, 2)},"
                f"{round(case.expected_decode_proxy, 2)})",
                flush=True,
            )
            run_composition_case(case, repeat_index, args, model, prompts, csv_lock)
            if args.case_sleep_sec > 0:
                time.sleep(args.case_sleep_sec)

    write_summary(args.raw_csv, args.summary_csv)
    write_validation_csv(
        args.raw_csv,
        args.iteration_jsonl,
        args.decode_timeline_jsonl,
        args.validation_csv,
    )
    write_composition_summary(
        cases,
        args.repeat_count,
        args.iteration_jsonl,
        args.decode_timeline_jsonl,
        args.composition_summary_csv,
        args.validation_tolerance_ratio,
    )
    print(f"[DONE] raw_csv={args.raw_csv}", flush=True)
    print(f"[DONE] summary_csv={args.summary_csv}", flush=True)
    print(f"[DONE] token_csv={args.token_csv}", flush=True)
    print(f"[DONE] validation_csv={args.validation_csv}", flush=True)
    print(f"[DONE] composition_summary_csv={args.composition_summary_csv}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
