#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import requests

from measure_iteration_causality_microbench import (
    CaseSpec,
    RequestSpec,
    build_exact_prompt,
    run_simultaneous_case,
    run_staged_case,
    write_summary,
    write_validation_csv,
)
from measure_vllm_run import get_model_id, tokenize_text


SWEEP_FIELDNAMES = [
    "axis",
    "repeat_index",
    "sweep_tokens",
    "case_id",
    "victim_label",
    "mixed_valid",
    "mixed_iteration_id",
    "mixed_iteration_model_wait_ms",
    "mixed_iteration_duration_ms",
    "mixed_scheduled_total_tokens",
    "mixed_scheduled_prefill_tokens",
    "mixed_scheduled_decode_tokens",
    "mixed_decode_context_max",
    "mixed_prefill_tokens",
    "victim_prefill_ms",
    "victim_decode_ms",
    "victim_ttft_ms",
    "victim_ttlt_ms",
    "victim_p95_tbt_ms",
    "victim_max_token_gap_ms",
    "victim_mean_token_gap_ms",
    "victim_first_prefill_iteration_id",
    "victim_first_decode_iteration_id",
    "victim_first_prefill_model_wait_ms",
    "victim_first_decode_model_wait_ms",
    "victim_max_iteration_model_wait_ms",
    "victim_mean_iteration_model_wait_ms",
    "observed_max_decode_context",
    "observed_max_prefill_tokens",
]


@dataclass(frozen=True)
class SweepCase:
    axis: str
    sweep_tokens: int
    case: CaseSpec
    victim_label: str
    victim_phase: str


def parse_token_sweep(value: str) -> List[int]:
    tokens = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        tokens.append(int(part))
    return sorted(set(tokens))


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


def as_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(row: Dict[str, Any], key: str, default: int = -1) -> int:
    return int(as_float(row, key, float(default)))


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_row(rows: List[Dict[str, str]], case_id: str, repeat_index: int,
             request_label: str) -> Dict[str, str] | None:
    for row in rows:
        if (
            row.get("case_id") == case_id
            and int(row.get("repeat_index", "0")) == repeat_index
            and row.get("request_label") == request_label
        ):
            return row
    return None


def token_gap_stats(token_rows: List[Dict[str, str]], case_id: str,
                    repeat_index: int, request_label: str) -> Tuple[float, float]:
    values = [
        as_float(row, "chunk_gap_ms")
        for row in token_rows
        if row.get("case_id") == case_id
        and int(row.get("repeat_index", "0")) == repeat_index
        and row.get("request_label") == request_label
        and row.get("chunk_gap_ms")
    ]
    if not values:
        return 0.0, 0.0
    return max(values), statistics.mean(values)


def build_sweep_cases(
    sweep_tokens: List[int],
    short_tokens: int,
    prefill_output_tokens: int,
    decode_output_tokens: int,
) -> List[SweepCase]:
    cases: List[SweepCase] = []
    for tokens in sweep_tokens:
        cases.append(
            SweepCase(
                axis="prefill_victim_decode_context_sweep",
                sweep_tokens=tokens,
                case=CaseSpec(
                    case_id=f"prefill_victim_decodectx_{tokens}",
                    launch_mode="staged_after_background_first_token",
                    requests=(),
                    background=RequestSpec(
                        f"decode_ctx_{tokens}", "peer", tokens,
                        decode_output_tokens),
                    after_first_token=(
                        RequestSpec("short_prefill_victim", "victim",
                                    short_tokens, prefill_output_tokens),
                    ),
                ),
                victim_label="short_prefill_victim",
                victim_phase="prefill",
            )
        )
    for tokens in sweep_tokens:
        cases.append(
            SweepCase(
                axis="decode_victim_prefill_size_sweep",
                sweep_tokens=tokens,
                case=CaseSpec(
                    case_id=f"decode_victim_prefill_{tokens}",
                    launch_mode="staged_after_background_first_token",
                    requests=(),
                    background=RequestSpec("short_decode_victim", "victim",
                                           short_tokens, decode_output_tokens),
                    after_first_token=(
                        RequestSpec(f"prefill_peer_{tokens}", "peer", tokens,
                                    prefill_output_tokens),
                    ),
                ),
                victim_label="short_decode_victim",
                victim_phase="decode",
            )
        )
    return cases


def build_control_cases(short_tokens: int, prefill_output_tokens: int,
                        decode_output_tokens: int) -> List[SweepCase]:
    return [
        SweepCase(
            axis="prefill_victim_alone_control",
            sweep_tokens=0,
            case=CaseSpec(
                case_id="prefill_victim_alone_control",
                launch_mode="simultaneous",
                requests=(
                    RequestSpec("short_prefill_victim", "victim",
                                short_tokens, prefill_output_tokens),
                ),
            ),
            victim_label="short_prefill_victim",
            victim_phase="prefill",
        ),
        SweepCase(
            axis="decode_victim_alone_control",
            sweep_tokens=0,
            case=CaseSpec(
                case_id="decode_victim_alone_control",
                launch_mode="simultaneous",
                requests=(
                    RequestSpec("short_decode_victim", "victim", short_tokens,
                                decode_output_tokens),
                ),
            ),
            victim_label="short_decode_victim",
            victim_phase="decode",
        ),
    ]


def matching_timeline_rows(
    validation_row: Dict[str, str],
    timeline_by_request: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    ids = [
        item for item in validation_row.get(
            "matched_timeline_request_ids", "").split(";")
        if item
    ]
    rows: List[Dict[str, Any]] = []
    for request_id in ids:
        rows.extend(timeline_by_request.get(request_id, []))
    rows.sort(key=lambda row: int(row.get("iteration_id", -1)))
    return rows


def find_mixed_timeline_row(
    validation_row: Dict[str, str],
    timeline_by_request: Dict[str, List[Dict[str, Any]]],
    axis: str,
    victim_phase: str,
    threshold_tokens: float,
) -> Dict[str, Any] | None:
    rows = matching_timeline_rows(validation_row, timeline_by_request)
    for row in rows:
        if row.get("phase") != victim_phase:
            continue
        if (
            axis == "prefill_victim_decode_context_sweep"
            and as_float(row, "token_batch_decode_context_max") >= threshold_tokens
        ):
            return row
        if (
            axis == "decode_victim_prefill_size_sweep"
            and as_float(row, "token_batch_prefill_tokens") >= threshold_tokens
        ):
            return row
    return None


def write_sweep_summary_csv(
    sweep_cases: List[SweepCase],
    repeat_count: int,
    raw_csv: str,
    token_csv: str,
    validation_csv: str,
    iteration_jsonl: str,
    decode_timeline_jsonl: str,
    output_csv: str,
    exposure_threshold_ratio: float,
) -> None:
    raw_rows = read_csv(raw_csv)
    token_rows = read_csv(token_csv)
    validation_rows = read_csv(validation_csv)
    iteration_rows = {
        int(row.get("iteration_id", -1)): row
        for row in iter_jsonl(Path(iteration_jsonl))
    }
    timeline_by_request: Dict[str, List[Dict[str, Any]]] = {}
    for row in iter_jsonl(Path(decode_timeline_jsonl)):
        request_id = str(row.get("request_id", ""))
        if request_id:
            timeline_by_request.setdefault(request_id, []).append(row)

    rows: List[Dict[str, Any]] = []
    for repeat_index in range(1, repeat_count + 1):
        for sweep_case in sweep_cases:
            case_id = sweep_case.case.case_id
            raw = find_row(raw_rows, case_id, repeat_index,
                           sweep_case.victim_label) or {}
            validation = find_row(validation_rows, case_id, repeat_index,
                                  sweep_case.victim_label) or {}
            threshold = sweep_case.sweep_tokens * exposure_threshold_ratio
            mixed_row = None
            if sweep_case.sweep_tokens > 0:
                mixed_row = find_mixed_timeline_row(
                    validation,
                    timeline_by_request,
                    sweep_case.axis,
                    sweep_case.victim_phase,
                    threshold,
                )
            mixed_iteration_id = as_int(mixed_row or {}, "iteration_id")
            mixed_iteration = iteration_rows.get(mixed_iteration_id, {})
            max_gap, mean_gap = token_gap_stats(
                token_rows, case_id, repeat_index, sweep_case.victim_label)
            observed_decode_context = as_float(
                validation, "max_token_batch_decode_context")
            observed_prefill_tokens = as_float(
                validation, "max_token_batch_prefill_tokens")
            mixed_valid = bool(mixed_row) if sweep_case.sweep_tokens > 0 else True
            rows.append({
                "axis": sweep_case.axis,
                "repeat_index": repeat_index,
                "sweep_tokens": sweep_case.sweep_tokens,
                "case_id": case_id,
                "victim_label": sweep_case.victim_label,
                "mixed_valid": int(mixed_valid),
                "mixed_iteration_id": mixed_iteration_id,
                "mixed_iteration_model_wait_ms": round(
                    as_float(mixed_iteration, "model_wait_duration_s") * 1000.0,
                    4),
                "mixed_iteration_duration_ms": round(
                    as_float(mixed_iteration, "iteration_duration_s") * 1000.0,
                    4),
                "mixed_scheduled_total_tokens": as_float(
                    mixed_iteration, "scheduled_total_tokens"),
                "mixed_scheduled_prefill_tokens": as_float(
                    mixed_iteration, "scheduled_prefill_tokens"),
                "mixed_scheduled_decode_tokens": as_float(
                    mixed_iteration, "scheduled_decode_tokens"),
                "mixed_decode_context_max": as_float(
                    mixed_iteration, "decode_context_max"),
                "mixed_prefill_tokens": as_float(
                    mixed_row or {}, "token_batch_prefill_tokens"),
                "victim_prefill_ms": as_float(raw, "prefill_time_ms"),
                "victim_decode_ms": as_float(raw, "decode_time_ms"),
                "victim_ttft_ms": as_float(raw, "ttft_ms_server"),
                "victim_ttlt_ms": as_float(raw, "ttlt_ms_server"),
                "victim_p95_tbt_ms": as_float(raw, "p95_tbt_ms_client"),
                "victim_max_token_gap_ms": round(max_gap, 4),
                "victim_mean_token_gap_ms": round(mean_gap, 4),
                "victim_first_prefill_iteration_id": as_int(
                    validation, "first_prefill_iteration_id"),
                "victim_first_decode_iteration_id": as_int(
                    validation, "first_decode_iteration_id"),
                "victim_first_prefill_model_wait_ms": as_float(
                    validation, "first_prefill_model_wait_ms"),
                "victim_first_decode_model_wait_ms": as_float(
                    validation, "first_decode_model_wait_ms"),
                "victim_max_iteration_model_wait_ms": as_float(
                    validation, "max_iteration_model_wait_ms"),
                "victim_mean_iteration_model_wait_ms": as_float(
                    validation, "mean_iteration_model_wait_ms"),
                "observed_max_decode_context": observed_decode_context,
                "observed_max_prefill_tokens": observed_prefill_tokens,
            })

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SWEEP_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--token-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--sweep-summary-csv", required=True)
    parser.add_argument("--metrics-jsonl", required=True)
    parser.add_argument("--iteration-jsonl", required=True)
    parser.add_argument("--decode-timeline-jsonl", required=True)
    parser.add_argument("--repeat-count", type=int, default=5)
    parser.add_argument("--short-prompt-tokens", type=int, default=512)
    parser.add_argument(
        "--sweep-prompt-tokens",
        default="512,1024,2048,4096,8192,16384",
    )
    parser.add_argument("--prefill-output-tokens", type=int, default=1)
    parser.add_argument("--decode-output-tokens", type=int, default=512)
    parser.add_argument("--request-timeout-sec", type=int, default=2400)
    parser.add_argument("--background-first-token-timeout-sec", type=int,
                        default=900)
    parser.add_argument("--launch-warmup-sec", type=float, default=0.2)
    parser.add_argument("--post-first-token-delay-ms", type=int, default=0)
    parser.add_argument("--case-sleep-sec", type=float, default=30.0)
    parser.add_argument("--exposure-threshold-ratio", type=float, default=0.9)
    parser.add_argument("--run-controls", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = requests.Session()
    model = get_model_id(session, args.base_url)
    sweep_tokens = parse_token_sweep(args.sweep_prompt_tokens)
    sweep_cases = build_sweep_cases(
        sweep_tokens,
        args.short_prompt_tokens,
        args.prefill_output_tokens,
        args.decode_output_tokens,
    )
    control_cases = build_control_cases(
        args.short_prompt_tokens,
        args.prefill_output_tokens,
        args.decode_output_tokens,
    ) if args.run_controls else []
    all_cases = [*control_cases, *sweep_cases]

    prompt_lengths = sorted({args.short_prompt_tokens, *sweep_tokens})
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
        args.sweep_summary_csv,
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    csv_lock = threading.Lock()
    for repeat_index in range(1, args.repeat_count + 1):
        for sweep_case in all_cases:
            case = sweep_case.case
            print(
                f"[CASE] repeat={repeat_index}/{args.repeat_count} "
                f"axis={sweep_case.axis} sweep_tokens={sweep_case.sweep_tokens} "
                f"case={case.case_id} mode={case.launch_mode}",
                flush=True,
            )
            if case.launch_mode == "simultaneous":
                run_simultaneous_case(
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
    write_sweep_summary_csv(
        all_cases,
        args.repeat_count,
        args.raw_csv,
        args.token_csv,
        args.validation_csv,
        args.iteration_jsonl,
        args.decode_timeline_jsonl,
        args.sweep_summary_csv,
        args.exposure_threshold_ratio,
    )
    print(f"[DONE] raw_csv={args.raw_csv}", flush=True)
    print(f"[DONE] summary_csv={args.summary_csv}", flush=True)
    print(f"[DONE] token_csv={args.token_csv}", flush=True)
    print(f"[DONE] validation_csv={args.validation_csv}", flush=True)
    print(f"[DONE] sweep_summary_csv={args.sweep_summary_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
