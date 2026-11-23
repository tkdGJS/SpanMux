#!/usr/bin/env python3
import argparse
import csv
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
    iter_jsonl,
    run_simultaneous_case,
    write_summary,
    write_validation_csv,
)
from measure_vllm_run import get_model_id, tokenize_text


PURE_FIELDNAMES = [
    "material",
    "repeat_index",
    "sweep_tokens",
    "case_id",
    "request_label",
    "status",
    "request_prefill_ms",
    "request_decode_ms",
    "request_ttft_ms",
    "request_ttlt_ms",
    "request_p95_tbt_ms",
    "phase_iteration_count",
    "phase_iteration_ids",
    "pure_phase_valid",
    "phase_model_wait_ms_mean",
    "phase_model_wait_ms_median",
    "phase_model_wait_ms_max",
    "phase_iteration_duration_ms_mean",
    "phase_iteration_duration_ms_median",
    "phase_iteration_duration_ms_max",
    "phase_scheduled_total_tokens_mean",
    "phase_scheduled_total_tokens_max",
    "phase_scheduled_prefill_tokens_mean",
    "phase_scheduled_prefill_tokens_max",
    "phase_scheduled_decode_tokens_mean",
    "phase_scheduled_decode_tokens_max",
    "phase_decode_context_max_mean",
    "phase_decode_context_max_max",
    "phase_prefill_attention_work_proxy_mean",
    "phase_prefill_attention_work_proxy_max",
    "phase_decode_kv_read_token_proxy_mean",
    "phase_decode_kv_read_token_proxy_max",
]


@dataclass(frozen=True)
class PureCase:
    material: str
    sweep_tokens: int
    case: CaseSpec
    request_label: str
    phase: str


def parse_token_sweep(value: str) -> List[int]:
    tokens = []
    for part in value.split(","):
        part = part.strip()
        if part:
            tokens.append(int(part))
    return sorted(set(tokens))


def as_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_row(rows: List[Dict[str, str]], case_id: str, repeat_index: int,
             request_label: str) -> Dict[str, str]:
    for row in rows:
        if (
            row.get("case_id") == case_id
            and int(row.get("repeat_index", "0")) == repeat_index
            and row.get("request_label") == request_label
        ):
            return row
    return {}


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def median(values: List[float]) -> float:
    return statistics.median(values) if values else 0.0


def max_or_zero(values: List[float]) -> float:
    return max(values) if values else 0.0


def phase_metric(rows: List[Dict[str, Any]], key: str) -> Tuple[float, float]:
    values = [as_float(row, key) for row in rows]
    return mean(values), max_or_zero(values)


def phase_wait_metrics(rows: List[Dict[str, Any]], key: str) -> Tuple[float, float, float]:
    values = [as_float(row, key) * 1000.0 for row in rows]
    return mean(values), median(values), max_or_zero(values)


def build_cases(
    sweep_tokens: Iterable[int],
    prefill_output_tokens: int,
    decode_output_tokens: int,
) -> List[PureCase]:
    cases: List[PureCase] = []
    for tokens in sweep_tokens:
        cases.append(
            PureCase(
                material="prefill_only",
                sweep_tokens=tokens,
                case=CaseSpec(
                    case_id=f"pure_prefill_{tokens}",
                    launch_mode="simultaneous",
                    requests=(
                        RequestSpec(
                            f"prefill_only_{tokens}",
                            "prefill_only",
                            tokens,
                            prefill_output_tokens,
                        ),
                    ),
                ),
                request_label=f"prefill_only_{tokens}",
                phase="prefill",
            )
        )
        cases.append(
            PureCase(
                material="decode_only",
                sweep_tokens=tokens,
                case=CaseSpec(
                    case_id=f"pure_decode_{tokens}",
                    launch_mode="simultaneous",
                    requests=(
                        RequestSpec(
                            f"decode_only_{tokens}",
                            "decode_only",
                            tokens,
                            decode_output_tokens,
                        ),
                    ),
                ),
                request_label=f"decode_only_{tokens}",
                phase="decode",
            )
        )
    return cases


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
    rows.sort(
        key=lambda row: (
            int(float(row.get("iteration_id", -1))),
            str(row.get("phase", "")),
        )
    )
    return rows


def matching_timeline_rows_from_raw(
    raw_row: Dict[str, str],
    timeline_by_request: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    candidates = {
        raw_row.get("metrics_request_id", ""),
        raw_row.get("client_request_id", ""),
        raw_row.get("server_response_id", ""),
    }
    candidates = {candidate for candidate in candidates if candidate}
    rows: List[Dict[str, Any]] = []
    for request_id, request_rows in timeline_by_request.items():
        if any(
            request_id == candidate
            or request_id.startswith(f"{candidate}-")
            or candidate.startswith(f"{request_id}-")
            or candidate in request_id
            for candidate in candidates
        ):
            rows.extend(request_rows)
    rows.sort(
        key=lambda row: (
            int(float(row.get("iteration_id", -1))),
            str(row.get("phase", "")),
        )
    )
    return rows


def write_pure_summary_csv(
    cases: List[PureCase],
    repeat_count: int,
    raw_csv: str,
    validation_csv: str,
    iteration_jsonl: str,
    decode_timeline_jsonl: str,
    output_csv: str,
) -> None:
    raw_rows = read_csv(raw_csv)
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

    out_rows: List[Dict[str, Any]] = []
    for repeat_index in range(1, repeat_count + 1):
        for pure_case in cases:
            case_id = pure_case.case.case_id
            raw = find_row(raw_rows, case_id, repeat_index,
                           pure_case.request_label)
            validation = find_row(validation_rows, case_id, repeat_index,
                                  pure_case.request_label)
            timeline_rows = matching_timeline_rows_from_raw(
                raw, timeline_by_request)
            if not timeline_rows:
                timeline_rows = matching_timeline_rows(
                    validation, timeline_by_request)
            phase_rows = [
                row for row in timeline_rows
                if row.get("phase") == pure_case.phase
            ]
            phase_iteration_ids = sorted({
                int(float(row.get("iteration_id", -1)))
                for row in phase_rows
                if int(float(row.get("iteration_id", -1))) >= 0
            })
            phase_iterations = [
                iteration_rows[iteration_id]
                for iteration_id in phase_iteration_ids
                if iteration_id in iteration_rows
            ]
            if pure_case.material == "prefill_only":
                pure_valid = all(
                    as_float(row, "scheduled_decode_tokens") == 0.0
                    for row in phase_iterations
                )
            else:
                pure_valid = all(
                    as_float(row, "scheduled_prefill_tokens") == 0.0
                    for row in phase_iterations
                )
            model_wait = phase_wait_metrics(
                phase_iterations, "model_wait_duration_s")
            iteration_duration = phase_wait_metrics(
                phase_iterations, "iteration_duration_s")
            scheduled_total = phase_metric(
                phase_iterations, "scheduled_total_tokens")
            scheduled_prefill = phase_metric(
                phase_iterations, "scheduled_prefill_tokens")
            scheduled_decode = phase_metric(
                phase_iterations, "scheduled_decode_tokens")
            decode_context = phase_metric(phase_iterations, "decode_context_max")
            prefill_proxy = phase_metric(
                phase_iterations, "prefill_attention_work_proxy")
            decode_proxy = phase_metric(
                phase_iterations, "decode_kv_read_token_proxy")
            out_rows.append({
                "material": pure_case.material,
                "repeat_index": repeat_index,
                "sweep_tokens": pure_case.sweep_tokens,
                "case_id": case_id,
                "request_label": pure_case.request_label,
                "status": raw.get("status", ""),
                "request_prefill_ms": as_float(raw, "prefill_time_ms"),
                "request_decode_ms": as_float(raw, "decode_time_ms"),
                "request_ttft_ms": as_float(raw, "ttft_ms_server"),
                "request_ttlt_ms": as_float(raw, "ttlt_ms_server"),
                "request_p95_tbt_ms": as_float(raw, "p95_tbt_ms_client"),
                "phase_iteration_count": len(phase_iteration_ids),
                "phase_iteration_ids": ";".join(
                    str(iteration_id) for iteration_id in phase_iteration_ids),
                "pure_phase_valid": int(bool(phase_iterations) and pure_valid),
                "phase_model_wait_ms_mean": round(model_wait[0], 4),
                "phase_model_wait_ms_median": round(model_wait[1], 4),
                "phase_model_wait_ms_max": round(model_wait[2], 4),
                "phase_iteration_duration_ms_mean": round(iteration_duration[0], 4),
                "phase_iteration_duration_ms_median": round(iteration_duration[1], 4),
                "phase_iteration_duration_ms_max": round(iteration_duration[2], 4),
                "phase_scheduled_total_tokens_mean": round(scheduled_total[0], 4),
                "phase_scheduled_total_tokens_max": round(scheduled_total[1], 4),
                "phase_scheduled_prefill_tokens_mean": round(
                    scheduled_prefill[0], 4),
                "phase_scheduled_prefill_tokens_max": round(
                    scheduled_prefill[1], 4),
                "phase_scheduled_decode_tokens_mean": round(scheduled_decode[0], 4),
                "phase_scheduled_decode_tokens_max": round(scheduled_decode[1], 4),
                "phase_decode_context_max_mean": round(decode_context[0], 4),
                "phase_decode_context_max_max": round(decode_context[1], 4),
                "phase_prefill_attention_work_proxy_mean": round(
                    prefill_proxy[0], 4),
                "phase_prefill_attention_work_proxy_max": round(
                    prefill_proxy[1], 4),
                "phase_decode_kv_read_token_proxy_mean": round(decode_proxy[0], 4),
                "phase_decode_kv_read_token_proxy_max": round(decode_proxy[1], 4),
            })

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PURE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--token-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--pure-summary-csv", required=True)
    parser.add_argument("--metrics-jsonl", required=True)
    parser.add_argument("--iteration-jsonl", required=True)
    parser.add_argument("--decode-timeline-jsonl", required=True)
    parser.add_argument("--repeat-count", type=int, default=10)
    parser.add_argument(
        "--sweep-prompt-tokens",
        default="512,1024,2048,4096,8192,16384",
    )
    parser.add_argument("--prefill-output-tokens", type=int, default=1)
    parser.add_argument("--decode-output-tokens", type=int, default=512)
    parser.add_argument("--request-timeout-sec", type=int, default=2400)
    parser.add_argument("--launch-warmup-sec", type=float, default=0.2)
    parser.add_argument("--case-sleep-sec", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = requests.Session()
    model = get_model_id(session, args.base_url)
    sweep_tokens = parse_token_sweep(args.sweep_prompt_tokens)
    cases = build_cases(
        sweep_tokens,
        args.prefill_output_tokens,
        args.decode_output_tokens,
    )

    prompts: Dict[int, str] = {}
    for length in sweep_tokens:
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
        args.pure_summary_csv,
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    csv_lock = threading.Lock()
    for repeat_index in range(1, args.repeat_count + 1):
        for pure_case in cases:
            print(
                f"[CASE] repeat={repeat_index}/{args.repeat_count} "
                f"material={pure_case.material} "
                f"sweep_tokens={pure_case.sweep_tokens} "
                f"case={pure_case.case.case_id}",
                flush=True,
            )
            run_simultaneous_case(
                pure_case.case,
                repeat_index,
                args,
                model,
                prompts,
                csv_lock,
            )
            if args.case_sleep_sec > 0:
                time.sleep(args.case_sleep_sec)

    write_summary(args.raw_csv, args.summary_csv)
    write_validation_csv(
        args.raw_csv,
        args.iteration_jsonl,
        args.decode_timeline_jsonl,
        args.validation_csv,
    )
    write_pure_summary_csv(
        cases,
        args.repeat_count,
        args.raw_csv,
        args.validation_csv,
        args.iteration_jsonl,
        args.decode_timeline_jsonl,
        args.pure_summary_csv,
    )
    print(f"[DONE] raw_csv={args.raw_csv}", flush=True)
    print(f"[DONE] summary_csv={args.summary_csv}", flush=True)
    print(f"[DONE] token_csv={args.token_csv}", flush=True)
    print(f"[DONE] validation_csv={args.validation_csv}", flush=True)
    print(f"[DONE] pure_summary_csv={args.pure_summary_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
