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
    run_coalesced_after_prefill_barrier_case,
    run_simultaneous_case,
    run_staged_case,
    write_summary,
    write_validation_csv,
)
from measure_vllm_run import get_model_id, tokenize_text


ITERATION_METRIC_KEYS = [
    "iteration_duration_s",
    "schedule_duration_s",
    "model_wait_duration_s",
    "output_process_duration_s",
    "gpu_forward_elapsed_ms",
    "worker_kv_cache_prepare_duration_s",
    "worker_attention_metadata_build_duration_s",
    "worker_forward_duration_s",
    "worker_model_forward_duration_s",
    "worker_observed_duration_s",
    "executor_unattributed_wait_duration_s",
    "scheduled_total_tokens",
    "scheduled_prefill_tokens",
    "scheduled_decode_tokens",
    "num_prefill_reqs",
    "num_decode_reqs",
    "decode_context_sum",
    "decode_context_max",
    "decode_context_p50",
    "decode_context_p95",
    "prefill_context_sum",
    "prefill_context_max",
    "prefill_context_p95",
    "prefill_attention_work_proxy",
    "decode_kv_read_token_proxy",
    "layer_kernel_timing_qkv_proj_ms",
    "layer_kernel_timing_attention_ms",
    "layer_kernel_timing_o_proj_ms",
    "layer_kernel_timing_mlp_total_ms",
    "layer_kernel_timing_projection_total_ms",
]

OP_PD_METRIC_KEYS = [
    "op_pd_enabled",
    "op_pd_mode",
    "op_pd_attention_enabled",
    "op_pd_phase_baseline_enabled",
    "op_pd_green_enabled",
    "op_pd_overlap_enabled",
    "op_pd_fallback_count",
    "op_pd_fallback_reason",
    "op_pd_layers",
    "op_pd_attention_layers",
    "op_pd_phase_layers",
    "op_pd_num_decode_tokens",
    "op_pd_num_prefill_tokens",
    "op_pd_decode_token_work",
    "op_pd_prefill_token_work",
    "op_pd_decode_attention_ms",
    "op_pd_prefill_attention_ms",
    "op_pd_attention_union_ms",
    "op_pd_attention_overlap_ms",
    "op_pd_attention_join_wait_ms",
    "op_pd_decode_phase_ms",
    "op_pd_prefill_phase_ms",
    "op_pd_phase_union_ms",
    "op_pd_phase_overlap_ms",
    "op_pd_phase_join_wait_ms",
    "green_context_enabled",
    "green_context_active",
    "green_context_fallback",
    "green_decode_sms",
    "green_prefill_sms",
    "green_stream_priority_decode",
    "green_stream_priority_prefill",
]

MOTIVATION_FIELDNAMES = [
    "execution_mode",
    "mode_label",
    "mix_case",
    "target_phase",
    "peer_phase",
    "peer_tokens",
    "repeat_index",
    "case_id",
    "launch_mode",
    "target_label",
    "peer_label",
    "target_role",
    "peer_role",
    "target_status",
    "peer_status",
    "valid_same_iteration",
    "validation_failure_reason",
    "target_iteration_id",
    "target_prefill_ms",
    "target_decode_ms",
    "target_ttft_ms",
    "target_ttlt_ms",
    "target_p95_tbt_ms",
    "target_max_token_gap_ms",
    "target_mean_token_gap_ms",
    "target_first_prefill_iteration_id",
    "target_first_decode_iteration_id",
    "target_first_prefill_model_wait_ms",
    "target_first_decode_model_wait_ms",
    "target_max_iteration_model_wait_ms",
    "target_mean_iteration_model_wait_ms",
    "observed_max_decode_context",
    "observed_max_prefill_tokens",
    "matched_timeline_request_ids",
    *ITERATION_METRIC_KEYS,
    *OP_PD_METRIC_KEYS,
]


@dataclass(frozen=True)
class MotivationCase:
    mix_case: str
    target_phase: str
    peer_phase: str
    peer_tokens: int
    case: CaseSpec
    target_label: str
    peer_label: str


def parse_token_sweep(value: str) -> List[int]:
    tokens = []
    for part in value.split(","):
        part = part.strip()
        if part:
            tokens.append(int(part))
    return sorted(set(tokens))


def parse_mix_cases(value: str) -> List[str]:
    cases = []
    for part in value.split(","):
        part = part.strip().upper()
        if part:
            cases.append(part)
    valid = {"SOLO", "D/D", "D/P", "P/D", "P/P"}
    unknown = sorted(set(cases) - valid)
    if unknown:
        raise ValueError(f"Unknown mix cases: {unknown}")
    return cases


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


def find_row(
    rows: List[Dict[str, str]],
    case_id: str,
    repeat_index: int,
    request_label: str,
) -> Dict[str, str] | None:
    for row in rows:
        if (
            row.get("case_id") == case_id
            and int(row.get("repeat_index", "0")) == repeat_index
            and row.get("request_label") == request_label
        ):
            return row
    return None


def token_gap_stats(
    token_rows: List[Dict[str, str]],
    case_id: str,
    repeat_index: int,
    request_label: str,
) -> Tuple[float, float]:
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


def target_timeline_phase(target_phase: str) -> str:
    if target_phase == "D":
        return "decode"
    if target_phase == "P":
        return "prefill"
    raise ValueError(f"Unknown target phase: {target_phase}")


def build_cases(
    sweep_tokens: List[int],
    short_tokens: int,
    prefill_output_tokens: int,
    decode_output_tokens: int,
    mix_cases: List[str],
    same_prefill_launch_mode: str,
) -> List[MotivationCase]:
    cases: List[MotivationCase] = []
    if "SOLO" in mix_cases:
        cases.extend([
            MotivationCase(
                mix_case="solo",
                target_phase="D",
                peer_phase="none",
                peer_tokens=0,
                case=CaseSpec(
                    case_id="solo_decode_target",
                    launch_mode="simultaneous",
                    requests=(
                        RequestSpec(
                            "short_decode_target",
                            "target",
                            short_tokens,
                            decode_output_tokens,
                        ),
                    ),
                ),
                target_label="short_decode_target",
                peer_label="",
            ),
            MotivationCase(
                mix_case="solo",
                target_phase="P",
                peer_phase="none",
                peer_tokens=0,
                case=CaseSpec(
                    case_id="solo_prefill_target",
                    launch_mode="simultaneous",
                    requests=(
                        RequestSpec(
                            "short_prefill_target",
                            "target",
                            short_tokens,
                            prefill_output_tokens,
                        ),
                    ),
                ),
                target_label="short_prefill_target",
                peer_label="",
            ),
        ])

    for tokens in sweep_tokens:
        if "D/D" in mix_cases:
            cases.append(
                MotivationCase(
                    mix_case="D/D",
                    target_phase="D",
                    peer_phase="D",
                    peer_tokens=tokens,
                    case=CaseSpec(
                        case_id=f"dd_decode_target_decode_peer_{tokens}",
                        launch_mode="staged_after_background_first_token",
                        requests=(),
                        background=RequestSpec(
                            f"decode_peer_{tokens}",
                            "peer",
                            tokens,
                            decode_output_tokens,
                        ),
                        after_first_token=(
                            RequestSpec(
                                "short_decode_target",
                                "target",
                                short_tokens,
                                decode_output_tokens,
                            ),
                        ),
                    ),
                    target_label="short_decode_target",
                    peer_label=f"decode_peer_{tokens}",
                )
            )
        if "D/P" in mix_cases:
            cases.append(
                MotivationCase(
                    mix_case="D/P",
                    target_phase="D",
                    peer_phase="P",
                    peer_tokens=tokens,
                    case=CaseSpec(
                        case_id=f"dp_decode_target_prefill_peer_{tokens}",
                        launch_mode="staged_after_background_first_token",
                        requests=(),
                        background=RequestSpec(
                            "short_decode_target",
                            "target",
                            short_tokens,
                            decode_output_tokens,
                        ),
                        after_first_token=(
                            RequestSpec(
                                f"prefill_peer_{tokens}",
                                "peer",
                                tokens,
                                prefill_output_tokens,
                            ),
                        ),
                    ),
                    target_label="short_decode_target",
                    peer_label=f"prefill_peer_{tokens}",
                )
            )
        if "P/D" in mix_cases:
            cases.append(
                MotivationCase(
                    mix_case="P/D",
                    target_phase="P",
                    peer_phase="D",
                    peer_tokens=tokens,
                    case=CaseSpec(
                        case_id=f"pd_prefill_target_decode_peer_{tokens}",
                        launch_mode="staged_after_background_first_token",
                        requests=(),
                        background=RequestSpec(
                            f"decode_peer_{tokens}",
                            "peer",
                            tokens,
                            decode_output_tokens,
                        ),
                        after_first_token=(
                            RequestSpec(
                                "short_prefill_target",
                                "target",
                                short_tokens,
                                prefill_output_tokens,
                            ),
                        ),
                    ),
                    target_label="short_prefill_target",
                    peer_label=f"decode_peer_{tokens}",
                )
            )
        if "P/P" in mix_cases:
            cases.append(
                MotivationCase(
                    mix_case="P/P",
                    target_phase="P",
                    peer_phase="P",
                    peer_tokens=tokens,
                    case=CaseSpec(
                        case_id=f"pp_prefill_target_prefill_peer_{tokens}",
                        launch_mode=same_prefill_launch_mode,
                        requests=(
                            RequestSpec(
                                "short_prefill_target",
                                "target",
                                short_tokens,
                                prefill_output_tokens,
                            ),
                            RequestSpec(
                                f"prefill_peer_{tokens}",
                                "peer",
                                tokens,
                                prefill_output_tokens,
                            ),
                        ),
                    ),
                    target_label="short_prefill_target",
                    peer_label=f"prefill_peer_{tokens}",
                )
            )
    return cases


def matching_timeline_rows(
    validation_row: Dict[str, str],
    timeline_by_request: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    ids = [
        item
        for item in validation_row.get("matched_timeline_request_ids", "").split(";")
        if item
    ]
    rows: List[Dict[str, Any]] = []
    for request_id in ids:
        rows.extend(timeline_by_request.get(request_id, []))
    if rows:
        rows.sort(key=lambda row: int(row.get("iteration_id", -1)))
        return rows

    candidates = {
        validation_row.get("metrics_request_id", ""),
        validation_row.get("client_request_id", ""),
        validation_row.get("server_response_id", ""),
    }
    for candidate in list(candidates):
        if candidate:
            candidates.add(f"cmpl-{candidate}")
            candidates.add(f"cmpl-{candidate}-0")
            if candidate.endswith("-0"):
                candidates.add(candidate[:-2])
    candidates = {candidate for candidate in candidates if candidate}

    for timeline_request_id, timeline_rows in timeline_by_request.items():
        for candidate in candidates:
            if (
                timeline_request_id == candidate
                or timeline_request_id.startswith(f"{candidate}-")
                or candidate.startswith(f"{timeline_request_id}-")
                or candidate in timeline_request_id
            ):
                rows.extend(timeline_rows)
                break
    rows.sort(key=lambda row: int(row.get("iteration_id", -1)))
    return rows


def first_phase_row(
    validation_row: Dict[str, str],
    timeline_by_request: Dict[str, List[Dict[str, Any]]],
    phase: str,
) -> Dict[str, Any] | None:
    for row in matching_timeline_rows(validation_row, timeline_by_request):
        if row.get("phase") == phase:
            return row
    return None


def find_selected_timeline_row(
    motivation_case: MotivationCase,
    validation_row: Dict[str, str],
    timeline_by_request: Dict[str, List[Dict[str, Any]]],
    iteration_rows: Dict[int, Dict[str, Any]],
    short_tokens: int,
    exposure_threshold_ratio: float,
) -> Tuple[Dict[str, Any] | None, str]:
    phase = target_timeline_phase(motivation_case.target_phase)
    if motivation_case.mix_case == "solo":
        row = first_phase_row(validation_row, timeline_by_request, phase)
        if row is None:
            return None, "no_target_phase_timeline"
        return row, ""

    peer_threshold = motivation_case.peer_tokens * exposure_threshold_ratio
    rows = matching_timeline_rows(validation_row, timeline_by_request)
    target_phase_rows = [row for row in rows if row.get("phase") == phase]
    if not target_phase_rows:
        return None, "no_target_phase_timeline"

    for row in target_phase_rows:
        iteration_id = as_int(row, "iteration_id")
        iteration = iteration_rows.get(iteration_id, {})
        if motivation_case.mix_case == "D/D":
            if (
                as_float(iteration, "num_decode_reqs") >= 2
                and as_float(iteration, "decode_context_max") >= peer_threshold
            ):
                return row, ""
        elif motivation_case.mix_case == "D/P":
            if (
                as_float(row, "token_batch_prefill_tokens")
                >= peer_threshold
                or as_float(iteration, "scheduled_prefill_tokens")
                >= peer_threshold
            ):
                return row, ""
        elif motivation_case.mix_case == "P/D":
            if (
                as_float(row, "token_batch_decode_context_max")
                >= peer_threshold
                or as_float(iteration, "decode_context_max")
                >= peer_threshold
            ):
                return row, ""
        elif motivation_case.mix_case == "P/P":
            required_prefill_tokens = short_tokens + peer_threshold
            if (
                as_float(iteration, "num_prefill_reqs") >= 2
                and as_float(iteration, "scheduled_prefill_tokens")
                >= required_prefill_tokens
            ):
                return row, ""
    return None, "peer_not_observed_in_target_phase_iteration"


def metric_value(row: Dict[str, Any], key: str) -> Any:
    value = row.get(key, "")
    if value is None:
        return ""
    return value


def build_summary_row(
    motivation_case: MotivationCase,
    repeat_index: int,
    raw_rows: List[Dict[str, str]],
    token_rows: List[Dict[str, str]],
    validation_rows: List[Dict[str, str]],
    timeline_by_request: Dict[str, List[Dict[str, Any]]],
    iteration_rows: Dict[int, Dict[str, Any]],
    execution_mode: str,
    mode_label: str,
    short_tokens: int,
    exposure_threshold_ratio: float,
) -> Dict[str, Any]:
    case_id = motivation_case.case.case_id
    target_raw = find_row(
        raw_rows, case_id, repeat_index, motivation_case.target_label
    ) or {}
    peer_raw = (
        find_row(raw_rows, case_id, repeat_index, motivation_case.peer_label)
        if motivation_case.peer_label
        else None
    ) or {}
    validation = find_row(
        validation_rows, case_id, repeat_index, motivation_case.target_label
    ) or {}

    failure_reason = ""
    if not target_raw:
        failure_reason = "target_raw_missing"
    elif target_raw.get("status") != "success":
        failure_reason = "target_failed"
    elif not validation:
        failure_reason = "target_validation_missing"

    selected_row = None
    selected_reason = ""
    if not failure_reason:
        selected_row, selected_reason = find_selected_timeline_row(
            motivation_case,
            validation,
            timeline_by_request,
            iteration_rows,
            short_tokens,
            exposure_threshold_ratio,
        )
        failure_reason = selected_reason

    target_iteration_id = as_int(selected_row or {}, "iteration_id")
    iteration = iteration_rows.get(target_iteration_id, {})
    valid_same_iteration = 1 if selected_row is not None and not failure_reason else 0
    max_gap, mean_gap = token_gap_stats(
        token_rows, case_id, repeat_index, motivation_case.target_label
    )

    out: Dict[str, Any] = {
        "execution_mode": execution_mode,
        "mode_label": mode_label,
        "mix_case": motivation_case.mix_case,
        "target_phase": motivation_case.target_phase,
        "peer_phase": motivation_case.peer_phase,
        "peer_tokens": motivation_case.peer_tokens,
        "repeat_index": repeat_index,
        "case_id": case_id,
        "launch_mode": motivation_case.case.launch_mode,
        "target_label": motivation_case.target_label,
        "peer_label": motivation_case.peer_label,
        "target_role": target_raw.get("role", ""),
        "peer_role": peer_raw.get("role", ""),
        "target_status": target_raw.get("status", ""),
        "peer_status": peer_raw.get("status", ""),
        "valid_same_iteration": valid_same_iteration,
        "validation_failure_reason": failure_reason,
        "target_iteration_id": target_iteration_id,
        "target_prefill_ms": as_float(target_raw, "prefill_time_ms"),
        "target_decode_ms": as_float(target_raw, "decode_time_ms"),
        "target_ttft_ms": as_float(target_raw, "ttft_ms_server"),
        "target_ttlt_ms": as_float(target_raw, "ttlt_ms_server"),
        "target_p95_tbt_ms": as_float(target_raw, "p95_tbt_ms_client"),
        "target_max_token_gap_ms": round(max_gap, 4),
        "target_mean_token_gap_ms": round(mean_gap, 4),
        "target_first_prefill_iteration_id": as_int(
            validation, "first_prefill_iteration_id"
        ),
        "target_first_decode_iteration_id": as_int(
            validation, "first_decode_iteration_id"
        ),
        "target_first_prefill_model_wait_ms": as_float(
            validation, "first_prefill_model_wait_ms"
        ),
        "target_first_decode_model_wait_ms": as_float(
            validation, "first_decode_model_wait_ms"
        ),
        "target_max_iteration_model_wait_ms": as_float(
            validation, "max_iteration_model_wait_ms"
        ),
        "target_mean_iteration_model_wait_ms": as_float(
            validation, "mean_iteration_model_wait_ms"
        ),
        "observed_max_decode_context": as_float(
            validation, "max_token_batch_decode_context"
        ),
        "observed_max_prefill_tokens": as_float(
            validation, "max_token_batch_prefill_tokens"
        ),
        "matched_timeline_request_ids": validation.get(
            "matched_timeline_request_ids", ""
        ),
    }
    for key in ITERATION_METRIC_KEYS:
        out[key] = metric_value(iteration, key)
    for key in OP_PD_METRIC_KEYS:
        out[key] = metric_value(iteration, key)
    return out


def write_motivation_summary_csv(
    motivation_cases: List[MotivationCase],
    repeat_count: int,
    raw_csv: str,
    token_csv: str,
    validation_csv: str,
    iteration_jsonl: str,
    decode_timeline_jsonl: str,
    output_csv: str,
    execution_mode: str,
    mode_label: str,
    short_tokens: int,
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

    rows = []
    for repeat_index in range(1, repeat_count + 1):
        for motivation_case in motivation_cases:
            rows.append(
                build_summary_row(
                    motivation_case,
                    repeat_index,
                    raw_rows,
                    token_rows,
                    validation_rows,
                    timeline_by_request,
                    iteration_rows,
                    execution_mode,
                    mode_label,
                    short_tokens,
                    exposure_threshold_ratio,
                )
            )

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MOTIVATION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--token-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--motivation-summary-csv", required=True)
    parser.add_argument("--metrics-jsonl", required=True)
    parser.add_argument("--iteration-jsonl", required=True)
    parser.add_argument("--decode-timeline-jsonl", required=True)
    parser.add_argument("--execution-mode", required=True)
    parser.add_argument("--mode-label", default="")
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--short-prompt-tokens", type=int, default=512)
    parser.add_argument(
        "--sweep-prompt-tokens",
        default="512,2048,4096,8192,16384",
    )
    parser.add_argument("--prefill-output-tokens", type=int, default=2)
    parser.add_argument("--decode-output-tokens", type=int, default=512)
    parser.add_argument("--request-timeout-sec", type=int, default=2400)
    parser.add_argument("--background-first-token-timeout-sec", type=int,
                        default=900)
    parser.add_argument("--launch-warmup-sec", type=float, default=0.2)
    parser.add_argument("--post-first-token-delay-ms", type=int, default=0)
    parser.add_argument("--case-sleep-sec", type=float, default=30.0)
    parser.add_argument("--exposure-threshold-ratio", type=float, default=0.9)
    parser.add_argument("--coalesce-barrier-prompt-tokens", type=int, default=0)
    parser.add_argument("--coalesce-barrier-output-tokens", type=int, default=1)
    parser.add_argument("--coalesce-barrier-settle-ms", type=int, default=1000)
    parser.add_argument(
        "--same-prefill-launch-mode",
        choices=("simultaneous", "coalesced_after_prefill_barrier"),
        default="coalesced_after_prefill_barrier",
    )
    parser.add_argument(
        "--mix-cases",
        default="solo,D/D,D/P,P/D,P/P",
        help="Comma-separated subset of solo,D/D,D/P,P/D,P/P",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session = requests.Session()
    model = get_model_id(session, args.base_url)
    sweep_tokens = parse_token_sweep(args.sweep_prompt_tokens)
    mix_cases = parse_mix_cases(args.mix_cases)
    if args.coalesce_barrier_prompt_tokens <= 0:
        args.coalesce_barrier_prompt_tokens = max(sweep_tokens)
    motivation_cases = build_cases(
        sweep_tokens,
        args.short_prompt_tokens,
        args.prefill_output_tokens,
        args.decode_output_tokens,
        mix_cases,
        args.same_prefill_launch_mode,
    )

    prompt_lengths = sorted({
        args.short_prompt_tokens,
        *sweep_tokens,
        args.coalesce_barrier_prompt_tokens,
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
        args.motivation_summary_csv,
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    csv_lock = threading.Lock()
    for repeat_index in range(1, args.repeat_count + 1):
        for motivation_case in motivation_cases:
            case = motivation_case.case
            print(
                f"[CASE] mode={args.execution_mode} "
                f"repeat={repeat_index}/{args.repeat_count} "
                f"mix={motivation_case.mix_case} "
                f"peer_tokens={motivation_case.peer_tokens} "
                f"case={case.case_id} launch={case.launch_mode}",
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
            elif case.launch_mode == "coalesced_after_prefill_barrier":
                run_coalesced_after_prefill_barrier_case(
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
    write_motivation_summary_csv(
        motivation_cases,
        args.repeat_count,
        args.raw_csv,
        args.token_csv,
        args.validation_csv,
        args.iteration_jsonl,
        args.decode_timeline_jsonl,
        args.motivation_summary_csv,
        args.execution_mode,
        args.mode_label or args.execution_mode,
        args.short_prompt_tokens,
        args.exposure_threshold_ratio,
    )
    print(f"[DONE] raw_csv={args.raw_csv}", flush=True)
    print(f"[DONE] summary_csv={args.summary_csv}", flush=True)
    print(f"[DONE] token_csv={args.token_csv}", flush=True)
    print(f"[DONE] validation_csv={args.validation_csv}", flush=True)
    print(f"[DONE] motivation_summary_csv={args.motivation_summary_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
