#!/usr/bin/env python3
"""Measure one fixed P/D SM partition with decode/prefill overlap.

For every context bucket, one or more decode anchors are allowed to reach
decode.  A second request is then submitted as a prefill probe.  Client token
gaps measure decode TBT while the probe is in prefill, while server iteration
metrics provide phase-local prefill chunk service times for offline profile
generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import threading
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
    parse_sse_json,
    percentile,
)


def exact_completion_token_ids(choice: dict[str, Any]) -> list[int]:
    """Use only token IDs returned by vLLM for per-token TBT.

    Keep this small compatibility helper local to the partition profiler. Some
    remote experiment bundles predate the same helper in
    ``measure_real_workload_motivation.py``; importing it from there made the
    microbenchmark fail before the first request. Retokenizing text remains
    forbidden because it is not token-count preserving.
    """

    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list):
        raise RuntimeError(
            "completion response omitted token_ids; exact TBT measurement "
            "requires return_token_ids=true"
        )
    return [int(token_id) for token_id in token_ids]


def validate_completion_contract(
    *,
    output_tokens: int,
    requested_min_tokens: int,
    finish_reason: str | None,
    saw_done: bool,
    streaming: bool,
) -> None:
    """Fail closed on truncated or prematurely terminated measurements."""

    if streaming and not saw_done:
        raise RuntimeError("stream ended before the OpenAI [DONE] marker")
    if not finish_reason:
        raise RuntimeError("completion ended without a finish_reason")
    if requested_min_tokens > 0 and output_tokens < requested_min_tokens:
        raise RuntimeError(
            "completion returned fewer tokens than requested: "
            f"output_tokens={output_tokens}, "
            f"requested_min_tokens={requested_min_tokens}, "
            f"finish_reason={finish_reason}"
        )


RAW_FIELDNAMES = [
    "case_id",
    "profile_id",
    "decode_sms",
    "prefill_sms",
    "prefill_mode",
    "phase",
    "context_tokens",
    "active_decode_requests",
    "repeat_index",
    "role",
    "request_id",
    "status",
    "error_message",
    "submit_monotonic_ts",
    "first_token_monotonic_ts",
    "completion_monotonic_ts",
    "input_tokens",
    "output_tokens",
    "ttft_ms",
    "mean_tbt_ms",
    "p95_tbt_ms",
    "p99_tbt_ms",
    "completion_latency_ms",
    "ttft_prefill_deadline_ms",
    "tbt_slo_ms",
    "early_stopped",
]

TOKEN_FIELDNAMES = [
    "case_id",
    "profile_id",
    "prefill_mode",
    "phase",
    "context_tokens",
    "active_decode_requests",
    "repeat_index",
    "role",
    "request_id",
    "chunk_index",
    "chunk_tokens",
    "gap_start_monotonic_ts",
    "arrival_monotonic_ts",
    "estimated_per_token_gap_ms",
    "before_probe_control",
    "during_probe_prefill",
]

TRIAL_FIELDNAMES = [
    "case_id",
    "profile_id",
    "decode_sms",
    "prefill_sms",
    "prefill_mode",
    "phase",
    "context_tokens",
    "probe_context_tokens",
    "active_decode_requests",
    "repeat_index",
    "status",
    "error_message",
    "trial_start_monotonic_ts",
    "probe_submit_monotonic_ts",
    "probe_first_token_monotonic_ts",
    "all_decode_anchors_active_end_monotonic_ts",
    "trial_end_monotonic_ts",
    "probe_ttft_ms",
    "probe_effective_prefill_tokens_per_s",
    "decode_only_control_gap_count",
    "decode_only_control_p50_tbt_ms",
    "decode_only_control_p90_tbt_ms",
    "decode_only_control_p99_tbt_ms",
    "overlap_decode_gap_count",
    "overlap_decode_p50_tbt_ms",
    "overlap_decode_p90_tbt_ms",
    "overlap_decode_mean_tbt_ms",
    "overlap_decode_p95_tbt_ms",
    "overlap_decode_p99_tbt_ms",
    "overlap_decode_tokens_per_s",
    "ttft_slo_ms",
    "tbt_slo_ms",
    "ttft_slo_met",
    "tbt_token_slo_attainment",
    "overlap_p50_over_control_p50",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rounded(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return str(round(float(value), digits))


def load_slo_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    buckets = payload.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        raise ValueError("request SLO policy must contain non-empty buckets")
    contract = payload.get("slo_contract")
    if not isinstance(contract, dict):
        raise ValueError(
            "request SLO policy must include slo_contract with explicit "
            "TTFT/TBT measurement definitions"
        )
    expected = {
        "tbt_measurement_source": "choice.token_ids",
        "tbt_gap_field": "estimated_per_token_gap_ms",
        "tbt_unit": "ms_per_output_token",
        "tbt_requires_return_token_ids": True,
    }
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = contract.get(key)
        if actual_value != expected_value:
            mismatches.append(
                f"{key}={actual_value!r}, expected {expected_value!r}"
            )
    if mismatches:
        raise ValueError(
            "request SLO policy measurement contract does not match the "
            "vendor_v8 evaluation measurement path: "
            + "; ".join(mismatches)
        )
    return payload


def lookup_request_slo(
    policy: dict[str, Any],
    context_tokens: int,
) -> dict[str, float]:
    covering = sorted(
        (
            bucket
            for bucket in policy["buckets"]
            if int(bucket["max_context_tokens"]) >= context_tokens
        ),
        key=lambda bucket: int(bucket["max_context_tokens"]),
    )
    if not covering:
        maximum = max(
            int(bucket["max_context_tokens"])
            for bucket in policy["buckets"]
        )
        raise ValueError(
            f"context {context_tokens} exceeds largest SLO bucket {maximum}"
        )
    selected = covering[0]
    return {
        "ttft_prefill_deadline_ms": float(
            selected["ttft_prefill_deadline_ms"]
        ),
        "tbt_slo_ms": float(selected["tbt_slo_ms"]),
    }


@dataclass
class StreamResult:
    row: dict[str, Any]
    token_rows: list[dict[str, Any]]


def stream_request(
    *,
    base_url: str,
    model: str,
    request: PreparedRequest,
    request_id: str,
    role: str,
    tenant_id: str,
    max_tokens: int,
    min_tokens: int,
    ignore_eos: bool,
    request_timeout_sec: int,
    xargs: dict[str, Any],
    first_token_event: threading.Event | None = None,
    stop_event: threading.Event | None = None,
    allow_early_stop: bool = False,
) -> StreamResult:
    http = get_requests_module()
    submit_ts = time.monotonic()
    payload: dict[str, Any] = {
        "model": model,
        "prompt": request.prompt,
        "max_tokens": max_tokens,
        "min_tokens": min_tokens,
        "ignore_eos": ignore_eos,
        "stream": True,
        "return_token_ids": True,
        "request_id": request_id,
        "vllm_xargs": {"tenant_id": tenant_id, **xargs},
    }
    first_token_ts: float | None = None
    previous_token_ts: float | None = None
    token_rows: list[dict[str, Any]] = []
    per_token_gaps: list[float] = []
    finish_reason: str | None = None
    saw_done = False
    stopped_early = False
    output_tokens = 0
    with http.Session() as session:
        response = session.post(
            base_url.rstrip("/") + "/v1/completions",
            json=payload,
            headers={"X-Request-Id": request_id},
            stream=True,
            timeout=(10, request_timeout_sec),
        )
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if stop_event is not None and stop_event.is_set():
                stopped_early = True
                break
            if not raw_line:
                continue
            if raw_line.startswith("data: "):
                text = raw_line[6:].strip()
                if text == "[DONE]":
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
                if first_token_event is not None:
                    first_token_event.set()
            chunk_tokens = len(delta_token_ids)
            output_tokens += chunk_tokens
            gap_start = previous_token_ts
            per_token_gap: float | None = None
            if previous_token_ts is not None:
                gap_ms = (now - previous_token_ts) * 1000.0
                per_token_gap = gap_ms / chunk_tokens
                per_token_gaps.extend([per_token_gap] * chunk_tokens)
            token_rows.append(
                {
                    "request_id": request_id,
                    "role": role,
                    "chunk_index": len(token_rows),
                    "chunk_tokens": chunk_tokens,
                    "gap_start_monotonic_ts": (
                        "" if gap_start is None else gap_start
                    ),
                    "arrival_monotonic_ts": now,
                    "estimated_per_token_gap_ms": (
                        "" if per_token_gap is None else per_token_gap
                    ),
                }
            )
            previous_token_ts = now
        completion_ts = time.monotonic()
        if first_token_ts is None:
            raise RuntimeError("stream returned no completion token")
        if stopped_early:
            if not allow_early_stop:
                raise RuntimeError("request stream stopped unexpectedly")
        else:
            validate_completion_contract(
                output_tokens=output_tokens,
                requested_min_tokens=min_tokens,
                finish_reason=finish_reason,
                saw_done=saw_done,
                streaming=True,
            )
    row = {
        "request_id": request_id,
        "role": role,
        "status": "success",
        "error_message": "",
        "submit_monotonic_ts": submit_ts,
        "first_token_monotonic_ts": first_token_ts,
        "completion_monotonic_ts": completion_ts,
        "input_tokens": request.input_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": (first_token_ts - submit_ts) * 1000.0,
        "mean_tbt_ms": (
            statistics.fmean(per_token_gaps) if per_token_gaps else None
        ),
        "p95_tbt_ms": percentile(per_token_gaps, 95),
        "p99_tbt_ms": percentile(per_token_gaps, 99),
        "completion_latency_ms": (completion_ts - submit_ts) * 1000.0,
        "ttft_prefill_deadline_ms": xargs.get(
            "ttft_prefill_deadline_ms", ""
        ),
        "tbt_slo_ms": xargs.get("tbt_slo_ms", ""),
        "early_stopped": int(stopped_early),
    }
    return StreamResult(row=row, token_rows=token_rows)


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames, extrasaction="ignore"
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _wait_for_anchor_decode(
    events: Sequence[threading.Event],
    futures: Sequence[Future[StreamResult]],
    timeout_sec: float,
) -> None:
    deadline = time.monotonic() + timeout_sec
    for event, future in zip(events, futures):
        while not event.is_set():
            if future.done():
                # Surface an HTTP/stream failure without waiting for the full
                # first-token timeout.
                future.result()
                raise RuntimeError(
                    "decode anchor completed without a first-token signal"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "decode anchor did not produce its first token"
                )
            event.wait(min(0.1, remaining))


def collect_all_anchor_overlap_gaps(
    anchor_results: Sequence[StreamResult],
    *,
    probe_submit: float,
    probe_first: float,
) -> tuple[list[float], float]:
    """Return gaps observed while the probe and every anchor are active."""

    if not anchor_results:
        raise ValueError("at least one decode anchor is required")
    all_anchors_active_end = min(
        float(result.row["completion_monotonic_ts"])
        for result in anchor_results
    )
    overlap_end = min(probe_first, all_anchors_active_end)
    overlap_gaps: list[float] = []
    for result in anchor_results:
        for token_row in result.token_rows:
            gap_start = token_row["gap_start_monotonic_ts"]
            gap_value = token_row["estimated_per_token_gap_ms"]
            during_probe = (
                gap_start != ""
                and gap_value != ""
                and float(gap_start) >= probe_submit
                and float(token_row["arrival_monotonic_ts"]) <= overlap_end
            )
            token_row["during_probe_prefill"] = int(during_probe)
            if during_probe:
                overlap_gaps.extend(
                    [float(gap_value)] * int(token_row["chunk_tokens"])
                )
    return overlap_gaps, all_anchors_active_end


def collect_anchor_decode_only_control_gaps(
    anchor_results: Sequence[StreamResult],
    *,
    probe_submit: float,
    gaps_per_anchor: int,
) -> list[float]:
    """Select the last full Decode gaps before Prefill probe submission.

    Selecting a fixed tail count removes first-token/transient gaps while
    keeping the control immediately adjacent to the overlap window.  The
    result is paired with overlap gaps from the same Decode request(s).
    """

    if gaps_per_anchor <= 0:
        return []
    control_gaps: list[float] = []
    for result in anchor_results:
        candidates = [
            row
            for row in result.token_rows
            if row["gap_start_monotonic_ts"] != ""
            and row["estimated_per_token_gap_ms"] != ""
            and float(row["arrival_monotonic_ts"]) <= probe_submit
        ]
        if len(candidates) < gaps_per_anchor:
            raise RuntimeError(
                "insufficient pre-probe Decode control gaps: "
                f"request_id={result.row['request_id']} "
                f"required={gaps_per_anchor} observed={len(candidates)}; "
                "increase --probe-delay-sec"
            )
        for row in candidates[-gaps_per_anchor:]:
            row["before_probe_control"] = 1
            control_gaps.extend(
                [float(row["estimated_per_token_gap_ms"])]
                * int(row["chunk_tokens"])
            )
    return control_gaps


def build_profile_state_plan(
    context_tokens: Sequence[int],
    active_decode_requests: Sequence[int],
    *,
    auxiliary_context_tokens: int,
    max_total_input_tokens: int,
    probe_context_tokens: int | None = None,
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    state_plan = [
        (
            context,
            decode_count,
            context
            + (probe_context_tokens or context)
            + (decode_count - 1) * auxiliary_context_tokens,
        )
        for context in context_tokens
        for decode_count in active_decode_requests
    ]
    skipped = [
        {
            "context_tokens": context,
            "active_decode_requests": decode_count,
            "estimated_total_input_tokens": total_input_tokens,
            "reason": "resident_input_budget",
        }
        for context, decode_count, total_input_tokens in state_plan
        if max_total_input_tokens > 0
        and total_input_tokens > max_total_input_tokens
    ]
    admitted = [
        (context, decode_count)
        for context, decode_count, total_input_tokens in state_plan
        if max_total_input_tokens <= 0
        or total_input_tokens <= max_total_input_tokens
    ]
    return admitted, skipped


def run_trial(
    *,
    executor: ThreadPoolExecutor,
    base_url: str,
    model: str,
    request: PreparedRequest,
    probe_request: PreparedRequest,
    auxiliary_request: PreparedRequest,
    case_id: str,
    profile_id: str,
    phase: str,
    repeat_index: int,
    active_decode_requests: int,
    anchor_output_tokens: int,
    probe_output_tokens: int,
    probe_delay_sec: float,
    decode_only_control_gaps: int,
    first_token_timeout_sec: float,
    request_timeout_sec: int,
    ignore_eos: bool,
    xargs: dict[str, Any],
    probe_xargs: dict[str, Any],
    auxiliary_xargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    trial_start = time.monotonic()
    anchor_events = [
        threading.Event() for _ in range(active_decode_requests)
    ]
    anchor_stop_event = threading.Event()
    anchor_futures: list[Future[StreamResult]] = []

    def submit_anchor(
        index: int,
        event: threading.Event,
        anchor_request: PreparedRequest,
        anchor_xargs: dict[str, Any],
    ) -> Future[StreamResult]:
        request_id = (
            f"{case_id}-ctx{request.input_tokens}-{phase}-r{repeat_index}-"
            f"anchor{index}-ctx{anchor_request.input_tokens}"
        )
        return executor.submit(
            stream_request,
            base_url=base_url,
            model=model,
            request=anchor_request,
            request_id=request_id,
            role="decode_anchor",
            tenant_id=f"profile_decode_{index}",
            max_tokens=anchor_output_tokens,
            min_tokens=anchor_output_tokens,
            ignore_eos=ignore_eos,
            request_timeout_sec=request_timeout_sec,
            xargs=anchor_xargs,
            first_token_event=event,
            stop_event=anchor_stop_event,
            allow_early_stop=True,
        )

    # Establish the target-context anchor first. Auxiliary 10K anchors are
    # then cheap to bring into Decode and model the target validation state:
    # one long request plus N-1 short requests.
    anchor_futures.append(
        submit_anchor(0, anchor_events[0], request, xargs)
    )
    _wait_for_anchor_decode(
        anchor_events[:1], anchor_futures[:1], first_token_timeout_sec
    )
    for index in range(1, active_decode_requests):
        anchor_futures.append(
            submit_anchor(
                index,
                anchor_events[index],
                auxiliary_request,
                auxiliary_xargs,
            )
        )
    if active_decode_requests > 1:
        _wait_for_anchor_decode(
            anchor_events[1:],
            anchor_futures[1:],
            first_token_timeout_sec,
        )
    if probe_delay_sec > 0:
        time.sleep(probe_delay_sec)
    completed_before_probe = [
        index
        for index, future in enumerate(anchor_futures)
        if future.done()
    ]
    if completed_before_probe:
        for index in completed_before_probe:
            # Surface an underlying stream failure before reporting the
            # lifetime-contract error.
            anchor_futures[index].result()
        raise RuntimeError(
            "decode anchor(s) completed before probe submission: "
            f"{completed_before_probe}; increase --anchor-output-tokens"
        )
    probe_request_id = (
        f"{case_id}-ctx{request.input_tokens}-{phase}-r{repeat_index}-probe"
    )
    probe_future = executor.submit(
        stream_request,
        base_url=base_url,
        model=model,
        request=probe_request,
        request_id=probe_request_id,
        role="prefill_probe",
        tenant_id="profile_prefill",
        max_tokens=probe_output_tokens,
        min_tokens=probe_output_tokens,
        ignore_eos=ignore_eos,
        request_timeout_sec=request_timeout_sec,
        xargs=probe_xargs,
    )
    try:
        probe_result = probe_future.result()
    finally:
        # Closing the anchor streams after the probe avoids paying for every
        # requested anchor token once the measurement window is complete.
        anchor_stop_event.set()
    anchor_results = [future.result() for future in anchor_futures]
    trial_end = time.monotonic()

    probe_submit = float(probe_result.row["submit_monotonic_ts"])
    probe_first = float(probe_result.row["first_token_monotonic_ts"])
    for result in anchor_results:
        for token_row in result.token_rows:
            token_row["before_probe_control"] = 0
            token_row["during_probe_prefill"] = 0
    control_gaps = collect_anchor_decode_only_control_gaps(
        anchor_results,
        probe_submit=probe_submit,
        gaps_per_anchor=decode_only_control_gaps,
    )
    overlap_gaps, all_anchors_active_end = (
        collect_all_anchor_overlap_gaps(
            anchor_results,
            probe_submit=probe_submit,
            probe_first=probe_first,
        )
    )
    for token_row in probe_result.token_rows:
        token_row["before_probe_control"] = 0
        token_row["during_probe_prefill"] = 0

    probe_ttft_ms = float(probe_result.row["ttft_ms"])
    tbt_slo_ms = float(xargs["tbt_slo_ms"])
    control_p50 = percentile(control_gaps, 50)
    overlap_p50 = percentile(overlap_gaps, 50)
    trial_row = {
        "case_id": case_id,
        "profile_id": profile_id,
        "phase": phase,
        "context_tokens": request.input_tokens,
        "probe_context_tokens": probe_request.input_tokens,
        "active_decode_requests": active_decode_requests,
        "repeat_index": repeat_index,
        "status": "success",
        "error_message": "",
        "trial_start_monotonic_ts": trial_start,
        "probe_submit_monotonic_ts": probe_submit,
        "probe_first_token_monotonic_ts": probe_first,
        "all_decode_anchors_active_end_monotonic_ts": (
            all_anchors_active_end
        ),
        "trial_end_monotonic_ts": trial_end,
        "probe_ttft_ms": probe_ttft_ms,
        "probe_effective_prefill_tokens_per_s": (
            probe_request.input_tokens / (probe_ttft_ms / 1000.0)
        ),
        "decode_only_control_gap_count": len(control_gaps),
        "decode_only_control_p50_tbt_ms": control_p50 or "",
        "decode_only_control_p90_tbt_ms": (
            percentile(control_gaps, 90) or ""
        ),
        "decode_only_control_p99_tbt_ms": (
            percentile(control_gaps, 99) or ""
        ),
        "overlap_decode_gap_count": len(overlap_gaps),
        "overlap_decode_p50_tbt_ms": overlap_p50 or "",
        "overlap_decode_p90_tbt_ms": (
            percentile(overlap_gaps, 90) or ""
        ),
        "overlap_decode_mean_tbt_ms": (
            statistics.fmean(overlap_gaps) if overlap_gaps else ""
        ),
        "overlap_decode_p95_tbt_ms": (
            percentile(overlap_gaps, 95) or ""
        ),
        "overlap_decode_p99_tbt_ms": (
            percentile(overlap_gaps, 99) or ""
        ),
        "overlap_decode_tokens_per_s": (
            active_decode_requests * 1000.0 / statistics.fmean(overlap_gaps)
            if overlap_gaps
            else ""
        ),
        "ttft_slo_ms": probe_xargs["ttft_prefill_deadline_ms"],
        "tbt_slo_ms": tbt_slo_ms,
        "ttft_slo_met": int(
            probe_ttft_ms
            <= float(probe_xargs["ttft_prefill_deadline_ms"])
        ),
        "overlap_p50_over_control_p50": (
            overlap_p50 / control_p50
            if overlap_p50 is not None
            and control_p50 is not None
            and control_p50 > 0
            else ""
        ),
        "tbt_token_slo_attainment": (
            sum(value <= tbt_slo_ms for value in overlap_gaps)
            / len(overlap_gaps)
            if overlap_gaps
            else ""
        ),
    }
    request_rows = [
        *(result.row for result in anchor_results),
        probe_result.row,
    ]
    token_rows = [
        *(row for result in anchor_results for row in result.token_rows),
        *probe_result.token_rows,
    ]
    return request_rows, token_rows, trial_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile one fixed vendor_v8 P/D partition"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--slo-policy-json", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--decode-sms", type=int, required=True)
    parser.add_argument("--prefill-sms", type=int, required=True)
    parser.add_argument(
        "--prefill-mode", choices=["native", "ring"], required=True
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--active-decode-requests",
        type=int,
        nargs="+",
        default=[1],
    )
    parser.add_argument(
        "--decode-aux-context-tokens",
        type=int,
        default=10_000,
        help=(
            "Context length for each additional Decode anchor after the "
            "single target-context anchor"
        ),
    )
    parser.add_argument(
        "--probe-context-tokens",
        type=int,
        default=0,
        help=(
            "Use one fixed Prefill-probe context for every Decode context; "
            "zero keeps the historical same-context probe"
        ),
    )
    parser.add_argument(
        "--max-profile-total-input-tokens",
        type=int,
        default=0,
        help=(
            "Skip states whose target Decode anchor, auxiliary Decode "
            "anchors, and Prefill probe exceed this resident-input budget; "
            "zero disables the filter"
        ),
    )
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--warmup-count", type=int, default=1)
    parser.add_argument("--anchor-output-tokens", type=int, default=128)
    parser.add_argument("--probe-output-tokens", type=int, default=8)
    parser.add_argument("--probe-delay-sec", type=float, default=0.0)
    parser.add_argument(
        "--decode-only-control-gaps",
        type=int,
        default=0,
        help=(
            "Select this many full pre-probe Decode gaps per anchor as a "
            "paired Decode-only control; zero preserves historical output"
        ),
    )
    parser.add_argument("--cooldown-sec", type=float, default=3.0)
    parser.add_argument("--first-token-timeout-sec", type=float, default=7200)
    parser.add_argument("--request-timeout-sec", type=int, default=14400)
    parser.add_argument("--ignore-eos", type=int, choices=[0, 1], default=1)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--token-csv", type=Path, required=True)
    parser.add_argument("--trial-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.decode_sms <= 0 or args.prefill_sms <= 0:
        raise ValueError("partition profiler requires a concurrent P/D split")
    if args.repeat_count <= 0 or args.warmup_count < 0:
        raise ValueError("repeat count must be positive and warmup non-negative")
    if args.anchor_output_tokens < 2 or args.probe_output_tokens <= 0:
        raise ValueError("anchor output must be >=2 and probe output positive")
    if args.probe_delay_sec < 0 or args.cooldown_sec < 0:
        raise ValueError("probe delay and cooldown must be non-negative")
    if any(value <= 0 for value in args.active_decode_requests):
        raise ValueError("active decode request counts must be positive")
    if args.decode_aux_context_tokens <= 0:
        raise ValueError("decode auxiliary context must be positive")
    if args.probe_context_tokens < 0:
        raise ValueError("probe context must be non-negative")
    if args.decode_only_control_gaps < 0:
        raise ValueError("Decode-only control gaps must be non-negative")
    if args.decode_only_control_gaps > 0 and args.probe_delay_sec <= 0:
        raise ValueError(
            "Decode-only controls require a positive --probe-delay-sec"
        )
    if args.max_profile_total_input_tokens < 0:
        raise ValueError("profile resident-input budget must be non-negative")
    for path in (
        args.raw_csv,
        args.token_csv,
        args.trial_csv,
        args.metadata_json,
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    requests = load_prepared_jsonl(args.requests_jsonl)
    by_context = {request.input_tokens: request for request in requests}
    probe_context = args.probe_context_tokens or None
    required_contexts = [
        *args.context_tokens,
        args.decode_aux_context_tokens,
        *([probe_context] if probe_context is not None else []),
    ]
    missing = [
        value
        for value in dict.fromkeys(required_contexts)
        if value not in by_context
    ]
    if missing:
        raise ValueError(f"prepared requests missing contexts: {missing}")
    slo_policy = load_slo_policy(args.slo_policy_json)

    admitted_states, skipped_states = build_profile_state_plan(
        args.context_tokens,
        args.active_decode_requests,
        auxiliary_context_tokens=args.decode_aux_context_tokens,
        max_total_input_tokens=args.max_profile_total_input_tokens,
        probe_context_tokens=probe_context,
    )
    if not admitted_states:
        raise ValueError("resident-input filter skipped every profile state")
    trial_plan = [
        (context, decode_count, phase, repeat)
        for context, decode_count in admitted_states
        for phase, count in (
            ("warmup", args.warmup_count),
            ("measurement", args.repeat_count),
        )
        for repeat in range(1, count + 1)
    ]
    if args.dry_run:
        for skipped in skipped_states:
            print(
                "[DRY-RUN][SKIP] "
                f"context={skipped['context_tokens']} "
                f"decode_requests={skipped['active_decode_requests']} "
                f"estimated_total_input_tokens="
                f"{skipped['estimated_total_input_tokens']} "
                f"budget={args.max_profile_total_input_tokens}"
            )
        for context, decode_count, phase, repeat in trial_plan:
            print(
                "[DRY-RUN] "
                f"profile={args.profile_id} mode={args.prefill_mode} "
                f"context={context} decode_requests={decode_count} "
                f"phase={phase} repeat={repeat}"
            )
        return 0

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
        "profile_id": args.profile_id,
        "decode_sms": args.decode_sms,
        "prefill_sms": args.prefill_sms,
        "prefill_mode": args.prefill_mode,
        "contexts": args.context_tokens,
        "active_decode_requests": args.active_decode_requests,
        "decode_aux_context_tokens": args.decode_aux_context_tokens,
        "probe_context_tokens": probe_context,
        "decode_only_control_gaps_per_anchor": (
            args.decode_only_control_gaps
        ),
        "max_profile_total_input_tokens": (
            args.max_profile_total_input_tokens
        ),
        "skipped_states": skipped_states,
        "repeat_count": args.repeat_count,
        "warmup_count": args.warmup_count,
        "model": model,
        "requests_jsonl": str(args.requests_jsonl),
        "prompt_sha256_by_context": {
            str(request.input_tokens): hashlib.sha256(
                request.prompt.encode("utf-8")
            ).hexdigest()
            for request in requests
        },
        "slo_policy_json": str(args.slo_policy_json),
        "tbt_token_count_source": "choice.token_ids",
        "tbt_requires_return_token_ids": True,
        "cluster_unit": "independent_request_repeat",
        "independent_request_clusters": True,
        "profile_contract": (
            "one target-context anchor reaches Decode first, N-1 auxiliary "
            "short-context anchors then reach Decode, and every anchor "
            "remains active through Prefill probe submission; optional "
            "paired Decode-only controls use the last full gaps before the "
            "probe, and overlap TBT uses full gaps before the first anchor "
            "exits"
        ),
    }
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    failed = False
    max_workers = max(args.active_decode_requests) + 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, (context, decode_count, phase, repeat) in enumerate(
            trial_plan
        ):
            common = {
                "case_id": args.case_id,
                "profile_id": args.profile_id,
                "decode_sms": args.decode_sms,
                "prefill_sms": args.prefill_sms,
                "prefill_mode": args.prefill_mode,
                "phase": phase,
                "context_tokens": context,
                "active_decode_requests": decode_count,
                "repeat_index": repeat,
            }
            print(
                "[TRIAL] "
                f"{index + 1}/{len(trial_plan)} context={context} "
                f"decode_requests={decode_count} phase={phase} repeat={repeat}",
                flush=True,
            )
            try:
                xargs = lookup_request_slo(slo_policy, context)
                selected_probe_context = probe_context or context
                probe_xargs = lookup_request_slo(
                    slo_policy, selected_probe_context
                )
                auxiliary_xargs = lookup_request_slo(
                    slo_policy, args.decode_aux_context_tokens
                )
                request_rows, token_rows, trial_row = run_trial(
                    executor=executor,
                    base_url=args.base_url,
                    model=model,
                    request=by_context[context],
                    probe_request=by_context[selected_probe_context],
                    auxiliary_request=by_context[
                        args.decode_aux_context_tokens
                    ],
                    case_id=args.case_id,
                    profile_id=args.profile_id,
                    phase=phase,
                    repeat_index=repeat,
                    active_decode_requests=decode_count,
                    anchor_output_tokens=args.anchor_output_tokens,
                    probe_output_tokens=args.probe_output_tokens,
                    probe_delay_sec=args.probe_delay_sec,
                    decode_only_control_gaps=(
                        args.decode_only_control_gaps
                    ),
                    first_token_timeout_sec=args.first_token_timeout_sec,
                    request_timeout_sec=args.request_timeout_sec,
                    ignore_eos=bool(args.ignore_eos),
                    xargs=xargs,
                    probe_xargs=probe_xargs,
                    auxiliary_xargs=auxiliary_xargs,
                )
                write_csv(
                    args.raw_csv,
                    RAW_FIELDNAMES,
                    [{**common, **row} for row in request_rows],
                )
                write_csv(
                    args.token_csv,
                    TOKEN_FIELDNAMES,
                    [{**common, **row} for row in token_rows],
                )
                write_csv(
                    args.trial_csv,
                    TRIAL_FIELDNAMES,
                    [{**common, **trial_row}],
                )
                print(
                    "[RESULT] "
                    f"probe_ttft_ms={trial_row['probe_ttft_ms']:.3f} "
                    "overlap_p95_tbt_ms="
                    f"{trial_row['overlap_decode_p95_tbt_ms']}",
                    flush=True,
                )
            except Exception as exc:
                failed = True
                error_row = {
                    **common,
                    "status": "failed",
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
                write_csv(
                    args.trial_csv, TRIAL_FIELDNAMES, [error_row]
                )
                print(f"[ERROR] {error_row['error_message']}", flush=True)
            if index + 1 < len(trial_plan) and args.cooldown_sec > 0:
                time.sleep(args.cooldown_sec)

    metadata["status"] = "failed" if failed else "completed"
    metadata["completed_at_utc"] = utc_now()
    args.metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
