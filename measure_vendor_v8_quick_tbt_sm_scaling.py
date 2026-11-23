#!/usr/bin/env python3
"""Collect paired Decode-only/overlap TBT with one anchor per partition.

The full partition profiler recreates a long Decode anchor for every repeat.
This quick microbenchmark creates it once, then submits multiple short Prefill
probes while the same anchor remains in Decode. Repeat rows are probe windows,
not independent request repetitions; downstream analysis must label them as
descriptive screening evidence.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence

from measure_vendor_v8_partition_profile import (
    RAW_FIELDNAMES,
    TOKEN_FIELDNAMES,
    TRIAL_FIELDNAMES,
    StreamResult,
    _wait_for_anchor_decode,
    build_profile_state_plan,
    get_model_id_from_server,
    get_requests_module,
    load_prepared_jsonl,
    load_slo_policy,
    lookup_request_slo,
    parse_args,
    percentile,
    stream_request,
    utc_now,
    validate_args,
    write_csv,
)


@dataclass(frozen=True)
class ProbeWindow:
    phase: str
    repeat_index: int
    control_lower_bound_ts: float
    result: StreamResult


def expand_rows(rows: Sequence[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        values.extend(
            [float(row["estimated_per_token_gap_ms"])]
            * int(row["chunk_tokens"])
        )
    return values


def classify_probe_window(
    anchor_rows: Sequence[dict[str, Any]],
    window: ProbeWindow,
    *,
    control_gap_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return fixed-tail control rows and full probe-overlap rows."""

    probe_submit = float(window.result.row["submit_monotonic_ts"])
    probe_first = float(window.result.row["first_token_monotonic_ts"])
    candidates = [
        row
        for row in anchor_rows
        if row["gap_start_monotonic_ts"] != ""
        and row["estimated_per_token_gap_ms"] != ""
        and float(row["gap_start_monotonic_ts"])
        >= window.control_lower_bound_ts
        and float(row["arrival_monotonic_ts"]) <= probe_submit
    ]
    if len(candidates) < control_gap_count:
        raise RuntimeError(
            "insufficient quick pre-probe control gaps: "
            f"phase={window.phase} repeat={window.repeat_index} "
            f"required={control_gap_count} observed={len(candidates)}; "
            "increase PROBE_DELAY_SEC or reduce DECODE_ONLY_CONTROL_GAPS"
        )
    control = candidates[-control_gap_count:]
    overlap = [
        row
        for row in anchor_rows
        if row["gap_start_monotonic_ts"] != ""
        and row["estimated_per_token_gap_ms"] != ""
        and float(row["gap_start_monotonic_ts"]) >= probe_submit
        and float(row["arrival_monotonic_ts"]) <= probe_first
    ]
    if not overlap:
        raise RuntimeError(
            "Prefill probe produced no full overlapping Decode gap: "
            f"phase={window.phase} repeat={window.repeat_index}"
        )
    return control, overlap


def tagged_token_rows(
    rows: Sequence[dict[str, Any]],
    *,
    common: dict[str, Any],
    before_probe_control: int,
    during_probe_prefill: int,
) -> list[dict[str, Any]]:
    return [
        {
            **common,
            **row,
            "before_probe_control": before_probe_control,
            "during_probe_prefill": during_probe_prefill,
        }
        for row in rows
    ]


def run_context(
    *,
    args: Any,
    model: str,
    request: Any,
    probe_request: Any,
    xargs: dict[str, float],
    probe_xargs: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    anchor_event = threading.Event()
    anchor_stop = threading.Event()
    windows: list[ProbeWindow] = []
    probe_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        anchor_id = f"{args.case_id}-ctx{request.input_tokens}-quick-anchor"
        anchor_future = executor.submit(
            stream_request,
            base_url=args.base_url,
            model=model,
            request=request,
            request_id=anchor_id,
            role="decode_anchor",
            tenant_id="quick_profile_decode",
            max_tokens=args.anchor_output_tokens,
            min_tokens=args.anchor_output_tokens,
            ignore_eos=bool(args.ignore_eos),
            request_timeout_sec=args.request_timeout_sec,
            xargs=xargs,
            first_token_event=anchor_event,
            stop_event=anchor_stop,
            allow_early_stop=True,
        )
        _wait_for_anchor_decode(
            [anchor_event], [anchor_future], args.first_token_timeout_sec
        )
        control_lower_bound = time.monotonic()
        try:
            plan = [
                (phase, repeat)
                for phase, count in (
                    ("warmup", args.warmup_count),
                    ("measurement", args.repeat_count),
                )
                for repeat in range(1, count + 1)
            ]
            for phase, repeat in plan:
                if args.probe_delay_sec > 0:
                    time.sleep(args.probe_delay_sec)
                if anchor_future.done():
                    anchor_future.result()
                    raise RuntimeError(
                        "quick Decode anchor completed before all probes; "
                        "increase ANCHOR_OUTPUT_TOKENS"
                    )
                probe_id = (
                    f"{args.case_id}-ctx{request.input_tokens}-{phase}-"
                    f"r{repeat}-quick-probe-ctx{probe_request.input_tokens}"
                )
                probe_result = stream_request(
                    base_url=args.base_url,
                    model=model,
                    request=probe_request,
                    request_id=probe_id,
                    role="prefill_probe",
                    tenant_id="quick_profile_prefill",
                    max_tokens=args.probe_output_tokens,
                    min_tokens=args.probe_output_tokens,
                    ignore_eos=bool(args.ignore_eos),
                    request_timeout_sec=args.request_timeout_sec,
                    xargs=probe_xargs,
                )
                windows.append(
                    ProbeWindow(
                        phase=phase,
                        repeat_index=repeat,
                        control_lower_bound_ts=control_lower_bound,
                        result=probe_result,
                    )
                )
                probe_rows.append(
                    {
                        "case_id": args.case_id,
                        "profile_id": args.profile_id,
                        "decode_sms": args.decode_sms,
                        "prefill_sms": args.prefill_sms,
                        "prefill_mode": args.prefill_mode,
                        "phase": phase,
                        "context_tokens": request.input_tokens,
                        "active_decode_requests": 1,
                        "repeat_index": repeat,
                        **probe_result.row,
                    }
                )
                control_lower_bound = float(
                    probe_result.row["completion_monotonic_ts"]
                )
                if args.cooldown_sec > 0:
                    time.sleep(args.cooldown_sec)
                    control_lower_bound = time.monotonic()
        finally:
            anchor_stop.set()
        anchor_result = anchor_future.result()

    token_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    anchor_rows = anchor_result.token_rows
    anchor_completion = float(anchor_result.row["completion_monotonic_ts"])
    for window in windows:
        if anchor_completion < float(
            window.result.row["first_token_monotonic_ts"]
        ):
            raise RuntimeError(
                "quick Decode anchor ended before a probe first token; "
                "increase ANCHOR_OUTPUT_TOKENS"
            )
        control_rows, overlap_rows = classify_probe_window(
            anchor_rows,
            window,
            control_gap_count=args.decode_only_control_gaps,
        )
        control = expand_rows(control_rows)
        overlap = expand_rows(overlap_rows)
        control_p50 = percentile(control, 50)
        overlap_p50 = percentile(overlap, 50)
        common = {
            "case_id": args.case_id,
            "profile_id": args.profile_id,
            "prefill_mode": args.prefill_mode,
            "phase": window.phase,
            "context_tokens": request.input_tokens,
            "active_decode_requests": 1,
            "repeat_index": window.repeat_index,
            "role": "decode_anchor",
        }
        token_rows.extend(
            tagged_token_rows(
                control_rows,
                common=common,
                before_probe_control=1,
                during_probe_prefill=0,
            )
        )
        token_rows.extend(
            tagged_token_rows(
                overlap_rows,
                common=common,
                before_probe_control=0,
                during_probe_prefill=1,
            )
        )
        probe_submit = float(window.result.row["submit_monotonic_ts"])
        probe_first = float(window.result.row["first_token_monotonic_ts"])
        probe_ttft = float(window.result.row["ttft_ms"])
        tbt_slo = float(xargs["tbt_slo_ms"])
        trial_rows.append(
            {
                "case_id": args.case_id,
                "profile_id": args.profile_id,
                "decode_sms": args.decode_sms,
                "prefill_sms": args.prefill_sms,
                "prefill_mode": args.prefill_mode,
                "phase": window.phase,
                "context_tokens": request.input_tokens,
                "probe_context_tokens": probe_request.input_tokens,
                "active_decode_requests": 1,
                "repeat_index": window.repeat_index,
                "status": "success",
                "error_message": "",
                "trial_start_monotonic_ts": window.control_lower_bound_ts,
                "probe_submit_monotonic_ts": probe_submit,
                "probe_first_token_monotonic_ts": probe_first,
                "all_decode_anchors_active_end_monotonic_ts": anchor_completion,
                "trial_end_monotonic_ts": float(
                    window.result.row["completion_monotonic_ts"]
                ),
                "probe_ttft_ms": probe_ttft,
                "probe_effective_prefill_tokens_per_s": (
                    probe_request.input_tokens / (probe_ttft / 1000.0)
                ),
                "decode_only_control_gap_count": len(control),
                "decode_only_control_p50_tbt_ms": control_p50,
                "decode_only_control_p90_tbt_ms": percentile(control, 90),
                "decode_only_control_p99_tbt_ms": percentile(control, 99),
                "overlap_decode_gap_count": len(overlap),
                "overlap_decode_p50_tbt_ms": overlap_p50,
                "overlap_decode_p90_tbt_ms": percentile(overlap, 90),
                "overlap_decode_mean_tbt_ms": statistics.fmean(overlap),
                "overlap_decode_p95_tbt_ms": percentile(overlap, 95),
                "overlap_decode_p99_tbt_ms": percentile(overlap, 99),
                "overlap_decode_tokens_per_s": (
                    1000.0 / statistics.fmean(overlap)
                ),
                "ttft_slo_ms": probe_xargs["ttft_prefill_deadline_ms"],
                "tbt_slo_ms": tbt_slo,
                "ttft_slo_met": int(
                    probe_ttft
                    <= float(probe_xargs["ttft_prefill_deadline_ms"])
                ),
                "tbt_token_slo_attainment": (
                    sum(value <= tbt_slo for value in overlap) / len(overlap)
                ),
                "overlap_p50_over_control_p50": (
                    overlap_p50 / control_p50
                ),
            }
        )
    anchor_raw = {
        "case_id": args.case_id,
        "profile_id": args.profile_id,
        "decode_sms": args.decode_sms,
        "prefill_sms": args.prefill_sms,
        "prefill_mode": args.prefill_mode,
        "phase": "measurement",
        "context_tokens": request.input_tokens,
        "active_decode_requests": 1,
        "repeat_index": 0,
        **anchor_result.row,
    }
    return [anchor_raw, *probe_rows], token_rows, trial_rows


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.active_decode_requests != [1]:
        raise ValueError(
            "quick TBT scaling supports ACTIVE_DECODE_REQUEST_COUNTS=1 only"
        )
    if args.decode_only_control_gaps <= 0:
        raise ValueError("quick TBT scaling requires paired control gaps")
    requests = load_prepared_jsonl(args.requests_jsonl)
    by_context = {request.input_tokens: request for request in requests}
    probe_context = args.probe_context_tokens or None
    required = [
        *args.context_tokens,
        *([probe_context] if probe_context is not None else []),
    ]
    missing = [value for value in dict.fromkeys(required) if value not in by_context]
    if missing:
        raise ValueError(f"prepared requests missing contexts: {missing}")
    admitted, skipped = build_profile_state_plan(
        args.context_tokens,
        [1],
        auxiliary_context_tokens=args.decode_aux_context_tokens,
        max_total_input_tokens=args.max_profile_total_input_tokens,
        probe_context_tokens=probe_context,
    )
    if not admitted:
        raise ValueError("resident-input filter skipped every quick state")
    if args.dry_run:
        for context, _decode_count in admitted:
            print(
                "[DRY-RUN][QUICK] "
                f"context={context} probe={probe_context or context} "
                f"one_anchor=1 probes={args.warmup_count + args.repeat_count}"
            )
        return 0

    http = get_requests_module()
    with http.Session() as session:
        model = args.model or get_model_id_from_server(
            session, args.base_url.rstrip("/")
        )
    policy = load_slo_policy(args.slo_policy_json)
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
        "active_decode_requests": [1],
        "probe_context_tokens": probe_context,
        "decode_only_control_gaps_per_anchor": args.decode_only_control_gaps,
        "repeat_count": args.repeat_count,
        "warmup_count": args.warmup_count,
        "model": model,
        "requests_jsonl": str(args.requests_jsonl),
        "slo_policy_json": str(args.slo_policy_json),
        "skipped_states": skipped,
        "tbt_token_count_source": "choice.token_ids",
        "tbt_requires_return_token_ids": True,
        "cluster_unit": "probe_window_on_single_decode_anchor",
        "independent_request_clusters": False,
        "prompt_sha256_by_context": {
            str(request.input_tokens): hashlib.sha256(
                request.prompt.encode("utf-8")
            ).hexdigest()
            for request in requests
        },
        "profile_contract": (
            "one long Decode anchor per partition; multiple sequential short "
            "Prefill probes; repeat rows are descriptive windows on that "
            "single anchor"
        ),
    }
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        for context, _decode_count in admitted:
            selected_probe_context = probe_context or context
            request_rows, token_rows, trial_rows = run_context(
                args=args,
                model=model,
                request=by_context[context],
                probe_request=by_context[selected_probe_context],
                xargs=lookup_request_slo(policy, context),
                probe_xargs=lookup_request_slo(
                    policy, selected_probe_context
                ),
            )
            write_csv(args.raw_csv, RAW_FIELDNAMES, request_rows)
            write_csv(args.token_csv, TOKEN_FIELDNAMES, token_rows)
            write_csv(args.trial_csv, TRIAL_FIELDNAMES, trial_rows)
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error_message"] = f"{type(exc).__name__}: {exc}"
        metadata["completed_at_utc"] = utc_now()
        args.metadata_json.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    metadata["status"] = "completed"
    metadata["completed_at_utc"] = utc_now()
    args.metadata_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[DONE][QUICK] "
        f"contexts={len(admitted)} windows_per_context={args.repeat_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
