#!/usr/bin/env python3
"""Replay a prepared LoongServe Mixed trace against a vLLM HTTP server.

The trace is dispatched open-loop. ``--arrival-time-scale`` multiplies every
paper-trace offset: values below one make arrivals faster and values above one
make them slower. ``--termination-mode time`` preserves the deadline-based
contract. ``--termination-mode requests`` instead submits the first N trace
rows in their immutable file order, retains their original scaled arrival
offsets, and drains every submitted request before ending the observation.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PLAN_FIELDS = [
    "event_index",
    "request_id",
    "dataset",
    "role",
    "tenant_id",
    "base_scheduled_offset_s",
    "effective_scheduled_offset_s",
    "scheduled_monotonic_ts",
    "dispatch_monotonic_ts",
    "arrival_lag_ms",
    "input_tokens",
    "requested_output_tokens",
    "source_id",
    "submission_status",
]

RAW_FIELDS = PLAN_FIELDS + [
    "status",
    "error_message",
    "submit_monotonic_ts",
    "first_token_monotonic_ts",
    "completion_monotonic_ts",
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
    "dataset",
    "role",
    "input_tokens",
    "chunk_index",
    "chunk_tokens",
    "gap_start_monotonic_ts",
    "arrival_monotonic_ts",
    "arrival_offset_from_trace_start_s",
    "estimated_per_token_gap_ms",
    "tbt_slo_ms",
    "tbt_slo_met",
]

SUMMARY_FIELDS = [
    "dataset",
    "request_count",
    "submitted_count",
    "success_count",
    "failure_count",
    "censored_count",
    "not_submitted_count",
    "input_tokens_min",
    "input_tokens_max",
    "completed_output_tokens",
    "observed_output_tokens",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p99_ms",
    "request_p50_tbt_p50_ms",
    "request_p95_tbt_p50_ms",
    "request_p99_tbt_p50_ms",
    "completion_p50_ms",
    "completion_p90_ms",
    "completion_p99_ms",
    "ttft_slo_attainment",
    "tbt_token_slo_attainment",
]

NOT_SUBMITTED_STATUSES = frozenset(
    {
        "not_submitted_time_limit",
        "not_submitted_request_limit",
        "not_submitted_probe_control",
    }
)


class IncompleteCompletionError(RuntimeError):
    """The HTTP response ended without satisfying the token contract."""


def is_not_submitted_status(status: object) -> bool:
    """Return whether a trace row was intentionally outside the replay set."""

    return str(status) in NOT_SUBMITTED_STATUSES


def replay_selection_state(
    *,
    position: int,
    effective_offset_s: float,
    termination_mode: str,
    request_limit: int,
    time_limit_sec: float,
) -> str:
    """Classify one trace row without changing its immutable input order."""

    if termination_mode == "requests":
        return "submit" if position < request_limit else "not_submitted_request_limit"
    if time_limit_sec > 0 and effective_offset_s >= time_limit_sec:
        return "not_submitted_time_limit"
    return "submit"


def request_timeout_total_seconds(value: float) -> float | None:
    """Translate explicit zero into an unlimited HTTP processing duration."""

    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            "request-timeout-sec must be finite and non-negative; "
            "0 disables the request processing timeout"
        )
    return None if value == 0 else value


def exact_completion_token_ids(choice: dict[str, Any]) -> list[int]:
    """Return server-provided token IDs; text-delta retokenization is invalid."""

    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list):
        raise IncompleteCompletionError(
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
    if streaming and not saw_done:
        raise IncompleteCompletionError("stream ended before the OpenAI [DONE] marker")
    if not finish_reason:
        raise IncompleteCompletionError("completion ended without a finish_reason")
    if requested_min_tokens > 0 and output_tokens < requested_min_tokens:
        raise IncompleteCompletionError(
            "completion returned fewer tokens than requested: "
            f"output_tokens={output_tokens}, "
            f"requested_min_tokens={requested_min_tokens}, "
            f"finish_reason={finish_reason}"
        )


def percentile(values: Sequence[float], pct: float) -> float | None:
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
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


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
    mismatches = [
        f"{key}={contract.get(key)!r}, expected {expected_value!r}"
        for key, expected_value in expected.items()
        if contract.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            "request SLO policy measurement contract does not match the "
            "LoongServe evaluation measurement path: " + "; ".join(mismatches)
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
        maximum = max(int(bucket["max_context_tokens"]) for bucket in policy["buckets"])
        raise ValueError(
            f"context {context_tokens} exceeds largest SLO bucket {maximum}"
        )
    selected = covering[0]
    return {
        "ttft_prefill_deadline_ms": float(selected["ttft_prefill_deadline_ms"]),
        "tbt_slo_ms": float(selected["tbt_slo_ms"]),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LoongServeRequest:
    event_index: int
    request_id: str
    tenant_id: str
    dataset: str
    role: str
    prompt: str
    input_tokens: int
    output_tokens: int
    base_scheduled_offset_s: float
    source_id: str


@dataclass
class RequestObservation:
    request: LoongServeRequest
    effective_offset_s: float
    scheduled_ts: float = 0.0
    dispatch_ts: float | None = None
    submit_ts: float | None = None
    first_token_ts: float | None = None
    completion_ts: float | None = None
    output_tokens: int = 0
    finish_reason: str | None = None
    saw_done: bool = False
    status: str = "planned"
    error_message: str = ""
    token_rows: list[dict[str, Any]] = field(default_factory=list)
    per_token_gaps_ms: list[float] = field(default_factory=list)


def load_trace(path: Path) -> list[LoongServeRequest]:
    requests: list[LoongServeRequest] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            maximum = int(payload["max_tokens"])
            minimum = int(payload["min_tokens"])
            if maximum <= 0 or minimum != maximum:
                raise ValueError(
                    f"line {line_number}: expected positive " "min_tokens == max_tokens"
                )
            requests.append(
                LoongServeRequest(
                    event_index=int(payload.get("event_index", len(requests))),
                    request_id=str(payload["request_id"]),
                    tenant_id=str(payload.get("tenant_id", payload["request_id"])),
                    dataset=str(payload["dataset"]),
                    role=str(payload.get("role", "request")),
                    prompt=str(payload["prompt"]),
                    input_tokens=int(payload["input_tokens"]),
                    output_tokens=maximum,
                    base_scheduled_offset_s=float(payload["base_scheduled_offset_s"]),
                    source_id=str(payload.get("source_id", line_number)),
                )
            )
    if not requests:
        raise RuntimeError(f"no requests found in {path}")
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("trace request_id values must be unique")
    if any(
        request.input_tokens <= 0
        or request.base_scheduled_offset_s < 0
        or not math.isfinite(request.base_scheduled_offset_s)
        for request in requests
    ):
        raise ValueError("trace contains an invalid length or offset")
    if any(
        current.base_scheduled_offset_s < previous.base_scheduled_offset_s
        for previous, current in zip(requests, requests[1:])
    ):
        raise ValueError("trace offsets must be nondecreasing")
    return requests


def scaled_offset(base_offset_s: float, scale: float) -> float:
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("arrival-time-scale must be finite and positive")
    return base_offset_s * scale


def build_observations(
    requests: Sequence[LoongServeRequest],
    *,
    arrival_time_scale: float,
) -> list[RequestObservation]:
    return [
        RequestObservation(
            request=request,
            effective_offset_s=scaled_offset(
                request.base_scheduled_offset_s, arrival_time_scale
            ),
        )
        for request in requests
    ]


def select_slo(
    policy: dict[str, Any] | None,
    context_tokens: int,
    *,
    overflow_policy: str,
    default_ttft_slo_ms: float,
    default_tbt_slo_ms: float,
) -> dict[str, float]:
    if policy is not None:
        try:
            return lookup_request_slo(policy, context_tokens)
        except ValueError:
            if overflow_policy != "last_bucket":
                raise
            selected = max(
                policy["buckets"],
                key=lambda bucket: int(bucket["max_context_tokens"]),
            )
            return {
                "ttft_prefill_deadline_ms": float(selected["ttft_prefill_deadline_ms"]),
                "tbt_slo_ms": float(selected["tbt_slo_ms"]),
            }
    if default_ttft_slo_ms > 0 and default_tbt_slo_ms > 0:
        return {
            "ttft_prefill_deadline_ms": default_ttft_slo_ms,
            "tbt_slo_ms": default_tbt_slo_ms,
        }
    return {}


async def get_model_id(session: Any, base_url: str) -> str:
    async with session.get(base_url.rstrip("/") + "/v1/models") as response:
        response.raise_for_status()
        payload = await response.json()
    models = payload.get("data") or []
    if not models:
        raise RuntimeError("/v1/models returned no model")
    return str(models[0]["id"])


def parse_sse_json(line: str) -> dict[str, Any] | None:
    if not line.startswith("data: "):
        return None
    payload = line[6:].strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


def build_completion_payload(
    request: LoongServeRequest,
    *,
    model: str,
    slo: dict[str, float],
    ignore_eos: bool,
) -> dict[str, Any]:
    """Build the exact-token replay request without echoing prompt token IDs."""

    return {
        "model": model,
        "prompt": request.prompt,
        "max_tokens": request.output_tokens,
        "min_tokens": request.output_tokens,
        "ignore_eos": ignore_eos,
        "stream": True,
        "return_token_ids": True,
        "return_prompt_token_ids": False,
        "request_id": request.request_id,
        "vllm_xargs": {"tenant_id": request.tenant_id, **slo},
    }


async def execute_request(
    observation: RequestObservation,
    *,
    session: Any,
    base_url: str,
    model: str,
    slo: dict[str, float],
    trace_start: float,
    ignore_eos: bool,
) -> RequestObservation:
    request = observation.request
    payload = build_completion_payload(
        request,
        model=model,
        slo=slo,
        ignore_eos=ignore_eos,
    )
    observation.dispatch_ts = time.monotonic()
    observation.submit_ts = observation.dispatch_ts
    observation.status = "submitted"
    previous_token_ts: float | None = None
    response_content_type = ""
    response_request_id = ""
    sse_message_count = 0
    choice_chunk_count = 0
    text_chunk_count = 0
    token_ids_field_count = 0
    empty_token_ids_chunk_count = 0
    nonempty_token_ids_chunk_count = 0
    last_message_keys: list[str] = []
    last_choice_keys: list[str] = []

    def stream_diagnostics() -> str:
        return (
            "return_token_ids_requested=true, "
            "return_prompt_token_ids_requested=false, "
            f"content_type={response_content_type!r}, "
            f"response_request_id={response_request_id!r}, "
            f"sse_messages={sse_message_count}, "
            f"choice_chunks={choice_chunk_count}, "
            f"text_chunks={text_chunk_count}, "
            f"token_ids_fields={token_ids_field_count}, "
            f"nonempty_token_id_chunks={nonempty_token_ids_chunk_count}, "
            f"empty_token_id_chunks={empty_token_ids_chunk_count}, "
            f"saw_done={observation.saw_done}, "
            f"finish_reason={observation.finish_reason!r}, "
            f"last_message_keys={last_message_keys}, "
            f"last_choice_keys={last_choice_keys}"
        )

    try:
        async with session.post(
            base_url.rstrip("/") + "/v1/completions",
            json=payload,
            headers={"X-Request-Id": request.request_id},
        ) as response:
            response.raise_for_status()
            response_content_type = response.headers.get("Content-Type", "")
            response_request_id = response.headers.get("X-Request-Id", "")
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                if line.startswith("data: ") and line[6:].strip() == "[DONE]":
                    observation.saw_done = True
                    break
                message = parse_sse_json(line)
                if not message:
                    continue
                sse_message_count += 1
                last_message_keys = sorted(str(key) for key in message)
                choices = message.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                choice_chunk_count += 1
                last_choice_keys = sorted(str(key) for key in choice)
                if choice.get("text"):
                    text_chunk_count += 1
                if choice.get("finish_reason") is not None:
                    observation.finish_reason = str(choice["finish_reason"])
                token_ids = choice.get("token_ids")
                if token_ids is None:
                    continue
                token_ids_field_count += 1
                try:
                    delta_token_ids = exact_completion_token_ids(choice)
                except IncompleteCompletionError as exc:
                    raise IncompleteCompletionError(
                        f"{exc}; stream diagnostics: {stream_diagnostics()}"
                    ) from exc
                if not delta_token_ids:
                    empty_token_ids_chunk_count += 1
                    continue
                nonempty_token_ids_chunk_count += 1
                now = time.monotonic()
                if observation.first_token_ts is None:
                    observation.first_token_ts = now
                chunk_tokens = len(delta_token_ids)
                observation.output_tokens += chunk_tokens
                gap_ms: float | None = None
                if previous_token_ts is not None:
                    gap_ms = (now - previous_token_ts) * 1000.0
                    per_token_gap = gap_ms / chunk_tokens
                    observation.per_token_gaps_ms.extend([per_token_gap] * chunk_tokens)
                else:
                    per_token_gap = None
                observation.token_rows.append(
                    {
                        "event_index": request.event_index,
                        "request_id": request.request_id,
                        "dataset": request.dataset,
                        "role": request.role,
                        "input_tokens": request.input_tokens,
                        "chunk_index": len(observation.token_rows),
                        "chunk_tokens": chunk_tokens,
                        "gap_start_monotonic_ts": (
                            "" if previous_token_ts is None else previous_token_ts
                        ),
                        "arrival_monotonic_ts": now,
                        "arrival_offset_from_trace_start_s": now - trace_start,
                        "estimated_per_token_gap_ms": (
                            "" if per_token_gap is None else per_token_gap
                        ),
                        "tbt_slo_ms": slo.get("tbt_slo_ms", ""),
                        "tbt_slo_met": (
                            ""
                            if per_token_gap is None or "tbt_slo_ms" not in slo
                            else int(per_token_gap <= slo["tbt_slo_ms"])
                        ),
                    }
                )
                previous_token_ts = now
        observation.completion_ts = time.monotonic()
        if observation.first_token_ts is None:
            if choice_chunk_count == 0:
                cause = "server emitted no completion choice chunks"
            elif token_ids_field_count == 0 and text_chunk_count > 0:
                cause = (
                    "server emitted text but omitted choice.token_ids; the "
                    "deployed OpenAI serving schema/path is incompatible with "
                    "the exact-TBT contract"
                )
            elif token_ids_field_count == 0:
                cause = "server omitted choice.token_ids from every choice chunk"
            else:
                cause = "server emitted only empty choice.token_ids deltas"
            raise IncompleteCompletionError(
                "stream returned no completion token: "
                f"{cause}; stream diagnostics: {stream_diagnostics()}"
            )
        validate_completion_contract(
            output_tokens=observation.output_tokens,
            requested_min_tokens=request.output_tokens,
            finish_reason=observation.finish_reason,
            saw_done=observation.saw_done,
            streaming=True,
        )
        observation.status = "success"
    except asyncio.CancelledError:
        observation.completion_ts = time.monotonic()
        observation.status = "censored_time_limit"
        observation.error_message = "global time limit cancelled HTTP stream"
        return observation
    except Exception as exc:
        observation.completion_ts = time.monotonic()
        observation.status = "failed"
        observation.error_message = f"{type(exc).__name__}: {exc}"
    return observation


def observation_row(
    observation: RequestObservation,
    slo: dict[str, float],
) -> dict[str, Any]:
    request = observation.request
    submit_ts = observation.submit_ts
    first_ts = observation.first_token_ts
    completion_ts = observation.completion_ts
    gaps = observation.per_token_gaps_ms
    success = observation.status == "success"
    tbt_slo = slo.get("tbt_slo_ms")
    return {
        "event_index": request.event_index,
        "request_id": request.request_id,
        "dataset": request.dataset,
        "role": request.role,
        "tenant_id": request.tenant_id,
        "base_scheduled_offset_s": request.base_scheduled_offset_s,
        "effective_scheduled_offset_s": observation.effective_offset_s,
        "scheduled_monotonic_ts": observation.scheduled_ts or "",
        "dispatch_monotonic_ts": observation.dispatch_ts or "",
        "arrival_lag_ms": (
            ""
            if observation.dispatch_ts is None or not observation.scheduled_ts
            else (observation.dispatch_ts - observation.scheduled_ts) * 1000.0
        ),
        "input_tokens": request.input_tokens,
        "requested_output_tokens": request.output_tokens,
        "source_id": request.source_id,
        "submission_status": (
            "not_submitted"
            if is_not_submitted_status(observation.status)
            else "submitted"
        ),
        "status": observation.status,
        "error_message": observation.error_message,
        "submit_monotonic_ts": submit_ts or "",
        "first_token_monotonic_ts": first_ts or "",
        "completion_monotonic_ts": completion_ts or "",
        "output_tokens": observation.output_tokens,
        "ttft_ms": (
            ""
            if first_ts is None or submit_ts is None
            else (first_ts - submit_ts) * 1000.0
        ),
        "mean_tbt_ms": statistics.fmean(gaps) if gaps else "",
        "p50_tbt_ms": percentile(gaps, 50) or "",
        "p90_tbt_ms": percentile(gaps, 90) or "",
        "p95_tbt_ms": percentile(gaps, 95) or "",
        "p99_tbt_ms": percentile(gaps, 99) or "",
        "completion_latency_ms": (
            ""
            if completion_ts is None or submit_ts is None
            else (completion_ts - submit_ts) * 1000.0
        ),
        "ttft_slo_ms": slo.get("ttft_prefill_deadline_ms", ""),
        "tbt_slo_ms": slo.get("tbt_slo_ms", ""),
        "ttft_slo_met": (
            ""
            if not success
            or first_ts is None
            or submit_ts is None
            or "ttft_prefill_deadline_ms" not in slo
            else int((first_ts - submit_ts) * 1000.0 <= slo["ttft_prefill_deadline_ms"])
        ),
        "tbt_token_slo_attainment": (
            ""
            if not success or not gaps or tbt_slo is None
            else sum(gap <= tbt_slo for gap in gaps) / len(gaps)
        ),
    }


def summarize(
    raw_rows: Sequence[dict[str, Any]],
    token_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    datasets = ["all"] + sorted({str(row["dataset"]) for row in raw_rows})
    for dataset in datasets:
        selected = [
            row for row in raw_rows if dataset == "all" or row["dataset"] == dataset
        ]
        selected_ids = {str(row["request_id"]) for row in selected}
        successful = [row for row in selected if row["status"] == "success"]
        submitted = [
            row for row in selected if not is_not_submitted_status(row["status"])
        ]
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

        contexts = [int(row["input_tokens"]) for row in selected]
        ttft_slo_values = [
            float(row["ttft_slo_met"])
            for row in successful
            if row.get("ttft_slo_met") not in ("", None)
        ]
        token_slo_values = [
            float(row["tbt_slo_met"])
            for row in selected_tokens
            if row.get("tbt_slo_met") not in ("", None)
            for _ in range(int(row["chunk_tokens"]))
        ]
        summaries.append(
            {
                "dataset": dataset,
                "request_count": len(selected),
                "submitted_count": len(submitted),
                "success_count": len(successful),
                "failure_count": sum(row["status"] == "failed" for row in selected),
                "censored_count": sum(
                    row["status"] == "censored_time_limit" for row in selected
                ),
                "not_submitted_count": sum(
                    is_not_submitted_status(row["status"]) for row in selected
                ),
                "input_tokens_min": min(contexts) if contexts else "",
                "input_tokens_max": max(contexts) if contexts else "",
                "completed_output_tokens": sum(
                    int(row["output_tokens"]) for row in successful
                ),
                "observed_output_tokens": sum(
                    int(row["output_tokens"]) for row in submitted
                ),
                "ttft_p50_ms": percentile(values("ttft_ms"), 50) or "",
                "ttft_p90_ms": percentile(values("ttft_ms"), 90) or "",
                "ttft_p99_ms": percentile(values("ttft_ms"), 99) or "",
                "request_p50_tbt_p50_ms": percentile(values("p50_tbt_ms"), 50) or "",
                "request_p95_tbt_p50_ms": percentile(values("p95_tbt_ms"), 50) or "",
                "request_p99_tbt_p50_ms": percentile(values("p99_tbt_ms"), 50) or "",
                "completion_p50_ms": percentile(values("completion_latency_ms"), 50)
                or "",
                "completion_p90_ms": percentile(values("completion_latency_ms"), 90)
                or "",
                "completion_p99_ms": percentile(values("completion_latency_ms"), 99)
                or "",
                "ttft_slo_attainment": (
                    statistics.fmean(ttft_slo_values) if ttft_slo_values else ""
                ),
                "tbt_token_slo_attainment": (
                    statistics.fmean(token_slo_values) if token_slo_values else ""
                ),
            }
        )
    return summaries


def write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


async def replay(args: argparse.Namespace) -> int:
    import aiohttp

    request_timeout = request_timeout_total_seconds(args.request_timeout_sec)
    print(
        "[REPLAY] "
        f"request_timeout_sec={args.request_timeout_sec:g} "
        f"processing_timeout={'disabled' if request_timeout is None else 'enabled'} "
        "http_connect_timeout_sec=10",
        flush=True,
    )
    requests = load_trace(args.workload_jsonl)
    observations = build_observations(
        requests, arrival_time_scale=args.arrival_time_scale
    )
    probe_control = None
    probe_report = None
    if getattr(args, "tbt_probe_control", None):
        from tools.tbt_probe_dispatch import load_control
        if (args.termination_mode != "time" or args.time_limit_mode != "cancel"
                or not 0 < args.time_limit_sec < math.inf):
            raise ValueError("TBT probe control requires finite time/cancel mode")
        probe_control = load_control(args.tbt_probe_control, observations, args.workload_jsonl)
    if args.termination_mode == "requests" and args.request_limit > len(requests):
        raise ValueError(
            f"request-limit {args.request_limit} exceeds trace rows {len(requests)}"
        )
    policy = (
        load_slo_policy(args.slo_policy_json)
        if args.slo_policy_json is not None
        else None
    )
    request_slos = {
        request.request_id: select_slo(
            policy,
            request.input_tokens,
            overflow_policy=args.slo_overflow_policy,
            default_ttft_slo_ms=args.default_ttft_slo_ms,
            default_tbt_slo_ms=args.default_tbt_slo_ms,
        )
        for request in requests
    }
    if args.dry_run:
        for position, observation in enumerate(observations):
            state = replay_selection_state(
                position=position,
                effective_offset_s=observation.effective_offset_s,
                termination_mode=args.termination_mode,
                request_limit=args.request_limit,
                time_limit_sec=args.time_limit_sec,
            )
            print(
                "[DRY-RUN] "
                f"event={observation.request.event_index} "
                f"dataset={observation.request.dataset} "
                f"base_offset={observation.request.base_scheduled_offset_s:.6f}s "
                f"effective_offset={observation.effective_offset_s:.6f}s "
                f"input={observation.request.input_tokens} "
                f"output={observation.request.output_tokens} state={state}"
            )
        return 0

    timeout = aiohttp.ClientTimeout(
        total=request_timeout,
        connect=10.0,
        sock_read=None,
    )
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        model = args.model or await get_model_id(session, args.base_url)
        if args.warmup_count > 0:
            warmup = min(requests, key=lambda request: request.input_tokens)
            for warmup_index in range(args.warmup_count):
                warmup_request = LoongServeRequest(
                    **{
                        **warmup.__dict__,
                        "request_id": (f"{args.case_id}-warmup-{warmup_index + 1}"),
                        "output_tokens": args.warmup_output_tokens,
                    }
                )
                warmup_observation = RequestObservation(
                    request=warmup_request, effective_offset_s=0.0
                )
                await execute_request(
                    warmup_observation,
                    session=session,
                    base_url=args.base_url,
                    model=model,
                    slo=request_slos[warmup.request_id],
                    trace_start=time.monotonic(),
                    ignore_eos=True,
                )
                if warmup_observation.status != "success":
                    raise RuntimeError(
                        "warmup failed: " + warmup_observation.error_message
                    )
            if args.warmup_cooldown_sec > 0:
                await asyncio.sleep(args.warmup_cooldown_sec)

        trace_start = time.monotonic()
        deadline_ts = (
            trace_start + args.time_limit_sec
            if args.termination_mode == "time" and args.time_limit_sec > 0
            else None
        )
        tasks: list[asyncio.Task[RequestObservation]] = []
        if probe_control is not None:
            from tools.tbt_probe_dispatch import run_probe

            async def probe_execute(observation):
                return await execute_request(
                    observation, session=session, base_url=args.base_url, model=model,
                    slo=request_slos[observation.request.request_id], trace_start=trace_start,
                    ignore_eos=bool(args.ignore_eos))

            probe_report = await run_probe(observations, probe_control, execute=probe_execute,
                                           trace_start=trace_start, deadline=deadline_ts)
        # The historical open-loop dispatcher remains unchanged when no
        # explicit calibration control is supplied.
        for position, observation in enumerate(observations):
            if probe_control is not None:
                break
            observation.scheduled_ts = trace_start + observation.effective_offset_s
            if args.termination_mode == "requests" and position >= args.request_limit:
                observation.status = "not_submitted_request_limit"
                observation.error_message = (
                    "trace position is outside the configured request limit"
                )
                continue
            if deadline_ts is not None and observation.scheduled_ts >= deadline_ts:
                observation.status = "not_submitted_time_limit"
                observation.error_message = (
                    "scheduled arrival is outside the production time limit"
                )
                continue
            remaining = observation.scheduled_ts - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(remaining)
            if deadline_ts is not None and time.monotonic() >= deadline_ts:
                observation.status = "not_submitted_time_limit"
                observation.error_message = (
                    "dispatcher reached the production time limit"
                )
                continue
            tasks.append(
                asyncio.create_task(
                    execute_request(
                        observation,
                        session=session,
                        base_url=args.base_url,
                        model=model,
                        slo=request_slos[observation.request.request_id],
                        trace_start=trace_start,
                        ignore_eos=bool(args.ignore_eos),
                    )
                )
            )

        if probe_control is None and deadline_ts is not None and args.time_limit_mode == "cancel":
            remaining = deadline_ts - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(remaining)
            for task in tasks:
                if not task.done():
                    task.cancel()
        if tasks:
            await asyncio.gather(*tasks)
        observation_end = time.monotonic()

    raw_rows = [
        observation_row(
            observation,
            request_slos[observation.request.request_id],
        )
        for observation in observations
    ]
    token_rows = sorted(
        (token for observation in observations for token in observation.token_rows),
        key=lambda row: float(row["arrival_monotonic_ts"]),
    )
    summary_rows = summarize(raw_rows, token_rows)
    plan_rows = [{key: row.get(key, "") for key in PLAN_FIELDS} for row in raw_rows]
    write_csv(args.output_dir / "submission_plan.csv", PLAN_FIELDS, plan_rows)
    write_csv(args.output_dir / "request_results.csv", RAW_FIELDS, raw_rows)
    write_csv(
        args.output_dir / "client_token_timeline.csv",
        TOKEN_FIELDS,
        token_rows,
    )
    write_csv(
        args.output_dir / "workload_summary.csv",
        SUMMARY_FIELDS,
        summary_rows,
    )
    all_summary = next(row for row in summary_rows if row["dataset"] == "all")
    metadata = {
        "schema_version": 2,
        "created_at_utc": utc_now(),
        "case_id": args.case_id,
        "model": model,
        "workload": "LoongServe Mixed",
        "workload_jsonl": str(args.workload_jsonl),
        "workload_sha256": file_sha256(args.workload_jsonl),
        "submission_mode": "open_loop_trace_offsets",
        "arrival_time_scale": args.arrival_time_scale,
        "arrival_scale_semantics": (
            "effective_offset = base_offset * scale; <1 faster, >1 slower"
        ),
        "termination_mode": args.termination_mode,
        "request_limit": args.request_limit,
        "effective_request_limit": (
            args.request_limit if args.termination_mode == "requests" else 0
        ),
        "request_limit_contract": {
            "selection": "first N JSONL rows in immutable file order",
            "arrival_timing": "preserve each selected row's scaled trace offset",
            "completion": "drain all selected requests",
            "excluded_status": "not_submitted_request_limit",
        },
        "time_limit_sec": args.time_limit_sec,
        "time_limit_mode": args.time_limit_mode,
        "time_limit_origin": "first production trace clock after warmup",
        "time_limit_status_contract": {
            "drain": "stop new submissions; wait for admitted requests",
            "cancel": "stop new submissions and cancel admitted HTTP streams",
            "censored_requests_excluded_from_latency_and_SLO_aggregates": True,
        },
        "trace_start_monotonic_ts": trace_start,
        "observation_end_monotonic_ts": observation_end,
        "observation_duration_s": observation_end - trace_start,
        "request_timeout_sec": args.request_timeout_sec,
        "request_timeout_enabled": request_timeout is not None,
        "http_connect_timeout_sec": 10.0,
        "tbt_measurement": "vllm choice.token_ids",
        "completion_response_contract": {
            "return_token_ids": True,
            "return_prompt_token_ids": False,
            "effect": "omit prompt IDs only; preserve generated token IDs",
        },
        "slo_policy_json": (
            "" if args.slo_policy_json is None else str(args.slo_policy_json)
        ),
        "summary": all_summary,
    }
    if probe_report is not None:
        metadata["submission_mode"] = "controlled_event_gated_probe"
        metadata["tbt_probe_control_sha256"] = file_sha256(args.tbt_probe_control)
        metadata["tbt_probe_health"] = probe_report
        (args.output_dir / "tbt_probe_health.json").write_text(
            json.dumps(probe_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("[TBT-PROBE] " + json.dumps(probe_report, ensure_ascii=False), flush=True)
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(all_summary, ensure_ascii=False), flush=True)
    return 1 if int(all_summary["failure_count"]) > 0 else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--workload-jsonl", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arrival-time-scale", type=float, default=1.0)
    parser.add_argument(
        "--termination-mode", choices=("time", "requests"), default="time"
    )
    parser.add_argument("--request-limit", type=int, default=0)
    parser.add_argument("--time-limit-sec", type=float, default=0.0)
    parser.add_argument(
        "--time-limit-mode", choices=("drain", "cancel"), default="drain"
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=14400.0,
        help=(
            "HTTP request processing timeout in seconds (default: 14400); "
            "0 disables it. Connection and experiment deadlines remain separate."
        ),
    )
    parser.add_argument("--ignore-eos", type=int, choices=(0, 1), default=1)
    parser.add_argument("--warmup-count", type=int, default=0)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--warmup-cooldown-sec", type=float, default=3.0)
    parser.add_argument("--slo-policy-json", type=Path)
    parser.add_argument(
        "--slo-overflow-policy",
        choices=("error", "last_bucket"),
        default="error",
    )
    parser.add_argument("--default-ttft-slo-ms", type=float, default=0.0)
    parser.add_argument("--default-tbt-slo-ms", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tbt-probe-control", type=Path,
                        default=os.environ.get("LOONGSERVE_TBT_PROBE_CONTROL") or None,
                        help="Opt-in calibration-only event gates; not real-workload replay")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty output directory {args.output_dir}"
        )
    if args.time_limit_sec < 0:
        raise ValueError("time-limit-sec must be non-negative")
    if args.request_limit < 0:
        raise ValueError("request-limit must be non-negative")
    if args.termination_mode == "requests":
        if args.request_limit <= 0:
            raise ValueError("request-limit must be positive in requests mode")
        if args.time_limit_sec != 0:
            raise ValueError("time-limit-sec must be zero in requests mode")
        if args.time_limit_mode != "drain":
            raise ValueError("time-limit-mode must be drain in requests mode")
    elif args.request_limit != 0:
        raise ValueError("request-limit must be zero in time mode")
    request_timeout_total_seconds(args.request_timeout_sec)
    if args.warmup_count < 0 or args.warmup_output_tokens <= 0:
        raise ValueError("invalid warmup configuration")
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    return asyncio.run(replay(args))


if __name__ == "__main__":
    raise SystemExit(main())
