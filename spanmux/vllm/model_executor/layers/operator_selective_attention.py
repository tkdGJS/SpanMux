# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prototype operator-selective prefill/decode attention.

This module is intentionally env-gated and narrow. It keeps the normal vLLM
linear path fused, but lets Llama-like models split the attention operator into
decode and prefill phases for a reordered mixed batch:

    [decode requests..., prefill/extend requests...]

The implementation is a correctness-oriented prototype. By default it invokes
the existing attention layer twice with sliced attention metadata and then
concatenates the outputs back into the original token order. Optional
GreenContext stream overlap and MuxWise-style layer-phase baselines are
default-off diagnostics for memory-contention experiments.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Callable, TypeVar

import torch

from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.green_context import (
    green_context_pd_enabled,
    green_context_state,
    run_decode_prefill_on_green_streams,
)
from vllm.model_executor.layers.operator_memory_shaping import (
    filter_opmux_pd_execution,
)

logger = init_logger(__name__)

_ENABLE_ENV = "VLLM_OPERATOR_SELECTIVE_PD"
_STRICT_ENV = "VLLM_OPERATOR_SELECTIVE_PD_STRICT"
_DECODE_THRESHOLD_ENV = "VLLM_OPERATOR_SELECTIVE_PD_DECODE_THRESHOLD"
_TIMING_ENV = "VLLM_OPERATOR_SELECTIVE_PD_TIMING"
_OVERLAP_ENV = "VLLM_OPERATOR_SELECTIVE_PD_GREEN_OVERLAP"
_MUXWISE_PHASE_ENV = "VLLM_MUXWISE_PHASE_BASELINE"
_MUXWISE_PHASE_GREEN_ENV = "VLLM_MUXWISE_PHASE_BASELINE_GREEN_CONTEXT"
_PD_DECOUPLED_NO_MIX_ENV = "VLLM_PD_DECOUPLED_NO_MIX"

T = TypeVar("T")

try:
    _torch_compile_disable = torch.compiler.disable
except AttributeError:
    def _torch_compile_disable(fn):
        return fn


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def operator_selective_pd_enabled() -> bool:
    # A no-mix scheduler emits a phase-pure batch by construction. Installing
    # the per-layer mixed-batch selector in that mode only adds phase-plan
    # inspection overhead and, before the CPU-mirror fast path below, a
    # GPU-to-CPU synchronization on every layer.
    return _env_enabled(_ENABLE_ENV) and not _env_enabled(
        _PD_DECOUPLED_NO_MIX_ENV
    )


def muxwise_phase_baseline_enabled() -> bool:
    return os.environ.get(_MUXWISE_PHASE_ENV, "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def muxwise_phase_green_context_enabled() -> bool:
    return os.environ.get(_MUXWISE_PHASE_GREEN_ENV, "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _strict_mode_enabled() -> bool:
    return os.environ.get(_STRICT_ENV, "0").lower() in {"1", "true", "yes", "on"}


def operator_selective_pd_timing_enabled() -> bool:
    return (
        os.environ.get(_TIMING_ENV, "0").lower() in {"1", "true", "yes", "on"}
        or os.environ.get("VLLM_ITERATION_GPU_TIMING", "0") == "1"
        or os.environ.get("VLLM_LAYER_KERNEL_TIMING", "0") == "1"
    )


def operator_selective_pd_overlap_enabled() -> bool:
    return os.environ.get(_OVERLAP_ENV, "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _decode_threshold() -> int:
    value = os.environ.get(_DECODE_THRESHOLD_ENV, "1")
    try:
        return max(1, int(value))
    except ValueError:
        return 1


@dataclass
class PDPhasePlan:
    layer_name: str
    first_prefill_request: int
    num_requests: int
    num_actual_tokens: int
    num_decode_tokens: int
    num_prefill_tokens: int
    decode_metadata: Any
    prefill_metadata: Any


_OP_PD_METRICS: ContextVar[dict[str, Any] | None] = ContextVar(
    "vllm_operator_selective_pd_metrics",
    default=None,
)
_OP_PD_PHASE_PLAN_CACHE: ContextVar[dict[int, PDPhasePlan | None] | None] = (
    ContextVar(
        "vllm_operator_selective_pd_phase_plan_cache",
        default=None,
    )
)


def _zero_metric_record() -> dict[str, Any]:
    mode = "default"
    if muxwise_phase_baseline_enabled():
        mode = "muxwise_phase"
    elif operator_selective_pd_enabled():
        mode = "operator_selective_attention"
    record: dict[str, Any] = {
        "op_pd_enabled": float(
            operator_selective_pd_enabled() or muxwise_phase_baseline_enabled()
        ),
        "op_pd_attention_enabled": float(operator_selective_pd_enabled()),
        "op_pd_phase_baseline_enabled": float(muxwise_phase_baseline_enabled()),
        "op_pd_mode": mode,
        "op_pd_green_enabled": float(green_context_pd_enabled()),
        "op_pd_overlap_enabled": float(operator_selective_pd_overlap_enabled()),
        "op_pd_timing_enabled": float(operator_selective_pd_timing_enabled()),
        "op_pd_fallback_count": 0.0,
        "op_pd_fallback_reason": "",
        "op_pd_layers": 0.0,
        "op_pd_attention_layers": 0.0,
        "op_pd_phase_layers": 0.0,
        "op_pd_num_decode_tokens": 0.0,
        "op_pd_num_prefill_tokens": 0.0,
        "op_pd_decode_token_work": 0.0,
        "op_pd_prefill_token_work": 0.0,
        "op_pd_decode_attention_ms": 0.0,
        "op_pd_prefill_attention_ms": 0.0,
        "op_pd_attention_union_ms": 0.0,
        "op_pd_attention_overlap_ms": 0.0,
        "op_pd_attention_join_wait_ms": 0.0,
        "op_pd_decode_phase_ms": 0.0,
        "op_pd_prefill_phase_ms": 0.0,
        "op_pd_phase_union_ms": 0.0,
        "op_pd_phase_overlap_ms": 0.0,
        "op_pd_phase_join_wait_ms": 0.0,
        "op_pd_memory_contention_observable": 0.0,
        "op_pd_plan_build_count": 0.0,
        "op_pd_plan_cache_hit_count": 0.0,
        "op_pd_plan_fast_reject_count": 0.0,
        "op_pd_plan_cpu_mirror_count": 0.0,
        "op_pd_plan_device_to_host_fallback_count": 0.0,
        "op_pd_no_mix_bypass_count": float(
            _env_enabled(_ENABLE_ENV)
            and _env_enabled(_PD_DECOUPLED_NO_MIX_ENV)
        ),
    }
    record.update(green_context_state())
    return record


def reset_operator_selective_pd_metrics() -> None:
    _OP_PD_METRICS.set(_zero_metric_record())
    _OP_PD_PHASE_PLAN_CACHE.set({})


def consume_operator_selective_pd_metrics() -> dict[str, Any]:
    metrics = _OP_PD_METRICS.get()
    if metrics is None:
        return _zero_metric_record()
    record = dict(metrics)
    record.update(green_context_state())
    _OP_PD_METRICS.set(None)
    _OP_PD_PHASE_PLAN_CACHE.set(None)
    return record


def _metrics() -> dict[str, Any]:
    metrics = _OP_PD_METRICS.get()
    if metrics is None:
        reset_operator_selective_pd_metrics()
        metrics = _OP_PD_METRICS.get()
    assert metrics is not None
    return metrics


def record_operator_selective_pd_fallback(reason: str) -> None:
    record = _metrics()
    record["op_pd_fallback_count"] = float(record["op_pd_fallback_count"]) + 1.0
    previous = str(record.get("op_pd_fallback_reason", ""))
    if not previous:
        record["op_pd_fallback_reason"] = reason
    elif reason not in previous.split(","):
        record["op_pd_fallback_reason"] = f"{previous},{reason}"


def _add_metric(name: str, value: float) -> None:
    record = _metrics()
    record[name] = float(record.get(name, 0.0)) + float(value)


def _max_metric(name: str, value: float) -> None:
    record = _metrics()
    record[name] = max(float(record.get(name, 0.0)), float(value))


def _event_timings_from_ms(
    *,
    decode_start_ms: float,
    decode_end_ms: float,
    prefill_start_ms: float,
    prefill_end_ms: float,
) -> dict[str, float | str]:
    overlap_ms = max(
        0.0,
        min(decode_end_ms, prefill_end_ms) - max(decode_start_ms, prefill_start_ms),
    )
    return {
        "fallback_reason": "",
        "decode_ms": max(0.0, decode_end_ms - decode_start_ms),
        "prefill_ms": max(0.0, prefill_end_ms - prefill_start_ms),
        "decode_start_ms": decode_start_ms,
        "decode_end_ms": decode_end_ms,
        "prefill_start_ms": prefill_start_ms,
        "prefill_end_ms": prefill_end_ms,
        "union_ms": (
            max(decode_end_ms, prefill_end_ms)
            - min(decode_start_ms, prefill_start_ms)
        ),
        "overlap_ms": overlap_ms,
        "join_wait_ms": max(0.0, prefill_end_ms - decode_end_ms),
    }


def _run_direct_serial_pair(
    decode_fn: Callable[[], T],
    prefill_fn: Callable[[], T],
    *,
    collect_timing: bool,
) -> tuple[T, T, dict[str, float | str]]:
    if not collect_timing or not torch.cuda.is_available():
        return decode_fn(), prefill_fn(), {"fallback_reason": ""}

    current = torch.cuda.current_stream()
    base = torch.cuda.Event(enable_timing=True)
    decode_start = torch.cuda.Event(enable_timing=True)
    decode_end = torch.cuda.Event(enable_timing=True)
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    base.record(current)
    decode_start.record(current)
    decode_result = decode_fn()
    decode_end.record(current)
    decode_end.synchronize()
    prefill_start.record(current)
    prefill_result = prefill_fn()
    prefill_end.record(current)
    prefill_end.synchronize()
    timings = _event_timings_from_ms(
        decode_start_ms=base.elapsed_time(decode_start),
        decode_end_ms=base.elapsed_time(decode_end),
        prefill_start_ms=base.elapsed_time(prefill_start),
        prefill_end_ms=base.elapsed_time(prefill_end),
    )
    return decode_result, prefill_result, timings


def record_pd_pair_metrics(
    kind: str,
    plan: PDPhasePlan,
    timings: dict[str, float | str],
) -> None:
    _add_metric("op_pd_layers", 1.0)
    _add_metric(f"op_pd_{kind}_layers", 1.0)
    _max_metric("op_pd_num_decode_tokens", float(plan.num_decode_tokens))
    _max_metric("op_pd_num_prefill_tokens", float(plan.num_prefill_tokens))
    _add_metric("op_pd_decode_token_work", float(plan.num_decode_tokens))
    _add_metric("op_pd_prefill_token_work", float(plan.num_prefill_tokens))

    fallback_reason = str(timings.get("fallback_reason", ""))
    if fallback_reason:
        record_operator_selective_pd_fallback(fallback_reason)

    decode_ms = float(timings.get("decode_ms", 0.0))
    prefill_ms = float(timings.get("prefill_ms", 0.0))
    union_ms = float(timings.get("union_ms", 0.0))
    overlap_ms = float(timings.get("overlap_ms", 0.0))
    join_wait_ms = float(timings.get("join_wait_ms", 0.0))
    if kind == "attention":
        _add_metric("op_pd_decode_attention_ms", decode_ms)
        _add_metric("op_pd_prefill_attention_ms", prefill_ms)
        _add_metric("op_pd_attention_union_ms", union_ms)
        _add_metric("op_pd_attention_overlap_ms", overlap_ms)
        _add_metric("op_pd_attention_join_wait_ms", join_wait_ms)
    elif kind == "phase":
        _add_metric("op_pd_decode_phase_ms", decode_ms)
        _add_metric("op_pd_prefill_phase_ms", prefill_ms)
        _add_metric("op_pd_phase_union_ms", union_ms)
        _add_metric("op_pd_phase_overlap_ms", overlap_ms)
        _add_metric("op_pd_phase_join_wait_ms", join_wait_ms)

    if overlap_ms > 0.0 and green_context_pd_enabled():
        _metrics()["op_pd_memory_contention_observable"] = 1.0


def run_pd_pair(
    decode_fn: Callable[[], T],
    prefill_fn: Callable[[], T],
    *,
    kind: str,
    plan: PDPhasePlan,
    use_green_context: bool,
    overlap: bool,
) -> tuple[T, T]:
    collect_timing = operator_selective_pd_timing_enabled()
    use_green_context, overlap = filter_opmux_pd_execution(
        kind=kind,
        plan=plan,
        use_green_context=use_green_context,
        overlap=overlap,
    )
    if use_green_context and green_context_pd_enabled():
        result = run_decode_prefill_on_green_streams(
            decode_fn,
            prefill_fn,
            overlap=overlap,
            collect_timing=collect_timing,
        )
        decode_result = result.decode_result
        prefill_result = result.prefill_result
        timings = result.timings
    else:
        if overlap:
            record_operator_selective_pd_fallback("overlap_requires_green_context")
        decode_result, prefill_result, timings = _run_direct_serial_pair(
            decode_fn,
            prefill_fn,
            collect_timing=collect_timing,
        )
    record_pd_pair_metrics(kind, plan, timings)
    return decode_result, prefill_result


def _metadata_for_layer(context: ForwardContext, layer_name: str) -> Any | None:
    attn_metadata = context.attn_metadata
    if isinstance(attn_metadata, list):
        # DBO/microbatch metadata is a list of per-ubatch dictionaries. This
        # prototype keeps that path on the upstream attention implementation.
        return None
    if isinstance(attn_metadata, dict):
        return attn_metadata.get(layer_name)
    return attn_metadata


def _context_with_layer_metadata(
    context: ForwardContext, layer_name: str, phase_metadata: Any
) -> ForwardContext:
    attn_metadata = context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = dict(attn_metadata)
        attn_metadata[layer_name] = phase_metadata
    else:
        attn_metadata = phase_metadata
    return replace(context, attn_metadata=attn_metadata)


def _field_names(instance: Any) -> set[str]:
    if not is_dataclass(instance):
        return set()
    return {field.name for field in fields(instance)}


def _make_phase_metadata(
    attn_metadata: Any,
    *,
    request_start: int,
    request_stop: int,
    token_start: int,
    token_stop: int,
    query_lens_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
) -> Any | None:
    names = _field_names(attn_metadata)
    if not names:
        return None

    query_start_loc = attn_metadata.query_start_loc[request_start : request_stop + 1]
    query_start_loc = query_start_loc - query_start_loc[0]
    phase_seq_lens = attn_metadata.seq_lens[request_start:request_stop]
    phase_query_lens_cpu = query_lens_cpu[request_start:request_stop]
    phase_seq_lens_cpu = seq_lens_cpu[request_start:request_stop]

    if phase_query_lens_cpu.numel() == 0:
        return None

    block_table_field = (
        "block_table" if "block_table" in names else "block_table_tensor"
    )
    if not hasattr(attn_metadata, block_table_field):
        return None

    updates: dict[str, Any] = {
        "num_actual_tokens": int(token_stop - token_start),
        "max_query_len": int(phase_query_lens_cpu.max().item()),
        "query_start_loc": query_start_loc,
        "max_seq_len": int(phase_seq_lens_cpu.max().item()),
        "seq_lens": phase_seq_lens,
        "slot_mapping": attn_metadata.slot_mapping[token_start:token_stop],
        block_table_field: getattr(attn_metadata, block_table_field)[
            request_start:request_stop
        ],
    }

    # Optional CPU mirrors exist on CommonAttentionMetadata and some helper
    # metadata classes. Keep them coherent if the current backend exposes them.
    if "query_start_loc_cpu" in names:
        phase_query_start_loc_cpu = torch.zeros(
            phase_query_lens_cpu.numel() + 1, dtype=phase_query_lens_cpu.dtype
        )
        phase_query_start_loc_cpu[1:] = torch.cumsum(phase_query_lens_cpu, dim=0)
        updates["query_start_loc_cpu"] = phase_query_start_loc_cpu
    if "_seq_lens_cpu" in names:
        updates["_seq_lens_cpu"] = phase_seq_lens_cpu
    if "_num_computed_tokens_cpu" in names and hasattr(
        attn_metadata, "num_computed_tokens_cpu"
    ):
        updates["_num_computed_tokens_cpu"] = attn_metadata.num_computed_tokens_cpu[
            request_start:request_stop
        ]

    # Disable whole-batch attention variants for the sliced phase metadata.
    # Cascade/DCP/AOT scheduler metadata can be reintroduced later once the
    # prototype has correctness and performance baselines.
    optional_resets: dict[str, Any] = {
        "use_cascade": False,
        "common_prefix_len": 0,
        "cu_prefix_query_lens": None,
        "prefix_kv_lens": None,
        "suffix_kv_lens": None,
        "scheduler_metadata": None,
        "prefix_scheduler_metadata": None,
        "mm_prefix_range": None,
        "dcp_context_kv_lens": None,
        "max_dcp_context_kv_len": 0,
    }
    for name, value in optional_resets.items():
        if name in names:
            updates[name] = value

    filtered_updates = {k: v for k, v in updates.items() if k in names}
    return replace(attn_metadata, **filtered_updates)


def get_pd_phase_plan(layer_name: str) -> PDPhasePlan | None:
    context = get_forward_context()
    attn_metadata = _metadata_for_layer(context, layer_name)
    if attn_metadata is None:
        return None

    cache = _OP_PD_PHASE_PLAN_CACHE.get()
    if cache is None:
        cache = {}
        _OP_PD_PHASE_PLAN_CACHE.set(cache)
    cache_key = id(attn_metadata)
    if cache_key in cache:
        _add_metric("op_pd_plan_cache_hit_count", 1.0)
        cached_plan = cache[cache_key]
        if cached_plan is None:
            return None
        return replace(cached_plan, layer_name=layer_name)

    _add_metric("op_pd_plan_build_count", 1.0)

    def reject(*, fast: bool = False) -> None:
        cache[cache_key] = None
        if fast:
            _add_metric("op_pd_plan_fast_reject_count", 1.0)

    # A no-mix phase never contains both Decode and Prefill requests. Keep a
    # runtime guard as well as the model-construction guard so a process whose
    # env is changed after construction still cannot take the selector path.
    if _env_enabled(_PD_DECOUPLED_NO_MIX_ENV):
        _add_metric("op_pd_no_mix_bypass_count", 1.0)
        reject(fast=True)
        return None

    required_attrs = (
        "num_actual_tokens",
        "query_start_loc",
        "seq_lens",
        "slot_mapping",
    )
    if any(not hasattr(attn_metadata, attr) for attr in required_attrs):
        reject(fast=True)
        return None
    if getattr(attn_metadata, "use_cascade", False):
        reject(fast=True)
        return None

    num_actual_tokens = int(attn_metadata.num_actual_tokens)
    if num_actual_tokens <= 0:
        reject(fast=True)
        return None

    query_start_loc_cpu = getattr(attn_metadata, "query_start_loc_cpu", None)
    if query_start_loc_cpu is None:
        # Tensor shape is host metadata and does not synchronize CUDA. This
        # rejects the dominant batch-size-one Decode/Prefill case before any
        # value-dependent device read.
        num_requests = int(attn_metadata.query_start_loc.shape[0]) - 1
        if num_requests < 2:
            reject(fast=True)
            return None
        query_start_loc_cpu = attn_metadata.query_start_loc.detach().to("cpu")
        _add_metric("op_pd_plan_device_to_host_fallback_count", 1.0)
    else:
        _add_metric("op_pd_plan_cpu_mirror_count", 1.0)

    if query_start_loc_cpu.numel() < 3:
        reject(fast=True)
        return None

    seq_lens_cpu = getattr(attn_metadata, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = getattr(attn_metadata, "_seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = attn_metadata.seq_lens.detach().to("cpu")
        _add_metric("op_pd_plan_device_to_host_fallback_count", 1.0)
    else:
        _add_metric("op_pd_plan_cpu_mirror_count", 1.0)

    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    if query_lens_cpu.numel() < 2:
        reject(fast=True)
        return None

    decode_threshold = _decode_threshold()
    is_decode = query_lens_cpu <= decode_threshold
    if bool(torch.all(is_decode)) or not bool(torch.any(is_decode)):
        reject(fast=True)
        return None

    first_prefill = int((~is_decode).nonzero(as_tuple=False)[0].item())
    if first_prefill <= 0:
        reject()
        return None
    if bool(torch.any(is_decode[first_prefill:])):
        # This prototype expects the scheduler/model runner order used by vLLM
        # mixed batches: decodes first, then prefill/extend requests.
        reject()
        return None

    num_decode_tokens = int(query_start_loc_cpu[first_prefill].item())
    if num_decode_tokens <= 0 or num_decode_tokens >= num_actual_tokens:
        reject()
        return None

    decode_metadata = _make_phase_metadata(
        attn_metadata,
        request_start=0,
        request_stop=first_prefill,
        token_start=0,
        token_stop=num_decode_tokens,
        query_lens_cpu=query_lens_cpu,
        seq_lens_cpu=seq_lens_cpu,
    )
    prefill_metadata = _make_phase_metadata(
        attn_metadata,
        request_start=first_prefill,
        request_stop=query_lens_cpu.numel(),
        token_start=num_decode_tokens,
        token_stop=num_actual_tokens,
        query_lens_cpu=query_lens_cpu,
        seq_lens_cpu=seq_lens_cpu,
    )
    if decode_metadata is None or prefill_metadata is None:
        reject()
        return None

    plan = PDPhasePlan(
        layer_name=layer_name,
        first_prefill_request=first_prefill,
        num_requests=int(query_lens_cpu.numel()),
        num_actual_tokens=num_actual_tokens,
        num_decode_tokens=num_decode_tokens,
        num_prefill_tokens=num_actual_tokens - num_decode_tokens,
        decode_metadata=decode_metadata,
        prefill_metadata=prefill_metadata,
    )
    cache[cache_key] = plan
    return plan


def run_with_pd_phase_context(
    layer_name: str,
    phase_metadata: Any,
    fn: Callable[[], T],
) -> T:
    context = get_forward_context()
    with override_forward_context(
        _context_with_layer_metadata(context, layer_name, phase_metadata)
    ):
        return fn()


def maybe_operator_selective_pd_attention(
    attn_layer: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor | None:
    """Run decode attention before prefill attention for reordered mixed P/D.

    Returns None when the current batch/backend is outside the prototype scope,
    allowing the caller to fall back to the normal attention path.
    """
    try:
        return _maybe_operator_selective_pd_attention_impl(
            attn_layer, query, key, value
        )
    except Exception as exc:
        if _strict_mode_enabled():
            raise
        logger.warning_once(
            "Operator-selective P/D attention fell back to the upstream path: %s",
            str(exc),
            scope="local",
        )
        return None


@_torch_compile_disable
def _maybe_operator_selective_pd_attention_impl(
    attn_layer: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor | None:
    layer_name = getattr(attn_layer, "layer_name", "")
    plan = get_pd_phase_plan(layer_name)
    if plan is None:
        return None
    if plan.num_actual_tokens > query.shape[0]:
        return None

    decode_fn = lambda: run_with_pd_phase_context(
        layer_name,
        plan.decode_metadata,
        lambda: attn_layer(
            query[: plan.num_decode_tokens],
            key[: plan.num_decode_tokens],
            value[: plan.num_decode_tokens],
        ),
    )
    prefill_fn = lambda: run_with_pd_phase_context(
        layer_name,
        plan.prefill_metadata,
        lambda: attn_layer(
            query[plan.num_decode_tokens : plan.num_actual_tokens],
            key[plan.num_decode_tokens : plan.num_actual_tokens],
            value[plan.num_decode_tokens : plan.num_actual_tokens],
        ),
    )
    decode_output, prefill_output = run_pd_pair(
        decode_fn,
        prefill_fn,
        kind="attention",
        plan=plan,
        use_green_context=green_context_pd_enabled(),
        overlap=operator_selective_pd_overlap_enabled(),
    )

    output = query.new_empty((query.shape[0], decode_output.shape[-1]))
    output[: plan.num_decode_tokens] = decode_output
    output[plan.num_decode_tokens : plan.num_actual_tokens] = prefill_output
    return output
