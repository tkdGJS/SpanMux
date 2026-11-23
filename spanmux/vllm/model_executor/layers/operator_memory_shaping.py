# SPDX-License-Identifier: Apache-2.0
"""OpMux 0.4.x operator memory-shaping prototype.

This module is intentionally conservative. It does not replace vLLM's
attention kernels or KV cache layout. Instead it exposes an experimental
operator-memory contract that can be logged per iteration and, when explicitly
requested, can suppress GreenContext P/D overlap that would exceed a configured
memory-residency budget.

The prototype is a substrate for the next kernel/operator step:

* QKV materialization and K/V commit byte accounting.
* Attention and MLP temporary-memory proxies.
* P/D phase compatibility checks before overlapping decode and prefill.
* Default-off behavior, with profiling-only mode separated from policy mode.
"""

from __future__ import annotations

import os
from typing import Any

_ENABLE_ENV = "VLLM_OPMUX_ENABLED"
_ENABLE_ALIAS_ENV = "VLLM_OPERATOR_MEMORY_SHAPING"
_TIMING_ENV = "VLLM_OPMUX_TIMING"
_STRICT_ENV = "VLLM_OPMUX_STRICT"
_POLICY_ENV = "VLLM_OPMUX_POLICY"
_PRESSURE_ENV = "VLLM_OPMUX_MEMORY_PRESSURE_MODE"
_SEGMENT_TOKENS_ENV = "VLLM_OPMUX_SEGMENTED_KV_COMMIT_TOKENS"
_PREFILL_SPLIT_LIMIT_ENV = "VLLM_OPMUX_PREFILL_SPLIT_LIMIT"
_TEMP_BUDGET_BYTES_ENV = "VLLM_OPMUX_TEMP_BUDGET_BYTES"
_KV_COMMIT_BUDGET_TOKENS_ENV = "VLLM_OPMUX_KV_COMMIT_BUDGET_TOKENS"
_PREFILL_TEMP_BYTES_PER_TOKEN_ENV = "VLLM_OPMUX_PREFILL_TEMP_BYTES_PER_TOKEN"
_VERSION = "0.4.0"

_OPMUX_METRICS: dict[str, Any] | None = None


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return max(minimum, int(value))
    except ValueError:
        return default


def opmux_enabled() -> bool:
    return _env_enabled(_ENABLE_ENV) or _env_enabled(_ENABLE_ALIAS_ENV)


def opmux_timing_enabled() -> bool:
    return (
        opmux_enabled()
        and (
            _env_enabled(_TIMING_ENV)
            or os.environ.get("VLLM_ITERATION_GPU_TIMING", "0") == "1"
            or os.environ.get("VLLM_LAYER_KERNEL_TIMING", "0") == "1"
        )
    )


def opmux_strict_enabled() -> bool:
    return opmux_enabled() and _env_enabled(_STRICT_ENV)


def opmux_policy() -> str:
    return os.environ.get(_POLICY_ENV, "profile").strip().lower() or "profile"


def opmux_memory_pressure_mode() -> str:
    mode = os.environ.get(_PRESSURE_ENV, "low").strip().lower()
    if mode not in {"low", "medium", "high"}:
        return "low"
    return mode


def _segment_tokens() -> int:
    return _env_int(_SEGMENT_TOKENS_ENV, 0)


def _prefill_split_limit() -> int:
    return _env_int(_PREFILL_SPLIT_LIMIT_ENV, 0)


def _temp_budget_bytes() -> int:
    return _env_int(_TEMP_BUDGET_BYTES_ENV, 0)


def _kv_commit_budget_tokens() -> int:
    return _env_int(_KV_COMMIT_BUDGET_TOKENS_ENV, 0)


def _prefill_temp_bytes_per_token() -> int:
    # A conservative proxy for QKV + short-lived attention/MLP materialization.
    # Workloads can override this per model/GPU after profiling.
    return _env_int(_PREFILL_TEMP_BYTES_PER_TOKEN_ENV, 8192, minimum=1)


def _zero_metric_record() -> dict[str, Any]:
    return {
        "opmux_version": _VERSION,
        "opmux_enabled": float(opmux_enabled()),
        "opmux_timing_enabled": float(opmux_timing_enabled()),
        "opmux_policy": opmux_policy(),
        "opmux_memory_pressure_mode": opmux_memory_pressure_mode(),
        "opmux_segmented_kv_commit_tokens": float(_segment_tokens()),
        "opmux_prefill_split_limit": float(_prefill_split_limit()),
        "opmux_temp_budget_bytes": float(_temp_budget_bytes()),
        "opmux_kv_commit_budget_tokens": float(_kv_commit_budget_tokens()),
        "opmux_prefill_temp_bytes_per_token": float(
            _prefill_temp_bytes_per_token()
        ),
        "opmux_qkv_calls": 0.0,
        "opmux_attention_calls": 0.0,
        "opmux_o_proj_calls": 0.0,
        "opmux_mlp_calls": 0.0,
        "opmux_qkv_tokens": 0.0,
        "opmux_attention_tokens": 0.0,
        "opmux_mlp_tokens": 0.0,
        "opmux_qkv_materialized_bytes": 0.0,
        "opmux_q_tensor_bytes": 0.0,
        "opmux_k_tensor_bytes": 0.0,
        "opmux_v_tensor_bytes": 0.0,
        "opmux_kv_commit_bytes": 0.0,
        "opmux_attention_output_bytes": 0.0,
        "opmux_attention_workspace_proxy_bytes": 0.0,
        "opmux_attention_kv_read_proxy_bytes": 0.0,
        "opmux_o_proj_output_bytes": 0.0,
        "opmux_mlp_gate_up_bytes": 0.0,
        "opmux_mlp_activation_bytes": 0.0,
        "opmux_mlp_output_bytes": 0.0,
        "opmux_peak_temp_bytes": 0.0,
        "opmux_peak_resident_proxy_bytes": 0.0,
        "opmux_pd_pairs_seen": 0.0,
        "opmux_pd_attention_pairs_seen": 0.0,
        "opmux_pd_phase_pairs_seen": 0.0,
        "opmux_pd_decode_tokens": 0.0,
        "opmux_pd_prefill_tokens": 0.0,
        "opmux_pd_prefill_temp_proxy_bytes": 0.0,
        "opmux_memory_compatible": 1.0,
        "opmux_incompatible_pd_pairs": 0.0,
        "opmux_co_run_suppressed_count": 0.0,
        "opmux_green_context_suppressed_count": 0.0,
        "opmux_fallback_count": 0.0,
        "opmux_fallback_reason": "",
        "opmux_last_layer": "",
        "opmux_last_incompatible_reason": "",
        # Future kernel/operator features. These stay 0 in this prototype so
        # logs cannot be misread as a custom fused kernel implementation.
        "opmux_kv_first_commit_active": 0.0,
        "opmux_budgeted_attention_active": 0.0,
        "opmux_segmented_kv_commit_active": 0.0,
    }


def reset_opmux_metrics() -> None:
    global _OPMUX_METRICS
    if opmux_enabled():
        _OPMUX_METRICS = _zero_metric_record()
    else:
        _OPMUX_METRICS = None


def consume_opmux_metrics() -> dict[str, Any]:
    global _OPMUX_METRICS
    if not opmux_enabled():
        _OPMUX_METRICS = None
        return {}
    if _OPMUX_METRICS is None:
        return _zero_metric_record()
    record = dict(_OPMUX_METRICS)
    _OPMUX_METRICS = None
    return record


def _metrics() -> dict[str, Any] | None:
    global _OPMUX_METRICS
    if not opmux_enabled():
        return None
    if _OPMUX_METRICS is None:
        _OPMUX_METRICS = _zero_metric_record()
    return _OPMUX_METRICS


def _add_metric(name: str, value: float) -> None:
    record = _metrics()
    if record is None:
        return
    record[name] = float(record.get(name, 0.0)) + float(value)


def _max_metric(name: str, value: float) -> None:
    record = _metrics()
    if record is None:
        return
    record[name] = max(float(record.get(name, 0.0)), float(value))


def _min_metric(name: str, value: float) -> None:
    record = _metrics()
    if record is None:
        return
    record[name] = min(float(record.get(name, value)), float(value))


def _set_metric(name: str, value: Any) -> None:
    record = _metrics()
    if record is None:
        return
    record[name] = value


def _append_reason(reason: str) -> None:
    record = _metrics()
    if record is None or not reason:
        return
    previous = str(record.get("opmux_fallback_reason", ""))
    if not previous:
        record["opmux_fallback_reason"] = reason
    elif reason not in previous.split(","):
        record["opmux_fallback_reason"] = f"{previous},{reason}"


def _tensor_bytes(tensor: Any) -> int:
    if tensor is None:
        return 0
    numel = getattr(tensor, "numel", None)
    element_size = getattr(tensor, "element_size", None)
    if numel is None or element_size is None:
        return 0
    try:
        return int(numel() * element_size())
    except Exception:
        return 0


def _tensor_tokens(tensor: Any) -> int:
    shape = getattr(tensor, "shape", None)
    if not shape:
        return 0
    try:
        return int(shape[0])
    except Exception:
        return 0


def _record_temp_bytes(value: int) -> None:
    _max_metric("opmux_peak_temp_bytes", float(value))


def _record_resident_proxy_bytes(temp_bytes: int, kv_bytes: int = 0) -> None:
    _max_metric(
        "opmux_peak_resident_proxy_bytes",
        float(max(0, temp_bytes) + max(0, kv_bytes)),
    )


def record_opmux_qkv(
    layer_name: str,
    hidden_states: Any,
    qkv: Any,
    q: Any,
    k: Any,
    v: Any,
) -> None:
    if not opmux_enabled():
        return
    qkv_bytes = _tensor_bytes(qkv)
    q_bytes = _tensor_bytes(q)
    k_bytes = _tensor_bytes(k)
    v_bytes = _tensor_bytes(v)
    kv_commit_bytes = k_bytes + v_bytes
    _add_metric("opmux_qkv_calls", 1.0)
    _add_metric("opmux_qkv_tokens", float(_tensor_tokens(hidden_states)))
    _add_metric("opmux_qkv_materialized_bytes", float(qkv_bytes))
    _add_metric("opmux_q_tensor_bytes", float(q_bytes))
    _add_metric("opmux_k_tensor_bytes", float(k_bytes))
    _add_metric("opmux_v_tensor_bytes", float(v_bytes))
    _add_metric("opmux_kv_commit_bytes", float(kv_commit_bytes))
    _record_temp_bytes(qkv_bytes)
    _record_resident_proxy_bytes(qkv_bytes, kv_commit_bytes)
    _set_metric("opmux_last_layer", layer_name)


def record_opmux_attention(
    layer_name: str,
    q: Any,
    k: Any,
    v: Any,
    attn_output: Any,
) -> None:
    if not opmux_enabled():
        return
    q_bytes = _tensor_bytes(q)
    k_bytes = _tensor_bytes(k)
    v_bytes = _tensor_bytes(v)
    output_bytes = _tensor_bytes(attn_output)
    workspace_proxy_bytes = q_bytes + output_bytes
    kv_read_proxy_bytes = k_bytes + v_bytes
    _add_metric("opmux_attention_calls", 1.0)
    _add_metric("opmux_attention_tokens", float(_tensor_tokens(q)))
    _add_metric("opmux_attention_output_bytes", float(output_bytes))
    _add_metric(
        "opmux_attention_workspace_proxy_bytes",
        float(workspace_proxy_bytes),
    )
    _add_metric(
        "opmux_attention_kv_read_proxy_bytes",
        float(kv_read_proxy_bytes),
    )
    _record_temp_bytes(workspace_proxy_bytes)
    _record_resident_proxy_bytes(workspace_proxy_bytes, kv_read_proxy_bytes)
    _set_metric("opmux_last_layer", layer_name)


def record_opmux_o_proj(layer_name: str, attn_output: Any, output: Any) -> None:
    if not opmux_enabled():
        return
    output_bytes = _tensor_bytes(output)
    _add_metric("opmux_o_proj_calls", 1.0)
    _add_metric("opmux_o_proj_output_bytes", float(output_bytes))
    _record_temp_bytes(max(_tensor_bytes(attn_output), output_bytes))
    _set_metric("opmux_last_layer", layer_name)


def record_opmux_mlp(
    layer_name: str,
    input_tensor: Any,
    gate_up: Any,
    activation: Any,
    output: Any,
) -> None:
    if not opmux_enabled():
        return
    gate_up_bytes = _tensor_bytes(gate_up)
    activation_bytes = _tensor_bytes(activation)
    output_bytes = _tensor_bytes(output)
    temp_bytes = max(gate_up_bytes, activation_bytes) + output_bytes
    _add_metric("opmux_mlp_calls", 1.0)
    _add_metric("opmux_mlp_tokens", float(_tensor_tokens(input_tensor)))
    _add_metric("opmux_mlp_gate_up_bytes", float(gate_up_bytes))
    _add_metric("opmux_mlp_activation_bytes", float(activation_bytes))
    _add_metric("opmux_mlp_output_bytes", float(output_bytes))
    _record_temp_bytes(temp_bytes)
    _record_resident_proxy_bytes(temp_bytes)
    _set_metric("opmux_last_layer", layer_name)


def _plan_attr(plan: Any, name: str, default: int = 0) -> int:
    try:
        return int(getattr(plan, name))
    except Exception:
        return default


def _classify_memory_compatibility(plan: Any) -> tuple[bool, str]:
    prefill_tokens = _plan_attr(plan, "num_prefill_tokens")
    pressure_mode = opmux_memory_pressure_mode()
    policy = opmux_policy()
    if policy in {"profile", "off", "observe"}:
        return True, ""

    if pressure_mode == "high" and prefill_tokens > 0:
        return False, "high_pressure_decode_priority"

    kv_budget_tokens = _kv_commit_budget_tokens()
    if kv_budget_tokens > 0 and prefill_tokens > kv_budget_tokens:
        return False, "kv_commit_budget_tokens"

    segment_tokens = _segment_tokens()
    if pressure_mode in {"medium", "high"} and segment_tokens > 0:
        if prefill_tokens > segment_tokens:
            return False, "segmented_kv_commit_tokens"

    temp_budget_bytes = _temp_budget_bytes()
    prefill_temp_proxy_bytes = prefill_tokens * _prefill_temp_bytes_per_token()
    if temp_budget_bytes > 0 and prefill_temp_proxy_bytes > temp_budget_bytes:
        return False, "temp_budget_bytes"

    return True, ""


def filter_opmux_pd_execution(
    *,
    kind: str,
    plan: Any,
    use_green_context: bool,
    overlap: bool,
) -> tuple[bool, bool]:
    """Apply OpMux memory compatibility policy to a P/D pair.

    The profiling-only policy records the plan and leaves execution unchanged.
    `memory_guard`/`decode_priority` policies can suppress overlap for an
    incompatible pair. This is deliberately a coarse 0.4.x prototype until the
    operator path can actually split QKV/KV-commit/attention/MLP sub-phases.
    """
    if not opmux_enabled():
        return use_green_context, overlap

    decode_tokens = _plan_attr(plan, "num_decode_tokens")
    prefill_tokens = _plan_attr(plan, "num_prefill_tokens")
    prefill_temp_proxy_bytes = prefill_tokens * _prefill_temp_bytes_per_token()
    _add_metric("opmux_pd_pairs_seen", 1.0)
    _add_metric(f"opmux_pd_{kind}_pairs_seen", 1.0)
    _max_metric("opmux_pd_decode_tokens", float(decode_tokens))
    _max_metric("opmux_pd_prefill_tokens", float(prefill_tokens))
    _max_metric(
        "opmux_pd_prefill_temp_proxy_bytes",
        float(prefill_temp_proxy_bytes),
    )

    compatible, reason = _classify_memory_compatibility(plan)
    _min_metric("opmux_memory_compatible", 1.0 if compatible else 0.0)
    if compatible:
        return use_green_context, overlap

    _add_metric("opmux_incompatible_pd_pairs", 1.0)
    _set_metric("opmux_last_incompatible_reason", reason)

    if overlap:
        _add_metric("opmux_co_run_suppressed_count", 1.0)
        overlap = False
    if opmux_policy() in {"decode_only", "decode_priority_no_green"}:
        if use_green_context:
            _add_metric("opmux_green_context_suppressed_count", 1.0)
        use_green_context = False

    _add_metric("opmux_fallback_count", 1.0)
    _append_reason(reason)
    if opmux_strict_enabled() and opmux_policy() == "strict_memory_guard":
        raise RuntimeError(f"OpMux rejected P/D overlap: {reason}")
    return use_green_context, overlap
