# SPDX-License-Identifier: Apache-2.0
"""Default-off attention-output trace for context-ring correctness runs.

The trace is intentionally synchronous. Correctness runs must not publish a
manifest entry before the corresponding CUDA output has reached host memory.
Normal serving never enables this module and pays only one cached boolean
check at the attention return boundary.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_VALID_MODES = {"off", "record"}
_VALID_PATHS = {"native_hbm", "context_ring"}
_VALID_PHASES = {"prefill", "decode"}
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_STATE_LOCK = threading.Lock()
_MANIFEST_LOCK = threading.Lock()
_CAPTURE_INDEX = 0
_REQUEST_ORDINALS: dict[str, int] = {}
_SEMANTIC_OCCURRENCES: dict[tuple[Any, ...], int] = {}


@dataclass(frozen=True)
class AttentionCorrectnessTraceConfig:
    mode: str
    output_dir: Path | None
    execution_paths: tuple[str, ...]
    phases: tuple[str, ...]
    layer_patterns: tuple[str, ...]
    request_patterns: tuple[str, ...]
    file_chunk_tokens: int
    max_records: int
    strict: bool
    capture_qkv: bool
    rank: int

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


@dataclass(frozen=True)
class AttentionTraceRow:
    request_id: str
    request_ordinal: int
    request_row: int
    phase: str
    sequence_length: int
    logical_query_start: int
    query_length: int
    output_start: int
    output_end: int


def _csv_values(name: str, default: str) -> tuple[str, ...]:
    raw_value = os.environ.get(name, default)
    return tuple(value.strip() for value in raw_value.split(",") if value.strip())


def _positive_int(name: str, default: int, *, allow_zero: bool = False) -> int:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@lru_cache(maxsize=1)
def get_attention_correctness_trace_config() -> AttentionCorrectnessTraceConfig:
    mode = os.environ.get("VLLM_CONTEXT_RING_CORRECTNESS_MODE", "off").lower()
    if mode not in _VALID_MODES:
        raise ValueError(
            "VLLM_CONTEXT_RING_CORRECTNESS_MODE must be one of "
            f"{sorted(_VALID_MODES)}, got {mode!r}"
        )

    output_dir_value = os.environ.get(
        "VLLM_CONTEXT_RING_CORRECTNESS_DIR", ""
    ).strip()
    output_dir = Path(output_dir_value).expanduser() if output_dir_value else None
    if mode == "record" and output_dir is None:
        raise ValueError(
            "VLLM_CONTEXT_RING_CORRECTNESS_DIR is required in record mode"
        )

    execution_paths = _csv_values(
        "VLLM_CONTEXT_RING_CORRECTNESS_PATHS",
        "native_hbm,context_ring",
    )
    invalid_paths = set(execution_paths) - _VALID_PATHS
    if invalid_paths:
        raise ValueError(
            "invalid VLLM_CONTEXT_RING_CORRECTNESS_PATHS values: "
            f"{sorted(invalid_paths)}"
        )

    phases = tuple(
        value.lower()
        for value in _csv_values(
            "VLLM_CONTEXT_RING_CORRECTNESS_PHASES", "prefill,decode"
        )
    )
    invalid_phases = set(phases) - _VALID_PHASES
    if invalid_phases:
        raise ValueError(
            "invalid VLLM_CONTEXT_RING_CORRECTNESS_PHASES values: "
            f"{sorted(invalid_phases)}"
        )

    return AttentionCorrectnessTraceConfig(
        mode=mode,
        output_dir=output_dir,
        execution_paths=execution_paths,
        phases=phases,
        layer_patterns=_csv_values(
            "VLLM_CONTEXT_RING_CORRECTNESS_LAYERS", "*"
        ),
        request_patterns=_csv_values(
            "VLLM_CONTEXT_RING_CORRECTNESS_REQUESTS", "*"
        ),
        file_chunk_tokens=_positive_int(
            "VLLM_CONTEXT_RING_CORRECTNESS_FILE_CHUNK_TOKENS", 2048
        ),
        max_records=_positive_int(
            "VLLM_CONTEXT_RING_CORRECTNESS_MAX_RECORDS",
            0,
            allow_zero=True,
        ),
        strict=(
            os.environ.get("VLLM_CONTEXT_RING_CORRECTNESS_STRICT", "1").lower()
            in _TRUE_VALUES
        ),
        capture_qkv=(
            os.environ.get(
                "VLLM_CONTEXT_RING_CORRECTNESS_CAPTURE_QKV", "0"
            ).lower()
            in _TRUE_VALUES
        ),
        rank=int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))),
    )


def attention_correctness_trace_enabled() -> bool:
    return get_attention_correctness_trace_config().enabled


def reset_attention_correctness_trace_state_for_tests() -> None:
    global _CAPTURE_INDEX
    get_attention_correctness_trace_config.cache_clear()
    with _STATE_LOCK:
        _CAPTURE_INDEX = 0
        _REQUEST_ORDINALS.clear()
        _SEMANTIC_OCCURRENCES.clear()


def _matches(value: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def _request_ordinal(request_id: str) -> int:
    with _STATE_LOCK:
        ordinal = _REQUEST_ORDINALS.get(request_id)
        if ordinal is None:
            ordinal = len(_REQUEST_ORDINALS)
            _REQUEST_ORDINALS[request_id] = ordinal
        return ordinal


def build_attention_trace_rows(
    *,
    request_ids: Sequence[str],
    query_start_locs: Sequence[int],
    sequence_lengths: Sequence[int],
    phase: str,
) -> list[AttentionTraceRow]:
    """Build semantic token ranges without depending on scheduler iteration IDs."""

    request_count = min(
        len(request_ids),
        len(sequence_lengths),
        max(0, len(query_start_locs) - 1),
    )
    rows: list[AttentionTraceRow] = []
    normalized_phase = phase.lower() if phase else ""
    for request_row in range(request_count):
        output_start = int(query_start_locs[request_row])
        output_end = int(query_start_locs[request_row + 1])
        query_length = output_end - output_start
        if query_length <= 0:
            continue
        sequence_length = int(sequence_lengths[request_row])
        row_phase = normalized_phase
        if row_phase not in _VALID_PHASES:
            # Upstream/mixed baselines do not always annotate a global phase.
            # A one-token row after an existing context is decode; larger query
            # rows are prefill. A final one-token chunked-prefill row remains an
            # acknowledged ambiguity and is called out in the manifest docs.
            row_phase = (
                "decode"
                if query_length == 1 and sequence_length > 1
                else "prefill"
            )
        rows.append(
            AttentionTraceRow(
                request_id=str(request_ids[request_row]),
                request_ordinal=_request_ordinal(str(request_ids[request_row])),
                request_row=request_row,
                phase=row_phase,
                sequence_length=sequence_length,
                logical_query_start=max(0, sequence_length - query_length),
                query_length=query_length,
                output_start=output_start,
                output_end=output_end,
            )
        )
    return rows


def _to_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    detached = value.detach() if hasattr(value, "detach") else value
    if hasattr(detached, "device") and str(detached.device) != "cpu":
        detached = detached.to(device="cpu")
    raw_values = detached.tolist() if hasattr(detached, "tolist") else detached
    return [int(item) for item in raw_values]


def _safe_component(value: str, *, max_length: int = 96) -> str:
    normalized = _SAFE_COMPONENT.sub("_", value).strip("._") or "unknown"
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{normalized[:max_length]}-{digest}"


def _reserve_capture_identity(
    config: AttentionCorrectnessTraceConfig,
    semantic_key: tuple[Any, ...],
) -> tuple[int, int] | None:
    global _CAPTURE_INDEX
    with _STATE_LOCK:
        if config.max_records and _CAPTURE_INDEX >= config.max_records:
            return None
        capture_index = _CAPTURE_INDEX
        _CAPTURE_INDEX += 1
        occurrence = _SEMANTIC_OCCURRENCES.get(semantic_key, 0)
        _SEMANTIC_OCCURRENCES[semantic_key] = occurrence + 1
        return capture_index, occurrence


def _append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with _MANIFEST_LOCK, path.open("a", encoding="utf-8") as manifest:
        manifest.write(line)
        manifest.write("\n")
        manifest.flush()


def _record_attention_output(
    *,
    execution_path: str,
    layer_name: str,
    attn_metadata: Any,
    output: Any,
    query: Any | None,
    key: Any | None,
    value: Any | None,
    softmax_scale: float | None,
    config: AttentionCorrectnessTraceConfig,
) -> int:
    import torch

    if not layer_name:
        raise ValueError("attention correctness trace requires a layer name")
    request_ids = list(getattr(attn_metadata, "request_ids", None) or ())
    if not request_ids:
        raise ValueError("attention correctness trace requires request IDs")
    query_start_loc = getattr(attn_metadata, "query_start_loc_cpu", None)
    if query_start_loc is None:
        query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    sequence_lengths = getattr(attn_metadata, "seq_lens_cpu", None)
    if sequence_lengths is None:
        sequence_lengths = getattr(attn_metadata, "seq_lens", None)
    rows = build_attention_trace_rows(
        request_ids=request_ids,
        query_start_locs=_to_int_list(query_start_loc),
        sequence_lengths=_to_int_list(sequence_lengths),
        phase=str(getattr(attn_metadata, "attention_phase", None) or "unknown"),
    )
    if not rows:
        raise ValueError(
            "attention correctness trace could not build any request rows"
        )
    if config.capture_qkv and (query is None or key is None or value is None):
        raise ValueError(
            "attention correctness Q/K/V capture requires query, key, and "
            "value tensors"
        )

    assert config.output_dir is not None
    written = 0
    layer_component = _safe_component(layer_name)
    rank_dir = config.output_dir / f"rank_{config.rank:03d}"
    manifest_path = rank_dir / "manifest.jsonl"
    for row in rows:
        if row.phase not in config.phases:
            continue
        if not _matches(row.request_id, config.request_patterns):
            continue
        request_component = _safe_component(row.request_id)
        for token_offset in range(0, row.query_length, config.file_chunk_tokens):
            token_count = min(
                config.file_chunk_tokens,
                row.query_length - token_offset,
            )
            logical_start = row.logical_query_start + token_offset
            output_start = row.output_start + token_offset
            output_end = output_start + token_count
            semantic_key = (
                row.request_ordinal,
                layer_name,
                row.phase,
                logical_start,
                token_count,
                execution_path,
            )
            capture_identity = _reserve_capture_identity(config, semantic_key)
            if capture_identity is None:
                return written
            capture_index, occurrence = capture_identity
            relative_path = Path(f"request_{row.request_ordinal:05d}") / (
                request_component
            ) / layer_component / (
                f"{row.phase}_q{logical_start:09d}_n{token_count:06d}_"
                f"occ{occurrence:03d}_cap{capture_index:09d}.pt"
            )
            output_path = rank_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            host_output = (
                output[output_start:output_end]
                .detach()
                .to(device="cpu", copy=True)
                .contiguous()
            )
            payload = {
                "schema_version": 2,
                "tensor_role": "attention_output",
                "execution_path": execution_path,
                "layer_name": layer_name,
                "row": asdict(row),
                "logical_query_start": logical_start,
                "query_length": token_count,
                "capture_index": capture_index,
                "occurrence": occurrence,
                "softmax_scale": softmax_scale,
                "output": host_output,
            }
            if config.capture_qkv:
                payload.update({
                    "query": (
                        query[output_start:output_end]
                        .detach()
                        .to(device="cpu", copy=True)
                        .contiguous()
                    ),
                    "key": (
                        key[output_start:output_end]
                        .detach()
                        .to(device="cpu", copy=True)
                        .contiguous()
                    ),
                    "value": (
                        value[output_start:output_end]
                        .detach()
                        .to(device="cpu", copy=True)
                        .contiguous()
                    ),
                })
            temporary_path = output_path.with_suffix(
                f".tmp-{os.getpid()}-{threading.get_ident()}"
            )
            torch.save(payload, temporary_path)
            os.replace(temporary_path, output_path)
            manifest_record = {
                key: value
                for key, value in payload.items()
                if key not in {"output", "query", "key", "value"}
            }
            manifest_record.update({
                "rank": config.rank,
                "pid": os.getpid(),
                "dtype": str(host_output.dtype),
                "shape": list(host_output.shape),
                "file": str(relative_path),
                "capture_qkv": config.capture_qkv,
            })
            if config.capture_qkv:
                manifest_record.update({
                    "query_dtype": str(payload["query"].dtype),
                    "query_shape": list(payload["query"].shape),
                    "key_dtype": str(payload["key"].dtype),
                    "key_shape": list(payload["key"].shape),
                    "value_dtype": str(payload["value"].dtype),
                    "value_shape": list(payload["value"].shape),
                })
            _append_manifest(manifest_path, manifest_record)
            written += 1
    return written


def maybe_record_attention_output(
    *,
    execution_path: str,
    layer_name: str,
    attn_metadata: Any,
    output: Any,
    query: Any | None = None,
    key: Any | None = None,
    value: Any | None = None,
    softmax_scale: float | None = None,
) -> int:
    """Record one attention return, or do nothing when tracing is disabled."""

    config = get_attention_correctness_trace_config()
    if not config.enabled:
        return 0
    if execution_path not in config.execution_paths:
        return 0
    if not _matches(layer_name, config.layer_patterns):
        return 0
    try:
        return _record_attention_output(
            execution_path=execution_path,
            layer_name=layer_name,
            attn_metadata=attn_metadata,
            output=output,
            query=query,
            key=key,
            value=value,
            softmax_scale=softmax_scale,
            config=config,
        )
    except Exception:
        if config.strict:
            raise
        logger.exception(
            "Attention correctness trace failed for path=%s layer=%s",
            execution_path,
            layer_name,
        )
        return 0
