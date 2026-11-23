# SPDX-License-Identifier: Apache-2.0
"""Green Context stream bridge for vendor_v7 P/D multiplexing experiments.

This module is default-off and intentionally isolated from upstream vLLM. It
loads a small C ABI shared library that owns CUDA Driver API Green Context
objects, wraps the returned raw CUstream with torch.cuda.ExternalStream, and
supports both the inherited fixed split and immutable dynamic profile bank.
"""

from __future__ import annotations

import ctypes
import contextvars
import os
import threading
import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.pd_layerwise_handoff import (
    PD_LAYERWISE_HANDOFF_CONTROLLER,
    layerwise_profile_handoff_requested,
)
from vllm.model_executor.layers.pd_prefill_workspace import (
    handoff_active_prefill_workspace,
)
from vllm.pd_sm_profiles import PDSMProfile, parse_pd_sm_profiles, profile_by_id

logger = init_logger(__name__)

T = TypeVar("T")

_ENABLE_ENV = "VLLM_OPERATOR_SELECTIVE_PD_GREEN_CONTEXT"
_MUXWISE_PHASE_ENABLE_ENV = "VLLM_MUXWISE_PHASE_BASELINE_GREEN_CONTEXT"
_STRICT_ENV = "VLLM_OPERATOR_SELECTIVE_PD_GREEN_STRICT"
_LIB_ENV = "VLLM_OPERATOR_SELECTIVE_PD_GREEN_LIB"
_DEVICE_ENV = "VLLM_OPERATOR_SELECTIVE_PD_GREEN_DEVICE"
_DECODE_SMS_ENV = "VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS"
_PREFILL_SMS_ENV = "VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS"
_DECODE_PRIORITY_ENV = "VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_PRIORITY"
_PREFILL_PRIORITY_ENV = "VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_PRIORITY"
_EXECUTE_MODEL_ENV = "VLLM_OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_MODEL"
_EXECUTE_MODEL_NO_JOIN_ENV = "VLLM_OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_NO_JOIN"
_DYNAMIC_SM_ENABLE_ENV = "VLLM_PD_DYNAMIC_SM_ENABLED"
_DYNAMIC_SM_SHADOW_ENV = "VLLM_PD_DYNAMIC_SM_SHADOW"
_DYNAMIC_SM_STRICT_ENV = "VLLM_PD_DYNAMIC_SM_STRICT"
_DYNAMIC_SM_PROFILES_ENV = "VLLM_PD_DYNAMIC_SM_PROFILES"
_DYNAMIC_SM_ALIGNMENT_ENV = "VLLM_PD_DYNAMIC_SM_ALIGNMENT"
_DYNAMIC_SM_TOTAL_SMS_ENV = "VLLM_PD_DYNAMIC_SM_TOTAL_SMS"
_DYNAMIC_SM_EAGER_PROFILES_ENV = "VLLM_PD_DYNAMIC_SM_EAGER_PROFILES"
_PROCESS_LIFETIME_ENV = "VLLM_ISOLATION_GREEN_CONTEXT_PROCESS_LIFETIME"

_PHASE_DECODE = 0
_PHASE_PREFILL = 1

_ACTIVE_PD_SM_PROFILE: contextvars.ContextVar[tuple[str, int] | None] = (
    contextvars.ContextVar("vllm_active_pd_sm_profile", default=None)
)

# Keep one layer executing and at most one additional layer queued.  The wait
# runs outside the model-execution thread so it releases that thread's Python
# execution slot while CUDA advances.  This is intentionally fixed for the
# first correctness-oriented implementation; it is not a tuning policy.
_LAYERWISE_MAX_QUEUED_LAYERS = 2
_LAYERWISE_BOUNDARY_WAIT_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="vllm-pd-layer-boundary",
)


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning_once(
            "Invalid %s=%r; using default %d",
            name,
            value,
            default,
            scope="local",
        )
        return default


def green_context_pd_enabled() -> bool:
    return _env_bool(_ENABLE_ENV) or _env_bool(_MUXWISE_PHASE_ENABLE_ENV)


def green_context_execute_model_enabled() -> bool:
    return green_context_pd_enabled() and _env_bool(_EXECUTE_MODEL_ENV)


def green_context_execute_model_no_join_enabled() -> bool:
    return _env_bool(_EXECUTE_MODEL_NO_JOIN_ENV, "1")


def _green_context_strict() -> bool:
    return _env_bool(_STRICT_ENV)


def green_context_dynamic_sm_enabled() -> bool:
    return (
        green_context_pd_enabled()
        and green_context_execute_model_enabled()
        and _env_bool(_DYNAMIC_SM_ENABLE_ENV)
        and not _env_bool(_DYNAMIC_SM_SHADOW_ENV)
    )


def _dynamic_sm_strict() -> bool:
    return _env_bool(_DYNAMIC_SM_STRICT_ENV, "1")


def _phase_to_driver_id(phase: str | None) -> int | None:
    if phase is None:
        return None
    normalized = phase.strip().lower()
    if normalized == "decode":
        return _PHASE_DECODE
    if normalized == "prefill":
        return _PHASE_PREFILL
    return None


class GreenContextError(RuntimeError):
    pass


@dataclass
class GreenPairResult:
    decode_result: Any
    prefill_result: Any
    timings: dict[str, float | str]


@dataclass
class PDLayerwiseHandoffExecution:
    """Mutable result for one phase-pure Prefill execute_model call."""

    enabled: bool = False
    generation: int = 0
    source_profile_id: str = ""
    source_epoch: int = 0
    effective_profile_id: str = ""
    effective_profile_epoch: int = 0
    applied: bool = False
    applied_layer: int = -1
    applied_count: int = 0
    _profile_tokens: list[contextvars.Token] = field(
        default_factory=list,
        repr=False,
    )
    _boundary_events: deque[Any] = field(
        default_factory=deque,
        repr=False,
    )
    _boundary_wait_future: Future[None] | None = field(
        default=None,
        repr=False,
    )
    _boundary_wake_event: threading.Event | None = field(
        default=None,
        repr=False,
    )
    _handoff_event: Any | None = field(default=None, repr=False)

    def as_debug_dict(self) -> dict[str, int | float | str]:
        return {
            **PD_LAYERWISE_HANDOFF_CONTROLLER.snapshot(),
            "pd_layerwise_handoff_execution_enabled": float(self.enabled),
            "pd_layerwise_handoff_execution_generation": float(
                self.generation
            ),
            "pd_layerwise_handoff_execution_source_profile": (
                self.source_profile_id
            ),
            "pd_layerwise_handoff_execution_source_epoch": float(
                self.source_epoch
            ),
            "pd_layerwise_handoff_execution_effective_profile": (
                self.effective_profile_id
            ),
            "pd_layerwise_handoff_execution_effective_epoch": float(
                self.effective_profile_epoch
            ),
            "pd_layerwise_handoff_execution_applied": float(self.applied),
            "pd_layerwise_handoff_execution_applied_layer": float(
                self.applied_layer
            ),
            "pd_layerwise_handoff_execution_applied_count": float(
                self.applied_count
            ),
        }


_ACTIVE_LAYERWISE_HANDOFF: contextvars.ContextVar[
    PDLayerwiseHandoffExecution | None
] = contextvars.ContextVar("vllm_active_pd_layerwise_handoff", default=None)


class _BridgeLibrary:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        self._configure_abi()

    def _configure_abi(self) -> None:
        self.lib.vllm_green_pd_create.argtypes = [
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.vllm_green_pd_create.restype = ctypes.c_int

        self.lib.vllm_green_pd_destroy.argtypes = [ctypes.c_void_p]
        self.lib.vllm_green_pd_destroy.restype = ctypes.c_int

        self.lib.vllm_green_pd_synchronize.argtypes = [ctypes.c_void_p]
        self.lib.vllm_green_pd_synchronize.restype = ctypes.c_int

        self.lib.vllm_green_pd_stream.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.vllm_green_pd_stream.restype = ctypes.c_uint64

        self._sm_count_fn = getattr(self.lib, "vllm_green_pd_sm_count", None)
        if self._sm_count_fn is not None:
            self._sm_count_fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self._sm_count_fn.restype = ctypes.c_int

    def create(
        self,
        *,
        device: int,
        decode_sms: int,
        prefill_sms: int,
        decode_priority: int,
        prefill_priority: int,
    ) -> ctypes.c_void_p:
        handle = ctypes.c_void_p()
        error = ctypes.create_string_buffer(4096)
        rc = self.lib.vllm_green_pd_create(
            int(device),
            int(decode_sms),
            int(prefill_sms),
            int(decode_priority),
            int(prefill_priority),
            ctypes.byref(handle),
            error,
            ctypes.sizeof(error),
        )
        if rc != 0 or not handle.value:
            message = error.value.decode("utf-8", errors="replace")
            raise GreenContextError(message or f"vllm_green_pd_create failed: {rc}")
        return handle

    def destroy(self, handle: ctypes.c_void_p) -> None:
        if handle and handle.value:
            self.lib.vllm_green_pd_destroy(handle)

    def synchronize(self, handle: ctypes.c_void_p) -> None:
        rc = self.lib.vllm_green_pd_synchronize(handle)
        if rc != 0:
            raise GreenContextError(f"vllm_green_pd_synchronize failed: {rc}")

    def stream_ptr(self, handle: ctypes.c_void_p, phase: int) -> int:
        if not handle or not handle.value:
            return 0
        return int(self.lib.vllm_green_pd_stream(handle, int(phase)))

    def sm_count(self, handle: ctypes.c_void_p, phase: int) -> int | None:
        if not handle or not handle.value or self._sm_count_fn is None:
            return None
        count = int(self._sm_count_fn(handle, int(phase)))
        if count < 0:
            raise GreenContextError(
                f"vllm_green_pd_sm_count failed for phase {phase}"
            )
        return count

    @property
    def supports_sm_count(self) -> bool:
        return self._sm_count_fn is not None


def _default_library_path() -> Path:
    return Path(__file__).resolve().parent / "green_context_lib" / (
        "libvllm_green_context.so"
    )


def _library_path() -> Path:
    override = os.environ.get(_LIB_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return _default_library_path()


def _current_torch_stream() -> torch.cuda.Stream:
    try:
        from vllm.utils.torch_utils import current_stream

        return current_stream()
    except Exception:
        return torch.cuda.current_stream()


class _GreenContextStreams:
    def __init__(
        self,
        *,
        profile: PDSMProfile | None = None,
        library: _BridgeLibrary | None = None,
    ) -> None:
        self._process_owned = False
        self._closed = False
        self._process_lifetime = _env_bool(_PROCESS_LIFETIME_ENV)
        self.library = library or _BridgeLibrary(_library_path())
        current_device = (
            torch.cuda.current_device() if torch.cuda.is_available() else 0
        )
        self.device = _env_int(_DEVICE_ENV, current_device)
        self.profile_id = profile.profile_id if profile is not None else "static"
        self.decode_sms = (
            profile.decode_sms
            if profile is not None
            else max(0, _env_int(_DECODE_SMS_ENV, 8))
        )
        self.prefill_sms = (
            profile.prefill_sms
            if profile is not None
            else max(0, _env_int(_PREFILL_SMS_ENV, 0))
        )
        self.decode_priority = _env_int(_DECODE_PRIORITY_ENV, -1)
        self.prefill_priority = _env_int(_PREFILL_PRIORITY_ENV, 0)
        self.handle = self.library.create(
            device=self.device,
            decode_sms=self.decode_sms,
            prefill_sms=self.prefill_sms,
            decode_priority=self.decode_priority,
            prefill_priority=self.prefill_priority,
        )
        self.decode_stream = self._make_external_stream(_PHASE_DECODE)
        self.prefill_stream = self._make_external_stream(_PHASE_PREFILL)
        queried_decode_sms = self.library.sm_count(self.handle, _PHASE_DECODE)
        queried_prefill_sms = self.library.sm_count(self.handle, _PHASE_PREFILL)
        self.actual_decode_sms = (
            queried_decode_sms
            if queried_decode_sms is not None
            else self.decode_sms
        )
        self.actual_prefill_sms = (
            queried_prefill_sms
            if queried_prefill_sms is not None
            else self.prefill_sms
        )
        self.sm_count_verified = self.library.supports_sm_count
        if self.sm_count_verified and (
            self.actual_decode_sms != self.decode_sms
            or self.actual_prefill_sms != self.prefill_sms
        ):
            message = (
                "Green Context SM verification mismatch for "
                f"{self.profile_id}: requested={self.decode_sms}/"
                f"{self.prefill_sms} actual={self.actual_decode_sms}/"
                f"{self.actual_prefill_sms}"
            )
            self.close()
            raise GreenContextError(message)
        logger.info(
            "Green Context P/D streams enabled: profile=%s device=%d "
            "decode_sms=%d prefill_sms=%d verified=%s lib=%s lifetime=%s",
            self.profile_id,
            self.device,
            self.actual_decode_sms,
            self.actual_prefill_sms,
            self.sm_count_verified,
            self.library.path,
            "process" if self._process_owned else "python_owner",
        )

    def _make_external_stream(self, phase: int) -> torch.cuda.Stream | None:
        stream_ptr = self.library.stream_ptr(self.handle, phase)
        if stream_ptr == 0:
            return None
        external_stream_cls = getattr(torch.cuda, "ExternalStream", None)
        if external_stream_cls is None:
            raise GreenContextError("torch.cuda.ExternalStream is unavailable")
        if self._process_lifetime and not self._process_owned:
            # C10 may retain the external stream in allocator/event state even
            # after every Python Stream reference disappears. Adopt BEFORE the
            # call: a partially failed export cannot prove no reference escaped.
            with _INITIALIZATION_LOCK:
                if _PROCESS_OWNED_GREEN_RETIRED:
                    raise GreenContextError(
                        "GreenContext serving has retired; start a new worker process"
                    )
                _PROCESS_OWNED_GREEN_HANDLES.append((self.library, self.handle))
                self._process_owned = True
        return external_stream_cls(stream_ptr, device=self.device)

    def stream_for_phase(self, phase: int) -> torch.cuda.Stream | None:
        self._ensure_open()
        if phase == _PHASE_DECODE:
            return self.decode_stream
        if phase == _PHASE_PREFILL:
            return self.prefill_stream
        return None

    def _ensure_open(self) -> None:
        if self._closed:
            raise GreenContextError("GreenContext streams have retired")

    def close(self) -> None:
        if getattr(self, "_process_owned", False):
            # Logical retirement only. Tensors, graph pools and C10 allocator
            # events can still reference this context. The worker process (not
            # Python GC order) owns final driver cleanup. Retain NO tensors here.
            self._closed = True
            return
        if hasattr(self, "library") and hasattr(self, "handle"):
            self.library.destroy(self.handle)
            self.handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def run_on_phase_stream(self, phase: int, fn: Callable[[], T]) -> T:
        self._ensure_open()
        stream = self.decode_stream if phase == _PHASE_DECODE else self.prefill_stream
        if stream is None:
            return fn()

        current = _current_torch_stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            result = fn()
        current.wait_stream(stream)
        return result

    def run_decode_prefill(
        self,
        decode_fn: Callable[[], T],
        prefill_fn: Callable[[], T],
        *,
        overlap: bool,
        collect_timing: bool,
    ) -> GreenPairResult:
        self._ensure_open()
        decode_stream = self.decode_stream
        prefill_stream = self.prefill_stream
        if decode_stream is None or prefill_stream is None:
            timing_enabled = collect_timing and torch.cuda.is_available()
            base_event = None
            decode_start = None
            decode_end = None
            prefill_start = None
            prefill_end = None
            if timing_enabled:
                current = _current_torch_stream()
                base_event = torch.cuda.Event(enable_timing=True)
                decode_start = torch.cuda.Event(enable_timing=True)
                decode_end = torch.cuda.Event(enable_timing=True)
                prefill_start = torch.cuda.Event(enable_timing=True)
                prefill_end = torch.cuda.Event(enable_timing=True)
                base_event.record(current)
                decode_start.record(current)
            decode_result = self.run_on_phase_stream(_PHASE_DECODE, decode_fn)
            if decode_end is not None:
                decode_end.record(_current_torch_stream())
                decode_end.synchronize()
            if prefill_start is not None:
                prefill_start.record(_current_torch_stream())
            prefill_result = self.run_on_phase_stream(_PHASE_PREFILL, prefill_fn)
            if prefill_end is not None:
                prefill_end.record(_current_torch_stream())
                prefill_end.synchronize()
            timings: dict[str, float | str] = {
                "fallback_reason": (
                    "missing_green_stream_for_overlap" if overlap else ""
                )
            }
            if (
                base_event is not None
                and decode_start is not None
                and decode_end is not None
                and prefill_start is not None
                and prefill_end is not None
            ):
                decode_start_ms = base_event.elapsed_time(decode_start)
                decode_end_ms = base_event.elapsed_time(decode_end)
                prefill_start_ms = base_event.elapsed_time(prefill_start)
                prefill_end_ms = base_event.elapsed_time(prefill_end)
                timings.update(
                    {
                        "decode_ms": decode_start.elapsed_time(decode_end),
                        "prefill_ms": prefill_start.elapsed_time(prefill_end),
                        "decode_start_ms": decode_start_ms,
                        "decode_end_ms": decode_end_ms,
                        "prefill_start_ms": prefill_start_ms,
                        "prefill_end_ms": prefill_end_ms,
                        "union_ms": prefill_end_ms - decode_start_ms,
                        "overlap_ms": 0.0,
                        "join_wait_ms": max(0.0, prefill_end_ms - decode_end_ms),
                    }
                )
            return GreenPairResult(
                decode_result,
                prefill_result,
                timings,
            )

        current = _current_torch_stream()
        timing_enabled = collect_timing and torch.cuda.is_available()
        base_event = None
        decode_start = None
        decode_end = None
        prefill_start = None
        prefill_end = None
        if timing_enabled:
            base_event = torch.cuda.Event(enable_timing=True)
            decode_start = torch.cuda.Event(enable_timing=True)
            decode_end = torch.cuda.Event(enable_timing=True)
            prefill_start = torch.cuda.Event(enable_timing=True)
            prefill_end = torch.cuda.Event(enable_timing=True)
            base_event.record(current)

        if overlap:
            decode_stream.wait_stream(current)
            prefill_stream.wait_stream(current)
            with torch.cuda.stream(decode_stream):
                if decode_start is not None:
                    decode_start.record()
                decode_result = decode_fn()
                if decode_end is not None:
                    decode_end.record()
            with torch.cuda.stream(prefill_stream):
                if prefill_start is not None:
                    prefill_start.record()
                prefill_result = prefill_fn()
                if prefill_end is not None:
                    prefill_end.record()
            current.wait_stream(decode_stream)
            current.wait_stream(prefill_stream)
        else:
            decode_stream.wait_stream(current)
            with torch.cuda.stream(decode_stream):
                if decode_start is not None:
                    decode_start.record()
                decode_result = decode_fn()
                if decode_end is not None:
                    decode_end.record()
            current.wait_stream(decode_stream)

            prefill_stream.wait_stream(current)
            with torch.cuda.stream(prefill_stream):
                if prefill_start is not None:
                    prefill_start.record()
                prefill_result = prefill_fn()
                if prefill_end is not None:
                    prefill_end.record()
            current.wait_stream(prefill_stream)

        timings: dict[str, float | str] = {"fallback_reason": ""}
        if (
            base_event is not None
            and decode_start is not None
            and decode_end is not None
            and prefill_start is not None
            and prefill_end is not None
        ):
            decode_end.synchronize()
            prefill_end.synchronize()
            decode_start_ms = base_event.elapsed_time(decode_start)
            decode_end_ms = base_event.elapsed_time(decode_end)
            prefill_start_ms = base_event.elapsed_time(prefill_start)
            prefill_end_ms = base_event.elapsed_time(prefill_end)
            overlap_ms = max(
                0.0,
                min(decode_end_ms, prefill_end_ms)
                - max(decode_start_ms, prefill_start_ms),
            )
            timings.update(
                {
                    "decode_ms": decode_start.elapsed_time(decode_end),
                    "prefill_ms": prefill_start.elapsed_time(prefill_end),
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
            )
        return GreenPairResult(decode_result, prefill_result, timings)


class _GreenContextProfileBank:
    """Own one stream pair per configured immutable SM profile."""

    def __init__(self) -> None:
        self._closed = False
        self._process_lifetime = _env_bool(_PROCESS_LIFETIME_ENV)
        self._streams: dict[str, _GreenContextStreams] = {}
        self._failed_profiles: dict[str, str] = {}
        self._lock = threading.RLock()
        self._active_profile_id = ""
        self._active_epoch = 0
        self._handoff_count = 0
        self._handoff_sync_ms_total = 0.0
        self._last_handoff_sync_ms = 0.0
        self._layerwise_commit_count = 0
        current_device = (
            torch.cuda.current_device() if torch.cuda.is_available() else 0
        )
        self.device = _env_int(_DEVICE_ENV, current_device)
        if not torch.cuda.is_available():
            raise GreenContextError("CUDA is unavailable for dynamic SM profiles")
        self.total_sms = int(
            torch.cuda.get_device_properties(self.device).multi_processor_count
        )
        configured_total = max(0, _env_int(_DYNAMIC_SM_TOTAL_SMS_ENV, 0))
        if configured_total and configured_total != self.total_sms:
            raise GreenContextError(
                "VLLM_PD_DYNAMIC_SM_TOTAL_SMS does not match the CUDA device: "
                f"configured={configured_total} actual={self.total_sms}"
            )
        self.alignment = max(1, _env_int(_DYNAMIC_SM_ALIGNMENT_ENV, 2))
        self.profiles = parse_pd_sm_profiles(
            os.environ.get(_DYNAMIC_SM_PROFILES_ENV, "auto"),
            total_sms=self.total_sms,
            alignment=self.alignment,
        )
        self.library = _BridgeLibrary(_library_path())
        if _dynamic_sm_strict() and not self.library.supports_sm_count:
            raise GreenContextError(
                "dynamic SM strict mode requires the vendor_v7 GreenContext "
                "bridge with vllm_green_pd_sm_count; rebuild the bridge"
            )
        if _env_bool(_DYNAMIC_SM_EAGER_PROFILES_ENV, "1"):
            try:
                for profile in self.profiles:
                    self.get(profile.profile_id)
            except Exception:
                self.close()
                raise
        logger.info(
            "Dynamic P/D GreenContext profile bank enabled: device=%d "
            "total_sms=%d profiles=%s eager=%s",
            self.device,
            self.total_sms,
            ",".join(profile.profile_id for profile in self.profiles),
            _env_bool(_DYNAMIC_SM_EAGER_PROFILES_ENV, "1"),
        )

    def get(self, profile_id: str) -> _GreenContextStreams | None:
        with self._lock:
            if self._closed:
                raise GreenContextError(
                    "GreenContext bank has retired; start a new worker process"
                )
            streams = self._streams.get(profile_id)
            if streams is not None:
                return streams
            if profile_id in self._failed_profiles:
                if _dynamic_sm_strict():
                    raise GreenContextError(self._failed_profiles[profile_id])
                return None
            profile = profile_by_id(self.profiles, profile_id)
            if profile is None:
                message = f"unknown dynamic P/D SM profile {profile_id!r}"
                if _dynamic_sm_strict():
                    raise GreenContextError(message)
                self._failed_profiles[profile_id] = message
                return None
            try:
                streams = _GreenContextStreams(
                    profile=profile,
                    library=self.library,
                )
            except Exception as exc:
                message = f"failed to create profile {profile_id}: {exc}"
                self._failed_profiles[profile_id] = message
                if _dynamic_sm_strict():
                    raise GreenContextError(message) from exc
                logger.warning_once(message, scope="local")
                return None
            self._streams[profile_id] = streams
            return streams

    def activate(
        self,
        profile_id: str,
        profile_epoch: int,
    ) -> _GreenContextStreams | None:
        """Select an immutable profile at a quiescent handoff boundary.

        EngineCore normally prevents a profile change while a phase future is
        in flight.  The default-off layerwise Prefill-tail continuation bypasses
        this method while the future is active and uses a stream-local CUDA
        event instead.  EngineCore reaches activate() later at the normal
        quiescent commit.  The explicit synchronization below therefore closes
        the CUDA-side gap for whole-future transitions only.
        """
        with self._lock:
            streams = self.get(profile_id)
            if streams is None:
                return None

            requested_epoch = int(profile_epoch)
            if (
                self._active_profile_id
                and requested_epoch < self._active_epoch
            ):
                message = (
                    "dynamic P/D SM profile epoch moved backwards: "
                    f"active={self._active_profile_id}@{self._active_epoch} "
                    f"requested={profile_id}@{requested_epoch}"
                )
                if _dynamic_sm_strict():
                    raise GreenContextError(message)
                logger.warning_once(message, scope="local")
                return None

            if profile_id != self._active_profile_id:
                if self._active_profile_id:
                    previous = self._streams.get(self._active_profile_id)
                    if previous is None:
                        raise GreenContextError(
                            "active dynamic P/D SM profile disappeared: "
                            f"{self._active_profile_id!r}"
                        )
                    handoff_start = time.perf_counter()
                    previous.library.synchronize(previous.handle)
                    handoff_ms = (time.perf_counter() - handoff_start) * 1000.0
                    self._handoff_count += 1
                    self._last_handoff_sync_ms = handoff_ms
                    self._handoff_sync_ms_total += handoff_ms
                self._active_profile_id = profile_id
            self._active_epoch = max(self._active_epoch, requested_epoch)
            return streams

    def commit_completed_layerwise_boundary(
        self,
        *,
        source_profile_id: str,
        source_epoch: int,
        target_profile_id: str,
        target_epoch: int,
    ) -> _GreenContextStreams:
        """Commit a GPU-complete layer boundary without a handle-wide sync.

        EngineCore calls this only after the generation-owned Prefill CUDA
        event completed and the source Decode queue drained.  Unlike
        :meth:`activate`, this method is not a data-path fallback: every source
        and target field must exactly match the active bank and controller
        epochs before its scalar authority can advance.
        """

        with self._lock:
            if not PD_LAYERWISE_HANDOFF_CONTROLLER.completed_transition_matches(
                source_profile_id=source_profile_id,
                source_epoch=source_epoch,
                target_profile_id=target_profile_id,
                target_epoch=target_epoch,
            ):
                raise GreenContextError(
                    "layerwise GreenContext commit has no completed boundary "
                    f"ACK: {source_profile_id}@{source_epoch} -> "
                    f"{target_profile_id}@{target_epoch}"
                )
            if (
                self._active_profile_id != source_profile_id
                or self._active_epoch != int(source_epoch)
            ):
                raise GreenContextError(
                    "layerwise GreenContext source authority mismatch: "
                    f"bank={self._active_profile_id}@{self._active_epoch} "
                    f"requested={source_profile_id}@{source_epoch}"
                )
            if int(target_epoch) != int(source_epoch) + 1:
                raise GreenContextError(
                    "layerwise GreenContext target epoch is not consecutive: "
                    f"source={source_epoch} target={target_epoch}"
                )
            streams = self.get(target_profile_id)
            if streams is None:
                raise GreenContextError(
                    "layerwise GreenContext target profile is unavailable: "
                    f"{target_profile_id!r}"
                )
            self._active_profile_id = target_profile_id
            self._active_epoch = int(target_epoch)
            self._handoff_count += 1
            self._last_handoff_sync_ms = 0.0
            self._layerwise_commit_count += 1
            return streams

    def peek(self, profile_id: str) -> _GreenContextStreams | None:
        with self._lock:
            return self._streams.get(profile_id)

    @property
    def created_count(self) -> int:
        with self._lock:
            return len(self._streams)

    @property
    def failed_count(self) -> int:
        with self._lock:
            return len(self._failed_profiles)

    @property
    def all_sm_counts_verified(self) -> bool:
        with self._lock:
            return bool(self._streams) and all(
                streams.sm_count_verified for streams in self._streams.values()
            )

    @property
    def active_profile_id(self) -> str:
        with self._lock:
            return self._active_profile_id

    @property
    def active_epoch(self) -> int:
        with self._lock:
            return self._active_epoch

    @property
    def handoff_count(self) -> int:
        with self._lock:
            return self._handoff_count

    @property
    def handoff_sync_ms_total(self) -> float:
        with self._lock:
            return self._handoff_sync_ms_total

    @property
    def last_handoff_sync_ms(self) -> float:
        with self._lock:
            return self._last_handoff_sync_ms

    @property
    def layerwise_commit_count(self) -> int:
        with self._lock:
            return self._layerwise_commit_count

    def close(self) -> None:
        with self._lock:
            if self._process_lifetime:
                self._closed = True
            for streams in self._streams.values():
                streams.close()
            self._streams.clear()
            self._active_profile_id = ""
            self._active_epoch = 0
            self._layerwise_commit_count = 0

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


_STREAMS: _GreenContextStreams | None = None
_PROFILE_BANK: _GreenContextProfileBank | None = None
_INITIALIZATION_FAILED = False
_PROFILE_BANK_INITIALIZATION_FAILED = False
_INITIALIZATION_LOCK = threading.RLock()
# Only raw handles/bridge owners, never model tensors or temporary arenas.
# Deliberately no atexit destroy: C10 cleanup can run later than Python atexit.
_PROCESS_OWNED_GREEN_HANDLES: list[tuple[_BridgeLibrary, ctypes.c_void_p]] = []
_PROCESS_OWNED_GREEN_RETIRED = False


def retire_process_owned_green_contexts() -> None:
    """Stop bank reuse without invalidating late CUDA/allocator consumers.

    Called after the ordinary executor/worker shutdown path. This is NOT a
    GPU-completion acknowledgement or permission to release KV/workspace. An
    isolated worker process is required for the next experiment/engine.
    """
    global _PROCESS_OWNED_GREEN_RETIRED
    with _INITIALIZATION_LOCK:
        if _PROCESS_OWNED_GREEN_RETIRED or not _PROCESS_OWNED_GREEN_HANDLES:
            return
        _PROCESS_OWNED_GREEN_RETIRED = True
        if _PROFILE_BANK is not None:
            _PROFILE_BANK.close()
        if _STREAMS is not None:
            _STREAMS.close()
        logger.info(
            "GreenContext serving retired: process_owned_handles=%d; "
            "CUDA contexts remain alive for late allocator cleanup until process exit",
            len(_PROCESS_OWNED_GREEN_HANDLES),
        )


def _get_static_streams() -> _GreenContextStreams | None:
    global _STREAMS, _INITIALIZATION_FAILED
    if _PROCESS_OWNED_GREEN_RETIRED:
        raise GreenContextError("GreenContext serving has retired; restart the process")
    if _STREAMS is not None:
        return _STREAMS
    if _INITIALIZATION_FAILED:
        return None
    try:
        _STREAMS = _GreenContextStreams()
        return _STREAMS
    except Exception as exc:
        _INITIALIZATION_FAILED = True
        if _green_context_strict():
            raise
        logger.warning_once(
            "Green Context P/D stream initialization failed; falling back to "
            "normal streams: %s",
            str(exc),
            scope="local",
        )
        return None


def _get_profile_bank() -> _GreenContextProfileBank | None:
    global _PROFILE_BANK, _PROFILE_BANK_INITIALIZATION_FAILED
    if _PROCESS_OWNED_GREEN_RETIRED:
        raise GreenContextError("GreenContext serving has retired; restart the process")
    if _PROFILE_BANK is not None:
        return _PROFILE_BANK
    if _PROFILE_BANK_INITIALIZATION_FAILED:
        return None
    try:
        _PROFILE_BANK = _GreenContextProfileBank()
        return _PROFILE_BANK
    except Exception as exc:
        _PROFILE_BANK_INITIALIZATION_FAILED = True
        if _dynamic_sm_strict():
            raise
        logger.warning_once(
            "Dynamic P/D GreenContext profile-bank initialization failed; "
            "falling back to the fixed profile: %s",
            str(exc),
            scope="local",
        )
        return None


def initialize_dynamic_green_context_profiles() -> dict[str, int | str]:
    """Create and verify the dynamic profile bank before serving requests."""
    if not green_context_dynamic_sm_enabled():
        return {"enabled": 0, "total_sms": 0, "profile_ids": ""}
    with _INITIALIZATION_LOCK:
        bank = _get_profile_bank()
        if bank is None:
            if _dynamic_sm_strict():
                raise GreenContextError(
                    "dynamic P/D SM profile bank is unavailable"
                )
            return {"enabled": 0, "total_sms": 0, "profile_ids": ""}
        eager = _env_bool(_DYNAMIC_SM_EAGER_PROFILES_ENV, "1")
        complete = not eager or bank.created_count == len(bank.profiles)
        return {
            "enabled": int(complete),
            "total_sms": bank.total_sms,
            "profile_ids": ",".join(
                profile.profile_id for profile in bank.profiles
            ),
            "created_profiles": bank.created_count,
            "failed_profiles": bank.failed_count,
            "sm_counts_verified": int(bank.all_sm_counts_verified),
        }


def commit_dynamic_green_context_profile_at_layer_boundary(
    source_profile_id: str,
    source_epoch: int,
    target_profile_id: str,
    target_epoch: int,
) -> dict[str, int | str]:
    """Commit a controller-proven completed layer boundary on this worker."""

    with _INITIALIZATION_LOCK:
        bank = _get_profile_bank()
        if bank is None:
            raise GreenContextError(
                "dynamic GreenContext profile bank is unavailable for a "
                "layerwise boundary commit"
            )
        streams = bank.commit_completed_layerwise_boundary(
            source_profile_id=source_profile_id,
            source_epoch=source_epoch,
            target_profile_id=target_profile_id,
            target_epoch=target_epoch,
        )
        return {
            "committed": 1,
            "profile_id": streams.profile_id,
            "profile_epoch": bank.active_epoch,
            "layerwise_commit_count": bank.layerwise_commit_count,
        }


def _get_streams(
    profile_id: str | None = None,
    profile_epoch: int = 0,
) -> _GreenContextStreams | None:
    if not green_context_pd_enabled():
        return None
    with _INITIALIZATION_LOCK:
        dynamic_enabled = green_context_dynamic_sm_enabled()
        if dynamic_enabled and not profile_id:
            active_profile = _ACTIVE_PD_SM_PROFILE.get()
            if active_profile is not None and active_profile[0]:
                profile_id, profile_epoch = active_profile
        if profile_id and dynamic_enabled:
            bank = _get_profile_bank()
            if bank is not None:
                streams = bank.activate(profile_id, profile_epoch)
                if streams is not None:
                    return streams
            # An output carrying a dynamic profile must never execute on the
            # fixed singleton: that would make the trace claim one allocation
            # while the GPU used another. Non-strict startup may disable the
            # controller, but an already-routed dynamic output is fail-fast.
            raise GreenContextError(
                f"dynamic P/D SM profile {profile_id!r} is unavailable"
            )
        if dynamic_enabled:
            raise GreenContextError(
                "dynamic P/D SM execution reached GreenContext routing without "
                "an assigned profile"
            )
        return _get_static_streams()


def _get_precreated_layerwise_target_streams(
    profile_id: str,
) -> _GreenContextStreams:
    """Look up a handoff target without activating its profile bank entry.

    ``_GreenContextProfileBank.activate`` performs a whole-handle
    synchronization. A layer continuation waits for its exact CUDA event and
    EngineCore explicitly commits the bank before releasing target admission.
    """

    with _INITIALIZATION_LOCK:
        bank = _get_profile_bank()
        streams = bank.peek(profile_id) if bank is not None else None
    if streams is None:
        raise GreenContextError(
            "layerwise P/D SM handoff requires an eagerly initialized target "
            f"profile: {profile_id!r}"
        )
    return streams


def _submit_layer_boundary_wait(
    boundary_event: Any,
    wake_event: threading.Event,
) -> Future[None]:
    """Wait for one CUDA event off the model and EngineCore host threads."""

    future = _LAYERWISE_BOUNDARY_WAIT_EXECUTOR.submit(
        boundary_event.synchronize
    )
    future.add_done_callback(lambda _completed: wake_event.set())
    return future


@contextmanager
def green_context_layerwise_prefill_handoff(
    phase: str | None,
    *,
    active: bool,
    profile_id: str,
    profile_epoch: int,
) -> Iterator[PDLayerwiseHandoffExecution]:
    """Expose real CUDA layer boundaries for one armed Prefill future.

    The controller is armed before execute_model submission. CUDA events bound
    the host to a two-layer submission window. Each EngineCore request adds a
    physical source-layer completion at the next host-visible boundary. Source
    Decode can continue while that event is pending. The remaining Prefill
    layers change streams only after EngineCore drains the final source Decode
    Future and explicitly commits the target profile.
    """

    execution = PDLayerwiseHandoffExecution(
        source_profile_id=profile_id,
        source_epoch=int(profile_epoch),
        effective_profile_id=profile_id,
        effective_profile_epoch=int(profile_epoch),
    )
    if (
        not active
        or phase != "prefill"
        or not profile_id
        or not green_context_dynamic_sm_enabled()
        or not layerwise_profile_handoff_requested()
    ):
        yield execution
        return

    generation = PD_LAYERWISE_HANDOFF_CONTROLLER.begin_execution(
        source_profile_id=profile_id,
        source_epoch=profile_epoch,
    )
    if generation <= 0:
        yield execution
        return

    execution.enabled = True
    execution.generation = generation
    execution._boundary_wake_event = (
        PD_LAYERWISE_HANDOFF_CONTROLLER.wake_event_for(generation)
    )
    if execution._boundary_wake_event is None:
        execution.enabled = False
        yield execution
        return
    context_token = _ACTIVE_LAYERWISE_HANDOFF.set(execution)
    try:
        yield execution
    finally:
        try:
            while execution._profile_tokens:
                _ACTIVE_PD_SM_PROFILE.reset(
                    execution._profile_tokens.pop()
                )
        finally:
            _ACTIVE_LAYERWISE_HANDOFF.reset(context_token)
            PD_LAYERWISE_HANDOFF_CONTROLLER.complete_execution(generation)


def pd_layerwise_handoff_layer_boundary(layer_index: int) -> None:
    """Bound source submission or continue on a requested target profile."""

    execution = _ACTIVE_LAYERWISE_HANDOFF.get()
    if (
        execution is None
        or not execution.enabled
        or not PD_LAYERWISE_HANDOFF_CONTROLLER.should_bound_submission(
            execution.generation
        )
    ):
        return

    # Recording is asynchronous.  The event represents all source-profile work
    # through this completed Python layer submission, including its hidden
    # state producer kernels.
    current_stream = _current_torch_stream()
    boundary_event = torch.cuda.Event()
    boundary_event.record(current_stream)
    execution._boundary_events.append(boundary_event)
    target = PD_LAYERWISE_HANDOFF_CONTROLLER.target_for(
        execution.generation
    )

    # Before a request arrives, retain one executing plus one queued layer.
    # Waiting for the oldest event on a helper thread prevents the model thread
    # from submitting the whole model, while the request-owned wake event lets
    # EngineCore interrupt this host wait immediately after Decode becomes
    # empty.  It does not interrupt a CUDA kernel; the target stream is ordered
    # after the latest boundary below.
    if (
        target is None
        and len(execution._boundary_events)
        >= _LAYERWISE_MAX_QUEUED_LAYERS
    ):
        wake_event = execution._boundary_wake_event
        if wake_event is None:
            raise GreenContextError(
                "layerwise P/D SM handoff lost its boundary wake event"
            )
        wake_event.clear()
        target = PD_LAYERWISE_HANDOFF_CONTROLLER.target_for(
            execution.generation
        )
        if target is None:
            oldest_event = execution._boundary_events[0]
            wait_future = execution._boundary_wait_future
            if wait_future is None:
                wait_future = _submit_layer_boundary_wait(
                    oldest_event,
                    wake_event,
                )
                execution._boundary_wait_future = wait_future
            # Close both lost-wakeup windows: Future completion and a handoff
            # request can each race with clear()/callback registration.
            if (
                wait_future.done()
                or PD_LAYERWISE_HANDOFF_CONTROLLER.target_for(
                    execution.generation
                )
                is not None
            ):
                wake_event.set()
            wake_event.wait()
            target = PD_LAYERWISE_HANDOFF_CONTROLLER.target_for(
                execution.generation
            )
            if target is None:
                if not wait_future.done():
                    # Controller reset/transition completion is the only other
                    # legal wake source.  Do not strand this worker during
                    # shutdown or generation replacement.
                    if not PD_LAYERWISE_HANDOFF_CONTROLLER.should_bound_submission(
                        execution.generation
                    ):
                        return
                    raise GreenContextError(
                        "layerwise boundary wait woke without CUDA or control "
                        "progress"
                    )
                wait_future.result()
                execution._boundary_wait_future = None
                execution._boundary_events.popleft()

    if target is None:
        return

    PD_LAYERWISE_HANDOFF_CONTROLLER.note_boundary(
        execution.generation,
        sync_ms=0.0,
    )
    resume_event = PD_LAYERWISE_HANDOFF_CONTROLLER.transition_resume_event_for(
        execution.generation
    )
    if resume_event is None:
        raise GreenContextError(
            "layerwise P/D SM handoff lost its transition resume event"
        )
    if not PD_LAYERWISE_HANDOFF_CONTROLLER.mark_boundary_recorded(
        execution.generation,
        layer_index=int(layer_index),
    ):
        raise GreenContextError(
            "layerwise P/D SM handoff lost controller ownership while "
            "recording the CUDA boundary"
        )

    # Wait only for this transition's CUDA event, off the model thread. While
    # it is pending, EngineCore keeps source-epoch Decode admission open when
    # that immutable profile owns Decode SMs. This is intentionally different
    # from synchronizing the complete GreenContext handle.
    boundary_wait_start = time.perf_counter()
    transition_wait_future = _LAYERWISE_BOUNDARY_WAIT_EXECUTOR.submit(
        boundary_event.synchronize
    )
    transition_wait_future.result()
    boundary_gpu_wait_ms = (
        time.perf_counter() - boundary_wait_start
    ) * 1000.0
    if not PD_LAYERWISE_HANDOFF_CONTROLLER.mark_boundary_completed(
        execution.generation,
        layer_index=int(layer_index),
        gpu_wait_ms=boundary_gpu_wait_ms,
    ):
        raise GreenContextError(
            "layerwise P/D SM handoff lost controller ownership after the "
            "CUDA boundary completed"
        )

    # Boundary completion closes new source Decode admission. EngineCore then
    # drains at most the depth-one source Decode Future, commits the worker
    # profile bank and allocator epoch, and releases this exact event.
    resume_event.wait()
    if not PD_LAYERWISE_HANDOFF_CONTROLLER.committed_transition_matches(
        source_profile_id=execution.effective_profile_id,
        source_epoch=execution.effective_profile_epoch,
        target_profile_id=target.profile_id,
        target_epoch=target.profile_epoch,
    ):
        raise GreenContextError(
            "layerwise P/D SM handoff resumed before the Engine committed "
            f"{execution.effective_profile_id}@"
            f"{execution.effective_profile_epoch} -> "
            f"{target.profile_id}@{target.profile_epoch}"
        )

    streams = _get_precreated_layerwise_target_streams(target.profile_id)
    target_stream = streams.stream_for_phase(_PHASE_PREFILL)
    if target_stream is None:
        raise GreenContextError(
            "layerwise P/D SM handoff target has no Prefill stream: "
            f"{target.profile_id}@{target.profile_epoch}"
        )

    # The event is already complete and EngineCore has explicitly committed the
    # bank, but retain the stream dependency as a local ordering assertion for
    # the resumed hidden state and KV writes.
    target_stream.wait_event(boundary_event)
    # The fixed Prefill arena follows the same Future and the same physical
    # layer boundary.  Updating its logical owner here introduces no new host
    # or device synchronization: target_stream already waits on the producer
    # event above.
    handoff_active_prefill_workspace(
        handoff_generation=execution.generation,
        source_profile_id=execution.effective_profile_id,
        source_profile_epoch=execution.effective_profile_epoch,
        target_profile_id=target.profile_id,
        target_profile_epoch=target.profile_epoch,
    )
    execution._handoff_event = boundary_event
    torch.cuda.set_stream(target_stream)
    execution._profile_tokens.append(
        _ACTIVE_PD_SM_PROFILE.set(
            (target.profile_id, target.profile_epoch)
        )
    )
    execution.effective_profile_id = target.profile_id
    execution.effective_profile_epoch = target.profile_epoch
    execution.applied = True
    execution.applied_count += 1
    execution.applied_layer = int(layer_index)
    if execution._boundary_wait_future is not None:
        execution._boundary_wait_future.result()
        execution._boundary_wait_future = None
    execution._boundary_events.clear()


def green_context_state() -> dict[str, float | int | str]:
    active_profile = _ACTIVE_PD_SM_PROFILE.get()
    active_profile_id = active_profile[0] if active_profile else ""
    active_profile_epoch = active_profile[1] if active_profile else 0
    active_streams = (
        _PROFILE_BANK.peek(active_profile_id)
        if _PROFILE_BANK is not None and active_profile_id
        else _STREAMS
    )
    decode_sms = (
        active_streams.actual_decode_sms
        if active_streams is not None
        else max(0, _env_int(_DECODE_SMS_ENV, 8))
    )
    prefill_sms = (
        active_streams.actual_prefill_sms
        if active_streams is not None
        else max(0, _env_int(_PREFILL_SMS_ENV, 0))
    )
    decode_priority = _env_int(_DECODE_PRIORITY_ENV, -1)
    prefill_priority = _env_int(_PREFILL_PRIORITY_ENV, 0)
    return {
        "green_context_enabled": float(green_context_pd_enabled()),
        "green_context_active": float(
            active_streams is not None
            and not _INITIALIZATION_FAILED
            and not _PROFILE_BANK_INITIALIZATION_FAILED
        ),
        "green_context_fallback": float(
            _INITIALIZATION_FAILED or _PROFILE_BANK_INITIALIZATION_FAILED
        ),
        "green_context_strict": float(_green_context_strict()),
        "green_decode_sms": decode_sms,
        "green_prefill_sms": prefill_sms,
        "green_sm_count_verified": float(
            active_streams is not None and active_streams.sm_count_verified
        ),
        "green_stream_priority_decode": decode_priority,
        "green_stream_priority_prefill": prefill_priority,
        "green_execute_model_enabled": float(green_context_execute_model_enabled()),
        "green_execute_model_no_join": float(
            green_context_execute_model_no_join_enabled()
        ),
        "pd_dynamic_sm_green_context_enabled": float(
            green_context_dynamic_sm_enabled()
        ),
        "pd_dynamic_sm_green_context_profile_id": active_profile_id,
        "pd_dynamic_sm_green_context_profile_epoch": float(active_profile_epoch),
        "pd_dynamic_sm_green_context_created_profiles": float(
            _PROFILE_BANK.created_count if _PROFILE_BANK is not None else 0
        ),
        "pd_dynamic_sm_green_context_failed_profiles": float(
            _PROFILE_BANK.failed_count if _PROFILE_BANK is not None else 0
        ),
        "pd_dynamic_sm_green_context_all_sm_counts_verified": float(
            _PROFILE_BANK.all_sm_counts_verified
            if _PROFILE_BANK is not None
            else False
        ),
        "pd_dynamic_sm_green_context_bank_profile_id": (
            _PROFILE_BANK.active_profile_id if _PROFILE_BANK is not None else ""
        ),
        "pd_dynamic_sm_green_context_bank_profile_epoch": float(
            _PROFILE_BANK.active_epoch if _PROFILE_BANK is not None else 0
        ),
        "pd_dynamic_sm_green_context_handoff_count": float(
            _PROFILE_BANK.handoff_count if _PROFILE_BANK is not None else 0
        ),
        "pd_dynamic_sm_green_context_handoff_sync_ms_total": (
            _PROFILE_BANK.handoff_sync_ms_total
            if _PROFILE_BANK is not None
            else 0.0
        ),
        "pd_dynamic_sm_green_context_last_handoff_sync_ms": (
            _PROFILE_BANK.last_handoff_sync_ms
            if _PROFILE_BANK is not None
            else 0.0
        ),
        "pd_dynamic_sm_green_context_layerwise_commit_count": float(
            _PROFILE_BANK.layerwise_commit_count
            if _PROFILE_BANK is not None
            else 0
        ),
    }


@contextmanager
def green_context_phase_stream(
    phase: str | None,
    *,
    wait_for_completion: bool = True,
    wait_for_current_stream: bool = True,
    profile_id: str | None = None,
    profile_epoch: int = 0,
) -> Iterator[bool]:
    """Route CUDA work in this scope to the requested P/D GreenContext stream.

    `wait_for_completion=False` is the experimental execute_model path used by
    the phase-lane scheduler. It submits work to the phase stream without
    inserting a default-stream join at scope exit, so decode and prefill streams
    can make independent forward progress. Consumers must then run on the same
    phase stream or insert their own dependency.
    """
    if not green_context_execute_model_enabled():
        yield False
        return

    driver_phase = _phase_to_driver_id(phase)
    if driver_phase is None:
        yield False
        return

    streams = _get_streams(profile_id, profile_epoch)
    if streams is None:
        yield False
        return

    stream = streams.stream_for_phase(driver_phase)
    if stream is None:
        if profile_id and green_context_dynamic_sm_enabled():
            message = (
                f"dynamic P/D SM profile {profile_id!r} has no stream for "
                f"phase {phase!r}"
            )
            raise GreenContextError(message)
        yield False
        return

    current = _current_torch_stream()
    if wait_for_current_stream:
        stream.wait_stream(current)
    profile_token = _ACTIVE_PD_SM_PROFILE.set(
        (streams.profile_id if profile_id else "", int(profile_epoch))
    )
    try:
        with torch.cuda.stream(stream):
            yield True
    finally:
        try:
            if wait_for_completion:
                current.wait_stream(stream)
        finally:
            _ACTIVE_PD_SM_PROFILE.reset(profile_token)


def run_on_decode_green_stream(fn: Callable[[], T]) -> T:
    streams = _get_streams()
    if streams is None:
        return fn()
    return streams.run_on_phase_stream(_PHASE_DECODE, fn)


def run_on_prefill_green_stream(fn: Callable[[], T]) -> T:
    streams = _get_streams()
    if streams is None:
        return fn()
    return streams.run_on_phase_stream(_PHASE_PREFILL, fn)


def run_decode_prefill_on_green_streams(
    decode_fn: Callable[[], T],
    prefill_fn: Callable[[], T],
    *,
    overlap: bool,
    collect_timing: bool,
) -> GreenPairResult:
    streams = _get_streams()
    if streams is None:
        decode_result = decode_fn()
        prefill_result = prefill_fn()
        return GreenPairResult(
            decode_result,
            prefill_result,
            {"fallback_reason": "green_context_unavailable"},
        )
    return streams.run_decode_prefill(
        decode_fn,
        prefill_fn,
        overlap=overlap,
        collect_timing=collect_timing,
    )
