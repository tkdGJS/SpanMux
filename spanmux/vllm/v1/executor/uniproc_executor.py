# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cached_property
from multiprocessing import Lock
from threading import Lock as ThreadLock
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.ttft_layerwise_prefill import (
    KVRetentionExecutionBinding,
    KVRetentionPhaseOwner,
    TTFTLayerwisePrefillBoundary,
)
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.core.tiered_errors import TieredBackendEligibilityError
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.engine.pd_engine_progress import (
    PDAsyncFutureLifecycle,
    attach_pd_future_lifecycle,
)
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.pd_host_enqueue import (
    DecodeFirstHostEnqueueGate,
    PDHostEnqueueTicket,
)
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)

_ENV_TRUE = ("1", "true", "yes", "on")


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in _ENV_TRUE


def _pd_parse_execution_key(value: str | None) -> tuple[str | None, int]:
    if value is None:
        return None, 0
    phase, separator, raw_slot = value.partition(":")
    if phase not in {"decode", "prefill"}:
        return value, 0
    if not separator:
        return phase, 0
    try:
        return phase, max(0, int(raw_slot))
    except ValueError:
        return phase, 0


def _pd_execution_key(phase: str | None, slot: int) -> str | None:
    base_phase, encoded_slot = _pd_parse_execution_key(phase)
    if base_phase not in {"decode", "prefill"}:
        return phase
    selected_slot = max(encoded_slot, max(0, int(slot)))
    return base_phase if selected_slot == 0 else f"{base_phase}:{selected_slot}"


class UniProcExecutor(Executor):
    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        self.driver_worker = WorkerWrapperBase(rpc_rank=0)
        distributed_init_method, rank, local_rank = self._distributed_args()
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=True,
            shared_worker_lock=Lock(),
        )

        self.async_output_thread: ThreadPoolExecutor | None = None
        if self.max_concurrent_batches > 1:
            self.async_output_thread = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="WorkerAsyncOutput"
            )
        self.pd_async_uniproc_enabled = _env_flag(
            "VLLM_PD_ASYNC_UNIPROC_EXECUTOR"
        )
        self.pd_async_uniproc_executor: ThreadPoolExecutor | None = None
        self.pd_async_uniproc_phase_executors: dict[
            str, ThreadPoolExecutor
        ] = {}
        self.pd_async_uniproc_output_executor: ThreadPoolExecutor | None = None
        self.pd_async_uniproc_phase_output_executors: dict[
            str, ThreadPoolExecutor
        ] = {}
        self.pd_async_uniproc_phase_workers_enabled = False
        self.pd_async_uniproc_concurrent_phase_submit_enabled = False
        self.pd_decode_first_host_enqueue_enabled = False
        self.pd_future_liveness_audit_enabled = _env_flag(
            "VLLM_PD_FUTURE_LIVENESS_AUDIT"
        )
        lightweight_timeline = _env_flag("VLLM_LIGHTWEIGHT_TIMELINE_TRACE")
        self.pd_async_executor_debug_enabled = bool(
            _env_flag("VLLM_PD_EXECUTOR_DEBUG")
            or _env_flag("VLLM_FULL_ITERATION_DEBUG")
            or (
                os.environ.get("VLLM_ITERATION_METRICS_JSONL")
                and not lightweight_timeline
            )
        )
        try:
            self.pd_async_executor_debug_sample_every_n = max(
                1,
                int(
                    os.environ.get(
                        "VLLM_ITERATION_METRICS_SAMPLE_EVERY_N", "100"
                    )
                ),
            )
        except ValueError:
            self.pd_async_executor_debug_sample_every_n = 100
        self._pd_host_enqueue_gate = DecodeFirstHostEnqueueGate()
        self.pd_async_worker_submit_lock = ThreadLock()
        self.pd_async_phase_submit_locks = {
            "decode": ThreadLock(),
            "prefill": ThreadLock(),
        }
        self._pd_async_phase_execute_futures: dict[str, Future[Any]] = {}
        self._pd_async_phase_execute_owners: dict[str, KVRetentionPhaseOwner] = {}
        if self.pd_async_uniproc_enabled:
            phase_local_enabled = _env_flag(
                "VLLM_PD_PHASE_LOCAL_MODEL_RUNNER"
            ) or _env_flag("VLLM_PD_MUXWISE_EVENT_LOOP")
            single_inflight_enabled = _env_flag(
                "VLLM_PD_SINGLE_INFLIGHT_PER_PHASE"
            ) or _env_flag("VLLM_PD_MUXWISE_EVENT_LOOP")
            phase_depth_enabled = _env_flag("VLLM_PD_PHASE_DEPTH_ENABLED")
            if not phase_local_enabled:
                logger.warning(
                    "Disabling VLLM_PD_ASYNC_UNIPROC_EXECUTOR because "
                    "VLLM_PD_PHASE_LOCAL_MODEL_RUNNER is not active."
                )
                self.pd_async_uniproc_enabled = False
            elif not single_inflight_enabled and not phase_depth_enabled:
                logger.warning(
                    "Disabling VLLM_PD_ASYNC_UNIPROC_EXECUTOR because "
                    "VLLM_PD_SINGLE_INFLIGHT_PER_PHASE is not active."
                )
                self.pd_async_uniproc_enabled = False
            else:
                try:
                    max_workers = int(
                        os.environ.get("VLLM_PD_ASYNC_UNIPROC_WORKERS", "2")
                    )
                except ValueError:
                    max_workers = 2
                max_workers = max(2, max_workers)
                self.pd_async_uniproc_phase_workers_enabled = _env_flag(
                    "VLLM_PD_ASYNC_UNIPROC_PHASE_WORKERS", "1"
                )
                self.pd_async_uniproc_concurrent_phase_submit_enabled = _env_flag(
                    "VLLM_PD_ASYNC_UNIPROC_CONCURRENT_PHASE_SUBMIT"
                )
                self.pd_decode_first_host_enqueue_enabled = _env_flag(
                    "VLLM_PD_DECODE_FIRST_HOST_ENQUEUE"
                )
                self.pd_async_uniproc_executor = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="PDAsyncUniProc",
                )
                # AsyncModelRunnerOutput.get_output() can wait for a CUDA to
                # CPU copy event. Keep that wait off the ordered phase submit
                # executors; otherwise a single Decode result prevents the
                # next host enqueue even though the phase CUDA stream can
                # accept more work.
                self.pd_async_uniproc_output_executor = ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="PDAsyncUniProcOutput",
                )
                if self.pd_async_uniproc_phase_workers_enabled:
                    try:
                        phase_workers = int(
                            os.environ.get(
                                "VLLM_PD_ASYNC_UNIPROC_WORKERS_PER_PHASE", "1"
                            )
                        )
                    except ValueError:
                        phase_workers = 1
                    phase_workers = max(1, phase_workers)
                    if phase_depth_enabled and phase_workers != 1:
                        logger.warning(
                            "Forcing one Python worker per phase for P/D phase "
                            "depth; slot-local buffers permit queued submissions "
                            "but shared model state still requires ordered host "
                            "entry."
                        )
                        phase_workers = 1
                    self.pd_async_uniproc_phase_executors = {
                        "decode": ThreadPoolExecutor(
                            max_workers=phase_workers,
                            thread_name_prefix="PDAsyncUniProcDecode",
                        ),
                        "prefill": ThreadPoolExecutor(
                            max_workers=phase_workers,
                            thread_name_prefix="PDAsyncUniProcPrefill",
                        ),
                    }
                    self.pd_async_uniproc_phase_output_executors = {
                        "decode": ThreadPoolExecutor(
                            max_workers=1,
                            thread_name_prefix="PDAsyncUniProcDecodeOutput",
                        ),
                        "prefill": ThreadPoolExecutor(
                            max_workers=1,
                            thread_name_prefix="PDAsyncUniProcPrefillOutput",
                        ),
                    }
                if self.pd_async_uniproc_concurrent_phase_submit_enabled:
                    logger.warning(
                        "VLLM_PD_ASYNC_UNIPROC_CONCURRENT_PHASE_SUBMIT=1 is "
                        "experimental. It allows decode and prefill worker "
                        "submit paths to overlap using phase-local locks. Use "
                        "phase queue and handoff traces to validate correctness."
                    )
                if self.pd_decode_first_host_enqueue_enabled:
                    logger.info(
                        "Decode-first phase admission is enabled. Decode and "
                        "Prefill execute_model host bodies use independent "
                        "lanes; only same-phase entry remains ordered."
                    )
                if _env_flag("VLLM_ITERATION_GPU_TIMING"):
                    logger.warning(
                        "VLLM_PD_ASYNC_UNIPROC_EXECUTOR is active while "
                        "VLLM_ITERATION_GPU_TIMING=1. GPU timing synchronizes "
                        "the forward path and can hide P/D stream overlap."
                    )

        self.driver_worker.init_worker(all_kwargs=[kwargs])
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def _distributed_args(self) -> tuple[str, int, int]:
        """Return (distributed_init_method, rank, local_rank)."""
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
        # set local rank as the device index if specified
        device_info = self.vllm_config.device_config.device.__str__().split(":")
        local_rank = int(device_info[1]) if len(device_info) > 1 else 0
        return distributed_init_method, 0, local_rank

    @cached_property
    def max_concurrent_batches(self) -> int:
        if not self.scheduler_config.async_scheduling:
            return 1
        if not _env_flag("VLLM_PD_PHASE_DEPTH_ENABLED"):
            return 2
        try:
            requested = int(os.environ.get("VLLM_PD_GLOBAL_INFLIGHT_DEPTH", "2"))
        except ValueError:
            requested = 2
        return max(2, requested)

    def _run_pd_async_rpc(
        self,
        method: str | Callable,
        args: tuple,
        kwargs: dict,
        phase: str | None,
        host_enqueue_ticket: PDHostEnqueueTicket,
        task_submit_ts: float,
        collect_debug: bool,
        lifecycle: PDAsyncFutureLifecycle | None,
    ) -> tuple[Any, dict[str, float | str]]:
        # Phase-local model-runner contexts isolate mutable execution buffers.
        # The decode-first gate therefore orders each phase independently and
        # does not hold a global lock across a long Ring Prefill host body.
        if lifecycle is not None:
            lifecycle.advance("worker_thread_started")
        lock = self._pd_async_submit_lock_for_phase(phase)
        task_start_ts = time.monotonic() if collect_debug else 0.0
        lock_wait_start = task_start_ts
        if lifecycle is not None:
            lifecycle.advance("host_admission_wait")
        try:
            if host_enqueue_ticket.ordered:
                with self._pd_host_enqueue_gate.enter(host_enqueue_ticket):
                    if lifecycle is not None:
                        lifecycle.advance("worker_body_started")
                    lock_acquired_ts = (
                        time.monotonic() if collect_debug else 0.0
                    )
                    result = run_method(
                        self.driver_worker, method, args, kwargs
                    )
            else:
                with lock:
                    if lifecycle is not None:
                        lifecycle.advance("worker_body_started")
                    lock_acquired_ts = (
                        time.monotonic() if collect_debug else 0.0
                    )
                    result = run_method(
                        self.driver_worker, method, args, kwargs
                    )
        except BaseException:
            if lifecycle is not None:
                lifecycle.advance("worker_body_error")
            raise
        if lifecycle is not None:
            lifecycle.advance("worker_body_returned")
        worker_call_end_ts = time.monotonic() if collect_debug else 0.0
        if not collect_debug:
            return result, {}
        (
            decode_host_active,
            prefill_host_active,
            pending_decode,
        ) = self._pd_host_enqueue_gate.phase_snapshot()
        rpc_method = self._pd_rpc_method_name(method) or "rpc"
        rpc_metric_prefix = (
            rpc_method
            if rpc_method
            in {
                "execute_model",
                "execute_model_and_sample",
                "sample_tokens",
                "execute_model_pd_pair",
            }
            else "rpc"
        )
        task_queue_wait_s = max(0.0, task_start_ts - task_submit_ts)
        debug = {
            "pd_async_uniproc_executor_enabled": 1.0,
            "pd_async_uniproc_phase_workers_enabled": float(
                self.pd_async_uniproc_phase_workers_enabled
            ),
            "pd_async_uniproc_concurrent_phase_submit_enabled": float(
                self.pd_async_uniproc_concurrent_phase_submit_enabled
            ),
            "pd_async_uniproc_rpc_phase": phase or "",
            "pd_async_uniproc_submit_lock_mode": (
                "decode_first_phase_admission"
                if host_enqueue_ticket.ordered
                else "phase"
                if self._pd_async_uses_phase_submit_lock(phase)
                else "global"
            ),
            "pd_async_uniproc_submit_lock_wait_s": (
                lock_acquired_ts - lock_wait_start
            ),
            "pd_async_uniproc_rpc_method": rpc_method,
            "pd_async_uniproc_task_submit_ts": task_submit_ts,
            "pd_async_uniproc_task_start_ts": task_start_ts,
            "pd_async_uniproc_task_queue_wait_s": task_queue_wait_s,
            "pd_async_uniproc_submit_attempt_ts": task_start_ts,
            "pd_async_uniproc_worker_call_start_ts": lock_acquired_ts,
            "pd_async_uniproc_worker_call_end_ts": worker_call_end_ts,
            "pd_async_uniproc_worker_call_duration_s": (
                worker_call_end_ts - lock_acquired_ts
            ),
            "pd_async_uniproc_output_offload_enabled": 1.0,
            "pd_async_uniproc_output_wait_offloaded": 0.0,
            "pd_async_uniproc_output_wait_start_ts": 0.0,
            "pd_async_uniproc_output_wait_end_ts": 0.0,
            "pd_async_uniproc_output_wait_duration_s": 0.0,
            "pd_decode_first_host_enqueue_enabled": float(
                self.pd_decode_first_host_enqueue_enabled
            ),
            "pd_decode_first_host_enqueue_applied": float(
                host_enqueue_ticket.ordered
            ),
            "pd_decode_pending_host_enqueues_after_call": float(
                pending_decode
            ),
            "pd_decode_host_lane_active_after_call": float(
                decode_host_active
            ),
            "pd_prefill_host_lane_active_after_call": float(
                prefill_host_active
            ),
        }
        debug.update(
            {
                f"pd_async_{rpc_metric_prefix}_task_submit_ts": task_submit_ts,
                f"pd_async_{rpc_metric_prefix}_task_start_ts": task_start_ts,
                f"pd_async_{rpc_metric_prefix}_task_queue_wait_s": (
                    task_queue_wait_s
                ),
                f"pd_async_{rpc_metric_prefix}_lock_wait_s": (
                    lock_acquired_ts - lock_wait_start
                ),
                f"pd_async_{rpc_metric_prefix}_worker_call_start_ts": (
                    lock_acquired_ts
                ),
                f"pd_async_{rpc_metric_prefix}_worker_call_end_ts": (
                    worker_call_end_ts
                ),
                f"pd_async_{rpc_metric_prefix}_worker_call_duration_s": (
                    worker_call_end_ts - lock_acquired_ts
                ),
            }
        )
        return result, debug

    def _resolve_pd_async_rpc_result(
        self,
        result: Any,
        debug: dict[str, float | str],
        args: tuple,
        single_value: bool,
        phase: str | None,
        lifecycle: PDAsyncFutureLifecycle | None,
    ) -> Any:
        """Resolve an async device-to-host result outside the phase worker."""
        if isinstance(result, AsyncModelRunnerOutput):
            output_wait_start = (
                time.monotonic() if debug else 0.0
            )
            if lifecycle is not None:
                lifecycle.advance("output_wait_started")
            try:
                result = result.get_output()
            except BaseException:
                if lifecycle is not None:
                    lifecycle.advance("output_wait_error")
                raise
            if lifecycle is not None:
                lifecycle.advance("output_wait_finished")
            if debug:
                output_wait_end = time.monotonic()
                debug.update({
                    "pd_async_uniproc_output_wait_start_ts": output_wait_start,
                    "pd_async_uniproc_output_wait_end_ts": output_wait_end,
                    "pd_async_uniproc_output_wait_duration_s": (
                        output_wait_end - output_wait_start
                    ),
                    "pd_async_uniproc_output_wait_offloaded": 1.0,
                    "pd_async_uniproc_output_executor_phase": (
                        _pd_parse_execution_key(phase)[0] or "global"
                    ),
                })
        if debug:
            self._record_pd_async_debug(args, result, debug)
        if lifecycle is not None:
            lifecycle.advance("result_resolved")
        return result if single_value else [result]

    def _submit_pd_async_rpc(
        self,
        method: str | Callable,
        args: tuple,
        kwargs: dict,
        single_value: bool,
        dependency: Future[Any] | None = None,
        phase: str | None = None,
    ) -> Future[Any]:
        async_executor = self._pd_async_executor_for_phase(phase)
        assert async_executor is not None
        chained_future: Future[Any] = Future()
        lifecycle = (
            attach_pd_future_lifecycle(
                chained_future,
                method=self._pd_rpc_method_name(method) or "rpc",
                execution_key=phase,
            )
            if self.pd_future_liveness_audit_enabled
            else None
        )

        def copy_completion(completion: Future[Any]) -> None:
            if chained_future.cancelled():
                return
            if lifecycle is not None:
                lifecycle.advance("output_completion_callback")
            try:
                chained_future.set_result(completion.result())
            except Exception as exc:
                if lifecycle is not None:
                    lifecycle.advance("output_completion_error")
                chained_future.set_exception(exc)

        def finish_rpc(inner: Future[Any]) -> None:
            if chained_future.cancelled():
                return
            if lifecycle is not None:
                lifecycle.advance("worker_completion_callback")
            try:
                result, debug = inner.result()
                if isinstance(result, AsyncModelRunnerOutput):
                    output_executor = self._pd_async_output_executor_for_phase(
                        phase
                    )
                    assert output_executor is not None
                    if lifecycle is not None:
                        lifecycle.advance("output_wait_queued")
                    completion = output_executor.submit(
                        self._resolve_pd_async_rpc_result,
                        result,
                        debug,
                        args,
                        single_value,
                        phase,
                        lifecycle,
                    )
                    completion.add_done_callback(copy_completion)
                else:
                    chained_future.set_result(
                        self._resolve_pd_async_rpc_result(
                            result,
                            debug,
                            args,
                            single_value,
                            phase,
                            lifecycle,
                        )
                    )
            except Exception as exc:
                if lifecycle is not None:
                    lifecycle.advance("worker_completion_error")
                chained_future.set_exception(exc)

        def submit_rpc() -> None:
            base_phase, _ = _pd_parse_execution_key(phase)
            ordered = (
                self.pd_decode_first_host_enqueue_enabled
                and self.pd_async_uniproc_concurrent_phase_submit_enabled
                and self._pd_rpc_method_name(method)
                in {"execute_model", "execute_model_and_sample"}
                and base_phase in {"decode", "prefill"}
            )
            host_enqueue_ticket = self._pd_host_enqueue_gate.register(
                base_phase,
                ordered=ordered,
            )
            if lifecycle is not None:
                lifecycle.advance("host_admission_registered")
            collect_debug = self._pd_async_should_collect_debug(args)
            task_submit_ts = time.monotonic() if collect_debug else 0.0
            try:
                if lifecycle is not None:
                    lifecycle.advance("worker_queued")
                inner_future = async_executor.submit(
                    self._run_pd_async_rpc,
                    method,
                    args,
                    kwargs,
                    phase,
                    host_enqueue_ticket,
                    task_submit_ts,
                    collect_debug,
                    lifecycle,
                )
                inner_future.add_done_callback(finish_rpc)
            except Exception as exc:
                self._pd_host_enqueue_gate.cancel(host_enqueue_ticket)
                if lifecycle is not None:
                    lifecycle.advance("worker_submit_error")
                if not chained_future.cancelled():
                    chained_future.set_exception(exc)

        def submit_after_dependency(dep: Future[Any]) -> None:
            if chained_future.cancelled():
                return
            try:
                dep.result()
            except Exception as exc:
                if lifecycle is not None:
                    lifecycle.advance("dependency_error")
                chained_future.set_exception(exc)
                return
            if lifecycle is not None:
                lifecycle.advance("dependency_ready")
            submit_rpc()

        if dependency is None:
            submit_rpc()
        elif dependency.done():
            submit_after_dependency(dependency)
        else:
            if lifecycle is not None:
                lifecycle.advance("dependency_wait")
            dependency.add_done_callback(submit_after_dependency)
        return chained_future

    def _pd_async_executor_for_phase(
        self, phase: str | None
    ) -> ThreadPoolExecutor | None:
        phase, _ = _pd_parse_execution_key(phase)
        if (
            self.pd_async_uniproc_phase_workers_enabled
            and phase in self.pd_async_uniproc_phase_executors
        ):
            return self.pd_async_uniproc_phase_executors[phase]
        return self.pd_async_uniproc_executor

    def _pd_async_output_executor_for_phase(
        self, phase: str | None
    ) -> ThreadPoolExecutor | None:
        base_phase, _ = _pd_parse_execution_key(phase)
        return self.pd_async_uniproc_phase_output_executors.get(
            base_phase or "", self.pd_async_uniproc_output_executor
        )

    def _pd_async_uses_phase_submit_lock(self, phase: str | None) -> bool:
        phase, _ = _pd_parse_execution_key(phase)
        return (
            self.pd_async_uniproc_concurrent_phase_submit_enabled
            and phase in self.pd_async_phase_submit_locks
        )

    def _pd_async_submit_lock_for_phase(self, phase: str | None) -> ThreadLock:
        if self._pd_async_uses_phase_submit_lock(phase):
            base_phase, _ = _pd_parse_execution_key(phase)
            return self.pd_async_phase_submit_locks[base_phase]  # type: ignore[index]
        return self.pd_async_worker_submit_lock

    @staticmethod
    def _pd_rpc_method_name(method: str | Callable) -> str:
        return method if isinstance(method, str) else getattr(method, "__name__", "")

    def _pd_async_should_collect_debug(self, args: tuple) -> bool:
        if not self.pd_async_executor_debug_enabled:
            return False
        for arg in args[:2]:
            if not isinstance(arg, SchedulerOutput):
                continue
            iteration_id = int(getattr(arg, "scheduler_iteration_id", -1))
            if iteration_id < 0:
                return True
            return (
                iteration_id % self.pd_async_executor_debug_sample_every_n == 0
            )
        return True

    @staticmethod
    def _record_pd_async_debug(
        args: tuple,
        result: Any,
        debug: dict[str, float | str],
    ) -> None:
        debug_payload = dict(debug)
        if isinstance(result, dict):
            debug_payload.update(result)
        if args and isinstance(args[0], SchedulerOutput):
            if args[0].iteration_debug is None:
                args[0].iteration_debug = {}
            args[0].iteration_debug.update(debug_payload)
        if len(args) > 1 and isinstance(args[1], SchedulerOutput):
            if args[1].iteration_debug is None:
                args[1].iteration_debug = {}
            args[1].iteration_debug.update(debug_payload)
        if isinstance(result, ModelRunnerOutput | TTFTLayerwisePrefillBoundary):
            result.model_runner_debug_timings.update(debug_payload)

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
        dependency: Future[Any] | None = None,
        pd_green_context_phase: str | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        if not non_block:
            result = run_method(self.driver_worker, method, args, kwargs)
            return result if single_value else [result]

        if self.pd_async_uniproc_enabled:
            return self._submit_pd_async_rpc(
                method,
                args,
                kwargs,
                single_value,
                dependency,
                pd_green_context_phase,
            )

        try:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                if (async_thread := self.async_output_thread) is not None:
                    if single_value:
                        return async_thread.submit(result.get_output)

                    def get_output_list() -> list[Any]:
                        return [result.get_output()]

                    return async_thread.submit(get_output_list)
                result = result.get_output()
            future = Future[Any]()
            future.set_result(result if single_value else [result])
        except Exception as e:
            future = Future[Any]()
            future.set_exception(e)
        return future

    def execute_model(  # type: ignore[override]
        self,
        scheduler_output: SchedulerOutput,
        non_block: bool = False,
        pd_green_context_phase: str | None = None,
        kv_retention_binding: KVRetentionExecutionBinding | None = None,
    ) -> (
        ModelRunnerOutput
        | TTFTLayerwisePrefillBoundary
        | None
        | Future[ModelRunnerOutput | TTFTLayerwisePrefillBoundary | None]
    ):
        execution_key = _pd_execution_key(
            pd_green_context_phase,
            getattr(scheduler_output, "pd_phase_slot", 0),
        )
        base_phase, _ = _pd_parse_execution_key(execution_key)
        execute_owner = None
        if (
            self.pd_async_uniproc_enabled
            and non_block
            and base_phase in {"decode", "prefill"}
        ):
            assert execution_key is not None
            execute_owner = KVRetentionPhaseOwner.from_output(
                execution_key, scheduler_output, kv_retention_binding
            )
        binding_kwargs = {}
        if kv_retention_binding is not None:
            if execute_owner is None:
                raise RuntimeError("Prefill binding requires asynchronous phase execution")
            kv_retention_binding.validate_submission(execution_key, scheduler_output)
            binding_kwargs["kv_retention_binding"] = kv_retention_binding
        future_or_output = self.collective_rpc(
            "execute_model",
            args=(scheduler_output, execution_key),
            kwargs=binding_kwargs,
            non_block=non_block,
            single_value=True,
            pd_green_context_phase=execution_key,
        )
        if (
            self.pd_async_uniproc_enabled
            and non_block
            and base_phase in {"decode", "prefill"}
            and isinstance(future_or_output, Future)
        ):
            assert execution_key is not None
            assert execute_owner is not None
            self._pd_async_phase_execute_futures[execution_key] = future_or_output
            self._pd_async_phase_execute_owners[execution_key] = execute_owner
            if kv_retention_binding is not None:
                future_or_output.kv_retention_binding = kv_retention_binding
        return future_or_output

    def clear_pd_phase_execute_state(
        self, execution_key: str, expected_owner: KVRetentionPhaseOwner
    ) -> bool:
        """Clear a phase slot through its ordered phase worker.

        A yielded Prefill continuation is owned by the Prefill executor, not
        by EngineCore's caller thread.  Queue cleanup on that same executor so
        it cannot race another Prefill RPC that is using the shared model
        object. Remove only the captured dependency after a matching ACK;
        a failed or stale cleanup must not drop a newer execute dependency.
        """

        expected_owner.validate_key(execution_key)
        execution_key = expected_owner.execution_key
        dependency = self._pd_async_phase_execute_futures.get(execution_key)
        dependency_owner = self._pd_async_phase_execute_owners.get(execution_key)
        result = self.collective_rpc(
            "clear_pd_phase_execute_state",
            args=(execution_key, expected_owner),
            non_block=self.pd_async_uniproc_enabled,
            single_value=True,
            pd_green_context_phase=execution_key,
        )
        if isinstance(result, Future):
            result = result.result()
        if not isinstance(result, bool):
            raise RuntimeError(
                "phase-local execute-state cleanup returned an unexpected "
                f"result: key={execution_key!r}, result={result!r}"
            )
        if (
            result
            and dependency_owner == expected_owner
            and self._pd_async_phase_execute_owners.get(execution_key) == expected_owner
            and self._pd_async_phase_execute_futures.get(execution_key) is dependency
        ):
            self._pd_async_phase_execute_futures.pop(execution_key, None)
            self._pd_async_phase_execute_owners.pop(execution_key, None)
        return result

    def offload_parked_prefill(self, expected_owner: KVRetentionPhaseOwner):
        """A normal Future on the ordered Prefill worker; Engine never waits inline."""
        expected_owner.validate_key("prefill")
        return self.collective_rpc(
            "offload_parked_prefill", args=(expected_owner,), non_block=True,
            single_value=True, pd_green_context_phase="prefill")

    def release_parked_execute_dependency(self, expected_owner: KVRetentionPhaseOwner) -> None:
        """Compare/remove an already recovered export, without closing its frame."""
        key = expected_owner.execution_key
        expected_owner.validate_key(key)
        owner = self._pd_async_phase_execute_owners.get(key)
        dependency = self._pd_async_phase_execute_futures.get(key)
        if owner != expected_owner or dependency is None or not dependency.done():
            raise RuntimeError("park ACK does not own a completed execute dependency")
        self._pd_async_phase_execute_owners.pop(key)
        self._pd_async_phase_execute_futures.pop(key)

    def execute_model_pd_pair(
        self,
        decode_scheduler_output: SchedulerOutput,
        prefill_scheduler_output: SchedulerOutput,
        non_block: bool = False,
    ) -> Any:
        """Submit paired decode/prefill execute_model in one worker RPC.

        The worker enqueues decode and prefill forward work onto their
        GreenContext streams before either phase samples. This avoids two
        Python worker threads concurrently entering the same model object while
        still allowing device-side stream overlap.
        """
        decode_execution_key = _pd_execution_key(
            "decode", getattr(decode_scheduler_output, "pd_phase_slot", 0)
        )
        prefill_execution_key = _pd_execution_key(
            "prefill", getattr(prefill_scheduler_output, "pd_phase_slot", 0)
        )
        assert decode_execution_key is not None
        assert prefill_execution_key is not None
        decode_owner = KVRetentionPhaseOwner.from_output(
            decode_execution_key, decode_scheduler_output
        )
        prefill_owner = KVRetentionPhaseOwner.from_output(
            prefill_execution_key, prefill_scheduler_output
        )
        future_or_output = self.collective_rpc(
            "execute_model_pd_pair",
            args=(decode_scheduler_output, prefill_scheduler_output),
            non_block=non_block,
            single_value=True,
            pd_green_context_phase="pd_pair",
        )
        if self.pd_async_uniproc_enabled and non_block and isinstance(
            future_or_output, Future
        ):
            self._pd_async_phase_execute_futures[decode_execution_key] = (
                self._pd_pair_phase_dependency(future_or_output, "decode")
            )
            self._pd_async_phase_execute_futures[prefill_execution_key] = (
                self._pd_pair_phase_dependency(future_or_output, "prefill")
            )
            self._pd_async_phase_execute_owners[decode_execution_key] = decode_owner
            self._pd_async_phase_execute_owners[prefill_execution_key] = prefill_owner
        return future_or_output

    def execute_model_and_sample(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: GrammarOutput | None,
        non_block: bool = False,
        pd_green_context_phase: str | None = None,
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        """Run execute/sample as one persistent phase-worker RPC."""

        execution_key = _pd_execution_key(
            pd_green_context_phase,
            getattr(scheduler_output, "pd_phase_slot", 0),
        )
        return self.collective_rpc(
            "execute_model_and_sample",
            args=(scheduler_output, grammar_output, execution_key),
            non_block=non_block,
            single_value=True,
            pd_green_context_phase=execution_key,
        )

    def sample_tokens(  # type: ignore[override]
        self,
        grammar_output: GrammarOutput | None,
        non_block: bool = False,
        pd_green_context_phase: str | None = None,
        kv_retention_binding: KVRetentionExecutionBinding | None = None,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        dependency = None
        execution_key = _pd_execution_key(pd_green_context_phase, 0)
        base_phase, _ = _pd_parse_execution_key(execution_key)
        if (
            self.pd_async_uniproc_enabled
            and non_block
            and base_phase in {"decode", "prefill"}
        ):
            assert execution_key is not None
            owner = self._pd_async_phase_execute_owners.get(execution_key)
            if (getattr(owner, "binding", None) != kv_retention_binding):
                raise RuntimeError("sampling binding does not own execute dependency")
            dependency = self._pd_async_phase_execute_futures.pop(
                execution_key, None
            )
            self._pd_async_phase_execute_owners.pop(execution_key, None)
        future_or_output = self.collective_rpc(
            "sample_tokens",
            args=(grammar_output, execution_key),
            kwargs=({"kv_retention_binding": kv_retention_binding}
                    if kv_retention_binding is not None else {}),
            non_block=non_block,
            single_value=True,
            dependency=dependency,
            pd_green_context_phase=execution_key,
        )
        if kv_retention_binding is not None and isinstance(future_or_output, Future):
            future_or_output.kv_retention_binding = kv_retention_binding
        return future_or_output

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.collective_rpc("take_draft_token_ids", single_value=True)

    def check_health(self) -> None:
        # UniProcExecutor will always be healthy as long as
        # it's running.
        return

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        self.driver_worker.reinitialize_distributed(reconfig_request)
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()

    def shutdown(self) -> None:
        if self.pd_async_uniproc_executor is not None:
            self.pd_async_uniproc_executor.shutdown(
                wait=True, cancel_futures=True
            )
        for executor in self.pd_async_uniproc_phase_executors.values():
            executor.shutdown(wait=True, cancel_futures=True)
        if self.pd_async_uniproc_output_executor is not None:
            self.pd_async_uniproc_output_executor.shutdown(
                wait=True, cancel_futures=True
            )
        for executor in self.pd_async_uniproc_phase_output_executors.values():
            executor.shutdown(wait=True, cancel_futures=True)
        if worker := self.driver_worker:
            worker.shutdown()

    @staticmethod
    def _pd_pair_phase_dependency(
        pair_future: Future[Any],
        phase: str,
    ) -> Future[Any]:
        """Project one paired execute future into a phase-local dependency."""
        phase_future: Future[Any] = Future()

        def finish(shared: Future[Any]) -> None:
            if phase_future.cancelled():
                return
            try:
                result = shared.result()
                error = (
                    result.get(f"pd_paired_{phase}_tiered_error", "")
                    if isinstance(result, dict)
                    else ""
                )
                if error:
                    phase_future.set_exception(
                        TieredBackendEligibilityError(str(error))
                    )
                else:
                    phase_future.set_result(result)
            except Exception as error:
                phase_future.set_exception(error)

        if pair_future.done():
            finish(pair_future)
        else:
            pair_future.add_done_callback(finish)
        return phase_future


class ExecutorWithExternalLauncher(UniProcExecutor):
    """An executor that uses external launchers to launch engines,
    specially designed for torchrun-compatible launchers, for
    offline inference with tensor parallelism.

    see https://github.com/vllm-project/vllm/issues/11400 for
    the motivation, and examples/offline_inference/torchrun_example.py
    for the usage example.

    The key idea: although it is tensor-parallel inference, we only
    create one worker per executor, users will launch multiple
    engines with torchrun-compatible launchers, and all these engines
    work together to process the same prompts. When scheduling is
    deterministic, all the engines will generate the same outputs,
    and they don't need to synchronize the states with each other.
    """

    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        assert not envs.VLLM_ENABLE_V1_MULTIPROCESSING, (
            "To get deterministic execution, "
            "please set VLLM_ENABLE_V1_MULTIPROCESSING=0"
        )
        super()._init_executor()

    def _distributed_args(self) -> tuple[str, int, int]:
        # engines are launched in torchrun-compatible launchers
        # so we can use the env:// method.
        # required env vars:
        # - RANK
        # - LOCAL_RANK
        # - MASTER_ADDR
        # - MASTER_PORT
        distributed_init_method = "env://"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        return distributed_init_method, rank, local_rank

    def determine_available_memory(self) -> list[int]:  # in bytes
        # we need to get the min across all ranks.
        memory = super().determine_available_memory()
        from vllm.distributed.parallel_state import get_world_group

        cpu_group = get_world_group().cpu_group
        memory_tensor = torch.tensor([memory], device="cpu", dtype=torch.int64)
        dist.all_reduce(memory_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        return [memory_tensor.item()]
