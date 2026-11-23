# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import queue
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, TypeVar, cast

import msgspec
import zmq

from vllm.config import CompilationMode, ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.pd_layerwise_handoff import (
    PD_LAYERWISE_HANDOFF_CONTROLLER,
    layerwise_profile_handoff_decision,
    layerwise_profile_handoff_requested,
    layerwise_profile_handoff_strict,
)
from vllm.model_executor.layers.pd_prefill_workspace import (
    prefill_workspace_requested,
    prefill_workspace_slot_count,
)
from vllm.model_executor.layers.kv_retention_parked_prefill import (
    kv_retention_p1_hbm_only_requested,
    kv_retention_p1_parked_requested,
    kv_retention_parked_limits,
    validate_kv_retention_p1_hbm_only,
)
from vllm.model_executor.layers.kv_retention_resume_frame import (
    kv_retention_resume_frame_gpu_bytes,
    kv_retention_resume_frame_requested,
)
from vllm.model_executor.layers.ttft_layerwise_prefill import (
    KVRetentionExecutionBinding,
    KVRetentionPhaseOwner,
    KVRetentionPrefillWorkRegistry,
    TTFTLayerwisePrefillBoundary,
    kv_retention_layerwise_checkpoint_requested,
    kv_retention_prefill_work_binding_requested,
    cancel_ttft_layerwise_prefill_yield,
    request_ttft_layerwise_prefill_yield,
    should_preempt_ttft_layerwise_prefill_batch,
    ttft_layerwise_prefill_execution_requested,
    ttft_layerwise_prefill_preemption_requested,
    ttft_prefill_continuation_mode_allowed,
)
from vllm.model_executor.layers.kv_retention_layerwise import (
    KVRetentionLayerwiseTransfer,
    kv_retention_layerwise_offload_enabled,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import engine_receiver_cache_from_config
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.observability import (
    compact_worker_debug_timings,
    lightweight_timeline_enabled,
    strict_bounded_lightweight_timeline_enabled,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.tiered_errors import TieredBackendEligibilityError
from vllm.v1.core.tiered_execution import has_native_reload_authority
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    FinishReason,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.engine.async_jsonl_writer import AsyncJSONLWriter
from vllm.v1.engine.pd_tbt_profiling import TBTCaptureObserver
from vllm.v1.engine.pd_capacity_retry import PDPhaseCapacityRetryGate
from vllm.v1.engine.pd_engine_progress import (
    notify_pd_engine_progress,
    pd_engine_progress_generation,
    pd_future_lifecycle_snapshot,
    register_pd_engine_progress_event,
    unregister_pd_engine_progress_event,
    wait_for_pd_engine_progress,
)
from vllm.v1.engine.pd_sm_allocator import (
    PDComputeAllocationController,
    PDSMAllocatorConfig,
    PDSMPlan,
)
from vllm.v1.engine.pd_sm_transition import PDSMDrainTransition
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    get_device_indices,
)
from vllm.v1.executor import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import compute_iteration_details
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5
PD_EXTERNAL_PROGRESS_FALLBACK_WAIT_S = 0.01

_R = TypeVar("_R")  # Return type for collective_rpc


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
        self._pd_engine_progress_event = threading.Event()
        self._pd_engine_progress_registered = False
        self._pd_progress_waited_this_step = False
        self._pd_progress_generation_at_step_start = (
            pd_engine_progress_generation()
        )
        # plugins need to be loaded at the engine/scheduler level too
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        self.vllm_config = vllm_config
        if not vllm_config.parallel_config.data_parallel_rank_local:
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION,
                vllm_config,
            )

        self.log_stats = log_stats
        self.iteration_metrics_jsonl = os.environ.get(
            "VLLM_ITERATION_METRICS_JSONL", ""
        )
        self.decode_timeline_jsonl = os.environ.get(
            "VLLM_DECODE_TIMELINE_JSONL", ""
        )
        self.pd_phase_queue_trace_jsonl = os.environ.get(
            "VLLM_PD_PHASE_QUEUE_TRACE_JSONL", ""
        )
        tbt_capture_path = os.environ.get("VLLM_PD_TBT_PROFILING_JSONL", "")
        self.lightweight_timeline_enabled = lightweight_timeline_enabled()
        self.strict_bounded_lightweight_timeline_enabled = (
            strict_bounded_lightweight_timeline_enabled()
        )
        legacy_path_requested = bool(
            self.iteration_metrics_jsonl or self.decode_timeline_jsonl
        )
        self.full_iteration_debug_enabled = bool(
            os.environ.get("VLLM_FULL_ITERATION_DEBUG", "").lower()
            in ("1", "true", "yes", "on")
            or (legacy_path_requested and not self.lightweight_timeline_enabled)
            or os.environ.get("VLLM_PD_DECODE_GPU_COMPONENT_TRACE_JSONL")
        )
        try:
            self.iteration_metrics_sample_every_n = max(
                1,
                int(
                    os.environ.get(
                        "VLLM_ITERATION_METRICS_SAMPLE_EVERY_N", "100"
                    )
                ),
            )
        except ValueError:
            self.iteration_metrics_sample_every_n = 100
        try:
            debug_trace_queue_size = max(
                1,
                int(os.environ.get("VLLM_DEBUG_TRACE_QUEUE_SIZE", "4096")),
            )
        except ValueError:
            debug_trace_queue_size = 4096
        self._debug_trace_writer = (
            AsyncJSONLWriter(max_pending=debug_trace_queue_size)
            if (
                self.iteration_metrics_jsonl
                or self.decode_timeline_jsonl
                or self.pd_phase_queue_trace_jsonl
                or tbt_capture_path
            )
            else None
        )
        self._tbt_capture = (TBTCaptureObserver(
            lambda record: self._append_jsonl(tbt_capture_path, record)
        ) if tbt_capture_path else None)
        try:
            self.pd_phase_queue_trace_heartbeat_interval_s = max(
                0.0,
                float(
                    os.environ.get(
                        "VLLM_PD_PHASE_QUEUE_TRACE_HEARTBEAT_INTERVAL_SEC",
                        "0.25",
                    )
                ),
            )
        except ValueError:
            self.pd_phase_queue_trace_heartbeat_interval_s = 0.25
        self._pd_phase_queue_trace_last_heartbeat_ts: dict[str, float] = {}
        self._pd_phase_queue_trace_suppressed_heartbeats: dict[str, int] = {}
        self._pd_future_liveness_audit_requested = os.environ.get(
            "VLLM_PD_FUTURE_LIVENESS_AUDIT", "0"
        ).lower() in ("1", "true", "yes", "on")
        try:
            self.pd_future_liveness_audit_interval_s = float(
                os.environ.get(
                    "VLLM_PD_FUTURE_LIVENESS_AUDIT_INTERVAL_SEC", "5"
                )
            )
        except ValueError as exc:
            raise ValueError(
                "VLLM_PD_FUTURE_LIVENESS_AUDIT_INTERVAL_SEC must be a "
                "positive number."
            ) from exc
        if self.pd_future_liveness_audit_interval_s <= 0.0:
            raise ValueError(
                "VLLM_PD_FUTURE_LIVENESS_AUDIT_INTERVAL_SEC must be "
                "positive."
            )
        self._pd_future_liveness_audit_total = 0
        self._iteration_debug_timings: dict[int, dict[str, Any]] = {}
        self._tbt_controller_state: dict[str, int | float | str] = {}
        # A final Prefill chunk is a TTFT-critical handoff.  Keep unrelated
        # Prefill work out of the next phase queue slot until the request's
        # first sampled token is emitted.  This is deliberately request-local
        # state, so the global P/D future depth can remain at 3.
        self._pd_ttft_handoff_request_ids: set[str] = set()
        self._pd_ttft_handoff_started_at = 0.0
        self._pd_ttft_handoff_blocked_prefill_checks = 0

        # Setup Model.
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(
            vllm_config
        )

        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
        self.collective_rpc("initialize_cache", args=(num_gpu_blocks, num_cpu_blocks))

        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        if len(kv_cache_config.kv_cache_groups) == 0:  # noqa: SIM102
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            if vllm_config.scheduler_config.enable_chunked_prefill:
                logger.warning("Disabling chunked prefill for model without KVCache")
                vllm_config.scheduler_config.enable_chunked_prefill = False

        scheduler_block_size = (
            vllm_config.cache_config.block_size
            * vllm_config.parallel_config.decode_context_parallel_size
            * vllm_config.parallel_config.prefill_context_parallel_size
        )

        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
        )
        self.use_spec_decode = vllm_config.speculative_config is not None
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

        self.mm_registry = mm_registry = MULTIMODAL_REGISTRY
        self.mm_receiver_cache = engine_receiver_cache_from_config(
            vllm_config, mm_registry
        )

        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {tp_rank: metadata}
                # Merge all worker dicts into a single dict
                content: dict[int, Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata(content)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)
        requested_pd_muxwise_event_loop = os.environ.get(
            "VLLM_PD_MUXWISE_EVENT_LOOP", ""
        ).lower() in ("1", "true", "yes", "on")
        self.pd_muxwise_event_loop_enabled = requested_pd_muxwise_event_loop
        requested_pd_phase_depth = os.environ.get(
            "VLLM_PD_PHASE_DEPTH_ENABLED", ""
        ).lower() in ("1", "true", "yes", "on")
        try:
            requested_pd_prefill_inflight_depth = int(
                os.environ.get("VLLM_PD_PREFILL_INFLIGHT_DEPTH", "1")
            )
            requested_pd_decode_inflight_depth = int(
                os.environ.get("VLLM_PD_DECODE_INFLIGHT_DEPTH", "1")
            )
            requested_pd_global_inflight_depth = int(
                os.environ.get("VLLM_PD_GLOBAL_INFLIGHT_DEPTH", "2")
            )
        except ValueError as exc:
            raise ValueError(
                "P/D phase depth values must be integers."
            ) from exc
        if min(
            requested_pd_prefill_inflight_depth,
            requested_pd_decode_inflight_depth,
            requested_pd_global_inflight_depth,
        ) < 1:
            raise ValueError("P/D phase depth values must be positive.")
        self.pd_prefill_inflight_depth = requested_pd_prefill_inflight_depth
        self.pd_decode_inflight_depth = requested_pd_decode_inflight_depth
        self.pd_global_inflight_depth = requested_pd_global_inflight_depth
        self.pd_decoupled_phase_aware_recovery = os.environ.get(
            "VLLM_PD_DECOUPLED_PHASE_AWARE_RECOVERY", ""
        ).lower() in ("1", "true", "yes", "on")
        requested_pd_phase_queues = os.environ.get(
            "VLLM_PD_INDEPENDENT_PHASE_QUEUES", ""
        ).lower() in ("1", "true", "yes", "on")
        if requested_pd_muxwise_event_loop:
            requested_pd_phase_queues = True
        self.pd_independent_phase_queues_enabled = (
            requested_pd_phase_queues
            and self.batch_queue_size > 1
            and not self.use_spec_decode
        )
        self.pd_phase_depth_enabled = (
            requested_pd_phase_depth
            and self.pd_independent_phase_queues_enabled
        )
        requested_pd_fused_execute_sample = os.environ.get(
            "VLLM_PD_FUSED_EXECUTE_SAMPLE", ""
        ).lower() in ("1", "true", "yes", "on")
        self.pd_fused_execute_sample_enabled = bool(
            requested_pd_fused_execute_sample
            and self.pd_independent_phase_queues_enabled
            and not self.use_spec_decode
            and getattr(
                self.model_executor,
                "pd_async_uniproc_enabled",
                False,
            )
            and callable(
                getattr(
                    self.model_executor,
                    "execute_model_and_sample",
                    None,
                )
            )
        )
        if (
            requested_pd_fused_execute_sample
            and not self.pd_fused_execute_sample_enabled
        ):
            raise RuntimeError(
                "VLLM_PD_FUSED_EXECUTE_SAMPLE requires independent phase "
                "queues, async UniProc execution, and speculative decoding "
                "disabled."
            )
        if requested_pd_phase_depth and not self.pd_phase_depth_enabled:
            raise RuntimeError(
                "P/D phase depth requires independent phase queues, async "
                "scheduling, and speculative decoding disabled."
            )
        if self.pd_phase_depth_enabled:
            if os.environ.get(
                "VLLM_PD_PHASE_LOCAL_MODEL_RUNNER", ""
            ).lower() not in ("1", "true", "yes", "on"):
                raise RuntimeError(
                    "P/D phase depth requires "
                    "VLLM_PD_PHASE_LOCAL_MODEL_RUNNER=1."
                )
            if not bool(
                getattr(self.model_executor, "pd_async_uniproc_enabled", False)
            ):
                raise RuntimeError(
                    "P/D phase depth requires the async UniProc executor."
                )
            if (
                self.pd_prefill_inflight_depth
                + self.pd_decode_inflight_depth
                > self.pd_global_inflight_depth
            ):
                raise ValueError(
                    "Prefill and decode inflight depths cannot exceed the "
                    "global P/D inflight depth."
                )
            if self.batch_queue_size < self.pd_global_inflight_depth:
                raise RuntimeError(
                    "Executor max_concurrent_batches is smaller than "
                    "VLLM_PD_GLOBAL_INFLIGHT_DEPTH."
                )
            self.batch_queue_size = self.pd_global_inflight_depth
            self.batch_queue = deque(maxlen=self.batch_queue_size)
        requested_pd_separate_schedulers = os.environ.get(
            "VLLM_PD_SEPARATE_SCHEDULERS", ""
        ).lower() in ("1", "true", "yes", "on")
        if requested_pd_muxwise_event_loop:
            requested_pd_separate_schedulers = True
        self.pd_separate_scheduler_lanes_enabled = (
            requested_pd_separate_schedulers
            and self.pd_independent_phase_queues_enabled
        )
        requested_pd_dual_continuous_batching = os.environ.get(
            "VLLM_PD_DUAL_CONTINUOUS_BATCHING", ""
        ).lower() in ("1", "true", "yes", "on")
        if requested_pd_muxwise_event_loop:
            requested_pd_dual_continuous_batching = True
        self.pd_dual_continuous_batching_enabled = (
            requested_pd_dual_continuous_batching
            and self.pd_separate_scheduler_lanes_enabled
        )
        requested_pd_paired_phase_launch = os.environ.get(
            "VLLM_PD_PAIRED_PHASE_LAUNCH", ""
        ).lower() in ("1", "true", "yes", "on")
        self.pd_paired_phase_launch_requested = requested_pd_paired_phase_launch
        requested_pd_nonblocking_phase_recovery = os.environ.get(
            "VLLM_PD_NONBLOCKING_PHASE_RECOVERY", ""
        ).lower() in ("1", "true", "yes", "on")
        if requested_pd_muxwise_event_loop:
            requested_pd_nonblocking_phase_recovery = True
        self.pd_nonblocking_phase_recovery_enabled = (
            requested_pd_nonblocking_phase_recovery
            and self.pd_independent_phase_queues_enabled
        )
        self.pd_future_liveness_audit_enabled = bool(
            self._pd_future_liveness_audit_requested
            and self.pd_nonblocking_phase_recovery_enabled
        )
        if hasattr(
            self.model_executor, "pd_future_liveness_audit_enabled"
        ):
            self.model_executor.pd_future_liveness_audit_enabled = (
                self.pd_future_liveness_audit_enabled
            )
        if (
            self._pd_future_liveness_audit_requested
            and not self.pd_future_liveness_audit_enabled
        ):
            logger.warning(
                "Disabling VLLM_PD_FUTURE_LIVENESS_AUDIT because "
                "nonblocking independent P/D phase recovery is not active."
            )
        elif self.pd_future_liveness_audit_enabled:
            logger.warning(
                "P/D Future liveness audit is enabled at %.3fs. Audit "
                "wakeups preserve Future, request, KV, workspace, and output "
                "ownership; they do not retry or fabricate completion.",
                self.pd_future_liveness_audit_interval_s,
            )
        single_inflight_default = (
            "1" if requested_pd_nonblocking_phase_recovery else "0"
        )
        requested_pd_single_inflight_per_phase = os.environ.get(
            "VLLM_PD_SINGLE_INFLIGHT_PER_PHASE",
            single_inflight_default,
        ).lower() in ("1", "true", "yes", "on")
        if requested_pd_muxwise_event_loop and not self.pd_phase_depth_enabled:
            requested_pd_single_inflight_per_phase = True
        if self.pd_phase_depth_enabled:
            requested_pd_single_inflight_per_phase = False
        self.pd_single_inflight_per_phase_enabled = (
            requested_pd_single_inflight_per_phase
            and self.pd_independent_phase_queues_enabled
        )
        try:
            requested_decode_reserved_batches = int(
                os.environ.get("VLLM_PD_DECODE_RESERVED_BATCHES", "1")
            )
        except ValueError:
            requested_decode_reserved_batches = 1
        self.pd_decode_reserved_batches = max(
            0,
            min(requested_decode_reserved_batches, self.batch_queue_size),
        )
        if requested_pd_separate_schedulers and not self.pd_independent_phase_queues_enabled:
            logger.warning(
                "Disabling VLLM_PD_SEPARATE_SCHEDULERS because "
                "VLLM_PD_INDEPENDENT_PHASE_QUEUES is not active."
            )
        if (
            requested_pd_dual_continuous_batching
            and not self.pd_separate_scheduler_lanes_enabled
        ):
            logger.warning(
                "Disabling VLLM_PD_DUAL_CONTINUOUS_BATCHING because "
                "VLLM_PD_SEPARATE_SCHEDULERS is not active."
            )
        if (
            requested_pd_nonblocking_phase_recovery
            and not self.pd_independent_phase_queues_enabled
        ):
            logger.warning(
                "Disabling VLLM_PD_NONBLOCKING_PHASE_RECOVERY because "
                "VLLM_PD_INDEPENDENT_PHASE_QUEUES is not active."
            )
        if (
            requested_pd_single_inflight_per_phase
            and not self.pd_independent_phase_queues_enabled
        ):
            logger.warning(
                "Disabling VLLM_PD_SINGLE_INFLIGHT_PER_PHASE because "
                "VLLM_PD_INDEPENDENT_PHASE_QUEUES is not active."
            )
        if requested_pd_phase_queues and self.use_spec_decode:
            logger.warning(
                "Disabling VLLM_PD_INDEPENDENT_PHASE_QUEUES because "
                "speculative decoding is enabled."
            )
        if requested_pd_phase_queues and self.batch_queue_size <= 1:
            logger.warning(
                "Disabling VLLM_PD_INDEPENDENT_PHASE_QUEUES because "
                "model executor max_concurrent_batches=%d.",
                self.batch_queue_size,
            )
        self.pd_phase_queues: dict[
            str,
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]],
        ] | None = None
        self._pd_phase_queue_submit_seq = 0
        self._pd_capacity_retry_gate = PDPhaseCapacityRetryGate()
        self._pd_capacity_retry_suppressed_this_step = False
        if self.pd_independent_phase_queues_enabled:
            self.pd_phase_queues = {
                "decode": deque(maxlen=self.batch_queue_size),
                "prefill": deque(maxlen=self.batch_queue_size),
                "mixed": deque(maxlen=self.batch_queue_size),
                "idle": deque(maxlen=self.batch_queue_size),
            }

        self.is_ec_producer = (
            vllm_config.ec_transfer_config is not None
            and vllm_config.ec_transfer_config.is_ec_producer
        )
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"
        self.pd_paired_phase_launch_enabled = (
            requested_pd_paired_phase_launch
            and self.pd_dual_continuous_batching_enabled
            # The legacy paired RPC projects one shared execute future into
            # both phase dependencies. It is safe only with one in-flight
            # future per phase. Phase-depth mode must use independent phase
            # futures plus the Decode-first host-enqueue gate.
            and self.pd_single_inflight_per_phase_enabled
            and not self.pd_phase_depth_enabled
            and not self.is_ec_producer
            and not self.is_pooling_model
            and hasattr(self.model_executor, "execute_model_pd_pair")
        )
        if (
            requested_pd_paired_phase_launch
            and not self.pd_paired_phase_launch_enabled
        ):
            logger.warning(
                "Disabling VLLM_PD_PAIRED_PHASE_LAUNCH because dual P/D "
                "phase queues, non-depth single-inflight phase ownership, "
                "non-pooling/non-EC execution, and executor "
                "execute_model_pd_pair support are required. Phase-depth "
                "execution falls back to independent phase futures with "
                "Decode-first host enqueue ordering."
            )
        requested_pd_decoupled_paired_recovery = os.environ.get(
            "VLLM_PD_DECOUPLED_PAIRED_RECOVERY", ""
        ).lower() in ("1", "true", "yes", "on")
        self.pd_decoupled_paired_recovery_requested = (
            requested_pd_decoupled_paired_recovery
        )
        executor_has_phase_workers = bool(
            getattr(
                self.model_executor,
                "pd_async_uniproc_phase_workers_enabled",
                False,
            )
        )
        executor_has_concurrent_phase_submit = bool(
            getattr(
                self.model_executor,
                "pd_async_uniproc_concurrent_phase_submit_enabled",
                False,
            )
        )
        self.pd_decoupled_paired_recovery_enabled = (
            requested_pd_decoupled_paired_recovery
            and self.pd_paired_phase_launch_enabled
            and self.pd_nonblocking_phase_recovery_enabled
            and self.pd_single_inflight_per_phase_enabled
            and executor_has_phase_workers
            and executor_has_concurrent_phase_submit
        )
        if (
            requested_pd_decoupled_paired_recovery
            and not self.pd_decoupled_paired_recovery_enabled
        ):
            logger.warning(
                "Disabling VLLM_PD_DECOUPLED_PAIRED_RECOVERY because paired "
                "phase scheduling, nonblocking single-inflight phase queues, "
                "UniProc phase workers, and concurrent phase submit are all "
                "required."
            )

        pd_sm_allocator_config = PDSMAllocatorConfig.from_env()
        pd_green_context_enabled = (
            os.environ.get(
                "VLLM_OPERATOR_SELECTIVE_PD_GREEN_CONTEXT", "0"
            ).lower()
            in ("1", "true", "yes", "on")
            or os.environ.get(
                "VLLM_MUXWISE_PHASE_BASELINE_GREEN_CONTEXT", "0"
            ).lower()
            in ("1", "true", "yes", "on")
        )
        pd_green_execute_enabled = os.environ.get(
            "VLLM_OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_MODEL", "0"
        ).lower() in ("1", "true", "yes", "on")
        pd_dynamic_sm_prerequisites = (
            self.pd_independent_phase_queues_enabled
            and self.pd_separate_scheduler_lanes_enabled
            and (
                self.pd_single_inflight_per_phase_enabled
                or self.pd_phase_depth_enabled
            )
            and (
                pd_sm_allocator_config.shadow
                or (pd_green_context_enabled and pd_green_execute_enabled)
            )
        )
        if pd_sm_allocator_config.enabled and not pd_dynamic_sm_prerequisites:
            message = (
                "VLLM_PD_DYNAMIC_SM_ENABLED requires independent P/D queues, "
                "separate scheduler lanes, bounded phase depth, and "
                "execute-model GreenContext routing (except in shadow mode)."
            )
            if pd_sm_allocator_config.strict:
                raise RuntimeError(message)
            logger.warning("%s Disabling dynamic SM allocation.", message)
            pd_sm_allocator_config = pd_sm_allocator_config.with_effective_enabled(
                False
            )
        self.pd_sm_allocator = PDComputeAllocationController(
            pd_sm_allocator_config
        )
        if self.pd_sm_allocator.predictive_enabled and (
            not self.pd_nonblocking_phase_recovery_enabled
            or (self.pd_phase_depth_enabled and self.pd_decode_inflight_depth != 1)
        ):
            raise RuntimeError("tbt_slo_predictive requires nonblocking P/D recovery and Decode depth 1")
        if self.pd_sm_allocator.enabled and not pd_sm_allocator_config.shadow:
            profile_bank_results = self.model_executor.collective_rpc(
                "initialize_dynamic_green_context_profiles"
            )
            expected_profile_ids = ",".join(
                profile.profile_id for profile in pd_sm_allocator_config.profiles
            )
            profile_bank_matches = bool(profile_bank_results) and all(
                (
                    isinstance(result, dict)
                    and result.get("enabled") == 1
                    and result.get("total_sms")
                    == pd_sm_allocator_config.total_sms
                    and result.get("profile_ids") == expected_profile_ids
                )
                for result in profile_bank_results
            )
            if not profile_bank_matches:
                message = (
                    "Dynamic P/D GreenContext profile-bank configuration does "
                    "not match the EngineCore controller: "
                    f"engine_total_sms={pd_sm_allocator_config.total_sms} "
                    f"engine_profiles={expected_profile_ids} "
                    f"workers={profile_bank_results!r}."
                )
                if pd_sm_allocator_config.strict:
                    raise RuntimeError(message)
                logger.warning("%s Disabling dynamic SM allocation.", message)
                pd_sm_allocator_config = (
                    pd_sm_allocator_config.with_effective_enabled(False)
                )
                self.pd_sm_allocator = PDComputeAllocationController(
                    pd_sm_allocator_config
                )
        self.pd_dynamic_sm_enabled = self.pd_sm_allocator.enabled
        self.pd_dynamic_sm_enforced = (
            self.pd_dynamic_sm_enabled and not pd_sm_allocator_config.shadow
        )
        requested_layerwise_handoff = layerwise_profile_handoff_requested()
        self.kv_retention_p1_parked_enabled = kv_retention_p1_parked_requested()
        self.kv_retention_p1_hbm_only = kv_retention_p1_hbm_only_requested()
        layerwise_model_type = str(
            getattr(vllm_config.model_config.hf_config, "model_type", "")
            or ""
        ).lower()
        layerwise_global_eager = (
            vllm_config.compilation_config.mode == CompilationMode.NONE
        )
        layerwise_selective_eager = bool(
            (
                requested_layerwise_handoff
                or ttft_layerwise_prefill_execution_requested()
            )
            and layerwise_model_type == "llama"
        )
        layerwise_model_eager = bool(
            layerwise_global_eager or layerwise_selective_eager
        )
        if requested_layerwise_handoff:
            logger.info(
                "Layerwise handoff model execution: model_type=%s "
                "global_eager=%s selective_eager=%s",
                layerwise_model_type,
                layerwise_global_eager,
                layerwise_selective_eager,
            )
        layerwise_handoff_prerequisites = bool(
            self.pd_dynamic_sm_enforced
            and self.pd_phase_depth_enabled
            and self.pd_nonblocking_phase_recovery_enabled
            and self.pd_prefill_inflight_depth == 1
            and self.pd_decode_inflight_depth == 1
            and self.pd_global_inflight_depth >= 2
            and vllm_config.parallel_config.world_size == 1
            and getattr(
                self.model_executor, "pd_async_uniproc_phase_workers_enabled", False
            )
            and getattr(
                self.model_executor,
                "pd_async_uniproc_concurrent_phase_submit_enabled",
                False,
            )
            and layerwise_model_eager
            and layerwise_model_type == "llama"
        )
        if requested_layerwise_handoff:
            PD_LAYERWISE_HANDOFF_CONTROLLER.reset()
        if requested_layerwise_handoff and not layerwise_handoff_prerequisites:
            message = (
                "VLLM_PD_LAYERWISE_PROFILE_HANDOFF requires enforced dynamic "
                "SM allocation, global or selective eager Llama execution, "
                "phase-local async "
                "single-GPU UniProc workers with concurrent phase submission, "
                "event-driven nonblocking recovery, and Prefill/Decode inflight "
                "depth 1 with global depth >= 2."
            )
            if layerwise_profile_handoff_strict():
                raise RuntimeError(message)
            logger.warning("%s Disabling layerwise handoff.", message)
        self.pd_layerwise_handoff_enabled = bool(
            requested_layerwise_handoff and layerwise_handoff_prerequisites
            and not self.kv_retention_p1_parked_enabled
        )
        requested_ttft_prefill_preemption = (
            ttft_layerwise_prefill_preemption_requested()
        )
        requested_prefill_workspace = prefill_workspace_requested()
        ttft_prefill_workspace_compatible = bool(
            not requested_prefill_workspace
            or prefill_workspace_slot_count()
            >= self.pd_prefill_inflight_depth
        )
        ttft_prefill_scheduler = getattr(
            self.scheduler,
            "ttft_prefill_chunk_scheduler",
            None,
        )
        ttft_prefill_policy = getattr(
            ttft_prefill_scheduler,
            "policy",
            None,
        )
        ttft_prefill_policy_name = str(
            getattr(ttft_prefill_policy, "value", ttft_prefill_policy) or ""
        ).lower()
        ttft_prefill_preemption_prerequisites = bool(
            ttft_layerwise_prefill_execution_requested()
            and ttft_prefill_policy_name == "srpt"
            and self.pd_dynamic_sm_enforced
            and self.pd_phase_depth_enabled
            and self.pd_nonblocking_phase_recovery_enabled
            and self.pd_separate_scheduler_lanes_enabled
            and self.pd_dual_continuous_batching_enabled
            and (self.pd_prefill_inflight_depth >= 2
                 or (self.kv_retention_p1_parked_enabled and self.pd_prefill_inflight_depth == 1))
            and self.pd_decode_inflight_depth == 1
            and self.pd_global_inflight_depth
            >= self.pd_prefill_inflight_depth + self.pd_decode_inflight_depth
            and vllm_config.parallel_config.world_size == 1
            and layerwise_model_type == "llama"
            and layerwise_model_eager
            and not self.pd_fused_execute_sample_enabled
            and not self.pd_paired_phase_launch_enabled
            and (not requested_layerwise_handoff or self.kv_retention_p1_parked_enabled)
            and ttft_prefill_workspace_compatible
            and getattr(
                self.model_executor,
                "pd_async_uniproc_phase_workers_enabled",
                False,
            )
            and getattr(
                self.model_executor,
                "pd_async_uniproc_concurrent_phase_submit_enabled",
                False,
            )
        )
        if (
            requested_ttft_prefill_preemption
            and not ttft_prefill_preemption_prerequisites
        ):
            raise RuntimeError(
                "VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION requires its base "
                "execution flag, SRPT Prefill scheduling, enforced dynamic "
                "SM, eager single-GPU Llama, "
                "event-driven separate P/D lanes, Prefill depth >= 2 "
                "(or the complete explicit P1 parked bundle), Decode "
                "depth 1, sufficient global depth, phase-local concurrent "
                "UniProc execution, fused/paired execution disabled, and "
                "legacy in-Future handoff disabled outside the P1 bundle; when fixed Prefill "
                "workspace is enabled its slot count must cover Prefill depth."
            )
        self.ttft_layerwise_prefill_preemption_enabled = bool(
            requested_ttft_prefill_preemption
            and ttft_prefill_preemption_prerequisites
        )
        from vllm.isolation_memory import isolation_execution_enabled
        # Independent of the older optimization bundles: changes Engine/IPC
        # ordering, never CUDA completion or SchedulerOutput ownership.
        output_first = os.getenv("VLLM_ISOLATION_DECODE_OUTPUT_FIRST", "0")
        if output_first not in {"0", "1"}:
            raise ValueError("VLLM_ISOLATION_DECODE_OUTPUT_FIRST must be 0 or 1")
        bounded_planning = getattr(
            self.scheduler, "isolation_bounded_prefill_planning", False)
        if bounded_planning and not (
            self.pd_separate_scheduler_lanes_enabled
            and self.pd_nonblocking_phase_recovery_enabled
            and not self.pd_paired_phase_launch_enabled
        ):
            raise ValueError("Bounded Prefill planning requires nonblocking separate "
                             "P/D lanes with paired launch disabled")
        self.isolation_decode_output_first = output_first == "1" or bounded_planning
        self.isolation_p1_ring_sm_handoff_enabled = bool(
            self.kv_retention_p1_parked_enabled
            and self.ttft_layerwise_prefill_preemption_enabled
            and isolation_execution_enabled("P1_RING_SM_HANDOFF"))
        if self.isolation_p1_ring_sm_handoff_enabled:
            logger.info("P1 Ring SM-only continuation enabled: original slot/KV retained; "
                        "ordinary Ring request parking disabled")
        validate_kv_retention_p1_hbm_only(
            enabled=self.kv_retention_p1_hbm_only,
            parked=self.kv_retention_p1_parked_enabled,
            offload=kv_retention_layerwise_offload_enabled(),
            completion_guard=getattr(self.scheduler, "isolation_hbm_guard", None),
            virtual_context=self.scheduler.kv_cache_manager.virtual_context,
        )
        self._kv_retention_hbm_preemption_blocker = None
        if self.kv_retention_p1_hbm_only:
            logger.info("P1 HBM-only SRPT preemption: retained KV, capacity-checked "
                        "request yield, no offload/Ring; failed admission resumes parked work")
        requested_work_binding = kv_retention_prefill_work_binding_requested()
        if self.kv_retention_p1_parked_enabled:
            kv_retention_parked_limits()
            if not (self.ttft_layerwise_prefill_preemption_enabled
                    and requested_work_binding and kv_retention_resume_frame_requested()
                    and kv_retention_layerwise_checkpoint_requested()
                    and (kv_retention_layerwise_offload_enabled()
                         or self.kv_retention_p1_hbm_only)
                    and self.pd_prefill_inflight_depth == 1
                    and self.pd_decode_inflight_depth == 1 and self.pd_global_inflight_depth == 2
                    and requested_layerwise_handoff and requested_prefill_workspace
                    and prefill_workspace_slot_count() == 1):
                raise RuntimeError("P1 parked Prefill requires P1/D1/G2, Stage2 SRPT, work/frame/checkpoint, offload or explicit HBM-only mode, SM handoff and one workspace")
        if requested_work_binding and not self.ttft_layerwise_prefill_preemption_enabled:
            raise RuntimeError(
                "VLLM_KV_RETENTION_PREFILL_WORK_BINDING requires Stage-2 "
                "Prefill preemption; this option does not enable P1 parking.")
        self._kv_retention_prefill_work_registry = (
            KVRetentionPrefillWorkRegistry(self.pd_prefill_inflight_depth,
                parked_capacity=int(self.kv_retention_p1_parked_enabled))
            if requested_work_binding else None)
        if kv_retention_resume_frame_requested():
            if not (requested_work_binding
                    and kv_retention_layerwise_checkpoint_requested()
                    and vllm_config.speculative_config is None):
                raise RuntimeError(
                    "VLLM_KV_RETENTION_PREFILL_RESUME_FRAME requires work "
                    "binding, checkpoint plan and non-speculative Stage-2; "
                    "it does not enable P1 slot lending.")
            kv_retention_resume_frame_gpu_bytes()  # Fail invalid budget at startup.
        self.kv_retention_layerwise_offload_enabled = kv_retention_layerwise_offload_enabled()
        if self.kv_retention_layerwise_offload_enabled:
            from vllm.v1.core.tiered_hbm_source_block_lease import (
                tiered_hbm_source_block_lease_enabled,
            )
            from vllm.v1.core.virtual_context import (
                virtual_context_ring_backend_dram_provider_enabled,
                virtual_context_ring_backend_enabled,
            )
            if not (
                self.ttft_layerwise_prefill_preemption_enabled
                and kv_retention_layerwise_checkpoint_requested()
                and tiered_hbm_source_block_lease_enabled()
                and virtual_context_ring_backend_dram_provider_enabled()
                and virtual_context_ring_backend_enabled()
                and self.scheduler.kv_cache_manager.virtual_context is not None
                and self.scheduler.virtual_context_tiered_external_blocks
                and self.scheduler.virtual_context_reload_to_hbm
                and vllm_config.speculative_config is None
                and not vllm_config.model_config.enable_return_routed_experts
            ):
                raise RuntimeError(
                    "VLLM_KV_RETENTION_LAYERWISE_OFFLOAD requires Stage-2 "
                    "preemption, CHECKPOINT_PLAN, virtual-context DRAM provider "
                    "and D2H_SOURCE_BLOCK_LEASE, tiered external blocks and "
                    "native reload, without speculative decode/routed-expert "
                    "capture; no implicit prerequisite changes."
                )
        self._ttft_prefill_suspended: dict[
            str, tuple[TTFTLayerwisePrefillBoundary, SchedulerOutput]
        ] = {}
        self._kv_retention_parked_offload_future = None
        self._kv_retention_parked_offload_attempt = None
        self._ttft_prefill_boundary_count = 0
        self._ttft_prefill_preemption_signal_count = 0
        self._ttft_prefill_resume_submit_count = 0
        self._ttft_prefill_final_execute_count = 0
        self._ttft_prefill_deferred_abort_ids: set[str] = set()
        self._ttft_prefill_deferred_abort_count = 0
        self._ttft_prefill_deferred_abort_flush_count = 0
        self._pd_layerwise_handoff_fence_active = False
        self._pd_layerwise_handoff_fence_blocked_decode_checks = 0
        self._pd_layerwise_handoff_fence_blocked_prefill_checks = 0
        self._pd_layerwise_handoff_source_decode_admission_checks = 0
        self._pd_layerwise_handoff_source_decode_admission_allowed = 0
        self._pd_layerwise_handoff_decode_inflight_defer_checks = 0
        self._pd_layerwise_handoff_decode_backlog_defer_checks = 0
        self._pd_layerwise_handoff_prefill_only_retarget_count = 0
        self._pd_layerwise_handoff_prefill_only_retarget_miss_count = 0
        self.pd_sm_transition = PDSMDrainTransition()
        self._pd_sm_transition_commit_count = 0
        self._pd_sm_transition_wait_ms_total = 0.0
        self._pd_sm_transition_prefill_drain_ms_total = 0.0
        self._pd_sm_transition_all_fence_ms_total = 0.0
        self._pd_sm_transition_resume_submit_count = 0
        self._pd_sm_transition_resume_submit_ms_total = 0.0
        self._pd_sm_transition_resume_submit_ms_max = 0.0
        self._pd_sm_transition_resume_pending = False
        self._pd_sm_transition_resume_commit_ts = 0.0
        self._pd_sm_transition_last_commit_debug: dict[
            str, int | float | str
        ] = {}
        if self.pd_dynamic_sm_enabled:
            logger.warning(
                "Dynamic P/D SM allocation is experimental: policy=%s "
                "shadow=%s "
                "serialized_full_sm=%s initial_profile=%s profiles=%s",
                pd_sm_allocator_config.policy,
                pd_sm_allocator_config.shadow,
                pd_sm_allocator_config.serialized_full_sm,
                pd_sm_allocator_config.initial_profile.profile_id,
                ",".join(
                    profile.profile_id
                    for profile in pd_sm_allocator_config.profiles
                ),
            )
        if self.pd_layerwise_handoff_enabled:
            logger.warning(
                "Layer-boundary P/D SM handoff is experimental: every "
                "profile transition that overlaps one active Prefill future "
                "uses a GPU-complete layer continuation; source Decode stays "
                "admissible until that boundary completes, then drains before "
                "the target epoch is committed."
            )

        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                scheduler_block_size, caching_hash_fn
            )

        self.step_fn = self.step
        if self.batch_queue is not None:
            self.step_fn = (
                self.step_with_pd_phase_queues
                if self.pd_independent_phase_queues_enabled
                else self.step_with_batch_queue
            )
        if self.pd_nonblocking_phase_recovery_enabled:
            register_pd_engine_progress_event(
                self._pd_engine_progress_event
            )
            self._pd_engine_progress_registered = True
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling

        self.aborts_queue = queue.Queue[list[str]]()
        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        freeze_gc_heap()
        # If enable, attach GC debugger after static variable freeze.
        maybe_attach_gc_debug_callback()
        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()

    def _notify_pd_engine_progress(self) -> None:
        """Wake the process-local P/D event loop."""

        if not self._pd_engine_progress_registered:
            return
        # Shared generation tracking covers Future, input, and tiered-provider
        # callbacks and closes the Event.clear lost-wakeup race.
        notify_pd_engine_progress()

    def _register_pd_model_future_completion(
        self,
        future: Future[Any],
        scheduler_output: SchedulerOutput,
        *,
        collect_timing: bool,
    ) -> None:
        """Attach the completion callback shared by P/D phase futures."""

        output_id = id(scheduler_output)

        def completed(_future: Future[Any]) -> None:
            if self.pd_dynamic_sm_enabled or collect_timing:
                self._iteration_debug_timings.get(output_id, {}).setdefault(
                    "actual_model_future_done_ts",
                    time.monotonic(),
                )
            self._notify_pd_engine_progress()

        future.add_done_callback(completed)

    def _pd_input_pending(self) -> bool:
        input_queue = getattr(self, "input_queue", None)
        return bool(input_queue is not None and not input_queue.empty())

    def _pd_recoverable_future_ready(self, phase_queues) -> bool:
        """Return whether queue policy can recover a Future right now."""

        pending_copy = getattr(self, "_kv_retention_parked_offload_future", None)
        if pending_copy is not None and pending_copy[0].done():
            return True
        if not any(phase_queues.values()):
            return False
        phase, index, _ = self._select_pd_phase_queue_recovery_item(
            phase_queues,
            collect_debug=False,
        )
        return bool(phase_queues[phase][index][0].done())

    def _record_pd_future_liveness_audit(
        self,
        phase_queues,
        selected_future: Future[Any],
        selected_scheduler_output: SchedulerOutput | None,
    ) -> None:
        """Record a non-mutating audit of one still-owned model Future."""

        self._pd_future_liveness_audit_total += 1
        timing_record = (
            self._iteration_debug_timings.setdefault(
                id(selected_scheduler_output), {}
            )
            if selected_scheduler_output is not None
            else None
        )
        audit_count = 1
        if timing_record is not None:
            audit_count = int(
                timing_record.get("pd_future_liveness_audit_count", 0.0)
            ) + 1
            timing_record["pd_future_liveness_audit_count"] = float(
                audit_count
            )

        lifecycle = pd_future_lifecycle_snapshot(selected_future)
        phase = (
            self._scheduler_output_phase(selected_scheduler_output)
            if selected_scheduler_output is not None
            else ""
        )
        submit_seq = (
            int(selected_scheduler_output.pd_phase_submit_seq)
            if selected_scheduler_output is not None
            else -1
        )
        record: dict[str, Any] = {
            **lifecycle,
            "pd_future_liveness_audit_enabled": 1.0,
            "pd_future_liveness_audit_count": float(audit_count),
            "pd_future_liveness_audit_total": float(
                self._pd_future_liveness_audit_total
            ),
            "pd_future_liveness_audit_interval_s": float(
                self.pd_future_liveness_audit_interval_s
            ),
            "pd_future_liveness_audit_selected_phase": phase,
            "pd_future_liveness_audit_selected_submit_seq": float(
                submit_seq
            ),
            "pd_future_liveness_audit_raced_completion": float(
                selected_future.done()
            ),
            "pd_future_liveness_audit_ownership_preserved": 1.0,
        }
        if selected_scheduler_output is not None:
            record["pd_future_liveness_audit_request_count"] = float(
                len(
                    self._scheduler_output_request_ids(
                        selected_scheduler_output
                    )
                )
            )
        self._write_pd_phase_queue_trace(
            "future_liveness_audit",
            phase_queues,
            record,
        )

        # Log the first audit and powers of two thereafter. The JSONL trace
        # retains every configured audit interval without flooding server.log.
        if audit_count == 1 or audit_count & (audit_count - 1) == 0:
            logger.warning(
                "P/D Future liveness audit: phase=%s submit_seq=%d "
                "count=%d stage=%s stage_age=%.3fs future_age=%.3fs "
                "done=%s ownership=preserved",
                phase or "unknown",
                submit_seq,
                audit_count,
                lifecycle["pd_future_lifecycle_stage"],
                lifecycle["pd_future_lifecycle_stage_age_s"],
                lifecycle["pd_future_lifecycle_age_s"],
                selected_future.done(),
            )

    def _wait_for_pd_progress(
        self,
        phase_queues,
        *,
        selected_future: Future[Any] | None = None,
        selected_scheduler_output: SchedulerOutput | None = None,
    ) -> bool:
        """Block on callbacks or an explicitly enabled ownership-safe audit."""

        def ready() -> bool:
            if self._pd_input_pending():
                return True
            return self._pd_recoverable_future_ready(phase_queues)

        if ready():
            return True
        self._pd_progress_waited_this_step = True
        # A queued model Future has an in-process completion callback registered
        # by _register_pd_model_future_completion(). New requests and local
        # tiered-copy completions publish through the same progress generation.
        # The normal model-Future path remains callback-owned. The opt-in
        # liveness audit is intentionally infrequent and observational: it
        # reports the executor stage and lets phase selection reconsider an
        # already-ready independent head, but it cannot clear or reassign any
        # Future-owned resource.
        audit_selected_future = bool(
            selected_future is not None
            and self.pd_future_liveness_audit_enabled
        )
        timeout_s = (
            self.pd_future_liveness_audit_interval_s
            if audit_selected_future
            else None
            if selected_future is not None
            else PD_EXTERNAL_PROGRESS_FALLBACK_WAIT_S
        )
        allocator = getattr(self, "pd_sm_allocator", None)
        deadline_timeout = (allocator.tbt_deadline_wait_s(time.monotonic())
                            if getattr(allocator, "predictive_enabled", False) else None)
        if deadline_timeout is not None:
            timeout_s = deadline_timeout if timeout_s is None else min(timeout_s, deadline_timeout)
        progressed = wait_for_pd_engine_progress(
            self._pd_engine_progress_event,
            ready,
            timeout_s=timeout_s,
            observed_generation=(
                self._pd_progress_generation_at_step_start
            ),
        )
        deadline_fired = bool(
            deadline_timeout is not None and allocator.consume_tbt_deadline(time.monotonic()))
        if deadline_fired:
            # Observation only: no notification to the capacity gate, no
            # fabricated Future completion, no periodic retry loop.
            self._write_pd_phase_queue_trace("tbt_predictive_deadline_wake", phase_queues)
        if (
            not progressed
            and not deadline_fired
            and audit_selected_future
            and selected_future is not None
        ):
            self._record_pd_future_liveness_audit(
                phase_queues,
                selected_future,
                selected_scheduler_output,
            )
        return progressed

    def _initialize_kv_caches(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int, KVCacheConfig]:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            if os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1":
                dp_group = getattr(self, "dp_group", None)
                assert dp_group is not None
                self.available_gpu_memory_for_kv_cache = (
                    ParallelConfig.sync_kv_cache_memory_size(dp_group, -1)
                )
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                available_gpu_memory = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)

        # Track max_model_len before KV cache config to detect auto-fit changes
        max_model_len_before = vllm_config.model_config.max_model_len

        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )

        # If auto-fit reduced max_model_len, sync the new value to workers.
        # This is needed because workers were spawned before memory profiling
        # and have the original (larger) max_model_len cached.
        max_model_len_after = vllm_config.model_config.max_model_len
        if max_model_len_after != max_model_len_before:
            self.collective_rpc("update_max_model_len", args=(max_model_len_after,))

        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        num_cpu_blocks = 0

        # Initialize kv cache and warmup the execution
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info_once(
            "init engine (profile, create kv cache, warmup model) took %.2f seconds",
            elapsed,
            scope="local",
        )
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_executor.supported_tasks

    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        if pooling_params := request.pooling_params:
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks() if task in POOLING_TASKS
            ]

            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        # A new request can have a smaller HBM claim than a phase which was
        # previously capacity-blocked.  Invalidate the observation before the
        # scheduler sees the new ordering.
        self._pd_capacity_retry_gate.clear()
        self.scheduler.add_request(request)

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self._pd_capacity_retry_gate.clear()
        requested_ids = set(request_ids)
        if not requested_ids:
            return
        continuation_owned_ids = self._ttft_prefill_owned_request_ids()
        deferred_ids = requested_ids.intersection(continuation_owned_ids)
        immediate_ids = requested_ids.difference(deferred_ids)
        if deferred_ids:
            new_deferred_ids = deferred_ids.difference(
                self._ttft_prefill_deferred_abort_ids
            )
            self._ttft_prefill_deferred_abort_ids.update(deferred_ids)
            self._ttft_prefill_deferred_abort_count += len(new_deferred_ids)
        if immediate_ids:
            self.scheduler.finish_requests(
                immediate_ids,
                RequestStatus.FINISHED_ABORTED,
            )

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure."""
        try:
            yield
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(
                self.vllm_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    @contextmanager
    def log_iteration_details(self, scheduler_output: SchedulerOutput):
        if not self.vllm_config.observability_config.enable_logging_iteration_details:
            yield
            return
        self._iteration_index = getattr(self, "_iteration_index", 0)
        iteration_details = compute_iteration_details(scheduler_output)
        before = time.monotonic()
        yield
        logger.info(
            "".join(
                [
                    "Iteration(",
                    str(self._iteration_index),
                    "): ",
                    str(iteration_details.num_ctx_requests),
                    " context requests, ",
                    str(iteration_details.num_ctx_tokens),
                    " context tokens, ",
                    str(iteration_details.num_generation_requests),
                    " generation requests, ",
                    str(iteration_details.num_generation_tokens),
                    " generation tokens, iteration elapsed time: ",
                    format((time.monotonic() - before) * 1000, ".2f"),
                    " ms",
                ]
            )
        )
        self._iteration_index += 1

    @staticmethod
    def _scheduler_output_phase(scheduler_output: SchedulerOutput) -> str:
        return getattr(scheduler_output, "scheduled_phase", "unknown")

    @staticmethod
    def _scheduler_output_request_ids(
        scheduler_output: SchedulerOutput,
    ) -> set[str]:
        return set(scheduler_output.num_scheduled_tokens)

    @classmethod
    def _batch_queue_phase_counts(cls, batch_queue) -> dict[str, float]:
        counts = {
            "engine_queue_decode_future_count": 0.0,
            "engine_queue_prefill_future_count": 0.0,
            "engine_queue_mixed_future_count": 0.0,
            "engine_queue_idle_future_count": 0.0,
            "engine_queue_ready_decode_future_count": 0.0,
            "engine_queue_ready_prefill_future_count": 0.0,
        }
        for queued_future, scheduler_output, _ in batch_queue:
            phase = cls._scheduler_output_phase(scheduler_output)
            if phase == "decode":
                counts["engine_queue_decode_future_count"] += 1.0
                if queued_future.done():
                    counts["engine_queue_ready_decode_future_count"] += 1.0
            elif phase == "prefill":
                counts["engine_queue_prefill_future_count"] += 1.0
                if queued_future.done():
                    counts["engine_queue_ready_prefill_future_count"] += 1.0
            elif phase == "mixed":
                counts["engine_queue_mixed_future_count"] += 1.0
            elif phase == "idle":
                counts["engine_queue_idle_future_count"] += 1.0
        return counts

    def _pd_phase_queues_counts(self, phase_queues) -> dict[str, float]:
        counts = {
            "pd_independent_phase_queues_enabled": 1.0,
            "pd_muxwise_event_loop_enabled": float(
                self.pd_muxwise_event_loop_enabled
            ),
            "pd_phase_queue_total_depth": 0.0,
            "pd_phase_queue_decode_depth": 0.0,
            "pd_phase_queue_prefill_depth": 0.0,
            "pd_phase_queue_mixed_depth": 0.0,
            "pd_phase_queue_idle_depth": 0.0,
            "pd_phase_queue_ready_decode_count": 0.0,
            "pd_phase_queue_ready_prefill_count": 0.0,
            "pd_phase_queue_ready_mixed_count": 0.0,
            "pd_phase_queue_ready_idle_count": 0.0,
        }
        for phase, queue_ in phase_queues.items():
            phase_depth_key = f"pd_phase_queue_{phase}_depth"
            phase_ready_key = f"pd_phase_queue_ready_{phase}_count"
            counts[phase_depth_key] = float(len(queue_))
            counts["pd_phase_queue_total_depth"] += float(len(queue_))
            counts[phase_ready_key] = float(
                sum(1 for queued_future, _, _ in queue_ if queued_future.done())
            )
        suspended = float(self._ttft_prefill_suspended_count())
        counts["ttft_layerwise_prefill_suspended_depth"] = suspended
        counts["ttft_layerwise_prefill_deferred_abort_depth"] = float(
            len(self._ttft_prefill_deferred_abort_ids)
        )
        counts["ttft_layerwise_prefill_deferred_abort_count"] = float(
            self._ttft_prefill_deferred_abort_count
        )
        counts["ttft_layerwise_prefill_deferred_abort_flush_count"] = float(
            self._ttft_prefill_deferred_abort_flush_count
        )
        counts["pd_phase_queue_owned_prefill_depth"] = (
            float(self._pd_owned_phase_depth(phase_queues, "prefill"))
        )
        counts["pd_phase_queue_owned_total_depth"] = (
            float(self._pd_owned_total_depth(phase_queues))
        )
        tickets = [b.kv_retention_parked_ticket for b, _ in self._ttft_prefill_suspended.values()
                   if getattr(b, "kv_retention_parked_ticket", None) is not None]
        counts["kv_retention_p1_parked_batches"] = float(len(tickets))
        counts["kv_retention_p1_parked_requests"] = float(sum(len(t.request_ids) for t in tickets))
        counts["kv_retention_p1_parked_gpu_bytes"] = float(sum(t.gpu_bytes for t in tickets))
        return counts

    def _pd_phase_queue_items(self, phase_queues) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for phase, queue_ in phase_queues.items():
            for index, (future, scheduler_output, exec_future) in enumerate(queue_):
                submit_seq = scheduler_output.pd_phase_submit_seq
                items.append(
                    {
                        "phase": phase,
                        "index": index,
                        "future": future,
                        "scheduler_output": scheduler_output,
                        "exec_future": exec_future,
                        "submit_seq": float(submit_seq),
                    }
                )
        return items

    def _ttft_prefill_suspended_count(self) -> int:
        return len(self._ttft_prefill_suspended)

    def _ttft_prefill_owned_request_ids(self) -> set[str]:
        """Return requests whose activation is owned by one batch Future.

        External abort may revoke client output immediately, but its KV and
        scheduler state cannot be freed while a whole-batch continuation still
        retains the corresponding activation row. Such aborts are committed
        only after the batch reaches its final worker result.
        """

        owned_ids = self._ttft_prefill_suspended_request_ids()
        phase_queues = getattr(self, "pd_phase_queues", None)
        if phase_queues is None:
            return owned_ids
        for _, scheduler_output, _ in phase_queues.get("prefill", ()):
            if scheduler_output.ttft_layerwise_prefill_preemption:
                owned_ids.update(
                    self._scheduler_output_request_ids(scheduler_output)
                )
        return owned_ids

    def _flush_ttft_prefill_deferred_aborts(
        self,
        scheduler_output: SchedulerOutput,
    ) -> set[str]:
        """Finish client-aborted members after their whole batch completes."""

        output_request_ids = self._scheduler_output_request_ids(
            scheduler_output
        )
        deferred_ids = output_request_ids.intersection(
            self._ttft_prefill_deferred_abort_ids
        )
        if not deferred_ids:
            return set()
        self._ttft_prefill_deferred_abort_ids.difference_update(deferred_ids)
        self._ttft_prefill_deferred_abort_flush_count += len(deferred_ids)
        self.scheduler.finish_requests(
            deferred_ids,
            RequestStatus.FINISHED_ABORTED,
        )
        timings = self._iteration_debug_timings.setdefault(
            id(scheduler_output), {}
        )
        timings.update(
            {
                "ttft_layerwise_prefill_deferred_abort_count": float(
                    self._ttft_prefill_deferred_abort_count
                ),
                "ttft_layerwise_prefill_deferred_abort_flush_count": float(
                    self._ttft_prefill_deferred_abort_flush_count
                ),
                "ttft_layerwise_prefill_deferred_abort_batch_count": float(
                    len(deferred_ids)
                ),
            }
        )
        return deferred_ids

    def _discard_ttft_prefill_deferred_aborts(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Drop tombstones when another terminal path owns the whole batch."""

        self._ttft_prefill_deferred_abort_ids.difference_update(
            self._scheduler_output_request_ids(scheduler_output)
        )

    def _ttft_prefill_suspended_request_ids(self) -> set[str]:
        return {
            request_id
            for _, scheduler_output in self._ttft_prefill_suspended.values()
            for request_id in self._scheduler_output_request_ids(
                scheduler_output
            )
        }

    def _pd_owned_total_depth(self, phase_queues) -> int:
        """Count queued Futures plus worker slots retained at safe boundaries."""

        return (
            self._pd_phase_queue_total_depth(phase_queues)
            + sum(not bool(getattr(boundary, "kv_retention_parked_ticket", None))
                  for boundary, _ in self._ttft_prefill_suspended.values())
            + int(getattr(self, "_kv_retention_parked_offload_future", None) is not None)
        )

    def _pd_owned_phase_depth(self, phase_queues, phase: str) -> int:
        depth = len(phase_queues[phase])
        if phase == "prefill":
            depth += sum(not bool(getattr(boundary, "kv_retention_parked_ticket", None))
                         for boundary, _ in self._ttft_prefill_suspended.values())
            depth += int(getattr(self, "_kv_retention_parked_offload_future", None) is not None)
        return depth

    def _pd_phase_queue_pending_request_ids(self, phase_queues) -> set[str]:
        pending_ids: set[str] = set()
        for item in self._pd_phase_queue_items(phase_queues):
            pending_ids.update(
                self._scheduler_output_request_ids(item["scheduler_output"])
            )
        for _, scheduler_output in self._ttft_prefill_suspended.values():
            pending_ids.update(
                self._scheduler_output_request_ids(scheduler_output)
            )
        return pending_ids

    def _pd_phase_queue_pending_request_ids_by_phase(
        self,
        phase_queues,
    ) -> dict[str, set[str]]:
        pending_by_phase: dict[str, set[str]] = {}
        for item in self._pd_phase_queue_items(phase_queues):
            scheduler_output = item["scheduler_output"]
            request_ids = self._scheduler_output_request_ids(
                scheduler_output
            )
            pending_by_phase.setdefault(item["phase"], set()).update(
                request_ids
            )
            if (
                scheduler_output.ttft_layerwise_prefill_preemption
                and item["future"] is item["exec_future"]
            ):
                pending_by_phase.setdefault(
                    "prefill_layerwise_active", set()
                ).update(request_ids)
            final_prefill_ids = set(
                getattr(
                    scheduler_output,
                    "pd_ttft_handoff_request_ids",
                    (),
                )
            )
            if final_prefill_ids:
                pending_by_phase.setdefault("prefill_final", set()).update(
                    final_prefill_ids
                )
        for _, scheduler_output in self._ttft_prefill_suspended.values():
            request_ids = self._scheduler_output_request_ids(scheduler_output)
            pending_by_phase.setdefault("prefill", set()).update(request_ids)
            pending_by_phase.setdefault("prefill_suspended", set()).update(
                request_ids
            )
            final_prefill_ids = set(
                scheduler_output.pd_ttft_handoff_request_ids
            )
            if final_prefill_ids:
                pending_by_phase.setdefault("prefill_final", set()).update(
                    final_prefill_ids
                )
        return pending_by_phase

    def _pd_ttft_handoff_prune(self) -> None:
        """Drop handoff fences for requests removed by abort/finish paths."""

        if not self._pd_ttft_handoff_request_ids:
            return
        requests = getattr(self.scheduler, "requests", None)
        if isinstance(requests, dict):
            self._pd_ttft_handoff_request_ids.intersection_update(requests)
        if not self._pd_ttft_handoff_request_ids:
            self._pd_ttft_handoff_started_at = 0.0

    def _pd_ttft_handoff_active(self) -> bool:
        self._pd_ttft_handoff_prune()
        return bool(self._pd_ttft_handoff_request_ids)

    def _pd_ttft_handoff_mark_submitted(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        request_ids = set(
            getattr(scheduler_output, "pd_ttft_handoff_request_ids", ())
        )
        if not request_ids:
            return
        self._pd_ttft_handoff_request_ids.update(request_ids)
        if self._pd_ttft_handoff_started_at <= 0.0:
            self._pd_ttft_handoff_started_at = time.monotonic()
        if scheduler_output.iteration_debug is not None:
            scheduler_output.iteration_debug.update(
                {
                    "pd_ttft_handoff_armed": 1.0,
                    "pd_ttft_handoff_request_count": float(len(request_ids)),
                    "pd_ttft_handoff_active_request_count": float(
                        len(self._pd_ttft_handoff_request_ids)
                    ),
                }
            )

    def _pd_ttft_handoff_clear_from_outputs(
        self,
        phase_queues,
        engine_core_outputs: dict[int, EngineCoreOutputs],
    ) -> None:
        """Release a handoff fence only after first-token/terminal output."""

        if not self._pd_ttft_handoff_request_ids:
            return
        completed_request_ids: set[str] = set()
        for client_outputs in engine_core_outputs.values():
            completed_request_ids.update(client_outputs.finished_requests or ())
            for output in client_outputs.outputs:
                request_id = output.request_id
                if request_id not in self._pd_ttft_handoff_request_ids:
                    continue
                if output.new_token_ids or output.finish_reason is not None:
                    completed_request_ids.add(request_id)
        if not completed_request_ids:
            return
        cleared_request_ids = (
            self._pd_ttft_handoff_request_ids & completed_request_ids
        )
        self._pd_ttft_handoff_request_ids.difference_update(
            completed_request_ids
        )
        if not self._pd_ttft_handoff_request_ids:
            self._pd_ttft_handoff_started_at = 0.0
        self._write_pd_phase_queue_trace(
            "pd_ttft_handoff_released",
            phase_queues,
            {
                "pd_ttft_handoff_released_request_count": float(
                    len(cleared_request_ids)
                ),
                "pd_ttft_handoff_active_request_count": float(
                    len(self._pd_ttft_handoff_request_ids)
                ),
            },
        )

    def _set_scheduler_pd_pending_request_ids(self, request_ids: set[str]) -> None:
        set_pending = getattr(self.scheduler, "set_pd_pending_request_ids", None)
        if set_pending is not None:
            set_pending(request_ids)

    def _set_scheduler_pd_pending_phase_request_ids(
        self,
        request_ids_by_phase: dict[str, set[str]] | None,
    ) -> None:
        set_pending = getattr(
            self.scheduler,
            "set_pd_pending_phase_request_ids",
            None,
        )
        if set_pending is not None:
            set_pending(request_ids_by_phase or {})

    @staticmethod
    def _pd_phase_queue_total_depth(phase_queues) -> int:
        return sum(len(queue_) for queue_ in phase_queues.values())

    def _pd_capacity_retry_free_blocks(self) -> int:
        get_free_blocks = getattr(
            self.scheduler,
            "get_pd_capacity_retry_free_blocks",
            None,
        )
        if not callable(get_free_blocks):
            return -1
        return int(get_free_blocks())

    def _pd_capacity_retry_can_submit(self, phase: str) -> bool:
        can_retry = self._pd_capacity_retry_gate.can_retry(
            phase,
            self._pd_capacity_retry_free_blocks(),
        )
        if not can_retry:
            self._pd_capacity_retry_suppressed_this_step = True
        return can_retry

    def _record_pd_capacity_retry_block(
        self,
        phase: str,
        scheduler_output: SchedulerOutput,
    ) -> None:
        if not scheduler_output.pd_capacity_retry_blocked:
            return
        blocked = self._pd_capacity_retry_gate.block(
            phase,
            minimum_free_blocks=(
                scheduler_output.pd_capacity_retry_min_free_blocks
            ),
            observed_free_blocks=(
                scheduler_output.pd_capacity_retry_observed_free_blocks
            ),
            reason=scheduler_output.pd_capacity_retry_reason,
            scheduler_iteration_id=scheduler_output.scheduler_iteration_id,
            cause=scheduler_output.pd_capacity_retry_cause,
            wake_on_output_progress=(
                scheduler_output.pd_capacity_retry_wake_on_output_progress
            ),
        )
        self._pd_capacity_retry_suppressed_this_step |= blocked

    def _pd_capacity_retry_snapshot(self) -> dict[str, int | str]:
        return self._pd_capacity_retry_gate.snapshot(
            self._pd_capacity_retry_free_blocks()
        )

    @staticmethod
    def _pd_engine_outputs_finished(
        engine_core_outputs: dict[int, EngineCoreOutputs],
    ) -> bool:
        return any(
            bool(output_group.finished_requests)
            or any(output.finished for output in output_group.outputs)
            for output_group in engine_core_outputs.values()
        )

    def _notify_pd_capacity_retry_after_output(
        self,
        scheduler_output: SchedulerOutput,
        engine_core_outputs: dict[int, EngineCoreOutputs],
        free_blocks_before: int,
    ) -> int:
        """Publish only output changes that can alter phase admission.

        A non-terminal Decode token does not release future ownership needed
        by the Prefill lane and normally consumes, rather than creates, KV
        capacity. Waking a no-progress Prefill gate for every such token turns
        one failed admission into a per-token scheduler/allocator scan.
        """

        free_blocks_after = self._pd_capacity_retry_free_blocks()
        capacity_increased = (
            free_blocks_before >= 0
            and free_blocks_after > free_blocks_before
        )
        return self._pd_capacity_retry_gate.notify_output_progress(
            completed_phase=self._scheduler_output_phase(scheduler_output),
            request_finished=self._pd_engine_outputs_finished(
                engine_core_outputs
            ),
            capacity_increased=capacity_increased,
        )

    def _pd_phase_queue_snapshot(self, phase_queues) -> dict[str, float | str]:
        self._pd_ttft_handoff_prune()
        snapshot: dict[str, float | str] = self._pd_phase_queues_counts(
            phase_queues
        )
        for phase in ("decode", "prefill", "mixed", "idle"):
            queue_ = phase_queues.get(phase, ())
            ready_count = sum(1 for queued_future, _, _ in queue_
                              if queued_future.done())
            depth = len(queue_)
            submit_seqs = [
                scheduler_output.pd_phase_submit_seq
                for _, scheduler_output, _ in queue_
            ]
            snapshot[f"pd_phase_queue_{phase}_not_ready_count"] = float(
                depth - ready_count
            )
            snapshot[f"pd_phase_queue_{phase}_oldest_submit_seq"] = float(
                min(submit_seqs) if submit_seqs else -1.0
            )
            snapshot[f"pd_phase_queue_{phase}_newest_submit_seq"] = float(
                max(submit_seqs) if submit_seqs else -1.0
            )
            snapshot[f"pd_phase_queue_{phase}_can_submit"] = float(
                self._pd_phase_queue_can_submit_phase(phase_queues, phase)
            )
        snapshot["pd_scheduler_has_requests"] = float(
            self.scheduler.has_requests()
        )
        snapshot["pd_scheduler_has_decode_work"] = float(
            self._scheduler_has_pd_phase_work(self.scheduler, "decode")
        )
        snapshot["pd_scheduler_has_prefill_work"] = float(
            self._scheduler_has_pd_phase_work(self.scheduler, "prefill")
        )
        snapshot.update(self._pd_capacity_retry_snapshot())
        snapshot.update(
            {
                "pd_future_liveness_audit_enabled": float(
                    self.pd_future_liveness_audit_enabled
                ),
                "pd_future_liveness_audit_interval_s": float(
                    self.pd_future_liveness_audit_interval_s
                ),
                "pd_future_liveness_audit_total": float(
                    self._pd_future_liveness_audit_total
                ),
                "pd_ttft_handoff_active": float(
                    bool(self._pd_ttft_handoff_request_ids)
                ),
                "pd_ttft_handoff_request_count": float(
                    len(self._pd_ttft_handoff_request_ids)
                ),
                "pd_ttft_handoff_age_ms": (
                    max(
                        0.0,
                        (time.monotonic() - self._pd_ttft_handoff_started_at)
                        * 1000.0,
                    )
                    if self._pd_ttft_handoff_started_at > 0.0
                    else 0.0
                ),
                "pd_ttft_handoff_blocked_prefill_checks": float(
                    self._pd_ttft_handoff_blocked_prefill_checks
                ),
            }
        )
        return snapshot

    def _write_pd_phase_queue_trace(
        self,
        event: str,
        phase_queues,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.pd_phase_queue_trace_jsonl:
            return
        now = time.monotonic()
        heartbeat_events = {
            "nonblocking_yield",
            "idle_empty",
            "idle_no_submit",
        }
        if event in heartbeat_events:
            last_ts = self._pd_phase_queue_trace_last_heartbeat_ts.get(
                event, -1.0
            )
            if (
                last_ts >= 0.0
                and now - last_ts
                < self.pd_phase_queue_trace_heartbeat_interval_s
            ):
                self._pd_phase_queue_trace_suppressed_heartbeats[event] = (
                    self._pd_phase_queue_trace_suppressed_heartbeats.get(
                        event, 0
                    )
                    + 1
                )
                return
            self._pd_phase_queue_trace_last_heartbeat_ts[event] = now
        suppressed_heartbeats = self._pd_phase_queue_trace_suppressed_heartbeats.pop(
            event, 0
        )
        record: dict[str, Any] = {
            "timestamp": now,
            "event": event,
            "suppressed_heartbeat_count": suppressed_heartbeats,
            "pd_muxwise_event_loop_enabled": float(
                self.pd_muxwise_event_loop_enabled
            ),
            "pd_independent_phase_queues_enabled": float(
                self.pd_independent_phase_queues_enabled
            ),
            "pd_separate_scheduler_lanes_enabled": float(
                self.pd_separate_scheduler_lanes_enabled
            ),
            "pd_dual_continuous_batching_enabled": float(
                self.pd_dual_continuous_batching_enabled
            ),
            "pd_nonblocking_phase_recovery_enabled": float(
                self.pd_nonblocking_phase_recovery_enabled
            ),
            "pd_single_inflight_per_phase_enabled": float(
                self.pd_single_inflight_per_phase_enabled
            ),
            "pd_phase_depth_enabled": float(self.pd_phase_depth_enabled),
            "pd_prefill_inflight_depth": float(self.pd_prefill_inflight_depth),
            "pd_decode_inflight_depth": float(self.pd_decode_inflight_depth),
            "pd_global_inflight_depth": float(self.pd_global_inflight_depth),
            "pd_dual_continuous_decode_reserved_batches": float(
                self.pd_decode_reserved_batches
            ),
            "pd_dual_continuous_prefill_capacity": float(
                self._pd_phase_queue_prefill_capacity()
            ),
            "pd_paired_phase_launch_requested": float(
                self.pd_paired_phase_launch_requested
            ),
            "pd_paired_phase_launch_enabled": float(
                self.pd_paired_phase_launch_enabled
            ),
            "pd_decoupled_paired_recovery_requested": float(
                self.pd_decoupled_paired_recovery_requested
            ),
            "pd_decoupled_paired_recovery_enabled": float(
                self.pd_decoupled_paired_recovery_enabled
            ),
            **self._pd_phase_queue_snapshot(phase_queues),
        }
        if extra:
            record.update(extra)
        self._append_jsonl(self.pd_phase_queue_trace_jsonl, record)

    def _pd_phase_queue_prefill_capacity(self) -> int:
        if self.pd_phase_depth_enabled:
            return self.pd_prefill_inflight_depth
        if not self.pd_dual_continuous_batching_enabled:
            return self.batch_queue_size
        return max(0, self.batch_queue_size - self.pd_decode_reserved_batches)

    def _pd_phase_inflight_limit(self, phase: str) -> int:
        if not self.pd_phase_depth_enabled:
            return self.batch_queue_size
        if phase == "prefill":
            return self.pd_prefill_inflight_depth
        if phase == "decode":
            return self.pd_decode_inflight_depth
        return self.pd_global_inflight_depth

    def _pd_phase_queue_can_submit_phase(self, phase_queues, phase: str) -> bool:
        if self._pd_layerwise_handoff_fence_active:
            # A recorded CUDA event is not a completed profile move. Keep the
            # source Decode lane live while the source Prefill layer executes;
            # close it only after the exact event completes, then drain the
            # depth-one source Decode Future before committing the target.
            allow_source_decode, _ = (
                PD_LAYERWISE_HANDOFF_CONTROLLER.source_decode_admission(
                    source_decode_sms=(
                        self.pd_sm_transition.source_decode_sms
                        if self.pd_sm_transition.active
                        else 0
                    )
                )
            )
            if phase == "decode":
                self._pd_layerwise_handoff_source_decode_admission_checks += 1
                if allow_source_decode:
                    self._pd_layerwise_handoff_source_decode_admission_allowed += 1
                else:
                    self._pd_layerwise_handoff_fence_blocked_decode_checks += 1
                    return False
            else:
                if phase == "prefill":
                    self._pd_layerwise_handoff_fence_blocked_prefill_checks += 1
                return False
        if (
            self.pd_dynamic_sm_enforced
            and self.pd_sm_transition.active
            and not self.pd_sm_transition.allows_phase(phase)
        ):
            self.pd_sm_transition.note_blocked(phase)
            return False
        if phase == "prefill" and any(
            queued_output.ttft_layerwise_prefill_preemption
            and queued_future is exec_future
            and not queued_future.done()
            for queued_future, queued_output, exec_future in (
                phase_queues["prefill"]
            )
        ):
            # Keep the second Prefill slot available for a genuinely shorter
            # SRPT arrival. EngineCore arms the running Future first; only
            # after its layer boundary is recovered may the new request take
            # that slot. Queueing lower-priority lookahead here would make the
            # later Short inadmissible even though P=2 was configured.
            return False
        if phase == "prefill" and self._pd_ttft_handoff_active():
            handoff_ids_are_safely_suspended = bool(
                self.ttft_layerwise_prefill_preemption_enabled
                and self._pd_ttft_handoff_request_ids
                <= self._ttft_prefill_suspended_request_ids()
            )
            if not handoff_ids_are_safely_suspended:
                self._pd_ttft_handoff_blocked_prefill_checks += 1
                return False
        if self._pd_owned_total_depth(phase_queues) >= self.batch_queue_size:
            return False
        if (
            self.pd_phase_depth_enabled
            and self._pd_owned_phase_depth(phase_queues, phase)
            >= self._pd_phase_inflight_limit(phase)
        ):
            return False
        if self.pd_dynamic_sm_enforced and phase in {"decode", "prefill"}:
            current_profile = self.pd_sm_allocator.current_profile
            phase_sms = (
                current_profile.decode_sms
                if phase == "decode"
                else current_profile.prefill_sms
            )
            # A solo profile cannot accept work for the opposite phase while
            # the current profile still owns an in-flight future. Wait until a
            # quiescent boundary, then switch both streams to a paired profile.
            if (
                phase_sms == 0
                and self._pd_phase_queue_total_depth(phase_queues) > 0
            ):
                return False
        if (
            self.pd_single_inflight_per_phase_enabled
            and len(phase_queues[phase]) > 0
        ):
            return False
        if (
            self.pd_dual_continuous_batching_enabled
            and phase == "prefill"
            and self._pd_owned_phase_depth(phase_queues, "prefill")
            >= self._pd_phase_queue_prefill_capacity()
        ):
            return False
        return True

    def _pd_sm_transition_future_counts(
        self, phase_queues
    ) -> tuple[int, int, int]:
        """Count source/older and unexpected newer-epoch futures."""

        if not self.pd_sm_transition.active:
            return 0, 0, 0
        source_epoch = self.pd_sm_transition.source_epoch
        decode_count = 0
        prefill_count = 0
        newer_epoch_count = 0
        pending_copy = getattr(self, "_kv_retention_parked_offload_future", None)
        if pending_copy is not None:
            # D2H is physical Prefill work until its terminal ACK is consumed,
            # even though the parked model itself is not an execution slot.
            if int(pending_copy[2].pd_sm_profile_epoch) > source_epoch:
                newer_epoch_count += 1
            else:
                prefill_count += 1
        for phase, queue_ in phase_queues.items():
            for _, scheduler_output, _ in queue_:
                output_epoch = int(
                    getattr(scheduler_output, "pd_sm_profile_epoch", -1)
                )
                if output_epoch > source_epoch:
                    newer_epoch_count += 1
                    continue
                if phase == "decode":
                    decode_count += 1
                else:
                    # Conservatively drain Prefill, mixed, and idle outputs
                    # before changing a whole-forward GreenContext profile.
                    prefill_count += 1
        return decode_count, prefill_count, newer_epoch_count

    def _pd_layerwise_transition_boundary_ready(
        self,
        phase_queues,
        *,
        old_decode: int,
        old_prefill: int,
        newer_epoch: int,
    ) -> bool:
        """Prove that one migrated Prefill future is the only old work.

        The worker-side controller is the CUDA boundary ACK. The queue proof
        prevents a target epoch from becoming admissible while any other
        source-profile future can still execute.
        """

        transition = self.pd_sm_transition
        if (
            not self.pd_layerwise_handoff_enabled
            or not self._pd_layerwise_handoff_fence_active
            or not transition.active
            or old_decode != 0
            or old_prefill != 1
            or newer_epoch != 0
            or self._pd_phase_queue_total_depth(phase_queues) != 1
            or len(phase_queues["prefill"]) != 1
            or transition.source_prefill_sms <= 0
            or transition.target_prefill_sms <= 0
        ):
            return False
        return PD_LAYERWISE_HANDOFF_CONTROLLER.completed_transition_matches(
            source_profile_id=transition.source_profile_id,
            source_epoch=transition.source_epoch,
            target_profile_id=transition.target_profile_id,
            target_epoch=transition.source_epoch + 1,
        )

    def _advance_pd_sm_transition(self, phase_queues) -> bool:
        """Commit at quiescence or an acknowledged Prefill layer boundary."""

        transition = self.pd_sm_transition
        if not self.pd_dynamic_sm_enforced or not transition.active:
            return False
        old_decode, old_prefill, newer_epoch = (
            self._pd_sm_transition_future_counts(phase_queues)
        )
        global_queue_depth = self._pd_phase_queue_total_depth(phase_queues)
        global_queue_depth += int(
            getattr(self, "_kv_retention_parked_offload_future", None) is not None)
        layerwise_boundary_ready = (
            self._pd_layerwise_transition_boundary_ready(
                phase_queues,
                old_decode=old_decode,
                old_prefill=old_prefill,
                newer_epoch=newer_epoch,
            )
        )
        ready = transition.advance(
            old_decode_futures=old_decode,
            old_prefill_futures=old_prefill,
            global_queue_depth=global_queue_depth,
            newer_epoch_futures=newer_epoch,
        )
        if not ready and not layerwise_boundary_ready:
            if transition.stage == "drain_all" and global_queue_depth > 0:
                self._write_pd_phase_queue_trace(
                    "pd_sm_transition_drain_continues",
                    phase_queues,
                    transition.debug_snapshot(),
                )
            return False
        source_profile_id = transition.source_profile_id
        source_epoch = transition.source_epoch
        target_profile_id = transition.target_profile_id
        layerwise_bank_debug: dict[str, int | float | str] = {}
        if layerwise_boundary_ready:
            expected_epoch = source_epoch + 1
            try:
                if (
                    self.pd_sm_allocator.epoch != source_epoch
                    or self.pd_sm_allocator.current_profile.profile_id
                    != source_profile_id
                ):
                    raise RuntimeError(
                        "layerwise allocator source authority changed before "
                        "GreenContext commit: "
                        f"allocator="
                        f"{self.pd_sm_allocator.current_profile.profile_id}"
                        f"@{self.pd_sm_allocator.epoch} transition="
                        f"{source_profile_id}@{source_epoch}"
                    )
                bank_results = self.model_executor.collective_rpc(
                    "commit_dynamic_green_context_profile_at_layer_boundary",
                    args=(
                        source_profile_id,
                        source_epoch,
                        target_profile_id,
                        expected_epoch,
                    ),
                )
                bank_commit_matches = bool(bank_results) and all(
                    isinstance(result, dict)
                    and int(result.get("committed", 0)) == 1
                    and result.get("profile_id") == target_profile_id
                    and int(result.get("profile_epoch", -1)) == expected_epoch
                    for result in bank_results
                )
                if not bank_commit_matches:
                    raise RuntimeError(
                        "layerwise GreenContext bank commit did not match the "
                        "completed worker boundary: "
                        f"{source_profile_id}@{source_epoch} -> "
                        f"{target_profile_id}@{expected_epoch}; "
                        f"workers={bank_results!r}"
                    )
                layerwise_bank_debug = {
                    "pd_layerwise_handoff_bank_commit_count": float(
                        sum(
                            int(result.get("committed", 0))
                            for result in bank_results
                            if isinstance(result, dict)
                        )
                    ),
                    "pd_layerwise_handoff_bank_committed_profile": (
                        target_profile_id
                    ),
                    "pd_layerwise_handoff_bank_committed_epoch": float(
                        expected_epoch
                    ),
                }
                committed = self.pd_sm_allocator.commit_profile_at_boundary(
                    target_profile_id,
                    expected_epoch=transition.source_epoch,
                )
            except BaseException:
                # Never strand the model worker on its generation-owned resume
                # event during fatal authority/commit failure.  Waking it makes
                # its exact committed-transition assertion fail as well, so no
                # target continuation can run under split authority.
                PD_LAYERWISE_HANDOFF_CONTROLLER.finish_engine_transition(
                    source_profile_id=source_profile_id,
                    source_epoch=source_epoch,
                )
                self._pd_layerwise_handoff_fence_active = False
                raise
        else:
            committed = self.pd_sm_allocator.commit_profile_at_quiescence(
                target_profile_id,
                expected_epoch=transition.source_epoch,
            )
        # A policy-blocked phase may become admissible under the new profile.
        # Exact capacity observations remain armed by notify_state_change().
        self._pd_capacity_retry_gate.notify_state_change(
            "sm_profile_committed"
        )
        self._pd_sm_transition_commit_count += 1
        commit_debug = transition.finish()
        layerwise_debug = (
            PD_LAYERWISE_HANDOFF_CONTROLLER.snapshot()
            if self.pd_layerwise_handoff_enabled
            else {}
        )
        if layerwise_boundary_ready:
            acknowledged = (
                PD_LAYERWISE_HANDOFF_CONTROLLER.acknowledge_completed_transition(
                    source_profile_id=source_profile_id,
                    source_epoch=source_epoch,
                    target_profile_id=target_profile_id,
                    target_epoch=source_epoch + 1,
                )
            )
            if not acknowledged:
                PD_LAYERWISE_HANDOFF_CONTROLLER.finish_engine_transition(
                    source_profile_id=source_profile_id,
                    source_epoch=source_epoch,
                )
                self._pd_layerwise_handoff_fence_active = False
                raise RuntimeError(
                    "layerwise P/D profile commit lost its worker boundary ACK: "
                    f"{source_profile_id}@{source_epoch} -> "
                    f"{target_profile_id}@{source_epoch + 1}"
                )
        elif self.pd_layerwise_handoff_enabled:
            PD_LAYERWISE_HANDOFF_CONTROLLER.finish_engine_transition(
                source_profile_id=source_profile_id,
                source_epoch=source_epoch,
            )
        self._pd_layerwise_handoff_fence_active = False
        self._pd_sm_transition_wait_ms_total += float(
            commit_debug.get("pd_sm_transition_wait_ms", 0.0) or 0.0
        )
        self._pd_sm_transition_prefill_drain_ms_total += float(
            commit_debug.get("pd_sm_transition_prefill_drain_ms", 0.0)
            or 0.0
        )
        self._pd_sm_transition_all_fence_ms_total += float(
            commit_debug.get("pd_sm_transition_all_fence_ms", 0.0) or 0.0
        )
        commit_debug.update(
            {
                **layerwise_debug,
                **layerwise_bank_debug,
                "pd_sm_transition_source_profile": source_profile_id,
                "pd_sm_transition_committed_profile": committed.profile_id,
                "pd_sm_transition_committed_epoch": float(
                    self.pd_sm_allocator.epoch
                ),
                "pd_sm_transition_commit_count": float(
                    self._pd_sm_transition_commit_count
                ),
                "pd_sm_transition_layerwise_boundary_commit": float(
                    layerwise_boundary_ready
                ),
                "pd_sm_transition_wait_ms_total": (
                    self._pd_sm_transition_wait_ms_total
                ),
                "pd_sm_transition_prefill_drain_ms_total": (
                    self._pd_sm_transition_prefill_drain_ms_total
                ),
                "pd_sm_transition_all_fence_ms_total": (
                    self._pd_sm_transition_all_fence_ms_total
                ),
                "pd_layerwise_handoff_fence_blocked_decode_checks": float(
                    self._pd_layerwise_handoff_fence_blocked_decode_checks
                ),
                "pd_layerwise_handoff_fence_blocked_prefill_checks": float(
                    self._pd_layerwise_handoff_fence_blocked_prefill_checks
                ),
                "pd_layerwise_handoff_source_decode_admission_checks": float(
                    self._pd_layerwise_handoff_source_decode_admission_checks
                ),
                "pd_layerwise_handoff_source_decode_admission_allowed": float(
                    self._pd_layerwise_handoff_source_decode_admission_allowed
                ),
            }
        )
        self._pd_sm_transition_last_commit_debug = commit_debug
        if getattr(self, "_tbt_capture", None) is not None:
            self._tbt_capture.write("transition_commit", debug=dict(commit_debug))
        if getattr(self.pd_sm_allocator, "predictive_enabled", False):
            self.pd_sm_allocator.note_tbt_transition_commit(commit_debug)
        self._tbt_controller_state.update(commit_debug)
        self._pd_sm_transition_resume_pending = self.scheduler.has_requests()
        self._pd_sm_transition_resume_commit_ts = float(
            commit_debug.get("pd_sm_transition_committed_ts", 0.0) or 0.0
        )
        if self.pd_sm_allocator.config.policy == "tbt_slo_feedback":
            reset_feedback_window = getattr(
                self.scheduler, "reset_tbt_feedback_window", None
            )
            if callable(reset_feedback_window):
                reset_feedback_window()
        self._write_pd_phase_queue_trace(
            "pd_sm_transition_committed", phase_queues, commit_debug
        )
        return True

    def _request_pd_sm_transition(
        self,
        plan: PDSMPlan,
        *,
        layerwise_decode_backlog_empty: bool = False,
    ) -> None:
        if (
            not self.pd_dynamic_sm_enforced
            or plan.shadow
            or not plan.deferred
            or plan.reason != "inflight_profile_hold"
            or plan.target_profile == plan.profile
        ):
            return
        if self._pd_layerwise_handoff_fence_active:
            # The in-flight Prefill worker already owns the requested target.
            # Do not supersede it until EngineCore commits that exact target
            # at its acknowledged layer boundary.
            return
        target = plan.target_profile
        self.pd_sm_transition.request(
            source_profile_id=plan.profile.profile_id,
            source_epoch=plan.epoch,
            source_decode_sms=plan.profile.decode_sms,
            source_prefill_sms=plan.profile.prefill_sms,
            target_profile_id=target.profile_id,
            target_decode_sms=target.decode_sms,
            target_prefill_sms=target.prefill_sms,
            reason=(
                "layerwise_decode_idle_prefill_only"
                if layerwise_decode_backlog_empty
                and target.decode_sms == 0
                else plan.reason
            ),
            urgent=plan.tbt_slo_emergency or plan.tbt_slo_overload,
            decode_backlog_empty=layerwise_decode_backlog_empty,
        )
        if getattr(self.pd_sm_allocator, "predictive_enabled", False):
            self.pd_sm_allocator.note_tbt_transition_request(
                requested_at=self.pd_sm_transition.requested_at)
        if getattr(self, "_tbt_capture", None) is not None:
            self._tbt_capture.write(
                "transition_request", requested_at=self.pd_sm_transition.requested_at,
                source_profile=plan.profile.profile_id,
                target_profile=target.profile_id,
                prefills=self._tbt_capture.prefills(self.pd_phase_queues))

    def _maybe_request_pd_layerwise_handoff(
        self,
        *,
        decode_inflight: bool,
        prefill_inflight: bool,
    ) -> None:
        """Apply any safe profile move to an active Prefill continuation."""

        if (
            not self.pd_layerwise_handoff_enabled
            or self._pd_layerwise_handoff_fence_active
            or not self.pd_sm_transition.active
            or decode_inflight
            or not prefill_inflight
        ):
            return
        transition = self.pd_sm_transition
        snapshot = PD_LAYERWISE_HANDOFF_CONTROLLER.snapshot()
        eligible, _ = layerwise_profile_handoff_decision(
            armed=bool(snapshot.get("pd_layerwise_handoff_armed", 0.0)),
            prefill_inflight=prefill_inflight,
            decode_inflight=decode_inflight,
            source_prefill_sms=transition.source_prefill_sms,
            target_prefill_sms=transition.target_prefill_sms,
        )
        if not eligible:
            return
        accepted = PD_LAYERWISE_HANDOFF_CONTROLLER.request(
            source_profile_id=transition.source_profile_id,
            source_epoch=transition.source_epoch,
            target_profile_id=transition.target_profile_id,
            target_epoch=transition.source_epoch + 1,
        )
        if accepted:
            self._pd_layerwise_handoff_fence_active = True

    def _poll_pd_layerwise_handoff(self, phase_queues) -> None:
        """Re-evaluate every profile move while one Prefill future runs."""

        if getattr(getattr(self, "pd_sm_allocator", None), "predictive_enabled", False):
            # This control pass follows Decode output recovery even if no
            # Prefill is newly submitted. It also runs after a deadline wake.
            # Do not wait for the entire Prefill Future to terminate.
            self._assign_pd_sm_plan(
                phase_queues, force_prefill_ready=bool(phase_queues["prefill"]),
                layerwise_decode_backlog_empty=not (
                    phase_queues["decode"] or self._scheduler_has_pd_phase_work(self.scheduler, "decode")))
            return

        if getattr(self, "kv_retention_p1_parked_enabled", False):
            # One Engine-visible GPU-complete boundary serves both request
            # yields and SM moves. Never arm the legacy blocking worker gate.
            if phase_queues["prefill"]:
                self._assign_pd_sm_plan(
                    phase_queues, force_prefill_ready=True,
                    layerwise_decode_backlog_empty=not (
                        phase_queues["decode"] or self._scheduler_has_pd_phase_work(self.scheduler, "decode")))
            return

        if (
            not self.pd_layerwise_handoff_enabled
            or self._pd_layerwise_handoff_fence_active
        ):
            return
        snapshot = PD_LAYERWISE_HANDOFF_CONTROLLER.snapshot()
        if not bool(snapshot.get("pd_layerwise_handoff_armed", 0.0)):
            return
        decode_inflight = bool(phase_queues["decode"])
        prefill_inflight = bool(phase_queues["prefill"])
        decode_ready = self._scheduler_has_pd_phase_work(
            self.scheduler, "decode"
        )
        if not prefill_inflight:
            return
        if decode_inflight:
            self._pd_layerwise_handoff_decode_inflight_defer_checks += 1
            return

        # Re-run the allocator even for an active transition. A newly ready
        # Decode may retarget p40 to a concurrent profile; Decode becoming idle
        # may retarget that profile back to p40. No target is admitted until
        # the worker ACKs a real Prefill layer boundary.
        self._assign_pd_sm_plan(
            phase_queues,
            force_prefill_ready=True,
            layerwise_decode_backlog_empty=not decode_ready,
        )
        if self._pd_layerwise_handoff_fence_active:
            self._pd_layerwise_handoff_prefill_only_retarget_count += 1
        else:
            self._pd_layerwise_handoff_prefill_only_retarget_miss_count += 1
        self._tbt_controller_state.update(
            {
                "pd_layerwise_handoff_decode_inflight_defer_checks": float(
                    self._pd_layerwise_handoff_decode_inflight_defer_checks
                ),
                "pd_layerwise_handoff_decode_backlog_defer_checks": float(
                    self._pd_layerwise_handoff_decode_backlog_defer_checks
                ),
                "pd_layerwise_handoff_prefill_only_retarget_count": float(
                    self._pd_layerwise_handoff_prefill_only_retarget_count
                ),
                "pd_layerwise_handoff_prefill_only_retarget_miss_count": float(
                    self._pd_layerwise_handoff_prefill_only_retarget_miss_count
                ),
                "pd_layerwise_handoff_source_decode_admission_checks": float(
                    self._pd_layerwise_handoff_source_decode_admission_checks
                ),
                "pd_layerwise_handoff_source_decode_admission_allowed": float(
                    self._pd_layerwise_handoff_source_decode_admission_allowed
                ),
            }
        )

    def _record_pd_sm_transition_resume_submission(
        self,
        phase_queues,
        scheduler_output: SchedulerOutput,
        phase: str,
    ) -> None:
        """Record commit-to-first-new-future host admission delay."""

        if (phase == "decode" and scheduler_output.total_num_scheduled_tokens > 0
                and getattr(getattr(self, "pd_sm_allocator", None), "predictive_enabled", False)):
            # A Prefill resume must not consume the new-Decode timing sample.
            self.pd_sm_allocator.note_tbt_decode_submission(
                profile_id=scheduler_output.pd_sm_profile_id,
                epoch=scheduler_output.pd_sm_profile_epoch, now=time.monotonic())
        if (
            not self._pd_sm_transition_resume_pending
            or scheduler_output.total_num_scheduled_tokens <= 0
        ):
            return
        resumed_ts = time.monotonic()
        delay_ms = max(
            0.0,
            (resumed_ts - self._pd_sm_transition_resume_commit_ts) * 1000.0,
        )
        self._pd_sm_transition_resume_pending = False
        self._pd_sm_transition_resume_submit_count += 1
        self._pd_sm_transition_resume_submit_ms_total += delay_ms
        self._pd_sm_transition_resume_submit_ms_max = max(
            self._pd_sm_transition_resume_submit_ms_max,
            delay_ms,
        )
        resume_debug: dict[str, int | float | str] = {
            "pd_sm_transition_resumed_ts": resumed_ts,
            "pd_sm_transition_resume_submit_delay_ms": delay_ms,
            "pd_sm_transition_resume_submit_phase": phase,
            "pd_sm_transition_resume_submit_profile": (
                scheduler_output.pd_sm_profile_id
            ),
            "pd_sm_transition_resume_submit_epoch": float(
                scheduler_output.pd_sm_profile_epoch
            ),
            "pd_sm_transition_resume_submit_count": float(
                self._pd_sm_transition_resume_submit_count
            ),
            "pd_sm_transition_resume_submit_ms_total": (
                self._pd_sm_transition_resume_submit_ms_total
            ),
            "pd_sm_transition_resume_submit_ms_max": (
                self._pd_sm_transition_resume_submit_ms_max
            ),
        }
        self._pd_sm_transition_last_commit_debug.update(resume_debug)
        self._tbt_controller_state.update(resume_debug)
        if scheduler_output.iteration_debug is not None:
            scheduler_output.iteration_debug.update(resume_debug)
        self._write_pd_phase_queue_trace(
            "pd_sm_transition_resumed",
            phase_queues,
            resume_debug,
        )

    @staticmethod
    def _pd_phase_execution_key(
        scheduler_output: SchedulerOutput,
        phase: str,
    ) -> str:
        slot = max(0, int(scheduler_output.pd_phase_slot))
        return phase if slot == 0 else f"{phase}:{slot}"

    def _bind_pd_phase_execution_key(
        self,
        scheduler_output: SchedulerOutput,
        phase: str,
    ) -> str:
        execution_key = self._pd_phase_execution_key(
            scheduler_output,
            phase,
        )
        bind_output_context = getattr(
            self.scheduler,
            "bind_pd_output_execution_context",
            None,
        )
        if bind_output_context is not None:
            bind_output_context(scheduler_output, execution_key)
        return execution_key

    def _pd_pending_phase_owner_overlap(
        self,
        phase_queues,
        scheduler_output: SchedulerOutput,
        phase: str,
    ) -> bool:
        """Detect a post-commit overlap with an existing phase owner.

        Scheduler token/KV state has already advanced when this method runs,
        so EngineCore cannot safely manufacture a continuation here. The
        Scheduler must exclude every pending-owned request before commit.
        Report any overlap so a stale or mixed-version Scheduler fails closed
        instead of executing duplicate request ownership.
        """

        request_ids = self._scheduler_output_request_ids(scheduler_output)
        pending_ids = {
            request_id
            for _, queued_output, _ in phase_queues.get(phase, ())
            for request_id in self._scheduler_output_request_ids(queued_output)
        }
        if phase == "prefill":
            pending_ids.update(
                request_id
                for _, suspended_output in self._ttft_prefill_suspended.values()
                for request_id in self._scheduler_output_request_ids(
                    suspended_output
                )
            )
        return bool(request_ids & pending_ids)

    def _assign_pd_phase_submission(
        self,
        phase_queues,
        scheduler_output: SchedulerOutput,
        phase: str,
    ) -> str:
        self._record_pd_sm_transition_resume_submission(
            phase_queues,
            scheduler_output,
            phase,
        )
        self._pd_phase_queue_submit_seq += 1
        scheduler_output.pd_phase_submit_seq = self._pd_phase_queue_submit_seq
        if phase == "decode":
            self.pd_sm_transition.note_decode_submission(
                scheduler_output.total_num_scheduled_tokens
            )
        scheduler_output.pd_phase_slot = 0
        if self.pd_phase_depth_enabled and phase in {"decode", "prefill"}:
            request_ids = self._scheduler_output_request_ids(scheduler_output)
            pending_overlap = self._pd_pending_phase_owner_overlap(
                phase_queues,
                scheduler_output,
                phase,
            )
            if pending_overlap:
                raise RuntimeError(
                    "Unsafe same-request phase submission without an ordered "
                    f"Prefill chain: phase={phase}, request_ids="
                    f"{sorted(request_ids)}."
                )
            used_slots = {
                int(queued_output.pd_phase_slot)
                for _, queued_output, _ in phase_queues[phase]
            }
            if phase == "prefill":
                used_slots.update(
                    int(suspended_output.pd_phase_slot)
                    for boundary, suspended_output in (
                        self._ttft_prefill_suspended.values()
                    )
                    if not getattr(boundary, "kv_retention_parked_ticket", None)
                )
            slot = next(
                (
                    candidate
                    for candidate in range(self._pd_phase_inflight_limit(phase))
                    if candidate not in used_slots
                ),
                None,
            )
            if slot is None:
                raise RuntimeError(f"No free P/D phase slot for {phase}.")
            scheduler_output.pd_phase_slot = slot
        if getattr(self, "_tbt_capture", None) is not None:
            self._tbt_capture.submit(scheduler_output, phase,
                                     self.scheduler.requests, phase_queues)
        return self._bind_pd_phase_execution_key(
            scheduler_output,
            phase,
        )

    def _annotate_pd_component_overlap_candidate(
        self,
        phase_queues,
        scheduler_output: SchedulerOutput,
        *,
        paired_prefill_submit: bool = False,
    ) -> None:
        """Snapshot Prefill work that may overlap a submitted Decode."""
        if not self.full_iteration_debug_enabled:
            return
        phase = self._scheduler_output_phase(scheduler_output)
        prefill_depth = len(phase_queues["prefill"])
        prefill_not_ready = sum(
            1
            for queued_future, _, _ in phase_queues["prefill"]
            if not queued_future.done()
        )
        if paired_prefill_submit:
            prefill_depth += 1
            prefill_not_ready += 1
        scheduler_output.iteration_debug.update(
            {
                "pd_component_prefill_inflight_at_submit": float(
                    prefill_depth
                ),
                "pd_component_prefill_not_ready_at_submit": float(
                    prefill_not_ready
                ),
                "pd_component_paired_prefill_submit": float(
                    paired_prefill_submit
                ),
                "pd_component_prefill_overlap_candidate": float(
                    phase == "decode" and prefill_not_ready > 0
                ),
            }
        )

    @staticmethod
    def _scheduler_output_has_work(scheduler_output: SchedulerOutput) -> bool:
        return (
            scheduler_output.total_num_scheduled_tokens > 0
            or bool(scheduler_output.finished_req_ids)
        )

    @staticmethod
    def _scheduler_output_can_pair(scheduler_output: SchedulerOutput) -> bool:
        return (
            not scheduler_output.has_structured_output_requests
            and not scheduler_output.pending_structured_output_tokens
        )

    @staticmethod
    def _pd_sm_output_tokens(
        scheduler_output: SchedulerOutput,
    ) -> tuple[int, int]:
        debug = scheduler_output.iteration_debug or {}
        decode_tokens = int(debug.get("num_decode_tokens", 0) or 0)
        prefill_tokens = int(debug.get("num_prefill_tokens", 0) or 0)
        phase = EngineCore._scheduler_output_phase(scheduler_output)
        if phase == "decode" and decode_tokens <= 0:
            decode_tokens = scheduler_output.total_num_scheduled_tokens
        elif phase == "prefill" and prefill_tokens <= 0:
            prefill_tokens = scheduler_output.total_num_scheduled_tokens
        return max(0, decode_tokens), max(0, prefill_tokens)

    @staticmethod
    def _pd_sm_output_tbt_state(
        *scheduler_outputs: SchedulerOutput,
    ) -> tuple[int, int, str, float, float]:
        """Extract the conservative decode/TBT state carried by outputs."""

        decode_context_tokens = 0
        decode_request_ids: set[str] = set()
        tbt_constraints: list[tuple[float, float]] = []
        elapsed_without_slo_ms: list[float] = []
        prefill_mode = (
            "ring"
            if any(
                output.pd_prefill_mode == "ring"
                or getattr(
                    getattr(output, "kv_retention_layerwise_transfer", None),
                    "state", "",
                )
                in {"released", "published", "consumed"}
                for output in scheduler_outputs
            )
            else "native"
        )
        for output in scheduler_outputs:
            for request_id, record in (output.request_debug or {}).items():
                phase = str(record.get("phase", ""))
                decode_tokens = int(record.get("decode_tokens", 0) or 0)
                prefill_tokens = int(record.get("prefill_tokens", 0) or 0)
                if phase == "decode" or decode_tokens > 0:
                    decode_request_ids.add(request_id)
                    decode_context_tokens = max(
                        decode_context_tokens,
                        int(
                            record.get(
                                "num_computed_tokens_before", 0
                            )
                            or 0
                        ),
                    )
                    request_slo_ms = float(
                        record.get("tbt_slo_ms", -1.0) or -1.0
                    )
                    elapsed_ms = float(
                        record.get("tbt_current_elapsed_ms", 0.0) or 0.0
                    )
                    if request_slo_ms > 0.0:
                        tbt_constraints.append(
                            (request_slo_ms, max(0.0, elapsed_ms))
                        )
                    else:
                        elapsed_without_slo_ms.append(max(0.0, elapsed_ms))
                if prefill_tokens > 0 and (
                    int(
                        record.get(
                            "virtual_context_request_dram_pages", 0
                        )
                        or 0
                    )
                    > 0
                    or int(
                        record.get(
                            "virtual_context_deferred_logical_computed_tokens",
                            0,
                        )
                        or 0
                    )
                    > 0
                ):
                    prefill_mode = "ring"
        critical_slo_ms, critical_elapsed_ms = (
            min(
                tbt_constraints,
                key=lambda constraint: (
                    constraint[0] - constraint[1],
                    constraint[0],
                    -constraint[1],
                ),
            )
            if tbt_constraints
            else (
                0.0,
                max(elapsed_without_slo_ms)
                if elapsed_without_slo_ms
                else 0.0,
            )
        )
        return (
            decode_context_tokens,
            len(decode_request_ids),
            prefill_mode,
            critical_slo_ms,
            critical_elapsed_ms,
        )

    def _assign_pd_sm_plan(
        self,
        phase_queues,
        *scheduler_outputs: SchedulerOutput,
        force_prefill_ready: bool = False,
        layerwise_decode_backlog_empty: bool = False,
    ) -> PDSMPlan:
        # A zero-token SchedulerOutput is an admission observation, not ready
        # model work. Treating its nominal phase as runnable strands SMs on a
        # lane that cannot submit anything (for example d38/p2 while Prefill is
        # capacity-gated).
        output_phases = {
            self._scheduler_output_phase(output)
            for output in scheduler_outputs
            if output.total_num_scheduled_tokens > 0
        }
        invalid_work_phases = {
            self._scheduler_output_phase(output)
            for output in scheduler_outputs
            if output.total_num_scheduled_tokens > 0
            and self._scheduler_output_phase(output) not in {"decode", "prefill"}
        }
        if invalid_work_phases:
            raise RuntimeError(
                "dynamic P/D SM allocation requires phase-pure scheduler "
                f"outputs, got {sorted(invalid_work_phases)}"
            )
        suspended_prefill = None
        resume_reason = "none"
        if getattr(self, "_ttft_prefill_suspended", None):
            if "prefill" in output_phases:
                resume_reason = "prefill_selected"
            else:
                # Decode is submitted before the Prefill lane. A parked owner
                # is outside Scheduler.running/waiting and the physical queue,
                # but may resume later in this same control pass. See the same
                # pre-SM checks as that resume, without acquiring its slot or
                # modifying its original output/profile/binding here.
                # Anticipate the existing fallback resume after an unsuccessful
                # SRPT admission too; this does not reorder the normal SRPT pass.
                storage_key, resume_reason = self._ttft_prefill_resume_candidate(
                    phase_queues
                )
                if storage_key is not None:
                    suspended_prefill = self._ttft_prefill_suspended[
                        storage_key
                    ][1]
        decode_ready = "decode" in output_phases
        prefill_ready = "prefill" in output_phases
        if not decode_ready:
            decode_ready = self._scheduler_has_pd_phase_work(
                self.scheduler, "decode"
            )
        if (
            "decode" not in output_phases
            and self._pd_capacity_retry_gate.is_blocked(
                "decode",
                self._pd_capacity_retry_free_blocks(),
            )
        ):
            decode_ready = False
        if not prefill_ready:
            prefill_ready = self._scheduler_has_pd_phase_work(
                self.scheduler, "prefill"
            )
        if (
            "prefill" not in output_phases
            and not force_prefill_ready
            and self._pd_capacity_retry_gate.is_blocked(
                "prefill",
                self._pd_capacity_retry_free_blocks(),
            )
        ):
            # Logical Prefill work exists, but the last authoritative
            # admission pass proved that none can run at the current HBM
            # capacity.  Do not reserve Prefill SMs while only Decode can make
            # progress.
            prefill_ready = False
        # A gate for NEW Scheduler admission cannot erase this already-owned
        # continuation. Its original KV/frame (or private Ring receipt) remains
        # valid; its independent resume/sequence/slot checks passed above.
        prefill_ready = (
            prefill_ready
            or bool(force_prefill_ready)
            or suspended_prefill is not None
        )
        capacity_retry_free_blocks = self._pd_capacity_retry_free_blocks()
        prefill_capacity_serialized = (
            self._pd_capacity_retry_gate.blocking_cause(
                "prefill",
                capacity_retry_free_blocks,
            )
            == "prefill_kv_capacity"
        )

        decode_tokens = 0
        prefill_tokens = 0
        for output in scheduler_outputs:
            output_decode_tokens, output_prefill_tokens = self._pd_sm_output_tokens(
                output
            )
            decode_tokens += output_decode_tokens
            prefill_tokens += output_prefill_tokens
        if suspended_prefill is not None:
            prefill_tokens += suspended_prefill.total_num_scheduled_tokens
        if decode_ready and decode_tokens <= 0:
            decode_tokens = 1
        if prefill_ready and prefill_tokens <= 0:
            prefill_tokens = 1

        other_inflight = self._pd_phase_queue_total_depth(phase_queues) > (
            len(phase_queues["decode"]) + len(phase_queues["prefill"])
        )
        parked_offload_inflight = (
            getattr(self, "_kv_retention_parked_offload_future", None) is not None
        )
        decode_inflight = bool(phase_queues["decode"]) or other_inflight
        # The ordered D2H control Future is not in the model queue. Match the
        # transition's physical-work count so plan() cannot switch epoch on a
        # false quiescent boundary, even if copy is done but not yet recovered.
        prefill_inflight = (
            bool(phase_queues["prefill"])
            or other_inflight
            or parked_offload_inflight
        )
        # In-flight work remains allocation demand even when the scheduler
        # excludes its request id from the next lane selection. This prevents
        # a target profile from assigning zero SMs to a future that is still
        # continuing at a layer boundary.
        allocation_decode_ready = decode_ready or decode_inflight
        allocation_prefill_ready = prefill_ready or prefill_inflight
        if allocation_decode_ready and decode_tokens <= 0:
            decode_tokens = 1
        if allocation_prefill_ready and prefill_tokens <= 0:
            prefill_tokens = 1
        predictive = getattr(self.pd_sm_allocator, "predictive_enabled", False)
        prospective_outputs = (
            (*scheduler_outputs, suspended_prefill)
            if suspended_prefill is not None
            else scheduler_outputs
        )
        tbt_outputs = prospective_outputs
        if predictive:
            # A no-output control pass or Decode-only submit must still see
            # actual in-flight Ring and private offload-resume authority.
            tbt_outputs = (
                *prospective_outputs,
                *(item[1] for item in phase_queues["prefill"]),
            )
        (
            decode_context_tokens,
            decode_request_count,
            prefill_mode,
            decode_tbt_slo_ms,
            decode_tbt_elapsed_ms,
        ) = self._pd_sm_output_tbt_state(*tbt_outputs)
        if predictive:
            live_prefills = [item[1] for item in phase_queues["prefill"]]
            live_prefills.extend(
                output for output in prospective_outputs
                if self._scheduler_output_phase(output) == "prefill"
                and output.total_num_scheduled_tokens > 0)
            # DRAM mirrors can remain after an explicit Native reload. Use
            # execution authority, not incidental request_debug page counts.
            prefill_mode = "ring" if any(
                output.pd_prefill_mode == "ring" or getattr(
                    getattr(output, "kv_retention_layerwise_transfer", None), "state", ""
                ) in {"released", "published", "consumed"}
                for output in live_prefills) else "native"
        decode_min_tbt_slo_ms = 0.0
        scheduler_tbt_state_used = False
        decode_tbt_observed_p95_ms = 0.0
        decode_tbt_observed_slo_ms = 0.0
        decode_tbt_observed_ratio = 0.0
        decode_tbt_observation_count = 0
        decode_tbt_observation_epoch = 0
        if self.pd_sm_allocator.config.policy in {
            "tbt_slo_profiled",
            "tbt_slo_feedback",
            "tbt_slo_predictive",
        }:
            state_getter = getattr(
                self.scheduler, "get_tbt_allocation_state", None
            )
            if callable(state_getter):
                state_kwargs = {}
                if predictive:
                    decode_ids = {
                        rid for output in scheduler_outputs
                        if self._scheduler_output_phase(output) == "decode"
                        for rid in output.num_scheduled_tokens}
                    decode_ids.update(rid for item in phase_queues["decode"] for rid in item[1].num_scheduled_tokens)
                    if not decode_ids:
                        # Only runnable residents, not every DRAM-waiting
                        # logical Decode, stand for the next possible batch.
                        decode_ids = {r.request_id for r in self.scheduler.running}
                    state_kwargs["decode_request_ids"] = decode_ids
                scheduler_tbt_state = state_getter(
                    default_tbt_slo_ms=(
                        self.pd_sm_allocator.config.tbt_default_slo_ms
                    ),
                    feedback_window_tokens=(
                        self.pd_sm_allocator.config.feedback_window_tokens
                    ),
                    **state_kwargs,
                )
                if predictive:
                    decode_min_tbt_slo_ms = float(scheduler_tbt_state.get("decode_min_tbt_slo_ms", 0.0))
                    self.pd_sm_allocator.arm_tbt_deadline(
                        request_id=str(scheduler_tbt_state.get("critical_request_id", "")),
                        deadline_at=float(scheduler_tbt_state.get("decode_tbt_deadline_at", 0.0)))
                scheduler_tbt_state["decode_sm"] = (
                    self.pd_sm_allocator.current_profile.decode_sms
                )
                scheduler_tbt_state["prefill_sm"] = (
                    self.pd_sm_allocator.current_profile.prefill_sms
                )
                scheduler_tbt_state["prefill_inflight"] = int(
                    prefill_inflight
                )
                self._tbt_controller_state = scheduler_tbt_state
                scheduler_decode_count = int(
                    scheduler_tbt_state.get("decode_request_count", 0) or 0
                )
                if scheduler_decode_count > 0:
                    scheduler_tbt_state_used = True
                    decode_request_count = scheduler_decode_count
                    decode_context_tokens = max(
                        decode_context_tokens,
                        int(
                            scheduler_tbt_state.get(
                                "decode_context_tokens", 0
                            )
                            or 0
                        ),
                    )
                    scheduler_slo_ms = float(
                        scheduler_tbt_state.get(
                            "decode_tbt_slo_ms", 0.0
                        )
                        or 0.0
                    )
                    decode_tbt_slo_ms = scheduler_slo_ms
                    decode_tbt_elapsed_ms = float(
                        scheduler_tbt_state.get(
                            "decode_tbt_elapsed_ms", 0.0
                        )
                        or 0.0
                    )
                    decode_tbt_observed_p95_ms = float(
                        scheduler_tbt_state.get(
                            "decode_tbt_observed_p95_ms", 0.0
                        )
                        or 0.0
                    )
                    decode_tbt_observed_slo_ms = float(
                        scheduler_tbt_state.get(
                            "decode_tbt_observed_slo_ms", 0.0
                        )
                        or 0.0
                    )
                    decode_tbt_observed_ratio = float(
                        scheduler_tbt_state.get(
                            "decode_tbt_observed_ratio", 0.0
                        )
                        or 0.0
                    )
                    decode_tbt_observation_count = int(
                        scheduler_tbt_state.get(
                            "decode_tbt_observation_count", 0
                        )
                        or 0
                    )
                    decode_tbt_observation_epoch = int(
                        scheduler_tbt_state.get(
                            "decode_tbt_observation_epoch", 0
                        )
                        or 0
                    )
        predictive_kwargs = {}
        if predictive:
            predictive_kwargs = {
                "decode_min_tbt_slo_ms": decode_min_tbt_slo_ms,
                "transition_age_ms": (max(0.0, (time.monotonic() - self.pd_sm_transition.requested_at) * 1000.0)
                                      if self.pd_sm_transition.active else 0.0),
            }
        plan = self.pd_sm_allocator.plan(
            decode_ready=allocation_decode_ready,
            prefill_ready=allocation_prefill_ready,
            decode_inflight=decode_inflight,
            prefill_inflight=prefill_inflight,
            transition_active=(
                self.pd_dynamic_sm_enforced
                and self.pd_sm_transition.active
            ),
            decode_tokens=decode_tokens,
            prefill_tokens=prefill_tokens,
            decode_context_tokens=decode_context_tokens,
            decode_request_count=decode_request_count,
            prefill_mode=prefill_mode,
            decode_tbt_slo_ms=decode_tbt_slo_ms,
            decode_tbt_elapsed_ms=decode_tbt_elapsed_ms,
            decode_tbt_observed_p95_ms=decode_tbt_observed_p95_ms,
            decode_tbt_observed_slo_ms=decode_tbt_observed_slo_ms,
            decode_tbt_observed_ratio=decode_tbt_observed_ratio,
            decode_tbt_observation_count=decode_tbt_observation_count,
            decode_tbt_observation_epoch=decode_tbt_observation_epoch,
            capacity_serialized=prefill_capacity_serialized,
            **predictive_kwargs,
        )
        if predictive and plan.tbt_predictive:
            self._tbt_controller_state.update(plan.tbt_predictive)
        if (
            self.pd_dynamic_sm_enforced
            and self.pd_sm_transition.active
            and not self._pd_layerwise_handoff_fence_active
            and plan.target_profile == plan.profile
        ):
            # Demand can return to the immutable source before the worker has
            # accepted a layer move (for example Decode arrives while a p40
            # retarget is waiting for begin_execution). Keeping that stale
            # transition would incorrectly steal Decode SMs. No GPU work has
            # moved yet, so cancellation is transactionally safe.
            cancelled_debug = self.pd_sm_transition.finish()
            cancelled_debug.update(
                {
                    "pd_sm_transition_cancelled": 1.0,
                    "pd_sm_transition_cancel_reason": (
                        "allocator_returned_to_source_before_layer_ack"
                    ),
                }
            )
            self._pd_sm_transition_last_commit_debug = cancelled_debug
            if predictive:
                self.pd_sm_allocator.note_tbt_transition_commit(cancelled_debug)
            self._tbt_controller_state.update(cancelled_debug)
        self._request_pd_sm_transition(
            plan,
            layerwise_decode_backlog_empty=(
                layerwise_decode_backlog_empty
                and not decode_ready
                and not decode_inflight
                and prefill_inflight
            ),
        )
        self._maybe_request_pd_layerwise_handoff(
            decode_inflight=decode_inflight,
            prefill_inflight=prefill_inflight,
        )
        if (
            plan.switched
            and not plan.shadow
            and self.pd_sm_allocator.config.policy == "tbt_slo_feedback"
        ):
            reset_feedback_window = getattr(
                self.scheduler, "reset_tbt_feedback_window", None
            )
            if callable(reset_feedback_window):
                reset_feedback_window()
        execution_profile = (
            self.pd_sm_allocator.config.initial_profile
            if plan.shadow
            else plan.profile
        )
        execution_profile_id = plan.execution_profile_id
        execution_epoch = plan.epoch
        execution_decode_sms = execution_profile.decode_sms
        execution_prefill_sms = execution_profile.prefill_sms
        if self.pd_dynamic_sm_enforced and self.pd_sm_transition.active:
            # The active transition exclusively owns the source epoch. Decode
            # admitted during DRAIN_PREFILL must stay on that immutable source
            # profile until the global phase queue reaches quiescence.
            execution_profile_id = self.pd_sm_transition.source_profile_id
            execution_epoch = self.pd_sm_transition.source_epoch
            execution_decode_sms = self.pd_sm_transition.source_decode_sms
            execution_prefill_sms = self.pd_sm_transition.source_prefill_sms
        if (
            self.pd_layerwise_handoff_enabled
            and "prefill" in output_phases
            and prefill_tokens > 0
            and execution_prefill_sms > 0
        ):
            PD_LAYERWISE_HANDOFF_CONTROLLER.arm(
                source_profile_id=execution_profile_id,
                source_epoch=execution_epoch,
            )
        layerwise_debug = (
            PD_LAYERWISE_HANDOFF_CONTROLLER.snapshot()
            if self.pd_layerwise_handoff_enabled
            else {}
        )
        collect_debug = any(
            output.iteration_debug is not None for output in scheduler_outputs
        )
        debug = (
            {
                **plan.as_debug_dict(),
                **self._pd_sm_transition_last_commit_debug,
                **self.pd_sm_transition.debug_snapshot(),
                **layerwise_debug,
                "pd_dynamic_sm_execution_decode_sms": float(
                    execution_decode_sms
                ),
                "pd_dynamic_sm_execution_prefill_sms": float(
                    execution_prefill_sms
                ),
                "pd_dynamic_sm_decode_ready": float(decode_ready),
                "pd_dynamic_sm_prefill_ready": float(prefill_ready),
                "pd_dynamic_sm_suspended_prefill_ready": float(
                    suspended_prefill is not None
                ),
                "pd_dynamic_sm_suspended_prefill_reason": resume_reason,
                "pd_dynamic_sm_decode_allocation_ready": float(
                    allocation_decode_ready
                ),
                "pd_dynamic_sm_prefill_allocation_ready": float(
                    allocation_prefill_ready
                ),
                "pd_dynamic_sm_decode_inflight": float(decode_inflight),
                "pd_dynamic_sm_prefill_inflight": float(prefill_inflight),
                "pd_dynamic_sm_other_inflight": float(other_inflight),
                "pd_dynamic_sm_parked_offload_inflight": float(
                    parked_offload_inflight
                ),
                "pd_dynamic_sm_decode_work_tokens": float(decode_tokens),
                "pd_dynamic_sm_prefill_work_tokens": float(prefill_tokens),
                "pd_dynamic_sm_prefill_capacity_serialized": float(
                    prefill_capacity_serialized
                ),
                "pd_tbt_decode_context_tokens": float(decode_context_tokens),
                "pd_tbt_decode_request_count": float(decode_request_count),
                "pd_tbt_scheduler_state_used": float(
                    scheduler_tbt_state_used
                ),
                "pd_dynamic_sm_layerwise_handoff_enabled": float(
                    self.pd_layerwise_handoff_enabled
                ),
                "pd_layerwise_handoff_fence_active": float(
                    self._pd_layerwise_handoff_fence_active
                ),
                "pd_layerwise_handoff_decode_inflight_defer_checks": float(
                    self._pd_layerwise_handoff_decode_inflight_defer_checks
                ),
                "pd_layerwise_handoff_decode_backlog_defer_checks": float(
                    self._pd_layerwise_handoff_decode_backlog_defer_checks
                ),
                "pd_layerwise_handoff_prefill_only_retarget_count": float(
                    self._pd_layerwise_handoff_prefill_only_retarget_count
                ),
                "pd_layerwise_handoff_prefill_only_retarget_miss_count": float(
                    self._pd_layerwise_handoff_prefill_only_retarget_miss_count
                ),
            }
            if collect_debug
            else None
        )
        for output in scheduler_outputs:
            output.pd_sm_profile_id = execution_profile_id
            output.pd_sm_profile_epoch = execution_epoch
            output.pd_sm_decode_sms = execution_decode_sms
            output.pd_sm_prefill_sms = execution_prefill_sms
            if debug is not None and output.iteration_debug is not None:
                output.iteration_debug.update(debug)
        if self._tbt_controller_state:
            self._tbt_controller_state["decode_sm"] = (
                execution_decode_sms
            )
            self._tbt_controller_state["prefill_sm"] = (
                execution_prefill_sms
            )
            self._tbt_controller_state.update(
                {
                    "decode_tbt_measurement_unit": "ms_per_output_token",
                    "decode_tbt_measurement_source": (
                        "scheduler_output_token_ids"
                    ),
                    "pd_sm_transition_pending": int(
                        self.pd_sm_transition.active
                    ),
                    "pd_sm_transition_stage": self.pd_sm_transition.stage,
                    "pd_sm_transition_target_profile": (
                        self.pd_sm_transition.target_profile_id
                    ),
                }
            )
        return plan

    def _observe_pd_sm_plan(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
        engine_timings: dict[str, Any],
    ) -> None:
        phase = self._scheduler_output_phase(scheduler_output)
        if phase not in {"decode", "prefill"}:
            return
        decode_tokens, prefill_tokens = self._pd_sm_output_tokens(scheduler_output)
        num_tokens = decode_tokens if phase == "decode" else prefill_tokens
        model_debug = getattr(model_output, "model_runner_debug_timings", {})
        gpu_elapsed_ms = float(model_debug.get("gpu_forward_elapsed_ms", 0.0) or 0.0)
        if gpu_elapsed_ms > 0.0:
            elapsed_ms = gpu_elapsed_ms
            source = "cuda_event"
        else:
            submit_ts = float(engine_timings.get("execute_submit_ts", 0.0) or 0.0)
            done_ts = float(
                engine_timings.get("actual_model_future_done_ts", 0.0) or 0.0
            )
            elapsed_ms = max(0.0, (done_ts - submit_ts) * 1000.0)
            source = "engine_future_wall"
        if not self.pd_dynamic_sm_enabled:
            return
        profile_id = scheduler_output.pd_sm_profile_id
        if not profile_id:
            profile_id = self.pd_sm_allocator.config.initial_profile.profile_id
        observation = self.pd_sm_allocator.observe(
            phase=phase,
            num_tokens=num_tokens,
            elapsed_ms=elapsed_ms,
            profile_id=profile_id,
        )
        if phase == "decode" and getattr(self.pd_sm_allocator, "predictive_enabled", False):
            self.pd_sm_allocator.note_tbt_decode_completion(profile_id=profile_id, now=time.monotonic())
        observation["pd_dynamic_sm_observation_source"] = source
        engine_timings.update(observation)

    def _pd_phase_submit_timing_record(
        self,
        scheduler_output: SchedulerOutput,
        *,
        phase: str,
        execution_key: str,
        schedule_start_ts: float,
        schedule_end_ts: float,
        execute_submit_ts: float,
        queue_depth_before_submit: int,
        paired_launch: bool,
        pair_submit_seq: int = -1,
        decoupled_recovery: bool = False,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schedule_start_ts": schedule_start_ts,
            "schedule_end_ts": schedule_end_ts,
            "execute_submit_ts": execute_submit_ts,
        }
        record.update(self._pd_capacity_retry_snapshot())
        if not self.full_iteration_debug_enabled:
            return record
        record.update(
            {
                "submitted_scheduler_output_phase": phase,
                "outstanding_model_futures_before": float(
                    queue_depth_before_submit
                ),
                "engine_batch_queue_depth_before_submit": float(
                    queue_depth_before_submit
                ),
                "pd_independent_phase_queues_enabled": 1.0,
                "pd_muxwise_event_loop_enabled": float(
                    self.pd_muxwise_event_loop_enabled
                ),
                "pd_separate_scheduler_lanes_enabled": float(
                    self.pd_separate_scheduler_lanes_enabled
                ),
                "pd_dual_continuous_batching_enabled": float(
                    self.pd_dual_continuous_batching_enabled
                ),
                "pd_nonblocking_phase_recovery_enabled": float(
                    self.pd_nonblocking_phase_recovery_enabled
                ),
                "pd_single_inflight_per_phase_enabled": float(
                    self.pd_single_inflight_per_phase_enabled
                ),
                "pd_dual_continuous_decode_reserved_batches": float(
                    self.pd_decode_reserved_batches
                ),
                "pd_dual_continuous_prefill_capacity": float(
                    self._pd_phase_queue_prefill_capacity()
                ),
                "pd_paired_phase_launch_requested": float(
                    self.pd_paired_phase_launch_requested
                ),
                "pd_paired_phase_launch_enabled": float(
                    self.pd_paired_phase_launch_enabled
                ),
                "pd_paired_phase_launch_used": float(paired_launch),
                "pd_paired_phase_launch_seq": float(pair_submit_seq),
                "pd_decoupled_paired_recovery_requested": float(
                    self.pd_decoupled_paired_recovery_requested
                ),
                "pd_decoupled_paired_recovery_enabled": float(
                    self.pd_decoupled_paired_recovery_enabled
                ),
                "pd_decoupled_paired_recovery_used": float(
                    decoupled_recovery
                ),
                "pd_decoupled_phase_execute_future": 0.0,
                "pd_paired_shared_execute_future": float(paired_launch),
                "pd_forward_enqueue_barrier_used": float(
                    decoupled_recovery
                ),
                "pd_phase_local_sample_future": float(paired_launch),
                "pd_phase_depth_enabled": float(self.pd_phase_depth_enabled),
                "pd_prefill_inflight_depth": float(
                    self.pd_prefill_inflight_depth
                ),
                "pd_decode_inflight_depth": float(
                    self.pd_decode_inflight_depth
                ),
                "pd_global_inflight_depth": float(
                    self.pd_global_inflight_depth
                ),
                "pd_phase_queue_submit_phase": phase,
                "pd_phase_queue_submit_slot": float(
                    scheduler_output.pd_phase_slot
                ),
                "pd_phase_queue_submit_execution_key": execution_key,
                "pd_phase_queue_submit_seq": float(
                    scheduler_output.pd_phase_submit_seq
                ),
            }
        )
        return record

    def _append_pd_phase_queue_item_after_paired_exec(
        self,
        phase_queues,
        scheduler_output: SchedulerOutput,
        exec_future: Future[Any],
        *,
        schedule_start_ts: float,
        schedule_end_ts: float,
        execute_submit_ts: float,
        queue_depth_before_submit: int,
        pair_submit_seq: int,
        decoupled_recovery: bool = False,
    ) -> tuple[bool, Future[Any]]:
        phase = self._pd_phase_queue_for_output(scheduler_output)
        execution_key = self._pd_phase_execution_key(scheduler_output, phase)
        collect_detailed_timing = self.full_iteration_debug_enabled
        timing_record = self._pd_phase_submit_timing_record(
            scheduler_output,
            phase=phase,
            execution_key=execution_key,
            schedule_start_ts=schedule_start_ts,
            schedule_end_ts=schedule_end_ts,
            execute_submit_ts=execute_submit_ts,
            queue_depth_before_submit=queue_depth_before_submit,
            paired_launch=True,
            pair_submit_seq=pair_submit_seq,
            decoupled_recovery=decoupled_recovery,
        )
        self._iteration_debug_timings[id(scheduler_output)] = timing_record
        model_executed = False
        if not self.is_ec_producer:
            model_executed = scheduler_output.total_num_scheduled_tokens > 0

        if self.is_pooling_model or not model_executed:
            future = cast(Future[ModelRunnerOutput], exec_future)
        else:
            grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
            future = self.model_executor.sample_tokens(
                grammar_output,
                non_block=True,
                pd_green_context_phase=execution_key,
            )

        self._register_pd_model_future_completion(
            future,
            scheduler_output,
            collect_timing=collect_detailed_timing,
        )
        phase_queues[phase].appendleft((future, scheduler_output, exec_future))
        timing_record["outstanding_model_futures_after"] = float(
            self._pd_phase_queue_total_depth(phase_queues)
        )
        timing_record["engine_batch_queue_depth_after_submit"] = float(
            self._pd_phase_queue_total_depth(phase_queues)
        )
        timing_record.update(self._pd_phase_queues_counts(phase_queues))
        self._write_pd_phase_queue_trace(
            "submit_paired",
            phase_queues,
            {
                "submitted_phase": phase,
                "submitted_scheduler_output_id": id(scheduler_output),
                "submitted_total_tokens": float(
                    scheduler_output.total_num_scheduled_tokens
                ),
                "pd_paired_phase_launch_seq": float(pair_submit_seq),
                "submitted_phase_slot": float(scheduler_output.pd_phase_slot),
                "submitted_execution_key": execution_key,
                "submitted_phase_seq": float(
                    scheduler_output.pd_phase_submit_seq
                ),
            },
        )
        return model_executed, future

    def _submit_pd_phase_pair_to_queues(
        self,
        phase_queues,
        decode_scheduler_output: SchedulerOutput,
        prefill_scheduler_output: SchedulerOutput,
        *,
        decode_schedule_start_ts: float,
        decode_schedule_end_ts: float,
        prefill_schedule_start_ts: float,
        prefill_schedule_end_ts: float,
        queue_depth_before_submit: int,
    ) -> tuple[bool, Future[Any]]:
        overlapping_request_ids = self._scheduler_output_request_ids(
            decode_scheduler_output
        ) & self._scheduler_output_request_ids(prefill_scheduler_output)
        if overlapping_request_ids:
            # A request cannot own two model-runner states or two KV mutation
            # futures concurrently. The scheduler's pending-id exclusion must
            # prevent this before launch; fail loudly rather than corrupting
            # phase-local state on the GPU.
            raise RuntimeError(
                "paired P/D scheduler outputs overlap request ids: "
                f"{sorted(overlapping_request_ids)}"
            )
        if self.pd_dynamic_sm_enabled:
            self._assign_pd_sm_plan(
                phase_queues,
                decode_scheduler_output,
                prefill_scheduler_output,
            )
        self._annotate_pd_component_overlap_candidate(
            phase_queues,
            decode_scheduler_output,
            paired_prefill_submit=True,
        )
        # execute_model_pd_pair derives the worker-side phase-local context
        # from pd_phase_slot. Reserve both slots before the paired RPC so a
        # depth>1 launch cannot alias an older in-flight slot (slot 0 was
        # otherwise reused implicitly).
        self._assign_pd_phase_submission(
            phase_queues, decode_scheduler_output, "decode"
        )
        self._assign_pd_phase_submission(
            phase_queues, prefill_scheduler_output, "prefill"
        )
        execute_submit_ts = time.monotonic()
        exec_future = self.model_executor.execute_model_pd_pair(
            decode_scheduler_output,
            prefill_scheduler_output,
            non_block=True,
        )
        self._pd_phase_queue_submit_seq += 1
        pair_submit_seq = self._pd_phase_queue_submit_seq
        model_executed = False
        for scheduler_output, schedule_start_ts, schedule_end_ts in (
            (
                decode_scheduler_output,
                decode_schedule_start_ts,
                decode_schedule_end_ts,
            ),
            (
                prefill_scheduler_output,
                prefill_schedule_start_ts,
                prefill_schedule_end_ts,
            ),
        ):
            output_executed, _ = self._append_pd_phase_queue_item_after_paired_exec(
                phase_queues,
                scheduler_output,
                exec_future,
                schedule_start_ts=schedule_start_ts,
                schedule_end_ts=schedule_end_ts,
                execute_submit_ts=execute_submit_ts,
                queue_depth_before_submit=queue_depth_before_submit,
                pair_submit_seq=pair_submit_seq,
            )
            model_executed = model_executed or output_executed
        self._pd_ttft_handoff_mark_submitted(prefill_scheduler_output)
        return model_executed, exec_future

    def _submit_pd_phase_pair_decoupled_to_queues(
        self,
        phase_queues,
        decode_scheduler_output: SchedulerOutput,
        prefill_scheduler_output: SchedulerOutput,
        *,
        decode_schedule_start_ts: float,
        decode_schedule_end_ts: float,
        prefill_schedule_start_ts: float,
        prefill_schedule_end_ts: float,
    ) -> tuple[bool, Future[Any]]:
        """Enqueue both phase forwards before starting phase-local sampling.

        The paired worker RPC is a host-enqueue barrier, not a device join: it
        returns after decode and prefill work has been submitted to their
        GreenContext streams. Sampling and result recovery remain phase-local,
        so either phase can complete and advance without waiting for the other
        phase's GPU work.
        """
        overlapping_request_ids = self._scheduler_output_request_ids(
            decode_scheduler_output
        ) & self._scheduler_output_request_ids(prefill_scheduler_output)
        if overlapping_request_ids:
            raise RuntimeError(
                "decoupled paired P/D scheduler outputs overlap request ids: "
                f"{sorted(overlapping_request_ids)}"
            )

        if self.pd_dynamic_sm_enabled:
            self._assign_pd_sm_plan(
                phase_queues,
                decode_scheduler_output,
                prefill_scheduler_output,
            )
        self._annotate_pd_component_overlap_candidate(
            phase_queues,
            decode_scheduler_output,
            paired_prefill_submit=True,
        )
        self._assign_pd_phase_submission(
            phase_queues, decode_scheduler_output, "decode"
        )
        self._assign_pd_phase_submission(
            phase_queues, prefill_scheduler_output, "prefill"
        )
        execute_submit_ts = time.monotonic()
        exec_future = self.model_executor.execute_model_pd_pair(
            decode_scheduler_output,
            prefill_scheduler_output,
            non_block=True,
        )
        self._pd_phase_queue_submit_seq += 1
        pair_submit_seq = self._pd_phase_queue_submit_seq
        queue_depth_before_submit = self._pd_phase_queue_total_depth(phase_queues)
        model_executed = False
        for scheduler_output, schedule_start_ts, schedule_end_ts in (
            (
                decode_scheduler_output,
                decode_schedule_start_ts,
                decode_schedule_end_ts,
            ),
            (
                prefill_scheduler_output,
                prefill_schedule_start_ts,
                prefill_schedule_end_ts,
            ),
        ):
            output_executed, _ = self._append_pd_phase_queue_item_after_paired_exec(
                phase_queues,
                scheduler_output,
                exec_future,
                schedule_start_ts=schedule_start_ts,
                schedule_end_ts=schedule_end_ts,
                execute_submit_ts=execute_submit_ts,
                queue_depth_before_submit=queue_depth_before_submit,
                pair_submit_seq=pair_submit_seq,
                decoupled_recovery=True,
            )
            model_executed = model_executed or output_executed
        self._pd_ttft_handoff_mark_submitted(prefill_scheduler_output)
        self._write_pd_phase_queue_trace(
            "submit_paired_decoupled",
            phase_queues,
            {
                "pd_paired_phase_launch_seq": float(pair_submit_seq),
                "decode_scheduler_output_id": id(decode_scheduler_output),
                "prefill_scheduler_output_id": id(prefill_scheduler_output),
                "pd_paired_shared_execute_future": 1.0,
                "pd_forward_enqueue_barrier_used": 1.0,
                "pd_decoupled_paired_recovery_used": 1.0,
            },
        )
        return model_executed, exec_future

    def _try_submit_pd_paired_phase_launch(
        self,
        phase_queues,
    ) -> tuple[bool, SchedulerOutput | None, Future[Any] | None, bool]:
        if not self.pd_paired_phase_launch_enabled:
            return False, None, None, False
        if (
            self._pd_phase_queue_total_depth(phase_queues)
            > self.batch_queue_size - 2
        ):
            return False, None, None, False
        if not (
            self._pd_phase_queue_can_submit_phase(phase_queues, "decode")
            and self._pd_phase_queue_can_submit_phase(phase_queues, "prefill")
        ):
            return False, None, None, False
        if not (
            self._pd_capacity_retry_can_submit("decode")
            and self._pd_capacity_retry_can_submit("prefill")
        ):
            return False, None, None, False
        if not (
            self._scheduler_has_pd_phase_work(self.scheduler, "decode")
            and self._scheduler_has_pd_phase_work(self.scheduler, "prefill")
        ):
            return False, None, None, False

        pending_request_ids = self._pd_phase_queue_pending_request_ids(phase_queues)
        decode_scheduler_output = None
        prefill_scheduler_output = None
        decode_exec_future = None
        try:
            self._set_scheduler_pd_pending_request_ids(pending_request_ids)
            self._set_scheduler_pd_pending_phase_request_ids(
                self._pd_phase_queue_pending_request_ids_by_phase(phase_queues)
            )
            decode_schedule_start_ts = time.monotonic()
            decode_scheduler_output = self._scheduler_schedule_pd_phase(
                self.scheduler, "decode"
            )
            decode_schedule_end_ts = time.monotonic()
            if not self._scheduler_output_has_work(decode_scheduler_output):
                self._record_pd_capacity_retry_block(
                    "decode", decode_scheduler_output
                )
                return False, None, None, False
            self._pd_capacity_retry_gate.clear("decode")

            decode_request_ids = self._scheduler_output_request_ids(
                decode_scheduler_output
            )
            self._set_scheduler_pd_pending_request_ids(
                pending_request_ids | decode_request_ids
            )
            pending_by_phase = self._pd_phase_queue_pending_request_ids_by_phase(
                phase_queues
            )
            pending_by_phase.setdefault("decode", set()).update(
                decode_request_ids
            )
            self._set_scheduler_pd_pending_phase_request_ids(pending_by_phase)
            prefill_schedule_start_ts = time.monotonic()
            prefill_scheduler_output = self._scheduler_schedule_pd_phase(
                self.scheduler, "prefill"
            )
            prefill_schedule_end_ts = time.monotonic()
            if not self._scheduler_output_has_work(prefill_scheduler_output):
                self._record_pd_capacity_retry_block(
                    "prefill", prefill_scheduler_output
                )
                self._write_pd_phase_queue_trace(
                    "paired_prefill_empty_after_decode",
                    phase_queues,
                    {
                        "decode_request_count": float(len(decode_request_ids)),
                        "prefill_scheduler_output_id": id(
                            prefill_scheduler_output
                        ),
                    },
                )
                (
                    decode_model_executed,
                    decode_deferred_output,
                    decode_exec_future,
                ) = self._submit_to_pd_phase_queue(
                    phase_queues,
                    decode_scheduler_output,
                    decode_schedule_start_ts,
                    decode_schedule_end_ts,
                    self._pd_phase_queue_total_depth(phase_queues),
                )
                return (
                    decode_model_executed,
                    decode_deferred_output,
                    decode_exec_future,
                    True,
                )
            self._pd_capacity_retry_gate.clear("prefill")

            if not (
                self._scheduler_output_can_pair(decode_scheduler_output)
                and self._scheduler_output_can_pair(prefill_scheduler_output)
            ):
                (
                    decode_model_executed,
                    decode_deferred_output,
                    decode_exec_future,
                ) = self._submit_to_pd_phase_queue(
                    phase_queues,
                    decode_scheduler_output,
                    decode_schedule_start_ts,
                    decode_schedule_end_ts,
                    self._pd_phase_queue_total_depth(phase_queues),
                )
                (
                    prefill_model_executed,
                    prefill_deferred_output,
                    prefill_exec_future,
                ) = self._submit_to_pd_phase_queue(
                    phase_queues,
                    prefill_scheduler_output,
                    prefill_schedule_start_ts,
                    prefill_schedule_end_ts,
                    self._pd_phase_queue_total_depth(phase_queues),
                )
                return (
                    decode_model_executed or prefill_model_executed,
                    decode_deferred_output or prefill_deferred_output,
                    decode_exec_future or prefill_exec_future,
                    True,
                )

            if self.pd_decoupled_paired_recovery_enabled:
                pair_model_executed, pair_exec_future = (
                    self._submit_pd_phase_pair_decoupled_to_queues(
                        phase_queues,
                        decode_scheduler_output,
                        prefill_scheduler_output,
                        decode_schedule_start_ts=decode_schedule_start_ts,
                        decode_schedule_end_ts=decode_schedule_end_ts,
                        prefill_schedule_start_ts=prefill_schedule_start_ts,
                        prefill_schedule_end_ts=prefill_schedule_end_ts,
                    )
                )
            else:
                pair_model_executed, pair_exec_future = (
                    self._submit_pd_phase_pair_to_queues(
                        phase_queues,
                        decode_scheduler_output,
                        prefill_scheduler_output,
                        decode_schedule_start_ts=decode_schedule_start_ts,
                        decode_schedule_end_ts=decode_schedule_end_ts,
                        prefill_schedule_start_ts=prefill_schedule_start_ts,
                        prefill_schedule_end_ts=prefill_schedule_end_ts,
                        queue_depth_before_submit=self._pd_phase_queue_total_depth(
                            phase_queues
                        ),
                    )
                )
            return pair_model_executed, None, pair_exec_future, True
        finally:
            self._set_scheduler_pd_pending_request_ids(set())
            self._set_scheduler_pd_pending_phase_request_ids({})

    def _submit_pd_scheduler_lanes(
        self,
        phase_queues,
        lane_phases: tuple[str, ...],
        *,
        allow_multiple_rounds: bool,
    ) -> tuple[bool, SchedulerOutput | None, Future[Any] | None, bool]:
        model_executed = False
        deferred_scheduler_output = None
        deferred_exec_future = None
        submitted_any = False
        bounded_planning = getattr(
            self.scheduler, "isolation_bounded_prefill_planning", False)
        decode_output_first = (
            bounded_planning
            or getattr(self, "isolation_decode_output_first", False)
        )

        def ready_decode() -> bool:
            # A Future can finish AFTER the step-entry recovery check. Output
            # priority is independent of the optional CPU-readiness prewarmer.
            # Yield only at admission boundaries, never inside schedule().
            if not decode_output_first or not phase_queues["decode"]:
                return False
            phase, index, _ = self._select_pd_phase_queue_recovery_item(
                phase_queues, collect_debug=False)
            ready = phase == "decode" and phase_queues[phase][index][0].done()
            if ready:
                self._write_pd_phase_queue_trace(
                    "decode_ready_admission_yield", phase_queues,
                    {"bounded_planning": bool(bounded_planning)},
                )
            return ready

        while (
            self._pd_phase_queue_total_depth(phase_queues) < self.batch_queue_size
            and self.scheduler.has_requests()
            and deferred_scheduler_output is None
        ):
            if ready_decode():
                break
            submitted_this_round = False
            planning_yield = False
            (
                paired_model_executed,
                paired_deferred_output,
                paired_exec_future,
                paired_submitted,
            ) = self._try_submit_pd_paired_phase_launch(phase_queues)
            if paired_submitted:
                model_executed = model_executed or paired_model_executed
                submitted_any = True
                submitted_this_round = True
                if paired_deferred_output is not None:
                    deferred_scheduler_output = paired_deferred_output
                    deferred_exec_future = paired_exec_future
                if not allow_multiple_rounds:
                    break
                continue
            for lane_phase in lane_phases:
                if ready_decode():
                    planning_yield = True
                    break
                if (lane_phase == "prefill" and getattr(self, "kv_retention_p1_parked_enabled", False)
                        and self._try_resume_ttft_prefill(phase_queues, respect_srpt=True)):
                    model_executed = submitted_any = submitted_this_round = True
                    continue
                if not self._pd_phase_queue_can_submit_phase(
                    phase_queues, lane_phase
                ):
                    continue
                if not self._pd_capacity_retry_can_submit(lane_phase):
                    continue
                pending_request_ids = self._pd_phase_queue_pending_request_ids(
                    phase_queues
                )
                self._set_scheduler_pd_pending_request_ids(pending_request_ids)
                self._set_scheduler_pd_pending_phase_request_ids(
                    self._pd_phase_queue_pending_request_ids_by_phase(
                        phase_queues
                    )
                )
                if not self._scheduler_has_pd_phase_work(
                    self.scheduler, lane_phase
                ):
                    continue
                if bounded_planning and lane_phase == "prefill":
                    if not self.scheduler.prepare_pd_prefill_admission():
                        # A real finite CPU slice advanced, not an HBM/DMA wait.
                        # Wake the next control pass even with no model Future.
                        # Never arm a capacity gate or fabricate SchedulerOutput.
                        self._notify_pd_engine_progress()
                        self._write_pd_phase_queue_trace(
                            "prefill_readiness_yield", phase_queues)
                        planning_yield = True
                        break
                    if ready_decode():
                        planning_yield = True
                        break
                schedule_start_ts = time.monotonic()
                scheduler_output = self._scheduler_schedule_pd_phase(
                    self.scheduler, lane_phase
                )
                schedule_end_ts = time.monotonic()
                scheduled = (
                    scheduler_output.total_num_scheduled_tokens > 0
                    or bool(scheduler_output.finished_req_ids)
                )
                if (getattr(self, "isolation_decode_output_first", False)
                        and schedule_end_ts - schedule_start_ts >= 1.0):
                    logger.warning(
                        "isolation slow phase admission: phase=%s elapsed_s=%.6f "
                        "tokens=%d mode=%s; Decode output may have waited for host work",
                        lane_phase, schedule_end_ts - schedule_start_ts,
                        scheduler_output.total_num_scheduled_tokens,
                        getattr(scheduler_output, "pd_prefill_mode", ""))
                if not scheduled:
                    self._record_pd_capacity_retry_block(
                        lane_phase, scheduler_output
                    )
                    continue
                self._pd_capacity_retry_gate.clear(lane_phase)
                (
                    lane_model_executed,
                    lane_deferred_output,
                    lane_exec_future,
                ) = self._submit_to_pd_phase_queue(
                    phase_queues,
                    scheduler_output,
                    schedule_start_ts,
                    schedule_end_ts,
                    self._pd_phase_queue_total_depth(phase_queues),
                )
                model_executed = model_executed or lane_model_executed
                submitted_any = True
                submitted_this_round = True
                if lane_deferred_output is not None:
                    deferred_scheduler_output = lane_deferred_output
                    deferred_exec_future = lane_exec_future
                    break
            if (
                planning_yield or not allow_multiple_rounds
                or not submitted_this_round
            ):
                break

        self._set_scheduler_pd_pending_request_ids(set())
        self._set_scheduler_pd_pending_phase_request_ids({})
        return (
            model_executed,
            deferred_scheduler_output,
            deferred_exec_future,
            submitted_any,
        )

    def _ttft_prefill_execution_key(
        self, scheduler_output: SchedulerOutput
    ) -> str:
        return self._pd_phase_execution_key(scheduler_output, "prefill")

    @staticmethod
    def _cancel_ttft_prefill_yield_for_output(
        scheduler_output: SchedulerOutput,
    ) -> None:
        cancel_ttft_layerwise_prefill_yield(
            scheduler_iteration_id=scheduler_output.scheduler_iteration_id,
            phase_submit_seq=scheduler_output.pd_phase_submit_seq,
            phase_slot=scheduler_output.pd_phase_slot,
            scheduled_tokens=tuple(
                (str(request_id), int(num_scheduled_tokens))
                for request_id, num_scheduled_tokens in (
                    scheduler_output.num_scheduled_tokens.items()
                )
            ),
        )

    def _request_ttft_prefill_preemption_if_needed(
        self,
        phase_queues,
    ) -> bool:
        """Yield an immutable active batch when SRPT displaces its tail."""

        if not self.ttft_layerwise_prefill_preemption_enabled:
            return False
        p1 = getattr(self, "kv_retention_p1_parked_enabled", False)
        if not p1 and (
            self._pd_owned_phase_depth(phase_queues, "prefill")
            >= self._pd_phase_inflight_limit("prefill")
        ):
            return False
        active_items = [
            (future, scheduler_output)
            for future, scheduler_output, exec_future in (
                phase_queues["prefill"]
            )
            if scheduler_output.ttft_layerwise_prefill_preemption
            and future is exec_future
            and not future.done()
        ]
        if len(active_items) != 1:
            return False

        if p1 and self.pd_sm_transition.active:
            output = active_items[0][1]
            if output.kv_retention_layerwise_pressure_checked:
                return False
            if not output.kv_retention_boundary_reason:
                output.kv_retention_boundary_reason = "sm"
            output.kv_retention_layerwise_pressure_checked = True
            armed = request_ttft_layerwise_prefill_yield(
                scheduler_iteration_id=output.scheduler_iteration_id,
                phase_submit_seq=output.pd_phase_submit_seq, phase_slot=output.pd_phase_slot,
                scheduled_tokens=tuple(output.num_scheduled_tokens.items()))
            if armed:
                self._write_pd_phase_queue_trace(
                    "kv_retention_p1_sm_boundary_signal", phase_queues,
                    {"reason": output.kv_retention_boundary_reason,
                     "prefill_mode": output.pd_prefill_mode,
                     "source_profile": output.pd_sm_profile_id})
            return armed
        if (active_items[0][1].pd_prefill_mode != "native"
                or has_native_reload_authority(active_items[0][1])):
            # Ordinary Ring and in-flight Native reload may retain their slot
            # at an SM boundary only. Reload destination blocks are not proof
            # that upper-layer historical KV is valid for snapshot/offload.
            return False
        if p1 and (self._ttft_prefill_suspended
                   or active_items[0][1].kv_retention_p1_park_declined):
            return False

        prefill_scheduler = getattr(
            self.scheduler,
            "ttft_prefill_chunk_scheduler",
            None,
        )
        policy = getattr(prefill_scheduler, "policy", None)
        policy_name = str(getattr(policy, "value", policy) or "").lower()
        remaining_tokens = getattr(
            prefill_scheduler,
            "remaining_prefill_tokens",
            None,
        )
        if policy_name != "srpt" or not callable(remaining_tokens):
            return False

        _, active_output = active_items[0]
        if ((self.kv_retention_layerwise_offload_enabled
             or getattr(self, "kv_retention_p1_hbm_only", False))
                and active_output.kv_retention_layerwise_pressure_checked):
            # The worker may have consumed the yield while exporting its frame
            # or finishing D2H. A retained-HBM boundary needs the same latch.
            # Do not re-arm that same signal or scan blocks per Decode token.
            return False
        active_ids = tuple(active_output.num_scheduled_tokens)
        if not active_ids:
            return False
        requests = getattr(self.scheduler, "requests", {})
        active_requests = [requests.get(request_id) for request_id in active_ids]
        if any(request is None for request in active_requests):
            return False

        pending_ids = self._pd_phase_queue_pending_request_ids(phase_queues)

        def srpt_key(request: Request) -> tuple[int, float, str]:
            return (
                max(0, int(remaining_tokens(request))),
                float(request.arrival_time),
                str(request.request_id),
            )

        candidates = [
            request
            for request_id, request in requests.items()
            if request_id not in pending_ids
            and int(remaining_tokens(request)) > 0
        ]
        if not candidates:
            return False
        candidate = min(candidates, key=srpt_key)
        active_keys = tuple(
            (
                max(
                    0,
                    int(remaining_tokens(request))
                    + int(
                        active_output.num_scheduled_tokens[
                            request.request_id
                        ]
                    ),
                ),
                float(request.arrival_time),
                str(request.request_id),
            )
            for request in active_requests
            if request is not None
        )
        candidate_key = srpt_key(candidate)
        if not should_preempt_ttft_layerwise_prefill_batch(
            active_srpt_keys=active_keys,
            candidate_srpt_key=candidate_key,
        ):
            return False
        active_tail_key = max(active_keys)
        active_head_key = min(active_keys)

        if getattr(self, "kv_retention_p1_hbm_only", False):
            # This is a capacity proof, not a reservation. Parked physical KV
            # remains owned and the Scheduler rechecks admission after yield.
            guard = self.scheduler.isolation_hbm_guard
            if not guard.can_admit(candidate):
                blocker = (candidate.request_id, active_ids,
                           guard.last_required_blocks - guard.last_free_blocks)
                if blocker != self._kv_retention_hbm_preemption_blocker:
                    self._write_pd_phase_queue_trace(
                        "kv_retention_p1_hbm_preemption_capacity_wait", phase_queues,
                        {"candidate_request_id": candidate.request_id,
                         "active_request_ids": list(active_ids),
                         "required_blocks": guard.last_required_blocks,
                         "free_blocks": guard.last_free_blocks})
                    self._kv_retention_hbm_preemption_blocker = blocker
                return False
            self._kv_retention_hbm_preemption_blocker = None

        # Publish intent before signaling. With ample capacity keep Native KV.
        if p1:
            active_output.kv_retention_boundary_reason = "request"
        if self.kv_retention_layerwise_offload_enabled:
            active_output.kv_retention_layerwise_pressure_checked = True
            if active_output.kv_retention_layerwise_transfer is None:
                pressure_kwargs = {}
                if getattr(self.scheduler, "isolation_prefill_batch_kv_budget", False):
                    # Same SRPT order already used for this yield. The
                    # Scheduler projects the whole replacement batch under
                    # real token/sequence caps; this never allocates its KV.
                    pressure_kwargs["replacement_candidates"] = tuple(
                        sorted(candidates, key=srpt_key))
                active_output.kv_retention_layerwise_offload_requested = (
                    self.scheduler.layerwise_offload_needed(
                        active_output, candidate, **pressure_kwargs)
                )

        armed = request_ttft_layerwise_prefill_yield(
            scheduler_iteration_id=active_output.scheduler_iteration_id,
            phase_submit_seq=active_output.pd_phase_submit_seq,
            phase_slot=active_output.pd_phase_slot,
            scheduled_tokens=tuple(
                (str(request_id), int(num_scheduled_tokens))
                for request_id, num_scheduled_tokens in (
                    active_output.num_scheduled_tokens.items()
                )
            ),
        )
        if not armed:
            return False
        if getattr(self, "kv_retention_p1_hbm_only", False):
            active_output.kv_retention_layerwise_pressure_checked = True
        self._ttft_prefill_preemption_signal_count += 1
        timings = self._iteration_debug_timings.setdefault(
            id(active_output), {}
        )
        timings.update(
            {
                "ttft_layerwise_prefill_preemption_signal_count": float(
                    self._ttft_prefill_preemption_signal_count
                ),
                "ttft_layerwise_prefill_preemption_candidate_id": (
                    candidate.request_id
                ),
                "ttft_layerwise_prefill_preemption_active_remaining_tokens": (
                    float(active_tail_key[0])
                ),
                "ttft_layerwise_prefill_preemption_active_head_remaining_tokens": (
                    float(active_head_key[0])
                ),
                "ttft_layerwise_prefill_preemption_active_batch_width": float(
                    len(active_ids)
                ),
                "ttft_layerwise_prefill_preemption_candidate_remaining_tokens": (
                    float(candidate_key[0])
                ),
            }
        )
        self._write_pd_phase_queue_trace(
            "ttft_prefill_preemption_signal",
            phase_queues,
            {
                "active_request_ids": list(active_ids),
                "active_batch_width": float(len(active_ids)),
                "active_request_id": active_tail_key[2],
                "active_tail_request_id": active_tail_key[2],
                "candidate_request_id": candidate.request_id,
                "active_remaining_tokens": float(active_tail_key[0]),
                "candidate_remaining_tokens": float(candidate_key[0]),
            },
        )
        return True

    def _kv_retention_bind_prefill(self, output: SchedulerOutput, *, resume=False):
        registry = getattr(self, "_kv_retention_prefill_work_registry", None)
        if registry is None or not output.ttft_layerwise_prefill_preemption:
            return None
        epochs = {
            req_id: self.scheduler.requests[req_id].pd_output_placeholder_ledger.request_epoch
            for req_id in output.num_scheduled_tokens}
        return registry.resume(output, epochs) if resume else registry.start(output, epochs)

    def _kv_retention_future_binding(self, future, output):
        binding = getattr(future, "kv_retention_binding", None)
        work = getattr(output, "kv_retention_prefill_work", None)
        if binding is None and work is None:
            return None
        registry = getattr(self, "_kv_retention_prefill_work_registry", None)
        if binding is None or registry is None or binding.work != work:
            raise RuntimeError("Prefill Future lost its captured execution binding")
        record = registry.require(binding, output)
        if record.state not in {"active", "sampling"}:
            raise RuntimeError("Prefill Future recovered outside an active segment")
        # Missing owners follow the existing whole-cancel/partial-abort path.
        # A new request incarnation with the same ID must never consume work.
        for req_id, epoch, _, _ in work.rows:
            request = self.scheduler.requests.get(req_id)
            if request is not None and request.pd_output_placeholder_ledger.request_epoch != epoch:
                raise RuntimeError("Prefill result belongs to an old request epoch")
        return binding

    def _kv_retention_binding_debug(self, binding):
        if binding is None:
            return {}
        return {**binding.debug(), "kv_retention_live_work_count":
                len(self._kv_retention_prefill_work_registry)}

    def _cancel_ttft_prefill_worker_slot(
        self, execution_key: str, scheduler_output: SchedulerOutput,
        kv_retention_binding: KVRetentionExecutionBinding | None = None,
    ) -> None:
        clear_state = getattr(
            self.model_executor,
            "clear_pd_phase_execute_state",
            None,
        )
        if not callable(clear_state):
            raise RuntimeError(
                "TTFT Prefill continuation cleanup requires an executor "
                "with ordered phase-slot cleanup support."
            )
        if kv_retention_binding is not None:
            self._kv_retention_prefill_work_registry.require(
                kv_retention_binding, scheduler_output)
        cleared = clear_state(
            execution_key,
            KVRetentionPhaseOwner.from_output(
                execution_key, scheduler_output, kv_retention_binding),
        )
        if kv_retention_binding is not None:
            if not cleared:
                raise RuntimeError("owned Prefill cancellation was not acknowledged")
            self._kv_retention_prefill_work_registry.finish(
                kv_retention_binding, cancelled=True)

    def _prune_ttft_prefill_suspended(self) -> None:
        if not self._ttft_prefill_suspended:
            return
        requests = getattr(self.scheduler, "requests", {})
        for storage_key, (boundary, scheduler_output) in tuple(
            self._ttft_prefill_suspended.items()
        ):
            execution_key = self._ttft_prefill_execution_key(scheduler_output)
            request_ids = self._scheduler_output_request_ids(scheduler_output)
            if (getattr(boundary, "kv_retention_parked_ticket", None) is not None
                    and request_ids <= self._ttft_prefill_deferred_abort_ids):
                queues = getattr(self, "pd_phase_queues", {}) or {}
                if (queues.get("prefill")
                        or getattr(self, "_kv_retention_parked_offload_future", None) is not None):
                    continue  # Do not block Decode on C's ordered Prefill worker.
                self._cancel_ttft_prefill_yield_for_output(scheduler_output)
                self._cancel_ttft_prefill_worker_slot(
                    execution_key, scheduler_output, boundary.kv_retention_binding)
                self.scheduler.discard_kv_retention_parked(scheduler_output)
                self._ttft_prefill_suspended.pop(storage_key)
                self._iteration_debug_timings.pop(id(scheduler_output), None)
                self._discard_ttft_prefill_deferred_aborts(scheduler_output)
                self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)
                self._pd_capacity_retry_gate.clear()
                self._notify_pd_engine_progress()
                continue
            present_request_ids = request_ids.intersection(requests)
            if present_request_ids == request_ids:
                continue
            missing_request_ids = request_ids.difference(present_request_ids)
            # Every yielded layer group is GPU-complete, so dropping this
            # worker-owned generator cannot race allocator reuse.
            self._cancel_ttft_prefill_yield_for_output(scheduler_output)
            self._cancel_ttft_prefill_worker_slot(
                execution_key, scheduler_output, boundary.kv_retention_binding)
            self._ttft_prefill_suspended.pop(storage_key, None)
            self._iteration_debug_timings.pop(id(scheduler_output), None)
            self._discard_ttft_prefill_deferred_aborts(scheduler_output)
            if present_request_ids:
                raise RuntimeError(
                    "TTFT whole-batch Prefill continuation lost only part "
                    "of its scheduler ownership: "
                    f"present={sorted(present_request_ids)}, "
                    f"missing={sorted(missing_request_ids)}."
                )

    def _kv_retention_parked_has_precedence_candidate(self, output, phase_queues) -> bool:
        """Use pre-chunk work, never optimistic computed tokens, for A's rank."""
        selector = self.scheduler.ttft_prefill_chunk_scheduler
        pending = self._pd_phase_queue_pending_request_ids(phase_queues)
        remaining = selector.remaining_prefill_tokens
        tail = max((int(remaining(self.scheduler.requests[req_id])) + count,
                    self.scheduler.requests[req_id].arrival_time, req_id)
                   for req_id, count in output.num_scheduled_tokens.items())
        return any((int(remaining(request)), request.arrival_time, req_id) < tail
                   for req_id, request in self.scheduler.requests.items()
                   if req_id not in pending and int(remaining(request)) > 0
                   and not request.is_finished())

    def _poll_kv_retention_parked_offload(self, phase_queues) -> None:
        """Event-scoped capacity relief for a previously Native parked batch."""
        if not getattr(self, "kv_retention_p1_parked_enabled", False):
            return
        if getattr(self, "kv_retention_p1_hbm_only", False):
            if self._kv_retention_parked_offload_future is not None:
                raise RuntimeError("HBM-only parked work cannot own a KV copy")
            return  # Keep KV; failed replacement admission falls back to resume.
        pending = self._kv_retention_parked_offload_future
        if pending is not None:
            future, boundary, output = pending
            if not future.done():
                return
            transfer = future.result()  # Unknown copy outcome is fatal, not free.
            if transfer is not None:
                transfer.validate_output(output)
                transfer.validate_boundary(boundary)
                if self.scheduler.commit_layerwise_kv_offload(output, transfer):
                    output.kv_retention_layerwise_transfer = transfer
            self._kv_retention_parked_offload_future = None
            self._write_pd_phase_queue_trace(
                "kv_retention_p1_late_offload_complete", phase_queues,
                {"released": float(output.kv_retention_layerwise_transfer is not None)})
            self._pd_capacity_retry_gate.notify_state_change("parked_kv_copy_completed")
            self._notify_pd_engine_progress()
            return
        if phase_queues["prefill"] or self._pd_owned_total_depth(phase_queues) >= self.batch_queue_size:
            return
        free = self._pd_capacity_retry_free_blocks()
        causes = {self._pd_capacity_retry_gate.blocking_cause(phase, free)
                  for phase in ("decode", "prefill")}
        if not causes.intersection({"prefill_kv_capacity", "decode_handoff_capacity"}):
            return
        for boundary, output in self._ttft_prefill_suspended.values():
            ticket = getattr(boundary, "kv_retention_parked_ticket", None)
            if (ticket is None or output.kv_retention_layerwise_transfer is not None
                    or self._kv_retention_parked_offload_attempt == ticket.binding
                    or set(output.num_scheduled_tokens) <= self._ttft_prefill_deferred_abort_ids):
                continue
            # One attempt per parked segment. A refused copy must not spin on
            # every Decode token or pretend to have freed capacity.
            self._kv_retention_parked_offload_attempt = ticket.binding
            owner = KVRetentionPhaseOwner.from_output("prefill", output, ticket.binding)
            future = self.model_executor.offload_parked_prefill(owner)
            self._kv_retention_parked_offload_future = (future, boundary, output)
            future.add_done_callback(lambda _: self._notify_pd_engine_progress())
            self._write_pd_phase_queue_trace("kv_retention_p1_late_offload_submit", phase_queues,
                                            self._kv_retention_binding_debug(ticket.binding))
            return

    def _suspend_ttft_prefill_boundary(
        self,
        boundary: TTFTLayerwisePrefillBoundary,
        scheduler_output: SchedulerOutput,
    ) -> bool:
        """Retain one GPU-complete Prefill slot without committing output."""

        boundary.validate_owner(
            scheduler_iteration_id=scheduler_output.scheduler_iteration_id,
            phase_submit_seq=scheduler_output.pd_phase_submit_seq,
            phase_slot=scheduler_output.pd_phase_slot,
            request_ids=tuple(scheduler_output.num_scheduled_tokens),
            scheduled_tokens=tuple(
                (str(request_id), int(num_scheduled_tokens))
                for request_id, num_scheduled_tokens in (
                    scheduler_output.num_scheduled_tokens.items()
                )
            ),
            profile_id=scheduler_output.pd_sm_profile_id,
            profile_epoch=scheduler_output.pd_sm_profile_epoch,
            prefill_mode=scheduler_output.pd_prefill_mode,
        )
        execution_key = self._ttft_prefill_execution_key(scheduler_output)
        ticket = getattr(boundary, "kv_retention_parked_ticket", None)
        if getattr(self, "kv_retention_p1_parked_enabled", False):
            self._cancel_ttft_prefill_yield_for_output(scheduler_output)
        storage_key = f"parked:{ticket.binding.work.work_id}" if ticket is not None else execution_key
        if storage_key in self._ttft_prefill_suspended:
            raise RuntimeError(
                "TTFT Prefill continuation attempted to suspend an already "
                f"owned slot: {execution_key!r}."
            )
        timings = self._iteration_debug_timings.setdefault(
            id(scheduler_output), {}
        )
        previous_yield_count = int(
            timings.get("ttft_layerwise_prefill_last_model_yield_count", 0)
        )
        if boundary.model_yield_count != previous_yield_count + 1:
            raise RuntimeError(
                "TTFT Prefill continuation boundary sequence is not "
                "strictly increasing: "
                f"previous={previous_yield_count}, "
                f"current={boundary.model_yield_count}."
            )
        previous_next_layer = int(
            timings.get("ttft_layerwise_prefill_last_next_layer", -1)
        )
        if previous_next_layer >= 0 and boundary.next_layer <= previous_next_layer:
            raise RuntimeError(
                "TTFT Prefill continuation did not advance its decoder "
                f"layer: previous={previous_next_layer}, "
                f"current={boundary.next_layer}."
            )
        requests = getattr(self.scheduler, "requests", {})
        request_ids = self._scheduler_output_request_ids(scheduler_output)
        present_request_ids = request_ids.intersection(requests)
        if present_request_ids != request_ids:
            self._cancel_ttft_prefill_worker_slot(
                execution_key, scheduler_output, boundary.kv_retention_binding)
            self._discard_ttft_prefill_deferred_aborts(scheduler_output)
            if present_request_ids:
                raise RuntimeError(
                    "TTFT whole-batch Prefill boundary lost only part of "
                    "its scheduler ownership: "
                    f"present={sorted(present_request_ids)}, "
                    f"missing={sorted(request_ids - present_request_ids)}."
                )
            return False
        transfer = boundary.kv_retention_transfer
        if transfer is not None:
            if not isinstance(transfer, KVRetentionLayerwiseTransfer):
                raise RuntimeError("invalid layerwise copy receipt")
            transfer.validate_output(scheduler_output)
            transfer.validate_boundary(boundary)
            previous = scheduler_output.kv_retention_layerwise_transfer
            if previous is not None and previous is not transfer:
                raise RuntimeError("layerwise copy receipt owner was replaced")
            if transfer.state == "copy_ready":
                if self.scheduler.commit_layerwise_kv_offload(scheduler_output, transfer):
                    scheduler_output.kv_retention_layerwise_transfer = transfer
                    self._pd_capacity_retry_gate.notify_state_change("layerwise_kv_released")
                    self._notify_pd_engine_progress()
                else:
                    boundary.kv_retention_transfer = None
                    scheduler_output.kv_retention_layerwise_offload_requested = False
            elif transfer.state != "released":
                raise RuntimeError("invalid suspended layerwise transfer state")
            timings.update(transfer.debug())
        if ticket is not None:
            if not self.kv_retention_p1_parked_enabled:
                raise RuntimeError("unexpected P1 parked ticket")
            ticket.validate(boundary, scheduler_output)
            self.model_executor.release_parked_execute_dependency(
                KVRetentionPhaseOwner.from_output(execution_key, scheduler_output, ticket.binding))
            self.scheduler.park_kv_retention_prefill(scheduler_output)
            scheduler_output.kv_retention_p1_parked = True
            self._pd_capacity_retry_gate.clear()
            self._notify_pd_engine_progress()
        self._ttft_prefill_suspended[storage_key] = (
            boundary,
            scheduler_output,
        )
        self._ttft_prefill_boundary_count += 1
        timings.update(
            {
                "ttft_layerwise_prefill_boundary_count": float(
                    self._ttft_prefill_boundary_count
                ),
                "ttft_layerwise_prefill_boundary_next_layer": float(
                    boundary.next_layer
                ),
                "ttft_layerwise_prefill_boundary_end_layer": float(
                    boundary.end_layer
                ),
                "ttft_layerwise_prefill_scheduler_yield": 1.0,
                "ttft_layerwise_prefill_last_model_yield_count": float(
                    boundary.model_yield_count
                ),
                "ttft_layerwise_prefill_last_next_layer": float(
                    boundary.next_layer
                ),
            }
        )
        if getattr(self, "_tbt_capture", None) is not None:
            self._tbt_capture.write("prefill_boundary",
                                    submit_seq=int(scheduler_output.pd_phase_submit_seq))
        return True

    def _ttft_prefill_resume_candidate(
        self,
        phase_queues,
        *,
        respect_srpt: bool = False,
    ) -> tuple[str | None, str]:
        """Inspect pre-SM resume eligibility without changing any owner.

        Return a storage key, not a new work/slot reservation. Do not gate this
        on the current profile or transition: the planner consumes this demand
        to make that very profile change possible. Actual resume still checks
        committed SM authority afterwards. Normal admission respects SRPT;
        callers may use the existing fallback when no new work was admitted.
        """
        if not self._ttft_prefill_suspended:
            return None, "none"
        if phase_queues["prefill"]:
            return None, "prefill_inflight"
        if getattr(self, "_kv_retention_parked_offload_future", None) is not None:
            return None, "offload_pending"

        storage_key = next(reversed(self._ttft_prefill_suspended))
        boundary, scheduler_output = self._ttft_prefill_suspended[storage_key]
        parked = bool(getattr(boundary, "kv_retention_parked_ticket", None))
        request_ids = set(scheduler_output.num_scheduled_tokens)
        if not request_ids <= self.scheduler.requests.keys():
            # The existing prune path owns cleanup and the partial-loss fatal
            # assertion. SM planning must not run ordered Worker cleanup.
            return None, "owner_cleanup"
        if parked and request_ids <= self._ttft_prefill_deferred_abort_ids:
            return None, "cancel_cleanup"
        if getattr(self, "kv_retention_p1_parked_enabled", False):
            if (self._pd_ttft_handoff_active()
                    and not self._pd_ttft_handoff_request_ids
                    <= self._ttft_prefill_suspended_request_ids()):
                return None, "first_token_handoff"
            if (
                parked
                and self._pd_owned_total_depth(phase_queues) >= self.batch_queue_size
            ):
                return None, "global_slots_full"
        if parked:
            if not self.scheduler.can_resume_kv_retention_prefill(scheduler_output):
                return None, "sequence_budget"
            if respect_srpt and self._kv_retention_parked_has_precedence_candidate(
                scheduler_output, phase_queues
            ):
                return None, "srpt_precedence"
        return storage_key, "ready"

    def _try_resume_ttft_prefill(
        self,
        phase_queues,
        *, respect_srpt: bool = False,
    ) -> bool:
        """Resume the newest (therefore normally shortest) suspended Prefill."""

        if not self._ttft_prefill_suspended:
            return False
        self._prune_ttft_prefill_suspended()
        storage_key, reason = self._ttft_prefill_resume_candidate(
            phase_queues, respect_srpt=respect_srpt
        )
        if storage_key is None:
            if reason == "first_token_handoff":
                self._pd_ttft_handoff_blocked_prefill_checks += 1
            return False
        boundary, scheduler_output = self._ttft_prefill_suspended[storage_key]
        execution_key = self._ttft_prefill_execution_key(scheduler_output)
        parked = bool(getattr(boundary, "kv_retention_parked_ticket", None))
        if self.pd_dynamic_sm_enabled:
            self._assign_pd_sm_plan(
                phase_queues,
                scheduler_output,
                force_prefill_ready=True,
            )
        if self.pd_dynamic_sm_enforced and (
            self.pd_sm_transition.active
            or scheduler_output.pd_sm_prefill_sms <= 0
        ):
            return False

        resume_submit_ts = time.monotonic()
        if parked:
            self.scheduler.resume_kv_retention_prefill(scheduler_output)
            scheduler_output.kv_retention_p1_parked = False
        scheduler_output.kv_retention_layerwise_pressure_checked = False
        scheduler_output.kv_retention_boundary_reason = ""
        timings = self._iteration_debug_timings.setdefault(
            id(scheduler_output), {}
        )
        timings.pop("actual_model_future_done_ts", None)
        binding = self._kv_retention_bind_prefill(scheduler_output, resume=True)
        exec_future = self.model_executor.execute_model(
            scheduler_output,
            non_block=True,
            pd_green_context_phase=execution_key,
            **({"kv_retention_binding": binding} if binding is not None else {}),
        )
        future = cast(Future[ModelRunnerOutput], exec_future)
        self._register_pd_model_future_completion(
            future,
            scheduler_output,
            collect_timing=self.full_iteration_debug_enabled,
        )
        self._ttft_prefill_suspended.pop(storage_key)
        phase_queues["prefill"].appendleft(
            (future, scheduler_output, exec_future)
        )
        self._ttft_prefill_resume_submit_count += 1
        if getattr(self, "_tbt_capture", None) is not None:
            self._tbt_capture.write("prefill_resume",
                                    submit_seq=int(scheduler_output.pd_phase_submit_seq))
        timings.update(
            {
                "ttft_layerwise_prefill_resume_submit_count": float(
                    self._ttft_prefill_resume_submit_count
                ),
                "ttft_layerwise_prefill_last_resume_submit_ts": (
                    resume_submit_ts
                ),
                "ttft_layerwise_prefill_resume_profile": (
                    scheduler_output.pd_sm_profile_id
                ),
                "ttft_layerwise_prefill_resume_profile_epoch": float(
                    scheduler_output.pd_sm_profile_epoch
                ),
                "ttft_layerwise_prefill_resume_mode": scheduler_output.pd_prefill_mode,
            }
        )
        self._write_pd_phase_queue_trace(
            "ttft_prefill_resume_submit",
            phase_queues,
            {
                "execution_key": execution_key,
                "scheduler_output_id": id(scheduler_output),
                "prefill_mode": scheduler_output.pd_prefill_mode,
                "resume_profile": scheduler_output.pd_sm_profile_id,
                "resume_profile_epoch": scheduler_output.pd_sm_profile_epoch,
                **self._kv_retention_binding_debug(binding),
                "resume_submit_count": float(
                    self._ttft_prefill_resume_submit_count
                ),
            },
        )
        return True

    def _enqueue_ttft_prefill_final_sampling(
        self,
        phase_queues,
        scheduler_output: SchedulerOutput,
        exec_future: Future[Any],
    ) -> None:
        """Sample only after a resumed Prefill reaches its final layer."""

        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        execution_key = self._ttft_prefill_execution_key(scheduler_output)
        if execution_key in self._ttft_prefill_suspended:
            raise RuntimeError(
                "TTFT Prefill final sampling cannot begin while the same "
                f"phase slot remains suspended: {execution_key!r}."
            )
        self._iteration_debug_timings.setdefault(
            id(scheduler_output), {}
        ).pop("actual_model_future_done_ts", None)
        binding = self._kv_retention_future_binding(exec_future, scheduler_output)
        if binding is not None:
            self._kv_retention_prefill_work_registry.sampling(binding)
        future = self.model_executor.sample_tokens(
            grammar_output,
            non_block=True,
            pd_green_context_phase=execution_key,
            **({"kv_retention_binding": binding} if binding is not None else {}),
        )
        self._register_pd_model_future_completion(
            future,
            scheduler_output,
            collect_timing=self.full_iteration_debug_enabled,
        )
        phase_queues["prefill"].appendleft(
            (future, scheduler_output, exec_future)
        )
        self._ttft_prefill_final_execute_count += 1
        self._iteration_debug_timings.setdefault(
            id(scheduler_output), {}
        )["ttft_layerwise_prefill_final_execute_count"] = float(
            self._ttft_prefill_final_execute_count
        )

    def _submit_to_pd_phase_queue(
        self,
        phase_queues,
        scheduler_output: SchedulerOutput,
        schedule_start_ts: float,
        schedule_end_ts: float,
        queue_depth_before_submit: int,
    ) -> tuple[bool, SchedulerOutput | None, Future[Any]]:
        if self.pd_dynamic_sm_enabled:
            self._assign_pd_sm_plan(phase_queues, scheduler_output)
        execute_submit_ts = time.monotonic()
        phase = self._pd_phase_queue_for_output(scheduler_output)
        self._annotate_pd_component_overlap_candidate(
            phase_queues, scheduler_output
        )
        execution_key = self._assign_pd_phase_submission(
            phase_queues, scheduler_output, phase
        )
        model_executed = bool(
            not self.is_ec_producer
            and scheduler_output.total_num_scheduled_tokens > 0
        )
        scheduler_output.ttft_layerwise_prefill_preemption = bool(
            self.ttft_layerwise_prefill_preemption_enabled
            and phase == "prefill"
            and ttft_prefill_continuation_mode_allowed(
                scheduler_output.pd_prefill_mode,
                p1_ring_sm_handoff=getattr(
                    self, "isolation_p1_ring_sm_handoff_enabled", False))
            and bool(scheduler_output.num_scheduled_tokens)
            and model_executed
            and not scheduler_output.pending_structured_output_tokens
        )
        fused_execute_sample = bool(
            self.pd_fused_execute_sample_enabled
            and model_executed
            and not scheduler_output.ttft_layerwise_prefill_preemption
            and not self.is_pooling_model
            and not scheduler_output.pending_structured_output_tokens
        )
        binding = self._kv_retention_bind_prefill(scheduler_output)
        if fused_execute_sample:
            grammar_output = self.scheduler.get_grammar_bitmask(
                scheduler_output
            )
            future = self.model_executor.execute_model_and_sample(
                scheduler_output,
                grammar_output,
                non_block=True,
                pd_green_context_phase=execution_key,
            )
            # Queue recovery retains the execute Future only to surface a
            # worker exception when sampling unexpectedly returns None.  The
            # fused RPC owns both stages, so the same Future is authoritative.
            exec_future = future
        else:
            exec_future = self.model_executor.execute_model(
                scheduler_output,
                non_block=True,
                pd_green_context_phase=execution_key,
                **({"kv_retention_binding": binding} if binding is not None else {}),
            )
        # Arm the TTFT handoff before the submit loop can refill another
        # Prefill slot.  This also covers deferred grammar sampling: the model
        # forward is already in flight and must fence following Prefill work.
        self._pd_ttft_handoff_mark_submitted(scheduler_output)
        collect_detailed_timing = self.full_iteration_debug_enabled
        timing_record = self._pd_phase_submit_timing_record(
            scheduler_output,
            phase=phase,
            execution_key=execution_key,
            schedule_start_ts=schedule_start_ts,
            schedule_end_ts=schedule_end_ts,
            execute_submit_ts=execute_submit_ts,
            queue_depth_before_submit=queue_depth_before_submit,
            paired_launch=False,
        )
        self._iteration_debug_timings[id(scheduler_output)] = timing_record
        timing_record["pd_fused_execute_sample_enabled"] = float(
            self.pd_fused_execute_sample_enabled
        )
        timing_record["pd_fused_execute_sample_used"] = float(
            fused_execute_sample
        )

        deferred_scheduler_output = None
        if fused_execute_sample:
            pass
        elif scheduler_output.ttft_layerwise_prefill_preemption:
            # The execute Future is authoritative until the model reaches its
            # final layer. Intermediate results are Engine control boundaries,
            # never sampling dependencies.
            future = cast(Future[ModelRunnerOutput], exec_future)
        elif self.is_pooling_model or not model_executed:
            future = cast(Future[ModelRunnerOutput], exec_future)
        else:
            if not scheduler_output.pending_structured_output_tokens:
                grammar_output = self.scheduler.get_grammar_bitmask(
                    scheduler_output
                )
                future = self.model_executor.sample_tokens(
                    grammar_output,
                    non_block=True,
                    pd_green_context_phase=execution_key,
                )
            else:
                deferred_scheduler_output = scheduler_output

        if deferred_scheduler_output is None:
            self._register_pd_model_future_completion(
                future,
                scheduler_output,
                collect_timing=collect_detailed_timing,
            )
            phase_queues[phase].appendleft((future, scheduler_output, exec_future))
            if collect_detailed_timing:
                timing_record["outstanding_model_futures_after"] = float(
                    self._pd_phase_queue_total_depth(phase_queues)
                )
                timing_record["engine_batch_queue_depth_after_submit"] = float(
                    self._pd_phase_queue_total_depth(phase_queues)
                )
                timing_record.update(self._pd_phase_queues_counts(phase_queues))
            self._write_pd_phase_queue_trace(
                "submit",
                phase_queues,
                {
                    "submitted_phase": phase,
                    "submitted_scheduler_output_id": id(scheduler_output),
                    "submitted_total_tokens": float(
                        scheduler_output.total_num_scheduled_tokens
                    ),
                    "submitted_phase_slot": float(
                        scheduler_output.pd_phase_slot
                    ),
                    "submitted_execution_key": execution_key,
                    **self._kv_retention_binding_debug(binding),
                    "submitted_phase_seq": float(
                        scheduler_output.pd_phase_submit_seq
                    ),
                },
            )

        return model_executed, deferred_scheduler_output, exec_future

    def _enqueue_deferred_pd_phase_sampling(
        self,
        phase_queues,
        scheduler_output: SchedulerOutput,
        exec_future: Future[Any],
        *,
        schedule_start_ts: float,
        schedule_end_ts: float,
    ) -> None:
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        deferred_phase = self._pd_phase_queue_for_output(scheduler_output)
        execution_key = self._pd_phase_execution_key(
            scheduler_output, deferred_phase
        )
        self._iteration_debug_timings[id(scheduler_output)] = {
            "schedule_start_ts": schedule_start_ts,
            "schedule_end_ts": schedule_end_ts,
            "execute_submit_ts": time.monotonic(),
            "outstanding_model_futures_before": float(
                self._pd_phase_queue_total_depth(phase_queues)
            ),
            "engine_batch_queue_depth_before_submit": float(
                self._pd_phase_queue_total_depth(phase_queues)
            ),
            "pd_independent_phase_queues_enabled": 1.0,
            "pd_separate_scheduler_lanes_enabled": float(
                self.pd_separate_scheduler_lanes_enabled
            ),
            "pd_dual_continuous_batching_enabled": float(
                self.pd_dual_continuous_batching_enabled
            ),
            "pd_nonblocking_phase_recovery_enabled": float(
                self.pd_nonblocking_phase_recovery_enabled
            ),
            "pd_single_inflight_per_phase_enabled": float(
                self.pd_single_inflight_per_phase_enabled
            ),
            "pd_paired_phase_launch_requested": float(
                self.pd_paired_phase_launch_requested
            ),
            "pd_paired_phase_launch_enabled": float(
                self.pd_paired_phase_launch_enabled
            ),
            "pd_paired_phase_launch_used": 0.0,
            "pd_phase_depth_enabled": float(self.pd_phase_depth_enabled),
            "pd_prefill_inflight_depth": float(self.pd_prefill_inflight_depth),
            "pd_decode_inflight_depth": float(self.pd_decode_inflight_depth),
            "pd_global_inflight_depth": float(self.pd_global_inflight_depth),
            "pd_phase_queue_submit_phase": deferred_phase,
            "pd_phase_queue_submit_slot": float(scheduler_output.pd_phase_slot),
            "pd_phase_queue_submit_execution_key": execution_key,
            "pd_phase_queue_submit_seq": float(
                scheduler_output.pd_phase_submit_seq
            ),
        }
        future = self.model_executor.sample_tokens(
            grammar_output,
            non_block=True,
            pd_green_context_phase=execution_key,
        )
        self._register_pd_model_future_completion(
            future,
            scheduler_output,
            collect_timing=self.full_iteration_debug_enabled,
        )
        phase_queues[deferred_phase].appendleft(
            (future, scheduler_output, exec_future)
        )
        self._iteration_debug_timings[id(scheduler_output)][
            "outstanding_model_futures_after"
        ] = float(self._pd_phase_queue_total_depth(phase_queues))
        self._iteration_debug_timings[id(scheduler_output)][
            "engine_batch_queue_depth_after_submit"
        ] = float(self._pd_phase_queue_total_depth(phase_queues))
        self._iteration_debug_timings[id(scheduler_output)].update(
            self._pd_phase_queues_counts(phase_queues)
        )
        self._write_pd_phase_queue_trace(
            "submit_deferred_sampling",
            phase_queues,
            {
                "submitted_phase": deferred_phase,
                "submitted_scheduler_output_id": id(scheduler_output),
                "submitted_total_tokens": float(
                    scheduler_output.total_num_scheduled_tokens
                ),
            },
        )

    @staticmethod
    def _pd_phase_queue_for_output(scheduler_output: SchedulerOutput) -> str:
        phase = EngineCore._scheduler_output_phase(scheduler_output)
        if phase in {"decode", "prefill", "mixed", "idle"}:
            return phase
        return "mixed"

    def _fail_tiered_backend_scheduler_output(
        self,
        scheduler_output: SchedulerOutput,
        error: TieredBackendEligibilityError,
    ) -> dict[int, EngineCoreOutputs]:
        """Convert an expected tiered rejection into request-local outputs."""
        fail_scheduled_requests = getattr(
            self.scheduler, "fail_scheduled_requests", None
        )
        if fail_scheduled_requests is None:
            raise error
        message = f"{type(error).__name__}: {error}"
        logger.error(
            "Tiered backend rejected scheduled request(s); failing only %s: %s",
            sorted(scheduler_output.num_scheduled_tokens),
            message,
        )
        free_blocks_before = self._pd_capacity_retry_free_blocks()
        engine_core_outputs = fail_scheduled_requests(
            scheduler_output, message
        )
        retry_wake_count = self._notify_pd_capacity_retry_after_output(
            scheduler_output,
            engine_core_outputs,
            free_blocks_before,
        )
        free_blocks_after = self._pd_capacity_retry_free_blocks()
        if scheduler_output.iteration_debug is None:
            scheduler_output.iteration_debug = {}
        scheduler_output.iteration_debug.update({
            "tiered_failure_free_blocks_before": free_blocks_before,
            "tiered_failure_free_blocks_after": free_blocks_after,
            "tiered_failure_capacity_retry_wake_count": retry_wake_count,
        })
        # Future completion may have signalled before scheduler ownership and
        # KV blocks were released. Publish a second, post-cleanup generation so
        # an event-driven engine cannot go back to sleep with newly feasible
        # Prefill work still behind the capacity gate.
        self._notify_pd_engine_progress()
        return engine_core_outputs

    @staticmethod
    def _scheduler_has_pd_phase_work(
        scheduler: SchedulerInterface,
        phase: str,
    ) -> bool:
        has_phase_work = getattr(scheduler, "has_pd_phase_work", None)
        if has_phase_work is None:
            return scheduler.has_requests()
        return bool(has_phase_work(phase))

    @staticmethod
    def _scheduler_schedule_pd_phase(
        scheduler: SchedulerInterface,
        phase: str,
    ) -> SchedulerOutput:
        schedule_phase = getattr(scheduler, "schedule_pd_phase", None)
        if schedule_phase is None:
            return scheduler.schedule()
        return schedule_phase(phase)

    @staticmethod
    def _pop_pd_phase_queue_at(phase_queues, phase: str, index: int):
        queue_ = phase_queues[phase]
        if index == len(queue_) - 1:
            return queue_.pop()
        queue_.rotate(-index)
        item = queue_.popleft()
        queue_.rotate(index)
        return item

    def _oldest_pd_phase_queue_item(self, phase_queues) -> dict[str, Any]:
        items = self._pd_phase_queue_items(phase_queues)
        assert items
        return min(items, key=lambda item: item["submit_seq"])

    def _has_older_request_dependency(
        self,
        candidate: dict[str, Any],
        items: list[dict[str, Any]],
    ) -> bool:
        candidate_req_ids = self._scheduler_output_request_ids(
            candidate["scheduler_output"]
        )
        for item in items:
            if item["submit_seq"] >= candidate["submit_seq"]:
                continue
            older_req_ids = self._scheduler_output_request_ids(
                item["scheduler_output"]
            )
            if candidate_req_ids & older_req_ids:
                return True
        return False

    def _select_pd_phase_queue_recovery_item(
        self,
        phase_queues,
        *,
        collect_debug: bool = True,
    ) -> tuple[str, int, dict[str, Any]]:
        items = sorted(
            self._pd_phase_queue_items(phase_queues),
            key=lambda item: item["submit_seq"],
        )
        oldest = items[0]
        debug: dict[str, Any] = {}
        if collect_debug:
            debug = {
                **self._pd_phase_queues_counts(phase_queues),
                "pd_phase_queue_recovery_used": 0.0,
                "pd_muxwise_event_loop_enabled": float(
                    self.pd_muxwise_event_loop_enabled
                ),
                "pd_phase_queue_selected_phase": oldest["phase"],
                "pd_phase_queue_oldest_phase": oldest["phase"],
                "pd_phase_queue_selected_submit_seq": oldest["submit_seq"],
                "pd_phase_queue_oldest_submit_seq": oldest["submit_seq"],
                "pd_phase_queue_candidate_count": 0.0,
                "pd_phase_queue_blocked_dependency_count": 0.0,
                "pd_phase_queue_blocked_global_state": 0.0,
                "pd_phase_queue_skipped_older_future_count": 0.0,
                "pd_dual_continuous_batching_enabled": float(
                    self.pd_dual_continuous_batching_enabled
                ),
                "pd_dual_continuous_waiting_on_decode": 0.0,
                "pd_nonblocking_phase_recovery_enabled": float(
                    self.pd_nonblocking_phase_recovery_enabled
                ),
                "pd_single_inflight_per_phase_enabled": float(
                    self.pd_single_inflight_per_phase_enabled
                ),
            }

        if any(
            item["scheduler_output"].has_structured_output_requests
            or item["scheduler_output"].pending_structured_output_tokens
            for item in items
        ):
            if collect_debug:
                debug["pd_phase_queue_blocked_global_state"] = 1.0
            return oldest["phase"], int(oldest["index"]), debug

        for preferred_phase in ("decode", "prefill", "mixed", "idle"):
            for item in items:
                if item["phase"] != preferred_phase:
                    continue
                if not item["future"].done():
                    continue
                if collect_debug:
                    debug["pd_phase_queue_candidate_count"] += 1.0
                if self._has_older_request_dependency(item, items):
                    if collect_debug:
                        debug["pd_phase_queue_blocked_dependency_count"] += 1.0
                    continue

                if collect_debug:
                    skipped_older = sum(
                        1
                        for older in items
                        if older["submit_seq"] < item["submit_seq"]
                    )
                    debug.update(
                        {
                            "pd_phase_queue_recovery_used": float(
                                item is not oldest
                            ),
                            "pd_phase_queue_selected_phase": item["phase"],
                            "pd_phase_queue_selected_submit_seq": item[
                                "submit_seq"
                            ],
                            "pd_phase_queue_skipped_older_future_count": float(
                                skipped_older
                            ),
                        }
                    )
                return item["phase"], int(item["index"]), debug

        if self.pd_dual_continuous_batching_enabled:
            for item in items:
                if item["phase"] != "decode":
                    continue
                if self._has_older_request_dependency(item, items):
                    if collect_debug:
                        debug["pd_phase_queue_blocked_dependency_count"] += 1.0
                    continue
                if collect_debug:
                    skipped_older = sum(
                        1
                        for older in items
                        if older["submit_seq"] < item["submit_seq"]
                    )
                    debug.update(
                        {
                            "pd_phase_queue_recovery_used": float(
                                item is not oldest
                            ),
                            "pd_phase_queue_selected_phase": item["phase"],
                            "pd_phase_queue_selected_submit_seq": item[
                                "submit_seq"
                            ],
                            "pd_phase_queue_skipped_older_future_count": float(
                                skipped_older
                            ),
                            "pd_dual_continuous_waiting_on_decode": 1.0,
                        }
                    )
                return item["phase"], int(item["index"]), debug

        return oldest["phase"], int(oldest["index"]), debug

    @staticmethod
    def _pop_batch_queue_at(batch_queue, index: int):
        if index == len(batch_queue) - 1:
            return batch_queue.pop()
        batch_queue.rotate(-index)
        item = batch_queue.popleft()
        batch_queue.rotate(index)
        return item

    def _select_phase_aware_recovery_index(
        self,
        batch_queue,
    ) -> tuple[int, dict[str, Any]]:
        oldest_index = len(batch_queue) - 1
        oldest_future, oldest_output, _ = batch_queue[oldest_index]
        debug: dict[str, Any] = {
            "pd_phase_aware_recovery_enabled": float(
                self.pd_decoupled_phase_aware_recovery
            ),
            "phase_aware_recovery_used": 0.0,
            "phase_aware_recovery_candidate_count": 0.0,
            "phase_aware_recovery_blocked_dependency_count": 0.0,
            "phase_aware_recovery_blocked_global_state": 0.0,
            "phase_aware_recovery_pop_index": 0.0,
            "phase_aware_recovery_skipped_older_future_count": 0.0,
            "phase_aware_recovery_oldest_phase": self._scheduler_output_phase(
                oldest_output
            ),
            "phase_aware_recovery_selected_phase": self._scheduler_output_phase(
                oldest_output
            ),
        }

        if (
            not self.pd_decoupled_phase_aware_recovery
            or oldest_future.done()
        ):
            return oldest_index, debug

        # Spec decode and structured-output requests carry global ordering
        # assumptions through draft ids / grammar masks. Keep FIFO until a
        # dedicated dependency model is added for those paths.
        if self.use_spec_decode:
            debug["phase_aware_recovery_blocked_global_state"] = 1.0
            return oldest_index, debug
        if any(
            scheduler_output.has_structured_output_requests
            or scheduler_output.pending_structured_output_tokens
            for _, scheduler_output, _ in batch_queue
        ):
            debug["phase_aware_recovery_blocked_global_state"] = 1.0
            return oldest_index, debug

        for index in range(oldest_index - 1, -1, -1):
            future, scheduler_output, _ = batch_queue[index]
            if not future.done():
                continue
            if self._scheduler_output_phase(scheduler_output) != "decode":
                continue

            debug["phase_aware_recovery_candidate_count"] += 1.0
            candidate_req_ids = self._scheduler_output_request_ids(scheduler_output)
            has_dependency = False
            for older_index in range(index + 1, len(batch_queue)):
                _, older_output, _ = batch_queue[older_index]
                if candidate_req_ids & self._scheduler_output_request_ids(
                    older_output
                ):
                    has_dependency = True
                    break
            if has_dependency:
                debug["phase_aware_recovery_blocked_dependency_count"] += 1.0
                continue

            debug.update(
                {
                    "phase_aware_recovery_used": 1.0,
                    "phase_aware_recovery_pop_index": float(
                        oldest_index - index
                    ),
                    "phase_aware_recovery_skipped_older_future_count": float(
                        oldest_index - index
                    ),
                    "phase_aware_recovery_selected_phase": (
                        self._scheduler_output_phase(scheduler_output)
                    ),
                }
            )
            return index, debug

        return oldest_index, debug

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        schedule_start_ts = time.monotonic()
        scheduler_output = self.scheduler.schedule()
        schedule_end_ts = time.monotonic()
        execute_submit_ts = time.monotonic()
        scheduler_output_phase = self._scheduler_output_phase(scheduler_output)
        future = self.model_executor.execute_model(
            scheduler_output,
            non_block=True,
            pd_green_context_phase=scheduler_output_phase,
        )
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            result_wait_start_ts = execute_submit_ts
            tiered_eligibility_error = None
            try:
                model_output = future.result()
                model_ready_ts = time.monotonic()
                if model_output is None:
                    model_output = self.model_executor.sample_tokens(
                        grammar_output,
                        pd_green_context_phase=scheduler_output_phase,
                    )
                model_done_ts = time.monotonic()
            except TieredBackendEligibilityError as error:
                tiered_eligibility_error = error
                model_output = None
                model_ready_ts = model_done_ts = time.monotonic()

        if tiered_eligibility_error is not None:
            self._process_aborts_queue()
            engine_core_outputs = self._fail_tiered_backend_scheduler_output(
                scheduler_output,
                tiered_eligibility_error,
            )
            output_process_end_ts = time.monotonic()
            self._write_iteration_debug_jsonl(
                scheduler_output,
                schedule_start_ts,
                schedule_end_ts,
                execute_submit_ts,
                result_wait_start_ts,
                model_ready_ts,
                model_done_ts,
                output_process_end_ts,
                {
                    "scheduler_output_phase": scheduler_output_phase,
                    "tiered_backend_request_local_failure": 1.0,
                    "outstanding_model_futures_before": 0.0,
                    "outstanding_model_futures_after": 0.0,
                },
                {},
            )
            return (
                engine_core_outputs,
                scheduler_output.total_num_scheduled_tokens > 0,
            )

        assert model_output is not None

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        retry_free_blocks_before = self._pd_capacity_retry_free_blocks()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        self._notify_pd_capacity_retry_after_output(
            scheduler_output,
            engine_core_outputs,
            retry_free_blocks_before,
        )
        output_process_end_ts = time.monotonic()
        self._write_iteration_debug_jsonl(
            scheduler_output,
            schedule_start_ts,
            schedule_end_ts,
            execute_submit_ts,
            result_wait_start_ts,
            model_ready_ts,
            model_done_ts,
            output_process_end_ts,
            {
                "scheduler_output_phase": self._scheduler_output_phase(
                    scheduler_output
                ),
                "pd_phase_aware_recovery_enabled": float(
                    self.pd_decoupled_phase_aware_recovery
                ),
                "phase_aware_recovery_used": 0.0,
                "outstanding_model_futures_before": 0.0,
                "outstanding_model_futures_after": 0.0,
                "engine_batch_queue_depth_before_submit": 0.0,
                "engine_batch_queue_depth_after_submit": 0.0,
                "result_queue_depth_before_pop": 0.0,
                "engine_batch_queue_depth_before_pop": 0.0,
                "ready_future_count": float(future.done()),
                "not_ready_future_count": float(not future.done()),
                "future_done_before_wait_start": 0.0,
                "future_done_at_poll": 0.0,
                "future_poll_order_index": 0.0,
                "blocked_by_earlier_future_count": 0.0,
            },
            getattr(model_output, "model_runner_debug_timings", {}),
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def _append_jsonl(self, path: str, record: dict[str, Any]) -> None:
        writer = self._debug_trace_writer
        if writer is not None:
            writer.submit(path, record)

    def _write_iteration_debug_jsonl(
        self,
        scheduler_output: SchedulerOutput,
        schedule_start_ts: float,
        schedule_end_ts: float,
        execute_submit_ts: float,
        result_wait_start_ts: float,
        model_ready_ts: float,
        model_done_ts: float,
        output_process_end_ts: float,
        engine_debug_timings: dict[str, float] | None = None,
        model_runner_debug_timings: dict[str, Any] | None = None,
    ) -> None:
        if not self.iteration_metrics_jsonl and not self.decode_timeline_jsonl:
            return

        iteration_id = int(scheduler_output.scheduler_iteration_id)
        if iteration_id < 0 or (
            iteration_id % self.iteration_metrics_sample_every_n != 0
        ):
            return
        iteration_debug = scheduler_output.iteration_debug or {}
        if self.iteration_metrics_jsonl:
            engine_timings = engine_debug_timings or {}
            debug_timings = model_runner_debug_timings or {}
            serialized_debug_timings = (
                compact_worker_debug_timings(debug_timings)
                if (
                    self.strict_bounded_lightweight_timeline_enabled
                    and not self.full_iteration_debug_enabled
                )
                else debug_timings
            )
            worker_execute_duration = debug_timings.get(
                "worker_execute_duration_s", 0.0
            )
            worker_sample_tokens_duration = debug_timings.get(
                "worker_sample_tokens_duration_s", 0.0
            )
            worker_observed_duration = (
                worker_execute_duration + worker_sample_tokens_duration
            )
            actual_future_done_ts = engine_timings.get(
                "actual_model_future_done_ts", model_ready_ts
            )
            record: dict[str, Any] = {
                **iteration_debug,
                "scheduled_phase": self._scheduler_output_phase(scheduler_output),
                "schedule_start_ts": schedule_start_ts,
                "schedule_end_ts": schedule_end_ts,
                "execute_submit_ts": execute_submit_ts,
                "result_wait_start_ts": result_wait_start_ts,
                "model_ready_ts": model_ready_ts,
                "model_done_ts": model_done_ts,
                "output_process_end_ts": output_process_end_ts,
                **engine_timings,
                "actual_model_future_done_ts": actual_future_done_ts,
                "future_done_ts_available": float(
                    "actual_model_future_done_ts" in engine_timings
                ),
                "future_ready_to_poll_delay_s": max(
                    0.0, result_wait_start_ts - actual_future_done_ts
                ),
                "future_wait_after_poll_s": max(
                    0.0, actual_future_done_ts - result_wait_start_ts
                ),
                "schedule_duration_s": schedule_end_ts - schedule_start_ts,
                "model_wait_duration_s": model_ready_ts - execute_submit_ts,
                "batch_queue_delay_duration_s": (
                    result_wait_start_ts - execute_submit_ts
                ),
                "future_wait_duration_s": model_ready_ts - result_wait_start_ts,
                "sample_duration_s": model_done_ts - model_ready_ts,
                "output_process_duration_s": (
                    output_process_end_ts - model_done_ts
                ),
                "iteration_duration_s": (
                    output_process_end_ts - schedule_start_ts
                ),
                **serialized_debug_timings,
                "worker_observed_duration_s": worker_observed_duration,
                # The worker durations above measure host enqueue work unless
                # explicit CUDA timing is enabled.  Keep the legacy field for
                # analyzers, but also give the residual an accurate name: it
                # includes device execution/D2H completion and is not purely
                # executor idle time.
                "executor_async_completion_residual_duration_s": max(
                    0.0,
                    (model_ready_ts - execute_submit_ts)
                    - worker_observed_duration,
                ),
                "executor_unattributed_wait_duration_s": max(
                    0.0,
                    (model_ready_ts - execute_submit_ts) - worker_observed_duration,
                ),
            }
            self._append_jsonl(self.iteration_metrics_jsonl, record)

        if self.decode_timeline_jsonl:
            timestamp = float(iteration_debug.get("scheduled_ts", schedule_end_ts))
            for request_debug in (scheduler_output.request_debug or {}).values():
                record = {
                    "iteration_id": iteration_id,
                    "timestamp": timestamp,
                    **request_debug,
                }
                self._append_jsonl(self.decode_timeline_jsonl, record)

    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        if not self.async_scheduling and self.use_spec_decode and model_executed:
            # Take the draft token ids.
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    def step_with_pd_phase_queues(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with independent P/D phase queues.

        This is a conservative prototype for P/D decoupling. Decode and prefill
        SchedulerOutputs are submitted into separate result queues, but recovery
        may bypass an older future only when the newer ready future has no
        request-id dependency on older queued futures.
        """
        phase_queues = self.pd_phase_queues
        assert phase_queues is not None
        self._pd_capacity_retry_suppressed_this_step = False
        self._pd_progress_waited_this_step = False
        self._pd_progress_generation_at_step_start = (
            pd_engine_progress_generation()
        )

        # Observe ready Decode before optional control work. The existing
        # selector, not queue position alone, proves request independence.
        output_first = getattr(self, "isolation_decode_output_first", False)
        defer_control_for_decode = False
        if output_first and self._pd_phase_queue_total_depth(phase_queues):
            early_phase, early_index, _ = self._select_pd_phase_queue_recovery_item(
                phase_queues, collect_debug=False)
            defer_control_for_decode = (
                early_phase == "decode" and phase_queues[early_phase][early_index][0].done())
        if output_first:
            self._write_pd_phase_queue_trace("output_first_check", phase_queues, {
                "isolation_decode_control_deferred": float(defer_control_for_decode),
            })

        # A pending whole-forward profile change cannot rely on an accidental
        # empty queue: continuous phase admission keeps at least one future
        # alive. Advance the explicit drain fence before considering any new
        # scheduler submission.
        if not defer_control_for_decode:
            self._advance_pd_sm_transition(phase_queues)
            self._poll_kv_retention_parked_offload(phase_queues)
            self._poll_pd_layerwise_handoff(phase_queues)
            self._prune_ttft_prefill_suspended()
            self._request_ttft_prefill_preemption_if_needed(phase_queues)

        def total_depth() -> int:
            return self._pd_phase_queue_total_depth(phase_queues)

        # Try to schedule a new batch if the shared outstanding-future budget
        # is not full. In dual-continuous mode a post-decode resubmit can leave
        # the phase queues full at the next loop entry, so a full queue simply
        # falls through to result recovery.

        model_executed = False
        deferred_scheduler_output = None
        deferred_exec_future = None
        ready_recovery_before_submit = False
        decode_ready_recovery_before_submit = False
        if total_depth() > 0:
            early_phase, early_index, _ = (
                self._select_pd_phase_queue_recovery_item(
                    phase_queues,
                    collect_debug=False,
                )
            )
            ready_recovery_before_submit = phase_queues[early_phase][
                early_index
            ][0].done()
            decode_ready_recovery_before_submit = (
                ready_recovery_before_submit and early_phase == "decode"
            )
        scheduler_has_requests = self.scheduler.has_requests()
        if scheduler_has_requests and not ready_recovery_before_submit:
            submitted_any = False
            if self.pd_separate_scheduler_lanes_enabled:
                (
                    submitted_model_executed,
                    deferred_scheduler_output,
                    deferred_exec_future,
                    submitted_any,
                ) = self._submit_pd_scheduler_lanes(
                    phase_queues,
                    ("decode", "prefill"),
                    allow_multiple_rounds=True,
                )
                model_executed = model_executed or submitted_model_executed
            elif total_depth() < self.batch_queue_size:
                self._set_scheduler_pd_pending_request_ids(
                    self._pd_phase_queue_pending_request_ids(phase_queues)
                )
                self._set_scheduler_pd_pending_phase_request_ids(
                    self._pd_phase_queue_pending_request_ids_by_phase(
                        phase_queues
                    )
                )
                queue_depth_before_submit = total_depth()
                schedule_start_ts = time.monotonic()
                scheduler_output = self.scheduler.schedule()
                schedule_end_ts = time.monotonic()
                (
                    model_executed,
                    deferred_scheduler_output,
                    deferred_exec_future,
                ) = self._submit_to_pd_phase_queue(
                    phase_queues,
                    scheduler_output,
                    schedule_start_ts,
                    schedule_end_ts,
                    queue_depth_before_submit,
                )
                submitted_any = True
            self._set_scheduler_pd_pending_request_ids(set())
            self._set_scheduler_pd_pending_phase_request_ids({})

            if (
                self._ttft_prefill_suspended
                and not phase_queues["prefill"]
                and self._pd_owned_total_depth(phase_queues)
                < self.batch_queue_size
            ):
                resumed_prefill = self._try_resume_ttft_prefill(
                    phase_queues
                )
                model_executed = model_executed or resumed_prefill

            if submitted_any and deferred_scheduler_output is None:
                oldest = self._oldest_pd_phase_queue_item(phase_queues)
                if (
                    not self.pd_dual_continuous_batching_enabled
                    and model_executed
                    and total_depth() < self.batch_queue_size
                    and not oldest["future"].done()
                ):
                    return None, True
            if (
                not submitted_any
                and deferred_scheduler_output is None
                and total_depth() == 0
            ):
                if self._ttft_prefill_suspended:
                    if not self._try_resume_ttft_prefill(phase_queues):
                        # Planning a resume can request a whole-profile move.
                        # With no physical Future the yielded layer boundary is
                        # already quiescent, so commit and retry immediately.
                        self._advance_pd_sm_transition(phase_queues)
                        if not self._try_resume_ttft_prefill(phase_queues):
                            self._wait_for_pd_progress(phase_queues)
                    return None, True
                if self._pd_capacity_retry_suppressed_this_step:
                    self._pd_capacity_retry_gate.note_wait()
                self._wait_for_pd_progress(phase_queues)
                self._write_pd_phase_queue_trace(
                    "idle_no_submit",
                    phase_queues,
                    {"scheduler_has_requests": float(self.scheduler.has_requests())},
                )
                return None, self.scheduler.has_requests()
            if deferred_scheduler_output is not None and total_depth() == 0:
                # This can only happen for structured-output sampling. Keep the
                # upstream behavior by continuing to the deferred sample path
                # after an older queued future has been recovered.
                self._write_pd_phase_queue_trace(
                    "deferred_without_recoverable_queue",
                    phase_queues,
                    {"deferred_scheduler_output_id": id(deferred_scheduler_output)},
                )
                return None, True

        elif not scheduler_has_requests and total_depth() == 0:
            self._write_pd_phase_queue_trace("idle_empty", phase_queues)
            return None, False

        result_queue_depth_before_pop = total_depth()
        ready_future_count = sum(
            1
            for queue_ in phase_queues.values()
            for queued_future, _, _ in queue_
            if queued_future.done()
        )
        not_ready_future_count = result_queue_depth_before_pop - ready_future_count
        oldest = self._oldest_pd_phase_queue_item(phase_queues)
        oldest_future_done = oldest["future"].done()
        collect_recovery_debug = bool(
            self.pd_phase_queue_trace_jsonl
            or self.full_iteration_debug_enabled
        )
        phase, pop_index, phase_recovery_record = (
            self._select_pd_phase_queue_recovery_item(
                phase_queues,
                collect_debug=collect_recovery_debug,
            )
        )
        selected_future = phase_queues[phase][pop_index][0]
        selected_scheduler_output = phase_queues[phase][pop_index][1]
        selected_future_done = selected_future.done()
        prefill_pending_at_recovery = len(phase_queues["prefill"])
        prefill_not_ready_at_recovery = sum(
            1
            for queued_future, _, _ in phase_queues["prefill"]
            if not queued_future.done()
        )
        selected_timing_record = self._iteration_debug_timings.get(
            id(selected_scheduler_output), {}
        )
        selected_timing_record.update(self._pd_capacity_retry_snapshot())
        collect_detailed_timing = self.full_iteration_debug_enabled
        if collect_detailed_timing:
            selected_timing_record.update(
                {
                    **phase_recovery_record,
                    "scheduler_output_phase": self._scheduler_output_phase(
                        selected_scheduler_output
                    ),
                    "result_queue_depth_before_pop": float(
                        result_queue_depth_before_pop
                    ),
                    "engine_batch_queue_depth_before_pop": float(
                        result_queue_depth_before_pop
                    ),
                    "ready_future_count": float(ready_future_count),
                    "not_ready_future_count": float(not_ready_future_count),
                    "future_done_before_wait_start": float(
                        selected_future_done
                    ),
                    "future_done_at_poll": float(selected_future_done),
                    "future_poll_order_index": float(
                        phase_recovery_record.get(
                            "pd_phase_queue_skipped_older_future_count", 0.0
                        )
                    ),
                    "blocked_by_earlier_future_count": float(
                        ready_future_count if not oldest_future_done else 0
                    ),
                    "pd_decoupled_decode_recovered_with_prefill_pending": float(
                        self.pd_decoupled_paired_recovery_enabled
                        and phase == "decode"
                        and selected_future_done
                        and prefill_pending_at_recovery > 0
                    ),
                    "pd_decoupled_decode_recovered_with_prefill_not_ready": float(
                        self.pd_decoupled_paired_recovery_enabled
                        and phase == "decode"
                        and selected_future_done
                        and prefill_not_ready_at_recovery > 0
                    ),
                    "pd_decoupled_prefill_pending_at_decode_recovery": float(
                        prefill_pending_at_recovery
                        if phase == "decode"
                        else 0
                    ),
                    "pd_decoupled_prefill_not_ready_at_decode_recovery": float(
                        prefill_not_ready_at_recovery
                        if phase == "decode"
                        else 0
                    ),
                    "pd_ready_recovery_before_submit": float(
                        ready_recovery_before_submit
                    ),
                    "pd_decode_ready_recovery_before_submit": float(
                        decode_ready_recovery_before_submit
                    ),
                }
            )
        if (
            self.pd_nonblocking_phase_recovery_enabled
            and not selected_future_done
            and deferred_scheduler_output is None
        ):
            if collect_detailed_timing:
                selected_timing_record[
                    "pd_nonblocking_phase_recovery_yield"
                ] = 1.0
                selected_timing_record[
                    "pd_nonblocking_phase_recovery_yield_count"
                ] = (
                    selected_timing_record.get(
                        "pd_nonblocking_phase_recovery_yield_count", 0.0
                    )
                    + 1.0
                )
                selected_timing_record[
                    "pd_nonblocking_phase_recovery_no_ready_future"
                ] = float(ready_future_count == 0)
            self._write_pd_phase_queue_trace(
                "nonblocking_yield",
                phase_queues,
                {
                    "selected_phase": phase,
                    "selected_submit_seq": phase_recovery_record.get(
                        "pd_phase_queue_selected_submit_seq", -1.0
                    ),
                    "selected_future_done": float(selected_future_done),
                    "ready_future_count": float(ready_future_count),
                    "not_ready_future_count": float(not_ready_future_count),
                    "pd_nonblocking_phase_recovery_no_ready_future": float(
                        ready_future_count == 0
                    ),
                },
            )
            if self._pd_capacity_retry_suppressed_this_step:
                self._pd_capacity_retry_gate.note_wait()
            self._wait_for_pd_progress(
                phase_queues,
                selected_future=selected_future,
                selected_scheduler_output=selected_scheduler_output,
            )
            return None, model_executed
        future, scheduler_output, exec_model_fut = self._pop_pd_phase_queue_at(
            phase_queues, phase, pop_index
        )
        binding = self._kv_retention_future_binding(future, scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            result_wait_start_ts = time.monotonic()
            tiered_eligibility_error = None
            try:
                model_output = future.result()
                model_ready_ts = time.monotonic()
                if (
                    model_output is None
                    and not scheduler_output.ttft_layerwise_prefill_preemption
                ):
                    exec_model_fut.result()
                    raise RuntimeError("unexpected error")
                model_done_ts = time.monotonic()
            except TieredBackendEligibilityError as error:
                tiered_eligibility_error = error
                model_output = None
                model_ready_ts = model_done_ts = time.monotonic()

        if isinstance(model_output, TTFTLayerwisePrefillBoundary):
            if not scheduler_output.ttft_layerwise_prefill_preemption:
                raise RuntimeError(
                    "model returned a TTFT Prefill boundary for an output "
                    "that did not opt into continuation ownership"
                )
            if model_output.kv_retention_binding != binding:
                raise RuntimeError("Prefill boundary returned a different execution segment")
            suspended = self._suspend_ttft_prefill_boundary(
                model_output,
                scheduler_output,
            )
            if suspended and binding is not None:
                self._kv_retention_prefill_work_registry.suspend(
                    binding, model_output.next_layer)
                if getattr(model_output, "kv_retention_parked_ticket", None) is not None:
                    self._kv_retention_prefill_work_registry.park(binding)
            # Install the whole-batch owner before consuming client aborts.
            # Otherwise a partial abort can free one row in the gap between
            # queue pop and suspended-slot publication.
            self._process_aborts_queue()
            self._write_pd_phase_queue_trace(
                "ttft_prefill_boundary",
                phase_queues,
                {
                    "scheduler_output_id": id(scheduler_output),
                    "boundary_next_layer": float(model_output.next_layer),
                    "boundary_end_layer": float(model_output.end_layer),
                    "boundary_batch_width": float(
                        len(model_output.scheduled_tokens)
                    ),
                    "boundary_retained": float(suspended),
                    **self._kv_retention_binding_debug(binding),
                },
            )
            return None, True

        if (
            model_output is None
            and scheduler_output.ttft_layerwise_prefill_preemption
            and tiered_eligibility_error is None
        ):
            self._cancel_ttft_prefill_yield_for_output(scheduler_output)
            exec_model_fut.result()
            self._enqueue_ttft_prefill_final_sampling(
                phase_queues,
                scheduler_output,
                exec_model_fut,
            )
            self._write_pd_phase_queue_trace(
                "ttft_prefill_final_execute",
                phase_queues,
                {"scheduler_output_id": id(scheduler_output)},
            )
            return None, True

        if tiered_eligibility_error is not None:
            self._process_aborts_queue()
            self._discard_ttft_prefill_deferred_aborts(scheduler_output)
            engine_core_outputs = self._fail_tiered_backend_scheduler_output(
                scheduler_output,
                tiered_eligibility_error,
            )
            if binding is not None:
                self._kv_retention_prefill_work_registry.finish(binding)
            self._pd_ttft_handoff_clear_from_outputs(
                phase_queues, engine_core_outputs
            )
            output_process_end_ts = time.monotonic()
            timings = self._iteration_debug_timings.pop(
                id(scheduler_output), {}
            )
            timings["tiered_backend_request_local_failure"] = 1.0
            schedule_start_ts = timings.get(
                "schedule_start_ts", model_ready_ts
            )
            schedule_end_ts = timings.get("schedule_end_ts", model_ready_ts)
            execute_submit_ts = timings.get(
                "execute_submit_ts", schedule_end_ts
            )
            if deferred_scheduler_output:
                assert deferred_exec_future is not None
                self._enqueue_deferred_pd_phase_sampling(
                    phase_queues,
                    deferred_scheduler_output,
                    deferred_exec_future,
                    schedule_start_ts=timings.get(
                        "schedule_start_ts", model_ready_ts
                    ),
                    schedule_end_ts=timings.get(
                        "schedule_end_ts", model_ready_ts
                    ),
                )
            # The popped future no longer owns its request ids.  Queue-derived
            # pending-id state is rebuilt on the next submit; clear transient
            # scheduler guards now so the surviving lane cannot inherit a
            # ghost dependency from the failed phase.
            self._set_scheduler_pd_pending_request_ids(set())
            self._set_scheduler_pd_pending_phase_request_ids({})
            self._write_pd_phase_queue_trace(
                "recover_request_local_failure",
                phase_queues,
                {
                    "recovered_phase": phase,
                    "failed_scheduler_output_id": id(scheduler_output),
                    "failed_request_count": float(
                        len(scheduler_output.num_scheduled_tokens)
                    ),
                },
            )
            self._write_iteration_debug_jsonl(
                scheduler_output,
                schedule_start_ts,
                schedule_end_ts,
                execute_submit_ts,
                result_wait_start_ts,
                model_ready_ts,
                model_done_ts,
                output_process_end_ts,
                timings,
                {},
            )
            return engine_core_outputs, (
                model_executed
                or scheduler_output.total_num_scheduled_tokens > 0
            )

        assert model_output is not None

        self._process_aborts_queue()
        if scheduler_output.ttft_layerwise_prefill_preemption:
            self._flush_ttft_prefill_deferred_aborts(scheduler_output)
        retry_free_blocks_before = self._pd_capacity_retry_free_blocks()
        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        if scheduler_output.kv_retention_layerwise_transfer is not None:
            self.scheduler.complete_layerwise_kv_offload(scheduler_output)
        if binding is not None:
            self._kv_retention_prefill_work_registry.finish(binding)
        self._notify_pd_capacity_retry_after_output(
            scheduler_output,
            engine_core_outputs,
            retry_free_blocks_before,
        )
        output_process_end_ts = time.monotonic()
        timings = self._iteration_debug_timings.pop(id(scheduler_output), {})
        schedule_start_ts = timings.get("schedule_start_ts", model_ready_ts)
        schedule_end_ts = timings.get("schedule_end_ts", model_ready_ts)
        execute_submit_ts = timings.get("execute_submit_ts", schedule_end_ts)
        self._pd_ttft_handoff_clear_from_outputs(
            phase_queues, engine_core_outputs
        )
        self._observe_pd_sm_plan(scheduler_output, model_output, timings)
        if getattr(self, "_tbt_capture", None) is not None:
            self._tbt_capture.recover(scheduler_output, engine_core_outputs,
                                      phase_queues)
        if deferred_scheduler_output:
            assert deferred_exec_future is not None
            self._enqueue_deferred_pd_phase_sampling(
                phase_queues,
                deferred_scheduler_output,
                deferred_exec_future,
                schedule_start_ts=timings.get(
                    "schedule_start_ts", model_ready_ts
                ),
                schedule_end_ts=timings.get("schedule_end_ts", model_ready_ts),
            )

        # Never hold a recovered Decode output behind the next admission or
        # parked-Prefill resume. The caller publishes it to IPC; the next step
        # performs maintenance and normal Decode-first admission. Also skip
        # resubmit when an abort consumed the output after control was deferred.
        output_first_return = output_first and phase == "decode"
        timings["isolation_decode_control_deferred"] = float(defer_control_for_decode)
        timings["isolation_decode_output_first_return"] = float(output_first_return)
        if (
            self.pd_dual_continuous_batching_enabled
            and self.pd_separate_scheduler_lanes_enabled
            and deferred_scheduler_output is None
            and self.scheduler.has_requests()
            and not output_first_return
        ):
            (
                post_recover_model_executed,
                post_recover_deferred_output,
                post_recover_deferred_exec_future,
                post_recover_submitted,
            ) = self._submit_pd_scheduler_lanes(
                phase_queues,
                ("decode", "prefill"),
                allow_multiple_rounds=False,
            )
            model_executed = model_executed or post_recover_model_executed
            if post_recover_deferred_output is not None:
                assert post_recover_deferred_exec_future is not None
                post_recover_deferred_ts = time.monotonic()
                self._enqueue_deferred_pd_phase_sampling(
                    phase_queues,
                    post_recover_deferred_output,
                    post_recover_deferred_exec_future,
                    schedule_start_ts=post_recover_deferred_ts,
                    schedule_end_ts=post_recover_deferred_ts,
                )
                self._iteration_debug_timings[
                    id(post_recover_deferred_output)
                ]["pd_dual_continuous_post_recover_deferred"] = 1.0
                if phase == "decode":
                    self._iteration_debug_timings[
                        id(post_recover_deferred_output)
                    ]["pd_dual_continuous_post_decode_deferred"] = 1.0
            if post_recover_submitted:
                selected_timing_record[
                    "pd_dual_continuous_post_recover_resubmit"
                ] = 1.0
                if phase == "decode":
                    selected_timing_record[
                        "pd_dual_continuous_post_decode_resubmit"
                    ] = 1.0
                    selected_timing_record[
                        "pd_decoupled_decode_post_recover_resubmit"
                    ] = float(self.pd_decoupled_paired_recovery_enabled)

        self._write_pd_phase_queue_trace(
            "recover",
            phase_queues,
            {
                "recovered_phase": phase,
                "recovered_scheduler_output_id": id(scheduler_output),
                "recovered_total_tokens": float(
                    scheduler_output.total_num_scheduled_tokens
                ),
                "ready_future_count_before_pop": float(ready_future_count),
                "not_ready_future_count_before_pop": float(
                    not_ready_future_count
                ),
                "future_wait_duration_s": float(
                    model_ready_ts - result_wait_start_ts
                ),
                "prefill_pending_at_decode_recovery": float(
                    prefill_pending_at_recovery if phase == "decode" else 0
                ),
                "prefill_not_ready_at_decode_recovery": float(
                    prefill_not_ready_at_recovery if phase == "decode" else 0
                ),
            },
        )

        self._write_iteration_debug_jsonl(
            scheduler_output,
            schedule_start_ts,
            schedule_end_ts,
            execute_submit_ts,
            result_wait_start_ts,
            model_ready_ts,
            model_done_ts,
            output_process_end_ts,
            timings,
            getattr(model_output, "model_runner_debug_timings", {}),
        )

        return engine_core_outputs, model_executed

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        batch_queue = self.batch_queue
        assert batch_queue is not None

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if self.scheduler.has_requests():
            schedule_start_ts = time.monotonic()
            scheduler_output = self.scheduler.schedule()
            schedule_end_ts = time.monotonic()
            execute_submit_ts = time.monotonic()
            scheduler_output_phase = self._scheduler_output_phase(scheduler_output)
            exec_future = self.model_executor.execute_model(
                scheduler_output,
                non_block=True,
                pd_green_context_phase=scheduler_output_phase,
            )
            timing_record = {
                "schedule_start_ts": schedule_start_ts,
                "schedule_end_ts": schedule_end_ts,
                "execute_submit_ts": execute_submit_ts,
                "submitted_scheduler_output_phase": scheduler_output_phase,
                "outstanding_model_futures_before": float(len(batch_queue)),
                "engine_batch_queue_depth_before_submit": float(len(batch_queue)),
            }
            self._iteration_debug_timings[id(scheduler_output)] = timing_record
            if not self.is_ec_producer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output,
                        non_block=True,
                        pd_green_context_phase=scheduler_output_phase,
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                future.add_done_callback(
                    lambda _future, output_id=id(scheduler_output): (
                        self._iteration_debug_timings.get(output_id, {}).setdefault(
                            "actual_model_future_done_ts", time.monotonic()
                        )
                    )
                )
                batch_queue.appendleft((future, scheduler_output, exec_future))
                timing_record["outstanding_model_futures_after"] = float(
                    len(batch_queue)
                )
                timing_record["engine_batch_queue_depth_after_submit"] = float(
                    len(batch_queue)
                )
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, True

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        result_queue_depth_before_pop = len(batch_queue)
        ready_future_count = sum(1 for queued_future, _, _ in batch_queue
                                 if queued_future.done())
        not_ready_future_count = result_queue_depth_before_pop - ready_future_count
        oldest_future_done = batch_queue[-1][0].done()
        pop_index, phase_recovery_record = (
            self._select_phase_aware_recovery_index(batch_queue)
        )
        selected_future_done = batch_queue[pop_index][0].done()
        selected_timing_record = self._iteration_debug_timings.get(
            id(batch_queue[pop_index][1]), {}
        )
        phase_queue_counts = self._batch_queue_phase_counts(batch_queue)
        fifo_poll_order_index = len(batch_queue) - 1 - pop_index
        selected_timing_record.update(
            {
                **phase_queue_counts,
                **phase_recovery_record,
                "scheduler_output_phase": self._scheduler_output_phase(
                    batch_queue[pop_index][1]
                ),
                "result_queue_depth_before_pop": float(
                    result_queue_depth_before_pop
                ),
                "engine_batch_queue_depth_before_pop": float(
                    result_queue_depth_before_pop
                ),
                "ready_future_count": float(ready_future_count),
                "not_ready_future_count": float(not_ready_future_count),
                "future_done_before_wait_start": float(selected_future_done),
                "future_done_at_poll": float(selected_future_done),
                "future_poll_order_index": float(fifo_poll_order_index),
                "blocked_by_earlier_future_count": float(
                    ready_future_count if not oldest_future_done else 0
                ),
            }
        )
        future, scheduler_output, exec_model_fut = self._pop_batch_queue_at(
            batch_queue, pop_index
        )
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            result_wait_start_ts = time.monotonic()
            tiered_eligibility_error = None
            try:
                model_output = future.result()
                model_ready_ts = time.monotonic()
                if model_output is None:
                    # None from sample_tokens() implies that the original
                    # execute_model() call failed - raise that exception.
                    exec_model_fut.result()
                    raise RuntimeError("unexpected error")
                model_done_ts = time.monotonic()
            except TieredBackendEligibilityError as error:
                tiered_eligibility_error = error
                model_output = None
                model_ready_ts = model_done_ts = time.monotonic()

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()
        timings = self._iteration_debug_timings.pop(id(scheduler_output), {})
        schedule_start_ts = timings.get("schedule_start_ts", model_ready_ts)
        schedule_end_ts = timings.get("schedule_end_ts", model_ready_ts)
        execute_submit_ts = timings.get("execute_submit_ts", schedule_end_ts)
        if tiered_eligibility_error is not None:
            engine_core_outputs = self._fail_tiered_backend_scheduler_output(
                scheduler_output,
                tiered_eligibility_error,
            )
            timings["tiered_backend_request_local_failure"] = 1.0
            model_runner_debug_timings = {}
        else:
            assert model_output is not None
            engine_core_outputs = self.scheduler.update_from_output(
                scheduler_output, model_output
            )
            model_runner_debug_timings = getattr(
                model_output, "model_runner_debug_timings", {}
            )
        output_process_end_ts = time.monotonic()
        self._write_iteration_debug_jsonl(
            scheduler_output,
            schedule_start_ts,
            schedule_end_ts,
            execute_submit_ts,
            result_wait_start_ts,
            model_ready_ts,
            model_done_ts,
            output_process_end_ts,
            timings,
            model_runner_debug_timings,
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # If we are doing speculative decoding with structured output,
            # we need to get the draft token ids from the prior step before
            # we can compute the grammar bitmask for the deferred request.
            if self.use_spec_decode:
                draft_token_ids = self.model_executor.take_draft_token_ids()
                assert draft_token_ids is not None
                # Update the draft token ids in the scheduler output to
                # filter out the invalid spec tokens, which will be padded
                # with -1 and skipped by the grammar bitmask computation.
                self.scheduler.update_draft_token_ids_in_output(
                    draft_token_ids, deferred_scheduler_output
                )
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            self._iteration_debug_timings[id(deferred_scheduler_output)] = {
                "schedule_start_ts": timings.get(
                    "schedule_start_ts", model_ready_ts
                ),
                "schedule_end_ts": timings.get("schedule_end_ts", model_ready_ts),
                "execute_submit_ts": time.monotonic(),
                "outstanding_model_futures_before": float(len(batch_queue)),
                "engine_batch_queue_depth_before_submit": float(len(batch_queue)),
            }
            deferred_phase = self._scheduler_output_phase(deferred_scheduler_output)
            future = self.model_executor.sample_tokens(
                grammar_output,
                non_block=True,
                pd_green_context_phase=deferred_phase,
            )
            future.add_done_callback(
                lambda _future, output_id=id(deferred_scheduler_output): (
                    self._iteration_debug_timings.get(output_id, {}).setdefault(
                        "actual_model_future_done_ts", time.monotonic()
                    )
                )
            )
            batch_queue.appendleft((future, deferred_scheduler_output, exec_future))
            self._iteration_debug_timings[id(deferred_scheduler_output)][
                "outstanding_model_futures_after"
            ] = float(len(batch_queue))
            self._iteration_debug_timings[id(deferred_scheduler_output)][
                "engine_batch_queue_depth_after_submit"
            ] = float(len(batch_queue))

        return engine_core_outputs, model_executed

    def _process_aborts_queue(self):
        if not self.aborts_queue.empty():
            request_ids = []
            while not self.aborts_queue.empty():
                ids = self.aborts_queue.get_nowait()
                # Should be a list here, but also handle string just in case.
                request_ids.extend((ids,) if isinstance(ids, str) else ids)
            # More efficient to abort all as a single batch.
            self.abort_requests(request_ids)

    def shutdown(self):
        if getattr(self, "_tbt_capture", None) is not None:
            self._tbt_capture.close()
            self._tbt_capture = None
        if self._pd_engine_progress_registered:
            unregister_pd_engine_progress_event(
                self._pd_engine_progress_event
            )
            self._pd_engine_progress_registered = False
        if self._debug_trace_writer is not None:
            self._debug_trace_writer.close()
            self._debug_trace_writer = None
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # The cache either exists in EngineCore or WorkerWrapperBase
        if self.mm_receiver_cache is not None:
            self.mm_receiver_cache.clear_cache()

        self.model_executor.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.scheduler.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def sleep(self, level: int = 1):
        self.model_executor.sleep(level)

    def wake_up(self, tags: list[str] | None = None):
        self.model_executor.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        *,
        engine_index: int = 0,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()

        def executor_fail_callback() -> None:
            self.input_queue.put_nowait(
                (EngineCoreRequestType.EXECUTOR_FAILED, b"")
            )
            progress_event = getattr(
                self,
                "_pd_engine_progress_event",
                None,
            )
            if progress_event is not None:
                self._notify_pd_engine_progress()

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False

        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            vllm_config,
            client_handshake_address,
        ) as addresses:
            self.client_count = len(addresses.outputs)

            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            internal_dp_balancing = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes.
            self.publish_dp_lb_stats = internal_dp_balancing

            self._init_data_parallel(vllm_config)

            super().__init__(
                vllm_config,
                executor_class,
                log_stats,
                executor_fail_callback,
                internal_dp_balancing,
            )

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            ready_event = threading.Event()
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # Don't complete handshake until DP coordinator ready message is
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        if client_handshake_address is None:
            with handshake as addresses:
                yield addresses
        else:
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: ParallelConfig | None = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            yield addresses

            # Send ready message.
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            # We pass back the coordinator stats update address here for the
            # external LB case for our colocated front-end to use (coordinator
            # only runs with rank 0).
            dp_stats_address = self.frontend_stats_publish_address

            # Include config hash for DP configuration validation
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
                "num_gpu_blocks": num_gpu_blocks,
                "dp_stats_address": dp_stats_address,
            }
            if vllm_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = (
                    vllm_config.parallel_config.compute_hash()
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: EngineCoreProc | None = None
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config: ParallelConfig = vllm_config.parallel_config
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
            if data_parallel:
                parallel_config.data_parallel_rank_local = local_dp_rank
                set_process_title("EngineCore", f"DP{dp_rank}")
            else:
                set_process_title("EngineCore")
            decorate_logs()

            parallel_config.data_parallel_index = dp_rank
            if data_parallel and vllm_config.model_config.is_moe:
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                # Non-MoE DP ranks are completely independent, so treat like DP=1.
                # Note that parallel_config.data_parallel_index will still reflect
                # the original DP rank.
                parallel_config.data_parallel_size = 1
                parallel_config.data_parallel_size_local = 1
                parallel_config.data_parallel_rank = 0
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while (
            not self.engines_running
            and not self.scheduler.has_requests()
            and not self.batch_queue
        ):
            if self.input_queue.empty():
                # Drain aborts queue; all aborts are also processed via input_queue.
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
                if logger.isEnabledFor(DEBUG):
                    logger.debug("EngineCore waiting for work.")
                    waited = True
            req = self.input_queue.get()
            self._handle_client_request(*req)

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)

        # If no model execution happened but requests remain (for example,
        # WAITING_FOR_REMOTE_KVS), block the experimental P/D loop on its
        # progress event. Other Engine modes retain the upstream brief GIL
        # yield for background producers such as NIXL.
        if (
            not outputs
            and not model_executed
            and self.scheduler.has_unfinished_requests()
        ):
            if (
                self.pd_nonblocking_phase_recovery_enabled
                and self.pd_phase_queues is not None
            ):
                if not self._pd_progress_waited_this_step:
                    self._wait_for_pd_progress(self.pd_phase_queues)
            else:
                time.sleep(0.001)

        return model_executed

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
                result = method(*self._convert_msgspec_args(method, args))
                output.result = UtilityResult(result)
            except BaseException as e:
                logger.exception("Invocation of %s method failed", method_name)
                output.failure_message = (
                    f"Call to {method_name} method failed: {str(e)}"
                )
            self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=output))
            )
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                # Send subscription message to coordinator.
                coord_socket.send(b"\x01")

            # Register sockets with poller.
            poller = zmq.Poller()
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(b"")
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # Wait for ready message from coordinator.
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    request: Any
                    if request_type == EngineCoreRequestType.ADD:
                        req: EngineCoreRequest = add_request_decoder.decode(data_frames)
                        try:
                            request = self.preprocess_add_request(req)
                        except Exception:
                            self._handle_request_preproc_error(req)
                            continue
                    else:
                        request = generic_decoder.decode(data_frames)

                        if request_type == EngineCoreRequestType.ABORT:
                            # Aborts are added to *both* queues, allows us to eagerly
                            # process aborts while also ensuring ordering in the input
                            # queue to avoid leaking requests. This is ok because
                            # aborting in the scheduler is idempotent.
                            self.aborts_queue.put_nowait(request)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))
                    self._notify_pd_engine_progress()

    def process_output_sockets(
        self,
        output_paths: list[str],
        coord_output_path: str | None,
        engine_index: int,
    ):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)

    def _handle_request_preproc_error(self, request: EngineCoreRequest) -> None:
        """Log and return a request-scoped error response for exceptions raised
        from the add request preprocessing in the input socket processing thread.
        """
        logger.exception(
            "Unexpected error pre-processing request %s", request.request_id
        )
        self.output_queue.put_nowait(
            (
                request.client_index,
                EngineCoreOutputs(
                    engine_index=self.engine_index,
                    finished_requests={request.request_id},
                    outputs=[
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=[],
                            finish_reason=FinishReason.ERROR,
                        )
                    ],
                ),
            )
        )


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
    ):
        assert vllm_config.model_config.is_moe, (
            "DPEngineCoreProc should only be used for MoE models"
        )

        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.step_counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            engine_index=dp_rank,
        )

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # Configure GPUs and stateless process group for data parallel.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        if vllm_config.kv_transfer_config is not None:
            # modify the engine_id and append the local_dp_rank to it to ensure
            # that the kv_transfer_config is unique for each DP rank.
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            )
            logger.debug(
                "Setting kv_transfer_config.engine_id to %s",
                vllm_config.kv_transfer_config.engine_id,
            )

        self.dp_rank = dp_rank
        self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def add_request(self, request: Request, request_wave: int = 0):
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif not self.engines_running:
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

        super().add_request(request, request_wave)

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug("EngineCore starting idle loop for wave %d.", new_wave)
                    self.engines_running = True
        else:
            super()._handle_client_request(request_type, request)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, step_counter=self.step_counter, current_wave=self.current_wave
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            # 2) Step the engine core.
            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Optimization - only perform finish-sync all-reduce every 32 steps.
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        stateless_destroy_torch_distributed_process_group(self.dp_group)
        self.shutdown()

        parallel_config = self.vllm_config.parallel_config
        old_dp_size = parallel_config.data_parallel_size
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if reconfig_request.new_data_parallel_rank != -1:
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        # local rank specifies device visibility, it should not be changed
        assert (
            reconfig_request.new_data_parallel_rank_local
            == ReconfigureRankType.KEEP_CURRENT_RANK
        )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        if reconfig_request.new_data_parallel_rank != -2:
            self.dp_rank = parallel_config.data_parallel_rank
            self.dp_group = parallel_config.stateless_init_dp_group()
        reconfig_request.new_data_parallel_master_port = (
            parallel_config.data_parallel_master_port
        )

        self.model_executor.reinitialize_distributed(reconfig_request)
        if reconfig_request.new_data_parallel_size > old_dp_size:
            assert self.available_gpu_memory_for_kv_cache > 0
            # pass available_gpu_memory_for_kv_cache from existing
            # engine-cores to new engine-cores so they can directly
            # use it in _initialize_kv_caches() rather than profiling.
            ParallelConfig.sync_kv_cache_memory_size(
                self.dp_group, self.available_gpu_memory_for_kv_cache
            )
            # NOTE(yongji): newly joined workers require dummy_run even
            # CUDA graph is not used
            self.model_executor.collective_rpc("compile_or_warm_up_model")
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()
            logger.info("DPEngineCoreProc %s shutdown", self.dp_rank)
        else:
            logger.info(
                "Distributed environment reinitialized for DP rank %s", self.dp_rank
            )


class EngineCoreActorMixin:
    """
    Ray actor for running EngineCore in a data parallel context
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        addresses: EngineZmqAddresses,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_index = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_visible_devices(vllm_config, local_dp_rank)

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        from vllm.platforms import current_platform

        if current_platform.is_xpu():
            pass
        else:
            device_control_env_var = current_platform.device_control_env_var
            self._set_cuda_visible_devices(
                vllm_config, local_dp_rank, device_control_env_var
            )

    def _set_cuda_visible_devices(
        self, vllm_config: VllmConfig, local_dp_rank: int, device_control_env_var: str
    ):
        world_size = vllm_config.parallel_config.world_size
        # Set CUDA_VISIBLE_DEVICES or equivalent.
        try:
            value = get_device_indices(
                device_control_env_var, local_dp_rank, world_size
            )
            os.environ[device_control_env_var] = value
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()  # type: ignore[attr-defined]
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()  # type: ignore[attr-defined]


class DPMoEEngineCoreActor(EngineCoreActorMixin, DPEngineCoreProc):
    """Used for MoE model data parallel cases."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        vllm_config.parallel_config.data_parallel_rank = dp_rank

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        DPEngineCoreProc.__init__(
            self, vllm_config, local_client, "", executor_class, log_stats
        )


class EngineCoreActor(EngineCoreActorMixin, EngineCoreProc):
    """Used for non-MoE and/or non-DP cases."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.data_parallel_size_local = 1
        vllm_config.parallel_config.data_parallel_rank = 0

        EngineCoreActorMixin.__init__(
            self, vllm_config, addresses, dp_rank, local_dp_rank
        )
        EngineCoreProc.__init__(
            self,
            vllm_config,
            local_client,
            "",
            executor_class,
            log_stats,
            engine_index=dp_rank,
        )
