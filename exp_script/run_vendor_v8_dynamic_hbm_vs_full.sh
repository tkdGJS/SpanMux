#!/usr/bin/env bash
set -uo pipefail

# Four-way comparison on the same workload:
#   A. native 40-SM vLLM: mixed batching + upstream chunked prefill
#   B. full vendor_v8: dynamic SM + SRPT + Context Ring
#   C. Dynamic-HBM capacity-idle split: FCFS + HBM-only; normal solo work uses
#      all 40 SMs, but a Prefill lane blocked specifically by HBM capacity
#      retains the TBT-SLO split and leaves its SM share idle.
#   D. Dynamic-HBM capacity-full-SM: the same FCFS/HBM-only path, but when HBM
#      capacity serializes the lanes, the runnable Decode uses all 40 SMs.
#   Optional Baseline+. Native 40-SM mixed batching and upstream chunked
#      Prefill, with only vendor_v8's Prefill-SRPT and Decode-FCFS request
#      selectors enabled. P/D multiplexing, Dynamic SM, Context Ring, victim
#      handling, and layerwise handoff remain disabled.
#   Optional dynamic_hbm_srpt. Same HBM-only/full-SM capacity policy as D,
#      but enable Prefill SRPT with native token budgets; no Ring or layerwise
#      request preemption. This is independent of the FCFS control's policy.
#   Optional dynamic_hbm_srpt_layerwise. The same HBM SRPT policy with bounded
#      P1 whole-batch layer preemption; KV stays in HBM, no offload or Ring.
#
# By default every case uses vLLM FULL_AND_PIECEWISE CUDA Graph mode. Native
# uses upstream graphs; vendor cases add strict GreenContext Decode namespaces.
# BENCHMARK_CUDAGRAPH=0 disables graphs. In the matched preset it preserves
# compilation eligibility; non-matched cases retain legacy eager gating. The optional
# layerwise profile handoff is common to all three vendor cases; Native always
# keeps it disabled.
#
# WORKLOAD selects one common arrival pattern for every comparison case:
#   A. submit all Long requests at the origin, then inject five Shorts;
#   B. submit all five Shorts at the origin, then inject Longs every five min;
#   C. interleave S1,L1,S2,L2,S3,L3,S4,S5 at a fixed interval.
#   LOONGSERVE. replay one immutable LoongServe Mixed JSONL trace open-loop.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Diagnostic ablation: Native width2 versus one request in each P/D batch.
ISOLATION_BATCH2_PD11_COMPARISON="${ISOLATION_BATCH2_PD11_COMPARISON:-0}"
case "${ISOLATION_BATCH2_PD11_COMPARISON}" in
  0|1) ;;
  *) echo "[ERROR] ISOLATION_BATCH2_PD11_COMPARISON must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "${ISOLATION_BATCH2_PD11_COMPARISON}" == "1" ]]; then
  for batch_setting in \
    ISOLATION_GUARDED_2K_COMPARISON=1 \
    VLLM_MAX_NUM_SEQS=2 \
    VLLM_ISOLATION_PREFILL_BATCH_MAX_REQUESTS=1 \
    VLLM_ISOLATION_DECODE_BATCH_MAX_REQUESTS=1; do
    batch_name="${batch_setting%%=*}"
    batch_value="${batch_setting#*=}"
    if [[ -v "${batch_name}" && "${!batch_name}" != "${batch_value}" ]]; then
      echo "[ERROR] batch2/PD11 comparison requires ${batch_setting}; conflicting ${!batch_name}" >&2
      exit 2
    fi
    export "${batch_setting}"
  done
fi
for phase_name in PREFILL DECODE; do
  batch_flag="VLLM_ISOLATION_${phase_name}_BATCH_MAX_REQUESTS"
  batch_value="${!batch_flag:-0}"
  if [[ ! "${batch_value}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] ${batch_flag} must be a non-negative integer" >&2
    exit 2
  fi
done

# Opt-in five-case comparison. 2K is PER REQUEST, not the aggregate batch
# budget: up to eight 2K chunks still fit a16K Prefill batch.
ISOLATION_GUARDED_2K_COMPARISON="${ISOLATION_GUARDED_2K_COMPARISON:-0}"
case "${ISOLATION_GUARDED_2K_COMPARISON}" in
  0|1) ;;
  *) echo "[ERROR] ISOLATION_GUARDED_2K_COMPARISON must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "${ISOLATION_GUARDED_2K_COMPARISON}" == "1" ]]; then
  for guarded_setting in \
    ISOLATION_MATCHED_COMPARISON=1 \
    PREFILL_CHUNK_TOKENS=2048 \
    VLLM_LONG_PREFILL_TOKEN_THRESHOLD=2048 \
    VLLM_PD_EVAL_NATIVE_PREFILL_MAX_TOKENS=2048 \
    VLLM_PD_EVAL_ACTIVE_DECODE_NATIVE_PREFILL_TOKENS=2048 \
    VLLM_PD_EVAL_ACTIVE_DECODE_RING_PREFILL_TOKENS=2048 \
    VLLM_PD_EVAL_RING_BW_MAX_TOKENS=2048 \
    VLLM_PD_EVAL_ACTIVE_DECODE_ELASTIC_PREFILL=0 \
    VLLM_ISOLATION_RING_QUERY_EXPANSION=0 \
    VLLM_ISOLATION_P1_RING_SM_HANDOFF=1 \
    VLLM_ISOLATION_DRAM_REFRESH_PROBE=1 \
    VLLM_KV_RETENTION_P1_PARKED_PREFILL=1; do
    guarded_name="${guarded_setting%%=*}"
    guarded_value="${guarded_setting#*=}"
    if [[ -v "${guarded_name}" && "${!guarded_name}" != "${guarded_value}" ]]; then
      echo "[ERROR] guarded2K comparison requires ${guarded_setting}; conflicting ${!guarded_name}" >&2
      exit 2
    fi
    export "${guarded_setting}"
  done
fi

# Reversible, single-GPU/non-speculative matched-comparison preset. Resolve it
# before per-case defaults so inherited P2/Stage-2 settings cannot silently
# turn this into a different experiment. Native receives no vendor execution
# features; the per-case routing below still clears those flags.
ISOLATION_MATCHED_COMPARISON="${ISOLATION_MATCHED_COMPARISON:-0}"
ISOLATION_MEMORY_OPTIMIZATIONS="${VLLM_ISOLATION_MEMORY_OPTIMIZATIONS:-0}"
# Common startup controls, not vendor-only execution optimizations. Native and
# P/D receive identical values; neither switch inherits the memory bundle.
for startup_feature in STARTUP_CACHE_TRIM STARTUP_MEMORY_AUDIT; do
  startup_flag="VLLM_ISOLATION_${startup_feature}"
  startup_value="${!startup_flag:-0}"
  case "${startup_value}" in
    0|1) ;;
    *) echo "[ERROR] ${startup_flag} must be 0 or 1" >&2; exit 2 ;;
  esac
  export "${startup_flag}=${startup_value}"
done
# New scratch ownership modes never inherit an older optimization bundle.
VLLM_ISOLATION_PREFILL_WORKSPACE_RIGHTSIZE="${VLLM_ISOLATION_PREFILL_WORKSPACE_RIGHTSIZE:-0}"
VLLM_ISOLATION_RING_WORKSPACE_SHARING="${VLLM_ISOLATION_RING_WORKSPACE_SHARING:-0}"
VLLM_ISOLATION_GREEN_CONTEXT_PROCESS_LIFETIME="${VLLM_ISOLATION_GREEN_CONTEXT_PROCESS_LIFETIME:-1}"
ISOLATION_MEMORY_FEATURES=(
  FUSED_RMS_NORM PREFILL_EXTENDED_WORKSPACE RING_STATE_REUSE
  RING_DYNAMIC_KERNEL COMPACT_DECODE_BUFFERS RING_EARLY_PREFETCH
  GRAPH_MEMORY_AUDIT PREFILL_WORKSPACE_RIGHTSIZE RING_WORKSPACE_SHARING
  GREEN_CONTEXT_PROCESS_LIFETIME
)
for memory_feature in MEMORY_OPTIMIZATIONS "${ISOLATION_MEMORY_FEATURES[@]}"; do
  memory_flag="VLLM_ISOLATION_${memory_feature}"
  memory_value="${!memory_flag:-${ISOLATION_MEMORY_OPTIMIZATIONS}}"
  case "${memory_value}" in
    0|1) ;;
    *) echo "[ERROR] ${memory_flag} must be 0 or 1" >&2; exit 2 ;;
  esac
done
ISOLATION_EXECUTION_OPTIMIZATIONS="${VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS:-0}"
# This correctness experiment is independent of the existing optimization bundle.
VLLM_ISOLATION_WAITING_DECODE_RECLAIM="${VLLM_ISOLATION_WAITING_DECODE_RECLAIM:-0}"
ISOLATION_EXECUTION_FEATURES=(
  RING_QUERY_EXPANSION P1_BOUNDARY_FASTPATH PREFILL_FUSED_ROPE
  TEXT_INPUT_BUFFERS MEMORY_SNAPSHOTS P1_RING_SM_HANDOFF DRAM_REFRESH_PROBE
  D2H_EXACT_PLAN_PROGRESS
  NATIVE_SOFT_HEADROOM_PROGRESS
  PREFILL_BATCH_KV_BUDGET
  LOGICAL_RELEASE_READINESS
  BOUNDED_PREFILL_PLANNING
  WAITING_DECODE_RECLAIM
)
for execution_feature in EXECUTION_OPTIMIZATIONS "${ISOLATION_EXECUTION_FEATURES[@]}"; do
  execution_flag="VLLM_ISOLATION_${execution_feature}"
  execution_value="${!execution_flag:-${ISOLATION_EXECUTION_OPTIMIZATIONS}}"
  case "${execution_value}" in
    0|1) ;;
    *) echo "[ERROR] ${execution_flag} must be 0 or 1" >&2; exit 2 ;;
  esac
done
KV_RETENTION_P1_PROPOSAL="${VLLM_KV_RETENTION_P1_PARKED_PREFILL:-0}"
case "${KV_RETENTION_P1_PROPOSAL}" in
  0|1) ;;
  *) echo "[ERROR] VLLM_KV_RETENTION_P1_PARKED_PREFILL must be 0 or 1" >&2; exit 2 ;;
esac
case "${ISOLATION_MATCHED_COMPARISON}" in
  0|1) ;;
  *) echo "[ERROR] ISOLATION_MATCHED_COMPARISON must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "${ISOLATION_MATCHED_COMPARISON}" == "1" ]]; then
  # Graph OFF is a matched execution mode, not a request for global eager.
  BENCHMARK_CUDAGRAPH="${BENCHMARK_CUDAGRAPH:-1}"
  case "${BENCHMARK_CUDAGRAPH}" in
    1) MATCHED_CUDAGRAPH_MODE=FULL_AND_PIECEWISE ;;
    0) MATCHED_CUDAGRAPH_MODE=NONE ;;
    *) echo "[ERROR] BENCHMARK_CUDAGRAPH must be 0 or 1" >&2; exit 2 ;;
  esac
  for matched_setting in \
    VLLM_PD_EVAL_PHASE_DEPTH_ENABLED=1 \
    VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH=1 \
    VLLM_PD_EVAL_DECODE_INFLIGHT_DEPTH=1 \
    VLLM_PD_EVAL_GLOBAL_INFLIGHT_DEPTH=2 \
    VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF=1 \
    VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION=0 \
    VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION=0 \
    VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN=0 \
    VLLM_KV_RETENTION_LAYERWISE_OFFLOAD=0 \
    VLLM_PD_PREFILL_CHAIN_MAX_LOOKAHEAD_TOKENS=0 \
    VLLM_PD_PREFILL_WORKSPACE_ENABLED=1 \
    VLLM_PD_PREFILL_WORKSPACE_SLOTS=1 \
    VLLM_PD_PREFILL_WORKSPACE_STRICT=1 \
    VLLM_PD_EVAL_PHASE_SYNC_SAMPLE_OUTPUT=0 \
    VLLM_PD_EVAL_FUSED_EXECUTE_SAMPLE=0 \
    VLLM_PD_EVAL_GREEN_DECODE_CUDAGRAPH_STRICT=1 \
    VLLM_PD_EVAL_STRICT_BOUNDED_LIGHTWEIGHT_TIMELINE=1 \
    VLLM_PD_EVAL_VIRTUAL_CONTEXT_TIMING=0 \
    ENABLE_LIGHTWEIGHT_TIMELINE_TRACE=1 \
    ENABLE_FULL_ITERATION_TRACE=0 \
    ENABLE_PD_EXECUTOR_DEBUG=0 \
    ENABLE_PD_CUDA_EVENT_TRACE=0 \
    ENABLE_PD_NVTX_PHASE_TRACE=0 \
    DYNAMIC_HBM_PREFILL_SCHEDULING_POLICY=fcfs \
    FULL_VENDOR_V8_PREFILL_SCHEDULING_POLICY=srpt \
    VLLM_SCHEDULING_POLICY=fcfs \
    DYNAMIC_HBM_CONTENTION_MODE=guard_wait \
    BENCHMARK_CUDAGRAPH="${BENCHMARK_CUDAGRAPH}" \
    BENCHMARK_CUDAGRAPH_MODE="${MATCHED_CUDAGRAPH_MODE}"; do
    matched_name="${matched_setting%%=*}"
    matched_value="${matched_setting#*=}"
    if [[ -v "${matched_name}" && "${!matched_name}" != "${matched_value}" ]]; then
      echo "[ERROR] ISOLATION_MATCHED_COMPARISON requires ${matched_setting}; remove or change the conflicting setting (${!matched_name})" >&2
      exit 2
    fi
    export "${matched_setting}"
  done
  # Common sampling is configurable, unlike the forbidden detailed probes.
  export ITERATION_TRACE_SAMPLE_EVERY_N="${ITERATION_TRACE_SAMPLE_EVERY_N:-100}"
  export VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS="${VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS:-${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}}"
fi

BENCHMARK_MODE="${BENCHMARK_MODE:-smoke}"
DEFAULT_EXPERIMENT_NAME="vendor_v8_dynamic_hbm_vs_full_${BENCHMARK_MODE}_v1"
if [[ "${ISOLATION_GUARDED_2K_COMPARISON}" == "1" ]]; then
  DEFAULT_EXPERIMENT_NAME="vendor_v8_guarded_2k_${BENCHMARK_MODE}_v1"
fi
if [[ "${ISOLATION_BATCH2_PD11_COMPARISON}" == "1" ]]; then
  DEFAULT_EXPERIMENT_NAME="vendor_v8_guarded_2k_batch2_pd11_${BENCHMARK_MODE}_v1"
fi
BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-${DEFAULT_EXPERIMENT_NAME}}"
VLLM_VENDOR_DIR="${VLLM_VENDOR_DIR:-${ROOT_DIR}/vendor_v8}"
# Empty means that the vLLM executable imports its installed, unmodified
# package instead of prepending a repository fork to PYTHONPATH.
NATIVE_VLLM_VENDOR_DIR="${NATIVE_VLLM_VENDOR_DIR-}"
PD_DYNAMIC_SM_POLICY="${PD_DYNAMIC_SM_POLICY:-tbt_slo_feedback}"
VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF="${VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF:-0}"
VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION="${VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION:-0}"
VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION="${VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION:-0}"
VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS="${VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS:-2}"
VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN="${VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN:-0}"
VLLM_KV_RETENTION_LAYERWISE_OFFLOAD="${VLLM_KV_RETENTION_LAYERWISE_OFFLOAD:-0}"
VLLM_PD_PREFILL_WORKSPACE_ENABLED="${VLLM_PD_PREFILL_WORKSPACE_ENABLED:-0}"
VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS="${VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS:-16384}"
VLLM_PD_PREFILL_WORKSPACE_SLOTS="${VLLM_PD_PREFILL_WORKSPACE_SLOTS:-1}"
VLLM_PD_PREFILL_WORKSPACE_STRICT="${VLLM_PD_PREFILL_WORKSPACE_STRICT:-1}"
FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS="${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS:-${ISOLATION_MATCHED_COMPARISON}}"
DYNAMIC_HBM_PREFILL_SCHEDULING_POLICY="${DYNAMIC_HBM_PREFILL_SCHEDULING_POLICY:-fcfs}"
FULL_VENDOR_V8_PREFILL_SCHEDULING_POLICY="${FULL_VENDOR_V8_PREFILL_SCHEDULING_POLICY:-srpt}"
BASELINE_PLUS_PREFILL_QUANTUM_TOKENS="${BASELINE_PLUS_PREFILL_QUANTUM_TOKENS:-0}"
# Common request ceiling; token/KV budgets determine the actual admitted batch.
# Explicit10 reproduces archived runs; the batch2 preset above still selects2.
# Matched capture sizing includes the exact ceiling (including100).
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-100}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
VLLM_LONG_PREFILL_TOKEN_THRESHOLD="${VLLM_LONG_PREFILL_TOKEN_THRESHOLD:-0}"
MATCHED_NATIVE_PREFILL_CAP="${VLLM_MAX_NUM_BATCHED_TOKENS}"
if [[ "${ISOLATION_GUARDED_2K_COMPARISON}" == "1" ]]; then
  MATCHED_NATIVE_PREFILL_CAP=2048
fi
if [[ "${ISOLATION_MATCHED_COMPARISON}" == "1" ]]; then
  for matched_native_cap_name in \
    VLLM_PD_EVAL_NATIVE_PREFILL_MAX_TOKENS \
    VLLM_PD_EVAL_ACTIVE_DECODE_NATIVE_PREFILL_TOKENS; do
    if [[ -v "${matched_native_cap_name}" \
          && "${!matched_native_cap_name}" != "${MATCHED_NATIVE_PREFILL_CAP}" ]]; then
      echo "[ERROR] ISOLATION_MATCHED_COMPARISON requires ${matched_native_cap_name}=${MATCHED_NATIVE_PREFILL_CAP}; omit the conflicting Native cap" >&2
      exit 2
    fi
  done
fi
FULL_VENDOR_V8_RESIDENCY_VICTIM_POLICY="${FULL_VENDOR_V8_RESIDENCY_VICTIM_POLICY:-scheduled_prefill_fit}"
FULL_VENDOR_V8_SCHEDULED_PREFILL_FIT_CAPACITY_FRACTION="${FULL_VENDOR_V8_SCHEDULED_PREFILL_FIT_CAPACITY_FRACTION:-0.98}"
DYNAMIC_HBM_CONTENTION_MODE="${DYNAMIC_HBM_CONTENTION_MODE:-guard_wait}"
DEFAULT_BENCHMARK_VARIANTS="native_40sm_chunked full_vendor_v8 dynamic_hbm_serialized_fixed_split dynamic_hbm_serialized_fullsm"
if [[ "${ISOLATION_MATCHED_COMPARISON}" == "1" ]]; then
  DEFAULT_BENCHMARK_VARIANTS="native_40sm_chunked dynamic_hbm_serialized_fullsm full_vendor_v8"
fi
if [[ "${ISOLATION_GUARDED_2K_COMPARISON}" == "1" ]]; then
  DEFAULT_BENCHMARK_VARIANTS="native_40sm_chunked baseline_plus dynamic_hbm_serialized_fullsm dynamic_hbm_srpt full_vendor_v8"
fi
BENCHMARK_VARIANTS="${BENCHMARK_VARIANTS:-${DEFAULT_BENCHMARK_VARIANTS}}"
BENCHMARK_CUDAGRAPH="${BENCHMARK_CUDAGRAPH:-1}"
BENCHMARK_CUDAGRAPH_MODE="${BENCHMARK_CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
WORKLOAD="${WORKLOAD:-A}"
LOONGSERVE_TRACE_JSONL="${LOONGSERVE_TRACE_JSONL:-}"
LOONGSERVE_ARRIVAL_TIME_SCALE="${LOONGSERVE_ARRIVAL_TIME_SCALE:-1.0}"
LOONGSERVE_TERMINATION_MODE="${LOONGSERVE_TERMINATION_MODE:-time}"
LOONGSERVE_REQUEST_LIMIT="${LOONGSERVE_REQUEST_LIMIT:-0}"
LOONGSERVE_TIME_LIMIT_SEC="${LOONGSERVE_TIME_LIMIT_SEC:-0}"
LOONGSERVE_TIME_LIMIT_MODE="${LOONGSERVE_TIME_LIMIT_MODE:-drain}"
PORT="${PORT:-8000}"
SERVER_READY_TIMEOUT_SEC="${SERVER_READY_TIMEOUT_SEC:-900}"
SERVER_SHUTDOWN_TIMEOUT_SEC="${SERVER_SHUTDOWN_TIMEOUT_SEC:-30}"
PORT_RELEASE_TIMEOUT_SEC="${PORT_RELEASE_TIMEOUT_SEC:-30}"
CASE_RESTART_COOLDOWN_SEC="${CASE_RESTART_COOLDOWN_SEC:-5}"
INSTANCE_CLEANUP_FAILURE_EXIT_CODE=86
KILL_EXISTING_PORT_OWNER="${KILL_EXISTING_PORT_OWNER:-0}"
DRY_RUN=0

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      echo "Usage: BENCHMARK_MODE=smoke|full WORKLOAD=A|B|C|LOONGSERVE ISOLATION_MATCHED_COMPARISON=0|1 ISOLATION_GUARDED_2K_COMPARISON=0|1 ISOLATION_BATCH2_PD11_COMPARISON=0|1 $0 [--dry-run]"
      exit 0
      ;;
    *)
      echo "[ERROR] unknown option: $1" >&2
      exit 2
      ;;
  esac
done

case "${BENCHMARK_CUDAGRAPH}" in
  0|1) ;;
  *)
    echo "[ERROR] BENCHMARK_CUDAGRAPH must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF}" in
  0|1) ;;
  *)
    echo "[ERROR] VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION}" in
  0|1) ;;
  *)
    echo "[ERROR] VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION}" in
  0|1) ;;
  *)
    echo "[ERROR] VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION must be 0 or 1" >&2
    exit 2
    ;;
esac
if [[ "${VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION}" == "1" \
      && "${VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION}" != "1" ]]; then
  echo "[ERROR] TTFT layerwise Prefill preemption requires its execution flag" >&2
  exit 2
fi
if [[ ! "${VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS must be positive" >&2
  exit 2
fi
for flag_name in VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN VLLM_KV_RETENTION_LAYERWISE_OFFLOAD; do
  case "${!flag_name}" in
    0|1) ;;
    *) echo "[ERROR] ${flag_name} must be 0 or 1" >&2; exit 2 ;;
  esac
done
case "${VLLM_PD_PREFILL_WORKSPACE_ENABLED}" in
  0|1) ;;
  *)
    echo "[ERROR] VLLM_PD_PREFILL_WORKSPACE_ENABLED must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${VLLM_PD_PREFILL_WORKSPACE_STRICT}" in
  0|1) ;;
  *)
    echo "[ERROR] VLLM_PD_PREFILL_WORKSPACE_STRICT must be 0 or 1" >&2
    exit 2
    ;;
esac
if [[ ! "${VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS must be positive" >&2
  exit 2
fi
if [[ ! "${VLLM_PD_PREFILL_WORKSPACE_SLOTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] VLLM_PD_PREFILL_WORKSPACE_SLOTS must be positive" >&2
  exit 2
fi
for positive_timeout_name in \
  SERVER_READY_TIMEOUT_SEC \
  SERVER_SHUTDOWN_TIMEOUT_SEC \
  PORT_RELEASE_TIMEOUT_SEC; do
  positive_timeout_value="${!positive_timeout_name}"
  if [[ ! "${positive_timeout_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] ${positive_timeout_name} must be a positive integer" >&2
    exit 2
  fi
done
if [[ ! "${CASE_RESTART_COOLDOWN_SEC}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] CASE_RESTART_COOLDOWN_SEC must be non-negative" >&2
  exit 2
fi
case "${KILL_EXISTING_PORT_OWNER}" in
  0|1) ;;
  *)
    echo "[ERROR] KILL_EXISTING_PORT_OWNER must be 0 or 1" >&2
    exit 2
    ;;
esac
for unsupported_partial_prefill_name in \
  VLLM_MAX_NUM_PARTIAL_PREFILLS \
  VLLM_MAX_LONG_PARTIAL_PREFILLS; do
  unsupported_partial_prefill_value="${!unsupported_partial_prefill_name-}"
  if [[ -n "${unsupported_partial_prefill_value}" \
    && "${unsupported_partial_prefill_value}" != "1" ]]; then
    echo "[ERROR] ${unsupported_partial_prefill_name}=${unsupported_partial_prefill_value} is unsupported by vLLM V1; omit it and use max-num-seqs/max-num-batched-tokens for native budget-fill admission" >&2
    exit 2
  fi
done
if [[ ! "${VLLM_LONG_PREFILL_TOKEN_THRESHOLD}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] VLLM_LONG_PREFILL_TOKEN_THRESHOLD must be a non-negative integer" >&2
  exit 2
fi
WORKLOAD="${WORKLOAD^^}"
case "${WORKLOAD}" in
  A|B|C|LOONGSERVE) ;;
  *)
    echo "[ERROR] WORKLOAD must be A, B, C, or LOONGSERVE" >&2
    exit 2
    ;;
esac
if [[ "${WORKLOAD}" == "LOONGSERVE" ]]; then
  case "${LOONGSERVE_TERMINATION_MODE}" in
    time|requests) ;;
    *)
      echo "[ERROR] LOONGSERVE_TERMINATION_MODE must be time or requests" >&2
      exit 2
      ;;
  esac
  case "${LOONGSERVE_TIME_LIMIT_MODE}" in
    drain|cancel) ;;
    *)
      echo "[ERROR] LOONGSERVE_TIME_LIMIT_MODE must be drain or cancel" >&2
      exit 2
      ;;
  esac
  if [[ ! "${LOONGSERVE_REQUEST_LIMIT}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] LOONGSERVE_REQUEST_LIMIT must be a non-negative integer" >&2
    exit 2
  fi
  if ! [[ "${LOONGSERVE_TIME_LIMIT_SEC}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "[ERROR] LOONGSERVE_TIME_LIMIT_SEC must be non-negative" >&2
    exit 2
  fi
  if [[ "${LOONGSERVE_TERMINATION_MODE}" == "requests" ]]; then
    if [[ ! "${LOONGSERVE_REQUEST_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
      echo "[ERROR] LOONGSERVE_REQUEST_LIMIT must be positive in requests mode" >&2
      exit 2
    fi
    if [[ ! "${LOONGSERVE_TIME_LIMIT_SEC}" =~ ^0*([.]0*)?$ ]]; then
      echo "[ERROR] LOONGSERVE_TIME_LIMIT_SEC must be zero in requests mode" >&2
      exit 2
    fi
    if [[ "${LOONGSERVE_TIME_LIMIT_MODE}" != "drain" ]]; then
      echo "[ERROR] LOONGSERVE_TIME_LIMIT_MODE must be drain in requests mode" >&2
      exit 2
    fi
  elif [[ ! "${LOONGSERVE_REQUEST_LIMIT}" =~ ^0+$ ]]; then
    echo "[ERROR] LOONGSERVE_REQUEST_LIMIT must be zero in time mode" >&2
    exit 2
  fi
  if [[ "${DRY_RUN}" != "1" && ! -f "${LOONGSERVE_TRACE_JSONL}" ]]; then
    echo "[ERROR] missing LOONGSERVE_TRACE_JSONL=${LOONGSERVE_TRACE_JSONL}" >&2
    exit 2
  fi
fi
if [[ "${BENCHMARK_CUDAGRAPH}" == "1" ]]; then
  case "${BENCHMARK_CUDAGRAPH_MODE}" in
    FULL|FULL_DECODE_ONLY|FULL_AND_PIECEWISE) ;;
    *)
      echo "[ERROR] BENCHMARK_CUDAGRAPH_MODE must provide full Decode graphs: FULL, FULL_DECODE_ONLY, or FULL_AND_PIECEWISE" >&2
      exit 2
      ;;
  esac
else
  BENCHMARK_CUDAGRAPH_MODE=NONE
fi

LONG_CONTEXT_TOKENS="${LONG_CONTEXT_TOKENS:-100000}"
SHORT_CONTEXT_TOKENS="${SHORT_CONTEXT_TOKENS:-10000}"
LONG_REQUEST_COUNT="${LONG_REQUEST_COUNT:-3}"
SHORT_REQUEST_COUNT="${SHORT_REQUEST_COUNT:-5}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-4294967296}"

case "${BENCHMARK_MODE}" in
  smoke)
    LONG_OUTPUT_TOKENS="${LONG_OUTPUT_TOKENS:-64}"
    SHORT_OUTPUT_TOKENS="${SHORT_OUTPUT_TOKENS:-32}"
    SHORT_INJECTION_OFFSETS_SEC="${SHORT_INJECTION_OFFSETS_SEC:-2 4 6 8 10}"
    LONG_INJECTION_OFFSETS_SEC="${LONG_INJECTION_OFFSETS_SEC:-2 4 6}"
    INTERLEAVED_INJECTION_INTERVAL_SEC="${INTERLEAVED_INJECTION_INTERVAL_SEC:-2}"
    REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-7200}"
    GENERATE_ABCD="${GENERATE_ABCD:-0}"
    WARMUP_COUNT="${WARMUP_COUNT:-0}"
    ;;
  full)
    # Keep at least one long Decode alive through the final 25-minute short
    # injection on the measured ~100ms 40-SM 100K TBT baseline.
    LONG_OUTPUT_TOKENS="${LONG_OUTPUT_TOKENS:-16384}"
    SHORT_OUTPUT_TOKENS="${SHORT_OUTPUT_TOKENS:-512}"
    SHORT_INJECTION_OFFSETS_SEC="${SHORT_INJECTION_OFFSETS_SEC:-300 600 900 1200 1500}"
    LONG_INJECTION_OFFSETS_SEC="${LONG_INJECTION_OFFSETS_SEC:-300 600 900}"
    INTERLEAVED_INJECTION_INTERVAL_SEC="${INTERLEAVED_INJECTION_INTERVAL_SEC:-300}"
    REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-28800}"
    # The graph generator reads the recorded per-iteration SM profile; the
    # decode/prefill boundary is therefore dynamic rather than a static label.
    GENERATE_ABCD="${GENERATE_ABCD:-1}"
    WARMUP_COUNT="${WARMUP_COUNT:-1}"
    ;;
  *)
    echo "[ERROR] BENCHMARK_MODE must be smoke or full" >&2
    exit 2
    ;;
esac

echo "[COMPARE] mode=${BENCHMARK_MODE}"
echo "[COMPARE] isolation_matched_comparison=${ISOLATION_MATCHED_COMPARISON}"
echo "[COMPARE] workload=${WORKLOAD}"
echo "[COMPARE] long=${LONG_REQUEST_COUNT}x${LONG_CONTEXT_TOKENS} output=${LONG_OUTPUT_TOKENS}"
echo "[COMPARE] short=${SHORT_REQUEST_COUNT}x${SHORT_CONTEXT_TOKENS} output=${SHORT_OUTPUT_TOKENS}"
echo "[COMPARE] workload_A_short_offsets=${SHORT_INJECTION_OFFSETS_SEC}"
echo "[COMPARE] workload_B_long_offsets=${LONG_INJECTION_OFFSETS_SEC}"
echo "[COMPARE] workload_C_event_interval=${INTERLEAVED_INJECTION_INTERVAL_SEC}"
echo "[COMPARE] A=native_40sm_chunked B=full_vendor_v8 C=dynamic_hbm_serialized_fixed_split D=dynamic_hbm_serialized_fullsm optional=baseline_plus,dynamic_hbm_srpt,dynamic_hbm_srpt_layerwise"
echo "[COMPARE] selected_variants=${BENCHMARK_VARIANTS}"
echo "[COMPARE] cudagraph=${BENCHMARK_CUDAGRAPH} mode=${BENCHMARK_CUDAGRAPH_MODE}"
echo "[COMPARE] dynamic_sm_policy=${PD_DYNAMIC_SM_POLICY}"
echo "[COMPARE] layerwise_profile_handoff=${VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF}"
echo "[COMPARE] ttft_layerwise_prefill_execution=${VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION} preemption=${VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION} group_layers=${VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS}"
echo "[COMPARE] prefill_workspace=${VLLM_PD_PREFILL_WORKSPACE_ENABLED} max_tokens=${VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS} slots=${VLLM_PD_PREFILL_WORKSPACE_SLOTS} strict=${VLLM_PD_PREFILL_WORKSPACE_STRICT}"
echo "[COMPARE] full_vendor_v8_throughput_optimizations=${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS}"
echo "[COMPARE] common_prefill_budgets max_num_seqs=${VLLM_MAX_NUM_SEQS} max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS} long_threshold=${VLLM_LONG_PREFILL_TOKEN_THRESHOLD}"
echo "[COMPARE] A_prefill_scheduling_policy=native_fcfs_chunked"
echo "[COMPARE] B_prefill_scheduling_policy=${FULL_VENDOR_V8_PREFILL_SCHEDULING_POLICY}"
echo "[COMPARE] baseline_plus=prefill_srpt decode_fcfs native_mixed_40sm quantum_tokens=${BASELINE_PLUS_PREFILL_QUANTUM_TOKENS}"
echo "[COMPARE] dynamic_hbm_srpt=prefill_srpt decode_fcfs hbm_only capacity_serialized_full_sm=1 quantum_tokens=0"
echo "[COMPARE] B_residency_victim_policy=${FULL_VENDOR_V8_RESIDENCY_VICTIM_POLICY}"
echo "[COMPARE] B_scheduled_prefill_fit_capacity_fraction=${FULL_VENDOR_V8_SCHEDULED_PREFILL_FIT_CAPACITY_FRACTION}"
echo "[COMPARE] C_prefill_scheduling_policy=${DYNAMIC_HBM_PREFILL_SCHEDULING_POLICY} capacity_serialized_full_sm=0 normal_solo_full_sm=1"
echo "[COMPARE] D_prefill_scheduling_policy=${DYNAMIC_HBM_PREFILL_SCHEDULING_POLICY} capacity_serialized_full_sm=1 normal_solo_full_sm=1"
echo "[COMPARE] dynamic_hbm_contention_mode=${DYNAMIC_HBM_CONTENTION_MODE}"
echo "[COMPARE] instance_lifecycle=separate_process_group_per_case"
echo "[COMPARE] shutdown_timeout=${SERVER_SHUTDOWN_TIMEOUT_SEC}s port_release_timeout=${PORT_RELEASE_TIMEOUT_SEC}s restart_cooldown=${CASE_RESTART_COOLDOWN_SEC}s"
if [[ "${WORKLOAD}" == "LOONGSERVE" ]]; then
  echo "[COMPARE] loongserve_trace=${LOONGSERVE_TRACE_JSONL}"
  echo "[COMPARE] loongserve_arrival_time_scale=${LOONGSERVE_ARRIVAL_TIME_SCALE}"
  echo "[COMPARE] loongserve_termination_mode=${LOONGSERVE_TERMINATION_MODE}"\
" request_limit=${LOONGSERVE_REQUEST_LIMIT}"
  echo "[COMPARE] loongserve_time_limit_sec=${LOONGSERVE_TIME_LIMIT_SEC} mode=${LOONGSERVE_TIME_LIMIT_MODE}"
fi

workload_experiment_suffix=""
if [[ "${WORKLOAD}" != "A" ]]; then
  workload_experiment_suffix="_workload_${WORKLOAD,,}"
fi

RUNNER_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  RUNNER_ARGS+=(--dry-run)
fi

overall_status=0
read -r -a benchmark_variants <<< "${BENCHMARK_VARIANTS}"
case_index=0
case_count="${#benchmark_variants[@]}"

wait_for_sweep_port_release() {
  command -v lsof >/dev/null 2>&1 || {
    echo "[ERROR] missing lifecycle command: lsof" >&2
    return 1
  }
  local deadline=$((SECONDS + PORT_RELEASE_TIMEOUT_SEC))
  local pids=""
  while ((SECONDS < deadline)); do
    pids="$(
      lsof -t -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null \
        | tr '\n' ' ' || true
    )"
    [[ -z "${pids// /}" ]] && return 0
    sleep 1
  done
  echo "[ERROR] sweep port ${PORT} remained occupied by ${pids:-unknown}" >&2
  return 1
}

run_case_with_log() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    "$@"
    return "$?"
  fi
  # The child intentionally creates its directories only after deployment
  # probes. Keep this log outside those directories so an early Native import
  # failure is visible without making the child's overwrite guard fail.
  local case_log="${ROOT_DIR}/${experiment_name}_launcher.log"
  if ! printf '\n[CASE] begin variant=%s time=%s\n' \
      "${variant}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${case_log}"; then
    echo "[ERROR] cannot open case launcher log: ${case_log}" >&2
    return 2
  fi
  echo "[CASE] launcher_log=${case_log}"
  # Continue draining on a log/output error, allowing the child's normal owned
  # instance cleanup to finish. Never replace status 86 with tee's exit code.
  "$@" 2>&1 | tee --output-error=warn -a -- "${case_log}"
  local pipeline_status=("${PIPESTATUS[@]}")
  local child_status="${pipeline_status[0]}"
  local log_status="${pipeline_status[1]}"
  printf '[CASE] exit_status=%s log_status=%s\n' \
    "${child_status}" "${log_status}" >> "${case_log}" || log_status=1
  if ((child_status != 0)); then
    return "${child_status}"
  fi
  if ((log_status != 0)); then
    echo "[ERROR] case launcher log failed: ${case_log}" >&2
    return 2
  fi
}

for variant in "${benchmark_variants[@]}"; do
  experiment_name="${BASE_EXPERIMENT_NAME}${workload_experiment_suffix}_${variant}"
  case_system_variant="${variant}"
  native_prefill_capacity_mode="guard_wait"
  layerwise_profile_handoff=0
  ttft_layerwise_prefill_execution=0
  ttft_layerwise_prefill_preemption=0
  kv_layerwise_checkpoint_plan=0
  kv_layerwise_offload=0
  prefill_workspace_enabled="${VLLM_PD_PREFILL_WORKSPACE_ENABLED}"
  serialized_full_sm=1
  green_decode_cudagraph="${BENCHMARK_CUDAGRAPH}"
  # The optimization bundle is an implementation property of full_vendor_v8,
  # not a common workload condition. Keep every control free of bundle members
  # even when a matched sweep requests the bundle for the full-vendor case.
  case_throughput_optimizations=0
  case_p1_parked=0
  case_p1_hbm_only=0
  case_work_binding=0
  case_resume_frame=0
  case "${variant}" in
    dynamic_hbm_serialized_fixed_split)
      case_vendor_dir="${VLLM_VENDOR_DIR}"
      build_green_bridge="${BUILD_GREEN_BRIDGE:-auto}"
      native_prefill_capacity_mode="${DYNAMIC_HBM_CONTENTION_MODE}"
      prefill_scheduling_policy="${DYNAMIC_HBM_PREFILL_SCHEDULING_POLICY}"
      layerwise_profile_handoff="${VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF}"
      ttft_layerwise_prefill_execution="${VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION}"
      ttft_layerwise_prefill_preemption=0
      serialized_full_sm=0
      ;;
    dynamic_hbm_serialized_fullsm|dynamic_pd_hbm)
      # dynamic_pd_hbm is retained as a backward-compatible alias for the
      # current work-conserving Dynamic-HBM behavior.
      case_vendor_dir="${VLLM_VENDOR_DIR}"
      build_green_bridge="${BUILD_GREEN_BRIDGE:-auto}"
      native_prefill_capacity_mode="${DYNAMIC_HBM_CONTENTION_MODE}"
      prefill_scheduling_policy="${DYNAMIC_HBM_PREFILL_SCHEDULING_POLICY}"
      layerwise_profile_handoff="${VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF}"
      ttft_layerwise_prefill_execution="${VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION}"
      ttft_layerwise_prefill_preemption=0
      serialized_full_sm=1
      ;;
    dynamic_hbm_srpt|dynamic_hbm_srpt_layerwise)
      case_vendor_dir="${VLLM_VENDOR_DIR}"
      build_green_bridge="${BUILD_GREEN_BRIDGE:-auto}"
      native_prefill_capacity_mode="${DYNAMIC_HBM_CONTENTION_MODE}"
      prefill_scheduling_policy=srpt
      layerwise_profile_handoff="${VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF}"
      ttft_layerwise_prefill_execution="${VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION}"
      # Execution groups/SM handoff do not grant request-preemption authority.
      ttft_layerwise_prefill_preemption=0
      serialized_full_sm=1
      if [[ "${variant}" == "dynamic_hbm_srpt_layerwise" ]]; then
        # Keep the HBM-only policy environment; the distinct result name and
        # explicit mode describe slot lending without granting KV release.
        case_system_variant=dynamic_hbm_srpt
        case_p1_hbm_only=1
        case_p1_parked=1
        case_work_binding=1
        case_resume_frame=1
        ttft_layerwise_prefill_execution=1
        ttft_layerwise_prefill_preemption=1
        kv_layerwise_checkpoint_plan=1
      fi
      ;;
    full_vendor_v8)
      case_vendor_dir="${VLLM_VENDOR_DIR}"
      build_green_bridge="${BUILD_GREEN_BRIDGE:-auto}"
      prefill_scheduling_policy="${FULL_VENDOR_V8_PREFILL_SCHEDULING_POLICY}"
      layerwise_profile_handoff="${VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF}"
      ttft_layerwise_prefill_execution="${VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION}"
      ttft_layerwise_prefill_preemption="${VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION}"
      kv_layerwise_checkpoint_plan="${VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN}"
      kv_layerwise_offload="${VLLM_KV_RETENTION_LAYERWISE_OFFLOAD}"
      serialized_full_sm=1
      case_throughput_optimizations="${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS}"
      case_work_binding="${VLLM_KV_RETENTION_PREFILL_WORK_BINDING:-0}"
      case_resume_frame="${VLLM_KV_RETENTION_PREFILL_RESUME_FRAME:-0}"
      if [[ "${KV_RETENTION_P1_PROPOSAL}" == "1" ]]; then
        case_p1_parked=1
        case_work_binding=1
        case_resume_frame=1
        ttft_layerwise_prefill_execution=1
        ttft_layerwise_prefill_preemption=1
        kv_layerwise_checkpoint_plan=1
        kv_layerwise_offload=1
      fi
      ;;
    native_40sm_chunked)
      case_vendor_dir="${NATIVE_VLLM_VENDOR_DIR}"
      build_green_bridge=0
      prefill_scheduling_policy=fcfs
      serialized_full_sm=1
      # Native uses the same base vLLM CUDA Graph mode, not the vendor-only
      # GreenContext namespace/capture path.
      green_decode_cudagraph=0
      prefill_workspace_enabled=0
      ;;
    baseline_plus)
      # Import vendor_v8 only for its default-off request schedulers. The
      # validation runner enforces a native 40-SM mixed-iteration runtime and
      # rejects every P/D, GreenContext, Context Ring, victim, and layerwise
      # execution path for this variant.
      case_vendor_dir="${VLLM_VENDOR_DIR}"
      build_green_bridge=0
      prefill_scheduling_policy=srpt
      native_prefill_capacity_mode=legacy_preempt_recompute
      layerwise_profile_handoff=0
      serialized_full_sm=1
      green_decode_cudagraph=0
      prefill_workspace_enabled=0
      ;;
    *)
      overall_status=1
      echo "[CASE] unknown BENCHMARK_VARIANTS entry: ${variant}" >&2
      continue
      ;;
  esac
  echo "[CASE] begin variant=${variant} experiment=${experiment_name} serialized_full_sm=${serialized_full_sm} layerwise_profile_handoff=${layerwise_profile_handoff} ttft_layerwise_prefill_execution=${ttft_layerwise_prefill_execution} ttft_layerwise_prefill_preemption=${ttft_layerwise_prefill_preemption} cudagraph_mode=${BENCHMARK_CUDAGRAPH_MODE} green_decode_cudagraph=${green_decode_cudagraph} throughput_optimizations=${case_throughput_optimizations}"
  isolation_memory_env=()
  for memory_feature in MEMORY_OPTIMIZATIONS "${ISOLATION_MEMORY_FEATURES[@]}"; do
    memory_flag="VLLM_ISOLATION_${memory_feature}"
    memory_value="${!memory_flag:-${ISOLATION_MEMORY_OPTIMIZATIONS}}"
    case "${variant}" in
      native_40sm_chunked|baseline_plus) memory_value=0 ;;
    esac
    isolation_memory_env+=("${memory_flag}=${memory_value}")
  done
  for execution_feature in EXECUTION_OPTIMIZATIONS "${ISOLATION_EXECUTION_FEATURES[@]}"; do
    execution_flag="VLLM_ISOLATION_${execution_feature}"
    execution_value="${!execution_flag:-${ISOLATION_EXECUTION_OPTIMIZATIONS}}"
    case "${variant}" in
      native_40sm_chunked|baseline_plus) execution_value=0 ;;
      full_vendor_v8) ;;
      *)
        case "${execution_feature}" in
          RING_QUERY_EXPANSION|P1_BOUNDARY_FASTPATH|P1_RING_SM_HANDOFF|DRAM_REFRESH_PROBE|D2H_EXACT_PLAN_PROGRESS|NATIVE_SOFT_HEADROOM_PROGRESS|PREFILL_BATCH_KV_BUDGET|WAITING_DECODE_RECLAIM) execution_value=0 ;;
        esac ;;
    esac
    isolation_memory_env+=("${execution_flag}=${execution_value}")
  done
  case_status=0
  case_hbm_guard="${VLLM_ISOLATION_HBM_COMPLETION_GUARD:-0}"
  case_output_first="${VLLM_ISOLATION_DECODE_OUTPUT_FIRST:-0}"
  case_next_chunk_fit="${VLLM_ISOLATION_PREFILL_FIT_NEXT_CHUNK:-0}"
  case_prefill_batch_limit="${VLLM_ISOLATION_PREFILL_BATCH_MAX_REQUESTS:-0}"
  case_decode_batch_limit="${VLLM_ISOLATION_DECODE_BATCH_MAX_REQUESTS:-0}"
  if [[ "${ISOLATION_GUARDED_2K_COMPARISON}" == "1" ]]; then
    case_hbm_guard=1
    case_output_first=1
    case_next_chunk_fit=0
    if [[ "${variant}" == "full_vendor_v8" ]]; then
      case_hbm_guard=0
      case_next_chunk_fit=1
    fi
  fi
  case "${variant}" in
    native_40sm_chunked|baseline_plus)
      case_output_first=0
      case_prefill_batch_limit=0
      case_decode_batch_limit=0 ;;
  esac
  [[ "${variant}" == "full_vendor_v8" ]] || case_next_chunk_fit=0
  if run_case_with_log env \
    "${isolation_memory_env[@]}" \
    VLLM_ISOLATION_STARTUP_CACHE_TRIM="${VLLM_ISOLATION_STARTUP_CACHE_TRIM}" \
    VLLM_ISOLATION_STARTUP_MEMORY_AUDIT="${VLLM_ISOLATION_STARTUP_MEMORY_AUDIT}" \
    ISOLATION_BATCH2_PD11_COMPARISON="${ISOLATION_BATCH2_PD11_COMPARISON}" \
    VLLM_ISOLATION_PREFILL_BATCH_MAX_REQUESTS="${case_prefill_batch_limit}" \
    VLLM_ISOLATION_DECODE_BATCH_MAX_REQUESTS="${case_decode_batch_limit}" \
    ISOLATION_GUARDED_2K_COMPARISON="${ISOLATION_GUARDED_2K_COMPARISON}" \
    VLLM_ISOLATION_HBM_COMPLETION_GUARD="${case_hbm_guard}" \
    VLLM_ISOLATION_DECODE_OUTPUT_FIRST="${case_output_first}" \
    VLLM_ISOLATION_PREFILL_FIT_NEXT_CHUNK="${case_next_chunk_fit}" \
    ISOLATION_MATCHED_COMPARISON="${ISOLATION_MATCHED_COMPARISON}" \
    EXPERIMENT_NAME="${experiment_name}" \
    SYSTEM_VARIANT="${case_system_variant}" \
    WORKLOAD="${WORKLOAD}" \
    LOONGSERVE_TRACE_JSONL="${LOONGSERVE_TRACE_JSONL}" \
    LOONGSERVE_ARRIVAL_TIME_SCALE="${LOONGSERVE_ARRIVAL_TIME_SCALE}" \
    LOONGSERVE_TERMINATION_MODE="${LOONGSERVE_TERMINATION_MODE}" \
    LOONGSERVE_REQUEST_LIMIT="${LOONGSERVE_REQUEST_LIMIT}" \
    LOONGSERVE_TIME_LIMIT_SEC="${LOONGSERVE_TIME_LIMIT_SEC}" \
    LOONGSERVE_TIME_LIMIT_MODE="${LOONGSERVE_TIME_LIMIT_MODE}" \
    SUBMISSION_MODE=concurrent_longs \
    VLLM_VENDOR_DIR="${case_vendor_dir}" \
    BUILD_GREEN_BRIDGE="${build_green_bridge}" \
    PD_DYNAMIC_SM_POLICY="${PD_DYNAMIC_SM_POLICY}" \
    VLLM_PD_TBT_PREDICTIVE_LEAD_MS="${VLLM_PD_TBT_PREDICTIVE_LEAD_MS:-0}" \
    VLLM_PD_TBT_PREDICTIVE_CUTOVER_MS="${VLLM_PD_TBT_PREDICTIVE_CUTOVER_MS:--1}" \
    VLLM_PD_TBT_PREDICTIVE_MIN_STEP_MS="${VLLM_PD_TBT_PREDICTIVE_MIN_STEP_MS:-0}" \
    VLLM_PD_SERIALIZED_FULL_SM="${serialized_full_sm}" \
    VLLM_PD_EVAL_CUDAGRAPH_MODE="${BENCHMARK_CUDAGRAPH_MODE}" \
    VLLM_PD_EVAL_GREEN_DECODE_CUDAGRAPH="${green_decode_cudagraph}" \
    VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF="${layerwise_profile_handoff}" \
    VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION="${ttft_layerwise_prefill_execution}" \
    VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION="${ttft_layerwise_prefill_preemption}" \
    VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS="${VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS}" \
    VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN="${kv_layerwise_checkpoint_plan}" \
    VLLM_KV_RETENTION_LAYERWISE_OFFLOAD="${kv_layerwise_offload}" \
    VLLM_KV_RETENTION_P1_PARKED_PREFILL="${case_p1_parked}" \
    VLLM_KV_RETENTION_P1_HBM_ONLY="${case_p1_hbm_only}" \
    VLLM_KV_RETENTION_PREFILL_WORK_BINDING="${case_work_binding}" \
    VLLM_KV_RETENTION_PREFILL_RESUME_FRAME="${case_resume_frame}" \
    VLLM_PD_PREFILL_WORKSPACE_ENABLED="${prefill_workspace_enabled}" \
    VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS="${VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS}" \
    VLLM_PD_PREFILL_WORKSPACE_SLOTS="${VLLM_PD_PREFILL_WORKSPACE_SLOTS}" \
    VLLM_PD_PREFILL_WORKSPACE_STRICT="${VLLM_PD_PREFILL_WORKSPACE_STRICT}" \
    PREFILL_SCHEDULING_POLICY="${prefill_scheduling_policy}" \
    BASELINE_PLUS_PREFILL_QUANTUM_TOKENS="${BASELINE_PLUS_PREFILL_QUANTUM_TOKENS}" \
    VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" \
    VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS}" \
    VLLM_LONG_PREFILL_TOKEN_THRESHOLD="${VLLM_LONG_PREFILL_TOKEN_THRESHOLD}" \
    FULL_VENDOR_V8_RESIDENCY_VICTIM_POLICY="${FULL_VENDOR_V8_RESIDENCY_VICTIM_POLICY}" \
    FULL_VENDOR_V8_SCHEDULED_PREFILL_FIT_CAPACITY_FRACTION="${FULL_VENDOR_V8_SCHEDULED_PREFILL_FIT_CAPACITY_FRACTION}" \
    FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS="${case_throughput_optimizations}" \
    NATIVE_PREFILL_CAPACITY_MODE="${native_prefill_capacity_mode}" \
    LONG_CONTEXT_TOKENS="${LONG_CONTEXT_TOKENS}" \
    SHORT_CONTEXT_TOKENS="${SHORT_CONTEXT_TOKENS}" \
    LONG_REQUEST_COUNT="${LONG_REQUEST_COUNT}" \
    SHORT_REQUEST_COUNT="${SHORT_REQUEST_COUNT}" \
    SHORT_INJECTION_OFFSETS_SEC="${SHORT_INJECTION_OFFSETS_SEC}" \
    LONG_INJECTION_OFFSETS_SEC="${LONG_INJECTION_OFFSETS_SEC}" \
    INTERLEAVED_INJECTION_INTERVAL_SEC="${INTERLEAVED_INJECTION_INTERVAL_SEC}" \
    LONG_OUTPUT_TOKENS="${LONG_OUTPUT_TOKENS}" \
    SHORT_OUTPUT_TOKENS="${SHORT_OUTPUT_TOKENS}" \
    WARMUP_COUNT="${WARMUP_COUNT}" \
    KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES}" \
    REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC}" \
    PORT="${PORT}" \
    SERVER_READY_TIMEOUT_SEC="${SERVER_READY_TIMEOUT_SEC}" \
    SERVER_SHUTDOWN_TIMEOUT_SEC="${SERVER_SHUTDOWN_TIMEOUT_SEC}" \
    PORT_RELEASE_TIMEOUT_SEC="${PORT_RELEASE_TIMEOUT_SEC}" \
    INSTANCE_CLEANUP_FAILURE_EXIT_CODE="${INSTANCE_CLEANUP_FAILURE_EXIT_CODE}" \
    GENERATE_ABCD="${GENERATE_ABCD}" \
    ENABLE_LIGHTWEIGHT_TIMELINE_TRACE="${ENABLE_LIGHTWEIGHT_TIMELINE_TRACE:-1}" \
    ENABLE_PD_EXECUTOR_DEBUG="${ENABLE_PD_EXECUTOR_DEBUG:-0}" \
    ITERATION_TRACE_SAMPLE_EVERY_N="${ITERATION_TRACE_SAMPLE_EVERY_N:-1}" \
    ENABLE_PD_CUDA_EVENT_TRACE=0 \
    ENABLE_FULL_ITERATION_TRACE=0 \
    KILL_EXISTING_PORT_OWNER="${KILL_EXISTING_PORT_OWNER}" \
    PROFILE_RESULTS_DIR="${PROFILE_RESULTS_DIR:-${ROOT_DIR}/vendor_v8_each_partition_profile_v2_results}" \
    REQUEST_SLO_POLICY="${REQUEST_SLO_POLICY:-${ROOT_DIR}/vendor_v8_each_partition_profile_v2_results/slo/request_slo_policy.json}" \
    PREFILL_SERVICE_PROFILE="${PREFILL_SERVICE_PROFILE:-${ROOT_DIR}/vendor_v8_each_partition_profile_v2_results/runtime_profiles/prefill_service_profile.json}" \
    TBT_OVERLAP_PROFILE="${TBT_OVERLAP_PROFILE:-${ROOT_DIR}/vendor_v8_each_partition_profile_v2_results/runtime_profiles/tbt_overlap_profile.json}" \
    "${ROOT_DIR}/run_vendor_v8_short_validation.sh" \
      "${RUNNER_ARGS[@]}"; then
    echo "[CASE] completed variant=${variant}"
  else
    case_status="$?"
    overall_status=1
    echo "[CASE] failed variant=${variant} status=${case_status}" >&2
  fi

  if ((case_status == INSTANCE_CLEANUP_FAILURE_EXIT_CODE)); then
    echo "[SWEEP] aborting: prior vLLM instance was not safely released" >&2
    exit "${INSTANCE_CLEANUP_FAILURE_EXIT_CODE}"
  fi

  case_index=$((case_index + 1))
  if ((case_index < case_count)) && [[ "${DRY_RUN}" != "1" ]]; then
    if ! wait_for_sweep_port_release; then
      echo "[SWEEP] aborting before the next case" >&2
      exit "${INSTANCE_CLEANUP_FAILURE_EXIT_CODE}"
    fi
    if ((CASE_RESTART_COOLDOWN_SEC > 0)); then
      echo "[SWEEP] prior instance released; restarting next case after ${CASE_RESTART_COOLDOWN_SEC}s"
      sleep "${CASE_RESTART_COOLDOWN_SEC}"
    else
      echo "[SWEEP] prior instance released; restarting next case"
    fi
  fi
done

echo "[DONE] base_experiment=${BASE_EXPERIMENT_NAME} status=${overall_status}"
exit "${overall_status}"
