#!/usr/bin/env bash
set -euo pipefail

# Compact end-to-end validation for vendor_v8:
#   Long1 -> two short injections -> Long1 completion
#   Long2 -> two short injections -> Long2 completion
#   Long3 -> two short injections -> Long3 completion

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-vendor_v8_short_3long_6short}"
RESULT_DIR="${ROOT_DIR}/${EXPERIMENT_NAME}_results"
LOG_DIR="${ROOT_DIR}/${EXPERIMENT_NAME}_logs"
ARTIFACT_DIR="${ROOT_DIR}/${EXPERIMENT_NAME}_artifacts"

PROFILE_RESULTS_DIR="${PROFILE_RESULTS_DIR:-${ROOT_DIR}/vendor_v8_each_partition_profile_v1_results}"
REQUEST_SLO_POLICY="${REQUEST_SLO_POLICY:-${PROFILE_RESULTS_DIR}/slo/request_slo_policy.json}"
PREFILL_SERVICE_PROFILE="${PREFILL_SERVICE_PROFILE:-${PROFILE_RESULTS_DIR}/runtime_profiles/prefill_service_profile.json}"
TBT_OVERLAP_PROFILE="${TBT_OVERLAP_PROFILE:-${PROFILE_RESULTS_DIR}/runtime_profiles/tbt_overlap_profile.json}"

VENV_DIR="${VENV_DIR:-/home/noslab-gpu/tkdgjs/tkdgjs}"
VLLM_EXECUTABLE="${VLLM_EXECUTABLE:-${VENV_DIR}/bin/vllm}"
VLLM_PYTHON_EXECUTABLE="${VLLM_PYTHON_EXECUTABLE:-${VENV_DIR}/bin/python}"
if [[ ! -v VLLM_VENDOR_DIR ]]; then
  VLLM_VENDOR_DIR="${ROOT_DIR}/vendor_v8"
fi
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:${PORT}"

LONG_CONTEXT_TOKENS="${LONG_CONTEXT_TOKENS:-100000}"
SHORT_CONTEXT_TOKENS="${SHORT_CONTEXT_TOKENS:-10000}"
LONG_REQUEST_COUNT="${LONG_REQUEST_COUNT:-3}"
SHORT_REQUESTS_PER_LONG="${SHORT_REQUESTS_PER_LONG:-2}"
SHORT_INJECTION_OFFSETS_SEC="${SHORT_INJECTION_OFFSETS_SEC:-5 10}"
LONG_INJECTION_OFFSETS_SEC="${LONG_INJECTION_OFFSETS_SEC:-300 600 900}"
INTERLEAVED_INJECTION_INTERVAL_SEC="${INTERLEAVED_INJECTION_INTERVAL_SEC:-300}"
REQUESTED_SHORT_REQUEST_COUNT="${SHORT_REQUEST_COUNT-}"
WORKLOAD="${WORKLOAD:-legacy}"
LOONGSERVE_TRACE_JSONL="${LOONGSERVE_TRACE_JSONL:-}"
LOONGSERVE_ARRIVAL_TIME_SCALE="${LOONGSERVE_ARRIVAL_TIME_SCALE:-1.0}"
LOONGSERVE_TERMINATION_MODE="${LOONGSERVE_TERMINATION_MODE:-time}"
LOONGSERVE_REQUEST_LIMIT="${LOONGSERVE_REQUEST_LIMIT:-0}"
LOONGSERVE_TIME_LIMIT_SEC="${LOONGSERVE_TIME_LIMIT_SEC:-0}"
LOONGSERVE_TIME_LIMIT_MODE="${LOONGSERVE_TIME_LIMIT_MODE:-drain}"
SUBMISSION_MODE="${SUBMISSION_MODE:-completion_gated}"
SYSTEM_VARIANT="${SYSTEM_VARIANT:-full_vendor_v8}"
ISOLATION_MEMORY_REQUESTED="${VLLM_ISOLATION_MEMORY_OPTIMIZATIONS:-0}"
# Independent and common to Native and vendor; do not clear these controls in
# the vendor-only feature lists below.
for startup_feature in STARTUP_CACHE_TRIM STARTUP_MEMORY_AUDIT; do
  startup_flag="VLLM_ISOLATION_${startup_feature}"
  startup_value="${!startup_flag:-0}"
  case "${startup_value}" in
    0|1) ;;
    *) echo "[ERROR] ${startup_flag} must be 0 or 1" >&2; exit 2 ;;
  esac
  export "${startup_flag}=${startup_value}"
done
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
  memory_value="${!memory_flag:-${ISOLATION_MEMORY_REQUESTED}}"
  case "${memory_value}" in
    0|1) ;;
    *) echo "[ERROR] ${memory_flag} must be 0 or 1" >&2; exit 2 ;;
  esac
  case "${SYSTEM_VARIANT}" in
    native_40sm_chunked|baseline_plus) memory_value=0 ;;
  esac
  export "${memory_flag}=${memory_value}"
done
ISOLATION_EXECUTION_REQUESTED="${VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS:-0}"
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
  execution_value="${!execution_flag:-${ISOLATION_EXECUTION_REQUESTED}}"
  case "${execution_value}" in
    0|1) ;;
    *) echo "[ERROR] ${execution_flag} must be 0 or 1" >&2; exit 2 ;;
  esac
  case "${SYSTEM_VARIANT}" in
    native_40sm_chunked|baseline_plus) execution_value=0 ;;
    full_vendor_v8) ;;
    *)
      case "${execution_feature}" in
        RING_QUERY_EXPANSION|P1_BOUNDARY_FASTPATH|P1_RING_SM_HANDOFF|DRAM_REFRESH_PROBE|D2H_EXACT_PLAN_PROGRESS|NATIVE_SOFT_HEADROOM_PROGRESS|PREFILL_BATCH_KV_BUDGET|WAITING_DECODE_RECLAIM) execution_value=0 ;;
      esac ;;
  esac
  export "${execution_flag}=${execution_value}"
done
export VLLM_ISOLATION_RING_QUERY_SCRATCH_BYTES="${VLLM_ISOLATION_RING_QUERY_SCRATCH_BYTES:-335544320}"
if [[ ! "${VLLM_ISOLATION_RING_QUERY_SCRATCH_BYTES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] VLLM_ISOLATION_RING_QUERY_SCRATCH_BYTES must be positive bytes" >&2
  exit 2
fi
ISOLATION_MATCHED_COMPARISON="${ISOLATION_MATCHED_COMPARISON:-0}"
ISOLATION_GUARDED_2K_COMPARISON="${ISOLATION_GUARDED_2K_COMPARISON:-0}"
ISOLATION_BATCH2_PD11_COMPARISON="${ISOLATION_BATCH2_PD11_COMPARISON:-0}"
for phase_name in PREFILL DECODE; do
  batch_flag="VLLM_ISOLATION_${phase_name}_BATCH_MAX_REQUESTS"
  batch_value="${!batch_flag:-0}"
  if [[ ! "${batch_value}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] ${batch_flag} must be a non-negative integer" >&2
    exit 2
  fi
  export "${batch_flag}=${batch_value}"
done
for isolation_switch in ISOLATION_GUARDED_2K_COMPARISON ISOLATION_BATCH2_PD11_COMPARISON \
  VLLM_ISOLATION_HBM_COMPLETION_GUARD VLLM_ISOLATION_DECODE_OUTPUT_FIRST \
  VLLM_ISOLATION_PREFILL_FIT_NEXT_CHUNK; do
  isolation_value="${!isolation_switch:-0}"
  case "${isolation_value}" in
    0|1) ;;
    *) echo "[ERROR] ${isolation_switch} must be 0 or 1" >&2; exit 2 ;;
  esac
  export "${isolation_switch}=${isolation_value}"
done
case "${ISOLATION_MATCHED_COMPARISON}" in
  0|1) ;;
  *) echo "[ERROR] ISOLATION_MATCHED_COMPARISON must be 0 or 1" >&2; exit 2 ;;
esac
LONG_OUTPUT_TOKENS="${LONG_OUTPUT_TOKENS:-32}"
SHORT_OUTPUT_TOKENS="${SHORT_OUTPUT_TOKENS:-32}"
WARMUP_COUNT="${WARMUP_COUNT:-0}"
WARMUP_OUTPUT_TOKENS="${WARMUP_OUTPUT_TOKENS:-8}"
WARMUP_COOLDOWN_SEC="${WARMUP_COOLDOWN_SEC:-3}"
PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-2048}"
# Legacy HBM-only cases previously ran FCFS even when the default PLAN said
# SRPT. Preserve their implicit behavior while making explicit SRPT effective.
case "${SYSTEM_VARIANT}" in
  dynamic_pd_hbm|dynamic_hbm_serialized_fixed_split|dynamic_hbm_serialized_fullsm)
    DEFAULT_PREFILL_SCHEDULING_POLICY=fcfs ;;
  *) DEFAULT_PREFILL_SCHEDULING_POLICY=srpt ;;
esac
PREFILL_SCHEDULING_POLICY="${PREFILL_SCHEDULING_POLICY:-${DEFAULT_PREFILL_SCHEDULING_POLICY}}"
BASELINE_PLUS_PREFILL_QUANTUM_TOKENS="${BASELINE_PLUS_PREFILL_QUANTUM_TOKENS:-0}"
VLLM_SCHEDULING_POLICY="${VLLM_SCHEDULING_POLICY:-fcfs}"
FULL_VENDOR_V8_RESIDENCY_VICTIM_POLICY="${FULL_VENDOR_V8_RESIDENCY_VICTIM_POLICY:-scheduled_prefill_fit}"
FULL_VENDOR_V8_SCHEDULED_PREFILL_FIT_CAPACITY_FRACTION="${FULL_VENDOR_V8_SCHEDULED_PREFILL_FIT_CAPACITY_FRACTION:-0.99}"
NATIVE_PREFILL_CAPACITY_MODE="${NATIVE_PREFILL_CAPACITY_MODE:-guard_wait}"
PD_PREFILL_INFLIGHT_DEPTH="${VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH:-1}"
PD_DECODE_INFLIGHT_DEPTH="${VLLM_PD_EVAL_DECODE_INFLIGHT_DEPTH:-1}"
PD_GLOBAL_INFLIGHT_DEPTH="${VLLM_PD_EVAL_GLOBAL_INFLIGHT_DEPTH:-2}"
PD_PHASE_DEPTH_ENABLED="${VLLM_PD_EVAL_PHASE_DEPTH_ENABLED:-1}"
PD_PREFILL_CHAIN_MAX_LOOKAHEAD_TOKENS="${VLLM_PD_PREFILL_CHAIN_MAX_LOOKAHEAD_TOKENS:-0}"
PD_LAYERWISE_PROFILE_HANDOFF="${VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF:-0}"
TTFT_LAYERWISE_PREFILL_EXECUTION="${VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION:-0}"
TTFT_LAYERWISE_PREFILL_PREEMPTION="${VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION:-0}"
TTFT_LAYERWISE_PREFILL_GROUP_LAYERS="${VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS:-2}"
KV_LAYERWISE_CHECKPOINT_PLAN="${VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN:-0}"
KV_LAYERWISE_OFFLOAD="${VLLM_KV_RETENTION_LAYERWISE_OFFLOAD:-0}"
KV_P1_PARKED="${VLLM_KV_RETENTION_P1_PARKED_PREFILL:-0}"
KV_P1_HBM_ONLY="${VLLM_KV_RETENTION_P1_HBM_ONLY:-0}"
KV_PREFILL_WORK_BINDING="${VLLM_KV_RETENTION_PREFILL_WORK_BINDING:-0}"
KV_PREFILL_RESUME_FRAME="${VLLM_KV_RETENTION_PREFILL_RESUME_FRAME:-0}"
KV_PARKED_MAX_BATCHES="${VLLM_KV_RETENTION_PARKED_MAX_BATCHES:-1}"
KV_PARKED_GPU_BYTES="${VLLM_KV_RETENTION_PARKED_GPU_BYTES:-268435456}"
PD_PREFILL_WORKSPACE_ENABLED="${VLLM_PD_PREFILL_WORKSPACE_ENABLED:-0}"
PD_PREFILL_WORKSPACE_MAX_TOKENS="${VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS:-16384}"
PD_PREFILL_WORKSPACE_SLOTS="${VLLM_PD_PREFILL_WORKSPACE_SLOTS:-1}"
PD_PREFILL_WORKSPACE_STRICT="${VLLM_PD_PREFILL_WORKSPACE_STRICT:-1}"
REQUESTED_PD_SERIALIZED_FULL_SM="${VLLM_PD_SERIALIZED_FULL_SM-}"
PD_SERIALIZED_FULL_SM="${VLLM_PD_SERIALIZED_FULL_SM:-1}"
DECODE_CONTINUATION_ROLLING_RECLAIM="${VLLM_VIRTUAL_CONTEXT_DECODE_CONTINUATION_ROLLING_RECLAIM:-0}"
FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS="${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS:-0}"
EFFECTIVE_FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS="${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS}"
REQUESTED_NATIVE_PREFILL_MAX_TOKENS="${VLLM_PD_EVAL_NATIVE_PREFILL_MAX_TOKENS-}"
PD_GREEN_DECODE_CUDAGRAPH="${VLLM_PD_EVAL_GREEN_DECODE_CUDAGRAPH:-${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS}}"
PD_GREEN_DECODE_CUDAGRAPH_STRICT="${VLLM_PD_EVAL_GREEN_DECODE_CUDAGRAPH_STRICT:-1}"
CUDAGRAPH_MODE="${VLLM_PD_EVAL_CUDAGRAPH_MODE:-}"
PD_FUSED_EXECUTE_SAMPLE="${VLLM_PD_EVAL_FUSED_EXECUTE_SAMPLE:-${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS}}"
PD_PHASE_SYNC_SAMPLE_OUTPUT="${VLLM_PD_EVAL_PHASE_SYNC_SAMPLE_OUTPUT:-0}"
DEFAULT_RING_BATCHED_PARTIAL=0
DEFAULT_RING_BATCHED_PARTIAL_STRICT=0
DEFAULT_RING_STREAMING_WAVE_GROUPS=1
RING_BATCHED_PARTIAL="${VLLM_PD_EVAL_RING_BATCHED_PARTIAL:-${DEFAULT_RING_BATCHED_PARTIAL}}"
RING_BATCHED_PARTIAL_STRICT="${VLLM_PD_EVAL_RING_BATCHED_PARTIAL_STRICT:-${DEFAULT_RING_BATCHED_PARTIAL_STRICT}}"
RING_BATCHED_WORKSPACE_BYTES="${VLLM_PD_EVAL_RING_BATCHED_WORKSPACE_BYTES:-0}"
RING_STREAMING_WAVE_GROUPS="${VLLM_PD_EVAL_RING_STREAMING_WAVE_GROUPS:-${DEFAULT_RING_STREAMING_WAVE_GROUPS}}"
RING_BW_DYNAMIC="${VLLM_PD_EVAL_RING_BW_DYNAMIC:-${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS}}"
if [[ "${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS}" == "1" ]]; then
  DEFAULT_RING_BW_MAX_TOKENS=16384
  DEFAULT_VIRTUAL_CONTEXT_TIMING=0
else
  DEFAULT_RING_BW_MAX_TOKENS="${PREFILL_CHUNK_TOKENS}"
  DEFAULT_VIRTUAL_CONTEXT_TIMING=1
fi
RING_BW_MAX_TOKENS="${VLLM_PD_EVAL_RING_BW_MAX_TOKENS:-${DEFAULT_RING_BW_MAX_TOKENS}}"
RING_BW_POLICY="${VLLM_PD_EVAL_RING_BW_POLICY:-safe_capacity}"
RING_BW_SAFETY_BLOCKS="${VLLM_PD_EVAL_RING_BW_SAFETY_BLOCKS:-0}"
RING_BW_DECODE_GROWTH_BLOCKS="${VLLM_PD_EVAL_RING_BW_DECODE_GROWTH_BLOCKS:-0}"
RING_BW_USE_SCHEDULED_HEADROOM="${VLLM_PD_EVAL_RING_BW_USE_SCHEDULED_HEADROOM:-${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS}}"
ACTIVE_DECODE_ELASTIC_PREFILL="${VLLM_PD_EVAL_ACTIVE_DECODE_ELASTIC_PREFILL:-0}"
if [[ ! -v VLLM_PD_EVAL_ACTIVE_DECODE_ELASTIC_PREFILL \
      && "${VLLM_ISOLATION_RING_QUERY_EXPANSION}" == "1" ]]; then
  ACTIVE_DECODE_ELASTIC_PREFILL=1
fi
RING_COMPUTE_GROUP_TOKENS="${VLLM_PD_EVAL_RING_COMPUTE_GROUP_TOKENS:-${PREFILL_CHUNK_TOKENS}}"
VIRTUAL_CONTEXT_TIMING="${VLLM_PD_EVAL_VIRTUAL_CONTEXT_TIMING:-${DEFAULT_VIRTUAL_CONTEXT_TIMING}}"
if [[ "${RING_BATCHED_PARTIAL}" == "1" \
      && -z "${VLLM_PD_EVAL_RING_STREAMING_WAVE_GROUPS+x}" ]]; then
  RING_STREAMING_WAVE_GROUPS=1
fi
if [[ "${PD_PHASE_SYNC_SAMPLE_OUTPUT}" != "0" \
      && "${PD_PHASE_SYNC_SAMPLE_OUTPUT}" != "1" ]]; then
  echo "[ERROR] VLLM_PD_EVAL_PHASE_SYNC_SAMPLE_OUTPUT must be 0 or 1" >&2
  exit 2
fi
VERIFY_TIERED_DIRECT_OPS="${VLLM_PD_EVAL_VERIFY_DIRECT_OPS:-1}"

INITIAL_GREEN_SM_SPLIT="${INITIAL_GREEN_SM_SPLIT:-20/20}"
PD_DYNAMIC_SM_POLICY="${PD_DYNAMIC_SM_POLICY:-tbt_slo_profiled}"
if [[ -z "${PD_SM_PROFILES+x}" ]]; then
  case "${PD_DYNAMIC_SM_POLICY}" in
    feedback|tbt_feedback|tbt_slo_feedback|tbt_slo_predictive)
      if [[ "${SYSTEM_VARIANT}" != "native_40sm_chunked" \
            && "${SYSTEM_VARIANT}" != "baseline_plus" \
            && "${PD_GREEN_DECODE_CUDAGRAPH}" == "1" ]]; then
        # Full Decode graphs are captured once per GreenContext profile. Keep
        # the default graph-enabled matrix on the measured T4 SLO frontier
        # instead of creating/capturing all 21 dense profiles. Callers can
        # still override PD_SM_PROFILES for an explicit profile-bank study.
        PD_SM_PROFILES="0/40,30/10,32/8,34/6,36/4,38/2,40/0"
      else
        PD_SM_PROFILES="auto_dense"
      fi
      ;;
    *)
      PD_SM_PROFILES="0/40,8/32,16/24,20/20,24/16,32/8,40/0"
      ;;
  esac
fi
ACTIVE_TBT_OVERLAP_PROFILE=""
case "${PD_DYNAMIC_SM_POLICY}" in
  tbt|tbt_profiled|tbt_slo_profiled)
    ACTIVE_TBT_OVERLAP_PROFILE="${TBT_OVERLAP_PROFILE}"
    ;;
esac
TBT_PROFILE_MARGIN_MS="${TBT_PROFILE_MARGIN_MS:-5}"
TBT_PROFILE_STRICT="${TBT_PROFILE_STRICT:-1}"
TBT_PREDICTIVE_LEAD_MS="${VLLM_PD_TBT_PREDICTIVE_LEAD_MS:-0}"
TBT_PREDICTIVE_CUTOVER_MS="${VLLM_PD_TBT_PREDICTIVE_CUTOVER_MS:--1}"
TBT_PREDICTIVE_MIN_STEP_MS="${VLLM_PD_TBT_PREDICTIVE_MIN_STEP_MS:-0}"
if [[ "${PD_DYNAMIC_SM_POLICY}" == "tbt_slo_predictive" \
      && "${SYSTEM_VARIANT}" != "native_40sm_chunked" \
      && "${SYSTEM_VARIANT}" != "baseline_plus" ]]; then
  ACTIVE_TBT_OVERLAP_PROFILE="${TBT_OVERLAP_PROFILE}"
fi
PD_FEEDBACK_INITIAL_DECODE_SMS="${PD_FEEDBACK_INITIAL_DECODE_SMS:-30}"
PD_FEEDBACK_STEP_SMS="${PD_FEEDBACK_STEP_SMS:-2}"
PD_FEEDBACK_WINDOW_TOKENS="${PD_FEEDBACK_WINDOW_TOKENS:-32}"
PD_FEEDBACK_SLACK_RATIO="${PD_FEEDBACK_SLACK_RATIO:-0.90}"
PD_FEEDBACK_REPROBE_RATIO="${PD_FEEDBACK_REPROBE_RATIO:-0.80}"
PD_FEEDBACK_DECREASE_WINDOWS="${PD_FEEDBACK_DECREASE_WINDOWS:-3}"
PD_FEEDBACK_HOLD_WINDOWS="${PD_FEEDBACK_HOLD_WINDOWS:-16}"
PD_FEEDBACK_MIN_PREFILL_SMS="${PD_FEEDBACK_MIN_PREFILL_SMS:-2}"
PD_FEEDBACK_CONTEXT_BUCKET_TOKENS="${PD_FEEDBACK_CONTEXT_BUCKET_TOKENS:-25000}"
TTFT_PREFILL_SLO_GUARD_MS="${TTFT_PREFILL_SLO_GUARD_MS:-500}"
TTFT_PREFILL_STARVATION_THRESHOLD_MS="${TTFT_PREFILL_STARVATION_THRESHOLD_MS:-5000}"
TTFT_PREFILL_STARVATION_MAX_BYPASSES="${TTFT_PREFILL_STARVATION_MAX_BYPASSES:-8}"

SHAREGPT_RAW_PATH="${SHAREGPT_RAW_PATH:-${ROOT_DIR}/data/ShareGPT_V3_unfiltered_cleaned_split.json}"
DATASET_JSONL="${DATASET_JSONL:-${RESULT_DIR}/experiment_snapshot/sharegpt_3long_6short.jsonl}"
DATASET_TOKENIZER="${DATASET_TOKENIZER:-${MODEL_NAME}}"
PREPARE_DATASET="${PREPARE_DATASET:-1}"
SHAREGPT_CANDIDATE_POOL_SIZE="${SHAREGPT_CANDIDATE_POOL_SIZE:-256}"
SHAREGPT_UNDERLENGTH_POLICY="${SHAREGPT_UNDERLENGTH_POLICY:-pack}"

KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-4294967296}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
# Match the comparison wrapper even when this entry point is used directly.
# This is a ceiling, not a fixed batch size or permission to bypass KV admission.
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-100}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
VLLM_LONG_PREFILL_TOKEN_THRESHOLD="${VLLM_LONG_PREFILL_TOKEN_THRESHOLD:-0}"
if [[ -n "${REQUESTED_NATIVE_PREFILL_MAX_TOKENS}" ]]; then
  NATIVE_PREFILL_MAX_TOKENS="${REQUESTED_NATIVE_PREFILL_MAX_TOKENS}"
elif [[ "${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS}" == "1" ]]; then
  NATIVE_PREFILL_MAX_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS}"
else
  NATIVE_PREFILL_MAX_TOKENS=0
fi
if [[ ! "${NATIVE_PREFILL_MAX_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] VLLM_PD_EVAL_NATIVE_PREFILL_MAX_TOKENS must be a non-negative integer" >&2
  exit 2
fi
if ((NATIVE_PREFILL_MAX_TOKENS > 0)); then
  DEFAULT_ACTIVE_DECODE_NATIVE_PREFILL_TOKENS="${NATIVE_PREFILL_MAX_TOKENS}"
else
  DEFAULT_ACTIVE_DECODE_NATIVE_PREFILL_TOKENS="${PREFILL_CHUNK_TOKENS}"
fi
ACTIVE_DECODE_NATIVE_PREFILL_TOKENS="${VLLM_PD_EVAL_ACTIVE_DECODE_NATIVE_PREFILL_TOKENS:-${DEFAULT_ACTIVE_DECODE_NATIVE_PREFILL_TOKENS}}"
ACTIVE_DECODE_RING_PREFILL_TOKENS="${VLLM_PD_EVAL_ACTIVE_DECODE_RING_PREFILL_TOKENS:-${PREFILL_CHUNK_TOKENS}}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-14400}"
SERVER_READY_TIMEOUT_SEC="${SERVER_READY_TIMEOUT_SEC:-900}"
SERVER_SHUTDOWN_TIMEOUT_SEC="${SERVER_SHUTDOWN_TIMEOUT_SEC:-30}"
PORT_RELEASE_TIMEOUT_SEC="${PORT_RELEASE_TIMEOUT_SEC:-30}"
INSTANCE_CLEANUP_FAILURE_EXIT_CODE=86
BUILD_GREEN_BRIDGE="${BUILD_GREEN_BRIDGE:-auto}"
ENABLE_DMON="${ENABLE_DMON:-1}"
GENERATE_ABCD="${GENERATE_ABCD:-1}"
ENABLE_FULL_ITERATION_TRACE="${ENABLE_FULL_ITERATION_TRACE:-0}"
ENABLE_LIGHTWEIGHT_TIMELINE_TRACE="${ENABLE_LIGHTWEIGHT_TIMELINE_TRACE:-1}"
STRICT_BOUNDED_LIGHTWEIGHT_TIMELINE="${VLLM_PD_EVAL_STRICT_BOUNDED_LIGHTWEIGHT_TIMELINE:-${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS}}"
ENABLE_PD_EXECUTOR_DEBUG="${ENABLE_PD_EXECUTOR_DEBUG:-0}"
ITERATION_TRACE_SAMPLE_EVERY_N="${ITERATION_TRACE_SAMPLE_EVERY_N:-100}"
ENABLE_SCHEDULER_TIMING_TRACE="${ENABLE_SCHEDULER_TIMING_TRACE:-0}"
SCHEDULER_METRICS=""
case "${ENABLE_SCHEDULER_TIMING_TRACE}" in
  0) ;;
  1)
    if [[ ! "${ITERATION_TRACE_SAMPLE_EVERY_N}" =~ ^[1-9][0-9]*$ ]]; then
      echo "[ERROR] scheduler timing requires positive ITERATION_TRACE_SAMPLE_EVERY_N" >&2
      exit 2
    fi
    SCHEDULER_METRICS="${ARTIFACT_DIR}/scheduler_metrics.jsonl"
    ;;
  *) echo "[ERROR] ENABLE_SCHEDULER_TIMING_TRACE must be 0 or 1" >&2; exit 2 ;;
esac
DEBUG_TRACE_QUEUE_SIZE="${DEBUG_TRACE_QUEUE_SIZE:-4096}"
DECODE_FIRST_HOST_ENQUEUE="${DECODE_FIRST_HOST_ENQUEUE:-1}"
ENABLE_PD_NVTX_PHASE_TRACE="${ENABLE_PD_NVTX_PHASE_TRACE:-0}"
ENABLE_PD_CUDA_EVENT_TRACE="${ENABLE_PD_CUDA_EVENT_TRACE:-0}"
KILL_EXISTING_PORT_OWNER="${KILL_EXISTING_PORT_OWNER:-0}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./run_vendor_v8_short_validation.sh [--dry-run]

The standalone default follows the legacy completion-gated interpretation.
Set WORKLOAD=A, B, C, or LOONGSERVE to select a common open-loop workload used
by run_vendor_v8_dynamic_hbm_vs_full.sh. LOONGSERVE requires an immutable
LOONGSERVE_TRACE_JSONL prepared by prepare_loongserve_mixed_workload.py.
Use LOONGSERVE_TERMINATION_MODE=requests and a positive
LOONGSERVE_REQUEST_LIMIT to replay the first N trace rows and drain them.
EOF
}

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

read -r -a SHORT_OFFSET_ARRAY <<<"${SHORT_INJECTION_OFFSETS_SEC}"
read -r -a LONG_OFFSET_ARRAY <<<"${LONG_INJECTION_OFFSETS_SEC}"
if [[ "${WORKLOAD}" != "legacy" ]]; then
  WORKLOAD="${WORKLOAD^^}"
fi
case "${WORKLOAD}" in
  legacy|A|B|C|LOONGSERVE) ;;
  *)
    echo "[ERROR] WORKLOAD must be legacy, A, B, C, or LOONGSERVE" >&2
    exit 2
    ;;
esac
case "${SUBMISSION_MODE}" in
  completion_gated|concurrent_longs) ;;
  *)
    echo "[ERROR] unsupported SUBMISSION_MODE=${SUBMISSION_MODE}" >&2
    exit 2
    ;;
esac
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
case "${KILL_EXISTING_PORT_OWNER}" in
  0|1) ;;
  *)
    echo "[ERROR] KILL_EXISTING_PORT_OWNER must be 0 or 1" >&2
    exit 2
    ;;
esac
if [[ "${WORKLOAD}" == "legacy" ]]; then
  EFFECTIVE_SUBMISSION_MODE="${SUBMISSION_MODE}"
  if [[ "${SUBMISSION_MODE}" == "completion_gated" ]] \
    && ((${#SHORT_OFFSET_ARRAY[@]} != SHORT_REQUESTS_PER_LONG)); then
    echo "[ERROR] SHORT_INJECTION_OFFSETS_SEC needs "\
"${SHORT_REQUESTS_PER_LONG} values" >&2
    exit 2
  fi
  if [[ "${SUBMISSION_MODE}" == "completion_gated" ]]; then
    SHORT_REQUEST_COUNT=$((LONG_REQUEST_COUNT * SHORT_REQUESTS_PER_LONG))
  else
    SHORT_REQUEST_COUNT=${#SHORT_OFFSET_ARRAY[@]}
  fi
elif [[ "${WORKLOAD}" == "LOONGSERVE" ]]; then
  EFFECTIVE_SUBMISSION_MODE="loongserve_open_loop_trace"
  SHORT_REQUEST_COUNT=0
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
  if ! [[ "${LOONGSERVE_ARRIVAL_TIME_SCALE}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || [[ "${LOONGSERVE_ARRIVAL_TIME_SCALE}" =~ ^0*([.]0*)?$ ]]; then
    echo "[ERROR] LOONGSERVE_ARRIVAL_TIME_SCALE must be positive" >&2
    exit 2
  fi
  if ! [[ "${LOONGSERVE_TIME_LIMIT_SEC}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "[ERROR] LOONGSERVE_TIME_LIMIT_SEC must be non-negative" >&2
    exit 2
  fi
  if [[ ! "${LOONGSERVE_REQUEST_LIMIT}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] LOONGSERVE_REQUEST_LIMIT must be a non-negative integer" >&2
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
  DATASET_JSONL="${LOONGSERVE_TRACE_JSONL}"
  PREPARE_DATASET=0
else
  case "${WORKLOAD}" in
    A) EFFECTIVE_SUBMISSION_MODE=concurrent_longs ;;
    B) EFFECTIVE_SUBMISSION_MODE=short_burst_periodic_longs ;;
    C) EFFECTIVE_SUBMISSION_MODE=interleaved_open_loop ;;
  esac
  SHORT_REQUEST_COUNT="${REQUESTED_SHORT_REQUEST_COUNT:-5}"
  if ! [[ "${SHORT_REQUEST_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] SHORT_REQUEST_COUNT must be positive" >&2
    exit 2
  fi
  if [[ "${WORKLOAD}" == "A" ]] \
    && ((${#SHORT_OFFSET_ARRAY[@]} != SHORT_REQUEST_COUNT)); then
    echo "[ERROR] workload A requires SHORT_REQUEST_COUNT short offsets" >&2
    exit 2
  fi
  if [[ "${WORKLOAD}" == "B" ]] \
    && ((${#LONG_OFFSET_ARRAY[@]} != LONG_REQUEST_COUNT)); then
    echo "[ERROR] workload B requires LONG_REQUEST_COUNT long offsets" >&2
    exit 2
  fi
  if [[ "${WORKLOAD}" == "C" ]]; then
    if ! [[ "${INTERLEAVED_INJECTION_INTERVAL_SEC}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
      || [[ "${INTERLEAVED_INJECTION_INTERVAL_SEC}" =~ ^0*([.]0*)?$ ]]; then
      echo "[ERROR] workload C interval must be positive" >&2
      exit 2
    fi
  fi
fi
case "${SYSTEM_VARIANT}" in
  dynamic_pd_hbm|dynamic_hbm_serialized_fixed_split|dynamic_hbm_serialized_fullsm|dynamic_hbm_srpt|full_vendor_v8|native_40sm_chunked|baseline_plus) ;;
  *)
    echo "[ERROR] unsupported SYSTEM_VARIANT=${SYSTEM_VARIANT}" >&2
    exit 2
    ;;
esac
NATIVE_FULL_GPU_VARIANT=0
SCHEDULER_ONLY_BASELINE=0
case "${SYSTEM_VARIANT}" in
  native_40sm_chunked|baseline_plus)
    NATIVE_FULL_GPU_VARIANT=1
    ;;
esac
if [[ "${SYSTEM_VARIANT}" == "baseline_plus" ]]; then
  SCHEDULER_ONLY_BASELINE=1
fi
case "${SYSTEM_VARIANT}" in
  dynamic_hbm_serialized_fixed_split)
    expected_serialized_full_sm=0
    ;;
  dynamic_pd_hbm|dynamic_hbm_serialized_fullsm|dynamic_hbm_srpt|full_vendor_v8|native_40sm_chunked|baseline_plus)
    expected_serialized_full_sm=1
    ;;
esac
if [[ -n "${REQUESTED_PD_SERIALIZED_FULL_SM}" \
      && "${REQUESTED_PD_SERIALIZED_FULL_SM}" != "${expected_serialized_full_sm}" ]]; then
  echo "[ERROR] SYSTEM_VARIANT=${SYSTEM_VARIANT} requires VLLM_PD_SERIALIZED_FULL_SM=${expected_serialized_full_sm}, got ${REQUESTED_PD_SERIALIZED_FULL_SM}" >&2
  exit 2
fi
PD_SERIALIZED_FULL_SM="${expected_serialized_full_sm}"
HBM_ONLY_VARIANT=0
case "${SYSTEM_VARIANT}" in
  dynamic_pd_hbm|dynamic_hbm_serialized_fixed_split|dynamic_hbm_serialized_fullsm|dynamic_hbm_srpt)
    HBM_ONLY_VARIANT=1 ;;
esac
if [[ "${SYSTEM_VARIANT}" == "dynamic_hbm_serialized_fullsm" \
      || "${SYSTEM_VARIANT}" == "dynamic_hbm_srpt" ]]; then
  case "${PD_SM_PROFILES}" in
    auto|auto_dense|dense)
      ;;
    *)
      normalized_pd_sm_profiles=",${PD_SM_PROFILES//[[:space:]]/},"
      if [[ "${normalized_pd_sm_profiles}" != *",0/40,"* \
            && "${normalized_pd_sm_profiles}" != *",d0_p40,"* ]] \
          || [[ "${normalized_pd_sm_profiles}" != *",40/0,"* \
                && "${normalized_pd_sm_profiles}" != *",d40_p0,"* ]]; then
        echo "[ERROR] ${SYSTEM_VARIANT} requires both 0/40 and 40/0 profiles, got ${PD_SM_PROFILES}" >&2
        exit 2
      fi
      ;;
  esac
fi
if [[ "${SYSTEM_VARIANT}" != "full_vendor_v8" ]]; then
  # The four-case wrapper may explicitly enable GreenContext Decode graphs for
  # Dynamic-HBM. Every other throughput-bundle member remains full-vendor-only
  # so the controls differ only in the named experiment variables.
  if [[ "${NATIVE_FULL_GPU_VARIANT}" == "1" ]]; then
    PD_GREEN_DECODE_CUDAGRAPH=0
  fi
  PD_FUSED_EXECUTE_SAMPLE=0
  RING_BW_DYNAMIC=0
  RING_BW_MAX_TOKENS="${PREFILL_CHUNK_TOKENS}"
  RING_BW_USE_SCHEDULED_HEADROOM=0
  ACTIVE_DECODE_ELASTIC_PREFILL=0
  NATIVE_PREFILL_MAX_TOKENS=0
  ACTIVE_DECODE_NATIVE_PREFILL_TOKENS="${PREFILL_CHUNK_TOKENS}"
  ACTIVE_DECODE_RING_PREFILL_TOKENS="${PREFILL_CHUNK_TOKENS}"
  STRICT_BOUNDED_LIGHTWEIGHT_TIMELINE=0
  VIRTUAL_CONTEXT_TIMING=1
  EFFECTIVE_FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS=0
fi
if [[ "${ISOLATION_MATCHED_COMPARISON}" == "1" ]]; then
  # These are measurement/execution controls, not Context Ring policy. The
  # historical full-vendor bundle normalization above must not make V1 trace
  # more expensively or keep V2's ordinary HBM chunks smaller than V1's.
  STRICT_BOUNDED_LIGHTWEIGHT_TIMELINE=1
  VIRTUAL_CONTEXT_TIMING=0
  PD_FUSED_EXECUTE_SAMPLE=0
  case "${CUDAGRAPH_MODE}" in
    FULL_AND_PIECEWISE) MATCHED_GREEN_DECODE_CUDAGRAPH=1 ;;
    NONE) MATCHED_GREEN_DECODE_CUDAGRAPH=0 ;;
    *) echo "[ERROR] matched comparison requires FULL_AND_PIECEWISE or NONE" >&2; exit 2 ;;
  esac
  if [[ "${VLLM_EXTRA_ARGS:-}" == *"--enforce-eager"* ]]; then
    echo "[ERROR] matched Graph comparison preserves compilation; --enforce-eager is a separate experiment" >&2
    exit 2
  fi
  if [[ "${NATIVE_FULL_GPU_VARIANT}" != "1" ]]; then
    NATIVE_PREFILL_MAX_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS}"
    ACTIVE_DECODE_NATIVE_PREFILL_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS}"
    if [[ "${ISOLATION_GUARDED_2K_COMPARISON}" == "1" ]]; then
      NATIVE_PREFILL_MAX_TOKENS=2048
      ACTIVE_DECODE_NATIVE_PREFILL_TOKENS=2048
    fi
    if [[ "${PD_LAYERWISE_PROFILE_HANDOFF}" != "1" \
          || "${PD_PREFILL_WORKSPACE_ENABLED}" != "1" \
          || "${PD_PREFILL_WORKSPACE_SLOTS}" != "1" \
          || "${PD_PREFILL_WORKSPACE_STRICT}" != "1" \
          || "${PD_GREEN_DECODE_CUDAGRAPH}" != "${MATCHED_GREEN_DECODE_CUDAGRAPH}" ]]; then
      echo "[ERROR] matched vendor comparison requires SM handoff, one strict Prefill workspace and Green Decode graphs matching CUDAGRAPH_MODE; use run_vendor_v8_dynamic_hbm_vs_full.sh" >&2
      exit 2
    fi
  fi
  MATCHED_EXPECTED_OFFLOAD="${KV_P1_PARKED}"
  [[ "${KV_P1_HBM_ONLY}" != "1" ]] || MATCHED_EXPECTED_OFFLOAD=0
  if [[ "${PD_PHASE_DEPTH_ENABLED}/${PD_PREFILL_INFLIGHT_DEPTH}/${PD_DECODE_INFLIGHT_DEPTH}/${PD_GLOBAL_INFLIGHT_DEPTH}" != "1/1/1/2" \
        || "${TTFT_LAYERWISE_PREFILL_EXECUTION}" != "${KV_P1_PARKED}" \
        || "${TTFT_LAYERWISE_PREFILL_PREEMPTION}" != "${KV_P1_PARKED}" \
        || "${KV_LAYERWISE_CHECKPOINT_PLAN}" != "${KV_P1_PARKED}" \
        || "${KV_LAYERWISE_OFFLOAD}" != "${MATCHED_EXPECTED_OFFLOAD}" \
        || "${KV_PREFILL_WORK_BINDING}" != "${KV_P1_PARKED}" \
        || "${KV_PREFILL_RESUME_FRAME}" != "${KV_P1_PARKED}" \
        || "${PD_PHASE_SYNC_SAMPLE_OUTPUT}" != "0" \
        || "${PD_PREFILL_CHAIN_MAX_LOOKAHEAD_TOKENS}" != "0" \
        || "${PD_GREEN_DECODE_CUDAGRAPH_STRICT}" != "1" \
        || "${ENABLE_LIGHTWEIGHT_TIMELINE_TRACE}" != "1" \
        || "${ENABLE_FULL_ITERATION_TRACE}/${ENABLE_PD_EXECUTOR_DEBUG}/${ENABLE_PD_CUDA_EVENT_TRACE}/${ENABLE_PD_NVTX_PHASE_TRACE}" != "0/0/0/0" ]]; then
    echo "[ERROR] matched comparison requires P1/D1/G2, bounded traces, common Graph mode, and Stage-2 only for explicit P1 cases (offload is proposal-only); use the comparison wrapper preset" >&2
    exit 2
  fi
fi
for phase_name in PREFILL DECODE; do
  batch_flag="VLLM_ISOLATION_${phase_name}_BATCH_MAX_REQUESTS"
  batch_value="${!batch_flag}"
  if ((10#${batch_value} > VLLM_MAX_NUM_SEQS)); then
    echo "[ERROR] ${batch_flag} exceeds VLLM_MAX_NUM_SEQS" >&2
    exit 2
  fi
  if [[ "${NATIVE_FULL_GPU_VARIANT}" == "1" ]] && ((10#${batch_value} > 0)); then
    echo "[ERROR] phase batch limits are P/D-only; Native controls must use zero" >&2
    exit 2
  fi
done
if [[ "${ISOLATION_BATCH2_PD11_COMPARISON}" == "1" ]]; then
  expected_batch_limits=1/1
  [[ "${NATIVE_FULL_GPU_VARIANT}" == "1" ]] && expected_batch_limits=0/0
  if [[ "${ISOLATION_GUARDED_2K_COMPARISON}/${VLLM_MAX_NUM_SEQS}" != "1/2" \
        || "${VLLM_ISOLATION_PREFILL_BATCH_MAX_REQUESTS}/${VLLM_ISOLATION_DECODE_BATCH_MAX_REQUESTS}" != "${expected_batch_limits}" ]]; then
    echo "[ERROR] batch2/PD11 requires guarded2K, maxseq2 and case-local row caps; use comparison wrapper" >&2
    exit 2
  fi
fi
if [[ "${ISOLATION_GUARDED_2K_COMPARISON}" == "1" ]]; then
  if [[ "${ISOLATION_MATCHED_COMPARISON}" != "1" \
        || "${PREFILL_CHUNK_TOKENS}/${VLLM_LONG_PREFILL_TOKEN_THRESHOLD}" != "2048/2048" \
        || "${VLLM_ISOLATION_RING_QUERY_EXPANSION}" != "0" ]]; then
    echo "[ERROR] guarded2K comparison requires matched preset and per-request2K; use comparison wrapper" >&2
    exit 2
  fi
  expected_hbm_guard=1
  expected_output_first=1
  expected_next_fit=0
  if [[ "${SYSTEM_VARIANT}" == "full_vendor_v8" ]]; then
    expected_hbm_guard=0
    expected_next_fit=1
    if [[ "${KV_P1_PARKED}" != "1" || "${VLLM_ISOLATION_P1_RING_SM_HANDOFF}" != "1" ]]; then
      echo "[ERROR] guarded2K proposal requires complete P1 bundle and Ring SM handoff" >&2
      exit 2
    fi
  fi
  [[ "${NATIVE_FULL_GPU_VARIANT}" == "1" ]] && expected_output_first=0
  if [[ "${VLLM_ISOLATION_HBM_COMPLETION_GUARD}/${VLLM_ISOLATION_DECODE_OUTPUT_FIRST}/${VLLM_ISOLATION_PREFILL_FIT_NEXT_CHUNK}" != "${expected_hbm_guard}/${expected_output_first}/${expected_next_fit}" ]]; then
    echo "[ERROR] guarded2K case guard/output-first/next-fit mismatch" >&2
    exit 2
  fi
  if [[ "${NATIVE_FULL_GPU_VARIANT}" != "1" \
        && "${NATIVE_PREFILL_MAX_TOKENS}/${ACTIVE_DECODE_NATIVE_PREFILL_TOKENS}/${ACTIVE_DECODE_RING_PREFILL_TOKENS}/${RING_BW_MAX_TOKENS}" != "2048/2048/2048/2048" ]]; then
    echo "[ERROR] guarded2K vendor execution caps must all be2048" >&2
    exit 2
  fi
fi
if [[ "${NATIVE_FULL_GPU_VARIANT}" != "1" \
      && "${PD_GREEN_DECODE_CUDAGRAPH}" == "1" \
      && "${ENABLE_LIGHTWEIGHT_TIMELINE_TRACE}" != "1" ]]; then
  echo "[ERROR] GreenContext Decode CUDA Graph comparison runs require the lightweight timeline" >&2
  exit 2
fi
case "${RING_BATCHED_PARTIAL}" in
  0|1) ;;
  *)
    echo "[ERROR] VLLM_PD_EVAL_RING_BATCHED_PARTIAL must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${RING_BATCHED_PARTIAL_STRICT}" in
  0|1) ;;
  *)
    echo "[ERROR] VLLM_PD_EVAL_RING_BATCHED_PARTIAL_STRICT must be 0 or 1" >&2
    exit 2
    ;;
esac
for bool_value in \
  "${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS}" \
  "${PD_GREEN_DECODE_CUDAGRAPH}" \
  "${PD_GREEN_DECODE_CUDAGRAPH_STRICT}" \
  "${PD_FUSED_EXECUTE_SAMPLE}" \
  "${RING_BW_DYNAMIC}" \
  "${RING_BW_USE_SCHEDULED_HEADROOM}" \
  "${ACTIVE_DECODE_ELASTIC_PREFILL}" \
  "${STRICT_BOUNDED_LIGHTWEIGHT_TIMELINE}" \
  "${ENABLE_PD_EXECUTOR_DEBUG}" \
  "${VIRTUAL_CONTEXT_TIMING}"; do
  case "${bool_value}" in
    0|1) ;;
    *)
      echo "[ERROR] throughput optimization boolean flags must be 0 or 1" >&2
      exit 2
      ;;
  esac
done
case "${PD_SERIALIZED_FULL_SM}" in
  0|1) ;;
  *)
    echo "[ERROR] VLLM_PD_SERIALIZED_FULL_SM must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${PD_LAYERWISE_PROFILE_HANDOFF}" in
  0|1) ;;
  *)
    echo "[ERROR] VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${TTFT_LAYERWISE_PREFILL_EXECUTION}" in
  0|1) ;;
  *)
    echo "[ERROR] VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION must be 0 or 1" >&2
    exit 2
    ;;
esac
for flag_value in "${KV_P1_PARKED}" "${KV_P1_HBM_ONLY}" "${KV_PREFILL_WORK_BINDING}" "${KV_PREFILL_RESUME_FRAME}"; do
  case "${flag_value}" in
    0|1) ;;
    *) echo "[ERROR] P1/work-binding/resume-frame flags must be 0 or 1" >&2; exit 2 ;;
  esac
done
if [[ "${KV_P1_HBM_ONLY}" == "1" ]] && \
   [[ "${SYSTEM_VARIANT}" != "dynamic_hbm_srpt" \
      || "${KV_P1_PARKED}/${KV_LAYERWISE_OFFLOAD}/${VLLM_ISOLATION_HBM_COMPLETION_GUARD}" != "1/0/1" ]]; then
  echo "[ERROR] P1 HBM-only preemption requires dynamic_hbm_srpt, P1 parking, HBM completion guard, and KV offload=0" >&2
  exit 2
fi
if [[ "${KV_P1_PARKED}" == "1" ]]; then
  KV_P1_EXPECTED_OFFLOAD=1
  [[ "${KV_P1_HBM_ONLY}" != "1" ]] || KV_P1_EXPECTED_OFFLOAD=0
  if [[ ( "${SYSTEM_VARIANT}" != "full_vendor_v8" && "${KV_P1_HBM_ONLY}" != "1" ) \
        || "${PD_PREFILL_INFLIGHT_DEPTH}/${PD_DECODE_INFLIGHT_DEPTH}/${PD_GLOBAL_INFLIGHT_DEPTH}" != "1/1/2" \
        || "${KV_PREFILL_WORK_BINDING}/${KV_PREFILL_RESUME_FRAME}/${KV_LAYERWISE_CHECKPOINT_PLAN}" != "1/1/1" \
        || "${KV_LAYERWISE_OFFLOAD}" != "${KV_P1_EXPECTED_OFFLOAD}" \
        || "${PD_LAYERWISE_PROFILE_HANDOFF}/${TTFT_LAYERWISE_PREFILL_EXECUTION}/${TTFT_LAYERWISE_PREFILL_PREEMPTION}" != "1/1/1" \
        || "${PD_PREFILL_WORKSPACE_ENABLED}/${PD_PREFILL_WORKSPACE_SLOTS}/${PD_PREFILL_WORKSPACE_STRICT}" != "1/1/1" \
        || "${KV_PARKED_MAX_BATCHES}" != "1" \
        || ! "${KV_PARKED_GPU_BYTES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] P1 parking requires full_vendor_v8 offload or explicit HBM-only mode, P1/D1/G2, work/frame/checkpoint, SM/Stage2, one strict workspace and one bounded parked batch" >&2
    exit 2
  fi
fi
case "${TTFT_LAYERWISE_PREFILL_PREEMPTION}" in
  0|1) ;;
  *)
    echo "[ERROR] VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION must be 0 or 1" >&2
    exit 2
    ;;
esac
if [[ "${TTFT_LAYERWISE_PREFILL_PREEMPTION}" == "1" \
      && "${TTFT_LAYERWISE_PREFILL_EXECUTION}" != "1" ]]; then
  echo "[ERROR] TTFT layerwise Prefill preemption requires VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION=1" >&2
  exit 2
fi
for flag_value in "${KV_LAYERWISE_CHECKPOINT_PLAN}" "${KV_LAYERWISE_OFFLOAD}"; do
  case "${flag_value}" in
    0|1) ;;
    *) echo "[ERROR] layerwise checkpoint/offload flags must be 0 or 1" >&2; exit 2 ;;
  esac
done
if [[ "${SYSTEM_VARIANT}" != "full_vendor_v8" && "${KV_P1_HBM_ONLY}" != "1" ]] \
    && [[ "${TTFT_LAYERWISE_PREFILL_PREEMPTION}" != "0" \
          || "${KV_LAYERWISE_CHECKPOINT_PLAN}" != "0" \
          || "${KV_LAYERWISE_OFFLOAD}" != "0" ]]; then
  echo "[ERROR] Stage-2 request preemption and KV checkpoint/offload require SYSTEM_VARIANT=full_vendor_v8; controls must keep them disabled" >&2
  exit 2
fi
if [[ "${KV_LAYERWISE_CHECKPOINT_PLAN}" == "1" \
      && "${TTFT_LAYERWISE_PREFILL_PREEMPTION}" != "1" ]]; then
  echo "[ERROR] layerwise KV checkpoint requires Stage-2 Prefill preemption" >&2
  exit 2
fi
if [[ "${KV_LAYERWISE_OFFLOAD}" == "1" \
      && "${KV_LAYERWISE_CHECKPOINT_PLAN}" != "1" ]]; then
  echo "[ERROR] layerwise KV offload requires VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN=1" >&2
  exit 2
fi
if [[ ! "${TTFT_LAYERWISE_PREFILL_GROUP_LAYERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS must be positive" >&2
  exit 2
fi
case "${PD_PREFILL_WORKSPACE_ENABLED}" in
  0|1) ;;
  *)
    echo "[ERROR] VLLM_PD_PREFILL_WORKSPACE_ENABLED must be 0 or 1" >&2
    exit 2
    ;;
esac
case "${PD_PREFILL_WORKSPACE_STRICT}" in
  0|1) ;;
  *)
    echo "[ERROR] VLLM_PD_PREFILL_WORKSPACE_STRICT must be 0 or 1" >&2
    exit 2
    ;;
esac
for token_value in \
  "${NATIVE_PREFILL_MAX_TOKENS}" \
  "${ACTIVE_DECODE_NATIVE_PREFILL_TOKENS}" \
  "${ACTIVE_DECODE_RING_PREFILL_TOKENS}"; do
  if [[ ! "${token_value}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Native/Ring Prefill token controls must be non-negative integers" >&2
    exit 2
  fi
done
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
if ((VLLM_LONG_PREFILL_TOKEN_THRESHOLD > VLLM_MAX_MODEL_LEN)); then
  echo "[ERROR] VLLM_LONG_PREFILL_TOKEN_THRESHOLD cannot exceed VLLM_MAX_MODEL_LEN" >&2
  exit 2
fi
if ((NATIVE_PREFILL_MAX_TOKENS > 0)); then
  if ((NATIVE_PREFILL_MAX_TOKENS < PREFILL_CHUNK_TOKENS)); then
    echo "[ERROR] Native Prefill max tokens cannot be smaller than PREFILL_CHUNK_TOKENS" >&2
    exit 2
  fi
  if ((NATIVE_PREFILL_MAX_TOKENS > VLLM_MAX_NUM_BATCHED_TOKENS)); then
    echo "[ERROR] Native Prefill max tokens cannot exceed VLLM_MAX_NUM_BATCHED_TOKENS" >&2
    exit 2
  fi
  if [[ "${PREFILL_SCHEDULING_POLICY}" == "slo_srpt" ]]; then
    echo "[ERROR] Native Prefill chunk expansion is not calibrated for slo_srpt" >&2
    exit 2
  fi
fi
if [[ ! "${PD_PREFILL_WORKSPACE_MAX_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS must be positive" >&2
  exit 2
fi
if [[ ! "${PD_PREFILL_WORKSPACE_SLOTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] VLLM_PD_PREFILL_WORKSPACE_SLOTS must be positive" >&2
  exit 2
fi
if [[ "${PD_PREFILL_WORKSPACE_ENABLED}" == "1" ]]; then
  if [[ "${NATIVE_FULL_GPU_VARIANT}" == "1" ]]; then
    echo "[ERROR] ${SYSTEM_VARIANT} cannot enable the vendor fixed Prefill workspace" >&2
    exit 2
  fi
  if [[ "${PD_PHASE_DEPTH_ENABLED}" != "1" \
        || ! "${PD_PREFILL_INFLIGHT_DEPTH}" =~ ^[1-9][0-9]*$ ]] \
      || ((PD_PREFILL_WORKSPACE_SLOTS < PD_PREFILL_INFLIGHT_DEPTH)); then
    echo "[ERROR] fixed Prefill workspace requires phase depth enabled and slots >= Prefill depth" >&2
    exit 2
  fi
  if ((PD_PREFILL_WORKSPACE_MAX_TOKENS < VLLM_MAX_NUM_BATCHED_TOKENS)); then
    echo "[ERROR] fixed Prefill workspace max tokens (${PD_PREFILL_WORKSPACE_MAX_TOKENS}) must cover --max-num-batched-tokens (${VLLM_MAX_NUM_BATCHED_TOKENS})" >&2
    exit 2
  fi
fi
if [[ "${PD_LAYERWISE_PROFILE_HANDOFF}" == "1" ]]; then
  if [[ "${NATIVE_FULL_GPU_VARIANT}" == "1" ]]; then
    echo "[ERROR] ${SYSTEM_VARIANT} cannot enable vendor layerwise profile handoff" >&2
    exit 2
  fi
  if [[ "${PD_PHASE_DEPTH_ENABLED}" != "1" \
        || "${PD_PREFILL_INFLIGHT_DEPTH}" != "1" \
        || "${PD_DECODE_INFLIGHT_DEPTH}" != "1" \
        || ! "${PD_GLOBAL_INFLIGHT_DEPTH}" =~ ^[0-9]+$ ]] \
      || ((PD_GLOBAL_INFLIGHT_DEPTH < 2)); then
    echo "[ERROR] layerwise profile handoff requires phase depth enabled with P=1/D=1/G>=2" >&2
    exit 2
  fi
fi
if [[ "${TTFT_LAYERWISE_PREFILL_EXECUTION}" == "1" \
      && "${NATIVE_FULL_GPU_VARIANT}" == "1" ]]; then
  echo "[ERROR] ${SYSTEM_VARIANT} cannot enable vendor TTFT layer-group Prefill execution" >&2
  exit 2
fi
if [[ "${TTFT_LAYERWISE_PREFILL_PREEMPTION}" == "1" ]]; then
  if [[ "${NATIVE_FULL_GPU_VARIANT}" == "1" ]]; then
    echo "[ERROR] ${SYSTEM_VARIANT} cannot enable vendor TTFT Prefill preemption" >&2
    exit 2
  fi
  if [[ "${PD_PHASE_DEPTH_ENABLED}" != "1" \
        || ! "${PD_PREFILL_INFLIGHT_DEPTH}" =~ ^[0-9]+$ \
        || "${PD_DECODE_INFLIGHT_DEPTH}" != "1" \
        || ! "${PD_GLOBAL_INFLIGHT_DEPTH}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] TTFT Prefill preemption requires phase depth P>=2/D=1/G>=P+D" >&2
    exit 2
  fi
  if [[ "${PREFILL_SCHEDULING_POLICY}" != "srpt" ]]; then
    echo "[ERROR] TTFT Prefill preemption currently requires PREFILL_SCHEDULING_POLICY=srpt" >&2
    exit 2
  fi
  if (( (KV_P1_PARKED == 0 && PD_PREFILL_INFLIGHT_DEPTH < 2) \
       || PD_GLOBAL_INFLIGHT_DEPTH \
          < PD_PREFILL_INFLIGHT_DEPTH + PD_DECODE_INFLIGHT_DEPTH)); then
    echo "[ERROR] TTFT Prefill preemption requires phase depth P>=2/D=1/G>=P+D" >&2
    exit 2
  fi
  if [[ "${PD_LAYERWISE_PROFILE_HANDOFF}" != "${KV_P1_PARKED}" \
        || "${PD_FUSED_EXECUTE_SAMPLE}" != "0" ]]; then
    echo "[ERROR] TTFT Prefill preemption requires unified handoff only for P1 parking and fused execute/sample disabled" >&2
    exit 2
  fi
fi
case "${CUDAGRAPH_MODE}" in
  ""|NONE|PIECEWISE|FULL|FULL_DECODE_ONLY|FULL_AND_PIECEWISE) ;;
  *)
    echo "[ERROR] VLLM_PD_EVAL_CUDAGRAPH_MODE is invalid: ${CUDAGRAPH_MODE}" >&2
    exit 2
    ;;
esac
if [[ "${PD_GREEN_DECODE_CUDAGRAPH}" == "1" ]]; then
  case "${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}" in
    FULL|FULL_DECODE_ONLY|FULL_AND_PIECEWISE) ;;
    *)
      echo "[ERROR] GreenContext Decode CUDA Graph requires a mode with full Decode capture" >&2
      exit 2
      ;;
  esac
fi
if ! [[ "${RING_BATCHED_WORKSPACE_BYTES}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] VLLM_PD_EVAL_RING_BATCHED_WORKSPACE_BYTES must be non-negative" >&2
  exit 2
fi
if ! [[ "${RING_STREAMING_WAVE_GROUPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] VLLM_PD_EVAL_RING_STREAMING_WAVE_GROUPS must be positive" >&2
  exit 2
fi
for token_value in \
  "${RING_BW_MAX_TOKENS}" \
  "${RING_COMPUTE_GROUP_TOKENS}"; do
  if ! [[ "${token_value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Ring token controls must be positive integers" >&2
    exit 2
  fi
done
for block_value in \
  "${RING_BW_SAFETY_BLOCKS}" \
  "${RING_BW_DECODE_GROWTH_BLOCKS}"; do
  if ! [[ "${block_value}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Ring reserve controls must be non-negative integers" >&2
    exit 2
  fi
done
case "${RING_BW_POLICY}" in
  step|safe|capacity|safe_capacity) ;;
  *)
    echo "[ERROR] VLLM_PD_EVAL_RING_BW_POLICY must be step or safe_capacity" >&2
    exit 2
    ;;
esac
if [[ "${ACTIVE_DECODE_ELASTIC_PREFILL}" == "1" ]]; then
  if [[ "${RING_BW_DYNAMIC}" != "1" \
        || "${RING_BW_POLICY}" != "safe_capacity" ]]; then
    echo "[ERROR] active-Decode elastic Prefill requires dynamic safe_capacity Ring admission" >&2
    exit 2
  fi
  if [[ "${RING_BW_USE_SCHEDULED_HEADROOM}" != "1" \
        && "${RING_BW_SAFETY_BLOCKS}" == "0" ]]; then
    echo "[ERROR] active-Decode elastic Prefill requires scheduled headroom or an explicit safety reserve" >&2
    exit 2
  fi
fi
if ((RING_BW_MAX_TOKENS < PREFILL_CHUNK_TOKENS)); then
  echo "[ERROR] Ring max tokens cannot be smaller than PREFILL_CHUNK_TOKENS" >&2
  exit 2
fi
if [[ "${RING_BATCHED_PARTIAL}" == "1" \
      && "${RING_STREAMING_WAVE_GROUPS}" != "1" ]]; then
  echo "[ERROR] all-group batched partial and streaming waves are mutually exclusive" >&2
  exit 2
fi
case "${PREFILL_SCHEDULING_POLICY}" in
  fcfs|srpt|slo_srpt) ;;
  *)
    echo "[ERROR] PREFILL_SCHEDULING_POLICY must be fcfs, srpt or slo_srpt" >&2
    exit 2
    ;;
esac
if [[ "${SYSTEM_VARIANT}" == "dynamic_hbm_srpt" \
      && "${PREFILL_SCHEDULING_POLICY}" != "srpt" ]]; then
  echo "[ERROR] dynamic_hbm_srpt requires PREFILL_SCHEDULING_POLICY=srpt" >&2
  exit 2
fi
if [[ "${HBM_ONLY_VARIANT}" == "1" \
      && "${PREFILL_SCHEDULING_POLICY}" == "slo_srpt" ]]; then
  echo "[ERROR] HBM-only comparisons support fcfs or srpt, not fixed-quantum slo_srpt" >&2
  exit 2
fi
if ! [[ "${BASELINE_PLUS_PREFILL_QUANTUM_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] BASELINE_PLUS_PREFILL_QUANTUM_TOKENS must be a non-negative integer" >&2
  exit 2
fi
if [[ "${SCHEDULER_ONLY_BASELINE}" == "1" ]]; then
  if [[ "${PREFILL_SCHEDULING_POLICY}" != "srpt" ]]; then
    echo "[ERROR] baseline_plus requires PREFILL_SCHEDULING_POLICY=srpt" >&2
    exit 2
  fi
  if [[ "${BASELINE_PLUS_PREFILL_QUANTUM_TOKENS}" != "0" ]]; then
    echo "[ERROR] baseline_plus requires BASELINE_PLUS_PREFILL_QUANTUM_TOKENS=0 so scheduling does not change the native chunk size" >&2
    exit 2
  fi
  if [[ "${VLLM_SCHEDULING_POLICY}" != "fcfs" ]]; then
    echo "[ERROR] baseline_plus requires VLLM_SCHEDULING_POLICY=fcfs" >&2
    exit 2
  fi
  if [[ "${PD_LAYERWISE_PROFILE_HANDOFF}" != "0" \
        || "${PD_GREEN_DECODE_CUDAGRAPH}" != "0" \
        || "${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS}" != "0" ]]; then
    echo "[ERROR] baseline_plus permits only SRPT/FCFS request scheduling; layerwise, Green Decode graph, and vendor throughput bundles must stay disabled" >&2
    exit 2
  fi
  if [[ -z "${VLLM_VENDOR_DIR}" \
        || ! -f "${VLLM_VENDOR_DIR}/vllm/v1/core/sched/prefill_chunk_scheduler.py" \
        || ! -f "${VLLM_VENDOR_DIR}/vllm/v1/core/sched/decode_scheduler.py" ]]; then
    echo "[ERROR] baseline_plus requires VLLM_VENDOR_DIR with the vendor_v8 Prefill/Decode scheduler implementation" >&2
    exit 2
  fi
fi
case "${FULL_VENDOR_V8_RESIDENCY_VICTIM_POLICY}" in
  longest_logical_context_first|min_ring_chunks|jit_min_copy|scheduled_prefill_fit)
    ;;
  *)
    echo "[ERROR] unsupported FULL_VENDOR_V8_RESIDENCY_VICTIM_POLICY="\
"${FULL_VENDOR_V8_RESIDENCY_VICTIM_POLICY}" >&2
    exit 2
    ;;
esac
case "${NATIVE_PREFILL_CAPACITY_MODE}" in
  guard_wait|legacy_preempt_recompute) ;;
  *)
    echo "[ERROR] NATIVE_PREFILL_CAPACITY_MODE must be guard_wait or legacy_preempt_recompute" >&2
    exit 2
    ;;
esac
if [[ "${NATIVE_FULL_GPU_VARIANT}" == "1" ]]; then
  DECODE_SMS=40
  PREFILL_SMS=0
else
  if [[ "${INITIAL_GREEN_SM_SPLIT}" != */* ]]; then
    echo "[ERROR] INITIAL_GREEN_SM_SPLIT must be decode/prefill" >&2
    exit 2
  fi
  DECODE_SMS="${INITIAL_GREEN_SM_SPLIT%%/*}"
  PREFILL_SMS="${INITIAL_GREEN_SM_SPLIT##*/}"
  if ! [[ "${DECODE_SMS}" =~ ^[1-9][0-9]*$ \
          && "${PREFILL_SMS}" =~ ^[1-9][0-9]*$ ]] \
      || ((DECODE_SMS + PREFILL_SMS != 40)); then
    echo "[ERROR] invalid T4 initial split: ${INITIAL_GREEN_SM_SPLIT}" >&2
    exit 2
  fi
fi
if [[ "${SCHEDULER_ONLY_BASELINE}" == "1" ]]; then
  # Even disabled virtual-context timing can add vendor-only hot-path probes.
  # Pin both the evaluation input and low-level environment to zero so the
  # comparison changes request order only.
  VIRTUAL_CONTEXT_TIMING=0
fi

# Resolve the scheduling arguments before dry-run. The same variables are used
# by the real server launch below, so a policy label cannot stand in for an
# enabled selector. Only slo_srpt's measured cost arguments are appended later,
# after its existing profile-file read; those are not used by HBM-only controls.
VLLM_EXTRA_ARGS="--attention-backend TRITON_ATTN \
--kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES} \
--enable-tbt-decode-scheduling \
--tbt-decode-scheduling-policy fcfs"
PD_EVAL_POLICY="muxwise_dynamic"
RESIDENCY_VICTIM_POLICY="min_ring_chunks"
PD_DYNAMIC_SM_ENABLED=1
OPERATOR_SELECTIVE_PD=1
RING_ENABLE_CHUNKED_PREFILL=1
PREFILL_SELECTOR_QUANTUM_TOKENS=0
if [[ "${SYSTEM_VARIANT}" == "native_40sm_chunked" ]]; then
  VLLM_EXTRA_ARGS="--attention-backend TRITON_ATTN \
--kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES}"
  PD_EVAL_POLICY="manual"
  RESIDENCY_VICTIM_POLICY="none"
  PD_DYNAMIC_SM_ENABLED=0
  OPERATOR_SELECTIVE_PD=0
  RING_ENABLE_CHUNKED_PREFILL=0
elif [[ "${SCHEDULER_ONLY_BASELINE}" == "1" ]]; then
  VLLM_EXTRA_ARGS="--attention-backend TRITON_ATTN \
--kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES} \
--enable-ttft-prefill-chunk-scheduling \
--ttft-prefill-chunk-scheduling-policy srpt \
--ttft-prefill-chunk-quantum-tokens ${BASELINE_PLUS_PREFILL_QUANTUM_TOKENS} \
--enable-tbt-decode-scheduling \
--tbt-decode-scheduling-policy fcfs"
  PD_EVAL_POLICY="mixed_iteration"
  RESIDENCY_VICTIM_POLICY="none"
  PD_DYNAMIC_SM_ENABLED=0
  OPERATOR_SELECTIVE_PD=0
  RING_ENABLE_CHUNKED_PREFILL=0
elif [[ "${SYSTEM_VARIANT}" == "full_vendor_v8" ]]; then
  PREFILL_SELECTOR_QUANTUM_TOKENS="${PREFILL_CHUNK_TOKENS}"
  PD_EVAL_POLICY="muxwise_dynamic_virtual_ring_optimized"
  RESIDENCY_VICTIM_POLICY="${FULL_VENDOR_V8_RESIDENCY_VICTIM_POLICY}"
  VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS} \
--enable-ttft-prefill-chunk-scheduling \
--ttft-prefill-chunk-scheduling-policy ${PREFILL_SCHEDULING_POLICY} \
--ttft-prefill-chunk-quantum-tokens ${PREFILL_CHUNK_TOKENS} \
--ttft-prefill-native-chunk-max-tokens ${NATIVE_PREFILL_MAX_TOKENS}"
elif [[ "${HBM_ONLY_VARIANT}" == "1" \
        && ( "${PREFILL_SCHEDULING_POLICY}" == "srpt" \
             || "${ISOLATION_MATCHED_COMPARISON}" == "1" ) ]]; then
  # Quantum zero leaves token/sequence sizing to native admission. This enables
  # only request ordering; muxwise_dynamic still clears all virtual-KV paths.
  VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS} \
--enable-ttft-prefill-chunk-scheduling \
--ttft-prefill-chunk-scheduling-policy ${PREFILL_SCHEDULING_POLICY} \
--ttft-prefill-chunk-quantum-tokens 0"
fi

MATCHED_CUDAGRAPH_CAPTURE_SIZES=()
if [[ "${ISOLATION_MATCHED_COMPARISON}" == "1" ]]; then
  if [[ ! "${VLLM_MAX_NUM_SEQS}" =~ ^[1-9][0-9]*$ \
        || ! "${VLLM_MAX_NUM_BATCHED_TOKENS}" =~ ^[1-9][0-9]*$ ]] \
      || ((VLLM_MAX_NUM_BATCHED_TOKENS < VLLM_MAX_NUM_SEQS)); then
    echo "[ERROR] matched comparison requires positive max-num-seqs <= max-num-batched-tokens" >&2
    exit 2
  fi
fi
if [[ "${ISOLATION_MATCHED_COMPARISON}" == "1" && "${CUDAGRAPH_MODE}" != "NONE" ]]; then
  # Mirror upstream's default small shapes and add the exact Decode ceiling.
  # Matched default100 captures FULL Decode through100. Explicit10 still gets
  # [1,2,4,8,10,16], so9 pads to FULL10 rather than an uncaptured FULL16.
  # Above-ceiling sizes remain for Native mixed/non-Decode graph coverage.
  capture_limit=$((VLLM_MAX_NUM_SEQS * 2))
  ((capture_limit <= 512)) || capture_limit=512
  ((capture_limit <= VLLM_MAX_NUM_BATCHED_TOKENS)) || capture_limit="${VLLM_MAX_NUM_BATCHED_TOKENS}"
  ((capture_limit >= VLLM_MAX_NUM_SEQS)) || capture_limit="${VLLM_MAX_NUM_SEQS}"
  for ((capture_size = 1; capture_size <= capture_limit; capture_size++)); do
    if ((capture_size == 1 || capture_size == 2 || capture_size == 4 \
        || capture_size == VLLM_MAX_NUM_SEQS \
        || (capture_size >= 8 && capture_size < 256 && capture_size % 8 == 0) \
        || (capture_size >= 256 && capture_size <= 512 && capture_size % 16 == 0))); then
      MATCHED_CUDAGRAPH_CAPTURE_SIZES+=("${capture_size}")
    fi
  done
  VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS} --cudagraph-capture-sizes ${MATCHED_CUDAGRAPH_CAPTURE_SIZES[*]}"
fi

PREFILL_ADMISSION_LABEL=native_budget_fill
if [[ "${HBM_ONLY_VARIANT}" == "1" \
      && "${PREFILL_SCHEDULING_POLICY}" == "fcfs" \
      && "${ISOLATION_MATCHED_COMPARISON}" != "1" ]]; then
  PREFILL_ADMISSION_LABEL=legacy_fcfs_single_request_gate
fi
echo "[PLAN] isolation_matched_comparison=${ISOLATION_MATCHED_COMPARISON}"
echo "[PLAN] guarded_2k=${ISOLATION_GUARDED_2K_COMPARISON} hbm_completion_guard=${VLLM_ISOLATION_HBM_COMPLETION_GUARD} decode_output_first=${VLLM_ISOLATION_DECODE_OUTPUT_FIRST} next_chunk_fit=${VLLM_ISOLATION_PREFILL_FIT_NEXT_CHUNK}"
echo "[PLAN] batch2_pd11=${ISOLATION_BATCH2_PD11_COMPARISON} prefill_batch_max_requests=${VLLM_ISOLATION_PREFILL_BATCH_MAX_REQUESTS} decode_batch_max_requests=${VLLM_ISOLATION_DECODE_BATCH_MAX_REQUESTS} (0=unlimited)"
echo "[PLAN] isolation_memory=${VLLM_ISOLATION_MEMORY_OPTIMIZATIONS} norm=${VLLM_ISOLATION_FUSED_RMS_NORM} workspace=${VLLM_ISOLATION_PREFILL_EXTENDED_WORKSPACE} ring_state=${VLLM_ISOLATION_RING_STATE_REUSE} ring_kernel=${VLLM_ISOLATION_RING_DYNAMIC_KERNEL} decode_buffers=${VLLM_ISOLATION_COMPACT_DECODE_BUFFERS} prefetch=${VLLM_ISOLATION_RING_EARLY_PREFETCH} graph_audit=${VLLM_ISOLATION_GRAPH_MEMORY_AUDIT}"
echo "[PLAN] prefill_workspace_rightsize=${VLLM_ISOLATION_PREFILL_WORKSPACE_RIGHTSIZE} ring_workspace_sharing=${VLLM_ISOLATION_RING_WORKSPACE_SHARING}"
echo "[PLAN] green_context_process_lifetime=${VLLM_ISOLATION_GREEN_CONTEXT_PROCESS_LIFETIME}"
echo "[PLAN] startup_cache_trim=${VLLM_ISOLATION_STARTUP_CACHE_TRIM} startup_memory_audit=${VLLM_ISOLATION_STARTUP_MEMORY_AUDIT} scope=before_capture_only"
if [[ "${VLLM_ISOLATION_PREFILL_WORKSPACE_RIGHTSIZE}" == "1" || "${VLLM_ISOLATION_RING_WORKSPACE_SHARING}" == "1" ]]; then
  if [[ "${PD_PREFILL_WORKSPACE_ENABLED}" != "1" ]]; then
    echo "[ERROR] workspace rightsize/sharing requires PD_PREFILL_WORKSPACE_ENABLED=1" >&2
    exit 2
  fi
fi
if [[ "${VLLM_ISOLATION_RING_WORKSPACE_SHARING}" == "1" && "${VLLM_ISOLATION_RING_STATE_REUSE}" != "1" ]]; then
  echo "[ERROR] Ring workspace sharing requires VLLM_ISOLATION_RING_STATE_REUSE=1" >&2
  exit 2
fi
echo "[PLAN] isolation_execution=${VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS} ring_query=${VLLM_ISOLATION_RING_QUERY_EXPANSION} p1_boundary=${VLLM_ISOLATION_P1_BOUNDARY_FASTPATH} prefill_rope=${VLLM_ISOLATION_PREFILL_FUSED_ROPE} text_inputs=${VLLM_ISOLATION_TEXT_INPUT_BUFFERS} memory_snapshots=${VLLM_ISOLATION_MEMORY_SNAPSHOTS} scratch_bytes=${VLLM_ISOLATION_RING_QUERY_SCRATCH_BYTES}"
echo "[PLAN] isolation_p1_ring_sm_handoff=${VLLM_ISOLATION_P1_RING_SM_HANDOFF} (effective only with P1 parked Prefill)"
echo "[PLAN] isolation_dram_refresh_probe=${VLLM_ISOLATION_DRAM_REFRESH_PROBE} (producer only; consumers remain strict)"
echo "[PLAN] isolation_d2h_exact_plan_progress=${VLLM_ISOLATION_D2H_EXACT_PLAN_PROGRESS} (bounded independent copy owner)"
echo "[PLAN] isolation_native_soft_headroom_progress=${VLLM_ISOLATION_NATIVE_SOFT_HEADROOM_PROGRESS} (Decode-free resident Native only)"
echo "[PLAN] isolation_prefill_batch_kv_budget=${VLLM_ISOLATION_PREFILL_BATCH_KV_BUDGET} (shared batch KV / imminent Decode reserve)"
echo "[PLAN] isolation_logical_release_readiness=${VLLM_ISOLATION_LOGICAL_RELEASE_READINESS} (coverage summary; full release revalidation unchanged)"
echo "[PLAN] isolation_bounded_prefill_planning=${VLLM_ISOLATION_BOUNDED_PREFILL_PLANNING} (includes logical readiness + Decode-output-first; 64-page pre-admission slices; full commit validation unchanged)"
echo "[PLAN] isolation_waiting_decode_reclaim=${VLLM_ISOLATION_WAITING_DECODE_RECLAIM} (proposal-only; whole-request copy-ready transactional reclaim)"
echo "[PLAN] cudagraph_capture_sizes=${MATCHED_CUDAGRAPH_CAPTURE_SIZES[*]:-vllm_default}"
echo "[PLAN] long_context=${LONG_CONTEXT_TOKENS} count=${LONG_REQUEST_COUNT}"
echo "[PLAN] short_context=${SHORT_CONTEXT_TOKENS} count=${SHORT_REQUEST_COUNT}"
echo "[PLAN] workload=${WORKLOAD}"
echo "[PLAN] workload_A_short_offsets=${SHORT_OFFSET_ARRAY[*]}"
echo "[PLAN] workload_B_long_offsets=${LONG_OFFSET_ARRAY[*]}"
echo "[PLAN] workload_C_event_interval=${INTERLEAVED_INJECTION_INTERVAL_SEC}"
echo "[PLAN] submission_mode=${EFFECTIVE_SUBMISSION_MODE}"
if [[ "${WORKLOAD}" == "LOONGSERVE" ]]; then
  echo "[PLAN] loongserve_trace=${LOONGSERVE_TRACE_JSONL}"
  echo "[PLAN] loongserve_arrival_time_scale=${LOONGSERVE_ARRIVAL_TIME_SCALE}"
  echo "[PLAN] loongserve_termination_mode=${LOONGSERVE_TERMINATION_MODE}"\
" request_limit=${LOONGSERVE_REQUEST_LIMIT}"
  echo "[PLAN] loongserve_time_limit_sec=${LOONGSERVE_TIME_LIMIT_SEC} mode=${LOONGSERVE_TIME_LIMIT_MODE}"
fi
echo "[PLAN] system_variant=${SYSTEM_VARIANT}"
if [[ "${SYSTEM_VARIANT}" == "native_40sm_chunked" ]]; then
  echo "[PLAN] execution_profile=native_full_gpu_40sm"
  echo "[PLAN] vllm_source=${VLLM_VENDOR_DIR:-installed_package}"
elif [[ "${SCHEDULER_ONLY_BASELINE}" == "1" ]]; then
  echo "[PLAN] execution_profile=native_full_gpu_40sm_scheduler_only"
  echo "[PLAN] vllm_source=${VLLM_VENDOR_DIR}"
  echo "[PLAN] scheduler_only_contract=prefill_srpt_decode_fcfs quantum_tokens=${BASELINE_PLUS_PREFILL_QUANTUM_TOKENS}"
  echo "[PLAN] scheduler_only_disabled=pd_partition,dynamic_sm,greencontext,context_ring,victim,layerwise"
else
  echo "[PLAN] initial_split=${INITIAL_GREEN_SM_SPLIT}"
  echo "[PLAN] phase_future_depths="\
"${PD_PREFILL_INFLIGHT_DEPTH}/${PD_DECODE_INFLIGHT_DEPTH}/${PD_GLOBAL_INFLIGHT_DEPTH} (P/D/G)"
  echo "[PLAN] prefill_chain_max_lookahead_tokens=${PD_PREFILL_CHAIN_MAX_LOOKAHEAD_TOKENS}"
fi
echo "[PLAN] dynamic_policy=${PD_DYNAMIC_SM_POLICY}"
if [[ "${PD_DYNAMIC_SM_POLICY}" == "tbt_slo_predictive" ]]; then
  echo "[PLAN] tbt_predictive_applicable=$((1 - NATIVE_FULL_GPU_VARIANT)) lead_ms=${TBT_PREDICTIVE_LEAD_MS} cutover_ms=${TBT_PREDICTIVE_CUTOVER_MS} min_step_ms=${TBT_PREDICTIVE_MIN_STEP_MS} calibration_required=1"
fi
echo "[PLAN] serialized_full_sm=${PD_SERIALIZED_FULL_SM}"
echo "[PLAN] serialized_full_sm_scope=capacity_blocked_prefill_only normal_solo_full_sm=1"
echo "[PLAN] layerwise_profile_handoff=${PD_LAYERWISE_PROFILE_HANDOFF}"
echo "[PLAN] ttft_layerwise_prefill_execution=${TTFT_LAYERWISE_PREFILL_EXECUTION} preemption=${TTFT_LAYERWISE_PREFILL_PREEMPTION} group_layers=${TTFT_LAYERWISE_PREFILL_GROUP_LAYERS} scheduler_yield=${TTFT_LAYERWISE_PREFILL_PREEMPTION}"
echo "[PLAN] kv_layerwise_checkpoint_plan=${KV_LAYERWISE_CHECKPOINT_PLAN} kv_layerwise_offload=${KV_LAYERWISE_OFFLOAD}"
echo "[PLAN] kv_p1_parked=${KV_P1_PARKED} work_binding=${KV_PREFILL_WORK_BINDING} resume_frame=${KV_PREFILL_RESUME_FRAME} parked_batches=${KV_PARKED_MAX_BATCHES} parked_gpu_bytes=${KV_PARKED_GPU_BYTES}"
echo "[PLAN] kv_p1_hbm_only=${KV_P1_HBM_ONLY}"
echo "[PLAN] prefill_workspace=${PD_PREFILL_WORKSPACE_ENABLED} max_tokens=${PD_PREFILL_WORKSPACE_MAX_TOKENS} slots=${PD_PREFILL_WORKSPACE_SLOTS} strict=${PD_PREFILL_WORKSPACE_STRICT}"
echo "[PLAN] prefill_selection_quantum_tokens=${PREFILL_SELECTOR_QUANTUM_TOKENS} native_prefill_max_tokens=${NATIVE_PREFILL_MAX_TOKENS}"
echo "[PLAN] prefill_admission=${PREFILL_ADMISSION_LABEL} max_num_seqs=${VLLM_MAX_NUM_SEQS} max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS} long_threshold=${VLLM_LONG_PREFILL_TOKEN_THRESHOLD}"
echo "[PLAN] instance_lifecycle=setsid_group_term_kill_then_port_release"
echo "[PLAN] server_shutdown_timeout_sec=${SERVER_SHUTDOWN_TIMEOUT_SEC} port_release_timeout_sec=${PORT_RELEASE_TIMEOUT_SEC}"
echo "[PLAN] phase_sync_sample_output=${PD_PHASE_SYNC_SAMPLE_OUTPUT}"
echo "[PLAN] dynamic_profiles=${PD_SM_PROFILES}"
echo "[PLAN] decode_first_host_enqueue=${DECODE_FIRST_HOST_ENQUEUE}"
echo "[PLAN] ring_batched_partial=${RING_BATCHED_PARTIAL} strict=${RING_BATCHED_PARTIAL_STRICT} workspace_bytes=${RING_BATCHED_WORKSPACE_BYTES}"
echo "[PLAN] ring_streaming_wave_groups=${RING_STREAMING_WAVE_GROUPS} wave_tokens=$((RING_COMPUTE_GROUP_TOKENS * RING_STREAMING_WAVE_GROUPS))"
echo "[PLAN] ring_prefill_query_tokens=${PREFILL_CHUNK_TOKENS}..${RING_BW_MAX_TOKENS} active_decode_native_prefill_tokens=${ACTIVE_DECODE_NATIVE_PREFILL_TOKENS} active_decode_ring_prefill_tokens=${ACTIVE_DECODE_RING_PREFILL_TOKENS} elastic_active_decode=${ACTIVE_DECODE_ELASTIC_PREFILL} dynamic=${RING_BW_DYNAMIC} historical_group_tokens=${RING_COMPUTE_GROUP_TOKENS}"
if [[ "${SYSTEM_VARIANT}" == "full_vendor_v8" \
      && "${RING_BATCHED_PARTIAL}" != "1" \
      && "${RING_STREAMING_WAVE_GROUPS}" == "1" ]]; then
  echo "[PLAN] ring_execution=historical_group_streaming_double_buffered"
fi
if [[ "${SYSTEM_VARIANT}" == "full_vendor_v8" \
      && "${VERIFY_TIERED_DIRECT_OPS}" == "1" \
      && "${DRY_RUN}" != "1" ]]; then
  if [[ ! -x "${VLLM_PYTHON_EXECUTABLE}" ]]; then
    echo "[ERROR] direct-op verifier Python is not executable: "\
"${VLLM_PYTHON_EXECUTABLE}" >&2
    exit 2
  fi
  echo "[PLAN] verifying tiered direct ops in deployed vllm._C"
  env PYTHONPATH="${VLLM_VENDOR_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${VLLM_PYTHON_EXECUTABLE}" \
    "${ROOT_DIR}/tools/virtual_context_smoke/verify_tiered_direct_ops.py" \
    --vendor-dir "${VLLM_VENDOR_DIR}"
elif [[ "${SYSTEM_VARIANT}" == "full_vendor_v8" \
        && "${VERIFY_TIERED_DIRECT_OPS}" == "1" ]]; then
  echo "[PLAN] tiered direct-op verification deferred until real execution"
fi
echo "[PLAN] full_iteration_trace=${ENABLE_FULL_ITERATION_TRACE} sample_every=${ITERATION_TRACE_SAMPLE_EVERY_N}"
echo "[PLAN] lightweight_timeline_trace=${ENABLE_LIGHTWEIGHT_TIMELINE_TRACE}"
echo "[PLAN] scheduler_timing_trace=${ENABLE_SCHEDULER_TIMING_TRACE} sample_every=${ITERATION_TRACE_SAMPLE_EVERY_N} path=${SCHEDULER_METRICS:-disabled}"
echo "[PLAN] strict_bounded_lightweight_timeline=${STRICT_BOUNDED_LIGHTWEIGHT_TIMELINE}"
echo "[PLAN] virtual_context_timing=${VIRTUAL_CONTEXT_TIMING}"
echo "[PLAN] pd_executor_debug=${ENABLE_PD_EXECUTOR_DEBUG}"
echo "[PLAN] cudagraph_mode=${CUDAGRAPH_MODE:-vllm_default}"
echo "[PLAN] matched_graph_preserves_compile=${ISOLATION_MATCHED_COMPARISON}"
if [[ "${NATIVE_FULL_GPU_VARIANT}" == "1" ]]; then
  echo "[PLAN] cudagraph_path=upstream"
elif [[ "${PD_GREEN_DECODE_CUDAGRAPH}" == "1" ]]; then
  echo "[PLAN] cudagraph_path=greencontext_partitioned_decode"
else
  echo "[PLAN] cudagraph_path=disabled_or_eager"
fi
echo "[PLAN] green_decode_cudagraph=${PD_GREEN_DECODE_CUDAGRAPH}"
echo "[PLAN] throughput_optimizations_requested=${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS} effective=${EFFECTIVE_FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS}"
echo "[PLAN] fused_execute_sample=${PD_FUSED_EXECUTE_SAMPLE}"
echo "[PLAN] ring_bw=${PREFILL_CHUNK_TOKENS}..${RING_BW_MAX_TOKENS} dynamic=${RING_BW_DYNAMIC} policy=${RING_BW_POLICY}"
echo "[PLAN] pd_cuda_event_trace=${ENABLE_PD_CUDA_EVENT_TRACE}"
echo "[PLAN] pd_nvtx_phase_trace=${ENABLE_PD_NVTX_PHASE_TRACE}"
echo "[PLAN] request_slo=${REQUEST_SLO_POLICY}"
echo "[PLAN] prefill_scheduling_policy=${PREFILL_SCHEDULING_POLICY}"
echo "[PLAN] pd_eval_policy=${PD_EVAL_POLICY} hbm_only=${HBM_ONLY_VARIANT}"
echo "[PLAN] server_base_extra_args=${VLLM_EXTRA_ARGS}"
echo "[PLAN] residency_victim_policy=${FULL_VENDOR_V8_RESIDENCY_VICTIM_POLICY}"
echo "[PLAN] scheduled_prefill_fit_capacity_fraction=${FULL_VENDOR_V8_SCHEDULED_PREFILL_FIT_CAPACITY_FRACTION}"
echo "[PLAN] decode_continuation_rolling_reclaim=${DECODE_CONTINUATION_ROLLING_RECLAIM}"
echo "[PLAN] native_prefill_capacity_mode=${NATIVE_PREFILL_CAPACITY_MODE}"
if [[ "${SYSTEM_VARIANT}" == "full_vendor_v8" \
      && "${PREFILL_SCHEDULING_POLICY}" == "slo_srpt" ]]; then
  echo "[PLAN] prefill_profile=${PREFILL_SERVICE_PROFILE}"
else
  echo "[PLAN] prefill_profile=disabled"
fi
if [[ -n "${ACTIVE_TBT_OVERLAP_PROFILE}" ]]; then
  echo "[PLAN] tbt_profile=${ACTIVE_TBT_OVERLAP_PROFILE}"
else
  echo "[PLAN] tbt_profile=disabled_for_${PD_DYNAMIC_SM_POLICY}"
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY-RUN] no files, server, or GPU state will be changed"
  exit 0
fi

ensure_file() {
  [[ -f "$1" ]] || {
    echo "[ERROR] missing $2: $1" >&2
    exit 1
  }
}

if [[ -e "${RESULT_DIR}" || -e "${LOG_DIR}" || -e "${ARTIFACT_DIR}" ]]; then
  echo "[ERROR] output exists; choose a new EXPERIMENT_NAME" >&2
  exit 1
fi

FIXED_WORKLOAD_MEASURER="${ROOT_DIR}/measure_vendor_v8_short_validation.py"
LOONGSERVE_MEASURER="${ROOT_DIR}/measure_loongserve_mixed_workload.py"
if [[ "${WORKLOAD}" == "LOONGSERVE" ]]; then
  MEASURER="${LOONGSERVE_MEASURER}"
else
  MEASURER="${FIXED_WORKLOAD_MEASURER}"
fi
DATASET_PREPARER="${ROOT_DIR}/prepare_mixed_context_sharegpt.py"
SERVER_STARTER="${ROOT_DIR}/start_vllm_causality_microbench.sh"
POLICY_HELPER="${ROOT_DIR}/tools/pd_eval_policy_env.sh"
for path in \
  "${REQUEST_SLO_POLICY}" \
  "${MEASURER}" "${SERVER_STARTER}" "${POLICY_HELPER}"; do
  ensure_file "${path}" "required input"
done
if [[ "${WORKLOAD}" == "LOONGSERVE" ]]; then
  ensure_file "${LOONGSERVE_TRACE_JSONL}" "LoongServe Mixed trace"
else
  ensure_file "${DATASET_PREPARER}" "dataset preparer"
  ensure_file "${SHAREGPT_RAW_PATH}" "ShareGPT source"
fi
if [[ "${SYSTEM_VARIANT}" == "full_vendor_v8" \
      && "${PREFILL_SCHEDULING_POLICY}" == "slo_srpt" ]]; then
  ensure_file "${PREFILL_SERVICE_PROFILE}" "Prefill service profile"
fi
case "${PD_DYNAMIC_SM_POLICY}" in
  tbt|tbt_profiled|tbt_slo_profiled)
    ensure_file "${ACTIVE_TBT_OVERLAP_PROFILE}" "TBT overlap profile"
    ;;
  tbt_slo_predictive)
    if [[ "${NATIVE_FULL_GPU_VARIANT}" == "0" ]]; then
      ensure_file "${ACTIVE_TBT_OVERLAP_PROFILE}" "predictive TBT overlap profile"
    fi
    ;;
esac
[[ -x "${VLLM_EXECUTABLE}" ]] || {
  echo "[ERROR] vLLM executable is not executable: ${VLLM_EXECUTABLE}" >&2
  exit 1
}

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${PD_DYNAMIC_SM_POLICY}" == "tbt_slo_predictive" \
      && "${NATIVE_FULL_GPU_VARIANT}" == "0" ]]; then
  # Validate before loading a model/GPU. Threshold-only SLO JSON is not an
  # overlap predictor, and missing calibration must not silently enable one.
  if ! "${PYTHON_BIN}" - "${TBT_PREDICTIVE_LEAD_MS}" \
      "${TBT_PREDICTIVE_CUTOVER_MS}" "${TBT_PREDICTIVE_MIN_STEP_MS}" \
      "${ACTIVE_TBT_OVERLAP_PROFILE}" \
      "${VLLM_VENDOR_DIR}/vllm/v1/engine/pd_tbt_profile.py" <<'PY'
import math
import runpy
import sys
for name, text, allow_zero in zip(("LEAD_MS", "CUTOVER_MS", "MIN_STEP_MS"), sys.argv[1:4], (False, True, False)):
    try:
        value = float(text)
    except ValueError:
        raise SystemExit(f"[ERROR] VLLM_PD_TBT_PREDICTIVE_{name} must be calibrated milliseconds")
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise SystemExit(f"[ERROR] VLLM_PD_TBT_PREDICTIVE_{name} is uncalibrated/invalid: {text}")
profile_module = runpy.run_path(sys.argv[5])
profile_module["TBTOverlapProfile"].from_json_file(sys.argv[4])
print("[PLAN] tbt_predictive_profile_contract=validated calibration=explicit_not_a_guarantee")
PY
  then
    exit 2
  fi
fi

if ((10#${VLLM_ISOLATION_PREFILL_BATCH_MAX_REQUESTS} > 0 \
      || 10#${VLLM_ISOLATION_DECODE_BATCH_MAX_REQUESTS} > 0)); then
  PHASE_BATCH_LIMIT_PROBE='import ast, hashlib, json, pathlib, vllm
root = pathlib.Path(vllm.__file__).resolve().parent / "v1/core/sched"
scheduler = root / "scheduler.py"
frontier = root / "prefill_chunk_scheduler.py"
source = scheduler.read_text()
tree = ast.parse(source)
calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
if not all(x in calls for x in ("self._configure_isolation_phase_batch_limits", "self._isolation_phase_request_limit")):
    raise SystemExit("missing phase batch request limits in " + str(scheduler))
frontier_source = frontier.read_text()
functions = [n for n in ast.parse(frontier_source).body if isinstance(n, ast.FunctionDef)]
if not any(n.name == "build_native_prefill_admission_frontier" and any(a.arg == "request_limit" for a in n.args.kwonlyargs) for n in functions):
    raise SystemExit("missing request_limit in SRPT frontier: " + str(frontier))
print(json.dumps({"scheduler": str(scheduler), "frontier": str(frontier),
                 "scheduler_sha256": hashlib.sha256(source.encode()).hexdigest(),
                 "frontier_sha256": hashlib.sha256(frontier_source.encode()).hexdigest()}))'
  PHASE_BATCH_LIMIT_RESULT=$(PYTHONPATH="${VLLM_VENDOR_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${VLLM_PYTHON_EXECUTABLE}" -c "${PHASE_BATCH_LIMIT_PROBE}") || exit 2
  echo "[PLAN] phase_batch_limit_deployment=${PHASE_BATCH_LIMIT_RESULT}"
fi

if [[ "${VLLM_ISOLATION_STARTUP_CACHE_TRIM}" == "1" \
      || "${VLLM_ISOLATION_STARTUP_MEMORY_AUDIT}" == "1" ]]; then
  STARTUP_MEMORY_PROBE='import json, pathlib, vllm
from vllm.isolation_startup_memory import STARTUP_MEMORY_CONTRACT_VERSION, validate_startup_memory_deployment
if STARTUP_MEMORY_CONTRACT_VERSION != 1:
    raise SystemExit("unsupported startup memory contract")
print(json.dumps(validate_startup_memory_deployment(pathlib.Path(vllm.__file__).resolve().parent), sort_keys=True))'
  if [[ -n "${VLLM_VENDOR_DIR}" ]]; then
    STARTUP_MEMORY_RESULT=$(PYTHONPATH="${VLLM_VENDOR_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${VLLM_PYTHON_EXECUTABLE}" -c "${STARTUP_MEMORY_PROBE}") || exit 2
  else
    STARTUP_MEMORY_RESULT=$("${VLLM_PYTHON_EXECUTABLE}" -c "${STARTUP_MEMORY_PROBE}") || exit 2
  fi
  echo "[PLAN] startup_memory_deployment=${STARTUP_MEMORY_RESULT}"
fi

if [[ "${ENABLE_SCHEDULER_TIMING_TRACE}" == "1" ]]; then
  # Check the actual server import, especially when Native uses an installed
  # package instead of vendor_v8. No silent missing-Baseline measurements.
  SCHEDULER_TIMING_PROBE='import ast, hashlib, json, pathlib, vllm
root = pathlib.Path(vllm.__file__).resolve().parent / "v1/core/sched"
scheduler = root / "scheduler.py"
helper = root / "isolation_scheduler_timing.py"
if not helper.is_file():
    raise SystemExit("missing common scheduler timing helper: " + str(helper))
tree = ast.parse(scheduler.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
for method, target in (("__init__", "install_scheduler_timing"), ("shutdown", "close_scheduler_timing")):
    calls = [ast.unparse(n.func) for n in ast.walk(methods[method]) if isinstance(n, ast.Call)]
    if target not in calls:
        raise SystemExit("scheduler timing not wired into " + method + ": " + str(scheduler))
print(json.dumps({"scheduler": str(scheduler), "helper": str(helper),
                 "scheduler_sha256": hashlib.sha256(scheduler.read_bytes()).hexdigest(),
                 "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest()}))'
  if [[ -n "${VLLM_VENDOR_DIR}" ]]; then
    SCHEDULER_TIMING_RESULT=$(PYTHONPATH="${VLLM_VENDOR_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${VLLM_PYTHON_EXECUTABLE}" -c "${SCHEDULER_TIMING_PROBE}") || exit 2
  else
    SCHEDULER_TIMING_RESULT=$("${VLLM_PYTHON_EXECUTABLE}" -c "${SCHEDULER_TIMING_PROBE}") || exit 2
  fi
  echo "[PLAN] scheduler_timing_deployment=${SCHEDULER_TIMING_RESULT}"
fi

if [[ "${VLLM_ISOLATION_HBM_COMPLETION_GUARD}" == "1" ]]; then
  # Probe the package used by the server interpreter, not just this checkout.
  # An unpatched installed Native must not silently become an unguarded case.
  HBM_GUARD_PROBE='import ast, hashlib, json, pathlib, vllm
root = pathlib.Path(vllm.__file__).resolve().parent
scheduler = root / "v1/core/sched/scheduler.py"
helper = root / "v1/core/sched/isolation_hbm_guard.py"
if not helper.is_file():
    raise SystemExit("missing HBM completion guard helper in " + str(root))
source = scheduler.read_text()
tree = ast.parse(source)
calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
if not all(x in calls for x in (
    "make_isolation_hbm_completion_guard", "self.isolation_hbm_guard.can_admit")):
    raise SystemExit("HBM guard is not wired into imported Scheduler: " + str(scheduler))
print(json.dumps({"scheduler": str(scheduler), "helper": str(helper),
                 "scheduler_sha256": hashlib.sha256(source.encode()).hexdigest(),
                 "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest()}))'
  if [[ -n "${VLLM_VENDOR_DIR}" ]]; then
    HBM_GUARD_RESULT=$(PYTHONPATH="${VLLM_VENDOR_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${VLLM_PYTHON_EXECUTABLE}" -c "${HBM_GUARD_PROBE}") || exit 2
  else
    HBM_GUARD_RESULT=$("${VLLM_PYTHON_EXECUTABLE}" -c "${HBM_GUARD_PROBE}") || exit 2
  fi
  echo "[PLAN] hbm_guard_deployment=${HBM_GUARD_RESULT}"
fi

if [[ "${WORKLOAD}" == "LOONGSERVE" ]]; then
  TOKEN_ID_CONTRACT_PROBE='import inspect, json, vllm
from vllm.entrypoints.openai.protocol import CompletionRequest, CompletionResponseStreamChoice
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
def fields(model):
    model_fields = getattr(model, "model_fields", None)
    if model_fields is not None:
        return model_fields
    return getattr(model, "__fields__", {})
request_field = "return_token_ids" in fields(CompletionRequest)
prompt_request_field = "return_prompt_token_ids" in fields(CompletionRequest)
response_field = "token_ids" in fields(CompletionResponseStreamChoice)
prompt_response_field = "prompt_token_ids" in fields(CompletionResponseStreamChoice)
serving_source = inspect.getsource(OpenAIServingCompletion)
serving_path = all(fragment in serving_source for fragment in (
    "request.return_token_ids", "request.return_prompt_token_ids"))
legacy_request = CompletionRequest(prompt="contract-probe", return_token_ids=True)
optout_request = CompletionRequest(
    prompt="contract-probe",
    return_token_ids=True,
    return_prompt_token_ids=False,
)
legacy_default = legacy_request.return_prompt_token_ids is None
optout_value = optout_request.return_prompt_token_ids is False
contract = {
    "vllm_file": vllm.__file__,
    "vllm_version": getattr(vllm, "__version__", "unknown"),
    "request_field": request_field,
    "prompt_request_field": prompt_request_field,
    "response_field": response_field,
    "prompt_response_field": prompt_response_field,
    "serving_path": serving_path,
    "legacy_default": legacy_default,
    "prompt_optout_value": optout_value,
}
print(json.dumps(contract, sort_keys=True))
raise SystemExit(
    0 if all((request_field, prompt_request_field, response_field,
              prompt_response_field, serving_path, legacy_default,
              optout_value)) else 3)'
  if [[ -n "${VLLM_VENDOR_DIR}" ]]; then
    if ! TOKEN_ID_CONTRACT_RESULT=$(env \
      PYTHONPATH="${VLLM_VENDOR_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" -c "${TOKEN_ID_CONTRACT_PROBE}"); then
      echo "[ERROR] selected vLLM source does not implement the exact "\
"return_token_ids + prompt-ID opt-out streaming contract: "\
"${VLLM_VENDOR_DIR}" >&2
      exit 2
    fi
  elif ! TOKEN_ID_CONTRACT_RESULT=$(
    "${PYTHON_BIN}" -c "${TOKEN_ID_CONTRACT_PROBE}"
  ); then
    echo "[ERROR] installed vLLM does not implement the exact "\
"return_token_ids + prompt-ID opt-out streaming contract" >&2
    exit 2
  fi
  echo "[PLAN] loongserve_token_id_contract=${TOKEN_ID_CONTRACT_RESULT}"
fi

if [[ "${WORKLOAD}" == "LOONGSERVE" ]]; then
  "${PYTHON_BIN}" - "${LOONGSERVE_TRACE_JSONL}" \
    "${VLLM_MAX_MODEL_LEN}" "${LOONGSERVE_TERMINATION_MODE}" \
    "${LOONGSERVE_REQUEST_LIMIT}" <<'PY'
import json
import sys

path, limit_text, termination_mode, request_limit_text = sys.argv[1:]
limit = int(limit_text)
request_limit = int(request_limit_text)
count = 0
maximum = 0
with open(path, encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        total = int(row["input_tokens"]) + int(row["max_tokens"])
        if total > limit:
            raise SystemExit(
                f"trace line {line_number} needs {total} tokens, "
                f"above VLLM_MAX_MODEL_LEN={limit}"
            )
        count += 1
        maximum = max(maximum, total)
if count == 0:
    raise SystemExit("LoongServe trace is empty")
if termination_mode == "requests" and request_limit > count:
    raise SystemExit(
        f"LOONGSERVE_REQUEST_LIMIT={request_limit} exceeds trace rows={count}"
    )
selected = request_limit if termination_mode == "requests" else "time-controlled"
print(
    f"[PLAN] loongserve_trace_requests={count} "
    f"selected_requests={selected} max_total_tokens={maximum}"
)
PY
fi

PREFILL_FALLBACK_MS_PER_TOKEN="$(
  "${PYTHON_BIN}" - "${REQUEST_SLO_POLICY}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(payload["prefill_fallback_ms_per_token"])
PY
)"
TBT_DEFAULT_SLO_MS="$(
  "${PYTHON_BIN}" - "${REQUEST_SLO_POLICY}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(max(float(row["tbt_slo_ms"]) for row in payload["buckets"]))
PY
)"

mkdir -p "${RESULT_DIR}/experiment_snapshot/scripts" \
  "${LOG_DIR}" "${ARTIFACT_DIR}"
for memory_feature in MEMORY_OPTIMIZATIONS "${ISOLATION_MEMORY_FEATURES[@]}" STARTUP_CACHE_TRIM STARTUP_MEMORY_AUDIT; do
  memory_flag="VLLM_ISOLATION_${memory_feature}"
  printf '%s=%s\n' "${memory_flag}" "${!memory_flag}"
done >"${RESULT_DIR}/experiment_snapshot/isolation_memory.txt"
for execution_feature in EXECUTION_OPTIMIZATIONS "${ISOLATION_EXECUTION_FEATURES[@]}" RING_QUERY_SCRATCH_BYTES; do
  execution_flag="VLLM_ISOLATION_${execution_feature}"
  printf '%s=%s\n' "${execution_flag}" "${!execution_flag}"
done >"${RESULT_DIR}/experiment_snapshot/isolation_execution.txt"
if [[ "${ISOLATION_MATCHED_COMPARISON}" == "1" ]]; then
  # Write the resolved contract BEFORE model startup/replay. The historical
  # success-only settings.txt remains unchanged; crashes still leave this
  # evidence and the script snapshots for comparison/configuration audits.
  printf '%s\n' \
    "isolation_matched_comparison=1" \
    "isolation_guarded_2k_comparison=${ISOLATION_GUARDED_2K_COMPARISON}" \
    "isolation_batch2_pd11_comparison=${ISOLATION_BATCH2_PD11_COMPARISON}" \
    "prefill_batch_max_requests=${VLLM_ISOLATION_PREFILL_BATCH_MAX_REQUESTS}" \
    "decode_batch_max_requests=${VLLM_ISOLATION_DECODE_BATCH_MAX_REQUESTS}" \
    "phase_batch_limit_deployment=${PHASE_BATCH_LIMIT_RESULT:-disabled}" \
    "hbm_completion_guard=${VLLM_ISOLATION_HBM_COMPLETION_GUARD}" \
    "hbm_guard_deployment=${HBM_GUARD_RESULT:-disabled}" \
    "startup_cache_trim=${VLLM_ISOLATION_STARTUP_CACHE_TRIM}" \
    "startup_memory_audit=${VLLM_ISOLATION_STARTUP_MEMORY_AUDIT}" \
    "startup_memory_deployment=${STARTUP_MEMORY_RESULT:-disabled}" \
    "decode_output_first=${VLLM_ISOLATION_DECODE_OUTPUT_FIRST}" \
    "prefill_fit_next_chunk=${VLLM_ISOLATION_PREFILL_FIT_NEXT_CHUNK}" \
    "system_variant=${SYSTEM_VARIANT}" \
    "model_name=${MODEL_NAME}" \
    "kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES}" \
    "max_model_len=${VLLM_MAX_MODEL_LEN}" \
    "max_num_seqs=${VLLM_MAX_NUM_SEQS}" \
    "max_num_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}" \
    "long_prefill_token_threshold=${VLLM_LONG_PREFILL_TOKEN_THRESHOLD}" \
    "loongserve_trace_jsonl=${LOONGSERVE_TRACE_JSONL}" \
    "loongserve_termination_mode=${LOONGSERVE_TERMINATION_MODE}" \
    "loongserve_request_limit=${LOONGSERVE_REQUEST_LIMIT}" \
    "loongserve_arrival_time_scale=${LOONGSERVE_ARRIVAL_TIME_SCALE}" \
    "phase_depths=${PD_PREFILL_INFLIGHT_DEPTH}/${PD_DECODE_INFLIGHT_DEPTH}/${PD_GLOBAL_INFLIGHT_DEPTH}" \
    "phase_depths_applicable=$((1 - NATIVE_FULL_GPU_VARIANT))" \
    "dynamic_sm_policy=${PD_DYNAMIC_SM_POLICY}" \
    "tbt_predictive_lead_ms=${TBT_PREDICTIVE_LEAD_MS}" \
    "tbt_predictive_cutover_ms=${TBT_PREDICTIVE_CUTOVER_MS}" \
    "tbt_predictive_min_step_ms=${TBT_PREDICTIVE_MIN_STEP_MS}" \
    "layerwise_profile_handoff=${PD_LAYERWISE_PROFILE_HANDOFF}" \
    "layerwise_request_preemption=${TTFT_LAYERWISE_PREFILL_PREEMPTION}" \
    "layerwise_kv_offload=${KV_LAYERWISE_OFFLOAD}" \
    "kv_p1_parked=${KV_P1_PARKED}" \
    "kv_p1_hbm_only=${KV_P1_HBM_ONLY}" \
    "parked_gpu_bytes=${KV_PARKED_GPU_BYTES}" \
    "prefill_admission=${PREFILL_ADMISSION_LABEL}" \
    "native_prefill_max_tokens=${NATIVE_PREFILL_MAX_TOKENS}" \
    "prefill_workspace=${PD_PREFILL_WORKSPACE_ENABLED}" \
    "prefill_workspace_slots=${PD_PREFILL_WORKSPACE_SLOTS}" \
    "iteration_trace_sample_every_n=${ITERATION_TRACE_SAMPLE_EVERY_N}" \
    "scheduler_timing_trace=${ENABLE_SCHEDULER_TIMING_TRACE}" \
    "scheduler_timing_deployment=${SCHEDULER_TIMING_RESULT:-disabled}" \
    "strict_bounded_lightweight_timeline=${STRICT_BOUNDED_LIGHTWEIGHT_TIMELINE}" \
    "virtual_context_timing=${VIRTUAL_CONTEXT_TIMING}" \
    "fused_execute_sample=${PD_FUSED_EXECUTE_SAMPLE}" \
    "cudagraph_mode=${CUDAGRAPH_MODE}" \
    "matched_graph_preserves_compile=1" \
    "cudagraph_capture_sizes=${MATCHED_CUDAGRAPH_CAPTURE_SIZES[*]}" \
    "green_decode_cudagraph=${PD_GREEN_DECODE_CUDAGRAPH}" \
    "server_base_extra_args=${VLLM_EXTRA_ARGS}" \
    >"${RESULT_DIR}/experiment_snapshot/matched_comparison.txt"
fi
cp -a "${BASH_SOURCE[0]}" "${MEASURER}" \
  "${SERVER_STARTER}" "${POLICY_HELPER}" \
  "${RESULT_DIR}/experiment_snapshot/scripts/"
if [[ "${WORKLOAD}" != "LOONGSERVE" ]]; then
  cp -a "${DATASET_PREPARER}" \
    "${RESULT_DIR}/experiment_snapshot/scripts/"
elif [[ -f "${LOONGSERVE_TRACE_JSONL}.manifest.json" ]]; then
  cp -a "${LOONGSERVE_TRACE_JSONL}.manifest.json" \
    "${RESULT_DIR}/experiment_snapshot/loongserve_trace.manifest.json"
fi
PAIR_WRAPPER="${ROOT_DIR}/run_vendor_v8_dynamic_hbm_vs_full.sh"
if [[ -f "${PAIR_WRAPPER}" ]]; then
  cp -a "${PAIR_WRAPPER}" "${RESULT_DIR}/experiment_snapshot/scripts/"
fi
cp -a "${REQUEST_SLO_POLICY}" \
  "${RESULT_DIR}/experiment_snapshot/request_slo_policy.json"
if [[ "${SYSTEM_VARIANT}" == "full_vendor_v8" \
      && "${PREFILL_SCHEDULING_POLICY}" == "slo_srpt" ]]; then
  cp -a "${PREFILL_SERVICE_PROFILE}" \
    "${RESULT_DIR}/experiment_snapshot/prefill_service_profile.json"
fi
if [[ -n "${ACTIVE_TBT_OVERLAP_PROFILE}" \
      && -f "${ACTIVE_TBT_OVERLAP_PROFILE}" ]]; then
  cp -a "${ACTIVE_TBT_OVERLAP_PROFILE}" \
    "${RESULT_DIR}/experiment_snapshot/tbt_overlap_profile.json"
fi

TARGETS=()
case "${WORKLOAD}" in
  A)
    for ((cycle = 0; cycle < LONG_REQUEST_COUNT; cycle++)); do
      TARGETS+=("${LONG_CONTEXT_TOKENS}")
    done
    for ((index = 0; index < SHORT_REQUEST_COUNT; index++)); do
      TARGETS+=("${SHORT_CONTEXT_TOKENS}")
    done
    ;;
  B)
    for ((index = 0; index < SHORT_REQUEST_COUNT; index++)); do
      TARGETS+=("${SHORT_CONTEXT_TOKENS}")
    done
    for ((cycle = 0; cycle < LONG_REQUEST_COUNT; cycle++)); do
      TARGETS+=("${LONG_CONTEXT_TOKENS}")
    done
    ;;
  C)
    cycle=0
    index=0
    while ((cycle < LONG_REQUEST_COUNT || index < SHORT_REQUEST_COUNT)); do
      if ((index < SHORT_REQUEST_COUNT)); then
        TARGETS+=("${SHORT_CONTEXT_TOKENS}")
        ((index += 1))
      fi
      if ((cycle < LONG_REQUEST_COUNT)); then
        TARGETS+=("${LONG_CONTEXT_TOKENS}")
        ((cycle += 1))
      fi
    done
    ;;
  LOONGSERVE)
    # Request contents and open-loop offsets are owned by the immutable trace.
    ;;
  legacy)
    if [[ "${SUBMISSION_MODE}" == "concurrent_longs" ]]; then
      for ((cycle = 0; cycle < LONG_REQUEST_COUNT; cycle++)); do
        TARGETS+=("${LONG_CONTEXT_TOKENS}")
      done
      for ((index = 0; index < SHORT_REQUEST_COUNT; index++)); do
        TARGETS+=("${SHORT_CONTEXT_TOKENS}")
      done
    else
      for ((cycle = 0; cycle < LONG_REQUEST_COUNT; cycle++)); do
        TARGETS+=("${LONG_CONTEXT_TOKENS}")
        for ((index = 0; index < SHORT_REQUESTS_PER_LONG; index++)); do
          TARGETS+=("${SHORT_CONTEXT_TOKENS}")
        done
      done
    fi
    ;;
esac
if [[ "${WORKLOAD}" != "LOONGSERVE" && "${PREPARE_DATASET}" == "1" ]]; then
  "${PYTHON_BIN}" "${DATASET_PREPARER}" \
    --sharegpt-path "${SHAREGPT_RAW_PATH}" \
    --target-tokens "${TARGETS[@]}" \
    --output-path "${DATASET_JSONL}" \
    --tokenizer "${DATASET_TOKENIZER}" \
    --candidate-pool-size "${SHAREGPT_CANDIDATE_POOL_SIZE}" \
    --underlength-policy "${SHAREGPT_UNDERLENGTH_POLICY}" \
    --max-tokens "$(
      ((LONG_OUTPUT_TOKENS > SHORT_OUTPUT_TOKENS)) \
        && echo "${LONG_OUTPUT_TOKENS}" || echo "${SHORT_OUTPUT_TOKENS}"
    )" \
    --min-tokens 1
fi
ensure_file "${DATASET_JSONL}" "validation dataset"

BRIDGE_LIB="${VLLM_VENDOR_DIR}/vllm/model_executor/layers/green_context_lib/libvllm_green_context.so"
if [[ "${NATIVE_FULL_GPU_VARIANT}" != "1" \
      && ("${BUILD_GREEN_BRIDGE}" == "1" \
          || ("${BUILD_GREEN_BRIDGE}" == "auto" && ! -f "${BRIDGE_LIB}")) ]]; then
  BRIDGE_BUILDER="${ROOT_DIR}/tools/green_context_smoke/build_vendor_v8_bridge.sh"
  ensure_file "${BRIDGE_BUILDER}" "vendor_v8 GreenContext builder"
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" \
    bash "${BRIDGE_BUILDER}"
fi

wait_for_server_ready() {
  local pid="$1"
  "${PYTHON_BIN}" - "${BASE_URL}" "${SERVER_READY_TIMEOUT_SEC}" "${pid}" <<'PY'
import os
import sys
import time
import requests

base_url, timeout_text, pid_text = sys.argv[1:]
deadline = time.time() + int(timeout_text)
pid = int(pid_text)
while time.time() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit("vLLM exited before readiness")
    try:
        if requests.get(base_url + "/v1/models", timeout=5).ok:
            raise SystemExit(0)
    except requests.RequestException:
        pass
    time.sleep(2)
raise SystemExit("timed out waiting for vLLM")
PY
}

cleanup_pid() {
  local pid="${1:-}"
  # EXIT can run after the explicit cleanup path already cleared ownership.
  # Return an explicit success instead of inheriting the caller's non-zero
  # workload status through a bare `return`.
  [[ -z "${pid}" ]] && return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "${pid}" 2>/dev/null || break
      sleep 0.1
    done
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

list_port_owner_pids() {
  lsof -t -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null \
    | tr '\n' ' ' || true
}

wait_for_port_release() {
  local deadline=$((SECONDS + PORT_RELEASE_TIMEOUT_SEC))
  local pids=""
  while ((SECONDS < deadline)); do
    pids="$(list_port_owner_pids)"
    [[ -z "${pids// /}" ]] && return 0
    sleep 1
  done
  pids="$(list_port_owner_pids)"
  echo "[ERROR] port ${PORT} remained occupied by ${pids:-unknown}" >&2
  return 1
}

check_port() {
  local pids=""
  pids="$(list_port_owner_pids)"
  [[ -z "${pids// /}" ]] && return 0
  if [[ "${KILL_EXISTING_PORT_OWNER}" != "1" ]]; then
    echo "[ERROR] port ${PORT} is already used by ${pids}" >&2
    return 1
  fi
  # Emergency-only behavior for a dedicated experiment port.
  # shellcheck disable=SC2086
  kill -TERM ${pids} 2>/dev/null || true
  wait_for_port_release
}

process_group_has_live_members() {
  local pgid="$1"
  ps -eo pgid=,stat= | awk -v target="${pgid}" '
    $1 == target && $2 !~ /^Z/ { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

cleanup_process_group() {
  local leader_pid="${1:-}"
  local pgid="${2:-}"
  # This function is intentionally idempotent: the EXIT trap calls it again
  # after normal explicit cleanup. A bare return here can preserve the pending
  # `exit 1` status and falsely report lifecycle failure code 86.
  [[ -z "${leader_pid}" ]] && return 0
  if [[ -z "${pgid}" || ! "${pgid}" =~ ^[1-9][0-9]*$ ]]; then
    cleanup_pid "${leader_pid}"
    return
  fi
  local runner_pgid=""
  runner_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]')"
  if [[ "${pgid}" == "${runner_pgid}" ]]; then
    echo "[ERROR] refusing to signal runner process group ${pgid}" >&2
    cleanup_pid "${leader_pid}"
    return 1
  fi
  if process_group_has_live_members "${pgid}"; then
    echo "[CLEANUP] terminate server process group pgid=${pgid}" >&2
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    local deadline=$((SECONDS + SERVER_SHUTDOWN_TIMEOUT_SEC))
    while process_group_has_live_members "${pgid}" \
        && ((SECONDS < deadline)); do
      sleep 1
    done
  fi
  if process_group_has_live_members "${pgid}"; then
    echo "[CLEANUP] force-kill server process group pgid=${pgid}" >&2
    kill -KILL -- "-${pgid}" 2>/dev/null || true
    local kill_deadline=$((SECONDS + 5))
    while process_group_has_live_members "${pgid}" \
        && ((SECONDS < kill_deadline)); do
      sleep 1
    done
  fi
  wait "${leader_pid}" 2>/dev/null || true
  if process_group_has_live_members "${pgid}"; then
    echo "[ERROR] server process group ${pgid} survived cleanup" >&2
    return 1
  fi
}

ACTIVE_SERVER_PID=""
ACTIVE_SERVER_PGID=""
ACTIVE_DMON_PID=""
cleanup_active() {
  local had_server=0
  local cleanup_status=0
  [[ -n "${ACTIVE_SERVER_PID}" ]] && had_server=1
  cleanup_pid "${ACTIVE_DMON_PID}"
  cleanup_process_group \
    "${ACTIVE_SERVER_PID}" "${ACTIVE_SERVER_PGID}" || cleanup_status=1
  ACTIVE_DMON_PID=""
  ACTIVE_SERVER_PID=""
  ACTIVE_SERVER_PGID=""
  if ((had_server)) && ! wait_for_port_release; then
    cleanup_status=1
  fi
  return "${cleanup_status}"
}
cleanup_on_exit() {
  local status=$?
  local cleanup_status=0
  trap - EXIT
  cleanup_active || cleanup_status="$?"
  if ((cleanup_status != 0)); then
    exit "${INSTANCE_CLEANUP_FAILURE_EXIT_CODE}"
  fi
  exit "${status}"
}
cleanup_on_signal() {
  local cleanup_status=0
  trap - INT TERM
  cleanup_active || cleanup_status="$?"
  if ((cleanup_status != 0)); then
    exit "${INSTANCE_CLEANUP_FAILURE_EXIT_CODE}"
  fi
  exit 130
}
trap cleanup_on_exit EXIT
trap cleanup_on_signal INT TERM

for command_name in setsid lsof ps awk; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "[ERROR] missing lifecycle command: ${command_name}" >&2
    exit 1
  }
done
check_port

ITERATION_METRICS=""
REQUEST_METRICS="${ARTIFACT_DIR}/request_metrics.jsonl"
DECODE_TIMELINE=""
if [[ "${ENABLE_FULL_ITERATION_TRACE}" == "1" \
      || "${ENABLE_LIGHTWEIGHT_TIMELINE_TRACE}" == "1" ]]; then
  ITERATION_METRICS="${ARTIFACT_DIR}/iteration_metrics.jsonl"
  DECODE_TIMELINE="${ARTIFACT_DIR}/decode_timeline.jsonl"
fi
PD_CUDA_EVENT_TRACE=""
if [[ "${ENABLE_PD_CUDA_EVENT_TRACE}" == "1" ]]; then
  PD_CUDA_EVENT_TRACE="${ARTIFACT_DIR}/pd_cuda_event_trace.jsonl"
fi
SERVER_LOG="${LOG_DIR}/server.log"

if [[ "${SYSTEM_VARIANT}" == "full_vendor_v8" \
      && "${PREFILL_SCHEDULING_POLICY}" == "slo_srpt" ]]; then
  VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS} \
--ttft-prefill-estimated-ms-per-token ${PREFILL_FALLBACK_MS_PER_TOKEN} \
--ttft-prefill-service-profile-path ${PREFILL_SERVICE_PROFILE} \
--ttft-prefill-slo-guard-ms ${TTFT_PREFILL_SLO_GUARD_MS} \
--ttft-prefill-starvation-threshold-ms ${TTFT_PREFILL_STARVATION_THRESHOLD_MS} \
--ttft-prefill-starvation-max-bypasses ${TTFT_PREFILL_STARVATION_MAX_BYPASSES}"
fi
echo "[START] server_extra_args=${VLLM_EXTRA_ARGS}"

(
  cd "${ROOT_DIR}"
  exec setsid env \
  VLLM_VENDOR_DIR="${VLLM_VENDOR_DIR}" \
  VLLM_EXECUTABLE="${VLLM_EXECUTABLE}" \
  VLLM_SCHEDULING_POLICY="${VLLM_SCHEDULING_POLICY}" \
  VLLM_PD_EVAL_POLICY="${PD_EVAL_POLICY}" \
  VLLM_PD_EVAL_PHASE_DEPTH_ENABLED="${PD_PHASE_DEPTH_ENABLED}" \
  VLLM_PD_EVAL_PREFILL_INFLIGHT_DEPTH="${PD_PREFILL_INFLIGHT_DEPTH}" \
  VLLM_PD_EVAL_DECODE_INFLIGHT_DEPTH="${PD_DECODE_INFLIGHT_DEPTH}" \
  VLLM_PD_EVAL_GLOBAL_INFLIGHT_DEPTH="${PD_GLOBAL_INFLIGHT_DEPTH}" \
  VLLM_PD_PREFILL_CHAIN_MAX_LOOKAHEAD_TOKENS="${PD_PREFILL_CHAIN_MAX_LOOKAHEAD_TOKENS}" \
  VLLM_VIRTUAL_CONTEXT_DECODE_CONTINUATION_ROLLING_RECLAIM="${DECODE_CONTINUATION_ROLLING_RECLAIM}" \
  VLLM_PD_EVAL_RESIDENCY_VICTIM_POLICY="${RESIDENCY_VICTIM_POLICY}" \
  VLLM_PD_EVAL_SCHEDULED_PREFILL_FIT_CAPACITY_FRACTION="${FULL_VENDOR_V8_SCHEDULED_PREFILL_FIT_CAPACITY_FRACTION}" \
  VLLM_PD_EVAL_NATIVE_PREFILL_CAPACITY_MODE="${NATIVE_PREFILL_CAPACITY_MODE}" \
  VLLM_PD_EVAL_RING_ENABLE_CHUNKED_PREFILL="${RING_ENABLE_CHUNKED_PREFILL}" \
  VLLM_PD_EVAL_VIRTUAL_CONTEXT_TIMING="${VIRTUAL_CONTEXT_TIMING}" \
  VLLM_VIRTUAL_CONTEXT_TIMING="${VIRTUAL_CONTEXT_TIMING}" \
  VLLM_PD_EVAL_PREFILL_RING_REQUIRED_MODE=pressure \
  VLLM_PD_EVAL_RING_BW_MIN_TOKENS="${PREFILL_CHUNK_TOKENS}" \
  VLLM_PD_EVAL_RING_BW_MAX_TOKENS="${RING_BW_MAX_TOKENS}" \
  VLLM_PD_EVAL_RING_BW_DYNAMIC="${RING_BW_DYNAMIC}" \
  VLLM_PD_EVAL_RING_BW_POLICY="${RING_BW_POLICY}" \
  VLLM_PD_EVAL_RING_BW_SAFETY_BLOCKS="${RING_BW_SAFETY_BLOCKS}" \
  VLLM_PD_EVAL_RING_BW_DECODE_GROWTH_BLOCKS="${RING_BW_DECODE_GROWTH_BLOCKS}" \
  VLLM_PD_EVAL_RING_BW_USE_SCHEDULED_HEADROOM="${RING_BW_USE_SCHEDULED_HEADROOM}" \
  VLLM_PD_EVAL_RING_MIN_PROGRESS_TOKENS="${PREFILL_CHUNK_TOKENS}" \
  VLLM_PD_EVAL_RING_MIN_PREFILL_TOKENS_WITH_DECODE="${PREFILL_CHUNK_TOKENS}" \
  VLLM_PD_EVAL_ACTIVE_DECODE_NATIVE_PREFILL_TOKENS="${ACTIVE_DECODE_NATIVE_PREFILL_TOKENS}" \
  VLLM_PD_EVAL_ACTIVE_DECODE_RING_PREFILL_TOKENS="${ACTIVE_DECODE_RING_PREFILL_TOKENS}" \
  VLLM_PD_EVAL_ACTIVE_DECODE_ELASTIC_PREFILL="${ACTIVE_DECODE_ELASTIC_PREFILL}" \
  VLLM_PD_EVAL_RING_COMPUTE_GROUP_TOKENS="${RING_COMPUTE_GROUP_TOKENS}" \
  VLLM_PD_EVAL_RING_STREAMING_WAVE_GROUPS="${RING_STREAMING_WAVE_GROUPS}" \
  VLLM_PD_EVAL_RING_FULLY_OPTIMIZED=1 \
  VLLM_PD_EVAL_RING_BATCHED_PARTIAL="${RING_BATCHED_PARTIAL}" \
  VLLM_PD_EVAL_RING_BATCHED_PARTIAL_STRICT="${RING_BATCHED_PARTIAL_STRICT}" \
  VLLM_PD_EVAL_RING_BATCHED_WORKSPACE_BYTES="${RING_BATCHED_WORKSPACE_BYTES}" \
  VLLM_PD_EVAL_DRAM_PAGE_POOL_CONTIGUOUS=1 \
  VLLM_PD_EVAL_DRAM_REGISTER_DIRECT_MULTICOPY=1 \
  VLLM_PD_EVAL_RING_STAGING_DIRECT_MULTICOPY=1 \
  VLLM_PD_EVAL_RING_STAGING_DIRECT_MULTICOPY_STRICT=1 \
  VLLM_PD_EVAL_DIRECT_MULTICOPY_STRICT=1 \
  VLLM_PD_EVAL_RELOAD_DIRECT_MULTICOPY=1 \
  VLLM_PD_EVAL_RELOAD_STAGING_SCATTER=0 \
  VLLM_PD_EVAL_RELOAD_FUSED_SCATTER=0 \
  VLLM_PD_EVAL_RELOAD_FUSED_SCATTER_STRICT=0 \
  VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS="${DECODE_SMS}" \
  VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS="${PREFILL_SMS}" \
  VLLM_OPERATOR_SELECTIVE_PD="${OPERATOR_SELECTIVE_PD}" \
  VLLM_PD_DYNAMIC_SM_ENABLED="${PD_DYNAMIC_SM_ENABLED}" \
  VLLM_PD_DYNAMIC_SM_SHADOW=0 \
  VLLM_PD_DYNAMIC_SM_STRICT=1 \
  VLLM_PD_DYNAMIC_SM_TOTAL_SMS=40 \
  VLLM_PD_DYNAMIC_SM_ALIGNMENT=2 \
  VLLM_PD_DYNAMIC_SM_PROFILES="${PD_SM_PROFILES}" \
  VLLM_PD_DYNAMIC_SM_POLICY="${PD_DYNAMIC_SM_POLICY}" \
  VLLM_PD_SERIALIZED_FULL_SM="${PD_SERIALIZED_FULL_SM}" \
  VLLM_CUDAGRAPH_MODE="${CUDAGRAPH_MODE}" \
  VLLM_PD_GREEN_DECODE_CUDAGRAPH="${PD_GREEN_DECODE_CUDAGRAPH}" \
  VLLM_PD_GREEN_DECODE_CUDAGRAPH_STRICT="${PD_GREEN_DECODE_CUDAGRAPH_STRICT}" \
  VLLM_PD_FUSED_EXECUTE_SAMPLE="${PD_FUSED_EXECUTE_SAMPLE}" \
  VLLM_PD_PHASE_SYNC_SAMPLE_OUTPUT="${PD_PHASE_SYNC_SAMPLE_OUTPUT}" \
  VLLM_PD_LAYERWISE_PROFILE_HANDOFF="${PD_LAYERWISE_PROFILE_HANDOFF}" \
  VLLM_PD_LAYERWISE_PROFILE_HANDOFF_STRICT=1 \
  VLLM_TTFT_LAYERWISE_PREFILL_EXECUTION="${TTFT_LAYERWISE_PREFILL_EXECUTION}" \
  VLLM_TTFT_LAYERWISE_PREFILL_PREEMPTION="${TTFT_LAYERWISE_PREFILL_PREEMPTION}" \
  VLLM_TTFT_LAYERWISE_PREFILL_GROUP_LAYERS="${TTFT_LAYERWISE_PREFILL_GROUP_LAYERS}" \
  VLLM_KV_RETENTION_LAYERWISE_CHECKPOINT_PLAN="${KV_LAYERWISE_CHECKPOINT_PLAN}" \
  VLLM_KV_RETENTION_LAYERWISE_OFFLOAD="${KV_LAYERWISE_OFFLOAD}" \
  VLLM_KV_RETENTION_P1_PARKED_PREFILL="${KV_P1_PARKED}" \
  VLLM_KV_RETENTION_P1_HBM_ONLY="${KV_P1_HBM_ONLY}" \
  VLLM_KV_RETENTION_PREFILL_WORK_BINDING="${KV_PREFILL_WORK_BINDING}" \
  VLLM_KV_RETENTION_PREFILL_RESUME_FRAME="${KV_PREFILL_RESUME_FRAME}" \
  VLLM_KV_RETENTION_PARKED_MAX_BATCHES="${KV_PARKED_MAX_BATCHES}" \
  VLLM_KV_RETENTION_PARKED_GPU_BYTES="${KV_PARKED_GPU_BYTES}" \
  VLLM_PD_PREFILL_WORKSPACE_ENABLED="${PD_PREFILL_WORKSPACE_ENABLED}" \
  VLLM_PD_PREFILL_WORKSPACE_MAX_TOKENS="${PD_PREFILL_WORKSPACE_MAX_TOKENS}" \
  VLLM_PD_PREFILL_WORKSPACE_SLOTS="${PD_PREFILL_WORKSPACE_SLOTS}" \
  VLLM_PD_PREFILL_WORKSPACE_STRICT="${PD_PREFILL_WORKSPACE_STRICT}" \
  VLLM_PD_DECODE_FIRST_HOST_ENQUEUE="${DECODE_FIRST_HOST_ENQUEUE}" \
  VLLM_LIGHTWEIGHT_TIMELINE_TRACE="${ENABLE_LIGHTWEIGHT_TIMELINE_TRACE}" \
  VLLM_LIGHTWEIGHT_TIMELINE_STRICT_BOUNDED="${STRICT_BOUNDED_LIGHTWEIGHT_TIMELINE}" \
  VLLM_FULL_ITERATION_DEBUG="${ENABLE_FULL_ITERATION_TRACE}" \
  VLLM_PD_EXECUTOR_DEBUG="${ENABLE_PD_EXECUTOR_DEBUG}" \
  VLLM_ITERATION_METRICS_SAMPLE_EVERY_N="${ITERATION_TRACE_SAMPLE_EVERY_N}" \
  VLLM_ISOLATION_SCHEDULER_TIMING_JSONL="${SCHEDULER_METRICS}" \
  VLLM_DEBUG_TRACE_QUEUE_SIZE="${DEBUG_TRACE_QUEUE_SIZE}" \
  VLLM_ITERATION_GPU_TIMING=0 \
  VLLM_LAYER_KERNEL_TIMING=0 \
  VLLM_PD_DECODE_GPU_COMPONENT_TRACE_JSONL="" \
  VLLM_PD_DECODE_GPU_COMPONENT_TRACE_INCLUDE_LAYERS=0 \
  VLLM_PD_CUDA_EVENT_TRACE_JSONL="${PD_CUDA_EVENT_TRACE}" \
  VLLM_PD_NVTX_PHASE_TRACE="${ENABLE_PD_NVTX_PHASE_TRACE}" \
  VLLM_OPERATOR_SELECTIVE_PD_TIMING=0 \
  VLLM_OPMUX_TIMING=0 \
  VLLM_VIRTUAL_CONTEXT_RING_CUDA_EVENT_TIMING=0 \
  VLLM_CONTEXT_RING_INTERFERENCE_TRACE_ENABLED=0 \
  VLLM_CONTEXT_RING_NVTX_TRACE_ENABLED=0 \
  VLLM_PD_TBT_PROFILE_PATH="${ACTIVE_TBT_OVERLAP_PROFILE}" \
  VLLM_PD_TBT_DEFAULT_SLO_MS="${TBT_DEFAULT_SLO_MS}" \
  VLLM_PD_TBT_PROFILE_MARGIN_MS="${TBT_PROFILE_MARGIN_MS}" \
  VLLM_PD_TBT_PROFILE_STRICT="${TBT_PROFILE_STRICT}" \
  VLLM_PD_TBT_PREDICTIVE_LEAD_MS="${TBT_PREDICTIVE_LEAD_MS}" \
  VLLM_PD_TBT_PREDICTIVE_CUTOVER_MS="${TBT_PREDICTIVE_CUTOVER_MS}" \
  VLLM_PD_TBT_PREDICTIVE_MIN_STEP_MS="${TBT_PREDICTIVE_MIN_STEP_MS}" \
  VLLM_PD_FEEDBACK_INITIAL_DECODE_SMS="${PD_FEEDBACK_INITIAL_DECODE_SMS}" \
  VLLM_PD_FEEDBACK_STEP_SMS="${PD_FEEDBACK_STEP_SMS}" \
  VLLM_PD_FEEDBACK_WINDOW_TOKENS="${PD_FEEDBACK_WINDOW_TOKENS}" \
  VLLM_PD_FEEDBACK_SLACK_RATIO="${PD_FEEDBACK_SLACK_RATIO}" \
  VLLM_PD_FEEDBACK_REPROBE_RATIO="${PD_FEEDBACK_REPROBE_RATIO}" \
  VLLM_PD_FEEDBACK_DECREASE_WINDOWS="${PD_FEEDBACK_DECREASE_WINDOWS}" \
  VLLM_PD_FEEDBACK_HOLD_WINDOWS="${PD_FEEDBACK_HOLD_WINDOWS}" \
  VLLM_PD_FEEDBACK_MIN_PREFILL_SMS="${PD_FEEDBACK_MIN_PREFILL_SMS}" \
  VLLM_PD_FEEDBACK_CONTEXT_BUCKET_TOKENS="${PD_FEEDBACK_CONTEXT_BUCKET_TOKENS}" \
  VLLM_ITERATION_METRICS_JSONL="${ITERATION_METRICS}" \
  VLLM_REQUEST_METRICS_JSONL="${REQUEST_METRICS}" \
  VLLM_DECODE_TIMELINE_JSONL="${DECODE_TIMELINE}" \
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
  VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" \
  VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS}" \
  VLLM_LONG_PREFILL_TOKEN_THRESHOLD="${VLLM_LONG_PREFILL_TOKEN_THRESHOLD}" \
  VLLM_ENABLE_CHUNKED_PREFILL=1 \
  VLLM_DISABLE_PREFIX_CACHING=1 \
  VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS}" \
    "${SERVER_STARTER}" --port "${PORT}" --model "${MODEL_NAME}"
) >>"${SERVER_LOG}" 2>&1 &
ACTIVE_SERVER_PID="$!"
ACTIVE_SERVER_PGID="${ACTIVE_SERVER_PID}"
if ! wait_for_server_ready "${ACTIVE_SERVER_PID}"; then
  tail -n 180 "${SERVER_LOG}" >&2 || true
  exit 1
fi
observed_server_pgid="$(
  ps -o pgid= -p "${ACTIVE_SERVER_PID}" 2>/dev/null \
    | tr -d '[:space:]'
)"
if [[ "${observed_server_pgid}" != "${ACTIVE_SERVER_PGID}" ]]; then
  echo "[ERROR] server process group mismatch: expected="\
"${ACTIVE_SERVER_PGID} observed=${observed_server_pgid:-missing}" >&2
  runner_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]')"
  if [[ "${observed_server_pgid}" =~ ^[1-9][0-9]*$ \
        && "${observed_server_pgid}" != "${runner_pgid}" ]]; then
    ACTIVE_SERVER_PGID="${observed_server_pgid}"
  fi
  exit 1
fi
echo "[LIFECYCLE] server_pid=${ACTIVE_SERVER_PID} pgid=${ACTIVE_SERVER_PGID}"
if [[ "${ENABLE_DMON}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi dmon -s pucm -d 1 -o TD \
    >"${ARTIFACT_DIR}/nvidia_smi_dmon.log" 2>&1 &
  ACTIVE_DMON_PID="$!"
fi

measure_status=0
if [[ "${WORKLOAD}" == "LOONGSERVE" ]]; then
  "${PYTHON_BIN}" "${MEASURER}" \
    --base-url "${BASE_URL}" \
    --model "${MODEL_NAME}" \
    --workload-jsonl "${LOONGSERVE_TRACE_JSONL}" \
    --slo-policy-json "${REQUEST_SLO_POLICY}" \
    --slo-overflow-policy last_bucket \
    --case-id "${EXPERIMENT_NAME}" \
    --output-dir "${RESULT_DIR}/client" \
    --arrival-time-scale "${LOONGSERVE_ARRIVAL_TIME_SCALE}" \
    --termination-mode "${LOONGSERVE_TERMINATION_MODE}" \
    --request-limit "${LOONGSERVE_REQUEST_LIMIT}" \
    --time-limit-sec "${LOONGSERVE_TIME_LIMIT_SEC}" \
    --time-limit-mode "${LOONGSERVE_TIME_LIMIT_MODE}" \
    --request-timeout-sec "${REQUEST_TIMEOUT_SEC}" \
    --ignore-eos 1 \
    --warmup-count "${WARMUP_COUNT}" \
    --warmup-output-tokens "${WARMUP_OUTPUT_TOKENS}" \
    --warmup-cooldown-sec "${WARMUP_COOLDOWN_SEC}" \
    || measure_status="$?"
else
  "${PYTHON_BIN}" "${MEASURER}" \
    --base-url "${BASE_URL}" \
    --requests-jsonl "${DATASET_JSONL}" \
    --slo-policy-json "${REQUEST_SLO_POLICY}" \
    --case-id "${EXPERIMENT_NAME}" \
    --workload "${WORKLOAD}" \
    --submission-mode "${SUBMISSION_MODE}" \
    --long-context-tokens "${LONG_CONTEXT_TOKENS}" \
    --short-context-tokens "${SHORT_CONTEXT_TOKENS}" \
    --long-count "${LONG_REQUEST_COUNT}" \
    --short-per-long "${SHORT_REQUESTS_PER_LONG}" \
    --short-count "${SHORT_REQUEST_COUNT}" \
    --short-injection-offsets-sec "${SHORT_OFFSET_ARRAY[@]}" \
    --long-injection-offsets-sec "${LONG_OFFSET_ARRAY[@]}" \
    --interleaved-injection-interval-sec "${INTERLEAVED_INJECTION_INTERVAL_SEC}" \
    --long-output-tokens "${LONG_OUTPUT_TOKENS}" \
    --short-output-tokens "${SHORT_OUTPUT_TOKENS}" \
    --warmup-count "${WARMUP_COUNT}" \
    --warmup-output-tokens "${WARMUP_OUTPUT_TOKENS}" \
    --warmup-cooldown-sec "${WARMUP_COOLDOWN_SEC}" \
    --request-timeout-sec "${REQUEST_TIMEOUT_SEC}" \
    --plan-csv "${RESULT_DIR}/submission_plan.csv" \
    --raw-csv "${RESULT_DIR}/request_results.csv" \
    --token-csv "${RESULT_DIR}/client_token_timeline.csv" \
    --summary-csv "${RESULT_DIR}/slo_summary.csv" \
    --metadata-json "${RESULT_DIR}/run_metadata.json" \
    || measure_status="$?"
fi

cleanup_status=0
cleanup_active || cleanup_status="$?"
if ((cleanup_status != 0)); then
  echo "[ERROR] vLLM instance cleanup failed; refusing unsafe restart" >&2
  exit "${INSTANCE_CLEANUP_FAILURE_EXIT_CODE}"
fi
echo "[LIFECYCLE] vLLM instance stopped; port ${PORT} is free"
if ((measure_status != 0)); then
  echo "[ERROR] workload measurer exited with status ${measure_status}" >&2
  exit "${measure_status}"
fi

cat >"${RESULT_DIR}/experiment_snapshot/settings.txt" <<EOF
experiment_name=${EXPERIMENT_NAME}
isolation_matched_comparison=${ISOLATION_MATCHED_COMPARISON}
isolation_guarded_2k_comparison=${ISOLATION_GUARDED_2K_COMPARISON}
isolation_batch2_pd11_comparison=${ISOLATION_BATCH2_PD11_COMPARISON}
prefill_batch_max_requests=${VLLM_ISOLATION_PREFILL_BATCH_MAX_REQUESTS}
decode_batch_max_requests=${VLLM_ISOLATION_DECODE_BATCH_MAX_REQUESTS}
phase_batch_limit_deployment=${PHASE_BATCH_LIMIT_RESULT:-disabled}
hbm_completion_guard=${VLLM_ISOLATION_HBM_COMPLETION_GUARD}
hbm_guard_deployment=${HBM_GUARD_RESULT:-disabled}
decode_output_first=${VLLM_ISOLATION_DECODE_OUTPUT_FIRST}
prefill_fit_next_chunk=${VLLM_ISOLATION_PREFILL_FIT_NEXT_CHUNK}
cudagraph_capture_sizes=${MATCHED_CUDAGRAPH_CAPTURE_SIZES[*]:-vllm_default}
long_context_tokens=${LONG_CONTEXT_TOKENS}
long_request_count=${LONG_REQUEST_COUNT}
short_context_tokens=${SHORT_CONTEXT_TOKENS}
short_requests_per_long=${SHORT_REQUESTS_PER_LONG}
short_injection_offsets_sec=${SHORT_OFFSET_ARRAY[*]}
long_injection_offsets_sec=${LONG_OFFSET_ARRAY[*]}
interleaved_injection_interval_sec=${INTERLEAVED_INJECTION_INTERVAL_SEC}
short_request_count=${SHORT_REQUEST_COUNT}
warmup_count=${WARMUP_COUNT}
warmup_output_tokens=${WARMUP_OUTPUT_TOKENS}
warmup_cooldown_sec=${WARMUP_COOLDOWN_SEC}
workload=${WORKLOAD}
submission_mode=${EFFECTIVE_SUBMISSION_MODE}
loongserve_trace_jsonl=${LOONGSERVE_TRACE_JSONL}
loongserve_arrival_time_scale=${LOONGSERVE_ARRIVAL_TIME_SCALE}
loongserve_termination_mode=${LOONGSERVE_TERMINATION_MODE}
loongserve_request_limit=${LOONGSERVE_REQUEST_LIMIT}
loongserve_time_limit_sec=${LOONGSERVE_TIME_LIMIT_SEC}
loongserve_time_limit_mode=${LOONGSERVE_TIME_LIMIT_MODE}
server_shutdown_timeout_sec=${SERVER_SHUTDOWN_TIMEOUT_SEC}
port_release_timeout_sec=${PORT_RELEASE_TIMEOUT_SEC}
instance_cleanup_failure_exit_code=${INSTANCE_CLEANUP_FAILURE_EXIT_CODE}
system_variant=${SYSTEM_VARIANT}
pd_eval_policy=${PD_EVAL_POLICY}
server_extra_args=${VLLM_EXTRA_ARGS}
pd_phase_depth_enabled=${PD_PHASE_DEPTH_ENABLED}
pd_prefill_inflight_depth=${PD_PREFILL_INFLIGHT_DEPTH}
pd_decode_inflight_depth=${PD_DECODE_INFLIGHT_DEPTH}
pd_global_inflight_depth=${PD_GLOBAL_INFLIGHT_DEPTH}
pd_phase_sync_sample_output=${PD_PHASE_SYNC_SAMPLE_OUTPUT}
pd_prefill_chain_max_lookahead_tokens=${PD_PREFILL_CHAIN_MAX_LOOKAHEAD_TOKENS}
decode_continuation_rolling_reclaim=${DECODE_CONTINUATION_ROLLING_RECLAIM}
residency_victim_policy=${RESIDENCY_VICTIM_POLICY}
initial_green_sm_split=$(
  if [[ "${NATIVE_FULL_GPU_VARIANT}" == "1" ]]; then
    echo "disabled"
  else
    echo "${INITIAL_GREEN_SM_SPLIT}"
  fi
)
execution_sms=$((DECODE_SMS + PREFILL_SMS))
vllm_vendor_dir=${VLLM_VENDOR_DIR:-installed_package}
pd_sm_profiles=${PD_SM_PROFILES}
pd_serialized_full_sm=${PD_SERIALIZED_FULL_SM}
pd_serialized_full_sm_scope=capacity_blocked_prefill_only
cudagraph_mode=${CUDAGRAPH_MODE:-vllm_default}
matched_graph_preserves_compile=${ISOLATION_MATCHED_COMPARISON}
cudagraph_path=$(
  if [[ "${NATIVE_FULL_GPU_VARIANT}" == "1" ]]; then
    echo "upstream"
  elif [[ "${PD_GREEN_DECODE_CUDAGRAPH}" == "1" ]]; then
    echo "greencontext_partitioned_decode"
  else
    echo "disabled_or_eager"
  fi
)
request_slo_policy=${REQUEST_SLO_POLICY}
prefill_scheduling_policy=${PREFILL_SCHEDULING_POLICY}
baseline_plus_prefill_quantum_tokens=${BASELINE_PLUS_PREFILL_QUANTUM_TOKENS}
scheduler_only_baseline=${SCHEDULER_ONLY_BASELINE}
vllm_scheduling_policy=${VLLM_SCHEDULING_POLICY}
prefill_admission=${PREFILL_ADMISSION_LABEL}
long_prefill_token_threshold=${VLLM_LONG_PREFILL_TOKEN_THRESHOLD}
native_prefill_capacity_mode=${NATIVE_PREFILL_CAPACITY_MODE}
prefill_service_profile=$(
  if [[ "${SYSTEM_VARIANT}" == "full_vendor_v8" \
        && "${PREFILL_SCHEDULING_POLICY}" == "slo_srpt" ]]; then
    echo "${PREFILL_SERVICE_PROFILE}"
  else
    echo "disabled"
  fi
)
tbt_overlap_profile=${ACTIVE_TBT_OVERLAP_PROFILE:-disabled_for_${PD_DYNAMIC_SM_POLICY}}
tbt_predictive_lead_ms=${TBT_PREDICTIVE_LEAD_MS}
tbt_predictive_cutover_ms=${TBT_PREDICTIVE_CUTOVER_MS}
tbt_predictive_min_step_ms=${TBT_PREDICTIVE_MIN_STEP_MS}
prefill_fallback_ms_per_token=${PREFILL_FALLBACK_MS_PER_TOKEN}
tbt_default_slo_ms=${TBT_DEFAULT_SLO_MS}
profile_switch_boundary=$(
  if [[ "${PD_LAYERWISE_PROFILE_HANDOFF}" == "1" ]]; then
    echo "prefill_layer_boundary_then_quiescent_commit"
  else
    echo "quiescent_only"
  fi
)
layerwise_profile_handoff=${PD_LAYERWISE_PROFILE_HANDOFF}
ttft_layerwise_prefill_execution=${TTFT_LAYERWISE_PREFILL_EXECUTION}
ttft_layerwise_prefill_preemption=${TTFT_LAYERWISE_PREFILL_PREEMPTION}
ttft_layerwise_prefill_group_layers=${TTFT_LAYERWISE_PREFILL_GROUP_LAYERS}
kv_layerwise_checkpoint_plan=${KV_LAYERWISE_CHECKPOINT_PLAN}
kv_layerwise_offload=${KV_LAYERWISE_OFFLOAD}
kv_p1_parked=${KV_P1_PARKED}
kv_p1_hbm_only=${KV_P1_HBM_ONLY}
kv_prefill_work_binding=${KV_PREFILL_WORK_BINDING}
kv_prefill_resume_frame=${KV_PREFILL_RESUME_FRAME}
kv_parked_max_batches=${KV_PARKED_MAX_BATCHES}
kv_parked_gpu_bytes=${KV_PARKED_GPU_BYTES}
ttft_layerwise_prefill_scheduler_yield=${TTFT_LAYERWISE_PREFILL_PREEMPTION}
prefill_workspace_enabled=${PD_PREFILL_WORKSPACE_ENABLED}
prefill_workspace_max_tokens=${PD_PREFILL_WORKSPACE_MAX_TOKENS}
prefill_workspace_slots=${PD_PREFILL_WORKSPACE_SLOTS}
prefill_workspace_strict=${PD_PREFILL_WORKSPACE_STRICT}
isolation_memory_optimizations=${VLLM_ISOLATION_MEMORY_OPTIMIZATIONS}
isolation_fused_rms_norm=${VLLM_ISOLATION_FUSED_RMS_NORM}
isolation_prefill_extended_workspace=${VLLM_ISOLATION_PREFILL_EXTENDED_WORKSPACE}
isolation_prefill_workspace_rightsize=${VLLM_ISOLATION_PREFILL_WORKSPACE_RIGHTSIZE}
startup_cache_trim=${VLLM_ISOLATION_STARTUP_CACHE_TRIM}
startup_memory_audit=${VLLM_ISOLATION_STARTUP_MEMORY_AUDIT}
startup_memory_deployment=${STARTUP_MEMORY_RESULT:-disabled}
isolation_ring_workspace_sharing=${VLLM_ISOLATION_RING_WORKSPACE_SHARING}
isolation_green_context_process_lifetime=${VLLM_ISOLATION_GREEN_CONTEXT_PROCESS_LIFETIME}
isolation_ring_state_reuse=${VLLM_ISOLATION_RING_STATE_REUSE}
isolation_ring_dynamic_kernel=${VLLM_ISOLATION_RING_DYNAMIC_KERNEL}
isolation_compact_decode_buffers=${VLLM_ISOLATION_COMPACT_DECODE_BUFFERS}
isolation_ring_early_prefetch=${VLLM_ISOLATION_RING_EARLY_PREFETCH}
isolation_graph_memory_audit=${VLLM_ISOLATION_GRAPH_MEMORY_AUDIT}
isolation_execution_optimizations=${VLLM_ISOLATION_EXECUTION_OPTIMIZATIONS}
isolation_ring_query_expansion=${VLLM_ISOLATION_RING_QUERY_EXPANSION}
isolation_ring_query_scratch_bytes=${VLLM_ISOLATION_RING_QUERY_SCRATCH_BYTES}
isolation_p1_boundary_fastpath=${VLLM_ISOLATION_P1_BOUNDARY_FASTPATH}
isolation_p1_ring_sm_handoff=${VLLM_ISOLATION_P1_RING_SM_HANDOFF}
isolation_dram_refresh_probe=${VLLM_ISOLATION_DRAM_REFRESH_PROBE}
isolation_d2h_exact_plan_progress=${VLLM_ISOLATION_D2H_EXACT_PLAN_PROGRESS}
isolation_native_soft_headroom_progress=${VLLM_ISOLATION_NATIVE_SOFT_HEADROOM_PROGRESS}
isolation_prefill_batch_kv_budget=${VLLM_ISOLATION_PREFILL_BATCH_KV_BUDGET}
isolation_logical_release_readiness=${VLLM_ISOLATION_LOGICAL_RELEASE_READINESS}
isolation_bounded_prefill_planning=${VLLM_ISOLATION_BOUNDED_PREFILL_PLANNING}
isolation_waiting_decode_reclaim=${VLLM_ISOLATION_WAITING_DECODE_RECLAIM}
isolation_prefill_fused_rope=${VLLM_ISOLATION_PREFILL_FUSED_ROPE}
isolation_text_input_buffers=${VLLM_ISOLATION_TEXT_INPUT_BUFFERS}
isolation_memory_snapshots=${VLLM_ISOLATION_MEMORY_SNAPSHOTS}
prefill_selection_quantum_tokens=${PREFILL_SELECTOR_QUANTUM_TOKENS}
native_prefill_max_tokens=${NATIVE_PREFILL_MAX_TOKENS}
active_decode_native_prefill_tokens=${ACTIVE_DECODE_NATIVE_PREFILL_TOKENS}
active_decode_ring_prefill_tokens=${ACTIVE_DECODE_RING_PREFILL_TOKENS}
green_decode_cudagraph=${PD_GREEN_DECODE_CUDAGRAPH}
green_decode_cudagraph_strict=${PD_GREEN_DECODE_CUDAGRAPH_STRICT}
throughput_optimizations_requested=${FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS}
throughput_optimizations=${EFFECTIVE_FULL_VENDOR_V8_THROUGHPUT_OPTIMIZATIONS}
fused_execute_sample=${PD_FUSED_EXECUTE_SAMPLE}
tbt_component_monitoring=off
decode_first_host_enqueue=${DECODE_FIRST_HOST_ENQUEUE}
full_iteration_trace=${ENABLE_FULL_ITERATION_TRACE}
lightweight_timeline_trace=${ENABLE_LIGHTWEIGHT_TIMELINE_TRACE}
strict_bounded_lightweight_timeline=${STRICT_BOUNDED_LIGHTWEIGHT_TIMELINE}
pd_executor_debug=${ENABLE_PD_EXECUTOR_DEBUG}
iteration_trace_sample_every_n=${ITERATION_TRACE_SAMPLE_EVERY_N}
scheduler_timing_trace=${ENABLE_SCHEDULER_TIMING_TRACE}
scheduler_timing_deployment=${SCHEDULER_TIMING_RESULT:-disabled}
debug_trace_queue_size=${DEBUG_TRACE_QUEUE_SIZE}
ring_batched_partial=${RING_BATCHED_PARTIAL}
ring_batched_partial_strict=${RING_BATCHED_PARTIAL_STRICT}
ring_batched_workspace_bytes=${RING_BATCHED_WORKSPACE_BYTES}
ring_streaming_wave_groups=${RING_STREAMING_WAVE_GROUPS}
ring_streaming_wave_tokens=$((RING_COMPUTE_GROUP_TOKENS * RING_STREAMING_WAVE_GROUPS))
ring_bw_dynamic=${RING_BW_DYNAMIC}
ring_bw_min_tokens=${PREFILL_CHUNK_TOKENS}
ring_bw_max_tokens=${RING_BW_MAX_TOKENS}
ring_bw_policy=${RING_BW_POLICY}
ring_bw_safety_blocks=${RING_BW_SAFETY_BLOCKS}
ring_bw_decode_growth_blocks=${RING_BW_DECODE_GROWTH_BLOCKS}
ring_bw_use_scheduled_headroom=${RING_BW_USE_SCHEDULED_HEADROOM}
active_decode_elastic_prefill=${ACTIVE_DECODE_ELASTIC_PREFILL}
ring_compute_group_tokens=${RING_COMPUTE_GROUP_TOKENS}
virtual_context_timing=${VIRTUAL_CONTEXT_TIMING}
ring_staging_direct_multicopy=1
ring_staging_direct_multicopy_strict=1
direct_multicopy_strict=1
verify_tiered_direct_ops=${VERIFY_TIERED_DIRECT_OPS}
iteration_gpu_timing=off
layer_kernel_timing=off
pd_cuda_event_trace=$(
  if [[ "${ENABLE_PD_CUDA_EVENT_TRACE}" == "1" ]]; then echo "on"; else echo "off"; fi
)
context_ring_cuda_event_timing=off
context_ring_interference_trace=off
nvtx_trace=${ENABLE_PD_NVTX_PHASE_TRACE}
EOF

if [[ "${GENERATE_ABCD}" == "1" \
      && "${NATIVE_FULL_GPU_VARIANT}" != "1" \
      && -f "${ROOT_DIR}/graph_script/generate_pd_abcd_timeline_graphs.py" ]]; then
  mkdir -p "${RESULT_DIR}/abcd" "${ARTIFACT_DIR}/.matplotlib"
  MPLCONFIGDIR="${ARTIFACT_DIR}/.matplotlib" \
    "${PYTHON_BIN}" \
    "${ROOT_DIR}/graph_script/generate_pd_abcd_timeline_graphs.py" \
      --artifact-dir "${ARTIFACT_DIR}" \
      --output-dir "${RESULT_DIR}/abcd" \
      --decode-sms "${DECODE_SMS}" \
      --prefill-sms "${PREFILL_SMS}" || true
elif [[ "${GENERATE_ABCD}" == "1" \
        && "${NATIVE_FULL_GPU_VARIANT}" == "1" ]]; then
  echo "[INFO] skipping PD ABCD graph for ${SYSTEM_VARIANT}: the native mixed-batch execution contract has no P/D partition timeline"
fi

echo "[DONE] results=${RESULT_DIR}"
echo "[DONE] artifacts=${ARTIFACT_DIR}"
echo "[DONE] logs=${LOG_DIR}"
