#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${SCRIPT_DIR##*/}" == exp_script ]]; then
  SCRIPT_DIR="$(dirname "${SCRIPT_DIR}")"
fi

OPTIONS=$(getopt -o m:p:h --long model:,port:,help -- "$@") || exit 1
eval set -- "$OPTIONS"

MODEL="meta-llama/Llama-3.2-1B-Instruct"
PORT="${VLLM_PORT:-8000}"

while true; do
  case "$1" in
  -m | --model)
    MODEL="$2"
    shift 2
    ;;
  -p | --port)
    PORT="$2"
    shift 2
    ;;
  -h | --help)
    cat <<EOF
Usage: $0 [--model MODEL] [--port PORT]
EOF
    exit 0
    ;;
  --)
    shift
    break
    ;;
  *)
    echo "Unknown option: $1" >&2
    exit 1
    ;;
  esac
done

PD_EVAL_POLICY_HELPER="${SCRIPT_DIR}/tools/pd_eval_policy_env.sh"
if [[ -f "${PD_EVAL_POLICY_HELPER}" ]]; then
  # shellcheck source=/dev/null
  source "${PD_EVAL_POLICY_HELPER}"
  pd_eval_policy_apply
fi

# Causal microbench defaults:
# - no KV guarantee, no prefix cache, so batch composition is the intervention.
# - large batched-token budget so 24K-token prefills can fit together.
DTYPE="${VLLM_DTYPE:-half}"
KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-auto}"
BLOCK_SIZE="${VLLM_BLOCK_SIZE:-16}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-81920}"
LONG_PREFILL_TOKEN_THRESHOLD="${VLLM_LONG_PREFILL_TOKEN_THRESHOLD:-0}"
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
NUM_GPU_BLOCKS_OVERRIDE="${VLLM_NUM_GPU_BLOCKS_OVERRIDE-16384}"
TENANT_KV_MIN_BLOCKS="${VLLM_TENANT_KV_MIN_BLOCKS:-0}"
TENANT_KV_PROTECTION_MODE="${VLLM_TENANT_KV_PROTECTION_MODE:-indexed}"
ITERATION_GPU_TIMING="${VLLM_ITERATION_GPU_TIMING:-0}"
LAYER_KERNEL_TIMING="${VLLM_LAYER_KERNEL_TIMING:-0}"
PD_DECODE_GPU_COMPONENT_TRACE_JSONL="${VLLM_PD_DECODE_GPU_COMPONENT_TRACE_JSONL:-}"
PD_DECODE_GPU_COMPONENT_TRACE_MAX_PENDING="${VLLM_PD_DECODE_GPU_COMPONENT_TRACE_MAX_PENDING:-8192}"
PD_DECODE_GPU_COMPONENT_TRACE_EVERY_N="${VLLM_PD_DECODE_GPU_COMPONENT_TRACE_EVERY_N:-1}"
PD_DECODE_GPU_COMPONENT_TRACE_INCLUDE_LAYERS="${VLLM_PD_DECODE_GPU_COMPONENT_TRACE_INCLUDE_LAYERS:-0}"
PD_GREEN_DECODE_CUDAGRAPH="${VLLM_PD_GREEN_DECODE_CUDAGRAPH:-0}"
PD_GREEN_DECODE_CUDAGRAPH_STRICT="${VLLM_PD_GREEN_DECODE_CUDAGRAPH_STRICT:-1}"
CUDAGRAPH_MODE="${VLLM_CUDAGRAPH_MODE:-}"
OPERATOR_SELECTIVE_PD="${VLLM_OPERATOR_SELECTIVE_PD:-0}"
OPERATOR_SELECTIVE_PD_TIMING="${VLLM_OPERATOR_SELECTIVE_PD_TIMING:-0}"
OPERATOR_SELECTIVE_PD_GREEN_OVERLAP="${VLLM_OPERATOR_SELECTIVE_PD_GREEN_OVERLAP:-0}"
OPERATOR_SELECTIVE_GREEN_CONTEXT="${VLLM_OPERATOR_SELECTIVE_PD_GREEN_CONTEXT:-0}"
OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS="${VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS:-8}"
OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS="${VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS:-0}"
OPERATOR_SELECTIVE_PD_GREEN_STRICT="${VLLM_OPERATOR_SELECTIVE_PD_GREEN_STRICT:-0}"
OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_MODEL="${VLLM_OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_MODEL:-0}"
OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_NO_JOIN="${VLLM_OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_NO_JOIN:-1}"
MUXWISE_PHASE_BASELINE="${VLLM_MUXWISE_PHASE_BASELINE:-0}"
MUXWISE_PHASE_GREEN_CONTEXT="${VLLM_MUXWISE_PHASE_BASELINE_GREEN_CONTEXT:-0}"
PD_MULTIPLEXING_BASELINE="${VLLM_PD_MULTIPLEXING_BASELINE:-${VLLM_OPMUX_BASELINE:-off}}"
PD_DECOUPLED_NO_MIX="${VLLM_PD_DECOUPLED_NO_MIX:-0}"
OPERATOR_SELECTIVE_PD_REQUESTED="${OPERATOR_SELECTIVE_PD}"
if [[ "${PD_DECOUPLED_NO_MIX}" == "1" ]]; then
  # Phase-separated scheduling cannot produce a mixed P/D attention batch.
  # Export the effective value so model construction and eager-mode gating do
  # not install the redundant per-layer selector.
  OPERATOR_SELECTIVE_PD="0"
fi
PD_INDEPENDENT_PHASE_QUEUES="${VLLM_PD_INDEPENDENT_PHASE_QUEUES:-0}"
PD_SEPARATE_SCHEDULERS="${VLLM_PD_SEPARATE_SCHEDULERS:-0}"
PD_DUAL_CONTINUOUS_BATCHING="${VLLM_PD_DUAL_CONTINUOUS_BATCHING:-0}"
PD_DECODE_RESERVED_BATCHES="${VLLM_PD_DECODE_RESERVED_BATCHES:-1}"
PD_NONBLOCKING_PHASE_RECOVERY="${VLLM_PD_NONBLOCKING_PHASE_RECOVERY:-0}"
PD_SINGLE_INFLIGHT_PER_PHASE="${VLLM_PD_SINGLE_INFLIGHT_PER_PHASE:-0}"
PD_MUXWISE_EVENT_LOOP="${VLLM_PD_MUXWISE_EVENT_LOOP:-0}"
PD_PAIRED_PHASE_LAUNCH="${VLLM_PD_PAIRED_PHASE_LAUNCH:-0}"
PD_DECOUPLED_PAIRED_RECOVERY="${VLLM_PD_DECOUPLED_PAIRED_RECOVERY:-0}"
PD_PHASE_LOCAL_MODEL_RUNNER="${VLLM_PD_PHASE_LOCAL_MODEL_RUNNER:-${PD_MUXWISE_EVENT_LOOP}}"
PD_PHASE_SYNC_SAMPLE_OUTPUT="${VLLM_PD_PHASE_SYNC_SAMPLE_OUTPUT:-0}"
PD_PHASE_SAMPLER_DEBUG_SYNC="${VLLM_PD_PHASE_SAMPLER_DEBUG_SYNC:-0}"
PD_ASYNC_UNIPROC_EXECUTOR="${VLLM_PD_ASYNC_UNIPROC_EXECUTOR:-0}"
PD_ASYNC_UNIPROC_WORKERS="${VLLM_PD_ASYNC_UNIPROC_WORKERS:-2}"
PD_ASYNC_UNIPROC_PHASE_WORKERS="${VLLM_PD_ASYNC_UNIPROC_PHASE_WORKERS:-1}"
PD_ASYNC_UNIPROC_WORKERS_PER_PHASE="${VLLM_PD_ASYNC_UNIPROC_WORKERS_PER_PHASE:-1}"
PD_ASYNC_UNIPROC_CONCURRENT_PHASE_SUBMIT="${VLLM_PD_ASYNC_UNIPROC_CONCURRENT_PHASE_SUBMIT:-0}"
PD_FUSED_EXECUTE_SAMPLE="${VLLM_PD_FUSED_EXECUTE_SAMPLE:-0}"
PD_LAYERWISE_PROFILE_HANDOFF="${VLLM_PD_LAYERWISE_PROFILE_HANDOFF:-0}"
PD_LAYERWISE_PROFILE_HANDOFF_STRICT="${VLLM_PD_LAYERWISE_PROFILE_HANDOFF_STRICT:-1}"
PD_SERIALIZED_FULL_SM="${VLLM_PD_SERIALIZED_FULL_SM:-1}"
PD_WAIT_ON_PREFILL_KV_CAPACITY="${VLLM_PD_WAIT_ON_PREFILL_KV_CAPACITY:-0}"
VIRTUAL_CONTEXT_RING_PREFILL_ADMISSION_TOKENS="${VLLM_VIRTUAL_CONTEXT_RING_PREFILL_ADMISSION_TOKENS:-0}"
VIRTUAL_CONTEXT_RING_PREFILL_ADMISSION_TOKENS_MAX="${VLLM_VIRTUAL_CONTEXT_RING_PREFILL_ADMISSION_TOKENS_MAX:-0}"
VIRTUAL_CONTEXT_RING_BW_DYNAMIC="${VLLM_VIRTUAL_CONTEXT_RING_BW_DYNAMIC:-0}"
VIRTUAL_CONTEXT_RING_BW_MIN_TOKENS="${VLLM_VIRTUAL_CONTEXT_RING_BW_MIN_TOKENS:-${VIRTUAL_CONTEXT_RING_PREFILL_ADMISSION_TOKENS}}"
VIRTUAL_CONTEXT_RING_BW_MAX_TOKENS="${VLLM_VIRTUAL_CONTEXT_RING_BW_MAX_TOKENS:-${VIRTUAL_CONTEXT_RING_PREFILL_ADMISSION_TOKENS_MAX}}"
VIRTUAL_CONTEXT_RING_BW_LOW_FREE_BLOCKS="${VLLM_VIRTUAL_CONTEXT_RING_BW_LOW_FREE_BLOCKS:-${VLLM_VIRTUAL_CONTEXT_RING_PREFILL_ADMISSION_DYNAMIC_MIN_FREE_BLOCKS:-0}}"
VIRTUAL_CONTEXT_RING_BW_STEP_BLOCKS="${VLLM_VIRTUAL_CONTEXT_RING_BW_STEP_BLOCKS:-${VLLM_VIRTUAL_CONTEXT_RING_PREFILL_ADMISSION_DYNAMIC_STEP_BLOCKS:-0}}"
VIRTUAL_CONTEXT_RING_BW_STEP_TOKENS="${VLLM_VIRTUAL_CONTEXT_RING_BW_STEP_TOKENS:-${VLLM_VIRTUAL_CONTEXT_RING_PREFILL_ADMISSION_DYNAMIC_STEP_TOKENS:-0}}"
VIRTUAL_CONTEXT_RING_BW_POLICY="${VLLM_VIRTUAL_CONTEXT_RING_BW_POLICY:-step}"
VIRTUAL_CONTEXT_RING_BW_SAFETY_BLOCKS="${VLLM_VIRTUAL_CONTEXT_RING_BW_SAFETY_BLOCKS:-0}"
VIRTUAL_CONTEXT_RING_BW_DECODE_GROWTH_BLOCKS="${VLLM_VIRTUAL_CONTEXT_RING_BW_DECODE_GROWTH_BLOCKS:-0}"
VIRTUAL_CONTEXT_RING_MIN_PROGRESS_TOKENS="${VLLM_VIRTUAL_CONTEXT_RING_MIN_PROGRESS_TOKENS:-1024}"
VIRTUAL_CONTEXT_RING_BACKEND_FORCE_ALL="${VLLM_VIRTUAL_CONTEXT_RING_BACKEND_FORCE_ALL:-0}"
VIRTUAL_CONTEXT_PREFILL_RING_REQUIRED="${VLLM_VIRTUAL_CONTEXT_PREFILL_RING_REQUIRED:-0}"
VIRTUAL_CONTEXT_PREFILL_RING_REQUIRED_MODE="${VLLM_VIRTUAL_CONTEXT_PREFILL_RING_REQUIRED_MODE:-never}"
VIRTUAL_CONTEXT_RING_ASYNC_PREFETCH="${VLLM_VIRTUAL_CONTEXT_RING_ASYNC_PREFETCH:-0}"
VIRTUAL_CONTEXT_RING_CUDA_EVENT_TIMING="${VLLM_VIRTUAL_CONTEXT_RING_CUDA_EVENT_TIMING:-0}"
CONTEXT_RING_INTERFERENCE_TRACE_ENABLED="${VLLM_CONTEXT_RING_INTERFERENCE_TRACE_ENABLED:-0}"
CONTEXT_RING_NVTX_TRACE_ENABLED="${VLLM_CONTEXT_RING_NVTX_TRACE_ENABLED:-0}"
VIRTUAL_CONTEXT_RING_ATTENTION_PLAN_TENSORS="${VLLM_VIRTUAL_CONTEXT_RING_ATTENTION_PLAN_TENSORS:-0}"
VIRTUAL_CONTEXT_RING_PREFETCH_DISTANCE="${VLLM_VIRTUAL_CONTEXT_RING_PREFETCH_DISTANCE:-1}"
VIRTUAL_CONTEXT_RING_STAGING_SLOTS="${VLLM_VIRTUAL_CONTEXT_RING_STAGING_SLOTS:-2}"
VIRTUAL_CONTEXT_RING_STAGING_DIRECT_MULTICOPY="${VLLM_VIRTUAL_CONTEXT_RING_STAGING_DIRECT_MULTICOPY:-0}"
VIRTUAL_CONTEXT_RING_STAGING_DIRECT_MULTICOPY_STRICT="${VLLM_VIRTUAL_CONTEXT_RING_STAGING_DIRECT_MULTICOPY_STRICT:-0}"
VIRTUAL_CONTEXT_DIRECT_MULTICOPY_STRICT="${VLLM_VIRTUAL_CONTEXT_DIRECT_MULTICOPY_STRICT:-0}"
VIRTUAL_CONTEXT_RING_COMPUTE_GROUP_TOKENS="${VLLM_VIRTUAL_CONTEXT_RING_COMPUTE_GROUP_TOKENS:-0}"
VIRTUAL_CONTEXT_RING_STREAMING_WAVE_GROUPS="${VLLM_VIRTUAL_CONTEXT_RING_STREAMING_WAVE_GROUPS:-1}"
VIRTUAL_CONTEXT_RING_Q_TILE_TOKENS="${VLLM_VIRTUAL_CONTEXT_RING_Q_TILE_TOKENS:-0}"
VIRTUAL_CONTEXT_RING_PRESSURE_BOOTSTRAP_TOKENS="${VLLM_VIRTUAL_CONTEXT_RING_PRESSURE_BOOTSTRAP_TOKENS:-0}"
VIRTUAL_CONTEXT_RING_TRITON_FUSED_UPDATE="${VLLM_VIRTUAL_CONTEXT_RING_TRITON_FUSED_UPDATE:-0}"
VIRTUAL_CONTEXT_RING_NATIVE_REDUCTION="${VLLM_VIRTUAL_CONTEXT_RING_NATIVE_REDUCTION:-0}"
VIRTUAL_CONTEXT_RING_TRITON_BLOCK_M="${VLLM_VIRTUAL_CONTEXT_RING_TRITON_BLOCK_M:-16}"
VIRTUAL_CONTEXT_RING_TRITON_BLOCK_N="${VLLM_VIRTUAL_CONTEXT_RING_TRITON_BLOCK_N:-64}"
VIRTUAL_CONTEXT_RING_BATCHED_PARTIAL="${VLLM_VIRTUAL_CONTEXT_RING_BATCHED_PARTIAL:-0}"
VIRTUAL_CONTEXT_RING_BATCHED_PARTIAL_STRICT="${VLLM_VIRTUAL_CONTEXT_RING_BATCHED_PARTIAL_STRICT:-0}"
VIRTUAL_CONTEXT_RING_BATCHED_WORKSPACE_BYTES="${VLLM_VIRTUAL_CONTEXT_RING_BATCHED_WORKSPACE_BYTES:-1073741824}"
VIRTUAL_CONTEXT_RING_PAGED_HBM_UPDATE="${VLLM_VIRTUAL_CONTEXT_RING_PAGED_HBM_UPDATE:-0}"
VIRTUAL_CONTEXT_RING_FULLY_OPTIMIZED_STRICT="${VLLM_VIRTUAL_CONTEXT_RING_FULLY_OPTIMIZED_STRICT:-0}"
VIRTUAL_CONTEXT_RING_BULK_COPY_TOKENS="${VLLM_VIRTUAL_CONTEXT_RING_BULK_COPY_TOKENS:-0}"
VIRTUAL_CONTEXT_DRAM_REGISTER_BULK_TOKENS="${VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_BULK_TOKENS:-0}"
VIRTUAL_CONTEXT_DRAM_PROVIDER_PIN_MEMORY="${VLLM_VIRTUAL_CONTEXT_DRAM_PROVIDER_PIN_MEMORY:-0}"
VIRTUAL_CONTEXT_DRAM_REGISTER_ASYNC="${VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_ASYNC:-0}"
VIRTUAL_CONTEXT_DRAM_REGISTER_PRESSURE_ONLY="${VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_PRESSURE_ONLY:-0}"
VIRTUAL_CONTEXT_NATIVE_DECODE_CONTROL_FAST_PATH="${VLLM_VIRTUAL_CONTEXT_NATIVE_DECODE_CONTROL_FAST_PATH:-0}"
VIRTUAL_CONTEXT_DELTA_HBM_OBSERVE="${VLLM_VIRTUAL_CONTEXT_DELTA_HBM_OBSERVE:-0}"
VIRTUAL_CONTEXT_DRAM_REGISTER_ASYNC_WORKERS="${VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_ASYNC_WORKERS:-1}"
VIRTUAL_CONTEXT_DRAM_REGISTER_ASYNC_MAX_PENDING_PAGES="${VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_ASYNC_MAX_PENDING_PAGES:-4096}"
VIRTUAL_CONTEXT_DRAM_REGISTER_CONTROL_DISPATCHER="${VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_CONTROL_DISPATCHER:-0}"
VIRTUAL_CONTEXT_DRAM_REGISTER_CONTROL_DISPATCHER_WORKERS="${VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_CONTROL_DISPATCHER_WORKERS:-1}"
VIRTUAL_CONTEXT_D2H_SOURCE_PLAN="${VLLM_VIRTUAL_CONTEXT_D2H_SOURCE_PLAN:-0}"
VIRTUAL_CONTEXT_D2H_SOURCE_BLOCK_LEASE="${VLLM_VIRTUAL_CONTEXT_D2H_SOURCE_BLOCK_LEASE:-0}"
VIRTUAL_CONTEXT_D2H_SOURCE_VIEW_DISPATCH="${VLLM_VIRTUAL_CONTEXT_D2H_SOURCE_VIEW_DISPATCH:-0}"
VIRTUAL_CONTEXT_DRAM_PAGE_POOL="${VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL:-0}"
VIRTUAL_CONTEXT_DRAM_PAGE_POOL_MIRROR_DICT="${VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL_MIRROR_DICT:-1}"
VIRTUAL_CONTEXT_DRAM_PAGE_POOL_MAX_SLOTS="${VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL_MAX_SLOTS:-0}"
VIRTUAL_CONTEXT_DRAM_PAGE_POOL_CONTIGUOUS="${VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL_CONTIGUOUS:-0}"
VIRTUAL_CONTEXT_DRAM_PAGE_POOL_PAGE_TOKENS="${VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL_PAGE_TOKENS:-16}"
VIRTUAL_CONTEXT_DRAM_PAGE_POOL_SLAB_CHUNK_SLOTS="${VLLM_VIRTUAL_CONTEXT_DRAM_PAGE_POOL_SLAB_CHUNK_SLOTS:-4096}"
VIRTUAL_CONTEXT_LAYER_RELOAD_PREFETCH="${VLLM_VIRTUAL_CONTEXT_LAYER_RELOAD_PREFETCH:-0}"
VIRTUAL_CONTEXT_LAYER_RELOAD_PREFETCH_WORKERS="${VLLM_VIRTUAL_CONTEXT_LAYER_RELOAD_PREFETCH_WORKERS:-2}"
VIRTUAL_CONTEXT_RELOAD_TO_HBM="${VLLM_VIRTUAL_CONTEXT_RELOAD_TO_HBM:-0}"
VIRTUAL_CONTEXT_RELOAD_TO_HBM_MIN_FREE_BLOCKS="${VLLM_VIRTUAL_CONTEXT_RELOAD_TO_HBM_MIN_FREE_BLOCKS:-0}"
VIRTUAL_CONTEXT_RELOAD_BULK_TOKENS="${VLLM_VIRTUAL_CONTEXT_RELOAD_BULK_TOKENS:-0}"
VIRTUAL_CONTEXT_RELOAD_DIRECT_MULTICOPY="${VLLM_VIRTUAL_CONTEXT_RELOAD_DIRECT_MULTICOPY:-0}"
VIRTUAL_CONTEXT_RELOAD_STAGING_SCATTER="${VLLM_VIRTUAL_CONTEXT_RELOAD_STAGING_SCATTER:-0}"
VIRTUAL_CONTEXT_RELOAD_FUSED_SCATTER="${VLLM_VIRTUAL_CONTEXT_RELOAD_FUSED_SCATTER:-0}"
VIRTUAL_CONTEXT_RELOAD_FUSED_SCATTER_STRICT="${VLLM_VIRTUAL_CONTEXT_RELOAD_FUSED_SCATTER_STRICT:-0}"
VIRTUAL_CONTEXT_RELOAD_SCATTER_DESCRIPTOR_CACHE_ENTRIES="${VLLM_VIRTUAL_CONTEXT_RELOAD_SCATTER_DESCRIPTOR_CACHE_ENTRIES:-512}"
VIRTUAL_CONTEXT_DEBUG_SNAPSHOT_INTERVAL="${VLLM_VIRTUAL_CONTEXT_DEBUG_SNAPSHOT_INTERVAL:-1}"
CONTEXT_RING_CORRECTNESS_MODE="${VLLM_CONTEXT_RING_CORRECTNESS_MODE:-off}"
CONTEXT_RING_CORRECTNESS_DIR="${VLLM_CONTEXT_RING_CORRECTNESS_DIR:-}"
CONTEXT_RING_CORRECTNESS_PATHS="${VLLM_CONTEXT_RING_CORRECTNESS_PATHS:-native_hbm,context_ring}"
CONTEXT_RING_CORRECTNESS_PHASES="${VLLM_CONTEXT_RING_CORRECTNESS_PHASES:-prefill,decode}"
CONTEXT_RING_CORRECTNESS_LAYERS="${VLLM_CONTEXT_RING_CORRECTNESS_LAYERS:-*}"
CONTEXT_RING_CORRECTNESS_REQUESTS="${VLLM_CONTEXT_RING_CORRECTNESS_REQUESTS:-*}"
CONTEXT_RING_CORRECTNESS_FILE_CHUNK_TOKENS="${VLLM_CONTEXT_RING_CORRECTNESS_FILE_CHUNK_TOKENS:-2048}"
CONTEXT_RING_CORRECTNESS_MAX_RECORDS="${VLLM_CONTEXT_RING_CORRECTNESS_MAX_RECORDS:-0}"
CONTEXT_RING_CORRECTNESS_STRICT="${VLLM_CONTEXT_RING_CORRECTNESS_STRICT:-1}"
CONTEXT_RING_CORRECTNESS_CAPTURE_QKV="${VLLM_CONTEXT_RING_CORRECTNESS_CAPTURE_QKV:-0}"
OPMUX_ENABLED="${VLLM_OPMUX_ENABLED:-${VLLM_OPERATOR_MEMORY_SHAPING:-0}}"
OPMUX_TIMING="${VLLM_OPMUX_TIMING:-0}"
OPMUX_POLICY="${VLLM_OPMUX_POLICY:-profile}"
OPMUX_MEMORY_PRESSURE_MODE="${VLLM_OPMUX_MEMORY_PRESSURE_MODE:-low}"
OPMUX_SEGMENTED_KV_COMMIT_TOKENS="${VLLM_OPMUX_SEGMENTED_KV_COMMIT_TOKENS:-0}"
OPMUX_PREFILL_SPLIT_LIMIT="${VLLM_OPMUX_PREFILL_SPLIT_LIMIT:-0}"
OPMUX_TEMP_BUDGET_BYTES="${VLLM_OPMUX_TEMP_BUDGET_BYTES:-0}"
OPMUX_KV_COMMIT_BUDGET_TOKENS="${VLLM_OPMUX_KV_COMMIT_BUDGET_TOKENS:-0}"
OPMUX_STRICT="${VLLM_OPMUX_STRICT:-0}"
CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-0}"
SCHEDULING_POLICY="${VLLM_SCHEDULING_POLICY:-fcfs}"
ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
DISABLE_PREFIX_CACHING="${VLLM_DISABLE_PREFIX_CACHING:-1}"
EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
VENDOR_DIR="${VLLM_VENDOR_DIR:-}"
VLLM_EXECUTABLE="${VLLM_EXECUTABLE:-vllm}"

CHUNK_FLAG=""
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  CHUNK_FLAG="--enable-chunked-prefill"
fi

PREFIX_CACHE_FLAG=""
if [[ "${DISABLE_PREFIX_CACHING}" == "1" ]]; then
  PREFIX_CACHE_FLAG="--no-enable-prefix-caching"
fi

BLOCK_OVERRIDE_FLAG=""
if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE}" ]]; then
  BLOCK_OVERRIDE_FLAG="--num-gpu-blocks-override ${NUM_GPU_BLOCKS_OVERRIDE}"
fi

LAYER_KERNEL_EAGER_FLAG=""
CUDAGRAPH_CONFIG_ARGS=()
case "${CUDAGRAPH_MODE}" in
  "") ;;
  NONE|PIECEWISE|FULL|FULL_DECODE_ONLY|FULL_AND_PIECEWISE)
    if [[ "${EXTRA_ARGS}" == *"--compilation-config"* \
          || " ${EXTRA_ARGS} " == *" -cc "* \
          || " ${EXTRA_ARGS} " == *" -cc="* ]]; then
      echo "[ERROR] VLLM_CUDAGRAPH_MODE cannot be combined with a compilation config in VLLM_EXTRA_ARGS" >&2
      exit 1
    fi
    CUDAGRAPH_CONFIG_ARGS=(
      --compilation-config
      "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\"}"
    )
    ;;
  *)
    echo "[ERROR] unsupported VLLM_CUDAGRAPH_MODE=${CUDAGRAPH_MODE}" >&2
    exit 1
    ;;
esac
if [[ "${PD_GREEN_DECODE_CUDAGRAPH}" == "1" ]]; then
  if [[ "${LAYER_KERNEL_TIMING}" == "1" || -n "${PD_DECODE_GPU_COMPONENT_TRACE_JSONL}" ]]; then
    echo "[ERROR] GreenContext Decode CUDA graph cannot run with module/component hooks" >&2
    exit 1
  fi
  if [[ "${OPERATOR_SELECTIVE_GREEN_CONTEXT}" != "1" || "${OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_MODEL}" != "1" || "${PD_PHASE_LOCAL_MODEL_RUNNER}" != "1" ]]; then
    echo "[ERROR] GreenContext Decode CUDA graph requires GreenContext execute-model routing and phase-local buffers" >&2
    exit 1
  fi
  if [[ "${EXTRA_ARGS}" == *"--enforce-eager"* ]]; then
    echo "[ERROR] GreenContext Decode CUDA graph cannot be combined with --enforce-eager" >&2
    exit 1
  fi
  if [[ "${CUDAGRAPH_MODE}" == "NONE" \
        || "${CUDAGRAPH_MODE}" == "PIECEWISE" ]]; then
    echo "[ERROR] GreenContext Decode CUDA graph requires full Decode capture" >&2
    exit 1
  fi
fi
MATCHED_GRAPH_ONLY="${ISOLATION_MATCHED_COMPARISON:-0}"
if [[ "${MATCHED_GRAPH_ONLY}" == "1" && "${EXTRA_ARGS}" == *"--enforce-eager"* ]]; then
  echo "[ERROR] matched Graph comparison cannot silently disable torch.compile with --enforce-eager" >&2
  exit 1
fi
if [[ "${EXTRA_ARGS}" != *"--enforce-eager"* ]]; then
  # Matched P/D already selects eager Llama at the model boundary for live
  # layer handoff/workspace use. Disabling outer capture does not require
  # disabling compilation globally. Non-matched/diagnostic behavior is kept.
  if [[ "${LAYER_KERNEL_TIMING}" == "1" || -n "${PD_DECODE_GPU_COMPONENT_TRACE_JSONL}" || "${OPERATOR_SELECTIVE_PD}" == "1" || ( "${OPERATOR_SELECTIVE_GREEN_CONTEXT}" == "1" && "${PD_GREEN_DECODE_CUDAGRAPH}" != "1" && "${MATCHED_GRAPH_ONLY}" != "1" ) || "${MUXWISE_PHASE_BASELINE}" == "1" || ( "${MUXWISE_PHASE_GREEN_CONTEXT}" == "1" && "${PD_GREEN_DECODE_CUDAGRAPH}" != "1" && "${MATCHED_GRAPH_ONLY}" != "1" ) || "${OPMUX_ENABLED}" == "1" ]]; then
    LAYER_KERNEL_EAGER_FLAG="--enforce-eager"
  fi
fi
if [[ ( "${MATCHED_GRAPH_ONLY}" == "1" || "${PD_GREEN_DECODE_CUDAGRAPH}" == "1" \
        || ( -n "${CUDAGRAPH_MODE}" && "${CUDAGRAPH_MODE}" != "NONE" ) ) \
      && -n "${LAYER_KERNEL_EAGER_FLAG}" ]]; then
  echo "[ERROR] CUDA Graph requested, but an active diagnostic path requires --enforce-eager" >&2
  exit 1
fi

echo "[start_vllm_causality] model=${MODEL} port=${PORT}"
echo "[start_vllm_causality] dtype=${DTYPE}"
echo "[start_vllm_causality] kv_cache_dtype=${KV_CACHE_DTYPE}"
echo "[start_vllm_causality] block_size=${BLOCK_SIZE}"
echo "[start_vllm_causality] max_model_len=${MAX_MODEL_LEN}"
echo "[start_vllm_causality] max_num_seqs=${MAX_NUM_SEQS}"
echo "[start_vllm_causality] max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
echo "[start_vllm_causality] num_gpu_blocks_override=${NUM_GPU_BLOCKS_OVERRIDE}"
echo "[start_vllm_causality] tenant_kv_min_blocks=${TENANT_KV_MIN_BLOCKS}"
echo "[start_vllm_causality] tenant_kv_protection_mode=${TENANT_KV_PROTECTION_MODE}"
echo "[start_vllm_causality] iteration_gpu_timing=${ITERATION_GPU_TIMING}"
echo "[start_vllm_causality] layer_kernel_timing=${LAYER_KERNEL_TIMING}"
echo "[start_vllm_causality] pd_decode_gpu_component_trace_jsonl=${PD_DECODE_GPU_COMPONENT_TRACE_JSONL:-off}"
echo "[start_vllm_causality] pd_decode_gpu_component_trace_max_pending=${PD_DECODE_GPU_COMPONENT_TRACE_MAX_PENDING}"
echo "[start_vllm_causality] pd_decode_gpu_component_trace_every_n=${PD_DECODE_GPU_COMPONENT_TRACE_EVERY_N}"
echo "[start_vllm_causality] pd_decode_gpu_component_trace_include_layers=${PD_DECODE_GPU_COMPONENT_TRACE_INCLUDE_LAYERS}"
echo "[start_vllm_causality] pd_green_decode_cudagraph=${PD_GREEN_DECODE_CUDAGRAPH}"
echo "[start_vllm_causality] pd_green_decode_cudagraph_strict=${PD_GREEN_DECODE_CUDAGRAPH_STRICT}"
echo "[start_vllm_causality] cudagraph_mode=${CUDAGRAPH_MODE:-vllm_default}"
echo "[start_vllm_causality] operator_selective_pd=${OPERATOR_SELECTIVE_PD}"
echo "[start_vllm_causality] operator_selective_pd_requested=${OPERATOR_SELECTIVE_PD_REQUESTED}"
echo "[start_vllm_causality] operator_selective_pd_timing=${OPERATOR_SELECTIVE_PD_TIMING}"
echo "[start_vllm_causality] operator_selective_pd_green_context=${OPERATOR_SELECTIVE_GREEN_CONTEXT}"
echo "[start_vllm_causality] operator_selective_pd_green_overlap=${OPERATOR_SELECTIVE_PD_GREEN_OVERLAP}"
echo "[start_vllm_causality] operator_selective_pd_green_decode_sms=${OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS}"
echo "[start_vllm_causality] operator_selective_pd_green_prefill_sms=${OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS}"
echo "[start_vllm_causality] operator_selective_pd_green_strict=${OPERATOR_SELECTIVE_PD_GREEN_STRICT}"
echo "[start_vllm_causality] operator_selective_pd_green_execute_model=${OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_MODEL}"
echo "[start_vllm_causality] operator_selective_pd_green_execute_no_join=${OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_NO_JOIN}"
echo "[start_vllm_causality] muxwise_phase_baseline=${MUXWISE_PHASE_BASELINE}"
echo "[start_vllm_causality] muxwise_phase_green_context=${MUXWISE_PHASE_GREEN_CONTEXT}"
echo "[start_vllm_causality] pd_multiplexing_baseline=${PD_MULTIPLEXING_BASELINE}"
echo "[start_vllm_causality] pd_decoupled_no_mix=${PD_DECOUPLED_NO_MIX}"
echo "[start_vllm_causality] pd_independent_phase_queues=${PD_INDEPENDENT_PHASE_QUEUES}"
echo "[start_vllm_causality] pd_separate_schedulers=${PD_SEPARATE_SCHEDULERS}"
echo "[start_vllm_causality] pd_dual_continuous_batching=${PD_DUAL_CONTINUOUS_BATCHING}"
echo "[start_vllm_causality] pd_decode_reserved_batches=${PD_DECODE_RESERVED_BATCHES}"
echo "[start_vllm_causality] pd_nonblocking_phase_recovery=${PD_NONBLOCKING_PHASE_RECOVERY}"
echo "[start_vllm_causality] pd_single_inflight_per_phase=${PD_SINGLE_INFLIGHT_PER_PHASE}"
echo "[start_vllm_causality] pd_muxwise_event_loop=${PD_MUXWISE_EVENT_LOOP}"
echo "[start_vllm_causality] pd_paired_phase_launch=${PD_PAIRED_PHASE_LAUNCH}"
echo "[start_vllm_causality] pd_decoupled_paired_recovery=${PD_DECOUPLED_PAIRED_RECOVERY}"
echo "[start_vllm_causality] pd_phase_local_model_runner=${PD_PHASE_LOCAL_MODEL_RUNNER}"
echo "[start_vllm_causality] pd_phase_sync_sample_output=${PD_PHASE_SYNC_SAMPLE_OUTPUT}"
echo "[start_vllm_causality] pd_phase_sampler_debug_sync=${PD_PHASE_SAMPLER_DEBUG_SYNC}"
echo "[start_vllm_causality] pd_async_uniproc_executor=${PD_ASYNC_UNIPROC_EXECUTOR}"
echo "[start_vllm_causality] pd_async_uniproc_workers=${PD_ASYNC_UNIPROC_WORKERS}"
echo "[start_vllm_causality] pd_async_uniproc_phase_workers=${PD_ASYNC_UNIPROC_PHASE_WORKERS}"
echo "[start_vllm_causality] pd_async_uniproc_workers_per_phase=${PD_ASYNC_UNIPROC_WORKERS_PER_PHASE}"
echo "[start_vllm_causality] pd_async_uniproc_concurrent_phase_submit=${PD_ASYNC_UNIPROC_CONCURRENT_PHASE_SUBMIT}"
echo "[start_vllm_causality] pd_fused_execute_sample=${PD_FUSED_EXECUTE_SAMPLE}"
echo "[start_vllm_causality] pd_layerwise_profile_handoff=${PD_LAYERWISE_PROFILE_HANDOFF}"
echo "[start_vllm_causality] pd_serialized_full_sm=${PD_SERIALIZED_FULL_SM}"
echo "[start_vllm_causality] pd_wait_on_prefill_kv_capacity=${PD_WAIT_ON_PREFILL_KV_CAPACITY}"
echo "[start_vllm_causality] pd_eval_policy=${VLLM_PD_EVAL_POLICY_NORMALIZED:-manual}"
echo "[start_vllm_causality] pd_eval_policy_status=${VLLM_PD_EVAL_POLICY_STATUS:-manual}"
echo "[start_vllm_causality] pd_eval_multiplexing_mode=${VLLM_PD_EVAL_MULTIPLEXING_MODE:-manual}"
echo "[start_vllm_causality] pd_eval_context_mode=${VLLM_PD_EVAL_CONTEXT_MODE:-manual}"
echo "[start_vllm_causality] pd_eval_slo_aware=${VLLM_PD_EVAL_SLO_AWARE:-0}"
echo "[start_vllm_causality] virtual_context_ring_prefill_admission_tokens=${VIRTUAL_CONTEXT_RING_PREFILL_ADMISSION_TOKENS}"
echo "[start_vllm_causality] virtual_context_ring_prefill_admission_tokens_max=${VIRTUAL_CONTEXT_RING_PREFILL_ADMISSION_TOKENS_MAX}"
echo "[start_vllm_causality] virtual_context_ring_bw_dynamic=${VIRTUAL_CONTEXT_RING_BW_DYNAMIC}"
echo "[start_vllm_causality] virtual_context_ring_bw_min_tokens=${VIRTUAL_CONTEXT_RING_BW_MIN_TOKENS}"
echo "[start_vllm_causality] virtual_context_ring_bw_max_tokens=${VIRTUAL_CONTEXT_RING_BW_MAX_TOKENS}"
echo "[start_vllm_causality] virtual_context_ring_bw_low_free_blocks=${VIRTUAL_CONTEXT_RING_BW_LOW_FREE_BLOCKS}"
echo "[start_vllm_causality] virtual_context_ring_bw_step_blocks=${VIRTUAL_CONTEXT_RING_BW_STEP_BLOCKS}"
echo "[start_vllm_causality] virtual_context_ring_bw_step_tokens=${VIRTUAL_CONTEXT_RING_BW_STEP_TOKENS}"
echo "[start_vllm_causality] virtual_context_ring_bw_policy=${VIRTUAL_CONTEXT_RING_BW_POLICY}"
echo "[start_vllm_causality] virtual_context_ring_bw_safety_blocks=${VIRTUAL_CONTEXT_RING_BW_SAFETY_BLOCKS}"
echo "[start_vllm_causality] virtual_context_ring_bw_decode_growth_blocks=${VIRTUAL_CONTEXT_RING_BW_DECODE_GROWTH_BLOCKS}"
echo "[start_vllm_causality] virtual_context_ring_min_progress_tokens=${VIRTUAL_CONTEXT_RING_MIN_PROGRESS_TOKENS}"
echo "[start_vllm_causality] pd_ring_min_prefill_tokens_with_decode=${VLLM_PD_RING_MIN_PREFILL_TOKENS_WITH_DECODE:-0}"
echo "[start_vllm_causality] virtual_context_active_decode_native_prefill_tokens=${VLLM_VIRTUAL_CONTEXT_ACTIVE_DECODE_NATIVE_PREFILL_TOKENS:-0}"
echo "[start_vllm_causality] virtual_context_active_decode_ring_prefill_tokens=${VLLM_VIRTUAL_CONTEXT_ACTIVE_DECODE_RING_PREFILL_TOKENS:-0}"
echo "[start_vllm_causality] virtual_context_ring_backend_force_all=${VIRTUAL_CONTEXT_RING_BACKEND_FORCE_ALL}"
echo "[start_vllm_causality] virtual_context_prefill_ring_required=${VIRTUAL_CONTEXT_PREFILL_RING_REQUIRED}"
echo "[start_vllm_causality] virtual_context_prefill_ring_required_mode=${VIRTUAL_CONTEXT_PREFILL_RING_REQUIRED_MODE}"
echo "[start_vllm_causality] virtual_context_ring_async_prefetch=${VIRTUAL_CONTEXT_RING_ASYNC_PREFETCH}"
echo "[start_vllm_causality] virtual_context_ring_cuda_event_timing=${VIRTUAL_CONTEXT_RING_CUDA_EVENT_TIMING}"
echo "[start_vllm_causality] context_ring_interference_trace_enabled=${CONTEXT_RING_INTERFERENCE_TRACE_ENABLED}"
echo "[start_vllm_causality] context_ring_nvtx_trace_enabled=${CONTEXT_RING_NVTX_TRACE_ENABLED}"
echo "[start_vllm_causality] virtual_context_ring_attention_plan_tensors=${VIRTUAL_CONTEXT_RING_ATTENTION_PLAN_TENSORS}"
echo "[start_vllm_causality] virtual_context_ring_prefetch_distance=${VIRTUAL_CONTEXT_RING_PREFETCH_DISTANCE}"
echo "[start_vllm_causality] virtual_context_ring_staging_slots=${VIRTUAL_CONTEXT_RING_STAGING_SLOTS}"
echo "[start_vllm_causality] virtual_context_ring_staging_direct_multicopy=${VIRTUAL_CONTEXT_RING_STAGING_DIRECT_MULTICOPY}"
echo "[start_vllm_causality] virtual_context_ring_staging_direct_multicopy_strict=${VIRTUAL_CONTEXT_RING_STAGING_DIRECT_MULTICOPY_STRICT}"
echo "[start_vllm_causality] virtual_context_direct_multicopy_strict=${VIRTUAL_CONTEXT_DIRECT_MULTICOPY_STRICT}"
echo "[start_vllm_causality] virtual_context_ring_compute_group_tokens=${VIRTUAL_CONTEXT_RING_COMPUTE_GROUP_TOKENS}"
echo "[start_vllm_causality] virtual_context_ring_streaming_wave_groups=${VIRTUAL_CONTEXT_RING_STREAMING_WAVE_GROUPS}"
echo "[start_vllm_causality] virtual_context_ring_q_tile_tokens=${VIRTUAL_CONTEXT_RING_Q_TILE_TOKENS}"
echo "[start_vllm_causality] virtual_context_ring_pressure_bootstrap_tokens=${VIRTUAL_CONTEXT_RING_PRESSURE_BOOTSTRAP_TOKENS}"
echo "[start_vllm_causality] virtual_context_ring_triton_fused_update=${VIRTUAL_CONTEXT_RING_TRITON_FUSED_UPDATE}"
echo "[start_vllm_causality] virtual_context_ring_native_reduction=${VIRTUAL_CONTEXT_RING_NATIVE_REDUCTION}"
echo "[start_vllm_causality] virtual_context_ring_triton_block_m=${VIRTUAL_CONTEXT_RING_TRITON_BLOCK_M}"
echo "[start_vllm_causality] virtual_context_ring_triton_block_n=${VIRTUAL_CONTEXT_RING_TRITON_BLOCK_N}"
echo "[start_vllm_causality] virtual_context_ring_batched_partial=${VIRTUAL_CONTEXT_RING_BATCHED_PARTIAL}"
echo "[start_vllm_causality] virtual_context_ring_batched_partial_strict=${VIRTUAL_CONTEXT_RING_BATCHED_PARTIAL_STRICT}"
echo "[start_vllm_causality] virtual_context_ring_batched_workspace_bytes=${VIRTUAL_CONTEXT_RING_BATCHED_WORKSPACE_BYTES}"
echo "[start_vllm_causality] virtual_context_ring_paged_hbm_update=${VIRTUAL_CONTEXT_RING_PAGED_HBM_UPDATE}"
echo "[start_vllm_causality] virtual_context_ring_fully_optimized_strict=${VIRTUAL_CONTEXT_RING_FULLY_OPTIMIZED_STRICT}"
echo "[start_vllm_causality] virtual_context_ring_bulk_copy_tokens=${VIRTUAL_CONTEXT_RING_BULK_COPY_TOKENS}"
echo "[start_vllm_causality] virtual_context_dram_register_bulk_tokens=${VIRTUAL_CONTEXT_DRAM_REGISTER_BULK_TOKENS}"
echo "[start_vllm_causality] virtual_context_dram_provider_pin_memory=${VIRTUAL_CONTEXT_DRAM_PROVIDER_PIN_MEMORY}"
echo "[start_vllm_causality] virtual_context_dram_register_async=${VIRTUAL_CONTEXT_DRAM_REGISTER_ASYNC}"
echo "[start_vllm_causality] virtual_context_dram_register_pressure_only=${VIRTUAL_CONTEXT_DRAM_REGISTER_PRESSURE_ONLY}"
echo "[start_vllm_causality] virtual_context_native_decode_control_fast_path=${VIRTUAL_CONTEXT_NATIVE_DECODE_CONTROL_FAST_PATH}"
echo "[start_vllm_causality] virtual_context_delta_hbm_observe=${VIRTUAL_CONTEXT_DELTA_HBM_OBSERVE}"
echo "[start_vllm_causality] virtual_context_dram_register_async_workers=${VIRTUAL_CONTEXT_DRAM_REGISTER_ASYNC_WORKERS}"
echo "[start_vllm_causality] virtual_context_dram_register_async_max_pending_pages=${VIRTUAL_CONTEXT_DRAM_REGISTER_ASYNC_MAX_PENDING_PAGES}"
echo "[start_vllm_causality] virtual_context_dram_register_control_dispatcher=${VIRTUAL_CONTEXT_DRAM_REGISTER_CONTROL_DISPATCHER}"
echo "[start_vllm_causality] virtual_context_dram_register_control_dispatcher_workers=${VIRTUAL_CONTEXT_DRAM_REGISTER_CONTROL_DISPATCHER_WORKERS}"
echo "[start_vllm_causality] virtual_context_d2h_source_plan=${VIRTUAL_CONTEXT_D2H_SOURCE_PLAN}"
echo "[start_vllm_causality] virtual_context_d2h_source_block_lease=${VIRTUAL_CONTEXT_D2H_SOURCE_BLOCK_LEASE}"
echo "[start_vllm_causality] virtual_context_d2h_source_view_dispatch=${VIRTUAL_CONTEXT_D2H_SOURCE_VIEW_DISPATCH}"
echo "[start_vllm_causality] virtual_context_dram_page_pool=${VIRTUAL_CONTEXT_DRAM_PAGE_POOL}"
echo "[start_vllm_causality] virtual_context_dram_page_pool_mirror_dict=${VIRTUAL_CONTEXT_DRAM_PAGE_POOL_MIRROR_DICT}"
echo "[start_vllm_causality] virtual_context_dram_page_pool_max_slots=${VIRTUAL_CONTEXT_DRAM_PAGE_POOL_MAX_SLOTS}"
echo "[start_vllm_causality] virtual_context_dram_page_pool_contiguous=${VIRTUAL_CONTEXT_DRAM_PAGE_POOL_CONTIGUOUS}"
echo "[start_vllm_causality] virtual_context_dram_page_pool_page_tokens=${VIRTUAL_CONTEXT_DRAM_PAGE_POOL_PAGE_TOKENS}"
echo "[start_vllm_causality] virtual_context_dram_page_pool_slab_chunk_slots=${VIRTUAL_CONTEXT_DRAM_PAGE_POOL_SLAB_CHUNK_SLOTS}"
echo "[start_vllm_causality] virtual_context_layer_reload_prefetch=${VIRTUAL_CONTEXT_LAYER_RELOAD_PREFETCH}"
echo "[start_vllm_causality] virtual_context_layer_reload_prefetch_workers=${VIRTUAL_CONTEXT_LAYER_RELOAD_PREFETCH_WORKERS}"
echo "[start_vllm_causality] virtual_context_reload_to_hbm=${VIRTUAL_CONTEXT_RELOAD_TO_HBM}"
echo "[start_vllm_causality] virtual_context_reload_to_hbm_min_free_blocks=${VIRTUAL_CONTEXT_RELOAD_TO_HBM_MIN_FREE_BLOCKS}"
echo "[start_vllm_causality] virtual_context_reload_bulk_tokens=${VIRTUAL_CONTEXT_RELOAD_BULK_TOKENS}"
echo "[start_vllm_causality] virtual_context_reload_direct_multicopy=${VIRTUAL_CONTEXT_RELOAD_DIRECT_MULTICOPY}"
echo "[start_vllm_causality] virtual_context_reload_staging_scatter=${VIRTUAL_CONTEXT_RELOAD_STAGING_SCATTER}"
echo "[start_vllm_causality] virtual_context_reload_fused_scatter=${VIRTUAL_CONTEXT_RELOAD_FUSED_SCATTER}"
echo "[start_vllm_causality] virtual_context_reload_fused_scatter_strict=${VIRTUAL_CONTEXT_RELOAD_FUSED_SCATTER_STRICT}"
echo "[start_vllm_causality] virtual_context_reload_scatter_descriptor_cache_entries=${VIRTUAL_CONTEXT_RELOAD_SCATTER_DESCRIPTOR_CACHE_ENTRIES}"
echo "[start_vllm_causality] virtual_context_debug_snapshot_interval=${VIRTUAL_CONTEXT_DEBUG_SNAPSHOT_INTERVAL}"
echo "[start_vllm_causality] context_ring_correctness_mode=${CONTEXT_RING_CORRECTNESS_MODE}"
echo "[start_vllm_causality] context_ring_correctness_dir=${CONTEXT_RING_CORRECTNESS_DIR}"
echo "[start_vllm_causality] context_ring_correctness_paths=${CONTEXT_RING_CORRECTNESS_PATHS}"
echo "[start_vllm_causality] context_ring_correctness_phases=${CONTEXT_RING_CORRECTNESS_PHASES}"
echo "[start_vllm_causality] context_ring_correctness_layers=${CONTEXT_RING_CORRECTNESS_LAYERS}"
echo "[start_vllm_causality] context_ring_correctness_requests=${CONTEXT_RING_CORRECTNESS_REQUESTS}"
echo "[start_vllm_causality] context_ring_correctness_file_chunk_tokens=${CONTEXT_RING_CORRECTNESS_FILE_CHUNK_TOKENS}"
echo "[start_vllm_causality] context_ring_correctness_max_records=${CONTEXT_RING_CORRECTNESS_MAX_RECORDS}"
echo "[start_vllm_causality] context_ring_correctness_strict=${CONTEXT_RING_CORRECTNESS_STRICT}"
echo "[start_vllm_causality] context_ring_correctness_capture_qkv=${CONTEXT_RING_CORRECTNESS_CAPTURE_QKV}"
echo "[start_vllm_causality] opmux_enabled=${OPMUX_ENABLED}"
echo "[start_vllm_causality] opmux_timing=${OPMUX_TIMING}"
echo "[start_vllm_causality] opmux_policy=${OPMUX_POLICY}"
echo "[start_vllm_causality] opmux_memory_pressure_mode=${OPMUX_MEMORY_PRESSURE_MODE}"
echo "[start_vllm_causality] opmux_segmented_kv_commit_tokens=${OPMUX_SEGMENTED_KV_COMMIT_TOKENS}"
echo "[start_vllm_causality] opmux_prefill_split_limit=${OPMUX_PREFILL_SPLIT_LIMIT}"
echo "[start_vllm_causality] opmux_temp_budget_bytes=${OPMUX_TEMP_BUDGET_BYTES}"
echo "[start_vllm_causality] opmux_kv_commit_budget_tokens=${OPMUX_KV_COMMIT_BUDGET_TOKENS}"
echo "[start_vllm_causality] opmux_strict=${OPMUX_STRICT}"
echo "[start_vllm_causality] cpu_offload_gb=${CPU_OFFLOAD_GB}"
echo "[start_vllm_causality] scheduling_policy=${SCHEDULING_POLICY}"
echo "[start_vllm_causality] chunked_prefill=${ENABLE_CHUNKED_PREFILL}"
echo "[start_vllm_causality] prefill_admission=native_budget_fill"
echo "[start_vllm_causality] long_prefill_token_threshold=${LONG_PREFILL_TOKEN_THRESHOLD}"
echo "[start_vllm_causality] disable_prefix_caching=${DISABLE_PREFIX_CACHING}"
echo "[start_vllm_causality] layer_kernel_eager_flag=${LAYER_KERNEL_EAGER_FLAG}"
echo "[start_vllm_causality] extra_args=${EXTRA_ARGS}"
echo "[start_vllm_causality] vendor_dir=${VENDOR_DIR}"
echo "[start_vllm_causality] executable=${VLLM_EXECUTABLE}"

if [[ -n "${VENDOR_DIR}" ]]; then
  export PYTHONPATH="${VENDOR_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export VLLM_TENANT_KV_PROTECTION_MODE="${TENANT_KV_PROTECTION_MODE}"
export VLLM_TENANT_KV_MIN_BLOCKS="${TENANT_KV_MIN_BLOCKS}"
export VLLM_ITERATION_GPU_TIMING="${ITERATION_GPU_TIMING}"
export VLLM_LAYER_KERNEL_TIMING="${LAYER_KERNEL_TIMING}"
export VLLM_PD_DECODE_GPU_COMPONENT_TRACE_JSONL="${PD_DECODE_GPU_COMPONENT_TRACE_JSONL}"
export VLLM_PD_DECODE_GPU_COMPONENT_TRACE_MAX_PENDING="${PD_DECODE_GPU_COMPONENT_TRACE_MAX_PENDING}"
export VLLM_PD_DECODE_GPU_COMPONENT_TRACE_EVERY_N="${PD_DECODE_GPU_COMPONENT_TRACE_EVERY_N}"
export VLLM_PD_DECODE_GPU_COMPONENT_TRACE_INCLUDE_LAYERS="${PD_DECODE_GPU_COMPONENT_TRACE_INCLUDE_LAYERS}"
export VLLM_PD_GREEN_DECODE_CUDAGRAPH="${PD_GREEN_DECODE_CUDAGRAPH}"
export VLLM_PD_GREEN_DECODE_CUDAGRAPH_STRICT="${PD_GREEN_DECODE_CUDAGRAPH_STRICT}"
export VLLM_OPERATOR_SELECTIVE_PD="${OPERATOR_SELECTIVE_PD}"
export VLLM_OPERATOR_SELECTIVE_PD_TIMING="${OPERATOR_SELECTIVE_PD_TIMING}"
export VLLM_OPERATOR_SELECTIVE_PD_GREEN_CONTEXT="${OPERATOR_SELECTIVE_GREEN_CONTEXT}"
export VLLM_OPERATOR_SELECTIVE_PD_GREEN_OVERLAP="${OPERATOR_SELECTIVE_PD_GREEN_OVERLAP}"
export VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS="${OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS}"
export VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS="${OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS}"
export VLLM_OPERATOR_SELECTIVE_PD_GREEN_STRICT="${OPERATOR_SELECTIVE_PD_GREEN_STRICT}"
export VLLM_OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_MODEL="${OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_MODEL}"
export VLLM_OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_NO_JOIN="${OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_NO_JOIN}"
export VLLM_MUXWISE_PHASE_BASELINE="${MUXWISE_PHASE_BASELINE}"
export VLLM_MUXWISE_PHASE_BASELINE_GREEN_CONTEXT="${MUXWISE_PHASE_GREEN_CONTEXT}"
export VLLM_PD_MULTIPLEXING_BASELINE="${PD_MULTIPLEXING_BASELINE}"
export VLLM_PD_DECOUPLED_NO_MIX="${PD_DECOUPLED_NO_MIX}"
export VLLM_PD_INDEPENDENT_PHASE_QUEUES="${PD_INDEPENDENT_PHASE_QUEUES}"
export VLLM_PD_SEPARATE_SCHEDULERS="${PD_SEPARATE_SCHEDULERS}"
export VLLM_PD_DUAL_CONTINUOUS_BATCHING="${PD_DUAL_CONTINUOUS_BATCHING}"
export VLLM_PD_DECODE_RESERVED_BATCHES="${PD_DECODE_RESERVED_BATCHES}"
export VLLM_PD_NONBLOCKING_PHASE_RECOVERY="${PD_NONBLOCKING_PHASE_RECOVERY}"
export VLLM_PD_SINGLE_INFLIGHT_PER_PHASE="${PD_SINGLE_INFLIGHT_PER_PHASE}"
export VLLM_PD_MUXWISE_EVENT_LOOP="${PD_MUXWISE_EVENT_LOOP}"
export VLLM_PD_PAIRED_PHASE_LAUNCH="${PD_PAIRED_PHASE_LAUNCH}"
export VLLM_PD_DECOUPLED_PAIRED_RECOVERY="${PD_DECOUPLED_PAIRED_RECOVERY}"
export VLLM_PD_PHASE_LOCAL_MODEL_RUNNER="${PD_PHASE_LOCAL_MODEL_RUNNER}"
export VLLM_PD_PHASE_SYNC_SAMPLE_OUTPUT="${PD_PHASE_SYNC_SAMPLE_OUTPUT}"
export VLLM_PD_PHASE_SAMPLER_DEBUG_SYNC="${PD_PHASE_SAMPLER_DEBUG_SYNC}"
export VLLM_PD_ASYNC_UNIPROC_EXECUTOR="${PD_ASYNC_UNIPROC_EXECUTOR}"
export VLLM_PD_ASYNC_UNIPROC_WORKERS="${PD_ASYNC_UNIPROC_WORKERS}"
export VLLM_PD_ASYNC_UNIPROC_PHASE_WORKERS="${PD_ASYNC_UNIPROC_PHASE_WORKERS}"
export VLLM_PD_ASYNC_UNIPROC_WORKERS_PER_PHASE="${PD_ASYNC_UNIPROC_WORKERS_PER_PHASE}"
export VLLM_PD_ASYNC_UNIPROC_CONCURRENT_PHASE_SUBMIT="${PD_ASYNC_UNIPROC_CONCURRENT_PHASE_SUBMIT}"
export VLLM_PD_FUSED_EXECUTE_SAMPLE="${PD_FUSED_EXECUTE_SAMPLE}"
export VLLM_PD_LAYERWISE_PROFILE_HANDOFF="${PD_LAYERWISE_PROFILE_HANDOFF}"
export VLLM_PD_LAYERWISE_PROFILE_HANDOFF_STRICT="${PD_LAYERWISE_PROFILE_HANDOFF_STRICT}"
export VLLM_PD_SERIALIZED_FULL_SM="${PD_SERIALIZED_FULL_SM}"
export VLLM_PD_WAIT_ON_PREFILL_KV_CAPACITY="${PD_WAIT_ON_PREFILL_KV_CAPACITY}"
export VLLM_OPMUX_ENABLED="${OPMUX_ENABLED}"
export VLLM_OPERATOR_MEMORY_SHAPING="${OPMUX_ENABLED}"
export VLLM_OPMUX_TIMING="${OPMUX_TIMING}"
export VLLM_OPMUX_POLICY="${OPMUX_POLICY}"
export VLLM_OPMUX_MEMORY_PRESSURE_MODE="${OPMUX_MEMORY_PRESSURE_MODE}"
export VLLM_OPMUX_SEGMENTED_KV_COMMIT_TOKENS="${OPMUX_SEGMENTED_KV_COMMIT_TOKENS}"
export VLLM_OPMUX_PREFILL_SPLIT_LIMIT="${OPMUX_PREFILL_SPLIT_LIMIT}"
export VLLM_OPMUX_TEMP_BUDGET_BYTES="${OPMUX_TEMP_BUDGET_BYTES}"
export VLLM_OPMUX_KV_COMMIT_BUDGET_TOKENS="${OPMUX_KV_COMMIT_BUDGET_TOKENS}"
export VLLM_OPMUX_STRICT="${OPMUX_STRICT}"
export VLLM_CONTEXT_RING_INTERFERENCE_TRACE_ENABLED="${CONTEXT_RING_INTERFERENCE_TRACE_ENABLED}"
export VLLM_CONTEXT_RING_NVTX_TRACE_ENABLED="${CONTEXT_RING_NVTX_TRACE_ENABLED}"
export VLLM_CONTEXT_RING_CORRECTNESS_MODE="${CONTEXT_RING_CORRECTNESS_MODE}"
export VLLM_CONTEXT_RING_CORRECTNESS_DIR="${CONTEXT_RING_CORRECTNESS_DIR}"
export VLLM_CONTEXT_RING_CORRECTNESS_PATHS="${CONTEXT_RING_CORRECTNESS_PATHS}"
export VLLM_CONTEXT_RING_CORRECTNESS_PHASES="${CONTEXT_RING_CORRECTNESS_PHASES}"
export VLLM_CONTEXT_RING_CORRECTNESS_LAYERS="${CONTEXT_RING_CORRECTNESS_LAYERS}"
export VLLM_CONTEXT_RING_CORRECTNESS_REQUESTS="${CONTEXT_RING_CORRECTNESS_REQUESTS}"
export VLLM_CONTEXT_RING_CORRECTNESS_FILE_CHUNK_TOKENS="${CONTEXT_RING_CORRECTNESS_FILE_CHUNK_TOKENS}"
export VLLM_CONTEXT_RING_CORRECTNESS_MAX_RECORDS="${CONTEXT_RING_CORRECTNESS_MAX_RECORDS}"
export VLLM_CONTEXT_RING_CORRECTNESS_STRICT="${CONTEXT_RING_CORRECTNESS_STRICT}"
export VLLM_CONTEXT_RING_CORRECTNESS_CAPTURE_QKV="${CONTEXT_RING_CORRECTNESS_CAPTURE_QKV}"

exec "${VLLM_EXECUTABLE}" serve "${MODEL}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --block-size "${BLOCK_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}" \
  --cpu-offload-gb "${CPU_OFFLOAD_GB}" \
  --scheduling-policy "${SCHEDULING_POLICY}" \
  ${CHUNK_FLAG} \
  ${PREFIX_CACHE_FLAG} \
  ${BLOCK_OVERRIDE_FLAG} \
  ${LAYER_KERNEL_EAGER_FLAG} \
  "${CUDAGRAPH_CONFIG_ARGS[@]}" \
  ${EXTRA_ARGS}
