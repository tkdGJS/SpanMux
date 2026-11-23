#!/usr/bin/env bash
set -euo pipefail

# Prototype profiler:
#   1. build fixed request TTFT/TBT SLOs from a native 40-SM baseline;
#   2. run fixed P/D partition overlap cases;
#   3. emit the two JSON profiles consumed by vendor_v8.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-vendor_v8_partition_profile}"
RESULT_DIR="${ROOT_DIR}/${EXPERIMENT_NAME}_results"
LOG_DIR="${ROOT_DIR}/${EXPERIMENT_NAME}_logs"
ARTIFACT_DIR="${ROOT_DIR}/${EXPERIMENT_NAME}_artifacts"

VENV_DIR="${VENV_DIR:-/home/noslab-gpu/tkdgjs/tkdgjs}"
VLLM_EXECUTABLE="${VLLM_EXECUTABLE:-${VENV_DIR}/bin/vllm}"
VLLM_VENDOR_DIR="${VLLM_VENDOR_DIR:-${ROOT_DIR}/vendor_v8}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:${PORT}"

BASELINE_RESULTS_DIR="${BASELINE_RESULTS_DIR:-${ROOT_DIR}/last_exp/slo/native_40sm_profile_baseline_v1_results}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-25000 50000 75000 100000}"
GREEN_SM_SPLITS="${GREEN_SM_SPLITS:-8/32 16/24 20/20 24/16 32/8}"
PREFILL_MODES="${PREFILL_MODES:-native ring}"
ACTIVE_DECODE_REQUEST_COUNTS="${ACTIVE_DECODE_REQUEST_COUNTS:-1 2 3}"
DECODE_AUX_CONTEXT_TOKENS="${DECODE_AUX_CONTEXT_TOKENS:-10000}"
PREFILL_PROBE_CONTEXT_TOKENS="${PREFILL_PROBE_CONTEXT_TOKENS:-0}"
REPEAT_COUNT="${REPEAT_COUNT:-3}"
WARMUP_COUNT="${WARMUP_COUNT:-1}"
TTFT_SLO_MULTIPLIER="${TTFT_SLO_MULTIPLIER:-2.0}"
TBT_SLO_MULTIPLIER="${TBT_SLO_MULTIPLIER:-1.25}"
PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-2048}"
ANCHOR_OUTPUT_TOKENS="${ANCHOR_OUTPUT_TOKENS:-4096}"
PROBE_OUTPUT_TOKENS="${PROBE_OUTPUT_TOKENS:-8}"
PROBE_DELAY_SEC="${PROBE_DELAY_SEC:-0}"
DECODE_ONLY_CONTROL_GAPS="${DECODE_ONLY_CONTROL_GAPS:-0}"
SM_SCALING_BOOTSTRAP_ITERATIONS="${SM_SCALING_BOOTSTRAP_ITERATIONS:-10000}"
SM_SCALING_BOOTSTRAP_SEED="${SM_SCALING_BOOTSTRAP_SEED:-20260804}"
SM_SCALING_ALLOW_WINDOW_CLUSTERS="${SM_SCALING_ALLOW_WINDOW_CLUSTERS:-0}"
CASE_COOLDOWN_SEC="${CASE_COOLDOWN_SEC:-3}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-14400}"
FIRST_TOKEN_TIMEOUT_SEC="${FIRST_TOKEN_TIMEOUT_SEC:-7200}"
SERVER_READY_TIMEOUT_SEC="${SERVER_READY_TIMEOUT_SEC:-900}"
SERVER_SHUTDOWN_TIMEOUT_SEC="${SERVER_SHUTDOWN_TIMEOUT_SEC:-30}"
PORT_RELEASE_TIMEOUT_SEC="${PORT_RELEASE_TIMEOUT_SEC:-30}"

SHAREGPT_RAW_PATH="${SHAREGPT_RAW_PATH:-${ROOT_DIR}/data/ShareGPT_V3_unfiltered_cleaned_split.json}"
DATASET_JSONL="${DATASET_JSONL:-${RESULT_DIR}/experiment_snapshot/sharegpt_profile_requests.jsonl}"
DATASET_TOKENIZER="${DATASET_TOKENIZER:-${MODEL_NAME}}"
SHAREGPT_CANDIDATE_POOL_SIZE="${SHAREGPT_CANDIDATE_POOL_SIZE:-128}"
SHAREGPT_UNDERLENGTH_POLICY="${SHAREGPT_UNDERLENGTH_POLICY:-pack}"
PREPARE_DATASET="${PREPARE_DATASET:-1}"

KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-4294967296}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
# Submit every configured state by default. VLLM_MAX_MODEL_LEN is a
# per-request sequence bound, not an aggregate resident-KV capacity. A
# positive explicit override restores the legacy experiment-only prefilter.
NATIVE_PROFILE_MAX_RESIDENT_TOKENS="${NATIVE_PROFILE_MAX_RESIDENT_TOKENS:-0}"
EXPECTED_GPU_SMS="${EXPECTED_GPU_SMS:-40}"
BUILD_GREEN_BRIDGE="${BUILD_GREEN_BRIDGE:-auto}"
ENABLE_DMON="${ENABLE_DMON:-1}"
ENABLE_ITERATION_METRICS="${ENABLE_ITERATION_METRICS:-1}"
RUN_STANDARD_ANALYZERS="${RUN_STANDARD_ANALYZERS:-1}"
PD_CUDA_EVENT_TRACE_ENABLED="${VLLM_PD_CUDA_EVENT_TRACE_ENABLED:-0}"
DECODE_GPU_COMPONENT_TRACE="${VLLM_PD_DECODE_GPU_COMPONENT_TRACE:-0}"
DECODE_GPU_COMPONENT_TRACE_MAX_PENDING="${VLLM_PD_DECODE_GPU_COMPONENT_TRACE_MAX_PENDING:-8192}"
DECODE_GPU_COMPONENT_TRACE_EVERY_N="${VLLM_PD_DECODE_GPU_COMPONENT_TRACE_EVERY_N:-8}"
DECODE_GPU_COMPONENT_TRACE_INCLUDE_LAYERS="${VLLM_PD_DECODE_GPU_COMPONENT_TRACE_INCLUDE_LAYERS:-0}"
GREEN_DECODE_CUDAGRAPH="${VLLM_PD_GREEN_DECODE_CUDAGRAPH:-0}"
GREEN_DECODE_CUDAGRAPH_STRICT="${VLLM_PD_GREEN_DECODE_CUDAGRAPH_STRICT:-1}"
KILL_EXISTING_PORT_OWNER="${KILL_EXISTING_PORT_OWNER:-0}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./run_vendor_v8_partition_profiler.sh [--dry-run]

Important environment variables:
  BASELINE_RESULTS_DIR       Completed 40-SM baseline containing 25K..100K.
  GREEN_SM_SPLITS           decode/prefill splits, default five T4 splits.
  PREFILL_MODES             native and/or ring.
  ACTIVE_DECODE_REQUEST_COUNTS  Active decode cardinalities to profile.
  DECODE_AUX_CONTEXT_TOKENS Additional Decode-anchor context, default 10K.
  PREFILL_PROBE_CONTEXT_TOKENS Fixed Prefill probe context; 0 uses Decode context.
  DECODE_ONLY_CONTROL_GAPS Last pre-probe gaps per anchor used as paired control.
  SM_SCALING_BOOTSTRAP_ITERATIONS Cluster-bootstrap iterations, default 10000.
  SM_SCALING_ALLOW_WINDOW_CLUSTERS Allow descriptive same-anchor window analysis.
  FIRST_TOKEN_TIMEOUT_SEC    Decode-anchor first-token timeout, default 7200.
  NATIVE_PROFILE_MAX_RESIDENT_TOKENS  Optional aggregate-input prefilter;
                                default 0 disables it.
  REPEAT_COUNT              Measured overlap trials per state.
  VLLM_PD_DECODE_GPU_COMPONENT_TRACE  Set 1 to capture Decode GPU components.
  ENABLE_ITERATION_METRICS  Set 0 for client/CUDA-boundary-only microbenchmarks.
  RUN_STANDARD_ANALYZERS    Set 0 when an outer microbenchmark owns analysis.
  VLLM_PD_GREEN_DECODE_CUDAGRAPH  Set 1 for phase-slot GreenContext Decode graphs.
  SERVER_SHUTDOWN_TIMEOUT_SEC  Group TERM-to-KILL timeout, default 30.
  PORT_RELEASE_TIMEOUT_SEC  Listener release timeout, default 30.
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

read -r -a CONTEXT_ARRAY <<<"${CONTEXT_TOKENS}"
read -r -a SPLIT_ARRAY <<<"${GREEN_SM_SPLITS}"
read -r -a MODE_ARRAY <<<"${PREFILL_MODES}"
read -r -a DECODE_COUNT_ARRAY <<<"${ACTIVE_DECODE_REQUEST_COUNTS}"

ensure_file() {
  [[ -f "$1" ]] || {
    echo "[ERROR] missing $2: $1" >&2
    exit 1
  }
}

ensure_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]] || {
    echo "[ERROR] $2 must be a positive integer: $1" >&2
    exit 2
  }
}

ensure_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]] || {
    echo "[ERROR] $2 must be a non-negative integer: $1" >&2
    exit 2
  }
}

parse_split() {
  local split="$1"
  [[ "${split}" == */* ]] || {
    echo "[ERROR] split must be decode/prefill: ${split}" >&2
    return 2
  }
  DECODE_SMS="${split%%/*}"
  PREFILL_SMS="${split##*/}"
  ensure_positive_integer "${DECODE_SMS}" "decode SMs"
  ensure_positive_integer "${PREFILL_SMS}" "prefill SMs"
  ((DECODE_SMS + PREFILL_SMS == EXPECTED_GPU_SMS)) || {
    echo "[ERROR] split must use ${EXPECTED_GPU_SMS} SMs: ${split}" >&2
    return 2
  }
  PROFILE_ID="d${DECODE_SMS}_p${PREFILL_SMS}"
}

for value in "${CONTEXT_ARRAY[@]}"; do
  ensure_positive_integer "${value}" "context token bucket"
  ((value + ANCHOR_OUTPUT_TOKENS <= VLLM_MAX_MODEL_LEN)) || {
    echo "[ERROR] context/output exceeds VLLM_MAX_MODEL_LEN: ${value}" >&2
    exit 2
  }
done
ensure_positive_integer \
  "${DECODE_AUX_CONTEXT_TOKENS}" "DECODE_AUX_CONTEXT_TOKENS"
ensure_nonnegative_integer \
  "${PREFILL_PROBE_CONTEXT_TOKENS}" "PREFILL_PROBE_CONTEXT_TOKENS"
ensure_nonnegative_integer \
  "${DECODE_ONLY_CONTROL_GAPS}" "DECODE_ONLY_CONTROL_GAPS"
ensure_positive_integer \
  "${SM_SCALING_BOOTSTRAP_ITERATIONS}" "SM_SCALING_BOOTSTRAP_ITERATIONS"
ensure_positive_integer \
  "${FIRST_TOKEN_TIMEOUT_SEC}" "FIRST_TOKEN_TIMEOUT_SEC"
ensure_nonnegative_integer \
  "${SM_SCALING_BOOTSTRAP_SEED}" "SM_SCALING_BOOTSTRAP_SEED"
[[ "${SM_SCALING_ALLOW_WINDOW_CLUSTERS}" == "0" \
  || "${SM_SCALING_ALLOW_WINDOW_CLUSTERS}" == "1" ]] || {
  echo "[ERROR] SM_SCALING_ALLOW_WINDOW_CLUSTERS must be 0 or 1" >&2
  exit 2
}
if ((DECODE_ONLY_CONTROL_GAPS > 0)) && [[ "${PROBE_DELAY_SEC}" == "0" ]]; then
  echo "[ERROR] paired Decode-only controls require PROBE_DELAY_SEC > 0" >&2
  exit 2
fi
((DECODE_AUX_CONTEXT_TOKENS + ANCHOR_OUTPUT_TOKENS <= VLLM_MAX_MODEL_LEN)) || {
  echo "[ERROR] auxiliary context/output exceeds VLLM_MAX_MODEL_LEN" >&2
  exit 2
}
if ((PREFILL_PROBE_CONTEXT_TOKENS > 0)); then
  ((PREFILL_PROBE_CONTEXT_TOKENS + PROBE_OUTPUT_TOKENS <= VLLM_MAX_MODEL_LEN)) || {
    echo "[ERROR] probe context/output exceeds VLLM_MAX_MODEL_LEN" >&2
    exit 2
  }
fi
for value in "${DECODE_COUNT_ARRAY[@]}"; do
  ensure_positive_integer "${value}" "active decode request count"
done
ensure_positive_integer "${REPEAT_COUNT}" "REPEAT_COUNT"
[[ "${WARMUP_COUNT}" =~ ^[0-9]+$ ]] || {
  echo "[ERROR] WARMUP_COUNT must be non-negative" >&2
  exit 2
}
ensure_positive_integer "${PREFILL_CHUNK_TOKENS}" "PREFILL_CHUNK_TOKENS"
ensure_positive_integer \
  "${SERVER_SHUTDOWN_TIMEOUT_SEC}" "SERVER_SHUTDOWN_TIMEOUT_SEC"
ensure_positive_integer \
  "${PORT_RELEASE_TIMEOUT_SEC}" "PORT_RELEASE_TIMEOUT_SEC"
ensure_nonnegative_integer \
  "${NATIVE_PROFILE_MAX_RESIDENT_TOKENS}" \
  "NATIVE_PROFILE_MAX_RESIDENT_TOKENS"
[[ "${DECODE_GPU_COMPONENT_TRACE}" == "0" \
    || "${DECODE_GPU_COMPONENT_TRACE}" == "1" ]] || {
  echo "[ERROR] VLLM_PD_DECODE_GPU_COMPONENT_TRACE must be 0 or 1" >&2
  exit 2
}
[[ "${ENABLE_ITERATION_METRICS}" == "0" \
    || "${ENABLE_ITERATION_METRICS}" == "1" ]] || {
  echo "[ERROR] ENABLE_ITERATION_METRICS must be 0 or 1" >&2
  exit 2
}
[[ "${RUN_STANDARD_ANALYZERS}" == "0" \
    || "${RUN_STANDARD_ANALYZERS}" == "1" ]] || {
  echo "[ERROR] RUN_STANDARD_ANALYZERS must be 0 or 1" >&2
  exit 2
}
[[ "${PD_CUDA_EVENT_TRACE_ENABLED}" == "0" \
    || "${PD_CUDA_EVENT_TRACE_ENABLED}" == "1" ]] || {
  echo "[ERROR] VLLM_PD_CUDA_EVENT_TRACE_ENABLED must be 0 or 1" >&2
  exit 2
}
[[ "${GREEN_DECODE_CUDAGRAPH}" == "0" \
    || "${GREEN_DECODE_CUDAGRAPH}" == "1" ]] || {
  echo "[ERROR] VLLM_PD_GREEN_DECODE_CUDAGRAPH must be 0 or 1" >&2
  exit 2
}
[[ "${GREEN_DECODE_CUDAGRAPH_STRICT}" == "0" \
    || "${GREEN_DECODE_CUDAGRAPH_STRICT}" == "1" ]] || {
  echo "[ERROR] VLLM_PD_GREEN_DECODE_CUDAGRAPH_STRICT must be 0 or 1" >&2
  exit 2
}
if [[ "${GREEN_DECODE_CUDAGRAPH}" == "1" \
      && "${DECODE_GPU_COMPONENT_TRACE}" == "1" ]]; then
  echo "[ERROR] Decode component hooks and GreenContext CUDA graphs are mutually exclusive" >&2
  exit 2
fi
ensure_positive_integer \
  "${DECODE_GPU_COMPONENT_TRACE_MAX_PENDING}" \
  "VLLM_PD_DECODE_GPU_COMPONENT_TRACE_MAX_PENDING"
ensure_positive_integer \
  "${DECODE_GPU_COMPONENT_TRACE_EVERY_N}" \
  "VLLM_PD_DECODE_GPU_COMPONENT_TRACE_EVERY_N"
[[ "${DECODE_GPU_COMPONENT_TRACE_INCLUDE_LAYERS}" == "0" \
    || "${DECODE_GPU_COMPONENT_TRACE_INCLUDE_LAYERS}" == "1" ]] || {
  echo "[ERROR] VLLM_PD_DECODE_GPU_COMPONENT_TRACE_INCLUDE_LAYERS must be 0 or 1" >&2
  exit 2
}
for mode in "${MODE_ARRAY[@]}"; do
  [[ "${mode}" == "native" || "${mode}" == "ring" ]] || {
    echo "[ERROR] PREFILL_MODES accepts native/ring: ${mode}" >&2
    exit 2
  }
done
for split in "${SPLIT_ARRAY[@]}"; do
  parse_split "${split}"
done

echo "[PLAN] experiment=${EXPERIMENT_NAME}"
echo "[PLAN] baseline=${BASELINE_RESULTS_DIR}"
echo "[PLAN] contexts=${CONTEXT_ARRAY[*]}"
echo "[PLAN] splits=${SPLIT_ARRAY[*]}"
echo "[PLAN] modes=${MODE_ARRAY[*]}"
echo "[PLAN] active_decode_counts=${DECODE_COUNT_ARRAY[*]}"
echo "[PLAN] decode_aux_context=${DECODE_AUX_CONTEXT_TOKENS}"
echo "[PLAN] prefill_probe_context=${PREFILL_PROBE_CONTEXT_TOKENS}"
echo "[PLAN] decode_only_control_gaps=${DECODE_ONLY_CONTROL_GAPS}"
echo "[PLAN] native_profile_max_resident_tokens=${NATIVE_PROFILE_MAX_RESIDENT_TOKENS}"
echo "[PLAN] warmups=${WARMUP_COUNT} repeats=${REPEAT_COUNT}"
echo "[PLAN] decode_gpu_component_trace=${DECODE_GPU_COMPONENT_TRACE}"
echo "[PLAN] iteration_metrics=${ENABLE_ITERATION_METRICS}"
echo "[PLAN] standard_analyzers=${RUN_STANDARD_ANALYZERS}"
echo "[PLAN] pd_cuda_event_trace=${PD_CUDA_EVENT_TRACE_ENABLED}"
echo "[PLAN] decode_gpu_component_trace_every_n=${DECODE_GPU_COMPONENT_TRACE_EVERY_N}"
echo "[PLAN] green_decode_cudagraph=${GREEN_DECODE_CUDAGRAPH}"
echo "[PLAN] green_decode_cudagraph_strict=${GREEN_DECODE_CUDAGRAPH_STRICT}"
echo "[PLAN] total_profile_cases=$((${#SPLIT_ARRAY[@]} * ${#MODE_ARRAY[@]}))"
echo "[PLAN] trials_per_case=$((
  ${#CONTEXT_ARRAY[@]} * ${#DECODE_COUNT_ARRAY[@]} *
  (WARMUP_COUNT + REPEAT_COUNT)
))"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY-RUN] no files, server, or GPU state will be changed"
  exit 0
fi

if [[ -e "${RESULT_DIR}" || -e "${LOG_DIR}" || -e "${ARTIFACT_DIR}" ]]; then
  echo "[ERROR] output exists; choose a new EXPERIMENT_NAME" >&2
  exit 1
fi

SLO_GENERATOR="${ROOT_DIR}/tools/generate_vendor_v8_slo_profile.py"
MEASURER="${VENDOR_V8_PARTITION_MEASURER:-${ROOT_DIR}/measure_vendor_v8_partition_profile.py}"
ANALYZER="${ROOT_DIR}/tools/analyze_vendor_v8_partition_profiles.py"
SM_SCALING_ANALYZER="${ROOT_DIR}/tools/analyze_vendor_v8_tbt_sm_scaling.py"
COMPONENT_ANALYZER="${ROOT_DIR}/tools/analyze_decode_gpu_components.py"
DATASET_PREPARER="${ROOT_DIR}/prepare_mixed_context_sharegpt.py"
SERVER_STARTER="${ROOT_DIR}/start_vllm_causality_microbench.sh"
POLICY_HELPER="${ROOT_DIR}/tools/pd_eval_policy_env.sh"
for item in \
  "${SLO_GENERATOR}" "${MEASURER}" "${ANALYZER}" "${SM_SCALING_ANALYZER}" \
  "${COMPONENT_ANALYZER}" \
  "${DATASET_PREPARER}" "${SERVER_STARTER}" "${POLICY_HELPER}"; do
  ensure_file "${item}" "required script"
done
ensure_file "${BASELINE_RESULTS_DIR}/request_runs.csv" "baseline raw CSV"
ensure_file "${BASELINE_RESULTS_DIR}/bucket_summary.csv" "baseline summary CSV"
ensure_file "${BASELINE_RESULTS_DIR}/run_metadata.json" "baseline metadata"
ensure_file "${SHAREGPT_RAW_PATH}" "ShareGPT source"
[[ -x "${VLLM_EXECUTABLE}" ]] || {
  echo "[ERROR] vLLM executable is not executable: ${VLLM_EXECUTABLE}" >&2
  exit 1
}
for command_name in setsid lsof ps awk; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "[ERROR] required command is unavailable: ${command_name}" >&2
    exit 1
  }
done

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${RESULT_DIR}/experiment_snapshot/scripts" \
  "${RESULT_DIR}/slo" "${RESULT_DIR}/cases" \
  "${LOG_DIR}" "${ARTIFACT_DIR}"
cp -a "${BASH_SOURCE[0]}" "${SLO_GENERATOR}" "${MEASURER}" \
  "${ANALYZER}" "${SM_SCALING_ANALYZER}" "${COMPONENT_ANALYZER}" \
  "${DATASET_PREPARER}" "${SERVER_STARTER}" \
  "${POLICY_HELPER}" "${RESULT_DIR}/experiment_snapshot/scripts/"

"${PYTHON_BIN}" "${SLO_GENERATOR}" \
  --baseline-results-dir "${BASELINE_RESULTS_DIR}" \
  --output-dir "${RESULT_DIR}/slo" \
  --required-context-tokens "${CONTEXT_ARRAY[@]}" \
  --ttft-slo-multiplier "${TTFT_SLO_MULTIPLIER}" \
  --tbt-slo-multiplier "${TBT_SLO_MULTIPLIER}"
SLO_POLICY_JSON="${RESULT_DIR}/slo/request_slo_policy.json"

if [[ "${PREPARE_DATASET}" == "1" ]]; then
  PROFILE_DATASET_TARGETS=("${CONTEXT_ARRAY[@]}")
  AUX_TARGET_PRESENT=0
  for context in "${PROFILE_DATASET_TARGETS[@]}"; do
    if [[ "${context}" == "${DECODE_AUX_CONTEXT_TOKENS}" ]]; then
      AUX_TARGET_PRESENT=1
      break
    fi
  done
  if [[ "${AUX_TARGET_PRESENT}" == "0" ]]; then
    PROFILE_DATASET_TARGETS+=("${DECODE_AUX_CONTEXT_TOKENS}")
  fi
  if ((PREFILL_PROBE_CONTEXT_TOKENS > 0)); then
    PROBE_TARGET_PRESENT=0
    for context in "${PROFILE_DATASET_TARGETS[@]}"; do
      if [[ "${context}" == "${PREFILL_PROBE_CONTEXT_TOKENS}" ]]; then
        PROBE_TARGET_PRESENT=1
        break
      fi
    done
    if [[ "${PROBE_TARGET_PRESENT}" == "0" ]]; then
      PROFILE_DATASET_TARGETS+=("${PREFILL_PROBE_CONTEXT_TOKENS}")
    fi
  fi
  "${PYTHON_BIN}" "${DATASET_PREPARER}" \
    --sharegpt-path "${SHAREGPT_RAW_PATH}" \
    --target-tokens "${PROFILE_DATASET_TARGETS[@]}" \
    --output-path "${DATASET_JSONL}" \
    --tokenizer "${DATASET_TOKENIZER}" \
    --candidate-pool-size "${SHAREGPT_CANDIDATE_POOL_SIZE}" \
    --underlength-policy "${SHAREGPT_UNDERLENGTH_POLICY}" \
    --max-tokens "${ANCHOR_OUTPUT_TOKENS}" \
    --min-tokens "${ANCHOR_OUTPUT_TOKENS}"
fi
ensure_file "${DATASET_JSONL}" "profile dataset"

MAX_DECODE_COUNT="$(
  printf '%s\n' "${DECODE_COUNT_ARRAY[@]}" | sort -nr | head -1
)"
VLLM_MAX_NUM_SEQS="$((MAX_DECODE_COUNT + 2))"

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
  [[ -z "${pid}" ]] && return
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    sleep 3
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

process_group_has_live_members() {
  local pgid="$1"
  ps -eo pgid=,stat= | awk -v target="${pgid}" '
    $1 == target && $2 !~ /^Z/ {
      found = 1
    }
    END {
      exit(found ? 0 : 1)
    }
  '
}

cleanup_process_group() {
  local leader_pid="${1:-}"
  local pgid="${2:-}"
  [[ -z "${leader_pid}" ]] && return
  if [[ -z "${pgid}" || ! "${pgid}" =~ ^[1-9][0-9]*$ ]]; then
    cleanup_pid "${leader_pid}"
    return
  fi

  local runner_pgid
  runner_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]')"
  if [[ "${pgid}" == "${runner_pgid}" ]]; then
    echo "[ERROR] refusing to signal runner process group ${pgid}" >&2
    cleanup_pid "${leader_pid}"
    return
  fi

  if process_group_has_live_members "${pgid}"; then
    echo "[CLEANUP] terminate server process group pgid=${pgid}"
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
  fi
  wait "${leader_pid}" 2>/dev/null || true
}

list_port_owner_pids() {
  lsof -t -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null \
    | tr '\n' ' ' || true
}

wait_for_port_release() {
  local deadline=$((SECONDS + PORT_RELEASE_TIMEOUT_SEC))
  local pids
  while ((SECONDS < deadline)); do
    pids="$(list_port_owner_pids)"
    [[ -z "${pids// /}" ]] && return 0
    sleep 1
  done
  pids="$(list_port_owner_pids)"
  echo "[ERROR] port ${PORT} remained occupied by ${pids}" >&2
  return 1
}

check_port() {
  local pids
  pids="$(list_port_owner_pids)"
  [[ -z "${pids// /}" ]] && return
  if [[ "${KILL_EXISTING_PORT_OWNER}" != "1" ]]; then
    echo "[ERROR] port ${PORT} is already used by ${pids}" >&2
    return 1
  fi
  # shellcheck disable=SC2086
  kill ${pids} 2>/dev/null || true
  wait_for_port_release
}

ACTIVE_SERVER_PID=""
ACTIVE_SERVER_PGID=""
ACTIVE_DMON_PID=""
cleanup_active() {
  cleanup_pid "${ACTIVE_DMON_PID}"
  cleanup_process_group "${ACTIVE_SERVER_PID}" "${ACTIVE_SERVER_PGID}"
  ACTIVE_DMON_PID=""
  ACTIVE_SERVER_PID=""
  ACTIVE_SERVER_PGID=""
}
trap cleanup_active EXIT INT TERM

BRIDGE_LIB="${VLLM_VENDOR_DIR}/vllm/model_executor/layers/green_context_lib/libvllm_green_context.so"
if [[ "${BUILD_GREEN_BRIDGE}" == "1" \
      || ("${BUILD_GREEN_BRIDGE}" == "auto" && ! -f "${BRIDGE_LIB}") ]]; then
  BRIDGE_BUILDER="${ROOT_DIR}/tools/green_context_smoke/build_vendor_v8_bridge.sh"
  ensure_file "${BRIDGE_BUILDER}" "vendor_v8 GreenContext builder"
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" \
    bash "${BRIDGE_BUILDER}"
fi

BASE_VLLM_EXTRA_ARGS="--attention-backend TRITON_ATTN \
--kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES} \
--enable-ttft-prefill-chunk-scheduling \
--ttft-prefill-chunk-scheduling-policy round_robin \
--ttft-prefill-chunk-quantum-tokens ${PREFILL_CHUNK_TOKENS} \
--enable-tbt-decode-scheduling \
--tbt-decode-scheduling-policy fcfs"

run_case() {
  local mode="$1"
  local split="$2"
  parse_split "${split}"
  local case_id="${EXPERIMENT_NAME}_${mode}_${PROFILE_ID}"
  local case_dir="${RESULT_DIR}/cases/${mode}/${PROFILE_ID}"
  local case_log_dir="${LOG_DIR}/${mode}/${PROFILE_ID}"
  local case_artifact_dir="${ARTIFACT_DIR}/${mode}/${PROFILE_ID}"
  local server_log="${case_log_dir}/server.log"
  local iteration_metrics=""
  if [[ "${ENABLE_ITERATION_METRICS}" == "1" ]]; then
    iteration_metrics="${case_dir}/iteration_metrics.jsonl"
  fi
  local pd_cuda_event_trace_jsonl=""
  if [[ "${PD_CUDA_EVENT_TRACE_ENABLED}" == "1" ]]; then
    pd_cuda_event_trace_jsonl="${case_artifact_dir}/pd_cuda_events.jsonl"
  fi
  local decode_component_trace_jsonl=""
  if [[ "${DECODE_GPU_COMPONENT_TRACE}" == "1" ]]; then
    decode_component_trace_jsonl="${case_artifact_dir}/decode_gpu_components.jsonl"
  fi
  mkdir -p "${case_dir}" "${case_log_dir}" "${case_artifact_dir}"

  local policy
  local max_profile_total_input_tokens
  if [[ "${mode}" == "ring" ]]; then
    policy="muxwise_dynamic_virtual_ring_optimized"
    max_profile_total_input_tokens=0
  else
    policy="muxwise_dynamic"
    max_profile_total_input_tokens="${NATIVE_PROFILE_MAX_RESIDENT_TOKENS}"
  fi

  if ! check_port; then
    return 1
  fi
  echo "[CASE] mode=${mode} split=${split} profile=${PROFILE_ID}"
  (
    cd "${ROOT_DIR}"
    exec setsid env \
      VLLM_VENDOR_DIR="${VLLM_VENDOR_DIR}" \
      VLLM_EXECUTABLE="${VLLM_EXECUTABLE}" \
      VLLM_PD_EVAL_POLICY="${policy}" \
      VLLM_PD_EVAL_RING_ENABLE_CHUNKED_PREFILL=1 \
      VLLM_PD_EVAL_RING_BACKEND_FORCE_ALL="$(
        [[ "${mode}" == "ring" ]] && echo 1 || echo 0
      )" \
      VLLM_PD_EVAL_PREFILL_RING_REQUIRED_MODE="$(
        [[ "${mode}" == "ring" ]] && echo always || echo never
      )" \
      VLLM_PD_EVAL_RING_BW_DYNAMIC=0 \
      VLLM_PD_EVAL_RING_BW_MIN_TOKENS="${PREFILL_CHUNK_TOKENS}" \
      VLLM_PD_EVAL_RING_BW_MAX_TOKENS="${PREFILL_CHUNK_TOKENS}" \
      VLLM_PD_EVAL_RING_COMPUTE_GROUP_TOKENS="${PREFILL_CHUNK_TOKENS}" \
      VLLM_PD_EVAL_RING_FULLY_OPTIMIZED=1 \
      VLLM_PD_EVAL_DRAM_PAGE_POOL_CONTIGUOUS=1 \
      VLLM_PD_EVAL_RELOAD_DIRECT_MULTICOPY=1 \
      VLLM_PD_EVAL_RELOAD_STAGING_SCATTER=0 \
      VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS="${DECODE_SMS}" \
      VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS="${PREFILL_SMS}" \
      VLLM_PD_DYNAMIC_SM_ENABLED=0 \
      VLLM_ITERATION_METRICS_JSONL="${iteration_metrics}" \
      VLLM_REQUEST_METRICS_JSONL="${case_artifact_dir}/request_metrics.jsonl" \
      VLLM_PD_CUDA_EVENT_TRACE_JSONL="${pd_cuda_event_trace_jsonl}" \
      VLLM_PD_CUDA_EVENT_TRACE_ENABLED="${PD_CUDA_EVENT_TRACE_ENABLED}" \
      VLLM_PD_DECODE_GPU_COMPONENT_TRACE_JSONL="${decode_component_trace_jsonl}" \
      VLLM_PD_DECODE_GPU_COMPONENT_TRACE_MAX_PENDING="${DECODE_GPU_COMPONENT_TRACE_MAX_PENDING}" \
      VLLM_PD_DECODE_GPU_COMPONENT_TRACE_EVERY_N="${DECODE_GPU_COMPONENT_TRACE_EVERY_N}" \
      VLLM_PD_DECODE_GPU_COMPONENT_TRACE_INCLUDE_LAYERS="${DECODE_GPU_COMPONENT_TRACE_INCLUDE_LAYERS}" \
      VLLM_PD_GREEN_DECODE_CUDAGRAPH="${GREEN_DECODE_CUDAGRAPH}" \
      VLLM_PD_GREEN_DECODE_CUDAGRAPH_STRICT="${GREEN_DECODE_CUDAGRAPH_STRICT}" \
      VLLM_ITERATION_GPU_TIMING=0 \
      VLLM_LAYER_KERNEL_TIMING=0 \
      VLLM_PD_PHASE_SAMPLER_DEBUG_SYNC=0 \
      VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
      VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" \
      VLLM_MAX_NUM_BATCHED_TOKENS="${PREFILL_CHUNK_TOKENS}" \
      VLLM_ENABLE_CHUNKED_PREFILL=1 \
      VLLM_DISABLE_PREFIX_CACHING=1 \
      VLLM_EXTRA_ARGS="${BASE_VLLM_EXTRA_ARGS}" \
      "${SERVER_STARTER}" --port "${PORT}" --model "${MODEL_NAME}"
  ) >>"${server_log}" 2>&1 &
  ACTIVE_SERVER_PID="$!"
  ACTIVE_SERVER_PGID="${ACTIVE_SERVER_PID}"
  if ! wait_for_server_ready "${ACTIVE_SERVER_PID}"; then
    tail -n 160 "${server_log}" >&2 || true
    return 1
  fi
  local observed_server_pgid
  observed_server_pgid="$(
    ps -o pgid= -p "${ACTIVE_SERVER_PID}" 2>/dev/null \
      | tr -d '[:space:]'
  )"
  if [[ "${observed_server_pgid}" != "${ACTIVE_SERVER_PGID}" ]]; then
    echo "[ERROR] server process group mismatch: "\
"expected=${ACTIVE_SERVER_PGID} "\
"observed=${observed_server_pgid:-missing}" >&2
    return 1
  fi
  if [[ "${ENABLE_DMON}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi dmon -s pucm -d 1 -o TD \
      >"${case_artifact_dir}/nvidia_smi_dmon.log" 2>&1 &
    ACTIVE_DMON_PID="$!"
  fi

  local measure_status=0
  "${PYTHON_BIN}" "${MEASURER}" \
    --base-url "${BASE_URL}" \
    --requests-jsonl "${DATASET_JSONL}" \
    --slo-policy-json "${SLO_POLICY_JSON}" \
    --case-id "${case_id}" \
    --profile-id "${PROFILE_ID}" \
    --decode-sms "${DECODE_SMS}" \
    --prefill-sms "${PREFILL_SMS}" \
    --prefill-mode "${mode}" \
    --context-tokens "${CONTEXT_ARRAY[@]}" \
    --active-decode-requests "${DECODE_COUNT_ARRAY[@]}" \
    --decode-aux-context-tokens "${DECODE_AUX_CONTEXT_TOKENS}" \
    --probe-context-tokens "${PREFILL_PROBE_CONTEXT_TOKENS}" \
    --max-profile-total-input-tokens \
      "${max_profile_total_input_tokens}" \
    --repeat-count "${REPEAT_COUNT}" \
    --warmup-count "${WARMUP_COUNT}" \
    --anchor-output-tokens "${ANCHOR_OUTPUT_TOKENS}" \
    --probe-output-tokens "${PROBE_OUTPUT_TOKENS}" \
    --probe-delay-sec "${PROBE_DELAY_SEC}" \
    --decode-only-control-gaps "${DECODE_ONLY_CONTROL_GAPS}" \
    --cooldown-sec "${CASE_COOLDOWN_SEC}" \
    --request-timeout-sec "${REQUEST_TIMEOUT_SEC}" \
    --first-token-timeout-sec "${FIRST_TOKEN_TIMEOUT_SEC}" \
    --raw-csv "${case_dir}/request_runs.csv" \
    --token-csv "${case_dir}/client_token_timeline.csv" \
    --trial-csv "${case_dir}/trial_summary.csv" \
    --metadata-json "${case_dir}/run_metadata.json" \
    || measure_status="$?"
  cleanup_active
  if ! wait_for_port_release; then
    return 1
  fi
  return "${measure_status}"
}

FAILED_CASES=0
for mode in "${MODE_ARRAY[@]}"; do
  for split in "${SPLIT_ARRAY[@]}"; do
    if ! run_case "${mode}" "${split}"; then
      FAILED_CASES=$((FAILED_CASES + 1))
      cleanup_active
    fi
  done
done

if ((FAILED_CASES > 0)); then
  echo "[ERROR] partition profile failed_cases=${FAILED_CASES}" >&2
  exit 1
fi

if [[ "${RUN_STANDARD_ANALYZERS}" == "1" ]]; then
  "${PYTHON_BIN}" "${ANALYZER}" \
    --profile-root "${RESULT_DIR}/cases" \
    --output-dir "${RESULT_DIR}/runtime_profiles" \
    --chunk-tokens "${PREFILL_CHUNK_TOKENS}" \
    --native-baseline-results-dir "${BASELINE_RESULTS_DIR}"

  if ((DECODE_ONLY_CONTROL_GAPS > 0)); then
    SM_SCALING_EXTRA_ARGS=()
    if [[ "${SM_SCALING_ALLOW_WINDOW_CLUSTERS}" == "1" ]]; then
      SM_SCALING_EXTRA_ARGS+=(--allow-window-clusters)
    fi
    "${PYTHON_BIN}" "${SM_SCALING_ANALYZER}" \
      --baseline-results-dir "${BASELINE_RESULTS_DIR}" \
      --partition-results-dir "${RESULT_DIR}/cases" \
      --output-dir "${RESULT_DIR}/sm_scaling_proof" \
      --total-sms "${EXPECTED_GPU_SMS}" \
      --quantile 50 \
      --slo-factor 1.25 \
      --assumed-overlap-overhead 0.10 \
      --bootstrap-iterations "${SM_SCALING_BOOTSTRAP_ITERATIONS}" \
      --bootstrap-seed "${SM_SCALING_BOOTSTRAP_SEED}" \
      --min-repeats "${REPEAT_COUNT}" \
      "${SM_SCALING_EXTRA_ARGS[@]}"
  fi
fi

if [[ "${DECODE_GPU_COMPONENT_TRACE}" == "1" ]]; then
  "${PYTHON_BIN}" "${COMPONENT_ANALYZER}" \
    --artifact-root "${ARTIFACT_DIR}" \
    --output-dir "${RESULT_DIR}/decode_gpu_components"
fi

cat >"${RESULT_DIR}/experiment_snapshot/settings.txt" <<EOF
experiment_name=${EXPERIMENT_NAME}
baseline_results_dir=${BASELINE_RESULTS_DIR}
contexts=${CONTEXT_ARRAY[*]}
green_sm_splits=${SPLIT_ARRAY[*]}
prefill_modes=${MODE_ARRAY[*]}
active_decode_request_counts=${DECODE_COUNT_ARRAY[*]}
decode_aux_context_tokens=${DECODE_AUX_CONTEXT_TOKENS}
prefill_probe_context_tokens=${PREFILL_PROBE_CONTEXT_TOKENS}
decode_only_control_gaps=${DECODE_ONLY_CONTROL_GAPS}
sm_scaling_bootstrap_iterations=${SM_SCALING_BOOTSTRAP_ITERATIONS}
sm_scaling_bootstrap_seed=${SM_SCALING_BOOTSTRAP_SEED}
sm_scaling_allow_window_clusters=${SM_SCALING_ALLOW_WINDOW_CLUSTERS}
first_token_timeout_sec=${FIRST_TOKEN_TIMEOUT_SEC}
native_profile_max_resident_tokens=${NATIVE_PROFILE_MAX_RESIDENT_TOKENS}
repeat_count=${REPEAT_COUNT}
warmup_count=${WARMUP_COUNT}
ttft_slo_multiplier=${TTFT_SLO_MULTIPLIER}
tbt_slo_multiplier=${TBT_SLO_MULTIPLIER}
prefill_chunk_tokens=${PREFILL_CHUNK_TOKENS}
anchor_output_tokens=${ANCHOR_OUTPUT_TOKENS}
probe_output_tokens=${PROBE_OUTPUT_TOKENS}
server_shutdown_timeout_sec=${SERVER_SHUTDOWN_TIMEOUT_SEC}
port_release_timeout_sec=${PORT_RELEASE_TIMEOUT_SEC}
decode_gpu_component_trace=${DECODE_GPU_COMPONENT_TRACE}
iteration_metrics=${ENABLE_ITERATION_METRICS}
standard_analyzers=${RUN_STANDARD_ANALYZERS}
pd_cuda_event_trace=${PD_CUDA_EVENT_TRACE_ENABLED}
decode_gpu_component_trace_max_pending=${DECODE_GPU_COMPONENT_TRACE_MAX_PENDING}
decode_gpu_component_trace_every_n=${DECODE_GPU_COMPONENT_TRACE_EVERY_N}
decode_gpu_component_trace_include_layers=${DECODE_GPU_COMPONENT_TRACE_INCLUDE_LAYERS}
green_decode_cudagraph=${GREEN_DECODE_CUDAGRAPH}
green_decode_cudagraph_strict=${GREEN_DECODE_CUDAGRAPH_STRICT}
vendor_dir=${VLLM_VENDOR_DIR}
EOF

echo "[DONE] request_slo=${SLO_POLICY_JSON}"
if [[ "${RUN_STANDARD_ANALYZERS}" == "1" ]]; then
  echo "[DONE] prefill_profile=${RESULT_DIR}/runtime_profiles/prefill_service_profile.json"
  echo "[DONE] tbt_profile=${RESULT_DIR}/runtime_profiles/tbt_overlap_profile.json"
  echo "[DONE] summary=${RESULT_DIR}/runtime_profiles/partition_profile_summary.csv"
  if ((DECODE_ONLY_CONTROL_GAPS > 0)); then
    echo "[DONE] sm_scaling_proof=${RESULT_DIR}/sm_scaling_proof"
  fi
else
  echo "[DONE] standard_analyzers=skipped_by_runner"
fi
if [[ "${DECODE_GPU_COMPONENT_TRACE}" == "1" ]]; then
  echo "[DONE] decode_gpu_components=${RESULT_DIR}/decode_gpu_components"
fi
