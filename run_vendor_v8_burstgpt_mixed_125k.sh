#!/usr/bin/env bash
set -euo pipefail

# Paper-facing trace-driven hybrid workload:
# - BurstGPT: open-loop arrival offsets and exact output-token counts
# - LooGLE: distinct 125K long-document prompts
# - ShareGPT: distinct short prompts matched to BurstGPT request lengths

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-vendor_v8_burstgpt_mixed_125k_v1}"
RESULT_DIR="${ROOT_DIR}/${EXPERIMENT_NAME}_results"
LOG_DIR="${ROOT_DIR}/${EXPERIMENT_NAME}_logs"
ARTIFACT_DIR="${ROOT_DIR}/${EXPERIMENT_NAME}_artifacts"

BURSTGPT_CSV="${BURSTGPT_CSV:-${ROOT_DIR}/data/raw/burstgpt/BurstGPT_without_fails_1.csv}"
LOOGLE_PATH="${LOOGLE_PATH:-${ROOT_DIR}/data/raw/loogle/data/summarization.jsonl}"
SHAREGPT_PATH="${SHAREGPT_PATH:-${ROOT_DIR}/data/ShareGPT_V3_unfiltered_cleaned_split.json}"
PREPARE_WORKLOAD="${PREPARE_WORKLOAD:-1}"
WORKLOAD_JSONL="${WORKLOAD_JSONL:-${RESULT_DIR}/experiment_snapshot/burstgpt_hybrid.jsonl}"
DATASET_TOKENIZER="${DATASET_TOKENIZER:-meta-llama/Llama-3.2-1B-Instruct}"
LONG_REQUEST_COUNT="${LONG_REQUEST_COUNT:-4}"
SHORT_REQUEST_COUNT="${SHORT_REQUEST_COUNT:-16}"
LONG_EVENT_INDICES="${LONG_EVENT_INDICES:-0 1 2 3}"
LONG_CONTEXT_TOKENS="${LONG_CONTEXT_TOKENS:-125000}"
LOOGLE_UNDERLENGTH_POLICY="${LOOGLE_UNDERLENGTH_POLICY:-pack}"
SHORT_MIN_TOKENS="${SHORT_MIN_TOKENS:-512}"
SHORT_MAX_TOKENS="${SHORT_MAX_TOKENS:-16384}"
OUTPUT_MIN_TOKENS="${OUTPUT_MIN_TOKENS:-16}"
OUTPUT_MAX_TOKENS="${OUTPUT_MAX_TOKENS:-4096}"
TRACE_START_VALID_INDEX="${TRACE_START_VALID_INDEX:-0}"
ARRIVAL_TIME_SCALE="${ARRIVAL_TIME_SCALE:-1.0}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-512}"

PROFILE_RESULTS_DIR="${PROFILE_RESULTS_DIR:-${ROOT_DIR}/vendor_v8_each_partition_profile_v2_results}"
REQUEST_SLO_POLICY="${REQUEST_SLO_POLICY:-${PROFILE_RESULTS_DIR}/slo/request_slo_policy.json}"
PREFILL_SERVICE_PROFILE="${PREFILL_SERVICE_PROFILE:-${PROFILE_RESULTS_DIR}/runtime_profiles/prefill_service_profile.json}"
TBT_OVERLAP_PROFILE="${TBT_OVERLAP_PROFILE:-${PROFILE_RESULTS_DIR}/runtime_profiles/tbt_overlap_profile.json}"

VENV_DIR="${VENV_DIR:-/home/noslab-gpu/tkdgjs/tkdgjs}"
VLLM_EXECUTABLE="${VLLM_EXECUTABLE:-${VENV_DIR}/bin/vllm}"
VLLM_VENDOR_DIR="${VLLM_VENDOR_DIR:-${ROOT_DIR}/vendor_v8}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}"
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:${PORT}"

KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-4294967296}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-131072}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-2048}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
INITIAL_GREEN_SM_SPLIT="${INITIAL_GREEN_SM_SPLIT:-20/20}"
PD_DYNAMIC_SM_POLICY="${PD_DYNAMIC_SM_POLICY:-tbt_slo_feedback}"
PD_SM_PROFILES="${PD_SM_PROFILES:-auto_dense}"
PD_FEEDBACK_INITIAL_DECODE_SMS="${PD_FEEDBACK_INITIAL_DECODE_SMS:-30}"
PD_FEEDBACK_STEP_SMS="${PD_FEEDBACK_STEP_SMS:-2}"
PD_FEEDBACK_WINDOW_TOKENS="${PD_FEEDBACK_WINDOW_TOKENS:-32}"
PD_FEEDBACK_SLACK_RATIO="${PD_FEEDBACK_SLACK_RATIO:-0.90}"
PD_FEEDBACK_REPROBE_RATIO="${PD_FEEDBACK_REPROBE_RATIO:-0.80}"
PD_FEEDBACK_DECREASE_WINDOWS="${PD_FEEDBACK_DECREASE_WINDOWS:-3}"
PD_FEEDBACK_HOLD_WINDOWS="${PD_FEEDBACK_HOLD_WINDOWS:-16}"
PD_FEEDBACK_MIN_PREFILL_SMS="${PD_FEEDBACK_MIN_PREFILL_SMS:-2}"
PD_FEEDBACK_CONTEXT_BUCKET_TOKENS="${PD_FEEDBACK_CONTEXT_BUCKET_TOKENS:-25000}"
TBT_PROFILE_MARGIN_MS="${TBT_PROFILE_MARGIN_MS:-5}"
TBT_PROFILE_STRICT="${TBT_PROFILE_STRICT:-1}"
TTFT_PREFILL_SLO_GUARD_MS="${TTFT_PREFILL_SLO_GUARD_MS:-500}"
TTFT_PREFILL_STARVATION_THRESHOLD_MS="${TTFT_PREFILL_STARVATION_THRESHOLD_MS:-5000}"
TTFT_PREFILL_STARVATION_MAX_BYPASSES="${TTFT_PREFILL_STARVATION_MAX_BYPASSES:-8}"
RESIDENCY_VICTIM_POLICY="${RESIDENCY_VICTIM_POLICY:-jit_min_copy}"
REQUIRE_PREFILL_PROFILE_COVERAGE="${REQUIRE_PREFILL_PROFILE_COVERAGE:-1}"

WARMUP_COUNT="${WARMUP_COUNT:-1}"
WARMUP_OUTPUT_TOKENS="${WARMUP_OUTPUT_TOKENS:-8}"
WARMUP_COOLDOWN_SEC="${WARMUP_COOLDOWN_SEC:-5}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-14400}"
SERVER_READY_TIMEOUT_SEC="${SERVER_READY_TIMEOUT_SEC:-900}"
SERVER_SHUTDOWN_TIMEOUT_SEC="${SERVER_SHUTDOWN_TIMEOUT_SEC:-30}"
PORT_RELEASE_TIMEOUT_SEC="${PORT_RELEASE_TIMEOUT_SEC:-30}"
BUILD_GREEN_BRIDGE="${BUILD_GREEN_BRIDGE:-auto}"
ENABLE_DMON="${ENABLE_DMON:-1}"
GENERATE_ABCD="${GENERATE_ABCD:-0}"
KILL_EXISTING_PORT_OWNER="${KILL_EXISTING_PORT_OWNER:-0}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./run_vendor_v8_burstgpt_mixed_125k.sh [--dry-run]

The runner intentionally labels this as a trace-driven hybrid, not a raw
BurstGPT production replay. Set BURSTGPT_CSV to an official BurstGPT CSV.
REQUEST_SLO_POLICY must contain a bucket covering LONG_CONTEXT_TOKENS.
EOF
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

read -r -a LONG_INDEX_ARRAY <<<"${LONG_EVENT_INDICES}"
if ((${#LONG_INDEX_ARRAY[@]} != LONG_REQUEST_COUNT)); then
  echo "[ERROR] LONG_EVENT_INDICES count must equal LONG_REQUEST_COUNT" >&2
  exit 2
fi
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

echo "[PLAN] workload=trace-driven BurstGPT + LooGLE/ShareGPT hybrid"
echo "[PLAN] replay=open_loop arrival_scale=${ARRIVAL_TIME_SCALE}"
echo "[PLAN] long=${LONG_REQUEST_COUNT}x${LONG_CONTEXT_TOKENS} indices=${LONG_INDEX_ARRAY[*]}"
echo "[PLAN] loogle_underlength_policy=${LOOGLE_UNDERLENGTH_POLICY}"
echo "[PLAN] short=${SHORT_REQUEST_COUNT} range=${SHORT_MIN_TOKENS}-${SHORT_MAX_TOKENS}"
echo "[PLAN] output_lengths=BurstGPT range=${OUTPUT_MIN_TOKENS}-${OUTPUT_MAX_TOKENS}"
echo "[PLAN] scheduler=slo_srpt decode=fcfs"
echo "[PLAN] residency=${RESIDENCY_VICTIM_POLICY}"
echo "[PLAN] dynamic_sm=${PD_DYNAMIC_SM_POLICY} profiles=${PD_SM_PROFILES} initial=${INITIAL_GREEN_SM_SPLIT}"
echo "[PLAN] kv_cache_bytes=${KV_CACHE_MEMORY_BYTES}"
echo "[PLAN] request_slo=${REQUEST_SLO_POLICY}"
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

PREPARER="${ROOT_DIR}/prepare_burstgpt_loogle_sharegpt_workload.py"
MEASURER="${ROOT_DIR}/measure_burstgpt_mixed_workload.py"
SERVER_STARTER="${ROOT_DIR}/start_vllm_causality_microbench.sh"
POLICY_HELPER="${ROOT_DIR}/tools/pd_eval_policy_env.sh"
for path in "${PREPARER}" "${MEASURER}" "${SERVER_STARTER}" \
  "${POLICY_HELPER}" "${REQUEST_SLO_POLICY}" \
  "${PREFILL_SERVICE_PROFILE}"; do
  ensure_file "${path}" "required input"
done
if [[ "${PREPARE_WORKLOAD}" == "1" ]]; then
  ensure_file "${BURSTGPT_CSV}" "BurstGPT CSV"
  ensure_file "${LOOGLE_PATH}" "LooGLE source"
  ensure_file "${SHAREGPT_PATH}" "ShareGPT source"
fi
case "${PD_DYNAMIC_SM_POLICY}" in
  tbt|tbt_profiled|tbt_slo_profiled)
    ensure_file "${TBT_OVERLAP_PROFILE}" "TBT overlap profile"
    ;;
esac
[[ -x "${VLLM_EXECUTABLE}" ]] || {
  echo "[ERROR] vLLM executable is not executable: ${VLLM_EXECUTABLE}" >&2
  exit 1
}
for command_name in setsid lsof ps awk; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "[ERROR] required command unavailable: ${command_name}" >&2
    exit 1
  }
done

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" - "${REQUEST_SLO_POLICY}" "${LONG_CONTEXT_TOKENS}" <<'PY'
import json
import sys

path, required_text = sys.argv[1:]
required = int(required_text)
payload = json.load(open(path, encoding="utf-8"))
buckets = payload.get("buckets") or []
covering = [
    row for row in buckets
    if int(row.get("max_context_tokens", -1)) >= required
]
if not covering:
    maximum = max(
        (int(row.get("max_context_tokens", -1)) for row in buckets),
        default=-1,
    )
    raise SystemExit(
        f"request SLO policy covers only {maximum} tokens; "
        f"a measured bucket covering {required} is required"
    )
PY

if [[ "${REQUIRE_PREFILL_PROFILE_COVERAGE}" == "1" ]]; then
  "${PYTHON_BIN}" - "${PREFILL_SERVICE_PROFILE}" \
    "${LONG_CONTEXT_TOKENS}" "${PREFILL_CHUNK_TOKENS}" <<'PY'
import json
import sys

path, required_context_text, required_chunk_text = sys.argv[1:]
required_context = int(required_context_text)
required_chunk = int(required_chunk_text)
payload = json.load(open(path, encoding="utf-8"))
points = payload.get("points") or []
variants = {
    (str(row.get("profile_id", "any")), str(row.get("prefill_mode", "any")))
    for row in points
}
if not variants:
    raise SystemExit("prefill service profile contains no variants")
missing = []
for variant in sorted(variants):
    covered = any(
        (str(row.get("profile_id", "any")), str(row.get("prefill_mode", "any")))
        == variant
        and int(row.get("max_context_tokens", -1)) >= required_context
        and int(row.get("max_chunk_tokens", -1)) >= required_chunk
        for row in points
    )
    if not covered:
        missing.append("/".join(variant))
if missing:
    raise SystemExit(
        "prefill service profile does not cover the 125K workload for "
        f"variants={missing}; profile context>={required_context}, "
        f"chunk>={required_chunk}, or set "
        "REQUIRE_PREFILL_PROFILE_COVERAGE=0 only for a labeled fallback run"
    )
PY
fi

PREFILL_FALLBACK_MS_PER_TOKEN="$(
  "${PYTHON_BIN}" - "${REQUEST_SLO_POLICY}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(float(payload["prefill_fallback_ms_per_token"]))
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
cp -a "${BASH_SOURCE[0]}" "${PREPARER}" "${MEASURER}" \
  "${SERVER_STARTER}" "${POLICY_HELPER}" \
  "${RESULT_DIR}/experiment_snapshot/scripts/"
cp -a "${REQUEST_SLO_POLICY}" \
  "${RESULT_DIR}/experiment_snapshot/request_slo_policy.json"
cp -a "${PREFILL_SERVICE_PROFILE}" \
  "${RESULT_DIR}/experiment_snapshot/prefill_service_profile.json"
if [[ -f "${TBT_OVERLAP_PROFILE}" ]]; then
  cp -a "${TBT_OVERLAP_PROFILE}" \
    "${RESULT_DIR}/experiment_snapshot/tbt_overlap_profile.json"
fi

if [[ "${PREPARE_WORKLOAD}" == "1" ]]; then
  "${PYTHON_BIN}" "${PREPARER}" \
    --burstgpt-csv "${BURSTGPT_CSV}" \
    --loogle-path "${LOOGLE_PATH}" \
    --sharegpt-path "${SHAREGPT_PATH}" \
    --output-path "${WORKLOAD_JSONL}" \
    --tokenizer "${DATASET_TOKENIZER}" \
    --long-count "${LONG_REQUEST_COUNT}" \
    --short-count "${SHORT_REQUEST_COUNT}" \
    --long-event-indices "${LONG_INDEX_ARRAY[@]}" \
    --long-context-tokens "${LONG_CONTEXT_TOKENS}" \
    --loogle-underlength-policy "${LOOGLE_UNDERLENGTH_POLICY}" \
    --short-min-tokens "${SHORT_MIN_TOKENS}" \
    --short-max-tokens "${SHORT_MAX_TOKENS}" \
    --output-min-tokens "${OUTPUT_MIN_TOKENS}" \
    --output-max-tokens "${OUTPUT_MAX_TOKENS}" \
    --model-max-len "${VLLM_MAX_MODEL_LEN}" \
    --trace-start-valid-index "${TRACE_START_VALID_INDEX}" \
    --arrival-time-scale "${ARRIVAL_TIME_SCALE}" \
    --candidate-pool-size "${CANDIDATE_POOL_SIZE}"
fi
ensure_file "${WORKLOAD_JSONL}" "prepared workload"

BRIDGE_LIB="${VLLM_VENDOR_DIR}/vllm/model_executor/layers/green_context_lib/libvllm_green_context.so"
if [[ "${BUILD_GREEN_BRIDGE}" == "1" \
      || ("${BUILD_GREEN_BRIDGE}" == "auto" && ! -f "${BRIDGE_LIB}") ]]; then
  BRIDGE_BUILDER="${ROOT_DIR}/tools/green_context_smoke/build_vendor_v8_bridge.sh"
  ensure_file "${BRIDGE_BUILDER}" "vendor_v8 GreenContext builder"
  CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" bash "${BRIDGE_BUILDER}"
fi

list_port_owner_pids() {
  lsof -t -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null | tr '\n' ' ' || true
}

wait_for_port_release() {
  local deadline=$((SECONDS + PORT_RELEASE_TIMEOUT_SEC))
  while ((SECONDS < deadline)); do
    [[ -z "$(list_port_owner_pids | tr -d ' ')" ]] && return 0
    sleep 1
  done
  echo "[ERROR] port ${PORT} did not become free" >&2
  return 1
}

check_port() {
  local pids
  pids="$(list_port_owner_pids)"
  [[ -z "${pids// /}" ]] && return 0
  if [[ "${KILL_EXISTING_PORT_OWNER}" != "1" ]]; then
    echo "[ERROR] port ${PORT} is already used by ${pids}" >&2
    return 1
  fi
  # shellcheck disable=SC2086
  kill ${pids} 2>/dev/null || true
  wait_for_port_release
}

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
  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
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
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    local deadline=$((SECONDS + SERVER_SHUTDOWN_TIMEOUT_SEC))
    while process_group_has_live_members "${pgid}" \
        && ((SECONDS < deadline)); do
      sleep 1
    done
  fi
  if process_group_has_live_members "${pgid}"; then
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  fi
  wait "${leader_pid}" 2>/dev/null || true
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

check_port
ITERATION_METRICS="${ARTIFACT_DIR}/iteration_metrics.jsonl"
REQUEST_METRICS="${ARTIFACT_DIR}/request_metrics.jsonl"
DECODE_TIMELINE="${ARTIFACT_DIR}/decode_timeline.jsonl"
CONTEXT_RING_TRACE="${ARTIFACT_DIR}/context_ring_trace.jsonl"
PD_CUDA_EVENTS="${ARTIFACT_DIR}/pd_cuda_events.jsonl"
SERVER_LOG="${LOG_DIR}/server.log"

VLLM_EXTRA_ARGS="--attention-backend TRITON_ATTN \
--kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES} \
--enable-ttft-prefill-chunk-scheduling \
--ttft-prefill-chunk-scheduling-policy slo_srpt \
--ttft-prefill-chunk-quantum-tokens ${PREFILL_CHUNK_TOKENS} \
--ttft-prefill-estimated-ms-per-token ${PREFILL_FALLBACK_MS_PER_TOKEN} \
--ttft-prefill-service-profile-path ${PREFILL_SERVICE_PROFILE} \
--ttft-prefill-slo-guard-ms ${TTFT_PREFILL_SLO_GUARD_MS} \
--ttft-prefill-starvation-threshold-ms ${TTFT_PREFILL_STARVATION_THRESHOLD_MS} \
--ttft-prefill-starvation-max-bypasses ${TTFT_PREFILL_STARVATION_MAX_BYPASSES} \
--enable-tbt-decode-scheduling \
--tbt-decode-scheduling-policy fcfs"

(
  cd "${ROOT_DIR}"
  exec setsid env \
    VLLM_VENDOR_DIR="${VLLM_VENDOR_DIR}" \
    VLLM_EXECUTABLE="${VLLM_EXECUTABLE}" \
    VLLM_PD_EVAL_POLICY=muxwise_dynamic_virtual_ring_optimized \
    VLLM_PD_EVAL_RING_ENABLE_CHUNKED_PREFILL=1 \
    VLLM_PD_EVAL_PREFILL_RING_REQUIRED_MODE=pressure \
    VLLM_PD_EVAL_RESIDENCY_VICTIM_POLICY="${RESIDENCY_VICTIM_POLICY}" \
    VLLM_PD_EVAL_RESIDENCY_RING_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS}" \
    VLLM_PD_EVAL_RING_BW_DYNAMIC=0 \
    VLLM_PD_EVAL_RING_BW_MIN_TOKENS="${PREFILL_CHUNK_TOKENS}" \
    VLLM_PD_EVAL_RING_BW_MAX_TOKENS="${PREFILL_CHUNK_TOKENS}" \
    VLLM_PD_EVAL_RING_COMPUTE_GROUP_TOKENS="${PREFILL_CHUNK_TOKENS}" \
    VLLM_PD_EVAL_RING_FULLY_OPTIMIZED=1 \
    VLLM_PD_EVAL_RING_BATCHED_PARTIAL=0 \
    VLLM_PD_EVAL_RING_BATCHED_WORKSPACE_BYTES=0 \
    VLLM_PD_EVAL_DRAM_PAGE_POOL_CONTIGUOUS=1 \
    VLLM_PD_EVAL_DRAM_REGISTER_DIRECT_MULTICOPY=1 \
    VLLM_PD_EVAL_DRAM_REGISTER_ASYNC=1 \
    VLLM_PD_EVAL_DRAM_REGISTER_ASYNC_WORKERS=1 \
    VLLM_PD_EVAL_DRAM_REGISTER_ASYNC_MAX_PENDING_PAGES=4096 \
    VLLM_PD_EVAL_RELOAD_DIRECT_MULTICOPY=1 \
    VLLM_PD_EVAL_RELOAD_STAGING_SCATTER=0 \
    VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS="${DECODE_SMS}" \
    VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS="${PREFILL_SMS}" \
    VLLM_PD_DYNAMIC_SM_ENABLED=1 \
    VLLM_PD_DYNAMIC_SM_SHADOW=0 \
    VLLM_PD_DYNAMIC_SM_STRICT=1 \
    VLLM_PD_DYNAMIC_SM_TOTAL_SMS=40 \
    VLLM_PD_DYNAMIC_SM_ALIGNMENT=2 \
    VLLM_PD_DYNAMIC_SM_PROFILES="${PD_SM_PROFILES}" \
    VLLM_PD_DYNAMIC_SM_POLICY="${PD_DYNAMIC_SM_POLICY}" \
    VLLM_PD_TBT_PROFILE_PATH="${TBT_OVERLAP_PROFILE}" \
    VLLM_PD_TBT_DEFAULT_SLO_MS="${TBT_DEFAULT_SLO_MS}" \
    VLLM_PD_TBT_PROFILE_MARGIN_MS="${TBT_PROFILE_MARGIN_MS}" \
    VLLM_PD_TBT_PROFILE_STRICT="${TBT_PROFILE_STRICT}" \
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
    VLLM_CONTEXT_RING_TRACE_ENABLED=1 \
    VLLM_VIRTUAL_CONTEXT_RING_TRACE_JSONL="${CONTEXT_RING_TRACE}" \
    VLLM_PD_CUDA_EVENT_TRACE_ENABLED=0 \
    VLLM_PD_CUDA_EVENT_TRACE_JSONL="${PD_CUDA_EVENTS}" \
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
    VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" \
    VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS}" \
    VLLM_ENABLE_CHUNKED_PREFILL=1 \
    VLLM_DISABLE_PREFIX_CACHING=1 \
    VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS}" \
      "${SERVER_STARTER}" --port "${PORT}" --model "${MODEL_NAME}"
) >>"${SERVER_LOG}" 2>&1 &
ACTIVE_SERVER_PID="$!"
ACTIVE_SERVER_PGID="$(
  ps -o pgid= -p "${ACTIVE_SERVER_PID}" | tr -d '[:space:]'
)"
if ! wait_for_server_ready "${ACTIVE_SERVER_PID}"; then
  tail -n 180 "${SERVER_LOG}" >&2 || true
  exit 1
fi
if [[ "${ENABLE_DMON}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi dmon -s pucm -d 1 -o TD \
    >"${ARTIFACT_DIR}/nvidia_smi_dmon.log" 2>&1 &
  ACTIVE_DMON_PID="$!"
fi

"${PYTHON_BIN}" "${MEASURER}" \
  --base-url "${BASE_URL}" \
  --workload-jsonl "${WORKLOAD_JSONL}" \
  --slo-policy-json "${REQUEST_SLO_POLICY}" \
  --case-id "${EXPERIMENT_NAME}" \
  --request-timeout-sec "${REQUEST_TIMEOUT_SEC}" \
  --warmup-count "${WARMUP_COUNT}" \
  --warmup-output-tokens "${WARMUP_OUTPUT_TOKENS}" \
  --warmup-cooldown-sec "${WARMUP_COOLDOWN_SEC}" \
  --plan-csv "${RESULT_DIR}/submission_plan.csv" \
  --raw-csv "${RESULT_DIR}/request_results.csv" \
  --token-csv "${RESULT_DIR}/client_token_timeline.csv" \
  --summary-csv "${RESULT_DIR}/slo_summary.csv" \
  --metadata-json "${RESULT_DIR}/run_metadata.json"

cleanup_active
ACTIVE_SERVER_PID=""
ACTIVE_SERVER_PGID=""
ACTIVE_DMON_PID=""

cat >"${RESULT_DIR}/experiment_snapshot/settings.txt" <<EOF
experiment_name=${EXPERIMENT_NAME}
workload_kind=trace_driven_hybrid_not_raw_production_replay
burstgpt_csv=${BURSTGPT_CSV}
arrival_time_scale=${ARRIVAL_TIME_SCALE}
long_context_tokens=${LONG_CONTEXT_TOKENS}
long_request_count=${LONG_REQUEST_COUNT}
loogle_underlength_policy=${LOOGLE_UNDERLENGTH_POLICY}
short_request_count=${SHORT_REQUEST_COUNT}
output_length_source=BurstGPT
submission_mode=open_loop_trace_offsets
tbt_measurement=vllm_choice_token_ids
kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES}
prefill_scheduler=slo_srpt
decode_scheduler=fcfs
residency_victim_policy=${RESIDENCY_VICTIM_POLICY}
pd_dynamic_sm_policy=${PD_DYNAMIC_SM_POLICY}
pd_sm_profiles=${PD_SM_PROFILES}
initial_green_sm_split=${INITIAL_GREEN_SM_SPLIT}
request_slo_policy=${REQUEST_SLO_POLICY}
prefill_service_profile=${PREFILL_SERVICE_PROFILE}
require_prefill_profile_coverage=${REQUIRE_PREFILL_PROFILE_COVERAGE}
EOF

if [[ "${GENERATE_ABCD}" == "1" \
      && -f "${ROOT_DIR}/graph_script/generate_pd_abcd_timeline_graphs.py" ]]; then
  mkdir -p "${RESULT_DIR}/abcd" "${ARTIFACT_DIR}/.matplotlib"
  MPLCONFIGDIR="${ARTIFACT_DIR}/.matplotlib" \
    "${PYTHON_BIN}" \
    "${ROOT_DIR}/graph_script/generate_pd_abcd_timeline_graphs.py" \
      --artifact-dir "${ARTIFACT_DIR}" \
      --output-dir "${RESULT_DIR}/abcd" \
      --decode-sms "${DECODE_SMS}" \
      --prefill-sms "${PREFILL_SMS}" || true
fi

echo "[DONE] results=${RESULT_DIR}"
echo "[DONE] artifacts=${ARTIFACT_DIR}"
echo "[DONE] logs=${LOG_DIR}"
