#!/usr/bin/env bash
set -euo pipefail

# Fast boundary-level comparison of one fixed Decode partition:
#   A. token gaps immediately before a Prefill probe (Decode-only control)
#   B. token gaps and sampled Decode forward intervals that overlap the probe
#
# The smoke mode checks the plumbing with 10K Decode context.  The target mode
# reproduces the 100K/30-SM question but its one-time 100K Prefill dominates
# wall time on a T4.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICROBENCH_MODE="${MICROBENCH_MODE:-smoke}"
ANALYZE_EXISTING="${ANALYZE_EXISTING:-0}"
DECODE_SMS="${DECODE_SMS:-30}"
TOTAL_SMS="${EXPECTED_GPU_SMS:-40}"
PREFILL_SMS=$((TOTAL_SMS - DECODE_SMS))

if ((DECODE_SMS <= 0 || PREFILL_SMS <= 0)); then
  echo "[ERROR] DECODE_SMS must leave positive Decode and Prefill partitions" >&2
  exit 2
fi
if [[ "${ANALYZE_EXISTING}" != "0" && "${ANALYZE_EXISTING}" != "1" ]]; then
  echo "[ERROR] ANALYZE_EXISTING must be 0 or 1" >&2
  exit 2
fi

case "${MICROBENCH_MODE}" in
  smoke)
    CONTEXT_TOKENS="${CONTEXT_TOKENS:-10000}"
    REPEAT_COUNT="${REPEAT_COUNT:-1}"
    ANCHOR_OUTPUT_TOKENS="${ANCHOR_OUTPUT_TOKENS:-1024}"
    PROBE_DELAY_SEC="${PROBE_DELAY_SEC:-1}"
    DECODE_ONLY_CONTROL_GAPS="${DECODE_ONLY_CONTROL_GAPS:-16}"
    BASELINE_RESULTS_DIR="${BASELINE_RESULTS_DIR:-${ROOT_DIR}/native_40sm_tbt_sm_scaling_10k_exact_v1_results}"
    EXPECTED_RUNTIME="about 3-6 minutes on the T4, including server startup"
    ;;
  target)
    CONTEXT_TOKENS="${CONTEXT_TOKENS:-100000}"
    REPEAT_COUNT="${REPEAT_COUNT:-2}"
    ANCHOR_OUTPUT_TOKENS="${ANCHOR_OUTPUT_TOKENS:-512}"
    PROBE_DELAY_SEC="${PROBE_DELAY_SEC:-3}"
    DECODE_ONLY_CONTROL_GAPS="${DECODE_ONLY_CONTROL_GAPS:-16}"
    BASELINE_RESULTS_DIR="${BASELINE_RESULTS_DIR:-${ROOT_DIR}/native_40sm_tbt_sm_scaling_100k_exact_v1_results}"
    EXPECTED_RUNTIME="about 55-75 minutes on the T4; static 30/10 makes the one-time 100K Prefill the dominant cost"
    ;;
  *)
    echo "[ERROR] MICROBENCH_MODE must be smoke or target" >&2
    exit 2
    ;;
esac

EXPERIMENT_NAME="${EXPERIMENT_NAME:-vendor_v8_decode_overlap_${MICROBENCH_MODE}_d${DECODE_SMS}_v1}"
RESULT_DIR="${ROOT_DIR}/${EXPERIMENT_NAME}_results"
ARTIFACT_DIR="${ROOT_DIR}/${EXPERIMENT_NAME}_artifacts"
PROFILE_ID="d${DECODE_SMS}_p${PREFILL_SMS}"
MODE="native"

export EXPERIMENT_NAME
export BASELINE_RESULTS_DIR
export CONTEXT_TOKENS
export GREEN_SM_SPLITS="${DECODE_SMS}/${PREFILL_SMS}"
export PREFILL_MODES="${MODE}"
export ACTIVE_DECODE_REQUEST_COUNTS=1
export PREFILL_PROBE_CONTEXT_TOKENS="${PREFILL_PROBE_CONTEXT_TOKENS:-2048}"
export REPEAT_COUNT
export WARMUP_COUNT=0
export ANCHOR_OUTPUT_TOKENS
export PROBE_OUTPUT_TOKENS=1
export PROBE_DELAY_SEC
export DECODE_ONLY_CONTROL_GAPS
export CASE_COOLDOWN_SEC=0
export PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-2048}"
export NATIVE_PROFILE_MAX_RESIDENT_TOKENS=0
export SM_SCALING_ALLOW_WINDOW_CLUSTERS=1
export SM_SCALING_BOOTSTRAP_ITERATIONS="${SM_SCALING_BOOTSTRAP_ITERATIONS:-1000}"

# Keep the serving path intact: no module hooks, no synchronous GPU timing,
# no full per-iteration dictionaries.  Sample only forward boundaries; every
# Prefill forward is retained so Decode samples can be classified by exact
# shared-epoch device overlap.
export ENABLE_ITERATION_METRICS=0
export RUN_STANDARD_ANALYZERS=0
export ENABLE_DMON="${ENABLE_DMON:-0}"
export VLLM_PD_DECODE_GPU_COMPONENT_TRACE=0
export VLLM_ITERATION_GPU_TIMING=0
export VLLM_LAYER_KERNEL_TIMING=0
export VLLM_PD_CUDA_EVENT_TRACE_ENABLED=1
export VLLM_PD_CUDA_EVENT_TRACE_PHASES=decode,prefill
export VLLM_PD_CUDA_EVENT_TRACE_KINDS=forward
export VLLM_PD_CUDA_EVENT_TRACE_DECODE_EVERY_N="${VLLM_PD_CUDA_EVENT_TRACE_DECODE_EVERY_N:-4}"
export VLLM_PD_CUDA_EVENT_TRACE_PREFILL_EVERY_N=1
export VLLM_PD_CUDA_EVENT_TRACE_MAX_PENDING="${VLLM_PD_CUDA_EVENT_TRACE_MAX_PENDING:-4096}"
export VLLM_PD_GREEN_DECODE_CUDAGRAPH="${VLLM_PD_GREEN_DECODE_CUDAGRAPH:-1}"
export VLLM_PD_GREEN_DECODE_CUDAGRAPH_STRICT="${VLLM_PD_GREEN_DECODE_CUDAGRAPH_STRICT:-1}"
export VENDOR_V8_PARTITION_MEASURER="${ROOT_DIR}/measure_vendor_v8_quick_tbt_sm_scaling.py"

echo "[PLAN] mode=${MICROBENCH_MODE} context=${CONTEXT_TOKENS} split=${DECODE_SMS}/${PREFILL_SMS}"
echo "[PLAN] repeats=${REPEAT_COUNT} control_gaps=${DECODE_ONLY_CONTROL_GAPS} decode_trace_every_n=${VLLM_PD_CUDA_EVENT_TRACE_DECODE_EVERY_N}"
echo "[PLAN] expected_runtime=${EXPECTED_RUNTIME}"
echo "[PLAN] detailed_component_hooks=off iteration_metrics=off standard_analyzers=off client_tbt=choice.token_ids"
echo "[PLAN] analyze_existing=${ANALYZE_EXISTING}"

if [[ "${ANALYZE_EXISTING}" == "0" ]]; then
  "${ROOT_DIR}/run_vendor_v8_partition_profiler.sh" "$@"
else
  echo "[PLAN] reuse existing result/artifact directories; no server is started"
fi

for argument in "$@"; do
  if [[ "${argument}" == "--dry-run" ]]; then
    exit 0
  fi
done

CUDA_TRACE="${ARTIFACT_DIR}/${MODE}/${PROFILE_ID}/pd_cuda_events.jsonl"
TRIAL_CSV="${RESULT_DIR}/cases/${MODE}/${PROFILE_ID}/trial_summary.csv"
ANALYSIS_DIR="${RESULT_DIR}/decode_overlap_boundary_analysis"
if [[ ! -s "${CUDA_TRACE}" ]]; then
  echo "[ERROR] missing CUDA boundary trace: ${CUDA_TRACE}" >&2
  exit 1
fi
if [[ ! -s "${TRIAL_CSV}" ]]; then
  echo "[ERROR] missing client trial summary: ${TRIAL_CSV}" >&2
  exit 1
fi

python "${ROOT_DIR}/tools/analyze_vendor_v8_decode_overlap_microbench.py" \
  --cuda-trace "${CUDA_TRACE}" \
  --trial-csv "${TRIAL_CSV}" \
  --output-dir "${ANALYSIS_DIR}"

mkdir -p "${RESULT_DIR}/experiment_snapshot/scripts"
cp -a \
  "${BASH_SOURCE[0]}" \
  "${ROOT_DIR}/tools/analyze_vendor_v8_decode_overlap_microbench.py" \
  "${RESULT_DIR}/experiment_snapshot/scripts/"

echo "[DONE] boundary_analysis=${ANALYSIS_DIR}"
