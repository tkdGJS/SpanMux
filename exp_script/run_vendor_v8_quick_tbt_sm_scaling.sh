#!/usr/bin/env bash
set -euo pipefail

# Fast TBT screening:
# - one long Decode anchor is created per partition case;
# - one warm-up and several short Prefill probes reuse that anchor;
# - paired pre-probe and overlap gaps are emitted in the standard partition
#   profile schema.
#
# Probe-window repeats share one Decode request. They are descriptive screening
# clusters, not independent request repetitions. Use the full profiler to
# confirm only the boundary splits selected by this run.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-vendor_v8_quick_tbt_sm_scaling_100k_v1}"
export CONTEXT_TOKENS="${CONTEXT_TOKENS:-100000}"
export GREEN_SM_SPLITS="${GREEN_SM_SPLITS:-32/8 34/6 36/4 38/2}"
export PREFILL_MODES="${PREFILL_MODES:-ring}"
export ACTIVE_DECODE_REQUEST_COUNTS="${ACTIVE_DECODE_REQUEST_COUNTS:-1}"
export PREFILL_PROBE_CONTEXT_TOKENS="${PREFILL_PROBE_CONTEXT_TOKENS:-2048}"
export REPEAT_COUNT="${REPEAT_COUNT:-5}"
export WARMUP_COUNT="${WARMUP_COUNT:-1}"
export ANCHOR_OUTPUT_TOKENS="${ANCHOR_OUTPUT_TOKENS:-4096}"
export PROBE_OUTPUT_TOKENS="${PROBE_OUTPUT_TOKENS:-1}"
export PROBE_DELAY_SEC="${PROBE_DELAY_SEC:-8}"
export DECODE_ONLY_CONTROL_GAPS="${DECODE_ONLY_CONTROL_GAPS:-32}"
export CASE_COOLDOWN_SEC="${CASE_COOLDOWN_SEC:-0}"
export PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-2048}"
export NATIVE_PROFILE_MAX_RESIDENT_TOKENS="${NATIVE_PROFILE_MAX_RESIDENT_TOKENS:-0}"
export REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-43200}"
export FIRST_TOKEN_TIMEOUT_SEC="${FIRST_TOKEN_TIMEOUT_SEC:-43200}"
export SM_SCALING_BOOTSTRAP_ITERATIONS="${SM_SCALING_BOOTSTRAP_ITERATIONS:-10000}"
export SM_SCALING_BOOTSTRAP_SEED="${SM_SCALING_BOOTSTRAP_SEED:-20260804}"
export SM_SCALING_ALLOW_WINDOW_CLUSTERS=1
export VLLM_PD_DECODE_GPU_COMPONENT_TRACE=0
export VLLM_PD_GREEN_DECODE_CUDAGRAPH="${VLLM_PD_GREEN_DECODE_CUDAGRAPH:-1}"
export VLLM_PD_GREEN_DECODE_CUDAGRAPH_STRICT="${VLLM_PD_GREEN_DECODE_CUDAGRAPH_STRICT:-1}"
export VENDOR_V8_PARTITION_MEASURER="${ROOT_DIR}/measure_vendor_v8_quick_tbt_sm_scaling.py"

exec "${ROOT_DIR}/run_vendor_v8_partition_profiler.sh" "$@"
