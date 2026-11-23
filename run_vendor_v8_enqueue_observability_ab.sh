#!/usr/bin/env bash
set -uo pipefail

# Four-way ablation:
#   1. concurrent host enqueue + full per-iteration debug
#   2. decode-first host enqueue + full per-iteration debug
#   3. concurrent host enqueue + minimal TBT controller state
#   4. decode-first host enqueue + minimal TBT controller state

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME:-vendor_v8_enqueue_observability_ab_v1}"
AB_MODE="${AB_MODE:-smoke}"
CPU_DIR="${CPU_DIR:-${ROOT_DIR}/${BASE_EXPERIMENT_NAME}_cpu}"
RUNNER="${ROOT_DIR}/run_vendor_v8_short_validation.sh"
ANALYZER="${ROOT_DIR}/tools/analyze_vendor_v8_enqueue_observability_ab.py"
DRY_RUN=0

while (($#)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      echo "Usage: AB_MODE=smoke|full $0 [--dry-run]"
      exit 0
      ;;
    *)
      echo "[ERROR] unknown option: $1" >&2
      exit 2
      ;;
  esac
done

case "${AB_MODE}" in
  smoke)
    LONG_OUTPUT_TOKENS="${LONG_OUTPUT_TOKENS:-256}"
    SHORT_OUTPUT_TOKENS="${SHORT_OUTPUT_TOKENS:-64}"
    SHORT_INJECTION_OFFSETS_SEC="${SHORT_INJECTION_OFFSETS_SEC:-2 4 6 8 10}"
    WARMUP_COUNT="${WARMUP_COUNT:-0}"
    ;;
  full)
    LONG_OUTPUT_TOKENS="${LONG_OUTPUT_TOKENS:-4096}"
    SHORT_OUTPUT_TOKENS="${SHORT_OUTPUT_TOKENS:-512}"
    SHORT_INJECTION_OFFSETS_SEC="${SHORT_INJECTION_OFFSETS_SEC:-60 120 180 240 300}"
    WARMUP_COUNT="${WARMUP_COUNT:-1}"
    ;;
  *)
    echo "[ERROR] AB_MODE must be smoke or full" >&2
    exit 2
    ;;
esac

if [[ "${DRY_RUN}" == "0" ]]; then
  mkdir -p "${CPU_DIR}"
fi
overall_status=0
for case_name in \
  legacy_contention_debug \
  enqueue_ordered_debug \
  contention_minimal \
  optimized_minimal; do
  decode_first=0
  full_debug=0
  sample_every=100
  case "${case_name}" in
    legacy_contention_debug)
      full_debug=1
      sample_every=1
      ;;
    enqueue_ordered_debug)
      decode_first=1
      full_debug=1
      sample_every=1
      ;;
    contention_minimal) ;;
    optimized_minimal)
      decode_first=1
      ;;
  esac
  experiment="${BASE_EXPERIMENT_NAME}_${case_name}"
  echo "[CASE] ${experiment} decode_first=${decode_first} full_debug=${full_debug}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    env \
      EXPERIMENT_NAME="${experiment}" \
      SYSTEM_VARIANT=full_vendor_v8 \
      SUBMISSION_MODE=concurrent_longs \
      PREFILL_SCHEDULING_POLICY="${PREFILL_SCHEDULING_POLICY:-srpt}" \
      DECODE_FIRST_HOST_ENQUEUE="${decode_first}" \
      ENABLE_FULL_ITERATION_TRACE="${full_debug}" \
      ITERATION_TRACE_SAMPLE_EVERY_N="${sample_every}" \
      LONG_OUTPUT_TOKENS="${LONG_OUTPUT_TOKENS}" \
      SHORT_OUTPUT_TOKENS="${SHORT_OUTPUT_TOKENS}" \
      SHORT_INJECTION_OFFSETS_SEC="${SHORT_INJECTION_OFFSETS_SEC}" \
      WARMUP_COUNT="${WARMUP_COUNT}" \
      "${RUNNER}" --dry-run
    continue
  fi
  if ! /usr/bin/time -v -o "${CPU_DIR}/${experiment}.time.txt" \
    env \
      EXPERIMENT_NAME="${experiment}" \
      SYSTEM_VARIANT=full_vendor_v8 \
      SUBMISSION_MODE=concurrent_longs \
      PREFILL_SCHEDULING_POLICY="${PREFILL_SCHEDULING_POLICY:-srpt}" \
      DECODE_FIRST_HOST_ENQUEUE="${decode_first}" \
      ENABLE_FULL_ITERATION_TRACE="${full_debug}" \
      ITERATION_TRACE_SAMPLE_EVERY_N="${sample_every}" \
      LONG_OUTPUT_TOKENS="${LONG_OUTPUT_TOKENS}" \
      SHORT_OUTPUT_TOKENS="${SHORT_OUTPUT_TOKENS}" \
      SHORT_INJECTION_OFFSETS_SEC="${SHORT_INJECTION_OFFSETS_SEC}" \
      WARMUP_COUNT="${WARMUP_COUNT}" \
      ENABLE_DMON="${ENABLE_DMON:-1}" \
      GENERATE_ABCD=0 \
      "${RUNNER}"; then
    overall_status=1
    echo "[CASE] failed: ${experiment}" >&2
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY-RUN] no server or workload was started"
  exit 0
fi

python "${ANALYZER}" \
  --root "${ROOT_DIR}" \
  --base-experiment "${BASE_EXPERIMENT_NAME}" \
  --cpu-dir "${CPU_DIR}"
exit "${overall_status}"
