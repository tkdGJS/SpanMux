#!/usr/bin/env bash
set -euo pipefail

# Prepare and replay the public LoongServe Mixed workload against an already
# running OpenAI-compatible vLLM endpoint. This runner owns only dataset/trace
# and client-side measurement state; server launch flags remain independent so
# the exact same trace can be reused for Native and vendor variants.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-loongserve_mixed_realworld_v1}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/${EXPERIMENT_NAME}_results}"
CLIENT_RESULT_DIR="${CLIENT_RESULT_DIR:-${RESULT_DIR}/client}"
TRACE_JSONL="${TRACE_JSONL:-${RESULT_DIR}/experiment_snapshot/loongserve_mixed.jsonl}"

VENV_DIR="${VENV_DIR:-/home/noslab-gpu/tkdgjs/tkdgjs}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL_NAME="${MODEL_NAME:-}"

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data/raw/loongserve}"
SHAREGPT_PATH="${SHAREGPT_PATH:-${ROOT_DIR}/data/ShareGPT_V3_unfiltered_cleaned_split.json}"
LEVAL_PATH="${LEVAL_PATH:-${DATA_ROOT}/LEval/LEval-data}"
LVEVAL_PATH="${LVEVAL_PATH:-${DATA_ROOT}/LVEval}"
LVEVAL_SUBSETS="${LVEVAL_SUBSETS:-factrecall_en hotpotwikiqa_mixup loogle_CR_mixup loogle_MIR_mixup loogle_SD_mixup multifieldqa_en_mixup}"

PREPARE_TRACE="${PREPARE_TRACE:-1}"
TOKENIZER="${TOKENIZER:-${MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}}"
TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"
TOKENIZER_LOCAL_FILES_ONLY="${TOKENIZER_LOCAL_FILES_ONLY:-1}"
TOKENIZER_WORKERS="${TOKENIZER_WORKERS:-1}"
TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-32}"
TOKENIZER_PROGRESS_EVERY="${TOKENIZER_PROGRESS_EVERY:-1000}"
SEED="${SEED:-0}"
REQUESTS_PER_DATASET="${REQUESTS_PER_DATASET:-1500}"
REQUEST_RATE="${REQUEST_RATE:-0.5}"
INSUFFICIENT_POLICY="${INSUFFICIENT_POLICY:-error}"
LENGTH_CAP_POLICY="${LENGTH_CAP_POLICY:-filter_then_sample}"

# A structural model limit normally applies to prompt + generated tokens.
# Set MAX_REQUEST_TOKENS=0 for the uncapped paper distribution. With
# paper_sample_then_truncate, only MAX_INPUT_TOKENS changes prompts; total/output
# caps validate the result and never remove a selected request.
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-0}"
MAX_REQUEST_TOKENS="${MAX_REQUEST_TOKENS:-${VLLM_MAX_MODEL_LEN:-65536}}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-0}"

# Effective arrival offset = paper trace offset * ARRIVAL_TIME_SCALE.
# 0.5 is twice as fast; 2.0 is twice as slow.
ARRIVAL_TIME_SCALE="${ARRIVAL_TIME_SCALE:-1.0}"

# TIME_LIMIT_SEC is measured from the production trace start, after warmup.
# 0 disables the limit. drain stops submission then waits; cancel also closes
# outstanding client streams and labels them censored_time_limit.
TERMINATION_MODE="${TERMINATION_MODE:-time}"
REQUEST_LIMIT="${REQUEST_LIMIT:-0}"
TIME_LIMIT_SEC="${TIME_LIMIT_SEC:-0}"
TIME_LIMIT_MODE="${TIME_LIMIT_MODE:-drain}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-14400}"

WARMUP_COUNT="${WARMUP_COUNT:-0}"
WARMUP_OUTPUT_TOKENS="${WARMUP_OUTPUT_TOKENS:-8}"
WARMUP_COOLDOWN_SEC="${WARMUP_COOLDOWN_SEC:-3}"
IGNORE_EOS="${IGNORE_EOS:-1}"

SLO_POLICY_JSON="${SLO_POLICY_JSON:-}"
SLO_OVERFLOW_POLICY="${SLO_OVERFLOW_POLICY:-error}"
DEFAULT_TTFT_SLO_MS="${DEFAULT_TTFT_SLO_MS:-0}"
DEFAULT_TBT_SLO_MS="${DEFAULT_TBT_SLO_MS:-0}"
ENABLE_DMON="${ENABLE_DMON:-1}"

RUN_MODE="${RUN_MODE:-prepare_and_replay}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./run_loongserve_mixed_workload.sh [--dry-run]

The target vLLM server must already be running at BASE_URL. RUN_MODE may be
prepare_and_replay, prepare_only, or replay_only.

Important controls:
  ARRIVAL_TIME_SCALE=0.5   replay twice as fast
  ARRIVAL_TIME_SCALE=2.0   replay twice as slow
  MAX_INPUT_TOKENS=N       filter by prompt length
  MAX_REQUEST_TOKENS=N     filter by prompt + requested output length
  LENGTH_CAP_POLICY=paper_sample_then_truncate
                          keep uncapped sample/order; head/tail input truncation
                          MAX_INPUT_TOKENS must be positive; exact HF required
                          total/output caps validate only (no dropped requests)
  TOKENIZER_WORKERS=N      native fast-tokenizer CPU threads
  TOKENIZER_BATCH_SIZE=N   candidate texts per tokenizer batch
  TERMINATION_MODE=time    use the production-time cutoff (default)
  TERMINATION_MODE=requests submit the first REQUEST_LIMIT trace rows, then drain
  REQUEST_LIMIT=N          positive request count in requests mode
  TIME_LIMIT_SEC=N         zero means unlimited
  TIME_LIMIT_MODE=drain    stop arrivals, finish admitted requests
  TIME_LIMIT_MODE=cancel   stop arrivals and censor active streams
EOF
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${RUN_MODE}" in
  prepare_and_replay|prepare_only|replay_only) ;;
  *) echo "[ERROR] invalid RUN_MODE=${RUN_MODE}" >&2; exit 2 ;;
esac
case "${TIME_LIMIT_MODE}" in
  drain|cancel) ;;
  *) echo "[ERROR] TIME_LIMIT_MODE must be drain or cancel" >&2; exit 2 ;;
esac
case "${TERMINATION_MODE}" in
  time|requests) ;;
  *) echo "[ERROR] TERMINATION_MODE must be time or requests" >&2; exit 2 ;;
esac
if [[ ! "${REQUEST_LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] REQUEST_LIMIT must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "${TIME_LIMIT_SEC}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  echo "[ERROR] TIME_LIMIT_SEC must be non-negative" >&2
  exit 2
fi
if [[ "${TERMINATION_MODE}" == "requests" ]]; then
  if [[ ! "${REQUEST_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] REQUEST_LIMIT must be positive in requests mode" >&2
    exit 2
  fi
  if [[ ! "${TIME_LIMIT_SEC}" =~ ^0*([.]0*)?$ ]]; then
    echo "[ERROR] TIME_LIMIT_SEC must be zero in requests mode" >&2
    exit 2
  fi
  if [[ "${TIME_LIMIT_MODE}" != "drain" ]]; then
    echo "[ERROR] TIME_LIMIT_MODE must be drain in requests mode" >&2
    exit 2
  fi
elif [[ ! "${REQUEST_LIMIT}" =~ ^0+$ ]]; then
  echo "[ERROR] REQUEST_LIMIT must be zero in time mode" >&2
  exit 2
fi
case "${PREPARE_TRACE}" in
  0|1) ;;
  *) echo "[ERROR] PREPARE_TRACE must be 0 or 1" >&2; exit 2 ;;
esac
case "${TOKENIZER_MODE}" in
  hf|approx) ;;
  *) echo "[ERROR] TOKENIZER_MODE must be hf or approx" >&2; exit 2 ;;
esac
case "${LENGTH_CAP_POLICY}" in
  filter_then_sample|paper_sample_then_filter|paper_sample_then_truncate) ;;
  *) echo "[ERROR] invalid LENGTH_CAP_POLICY=${LENGTH_CAP_POLICY}" >&2; exit 2 ;;
esac
if [[ "${LENGTH_CAP_POLICY}" == "paper_sample_then_truncate" ]]; then
  if [[ "${TOKENIZER_MODE}" != "hf" || ! "${MAX_INPUT_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] truncation requires TOKENIZER_MODE=hf and positive MAX_INPUT_TOKENS" >&2
    exit 2
  fi
fi
if [[ ! "${TOKENIZER_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] TOKENIZER_WORKERS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${TOKENIZER_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] TOKENIZER_BATCH_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ ! "${TOKENIZER_PROGRESS_EVERY}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] TOKENIZER_PROGRESS_EVERY must be a non-negative integer" >&2
  exit 2
fi

echo "[PLAN] workload=LoongServe Mixed (ShareGPT:L-Eval:LV-Eval=1:1:1)"
echo "[PLAN] requests_per_dataset=${REQUESTS_PER_DATASET} seed=${SEED} base_rate_rps=${REQUEST_RATE}"
echo "[PLAN] arrival_time_scale=${ARRIVAL_TIME_SCALE} (<1 faster, >1 slower)"
echo "[PLAN] max_input_tokens=${MAX_INPUT_TOKENS} max_request_tokens=${MAX_REQUEST_TOKENS} max_output_tokens=${MAX_OUTPUT_TOKENS}"
echo "[PLAN] length_cap_policy=${LENGTH_CAP_POLICY} insufficient_policy=${INSUFFICIENT_POLICY}"
echo "[PLAN] termination_mode=${TERMINATION_MODE} request_limit=${REQUEST_LIMIT}"
echo "[PLAN] time_limit_sec=${TIME_LIMIT_SEC} time_limit_mode=${TIME_LIMIT_MODE}"
echo "[PLAN] tokenizer=${TOKENIZER} mode=${TOKENIZER_MODE} workers=${TOKENIZER_WORKERS} batch_size=${TOKENIZER_BATCH_SIZE}"
echo "[PLAN] server=${BASE_URL} model=${MODEL_NAME:-<discover-from-server>}"
if [[ "${TOKENIZER_MODE}" == "approx" ]]; then
  echo "[WARN] approximate token counts are for schema smoke tests only"
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY-RUN] no files, HTTP requests, or GPU state changed"
  exit 0
fi

PREPARER="${ROOT_DIR}/prepare_loongserve_mixed_workload.py"
MEASURER="${ROOT_DIR}/measure_loongserve_mixed_workload.py"
for path in "${PREPARER}" "${MEASURER}"; do
  [[ -f "${path}" ]] || {
    echo "[ERROR] missing required script: ${path}" >&2
    exit 1
  }
done

if [[ "${RUN_MODE}" != "replay_only" && "${PREPARE_TRACE}" == "1" ]]; then
  for dataset_path in "${SHAREGPT_PATH}" "${LEVAL_PATH}" "${LVEVAL_PATH}"; do
    if [[ ! -e "${dataset_path}" ]]; then
      echo "[ERROR] missing dataset path: ${dataset_path}" >&2
      echo "[ERROR] follow the LoongServe artifact dataset guide or override SHAREGPT_PATH/LEVAL_PATH/LVEVAL_PATH" >&2
      exit 1
    fi
  done
fi
if [[ -e "${RESULT_DIR}" ]]; then
  echo "[ERROR] result directory exists; choose a new EXPERIMENT_NAME: ${RESULT_DIR}" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}/experiment_snapshot"
cp -a "${BASH_SOURCE[0]}" "${PREPARER}" "${MEASURER}" \
  "${RESULT_DIR}/experiment_snapshot/"

read -r -a lveval_subset_array <<<"${LVEVAL_SUBSETS}"
if [[ "${RUN_MODE}" != "replay_only" && "${PREPARE_TRACE}" == "1" ]]; then
  prepare_args=(
    --sharegpt-path "${SHAREGPT_PATH}"
    --leval-path "${LEVAL_PATH}"
    --lveval-path "${LVEVAL_PATH}"
    --output-path "${TRACE_JSONL}"
    --tokenizer "${TOKENIZER}"
    --tokenizer-mode "${TOKENIZER_MODE}"
    --tokenizer-workers "${TOKENIZER_WORKERS}"
    --tokenizer-batch-size "${TOKENIZER_BATCH_SIZE}"
    --progress-every "${TOKENIZER_PROGRESS_EVERY}"
    --seed "${SEED}"
    --requests-per-dataset "${REQUESTS_PER_DATASET}"
    --request-rate "${REQUEST_RATE}"
    --max-input-tokens "${MAX_INPUT_TOKENS}"
    --max-total-tokens "${MAX_REQUEST_TOKENS}"
    --max-output-tokens "${MAX_OUTPUT_TOKENS}"
    --length-cap-policy "${LENGTH_CAP_POLICY}"
    --insufficient-policy "${INSUFFICIENT_POLICY}"
    --lveval-subsets "${lveval_subset_array[@]}"
  )
  if [[ "${TOKENIZER_LOCAL_FILES_ONLY}" == "1" ]]; then
    prepare_args+=(--local-files-only)
  fi
  "${PYTHON_BIN}" "${PREPARER}" "${prepare_args[@]}"
fi

[[ -f "${TRACE_JSONL}" ]] || {
  echo "[ERROR] missing prepared trace: ${TRACE_JSONL}" >&2
  exit 1
}

# Persist the complete replay contract before the client starts. If the server
# or client fails mid-run, the incomplete result directory still explains
# exactly which immutable trace and timing/cap controls were used.
cat >"${RESULT_DIR}/experiment_snapshot/settings.txt" <<EOF
experiment_name=${EXPERIMENT_NAME}
workload=LoongServe Mixed
trace_jsonl=${TRACE_JSONL}
sharegpt_path=${SHAREGPT_PATH}
leval_path=${LEVAL_PATH}
lveval_path=${LVEVAL_PATH}
tokenizer=${TOKENIZER}
tokenizer_mode=${TOKENIZER_MODE}
tokenizer_workers=${TOKENIZER_WORKERS}
tokenizer_batch_size=${TOKENIZER_BATCH_SIZE}
tokenizer_progress_every=${TOKENIZER_PROGRESS_EVERY}
seed=${SEED}
requests_per_dataset=${REQUESTS_PER_DATASET}
request_rate_rps=${REQUEST_RATE}
arrival_time_scale=${ARRIVAL_TIME_SCALE}
max_input_tokens=${MAX_INPUT_TOKENS}
max_request_tokens=${MAX_REQUEST_TOKENS}
max_output_tokens=${MAX_OUTPUT_TOKENS}
length_cap_policy=${LENGTH_CAP_POLICY}
termination_mode=${TERMINATION_MODE}
request_limit=${REQUEST_LIMIT}
time_limit_sec=${TIME_LIMIT_SEC}
time_limit_mode=${TIME_LIMIT_MODE}
base_url=${BASE_URL}
model_name=${MODEL_NAME}
EOF

if [[ "${RUN_MODE}" == "prepare_only" ]]; then
  echo "[DONE] prepared_trace=${TRACE_JSONL}"
  exit 0
fi

dmon_pid=""
cleanup() {
  if [[ -n "${dmon_pid}" ]]; then
    kill "${dmon_pid}" 2>/dev/null || true
    wait "${dmon_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
if [[ "${ENABLE_DMON}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi dmon -s pucm -d 1 -o TD \
    >"${RESULT_DIR}/nvidia_smi_dmon.log" 2>&1 &
  dmon_pid="$!"
fi

measure_args=(
  --base-url "${BASE_URL}"
  --workload-jsonl "${TRACE_JSONL}"
  --case-id "${EXPERIMENT_NAME}"
  --output-dir "${CLIENT_RESULT_DIR}"
  --arrival-time-scale "${ARRIVAL_TIME_SCALE}"
  --termination-mode "${TERMINATION_MODE}"
  --request-limit "${REQUEST_LIMIT}"
  --time-limit-sec "${TIME_LIMIT_SEC}"
  --time-limit-mode "${TIME_LIMIT_MODE}"
  --request-timeout-sec "${REQUEST_TIMEOUT_SEC}"
  --ignore-eos "${IGNORE_EOS}"
  --warmup-count "${WARMUP_COUNT}"
  --warmup-output-tokens "${WARMUP_OUTPUT_TOKENS}"
  --warmup-cooldown-sec "${WARMUP_COOLDOWN_SEC}"
  --slo-overflow-policy "${SLO_OVERFLOW_POLICY}"
  --default-ttft-slo-ms "${DEFAULT_TTFT_SLO_MS}"
  --default-tbt-slo-ms "${DEFAULT_TBT_SLO_MS}"
)
if [[ -n "${MODEL_NAME}" ]]; then
  measure_args+=(--model "${MODEL_NAME}")
fi
if [[ -n "${SLO_POLICY_JSON}" ]]; then
  [[ -f "${SLO_POLICY_JSON}" ]] || {
    echo "[ERROR] missing SLO policy: ${SLO_POLICY_JSON}" >&2
    exit 1
  }
  measure_args+=(--slo-policy-json "${SLO_POLICY_JSON}")
fi

"${PYTHON_BIN}" "${MEASURER}" "${measure_args[@]}"
cleanup
dmon_pid=""

echo "[DONE] trace=${TRACE_JSONL}"
echo "[DONE] results=${CLIENT_RESULT_DIR}"
