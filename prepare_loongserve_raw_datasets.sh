#!/usr/bin/env bash
set -euo pipefail

# Check and, when missing, download the raw datasets used by the LoongServe
# Mixed workload. Existing dataset files are validated but never overwritten.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOONGSERVE_REPO_ROOT="${LOONGSERVE_REPO_ROOT:-${SCRIPT_DIR}}"
LOONGSERVE_DATA_ROOT="${LOONGSERVE_DATA_ROOT:-${LOONGSERVE_REPO_ROOT}/data/raw/loongserve}"
LOONGSERVE_SHAREGPT_PATH="${LOONGSERVE_SHAREGPT_PATH:-${LOONGSERVE_REPO_ROOT}/data/ShareGPT_V3_unfiltered_cleaned_split.json}"
LOONGSERVE_LEVAL_ROOT="${LOONGSERVE_LEVAL_ROOT:-${LOONGSERVE_DATA_ROOT}/LEval}"
LOONGSERVE_LVEVAL_ROOT="${LOONGSERVE_LVEVAL_ROOT:-${LOONGSERVE_DATA_ROOT}/LVEval}"

LOONGSERVE_SHAREGPT_URL="${LOONGSERVE_SHAREGPT_URL:-https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json}"
LOONGSERVE_LEVAL_REPO_URL="${LOONGSERVE_LEVAL_REPO_URL:-https://github.com/OpenLMLab/LEval.git}"
LOONGSERVE_LVEVAL_BASE_URL="${LOONGSERVE_LVEVAL_BASE_URL:-https://huggingface.co/datasets/Infinigence/LVEval/resolve/main}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CHECK_ONLY=0
DRY_RUN=0
MISSING_COUNT=0
DOWNLOAD_COUNT=0
EXTRACT_COUNT=0
SHAREGPT_STATUS="unchecked"
LEVAL_STATUS="unchecked"
LVEVAL_STATUS="unchecked"

LVEVAL_SUBSETS=(
  factrecall_en
  hotpotwikiqa_mixup
  loogle_CR_mixup
  loogle_MIR_mixup
  loogle_SD_mixup
  multifieldqa_en_mixup
)
LVEVAL_LEVELS=(16k 32k 64k 128k 256k)

usage() {
  cat <<'EOF'
Usage: ./prepare_loongserve_raw_datasets.sh [--check-only] [--dry-run]

Without arguments, validate existing datasets, download only missing datasets,
extract the six English LV-Eval subsets, and print a final validation summary.

Options:
  --check-only  Never download; report every missing dataset and exit 2.
  --dry-run     Print missing data and planned actions without changing files.
  -h, --help    Show this help.

Useful environment overrides:
  LOONGSERVE_REPO_ROOT
  LOONGSERVE_DATA_ROOT
  LOONGSERVE_SHAREGPT_PATH
  LOONGSERVE_LEVAL_ROOT
  LOONGSERVE_LVEVAL_ROOT
  LOONGSERVE_SHAREGPT_URL
  LOONGSERVE_LEVAL_REPO_URL
  LOONGSERVE_LVEVAL_BASE_URL
  PYTHON_BIN
EOF
}

while (($#)); do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${CHECK_ONLY}" == "1" && "${DRY_RUN}" == "1" ]]; then
  echo "[ERROR] --check-only and --dry-run are mutually exclusive" >&2
  exit 2
fi

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

require_command() {
  if ! command_exists "$1"; then
    echo "[ERROR] required command is unavailable: $1" >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local destination="$2"
  local label="$3"
  local temporary_path

  mkdir -p "$(dirname "${destination}")"
  temporary_path="${TEMP_DIR}/$(basename "${destination}").download"
  echo "[DOWNLOAD] ${label}"
  echo "[DOWNLOAD] source=${url}"
  echo "[DOWNLOAD] destination=${destination}"
  if command_exists curl; then
    curl --fail --location --retry 5 --output "${temporary_path}" "${url}"
  elif command_exists wget; then
    wget --tries=5 --output-document="${temporary_path}" "${url}"
  else
    echo "[ERROR] curl or wget is required for dataset download" >&2
    exit 1
  fi
  if [[ ! -s "${temporary_path}" ]]; then
    echo "[ERROR] downloaded file is empty: ${label}" >&2
    exit 1
  fi
  mv "${temporary_path}" "${destination}"
  ((DOWNLOAD_COUNT += 1))
}

validate_sharegpt() {
  local path="$1"
  "${PYTHON_BIN}" - "${path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    with path.open(encoding="utf-8") as stream:
        rows = json.load(stream)
except Exception as exc:
    raise SystemExit(f"[ERROR] invalid ShareGPT JSON {path}: {exc}")
if not isinstance(rows, list) or not rows:
    raise SystemExit(f"[ERROR] ShareGPT must be a non-empty JSON list: {path}")
valid = sum(
    isinstance(row, dict)
    and isinstance(row.get("conversations"), list)
    and bool(row["conversations"])
    for row in rows
)
if valid == 0:
    raise SystemExit(f"[ERROR] no valid ShareGPT conversations found: {path}")
print(
    f"[VERIFY] dataset=ShareGPT rows={len(rows)} "
    f"valid_conversations={valid} bytes={path.stat().st_size}"
)
PY
}

validate_leval() {
  local path="$1"
  "${PYTHON_BIN}" - "${path}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(root.rglob("*.jsonl"))
if not files:
    raise SystemExit(f"[ERROR] no L-Eval JSONL files found: {root}")
rows = 0
valid = 0
for path in files:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                raise SystemExit(
                    f"[ERROR] invalid L-Eval JSONL {path}:{line_number}: {exc}"
                )
            rows += 1
            valid += int(
                isinstance(row, dict)
                and isinstance(row.get("input"), str)
                and isinstance(row.get("instructions"), list)
                and isinstance(row.get("outputs"), list)
            )
if rows == 0 or valid == 0:
    raise SystemExit(f"[ERROR] no valid L-Eval rows found: {root}")
print(f"[VERIFY] dataset=L-Eval files={len(files)} rows={rows} valid_rows={valid}")
PY
}

lveval_subset_complete() {
  local subset="$1"
  local level
  for level in "${LVEVAL_LEVELS[@]}"; do
    if [[ ! -s "${LOONGSERVE_LVEVAL_ROOT}/${subset}/${subset}_${level}.jsonl" ]]; then
      return 1
    fi
  done
  return 0
}

validate_lveval() {
  local path="$1"
  shift
  "${PYTHON_BIN}" - "${path}" "$@" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
subsets = sys.argv[2:]
levels = ("16k", "32k", "64k", "128k", "256k")
files = []
rows = 0
valid = 0
for subset in subsets:
    for level in levels:
        path = root / subset / f"{subset}_{level}.jsonl"
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"[ERROR] missing LV-Eval file: {path}")
        files.append(path)
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    raise SystemExit(
                        f"[ERROR] invalid LV-Eval JSONL {path}:{line_number}: {exc}"
                    )
                rows += 1
                valid += int(
                    isinstance(row, dict)
                    and isinstance(row.get("context"), str)
                    and isinstance(row.get("input"), str)
                    and isinstance(row.get("answers"), list)
                )
if rows == 0 or valid == 0:
    raise SystemExit(f"[ERROR] no valid LV-Eval rows found: {root}")
print(f"[VERIFY] dataset=LV-Eval files={len(files)} rows={rows} valid_rows={valid}")
PY
}

require_command "${PYTHON_BIN}"

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/loongserve_raw_datasets.XXXXXX")"
cleanup() {
  rm -rf -- "${TEMP_DIR}"
}
trap cleanup EXIT

echo "[PLAN] repo_root=${LOONGSERVE_REPO_ROOT}"
echo "[PLAN] data_root=${LOONGSERVE_DATA_ROOT}"
echo "[PLAN] sharegpt=${LOONGSERVE_SHAREGPT_PATH}"
echo "[PLAN] leval=${LOONGSERVE_LEVAL_ROOT}/LEval-data"
echo "[PLAN] lveval=${LOONGSERVE_LVEVAL_ROOT}"
echo "[PLAN] mode=$([[ "${CHECK_ONLY}" == "1" ]] && echo check_only || ([[ "${DRY_RUN}" == "1" ]] && echo dry_run || echo prepare))"

echo "[CHECK] ShareGPT"
if [[ -s "${LOONGSERVE_SHAREGPT_PATH}" ]]; then
  validate_sharegpt "${LOONGSERVE_SHAREGPT_PATH}"
  SHAREGPT_STATUS="found"
else
  echo "[MISSING] ${LOONGSERVE_SHAREGPT_PATH}"
  ((MISSING_COUNT += 1))
  SHAREGPT_STATUS="missing"
  if [[ "${CHECK_ONLY}" == "0" && "${DRY_RUN}" == "0" ]]; then
    download_file \
      "${LOONGSERVE_SHAREGPT_URL}" \
      "${TEMP_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json" \
      "ShareGPT"
    validate_sharegpt "${TEMP_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json"
    mkdir -p "$(dirname "${LOONGSERVE_SHAREGPT_PATH}")"
    mv "${TEMP_DIR}/ShareGPT_V3_unfiltered_cleaned_split.json" \
      "${LOONGSERVE_SHAREGPT_PATH}"
    SHAREGPT_STATUS="downloaded"
  elif [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] would download ShareGPT from ${LOONGSERVE_SHAREGPT_URL}"
  fi
fi

echo "[CHECK] L-Eval"
if [[ -d "${LOONGSERVE_LEVAL_ROOT}/LEval-data" ]]; then
  validate_leval "${LOONGSERVE_LEVAL_ROOT}/LEval-data"
  LEVAL_STATUS="found"
else
  echo "[MISSING] ${LOONGSERVE_LEVAL_ROOT}/LEval-data"
  ((MISSING_COUNT += 1))
  LEVAL_STATUS="missing"
  if [[ "${CHECK_ONLY}" == "0" && "${DRY_RUN}" == "0" ]]; then
    require_command git
    if [[ -e "${LOONGSERVE_LEVAL_ROOT}" ]]; then
      echo "[ERROR] incomplete L-Eval path already exists; refusing to overwrite: ${LOONGSERVE_LEVAL_ROOT}" >&2
      exit 1
    fi
    echo "[DOWNLOAD] L-Eval repository"
    echo "[DOWNLOAD] source=${LOONGSERVE_LEVAL_REPO_URL}"
    echo "[DOWNLOAD] destination=${LOONGSERVE_LEVAL_ROOT}"
    git clone --depth 1 "${LOONGSERVE_LEVAL_REPO_URL}" \
      "${TEMP_DIR}/LEval"
    validate_leval "${TEMP_DIR}/LEval/LEval-data"
    mkdir -p "$(dirname "${LOONGSERVE_LEVAL_ROOT}")"
    mv "${TEMP_DIR}/LEval" "${LOONGSERVE_LEVAL_ROOT}"
    ((DOWNLOAD_COUNT += 1))
    LEVAL_STATUS="downloaded"
  elif [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] would clone L-Eval from ${LOONGSERVE_LEVAL_REPO_URL}"
  fi
fi

echo "[CHECK] LV-Eval English subsets"
LVEVAL_STATUS="found"
mkdir_lveval=0
for subset in "${LVEVAL_SUBSETS[@]}"; do
  if lveval_subset_complete "${subset}"; then
    echo "[FOUND] LV-Eval subset=${subset} levels=5"
    continue
  fi
  echo "[MISSING] LV-Eval subset=${subset}"
  ((MISSING_COUNT += 1))
  LVEVAL_STATUS="missing"
  if [[ "${CHECK_ONLY}" == "1" ]]; then
    continue
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] would download and extract ${subset}.zip"
    continue
  fi
  require_command unzip
  if [[ "${mkdir_lveval}" == "0" ]]; then
    mkdir -p "${LOONGSERVE_LVEVAL_ROOT}"
    mkdir_lveval=1
  fi
  archive="${LOONGSERVE_LVEVAL_ROOT}/${subset}.zip"
  if [[ ! -e "${archive}" ]]; then
    download_file \
      "${LOONGSERVE_LVEVAL_BASE_URL}/${subset}.zip" \
      "${TEMP_DIR}/${subset}.zip" \
      "LV-Eval subset ${subset}"
    if ! unzip -tq "${TEMP_DIR}/${subset}.zip" >/dev/null 2>&1; then
      echo "[ERROR] downloaded LV-Eval archive failed validation: ${subset}.zip" >&2
      exit 1
    fi
    mv "${TEMP_DIR}/${subset}.zip" "${archive}"
  elif ! unzip -tq "${archive}" >/dev/null 2>&1; then
    echo "[ERROR] existing LV-Eval archive is invalid; refusing to overwrite: ${archive}" >&2
    exit 1
  else
    echo "[FOUND] valid archive=${archive}"
  fi
  if ! unzip -tq "${archive}" >/dev/null 2>&1; then
    echo "[ERROR] downloaded LV-Eval archive failed validation: ${archive}" >&2
    exit 1
  fi
  echo "[EXTRACT] subset=${subset} destination=${LOONGSERVE_LVEVAL_ROOT}"
  unzip -qn "${archive}" -d "${LOONGSERVE_LVEVAL_ROOT}"
  ((EXTRACT_COUNT += 1))
  if ! lveval_subset_complete "${subset}"; then
    echo "[ERROR] LV-Eval subset remains incomplete after extraction: ${subset}" >&2
    exit 1
  fi
  LVEVAL_STATUS="downloaded_or_completed"
done

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "[SUMMARY] sharegpt=${SHAREGPT_STATUS} leval=${LEVAL_STATUS} lveval=${LVEVAL_STATUS} missing=${MISSING_COUNT}"
  if ((MISSING_COUNT > 0)); then
    echo "[SUMMARY] ready=0 action=rerun_without_check_only"
    exit 2
  fi
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[SUMMARY] sharegpt=${SHAREGPT_STATUS} leval=${LEVAL_STATUS} lveval=${LVEVAL_STATUS} missing=${MISSING_COUNT}"
  echo "[SUMMARY] ready=$((MISSING_COUNT == 0 ? 1 : 0)) mode=dry_run files_changed=0"
  exit 0
fi

echo "[VERIFY] final dataset contract"
validate_sharegpt "${LOONGSERVE_SHAREGPT_PATH}"
validate_leval "${LOONGSERVE_LEVAL_ROOT}/LEval-data"
validate_lveval "${LOONGSERVE_LVEVAL_ROOT}" "${LVEVAL_SUBSETS[@]}"

echo "[SUMMARY] sharegpt=${SHAREGPT_STATUS} leval=${LEVAL_STATUS} lveval=${LVEVAL_STATUS}"
echo "[SUMMARY] downloads=${DOWNLOAD_COUNT} extractions=${EXTRACT_COUNT} ready=1"
echo "[OK] LoongServe Mixed raw datasets are ready"
