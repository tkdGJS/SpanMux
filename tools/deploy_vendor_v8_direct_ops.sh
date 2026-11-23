#!/usr/bin/env bash
set -euo pipefail

# Atomically deploy an already rebuilt vLLM CUDA extension into vendor_v8 and
# verify the exact file through the same PYTHONPATH used by benchmark workers.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_EXTENSION="${SOURCE_EXTENSION:-${ROOT_DIR}/src/vllm-0.14-src/vllm/_C.abi3.so}"
VLLM_VENDOR_DIR="${VLLM_VENDOR_DIR:-${ROOT_DIR}/vendor_v8}"
VLLM_PYTHON_EXECUTABLE="${VLLM_PYTHON_EXECUTABLE:-python}"
DESTINATION_EXTENSION="${VLLM_VENDOR_DIR}/vllm/_C.abi3.so"

if [[ ! -f "${SOURCE_EXTENSION}" ]]; then
  echo "[ERROR] rebuilt extension does not exist: ${SOURCE_EXTENSION}" >&2
  echo "[HINT] build src/vllm-0.14-src first, then set SOURCE_EXTENSION" >&2
  exit 2
fi
if [[ ! -d "${VLLM_VENDOR_DIR}/vllm" ]]; then
  echo "[ERROR] invalid VLLM_VENDOR_DIR: ${VLLM_VENDOR_DIR}" >&2
  exit 2
fi

temporary_extension="${DESTINATION_EXTENSION}.direct-ops-new"
backup_extension="${DESTINATION_EXTENSION}.direct-ops-backup"
cp -- "${SOURCE_EXTENSION}" "${temporary_extension}"
if [[ -f "${DESTINATION_EXTENSION}" ]]; then
  cp -- "${DESTINATION_EXTENSION}" "${backup_extension}"
fi
mv -- "${temporary_extension}" "${DESTINATION_EXTENSION}"

if ! env PYTHONPATH="${VLLM_VENDOR_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${VLLM_PYTHON_EXECUTABLE}" \
  "${ROOT_DIR}/tools/virtual_context_smoke/verify_tiered_direct_ops.py" \
  --vendor-dir "${VLLM_VENDOR_DIR}"; then
  echo "[ERROR] deployed extension failed verification" >&2
  if [[ -f "${backup_extension}" ]]; then
    mv -- "${backup_extension}" "${DESTINATION_EXTENSION}"
    echo "[RECOVERED] restored previous vendor_v8 extension" >&2
  fi
  exit 3
fi

rm -f -- "${backup_extension}"
echo "[OK] deployed verified direct-copy extension: ${DESTINATION_EXTENSION}"
