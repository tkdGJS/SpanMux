#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENDOR_DIR="${VLLM_VENDOR_DIR:-${ROOT_DIR}/vendor_v8}"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" \
  bash "${ROOT_DIR}/tools/green_context_smoke/build_vendor_v8_bridge.sh"

export PYTHONPATH="${VENDOR_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_OPERATOR_SELECTIVE_PD_GREEN_CONTEXT=1
export VLLM_OPERATOR_SELECTIVE_PD_GREEN_STRICT=1
export VLLM_OPERATOR_SELECTIVE_PD_GREEN_EXECUTE_MODEL=1
export VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS="${VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS:-32}"
export VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS="${VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS:-8}"

exec "${PYTHON_BIN:-python}" \
  "${ROOT_DIR}/tools/green_context_smoke/test_vendor_v8_green_decode_cudagraph.py"
