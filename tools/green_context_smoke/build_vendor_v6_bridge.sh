#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
CXX="${CXX:-g++}"
LIBCUDA_LDFLAGS="${LIBCUDA_LDFLAGS:--lcuda}"

OUT_DIR="${ROOT_DIR}/vendor_v6/vllm/model_executor/layers/green_context_lib"
SRC="${OUT_DIR}/green_context_bridge.cc"
OUT="${OUT_DIR}/libvllm_green_context.so"

CUDA_INCLUDE="${CUDA_HOME}/targets/x86_64-linux/include"
if [[ ! -f "${CUDA_INCLUDE}/cuda.h" ]]; then
  CUDA_INCLUDE="${CUDA_HOME}/include"
fi

CUDA_H="${CUDA_INCLUDE}/cuda.h"
CUDA_TYPEDEFS_H="${CUDA_INCLUDE}/cudaTypedefs.h"

if [[ ! -f "${SRC}" ]]; then
  echo "[ERROR] vendor_v6 Green Context source not found: ${SRC}" >&2
  exit 1
fi

if [[ ! -f "${CUDA_H}" ]]; then
  echo "[ERROR] cuda.h not found under CUDA_HOME=${CUDA_HOME}" >&2
  exit 1
fi

if ! grep -Eq "CUgreenCtx|cuGreenCtxCreate" \
  "${CUDA_H}" "${CUDA_TYPEDEFS_H}" 2>/dev/null; then
  echo "[ERROR] CUDA headers do not expose Green Context APIs: ${CUDA_INCLUDE}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "[BUILD] vendor=vendor_v6"
echo "[BUILD] CUDA_HOME=${CUDA_HOME}"
echo "[BUILD] include=${CUDA_INCLUDE}"
echo "[BUILD] source=${SRC}"
echo "[BUILD] output=${OUT}"
"${CXX}" -std=c++17 -O2 -Wall -Wextra -fPIC -shared \
  -I"${CUDA_INCLUDE}" "${SRC}" ${LIBCUDA_LDFLAGS} -o "${OUT}"

echo "[OK] built ${OUT}"
