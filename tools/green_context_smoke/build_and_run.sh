#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
CXX="${CXX:-g++}"
GREEN_CONTEXT_DEVICE="${GREEN_CONTEXT_DEVICE:-0}"
GREEN_CONTEXT_SMS="${GREEN_CONTEXT_SMS:-8}"
GREEN_CONTEXT_PRIORITY="${GREEN_CONTEXT_PRIORITY:--1}"
GREEN_CONTEXT_ELEMENTS="${GREEN_CONTEXT_ELEMENTS:-4096}"
LIBCUDA_LDFLAGS="${LIBCUDA_LDFLAGS:--lcuda}"

SRC="${ROOT_DIR}/tools/green_context_smoke/green_context_smoke.cc"
BUILD_DIR="${ROOT_DIR}/tools/green_context_smoke/build"
BIN="${BUILD_DIR}/green_context_smoke"

CUDA_INCLUDE="${CUDA_HOME}/targets/x86_64-linux/include"
if [[ ! -f "${CUDA_INCLUDE}/cuda.h" ]]; then
  CUDA_INCLUDE="${CUDA_HOME}/include"
fi

CUDA_H="${CUDA_INCLUDE}/cuda.h"
CUDA_TYPEDEFS_H="${CUDA_INCLUDE}/cudaTypedefs.h"

if [[ ! -f "${CUDA_H}" ]]; then
  echo "[ERROR] cuda.h not found under CUDA_HOME=${CUDA_HOME}" >&2
  exit 1
fi

if ! grep -Eq "CUgreenCtx|cuGreenCtxCreate" "${CUDA_H}" "${CUDA_TYPEDEFS_H}" 2>/dev/null; then
  echo "[ERROR] CUDA headers do not expose Green Context APIs: ${CUDA_INCLUDE}" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"

echo "[BUILD] CUDA_HOME=${CUDA_HOME}"
echo "[BUILD] include=${CUDA_INCLUDE}"
echo "[BUILD] output=${BIN}"
"${CXX}" -std=c++17 -O2 -Wall -Wextra -I"${CUDA_INCLUDE}" \
  "${SRC}" ${LIBCUDA_LDFLAGS} -o "${BIN}"

echo "[RUN] device=${GREEN_CONTEXT_DEVICE} green_sms=${GREEN_CONTEXT_SMS} priority=${GREEN_CONTEXT_PRIORITY}"
"${BIN}" \
  --device "${GREEN_CONTEXT_DEVICE}" \
  --green-sms "${GREEN_CONTEXT_SMS}" \
  --priority "${GREEN_CONTEXT_PRIORITY}" \
  --elements "${GREEN_CONTEXT_ELEMENTS}"
