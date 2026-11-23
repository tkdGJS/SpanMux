#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENDOR_DIR="${VLLM_VENDOR_DIR:-${ROOT_DIR}/vendor_v8}"
SMOKE="${ROOT_DIR}/tools/virtual_context_smoke/tiered_backend_provider_smoke.py"
STAGING_SMOKE="${ROOT_DIR}/tools/virtual_context_smoke/staging_cross_thread_inference_smoke.py"
MULTICOPY_CONTRACT="${ROOT_DIR}/tools/virtual_context_smoke/tiered_h2d_multicopy_contract_smoke.py"
MULTICOPY_SMOKE="${ROOT_DIR}/tools/virtual_context_smoke/tiered_h2d_multicopy_smoke.py"
BATCHED_ENQUEUE_CONTRACT="${ROOT_DIR}/tools/virtual_context_smoke/context_ring_batched_enqueue_contract_smoke.py"

# Catch first-touch allocation under model inference mode followed by reuse in
# the native-reload executor before running the longer CUDA exactness matrix.
python "${STAGING_SMOKE}"
python "${MULTICOPY_CONTRACT}"
python "${MULTICOPY_SMOKE}"
python "${BATCHED_ENQUEUE_CONTRACT}"

run_case() {
  local query_tokens="$1"
  local blocks="$2"
  local block_tokens="$3"
  local query_heads="$4"
  local kv_heads="$5"
  local head_dim="$6"
  python "${SMOKE}" \
    --vendor-dir "${VENDOR_DIR}" \
    --device cuda \
    --fully-optimized \
    --batched-partial \
    --batched-partial-strict \
    --register-from-hbm \
    --query-tokens "${query_tokens}" \
    --blocks "${blocks}" \
    --block-tokens "${block_tokens}" \
    --query-heads "${query_heads}" \
    --kv-heads "${kv_heads}" \
    --head-dim "${head_dim}"
}

run_case 1 4 16 8 2 64
run_case 7 5 16 8 2 64
run_case 17 6 16 16 4 64
run_case 31 8 16 8 8 128

python "${SMOKE}" \
  --vendor-dir "${VENDOR_DIR}" \
  --device cuda \
  --fully-optimized \
  --batched-partial \
  --batched-partial-strict \
  --register-from-hbm \
  --request-count 3 \
  --query-tokens 7 \
  --blocks 5 \
  --block-tokens 16 \
  --query-heads 8 \
  --kv-heads 2 \
  --head-dim 64

echo "[OK] fully optimized CUDA exactness matrix completed"
