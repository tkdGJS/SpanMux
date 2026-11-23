#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname "$script_dir")"
config="$script_dir/thesis.env"
if [[ "${1:-}" == --config ]]; then
  if (( $# < 2 )); then
    echo 'Usage: run_thesis.sh [--config FILE] [--dry-run]' >&2
    exit 2
  fi
  config="$2"
  shift 2
fi
case "$config" in
  /*) ;;
  *) config="$PWD/$config" ;;
esac
cd "$repo_root"

# The config is a trusted local shell file. Export its assignments to the sweep.
set -a
source "$config"
VLLM_EXECUTABLE="${VLLM_EXECUTABLE:-$VENV_DIR/bin/vllm}"
VLLM_PYTHON_EXECUTABLE="${VLLM_PYTHON_EXECUTABLE:-$VENV_DIR/bin/python}"
VLLM_VENDOR_DIR="${VLLM_VENDOR_DIR:-$repo_root/spanmux}"
NATIVE_VLLM_VENDOR_DIR="${NATIVE_VLLM_VENDOR_DIR-}"
if [[ -d "$repo_root/profiling" ]]; then
  default_profile_dir="$repo_root/profiling"
else
  default_profile_dir="$repo_root/vendor_v8_runtime_inputs_1token_splitwise_v2"
fi
PROFILE_RESULTS_DIR="${PROFILE_RESULTS_DIR:-$default_profile_dir}"
REQUEST_SLO_POLICY="${REQUEST_SLO_POLICY:-$PROFILE_RESULTS_DIR/slo/request_slo_policy.json}"
set +a

launcher="$script_dir/run_vendor_v8_dynamic_hbm_vs_full.sh"
if [[ ! -f "$launcher" ]]; then
  launcher="$repo_root/run_vendor_v8_dynamic_hbm_vs_full.sh"
fi
exec bash "$launcher" "$@"
