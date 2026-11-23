#!/usr/bin/env python3
"""Import-free source contract for direct Context Ring staging copies."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "src/vllm-0.14-src/csrc/cache_kernels.cu"
HEADER = ROOT / "src/vllm-0.14-src/csrc/cache.h"
BINDINGS = ROOT / "src/vllm-0.14-src/csrc/torch_bindings.cpp"
SOURCE_WRAPPER = ROOT / "src/vllm-0.14-src/vllm/_custom_ops.py"
VENDOR_WRAPPER = ROOT / "vendor_v8/vllm/_custom_ops.py"
RING = ROOT / "vendor_v8/vllm/v1/attention/ops/tiered_blockwise_attention.py"
POLICY = ROOT / "tools/pd_eval_policy_env.sh"


def require(path: Path, *markers: str) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    assert not missing, f"{path}: missing source markers {missing}"


def main() -> None:
    require(HEADER, "int64_t tiered_h2d_staging_multicopy(")
    require(
        KERNEL,
        "int64_t tiered_h2d_staging_multicopy(",
        "cudaMemcpyDeviceToDevice",
        "cudaMemcpyHostToDevice",
        "destination_token_offsets.data_ptr<int64_t>()",
        "at::cuda::getCurrentCUDAStream()",
    )
    require(
        BINDINGS,
        '"tiered_h2d_staging_multicopy(Tensor[] src_keys, "',
        'cache_ops.impl("tiered_h2d_staging_multicopy", torch::kCUDA',
    )
    for wrapper in (SOURCE_WRAPPER, VENDOR_WRAPPER):
        require(
            wrapper,
            "def tiered_h2d_staging_multicopy(",
            "torch.ops._C_cache_ops.tiered_h2d_staging_multicopy(",
        )
    require(
        RING,
        "VLLM_VIRTUAL_CONTEXT_RING_STAGING_DIRECT_MULTICOPY",
        "def _try_direct_multicopy(",
        "_custom_ops_module().tiered_h2d_staging_multicopy(",
        "destination_key=staged_key.unsqueeze(0)",
        "context_ring_staging_direct_multicopy_successes",
        "falling back to per-source Tensor.copy_",
    )
    require(
        POLICY,
        "VLLM_VIRTUAL_CONTEXT_RING_STAGING_DIRECT_MULTICOPY=0",
        "VLLM_PD_EVAL_RING_STAGING_DIRECT_MULTICOPY:-1",
        "VLLM_VIRTUAL_CONTEXT_DIRECT_MULTICOPY_STRICT",
    )
    print("[OK] tiered Context Ring staging multicopy source contract")


if __name__ == "__main__":
    main()
