#!/usr/bin/env python3
"""Import-free source contract for direct fragmented DRAM-to-HBM reload."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "src/vllm-0.14-src/csrc/cache_kernels.cu"
HEADER = ROOT / "src/vllm-0.14-src/csrc/cache.h"
BINDINGS = ROOT / "src/vllm-0.14-src/csrc/torch_bindings.cpp"
SOURCE_WRAPPER = ROOT / "src/vllm-0.14-src/vllm/_custom_ops.py"
VENDOR_WRAPPER = ROOT / "vendor_v8/vllm/_custom_ops.py"
PROVIDER = (
    ROOT
    / "vendor_v8/vllm/v1/attention/ops/tiered_residency_provider.py"
)
POLICY = ROOT / "tools/pd_eval_policy_env.sh"


def require(path: Path, *markers: str) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    assert not missing, f"{path}: missing source markers {missing}"


def main() -> None:
    require(HEADER, "int64_t tiered_h2d_multicopy(")
    require(
        KERNEL,
        "int64_t tiered_h2d_multicopy(",
        "destination_block_ids.data_ptr<int64_t>()",
        "at::cuda::getCurrentCUDAStream()",
        "cudaMemcpyHostToDevice",
        "cudaMemcpyAsync(",
        "has_contiguous_inner_page",
        "destination_pages_are_adjacent",
        "key_block_stride_bytes >= key_page_bytes",
        "block_ids[last_page] == block_ids[last_page - 1] + 1",
    )
    require(
        BINDINGS,
        '"tiered_h2d_multicopy(Tensor src_key, Tensor src_value, "',
        'cache_ops.impl("tiered_h2d_multicopy", torch::kCUDA',
    )
    for wrapper in (SOURCE_WRAPPER, VENDOR_WRAPPER):
        require(
            wrapper,
            "def tiered_h2d_multicopy(",
            "torch.ops._C_cache_ops.tiered_h2d_multicopy(",
        )
    require(
        PROVIDER,
        "VLLM_VIRTUAL_CONTEXT_RELOAD_DIRECT_MULTICOPY",
        "tiered_dram_reload_direct_multicopy_enabled()",
        "_custom_ops_module().tiered_h2d_multicopy(",
        "dram_provider_reload_to_hbm_direct_multicopy_ops",
        "dram_provider_reload_to_hbm_direct_multicopy_runs",
        "dram_provider_reload_to_hbm_direct_multicopy_pages",
        "dram_provider_reload_to_hbm_direct_multicopy_bytes",
    )
    require(
        POLICY,
        "VLLM_VIRTUAL_CONTEXT_RELOAD_DIRECT_MULTICOPY=0",
        "VLLM_PD_EVAL_RELOAD_DIRECT_MULTICOPY:-0",
    )
    print("[OK] tiered H2D multicopy source contract")


if __name__ == "__main__":
    main()
