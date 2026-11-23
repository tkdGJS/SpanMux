#!/usr/bin/env python3
"""Import-free source contract for fragmented HBM-to-DRAM multicopy."""

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
    require(HEADER, "int64_t tiered_d2h_multicopy(")
    require(
        KERNEL,
        "int64_t tiered_d2h_multicopy(",
        "at::cuda::getCurrentCUDAStream()",
        "cudaMemcpyDeviceToHost",
        "cudaMemcpyAsync(",
        "next_key != previous_key + key_page_bytes",
        "destinations must use pinned CPU memory",
    )
    require(
        BINDINGS,
        '"tiered_d2h_multicopy(Tensor[] src_keys, Tensor[] src_values, "',
        'cache_ops.impl("tiered_d2h_multicopy", torch::kCUDA',
    )
    for wrapper in (SOURCE_WRAPPER, VENDOR_WRAPPER):
        require(
            wrapper,
            "def tiered_d2h_multicopy(",
            "torch.ops._C_cache_ops.tiered_d2h_multicopy(",
        )
    require(
        PROVIDER,
        "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_DIRECT_MULTICOPY",
        "tiered_dram_register_direct_multicopy_enabled()",
        "_custom_ops_module().tiered_d2h_multicopy(",
        "not direct_multicopy_enabled",
        "dram_provider_register_from_hbm_direct_multicopy_ops",
        "dram_provider_register_from_hbm_direct_multicopy_runs",
        "dram_provider_register_from_hbm_direct_multicopy_pages",
        "dram_provider_register_from_hbm_direct_multicopy_bytes",
    )
    require(
        POLICY,
        "VLLM_VIRTUAL_CONTEXT_DRAM_REGISTER_DIRECT_MULTICOPY=0",
        "VLLM_PD_EVAL_DRAM_REGISTER_DIRECT_MULTICOPY:-0",
    )
    print("[OK] tiered D2H multicopy source contract")


if __name__ == "__main__":
    main()
