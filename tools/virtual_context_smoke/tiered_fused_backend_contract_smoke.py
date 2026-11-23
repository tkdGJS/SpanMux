#!/usr/bin/env python3
"""Import-free contract check for the fused tiered attention patch.

The development host does not always have PyTorch/CUDA.  This check therefore
verifies the source-level integration contract; the numerical CUDA comparison
remains in ``tiered_blockwise_attention_smoke.py --triton-fused-update``.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FUSED_MODULE = (
    ROOT
    / "vendor_v4/vllm/v1/attention/ops/tiered_triton_fused_attention.py"
)
EXECUTOR_MODULE = (
    ROOT / "vendor_v4/vllm/v1/attention/ops/tiered_blockwise_attention.py"
)


def defined_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def require_text(path: Path, needles: tuple[str, ...]) -> None:
    source = path.read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in source]
    if missing:
        raise AssertionError(f"{path}: missing contract markers: {missing}")


def main() -> int:
    required_functions = {
        "_fused_online_softmax_update_kernel",
        "_finalize_online_softmax_kernel",
        "create_triton_online_softmax_state",
        "update_triton_online_softmax_state",
        "finalize_triton_online_softmax_state",
    }
    missing_functions = required_functions - defined_functions(FUSED_MODULE)
    if missing_functions:
        raise AssertionError(
            f"missing fused backend functions: {sorted(missing_functions)}"
        )

    require_text(
        FUSED_MODULE,
        (
            "m_i = tl.load(",
            "l_i = tl.load(",
            "acc = tl.load(",
            "tl.store(STATE_MAX",
            "tl.store(STATE_EXP_SUM",
            "tl.dot(p.to(v.dtype), v)",
            "native_reduction: tl.constexpr",
            "qk = tl.dot(q, k, acc=qk)",
            'input_precision="ieee"',
        ),
    )
    require_text(
        EXECUTOR_MODULE,
        (
            "VLLM_VIRTUAL_CONTEXT_RING_TRITON_FUSED_UPDATE",
            "VLLM_VIRTUAL_CONTEXT_RING_NATIVE_REDUCTION",
            "native_reduction=native_reduction_enabled",
            "triton_fused_update_successes == compute_group_count",
            "context_ring_triton_fused_q_tile_loop_bypassed",
            "fused tiered attention failed after mutating",
        ),
    )
    print("[OK] fused tiered attention source contract validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
