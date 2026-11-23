#!/usr/bin/env python3
"""Verify the direct-copy ops in the exact vLLM extension used by a worker."""

import argparse
import importlib
import sys
from pathlib import Path


REQUIRED_OPS = (
    "tiered_d2h_multicopy",
    "tiered_h2d_multicopy",
    "tiered_h2d_staging_multicopy",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        required=True,
        help="Directory placed in PYTHONPATH by the benchmark worker",
    )
    args = parser.parse_args()
    vendor_dir = args.vendor_dir.resolve()
    sys.path.insert(0, str(vendor_dir))

    import torch

    importlib.import_module("vllm._custom_ops")
    extension = importlib.import_module("vllm._C")
    extension_path = Path(extension.__file__).resolve()
    namespace = torch.ops._C_cache_ops
    missing = [name for name in REQUIRED_OPS if not hasattr(namespace, name)]

    print(f"vendor_dir={vendor_dir}")
    print(f"loaded_extension={extension_path}")
    print(f"loaded_extension_size={extension_path.stat().st_size}")
    for name in REQUIRED_OPS:
        print(f"op.{name}={'present' if name not in missing else 'missing'}")
    if missing:
        print(
            "[ERROR] rebuilt vllm/_C.abi3.so was not deployed with the "
            f"vendor tree; missing={','.join(missing)}",
            file=sys.stderr,
        )
        return 2
    print("[OK] all tiered direct-copy ops are loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
