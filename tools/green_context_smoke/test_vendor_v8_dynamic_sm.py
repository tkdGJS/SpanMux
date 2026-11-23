#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the vendor_v7 GreenContext GPU contract against vendor_v8."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("VLLM_VENDOR_DIR", str(root / "vendor_v8"))
    script = Path(__file__).with_name("test_vendor_v7_dynamic_sm.py")
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
        if exit_code:
            return exit_code
    print("[OK] vendor_v8 inherited dynamic P/D profile bank validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
