#!/usr/bin/env python3
"""Validate prospective patch-effect metrics and impact-record coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

from patch_metrics import REPO_ROOT, validate_repository_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    errors, records = validate_repository_records(repo_root=repo_root)
    if errors:
        print(f"[ERROR] patch metrics validation failed ({len(errors)} errors)")
        for error in errors:
            print(f"- {error}")
        return 1
    complete = sum(record.get("status") == "complete" for record in records)
    print(
        f"[OK] patch metrics are valid: records={len(records)} "
        f"complete={complete} in_progress={len(records) - complete}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
