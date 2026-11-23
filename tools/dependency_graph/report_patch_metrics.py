#!/usr/bin/env python3
"""Print descriptive patch-effect measurements with causal-claim guards."""

from __future__ import annotations

from patch_metrics import summarize_records, validate_repository_records


def main() -> int:
    errors, records = validate_repository_records()
    if errors:
        print("[ERROR] metrics are invalid; run validate_patch_metrics.py")
        for error in errors:
            print(f"- {error}")
        return 1
    print(summarize_records(records), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
