#!/usr/bin/env python3
"""Validate dependency graph schema and referenced checkout symbols."""

from __future__ import annotations

import argparse
from pathlib import Path

from graph_lib import (
    DEFAULT_GRAPH_PATH,
    REPO_ROOT,
    format_validation_result,
    load_graph,
    validate_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--skip-symbol-validation",
        action="store_true",
        help="Validate schema and paths without parsing qualified symbols.",
    )
    parser.add_argument("--warnings-as-errors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        graph = load_graph(args.graph)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2
    errors, warnings = validate_graph(
        graph,
        repo_root=args.repo_root.resolve(),
        validate_symbols=not args.skip_symbol_validation,
    )
    print(format_validation_result(errors, warnings))
    if errors or (warnings and args.warnings_as_errors):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
