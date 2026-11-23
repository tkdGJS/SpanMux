#!/usr/bin/env python3
"""Expand explicit patch seeds through the semantic dependency graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graph_lib import (
    DEFAULT_GRAPH_PATH,
    REPO_ROOT,
    build_impact_slice,
    impact_slice_markdown,
    load_graph,
    validate_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--node", action="append", default=[])
    parser.add_argument("--symbol", action="append", default=[])
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Explicit task-owned file; do not infer from a dirty global diff.",
    )
    parser.add_argument("--flag", action="append", default=[])
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--direction", choices=("both", "in", "out"), default="both")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-unmapped",
        action="store_true",
        help=(
            "Report but do not fail unresolved selectors "
            "(not valid for pre-edit approval)."
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not any((args.node, args.symbol, args.file, args.flag)):
        print("[ERROR] at least one --node/--symbol/--file/--flag seed is required")
        return 2
    try:
        graph = load_graph(args.graph)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2
    errors, warnings = validate_graph(graph, repo_root=args.repo_root.resolve())
    for warning in warnings:
        print(f"[WARN] {warning}")
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        return 2
    try:
        impact = build_impact_slice(
            graph,
            node_seeds=args.node,
            symbol_seeds=args.symbol,
            file_seeds=args.file,
            flag_seeds=args.flag,
            depth=args.depth,
            direction=args.direction,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 2

    unresolved = (
        impact.unresolved_nodes
        + impact.unresolved_symbols
        + impact.unresolved_files
        + impact.unresolved_flags
    )
    if not impact.seeds:
        print("[ERROR] no graph node matched the supplied patch seeds")
        return 3

    if args.format == "json":
        content = json.dumps(
            {
                "seeds": impact.seeds,
                "distances": impact.distances,
                "edges": impact.edges,
                "unresolved_nodes": impact.unresolved_nodes,
                "unresolved_symbols": impact.unresolved_symbols,
                "unresolved_files": impact.unresolved_files,
                "unresolved_flags": impact.unresolved_flags,
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    else:
        content = impact_slice_markdown(graph, impact)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"[OK] wrote impact slice: {args.output}")
    else:
        print(content, end="")
    if unresolved and not args.allow_unmapped:
        print("[ERROR] unresolved patch seeds must be mapped before editing")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
