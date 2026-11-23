#!/usr/bin/env python3
"""Render or freshness-check the semantic dependency graph documentation."""

from __future__ import annotations

import argparse
from pathlib import Path

from graph_lib import (
    DEFAULT_GRAPH_PATH,
    DEFAULT_RENDERED_PATH,
    REPO_ROOT,
    load_graph,
    render_graph_markdown,
    validate_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RENDERED_PATH)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if output is absent or differs; do not write it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    rendered = render_graph_markdown(graph)
    if args.check:
        try:
            current = args.output.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"[ERROR] generated graph is missing: {args.output}")
            return 1
        if current != rendered:
            print(f"[ERROR] generated graph is stale: {args.output}")
            return 1
        print(f"[OK] generated graph is current: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"[OK] rendered dependency graph: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
