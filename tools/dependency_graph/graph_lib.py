"""Shared helpers for the repository semantic dependency graph.

The graph intentionally models contracts that a static import graph cannot
recover: ownership, terminal paths, asynchronous wakeups, and policy/resource
coupling.  This module uses only the Python standard library so the pre-edit
check works in the lightweight repository interpreter.
"""

from __future__ import annotations

import ast
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_PATH = REPO_ROOT / "docs/dependency_graph/system_dependencies.json"
DEFAULT_RENDERED_PATH = (
    REPO_ROOT / "docs/dependency_graph/generated/system_dependencies.md"
)


@dataclass(frozen=True)
class ImpactSlice:
    """A bounded, deterministic graph expansion from explicit patch seeds."""

    seeds: tuple[str, ...]
    distances: Mapping[str, int]
    edges: tuple[Mapping[str, Any], ...]
    unresolved_nodes: tuple[str, ...] = ()
    unresolved_symbols: tuple[str, ...] = ()
    unresolved_files: tuple[str, ...] = ()
    unresolved_flags: tuple[str, ...] = ()


def load_graph(path: Path | str = DEFAULT_GRAPH_PATH) -> dict[str, Any]:
    graph_path = Path(path)
    try:
        data = json.loads(graph_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"dependency graph does not exist: {graph_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"dependency graph is not valid JSON: {graph_path}:{exc.lineno}:"
            f"{exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError("dependency graph root must be a JSON object")
    return data


def _is_safe_relative_path(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


class _PythonSymbolCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.symbols: set[str] = set()

    def _visit_named(self, node: ast.AST, name: str) -> None:
        self.stack.append(name)
        self.symbols.add(".".join(self.stack))
        self.generic_visit(node)
        self.stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._visit_named(node, node.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_named(node, node.name)

    def visit_AsyncFunctionDef(  # noqa: N802
        self, node: ast.AsyncFunctionDef
    ) -> None:
        self._visit_named(node, node.name)


def collect_python_symbols(path: Path) -> set[str]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise ValueError(f"cannot parse Python symbol source {path}: {exc}") from exc
    collector = _PythonSymbolCollector()
    collector.visit(tree)
    return collector.symbols


def _validate_code_reference(
    reference: object,
    *,
    repo_root: Path,
    symbol_cache: dict[Path, set[str]],
) -> str | None:
    if not isinstance(reference, str):
        return f"code reference must be a string: {reference!r}"
    path_text, separator, symbol = reference.partition("::")
    if not _is_safe_relative_path(path_text):
        return f"code reference path must be repository-relative: {reference!r}"
    path = repo_root / path_text
    if not path.is_file():
        return f"code reference path does not exist: {reference}"
    if not separator or not symbol:
        return f"code reference must use path::qualified_symbol: {reference!r}"
    if path.suffix == ".py":
        try:
            if path not in symbol_cache:
                symbol_cache[path] = collect_python_symbols(path)
            symbols = symbol_cache[path]
        except ValueError as exc:
            return str(exc)
        if symbol not in symbols:
            return f"Python symbol does not exist: {reference}"
        return None
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"cannot read code reference {reference}: {exc}"
    if symbol not in source:
        return f"symbol text does not exist: {reference}"
    return None


def validate_graph(
    graph: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    validate_symbols: bool = True,
) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for schema and checkout consistency."""

    errors: list[str] = []
    warnings: list[str] = []
    if graph.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    node_kinds = graph.get("node_kinds")
    edge_types = graph.get("edge_types")
    required_terminal_paths = graph.get("required_terminal_paths")
    if not isinstance(node_kinds, list) or not all(
        isinstance(value, str) and value for value in node_kinds
    ):
        errors.append("node_kinds must be a non-empty string list")
        node_kinds = []
    if not isinstance(edge_types, list) or not all(
        isinstance(value, str) and value for value in edge_types
    ):
        errors.append("edge_types must be a non-empty string list")
        edge_types = []
    if not isinstance(required_terminal_paths, list) or not all(
        isinstance(value, str) and value for value in required_terminal_paths
    ):
        errors.append("required_terminal_paths must be a string list")
        required_terminal_paths = []

    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not nodes:
        errors.append("nodes must be a non-empty list")
        nodes = []
    if not isinstance(edges, list) or not edges:
        errors.append("edges must be a non-empty list")
        edges = []

    node_ids: set[str] = set()
    symbol_cache: dict[Path, set[str]] = {}
    for index, raw_node in enumerate(nodes):
        label = f"nodes[{index}]"
        if not isinstance(raw_node, dict):
            errors.append(f"{label} must be an object")
            continue
        node_id = raw_node.get("id")
        if not isinstance(node_id, str) or not node_id:
            errors.append(f"{label}.id must be a non-empty string")
            continue
        label = f"node {node_id!r}"
        if node_id in node_ids:
            errors.append(f"duplicate node id: {node_id}")
        node_ids.add(node_id)
        if raw_node.get("kind") not in node_kinds:
            errors.append(f"{label} has unknown kind {raw_node.get('kind')!r}")
        if not isinstance(raw_node.get("subsystem"), str) or not raw_node.get(
            "subsystem"
        ):
            errors.append(f"{label} requires subsystem")
        if raw_node.get("risk") not in {"low", "medium", "high"}:
            errors.append(f"{label} risk must be low, medium, or high")
        if not isinstance(raw_node.get("summary"), str) or not raw_node.get(
            "summary"
        ):
            errors.append(f"{label} requires summary")

        symbols = raw_node.get("symbols", [])
        flags = raw_node.get("flags", [])
        invariants = raw_node.get("invariants", [])
        tests = raw_node.get("tests", [])
        for field_name, values in (
            ("symbols", symbols),
            ("flags", flags),
            ("invariants", invariants),
            ("tests", tests),
        ):
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value for value in values
            ):
                errors.append(f"{label}.{field_name} must be a string list")

        if raw_node.get("risk") == "high":
            if not invariants:
                errors.append(f"{label} is high-risk but has no invariants")
            if not tests:
                errors.append(f"{label} is high-risk but has no tests")
        if raw_node.get("stateful"):
            terminal_paths = raw_node.get("terminal_paths")
            if not isinstance(terminal_paths, dict):
                errors.append(f"{label} is stateful but has no terminal_paths")
            else:
                for terminal in required_terminal_paths:
                    value = terminal_paths.get(terminal)
                    if not isinstance(value, str) or not value:
                        errors.append(
                            f"{label}.terminal_paths requires {terminal!r}"
                        )

        if validate_symbols and isinstance(symbols, list):
            for reference in symbols:
                problem = _validate_code_reference(
                    reference, repo_root=repo_root, symbol_cache=symbol_cache
                )
                if problem:
                    errors.append(f"{label}: {problem}")
        if isinstance(tests, list):
            for test_path in tests:
                if not isinstance(test_path, str):
                    continue
                if not _is_safe_relative_path(test_path):
                    errors.append(
                        f"{label} test path must be repository-relative: "
                        f"{test_path!r}"
                    )
                elif not (repo_root / test_path).is_file():
                    errors.append(f"{label} test path does not exist: {test_path}")

    seen_edges: set[tuple[str, str, str]] = set()
    connected: set[str] = set()
    for index, raw_edge in enumerate(edges):
        label = f"edges[{index}]"
        if not isinstance(raw_edge, dict):
            errors.append(f"{label} must be an object")
            continue
        source = raw_edge.get("from")
        target = raw_edge.get("to")
        edge_type = raw_edge.get("type")
        contract = raw_edge.get("contract")
        if source not in node_ids:
            errors.append(f"{label} has unknown source {source!r}")
        if target not in node_ids:
            errors.append(f"{label} has unknown target {target!r}")
        if edge_type not in edge_types:
            errors.append(f"{label} has unknown type {edge_type!r}")
        if not isinstance(contract, str) or not contract:
            errors.append(f"{label} requires a non-empty contract")
        if isinstance(source, str) and isinstance(target, str):
            connected.update((source, target))
        if all(isinstance(value, str) for value in (source, target, edge_type)):
            key = (source, target, edge_type)
            if key in seen_edges:
                errors.append(f"duplicate semantic edge: {key}")
            seen_edges.add(key)

    for node_id in sorted(node_ids - connected):
        warnings.append(f"isolated dependency node: {node_id}")
    return errors, warnings


def nodes_by_id(graph: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(node["id"]): node
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and "id" in node
    }


def _normalize_path(value: str) -> str:
    value = value.strip()
    while value.startswith("./"):
        value = value[2:]
    return Path(value).as_posix()


def build_impact_slice(
    graph: Mapping[str, Any],
    *,
    node_seeds: Sequence[str] = (),
    symbol_seeds: Sequence[str] = (),
    file_seeds: Sequence[str] = (),
    flag_seeds: Sequence[str] = (),
    depth: int = 2,
    direction: str = "both",
) -> ImpactSlice:
    """Expand explicit patch seeds through typed semantic edges."""

    if depth < 0:
        raise ValueError("impact depth must be non-negative")
    if direction not in {"both", "in", "out"}:
        raise ValueError("impact direction must be one of: both, in, out")
    node_map = nodes_by_id(graph)
    seeds: set[str] = set()

    unresolved_nodes = tuple(sorted(set(node_seeds) - node_map.keys()))
    seeds.update(set(node_seeds) & node_map.keys())

    unresolved_symbols: list[str] = []
    for selector in symbol_seeds:
        matches = {
            node_id
            for node_id, node in node_map.items()
            if any(
                reference == selector or reference.endswith(f"::{selector}")
                for reference in node.get("symbols", [])
            )
        }
        if matches:
            seeds.update(matches)
        else:
            unresolved_symbols.append(selector)

    unresolved_files: list[str] = []
    for selector in file_seeds:
        normalized = _normalize_path(selector)
        matches = {
            node_id
            for node_id, node in node_map.items()
            if any(
                reference.partition("::")[0] == normalized
                or reference.partition("::")[0].startswith(f"{normalized}/")
                for reference in node.get("symbols", [])
            )
        }
        if matches:
            seeds.update(matches)
        else:
            unresolved_files.append(selector)

    unresolved_flags: list[str] = []
    for selector in flag_seeds:
        matches = {
            node_id
            for node_id, node in node_map.items()
            if selector in node.get("flags", [])
        }
        if matches:
            seeds.update(matches)
        else:
            unresolved_flags.append(selector)

    outgoing: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    incoming: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    all_edges = [edge for edge in graph.get("edges", []) if isinstance(edge, dict)]
    for edge in all_edges:
        source = str(edge.get("from", ""))
        target = str(edge.get("to", ""))
        outgoing.setdefault(source, []).append((target, edge))
        incoming.setdefault(target, []).append((source, edge))

    distances: dict[str, int] = {node_id: 0 for node_id in sorted(seeds)}
    queue = deque(sorted(seeds))
    while queue:
        current = queue.popleft()
        current_depth = distances[current]
        if current_depth >= depth:
            continue
        neighbors: list[tuple[str, Mapping[str, Any]]] = []
        if direction in {"both", "out"}:
            neighbors.extend(outgoing.get(current, ()))
        if direction in {"both", "in"}:
            neighbors.extend(incoming.get(current, ()))
        for neighbor, _edge in sorted(neighbors, key=lambda item: item[0]):
            if neighbor not in node_map or neighbor in distances:
                continue
            distances[neighbor] = current_depth + 1
            queue.append(neighbor)

    impacted = set(distances)
    selected_edges = tuple(
        sorted(
            (
                edge
                for edge in all_edges
                if edge.get("from") in impacted and edge.get("to") in impacted
            ),
            key=lambda edge: (
                str(edge.get("from")),
                str(edge.get("to")),
                str(edge.get("type")),
            ),
        )
    )
    return ImpactSlice(
        seeds=tuple(sorted(seeds)),
        distances=dict(sorted(distances.items(), key=lambda item: (item[1], item[0]))),
        edges=selected_edges,
        unresolved_nodes=unresolved_nodes,
        unresolved_symbols=tuple(sorted(unresolved_symbols)),
        unresolved_files=tuple(sorted(unresolved_files)),
        unresolved_flags=tuple(sorted(unresolved_flags)),
    )


def impact_slice_markdown(
    graph: Mapping[str, Any], impact: ImpactSlice
) -> str:
    node_map = nodes_by_id(graph)
    lines = ["# Dependency Impact Slice", "", "## Seeds", ""]
    if impact.seeds:
        lines.extend(f"- `{seed}`" for seed in impact.seeds)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Impacted nodes",
            "",
            "| Distance | Node | Kind | Subsystem | Risk | Summary |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for node_id, distance in impact.distances.items():
        node = node_map[node_id]
        summary = str(node.get("summary", "")).replace("|", "\\|")
        lines.append(
            f"| {distance} | `{node_id}` | {node.get('kind', '')} | "
            f"{node.get('subsystem', '')} | {node.get('risk', '')} | {summary} |"
        )

    lines.extend(["", "## Invariants and terminal paths", ""])
    for node_id in impact.distances:
        node = node_map[node_id]
        lines.append(f"### `{node_id}`")
        lines.append("")
        for invariant in node.get("invariants", []):
            lines.append(f"- Invariant: {invariant}")
        for terminal, contract in node.get("terminal_paths", {}).items():
            lines.append(f"- `{terminal}`: {contract}")
        lines.append("")

    lines.extend(["## Dependency edges", ""])
    for edge in impact.edges:
        lines.append(
            f"- `{edge['from']}` --**{edge['type']}**--> `{edge['to']}`: "
            f"{edge['contract']}"
        )

    symbols = sorted(
        {
            symbol
            for node_id in impact.distances
            for symbol in node_map[node_id].get("symbols", [])
        }
    )
    tests = sorted(
        {
            test
            for node_id in impact.distances
            for test in node_map[node_id].get("tests", [])
        }
    )
    flags = sorted(
        {
            flag
            for node_id in impact.distances
            for flag in node_map[node_id].get("flags", [])
        }
    )
    lines.extend(["", "## Code symbols", ""])
    lines.extend(f"- `{symbol}`" for symbol in symbols)
    lines.extend(["", "## Flags", ""])
    lines.extend(f"- `{flag}`" for flag in flags) if flags else lines.append("- None")
    lines.extend(["", "## Existing validation targets", ""])
    lines.extend(f"- `{test}`" for test in tests)

    unresolved = {
        "node": impact.unresolved_nodes,
        "symbol": impact.unresolved_symbols,
        "file": impact.unresolved_files,
        "flag": impact.unresolved_flags,
    }
    if any(unresolved.values()):
        lines.extend(["", "## Unresolved selectors", ""])
        for selector_type, values in unresolved.items():
            for value in values:
                lines.append(f"- {selector_type}: `{value}`")
    return "\n".join(lines).rstrip() + "\n"


def _mermaid_id(node_id: str) -> str:
    return "n_" + "".join(
        character if character.isalnum() else "_" for character in node_id
    )


def render_graph_markdown(graph: Mapping[str, Any]) -> str:
    """Render a deterministic Mermaid overview plus review tables."""

    node_map = nodes_by_id(graph)
    subsystems: dict[str, list[str]] = {}
    for node_id, node in node_map.items():
        subsystems.setdefault(str(node.get("subsystem", "other")), []).append(node_id)

    lines = [
        f"# {graph.get('title', 'System dependency graph')}",
        "",
        "> Generated from `docs/dependency_graph/system_dependencies.json`.",
        "> Do not edit this file directly.",
        "",
        "```mermaid",
        "flowchart LR",
    ]
    for subsystem in sorted(subsystems):
        subgraph_id = "sg_" + "".join(
            character if character.isalnum() else "_" for character in subsystem
        )
        lines.append(f"  subgraph {subgraph_id}[\"{subsystem}\"]")
        for node_id in sorted(subsystems[subsystem]):
            node = node_map[node_id]
            label = f"{node_id}\\n{node.get('kind', '')} / {node.get('risk', '')}"
            lines.append(f"    {_mermaid_id(node_id)}[\"{label}\"]")
        lines.append("  end")
    for edge in sorted(
        graph.get("edges", []),
        key=lambda item: (item["from"], item["to"], item["type"]),
    ):
        lines.append(
            f"  {_mermaid_id(edge['from'])} -->|{edge['type']}| "
            f"{_mermaid_id(edge['to'])}"
        )
    high_nodes = [
        _mermaid_id(node_id)
        for node_id, node in sorted(node_map.items())
        if node.get("risk") == "high"
    ]
    lines.append("  classDef highRisk fill:#ffe0e0,stroke:#b30000,stroke-width:2px")
    if high_nodes:
        lines.append(f"  class {','.join(high_nodes)} highRisk")
    lines.extend(["```", "", "## Nodes", ""])

    for node_id, node in sorted(node_map.items()):
        lines.extend(
            [
                f"### `{node_id}`",
                "",
                f"- Kind: `{node.get('kind')}`",
                f"- Subsystem: `{node.get('subsystem')}`",
                f"- Risk: `{node.get('risk')}`",
                f"- Summary: {node.get('summary')}",
            ]
        )
        for invariant in node.get("invariants", []):
            lines.append(f"- Invariant: {invariant}")
        for symbol in node.get("symbols", []):
            lines.append(f"- Symbol: `{symbol}`")
        for test in node.get("tests", []):
            lines.append(f"- Test: `{test}`")
        lines.append("")

    lines.extend(["## Edge contracts", ""])
    for edge in sorted(
        graph.get("edges", []),
        key=lambda item: (item["from"], item["to"], item["type"]),
    ):
        lines.append(
            f"- `{edge['from']}` --**{edge['type']}**--> `{edge['to']}`: "
            f"{edge['contract']}"
        )
    return "\n".join(lines).rstrip() + "\n"


def format_validation_result(errors: Iterable[str], warnings: Iterable[str]) -> str:
    lines: list[str] = []
    for warning in warnings:
        lines.append(f"[WARN] {warning}")
    for error in errors:
        lines.append(f"[ERROR] {error}")
    if not lines:
        lines.append("[OK] dependency graph is valid")
    return "\n".join(lines)
