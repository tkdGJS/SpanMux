#!/usr/bin/env python3
"""Import-free smoke for superseded provider-version reconciliation."""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vendor-dir",
        default=os.environ.get("VLLM_VENDOR_DIR"),
    )
    return parser.parse_args()


def load_coverage_helper(provider_path: Path):
    tree = ast.parse(provider_path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_provider_snapshot_covers_pinned_version"
    )
    namespace: dict[str, object] = {}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(provider_path),
            "exec",
        ),
        namespace,
    )
    return namespace["_provider_snapshot_covers_pinned_version"]


def assert_required_runtime_imports(provider_path: Path) -> None:
    """Catch undefined globals that bytecode compilation cannot detect."""

    tree = ast.parse(provider_path.read_text())
    collections_imports = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "collections"
        for alias in node.names
    }
    assert "defaultdict" in collections_imports, (
        "tiered provider uses defaultdict during registration reconciliation "
        "but does not import it from collections"
    )


def main() -> int:
    args = parse_args()
    vendor_dir = (
        Path(args.vendor_dir).resolve()
        if args.vendor_dir
        else Path(__file__).resolve().parents[2] / "vendor_v8"
    )
    provider_path = (
        vendor_dir / "vllm/v1/attention/ops/tiered_residency_provider.py"
    )
    virtual_context_path = vendor_dir / "vllm/v1/core/virtual_context.py"
    provider_source = provider_path.read_text()
    virtual_context_source = virtual_context_path.read_text()

    assert_required_runtime_imports(provider_path)
    covers = load_coverage_helper(provider_path)
    assert covers(
        available_version=9,
        available_token_count=16,
        pinned_version=7,
        expected_token_count=16,
    )
    assert covers(
        available_version=7,
        available_token_count=16,
        pinned_version=7,
        expected_token_count=16,
    )
    assert not covers(
        available_version=6,
        available_token_count=16,
        pinned_version=7,
        expected_token_count=16,
    )
    assert not covers(
        available_version=9,
        available_token_count=8,
        pinned_version=7,
        expected_token_count=16,
    )

    required_provider_contracts = (
        "allow_newer_version=True",
        "reloaded_provider_versions_by_request",
        "pages=pages",
        "else max(0, len(pages) - verified)",
        "raw_failed\n            if verified is None",
        "dram_provider_reload_to_hbm_newer_version_fallback_pages",
        "dram_provider_register_from_hbm_reconcile_newer_version_pages",
    )
    for contract in required_provider_contracts:
        assert contract in provider_source, contract
    required_sidecar_contracts = (
        "provider_versions: dict[int, int] | None = None",
        "old.dram_copy_version = provider_version",
        "completed_provider_versions",
    )
    for contract in required_sidecar_contracts:
        assert contract in virtual_context_source, contract

    print("[provider_version_reconciliation_smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
