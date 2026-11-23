#!/usr/bin/env python3
"""CPU smoke checks for virtual-context release/reload correctness."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vendor-dir",
        default=os.environ.get("VLLM_VENDOR_DIR"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vendor_dir = (
        Path(args.vendor_dir).resolve()
        if args.vendor_dir
        else Path(__file__).resolve().parents[2] / "vendor_v4"
    )
    module_path = vendor_dir / "vllm/v1/core/virtual_context.py"
    spec = importlib.util.spec_from_file_location(
        "virtual_context_lifecycle_smoke_module",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    VirtualContextTable = module.VirtualContextTable
    count_full_virtual_external_blocks = (
        module.count_full_virtual_external_blocks
    )
    resolve_coverage_block_size = (
        module.resolve_virtual_context_coverage_block_size
    )

    # The first pressure transition has no DRAM copy yet. Empty page-size
    # input must retain the native block size instead of calling max(int).
    assert resolve_coverage_block_size(16, []) == 16
    assert resolve_coverage_block_size(16, [8, 16]) == 16
    assert resolve_coverage_block_size(16, iter([32])) == 32
    try:
        resolve_coverage_block_size(0, [])
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive block size must be rejected")

    # Only pages fully covered by virtual external tokens may use null block
    # placeholders. A partial tail needs a concrete HBM append block.
    assert count_full_virtual_external_blocks(10, 10, 16) == 0
    assert count_full_virtual_external_blocks(32, 32, 16) == 2
    assert count_full_virtual_external_blocks(25, 20, 16) == 0
    assert count_full_virtual_external_blocks(45, 40, 16) == 1

    request_id = "partial-tail-request"
    table = VirtualContextTable()
    table.observe_hbm_blocks(
        request_id,
        ([7],),
        block_size=16,
        logical_tokens=10,
    )
    for layer_id in (0, 1):
        table.observe_provider_layer(request_id, layer_id)

    for layer_id in (0, 1):
        table.record_dram_copies_ready(
            request_id,
            {0: 100 + layer_id},
            block_size=16,
            layer_id=layer_id,
            page_to_block={0: 7},
            page_token_counts={0: 10},
        )
        if layer_id == 0:
            assert not table.plan_hbm_release_candidates(
                request_ids={request_id},
                allow_partial_pages=True,
            ), "all expected layer copies are required before HBM release"

    assert not table.plan_hbm_release_candidates(
        request_ids={request_id}
    ), "partial tail must require explicit partial release permission"
    partial_candidates = table.plan_hbm_release_candidates(
        request_ids={request_id},
        allow_partial_pages=True,
    )
    assert len(partial_candidates) == 2, partial_candidates
    released = table.mark_hbm_released_to_dram(
        request_ids={request_id},
        allow_partial_pages=True,
    )
    assert len(released) == 2, released
    assert (
        table.get_debug_stats()[
            "virtual_context_hbm_partial_release_marked_pages_total"
        ]
        == 1
    )
    assert len(table.plan_hbm_reload_pages(request_ids={request_id})) == 3
    assert table.has_hbm_reload_required_pages({request_id})

    table.mark_hbm_reload_complete(
        request_id,
        {0: 9},
        block_size=16,
        layer_id=0,
        page_token_counts={0: 10},
        provider_versions={0: 200},
    )
    assert len(table.plan_hbm_reload_pages(request_ids={request_id})) == 2
    table.mark_hbm_reload_complete(
        request_id,
        {0: 9},
        block_size=16,
        layer_id=1,
        page_token_counts={0: 10},
        provider_versions={0: 201},
    )
    assert not table.plan_hbm_reload_pages(request_ids={request_id})
    assert not table.has_hbm_reload_required_pages({request_id})

    # A native H2D reload may consume a provider version newer than the one
    # pinned when HBM was released. A later pressure release must reuse the
    # provider version that was actually copied, not the evicted old version.
    rereleased = table.mark_hbm_released_to_dram(
        request_ids={request_id},
        allow_partial_pages=True,
    )
    assert len(rereleased) == 3, rereleased
    rerelease_versions = {
        (page.layer_id, page.provider_version)
        for page in table.plan_hbm_reload_pages(request_ids={request_id})
        if page.layer_id >= 0
    }
    assert rerelease_versions == {(0, 200), (1, 201)}, rerelease_versions
    aggregate_versions = {
        page.provider_version
        for page in table.plan_hbm_reload_pages(request_ids={request_id})
        if page.layer_id == -1
    }
    assert aggregate_versions == {201}, aggregate_versions

    # A copy of an 8-token tail cannot release a page after the logical tail
    # grows to 12 tokens. Both layer copies must be refreshed first.
    growing_id = "growing-tail-request"
    growing = VirtualContextTable()
    growing.observe_hbm_blocks(
        growing_id,
        ([11],),
        block_size=16,
        logical_tokens=8,
    )
    for layer_id in (0, 1):
        growing.observe_provider_layer(growing_id, layer_id)
        growing.record_dram_copies_ready(
            growing_id,
            {0: 200 + layer_id},
            block_size=16,
            layer_id=layer_id,
            page_to_block={0: 11},
            page_token_counts={0: 8},
        )
    growing.observe_hbm_block_delta(
        growing_id,
        ([],),
        (1,),
        block_size=16,
        logical_tokens=12,
    )
    # Even a numerically newer async completion is stale if it covers fewer
    # tokens than the aggregate logical tail observed by the scheduler.
    for layer_id in (0, 1):
        growing.record_dram_copies_ready(
            growing_id,
            {0: 300 + layer_id},
            block_size=16,
            layer_id=layer_id,
            page_to_block={0: 11},
            page_token_counts={0: 8},
        )
    assert not growing.plan_hbm_release_candidates(
        request_ids={growing_id},
        allow_partial_pages=True,
    )
    for layer_id in (0, 1):
        growing.record_dram_copies_ready(
            growing_id,
            {0: 400 + layer_id},
            block_size=16,
            layer_id=layer_id,
            page_to_block={0: 11},
            page_token_counts={0: 12},
        )
    assert len(
        growing.plan_hbm_release_candidates(
            request_ids={growing_id},
            allow_partial_pages=True,
        )
    ) == 2
    assert (
        growing.get_debug_stats()[
            "virtual_context_dram_copy_short_rejected_pages_total"
        ]
        == 2
    )

    # Reloading an 11-token provider tail and then appending five Prefill
    # tokens creates a 16-token HBM page backed by only 11 DRAM tokens. Keep
    # those facts separate: the page cannot be victimized again until every
    # layer publishes a fresh 16-token D2H snapshot.
    resumed_id = "resumed-growing-tail"
    resumed = VirtualContextTable()
    resumed.observe_hbm_blocks(
        resumed_id,
        ([21],),
        block_size=16,
        logical_tokens=11,
    )
    for layer_id in (0, 1):
        resumed.observe_provider_layer(resumed_id, layer_id)
        resumed.record_dram_copies_ready(
            resumed_id,
            {0: 500 + layer_id},
            block_size=16,
            layer_id=layer_id,
            page_to_block={0: 21},
            page_token_counts={0: 11},
        )
    assert len(
        resumed.mark_hbm_released_to_dram(
            request_ids={resumed_id},
            allow_partial_pages=True,
        )
    ) == 2
    for layer_id in (0, 1):
        resumed.mark_hbm_reload_complete(
            resumed_id,
            {0: 22},
            block_size=16,
            layer_id=layer_id,
            page_token_counts={0: 16},
            provider_token_counts={0: 11},
            provider_versions={0: 600 + layer_id},
        )
    assert not resumed.plan_hbm_reload_pages(request_ids={resumed_id})
    assert not resumed.plan_hbm_release_candidates(
        request_ids={resumed_id},
        allow_partial_pages=True,
    ), "an 11-token provider copy cannot release a 16-token HBM page"
    for layer_id in (0, 1):
        resumed.record_dram_copies_ready(
            resumed_id,
            {0: 700 + layer_id},
            block_size=16,
            layer_id=layer_id,
            page_to_block={0: 22},
            page_token_counts={0: 16},
        )
    assert len(
        resumed.plan_hbm_release_candidates(
            request_ids={resumed_id},
            allow_partial_pages=True,
        )
    ) == 2

    # scheduled_prefill_fit limits by distinct physical logical pages, not by
    # per-layer provider records. Releasing one page must transition every
    # layer for page 0 while pages 1 and 2 remain HBM-authoritative.
    targeted_id = "targeted-partial-victim"
    targeted = VirtualContextTable()
    targeted.observe_hbm_blocks(
        targeted_id,
        ([31, 32, 33],),
        block_size=16,
        logical_tokens=48,
    )
    for layer_id in (0, 1):
        targeted.observe_provider_layer(targeted_id, layer_id)
        targeted.record_dram_copies_ready(
            targeted_id,
            {0: 500 + layer_id, 1: 510 + layer_id, 2: 520 + layer_id},
            block_size=16,
            layer_id=layer_id,
            page_to_block={0: 31, 1: 32, 2: 33},
        )
    targeted_released = targeted.mark_hbm_released_to_dram(
        request_ids={targeted_id},
        allow_partial_pages=True,
        max_logical_pages=1,
    )
    assert {
        candidate.logical_token_page for candidate in targeted_released
    } == {0}, targeted_released
    assert {
        page.logical_token_page
        for page in targeted.plan_hbm_reload_pages(
            request_ids={targeted_id}
        )
    } == {0}
    assert {
        candidate.logical_token_page
        for candidate in targeted.plan_hbm_release_candidates(
            request_ids={targeted_id},
            allow_partial_pages=True,
        )
    } == {1, 2}

    # scheduled_prefill_fit's partially released page rejoins the same native
    # reload completion path as a whole-request victim. A provider snapshot
    # may supersede the version pinned at partial release; both layer records
    # and the aggregate must retain the versions actually copied to HBM.
    targeted.mark_hbm_reload_complete(
        targeted_id,
        {0: 31},
        block_size=16,
        layer_id=0,
        provider_versions={0: 600},
    )
    assert {
        page.logical_token_page
        for page in targeted.plan_hbm_reload_pages(request_ids={targeted_id})
    } == {0}
    targeted.mark_hbm_reload_complete(
        targeted_id,
        {0: 31},
        block_size=16,
        layer_id=1,
        provider_versions={0: 601},
    )
    assert not targeted.plan_hbm_reload_pages(request_ids={targeted_id})
    targeted_rereleased = targeted.mark_hbm_released_to_dram(
        request_ids={targeted_id},
        allow_partial_pages=True,
        max_logical_pages=1,
    )
    assert {
        candidate.logical_token_page for candidate in targeted_rereleased
    } == {0}, targeted_rereleased
    assert {
        (candidate.layer_id, candidate.dram_copy_version)
        for candidate in targeted_rereleased
    } == {(-1, 601), (0, 600), (1, 601)}, targeted_rereleased

    # A monotonic allocation appends only the new page and updates the old/new
    # tail lengths. It must not require a full request block-table traversal.
    delta_id = "delta-page-request"
    delta = VirtualContextTable()
    delta.observe_hbm_blocks(
        delta_id,
        ([21],),
        block_size=16,
        logical_tokens=16,
    )
    delta.observe_hbm_block_delta(
        delta_id,
        ([22],),
        (1,),
        block_size=16,
        logical_tokens=20,
    )
    delta_stats = delta.get_request_debug_stats(delta_id)
    assert delta_stats["virtual_context_request_pages"] == 2, delta_stats
    global_stats = delta.get_debug_stats()
    assert global_stats["virtual_context_observe_hbm_delta_calls"] == 1
    assert global_stats["virtual_context_observe_hbm_delta_pages_total"] == 2

    print("[virtual_context_lifecycle_smoke] OK")
    print(f"vendor_dir={vendor_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
