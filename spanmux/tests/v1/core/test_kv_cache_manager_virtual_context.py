# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from vllm.v1.core.kv_cache_manager import KVCacheManager


def test_reload_required_count_uses_index_without_materializing_pages() -> None:
    manager = object.__new__(KVCacheManager)
    virtual_context = Mock()
    virtual_context.hbm_reload_required_page_count.return_value = 106_250
    manager.virtual_context = virtual_context

    request_ids = {"long-2", "long-3"}
    assert (
        manager.get_virtual_context_hbm_reload_required_page_count(request_ids)
        == 106_250
    )
    virtual_context.hbm_reload_required_page_count.assert_called_once_with(
        request_ids
    )
    virtual_context.plan_hbm_reload_pages.assert_not_called()


def test_reload_required_count_without_virtual_context_is_zero() -> None:
    manager = object.__new__(KVCacheManager)
    manager.virtual_context = None

    assert manager.get_virtual_context_hbm_reload_required_page_count({"req"}) == 0
