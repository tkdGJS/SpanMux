from pathlib import Path

import pytest

from prepare_ctx100k_dataset_examples import PromptSource, select_sharegpt_sources


class _WhitespaceTokenizer:

    def encode(self, text: str, add_special_tokens: bool = True) -> list[str]:
        del add_special_tokens
        return text.split()


def _source(source_id: str, token: str, count: int) -> PromptSource:
    return PromptSource(
        workload="sharegpt",
        prefix="",
        body=" ".join([token] * count),
        suffix=" Assistant:",
        source_ids=(f"{source_id}:0",),
        source_path=Path("sharegpt.json"),
        role_default="foreground",
        stream_default=True,
        construction="single_conversation_prefix_truncated",
        source_faithful=True,
    )


def test_selects_three_disjoint_sources_with_explicit_packing() -> None:
    candidates = [
        _source("long", "long", 110),
        _source("short-a", "a", 45),
        _source("short-b", "b", 45),
        _source("short-c", "c", 45),
        _source("short-d", "d", 45),
        _source("duplicate-body", "d", 45),
    ]

    selected = select_sharegpt_sources(
        _WhitespaceTokenizer(),
        candidates,
        target_tokens=100,
        request_count=3,
        underlength_policy="pack",
    )

    assert len(selected) == 3
    assert selected[0].source_faithful is True
    assert [source.source_faithful for source in selected[1:]] == [False, False]
    source_sets = [set(source.source_ids) for source in selected]
    assert all(
        source_sets[left].isdisjoint(source_sets[right])
        for left in range(len(source_sets))
        for right in range(left + 1, len(source_sets))
    )
    assert "duplicate-body:0" not in set().union(*source_sets)


def test_rejects_implicit_packing() -> None:
    candidates = [
        _source("long", "long", 110),
        _source("short-a", "a", 45),
        _source("short-b", "b", 45),
    ]

    with pytest.raises(ValueError, match="Pass --sharegpt-underlength-policy=pack"):
        select_sharegpt_sources(
            _WhitespaceTokenizer(),
            candidates,
            target_tokens=100,
            request_count=2,
            underlength_policy="error",
        )
