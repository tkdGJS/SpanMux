from pathlib import Path

import pytest

from prepare_ctx100k_dataset_examples import PromptSource, fit_exact_tokens
from prepare_mixed_context_sharegpt import select_distinct_sources


class _WhitespaceTokenizer:

    def encode(self, text: str, add_special_tokens: bool = True) -> list[str]:
        del add_special_tokens
        return text.split()


class _FramingGapTokenizer:
    """Toy tokenizer whose body/suffix boundary skips odd token counts."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[str]:
        if not add_special_tokens:
            return list(text)
        body_units = sum(character in "abcd" for character in text)
        newline_units = text.count("\n")
        return ["token"] * (body_units * 2 + newline_units)

    def decode(
        self,
        token_ids: list[str],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(token_ids)


def _source(source_id: str, token_count: int) -> PromptSource:
    return PromptSource(
        workload="sharegpt",
        prefix="",
        body=" ".join([source_id] * token_count),
        suffix=" Assistant:",
        source_ids=(f"{source_id}:0",),
        source_path=Path("sharegpt.json"),
        role_default="foreground",
        stream_default=True,
        construction="single_conversation_prefix_truncated",
        source_faithful=True,
    )


def test_selects_closest_distinct_source_in_requested_order() -> None:
    selected = select_distinct_sources(
        _WhitespaceTokenizer(),
        [
            _source("a", 30),
            _source("b", 55),
            _source("c", 80),
            _source("d", 110),
            _source("e", 135),
        ],
        [25, 50, 75, 100, 125],
    )

    assert [target for target, _source_row, _full in selected] == [
        25,
        50,
        75,
        100,
        125,
    ]
    assert [source.source_ids[0] for _target, source, _full in selected] == [
        "a:0",
        "b:0",
        "c:0",
        "d:0",
        "e:0",
    ]


def test_rejects_when_distinct_conversations_are_insufficient() -> None:
    with pytest.raises(ValueError, match="no unused ShareGPT conversation"):
        select_distinct_sources(
            _WhitespaceTokenizer(),
            [_source("only", 110)],
            [50, 100],
        )


def test_pack_policy_uses_disjoint_sources_and_labels_synthetic_row() -> None:
    selected = select_distinct_sources(
        _WhitespaceTokenizer(),
        [_source("a", 40), _source("b", 40), _source("c", 40)],
        [100],
        underlength_policy="pack",
    )

    target, source, full_tokens = selected[0]
    assert target == 100
    assert full_tokens >= target
    assert source.source_ids == ("a:0", "b:0", "c:0")
    assert source.construction == "packed_multi_conversation_stress_example"
    assert source.source_faithful is False


def test_exact_fit_uses_whitespace_framing_when_bpe_count_skips_target() -> None:
    prompt = fit_exact_tokens(
        _FramingGapTokenizer(),
        prefix="",
        body="abcd",
        suffix="<suffix>",
        target_tokens=5,
    )

    assert prompt == "ab\n<suffix>"
