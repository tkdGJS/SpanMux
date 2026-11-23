#!/usr/bin/env python3
"""Prepare a reproducible LoongServe Mixed open-loop workload trace.

The paper workload contains equal request counts from ShareGPT, L-Eval, and
LV-Eval.  This script follows the public LoongServe artifact's preprocessing
shape, samples each dataset deterministically, and writes a stable JSONL trace
that can be replayed unchanged by multiple serving systems.

By default, instance feasibility caps filter whole requests. The opt-in
``paper_sample_then_truncate`` policy instead keeps the uncapped sample/order
and truncates only oversized inputs, retaining the prompt head and tail. Both
capped policies are explicitly identified as derivatives in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar


PAPER_DATASETS = ("sharegpt", "leval", "lveval")
LOONGSERVE_ARTIFACT_COMMIT = "f6e8fc15f31ea6913b353b627545058db5eec321"
DEFAULT_LVEVAL_SUBSETS = (
    "factrecall_en",
    "hotpotwikiqa_mixup",
    "loogle_CR_mixup",
    "loogle_MIR_mixup",
    "loogle_SD_mixup",
    "multifieldqa_en_mixup",
)
T = TypeVar("T")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[int], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


class TokenCounter:
    """Count tokens deterministically with bounded native batch parallelism.

    Candidate order remains owned by the caller. Hugging Face fast-tokenizer
    batches use the Rust/Rayon pool, while results are consumed in input order.
    A content-addressed cache avoids repeating the exact same tokenization work
    during the preparer's count and collection passes.
    """

    def __init__(
        self,
        *,
        mode: str,
        tokenizer_name: str,
        local_files_only: bool,
        workers: int,
        batch_size: int,
    ) -> None:
        self.mode = mode
        self.tokenizer_name = tokenizer_name
        self.local_files_only = local_files_only
        self.workers = workers
        self.batch_size = batch_size
        self.resolved_tokenizer = tokenizer_name
        self.is_fast_tokenizer = False
        self._tokenizer_source = tokenizer_name
        self._tokenizer: Any = None
        self._count_cache: dict[tuple[str, int, bytes], int] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._piece_batch_verified = False
        if mode == "hf":
            # These must be set before the tokenizers Rayon pool is first used.
            # This script is a standalone process, so TOKENIZER_WORKERS is the
            # unambiguous owner of its offline tokenization CPU budget.
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            os.environ["RAYON_NUM_THREADS"] = str(workers)
            self._tokenizer_source = self._resolve_tokenizer_source()
            self._tokenizer = self._load_tokenizer()
            self.resolved_tokenizer = str(
                getattr(self._tokenizer, "name_or_path", tokenizer_name)
            )
            self.is_fast_tokenizer = bool(getattr(self._tokenizer, "is_fast", False))

    def _resolve_tokenizer_source(self) -> str:
        source_path = Path(self.tokenizer_name).expanduser()
        if source_path.exists() or not self.local_files_only:
            return str(source_path) if source_path.exists() else self.tokenizer_name
        # Some transformers releases perform a compatibility model-info call
        # even with local_files_only=True when given a repository ID. Resolve
        # the cached snapshot once, then give every worker a concrete path.
        from huggingface_hub import snapshot_download

        return snapshot_download(
            self.tokenizer_name,
            local_files_only=True,
        )

    def _load_tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self._tokenizer_source,
            trust_remote_code=False,
            local_files_only=self.local_files_only,
        )
        tokenizer.model_max_length = 1_000_000_000
        return tokenizer

    @staticmethod
    def _cache_key(text: str, operation: str) -> tuple[str, int, bytes]:
        payload = text.encode("utf-8")
        return operation, len(payload), hashlib.sha256(payload).digest()

    def _cached_count_many(
        self,
        texts: Sequence[str],
        *,
        operation: str,
        compute: Callable[[Sequence[str]], list[int]],
    ) -> list[int]:
        results: list[int | None] = [None] * len(texts)
        missing: dict[tuple[str, int, bytes], tuple[str, list[int]]] = {}
        for index, text in enumerate(texts):
            key = self._cache_key(text, operation)
            cached = self._count_cache.get(key)
            if cached is not None:
                self._cache_hits += 1
                results[index] = cached
                continue
            if key in missing:
                self._cache_hits += 1
                missing[key][1].append(index)
                continue
            self._cache_misses += 1
            missing[key] = (text, [index])

        if missing:
            keys = list(missing)
            values = compute([missing[key][0] for key in keys])
            if len(values) != len(keys):
                raise RuntimeError("tokenizer batch result length mismatch")
            for key, value in zip(keys, values):
                existing = self._count_cache.setdefault(key, value)
                if existing != value:
                    raise RuntimeError("token count cache invariant violated")
                for index in missing[key][1]:
                    results[index] = existing

        if any(value is None for value in results):
            raise RuntimeError("token count result was not populated")
        return [int(value) for value in results]

    def _encode_many(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        if self.mode == "approx":
            return [max(1, math.ceil(len(text.split()) * 1.35)) for text in texts]
        encoded = self._tokenizer(
            list(texts),
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return [len(token_ids) for token_ids in encoded["input_ids"]]

    def count_many(
        self,
        texts: Sequence[str],
        *,
        add_special_tokens: bool = True,
    ) -> list[int]:
        operation = f"encode:special={int(add_special_tokens)}"
        return self._cached_count_many(
            texts,
            operation=operation,
            compute=lambda missing: self._encode_many(
                missing,
                add_special_tokens=add_special_tokens,
            ),
        )

    def count(self, text: str, *, add_special_tokens: bool = True) -> int:
        return self.count_many([text], add_special_tokens=add_special_tokens)[0]

    def count_pieces_many(self, texts: Sequence[str]) -> list[int]:
        def compute(missing: Sequence[str]) -> list[int]:
            values = self._encode_many(
                missing,
                add_special_tokens=False,
            )
            if self.mode == "hf" and not self._piece_batch_verified:
                for text, value in zip(missing[:3], values[:3]):
                    if len(self._tokenizer.tokenize(text)) != value:
                        raise RuntimeError(
                            "batched no-special encoding does not match "
                            "tokenizer.tokenize length"
                        )
                self._piece_batch_verified = True
            return values

        return self._cached_count_many(
            texts,
            operation="tokenize:pieces",
            compute=compute,
        )

    def count_pieces(self, text: str) -> int:
        """Match the artifact's ``len(tokenizer.tokenize(text))`` calls."""

        return self.count_pieces_many([text])[0]

    def cache_stats(self) -> dict[str, int]:
        return {
            "entries": len(self._count_cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

    def truncate_prompt_head_tail(self, text: str, max_tokens: int) -> tuple[str, int]:
        """Keep both ends, then verify the *served text* including special tokens.

        Character offsets avoid decoding partial UTF-8 byte tokens into U+FFFD.
        Joining the ends may change BPE segmentation, so retained source-token
        count alone is not authority for the final input length.
        """
        if self.mode != "hf" or not self.is_fast_tokenizer:
            raise ValueError("prompt truncation requires an exact HF fast tokenizer")
        if max_tokens <= 0:
            raise ValueError("prompt truncation requires positive max-input-tokens")
        original_count = self.count(text)
        if original_count <= max_tokens:
            return text, original_count
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        offsets = encoded["offset_mapping"]
        if len(offsets) != len(encoded["input_ids"]) or any(
            not 0 <= start <= end <= len(text) for start, end in offsets
        ):
            raise RuntimeError("invalid tokenizer character offsets for truncation")
        special_count = self.count("")
        keep = min(len(offsets) - 1, max_tokens - special_count)
        while keep > 0:
            head = (keep + 1) // 2
            tail = keep // 2
            left_end = offsets[head - 1][1]
            right_start = offsets[-tail][0] if tail else len(text)
            if left_end < right_start:
                shortened = text[:left_end] + text[right_start:]
                actual_count = self.count(shortened)
                if 0 < actual_count <= max_tokens:
                    return shortened, actual_count
                keep -= max(1, actual_count - max_tokens)
            else:
                # Repeated offsets are legal for byte-level tokenizers. Do not
                # duplicate an overlapping character at the cut boundary.
                keep -= 1
        raise ValueError(
            f"cannot retain nonempty prompt text within {max_tokens} tokens "
            "including normal special tokens"
        )

    def close(self) -> None:
        return None


def iter_batches(values: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for value in values:
        batch.append(value)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass(frozen=True)
class Candidate:
    dataset: str
    source_id: str
    source_path: str
    prompt: str
    reference_output: str
    # Serving length includes the tokenizer's normal prompt special tokens and
    # is the value used for instance-capacity checks and request reporting.
    input_tokens: int
    # The requested generation length follows the artifact preprocessing.
    output_tokens: int
    # Paper lengths retain the artifact's dataset-specific counting rules.
    paper_input_tokens: int
    paper_output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def iter_json_rows(path: Path) -> Iterator[tuple[dict[str, Any], Path, int]]:
    """Yield dictionaries from one JSON/JSONL file or a directory tree."""

    if not path.exists():
        raise FileNotFoundError(path)
    files = (
        [path]
        if path.is_file()
        else sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and candidate.suffix.lower() in {".json", ".jsonl", ".ndjson"}
        )
    )
    if not files:
        raise RuntimeError(f"no JSON/JSONL files found under {path}")
    for source in files:
        if source.suffix.lower() in {".jsonl", ".ndjson"}:
            with source.open("r", encoding="utf-8") as stream:
                for line_index, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        yield payload, source, line_index
            continue
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if isinstance(payload, dict):
            rows: Any = next(
                (
                    payload[key]
                    for key in ("data", "instances", "examples", "records")
                    if isinstance(payload.get(key), list)
                ),
                [payload],
            )
        else:
            rows = payload
        if not isinstance(rows, list):
            continue
        for row_index, row in enumerate(rows):
            if isinstance(row, dict):
                yield row, source, row_index


def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(scalar_text(item) for item in value).strip()
    return str(value)


def list_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def sharegpt_candidates(
    path: Path,
    counter: TokenCounter,
    *,
    seed: int,
) -> Iterator[Candidate]:
    """Mirror the artifact's one-random-continuation-per-dialogue rule."""

    def raw_items() -> Iterator[tuple[str, str, str, str]]:
        rng = random.Random(seed)
        for row, source, row_index in iter_json_rows(path):
            messages = (
                row.get("conversations")
                or row.get("conversation")
                or row.get("messages")
            )
            if not isinstance(messages, list) or len(messages) < 3:
                continue
            values = [
                scalar_text(
                    message.get("value")
                    or message.get("content")
                    or message.get("text")
                ).strip()
                for message in messages
                if isinstance(message, dict)
            ]
            if len(values) < 3 or any(not value for value in values):
                continue
            prompt_turns = rng.randint(1, len(values) - 1)
            yield (
                str(row.get("id") or row_index),
                str(source),
                "\n".join(values[:prompt_turns]),
                values[prompt_turns],
            )

    for batch in iter_batches(raw_items(), counter.batch_size):
        input_lengths = counter.count_many([item[2] for item in batch])
        output_lengths = counter.count_many([item[3] for item in batch])
        for item, input_tokens, output_tokens in zip(
            batch, input_lengths, output_lengths
        ):
            source_id, source_path, prompt, output = item
            # These are the ShareGPT-specific public artifact filters. The
            # common 16/4 filter is applied later to all three datasets.
            if (
                input_tokens < 4
                or output_tokens < 4
                or input_tokens + output_tokens >= 200_000
                or output_tokens >= 1_000
            ):
                continue
            yield Candidate(
                dataset="sharegpt",
                source_id=source_id,
                source_path=source_path,
                prompt=prompt,
                reference_output=output,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                paper_input_tokens=input_tokens,
                paper_output_tokens=output_tokens,
            )


def leval_candidates(
    path: Path,
    counter: TokenCounter,
) -> Iterator[Candidate]:
    def raw_items() -> Iterator[tuple[str, str, str, str, str]]:
        for row, source, row_index in iter_json_rows(path):
            document = scalar_text(row.get("input"))
            instructions = list_values(row.get("instructions", row.get("instruction")))
            outputs = list_values(row.get("outputs", row.get("output")))
            if not document or not instructions or not outputs:
                continue
            for pair_index, (instruction, output_value) in enumerate(
                zip(instructions, outputs)
            ):
                instruction_text = scalar_text(instruction)
                output = scalar_text(output_value)
                if not instruction_text or not output:
                    continue
                yield (
                    f"{row.get('_id') or row.get('id') or row_index}:" f"{pair_index}",
                    str(source),
                    document,
                    instruction_text,
                    output,
                )

    for batch in iter_batches(raw_items(), counter.batch_size):
        documents = [item[2] for item in batch]
        instructions = [item[3] for item in batch]
        outputs = [item[4] for item in batch]
        prompts = [
            document + instruction
            for document, instruction in zip(documents, instructions)
        ]
        paper_document_lengths = counter.count_pieces_many(documents)
        paper_instruction_lengths = counter.count_pieces_many(instructions)
        paper_output_lengths = counter.count_pieces_many(outputs)
        input_lengths = counter.count_many(prompts)
        for index, item in enumerate(batch):
            source_id, source_path, _document, _instruction, output = item
            paper_input_tokens = (
                paper_document_lengths[index] + paper_instruction_lengths[index]
            )
            paper_output_tokens = paper_output_lengths[index]
            yield Candidate(
                dataset="leval",
                source_id=source_id,
                source_path=source_path,
                prompt=prompts[index],
                reference_output=output,
                input_tokens=input_lengths[index],
                output_tokens=paper_output_tokens,
                paper_input_tokens=paper_input_tokens,
                paper_output_tokens=paper_output_tokens,
            )


def select_lveval_files(
    path: Path,
    subsets: Sequence[str],
) -> list[Path]:
    if path.is_file():
        return [path]
    all_files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in {".json", ".jsonl", ".ndjson"}
    )
    selected = [
        candidate
        for candidate in all_files
        if any(subset.lower() in str(candidate).lower() for subset in subsets)
    ]
    if not selected:
        raise RuntimeError(
            "none of the requested LV-Eval English subsets were found: "
            + ", ".join(subsets)
        )
    return selected


def lveval_candidates(
    path: Path,
    counter: TokenCounter,
    *,
    subsets: Sequence[str],
) -> Iterator[Candidate]:
    def raw_items() -> Iterator[tuple[str, str, str, str]]:
        for selected_file in select_lveval_files(path, subsets):
            for row, source, row_index in iter_json_rows(selected_file):
                context = scalar_text(row.get("context"))
                question = scalar_text(row.get("input"))
                answers = list_values(row.get("answers", row.get("answer")))
                output = " ".join(scalar_text(answer) for answer in answers).strip()
                if not context or not question or not output:
                    continue
                yield (
                    str(row.get("_id") or row.get("id") or row_index),
                    str(source),
                    context + question,
                    output,
                )

    for batch in iter_batches(raw_items(), counter.batch_size):
        prompts = [item[2] for item in batch]
        outputs = [item[3] for item in batch]
        paper_input_lengths = counter.count_pieces_many(prompts)
        paper_output_lengths = counter.count_pieces_many(outputs)
        input_lengths = counter.count_many(prompts)
        for index, item in enumerate(batch):
            source_id, source_path, prompt, output = item
            paper_output_tokens = paper_output_lengths[index]
            yield Candidate(
                dataset="lveval",
                source_id=source_id,
                source_path=source_path,
                prompt=prompt,
                reference_output=output,
                input_tokens=input_lengths[index],
                output_tokens=paper_output_tokens,
                paper_input_tokens=paper_input_lengths[index],
                paper_output_tokens=paper_output_tokens,
            )


def common_paper_filter(candidate: Candidate) -> bool:
    return candidate.paper_input_tokens >= 16 and candidate.paper_output_tokens >= 4


def within_instance_caps(
    candidate: Candidate,
    *,
    max_input_tokens: int,
    max_total_tokens: int,
    max_output_tokens: int,
) -> bool:
    return (
        (max_input_tokens <= 0 or candidate.input_tokens <= max_input_tokens)
        and (max_total_tokens <= 0 or candidate.total_tokens <= max_total_tokens)
        and (max_output_tokens <= 0 or candidate.output_tokens <= max_output_tokens)
    )


def truncate_candidate(
    candidate: Candidate,
    counter: TokenCounter,
    *,
    max_input_tokens: int,
    max_total_tokens: int,
    max_output_tokens: int,
) -> tuple[Candidate, dict[str, Any]]:
    """Transform one selected input without replacing, dropping or reranking it."""
    prompt, input_tokens = counter.truncate_prompt_head_tail(
        candidate.prompt, max_input_tokens
    )
    effective = replace(candidate, prompt=prompt, input_tokens=input_tokens)
    if not within_instance_caps(
        effective,
        max_input_tokens=max_input_tokens,
        max_total_tokens=max_total_tokens,
        max_output_tokens=max_output_tokens,
    ):
        raise ValueError(
            "selected request exceeds output/total cap after input truncation: "
            f"dataset={candidate.dataset} source_id={candidate.source_id} "
            f"input={input_tokens} output={candidate.output_tokens} "
            f"max_total={max_total_tokens} max_output={max_output_tokens}; "
            "no request was dropped and no output target was shortened"
        )
    return effective, {
        "original_input_tokens": candidate.input_tokens,
        "input_truncated": prompt != candidate.prompt,
        "removed_input_tokens": candidate.input_tokens - input_tokens,
        "input_truncation_method": (
            "head_tail_character_offsets" if prompt != candidate.prompt else "none"
        ),
    }


def scan_candidates(
    factory: Callable[[], Iterable[Candidate]],
    predicate: Callable[[Candidate], bool],
    *,
    dataset: str,
    progress_every: int,
) -> tuple[int, list[int], list[int]]:
    count = 0
    scanned = 0
    input_lengths: list[int] = []
    output_lengths: list[int] = []
    for candidate in factory():
        scanned += 1
        if not common_paper_filter(candidate) or not predicate(candidate):
            pass
        else:
            count += 1
            input_lengths.append(candidate.input_tokens)
            output_lengths.append(candidate.output_tokens)
        if progress_every > 0 and scanned % progress_every == 0:
            print(
                "[PROGRESS] "
                f"phase=scan dataset={dataset} "
                f"candidates={scanned} eligible={count}",
                flush=True,
            )
    return count, input_lengths, output_lengths


def collect_sampled_candidates(
    factory: Callable[[], Iterable[Candidate]],
    predicate: Callable[[Candidate], bool],
    selected_indices: Sequence[int],
    *,
    dataset: str,
    progress_every: int,
) -> list[Candidate]:
    selected_set = set(selected_indices)
    selected_by_index: dict[int, Candidate] = {}
    eligible_index = 0
    scanned = 0
    for candidate in factory():
        scanned += 1
        if not common_paper_filter(candidate) or not predicate(candidate):
            pass
        else:
            if eligible_index in selected_set:
                selected_by_index[eligible_index] = candidate
            eligible_index += 1
        if progress_every > 0 and scanned % progress_every == 0:
            print(
                "[PROGRESS] "
                f"phase=collect dataset={dataset} "
                f"candidates={scanned} selected={len(selected_by_index)}",
                flush=True,
            )
    if len(selected_by_index) != len(selected_indices):
        raise RuntimeError(
            "candidate source changed between count and collection passes"
        )
    # random.sample returns a randomized order. Preserve it before the public
    # artifact's later whole-mixture shuffle instead of accidentally restoring
    # source-file order through a set.
    return [selected_by_index[index] for index in selected_indices]


def generate_poisson_offsets(
    request_count: int,
    request_rate: float,
    seed: int,
) -> list[float]:
    """Use LoongServe's normalized exponential-arrival construction."""

    if request_count <= 0:
        return []
    if request_rate <= 0 or not math.isfinite(request_rate):
        raise ValueError("request_rate must be finite and positive")
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required for LoongServe-compatible Poisson arrivals"
        ) from exc
    rng = np.random.RandomState(seed)
    if request_count == 1:
        return [0.0]
    intervals = rng.exponential(1.0 / request_rate, request_count)
    cumulative = np.cumsum(intervals)
    # The artifact makes the first issued request the trace origin, then
    # rescales the retained prefix so its last timestamp is N / request_rate.
    cumulative -= cumulative[0]
    normalized = cumulative / cumulative[-1] * (request_count / request_rate)
    return [float(value) for value in normalized]


def length_summary(values: Sequence[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values) if values else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sharegpt-path", type=Path, required=True)
    parser.add_argument("--leval-path", type=Path, required=True)
    parser.add_argument("--lveval-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-mode", choices=("hf", "approx"), default="hf")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--tokenizer-workers",
        type=int,
        default=1,
        help=("number of native fast-tokenizer CPU threads; results remain " "ordered"),
    )
    parser.add_argument(
        "--tokenizer-batch-size",
        type=int,
        default=32,
        help="maximum candidate texts per native tokenizer batch",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="print scan/collection progress every N candidates; 0 disables",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--requests-per-dataset", type=int, default=1500)
    parser.add_argument("--request-rate", type=float, default=0.5)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--max-total-tokens", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=0)
    parser.add_argument(
        "--length-cap-policy",
        choices=(
            "filter_then_sample", "paper_sample_then_filter",
            "paper_sample_then_truncate",
        ),
        default="filter_then_sample",
        help=("truncate preserves the uncapped sample/order and output targets; "
              "requires HF fast tokenization and a positive input cap"),
    )
    parser.add_argument(
        "--insufficient-policy",
        choices=("error", "downsample_all"),
        default="error",
    )
    parser.add_argument(
        "--lveval-subsets",
        nargs="+",
        default=list(DEFAULT_LVEVAL_SUBSETS),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.output_path}")
    if args.requests_per_dataset <= 0:
        raise ValueError("requests-per-dataset must be positive")
    if args.tokenizer_workers <= 0:
        raise ValueError("tokenizer-workers must be positive")
    if args.tokenizer_batch_size <= 0:
        raise ValueError("tokenizer-batch-size must be positive")
    if args.progress_every < 0:
        raise ValueError("progress-every must be non-negative")
    for name in (
        "max_input_tokens",
        "max_total_tokens",
        "max_output_tokens",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")

    truncate_inputs = args.length_cap_policy == "paper_sample_then_truncate"
    if truncate_inputs and (
        args.max_input_tokens <= 0 or args.tokenizer_mode != "hf"
    ):
        raise ValueError(
            "paper_sample_then_truncate requires --max-input-tokens > 0 "
            "and --tokenizer-mode hf"
        )

    counter = TokenCounter(
        mode=args.tokenizer_mode,
        tokenizer_name=args.tokenizer,
        local_files_only=args.local_files_only,
        workers=args.tokenizer_workers,
        batch_size=args.tokenizer_batch_size,
    )
    if truncate_inputs and not counter.is_fast_tokenizer:
        raise ValueError("paper_sample_then_truncate requires a fast tokenizer")
    print(
        "[TOKENIZER] "
        f"mode={args.tokenizer_mode} workers={args.tokenizer_workers} "
        f"batch_size={args.tokenizer_batch_size} "
        f"fast={int(counter.is_fast_tokenizer)} "
        "ordered_results=1 count_cache=sha256",
        flush=True,
    )
    if (
        args.tokenizer_mode == "hf"
        and args.tokenizer_workers > 1
        and not counter.is_fast_tokenizer
    ):
        print(
            "[WARN] tokenizer is not a Hugging Face fast tokenizer; "
            "TOKENIZER_WORKERS may not improve throughput",
            flush=True,
        )
    factories: dict[str, Callable[[], Iterable[Candidate]]] = {
        "sharegpt": lambda: sharegpt_candidates(
            args.sharegpt_path, counter, seed=args.seed
        ),
        "leval": lambda: leval_candidates(args.leval_path, counter),
        "lveval": lambda: lveval_candidates(
            args.lveval_path,
            counter,
            subsets=args.lveval_subsets,
        ),
    }
    cap_predicate = lambda candidate: within_instance_caps(
        candidate,
        max_input_tokens=args.max_input_tokens,
        max_total_tokens=args.max_total_tokens,
        max_output_tokens=args.max_output_tokens,
    )
    sample_predicate = (
        cap_predicate
        if args.length_cap_policy == "filter_then_sample"
        else lambda _candidate: True
    )

    scans: dict[str, tuple[int, list[int], list[int]]] = {}
    for dataset in PAPER_DATASETS:
        scans[dataset] = scan_candidates(
            factories[dataset],
            sample_predicate,
            dataset=dataset,
            progress_every=args.progress_every,
        )
        cache_stats = counter.cache_stats()
        print(
            f"[SCAN] dataset={dataset} eligible={scans[dataset][0]} "
            f"cache_entries={cache_stats['entries']} "
            f"cache_hits={cache_stats['hits']}",
            flush=True,
        )
    sample_count = args.requests_per_dataset
    smallest = min(scan[0] for scan in scans.values())
    if smallest < sample_count:
        if args.insufficient_policy == "error":
            counts = ", ".join(
                f"{dataset}={scans[dataset][0]}" for dataset in PAPER_DATASETS
            )
            raise RuntimeError(
                f"need {sample_count} eligible requests per dataset; {counts}"
            )
        sample_count = smallest
    if sample_count <= 0:
        raise RuntimeError("length filters removed every request")

    mix_rng = random.Random(args.seed)
    selected: list[Candidate] = []
    sampled_before_cap: dict[str, int] = {}
    selected_after_cap: dict[str, int] = {}
    for dataset in PAPER_DATASETS:
        count = scans[dataset][0]
        indices = mix_rng.sample(range(count), sample_count)
        candidates = collect_sampled_candidates(
            factories[dataset],
            sample_predicate,
            indices,
            dataset=dataset,
            progress_every=args.progress_every,
        )
        sampled_before_cap[dataset] = len(candidates)
        if args.length_cap_policy == "paper_sample_then_filter":
            candidates = [
                candidate for candidate in candidates if cap_predicate(candidate)
            ]
        selected_after_cap[dataset] = len(candidates)
        selected.extend(candidates)
    if not selected:
        raise RuntimeError("length filters removed every sampled request")
    mix_rng.shuffle(selected)

    # The artifact benchmark shuffles the loaded mixed dataset once more with
    # the benchmark seed before constructing request arrivals.
    benchmark_rng = random.Random(args.seed)
    benchmark_rng.shuffle(selected)
    offsets = generate_poisson_offsets(len(selected), args.request_rate, args.seed)

    rows: list[dict[str, Any]] = []
    for event_index, (candidate, offset) in enumerate(zip(selected, offsets)):
        audit: dict[str, Any] = {}
        if truncate_inputs:
            candidate, audit = truncate_candidate(
                candidate,
                counter,
                max_input_tokens=args.max_input_tokens,
                max_total_tokens=args.max_total_tokens,
                max_output_tokens=args.max_output_tokens,
            )
        request_id = f"loongserve-mixed-{event_index + 1:05d}"
        rows.append(
            {
                "schema_version": 1,
                "event_index": event_index,
                "request_id": request_id,
                "tenant_id": request_id,
                "dataset": candidate.dataset,
                "role": ("short" if candidate.dataset == "sharegpt" else "long"),
                "prompt": candidate.prompt,
                "reference_output": candidate.reference_output,
                "input_tokens": candidate.input_tokens,
                "max_tokens": candidate.output_tokens,
                "min_tokens": candidate.output_tokens,
                "paper_input_tokens": candidate.paper_input_tokens,
                "paper_output_tokens": candidate.paper_output_tokens,
                "base_scheduled_offset_s": offset,
                "source_id": candidate.source_id,
                "source_path": candidate.source_path,
                **audit,
            }
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    caps_enabled = any(
        value > 0
        for value in (
            args.max_input_tokens,
            args.max_total_tokens,
            args.max_output_tokens,
        )
    )
    selected_lengths = {
        dataset: {
            "input_tokens": length_summary(
                [row["input_tokens"] for row in rows if row["dataset"] == dataset]
            ),
            "output_tokens": length_summary(
                [row["max_tokens"] for row in rows if row["dataset"] == dataset]
            ),
        }
        for dataset in PAPER_DATASETS
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "workload": "LoongServe Mixed",
        "paper_contract": {
            "dataset_ratio": "ShareGPT:L-Eval:LV-Eval = 1:1:1",
            "paper_requests_per_dataset": 1500,
            "sampling_seed": args.seed,
            "arrival_process": "normalized exponential (Poisson)",
        },
        "artifact_reference": {
            "repository": "https://github.com/LoongServe/LoongServe",
            "commit": LOONGSERVE_ARTIFACT_COMMIT,
            "preprocessor": "test/longserve/4-preprocess-dataset.py",
            "mixer": "test/longserve/4-mix-dataset.py",
            "arrival_driver": "test/longserve/5-benchmark-serving.py",
        },
        "reproduction_class": (
            "model_feasible_derivative"
            if caps_enabled or args.tokenizer_mode != "hf"
            else "artifact_rule_reproduction_with_selected_tokenizer"
        ),
        "tokenizer": {
            "mode": args.tokenizer_mode,
            "requested": args.tokenizer,
            "resolved": counter.resolved_tokenizer,
        },
        "preparation_execution": {
            "tokenizer_workers": args.tokenizer_workers,
            "tokenizer_batch_size": args.tokenizer_batch_size,
            "parallelism_backend": (
                "huggingface-fast-tokenizer-rayon"
                if counter.is_fast_tokenizer
                else (
                    "tokenizer-batch-without-fast-rayon"
                    if args.tokenizer_mode == "hf"
                    else "approximate-python"
                )
            ),
            "is_fast_tokenizer": counter.is_fast_tokenizer,
            "ordered_batch_results": True,
            "token_count_cache": "sha256-content-addressed",
            "token_count_cache_stats": counter.cache_stats(),
        },
        "request_rate_rps": args.request_rate,
        "base_trace_duration_s": offsets[-1] if offsets else 0.0,
        "requested_requests_per_dataset": args.requests_per_dataset,
        "effective_sample_count_per_dataset": sample_count,
        "sampled_before_cap": sampled_before_cap,
        "selected_after_cap": selected_after_cap,
        "total_requests": len(rows),
        "length_cap_policy": args.length_cap_policy,
        "insufficient_policy": args.insufficient_policy,
        "length_caps": {
            "max_input_tokens": args.max_input_tokens,
            "max_total_tokens": args.max_total_tokens,
            "max_output_tokens": args.max_output_tokens,
            "semantics": (
                "input head/tail truncation after uncapped sampling; "
                "no upper-length request filtering; output/total caps validate only"
                if truncate_inputs
                else "whole-request filtering; no prompt truncation"
            ),
            "input_length_basis": (
                "serving tokenizer encode with normal special tokens"
            ),
        },
        "paper_length_semantics": {
            "sharegpt": "tokenizer(prompt).input_ids",
            "leval": ("len(tokenize(document)) + len(tokenize(instruction))"),
            "lveval": "len(tokenize(context + question))",
            "output": "artifact dataset-specific token count",
        },
        "eligible_length_summary": {
            dataset: {
                "input_tokens": length_summary(scans[dataset][1]),
                "output_tokens": length_summary(scans[dataset][2]),
            }
            for dataset in PAPER_DATASETS
        },
        "selected_length_summary": selected_lengths,
        "lveval_subsets": args.lveval_subsets,
        "trace_jsonl": str(args.output_path),
        "trace_sha256": file_sha256(args.output_path),
    }
    if truncate_inputs:
        manifest["prompt_truncation"] = {
            "method": "head_tail_character_offsets",
            "final_length_verified_with_special_tokens": True,
            "order_and_arrivals_match_uncapped_sample": True,
            "output_targets_preserved": True,
            "truncated_requests": sum(row["input_truncated"] for row in rows),
            "truncated_by_dataset": {
                dataset: sum(
                    row["input_truncated"] for row in rows
                    if row["dataset"] == dataset
                ) for dataset in PAPER_DATASETS
            },
            "original_selected_input_lengths": {
                dataset: length_summary([
                    row["original_input_tokens"] for row in rows
                    if row["dataset"] == dataset
                ]) for dataset in PAPER_DATASETS
            },
        }
    manifest_path = args.output_path.with_suffix(
        args.output_path.suffix + ".manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[DONE] "
        f"trace={args.output_path} requests={len(rows)} "
        f"duration_s={manifest['base_trace_duration_s']:.3f}",
        flush=True,
    )
    cache_stats = counter.cache_stats()
    print(
        "[DONE] token_count_cache "
        f"entries={cache_stats['entries']} hits={cache_stats['hits']} "
        f"misses={cache_stats['misses']}",
        flush=True,
    )
    print(f"[DONE] manifest={manifest_path}", flush=True)
    if truncate_inputs:
        print(
            "[TRUNCATION] "
            f"input_cap={args.max_input_tokens} "
            f"truncated={manifest['prompt_truncation']['truncated_requests']} "
            "length_filtered=0 output_targets_unchanged=1 order_unchanged=1 "
            f"dataset_counts={json.dumps(selected_after_cap, sort_keys=True)}",
            flush=True,
        )
    counter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
