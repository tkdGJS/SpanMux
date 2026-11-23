#!/usr/bin/env python3
"""CPU/mock smoke checks for the attention correctness trace."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


VENDOR_DIR = Path(
    os.environ.get(
        "VLLM_VENDOR_DIR",
        Path(__file__).resolve().parents[2] / "vendor_v4",
    )
).resolve()
MODULE_PATH = (
    VENDOR_DIR
    / "vllm/v1/attention/ops/context_ring_correctness.py"
)
SPEC = importlib.util.spec_from_file_location(
    "context_ring_correctness_under_test",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load attention trace module: {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

attention_correctness_trace_enabled = (
    MODULE.attention_correctness_trace_enabled
)
build_attention_trace_rows = MODULE.build_attention_trace_rows
maybe_record_attention_output = MODULE.maybe_record_attention_output
reset_attention_correctness_trace_state_for_tests = (
    MODULE.reset_attention_correctness_trace_state_for_tests
)


_TRACE_ENV_NAMES = (
    "VLLM_CONTEXT_RING_CORRECTNESS_MODE",
    "VLLM_CONTEXT_RING_CORRECTNESS_DIR",
    "VLLM_CONTEXT_RING_CORRECTNESS_PATHS",
    "VLLM_CONTEXT_RING_CORRECTNESS_PHASES",
    "VLLM_CONTEXT_RING_CORRECTNESS_LAYERS",
    "VLLM_CONTEXT_RING_CORRECTNESS_REQUESTS",
    "VLLM_CONTEXT_RING_CORRECTNESS_FILE_CHUNK_TOKENS",
    "VLLM_CONTEXT_RING_CORRECTNESS_MAX_RECORDS",
    "VLLM_CONTEXT_RING_CORRECTNESS_STRICT",
    "VLLM_CONTEXT_RING_CORRECTNESS_CAPTURE_QKV",
)


class _FakeTensor:
    def __init__(self, rows: int, width: int = 4) -> None:
        self.rows = rows
        self.width = width
        self.dtype = "fake.float16"
        self.shape = (rows, width)

    def __getitem__(self, index: slice) -> "_FakeTensor":
        start, stop, step = index.indices(self.rows)
        return _FakeTensor(len(range(start, stop, step)), self.width)

    def detach(self) -> "_FakeTensor":
        return self

    def to(self, **_: object) -> "_FakeTensor":
        return _FakeTensor(self.rows, self.width)

    def contiguous(self) -> "_FakeTensor":
        return self


class AttentionOutputTraceSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_env = {
            name: os.environ.get(name) for name in _TRACE_ENV_NAMES
        }
        for name in _TRACE_ENV_NAMES:
            os.environ.pop(name, None)
        reset_attention_correctness_trace_state_for_tests()

    def tearDown(self) -> None:
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        reset_attention_correctness_trace_state_for_tests()

    def test_default_off_does_not_touch_output(self) -> None:
        class _ExplodingOutput:
            def __getitem__(self, _: object) -> object:
                raise AssertionError("disabled trace touched the output tensor")

        self.assertFalse(attention_correctness_trace_enabled())
        self.assertEqual(
            maybe_record_attention_output(
                execution_path="native_hbm",
                layer_name="model.layers.0.self_attn.attn",
                attn_metadata=object(),
                output=_ExplodingOutput(),
            ),
            0,
        )

    def test_rows_use_logical_query_ranges(self) -> None:
        rows = build_attention_trace_rows(
            request_ids=["req-a", "req-b"],
            query_start_locs=[0, 4, 5],
            sequence_lengths=[100, 11],
            phase="prefill",
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].logical_query_start, 96)
        self.assertEqual(rows[0].query_length, 4)
        self.assertEqual((rows[0].output_start, rows[0].output_end), (0, 4))
        self.assertEqual(rows[1].logical_query_start, 10)
        self.assertEqual(rows[1].query_length, 1)
        self.assertEqual((rows[0].request_ordinal, rows[1].request_ordinal), (0, 1))

    def test_unannotated_mixed_rows_infer_phase(self) -> None:
        rows = build_attention_trace_rows(
            request_ids=["prefill", "decode"],
            query_start_locs=[0, 4, 5],
            sequence_lengths=[4, 101],
            phase="mixed",
        )
        self.assertEqual([row.phase for row in rows], ["prefill", "decode"])

    def test_record_splits_files_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            os.environ.update({
                "VLLM_CONTEXT_RING_CORRECTNESS_MODE": "record",
                "VLLM_CONTEXT_RING_CORRECTNESS_DIR": temporary_dir,
                "VLLM_CONTEXT_RING_CORRECTNESS_PATHS": "context_ring",
                "VLLM_CONTEXT_RING_CORRECTNESS_PHASES": "prefill",
                "VLLM_CONTEXT_RING_CORRECTNESS_FILE_CHUNK_TOKENS": "2",
            })
            reset_attention_correctness_trace_state_for_tests()
            metadata = SimpleNamespace(
                request_ids=["req/0"],
                query_start_loc_cpu=[0, 5],
                query_start_loc=None,
                seq_lens_cpu=[100],
                seq_lens=None,
                attention_phase="prefill",
            )

            def fake_save(_: object, path: Path) -> None:
                Path(path).write_bytes(b"fake-torch-payload")

            fake_torch = SimpleNamespace(save=fake_save)
            with patch.dict(sys.modules, {"torch": fake_torch}):
                written = maybe_record_attention_output(
                    execution_path="context_ring",
                    layer_name="model.layers.0.self_attn.attn",
                    attn_metadata=metadata,
                    output=_FakeTensor(5),
                )

            self.assertEqual(written, 3)
            rank_dir = Path(temporary_dir) / "rank_000"
            manifest_rows = [
                json.loads(line)
                for line in (rank_dir / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [row["logical_query_start"] for row in manifest_rows],
                [95, 97, 99],
            )
            self.assertEqual(
                [row["query_length"] for row in manifest_rows],
                [2, 2, 1],
            )
            for row in manifest_rows:
                self.assertTrue((rank_dir / row["file"]).is_file())

    def test_max_records_limits_payloads_and_manifest_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            os.environ.update({
                "VLLM_CONTEXT_RING_CORRECTNESS_MODE": "record",
                "VLLM_CONTEXT_RING_CORRECTNESS_DIR": temporary_dir,
                "VLLM_CONTEXT_RING_CORRECTNESS_FILE_CHUNK_TOKENS": "2",
                "VLLM_CONTEXT_RING_CORRECTNESS_MAX_RECORDS": "2",
            })
            reset_attention_correctness_trace_state_for_tests()
            metadata = SimpleNamespace(
                request_ids=["req-0"],
                query_start_loc_cpu=[0, 5],
                query_start_loc=None,
                seq_lens_cpu=[5],
                seq_lens=None,
                attention_phase="prefill",
            )

            def fake_save(_: object, path: Path) -> None:
                Path(path).write_bytes(b"fake-torch-payload")

            with patch.dict(
                sys.modules,
                {"torch": SimpleNamespace(save=fake_save)},
            ):
                written = maybe_record_attention_output(
                    execution_path="native_hbm",
                    layer_name="model.layers.0.self_attn.attn",
                    attn_metadata=metadata,
                    output=_FakeTensor(5),
                )

            self.assertEqual(written, 2)
            manifest_path = (
                Path(temporary_dir) / "rank_000" / "manifest.jsonl"
            )
            self.assertEqual(
                len(manifest_path.read_text(encoding="utf-8").splitlines()),
                2,
            )

    def test_optional_qkv_capture_uses_matching_logical_slices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            os.environ.update({
                "VLLM_CONTEXT_RING_CORRECTNESS_MODE": "record",
                "VLLM_CONTEXT_RING_CORRECTNESS_DIR": temporary_dir,
                "VLLM_CONTEXT_RING_CORRECTNESS_FILE_CHUNK_TOKENS": "2",
                "VLLM_CONTEXT_RING_CORRECTNESS_CAPTURE_QKV": "1",
            })
            reset_attention_correctness_trace_state_for_tests()
            metadata = SimpleNamespace(
                request_ids=["req-0"],
                query_start_loc_cpu=[0, 3],
                query_start_loc=None,
                seq_lens_cpu=[13],
                seq_lens=None,
                attention_phase="prefill",
            )
            payloads: list[dict[str, object]] = []

            def fake_save(payload: object, path: Path) -> None:
                self.assertIsInstance(payload, dict)
                payloads.append(payload)
                Path(path).write_bytes(b"fake-torch-payload")

            with patch.dict(
                sys.modules,
                {"torch": SimpleNamespace(save=fake_save)},
            ):
                written = maybe_record_attention_output(
                    execution_path="native_hbm",
                    layer_name="model.layers.0.self_attn.attn",
                    attn_metadata=metadata,
                    output=_FakeTensor(3, 8),
                    query=_FakeTensor(3, 8),
                    key=_FakeTensor(3, 4),
                    value=_FakeTensor(3, 4),
                )

            self.assertEqual(written, 2)
            self.assertEqual([payload["query"].rows for payload in payloads], [2, 1])
            self.assertEqual([payload["key"].width for payload in payloads], [4, 4])
            manifest_path = (
                Path(temporary_dir) / "rank_000" / "manifest.jsonl"
            )
            manifest_rows = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(all(row["capture_qkv"] for row in manifest_rows))
            self.assertEqual(manifest_rows[0]["query_shape"], [2, 8])
            self.assertEqual(manifest_rows[0]["key_shape"], [2, 4])


if __name__ == "__main__":
    unittest.main()
