# SPDX-License-Identifier: Apache-2.0

import json

from vllm.v1.engine.async_jsonl_writer import AsyncJSONLWriter


def test_async_jsonl_writer_reuses_handle_and_drains_on_close(tmp_path) -> None:
    path = tmp_path / "trace" / "iteration.jsonl"
    writer = AsyncJSONLWriter(max_pending=8, flush_every=8)
    for index in range(3):
        assert writer.submit(str(path), {"iteration": index})
    writer.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [{"iteration": 0}, {"iteration": 1}, {"iteration": 2}]
    assert writer.stats() == {
        "submitted": 3,
        "written": 3,
        "dropped": 0,
        "errors": 0,
        "opened_files": 1,
        "open_handles": 0,
    }
