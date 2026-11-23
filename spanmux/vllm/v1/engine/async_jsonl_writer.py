# SPDX-License-Identifier: Apache-2.0
"""Bounded asynchronous JSONL writer for opt-in debug traces."""

import json
import logging
import os
import queue
import threading
import time
from typing import Any, TextIO

logger = logging.getLogger(__name__)

_STOP = object()


class AsyncJSONLWriter:
    """Write debug records without blocking the engine's critical path.

    Queue saturation drops trace records instead of stalling inference.  File
    handles are opened once by the background thread and reused until close.
    Submitted records transfer ownership to the writer and must not be mutated
    afterwards.
    """

    def __init__(
        self,
        *,
        max_pending: int = 4096,
        flush_every: int = 64,
        flush_interval_s: float = 0.25,
    ) -> None:
        self._queue: queue.Queue[object] = queue.Queue(
            maxsize=max(1, int(max_pending))
        )
        self._flush_every = max(1, int(flush_every))
        self._flush_interval_s = max(0.01, float(flush_interval_s))
        self._handles: dict[str, TextIO] = {}
        self._closing = False
        self.submitted = 0
        self.written = 0
        self.dropped = 0
        self.errors = 0
        self.opened_files = 0
        self._thread = threading.Thread(
            target=self._run,
            name="VLLMDebugJSONLWriter",
            daemon=True,
        )
        self._thread.start()

    def submit(self, path: str, record: dict[str, Any]) -> bool:
        if not path or self._closing:
            return False
        try:
            self._queue.put_nowait((path, record))
        except queue.Full:
            self.dropped += 1
            return False
        self.submitted += 1
        return True

    def _handle_for_path(self, path: str) -> TextIO:
        handle = self._handles.get(path)
        if handle is not None:
            return handle
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        handle = open(path, "a", encoding="utf-8")
        self._handles[path] = handle
        self.opened_files += 1
        return handle

    def _flush_all(self) -> None:
        for handle in self._handles.values():
            handle.flush()

    def _run(self) -> None:
        unflushed = 0
        last_flush = time.monotonic()
        while True:
            try:
                item = self._queue.get(timeout=self._flush_interval_s)
            except queue.Empty:
                if unflushed:
                    self._flush_all()
                    unflushed = 0
                    last_flush = time.monotonic()
                continue

            try:
                if item is _STOP:
                    break
                path, record = item  # type: ignore[misc]
                handle = self._handle_for_path(path)
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                self.written += 1
                unflushed += 1
                now = time.monotonic()
                if (
                    unflushed >= self._flush_every
                    or now - last_flush >= self._flush_interval_s
                ):
                    self._flush_all()
                    unflushed = 0
                    last_flush = now
            except Exception:
                self.errors += 1
                logger.exception("Failed to append an asynchronous JSONL record")
            finally:
                self._queue.task_done()

        if unflushed:
            self._flush_all()
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._queue.join()
        self._queue.put(_STOP)
        self._thread.join()
        if self.dropped or self.errors:
            logger.warning(
                "Debug JSONL writer closed: submitted=%d written=%d "
                "dropped=%d errors=%d",
                self.submitted,
                self.written,
                self.dropped,
                self.errors,
            )

    def stats(self) -> dict[str, int]:
        return {
            "submitted": self.submitted,
            "written": self.written,
            "dropped": self.dropped,
            "errors": self.errors,
            "opened_files": self.opened_files,
            "open_handles": len(self._handles),
        }
