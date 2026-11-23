# SPDX-License-Identifier: Apache-2.0
"""Single-owner bounded Ring scratch, with explicit CUDA/allocator ordering."""
from __future__ import annotations


class IsolationRingStateArena:
    def __init__(self, torch, state_type, borrow_factory=None):
        self.torch = torch
        self.state_type = state_type
        self.storage = None
        self.geometry = None
        self.capacity = 0
        self.last_stream = None
        self.last_event = None
        self.active = False
        self.borrow_factory = borrow_factory
        self.borrow = None
        self.borrow_failed = False
        self.shared_bytes = 0

    @property
    def bytes(self):
        if self.storage is None:
            return 0
        return sum(t.numel() * t.element_size() for t in self.storage)

    def acquire(self, query):
        if self.borrow_failed:
            raise RuntimeError("Ring workspace owner failed; cannot reuse arena")
        if self.active:
            raise RuntimeError("Ring state arena is already owned by another run")
        tokens, heads, dim = map(int, query.shape)
        if tokens <= 0:
            raise ValueError("Ring state requires non-empty query")
        stream = self.torch.cuda.current_stream(query.device)
        if self.last_event is not None and stream != self.last_stream:
            stream.wait_event(self.last_event)
        if self.borrow_factory is not None:
            try:
                borrow = self.borrow_factory(query)
            except Exception:
                # Ownership errors cannot silently fall back to fresh tensors.
                self.borrow_failed = True
                raise
            if borrow is not None:
                self.borrow = borrow
                self.active = True
                self.last_stream = stream
                self.shared_bytes = borrow.bytes
                # Prior private tensors record every consumer stream. They are
                # no longer retained when a bounded shared lease is available.
                self.storage, self.geometry, self.capacity = None, None, tokens
                return self.state_type(*borrow.tensors), True
        self.shared_bytes = 0
        geometry = (str(query.device), heads, dim)
        reused = (self.storage is not None and self.geometry == geometry
                  and self.capacity >= tokens)
        if not reused:
            capacity = 1 << (tokens - 1).bit_length()
            # Old allocations were record_stream'ed on every reader; deleting
            # the owner cannot return storage still used by an asynchronous run.
            self.storage = None
            self.geometry, self.capacity = None, 0
            self.storage = tuple(self.torch.empty(
                shape, dtype=self.torch.float32, device=query.device
            ) for shape in ((capacity, heads), (capacity, heads),
                            (capacity, heads, dim)))
            self.geometry, self.capacity = geometry, capacity
        for tensor in self.storage:
            tensor.record_stream(stream)
        self.active = True
        self.last_stream = stream
        return self.state_type(*(tensor[:tokens] for tensor in self.storage)), reused

    def release(self):
        if not self.active:
            return
        # Keep active=True if recording fails: ambiguous completion is fatal,
        # not permission for a later acquire to overwrite the accumulator.
        if self.borrow is not None and (
            self.torch.cuda.current_stream(self.borrow.tensors[0].device)
            != self.last_stream
        ):
            raise RuntimeError("Ring finalize changed its borrowed workspace stream")
        event = self.torch.cuda.Event()
        event.record(self.last_stream)
        if self.borrow is not None:
            self.borrow.release()
            # Final output/LSE is separate; never cache a borrowed tensor view.
            self.borrow = None
        self.last_event = event
        self.active = False
