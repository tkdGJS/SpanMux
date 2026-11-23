# SPDX-License-Identifier: Apache-2.0
"""Opt-in, scalar-only capture for offline exact-token TBT calibration.

No scheduler/allocator authority, CUDA calls, or execution-object retention.
The existing bounded writer may drop records; the analyzer requires consecutive
record numbers and a terminal footer, so loss never looks like clean evidence.
"""
from __future__ import annotations

import time
from typing import Any, Callable


class TBTCaptureObserver:
    def __init__(self, emit: Callable[[dict], Any]):
        self._emit = emit
        self.sequence = 0
        self.write("start", contract="tbt-capture-v1")

    def write(self, event: str, **fields) -> None:
        self.sequence += 1
        self._emit(dict(fields, event=event, sequence=self.sequence,
                        timestamp=time.monotonic()))

    @staticmethod
    def prefills(phase_queues) -> list[dict]:
        result = []
        for future, output, _ in phase_queues.get("prefill", ()):
            if future.done():
                continue
            transfer = getattr(output, "kv_retention_layerwise_transfer", None)
            mode = "ring" if (output.pd_prefill_mode == "ring" or
                              getattr(transfer, "state", "") in
                              {"released", "published", "consumed"}) else "native"
            result.append({"submit_seq": int(output.pd_phase_submit_seq),
                           "mode": mode,
                           "query_tokens": int(output.total_num_scheduled_tokens)})
        return result

    def submit(self, output, phase, requests, phase_queues) -> None:
        if output.total_num_scheduled_tokens <= 0:
            return
        rows = []
        for rid, scheduled in output.num_scheduled_tokens.items():
            request = requests.get(rid)
            rows.append({"request_id": rid, "scheduled_tokens": int(scheduled),
                         "context_tokens": (int(request.num_computed_tokens)
                                            if request is not None else -1),
                         "output_start": (int(request.num_output_tokens)
                                          if request is not None else -1)})
        self.write("submit", submit_seq=int(output.pd_phase_submit_seq),
                   phase=phase, profile_id=output.pd_sm_profile_id,
                   epoch=int(output.pd_sm_profile_epoch), rows=rows,
                   prefills=self.prefills(phase_queues))

    def recover(self, output, engine_outputs, phase_queues) -> None:
        # Committed outputs only: cancelled raw model samples are NOT tokens.
        tokens = [{"request_id": item.request_id, "count": len(item.new_token_ids)}
                  for client in engine_outputs.values() for item in client.outputs
                  if item.new_token_ids]
        self.write("recover", submit_seq=int(output.pd_phase_submit_seq),
                   tokens=tokens, prefills=self.prefills(phase_queues))

    def close(self) -> None:
        self.write("end")
