# SPDX-License-Identifier: Apache-2.0

import threading

from vllm.v1.executor.pd_host_enqueue import DecodeFirstHostEnqueueGate


def test_decode_and_prefill_host_enqueue_overlap_after_decode_admission() -> None:
    gate = DecodeFirstHostEnqueueGate()
    # Register Decode before phase-local executor threads start.  This is the
    # same point at which UniProcExecutor accepts the engine submission.
    decode_ticket = gate.register("decode", ordered=True)
    prefill_ticket = gate.register("prefill", ordered=True)

    decode_entered = threading.Event()
    allow_decode_host_return = threading.Event()
    decode_host_returned = threading.Event()
    prefill_entered = threading.Event()
    simulated_decode_gpu_done = threading.Event()

    def run_decode() -> None:
        with gate.enter(decode_ticket):
            decode_entered.set()
            assert allow_decode_host_return.wait(timeout=2)
        # The host enqueue section has ended, but device completion has not.
        assert not simulated_decode_gpu_done.is_set()
        decode_host_returned.set()

    def run_prefill() -> None:
        with gate.enter(prefill_ticket):
            prefill_entered.set()
            assert not simulated_decode_gpu_done.is_set()
            assert gate.phase_snapshot() == (True, True, 0)

    prefill_thread = threading.Thread(target=run_prefill)
    decode_thread = threading.Thread(target=run_decode)
    prefill_thread.start()
    decode_thread.start()

    assert decode_entered.wait(timeout=2)
    # Prefill waits only until the already-registered Decode has entered its
    # phase-local lane. It does not wait for the Decode execute_model call to
    # return, and neither lane waits for GPU completion.
    assert prefill_entered.wait(timeout=2)
    allow_decode_host_return.set()
    assert decode_host_returned.wait(timeout=2)
    simulated_decode_gpu_done.set()
    decode_thread.join(timeout=2)
    prefill_thread.join(timeout=2)
    assert not decode_thread.is_alive()
    assert not prefill_thread.is_alive()
    assert gate.snapshot() == (False, 0)


def test_prefill_waits_until_all_already_submitted_decodes_enter() -> None:
    gate = DecodeFirstHostEnqueueGate()
    first = gate.register("decode", ordered=True)
    second = gate.register("decode", ordered=True)
    prefill = gate.register("prefill", ordered=True)
    order: list[str] = []

    def enter(ticket, label: str) -> None:
        with gate.enter(ticket):
            order.append(label)

    threads = [
        threading.Thread(target=enter, args=(prefill, "prefill")),
        threading.Thread(target=enter, args=(first, "decode-1")),
        threading.Thread(target=enter, args=(second, "decode-2")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert set(order[:2]) == {"decode-1", "decode-2"}
    assert order[-1] == "prefill"
    assert gate.snapshot() == (False, 0)


def test_late_decode_enters_while_prefill_is_active() -> None:
    gate = DecodeFirstHostEnqueueGate()
    prefill = gate.register("prefill", ordered=True)
    prefill_entered = threading.Event()
    release_prefill = threading.Event()
    decode_entered = threading.Event()

    def run_prefill() -> None:
        with gate.enter(prefill):
            prefill_entered.set()
            assert release_prefill.wait(timeout=2)

    prefill_thread = threading.Thread(target=run_prefill)
    prefill_thread.start()
    assert prefill_entered.wait(timeout=2)

    decode = gate.register("decode", ordered=True)

    def run_decode() -> None:
        with gate.enter(decode):
            decode_entered.set()
            assert gate.phase_snapshot() == (True, True, 0)

    decode_thread = threading.Thread(target=run_decode)
    decode_thread.start()
    assert decode_entered.wait(timeout=2)

    release_prefill.set()
    decode_thread.join(timeout=2)
    prefill_thread.join(timeout=2)
    assert gate.snapshot() == (False, 0)
