# vendor_v6 Prefill Chunk Scheduler

This directory is a copy-on-write descendant of `vendor_v4`. The new behavior
is disabled unless all of the following scheduler options are supplied:

```bash
--scheduling-policy fcfs \
--enable-chunked-prefill \
--enable-ttft-prefill-chunk-scheduling \
--ttft-prefill-chunk-scheduling-policy round_robin \
--ttft-prefill-chunk-quantum-tokens 2048
```

The policy value can be `fcfs`, `edf`, `srpt`, or `round_robin`. EDF request
deadlines are positive milliseconds relative to request arrival:

```json
{"vllm_xargs": {"ttft_prefill_deadline_ms": 5000}}
```

The selector chooses at most one prefill request per scheduler iteration.
Decode scheduling and the existing KV/context-ring admission checks remain in
control. Round Robin advances only when the selected chunk is actually
scheduled. The default quantum `0` keeps existing scheduler/ring chunk caps.

Run the CPU-only policy and source-contract checks from the repository root:

```bash
python tools/prefill_chunk_scheduler_smoke.py
```

Build and validate the fixed-split Green Context bridge on the CUDA host:

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
./tools/green_context_smoke/build_vendor_v6_bridge.sh

VLLM_VENDOR_DIR=/home/noslab-gpu/zoneatten_gpu/vendor_v6 \
VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS=20 \
VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS=20 \
python tools/green_context_smoke/test_vendor_v6_bridge.py
```

The build writes
`vendor_v6/vllm/model_executor/layers/green_context_lib/libvllm_green_context.so`.
It does not rebuild vLLM's `_C.abi3.so`.
