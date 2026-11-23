# ZoneAtten vendor_v8: SLO-Guarded Scheduling and TBT-First SM Allocation

`vendor_v8` is a copy-on-write descendant of `vendor_v7`. It keeps the
vendor_v7 context-ring, Min-Ring-Chunks, decode FCFS, phase-depth, and dynamic
GreenContext paths, then adds two opt-in policies:

```text
slo_srpt
  profiled remaining chunks
    -> infeasible current set: EDF overload
    -> feasible: TTFT critical -> projected-starved -> safe SRPT

tbt_slo_profiled
  measured overlap p95 -> minimum Decode SM satisfying TBT SLO
```

The complete design and claim boundary are in
`docs/54_VENDOR_V8_SLO_JOINT_SCHEDULER_PLAN.md`.

## Default behavior

All new behavior is disabled by default.

- Without `--enable-ttft-prefill-chunk-scheduling`, the existing vLLM
  prefill order is unchanged.
- Existing prefill policies `fcfs`, `edf`, `srpt`, and `round_robin` retain
  their vendor_v7 behavior.
- `VLLM_PD_DYNAMIC_SM_POLICY` defaults to `ewma_best_fit`.
- `VLLM_VIRTUAL_CONTEXT_RESIDENCY_VICTIM_POLICY` still defaults to
  `longest_logical_context_first`.

## SLO-guarded prefill

Example server arguments:

```bash
--scheduling-policy fcfs \
--enable-chunked-prefill \
--enable-ttft-prefill-chunk-scheduling \
--ttft-prefill-chunk-scheduling-policy slo_srpt \
--ttft-prefill-chunk-quantum-tokens 2048 \
--ttft-prefill-estimated-ms-per-token 0.20 \
--ttft-prefill-service-profile-path /absolute/path/prefill_profile.json \
--ttft-prefill-slo-guard-ms 500 \
--ttft-prefill-starvation-threshold-ms 5000 \
--ttft-prefill-starvation-max-bypasses 8 \
--enable-tbt-decode-scheduling \
--tbt-decode-scheduling-policy fcfs
```

`0.20 ms/token` is only a syntax example and remains the explicit fallback
for an uncovered profile bucket. The Prefill profile records resulting
context, chunk size, P/D profile ID, native/ring mode, and measured p95 chunk
time. The scheduler sums every remaining chunk and takes the maximum across
the execution variants declared in the file. If current-set EDF predicts any
TTFT miss, the iteration runs in `overload_edf` order and reports the predicted
violation count/sum/maximum.

Prefill profile schema:

```json
{
  "schema_version": 1,
  "points": [
    {
      "profile_id": "d20_p20",
      "prefill_mode": "ring",
      "max_context_tokens": 100000,
      "max_chunk_tokens": 2048,
      "p95_prefill_chunk_ms": 812.4
    }
  ]
}
```

Measure and supply conservative profile/fallback values for the model, GPU,
allowed P/D profiles, and execution modes under test.

Per-request SLOs use `vllm_xargs`:

```json
{
  "vllm_xargs": {
    "ttft_prefill_deadline_ms": 120000,
    "tbt_slo_ms": 128.0
  }
}
```

## Profiled TBT-first SM allocation

Required environment:

```bash
VLLM_PD_DYNAMIC_SM_ENABLED=1
VLLM_PD_DYNAMIC_SM_POLICY=tbt_slo_profiled
VLLM_PD_TBT_PROFILE_PATH=/absolute/path/tbt_overlap_profile.json
VLLM_PD_TBT_DEFAULT_SLO_MS=128
VLLM_PD_TBT_PROFILE_MARGIN_MS=5
VLLM_PD_TBT_PROFILE_STRICT=1
```

Start with `VLLM_PD_DYNAMIC_SM_SHADOW=1`. Enforced mode additionally needs
the same independent P/D queues, bounded phase ownership, and GreenContext
execute routing as vendor_v7.

Profile schema:

```json
{
  "schema_version": 1,
  "points": [
    {
      "profile_id": "d24_p16",
      "max_decode_context_tokens": 100000,
      "max_decode_requests": 3,
      "prefill_mode": "ring",
      "p95_decode_step_ms": 118.4
    }
  ]
}
```

The runtime does not extrapolate an uncovered state. Strict mode uses the
largest Decode-SM concurrent profile and records overload; non-strict mode
records a fallback to vendor_v7 EWMA selection.

## Profile generation and short validation

Use the completed native 40-SM reference containing the four currently
required context buckets:

```bash
EXPERIMENT_NAME=native_40sm_profile_baseline_4bucket_v1 \
CONTEXT_TOKENS="25000 50000 75000 100000" \
REPEAT_COUNT=5 \
WARMUP_COUNT_PER_BUCKET=1 \
./run_native_40sm_profile_baseline.sh
```

Generate the fixed request SLO and fixed-partition runtime profiles:

```bash
EXPERIMENT_NAME=vendor_v8_each_partition_profile_v1 \
BASELINE_RESULTS_DIR=/home/noslab-gpu/zoneatten_gpu/native_40sm_profile_baseline_5bucket_v1_results \
VLLM_VENDOR_DIR=/home/noslab-gpu/zoneatten_gpu/vendor_v8 \
CONTEXT_TOKENS="25000 50000 75000 100000" \
GREEN_SM_SPLITS="8/32 16/24 20/20 24/16 32/8" \
PREFILL_MODES="native ring" \
ACTIVE_DECODE_REQUEST_COUNTS="1 2 3" \
DECODE_AUX_CONTEXT_TOKENS=10000 \
./run_vendor_v8_partition_profiler.sh
```

Then run the completion-gated 3x100K plus 6x10K workload:

```bash
EXPERIMENT_NAME=vendor_v8_short_3long_6short_v1 \
PROFILE_RESULTS_DIR=/home/noslab-gpu/zoneatten_gpu/vendor_v8_each_partition_profile_v1_results \
VLLM_VENDOR_DIR=/home/noslab-gpu/zoneatten_gpu/vendor_v8 \
SHORT_INJECTION_OFFSETS_SEC="5 10" \
./run_vendor_v8_short_validation.sh
```

`request_slo_policy.json` is fixed from the 40-SM no-contention reference.
Partition measurements evaluate that contract; they do not loosen it by
`40 / partition_sms`. The current default stops at the measured 100K bucket;
125K can be restored explicitly only after its baseline exists. Profile
switching in the validation remains
quiescent-boundary only. Each Decode state uses one target-context anchor plus
`N-1` auxiliary 10K anchors; Round Robin is used only during their setup. A
trial fails if any anchor exits before probe submission, and only gaps
observed while every declared anchor remains active and actual Prefill worker
execution overlaps the gap enter the TBT profile.

Native states whose target Decode anchor, same-context Prefill probe, and
auxiliary anchors exceed `NATIVE_PROFILE_MAX_RESIDENT_TOKENS` are recorded as
skipped and remain uncovered. Ring states are not filtered by that native
resident budget.

## CPU-only validation

```bash
python tools/vendor_v8_smoke/test_slo_prefill_scheduler.py
python tools/vendor_v8_smoke/test_tbt_profiled_sm_allocator.py
python tools/vendor_v8_smoke/test_vendor_v8_contract.py
python tools/vendor_v8_smoke/test_vendor_v8_profile_workflows.py

VLLM_VENDOR_DIR=/home/yuhwa2323/zoneatten_gpu/vendor_v8 \
python tools/prefill_chunk_scheduler_smoke.py

VLLM_VENDOR_DIR=/home/yuhwa2323/zoneatten_gpu/vendor_v8 \
python tools/decode_scheduler_smoke.py

VLLM_VENDOR_DIR=/home/yuhwa2323/zoneatten_gpu/vendor_v8 \
python tools/green_context_smoke/test_pd_sm_allocator.py
```

## CUDA bridge

Build the copied GreenContext bridge on the CUDA host:

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
./tools/green_context_smoke/build_vendor_v8_bridge.sh

VLLM_VENDOR_DIR=/home/noslab-gpu/zoneatten_gpu/vendor_v8 \
python tools/green_context_smoke/test_vendor_v8_dynamic_sm.py
```

The bridge build is independent of vLLM `_C.abi3.so`; vendor_v8 does not add
a new custom CUDA operator.
