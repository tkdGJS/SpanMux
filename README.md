# SpanMux Experiment Guide

This guide runs the five thesis configurations on the first 100 requests of an
existing LoongServe Mixed trace, with a 2 GiB KV cache and a maximum context
length of 65,536 tokens.

## Server Setup

```bash
python -m pip install -r src/vllm-0.14-src/requirements/cuda.txt \
  -r src/vllm-0.14-src/requirements/build.txt

(
  cd src/vllm-0.14-src
  CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=8 VLLM_VERSION_OVERRIDE=0.14.0 \
    python -m pip install --no-build-isolation --no-deps -e .
)

cp src/vllm-0.14-src/vllm/*.so spanmux/vllm/
cp src/vllm-0.14-src/vllm/vllm_flash_attn/*.so spanmux/vllm/vllm_flash_attn/
```

## Run the Experiment

From the repository root, copy the preset and edit it to point to your Python
environment and existing LoongServe Mixed trace.

```bash
cp exp_script/thesis.env exp_script/thesis.local.env
bash exp_script/run_thesis.sh --config exp_script/thesis.local.env --dry-run
bash exp_script/run_thesis.sh --config exp_script/thesis.local.env
```

If the preset already matches your setup, run it directly.

```bash
bash exp_script/run_thesis.sh
```

The dry run prints the plan for each configuration. Choose a new experiment name
for each run to keep its outputs separate.

| Configuration | Variant ID | Execution |
|---|---|---|
| Baseline | `native_40sm_chunked` | Native mixed batches on 40 SMs with chunked prefill |
| Baseline+SRPT | `baseline_plus` | Native mixed batches with SRPT prefill and FCFS decode |
| P/D+FCFS | `dynamic_hbm_serialized_fullsm` | Separate prefill and decode, with FCFS prefill and serialization under KV pressure |
| P/D+SRPT | `dynamic_hbm_srpt` | Separate prefill and decode, with SRPT prefill and capacity guards |
| SpanMux | `full_vendor_v8` | P/D+SRPT with layer-boundary preemption, KV reclamation, and Context Ring |

## Experiment Settings

Edit [thesis.env](exp_script/thesis.env) or the local copy passed to `--config`.

| Setting | Meaning |
|---|---|
| `VENV_DIR`, `MODEL_NAME` | Python environment and model identifier or local model path |
| `LOONGSERVE_TRACE_JSONL` | Path to the existing LoongServe Mixed trace |
| `BASE_EXPERIMENT_NAME`, `BENCHMARK_VARIANTS` | Output prefix and space-separated list of configurations to run |
| `LOONGSERVE_TERMINATION_MODE=requests`, `LOONGSERVE_REQUEST_LIMIT=100` | Submit the first 100 trace requests in order and wait for submitted requests to finish |
| `LOONGSERVE_ARRIVAL_TIME_SCALE=1` | Multiply trace arrival offsets by this factor. A value of 0.5 halves the gaps |
| `KV_CACHE_MEMORY_BYTES=2147483648` | 2 GiB reserved for the KV cache. Total GPU memory also includes model weights and workspaces |
| `VLLM_MAX_MODEL_LEN=65536` | Maximum total context length per request |
| `VLLM_MAX_NUM_SEQS=100`, `VLLM_MAX_NUM_BATCHED_TOKENS=16384` | Per-iteration limits on sequences and tokens. Available KV capacity may lower the actual batch size |
| `PREFILL_CHUNK_TOKENS=2048`, `VLLM_LONG_PREFILL_TOKEN_THRESHOLD=2048` | Set the per-request prefill chunk limit for the guarded comparison |
| `ISOLATION_GUARDED_2K_COMPARISON=1`, `ISOLATION_MATCHED_COMPARISON=1` | Apply shared comparison settings and each variant's execution and capacity guards |
| `BENCHMARK_CUDAGRAPH=1`, `BENCHMARK_CUDAGRAPH_MODE=FULL_AND_PIECEWISE` | Enable CUDA Graphs. The dry run and saved settings show the path selected for each variant |
| `PD_SM_PROFILES`, `PD_DYNAMIC_SM_POLICY=tbt_slo_feedback` | Candidate Decode/Prefill SM splits and the policy that adjusts them using observed decode TBT |
| `VLLM_PD_EVAL_*_INFLIGHT_DEPTH` | Limits of one prefill future, one decode future, and two futures overall |
| `VLLM_PD_EVAL_LAYERWISE_PROFILE_HANDOFF=1` | Allow SM changes at eligible prefill layer boundaries. SpanMux enables request preemption and KV migration through its variant settings |
| `FULL_VENDOR_V8_RESIDENCY_VICTIM_POLICY=scheduled_prefill_fit` | Select prefill KV reclamation based on the space needed for upcoming work |
| `FULL_VENDOR_V8_SCHEDULED_PREFILL_FIT_CAPACITY_FRACTION=0.98` | Account for decode growth headroom in the reclamation plan |
| `VLLM_PD_EVAL_RING_COMPUTE_GROUP_TOKENS=2048` | Number of historical KV tokens in each Context Ring attention group |
| `VLLM_PD_EVAL_DRAM_REGISTER_DIRECT_MULTICOPY`, `VLLM_PD_EVAL_RELOAD_DIRECT_MULTICOPY`, `VLLM_PD_EVAL_VERIFY_DIRECT_OPS` | Enable direct GPU-to-host and host-to-GPU transfers and verify the compiled operators |
| `WARMUP_COUNT=1`, `WARMUP_OUTPUT_TOKENS=8` | Run one eight-token warmup before the measured requests |
| `REQUEST_TIMEOUT_SEC=28800`, `SERVER_READY_TIMEOUT_SEC=900` | Set the request processing and server startup timeouts, respectively |

## Monitoring and Results

| Option | Measurement |
|---|---|
| `ENABLE_DMON=1` | Samples device GPU utilization, framebuffer memory, power, temperature, and clocks once per second in `nvidia_smi_dmon.log` |
| `ENABLE_LIGHTWEIGHT_TIMELINE_TRACE=1` | Records execution phases, requests, and SM profiles where supported |
| `ITERATION_TRACE_SAMPLE_EVERY_N=100` | Samples iteration diagnostics every 100 iterations. Client latency measurements are collected separately |
| `ENABLE_SCHEDULER_TIMING_TRACE=1` | Records sampled scheduler call times in `scheduler_metrics.jsonl` on supported builds |
| `ENABLE_PD_CUDA_EVENT_TRACE=1` | Records CUDA event timing in `pd_cuda_event_trace.jsonl` on supported execution paths |
| `ENABLE_FULL_ITERATION_TRACE=1`, `ENABLE_PD_EXECUTOR_DEBUG=1` | Enable detailed iteration and executor diagnostics. Both are off in the preset to limit measurement overhead |

Use the same monitoring settings across configurations. Run `--dry-run` before
enabling detailed tracing to check that the selected options can be combined.

Each configuration uses the prefix `<experiment>_workload_loongserve_<variant>`.

| Output | Contents |
|---|---|
| `*_results/client/run_metadata.json` | Observation duration, trace checksum, request counts, and termination settings |
| `*_results/client/request_results.csv` | Request status, token counts, TTFT, TBT, end-to-end latency, and TTFT SLO outcome |
| `*_results/client/client_token_timeline.csv` | Streamed token gaps, token weights, and TBT SLO outcomes |
| `*_results/client/workload_summary.csv` | Completion counts, latency percentiles, and SLO attainment by dataset |
| `*_results/experiment_snapshot/` | Effective settings, SLO policy, and copies of experiment scripts |
| `*_artifacts/`, `*_logs/server.log` | GPU and engine measurements, server errors, and lifecycle logs |

Use `observation_duration_s` for experiment elapsed time. Divide the completed
requests' input or output token total by that duration to calculate throughput.
TTFT attainment counts requests. TBT attainment uses token-weighted intervals
after the first streamed chunk. SLO goodput reports qualifying requests or
tokens per unit time.
