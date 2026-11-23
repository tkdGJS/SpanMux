# ZoneAtten vendor_v7: vendor_v6 Scheduling + Dynamic P/D SM Allocation

`vendor_v7` is a copy-on-write descendant of `vendor_v6`. It preserves the
vendor_v6 request-level prefill/decode scheduling, context-ring residency, and
joint-admission behavior, and adds the quiescent-boundary dynamic P/D SM profile
controller previously implemented in `vendor_v5`.

## Source Lineage

```text
vendor_v6
  + selectable prefill chunk scheduler
  + decode-ready FCFS scheduler boundary
  + min-ring-chunks residency policy
  + context-ring progress and decode-handoff reclaim

vendor_v5 dynamic-SM feature
  + immutable full-device GreenContext profiles
  + EngineCore profile controller and EWMA feedback
  + profile ID/epoch execute-to-sample routing
  + driver-reported SM-count verification

  -> vendor_v7
```

`vendor_v5` and `vendor_v6` remain unchanged baselines.

## Dynamic-SM Boundary

Profile changes occur only when neither P/D phase owns an in-flight future.
On a 40-SM Tesla T4, the automatic profile set is:

```text
0/40, 8/32, 16/24, 20/20, 24/16, 32/8, 40/0
```

The controller chooses `40/0` for decode-only work, `0/40` for prefill-only
work, and an EWMA-based concurrent profile when both phases are runnable. A
paired P/D launch carries one immutable profile ID and epoch. The worker
synchronizes both streams of the previous GreenContext handle before activating
a new profile epoch.

The implementation does not move kernels already submitted by an in-flight
whole-model forward. Layer-wise profile handoff is out of scope.

## Default Behavior

Dynamic allocation is default-off:

```text
VLLM_PD_DYNAMIC_SM_ENABLED=0
```

With the flag off, vendor_v7 follows vendor_v6 fixed-split behavior. Shadow mode
records controller decisions without routing execution to the profile bank:

```text
VLLM_PD_DYNAMIC_SM_ENABLED=1
VLLM_PD_DYNAMIC_SM_SHADOW=1
```

## Required Runtime Configuration

Enforced mode requires the vendor_v6 independent P/D lane configuration:

```text
independent phase queues
separate P/D scheduler lanes
single in-flight future per phase
execute-model GreenContext routing
```

Typical T4 additions are:

```bash
VLLM_PD_DYNAMIC_SM_ENABLED=1
VLLM_PD_DYNAMIC_SM_STRICT=1
VLLM_PD_DYNAMIC_SM_TOTAL_SMS=40
VLLM_PD_DYNAMIC_SM_PROFILES=auto
VLLM_PD_DYNAMIC_SM_EAGER_PROFILES=1
```

## Build and Validation

The source tree does not ship a built vendor_v7 bridge. A vendor_v6 bridge
copied into this directory would also lack the new SM-count ABI. Build the
vendor_v7 bridge on the CUDA 12.8 GPU host before enforced runs:

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
./tools/green_context_smoke/build_vendor_v7_bridge.sh

VLLM_VENDOR_DIR=/home/noslab-gpu/zoneatten_gpu/vendor_v7 \
python tools/green_context_smoke/test_vendor_v7_dynamic_sm.py
```

CPU-only checks:

```bash
VLLM_VENDOR_DIR=/home/yuhwa2323/zoneatten_gpu/vendor_v7 \
python tools/green_context_smoke/test_pd_sm_allocator.py

python tools/green_context_smoke/test_vendor_v7_contract.py
```
