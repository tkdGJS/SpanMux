# Green Context Smoke Test

This standalone CUDA Driver API smoke test verifies whether the current machine
can create a Green Context, create a stream attached to it, and run a simple
kernel on that stream.

It intentionally does not import vLLM or PyTorch. Passing this test is the first
prerequisite before wiring Green Context streams into the `vendor_v2`
operator-selective prefill/decode prototype.

## Run

On the remote GPU server:

```bash
cd /home/yuhwa2323/zoneatten_gpu
CUDA_HOME=/usr/local/cuda-12.8 \
GREEN_CONTEXT_DEVICE=0 \
GREEN_CONTEXT_SMS=8 \
GREEN_CONTEXT_PRIORITY=-1 \
./tools/green_context_smoke/build_and_run.sh
```

Expected success:

```text
[INFO] device=0 name=...
[INFO] requested_green_sms=8 priority=-1
[INFO] split_groups_created=...
[OK] Green Context stream created and kernel validated
```

Validated result on the remote Tesla T4 server:

```text
[INFO] device=0 name=Tesla T4 cc=7.5 sms=40
[INFO] requested_green_sms=8 priority=-1
[INFO] split_groups_created=1
[OK] Green Context stream created and kernel validated
```

## Environment Variables

| Variable | Default | Meaning |
|---|---:|---|
| `CUDA_HOME` | `/usr/local/cuda-12.8` | CUDA toolkit path used for headers |
| `CXX` | `g++` | Host C++ compiler |
| `GREEN_CONTEXT_DEVICE` | `0` | CUDA device index |
| `GREEN_CONTEXT_SMS` | `8` | Number of SMs requested for the Green Context |
| `GREEN_CONTEXT_PRIORITY` | `-1` | Green stream priority |
| `GREEN_CONTEXT_ELEMENTS` | `4096` | Number of uint32 elements written by the test kernel |
| `LIBCUDA_LDFLAGS` | `-lcuda` | Linker flags for the CUDA driver library |

If linking fails because `libcuda.so` is not on the default linker path, retry
with a system-specific path, for example:

```bash
LIBCUDA_LDFLAGS="-L/usr/lib/x86_64-linux-gnu -lcuda" \
./tools/green_context_smoke/build_and_run.sh
```

## Failure Interpretation

- Header check fails: the selected `CUDA_HOME` does not expose the CUDA 12.8
  Green Context APIs used by this smoke test.
- `cuDevSmResourceSplitByCount` fails: the driver/device rejected the requested
  SM split. Retry with a smaller `GREEN_CONTEXT_SMS`.
- `cuGreenCtxCreate` or `cuGreenCtxStreamCreate` fails: the driver exposes the
  API but does not support this device/runtime combination.
- Kernel validation fails: the stream was created, but work submission or memory
  operations did not execute correctly on the Green Context stream.

## Next Integration Step

Only after this passes should vLLM integration proceed:

1. Build the `vendor_v2` Green Context bridge:

   ```bash
   CUDA_HOME=/usr/local/cuda-12.8 \
   ./tools/green_context_smoke/build_vendor_v2_bridge.sh
   ```

2. Validate that PyTorch can wrap and use the decode/prefill Green Context
   stream pair. This is the important smoke test for P/D multiplexing because a
   decode-only stream can pass even when a two-way SM split is invalid:

   ```bash
   VLLM_OPERATOR_SELECTIVE_PD_GREEN_CONTEXT=1 \
   VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS=8 \
   VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS=32 \
   VLLM_OPERATOR_SELECTIVE_PD_GREEN_STRICT=1 \
   ./tools/green_context_smoke/test_vendor_v2_bridge.py
   ```

   Expected success:

   ```text
   [STATE] {'green_context_enabled': 1.0, 'green_context_active': 1.0, ...}
   [OK] vendor_v2 Green Context Python bridge validated
   ```

3. Run only with the default-off `vendor_v2` flags:

   ```bash
   VLLM_VENDOR_DIR=/home/noslab-gpu/zoneatten_gpu/vendor_v2 \
   VLLM_OPERATOR_SELECTIVE_PD=1 \
   VLLM_OPERATOR_SELECTIVE_PD_GREEN_CONTEXT=1 \
   VLLM_OPERATOR_SELECTIVE_PD_GREEN_DECODE_SMS=8 \
   VLLM_OPERATOR_SELECTIVE_PD_GREEN_PREFILL_SMS=32 \
   VLLM_OPERATOR_SELECTIVE_PD_GREEN_STRICT=1 \
   <experiment script>
   ```

4. If PyTorch cannot safely wrap the Green Context stream, set
   `VLLM_OPERATOR_SELECTIVE_PD_GREEN_STRICT=1` to fail fast and move the selected
   attention path to a direct C++/CUDA launch.

## vendor_v5 Dynamic Profile Bank

Build and validate the dynamic bridge separately from vendor_v2-v4:

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
./tools/green_context_smoke/build_vendor_v5_bridge.sh

VLLM_VENDOR_DIR=/home/noslab-gpu/zoneatten_gpu/vendor_v5 \
VLLM_PD_DYNAMIC_SM_TOTAL_SMS=40 \
VLLM_PD_DYNAMIC_SM_PROFILES="40/0,32/8,24/16,20/20,16/24,8/32,0/40" \
python tools/green_context_smoke/test_vendor_v5_dynamic_sm.py
```

The test fails unless the bridge exposes driver-reported SM counts and every
non-empty phase stream executes correctly. It also traverses all seven profiles
and requires six synchronized handoffs. Run the CPU-only controller and
import-free integration-contract tests independently with:

```bash
python tools/green_context_smoke/test_pd_sm_allocator.py
python tools/green_context_smoke/test_vendor_v5_contract.py
```
