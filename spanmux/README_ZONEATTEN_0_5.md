# ZoneAtten vendor_v4: Virtual Context 0.5.x

`vendor_v4` is the 0.5.x implementation line for virtual-context and
tiered-KV execution experiments.

## Source Lineage

```text
vendor_v3  ->  vendor_v4
0.4.x OpMux / GreenContext P-D multiplexing baseline
       -> 0.5.x logical context virtualization and tiered blockwise execution
```

`vendor_v3` must remain reproducible. Do not modify `vendor_v3` for 0.5.x
work. All 0.5.x runtime changes must land under `vendor_v4`.

## GreenContext Rule

GreenContext code was copied with the rest of `vendor_v3`:

```text
vendor_v4/vllm/model_executor/layers/green_context.py
vendor_v4/vllm/model_executor/layers/green_context_lib/green_context_bridge.cc
```

If 0.5.x needs GreenContext changes, patch only these `vendor_v4` files or add
new `vendor_v4`-local files. Do not mutate the `vendor_v2` or `vendor_v3`
GreenContext implementations.

## Activation

Use the normal vendor override:

```bash
VLLM_VENDOR_DIR=/home/yuhwa2323/zoneatten_gpu/vendor_v4 \
./start_vllm_causality_microbench.sh \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --port 8000
```

Until explicit `VLLM_VIRTUAL_CONTEXT_*` flags are added, `vendor_v4` should
behave like the copied `vendor_v3` baseline.

## Implementation Boundary

The 0.5.x goal is not ordinary KV offloading. The intended split is:

```text
Logical Context:
  complete causal KV history of a request

Physical Residency:
  HBM/DRAM/staging placement of logical KV pages

Attention Plan:
  prefill/decode phase plan that materializes and traverses pages under HBM
  and DMA/DRAM budgets
```

