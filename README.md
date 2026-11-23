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
