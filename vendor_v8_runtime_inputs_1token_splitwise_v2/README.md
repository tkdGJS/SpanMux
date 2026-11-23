# vendor_v8 one-token Splitwise runtime inputs v2

## Active input

- `slo/request_slo_policy.json`: 10K/25K/50K/100K native 40-SM bucket
  policy. Runtime uses P50 TTFT × 2 and P50 one-token TBT × 1.25.

## Dynamic-SM policy compatibility

`tbt_slo_feedback` does not consume a static overlap profile. It starts from
the configured Decode SM count, observes exact per-token TBT online, and moves
the split using the request's bucket SLO. Therefore this input set intentionally
does not invent `runtime_profiles/tbt_overlap_profile.json` from reconstructed
legacy partition data.

`tbt_slo_profiled` is different: it requires a direct `choice.token_ids`
overlap profile and must not use the reconstructed partition archive.

## Calibration provenance

The four current native baseline archives were validated by removing one
tokenizer-added BOS per SSE fragment and checking every fixed-output request
total. This is the strongest recoverable historical calibration, but it is not
a direct exact-ID baseline. New benchmark requests themselves still calculate
TBT from returned `choice.token_ids`.
