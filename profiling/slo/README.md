# Native 40SM one-token performance and SLO

TBT is the pooled distribution of individual generated-token gaps. The first output token is excluded as TTFT.

| Context | Q | TTFT profile (ms) | TTFT SLO (ms) | 1-token TBT profile (ms) | TBT SLO (ms) | E2E profile (ms) | E2E SLO (ms) |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 10K | P50 | 9596.919 | 19193.838 | 22.8093 | 28.5116 | 103067.033 | 128833.792 |
| 10K | P90 | 9617.606 | 28852.818 | 24.3774 | 36.5661 | 103429.698 | 155144.547 |
| 10K | P99 | 9622.261 | 57733.564 | 24.9910 | 124.9551 | 103511.297 | 517556.486 |
| 25K | P50 | 55773.973 | 111547.945 | 35.9703 | 44.9629 | 203258.910 | 254073.637 |
| 25K | P90 | 55804.836 | 167414.508 | 37.4875 | 56.2313 | 203265.992 | 304898.987 |
| 25K | P99 | 55811.780 | 334870.681 | 38.1219 | 190.6097 | 203267.585 | 1016337.925 |
| 50K | P50 | 224945.132 | 449890.264 | 58.4886 | 73.1107 | 465107.997 | 581384.996 |
| 50K | P90 | 225234.514 | 675703.543 | 60.2067 | 90.3100 | 465132.374 | 697698.561 |
| 50K | P99 | 225299.626 | 1351797.753 | 61.1918 | 305.9592 | 465137.859 | 2325689.293 |
| 100K | P50 | 883266.727 | 1766533.453 | 100.4235 | 125.5294 | 1293920.080 | 1617400.100 |
| 100K | P90 | 884118.168 | 2652354.503 | 101.6926 | 152.5388 | 1294974.474 | 1942461.711 |
| 100K | P99 | 884309.742 | 5305858.451 | 102.7486 | 513.7430 | 1295211.713 | 6476058.565 |

## SLO multipliers

| Metric | P50 | P90 | P99 |
|---|---:|---:|---:|
| TTFT | 2x | 3x | 6x |
| TBT | 1.25x | 1.5x | 5x |
| E2E | 1.25x | 1.5x | 5x |

## Provenance

`choice_token_ids` is a direct exact-ID measurement. `legacy_archive_bos_reconstruction` removes one tokenizer-added BOS per SSE fragment after validating every request's fixed output-token total. Do not label the latter as a direct exact-ID measurement.

## Runtime policy

`request_slo_policy.json` selects the quantile-matched P50 contract: TTFT = native P50 × 2 and one-token TBT = native P50 × 1.25. Runtime TBT remains measured directly from `choice.token_ids`; the calibration provenance is recorded separately.
