#!/usr/bin/env python3
"""Generate prior-work SLO definitions without conflating their semantics.

Splitwise defines quantile-specific slowdown relative to a no-contention
request, so it can be recomputed from the local native 40-SM baseline.
DistServe and Llumnix take application/operator SLO targets as inputs;
their published values are emitted as reported references and are not
silently transferred to the local T4/Llama workload.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-zoneatten")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPLITWISE_FACTORS = {
    50: {"ttft": 2.0, "tbt": 1.25, "e2e": 1.25},
    90: {"ttft": 3.0, "tbt": 1.5, "e2e": 1.5},
    99: {"ttft": 6.0, "tbt": 5.0, "e2e": 5.0},
}
SPLITWISE_SOURCE = (
    "Splitwise Table VI: slowdown versus a no-contention DGX-A100 request"
)
DIST_SERVE_SOURCE = "DistServe Table 1, OSDI 2024"
LLUMNIX_SOURCE = "Llumnix deployment guide, Qwen3-32B on H20 defaults"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def write_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile requires observations")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


def load_native_samples(
    baseline_dir: Path,
) -> tuple[
    dict[int, list[float]],
    dict[int, list[float]],
    dict[int, list[float]],
    dict[int, list[float]],
]:
    request_path = baseline_dir / "request_runs.csv"
    token_path = baseline_dir / "token_timeline.csv"
    metadata_path = baseline_dir / "run_metadata.json"
    for path in (request_path, token_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError("native baseline must be completed")

    ttft: dict[int, list[float]] = defaultdict(list)
    request_mean_tbt: dict[int, list[float]] = defaultdict(list)
    e2e: dict[int, list[float]] = defaultdict(list)
    for row in read_csv(request_path):
        if row.get("phase") != "measurement":
            continue
        if row.get("status") != "success":
            raise ValueError("all measured native requests must succeed")
        context = int(row["context_target_tokens"])
        ttft[context].append(float(row["ttft_ms"]))
        request_mean_tbt[context].append(float(row["mean_tbt_ms"]))
        e2e[context].append(float(row["completion_latency_ms"]))

    pooled_token_tbt: dict[int, list[float]] = defaultdict(list)
    for row in read_csv(token_path):
        if row.get("phase") != "measurement":
            continue
        raw_gap = row.get("estimated_per_token_gap_ms")
        if raw_gap in ("", None):
            continue
        context = int(row["context_target_tokens"])
        chunk_tokens = max(1, int(row.get("chunk_tokens", 1) or 1))
        pooled_token_tbt[context].extend(
            [float(raw_gap)] * chunk_tokens
        )

    contexts = sorted(ttft)
    if not contexts:
        raise ValueError("native baseline has no measured requests")
    for context in contexts:
        if (
            not request_mean_tbt.get(context)
            or not pooled_token_tbt.get(context)
            or not e2e.get(context)
        ):
            raise ValueError(f"incomplete native context {context}")
    return (
        dict(ttft),
        dict(request_mean_tbt),
        dict(pooled_token_tbt),
        dict(e2e),
    )


SPLITWISE_FIELDS = [
    "context_tokens",
    "quantile",
    "measurement_count",
    "profiling_ttft_ms",
    "ttft_slowdown",
    "splitwise_ttft_slo_ms",
    "profiling_request_mean_tbt_ms",
    "profiling_pooled_token_gap_tbt_ms",
    "tbt_slowdown",
    "splitwise_request_mean_tbt_slo_ms",
    "splitwise_pooled_token_gap_tbt_slo_ms",
    "profiling_e2e_ms",
    "e2e_slowdown",
    "splitwise_e2e_slo_ms",
    "paper_alignment",
    "source",
]


def build_splitwise_rows(
    ttft: dict[int, list[float]],
    request_mean_tbt: dict[int, list[float]],
    pooled_token_tbt: dict[int, list[float]],
    e2e: dict[int, list[float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for context in sorted(ttft):
        for quantile, factors in SPLITWISE_FACTORS.items():
            ttft_value = percentile(ttft[context], quantile)
            request_tbt_value = percentile(
                request_mean_tbt[context], quantile
            )
            token_tbt_value = percentile(
                pooled_token_tbt[context], quantile
            )
            e2e_value = percentile(e2e[context], quantile)
            rows.append(
                {
                    "context_tokens": context,
                    "quantile": f"P{quantile}",
                    "measurement_count": len(ttft[context]),
                    "profiling_ttft_ms": round(ttft_value, 6),
                    "ttft_slowdown": factors["ttft"],
                    "splitwise_ttft_slo_ms": round(
                        ttft_value * factors["ttft"], 6
                    ),
                    "profiling_request_mean_tbt_ms": round(
                        request_tbt_value, 6
                    ),
                    "profiling_pooled_token_gap_tbt_ms": round(
                        token_tbt_value, 6
                    ),
                    "tbt_slowdown": factors["tbt"],
                    "splitwise_request_mean_tbt_slo_ms": round(
                        request_tbt_value * factors["tbt"], 6
                    ),
                    "splitwise_pooled_token_gap_tbt_slo_ms": round(
                        token_tbt_value * factors["tbt"], 6
                    ),
                    "profiling_e2e_ms": round(e2e_value, 6),
                    "e2e_slowdown": factors["e2e"],
                    "splitwise_e2e_slo_ms": round(
                        e2e_value * factors["e2e"], 6
                    ),
                    "paper_alignment": (
                        "request-level TBT is closer to the paper's "
                        "per-request metric; pooled token-gap TBT is a local "
                        "TBT-isolation adaptation"
                    ),
                    "source": SPLITWISE_SOURCE,
                }
            )
    return rows


DIST_SERVE_FIELDS = [
    "application",
    "model",
    "dataset",
    "base_ttft_slo_ms",
    "base_tpot_slo_ms",
    "slo_scale",
    "scaled_ttft_slo_ms",
    "scaled_tpot_slo_ms",
    "local_t4_recomputed",
    "source",
]


DIST_SERVE_TARGETS = (
    ("chatbot", "OPT-13B", "ShareGPT", 250.0, 100.0),
    ("chatbot", "OPT-66B", "ShareGPT", 2500.0, 150.0),
    ("chatbot", "OPT-175B", "ShareGPT", 4000.0, 200.0),
    ("code_completion", "OPT-66B", "HumanEval", 125.0, 200.0),
    ("summarization", "OPT-66B", "LongBench", 15000.0, 150.0),
)


def build_distserve_rows(scales: Sequence[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for application, model, dataset, ttft_ms, tpot_ms in DIST_SERVE_TARGETS:
        for scale in scales:
            rows.append(
                {
                    "application": application,
                    "model": model,
                    "dataset": dataset,
                    "base_ttft_slo_ms": ttft_ms,
                    "base_tpot_slo_ms": tpot_ms,
                    "slo_scale": scale,
                    "scaled_ttft_slo_ms": round(ttft_ms * scale, 6),
                    "scaled_tpot_slo_ms": round(tpot_ms * scale, 6),
                    "local_t4_recomputed": 0,
                    "source": DIST_SERVE_SOURCE,
                }
            )
    return rows


LLUMNIX_FIELDS = [
    "model",
    "hardware",
    "ttft_slo_ms",
    "tpot_slo_ms",
    "ttft_dispatch_threshold",
    "tpot_dispatch_threshold",
    "local_t4_recomputed",
    "source",
]


def build_llumnix_rows() -> list[dict[str, Any]]:
    ttft_slo = 6000.0
    tpot_slo = 50.0
    ttft_threshold = 0.9
    tpot_threshold = 0.9
    return [
        {
            "model": "Qwen3-32B",
            "hardware": "H20",
            "ttft_slo_ms": ttft_slo,
            "tpot_slo_ms": tpot_slo,
            "ttft_dispatch_threshold": ttft_threshold,
            "tpot_dispatch_threshold": tpot_threshold,
            "local_t4_recomputed": 0,
            "source": LLUMNIX_SOURCE,
        }
    ]


CURRENT_FIELDS = [
    "context_tokens",
    "ttft_reference_p95_ms",
    "ttft_slo_ms",
    "tbt_reference_request_p99_p95_ms",
    "tbt_slo_ms",
    "operational_headroom",
    "method",
]


def load_current_vendor_rows(policy_csv: Path) -> list[dict[str, Any]]:
    if not policy_csv.is_file():
        raise FileNotFoundError(policy_csv)
    rows: list[dict[str, Any]] = []
    for row in read_csv(policy_csv):
        rows.append(
            {
                "context_tokens": int(row["context_tokens"]),
                "ttft_reference_p95_ms": float(
                    row["ttft_reference_p95_ms"]
                ),
                "ttft_slo_ms": float(row["ttft_slo_ms"]),
                "tbt_reference_request_p99_p95_ms": float(
                    row["tbt_reference_request_p99_p95_ms"]
                ),
                "tbt_slo_ms": float(row["tbt_slo_ms"]),
                "operational_headroom": float(
                    row["operational_headroom"]
                ),
                "method": (
                    "OLS context envelope + maximum positive residual + "
                    "operational headroom"
                ),
            }
        )
    return rows


METHOD_FIELDS = [
    "method",
    "slo_origin",
    "uses_local_profile_to_set_slo",
    "quantiles",
    "formula",
    "local_t4_output",
    "important_limit",
]


def build_method_rows() -> list[dict[str, Any]]:
    return [
        {
            "method": "Splitwise",
            "slo_origin": "no-contention request latency",
            "uses_local_profile_to_set_slo": 1,
            "quantiles": "P50,P90,P99",
            "formula": "SLO(metric,q)=profile_no_contention(metric,q)*alpha(metric,q)",
            "local_t4_output": "recomputed",
            "important_limit": (
                "paper uses DGX-A100; local request-level and pooled-token "
                "TBT interpretations are both emitted"
            ),
        },
        {
            "method": "DistServe",
            "slo_origin": "empirical application service target",
            "uses_local_profile_to_set_slo": 0,
            "quantiles": "attainment target over requests",
            "formula": "SLO(scale)=Table1_application_target*scale",
            "local_t4_output": "not derivable without a chosen application target",
            "important_limit": (
                "reported OPT targets are not transferable to "
                "Llama-3.2-1B 25K..100K on T4"
            ),
        },
        {
            "method": "Llumnix",
            "slo_origin": "operator-configured TTFT/TPOT target",
            "uses_local_profile_to_set_slo": 0,
            "quantiles": "dispatch satisfaction threshold",
            "formula": "dispatch_budget=SLO*dispatch_threshold",
            "local_t4_output": "not derivable without configured TTFT/TPOT",
            "important_limit": (
                "profiling predicts schedulability; it does not generate "
                "the configured SLO"
            ),
        },
        {
            "method": "vendor_v8_current",
            "slo_origin": "native 40-SM regression envelope",
            "uses_local_profile_to_set_slo": 1,
            "quantiles": "TTFT request P95; request-P99-TBT P95",
            "formula": "SLO=headroom*(OLS(context)+max_positive_residual)",
            "local_t4_output": "already generated",
            "important_limit": "engineering policy, not an exact prior-work formula",
        },
    ]


def plot_splitwise(
    rows: Sequence[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    contexts = sorted({int(row["context_tokens"]) for row in rows})
    quantiles = ("P50", "P90", "P99")
    figure, axes = plt.subplots(
        3,
        len(contexts),
        figsize=(20, 13.5),
        squeeze=False,
    )
    x_values = np.arange(len(quantiles))
    width = 0.34
    for column, context in enumerate(contexts):
        selected = {
            str(row["quantile"]): row
            for row in rows
            if int(row["context_tokens"]) == context
        }
        ttft_profile = [
            float(selected[q]["profiling_ttft_ms"]) / 1000.0
            for q in quantiles
        ]
        ttft_slo = [
            float(selected[q]["splitwise_ttft_slo_ms"]) / 1000.0
            for q in quantiles
        ]
        tbt_profile = [
            float(selected[q]["profiling_request_mean_tbt_ms"])
            for q in quantiles
        ]
        tbt_slo = [
            float(selected[q]["splitwise_request_mean_tbt_slo_ms"])
            for q in quantiles
        ]
        e2e_profile = [
            float(selected[q]["profiling_e2e_ms"]) / 1000.0
            for q in quantiles
        ]
        e2e_slo = [
            float(selected[q]["splitwise_e2e_slo_ms"]) / 1000.0
            for q in quantiles
        ]
        for row_index, (profile, slo, ylabel) in enumerate(
            (
                (ttft_profile, ttft_slo, "TTFT (seconds)"),
                (tbt_profile, tbt_slo, "request-mean TBT (ms)"),
                (e2e_profile, e2e_slo, "E2E latency (seconds)"),
            )
        ):
            axis = axes[row_index][column]
            axis.bar(
                x_values - width / 2,
                profile,
                width,
                color=("#2563EB", "#059669", "#7C3AED")[row_index],
                label="No-contention profile",
            )
            axis.bar(
                x_values + width / 2,
                slo,
                width,
                color=("#93C5FD", "#6EE7B7", "#C4B5FD")[row_index],
                label="Splitwise SLO",
            )
            axis.set_xticks(x_values, quantiles)
            axis.set_ylabel(ylabel)
            axis.set_title(
                f"{context // 1000}K context — "
                f"{('TTFT', 'TBT', 'E2E')[row_index]}"
            )
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", color="#CBD5E1", alpha=0.65)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if column == 0:
                axis.legend(frameon=False, loc="upper left")
    figure.suptitle(
        "Splitwise Quantile Slowdown Applied to Native 40-SM T4 Profile",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "TTFT factors P50/P90/P99 = 2×/3×/6×; "
        "TBT and E2E factors = 1.25×/1.5×/5×.",
        ha="center",
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.95))
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def markdown_table(
    rows: Sequence[dict[str, Any]],
    fields: Sequence[str],
) -> str:
    header = "| " + " | ".join(fields) + " |"
    separator = "|" + "|".join("---" for _ in fields) + "|"
    body = [
        "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        for row in rows
    ]
    return "\n".join((header, separator, *body))


def write_report(
    output_dir: Path,
    splitwise_rows: Sequence[dict[str, Any]],
    distserve_rows: Sequence[dict[str, Any]],
    llumnix_rows: Sequence[dict[str, Any]],
    current_rows: Sequence[dict[str, Any]],
) -> None:
    splitwise_view = [
        {
            "ctx": f"{int(row['context_tokens']) // 1000}K",
            "q": row["quantile"],
            "TTFT profile ms": row["profiling_ttft_ms"],
            "TTFT SLO ms": row["splitwise_ttft_slo_ms"],
            "request-TBT profile ms": row[
                "profiling_request_mean_tbt_ms"
            ],
            "request-TBT SLO ms": row[
                "splitwise_request_mean_tbt_slo_ms"
            ],
            "token-gap TBT SLO ms": row[
                "splitwise_pooled_token_gap_tbt_slo_ms"
            ],
            "E2E profile ms": row["profiling_e2e_ms"],
            "E2E SLO ms": row["splitwise_e2e_slo_ms"],
        }
        for row in splitwise_rows
    ]
    distserve_base = [
        row for row in distserve_rows if float(row["slo_scale"]) == 1.0
    ]
    report = f"""# Prior-Work SLO Recalculation

## Correctness boundary

- Splitwise can be recomputed from the local native 40-SM no-contention
  measurements because the paper defines SLOs as quantile-specific slowdown.
- DistServe sets application SLOs empirically and then scales those targets.
  Its OPT values are emitted as reported, not transferred to the local model.
- Llumnix takes operator-configured SLOs and uses profile curves to predict
  schedulability. Its Qwen3-32B/H20 defaults are emitted as reported.

## Splitwise applied to the local T4 profile

{markdown_table(splitwise_view, tuple(splitwise_view[0]))}

## DistServe reported base targets

{markdown_table(distserve_base, DIST_SERVE_FIELDS[:7])}

## Llumnix reported defaults

{markdown_table(llumnix_rows, LLUMNIX_FIELDS[:8])}

## Current vendor_v8 regression SLO

{markdown_table(current_rows, CURRENT_FIELDS[:6])}

## Files

- `splitwise_local_t4_slo.csv` and `.json`
- `splitwise_local_t4_slo_bars.png`
- `distserve_reported_slo_scale.csv`
- `llumnix_reported_defaults.csv`
- `vendor_v8_current_regression_slo.csv`
- `method_comparison.csv`
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute applicable prior-work SLOs and emit non-transferable "
            "reported targets separately"
        )
    )
    parser.add_argument(
        "--baseline-results-dir",
        type=Path,
        default=ROOT / "native_40sm_profile_baseline_5bucket_v1_results",
    )
    parser.add_argument(
        "--current-policy-csv",
        type=Path,
        default=(
            ROOT
            / "vendor_v8_each_partition_profile_v1_results"
            / "slo"
            / "request_slo_policy.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "graphs" / "vendor_v8_prior_work_slo_comparison",
    )
    parser.add_argument(
        "--distserve-scales",
        type=float,
        nargs="+",
        default=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0),
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    scales = list(dict.fromkeys(float(value) for value in args.distserve_scales))
    if not scales or any(
        not math.isfinite(value) or value <= 0 for value in scales
    ):
        raise ValueError("DistServe scales must be positive finite values")

    baseline_dir = args.baseline_results_dir.expanduser().resolve()
    policy_csv = args.current_policy_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ttft, request_tbt, token_tbt, e2e = load_native_samples(baseline_dir)
    splitwise_rows = build_splitwise_rows(
        ttft,
        request_tbt,
        token_tbt,
        e2e,
    )
    distserve_rows = build_distserve_rows(scales)
    llumnix_rows = build_llumnix_rows()
    current_rows = load_current_vendor_rows(policy_csv)
    method_rows = build_method_rows()

    write_csv(
        output_dir / "splitwise_local_t4_slo.csv",
        splitwise_rows,
        SPLITWISE_FIELDS,
    )
    (output_dir / "splitwise_local_t4_slo.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "Splitwise Table VI slowdown",
                "source": SPLITWISE_SOURCE,
                "rows": splitwise_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "distserve_reported_slo_scale.csv",
        distserve_rows,
        DIST_SERVE_FIELDS,
    )
    write_csv(
        output_dir / "llumnix_reported_defaults.csv",
        llumnix_rows,
        LLUMNIX_FIELDS,
    )
    write_csv(
        output_dir / "vendor_v8_current_regression_slo.csv",
        current_rows,
        CURRENT_FIELDS,
    )
    write_csv(
        output_dir / "method_comparison.csv",
        method_rows,
        METHOD_FIELDS,
    )
    plot_splitwise(
        splitwise_rows,
        output_dir / "splitwise_local_t4_slo_bars.png",
        args.dpi,
    )
    write_report(
        output_dir,
        splitwise_rows,
        distserve_rows,
        llumnix_rows,
        current_rows,
    )
    print(
        "[DONE] "
        f"splitwise_rows={len(splitwise_rows)} "
        f"distserve_rows={len(distserve_rows)} "
        f"llumnix_rows={len(llumnix_rows)} "
        f"output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
