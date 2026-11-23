#!/usr/bin/env python3
"""Generate compact TBT tables and grayscale graphs after archive repair."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-zoneatten")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


QUANTILES = ("P50", "P90", "P99")
COLORS = ("#d9d9d9", "#969696", "#525252")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-40sm-summary", type=Path, required=True)
    parser.add_argument("--decode-20sm-summary", type=Path, required=True)
    parser.add_argument("--partition-quantiles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def grouped_bar(
    rows: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    contexts = [str(int(row["context_tokens"]) // 1000) + "K" for row in rows]
    x = np.arange(len(rows), dtype=float)
    width = 0.23
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for index, (quantile, color) in enumerate(zip(QUANTILES, COLORS)):
        values = [float(row[f"tbt_{quantile.lower()}_ms"]) for row in rows]
        axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=quantile,
            color=color,
            edgecolor="#333333",
            linewidth=0.7,
        )
    axis.set_xticks(x, contexts)
    axis.set_xlabel("Input context tokens")
    axis.set_ylabel("TBT (ms)")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25, linewidth=0.7)
    axis.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    native_rows: list[dict[str, Any]] = []
    for row in read_csv(args.native_40sm_summary):
        native_rows.append(
            {
                "context_tokens": int(row["context_target_tokens"]),
                "repeat_count": int(row["measurement_successes"]),
                "tbt_p50_ms": float(row["pooled_p50_tbt_ms"]),
                "tbt_p90_ms": float(row["pooled_p90_tbt_ms"]),
                "tbt_p99_ms": float(row["pooled_p99_tbt_ms"]),
                "measurement_source": "legacy_archive_bos_reconstruction",
            }
        )
    native_rows.sort(key=lambda row: row["context_tokens"])

    decode_rows: list[dict[str, Any]] = []
    for row in read_csv(args.decode_20sm_summary):
        decode_rows.append(
            {
                "context_tokens": int(row["input_tokens"]),
                "request": row["request"],
                "tbt_p50_ms": float(row["corrected_p50_tbt_ms"]),
                "tbt_p90_ms": float(row["corrected_p90_tbt_ms"]),
                "tbt_p99_ms": float(row["corrected_p99_tbt_ms"]),
                "measurement_source": "legacy_archive_bos_reconstruction",
            }
        )
    decode_rows.sort(key=lambda row: row["context_tokens"])

    partition_by_state: dict[
        tuple[str, int, int, str, int, int], dict[str, Any]
    ] = {}
    for row in read_csv(args.partition_quantiles):
        key = (
            row["profile_id"],
            int(row["decode_sms"]),
            int(row["prefill_sms"]),
            row["prefill_mode"],
            int(row["context_tokens"]),
            int(row["active_decode_requests"]),
        )
        state = partition_by_state.setdefault(
            key,
            {
                "profile_id": key[0],
                "decode_sms": key[1],
                "prefill_sms": key[2],
                "prefill_mode": key[3],
                "context_tokens": key[4],
                "active_decode_requests": key[5],
                "measurement_trial_count": int(
                    row["measurement_trial_count"]
                ),
                "overlap_token_gap_count": int(
                    row["overlap_token_gap_count"]
                ),
                "measurement_source": "legacy_archive_bos_reconstruction",
            },
        )
        quantile = row["quantile"].lower()
        state[f"ttft_{quantile}_ms"] = float(row["profile_ttft_ms"])
        state[f"overlap_tbt_{quantile}_ms"] = float(
            row["profile_overlap_tbt_ms"]
        )
    partition_rows = sorted(
        partition_by_state.values(),
        key=lambda row: (
            row["prefill_mode"],
            row["decode_sms"],
            row["context_tokens"],
            row["active_decode_requests"],
        ),
    )
    partition_case_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in partition_rows:
        partition_case_groups.setdefault(
            (row["profile_id"], row["prefill_mode"]), []
        ).append(row)
    partition_case_rows: list[dict[str, Any]] = []
    for (profile_id, mode), states in sorted(partition_case_groups.items()):
        summary: dict[str, Any] = {
            "profile_id": profile_id,
            "decode_sms": states[0]["decode_sms"],
            "prefill_sms": states[0]["prefill_sms"],
            "prefill_mode": mode,
            "measured_state_count": len(states),
            "context_tokens_min": min(row["context_tokens"] for row in states),
            "context_tokens_max": max(row["context_tokens"] for row in states),
            "active_decode_requests_min": min(
                row["active_decode_requests"] for row in states
            ),
            "active_decode_requests_max": max(
                row["active_decode_requests"] for row in states
            ),
            "measurement_source": "legacy_archive_bos_reconstruction",
        }
        for quantile in QUANTILES:
            field = f"overlap_tbt_{quantile.lower()}_ms"
            values = [float(row[field]) for row in states]
            summary[f"{field}_state_mean"] = round(
                statistics.fmean(values), 6
            )
            summary[f"{field}_state_min"] = round(min(values), 6)
            summary[f"{field}_state_max"] = round(max(values), 6)
        partition_case_rows.append(summary)

    native_fields = (
        "context_tokens",
        "repeat_count",
        "tbt_p50_ms",
        "tbt_p90_ms",
        "tbt_p99_ms",
        "measurement_source",
    )
    decode_fields = (
        "context_tokens",
        "request",
        "tbt_p50_ms",
        "tbt_p90_ms",
        "tbt_p99_ms",
        "measurement_source",
    )
    partition_fields = (
        "profile_id",
        "decode_sms",
        "prefill_sms",
        "prefill_mode",
        "context_tokens",
        "active_decode_requests",
        "measurement_trial_count",
        "overlap_token_gap_count",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "ttft_p99_ms",
        "overlap_tbt_p50_ms",
        "overlap_tbt_p90_ms",
        "overlap_tbt_p99_ms",
        "measurement_source",
    )
    partition_case_fields = (
        "profile_id",
        "decode_sms",
        "prefill_sms",
        "prefill_mode",
        "measured_state_count",
        "context_tokens_min",
        "context_tokens_max",
        "active_decode_requests_min",
        "active_decode_requests_max",
        "overlap_tbt_p50_ms_state_mean",
        "overlap_tbt_p50_ms_state_min",
        "overlap_tbt_p50_ms_state_max",
        "overlap_tbt_p90_ms_state_mean",
        "overlap_tbt_p90_ms_state_min",
        "overlap_tbt_p90_ms_state_max",
        "overlap_tbt_p99_ms_state_mean",
        "overlap_tbt_p99_ms_state_min",
        "overlap_tbt_p99_ms_state_max",
        "measurement_source",
    )
    write_csv(output_dir / "native_40sm_tbt_table.csv", native_fields, native_rows)
    write_csv(output_dir / "decode_only_20sm_tbt_table.csv", decode_fields, decode_rows)
    write_csv(
        output_dir / "partition_profile_tbt_table.csv",
        partition_fields,
        partition_rows,
    )
    write_csv(
        output_dir / "partition_profile_case_tbt_ranges.csv",
        partition_case_fields,
        partition_case_rows,
    )
    grouped_bar(
        native_rows,
        output_dir / "native_40sm_tbt_p50_p90_p99.png",
        title="Native Decode — 40 SM",
        dpi=args.dpi,
    )
    grouped_bar(
        decode_rows,
        output_dir / "decode_only_20sm_tbt_p50_p90_p99.png",
        title="Decode-only — 20 Decode SM",
        dpi=args.dpi,
    )
    print(f"[DONE] native_rows={len(native_rows)}")
    print(f"[DONE] decode_rows={len(decode_rows)}")
    print(f"[DONE] partition_rows={len(partition_rows)}")
    print(f"[DONE] output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
