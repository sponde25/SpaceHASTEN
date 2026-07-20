#!/usr/bin/env python3
"""Plot observed-hit diversity effects against matched random seed samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#0072B2"
ORANGE = "#D55E00"

METRICS = [
    ("internal_diversity", "Internal diversity"),
    ("typed_scaffold_count", "Typed scaffold richness"),
    ("largest_typed_scaffold_fraction", "Typed scaffold dispersion"),
    ("generic_framework_count", "Generic framework richness"),
    ("largest_generic_framework_fraction", "Generic framework dispersion"),
    ("atlas_occupied_cluster_count", "Fixed-atlas cluster richness"),
    ("atlas_largest_cluster_fraction", "Fixed-atlas cluster dispersion"),
    ("atlas_normalized_cluster_entropy", "Fixed-atlas cluster evenness"),
]


def favorable_log2_ratio(value: float, reference: float, direction: str) -> float:
    if value <= 0 or reference <= 0:
        raise ValueError("diversity ratios require positive values")
    if direction == "higher":
        return float(np.log2(value / reference))
    return float(np.log2(reference / value))


def favorable_interval(values: dict[str, float | bool]) -> tuple[float, float]:
    mean = float(values["random_mean"])
    low = float(values["random_ci95_low"])
    high = float(values["random_ci95_high"])
    direction = str(values["diversity_direction"])
    interval = (
        favorable_log2_ratio(low, mean, direction),
        favorable_log2_ratio(high, mean, direction),
    )
    return min(interval), max(interval)


def plot(args: argparse.Namespace) -> None:
    with args.comparison.open("rt", encoding="utf-8") as handle:
        data = json.load(handle)
    comparison = data["comparison"]

    labels: list[str] = []
    observed: list[float] = []
    interval_low: list[float] = []
    interval_high: list[float] = []
    outside: list[bool] = []
    for key, label in METRICS:
        values = comparison[key]
        low, high = favorable_interval(values)
        labels.append(label)
        observed.append(
            favorable_log2_ratio(
                float(values["observed"]),
                float(values["random_mean"]),
                str(values["diversity_direction"]),
            )
        )
        interval_low.append(low)
        interval_high.append(high)
        outside.append(bool(values["outside_random_ci95"]))

    observed_array = np.asarray(observed)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.hlines(y, interval_low, interval_high, color="#888888", linewidth=3, alpha=0.7)
    ax.scatter(
        observed_array,
        y,
        s=55,
        c=np.where(observed_array >= 0, BLUE, ORANGE),
        edgecolors=np.where(outside, "black", "none"),
        linewidths=0.8,
        zorder=3,
    )
    ax.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("log2 diversity-favorable ratio to random-seed mean")
    ax.set_title(f"{args.label}: virtual hits versus matched random seed samples")
    ax.text(
        0.01,
        0.01,
        "Positive values favor observed hits; gray bars show the random-seed 95% interval",
        transform=ax.transAxes,
        fontsize=7,
        color="#444444",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output_dir / f"{args.output_stem}.png",
        dpi=args.dpi,
        bbox_inches="tight",
    )
    fig.savefig(
        args.output_dir / f"{args.output_stem}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="SpaceHASTEN run")
    parser.add_argument("--output-stem", default="05_hits_vs_random_seed_diversity")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


if __name__ == "__main__":
    plot(parse_args())
