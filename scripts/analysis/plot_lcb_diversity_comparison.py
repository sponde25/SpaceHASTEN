#!/usr/bin/env python3
"""Create final round, exclusive-hit, and potency-matched diversity figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#777777"
METRICS = (
    ("sampled_pair_internal_diversity", "Internal diversity"),
    ("typed_scaffold_richness_per_molecule", "Typed scaffolds per hit"),
    ("generic_framework_richness_per_molecule", "Generic frameworks per hit"),
    ("atlas_richness_per_molecule", "Common-atlas clusters per hit"),
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rounds(intervals: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4))
    for axis, (metric, label) in zip(axes.flat, METRICS, strict=True):
        subset = intervals[intervals.metric == metric].sort_values("round")
        for row in subset.itertuples():
            x = int(row.round)
            sampled_color = BLUE if row.sampled_run == "Greedy" else ORANGE
            fixed_color = BLUE if row.fixed_run == "Greedy" else ORANGE
            axis.errorbar(
                x - 0.08,
                row.reference_median,
                yerr=[
                    [row.reference_median - row.reference_q025],
                    [row.reference_q975 - row.reference_median],
                ],
                fmt="o",
                color=sampled_color,
                capsize=3,
                label=f"{row.sampled_run} matched" if x == 1 else None,
            )
            axis.scatter(
                x + 0.08,
                row.observed_value,
                marker="s",
                s=38,
                color=fixed_color,
                label=f"{row.fixed_run} observed" if x == 1 else None,
                zorder=3,
            )
            axis.text(
                x,
                max(row.reference_q975, row.observed_value),
                f"n={int(row.matched_n):,}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
        axis.set_xticks([1, 2], ["Round 1", "Round 2"])
        axis.set_ylabel(label)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=BLUE, label="Greedy"),
        plt.Line2D([], [], marker="s", linestyle="none", color=ORANGE, label="SVDKL-LCB"),
    ]
    axes[0, 0].legend(handles=handles, frameon=False)
    fig.suptitle("Round-specific diversity at matched hit counts", y=1.01)
    fig.tight_layout()
    save(fig, output_dir, "07_round_matched_diversity", dpi)


def plot_cohort_bars(
    frame: pd.DataFrame,
    output_dir: Path,
    stem: str,
    title: str,
    cohort_order: list[str],
    labels: list[str],
    colors: list[str],
    dpi: int,
) -> None:
    indexed = frame.set_index("cohort").loc[cohort_order]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.4))
    x = np.arange(len(indexed))
    for axis, (metric, label) in zip(axes.flat, METRICS, strict=True):
        values = indexed[metric].to_numpy(dtype=float)
        axis.bar(x, values, color=colors, width=0.68)
        axis.set_xticks(x, labels)
        axis.set_ylabel(label)
        for position, value in zip(x, values, strict=True):
            axis.text(position, value, f"{value:.3g}", ha="center", va="bottom", fontsize=6.5)
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    save(fig, output_dir, stem, dpi)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-intervals", type=Path, required=True)
    parser.add_argument("--exclusive", type=Path, required=True)
    parser.add_argument("--potency", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi must be positive")

    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rounds = pd.read_csv(args.round_intervals)
    exclusive = pd.read_csv(args.exclusive)
    potency = pd.read_csv(args.potency)
    plot_rounds(rounds, args.output_dir, args.dpi)
    plot_cohort_bars(
        exclusive,
        args.output_dir,
        "08_exclusive_shared_diversity",
        "Diversity of strategy-exclusive and shared hits",
        ["greedy_only", "lcb_only", "shared"],
        ["Greedy-only", "LCB-only", "Shared"],
        [BLUE, ORANGE, GREY],
        args.dpi,
    )
    potency = potency.assign(cohort=potency.run)
    plot_cohort_bars(
        potency,
        args.output_dir,
        "09_top50k_potency_matched_diversity",
        "Diversity among the top 50,000 docking scores",
        ["Greedy", "SVDKL-LCB"],
        ["Greedy", "SVDKL-LCB"],
        [BLUE, ORANGE],
        args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
