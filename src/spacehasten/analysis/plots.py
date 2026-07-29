"""Canonical analysis plots rendered only from generated tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def render_plots(root: Path, dpi: int, hit_threshold: float) -> list[str]:
    import matplotlib.pyplot as plt

    plt.style.use("tableau-colorblind10")
    round_rows = read_csv(root / "round_metrics.csv")
    budget_rows = read_csv(root / "budget_curve.csv")
    cutoff_rows = read_csv(root / "cutoff_curve.csv")
    family_rows = read_csv(root / "family_metrics.csv")
    acquisition_rows = read_csv(root / "acquisition_metrics.csv")
    score_rows = read_csv(root / "score_distribution.csv")
    calibration_rows = read_csv(root / "calibration_curve.csv")
    plots: list[str] = []

    def save(name: str, figure: Any) -> None:
        for suffix in ("png", "pdf"):
            filename = f"{name}.{suffix}"
            figure.savefig(root / filename, dpi=dpi, bbox_inches="tight")
            plots.append(filename)
        plt.close(figure)

    figure, axis = plt.subplots()
    axis.plot(
        [int(row["selected_budget"]) for row in budget_rows],
        [int(row["cumulative_hits"]) for row in budget_rows],
    )
    axis.set(xlabel="Selected compounds", ylabel="Cumulative hits")
    save("cumulative_hits", figure)
    rounds = [int(row["round"]) for row in round_rows]
    figure, axis = plt.subplots()
    rates = [float_or_zero(row["hit_rate_selected"]) for row in round_rows]
    lower = [float_or_zero(row["hit_rate_selected_ci_low"]) for row in round_rows]
    upper = [float_or_zero(row["hit_rate_selected_ci_high"]) for row in round_rows]
    axis.errorbar(
        rounds,
        rates,
        yerr=[
            [rate - bound for rate, bound in zip(rates, lower, strict=True)],
            [bound - rate for bound, rate in zip(upper, rates, strict=True)],
        ],
        fmt="o-",
    )
    axis.set(xlabel="Round", ylabel="Hit rate (selected)")
    save("round_hit_rates", figure)
    figure, axis = plt.subplots()
    for round_id in sorted({int(row["round"]) for row in cutoff_rows}):
        subset = [row for row in cutoff_rows if int(row["round"]) == round_id]
        axis.plot(
            [float(row["cutoff"]) for row in subset],
            [float_or_zero(row["hit_rate_scored"]) for row in subset],
            marker="o",
            label=f"round {round_id}",
        )
    axis.axvline(
        hit_threshold,
        color="0.25",
        linestyle="--",
        linewidth=1,
        label=f"Primary cutoff ({hit_threshold:g})",
    )
    axis.legend()
    axis.set(
        xlabel="Docking-score cutoff (hit if score <= cutoff)",
        ylabel="Cumulative hit rate among scored selections",
        title="Cumulative yield sensitivity to the hit definition",
    )
    save("cutoff_sensitivity", figure)
    figure, axis = plt.subplots()
    for round_id in sorted({int(row["round"]) for row in score_rows}):
        subset = [row for row in score_rows if int(row["round"]) == round_id]
        axis.plot(
            [float(row["score"]) for row in subset],
            [float(row["quantile"]) for row in subset],
            label=f"round {round_id}",
        )
    axis.legend()
    axis.set(xlabel="Docking score", ylabel="Empirical cumulative fraction")
    save("score_ecdf", figure)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(
        [int(row["round"]) for row in family_rows],
        [float_or_zero(row["internal_diversity"]) for row in family_rows],
        marker="o",
    )
    axes[0].set(xlabel="Round", ylabel="Internal diversity")
    axes[1].plot(
        [int(row["round"]) for row in family_rows],
        [float_or_zero(row["typed_effective_q1"]) for row in family_rows],
        marker="o",
        label="typed scaffold q1",
    )
    axes[1].plot(
        [int(row["round"]) for row in family_rows],
        [float_or_zero(row["generic_effective_q1"]) for row in family_rows],
        marker="s",
        label="generic framework q1",
    )
    axes[1].legend(frameon=False)
    axes[1].set(xlabel="Round", ylabel="Effective families")
    figure.tight_layout()
    save("diversity_concentration", figure)
    if acquisition_rows and any(row.get("candidate_count") for row in acquisition_rows):
        figure, axes = plt.subplots(2, 2, figsize=(9, 6.5))
        acquisition_rounds = [int(row["round"]) for row in acquisition_rows]
        panels = (
            ("candidate_count", "Candidate pool", "o"),
            ("selected_cluster_largest_fraction", "Largest selected-cluster fraction", "s"),
            ("selected_cluster_effective_q1", "Effective selected clusters", "^"),
            ("cluster_penalty_nonzero_fraction", "Nonzero penalty fraction", "D"),
        )
        for axis, (column, label, marker) in zip(axes.flat, panels, strict=True):
            axis.plot(
                acquisition_rounds,
                [float_or_zero(row.get(column, "")) for row in acquisition_rows],
                marker=marker,
            )
            axis.set(xlabel="Round", ylabel=label)
        figure.tight_layout()
        save("acquisition_diagnostics", figure)
    if calibration_rows:
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1], color="0.4", linestyle="--", label="perfect calibration")
        for round_id in sorted({int(row["round"]) for row in calibration_rows}):
            subset = [row for row in calibration_rows if int(row["round"]) == round_id]
            line = axis.plot(
                [float(row["mean_predicted_probability"]) for row in subset],
                [float(row["observed_hit_fraction"]) for row in subset],
                label=f"round {round_id}",
            )[0]
            counts = [int(row["count"]) for row in subset]
            maximum = max(counts)
            axis.scatter(
                [float(row["mean_predicted_probability"]) for row in subset],
                [float(row["observed_hit_fraction"]) for row in subset],
                s=[20 + 80 * count / maximum for count in counts],
                color=line.get_color(),
            )
        axis.legend()
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Mean raw Gaussian hit probability",
            ylabel="Observed hit fraction",
            title="Raw Gaussian probability reliability",
        )
        axis.text(
            0.02,
            0.98,
            "Marker area proportional to bin count",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize="small",
        )
        save("calibration_reliability", figure)
    return plots


def float_or_zero(value: str | None) -> float:
    return float(value) if value else 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
