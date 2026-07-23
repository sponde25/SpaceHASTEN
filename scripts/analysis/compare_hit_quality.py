#!/usr/bin/env python3
"""Compare virtual-hit quality across two or more acquisition runs."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.switch_backend("Agg")

LOGGER = logging.getLogger("compare_hit_quality")
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")
LINESTYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D", "v")
QUANTILES = (0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999)
TOP_KS = (100, 500, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000)


@dataclass(frozen=True)
class RunData:
    label: str
    database: Path
    virtual: pd.DataFrame
    seeds: pd.DataFrame
    iteration_selected: dict[int, int]
    acquisition_directory: Path | None
    cross_round_reused_ids: int

    @property
    def virtual_selected(self) -> int:
        return sum(self.iteration_selected.values())


def parse_assignments(
    values: list[str], converter: Callable[[str], Any]
) -> dict[str, Any]:
    assignments: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=VALUE, got {value!r}")
        label, raw = value.split("=", 1)
        if not label or not raw:
            raise ValueError(f"Expected non-empty LABEL=VALUE, got {value!r}")
        if label in assignments:
            raise ValueError(f"Duplicate label: {label}")
        assignments[label] = converter(raw)
    return assignments


def readonly_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def acquisition_iterations(directory: Path) -> tuple[dict[int, set[int]], set[int]]:
    files = sorted(
        directory.glob("iter*/acquisition.csv"),
        key=lambda path: int(path.parent.name.removeprefix("iter")),
    )
    if not files:
        raise ValueError(f"No iter*/acquisition.csv files under {directory}")

    result: dict[int, set[int]] = {}
    all_ids: set[int] = set()
    reused_ids: set[int] = set()
    for path in files:
        iteration = int(path.parent.name.removeprefix("iter"))
        frame = pd.read_csv(path, usecols=["spacehastenid"])
        if frame["spacehastenid"].isna().any():
            raise ValueError(f"Missing spacehastenid values in {path}")
        ids = set(frame["spacehastenid"].astype(np.int64))
        if len(ids) != len(frame):
            raise ValueError(f"Duplicate spacehastenid values in {path}")
        reused_ids.update(all_ids.intersection(ids))
        result[iteration] = ids
        all_ids.update(ids)
    return result, reused_ids


def load_run(
    label: str,
    database: Path,
    acquisition_directory: Path | None,
) -> RunData:
    started = time.monotonic()
    database = database.resolve()
    if not database.is_file():
        raise FileNotFoundError(database)

    with readonly_connection(database) as connection:
        database_selected = {
            int(iteration): int(count)
            for iteration, count in connection.execute(
                "SELECT dock_iteration, COUNT(*) FROM data WHERE dock_iteration > 0 "
                "GROUP BY dock_iteration ORDER BY dock_iteration"
            ).fetchall()
        }
        virtual = pd.read_sql_query(
            "SELECT spacehastenid, reghash, dock_score, dock_iteration FROM data "
            "WHERE dock_iteration > 0 AND dock_score IS NOT NULL",
            connection,
        )
        seeds = pd.read_sql_query(
            "SELECT reghash, dock_score FROM data "
            "WHERE dock_iteration = 0 AND dock_score IS NOT NULL",
            connection,
        )

    if virtual.empty:
        raise ValueError(f"{label}: no scored virtual compounds")
    if virtual["reghash"].isna().any() or (virtual["reghash"].str.strip() == "").any():
        raise ValueError(f"{label}: scored virtual cohort has missing reghashes")
    if virtual["reghash"].duplicated().any():
        raise ValueError(f"{label}: scored virtual cohort has duplicate reghashes")
    if virtual["spacehastenid"].duplicated().any():
        raise ValueError(f"{label}: scored virtual cohort has duplicate IDs")
    if not np.isfinite(virtual["dock_score"].to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: scored virtual cohort has non-finite scores")
    if seeds["reghash"].isna().any() or seeds["reghash"].duplicated().any():
        raise ValueError(f"{label}: scored seed cohort has missing or duplicate reghashes")
    if not np.isfinite(seeds["dock_score"].to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: scored seed cohort has non-finite scores")

    selected = database_selected
    cross_round_reused_ids = 0
    if acquisition_directory is not None:
        acquisition_directory = acquisition_directory.resolve()
        selected_ids, reused_ids = acquisition_iterations(acquisition_directory)
        cross_round_reused_ids = len(reused_ids)
        selected = {iteration: len(ids) for iteration, ids in selected_ids.items()}
        scored_ids = {
            iteration: set(
                virtual.loc[
                    virtual["dock_iteration"] == iteration, "spacehastenid"
                ].astype(np.int64)
            )
            for iteration in selected_ids
        }
        for iteration, ids in selected_ids.items():
            unexpected = scored_ids[iteration].difference(ids)
            if unexpected:
                raise ValueError(
                    f"{label} round {iteration}: {len(unexpected)} scored IDs were not selected"
                )
            scored_count = len(scored_ids[iteration])
            if scored_count > len(ids):
                raise ValueError(f"{label} round {iteration}: scored count exceeds selected")
        scored_iteration = virtual.set_index("spacehastenid")["dock_iteration"]
        for compound_id in reused_ids.intersection(scored_iteration.index):
            selected_rounds = [
                iteration
                for iteration, ids in selected_ids.items()
                if compound_id in ids
            ]
            observed_round = int(scored_iteration.loc[compound_id])
            if observed_round != max(selected_rounds):
                raise ValueError(
                    f"{label}: reused ID {compound_id} was scored in round "
                    f"{observed_round} before its final selection event"
                )
        if cross_round_reused_ids:
            LOGGER.info(
                "%s: %d unscored IDs were selected in more than one round",
                label,
                cross_round_reused_ids,
            )

    iterations = tuple(sorted(int(value) for value in virtual["dock_iteration"].unique()))
    if tuple(sorted(selected)) != iterations:
        raise ValueError(
            f"{label}: selected rounds {tuple(sorted(selected))} do not match scored rounds "
            f"{iterations}"
        )

    LOGGER.info(
        "Loaded %s: %d selected, %d scored virtual rows, %d scored seeds in %.1f s",
        label,
        sum(selected.values()),
        len(virtual),
        len(seeds),
        time.monotonic() - started,
    )
    return RunData(
        label=label,
        database=database,
        virtual=virtual,
        seeds=seeds,
        iteration_selected=selected,
        acquisition_directory=acquisition_directory,
        cross_round_reused_ids=cross_round_reused_ids,
    )


def wilson_interval(hits: int, total: int) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    z = 1.959963984540054
    proportion = hits / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z**2 / (4 * total**2)
        )
        / denominator
    )
    return center - half_width, center + half_width


def score_summary(scores: np.ndarray) -> dict[str, float]:
    values = np.asarray(scores, dtype=float)
    result: dict[str, float] = {
        "mean_score": float(np.mean(values)),
        "sd_score": float(np.std(values, ddof=1)),
        "min_score": float(np.min(values)),
        "max_score": float(np.max(values)),
        "median_score": float(np.median(values)),
        "iqr_score": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
    }
    result.update(
        {f"q_{quantile * 100:g}": float(np.quantile(values, quantile)) for quantile in QUANTILES}
    )
    return result


def summary_row(
    run: RunData,
    cohort: pd.DataFrame,
    iteration: str,
    selected: int,
    cutoff: float,
) -> dict[str, Any]:
    scored = len(cohort)
    hits = int((cohort["dock_score"] <= cutoff).sum())
    ci_low, ci_high = wilson_interval(hits, scored)
    return {
        "run": run.label,
        "iteration": iteration,
        "selected": selected,
        "scored": scored,
        "missing_score_count": selected - scored,
        "missing_score_rate": (selected - scored) / selected if selected else math.nan,
        "unique_reghashes": int(cohort["reghash"].nunique()),
        "hit_count": hits,
        "hit_rate_scored": hits / scored if scored else math.nan,
        "hit_rate_selected": hits / selected if selected else math.nan,
        "wilson95_low_scored": ci_low,
        "wilson95_high_scored": ci_high,
        **score_summary(cohort["dock_score"].to_numpy(dtype=float)),
    }


def rate_comparison(
    reference_hits: int,
    reference_n: int,
    comparison_hits: int,
    comparison_n: int,
) -> dict[str, float]:
    reference_rate = reference_hits / reference_n
    comparison_rate = comparison_hits / comparison_n
    difference = comparison_rate - reference_rate
    standard_error = math.sqrt(
        reference_rate * (1 - reference_rate) / reference_n
        + comparison_rate * (1 - comparison_rate) / comparison_n
    )
    odds_ratio = (
        comparison_hits / (comparison_n - comparison_hits)
    ) / (reference_hits / (reference_n - reference_hits))
    return {
        "reference_hit_rate": reference_rate,
        "comparison_hit_rate": comparison_rate,
        "absolute_difference_comparison_minus_reference_pp": 100 * difference,
        "difference95_low_pp": 100 * (difference - 1.959963984540054 * standard_error),
        "difference95_high_pp": 100 * (difference + 1.959963984540054 * standard_error),
        "relative_risk_comparison_over_reference": comparison_rate / reference_rate,
        "odds_ratio_comparison_over_reference": odds_ratio,
    }


def score_comparison(reference: np.ndarray, comparison: np.ndarray) -> dict[str, float]:
    n_reference, n_comparison = len(reference), len(comparison)
    u_comparison_greater = float(
        stats.mannwhitneyu(comparison, reference, alternative="two-sided").statistic
    )
    probability_comparison_better = 1.0 - u_comparison_greater / (
        n_comparison * n_reference
    )
    return {
        "ks_statistic": float(
            stats.ks_2samp(comparison, reference, method="asymp").statistic
        ),
        "wasserstein_distance": float(
            stats.wasserstein_distance(comparison, reference)
        ),
        "common_language_probability_comparison_better": probability_comparison_better,
        "rank_biserial_cliffs_delta_comparison_better": (
            2 * probability_comparison_better - 1
        ),
        "mean_difference_comparison_minus_reference": float(
            np.mean(comparison) - np.mean(reference)
        ),
        "median_difference_comparison_minus_reference": float(
            np.median(comparison) - np.median(reference)
        ),
    }


def cutoff_table(runs: list[RunData], cutoff: float) -> pd.DataFrame:
    thresholds = sorted(set(np.round(np.arange(-12, -7.999, 0.25), 2)).union({cutoff}))
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for run in runs:
            cohorts: list[tuple[str, pd.DataFrame, int]] = [
                ("overall", run.virtual, run.virtual_selected)
            ]
            cohorts.extend(
                (
                    str(iteration),
                    group,
                    run.iteration_selected[int(iteration)],
                )
                for iteration, group in run.virtual.groupby("dock_iteration")
            )
            for iteration, cohort, selected in cohorts:
                hits = int((cohort["dock_score"] <= threshold).sum())
                rows.append(
                    {
                        "run": run.label,
                        "iteration": iteration,
                        "cutoff": threshold,
                        "selected": selected,
                        "scored": len(cohort),
                        "hit_count": hits,
                        "hit_rate_scored": hits / len(cohort),
                        "hit_rate_selected": hits / selected,
                    }
                )
    return pd.DataFrame(rows)


def top_k_table(runs: list[RunData], cutoff: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        sorted_scores = np.sort(run.virtual["dock_score"].to_numpy(dtype=float))
        for k in (*TOP_KS, len(sorted_scores)):
            if k > len(sorted_scores):
                continue
            top = sorted_scores[:k]
            rows.append(
                {
                    "run": run.label,
                    "k": k,
                    "is_full_scored_cohort": k == len(sorted_scores),
                    "kth_worst_score": float(top[-1]),
                    "mean_score": float(np.mean(top)),
                    "median_score": float(np.median(top)),
                    "hit_count": int((top <= cutoff).sum()),
                    "hit_rate": float(np.mean(top <= cutoff)),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(["run", "k"])


def overlap_rows(
    reference: RunData,
    comparison: RunData,
    cutoff: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_scores = reference.virtual.set_index("reghash")["dock_score"]
    comparison_scores = comparison.virtual.set_index("reghash")["dock_score"]
    shared = reference_scores.index.intersection(comparison_scores.index)
    scored_union = reference_scores.index.union(comparison_scores.index)
    reference_hits = reference_scores[reference_scores <= cutoff].index
    comparison_hits = comparison_scores[comparison_scores <= cutoff].index
    shared_hits = reference_hits.intersection(comparison_hits)
    hit_union = reference_hits.union(comparison_hits)
    differences = (
        reference_scores.loc[shared] - comparison_scores.loc[shared]
    ).abs().to_numpy()
    label = f"{comparison.label} vs {reference.label}"
    agreement: dict[str, Any] = {
        "comparison": label,
        "reference_run": reference.label,
        "comparison_run": comparison.label,
        "shared_scored_count": len(shared),
        "scored_union_count": len(scored_union),
        "scored_jaccard": len(shared) / len(scored_union),
        "reference_only_scored": len(reference_scores.index.difference(comparison_scores.index)),
        "comparison_only_scored": len(comparison_scores.index.difference(reference_scores.index)),
        "shared_hit_count": len(shared_hits),
        "hit_union_count": len(hit_union),
        "hit_jaccard": len(shared_hits) / len(hit_union),
        "reference_only_hits": len(reference_hits.difference(comparison_hits)),
        "comparison_only_hits": len(comparison_hits.difference(reference_hits)),
        "score_exact_match_count": int(np.sum(differences == 0)),
        "score_tolerance_1e-8_count": int(np.sum(differences <= 1e-8)),
        "score_abs_difference_mean": float(np.mean(differences)),
        "score_abs_difference_median": float(np.median(differences)),
        "score_abs_difference_max": float(np.max(differences)),
        "score_pearson_r": float(
            stats.pearsonr(reference_scores.loc[shared], comparison_scores.loc[shared]).statistic
        ),
        "score_spearman_r": float(
            stats.spearmanr(reference_scores.loc[shared], comparison_scores.loc[shared]).statistic
        ),
    }
    rows = [
        {
            "comparison": label,
            "reference_run": reference.label,
            "comparison_run": comparison.label,
            "metric": key,
            "value": value,
        }
        for key, value in agreement.items()
        if key not in {"comparison", "reference_run", "comparison_run"}
    ]
    for k in TOP_KS:
        reference_top = set(
            reference_scores.nsmallest(min(k, len(reference_scores))).index
        )
        comparison_top = set(
            comparison_scores.nsmallest(min(k, len(comparison_scores))).index
        )
        overlap = len(reference_top.intersection(comparison_top))
        union = len(reference_top.union(comparison_top))
        rows.extend(
            [
                {
                    "comparison": label,
                    "reference_run": reference.label,
                    "comparison_run": comparison.label,
                    "metric": f"top_{k}_overlap_count",
                    "value": overlap,
                },
                {
                    "comparison": label,
                    "reference_run": reference.label,
                    "comparison_run": comparison.label,
                    "metric": f"top_{k}_jaccard",
                    "value": overlap / union,
                },
            ]
        )
    return rows, agreement


def pairwise_tables(
    runs: list[RunData], cutoff: float
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    overlap_records: list[dict[str, Any]] = []
    statistical_records: list[dict[str, Any]] = []
    overlap_summaries: list[dict[str, Any]] = []
    for reference, comparison in combinations(runs, 2):
        rows, summary = overlap_rows(reference, comparison, cutoff)
        overlap_records.extend(rows)
        overlap_summaries.append(summary)
        reference_scores = reference.virtual["dock_score"].to_numpy(dtype=float)
        comparison_scores = comparison.virtual["dock_score"].to_numpy(dtype=float)
        reference_hits = int(np.sum(reference_scores <= cutoff))
        comparison_hits = int(np.sum(comparison_scores <= cutoff))
        statistical_records.append(
            {
                "comparison": f"{comparison.label} vs {reference.label}",
                "reference_run": reference.label,
                "comparison_run": comparison.label,
                **score_comparison(reference_scores, comparison_scores),
                **rate_comparison(
                    reference_hits,
                    len(reference_scores),
                    comparison_hits,
                    len(comparison_scores),
                ),
            }
        )
    return (
        pd.DataFrame(overlap_records),
        pd.DataFrame(statistical_records),
        overlap_summaries,
    )


def stage_hours(path: Path) -> float:
    frame = pd.read_csv(path)
    classic_columns = {"category", "hours", "minutes", "seconds"}
    if classic_columns.issubset(frame.columns):
        total_rows = frame[frame["category"].astype(str).str.upper() == "TOTAL"]
        if len(total_rows) != 1:
            raise ValueError(f"Timing CSV must contain exactly one TOTAL row: {path}")
        total = total_rows.iloc[0]
        return float(total["hours"] + total["minutes"] / 60 + total["seconds"] / 3600)

    elapsed_columns = {"category", "elapsed_seconds"}
    if elapsed_columns.issubset(frame.columns):
        total_rows = frame[
            frame["category"].astype(str).str.lower() == "measured_stage_total"
        ]
        if len(total_rows) != 1:
            raise ValueError(
                f"Timing CSV must contain one measured_stage_total row: {path}"
            )
        return float(total_rows.iloc[0]["elapsed_seconds"] / 3600)
    raise ValueError(f"Unsupported timing CSV schema: {path}")


def operational_table(
    runs: list[RunData], cutoff: float, timing_paths: dict[str, Path]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        timing = timing_paths[run.label].resolve()
        hits = int((run.virtual["dock_score"] <= cutoff).sum())
        hours = stage_hours(timing)
        rows.append(
            {
                "run": run.label,
                "selected": run.virtual_selected,
                "scored": len(run.virtual),
                "missing_docking_scores": run.virtual_selected - len(run.virtual),
                "hit_count": hits,
                "hits_per_100k_selected": 100_000 * hits / run.virtual_selected,
                "measured_stage_hours": hours,
                "hits_per_measured_stage_hour": hits / hours,
                "timing_csv": str(timing),
            }
        )
    return pd.DataFrame(rows)


def validate_seed_reference(runs: list[RunData], cutoff: float) -> dict[str, Any]:
    reference = runs[0]
    reference_scores = reference.seeds.set_index("reghash")["dock_score"].sort_index()
    per_run: list[dict[str, Any]] = []
    for run in runs:
        scores = run.seeds.set_index("reghash")["dock_score"].sort_index()
        sets_equal = scores.index.equals(reference_scores.index)
        scores_equal = sets_equal and np.array_equal(
            scores.to_numpy(dtype=float), reference_scores.to_numpy(dtype=float)
        )
        per_run.append(
            {
                "run": run.label,
                "scored_seed_count": len(scores),
                "seed_hit_count": int((scores <= cutoff).sum()),
                "seed_hash_set_matches_reference": sets_equal,
                "seed_scores_match_reference": scores_equal,
            }
        )
        if not sets_equal or not scores_equal:
            raise ValueError(f"{run.label}: seed reference differs from {reference.label}")

    seed_hits = per_run[0]["seed_hit_count"]
    seed_count = per_run[0]["scored_seed_count"]
    return {
        "reference_run": reference.label,
        "shared_seed_count": seed_count,
        "seed_hit_count": seed_hits,
        "seed_hit_rate": seed_hits / seed_count,
        "all_seed_hash_sets_equal": True,
        "all_seed_scores_exactly_agree": True,
        "runs": per_run,
        "virtual_enrichment_over_seed": {
            run.label: float(
                np.mean(run.virtual["dock_score"].to_numpy(dtype=float) <= cutoff)
                / (seed_hits / seed_count)
            )
            for run in runs
        },
    }


def configure_figure_style() -> None:
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, stem: Path, dpi: int) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def make_figures(
    output_dir: Path,
    runs: list[RunData],
    cutoffs: pd.DataFrame,
    top_k: pd.DataFrame,
    cutoff: float,
    dpi: int,
) -> None:
    configure_figure_style()
    styles = {
        run.label: (
            COLORS[index % len(COLORS)],
            LINESTYLES[index % len(LINESTYLES)],
            MARKERS[index % len(MARKERS)],
        )
        for index, run in enumerate(runs)
    }

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    all_scores = np.concatenate(
        [run.virtual["dock_score"].to_numpy(dtype=float) for run in runs]
    )
    low, high = np.quantile(all_scores, [0.001, 0.999])
    bins = np.linspace(low, high, 80)
    for run in runs:
        color, linestyle, _ = styles[run.label]
        scores = run.virtual["dock_score"].to_numpy(dtype=float)
        visible = scores[(scores >= low) & (scores <= high)]
        axes[0].hist(
            visible,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            color=color,
            linestyle=linestyle,
            label=run.label,
        )
        ordered = np.sort(scores)
        axes[1].step(
            ordered,
            np.arange(1, len(ordered) + 1) / len(ordered),
            where="post",
            color=color,
            linestyle=linestyle,
            label=run.label,
        )
    for axis in axes:
        axis.axvline(cutoff, color="black", linestyle=":", linewidth=1)
        axis.set_xlabel("Docking score (lower is better)")
        axis.legend(frameon=False)
    axes[0].set_ylabel("Normalized density")
    axes[0].set_title("Central 99.8% score range shown")
    axes[1].set_ylabel("ECDF")
    axes[1].set_title("Full scored cohorts")
    figure.tight_layout()
    save_figure(figure, output_dir / "01_score_distributions", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for run in runs:
        color, linestyle, marker = styles[run.label]
        subset = cutoffs[
            (cutoffs["run"] == run.label) & (cutoffs["iteration"] == "overall")
        ]
        axes[0].plot(
            subset["cutoff"],
            100 * subset["hit_rate_scored"],
            color=color,
            linestyle=linestyle,
            label=run.label,
        )
        primary = subset[np.isclose(subset["cutoff"], cutoff)].iloc[0]
        axes[0].scatter(
            [cutoff], [100 * primary["hit_rate_scored"]], color=color, zorder=3
        )
        for iteration in sorted(run.iteration_selected):
            row = cutoffs[
                (cutoffs["run"] == run.label)
                & (cutoffs["iteration"] == str(iteration))
                & np.isclose(cutoffs["cutoff"], cutoff)
            ].iloc[0]
            low_ci, high_ci = wilson_interval(int(row["hit_count"]), int(row["scored"]))
            x_value = f"{run.label}\nround {iteration}"
            axes[1].errorbar(
                x_value,
                100 * row["hit_rate_scored"],
                yerr=[
                    [100 * (row["hit_rate_scored"] - low_ci)],
                    [100 * (high_ci - row["hit_rate_scored"])],
                ],
                fmt=marker,
                color=color,
                capsize=3,
            )
    axes[0].axvline(cutoff, color="black", linestyle=":", linewidth=1)
    axes[0].set(
        xlabel="Hit cutoff (docking score; lower is stricter)",
        ylabel="Hit rate among scored compounds (%)",
    )
    axes[0].legend(frameon=False)
    axes[1].set(
        ylabel=f"Hit rate at {cutoff:g} (%)",
        title="Error bars: Wilson 95% CI",
    )
    axes[1].tick_params(axis="x", labelrotation=25)
    figure.tight_layout()
    save_figure(figure, output_dir / "02_hit_yield", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for run in runs:
        color, linestyle, marker = styles[run.label]
        subset = top_k[
            (top_k["run"] == run.label) & (~top_k["is_full_scored_cohort"])
        ].sort_values("k")
        axes[0].plot(
            subset["k"],
            subset["kth_worst_score"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=3,
            label=run.label,
        )
        axes[1].plot(
            subset["k"],
            subset["mean_score"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=3,
            label=run.label,
        )
    for axis, ylabel in zip(
        axes,
        ("Kth (worst) docking score", "Mean docking score among top k"),
        strict=True,
    ):
        axis.set_xscale("log")
        axis.set(xlabel="k (log scale)", ylabel=ylabel)
        axis.axhline(cutoff, color="black", linestyle=":", linewidth=1)
        axis.legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, output_dir / "03_top_k_quality", dpi)


def validate_outputs(output_dir: Path, expected_files: list[str]) -> None:
    for filename in expected_files:
        path = output_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty output: {path}")
    table_names = (
        "run_summary.csv",
        "iteration_summary.csv",
        "score_quantiles.csv",
        "cutoff_sensitivity.csv",
        "top_k_summary.csv",
        "overlap_summary.csv",
        "statistical_comparison.csv",
        "operational_efficiency.csv",
    )
    for filename in table_names:
        frame = pd.read_csv(output_dir / filename)
        if frame.empty or len(frame.columns) == 0:
            raise ValueError(f"Invalid CSV schema: {filename}")
    LOGGER.info("Validated %d output artifacts", len(expected_files))


def calculate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    run_paths = parse_assignments(args.run, Path)
    timing_paths = parse_assignments(args.timing, Path)
    acquisition_paths = parse_assignments(args.acquisition_dir, Path)
    expected_hits = parse_assignments(args.expected_hits, int)
    if len(run_paths) < 2:
        raise ValueError("At least two --run arguments are required")
    if set(timing_paths) != set(run_paths):
        raise ValueError("--timing labels must exactly match --run labels")
    if not set(acquisition_paths).issubset(run_paths):
        raise ValueError("--acquisition-dir includes an unknown run label")
    if not set(expected_hits).issubset(run_paths):
        raise ValueError("--expected-hits includes an unknown run label")

    runs = [
        load_run(label, path, acquisition_paths.get(label))
        for label, path in run_paths.items()
    ]
    run_rows = [
        summary_row(run, run.virtual, "overall", run.virtual_selected, args.cutoff)
        for run in runs
    ]
    iteration_rows = [
        summary_row(
            run,
            group,
            str(iteration),
            run.iteration_selected[int(iteration)],
            args.cutoff,
        )
        for run in runs
        for iteration, group in run.virtual.groupby("dock_iteration")
    ]
    run_summary = pd.DataFrame(run_rows)
    iteration_summary = pd.DataFrame(iteration_rows)

    for run in runs:
        round_rows = iteration_summary[iteration_summary["run"] == run.label]
        if int(round_rows["hit_count"].sum()) != int(
            run_summary.loc[run_summary["run"] == run.label, "hit_count"].iloc[0]
        ):
            raise ValueError(f"{run.label}: iteration hits do not sum to overall")
        if int(round_rows["selected"].sum()) != run.virtual_selected:
            raise ValueError(f"{run.label}: iteration selections do not sum to overall")

    observed_hits = {
        str(row["run"]): int(row["hit_count"]) for row in run_rows
    }
    for label, expected in expected_hits.items():
        if observed_hits[label] != expected:
            raise ValueError(
                f"{label}: expected {expected} hits, observed {observed_hits[label]}"
            )

    quantiles = run_summary[
        ["run", *[f"q_{quantile * 100:g}" for quantile in QUANTILES]]
    ].melt("run", var_name="quantile", value_name="dock_score")
    cutoffs = cutoff_table(runs, args.cutoff)
    top_k = top_k_table(runs, args.cutoff)
    overlap, statistics, overlap_summaries = pairwise_tables(runs, args.cutoff)
    operational = operational_table(runs, args.cutoff, timing_paths)
    seed_reference = validate_seed_reference(runs, args.cutoff)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "run_summary.csv": run_summary,
        "iteration_summary.csv": iteration_summary,
        "score_quantiles.csv": quantiles,
        "cutoff_sensitivity.csv": cutoffs,
        "top_k_summary.csv": top_k,
        "overlap_summary.csv": overlap,
        "statistical_comparison.csv": statistics,
        "operational_efficiency.csv": operational,
    }
    for filename, frame in tables.items():
        frame.to_csv(output_dir / filename, index=False)

    make_figures(output_dir, runs, cutoffs, top_k, args.cutoff, args.dpi)
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "runs": {label: str(path.resolve()) for label, path in run_paths.items()},
            "timings": {
                label: str(path.resolve()) for label, path in timing_paths.items()
            },
            "acquisition_directories": {
                label: str(path.resolve())
                for label, path in acquisition_paths.items()
            },
            "cutoff": args.cutoff,
        },
        "validation": {
            "expected_primary_hits": expected_hits,
            "observed_primary_hits": observed_hits,
            "iteration_hit_sums_equal_overall": True,
            "iteration_selection_sums_equal_overall": True,
            "scored_virtual_reghashes_unique": True,
            "seed_reference_equal_across_runs": True,
            "selection_events": {
                run.label: {
                    "selected_by_iteration": run.iteration_selected,
                    "cross_round_reused_ids": run.cross_round_reused_ids,
                }
                for run in runs
            },
        },
        "run_summary": run_rows,
        "score_comparisons": statistics.to_dict(orient="records"),
        "overlap": overlap_summaries,
        "seed_reference": seed_reference,
        "notes": [
            "Scores are docking scores; lower is better.",
            "Hits are virtual compounds with dock_iteration > 0 and dock_score <= cutoff.",
            (
                "No molecule-level p-values are reported because cohorts overlap and are "
                "adaptively selected."
            ),
            "LCB and EI selected counts come from authoritative acquisition files.",
            (
                "EI timing is an operational replay cost because model v0 and the seed "
                "atlas were reused."
            ),
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )

    expected_files = [
        *tables,
        "analysis_summary.json",
        *[
            f"{stem}.{suffix}"
            for stem in (
                "01_score_distributions",
                "02_hit_yield",
                "03_top_k_quality",
            )
            for suffix in ("png", "pdf")
        ],
    ]
    validate_outputs(output_dir, expected_files)
    LOGGER.info("Complete in %.1f s: %s", time.monotonic() - started, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=DATABASE",
        help="Run label and SQLite database; repeat in display order",
    )
    parser.add_argument(
        "--timing",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="Run label and stage-timing CSV; repeat for every run",
    )
    parser.add_argument(
        "--acquisition-dir",
        action="append",
        default=[],
        metavar="LABEL=DIRECTORY",
        help="Optional directory containing iter*/acquisition.csv",
    )
    parser.add_argument(
        "--expected-hits",
        action="append",
        default=[],
        metavar="LABEL=COUNT",
        help="Optional expected virtual-hit count for validation",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=-9.7)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    try:
        calculate(parse_args())
    except Exception:
        LOGGER.exception("Hit-quality comparison failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
