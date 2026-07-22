#!/usr/bin/env python3
# ruff: noqa: E501
"""Compare pre-diversity virtual-hit quality for greedy and LCB runs."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger("compare_hit_quality")
BLUE = "#0072B2"  # Okabe-Ito blue
ORANGE = "#E69F00"  # Okabe-Ito orange
QUANTILES = (0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999)
TOP_KS = (100, 500, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000)


@dataclass(frozen=True)
class RunData:
    """Scored virtual cohort and scored shared-seed reference cohort."""

    label: str
    database: Path
    virtual: pd.DataFrame
    seeds: pd.DataFrame
    iteration_selected: dict[int, int]
    virtual_selected: int


def readonly_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def load_run(label: str, database: Path) -> RunData:
    """Load only scored virtual/seed reghashes and scores, plus SQL cohort counts."""
    started = time.monotonic()
    with readonly_connection(database) as connection:
        selected_rows = connection.execute(
            "SELECT dock_iteration, COUNT(*) FROM data WHERE dock_iteration > 0 "
            "GROUP BY dock_iteration ORDER BY dock_iteration"
        ).fetchall()
        virtual = pd.read_sql_query(
            "SELECT reghash, dock_score, dock_iteration FROM data "
            "WHERE dock_iteration > 0 AND dock_score IS NOT NULL",
            connection,
        )
        seeds = pd.read_sql_query(
            "SELECT reghash, dock_score FROM data "
            "WHERE dock_iteration = 0 AND dock_score IS NOT NULL",
            connection,
        )
    if virtual["reghash"].isna().any() or (virtual["reghash"].str.strip() == "").any():
        raise ValueError(f"{label}: scored virtual cohort has missing reghashes")
    if virtual["reghash"].duplicated().any():
        raise ValueError(f"{label}: scored virtual cohort has duplicate reghashes")
    if seeds["reghash"].isna().any() or seeds["reghash"].duplicated().any():
        raise ValueError(f"{label}: scored seed cohort has missing or duplicate reghashes")
    iteration_selected = {int(iteration): int(count) for iteration, count in selected_rows}
    LOGGER.info(
        "Loaded %s: %d scored virtual rows, %d scored seeds in %.1f s",
        label,
        len(virtual),
        len(seeds),
        time.monotonic() - started,
    )
    return RunData(
        label=label,
        database=database,
        virtual=virtual,
        seeds=seeds,
        iteration_selected=iteration_selected,
        virtual_selected=sum(iteration_selected.values()),
    )


def wilson_interval(hits: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    z = 1.959963984540054
    proportion = hits / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
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
    result.update({f"q_{q * 100:g}": float(np.quantile(values, q)) for q in QUANTILES})
    return result


def summary_row(run: RunData, cohort: pd.DataFrame, iteration: str, selected: int, cutoff: float) -> dict[str, Any]:
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
        **score_summary(cohort["dock_score"].to_numpy()),
    }


def rate_comparison(greedy_hits: int, greedy_n: int, lcb_hits: int, lcb_n: int) -> dict[str, float]:
    greedy_rate, lcb_rate = greedy_hits / greedy_n, lcb_hits / lcb_n
    difference = lcb_rate - greedy_rate
    standard_error = math.sqrt(greedy_rate * (1 - greedy_rate) / greedy_n + lcb_rate * (1 - lcb_rate) / lcb_n)
    odds_ratio = (lcb_hits / (lcb_n - lcb_hits)) / (greedy_hits / (greedy_n - greedy_hits))
    return {
        "greedy_hit_rate": greedy_rate,
        "lcb_hit_rate": lcb_rate,
        "absolute_difference_pp": 100 * difference,
        "difference95_low_pp": 100 * (difference - 1.959963984540054 * standard_error),
        "difference95_high_pp": 100 * (difference + 1.959963984540054 * standard_error),
        "relative_risk_lcb_over_greedy": lcb_rate / greedy_rate,
        "odds_ratio_lcb_over_greedy": odds_ratio,
    }


def score_comparison(greedy: np.ndarray, lcb: np.ndarray) -> dict[str, float]:
    n_greedy, n_lcb = len(greedy), len(lcb)
    u_lcb_greater = float(stats.mannwhitneyu(lcb, greedy, alternative="two-sided").statistic)
    probability_lcb_better = 1.0 - u_lcb_greater / (n_lcb * n_greedy)
    return {
        "ks_statistic": float(stats.ks_2samp(lcb, greedy, method="asymp").statistic),
        "wasserstein_distance": float(stats.wasserstein_distance(lcb, greedy)),
        "common_language_probability_lcb_better": probability_lcb_better,
        "rank_biserial_cliffs_delta_lcb_better": 2 * probability_lcb_better - 1,
        "mean_difference_lcb_minus_greedy": float(np.mean(lcb) - np.mean(greedy)),
        "median_difference_lcb_minus_greedy": float(np.median(lcb) - np.median(greedy)),
    }


def cutoff_table(greedy: RunData, lcb: RunData, cutoff: float) -> pd.DataFrame:
    thresholds = sorted(set(np.round(np.arange(-12, -7.999, 0.25), 2)).union({cutoff}))
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for run in (greedy, lcb):
            cohorts: list[tuple[str, pd.DataFrame, int]] = [("overall", run.virtual, run.virtual_selected)]
            cohorts.extend(
                (str(iteration), group, run.iteration_selected[iteration])
                for iteration, group in run.virtual.groupby("dock_iteration")
            )
            for iteration, cohort, selected in cohorts:
                hits = int((cohort["dock_score"] <= threshold).sum())
                rows.append({
                    "run": run.label, "iteration": iteration, "cutoff": threshold,
                    "selected": selected, "scored": len(cohort), "hit_count": hits,
                    "hit_rate_scored": hits / len(cohort), "hit_rate_selected": hits / selected,
                })
    return pd.DataFrame(rows)


def top_k_table(greedy: RunData, lcb: RunData, cutoff: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in (greedy, lcb):
        sorted_scores = np.sort(run.virtual["dock_score"].to_numpy())
        for k in (*TOP_KS, len(sorted_scores)):
            if k > len(sorted_scores):
                continue
            top = sorted_scores[:k]
            rows.append({"run": run.label, "k": k, "is_full_scored_cohort": k == len(sorted_scores),
                         "kth_worst_score": float(top[-1]), "mean_score": float(np.mean(top)),
                         "median_score": float(np.median(top)), "hit_count": int((top <= cutoff).sum()),
                         "hit_rate": float(np.mean(top <= cutoff))})
    return pd.DataFrame(rows).drop_duplicates(["run", "k"])


def overlap_table(greedy: RunData, lcb: RunData, cutoff: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    greedy_scores = greedy.virtual.set_index("reghash")["dock_score"]
    lcb_scores = lcb.virtual.set_index("reghash")["dock_score"]
    shared = greedy_scores.index.intersection(lcb_scores.index)
    scored_union = greedy_scores.index.union(lcb_scores.index)
    greedy_hits = greedy_scores[greedy_scores <= cutoff].index
    lcb_hits = lcb_scores[lcb_scores <= cutoff].index
    hit_shared = greedy_hits.intersection(lcb_hits)
    hit_union = greedy_hits.union(lcb_hits)
    differences = (greedy_scores.loc[shared] - lcb_scores.loc[shared]).abs().to_numpy()
    agreement: dict[str, Any] = {
        "shared_scored_count": len(shared), "scored_union_count": len(scored_union),
        "scored_jaccard": len(shared) / len(scored_union), "greedy_only_scored": len(greedy_scores.index.difference(lcb_scores.index)),
        "lcb_only_scored": len(lcb_scores.index.difference(greedy_scores.index)), "shared_hit_count": len(hit_shared),
        "hit_union_count": len(hit_union), "hit_jaccard": len(hit_shared) / len(hit_union),
        "greedy_only_hits": len(greedy_hits.difference(lcb_hits)), "lcb_only_hits": len(lcb_hits.difference(greedy_hits)),
        "score_exact_match_count": int(np.sum(differences == 0)), "score_tolerance_1e-8_count": int(np.sum(differences <= 1e-8)),
        "score_abs_difference_mean": float(np.mean(differences)), "score_abs_difference_median": float(np.median(differences)),
        "score_abs_difference_max": float(np.max(differences)),
        "score_pearson_r": float(stats.pearsonr(greedy_scores.loc[shared], lcb_scores.loc[shared]).statistic),
        "score_spearman_r": float(stats.spearmanr(greedy_scores.loc[shared], lcb_scores.loc[shared]).statistic),
    }
    rows = [{"metric": key, "value": value} for key, value in agreement.items()]
    for k in TOP_KS:
        greedy_top = set(greedy_scores.nsmallest(min(k, len(greedy_scores))).index)
        lcb_top = set(lcb_scores.nsmallest(min(k, len(lcb_scores))).index)
        rows.append({"metric": f"top_{k}_overlap_count", "value": len(greedy_top & lcb_top)})
        rows.append({"metric": f"top_{k}_jaccard", "value": len(greedy_top & lcb_top) / len(greedy_top | lcb_top)})
    return pd.DataFrame(rows), agreement


def stage_hours(path: Path) -> float:
    frame = pd.read_csv(path)
    required = {"category", "hours", "minutes", "seconds"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Timing CSV lacks {sorted(required)}: {path}")
    total_rows = frame[frame["category"].astype(str).str.upper() == "TOTAL"]
    if len(total_rows) != 1:
        raise ValueError(f"Timing CSV must contain exactly one TOTAL row: {path}")
    total = total_rows.iloc[0]
    return float(total["hours"] + total["minutes"] / 60 + total["seconds"] / 3600)


def operational_table(greedy: RunData, lcb: RunData, cutoff: float, greedy_timing: Path, lcb_timing: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run, timing in ((greedy, greedy_timing), (lcb, lcb_timing)):
        hits = int((run.virtual["dock_score"] <= cutoff).sum())
        hours = stage_hours(timing)
        rows.append({"run": run.label, "selected": run.virtual_selected, "scored": len(run.virtual),
                     "missing_docking_scores": run.virtual_selected - len(run.virtual), "hit_count": hits,
                     "hits_per_100k_selected": 100_000 * hits / run.virtual_selected,
                     "measured_stage_hours": hours, "hits_per_measured_stage_hour": hits / hours,
                     "timing_csv": str(timing)})
    return pd.DataFrame(rows)


def save_figure(figure: plt.Figure, stem: Path, dpi: int) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def make_figures(output_dir: Path, greedy: RunData, lcb: RunData, cutoffs: pd.DataFrame, top_k: pd.DataFrame, cutoff: float, dpi: int) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    all_scores = np.concatenate([greedy.virtual.dock_score.to_numpy(), lcb.virtual.dock_score.to_numpy()])
    lo, hi = np.quantile(all_scores, [0.001, 0.999])
    bins = np.linspace(lo, hi, 80)
    for run, color in ((greedy, BLUE), (lcb, ORANGE)):
        scores = run.virtual.dock_score.to_numpy()
        axes[0].hist(scores[(scores >= lo) & (scores <= hi)], bins=bins, density=True, histtype="step", linewidth=1.5, color=color, label=run.label)
        ordered = np.sort(scores)
        axes[1].step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post", color=color, label=run.label)
    for axis in axes:
        axis.axvline(cutoff, color="black", linestyle="--", linewidth=1, label=f"cutoff {cutoff:g}")
        axis.set_xlabel("Docking score (lower is better)")
        axis.legend(frameon=False)
    axes[0].set_ylabel("Normalized density")
    axes[0].set_title("Central 99.8% score range shown")
    axes[1].set_ylabel("ECDF")
    save_figure(fig, output_dir / "01_score_distributions", dpi)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for run, color in ((greedy, BLUE), (lcb, ORANGE)):
        subset = cutoffs[(cutoffs.run == run.label) & (cutoffs.iteration == "overall")]
        axes[0].plot(subset.cutoff, 100 * subset.hit_rate_scored, color=color, label=run.label)
        primary = subset[np.isclose(subset.cutoff, cutoff)].iloc[0]
        axes[0].scatter([cutoff], [100 * primary.hit_rate_scored], color=color, zorder=3)
        for iteration, marker in ((1, "o"), (2, "s")):
            row = cutoffs[(cutoffs.run == run.label) & (cutoffs.iteration == str(iteration)) & np.isclose(cutoffs.cutoff, cutoff)].iloc[0]
            low, high = wilson_interval(int(row.hit_count), int(row.scored))
            axes[1].errorbar(f"{run.label}\nround {iteration}", 100 * row.hit_rate_scored, yerr=[[100 * (row.hit_rate_scored - low)], [100 * (high - row.hit_rate_scored)]], fmt=marker, color=color, capsize=3)
    axes[0].axvline(cutoff, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Hit cutoff (docking score; lower is stricter)", ylabel="Hit rate among scored compounds (%)")
    axes[0].legend(frameon=False)
    axes[1].set(ylabel="Hit rate at -9.7 (%)", title="Error bars: Wilson 95% CI")
    save_figure(fig, output_dir / "02_hit_yield", dpi)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for run, color in ((greedy, BLUE), (lcb, ORANGE)):
        subset = top_k[top_k.run == run.label].sort_values("k")
        axes[0].plot(subset.k, subset.kth_worst_score, "o-", color=color, label=run.label)
        axes[1].plot(subset.k, subset.mean_score, "o-", color=color, label=run.label)
    for axis, ylabel in zip(axes, ("Kth (worst) docking score", "Mean docking score among top k"), strict=True):
        axis.set_xscale("log")
        axis.set(xlabel="k (log scale)", ylabel=ylabel)
        axis.axhline(cutoff, color="black", linestyle="--", linewidth=1)
        axis.legend(frameon=False)
    save_figure(fig, output_dir / "03_top_k_quality", dpi)


def validate_outputs(output_dir: Path, expected_files: list[str]) -> None:
    for filename in expected_files:
        path = output_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty output: {path}")
    for filename in ("run_summary.csv", "iteration_summary.csv", "score_quantiles.csv", "cutoff_sensitivity.csv", "top_k_summary.csv", "overlap_summary.csv", "statistical_comparison.csv", "operational_efficiency.csv"):
        frame = pd.read_csv(output_dir / filename)
        if frame.empty or len(frame.columns) == 0:
            raise ValueError(f"Invalid CSV schema: {filename}")
    LOGGER.info("Validated %d output artifacts", len(expected_files))


def calculate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    greedy, lcb = load_run("Greedy", args.greedy_db), load_run("SVDKL-LCB", args.lcb_db)
    for run, expected in ((greedy, args.greedy_selected), (lcb, args.lcb_selected)):
        if len(run.virtual) > expected:
            raise ValueError(
                f"{run.label}: scored count {len(run.virtual)} exceeds selected count {expected}"
            )
    greedy = replace(greedy, virtual_selected=args.greedy_selected)
    lcb = replace(lcb, virtual_selected=args.lcb_selected)
    run_rows = [summary_row(run, run.virtual, "overall", run.virtual_selected, args.cutoff) for run in (greedy, lcb)]
    iteration_rows = [summary_row(run, group, str(iteration), run.iteration_selected[iteration], args.cutoff) for run in (greedy, lcb) for iteration, group in run.virtual.groupby("dock_iteration")]
    run_summary, iteration_summary = pd.DataFrame(run_rows), pd.DataFrame(iteration_rows)
    for run in (greedy, lcb):
        summed = iteration_summary[iteration_summary.run == run.label].hit_count.sum()
        overall = run_summary[run_summary.run == run.label].hit_count.iloc[0]
        if summed != overall:
            raise ValueError(f"{run.label}: iteration hits do not sum to overall")
    expected_hits = {
        "Greedy": args.expected_greedy_hits,
        "SVDKL-LCB": args.expected_lcb_hits,
    }
    for row in run_rows:
        expected = expected_hits[row["run"]]
        if expected is not None and row["hit_count"] != expected:
            raise ValueError(f"{row['run']}: unexpected primary hit count {row['hit_count']}")
    primary_hits = {str(row["run"]): int(row["hit_count"]) for row in run_rows}
    quantiles = run_summary[["run", *[f"q_{q * 100:g}" for q in QUANTILES]]].melt("run", var_name="quantile", value_name="dock_score")
    cutoffs = cutoff_table(greedy, lcb, args.cutoff)
    top_k = top_k_table(greedy, lcb, args.cutoff)
    overlap, overlap_data = overlap_table(greedy, lcb, args.cutoff)
    statistics = pd.DataFrame([{"comparison": "SVDKL-LCB vs Greedy", **score_comparison(greedy.virtual.dock_score.to_numpy(), lcb.virtual.dock_score.to_numpy()), **rate_comparison(primary_hits["Greedy"], len(greedy.virtual), primary_hits["SVDKL-LCB"], len(lcb.virtual))}])
    operational = operational_table(greedy, lcb, args.cutoff, args.greedy_timing, args.lcb_timing)
    seed_merged = greedy.seeds.merge(lcb.seeds, on="reghash", suffixes=("_greedy", "_lcb"))
    seed_hits_greedy = int((greedy.seeds.dock_score <= args.cutoff).sum())
    seed_hits_lcb = int((lcb.seeds.dock_score <= args.cutoff).sum())
    seed_reference = {"scored_seed_count_greedy": len(greedy.seeds), "scored_seed_count_lcb": len(lcb.seeds), "shared_seed_count": len(seed_merged), "seed_hash_sets_equal": set(greedy.seeds.reghash) == set(lcb.seeds.reghash), "seed_hit_count_greedy": seed_hits_greedy, "seed_hit_count_lcb": seed_hits_lcb, "seed_hit_rate": seed_hits_greedy / len(greedy.seeds), "seed_scores_exactly_agree": bool(np.array_equal(np.sort(seed_merged.dock_score_greedy), np.sort(seed_merged.dock_score_lcb))), "greedy_virtual_enrichment_over_seed": run_rows[0]["hit_rate_scored"] / (seed_hits_greedy / len(greedy.seeds)), "lcb_virtual_enrichment_over_seed": run_rows[1]["hit_rate_scored"] / (seed_hits_greedy / len(greedy.seeds))}
    for name, frame in (("run_summary.csv", run_summary), ("iteration_summary.csv", iteration_summary), ("score_quantiles.csv", quantiles), ("cutoff_sensitivity.csv", cutoffs), ("top_k_summary.csv", top_k), ("overlap_summary.csv", overlap), ("statistical_comparison.csv", statistics), ("operational_efficiency.csv", operational)):
        frame.to_csv(output_dir / name, index=False)
    make_figures(output_dir, greedy, lcb, cutoffs, top_k, args.cutoff, args.dpi)
    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": {"greedy_db": str(args.greedy_db), "lcb_db": str(args.lcb_db), "cutoff": args.cutoff},
        "validation": {
            "expected_primary_hits": expected_hits,
            "observed_primary_hits": primary_hits,
            "iteration_hit_sums_equal_overall": True,
            "scored_virtual_reghashes_unique": True,
        },
        "run_summary": run_rows,
        "score_comparison": statistics.iloc[0].to_dict(),
        "overlap": overlap_data,
        "seed_reference": seed_reference,
        "notes": [
            "Scores are docking scores; lower is better.",
            "No p-values are reported because the cohorts are very large and may overlap.",
            "LCB full-space clustering can inflate total wall time.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    expected = [
        "run_summary.csv",
        "iteration_summary.csv",
        "score_quantiles.csv",
        "cutoff_sensitivity.csv",
        "top_k_summary.csv",
        "overlap_summary.csv",
        "statistical_comparison.csv",
        "operational_efficiency.csv",
        "analysis_summary.json",
        *[
            f"{stem}.{suffix}"
            for stem in ("01_score_distributions", "02_hit_yield", "03_top_k_quality")
            for suffix in ("png", "pdf")
        ],
    ]
    validate_outputs(output_dir, expected)
    LOGGER.info("Complete in %.1f s: %s", time.monotonic() - started, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--greedy-db", type=Path, required=True)
    parser.add_argument("--lcb-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--greedy-timing", type=Path, required=True)
    parser.add_argument("--lcb-timing", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=-9.7)
    parser.add_argument("--greedy-selected", type=int, default=200_000)
    parser.add_argument("--lcb-selected", type=int, default=200_000)
    parser.add_argument("--expected-greedy-hits", type=int)
    parser.add_argument("--expected-lcb-hits", type=int)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        calculate(parse_args())
    except Exception:
        LOGGER.exception("Hit-quality comparison failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
