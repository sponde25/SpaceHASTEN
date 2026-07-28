#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Analyze portfolio utility, support, cap behavior, and productive coverage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from spacehasten.analysis.artifacts import write_csv, write_json
from spacehasten.analysis.coverage import coverage_depth, coverage_summary, hit_depth_bins
from spacehasten.analysis.discovery import discover_run
from spacehasten.analysis.database import ReadOnlyDatabase

SUPPORT_BINS = [-np.inf, 1, 5, 10, 20, 40, np.inf]
SUPPORT_LABELS = ["[0,1)", "[1,5)", "[5,10)", "[10,20)", "[20,40)", "[40,inf)"]
STATE_LABELS = ["0", "1-4", "5-9", "10-19", "20-39", ">=40"]
COMPONENTS = ["quality", "marginal_reward", "crowding_penalty", "final_utility"]
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]


def load_history(database: ReadOnlyDatabase) -> pd.DataFrame:
    required = {
        "acquisition_batches",
        "acquisition_selections",
        "acquisition_outcomes",
    }
    if not required <= set(database.capabilities().tables):
        raise ValueError("portfolio history tables are unavailable")
    frame = pd.read_sql_query(
        "SELECT b.dock_iteration AS round,s.selection_rank AS rank,s.spacehastenid,"
        "s.clusterid,s.model_version,s.raw_mean,s.raw_epistemic_std,s.calibrated_mean,"
        "s.calibrated_std,s.p_hit,s.expected_improvement,s.quality,s.support_before,"
        "s.support_after,s.marginal_reward,s.crowding_penalty,s.final_utility,"
        "s.cluster_count_before,s.cap_reached_after,o.status,o.dock_score,b.cap_limit,"
        "b.candidate_count,b.atlas_version FROM acquisition_batches b "
        "JOIN acquisition_selections s ON s.batch_id=b.batch_id "
        "JOIN acquisition_outcomes o ON o.batch_id=s.batch_id "
        "AND o.spacehastenid=s.spacehastenid WHERE b.strategy='portfolio' "
        "ORDER BY b.dock_iteration,s.selection_rank",
        database.connection,
    )
    if frame.empty:
        raise ValueError("portfolio history is empty")
    if frame[["round", "rank"]].duplicated().any():
        raise ValueError("portfolio history has duplicate round/rank rows")
    return frame


def state(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        [-0.5, 0.5, 4.5, 9.5, 19.5, 39.5, np.inf],
        labels=STATE_LABELS,
    )


def calibration(values: pd.DataFrame) -> dict[str, float | int]:
    probability = values["p_hit"].to_numpy(float)
    observed = values["hit"].to_numpy(float)
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    assignments = np.minimum((probability * 10).astype(int), 9)
    ece = 0.0
    for index in range(10):
        mask = assignments == index
        if mask.any():
            ece += float(mask.mean() * abs(probability[mask].mean() - observed[mask].mean()))
    return {
        "selected": len(values),
        "hits": int(observed.sum()),
        "predicted_p_hit": float(probability.mean()),
        "hit_rate": float(observed.mean()),
        "brier": float(np.mean((probability - observed) ** 2)),
        "log_loss": float(
            -np.mean(observed * np.log(clipped) + (1 - observed) * np.log(1 - clipped))
        ),
        "ece_10": ece,
        "calibration_in_the_large": float(probability.mean() - observed.mean()),
    }


def summary_tables(
    frame: pd.DataFrame, hit_threshold: float, strict_threshold: float
) -> dict[str, list[dict[str, Any]]]:
    frame = frame.copy()
    frame["hit"] = frame["dock_score"].notna() & (frame["dock_score"] <= hit_threshold)
    frame["strict_hit"] = frame["dock_score"].notna() & (frame["dock_score"] <= strict_threshold)
    frame["support_stratum"] = pd.cut(
        frame["support_before"], SUPPORT_BINS, labels=SUPPORT_LABELS, right=False
    )

    contributions = []
    for round_id, current in frame.groupby("round", sort=True):
        for component in COMPONENTS:
            values = current[component].to_numpy(float)
            contributions.append(
                {
                    "round": int(round_id),
                    "component": component,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "q05": float(np.quantile(values, 0.05)),
                    "q95": float(np.quantile(values, 0.95)),
                    "nonzero_fraction": float(np.mean(values != 0)),
                }
            )

    support_rows = []
    for (round_id, label), current in frame.groupby(
        ["round", "support_stratum"], observed=False, sort=True
    ):
        if current.empty:
            row = {
                "selected": 0,
                "hits": 0,
                "predicted_p_hit": math.nan,
                "hit_rate": math.nan,
                "brier": math.nan,
                "log_loss": math.nan,
                "ece_10": math.nan,
                "calibration_in_the_large": math.nan,
            }
        else:
            row = calibration(current)
        support_rows.append({"round": int(round_id), "support_stratum": str(label), **row})

    expected_observed = []
    cluster_rows = frame.groupby(["round", "clusterid"], sort=True).agg(
        selected=("spacehastenid", "size"),
        expected_hits=("p_hit", "sum"),
        observed_hits=("hit", "sum"),
        strict_hits=("strict_hit", "sum"),
        prior_support=("support_before", "min"),
        final_support=("support_after", "max"),
        cap_reached=("cap_reached_after", "max"),
    )
    for (round_id, clusterid), row in cluster_rows.iterrows():
        expected_observed.append(
            {
                "round": int(round_id),
                "clusterid": int(clusterid),
                **{
                    name: float(value)
                    if name in {"expected_hits", "prior_support", "final_support"}
                    else int(value)
                    for name, value in row.items()
                },
            }
        )

    cap_rows = []
    for round_id, current in frame.groupby("round", sort=True):
        grouped = current.groupby("clusterid")
        counts = grouped.size()
        limit = (
            int(current["cap_limit"].dropna().iloc[0]) if current["cap_limit"].notna().any() else 0
        )
        binding = set(counts[counts >= limit].index) if limit else set()
        selected = current[current["clusterid"].isin(binding)]
        cap_rows.append(
            {
                "round": int(round_id),
                "cap_limit": limit,
                "selected_regions": len(counts),
                "binding_regions": len(binding),
                "selections_in_binding_regions": len(selected),
                "hits_in_binding_regions": int(selected["hit"].sum()),
                "hit_rate_in_binding_regions": float(selected["hit"].mean())
                if len(selected)
                else math.nan,
            }
        )

    coverage_rows = []
    depth_rows = []
    atlas_rows = []
    cumulative = []
    for round_id, current in frame.groupby("round", sort=True):
        cumulative.append(current)
        observed = pd.concat(cumulative, ignore_index=True)
        hits = observed[observed["hit"]]
        counts = hits.groupby("clusterid").size().to_numpy(np.int64)
        metrics = coverage_summary(counts)
        atlas_rows.append(
            {
                "scope": "cumulative",
                "round": int(round_id),
                "strict_hits": int(observed["strict_hit"].sum()),
                **metrics,
            }
        )
        coverage_rows.extend(
            {"scope": "cumulative", "round": int(round_id), **row} for row in coverage_depth(counts)
        )
        depth_rows.extend(
            {"scope": "cumulative", "round": int(round_id), **row} for row in hit_depth_bins(counts)
        )

    all_clusters = np.sort(frame["clusterid"].unique())
    running = pd.Series(0, index=all_clusters, dtype=int)
    transitions = []
    first_crossing: dict[tuple[int, int], int] = {}
    for round_id, current in frame.groupby("round", sort=True):
        before = running.copy()
        additions = current[current["hit"]].groupby("clusterid").size()
        running = running.add(additions, fill_value=0).astype(int)
        transition = pd.DataFrame({"from_state": state(before), "to_state": state(running)})
        for (from_state, to_state), count in transition.value_counts().items():
            transitions.append(
                {
                    "round": int(round_id),
                    "from_state": str(from_state),
                    "to_state": str(to_state),
                    "regions": int(count),
                }
            )
        for threshold in (1, 5, 10, 20, 40):
            crossed = running[(before < threshold) & (running >= threshold)].index
            for clusterid in crossed:
                first_crossing[(int(clusterid), threshold)] = int(round_id)
    crossings = [
        {"clusterid": clusterid, "threshold": threshold, "first_round": round_id}
        for (clusterid, threshold), round_id in sorted(first_crossing.items())
    ]
    return {
        "contribution_summary": contributions,
        "support_outcomes": support_rows,
        "expected_observed_regions": expected_observed,
        "cap_binding_summary": cap_rows,
        "production_atlas_metrics": atlas_rows,
        "coverage_depth": coverage_rows,
        "hit_depth_bins": depth_rows,
        "region_transitions": transitions,
        "threshold_crossings": crossings,
    }


def save_figures(root: Path, tables: dict[str, list[dict[str, Any]]], dpi: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    contributions = pd.DataFrame(tables["contribution_summary"])
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    groups = (("quality", "final_utility"), ("marginal_reward", "crowding_penalty"))
    colors = dict(zip(COMPONENTS, OKABE_ITO, strict=True))
    for axis, components in zip(axes, groups, strict=True):
        for component in components:
            current = contributions[contributions["component"] == component]
            axis.plot(
                current["round"],
                current["mean"],
                marker="o",
                color=colors[component],
                label=component.replace("_", " "),
            )
        axis.set(
            xlabel="Round",
            ylabel="Mean contribution",
            xticks=sorted(contributions["round"].unique()),
        )
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    axes[0].set_title("Quality and final utility")
    axes[1].set_title("Reward and crowding")
    figure.tight_layout()
    _save(figure, root / "portfolio_contributions", dpi)

    support = pd.DataFrame(tables["support_outcomes"])
    final_round = support[support["round"] == support["round"].max()]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(final_round))
    axis.plot(x, final_round["predicted_p_hit"], marker="o", label="Mean predicted p(hit)")
    axis.plot(x, final_round["hit_rate"], marker="s", label="Observed hit rate")
    axis.set(
        xticks=x,
        xticklabels=final_round["support_stratum"],
        xlabel="Support before selection",
        ylabel="Rate",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save(figure, root / "support_calibration", dpi)

    coverage = pd.DataFrame(tables["coverage_depth"])
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for round_id, current in coverage.groupby("round"):
        axis.plot(current["threshold"], current["regions"], label=f"Through round {round_id}")
    axis.set(xlabel="Observed hits per region, k", ylabel="Regions with at least k hits")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(color="#E6E6E6", linewidth=0.6)
    figure.tight_layout()
    _save(figure, root / "coverage_depth", dpi)

    transition = pd.DataFrame(tables["region_transitions"])
    matrix = transition.groupby(["from_state", "to_state"])["regions"].sum().unstack(fill_value=0)
    matrix = matrix.reindex(index=STATE_LABELS, columns=STATE_LABELS, fill_value=0)
    figure, axis = plt.subplots(figsize=(6.0, 5.0))
    image = axis.imshow(matrix.to_numpy(), cmap="viridis", aspect="auto")
    axis.set(
        xticks=np.arange(6),
        xticklabels=STATE_LABELS,
        yticks=np.arange(6),
        yticklabels=STATE_LABELS,
        xlabel="End state",
        ylabel="Start state",
    )
    figure.colorbar(image, ax=axis, label="Region transitions")
    figure.tight_layout()
    _save(figure, root / "region_transitions", dpi)


def _save(figure: plt.Figure, path: Path, dpi: int) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_or_db")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hit-threshold", type=float, required=True)
    parser.add_argument("--strict-threshold", type=float, default=-11.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output root is non-empty; pass --overwrite: {root}")
    root.mkdir(parents=True, exist_ok=True)
    context = discover_run(args.run_or_db, database_path=args.database)
    with ReadOnlyDatabase(context.database_path) as database:
        if database.quick_check() != "ok":
            raise ValueError("sqlite quick_check failed")
        frame = load_history(database)
    tables = summary_tables(frame, args.hit_threshold, args.strict_threshold)
    for name, rows in tables.items():
        write_csv(root / f"{name}.csv", rows)
    save_figures(root / "figures", tables, args.dpi)
    final = tables["production_atlas_metrics"][-1]
    receipt = {
        "status": "ok",
        "database": str(context.database_path),
        "selected": len(frame),
        "scored": int(frame["dock_score"].notna().sum()),
        "hits": int((frame["dock_score"] <= args.hit_threshold).sum()),
        "strict_hits": int((frame["dock_score"] <= args.strict_threshold).sum()),
        "rounds": sorted(frame["round"].astype(int).unique().tolist()),
        "final_coverage": final,
        "tables": {name: len(rows) for name, rows in tables.items()},
    }
    write_json(root / "_SUCCESS.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
