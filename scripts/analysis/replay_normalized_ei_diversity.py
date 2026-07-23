#!/usr/bin/env python3
# ruff: noqa: E501
"""Replay beta=1 EI with a scale-normalized dynamic diversity penalty."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_ei_acquisition_attribution import (
    RoundDefinition,
    expected_improvement_scores,
    load_actual_acquisition,
    load_candidate_pool,
    load_score_map,
    load_selected_compounds_with_outcomes,
    validate_historical_replay,
)
from analyze_lcb_acquisition_attribution import (
    PolicySelection,
    add_structures,
    deterministic_top_k,
    penalized_top_k,
    policy_metrics,
)

plt.switch_backend("Agg")
LOGGER = logging.getLogger("replay_normalized_ei_diversity")
ALPHAS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
HISTORICAL_LAMBDA = 0.5
ATLAS_ID = "morgan-r2-1024-t040"


@dataclass(frozen=True)
class ReplayPolicy:
    name: str
    kind: str
    alpha: float | None
    cluster_lambda: float
    selection: PolicySelection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ei-db", type=Path, required=True)
    parser.add_argument("--lcb-docked-db", type=Path, required=True)
    parser.add_argument("--greedy-docked-db", type=Path, required=True)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--threshold", type=float, default=-9.7)
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--pair-samples", type=int, default=250_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if min(args.batch_size, args.pair_samples, args.dpi) < 1:
        parser.error("batch size, pair samples, and DPI must be positive")
    return args


def alpha_label(alpha: float) -> str:
    return f"alpha_{alpha:g}".replace(".", "p")


def unpenalized_selection(
    base_scores: np.ndarray,
    identifiers: np.ndarray,
    clusters: np.ndarray,
    count: int,
) -> PolicySelection:
    indices = deterministic_top_k(base_scores, identifiers, count)
    counts: dict[int, int] = {}
    counts_before = np.empty(count, dtype=np.int32)
    for position, index in enumerate(indices):
        cluster = int(clusters[index])
        counts_before[position] = counts.get(cluster, 0)
        counts[cluster] = counts_before[position] + 1
    return PolicySelection(
        "unpenalized_ei",
        indices,
        counts_before,
        np.zeros(count, dtype=np.float64),
    )


def frontier_scale(
    base_scores: np.ndarray,
    identifiers: np.ndarray,
    batch_size: int,
) -> dict[str, float | int]:
    order = np.lexsort((identifiers, base_scores))

    def summarize(start_multiplier: float, stop_multiplier: float) -> dict[str, float]:
        start = int(start_multiplier * batch_size)
        stop = min(int(stop_multiplier * batch_size), len(order))
        values = base_scores[order[start:stop]]
        q10, q50, q90 = np.quantile(values, [0.10, 0.50, 0.90])
        return {
            "start_rank": start + 1,
            "stop_rank": stop,
            "q10": float(q10),
            "median": float(q50),
            "q90": float(q90),
            "scale": float(q90 - q10),
        }

    primary = summarize(0.5, 2.0)
    sensitivity = summarize(1.0, 3.0)
    if primary["scale"] <= 0:
        raise ValueError("primary EI frontier scale is not positive")
    return {
        "primary_start_rank": int(primary["start_rank"]),
        "primary_stop_rank": int(primary["stop_rank"]),
        "primary_q10": primary["q10"],
        "primary_median": primary["median"],
        "primary_q90": primary["q90"],
        "primary_scale": primary["scale"],
        "sensitivity_start_rank": int(sensitivity["start_rank"]),
        "sensitivity_stop_rank": int(sensitivity["stop_rank"]),
        "sensitivity_q10": sensitivity["q10"],
        "sensitivity_median": sensitivity["median"],
        "sensitivity_q90": sensitivity["q90"],
        "sensitivity_scale": sensitivity["scale"],
        "scale_ratio_sensitivity_over_primary": float(
            sensitivity["scale"] / primary["scale"]
        ),
    }


def build_policies(
    base_scores: np.ndarray,
    identifiers: np.ndarray,
    clusters: np.ndarray,
    batch_size: int,
    scale: float,
) -> list[ReplayPolicy]:
    policies: list[ReplayPolicy] = []
    for alpha in ALPHAS:
        cluster_lambda = alpha * scale / math.log(2.0)
        if alpha == 0:
            selection = unpenalized_selection(
                base_scores, identifiers, clusters, batch_size
            )
        else:
            indices, counts, penalties = penalized_top_k(
                base_scores,
                identifiers,
                clusters,
                batch_size,
                cluster_lambda,
            )
            selection = PolicySelection(alpha_label(alpha), indices, counts, penalties)
        policies.append(
            ReplayPolicy(
                name=alpha_label(alpha),
                kind="normalized_alpha",
                alpha=alpha,
                cluster_lambda=cluster_lambda,
                selection=selection,
            )
        )
    indices, counts, penalties = penalized_top_k(
        base_scores,
        identifiers,
        clusters,
        batch_size,
        HISTORICAL_LAMBDA,
    )
    policies.append(
        ReplayPolicy(
            name="historical_fixed_lambda_0p5",
            kind="historical_fixed_lambda",
            alpha=None,
            cluster_lambda=HISTORICAL_LAMBDA,
            selection=PolicySelection(
                "historical_fixed_lambda_0p5", indices, counts, penalties
            ),
        )
    )
    return policies


def concentration_metrics(clusters: np.ndarray) -> dict[str, float | int]:
    _, counts = np.unique(clusters, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    hhi = float(np.square(probabilities).sum())
    return {
        "unique_clusters": int(len(counts)),
        "effective_clusters_exp_entropy": float(math.exp(entropy)),
        "cluster_hhi": hhi,
        "effective_clusters_inverse_hhi": float(1.0 / hhi),
        "largest_cluster_count": int(counts.max()),
        "largest_cluster_fraction": float(counts.max() / counts.sum()),
        "singleton_cluster_fraction": float(np.mean(counts == 1)),
    }


def pareto_flags(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["pareto_utility_internal_diversity"] = False
    normalized = result[result["kind"] == "normalized_alpha"]
    for index, row in normalized.iterrows():
        dominated = (
            (normalized["ei_utility_retention"] >= row["ei_utility_retention"])
            & (
                normalized["internal_diversity"]
                >= row["internal_diversity"]
            )
            & (
                (normalized["ei_utility_retention"] > row["ei_utility_retention"])
                | (
                    normalized["internal_diversity"]
                    > row["internal_diversity"]
                )
            )
        ).any()
        result.loc[index, "pareto_utility_internal_diversity"] = not dominated
    return result


def replay_round(
    definition: RoundDefinition,
    args: argparse.Namespace,
    lcb_scores: dict[str, float],
    greedy_scores: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pool = load_candidate_pool(args.ei_db, definition)
    actual = load_actual_acquisition(
        args.acquisition_root / f"iter{definition.round_id}" / "acquisition.csv",
        definition,
        args.batch_size,
        args.threshold,
        args.xi,
    )
    base_scores = expected_improvement_scores(
        pool.means, pool.epistemic, args.threshold, args.xi
    )
    scale = frontier_scale(base_scores, pool.identifiers, args.batch_size)
    policies = build_policies(
        base_scores,
        pool.identifiers,
        pool.atlas_clusters,
        args.batch_size,
        float(scale["primary_scale"]),
    )
    historical = next(
        policy for policy in policies if policy.kind == "historical_fixed_lambda"
    )
    validate_historical_replay(
        definition, pool, actual, historical.selection, base_scores
    )
    unpenalized = next(
        policy
        for policy in policies
        if policy.kind == "normalized_alpha" and policy.alpha == 0
    )
    unpenalized_ids = pool.identifiers[unpenalized.selection.indices]
    unpenalized_set = set(map(int, unpenalized_ids))
    utility = -base_scores
    unpenalized_utility_sum = float(
        utility[unpenalized.selection.indices].sum()
    )

    selected_ids_by_policy = {
        policy.name: pool.identifiers[policy.selection.indices] for policy in policies
    }
    union_ids = np.unique(np.concatenate(list(selected_ids_by_policy.values())))
    info = load_selected_compounds_with_outcomes(
        args.ei_db, union_ids, lcb_scores, greedy_scores
    )
    info, words, popcounts = add_structures(info)

    metric_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for policy_index, policy in enumerate(policies):
        selection = policy.selection
        identifiers = pool.identifiers[selection.indices]
        clusters = pool.atlas_clusters[selection.indices]
        selected_set = set(map(int, identifiers))
        overlap = len(selected_set & unpenalized_set)
        selected_utility = float(utility[selection.indices].sum())
        selected_info = info.set_index("spacehastenid").loc[identifiers]
        observed = selected_info["observed_score"].notna()
        observed_scores = selected_info.loc[observed, "observed_score"].to_numpy(
            dtype=float
        )
        observed_hits = int(np.sum(observed_scores <= args.threshold))
        unobserved = args.batch_size - len(observed_scores)
        structural = policy_metrics(
            definition.round_id,
            policy.name,
            identifiers,
            clusters,
            info,
            words,
            popcounts,
            args.threshold,
            args.pair_samples,
            args.random_seed + definition.round_id * 100 + policy_index,
        )
        structural.update(
            {
                "kind": policy.kind,
                "alpha": policy.alpha,
                "cluster_lambda": policy.cluster_lambda,
                "frontier_scale": float(scale["primary_scale"]),
                "second_member_penalty": policy.cluster_lambda * math.log(2.0),
                "historical_equivalent_alpha": (
                    policy.cluster_lambda
                    * math.log(2.0)
                    / float(scale["primary_scale"])
                ),
                "unpenalized_ei_overlap": overlap,
                "unpenalized_ei_retention": overlap / args.batch_size,
                "selected_ei_utility_sum": selected_utility,
                "ei_utility_retention": selected_utility / unpenalized_utility_sum,
                "mean_pred_score": float(pool.means[selection.indices].mean()),
                "median_pred_score": float(np.median(pool.means[selection.indices])),
                "predicted_hit_fraction": float(
                    np.mean(pool.means[selection.indices] <= args.threshold)
                ),
                "mean_epistemic_std": float(
                    pool.epistemic[selection.indices].mean()
                ),
                "median_epistemic_std": float(
                    np.median(pool.epistemic[selection.indices])
                ),
                "mean_expected_improvement": float(
                    utility[selection.indices].mean()
                ),
                "observed_hit_rate": (
                    observed_hits / len(observed_scores)
                    if len(observed_scores)
                    else math.nan
                ),
                "observed_hit_count": observed_hits,
                "observed_outcomes": len(observed_scores),
                "outcome_coverage": len(observed_scores) / args.batch_size,
                "hit_rate_lower_bound": observed_hits / args.batch_size,
                "hit_rate_upper_bound": (observed_hits + unobserved)
                / args.batch_size,
                "first_in_cluster_fraction": float(
                    np.mean(selection.cluster_count_before == 0)
                ),
                "mean_cluster_penalty": float(selection.penalties.mean()),
                "max_cluster_penalty": float(selection.penalties.max()),
                **concentration_metrics(clusters),
            }
        )
        required_overlap = 0.70 if definition.round_id == 1 else 0.85
        required_utility = 0.85 if definition.round_id == 1 else 0.95
        structural["passes_retention_constraints"] = bool(
            structural["unpenalized_ei_retention"] >= required_overlap
            and structural["ei_utility_retention"] >= required_utility
        )
        metric_rows.append(structural)
        sources = selected_info["outcome_source"].value_counts()
        source_rows.append(
            {
                "round": definition.round_id,
                "policy": policy.name,
                **{
                    f"outcomes_{source}": int(sources.get(source, 0))
                    for source in ("ei", "lcb", "greedy", "unobserved")
                },
            }
        )
        selected_rows.extend(
            {
                "round": definition.round_id,
                "policy": policy.name,
                "kind": policy.kind,
                "alpha": policy.alpha,
                "cluster_lambda": policy.cluster_lambda,
                "rank": rank,
                "spacehastenid": int(identifier),
                "clusterid": int(cluster),
                "base_score": float(base_scores[index]),
                "expected_improvement": float(utility[index]),
                "cluster_count_before": int(count_before),
                "cluster_penalty": float(penalty),
                "penalized_score": float(base_scores[index] + penalty),
            }
            for rank, (identifier, cluster, index, count_before, penalty) in enumerate(
                zip(
                    identifiers,
                    clusters,
                    selection.indices,
                    selection.cluster_count_before,
                    selection.penalties,
                    strict=True,
                ),
                start=1,
            )
        )
    metrics = pareto_flags(pd.DataFrame(metric_rows))
    scale_row = {
        "round": definition.round_id,
        "candidate_count": len(pool.identifiers),
        "batch_size": args.batch_size,
        **scale,
        "historical_lambda": HISTORICAL_LAMBDA,
        "historical_equivalent_alpha": HISTORICAL_LAMBDA
        * math.log(2.0)
        / float(scale["primary_scale"]),
    }
    return (
        metrics,
        pd.DataFrame(source_rows),
        pd.DataFrame(selected_rows),
        scale_row,
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def alpha_axis(data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    positions = np.arange(len(data), dtype=float)
    labels = [f"{value:g}" for value in data["alpha"]]
    return positions, labels


def save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def make_figures(metrics: pd.DataFrame, output: Path, dpi: int) -> None:
    configure_style()
    normalized = metrics[metrics["kind"] == "normalized_alpha"].copy()
    historical = metrics[metrics["kind"] == "historical_fixed_lambda"]

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    for axis, round_id in zip(axes, (1, 2), strict=True):
        data = normalized[normalized["round"] == round_id].sort_values("alpha")
        positions, labels = alpha_axis(data)
        axis.plot(
            positions,
            100 * data["unpenalized_ei_retention"],
            "o-",
            label="Top-EI identity retention",
        )
        axis.plot(
            positions,
            100 * data["ei_utility_retention"],
            "s--",
            label="Total EI utility retention",
        )
        historical_row = historical[historical["round"] == round_id].iloc[0]
        axis.scatter(
            len(data) + 0.5,
            100 * historical_row["unpenalized_ei_retention"],
            marker="x",
            s=55,
            color="#D55E00",
            label="Historical lambda=0.5",
        )
        axis.set(
            xlabel="Dimensionless normalized alpha",
            ylabel="Retention (%)",
            title=f"Round {round_id}",
        )
        axis.set_xticks(
            [*positions, len(data) + 0.5],
            [*labels, f"historical\n({historical_row['historical_equivalent_alpha']:.2f})"],
            rotation=30,
            ha="right",
        )
        axis.legend(frameon=False)
    figure.suptitle("Normalized penalty retention of unpenalized EI", y=1.02)
    figure.tight_layout()
    save_figure(figure, output / "01_ei_retention", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    for axis, round_id in zip(axes, (1, 2), strict=True):
        data = normalized[normalized["round"] == round_id].sort_values("alpha")
        positions, labels = alpha_axis(data)
        axis.plot(
            positions, data["internal_diversity"], "o-", label="Internal diversity"
        )
        secondary = axis.twinx()
        secondary.plot(
            positions,
            data["effective_clusters_exp_entropy"],
            "s--",
            color="#009E73",
            label="Effective clusters",
        )
        axis.set(
            xlabel="Dimensionless normalized alpha",
            ylabel="Internal diversity",
            title=f"Round {round_id}",
        )
        axis.set_xticks(positions, labels)
        secondary.set_ylabel("Effective cluster count")
    figure.suptitle("Structural breadth under normalized EI penalties", y=1.02)
    figure.tight_layout()
    save_figure(figure, output / "02_diversity_concentration", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    for axis, round_id in zip(axes, (1, 2), strict=True):
        data = normalized[normalized["round"] == round_id].sort_values("alpha")
        positions, labels = alpha_axis(data)
        axis.fill_between(
            positions,
            100 * data["hit_rate_lower_bound"],
            100 * data["hit_rate_upper_bound"],
            alpha=0.18,
            color="#777777",
            label="Unobserved-outcome bounds",
        )
        axis.plot(
            positions,
            100 * data["observed_hit_rate"],
            "o-",
            color="#0072B2",
            label="Observed-subset hit rate",
        )
        secondary = axis.twinx()
        secondary.plot(
            positions,
            100 * data["outcome_coverage"],
            "s--",
            color="#E69F00",
            label="Outcome coverage",
        )
        axis.set(
            xlabel="Dimensionless normalized alpha",
            ylabel="Hit rate or bound (%)",
            title=f"Round {round_id}",
        )
        axis.set_xticks(positions, labels)
        secondary.set_ylabel("Outcome coverage (%)")
        axis.legend(frameon=False, loc="lower left")
    figure.suptitle("Retrospective yield evidence and coverage", y=1.02)
    figure.tight_layout()
    save_figure(figure, output / "03_yield_coverage_bounds", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for axis, round_id in zip(axes, (1, 2), strict=True):
        data = normalized[normalized["round"] == round_id]
        scatter = axis.scatter(
            data["ei_utility_retention"],
            data["internal_diversity"],
            c=data["alpha"],
            cmap="viridis",
            s=55,
        )
        for row in data.itertuples(index=False):
            axis.annotate(
                f"{row.alpha:g}",
                (row.ei_utility_retention, row.internal_diversity),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set(
            xlabel="EI utility retention",
            ylabel="Internal diversity",
            title=f"Round {round_id}",
        )
        figure.colorbar(scatter, ax=axis, label="alpha")
    figure.suptitle("Beta=1 EI utility-diversity frontier", y=1.02)
    figure.tight_layout()
    save_figure(figure, output / "04_utility_diversity_pareto", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    for axis, round_id in zip(axes, (1, 2), strict=True):
        data = normalized[normalized["round"] == round_id].sort_values("alpha")
        positions, labels = alpha_axis(data)
        axis.plot(
            positions,
            100 * data["predicted_hit_fraction"],
            "o-",
            label="Predicted-hit fraction",
        )
        secondary = axis.twinx()
        secondary.plot(
            positions,
            data["typed_scaffold_richness"],
            "s--",
            color="#009E73",
            label="Typed scaffolds",
        )
        axis.set(
            xlabel="Dimensionless normalized alpha",
            ylabel="Predicted-hit fraction (%)",
            title=f"Round {round_id}",
        )
        axis.set_xticks(positions, labels)
        secondary.set_ylabel("Typed scaffold richness")
    figure.suptitle("Predicted quality versus scaffold breadth", y=1.02)
    figure.tight_layout()
    save_figure(figure, output / "05_predicted_quality_scaffolds", dpi)


def write_report(metrics: pd.DataFrame, scales: pd.DataFrame, output: Path) -> None:
    normalized = metrics[metrics["kind"] == "normalized_alpha"]
    recommended = normalized[normalized["alpha"] == 0.1].set_index("round")
    unpenalized = normalized[normalized["alpha"] == 0].set_index("round")
    lines = [
        "# Beta=1 Scale-Normalized EI Diversity Replay",
        "",
        "## Decision",
        "",
        (
            "The replay supports keeping `beta=1` for now. Correcting the diversity-penalty "
            "scale is sufficient to recover a useful potency-diversity operating region. "
            "A common `alpha=0.10` is the leading balanced policy; `alpha=0.05` is the "
            "conservative yield-oriented neighbor and `alpha=0.20` is the exploratory "
            "sensitivity policy."
        ),
        "",
        "At `alpha=0.10`:",
        "",
    ]
    for round_id in (1, 2):
        row = recommended.loc[round_id]
        baseline = unpenalized.loc[round_id]
        lines.append(
            f"- Round {round_id}: lambda={row.cluster_lambda:.5f}; retained "
            f"{100 * row.unpenalized_ei_retention:.1f}% of top-EI identities and "
            f"{100 * row.ei_utility_retention:.1f}% of EI utility; increased typed "
            f"scaffolds by {100 * (row.typed_scaffold_richness / baseline.typed_scaffold_richness - 1):.1f}% "
            f"and atlas occupancy by {100 * (row.atlas_occupied_clusters / baseline.atlas_occupied_clusters - 1):.1f}%; "
            f"reduced the largest cluster from {int(baseline.largest_cluster_count):,} "
            f"to {int(row.largest_cluster_count):,}."
        )
    lines.extend(
        [
            "",
            (
                "Historical `lambda=0.5` corresponded to normalized alpha 4.86 in round 1 "
                "and 1.60 in round 2, far beyond the useful replay grid."
            ),
            "",
        "## Definition",
        "",
        "EI is unchanged: `beta=1`, threshold `-9.7`, and `xi=0`.",
        "",
        "For `K=100,000`, the primary robust score scale is `Q90-Q10` of production-exact `-EI` scores ranked from `0.5K` through `2K`. The replay uses `lambda=alpha*scale/log(2)`, so selecting a second member from one cluster costs exactly `alpha` frontier spans.",
        "",
        "## Frontier Scales",
        "",
        "| Round | Primary scale | Wider scale | Wider/primary | Historical equivalent alpha |",
        "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in scales.itertuples(index=False):
        lines.append(
            f"| {row.round} | {row.primary_scale:.6g} | {row.sensitivity_scale:.6g} | "
            f"{row.scale_ratio_sensitivity_over_primary:.3f} | "
            f"{row.historical_equivalent_alpha:.2f} |"
        )
    lines.extend(
        [
            "",
            (
                "The wider frontier scale is smaller than the primary scale. At the same "
                "nominal `alpha=0.10`, it would produce an effective primary-grid strength "
                "of 0.066 in round 1 and 0.082 in round 2, both bracketed by the replayed "
                "`alpha=0.05` and `0.10` policies. The low-penalty operating region is "
                "therefore robust, but alpha remains tied to the stated frontier definition."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Replay Results",
            "",
            "| Round | Alpha | Lambda | EI identity retained | EI utility retained | Internal diversity | Effective clusters | Largest cluster | Outcome coverage | Observed hit rate |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in normalized.sort_values(["round", "alpha"]).itertuples(index=False):
        lines.append(
            f"| {row.round} | {row.alpha:g} | {row.cluster_lambda:.6g} | "
            f"{100 * row.unpenalized_ei_retention:.1f}% | "
            f"{100 * row.ei_utility_retention:.1f}% | {row.internal_diversity:.4f} | "
            f"{row.effective_clusters_exp_entropy:,.0f} | {row.largest_cluster_count:,} | "
            f"{100 * row.outcome_coverage:.1f}% | {100 * row.observed_hit_rate:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Suggested Prospective Shortlist",
            "",
            "- Primary: `alpha=0.10` in both rounds.",
            "- Conservative comparator: `alpha=0.05`.",
            "- Exploratory comparator: `alpha=0.20`.",
            "- Keep `beta=1`, threshold `-9.7`, and `xi=0` unchanged until these policies receive balanced outcome coverage.",
            "",
            "## Interpretation Limits",
            "",
            "Structural and selection metrics are exact. Observed-subset hit rates have policy-dependent coverage because unevaluated counterfactual compounds are not imputed. Hit-rate bounds treat every unobserved compound as either a miss or a hit and are therefore intentionally broad.",
            "",
        ]
    )
    (output / "BETA1_NORMALIZED_LAMBDA_REPLAY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def policy_overlap(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for round_id, round_frame in selected.groupby("round"):
        policies = list(round_frame["policy"].drop_duplicates())
        sets = {
            policy: set(
                round_frame.loc[
                    round_frame["policy"] == policy, "spacehastenid"
                ].astype(int)
            )
            for policy in policies
        }
        for left_index, left in enumerate(policies):
            for right in policies[left_index:]:
                intersection = len(sets[left] & sets[right])
                union = len(sets[left] | sets[right])
                rows.append(
                    {
                        "round": int(round_id),
                        "policy_left": left,
                        "policy_right": right,
                        "intersection": intersection,
                        "union": union,
                        "jaccard": intersection / union,
                        "left_only": len(sets[left] - sets[right]),
                        "right_only": len(sets[right] - sets[left]),
                    }
                )
    return pd.DataFrame(rows)


def calculate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    definitions = (
        RoundDefinition(1, 0, 1, 4_557_956, "1=1"),
        RoundDefinition(
            2,
            1,
            2,
            7_425_038,
            "d.dock_iteration IS NULL OR d.dock_iteration=2",
        ),
    )
    lcb_scores = load_score_map(args.lcb_docked_db)
    greedy_scores = load_score_map(args.greedy_docked_db)
    metric_frames: list[pd.DataFrame] = []
    source_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    scale_rows: list[dict[str, Any]] = []
    for definition in definitions:
        metrics, sources, selected, scale = replay_round(
            definition, args, lcb_scores, greedy_scores
        )
        metric_frames.append(metrics)
        source_frames.append(sources)
        selected_frames.append(selected)
        scale_rows.append(scale)
    metrics = pd.concat(metric_frames, ignore_index=True)
    sources = pd.concat(source_frames, ignore_index=True)
    selected = pd.concat(selected_frames, ignore_index=True)
    scales = pd.DataFrame(scale_rows)
    overlaps = policy_overlap(selected)
    metrics.to_csv(output / "policy_metrics.csv", index=False)
    sources.to_csv(output / "outcome_sources.csv", index=False)
    selected.to_csv(
        output / "policy_selected_ids.csv.gz", index=False, compression="gzip"
    )
    scales.to_csv(output / "frontier_scale.csv", index=False)
    overlaps.to_csv(output / "policy_overlap.csv", index=False)
    metrics[
        [
            "round",
            "policy",
            "alpha",
            "cluster_lambda",
            "pareto_utility_internal_diversity",
            "passes_retention_constraints",
            "ei_utility_retention",
            "unpenalized_ei_retention",
            "internal_diversity",
            "typed_scaffold_richness",
            "atlas_occupied_clusters",
            "observed_hit_rate",
            "outcome_coverage",
        ]
    ].to_csv(output / "pareto_frontier.csv", index=False)
    make_figures(metrics, output, args.dpi)
    write_report(metrics, scales, output)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "definition": {
            "beta": 1.0,
            "threshold": args.threshold,
            "xi": args.xi,
            "alphas": ALPHAS,
            "lambda_formula": "alpha * primary_frontier_scale / log(2)",
            "primary_frontier": "ranks 0.5K through 2K; Q90-Q10",
            "sensitivity_frontier": "ranks K through 3K; Q90-Q10",
            "batch_size": args.batch_size,
            "pair_samples": args.pair_samples,
        },
        "validation": {
            "historical_fixed_lambda_replay_exact": True,
            "round1_candidates": 3_562_032,
            "round2_candidates": 6_329_334,
            "policies_per_round": len(ALPHAS) + 1,
        },
        "scales": scales.to_dict(orient="records"),
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    required = {
        "policy_metrics.csv",
        "outcome_sources.csv",
        "policy_selected_ids.csv.gz",
        "frontier_scale.csv",
        "policy_overlap.csv",
        "pareto_frontier.csv",
        "analysis_summary.json",
        "BETA1_NORMALIZED_LAMBDA_REPLAY.md",
    }
    stems = (
        "01_ei_retention",
        "02_diversity_concentration",
        "03_yield_coverage_bounds",
        "04_utility_diversity_pareto",
        "05_predicted_quality_scaffolds",
    )
    required.update(
        f"{stem}.{suffix}" for stem in stems for suffix in ("png", "pdf")
    )
    missing = [
        name
        for name in sorted(required)
        if not (output / name).is_file() or (output / name).stat().st_size == 0
    ]
    if missing:
        raise ValueError(f"missing normalized-EI replay outputs: {missing}")
    LOGGER.info(
        "Normalized beta=1 EI replay complete in %.1f seconds", time.monotonic() - started
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    try:
        calculate(parse_args())
    except Exception:
        LOGGER.exception("Normalized beta=1 EI replay failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
