#!/usr/bin/env python3
# ruff: noqa: E501
"""Decompose EI acquisition into mean, uncertainty, and diversity effects."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_lcb_acquisition_attribution import (
    PolicySelection,
    add_structures,
    cluster_metrics,
    deterministic_top_k,
    family_metrics,
    internal_diversity,
    load_greedy_score_map,
    pairwise_policy_overlap,
    penalized_top_k,
    policy_metrics,
)
from scipy import stats
from tqdm import tqdm

from spacehasten.core.acquisition import expected_improvement

LOGGER = logging.getLogger("analyze_ei_acquisition_attribution")
ATLAS_ID = "morgan-r2-1024-t040"
SEED_COUNT = 995_924
POLICIES = (
    "mean_only",
    "deterministic_improvement",
    "expected_improvement",
    "mean_plus_diversity",
    "deterministic_improvement_plus_diversity",
    "full_ei_plus_diversity",
    "actual_ei_historical",
)


@dataclass(frozen=True)
class RoundDefinition:
    round_id: int
    model_version: int
    atlas_version: int
    upper_id: int
    candidate_filter: str


@dataclass
class CandidatePool:
    identifiers: np.ndarray
    means: np.ndarray
    epistemic: np.ndarray
    atlas_clusters: np.ndarray
    dock_scores: np.ndarray
    dock_iterations: np.ndarray


def readonly_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def expected_improvement_scores(
    means: np.ndarray,
    epistemic: np.ndarray,
    threshold: float,
    xi: float,
) -> np.ndarray:
    """Return production-exact negative expected improvement for minimization."""
    return np.fromiter(
        (
            -expected_improvement(float(mean), float(sigma), threshold, xi)
            for mean, sigma in zip(means, epistemic, strict=True)
        ),
        dtype=np.float64,
        count=len(means),
    )


def deterministic_improvement_scores(
    means: np.ndarray, threshold: float, xi: float
) -> np.ndarray:
    return -np.maximum(threshold - means.astype(np.float64) - xi, 0.0)


def load_candidate_pool(
    database: Path, definition: RoundDefinition
) -> CandidatePool:
    started = time.monotonic()
    where = (
        f"d.spacehastenid > {SEED_COUNT} AND d.spacehastenid <= ? "
        f"AND ({definition.candidate_filter})"
    )
    with readonly_connection(database) as connection:
        count = int(
            connection.execute(
                "SELECT COUNT(*) FROM data AS d "
                "JOIN predictions AS p ON p.spacehastenid=d.spacehastenid "
                "AND p.model_version=? "
                "JOIN cluster_atlas_assignments AS a "
                "ON a.atlas_id=? AND a.spacehastenid=d.spacehastenid "
                f"WHERE {where}",
                (definition.model_version, ATLAS_ID, definition.upper_id),
            ).fetchone()[0]
        )
        identifiers = np.empty(count, dtype=np.int64)
        means = np.empty(count, dtype=np.float64)
        epistemic = np.empty(count, dtype=np.float64)
        clusters = np.empty(count, dtype=np.int64)
        dock_scores = np.full(count, np.nan, dtype=np.float64)
        dock_iterations = np.empty(count, dtype=np.int8)
        rows = connection.execute(
            "SELECT d.spacehastenid,p.pred_score,p.epistemic_std,a.clusterid,"
            "d.dock_score,COALESCE(d.dock_iteration,-1) FROM data AS d "
            "JOIN predictions AS p ON p.spacehastenid=d.spacehastenid "
            "AND p.model_version=? "
            "JOIN cluster_atlas_assignments AS a "
            "ON a.atlas_id=? AND a.spacehastenid=d.spacehastenid "
            f"WHERE {where} ORDER BY d.spacehastenid",
            (definition.model_version, ATLAS_ID, definition.upper_id),
        )
        for index, (sid, mean, sigma, cluster, score, iteration) in enumerate(
            tqdm(
                rows,
                total=count,
                desc=f"EI round {definition.round_id} candidates",
                unit="mol",
            )
        ):
            identifiers[index] = int(sid)
            means[index] = float(mean)
            epistemic[index] = float(sigma)
            clusters[index] = int(cluster)
            if score is not None:
                dock_scores[index] = float(score)
            dock_iterations[index] = int(iteration)
    if len(np.unique(identifiers)) != count or not np.all(np.diff(identifiers) > 0):
        raise ValueError(f"round {definition.round_id}: candidate IDs are not unique")
    if not np.isfinite(means).all() or not np.isfinite(epistemic).all():
        raise ValueError(f"round {definition.round_id}: predictions are not finite")
    if np.any(epistemic < 0):
        raise ValueError(f"round {definition.round_id}: negative epistemic uncertainty")
    LOGGER.info(
        "Loaded EI round %d candidate pool: %d rows in %.1f s",
        definition.round_id,
        count,
        time.monotonic() - started,
    )
    return CandidatePool(
        identifiers,
        means,
        epistemic,
        clusters,
        dock_scores,
        dock_iterations,
    )


def load_actual_acquisition(
    path: Path,
    definition: RoundDefinition,
    batch_size: int,
    threshold: float,
    xi: float,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "rank",
        "method",
        "spacehastenid",
        "model_version",
        "pred_score",
        "epistemic_std",
        "base_score",
        "clusterid",
        "cluster_count_before",
        "cluster_penalty",
        "penalized_score",
        "ei_hit_threshold",
        "ei_xi",
        "cluster_lambda",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"missing acquisition columns: {sorted(required - set(frame))}")
    if len(frame) != batch_size or frame["spacehastenid"].duplicated().any():
        raise ValueError(f"round {definition.round_id}: invalid historical batch")
    if not np.array_equal(frame["rank"], np.arange(1, batch_size + 1)):
        raise ValueError(f"round {definition.round_id}: non-contiguous ranks")
    if set(frame["method"]) != {"ei"}:
        raise ValueError(f"round {definition.round_id}: acquisition method is not EI")
    if set(frame["model_version"]) != {definition.model_version}:
        raise ValueError(f"round {definition.round_id}: model version mismatch")
    if not np.allclose(frame["ei_hit_threshold"], threshold, atol=0, rtol=0):
        raise ValueError(f"round {definition.round_id}: hit threshold mismatch")
    if not np.allclose(frame["ei_xi"], xi, atol=0, rtol=0):
        raise ValueError(f"round {definition.round_id}: xi mismatch")
    expected_base = expected_improvement_scores(
        frame["pred_score"].to_numpy(dtype=float),
        frame["epistemic_std"].to_numpy(dtype=float),
        threshold,
        xi,
    )
    expected_penalty = frame["cluster_lambda"] * np.log1p(
        frame["cluster_count_before"]
    )
    if not np.allclose(frame["base_score"], expected_base, atol=1e-10, rtol=1e-9):
        difference = np.max(np.abs(frame["base_score"] - expected_base))
        raise ValueError(
            f"round {definition.round_id}: invalid EI base scores (max diff {difference})"
        )
    if not np.allclose(frame["cluster_penalty"], expected_penalty, atol=1e-12):
        raise ValueError(f"round {definition.round_id}: invalid diversity penalties")
    if not np.allclose(
        frame["penalized_score"],
        frame["base_score"] + frame["cluster_penalty"],
        atol=1e-9,
    ):
        raise ValueError(f"round {definition.round_id}: invalid penalized scores")
    return frame


def candidate_positions(
    candidate_ids: np.ndarray, selected_ids: np.ndarray
) -> np.ndarray:
    positions = np.searchsorted(candidate_ids, selected_ids)
    valid = positions < len(candidate_ids)
    if not valid.all() or not np.array_equal(candidate_ids[positions], selected_ids):
        raise ValueError("historical selection is missing from the candidate pool")
    return positions


def select_policies(
    pool: CandidatePool,
    batch_size: int,
    cluster_lambda: float,
    threshold: float,
    xi: float,
) -> tuple[dict[str, PolicySelection], dict[str, np.ndarray]]:
    mean_scores = pool.means.astype(np.float64)
    deterministic_scores = deterministic_improvement_scores(pool.means, threshold, xi)
    ei_scores = expected_improvement_scores(
        pool.means, pool.epistemic, threshold, xi
    )
    zero_counts = np.zeros(batch_size, dtype=np.int32)
    zero_penalties = np.zeros(batch_size, dtype=np.float64)

    mean_indices = deterministic_top_k(mean_scores, pool.identifiers, batch_size)
    deterministic_indices = deterministic_top_k(
        deterministic_scores, pool.identifiers, batch_size
    )
    ei_indices = deterministic_top_k(ei_scores, pool.identifiers, batch_size)
    mean_diverse, mean_counts, mean_penalties = penalized_top_k(
        mean_scores,
        pool.identifiers,
        pool.atlas_clusters,
        batch_size,
        cluster_lambda,
    )
    deterministic_diverse, deterministic_counts, deterministic_penalties = (
        penalized_top_k(
            deterministic_scores,
            pool.identifiers,
            pool.atlas_clusters,
            batch_size,
            cluster_lambda,
        )
    )
    full_indices, full_counts, full_penalties = penalized_top_k(
        ei_scores,
        pool.identifiers,
        pool.atlas_clusters,
        batch_size,
        cluster_lambda,
    )
    policies = {
        "mean_only": PolicySelection(
            "mean_only", mean_indices, zero_counts.copy(), zero_penalties.copy()
        ),
        "deterministic_improvement": PolicySelection(
            "deterministic_improvement",
            deterministic_indices,
            zero_counts.copy(),
            zero_penalties.copy(),
        ),
        "expected_improvement": PolicySelection(
            "expected_improvement",
            ei_indices,
            zero_counts.copy(),
            zero_penalties.copy(),
        ),
        "mean_plus_diversity": PolicySelection(
            "mean_plus_diversity", mean_diverse, mean_counts, mean_penalties
        ),
        "deterministic_improvement_plus_diversity": PolicySelection(
            "deterministic_improvement_plus_diversity",
            deterministic_diverse,
            deterministic_counts,
            deterministic_penalties,
        ),
        "full_ei_plus_diversity": PolicySelection(
            "full_ei_plus_diversity", full_indices, full_counts, full_penalties
        ),
    }
    return policies, {
        "mean": mean_scores,
        "deterministic_improvement": deterministic_scores,
        "expected_improvement": ei_scores,
    }


def validate_historical_replay(
    definition: RoundDefinition,
    pool: CandidatePool,
    actual: pd.DataFrame,
    replay: PolicySelection,
    ei_scores: np.ndarray,
) -> None:
    actual_ids = actual["spacehastenid"].to_numpy(dtype=np.int64)
    replay_ids = pool.identifiers[replay.indices]
    if not np.array_equal(replay_ids, actual_ids):
        mismatch = np.flatnonzero(replay_ids != actual_ids)
        first = int(mismatch[0]) if len(mismatch) else -1
        overlap = len(set(map(int, replay_ids)) & set(map(int, actual_ids)))
        raise ValueError(
            f"round {definition.round_id}: EI replay differs from history at rank "
            f"{first + 1}; overlap={overlap}"
        )
    if not np.array_equal(
        replay.cluster_count_before,
        actual["cluster_count_before"].to_numpy(dtype=np.int32),
    ):
        raise ValueError(f"round {definition.round_id}: cluster-count replay mismatch")
    if not np.allclose(
        replay.penalties,
        actual["cluster_penalty"].to_numpy(dtype=float),
        atol=1e-12,
    ):
        raise ValueError(f"round {definition.round_id}: penalty replay mismatch")
    replay_base = ei_scores[replay.indices]
    if not np.allclose(replay_base, actual["base_score"], atol=1e-10, rtol=1e-9):
        raise ValueError(f"round {definition.round_id}: replayed EI scores mismatch")


def load_score_map(path: Path) -> dict[str, float]:
    return load_greedy_score_map(path)


def load_selected_compounds_with_outcomes(
    database: Path,
    identifiers: np.ndarray,
    lcb_scores: dict[str, float],
    greedy_scores: dict[str, float],
) -> pd.DataFrame:
    workspace = sqlite3.connect(":memory:")
    try:
        workspace.execute("ATTACH DATABASE ? AS source", (str(database.resolve()),))
        workspace.execute("CREATE TABLE selected(spacehastenid INTEGER PRIMARY KEY)")
        for start in range(0, len(identifiers), 10_000):
            workspace.executemany(
                "INSERT INTO selected(spacehastenid) VALUES (?)",
                ((int(value),) for value in identifiers[start : start + 10_000]),
            )
        frame = pd.read_sql_query(
            "SELECT d.spacehastenid,d.reghash,d.smiles,d.dock_score,d.dock_iteration "
            "FROM source.data AS d JOIN selected AS s USING(spacehastenid) "
            "ORDER BY d.spacehastenid",
            workspace,
        )
    finally:
        workspace.close()
    if len(frame) != len(identifiers):
        raise ValueError("failed to load every replay-selected compound")
    lcb = frame["reghash"].map(lcb_scores)
    greedy = frame["reghash"].map(greedy_scores)
    frame["observed_score"] = frame["dock_score"].fillna(lcb).fillna(greedy)
    frame["outcome_source"] = np.select(
        [frame["dock_score"].notna(), lcb.notna(), greedy.notna()],
        ["ei", "lcb", "greedy"],
        default="unobserved",
    )
    return frame


def actual_group_summary(
    actual: pd.DataFrame,
    words: np.ndarray,
    popcounts: np.ndarray,
    round_id: int,
    cutoff: float,
    pair_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = (
        "mean_and_ei_core",
        "ei_uncertainty_promoted",
        "diversity_rescued_mean",
        "penalty_enabled_outside_both",
    )
    for group_index, group in enumerate(groups):
        frame = actual[actual["selection_group"] == group]
        observed = frame["dock_score"].notna()
        scores = frame.loc[observed, "dock_score"].to_numpy(dtype=float)
        structure_indices = frame["structure_index"].to_numpy(dtype=np.int64)
        deterministic = np.maximum(
            frame["ei_hit_threshold"] - frame["pred_score"] - frame["ei_xi"],
            0.0,
        )
        ei = -frame["base_score"]
        metrics: dict[str, Any] = {
            "round": round_id,
            "selection_group": group,
            "count": len(frame),
            "batch_fraction": len(frame) / len(actual),
            "observed_outcomes": int(observed.sum()),
            "hit_count": int(np.sum(scores <= cutoff)),
            "hit_rate": float(np.mean(scores <= cutoff)) if len(scores) else math.nan,
            "mean_dock_score": float(np.mean(scores)) if len(scores) else math.nan,
            "median_dock_score": float(np.median(scores)) if len(scores) else math.nan,
            "mean_pred_score": float(frame["pred_score"].mean()),
            "mean_epistemic_std": float(frame["epistemic_std"].mean()),
            "mean_expected_improvement": float(ei.mean()),
            "mean_deterministic_improvement": float(deterministic.mean()),
            "mean_uncertainty_increment_to_ei": float((ei - deterministic).mean()),
            "predicted_nonhit_fraction": float(
                np.mean(frame["pred_score"] > frame["ei_hit_threshold"])
            ),
            "mean_cluster_penalty": float(frame["cluster_penalty"].mean()),
            "internal_diversity": (
                internal_diversity(
                    structure_indices,
                    words,
                    popcounts,
                    pair_samples,
                    np.random.default_rng(10_000 + round_id * 10 + group_index),
                )
                if len(frame) > 1
                else math.nan
            ),
            **(family_metrics(frame["typed_scaffold"], "typed_scaffold") if len(frame) else {}),
            **(
                family_metrics(frame["generic_framework"], "generic_framework")
                if len(frame)
                else {}
            ),
            **(
                cluster_metrics(frame["atlas_cluster"].to_numpy(dtype=np.int64))
                if len(frame)
                else {}
            ),
        }
        rows.append(metrics)
    result = pd.DataFrame(rows)
    total_hits = result["hit_count"].sum()
    result["hit_contribution_fraction"] = result["hit_count"] / total_hits
    return result


def quantile_yield(
    actual: pd.DataFrame,
    column: str,
    label: str,
    cutoff: float,
    round_id: int,
    bins: int = 5,
) -> pd.DataFrame:
    frame = actual[actual["dock_score"].notna()].copy()
    frame["bin"] = pd.qcut(
        frame[column].rank(method="first"), bins, labels=False
    ) + 1
    rows: list[dict[str, Any]] = []
    for bin_id, group in frame.groupby("bin"):
        residual = group["dock_score"] - group["pred_score"]
        rows.append(
            {
                "round": round_id,
                "bin_type": label,
                "bin": int(bin_id),
                "count": len(group),
                "value_min": float(group[column].min()),
                "value_median": float(group[column].median()),
                "value_max": float(group[column].max()),
                "hit_count": int(np.sum(group["dock_score"] <= cutoff)),
                "hit_rate": float(np.mean(group["dock_score"] <= cutoff)),
                "mean_dock_score": float(group["dock_score"].mean()),
                "mean_residual": float(residual.mean()),
                "mean_absolute_error": float(np.abs(residual).mean()),
                "favorable_surprise_rate_residual_lt_minus1": float(
                    np.mean(residual < -1.0)
                ),
            }
        )
    return pd.DataFrame(rows)


def mean_decile_uncertainty_control(
    actual: pd.DataFrame, cutoff: float, round_id: int
) -> pd.DataFrame:
    frame = actual[actual["dock_score"].notna()].copy()
    frame["mean_decile"] = pd.qcut(
        frame["pred_score"].rank(method="first"), 10, labels=False
    ) + 1
    frame["uncertainty_half"] = frame.groupby("mean_decile")[
        "epistemic_std"
    ].transform(lambda values: np.where(values >= values.median(), "high", "low"))
    rows: list[dict[str, Any]] = []
    for (decile, half), group in frame.groupby(
        ["mean_decile", "uncertainty_half"]
    ):
        residual = group["dock_score"] - group["pred_score"]
        rows.append(
            {
                "round": round_id,
                "mean_decile": int(decile),
                "uncertainty_half": half,
                "count": len(group),
                "mean_pred_score": float(group["pred_score"].mean()),
                "mean_epistemic_std": float(group["epistemic_std"].mean()),
                "hit_rate": float(np.mean(group["dock_score"] <= cutoff)),
                "mean_residual": float(residual.mean()),
                "mean_absolute_error": float(np.abs(residual).mean()),
            }
        )
    return pd.DataFrame(rows)


def uncertainty_calibration(
    actual: pd.DataFrame, round_id: int
) -> dict[str, Any]:
    frame = actual[actual["dock_score"].notna()].copy()
    residual = frame["dock_score"] - frame["pred_score"]
    return {
        "round": round_id,
        "actual_selected": len(actual),
        "observed_actual": len(frame),
        "spearman_uncertainty_absolute_error": float(
            stats.spearmanr(frame["epistemic_std"], np.abs(residual)).statistic
        ),
        "spearman_uncertainty_residual": float(
            stats.spearmanr(frame["epistemic_std"], residual).statistic
        ),
        "mean_absolute_error": float(np.abs(residual).mean()),
        "mean_residual": float(residual.mean()),
        "favorable_surprise_rate_residual_lt_minus1": float(np.mean(residual < -1)),
    }


def factorial_attribution(policy_frame: pd.DataFrame) -> pd.DataFrame:
    ignored = {
        "round",
        "policy",
        "selected",
        "observed_outcomes",
        "outcome_coverage",
        "observed_hit_count",
    }
    rows: list[dict[str, Any]] = []
    for round_id, group in policy_frame.groupby("round"):
        indexed = group.set_index("policy")
        for metric in group.columns:
            if metric in ignored or not pd.api.types.is_numeric_dtype(group[metric]):
                continue
            no_uncertainty = float(indexed.loc["deterministic_improvement", metric])
            uncertainty = float(indexed.loc["expected_improvement", metric])
            diversity = float(
                indexed.loc["deterministic_improvement_plus_diversity", metric]
            )
            full = float(indexed.loc["full_ei_plus_diversity", metric])
            rows.append(
                {
                    "round": int(round_id),
                    "metric": metric,
                    "deterministic_improvement": no_uncertainty,
                    "expected_improvement": uncertainty,
                    "deterministic_improvement_plus_diversity": diversity,
                    "full_ei_plus_diversity": full,
                    "uncertainty_main_effect": 0.5
                    * ((uncertainty - no_uncertainty) + (full - diversity)),
                    "diversity_main_effect": 0.5
                    * ((diversity - no_uncertainty) + (full - uncertainty)),
                    "interaction": full - uncertainty - diversity + no_uncertainty,
                }
            )
    return pd.DataFrame(rows)


def yield_composition_decomposition(groups: pd.DataFrame) -> pd.DataFrame:
    round1 = groups[groups["round"] == 1].set_index("selection_group")
    round2 = groups[groups["round"] == 2].set_index("selection_group")
    weights1, weights2 = round1["batch_fraction"], round2["batch_fraction"]
    rates1, rates2 = round1["hit_rate"], round2["hit_rate"]
    composition = 0.5 * (
        np.sum((weights2 - weights1) * rates1)
        + np.sum((weights2 - weights1) * rates2)
    )
    within = 0.5 * (
        np.sum(weights2 * (rates2 - rates1))
        + np.sum(weights1 * (rates2 - rates1))
    )
    total = float(np.sum(weights2 * rates2) - np.sum(weights1 * rates1))
    return pd.DataFrame(
        [
            {
                "round1_hit_rate": float(np.sum(weights1 * rates1)),
                "round2_hit_rate": float(np.sum(weights2 * rates2)),
                "total_change": total,
                "composition_effect_shapley": float(composition),
                "within_group_yield_effect_shapley": float(within),
                "composition_share_of_change": float(composition / total),
                "within_group_share_of_change": float(within / total),
            }
        ]
    )


def strategy_yield_decomposition(group_comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for round_id, group in group_comparison.groupby("round"):
        weights_lcb = group["lcb_batch_fraction"].to_numpy(dtype=float)
        weights_ei = group["ei_batch_fraction"].to_numpy(dtype=float)
        rates_lcb = group["lcb_hit_rate"].to_numpy(dtype=float)
        rates_ei = group["ei_hit_rate"].to_numpy(dtype=float)
        composition = 0.5 * (
            np.sum((weights_ei - weights_lcb) * rates_lcb)
            + np.sum((weights_ei - weights_lcb) * rates_ei)
        )
        within = 0.5 * (
            np.sum(weights_ei * (rates_ei - rates_lcb))
            + np.sum(weights_lcb * (rates_ei - rates_lcb))
        )
        lcb_rate = float(np.sum(weights_lcb * rates_lcb))
        ei_rate = float(np.sum(weights_ei * rates_ei))
        total = ei_rate - lcb_rate
        rows.append(
            {
                "round": int(round_id),
                "lcb_hit_rate": lcb_rate,
                "ei_hit_rate": ei_rate,
                "ei_minus_lcb": total,
                "composition_effect_shapley": float(composition),
                "within_group_yield_effect_shapley": float(within),
                "composition_share_of_difference": float(composition / total),
                "within_group_share_of_difference": float(within / total),
            }
        )
    return pd.DataFrame(rows)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"failed to write {path}")


def lcb_ei_comparison_tables(
    lcb_root: Path,
    ei_candidates: pd.DataFrame,
    ei_calibration: pd.DataFrame,
    ei_groups: pd.DataFrame,
    ei_factorial: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lcb_candidates = pd.read_csv(lcb_root / "round_candidate_summary.csv")
    lcb_calibration = pd.read_csv(lcb_root / "uncertainty_calibration.csv")
    lcb_groups = pd.read_csv(lcb_root / "actual_selection_groups.csv")
    lcb_factorial = pd.read_csv(lcb_root / "factorial_attribution.csv")

    lcb_round = lcb_candidates.merge(
        lcb_calibration.drop(
            columns=["candidate_count", "actual_selected"], errors="ignore"
        ),
        on="round",
    )
    ei_round = ei_candidates.merge(ei_calibration, on="round")
    round_metrics = {
        "candidate_count": ("candidate_count", "candidate_count"),
        "candidate_epistemic_median": ("epistemic_median", "epistemic_median"),
        "selected_epistemic_median": (
            "actual_epistemic_median",
            "actual_epistemic_median",
        ),
        "mean_vs_uncertainty_shortlist_replacements": (
            "uncertainty_added_vs_mean",
            "ei_added_vs_mean",
        ),
        "diversity_replacements_from_uncertainty_shortlist": (
            "historical_diversity_added_vs_uncertainty",
            "diversity_added_vs_ei",
        ),
        "selected_unique_clusters": (
            "actual_unique_historical_clusters",
            "actual_unique_clusters",
        ),
        "first_in_cluster_fraction": (
            "actual_first_in_cluster_fraction",
            "actual_first_in_cluster_fraction",
        ),
        "spearman_uncertainty_absolute_error": (
            "spearman_uncertainty_absolute_error",
            "spearman_uncertainty_absolute_error",
        ),
        "mean_absolute_error": ("mean_absolute_error", "mean_absolute_error"),
    }
    round_rows: list[dict[str, Any]] = []
    for round_id in (1, 2):
        left = lcb_round[lcb_round["round"] == round_id].iloc[0]
        right = ei_round[ei_round["round"] == round_id].iloc[0]
        for metric, (lcb_column, ei_column) in round_metrics.items():
            lcb_value = float(left[lcb_column])
            ei_value = float(right[ei_column])
            round_rows.append(
                {
                    "round": round_id,
                    "metric": metric,
                    "lcb_value": lcb_value,
                    "ei_value": ei_value,
                    "ei_minus_lcb": ei_value - lcb_value,
                    "ei_over_lcb": ei_value / lcb_value if lcb_value else math.nan,
                }
            )
    round_comparison = pd.DataFrame(round_rows)

    group_map = {
        "mean_and_uncertainty_core": "mean_and_ei_core",
        "uncertainty_promoted": "ei_uncertainty_promoted",
        "diversity_rescued_mean": "diversity_rescued_mean",
        "diversity_only": "penalty_enabled_outside_both",
    }
    group_rows: list[dict[str, Any]] = []
    for round_id in (1, 2):
        for lcb_group, ei_group in group_map.items():
            left = lcb_groups[
                (lcb_groups["round"] == round_id)
                & (lcb_groups["selection_group"] == lcb_group)
            ].iloc[0]
            right = ei_groups[
                (ei_groups["round"] == round_id)
                & (ei_groups["selection_group"] == ei_group)
            ].iloc[0]
            group_rows.append(
                {
                    "round": round_id,
                    "mechanism_group": ei_group,
                    "lcb_group": lcb_group,
                    "lcb_count": int(left["count"]),
                    "ei_count": int(right["count"]),
                    "lcb_batch_fraction": float(left["batch_fraction"]),
                    "ei_batch_fraction": float(right["batch_fraction"]),
                    "lcb_hit_rate": float(left["hit_rate"]),
                    "ei_hit_rate": float(right["hit_rate"]),
                    "ei_minus_lcb_hit_rate": float(
                        right["hit_rate"] - left["hit_rate"]
                    ),
                    "lcb_internal_diversity": float(left["internal_diversity"]),
                    "ei_internal_diversity": float(right["internal_diversity"]),
                }
            )
    group_comparison = pd.DataFrame(group_rows)

    factorial_metrics = (
        "observed_hit_rate",
        "internal_diversity",
        "typed_scaffold_richness_per_compound",
        "generic_framework_richness_per_compound",
        "atlas_clusters_per_compound",
    )
    factorial_rows: list[dict[str, Any]] = []
    for round_id in (1, 2):
        for metric in factorial_metrics:
            left = lcb_factorial[
                (lcb_factorial["round"] == round_id)
                & (lcb_factorial["metric"] == metric)
            ].iloc[0]
            right = ei_factorial[
                (ei_factorial["round"] == round_id)
                & (ei_factorial["metric"] == metric)
            ].iloc[0]
            for effect in (
                "uncertainty_main_effect",
                "diversity_main_effect",
                "interaction",
            ):
                factorial_rows.append(
                    {
                        "round": round_id,
                        "metric": metric,
                        "effect": effect,
                        "lcb_value": float(left[effect]),
                        "ei_value": float(right[effect]),
                        "ei_minus_lcb": float(right[effect] - left[effect]),
                    }
                )
    return round_comparison, group_comparison, pd.DataFrame(factorial_rows)


def calculate(args: argparse.Namespace) -> dict[str, Any]:
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
    greedy_scores = load_score_map(args.greedy_docked_db)
    lcb_scores = load_score_map(args.lcb_docked_db)
    with readonly_connection(args.ei_db) as connection:
        historical_outcomes = {
            int(round_id): (int(scored), int(hits))
            for round_id, scored, hits in connection.execute(
                "SELECT dock_iteration,COUNT(*),"
                "SUM(CASE WHEN dock_score <= ? THEN 1 ELSE 0 END) "
                "FROM data WHERE dock_iteration > 0 GROUP BY dock_iteration",
                (args.hit_threshold,),
            )
        }

    candidate_rows: list[dict[str, Any]] = []
    overlap_frames: list[pd.DataFrame] = []
    policy_metric_rows: list[dict[str, Any]] = []
    policy_source_rows: list[dict[str, Any]] = []
    selected_id_rows: list[dict[str, Any]] = []
    actual_groups: list[pd.DataFrame] = []
    actual_members: list[pd.DataFrame] = []
    uncertainty_bins: list[pd.DataFrame] = []
    ei_bins: list[pd.DataFrame] = []
    penalty_bins: list[pd.DataFrame] = []
    mean_controls: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    candidate_samples: list[pd.DataFrame] = []

    for definition in definitions:
        pool = load_candidate_pool(args.ei_db, definition)
        actual = load_actual_acquisition(
            args.acquisition_root / f"iter{definition.round_id}" / "acquisition.csv",
            definition,
            args.batch_size,
            args.hit_threshold,
            args.xi,
        )
        positions = candidate_positions(
            pool.identifiers, actual["spacehastenid"].to_numpy(dtype=np.int64)
        )
        if not np.allclose(pool.means[positions], actual["pred_score"], atol=1e-6):
            raise ValueError(f"round {definition.round_id}: means differ from history")
        if not np.allclose(
            pool.epistemic[positions], actual["epistemic_std"], atol=1e-6
        ):
            raise ValueError(
                f"round {definition.round_id}: uncertainty differs from history"
            )
        policies, score_arrays = select_policies(
            pool,
            args.batch_size,
            args.cluster_lambda,
            args.hit_threshold,
            args.xi,
        )
        validate_historical_replay(
            definition,
            pool,
            actual,
            policies["full_ei_plus_diversity"],
            score_arrays["expected_improvement"],
        )
        actual_ids = actual["spacehastenid"].to_numpy(dtype=np.int64)
        policy_ids = {
            name: pool.identifiers[selection.indices]
            for name, selection in policies.items()
        }
        policy_ids["actual_ei_historical"] = actual_ids
        overlap_frames.append(pairwise_policy_overlap(definition.round_id, policy_ids))

        mean_set = set(map(int, policy_ids["mean_only"]))
        ei_set = set(map(int, policy_ids["expected_improvement"]))
        actual_set = set(map(int, actual_ids))
        actual["selection_group"] = [
            (
                "mean_and_ei_core"
                if int(sid) in mean_set and int(sid) in ei_set
                else "ei_uncertainty_promoted"
                if int(sid) in ei_set
                else "diversity_rescued_mean"
                if int(sid) in mean_set
                else "penalty_enabled_outside_both"
            )
            for sid in actual_ids
        ]
        actual["atlas_cluster"] = pool.atlas_clusters[positions]
        actual["dock_score"] = pool.dock_scores[positions]
        actual["final_dock_iteration"] = pool.dock_iterations[positions]
        actual.loc[
            actual["final_dock_iteration"] != definition.round_id, "dock_score"
        ] = np.nan
        observed_scores = actual["dock_score"].dropna().to_numpy(dtype=float)
        observed_counts = (
            len(observed_scores),
            int(np.sum(observed_scores <= args.hit_threshold)),
        )
        if observed_counts != historical_outcomes[definition.round_id]:
            raise ValueError(
                f"round {definition.round_id}: historical outcome mismatch "
                f"{observed_counts} != {historical_outcomes[definition.round_id]}"
            )

        mean_order = np.lexsort((pool.identifiers, score_arrays["mean"]))
        deterministic_order = np.lexsort(
            (pool.identifiers, score_arrays["deterministic_improvement"])
        )
        ei_order = np.lexsort(
            (pool.identifiers, score_arrays["expected_improvement"])
        )
        for name, order in (
            ("candidate_mean_rank", mean_order),
            ("candidate_deterministic_improvement_rank", deterministic_order),
            ("candidate_ei_rank", ei_order),
        ):
            ranks = np.empty(len(pool.identifiers), dtype=np.int32)
            ranks[order] = np.arange(1, len(pool.identifiers) + 1, dtype=np.int32)
            actual[name] = ranks[positions]
        actual["ei_rank_promotion_vs_mean"] = (
            actual["candidate_mean_rank"] - actual["candidate_ei_rank"]
        )

        selected_union = np.unique(np.concatenate(list(policy_ids.values())))
        info = load_selected_compounds_with_outcomes(
            args.ei_db, selected_union, lcb_scores, greedy_scores
        )
        info, words, popcounts = add_structures(info)
        actual = actual.merge(
            info[
                [
                    "spacehastenid",
                    "reghash",
                    "structure_index",
                    "typed_scaffold",
                    "generic_framework",
                ]
            ],
            on="spacehastenid",
            how="left",
            validate="one_to_one",
        )
        actual["round"] = definition.round_id
        actual_members.append(actual)
        actual_groups.append(
            actual_group_summary(
                actual,
                words,
                popcounts,
                definition.round_id,
                args.hit_threshold,
                args.pair_samples,
            )
        )

        for name, selection in policies.items():
            identifiers = pool.identifiers[selection.indices]
            clusters = pool.atlas_clusters[selection.indices]
            metrics = policy_metrics(
                definition.round_id,
                name,
                identifiers,
                clusters,
                info,
                words,
                popcounts,
                args.hit_threshold,
                args.pair_samples,
                args.random_seed + definition.round_id * 100 + len(policy_metric_rows),
            )
            policy_metric_rows.append(metrics)
            selected_info = info.set_index("spacehastenid").loc[identifiers]
            sources = selected_info["outcome_source"].value_counts()
            policy_source_rows.append(
                {
                    "round": definition.round_id,
                    "policy": name,
                    **{f"outcomes_{source}": int(sources.get(source, 0)) for source in (
                        "ei",
                        "lcb",
                        "greedy",
                        "unobserved",
                    )},
                }
            )
            selected_id_rows.extend(
                {
                    "round": definition.round_id,
                    "policy": name,
                    "spacehastenid": int(sid),
                }
                for sid in identifiers
            )
        actual_metrics = policy_metrics(
            definition.round_id,
            "actual_ei_historical",
            actual_ids,
            actual["atlas_cluster"].to_numpy(dtype=np.int64),
            info,
            words,
            popcounts,
            args.hit_threshold,
            args.pair_samples,
            args.random_seed + definition.round_id * 100 + 99,
        )
        historical_scores = actual["dock_score"].dropna().to_numpy(dtype=float)
        actual_metrics.update(
            {
                "observed_outcomes": len(historical_scores),
                "outcome_coverage": len(historical_scores) / len(actual),
                "observed_hit_count": int(
                    np.sum(historical_scores <= args.hit_threshold)
                ),
                "observed_hit_rate": float(
                    np.mean(historical_scores <= args.hit_threshold)
                ),
                "observed_mean_dock_score": float(np.mean(historical_scores)),
            }
        )
        policy_metric_rows.append(actual_metrics)

        calibration_rows.append(uncertainty_calibration(actual, definition.round_id))
        uncertainty_bins.append(
            quantile_yield(
                actual,
                "epistemic_std",
                "epistemic_uncertainty",
                args.hit_threshold,
                definition.round_id,
            )
        )
        ei_bins.append(
            quantile_yield(
                actual,
                "base_score",
                "negative_expected_improvement",
                args.hit_threshold,
                definition.round_id,
            )
        )
        penalty_bins.append(
            quantile_yield(
                actual,
                "cluster_penalty",
                "cluster_penalty",
                args.hit_threshold,
                definition.round_id,
            )
        )
        mean_controls.append(
            mean_decile_uncertainty_control(
                actual, args.hit_threshold, definition.round_id
            )
        )

        candidate_ei = -score_arrays["expected_improvement"]
        candidate_deterministic = -score_arrays["deterministic_improvement"]
        actual_ei = -actual["base_score"].to_numpy(dtype=float)
        actual_deterministic = np.maximum(
            args.hit_threshold - actual["pred_score"].to_numpy(dtype=float) - args.xi,
            0.0,
        )
        base_q05, base_q95 = np.quantile(actual["base_score"], [0.05, 0.95])
        penalty_q05, penalty_q95 = np.quantile(
            actual["cluster_penalty"], [0.05, 0.95]
        )
        candidate_rows.append(
            {
                "round": definition.round_id,
                "model_version": definition.model_version,
                "atlas_version": definition.atlas_version,
                "candidate_count": len(pool.identifiers),
                "pred_score_mean": float(np.mean(pool.means)),
                "pred_score_median": float(np.median(pool.means)),
                "epistemic_mean": float(np.mean(pool.epistemic)),
                "epistemic_median": float(np.median(pool.epistemic)),
                "candidate_predicted_hit_fraction": float(
                    np.mean(pool.means <= args.hit_threshold)
                ),
                "candidate_predicted_hit_count": int(
                    np.sum(pool.means <= args.hit_threshold)
                ),
                "candidate_expected_improvement_mean": float(np.mean(candidate_ei)),
                "candidate_uncertainty_increment_mean": float(
                    np.mean(candidate_ei - candidate_deterministic)
                ),
                "actual_epistemic_median": float(actual["epistemic_std"].median()),
                "actual_predicted_hit_fraction": float(
                    np.mean(actual["pred_score"] <= args.hit_threshold)
                ),
                "actual_expected_improvement_mean": float(np.mean(actual_ei)),
                "actual_uncertainty_increment_mean": float(
                    np.mean(actual_ei - actual_deterministic)
                ),
                "actual_penalty_mean": float(actual["cluster_penalty"].mean()),
                "actual_penalty_median": float(actual["cluster_penalty"].median()),
                "actual_base_q90_span": float(base_q95 - base_q05),
                "actual_penalty_q90_span": float(penalty_q95 - penalty_q05),
                "penalty_span_over_base_span": float(
                    (penalty_q95 - penalty_q05) / (base_q95 - base_q05)
                ),
                "actual_first_in_cluster_fraction": float(
                    np.mean(actual["cluster_count_before"] == 0)
                ),
                "actual_unique_clusters": int(actual["clusterid"].nunique()),
                "mean_ei_overlap": len(mean_set & ei_set),
                "ei_added_vs_mean": len(ei_set - mean_set),
                "deterministic_ei_overlap": len(
                    set(map(int, policy_ids["deterministic_improvement"])) & ei_set
                ),
                "ei_added_vs_deterministic": len(
                    ei_set
                    - set(map(int, policy_ids["deterministic_improvement"]))
                ),
                "diversity_added_vs_ei": len(actual_set - ei_set),
                "diversity_displaced_from_ei": len(ei_set - actual_set),
                "full_replay_actual_overlap": len(
                    set(map(int, policy_ids["full_ei_plus_diversity"])) & actual_set
                ),
            }
        )
        sample_rng = np.random.default_rng(args.random_seed + definition.round_id)
        sample = sample_rng.choice(
            len(pool.identifiers), min(200_000, len(pool.identifiers)), replace=False
        )
        candidate_samples.append(
            pd.DataFrame(
                {
                    "round": definition.round_id,
                    "pred_score": pool.means[sample],
                    "epistemic_std": pool.epistemic[sample],
                    "expected_improvement": candidate_ei[sample],
                }
            )
        )
        del info, words, popcounts, pool

    candidate_summary = pd.DataFrame(candidate_rows)
    overlaps = pd.concat(overlap_frames, ignore_index=True)
    policy_frame = pd.DataFrame(policy_metric_rows)
    group_frame = pd.concat(actual_groups, ignore_index=True)
    actual_frame = pd.concat(actual_members, ignore_index=True)
    uncertainty_frame = pd.concat(uncertainty_bins, ignore_index=True)
    ei_frame = pd.concat(ei_bins, ignore_index=True)
    penalty_frame = pd.concat(penalty_bins, ignore_index=True)
    mean_control = pd.concat(mean_controls, ignore_index=True)
    calibration = pd.DataFrame(calibration_rows)
    source_frame = pd.DataFrame(policy_source_rows)
    factorial = factorial_attribution(policy_frame)
    yield_decomposition = yield_composition_decomposition(group_frame)
    round_comparison, group_comparison, factorial_comparison = (
        lcb_ei_comparison_tables(
            args.lcb_attribution,
            candidate_summary,
            calibration,
            group_frame,
            factorial,
        )
    )
    strategy_decomposition = strategy_yield_decomposition(group_comparison)

    tables = {
        "round_candidate_summary.csv": candidate_summary,
        "policy_selection_overlap.csv": overlaps,
        "actual_selection_groups.csv": group_frame,
        "uncertainty_quantile_yield.csv": uncertainty_frame,
        "expected_improvement_quantile_yield.csv": ei_frame,
        "penalty_quantile_yield.csv": penalty_frame,
        "mean_decile_uncertainty_control.csv": mean_control,
        "uncertainty_calibration.csv": calibration,
        "counterfactual_policy_metrics.csv": policy_frame,
        "counterfactual_outcome_sources.csv": source_frame,
        "factorial_attribution.csv": factorial,
        "round_yield_composition_decomposition.csv": yield_decomposition,
        "lcb_ei_round_comparison.csv": round_comparison,
        "lcb_ei_group_comparison.csv": group_comparison,
        "lcb_ei_factorial_comparison.csv": factorial_comparison,
        "lcb_ei_yield_decomposition.csv": strategy_decomposition,
    }
    for filename, frame in tables.items():
        write_csv(output / filename, frame)
    actual_frame.to_csv(
        output / "actual_selection_membership.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(selected_id_rows).to_csv(
        output / "policy_selected_ids.csv.gz", index=False, compression="gzip"
    )
    pd.concat(candidate_samples, ignore_index=True).to_csv(
        output / "candidate_sample.csv.gz", index=False, compression="gzip"
    )

    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "definition": {
            "rounds": [definition.__dict__ for definition in definitions],
            "batch_size": args.batch_size,
            "hit_threshold": args.hit_threshold,
            "xi": args.xi,
            "cluster_lambda": args.cluster_lambda,
            "atlas_id": ATLAS_ID,
            "factorial_no_uncertainty": "deterministic improvement max(threshold - mean - xi, 0)",
            "factorial_with_uncertainty": "Gaussian expected improvement using epistemic standard deviation",
            "outcomes": (
                "EI docking score when present, otherwise LCB then Greedy score by "
                "reghash; unobserved counterfactual outcomes are not imputed"
            ),
        },
        "validation": {
            "historical_replay_exact": True,
            "round1_candidates": int(candidate_summary.iloc[0]["candidate_count"]),
            "round2_candidates": int(candidate_summary.iloc[1]["candidate_count"]),
            "round1_selected": int(
                actual_frame[actual_frame["round"] == 1]["spacehastenid"].nunique()
            ),
            "round2_selected": int(
                actual_frame[actual_frame["round"] == 2]["spacehastenid"].nunique()
            ),
        },
        "candidate_summary": candidate_summary.to_dict(orient="records"),
        "calibration": calibration.to_dict(orient="records"),
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return {
        "summary": summary,
        "candidate_samples": pd.concat(candidate_samples, ignore_index=True),
        "actual": actual_frame,
        "groups": group_frame,
        "uncertainty": uncertainty_frame,
        "ei_bins": ei_frame,
        "penalty": penalty_frame,
        "policies": policy_frame,
        "factorial": factorial,
        "round_comparison": round_comparison,
        "group_comparison": group_comparison,
        "output": output,
    }


GROUP_ORDER = (
    "mean_and_ei_core",
    "ei_uncertainty_promoted",
    "diversity_rescued_mean",
    "penalty_enabled_outside_both",
)
GROUP_LABELS = {
    "mean_and_ei_core": "Mean + EI core",
    "ei_uncertainty_promoted": "EI uncertainty-promoted",
    "diversity_rescued_mean": "Diversity-rescued mean",
    "penalty_enabled_outside_both": "Penalty-enabled outside both",
}
GROUP_COLORS = {
    "mean_and_ei_core": "#0072B2",
    "ei_uncertainty_promoted": "#E69F00",
    "diversity_rescued_mean": "#009E73",
    "penalty_enabled_outside_both": "#CC79A7",
}
POLICY_LABELS = {
    "mean_only": "Mean",
    "deterministic_improvement": "Deterministic\nimprovement",
    "expected_improvement": "Expected\nimprovement",
    "mean_plus_diversity": "Mean +\ndiversity",
    "deterministic_improvement_plus_diversity": "Deterministic +\ndiversity",
    "full_ei_plus_diversity": "EI +\ndiversity",
    "actual_ei_historical": "Historical EI",
}


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


def save_figure(figure: plt.Figure, output: Path, dpi: int) -> None:
    figure.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def make_figures(result: dict[str, Any], dpi: int) -> None:
    configure_style()
    output: Path = result["output"]
    candidate_samples: pd.DataFrame = result["candidate_samples"]
    actual: pd.DataFrame = result["actual"]
    groups: pd.DataFrame = result["groups"]
    uncertainty: pd.DataFrame = result["uncertainty"]
    policies: pd.DataFrame = result["policies"]
    factorial: pd.DataFrame = result["factorial"]
    candidate_summary = pd.DataFrame(result["summary"]["candidate_summary"])
    group_comparison: pd.DataFrame = result["group_comparison"]

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    for axis, round_id in zip(axes, (1, 2), strict=True):
        sample = candidate_samples[candidate_samples["round"] == round_id]
        selected = actual[actual["round"] == round_id]
        axis.hexbin(
            sample["pred_score"],
            sample["epistemic_std"],
            gridsize=65,
            mincnt=1,
            cmap="Greys",
            bins="log",
        )
        axis.scatter(
            selected["pred_score"],
            selected["epistemic_std"],
            s=1,
            alpha=0.12,
            color="#009E73",
            rasterized=True,
            label="Historical EI selections",
        )
        axis.axvline(-9.7, color="#555555", linestyle=":", linewidth=1)
        axis.set(
            xlabel="Predicted docking score (lower is better)",
            ylabel="Epistemic standard deviation",
            title=f"Round {round_id}",
        )
        axis.legend(frameon=False)
    figure.suptitle("EI selection across predicted mean and uncertainty", y=1.02)
    figure.tight_layout()
    save_figure(figure, output / "01_candidate_mean_uncertainty_selection", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for axis, round_id in zip(axes, (1, 2), strict=True):
        data = groups[groups["round"] == round_id].set_index("selection_group")
        rates = 100 * data.loc[list(GROUP_ORDER), "hit_rate"].to_numpy()
        counts = data.loc[list(GROUP_ORDER), "count"].to_numpy(dtype=int)
        x = np.arange(len(GROUP_ORDER))
        axis.bar(x, rates, color=[GROUP_COLORS[group] for group in GROUP_ORDER])
        axis.set_xticks(
            x,
            [GROUP_LABELS[group] for group in GROUP_ORDER],
            rotation=24,
            ha="right",
        )
        axis.set(title=f"Round {round_id}", ylabel="Observed hit rate (%)")
        for position, rate, count in zip(x, rates, counts, strict=True):
            axis.text(
                position,
                rate,
                f"{rate:.1f}%\nn={count:,}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
    figure.suptitle("Historical EI yield by selection mechanism", y=1.02)
    figure.tight_layout()
    save_figure(figure, output / "02_actual_yield_by_selection_mechanism", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for axis, round_id in zip(axes, (1, 2), strict=True):
        data = uncertainty[uncertainty["round"] == round_id]
        axis.plot(
            data["bin"],
            100 * data["hit_rate"],
            "o-",
            color="#E69F00",
            label="Hit rate",
        )
        secondary = axis.twinx()
        secondary.plot(
            data["bin"],
            data["mean_absolute_error"],
            "s--",
            color="#0072B2",
            label="Absolute error",
        )
        axis.set(
            xlabel="Epistemic uncertainty quintile (low to high)",
            ylabel="Hit rate (%)",
            title=f"Round {round_id}",
        )
        secondary.set_ylabel("Mean absolute prediction error")
    figure.suptitle("EI uncertainty, observed yield, and calibration", y=1.02)
    figure.tight_layout()
    save_figure(figure, output / "03_uncertainty_yield_calibration", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    x = np.arange(2)
    width = 0.35
    axes[0].bar(
        x - width / 2,
        candidate_summary["actual_base_q90_span"],
        width,
        label="EI base-score span",
        color="#0072B2",
    )
    axes[0].bar(
        x + width / 2,
        candidate_summary["actual_penalty_q90_span"],
        width,
        label="Cluster-penalty span",
        color="#009E73",
    )
    axes[0].set_xticks(x, ["Round 1", "Round 2"])
    axes[0].set_ylabel("Central 90% score span")
    axes[0].set_title("Acquisition score scale")
    axes[0].legend(frameon=False)
    axes[1].bar(
        x - width / 2,
        candidate_summary["ei_added_vs_mean"],
        width,
        label="EI replaces mean shortlist",
        color="#E69F00",
    )
    axes[1].bar(
        x + width / 2,
        candidate_summary["diversity_added_vs_ei"],
        width,
        label="Diversity replaces EI shortlist",
        color="#CC79A7",
    )
    axes[1].set_xticks(x, ["Round 1", "Round 2"])
    axes[1].set_ylabel("Compounds replaced from top 100,000")
    axes[1].set_title("Shortlist reshuffling")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, output / "04_score_scale_and_shortlist_replacement", dpi)

    metrics = (
        ("observed_hit_rate", "Observed-subset hit rate", 100),
        ("internal_diversity", "Internal diversity", 1),
        ("typed_scaffold_richness_per_compound", "Typed scaffolds per compound", 1),
        ("atlas_clusters_per_compound", "Atlas clusters per compound", 1),
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 6.6))
    policy_order = (
        "mean_only",
        "deterministic_improvement",
        "expected_improvement",
        "deterministic_improvement_plus_diversity",
        "full_ei_plus_diversity",
        "actual_ei_historical",
    )
    for axis, (metric, label, scale) in zip(axes.flat, metrics, strict=True):
        for round_id, marker in ((1, "o"), (2, "s")):
            data = policies[policies["round"] == round_id].set_index("policy")
            axis.plot(
                np.arange(len(policy_order)),
                scale * data.loc[list(policy_order), metric],
                marker=marker,
                label=f"Round {round_id}",
            )
        axis.set_xticks(
            np.arange(len(policy_order)),
            [POLICY_LABELS[policy] for policy in policy_order],
            rotation=25,
            ha="right",
        )
        axis.set_ylabel(label + (" (%)" if scale == 100 else ""))
    axes[0, 0].legend(frameon=False)
    figure.suptitle("EI policy replay: uncertainty and diversity", y=1.01)
    figure.text(
        0.5,
        0.005,
        "Counterfactual hit rates use observed EI/LCB/Greedy outcomes only; "
        "structural metrics cover all selections.",
        ha="center",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(figure, output / "05_policy_replay_potency_diversity", dpi)

    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.5))
    factorial_metrics = {
        "observed_hit_rate": "Observed-subset hit rate",
        "internal_diversity": "Internal diversity",
        "typed_scaffold_richness_per_compound": "Typed scaffolds per compound",
        "atlas_clusters_per_compound": "Atlas clusters per compound",
    }
    for axis, (metric, label) in zip(axes.flat, factorial_metrics.items(), strict=True):
        data = factorial[factorial["metric"] == metric].sort_values("round")
        positions = np.arange(len(data))
        width = 0.24
        axis.bar(
            positions - width,
            data["uncertainty_main_effect"],
            width,
            label="Uncertainty",
            color="#E69F00",
        )
        axis.bar(
            positions,
            data["diversity_main_effect"],
            width,
            label="Diversity",
            color="#009E73",
        )
        axis.bar(
            positions + width,
            data["interaction"],
            width,
            label="Interaction",
            color="#CC79A7",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(positions, [f"Round {value}" for value in data["round"]])
        axis.set_ylabel(f"Effect on {label}")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("EI factorial attribution", y=1.01)
    figure.tight_layout()
    save_figure(figure, output / "06_factorial_mechanism_decomposition", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    for axis, round_id in zip(axes, (1, 2), strict=True):
        data = group_comparison[group_comparison["round"] == round_id]
        positions = np.arange(len(data))
        width = 0.36
        axis.bar(
            positions - width / 2,
            100 * data["lcb_hit_rate"],
            width,
            label="LCB",
            color="#D55E00",
        )
        axis.bar(
            positions + width / 2,
            100 * data["ei_hit_rate"],
            width,
            label="EI",
            color="#009E73",
        )
        axis.set_xticks(
            positions,
            [GROUP_LABELS[value] for value in data["mechanism_group"]],
            rotation=25,
            ha="right",
        )
        axis.set(title=f"Round {round_id}", ylabel="Observed hit rate (%)")
    axes[0].legend(frameon=False)
    figure.suptitle("LCB versus EI yield within comparable mechanism groups", y=1.02)
    figure.tight_layout()
    save_figure(figure, output / "07_lcb_ei_mechanism_comparison", dpi)


def write_report(result: dict[str, Any]) -> None:
    output: Path = result["output"]
    summary = result["summary"]
    candidates = pd.DataFrame(summary["candidate_summary"]).set_index("round")
    calibration = pd.DataFrame(summary["calibration"]).set_index("round")
    groups: pd.DataFrame = result["groups"]
    policies: pd.DataFrame = result["policies"]
    factorial: pd.DataFrame = result["factorial"]
    comparison: pd.DataFrame = result["group_comparison"]
    decomposition = pd.read_csv(output / "round_yield_composition_decomposition.csv").iloc[0]
    strategy_decomposition = pd.read_csv(
        output / "lcb_ei_yield_decomposition.csv"
    ).set_index("round")

    lines = [
        "# TGFR1 SVDKL-EI Acquisition Attribution",
        "",
        "## Executive Conclusion",
        "",
        (
            "EI made two distinct changes to acquisition. Gaussian epistemic uncertainty "
            "reshaped the unpenalized shortlist through expected improvement, while the "
            "dynamic cluster penalty spread that shortlist over chemical space. The replay "
            "separates these mechanisms exactly for structural selection; counterfactual "
            "yield remains descriptive because many alternatives were never docked."
        ),
        (
            "The dominant low-yield mechanism was the diversity penalty, not uncertainty "
            "alone. It displaced most of the unpenalized EI shortlist and made the majority "
            "of each historical batch penalty-enabled chemistry outside both the mean and "
            "EI top-100k references. Those compounds were highly diverse but had the lowest "
            "observed hit rates."
        ),
        "",
        "## Production Decision Rule",
        "",
        "For candidate mean `mu`, epistemic standard deviation `sigma`, threshold `t=-9.7`, and `xi=0`:",
        "",
        "`EI = (t - mu) Phi((t - mu)/sigma) + sigma phi((t - mu)/sigma)`",
        "",
        "SpaceHASTEN minimized `-EI + 0.5 log(1+n_cluster)` dynamically within each 100,000-compound batch.",
        "",
        "The historical replay reproduced every selected ID, rank, cluster count, base score, and penalty in both rounds.",
        "",
        "## Candidate And Selection Summary",
        "",
        "| Round | Candidates | EI replacements vs mean | Diversity replacements vs EI | Selected clusters | First selections | Penalty/base span | Hit rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    actual_rates = policies[policies["policy"] == "actual_ei_historical"].set_index(
        "round"
    )
    for round_id in (1, 2):
        row = candidates.loc[round_id]
        lines.append(
            f"| {round_id} | {int(row.candidate_count):,} | "
            f"{int(row.ei_added_vs_mean):,} | {int(row.diversity_added_vs_ei):,} | "
            f"{int(row.actual_unique_clusters):,} | "
            f"{100 * row.actual_first_in_cluster_fraction:.1f}% | "
            f"{row.penalty_span_over_base_span:.2f} | "
            f"{100 * actual_rates.loc[round_id, 'observed_hit_rate']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Role Of Uncertainty",
            "",
            (
                "Expected improvement does not apply a linear uncertainty bonus. It "
                "integrates the Gaussian probability and magnitude of crossing the fixed "
                "-9.7 threshold. The deterministic-improvement policies set `sigma=0`; "
                "their contrast with EI is the uncertainty effect used in the factorial."
            ),
            "",
            "| Round | Candidate sigma median | Selected sigma median | Sigma-error Spearman | MAE |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for round_id in (1, 2):
        lines.append(
            f"| {round_id} | {candidates.loc[round_id, 'epistemic_median']:.4f} | "
            f"{candidates.loc[round_id, 'actual_epistemic_median']:.4f} | "
            f"{calibration.loc[round_id, 'spearman_uncertainty_absolute_error']:.3f} | "
            f"{calibration.loc[round_id, 'mean_absolute_error']:.3f} |"
        )
    for round_id in (1, 2):
        row = candidates.loc[round_id]
        uncertainty_share = (
            row.actual_uncertainty_increment_mean / row.actual_expected_improvement_mean
        )
        lines.append(
            f"- Round {round_id}: EI replaced {int(row.ei_added_vs_mean):,} mean-only "
            f"shortlist members. Uncertainty contributed {100 * uncertainty_share:.1f}% "
            "of mean EI among the historical selections."
        )
    lines.extend(
        [
            "",
            (
                f"Only {int(candidates.loc[1, 'candidate_predicted_hit_count']):,} of the "
                "round-1 candidates had predicted means at or below -9.7. Consequently, "
                "deterministic improvement has a large zero-score tie when forced to return "
                "100,000 compounds. Its round-1 factorial uncertainty effect partly reflects "
                "resolution of this tie; mean-only versus EI is the cleaner direct shortlist "
                "comparison."
            ),
            "",
            "## Role Of The Diversity Penalty",
            "",
        ]
    )
    for round_id in (1, 2):
        row = candidates.loc[round_id]
        unpenalized = policies[
            (policies["round"] == round_id)
            & (policies["policy"] == "expected_improvement")
        ].iloc[0]
        full = policies[
            (policies["round"] == round_id)
            & (policies["policy"] == "full_ei_plus_diversity")
        ].iloc[0]
        lines.append(
            f"- Round {round_id}: the penalty replaced "
            f"{int(row.diversity_added_vs_ei):,} of 100,000 unpenalized EI selections, "
            f"increased internal diversity from {unpenalized.internal_diversity:.4f} to "
            f"{full.internal_diversity:.4f}, and increased atlas occupancy from "
            f"{int(unpenalized.atlas_occupied_clusters):,} to "
            f"{int(full.atlas_occupied_clusters):,} clusters."
        )
    lines.extend(
        [
            (
                "The same fixed `lambda=0.5` was much stronger relative to EI's compressed "
                f"base-score scale: the penalty/base central-span ratio was "
                f"{candidates.loc[1, 'penalty_span_over_base_span']:.2f} in round 1 and "
                f"{candidates.loc[2, 'penalty_span_over_base_span']:.2f} in round 2. This "
                "explains why EI spread over far more clusters than LCB."
            ),
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## Historical Selection Groups",
            "",
            "Groups compare each actual EI selection with the unpenalized mean-only and EI top-100k lists.",
            "",
            "| Round | Group | Batch fraction | Hit rate | Internal diversity |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in groups.itertuples(index=False):
        lines.append(
            f"| {row.round} | {GROUP_LABELS[row.selection_group]} | "
            f"{100 * row.batch_fraction:.2f}% | {100 * row.hit_rate:.2f}% | "
            f"{row.internal_diversity:.4f} |"
        )
    for round_id in (1, 2):
        broad = groups[
            (groups["round"] == round_id)
            & (groups["selection_group"] == "penalty_enabled_outside_both")
        ].iloc[0]
        lines.append(
            f"- Round {round_id}: penalty-enabled selections were "
            f"{100 * broad.batch_fraction:.1f}% of the batch but had only a "
            f"{100 * broad.hit_rate:.1f}% hit rate. They contributed "
            f"{100 * broad.hit_contribution_fraction:.1f}% of observed hits."
        )
    lines.extend(
        [
            "",
            "## Factorial Replay",
            "",
            (
                "The 2x2 replay crosses deterministic improvement versus uncertainty-aware "
                "EI with no diversity penalty versus the dynamic cluster penalty. Structural "
                "metrics use complete selections. Hit-rate effects use only observed EI, "
                "LCB, or Greedy outcomes and must not be read as exact causal yield effects."
            ),
            "",
            "| Round | Metric | Uncertainty effect | Diversity effect | Interaction |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for metric in (
        "observed_hit_rate",
        "internal_diversity",
        "typed_scaffold_richness_per_compound",
        "atlas_clusters_per_compound",
    ):
        for round_id in (1, 2):
            row = factorial[
                (factorial["round"] == round_id) & (factorial["metric"] == metric)
            ].iloc[0]
            lines.append(
                f"| {round_id} | {metric} | {row.uncertainty_main_effect:.4f} | "
                f"{row.diversity_main_effect:.4f} | {row.interaction:.4f} |"
            )
    lines.extend(
        [
            "",
            "### Counterfactual Outcome Coverage",
            "",
            "| Round | Policy | Outcome coverage | Observed-subset hit rate |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in policies.itertuples(index=False):
        if row.policy not in {
            "mean_only",
            "expected_improvement",
            "mean_plus_diversity",
            "full_ei_plus_diversity",
        }:
            continue
        lines.append(
            f"| {row.round} | {POLICY_LABELS[row.policy].replace(chr(10), ' ')} | "
            f"{100 * row.outcome_coverage:.1f}% | {100 * row.observed_hit_rate:.1f}% |"
        )
    lines.append(
        "Coverage differs substantially among policies, so these observed-subset hit rates "
        "support direction and mechanism but are not unbiased counterfactual yield estimates."
    )
    lines.extend(
        [
            "",
            "## Why Round 2 Improved",
            "",
            (
                f"The hit-rate increase was {100 * decomposition.total_change:.2f} percentage "
                f"points. A Shapley composition/yield decomposition assigns "
                f"{100 * decomposition.composition_effect_shapley:.2f} points "
                f"({100 * decomposition.composition_share_of_change:.1f}%) to changed group "
                f"proportions and {100 * decomposition.within_group_yield_effect_shapley:.2f} "
                f"points ({100 * decomposition.within_group_share_of_change:.1f}%) to improved "
                "yield within the same mechanisms."
            ),
            "",
            "## LCB Versus EI",
            "",
            "| Round | Comparable group | LCB hit rate | EI hit rate | Difference |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.round} | {GROUP_LABELS[row.mechanism_group]} | "
            f"{100 * row.lcb_hit_rate:.2f}% | {100 * row.ei_hit_rate:.2f}% | "
            f"{100 * row.ei_minus_lcb_hit_rate:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            (
                "The exploitation-supported groups had similar LCB and EI yields in round 1. "
                "The large aggregate EI deficit arose because EI assigned 84.5% of round 1 "
                "and 79.1% of round 2 to the penalty-enabled outside-both group, versus "
                "60.8% and 55.5% for LCB, and EI's broad group converted substantially worse."
            ),
            "",
            "### Decomposition Of The EI-LCB Yield Gap",
            "",
            "| Round | EI minus LCB | Group-composition effect | Within-group yield effect |",
            "|---:|---:|---:|---:|",
        ]
    )
    for round_id in (1, 2):
        row = strategy_decomposition.loc[round_id]
        lines.append(
            f"| {round_id} | {100 * row.ei_minus_lcb:+.2f} pp | "
            f"{100 * row.composition_effect_shapley:+.2f} pp "
            f"({100 * row.composition_share_of_difference:.1f}%) | "
            f"{100 * row.within_group_yield_effect_shapley:+.2f} pp "
            f"({100 * row.within_group_share_of_difference:.1f}%) |"
        )
    lines.extend(
        [
            "",
            (
                "Thus the EI deficit was not solely a composition effect. In round 1, the "
                "larger exploratory group and poorer yield within comparable groups each "
                "explained about half the gap. By round 2, poorer within-group conversion "
                "was the larger component."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation Limits",
            "",
            "- Historical group yields are directly observed; their labels describe shortlist membership, not isolated causal treatments.",
            "- Counterfactual structural effects are exact, but counterfactual hit rates have policy-dependent outcome coverage.",
            "- Round-2 models and candidate pools differ between LCB and EI because each was retrained on its own round-1 batch.",
            "- Causal attribution of learning requires fixed external evaluation or retraining ablations.",
            "",
            "## Recommended Follow-up",
            "",
            "Replay or prospectively test EI with a normalized cluster penalty, including `lambda=0`, a smaller fixed lambda, and a penalty scaled to the EI base-score spread. Evaluate all policies on one balanced docking sample rather than imputing unobserved outcomes.",
            "",
        ]
    )
    (output / "EI_ACQUISITION_ATTRIBUTION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def validate_outputs(output: Path, dpi: int) -> None:
    required = {
        "round_candidate_summary.csv",
        "policy_selection_overlap.csv",
        "actual_selection_groups.csv",
        "actual_selection_membership.csv.gz",
        "uncertainty_quantile_yield.csv",
        "expected_improvement_quantile_yield.csv",
        "penalty_quantile_yield.csv",
        "mean_decile_uncertainty_control.csv",
        "uncertainty_calibration.csv",
        "counterfactual_policy_metrics.csv",
        "counterfactual_outcome_sources.csv",
        "factorial_attribution.csv",
        "round_yield_composition_decomposition.csv",
        "lcb_ei_round_comparison.csv",
        "lcb_ei_group_comparison.csv",
        "lcb_ei_factorial_comparison.csv",
        "lcb_ei_yield_decomposition.csv",
        "policy_selected_ids.csv.gz",
        "candidate_sample.csv.gz",
        "analysis_summary.json",
        "EI_ACQUISITION_ATTRIBUTION_REPORT.md",
    }
    stems = (
        "01_candidate_mean_uncertainty_selection",
        "02_actual_yield_by_selection_mechanism",
        "03_uncertainty_yield_calibration",
        "04_score_scale_and_shortlist_replacement",
        "05_policy_replay_potency_diversity",
        "06_factorial_mechanism_decomposition",
        "07_lcb_ei_mechanism_comparison",
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
        raise ValueError(f"missing or empty EI attribution outputs: {missing}")
    LOGGER.info("Validated %d EI attribution artifacts at %d dpi", len(required), dpi)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    try:
        args = parse_args()
        result = calculate(args)
        make_figures(result, args.dpi)
        write_report(result)
        validate_outputs(result["output"], args.dpi)
    except Exception:
        LOGGER.exception("EI acquisition attribution failed")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ei-db", type=Path, required=True)
    parser.add_argument("--lcb-docked-db", type=Path, required=True)
    parser.add_argument("--greedy-docked-db", type=Path, required=True)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--lcb-attribution", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--cluster-lambda", type=float, default=0.5)
    parser.add_argument("--hit-threshold", type=float, default=-9.7)
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--pair-samples", type=int, default=250_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if min(args.batch_size, args.pair_samples, args.dpi) < 1:
        parser.error("batch size, pair samples, and DPI must be positive")
    if args.cluster_lambda < 0 or args.xi < 0:
        parser.error("cluster lambda and xi must be non-negative")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
