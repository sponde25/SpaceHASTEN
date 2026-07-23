#!/usr/bin/env python3
"""Decompose LCB acquisition into mean, uncertainty, and diversity effects."""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from tqdm import tqdm

LOGGER = logging.getLogger("analyze_lcb_acquisition_attribution")
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#777777"
POLICY_COLORS = {
    "mean_only": BLUE,
    "uncertainty_only": ORANGE,
    "diversity_only": GREEN,
    "full_lcb_atlas": PURPLE,
    "actual_lcb_historical": "#000000",
}
POLICY_LABELS = {
    "mean_only": "Mean only",
    "uncertainty_only": "Mean - uncertainty",
    "diversity_only": "Mean + diversity",
    "full_lcb_atlas": "Full LCB + diversity",
    "actual_lcb_historical": "Actual historical LCB",
}
ACTUAL_GROUPS = (
    "mean_and_uncertainty_core",
    "uncertainty_promoted",
    "diversity_rescued_mean",
    "diversity_only",
)
ACTUAL_GROUP_LABELS = {
    "mean_and_uncertainty_core": "Mean + uncertainty core",
    "uncertainty_promoted": "Uncertainty-promoted",
    "diversity_rescued_mean": "Diversity-rescued mean",
    "diversity_only": "Diversity-only",
}
ACTUAL_GROUP_COLORS = {
    "mean_and_uncertainty_core": BLUE,
    "uncertainty_promoted": ORANGE,
    "diversity_rescued_mean": GREEN,
    "diversity_only": PURPLE,
}
BYTE_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


@dataclass(frozen=True)
class RoundDefinition:
    round_id: int
    model_version: int
    upper_id: int
    candidate_filter: str


@dataclass
class CandidatePool:
    identifiers: np.ndarray
    means: np.ndarray
    epistemic: np.ndarray
    atlas_clusters: np.ndarray
    lcb_dock_scores: np.ndarray
    final_dock_iterations: np.ndarray


@dataclass(frozen=True)
class PolicySelection:
    name: str
    indices: np.ndarray
    cluster_count_before: np.ndarray
    penalties: np.ndarray


def readonly_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcb-db", type=Path, required=True)
    parser.add_argument("--greedy-docked-db", type=Path, required=True)
    parser.add_argument("--atlas-db", type=Path, required=True)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--nearest-seed", type=Path, required=True)
    parser.add_argument("--umap-coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--cluster-lambda", type=float, default=0.5)
    parser.add_argument("--hit-cutoff", type=float, default=-9.7)
    parser.add_argument("--pair-samples", type=int, default=1_000_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if min(args.batch_size, args.pair_samples, args.dpi) < 1:
        parser.error("batch size, pair samples, and dpi must be positive")
    if args.cluster_lambda < 0:
        parser.error("cluster lambda must be non-negative")
    return args


def load_candidate_pool(
    database: Path,
    atlas_database: Path,
    definition: RoundDefinition,
) -> CandidatePool:
    started = time.monotonic()
    with readonly_connection(database) as connection:
        connection.execute(
            "ATTACH DATABASE ? AS atlas",
            (f"file:{atlas_database.resolve()}?mode=ro",),
        )
        where = (
            f"d.spacehastenid > 995924 AND d.spacehastenid <= ? AND ({definition.candidate_filter})"
        )
        count = int(
            connection.execute(
                "SELECT COUNT(*) FROM data AS d "
                "JOIN predictions AS p ON p.spacehastenid=d.spacehastenid "
                "AND p.model_version=? "
                "JOIN atlas.cluster_atlas_assignments AS a "
                "ON a.atlas_id='morgan-r2-1024-t040' "
                "AND a.spacehastenid=d.spacehastenid "
                f"WHERE {where}",
                (definition.model_version, definition.upper_id),
            ).fetchone()[0]
        )
        identifiers = np.empty(count, dtype=np.int64)
        means = np.empty(count, dtype=np.float32)
        epistemic = np.empty(count, dtype=np.float32)
        clusters = np.empty(count, dtype=np.int64)
        dock_scores = np.empty(count, dtype=np.float32)
        dock_scores.fill(np.nan)
        dock_iterations = np.empty(count, dtype=np.int8)
        query = (
            "SELECT d.spacehastenid,p.pred_score,p.epistemic_std,a.clusterid,"
            "d.dock_score,COALESCE(d.dock_iteration,-1) FROM data AS d "
            "JOIN predictions AS p ON p.spacehastenid=d.spacehastenid "
            "AND p.model_version=? "
            "JOIN atlas.cluster_atlas_assignments AS a "
            "ON a.atlas_id='morgan-r2-1024-t040' "
            "AND a.spacehastenid=d.spacehastenid "
            f"WHERE {where} ORDER BY d.spacehastenid"
        )
        rows = connection.execute(query, (definition.model_version, definition.upper_id))
        for index, (sid, mean, sigma, cluster, score, iteration) in enumerate(
            tqdm(rows, total=count, desc=f"Round {definition.round_id} candidates", unit="mol")
        ):
            identifiers[index] = int(sid)
            means[index] = float(mean)
            epistemic[index] = float(sigma)
            clusters[index] = int(cluster)
            if score is not None:
                dock_scores[index] = float(score)
            dock_iterations[index] = int(iteration)
    if len(np.unique(identifiers)) != count or not np.all(np.diff(identifiers) > 0):
        raise ValueError(f"round {definition.round_id}: candidate IDs are not unique and sorted")
    if not np.isfinite(means).all() or not np.isfinite(epistemic).all() or np.any(epistemic < 0):
        raise ValueError(f"round {definition.round_id}: invalid predictions")
    LOGGER.info(
        "Loaded round %d candidate pool: %d rows in %.1fs",
        definition.round_id,
        count,
        time.monotonic() - started,
    )
    return CandidatePool(identifiers, means, epistemic, clusters, dock_scores, dock_iterations)


def load_actual_acquisition(path: Path, expected_round: int, expected_size: int) -> pd.DataFrame:
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
        "lcb_beta",
        "cluster_lambda",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"missing acquisition columns: {sorted(required - set(frame.columns))}")
    if len(frame) != expected_size or frame.spacehastenid.duplicated().any():
        raise ValueError(f"round {expected_round}: invalid acquisition size or duplicate IDs")
    if not np.array_equal(frame["rank"].to_numpy(), np.arange(1, expected_size + 1)):
        raise ValueError(f"round {expected_round}: acquisition ranks are not contiguous")
    if set(frame.method) != {"lcb"} or set(frame.model_version) != {expected_round - 1}:
        raise ValueError(f"round {expected_round}: unexpected method/model version")
    expected_base = frame.pred_score - frame.lcb_beta * frame.epistemic_std
    expected_penalty = frame.cluster_lambda * np.log1p(frame.cluster_count_before)
    if not np.allclose(frame.base_score, expected_base, atol=1e-7):
        raise ValueError(f"round {expected_round}: invalid historical LCB base scores")
    if not np.allclose(frame.cluster_penalty, expected_penalty, atol=1e-12):
        raise ValueError(f"round {expected_round}: invalid historical cluster penalties")
    if not np.allclose(frame.penalized_score, frame.base_score + frame.cluster_penalty, atol=1e-9):
        raise ValueError(f"round {expected_round}: invalid historical penalized scores")
    return frame


def deterministic_top_k(base_scores: np.ndarray, identifiers: np.ndarray, count: int) -> np.ndarray:
    return np.lexsort((identifiers, base_scores))[:count].astype(np.int64, copy=False)


def penalized_top_k(
    base_scores: np.ndarray,
    identifiers: np.ndarray,
    clusters: np.ndarray,
    count: int,
    cluster_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((identifiers, base_scores, clusters))
    ordered_clusters = clusters[order]
    starts = np.flatnonzero(np.r_[True, ordered_clusters[1:] != ordered_clusters[:-1]])
    stops = np.r_[starts[1:], len(order)]
    heap: list[tuple[float, float, int, int, int, int]] = []
    for start, stop in zip(starts, stops, strict=True):
        candidate_index = int(order[start])
        heapq.heappush(
            heap,
            (
                float(base_scores[candidate_index]),
                float(base_scores[candidate_index]),
                int(identifiers[candidate_index]),
                int(start),
                int(stop),
                0,
            ),
        )
    selected = np.empty(count, dtype=np.int64)
    counts_before = np.empty(count, dtype=np.int32)
    penalties = np.empty(count, dtype=np.float64)
    for selected_index in range(count):
        penalized, base, _sid, position, stop, count_before = heapq.heappop(heap)
        candidate_index = int(order[position])
        selected[selected_index] = candidate_index
        counts_before[selected_index] = count_before
        penalties[selected_index] = penalized - base
        next_position = position + 1
        if next_position < stop:
            next_candidate = int(order[next_position])
            next_count = count_before + 1
            next_base = float(base_scores[next_candidate])
            heapq.heappush(
                heap,
                (
                    next_base + cluster_lambda * math.log1p(next_count),
                    next_base,
                    int(identifiers[next_candidate]),
                    next_position,
                    stop,
                    next_count,
                ),
            )
    return selected, counts_before, penalties


def select_policies(
    pool: CandidatePool, batch_size: int, cluster_lambda: float
) -> dict[str, PolicySelection]:
    mean_scores = pool.means.astype(np.float64)
    lcb_scores = mean_scores - pool.epistemic.astype(np.float64)
    zero_counts = np.zeros(batch_size, dtype=np.int32)
    zero_penalties = np.zeros(batch_size, dtype=np.float64)
    mean_indices = deterministic_top_k(mean_scores, pool.identifiers, batch_size)
    uncertainty_indices = deterministic_top_k(lcb_scores, pool.identifiers, batch_size)
    diversity_indices, diversity_counts, diversity_penalties = penalized_top_k(
        mean_scores, pool.identifiers, pool.atlas_clusters, batch_size, cluster_lambda
    )
    full_indices, full_counts, full_penalties = penalized_top_k(
        lcb_scores, pool.identifiers, pool.atlas_clusters, batch_size, cluster_lambda
    )
    return {
        "mean_only": PolicySelection("mean_only", mean_indices, zero_counts, zero_penalties),
        "uncertainty_only": PolicySelection(
            "uncertainty_only", uncertainty_indices, zero_counts.copy(), zero_penalties.copy()
        ),
        "diversity_only": PolicySelection(
            "diversity_only", diversity_indices, diversity_counts, diversity_penalties
        ),
        "full_lcb_atlas": PolicySelection(
            "full_lcb_atlas", full_indices, full_counts, full_penalties
        ),
    }


def candidate_positions(candidate_ids: np.ndarray, selected_ids: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(candidate_ids, selected_ids)
    if np.any(positions == len(candidate_ids)) or not np.array_equal(
        candidate_ids[positions], selected_ids
    ):
        raise ValueError("selected IDs are missing from candidate pool")
    return positions


def load_selected_compounds(
    database: Path,
    identifiers: np.ndarray,
    greedy_scores: dict[str, float],
) -> pd.DataFrame:
    workspace = sqlite3.connect(":memory:", uri=True)
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
    if len(frame) != len(identifiers) or frame.reghash.isna().any() or frame.smiles.isna().any():
        raise ValueError("failed to load all selected compound structures")
    greedy = frame.reghash.map(greedy_scores)
    frame["observed_score"] = frame.dock_score.fillna(greedy)
    frame["outcome_source"] = np.where(
        frame.dock_score.notna(), "lcb", np.where(greedy.notna(), "greedy", "unobserved")
    )
    return frame


def scaffold(smiles: str) -> tuple[str, str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"unparseable SMILES: {smiles}")
    core = MurckoScaffold.GetScaffoldForMol(molecule)
    if core.GetNumAtoms() == 0:
        return "[ACYCLIC]", "[ACYCLIC]"
    typed = Chem.MolToSmiles(core, canonical=True, isomericSmiles=False)
    generic = Chem.MolToSmiles(
        MurckoScaffold.MakeScaffoldGeneric(core), canonical=True, isomericSmiles=False
    )
    return typed, generic


def add_structures(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    result = frame.copy()
    scaffolds = [
        scaffold(smiles)
        for smiles in tqdm(result.smiles, total=len(result), desc="Selected scaffolds", unit="mol")
    ]
    result[["typed_scaffold", "generic_framework"]] = pd.DataFrame(scaffolds, index=result.index)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fingerprints = [
        generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        for smiles in tqdm(
            result.smiles, total=len(result), desc="Selected fingerprints", unit="mol"
        )
    ]
    packed = b"".join(DataStructs.BitVectToBinaryText(fp) for fp in fingerprints)
    words = np.frombuffer(packed, dtype=np.uint64).reshape(len(result), 16)
    popcounts = np.asarray([fp.GetNumOnBits() for fp in fingerprints], dtype=np.int16)
    result["structure_index"] = np.arange(len(result), dtype=np.int64)
    return result, words, popcounts


def internal_diversity(
    row_indices: np.ndarray,
    words: np.ndarray,
    popcounts: np.ndarray,
    pair_samples: int,
    rng: np.random.Generator,
) -> float:
    total = 0.0
    completed = 0
    while completed < pair_samples:
        size = min(250_000, pair_samples - completed)
        left = rng.integers(0, len(row_indices), size=size)
        right = rng.integers(0, len(row_indices), size=size)
        equal = left == right
        while equal.any():
            right[equal] = rng.integers(0, len(row_indices), size=int(equal.sum()))
            equal = left == right
        left_rows, right_rows = row_indices[left], row_indices[right]
        intersection = BYTE_POPCOUNT[
            np.bitwise_and(words[left_rows], words[right_rows]).view(np.uint8)
        ].sum(axis=1)
        union = popcounts[left_rows] + popcounts[right_rows] - intersection
        total += float(np.sum(intersection / union))
        completed += size
    return 1.0 - total / pair_samples


def family_metrics(values: pd.Series, prefix: str) -> dict[str, float | int]:
    counts = values.value_counts().to_numpy()
    probabilities = counts / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return {
        f"{prefix}_richness": len(counts),
        f"{prefix}_richness_per_compound": len(counts) / counts.sum(),
        f"{prefix}_largest_fraction": counts.max() / counts.sum(),
        f"{prefix}_entropy": entropy,
        f"{prefix}_normalized_entropy": entropy / math.log(len(counts)) if len(counts) > 1 else 0,
    }


def cluster_metrics(clusters: np.ndarray) -> dict[str, float | int]:
    _, counts = np.unique(clusters, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return {
        "atlas_occupied_clusters": len(counts),
        "atlas_clusters_per_compound": len(counts) / counts.sum(),
        "atlas_largest_fraction": counts.max() / counts.sum(),
        "atlas_entropy": entropy,
        "atlas_normalized_entropy": entropy / math.log(len(counts)) if len(counts) > 1 else 0,
    }


def policy_metrics(
    round_id: int,
    name: str,
    selected_ids: np.ndarray,
    selected_clusters: np.ndarray,
    info: pd.DataFrame,
    words: np.ndarray,
    popcounts: np.ndarray,
    cutoff: float,
    pair_samples: int,
    random_seed: int,
) -> dict[str, object]:
    indexed = info.set_index("spacehastenid")
    selected = indexed.loc[selected_ids]
    structure_indices = selected.structure_index.to_numpy(dtype=np.int64)
    observed = selected.observed_score.notna().to_numpy()
    observed_scores = selected.observed_score.to_numpy(dtype=float)[observed]
    row: dict[str, object] = {
        "round": round_id,
        "policy": name,
        "selected": len(selected),
        "observed_outcomes": int(observed.sum()),
        "outcome_coverage": float(observed.mean()),
        "observed_hit_count": int(np.sum(observed_scores <= cutoff)),
        "observed_hit_rate": float(np.mean(observed_scores <= cutoff))
        if len(observed_scores)
        else math.nan,
        "observed_mean_dock_score": float(np.mean(observed_scores))
        if len(observed_scores)
        else math.nan,
        "internal_diversity": internal_diversity(
            structure_indices,
            words,
            popcounts,
            pair_samples,
            np.random.default_rng(random_seed),
        ),
        **family_metrics(selected.typed_scaffold, "typed_scaffold"),
        **family_metrics(selected.generic_framework, "generic_framework"),
        **cluster_metrics(selected_clusters),
    }
    return row


def load_greedy_score_map(path: Path) -> dict[str, float]:
    with readonly_connection(path) as connection:
        rows = connection.execute(
            "SELECT reghash,dock_score FROM data WHERE dock_iteration > 0 "
            "AND dock_score IS NOT NULL"
        )
        return {str(reghash): float(score) for reghash, score in rows}


def load_vector_artifact(path: Path, value: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        identifiers = data["spacehastenid"].astype(np.int64)
        values = data[value].copy()
    order = np.argsort(identifiers)
    return identifiers[order], values[order]


def map_values(
    artifact_ids: np.ndarray,
    artifact_values: np.ndarray,
    identifiers: np.ndarray,
    *,
    allow_missing: bool = False,
) -> np.ndarray:
    positions = np.searchsorted(artifact_ids, identifiers)
    valid = (positions < len(artifact_ids)) & (
        artifact_ids[np.minimum(positions, len(artifact_ids) - 1)] == identifiers
    )
    if not valid.all() and not allow_missing:
        raise ValueError(f"{np.count_nonzero(~valid)} selected IDs missing from artifact")
    output_shape = (len(identifiers), *artifact_values.shape[1:])
    output = np.full(output_shape, np.nan, dtype=np.float32)
    output[valid] = artifact_values[positions[valid]]
    return output


def summarize_actual_groups(
    actual: pd.DataFrame,
    cutoff: float,
    pair_samples: int,
    words: np.ndarray,
    popcounts: np.ndarray,
    round_id: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in ACTUAL_GROUPS:
        frame = actual[actual.selection_group == group]
        lcb_observed = frame.dock_score.notna()
        scores = frame.loc[lcb_observed, "dock_score"].to_numpy(dtype=float)
        structure_indices = frame.structure_index.to_numpy(dtype=np.int64)
        metrics: dict[str, object] = {
            "round": round_id,
            "selection_group": group,
            "count": len(frame),
            "batch_fraction": len(frame) / len(actual),
            "observed_outcomes": int(lcb_observed.sum()),
            "hit_count": int(np.sum(scores <= cutoff)),
            "hit_rate": float(np.mean(scores <= cutoff)) if len(scores) else math.nan,
            "mean_dock_score": float(np.mean(scores)) if len(scores) else math.nan,
            "median_dock_score": float(np.median(scores)) if len(scores) else math.nan,
            "mean_pred_score": frame.pred_score.mean(),
            "mean_epistemic_std": frame.epistemic_std.mean(),
            "mean_cluster_penalty": frame.cluster_penalty.mean(),
            "internal_diversity": internal_diversity(
                structure_indices,
                words,
                popcounts,
                pair_samples,
                np.random.default_rng(10_000 + round_id * 10 + ACTUAL_GROUPS.index(group)),
            )
            if len(frame) > 1
            else math.nan,
            **(family_metrics(frame.typed_scaffold, "typed_scaffold") if len(frame) else {}),
            **(family_metrics(frame.generic_framework, "generic_framework") if len(frame) else {}),
            **(cluster_metrics(frame.atlas_cluster.to_numpy(dtype=np.int64)) if len(frame) else {}),
            "nearest_seed_mean": frame.nearest_seed.mean(),
            "nearest_seed_median": frame.nearest_seed.median(),
            "umap_grid_cells": int(
                np.count_nonzero(
                    np.histogram2d(
                        frame.loc[frame.umap_x.notna(), "umap_x"],
                        frame.loc[frame.umap_y.notna(), "umap_y"],
                        bins=60,
                    )[0]
                )
            ),
        }
        rows.append(metrics)
    return pd.DataFrame(rows)


def quantile_summary(
    frame: pd.DataFrame,
    column: str,
    label: str,
    cutoff: float,
    round_id: int,
    bins: int = 5,
) -> pd.DataFrame:
    result = frame[frame.dock_score.notna()].copy()
    result["bin"] = pd.qcut(result[column], bins, labels=False, duplicates="drop") + 1
    rows = []
    for bin_id, group in result.groupby("bin"):
        residual = group.dock_score - group.pred_score
        rows.append(
            {
                "round": round_id,
                "bin_type": label,
                "bin": int(bin_id),
                "count": len(group),
                "value_min": group[column].min(),
                "value_median": group[column].median(),
                "value_max": group[column].max(),
                "hit_count": int(np.sum(group.dock_score <= cutoff)),
                "hit_rate": float(np.mean(group.dock_score <= cutoff)),
                "mean_dock_score": group.dock_score.mean(),
                "mean_residual": residual.mean(),
                "median_residual": residual.median(),
                "mean_absolute_error": np.abs(residual).mean(),
                "favorable_surprise_rate_residual_lt_minus1": float(np.mean(residual < -1.0)),
            }
        )
    return pd.DataFrame(rows)


def mean_decile_uncertainty_summary(
    frame: pd.DataFrame, cutoff: float, round_id: int
) -> pd.DataFrame:
    result = frame[frame.dock_score.notna()].copy()
    result["mean_decile"] = pd.qcut(result.pred_score, 10, labels=False, duplicates="drop") + 1
    result["uncertainty_half"] = result.groupby("mean_decile").epistemic_std.transform(
        lambda values: np.where(values >= values.median(), "high", "low")
    )
    rows = []
    for (decile, half), group in result.groupby(["mean_decile", "uncertainty_half"]):
        residual = group.dock_score - group.pred_score
        rows.append(
            {
                "round": round_id,
                "mean_decile": int(decile),
                "uncertainty_half": half,
                "count": len(group),
                "mean_pred_score": group.pred_score.mean(),
                "mean_epistemic_std": group.epistemic_std.mean(),
                "hit_rate": float(np.mean(group.dock_score <= cutoff)),
                "mean_residual": residual.mean(),
                "mean_absolute_error": np.abs(residual).mean(),
            }
        )
    return pd.DataFrame(rows)


def cluster_count_summary(frame: pd.DataFrame, cutoff: float, round_id: int) -> pd.DataFrame:
    result = frame[frame.dock_score.notna()].copy()
    result["count_group"] = pd.cut(
        result.cluster_count_before,
        bins=[-1, 0, 1, 2, 4, np.inf],
        labels=["0", "1", "2", "3-4", "5+"],
    )
    rows = []
    for group_name, group in result.groupby("count_group", observed=True):
        residual = group.dock_score - group.pred_score
        rows.append(
            {
                "round": round_id,
                "cluster_count_before": str(group_name),
                "count": len(group),
                "hit_rate": float(np.mean(group.dock_score <= cutoff)),
                "mean_dock_score": group.dock_score.mean(),
                "mean_residual": residual.mean(),
                "mean_penalty": group.cluster_penalty.mean(),
            }
        )
    return pd.DataFrame(rows)


def cluster_occupancy_overall(frame: pd.DataFrame, round_id: int) -> dict[str, object]:
    counts = frame.clusterid.value_counts().to_numpy()
    probabilities = counts / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    return {
        "round": round_id,
        "selected": len(frame),
        "historical_clusters": len(counts),
        "effective_cluster_count_exp_entropy": math.exp(entropy),
        "normalized_cluster_entropy": entropy / math.log(len(counts)) if len(counts) > 1 else 0,
        "largest_cluster_contribution": int(counts.max()),
        "largest_cluster_fraction": float(counts.max() / counts.sum()),
        "first_in_cluster_count": int(np.sum(frame.cluster_count_before == 0)),
        "first_in_cluster_fraction": float(np.mean(frame.cluster_count_before == 0)),
        "median_cluster_count_before": float(frame.cluster_count_before.median()),
        "maximum_cluster_count_before": int(frame.cluster_count_before.max()),
        "mean_penalty": float(frame.cluster_penalty.mean()),
        "median_penalty": float(frame.cluster_penalty.median()),
        "maximum_penalty": float(frame.cluster_penalty.max()),
    }


def pairwise_policy_overlap(
    round_id: int,
    selections: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    names = list(selections)
    for left_index, left in enumerate(names):
        left_set = set(map(int, selections[left]))
        for right in names[left_index:]:
            right_set = set(map(int, selections[right]))
            intersection = len(left_set & right_set)
            union = len(left_set | right_set)
            rows.append(
                {
                    "round": round_id,
                    "policy_left": left,
                    "policy_right": right,
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union,
                    "left_only": len(left_set - right_set),
                    "right_only": len(right_set - left_set),
                }
            )
    return pd.DataFrame(rows)


def factorial_attribution(policy_metrics_frame: pd.DataFrame) -> pd.DataFrame:
    ignored = {
        "round",
        "policy",
        "selected",
        "observed_outcomes",
        "outcome_coverage",
        "observed_hit_count",
    }
    rows = []
    for round_id, group in policy_metrics_frame.groupby("round"):
        indexed = group.set_index("policy")
        for metric in group.columns:
            if metric in ignored or not pd.api.types.is_numeric_dtype(group[metric]):
                continue
            mean = float(indexed.loc["mean_only", metric])
            uncertainty = float(indexed.loc["uncertainty_only", metric])
            diversity = float(indexed.loc["diversity_only", metric])
            full = float(indexed.loc["full_lcb_atlas", metric])
            rows.append(
                {
                    "round": int(round_id),
                    "metric": metric,
                    "mean_only": mean,
                    "uncertainty_only": uncertainty,
                    "diversity_only": diversity,
                    "full_lcb_atlas": full,
                    "uncertainty_main_effect": 0.5 * ((uncertainty - mean) + (full - diversity)),
                    "diversity_main_effect": 0.5 * ((diversity - mean) + (full - uncertainty)),
                    "interaction": full - uncertainty - diversity + mean,
                }
            )
    return pd.DataFrame(rows)


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
        }
    )


def save_figure(fig: plt.Figure, output: Path, dpi: int) -> None:
    fig.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_figures(
    output: Path,
    candidate_samples: pd.DataFrame,
    actual_members: pd.DataFrame,
    group_summary: pd.DataFrame,
    uncertainty_bins: pd.DataFrame,
    penalty_bins: pd.DataFrame,
    policy_frame: pd.DataFrame,
    factorial: pd.DataFrame,
    dpi: int,
) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    for axis, round_id in zip(axes, (1, 2), strict=True):
        pool = candidate_samples[candidate_samples["round"] == round_id]
        selected = actual_members[actual_members["round"] == round_id]
        axis.hexbin(
            pool.pred_score,
            pool.epistemic_std,
            gridsize=70,
            mincnt=1,
            cmap="Greys",
            bins="log",
        )
        axis.scatter(
            selected.pred_score,
            selected.epistemic_std,
            s=1,
            alpha=0.12,
            color=ORANGE,
            rasterized=True,
            label="Actual LCB selections",
        )
        axis.set(
            xlabel="Predicted docking score (lower is better)",
            ylabel="Epistemic standard deviation",
            title=f"Round {round_id}",
        )
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output / "01_candidate_mean_uncertainty_selection", dpi)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), sharey=True)
    for axis, round_id in zip(axes, (1, 2), strict=True):
        data = group_summary[group_summary["round"] == round_id].set_index("selection_group")
        groups = [group for group in ACTUAL_GROUPS if group in data.index]
        x = np.arange(len(groups))
        rates = 100 * data.loc[groups, "hit_rate"].to_numpy(dtype=float)
        counts = data.loc[groups, "count"].to_numpy(dtype=int)
        axis.bar(x, rates, color=[ACTUAL_GROUP_COLORS[group] for group in groups])
        axis.set_xticks(
            x, [ACTUAL_GROUP_LABELS[group] for group in groups], rotation=25, ha="right"
        )
        axis.set(title=f"Round {round_id}", ylabel="Observed hit rate (%)")
        for position, rate, count in zip(x, rates, counts, strict=True):
            axis.text(
                position, rate, f"{rate:.1f}%\nn={count:,}", ha="center", va="bottom", fontsize=6.5
            )
    fig.suptitle("Actual LCB yield by selection mechanism", y=1.02)
    fig.tight_layout()
    save_figure(fig, output / "02_actual_yield_by_selection_mechanism", dpi)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    for axis, round_id in zip(axes, (1, 2), strict=True):
        data = uncertainty_bins[uncertainty_bins["round"] == round_id]
        axis.plot(data["bin"], 100 * data.hit_rate, "o-", color=ORANGE, label="Hit rate")
        secondary = axis.twinx()
        secondary.plot(
            data["bin"], data.mean_absolute_error, "s--", color=BLUE, label="Absolute error"
        )
        axis.set(
            xlabel="Epistemic uncertainty quintile (low to high)",
            ylabel="Hit rate (%)",
            title=f"Round {round_id}",
        )
        secondary.set_ylabel("Mean absolute prediction error")
    fig.suptitle("Uncertainty, hit yield, and calibration", y=1.02)
    fig.tight_layout()
    save_figure(fig, output / "03_uncertainty_yield_calibration", dpi)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), sharey=True)
    for axis, round_id in zip(axes, (1, 2), strict=True):
        data = penalty_bins[penalty_bins["round"] == round_id]
        axis.plot(data["bin"], 100 * data.hit_rate, "o-", color=GREEN)
        axis.set(
            xlabel="Diversity-penalty quintile (low to high)",
            ylabel="Hit rate (%)",
            title=f"Round {round_id}",
        )
    fig.suptitle("Observed hit yield versus diversity penalty", y=1.02)
    fig.text(
        0.5,
        0.01,
        "Association, not a causal penalty effect: high penalties mark repeatedly "
        "sampled, candidate-rich clusters.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, output / "04_penalty_quantile_yield", dpi)

    metrics = (
        ("observed_hit_rate", "Observed hit rate", 100),
        ("internal_diversity", "Internal diversity", 1),
        ("typed_scaffold_richness_per_compound", "Typed scaffolds per compound", 1),
        ("atlas_clusters_per_compound", "Atlas clusters per compound", 1),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    policies = ["mean_only", "uncertainty_only", "diversity_only", "full_lcb_atlas"]
    x = np.arange(len(policies))
    for axis, (metric, label, scale) in zip(axes.flat, metrics, strict=True):
        for round_id, marker in ((1, "o"), (2, "s")):
            data = policy_frame[policy_frame["round"] == round_id].set_index("policy")
            axis.plot(
                x,
                scale * data.loc[policies, metric],
                marker=marker,
                label=f"Round {round_id}",
            )
        axis.set_xticks(x, [POLICY_LABELS[policy] for policy in policies], rotation=22, ha="right")
        axis.set_ylabel(label + (" (%)" if scale == 100 else ""))
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Standardized mean/uncertainty/diversity policy replay", y=1.01)
    fig.text(
        0.5,
        0.01,
        "Hit rates use only compounds with observed LCB or greedy docking outcomes; "
        "structural metrics cover full selections.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, output / "05_four_policy_potency_diversity", dpi)

    selected_metrics = {
        "observed_hit_rate": "Observed hit rate",
        "internal_diversity": "Internal diversity",
        "typed_scaffold_richness_per_compound": "Typed scaffolds per compound",
        "atlas_clusters_per_compound": "Atlas clusters per compound",
    }
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5))
    for axis, (metric, label) in zip(axes.flat, selected_metrics.items(), strict=True):
        data = factorial[factorial.metric == metric].sort_values("round")
        x = np.arange(len(data))
        width = 0.24
        axis.bar(x - width, data.uncertainty_main_effect, width, label="Uncertainty", color=ORANGE)
        axis.bar(x, data.diversity_main_effect, width, label="Diversity", color=GREEN)
        axis.bar(x + width, data.interaction, width, label="Interaction", color=PURPLE)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, [f"Round {value}" for value in data["round"]])
        axis.set_ylabel(f"Effect on {label}")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Factorial attribution of standardized policy effects", y=1.01)
    fig.text(
        0.5,
        0.01,
        "Hit-rate attribution is observed-subset only; diversity attribution uses "
        "all selected compounds.",
        ha="center",
        fontsize=7,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, output / "06_factorial_mechanism_decomposition", dpi)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    for axis, round_id in zip(axes, (1, 2), strict=True):
        data = actual_members[actual_members["round"] == round_id]
        for group in ACTUAL_GROUPS:
            subset = data[data.selection_group == group]
            axis.scatter(
                subset.umap_x,
                subset.umap_y,
                s=2,
                alpha=0.35,
                color=ACTUAL_GROUP_COLORS[group],
                label=ACTUAL_GROUP_LABELS[group],
                rasterized=True,
            )
        axis.set(xlabel="Fixed UMAP-1", ylabel="Fixed UMAP-2", title=f"Round {round_id}")
    axes[0].legend(frameon=False, markerscale=3, fontsize=7)
    fig.suptitle("Actual LCB selections by mechanism", y=1.02)
    fig.tight_layout()
    save_figure(fig, output / "07_selection_mechanism_umap", dpi)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"failed to write {path}")


def calculate(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    definitions = (
        RoundDefinition(1, 0, 4_557_956, "1=1"),
        RoundDefinition(2, 1, 7_266_854, "d.dock_iteration IS NULL OR d.dock_iteration=2"),
    )
    greedy_scores = load_greedy_score_map(args.greedy_docked_db)
    nearest_ids, nearest_values = load_vector_artifact(args.nearest_seed, "tanimoto")
    umap_ids, umap_values = load_vector_artifact(args.umap_coordinates, "umap")

    candidate_summary_rows: list[dict[str, object]] = []
    overlap_frames: list[pd.DataFrame] = []
    policy_metric_rows: list[dict[str, object]] = []
    actual_group_frames: list[pd.DataFrame] = []
    actual_member_frames: list[pd.DataFrame] = []
    uncertainty_frames: list[pd.DataFrame] = []
    penalty_frames: list[pd.DataFrame] = []
    cluster_count_frames: list[pd.DataFrame] = []
    cluster_occupancy_rows: list[dict[str, object]] = []
    mean_control_frames: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, object]] = []
    selected_id_rows: list[dict[str, object]] = []
    candidate_sample_frames: list[pd.DataFrame] = []

    for definition in definitions:
        pool = load_candidate_pool(args.lcb_db, args.atlas_db, definition)
        actual_path = args.acquisition_root / f"iter{definition.round_id}" / "acquisition.csv"
        actual = load_actual_acquisition(actual_path, definition.round_id, args.batch_size)
        actual_positions = candidate_positions(
            pool.identifiers, actual.spacehastenid.to_numpy(dtype=np.int64)
        )
        if not np.allclose(pool.means[actual_positions], actual.pred_score, atol=1e-6):
            raise ValueError(
                f"round {definition.round_id}: candidate means differ from acquisition"
            )
        if not np.allclose(pool.epistemic[actual_positions], actual.epistemic_std, atol=1e-6):
            raise ValueError(f"round {definition.round_id}: candidate uncertainties differ")
        policies = select_policies(pool, args.batch_size, args.cluster_lambda)
        policy_ids = {
            name: pool.identifiers[selection.indices] for name, selection in policies.items()
        }
        policy_ids["actual_lcb_historical"] = actual.spacehastenid.to_numpy(dtype=np.int64)
        overlap_frames.append(pairwise_policy_overlap(definition.round_id, policy_ids))

        mean_set = set(map(int, policy_ids["mean_only"]))
        uncertainty_set = set(map(int, policy_ids["uncertainty_only"]))
        actual_set = set(map(int, policy_ids["actual_lcb_historical"]))
        actual["selection_group"] = [
            (
                "mean_and_uncertainty_core"
                if int(sid) in mean_set and int(sid) in uncertainty_set
                else "uncertainty_promoted"
                if int(sid) in uncertainty_set
                else "diversity_rescued_mean"
                if int(sid) in mean_set
                else "diversity_only"
            )
            for sid in actual.spacehastenid
        ]
        actual["atlas_cluster"] = pool.atlas_clusters[actual_positions]
        actual["dock_score"] = pool.lcb_dock_scores[actual_positions]
        actual["dock_iteration"] = pool.final_dock_iterations[actual_positions]
        actual["candidate_mean_rank"] = 0
        actual["candidate_lcb_rank"] = 0
        mean_order = np.lexsort((pool.identifiers, pool.means))
        lcb_order = np.lexsort((pool.identifiers, pool.means - pool.epistemic))
        mean_rank = np.empty(len(pool.identifiers), dtype=np.int32)
        lcb_rank = np.empty(len(pool.identifiers), dtype=np.int32)
        mean_rank[mean_order] = np.arange(1, len(pool.identifiers) + 1, dtype=np.int32)
        lcb_rank[lcb_order] = np.arange(1, len(pool.identifiers) + 1, dtype=np.int32)
        actual["candidate_mean_rank"] = mean_rank[actual_positions]
        actual["candidate_lcb_rank"] = lcb_rank[actual_positions]
        actual["uncertainty_rank_promotion"] = (
            actual.candidate_mean_rank - actual.candidate_lcb_rank
        )

        selected_union = np.unique(np.concatenate(list(policy_ids.values())))
        info = load_selected_compounds(args.lcb_db, selected_union, greedy_scores)
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
        actual["nearest_seed"] = map_values(
            nearest_ids,
            nearest_values,
            actual.spacehastenid.to_numpy(dtype=np.int64),
            allow_missing=True,
        )
        actual_umap = map_values(
            umap_ids,
            umap_values,
            actual.spacehastenid.to_numpy(dtype=np.int64),
            allow_missing=True,
        )
        actual["umap_x"] = actual_umap[:, 0]
        actual["umap_y"] = actual_umap[:, 1]
        actual["round"] = definition.round_id
        actual_member_frames.append(actual)
        actual_group_frames.append(
            summarize_actual_groups(
                actual,
                args.hit_cutoff,
                args.pair_samples,
                words,
                popcounts,
                definition.round_id,
            )
        )

        for name, selection in policies.items():
            identifiers = pool.identifiers[selection.indices]
            clusters = pool.atlas_clusters[selection.indices]
            policy_metric_rows.append(
                policy_metrics(
                    definition.round_id,
                    name,
                    identifiers,
                    clusters,
                    info,
                    words,
                    popcounts,
                    args.hit_cutoff,
                    args.pair_samples,
                    args.random_seed + definition.round_id * 100 + len(policy_metric_rows),
                )
            )
            for sid in identifiers:
                selected_id_rows.append(
                    {"round": definition.round_id, "policy": name, "spacehastenid": int(sid)}
                )
        actual_clusters = actual.atlas_cluster.to_numpy(dtype=np.int64)
        policy_metric_rows.append(
            policy_metrics(
                definition.round_id,
                "actual_lcb_historical",
                actual.spacehastenid.to_numpy(dtype=np.int64),
                actual_clusters,
                info,
                words,
                popcounts,
                args.hit_cutoff,
                args.pair_samples,
                args.random_seed + definition.round_id * 100 + 99,
            )
        )

        observed_actual = actual[actual.dock_score.notna()].copy()
        residual = observed_actual.dock_score - observed_actual.pred_score
        calibration_rows.append(
            {
                "round": definition.round_id,
                "candidate_count": len(pool.identifiers),
                "actual_selected": len(actual),
                "observed_actual": len(observed_actual),
                "spearman_uncertainty_absolute_error": float(
                    stats.spearmanr(observed_actual.epistemic_std, np.abs(residual)).statistic
                ),
                "spearman_uncertainty_residual": float(
                    stats.spearmanr(observed_actual.epistemic_std, residual).statistic
                ),
                "mean_absolute_error": float(np.mean(np.abs(residual))),
                "mean_residual": float(np.mean(residual)),
                "favorable_surprise_rate_residual_lt_minus1": float(np.mean(residual < -1)),
            }
        )
        uncertainty_frames.append(
            quantile_summary(
                actual,
                "epistemic_std",
                "epistemic_uncertainty",
                args.hit_cutoff,
                definition.round_id,
            )
        )
        penalty_frames.append(
            quantile_summary(
                actual, "cluster_penalty", "cluster_penalty", args.hit_cutoff, definition.round_id
            )
        )
        cluster_count_frames.append(
            cluster_count_summary(actual, args.hit_cutoff, definition.round_id)
        )
        cluster_occupancy_rows.append(cluster_occupancy_overall(actual, definition.round_id))
        mean_control_frames.append(
            mean_decile_uncertainty_summary(actual, args.hit_cutoff, definition.round_id)
        )
        candidate_summary_rows.append(
            {
                "round": definition.round_id,
                "model_version": definition.model_version,
                "candidate_count": len(pool.identifiers),
                "pred_score_mean": float(np.mean(pool.means)),
                "pred_score_median": float(np.median(pool.means)),
                "epistemic_mean": float(np.mean(pool.epistemic)),
                "epistemic_median": float(np.median(pool.epistemic)),
                "epistemic_q05": float(np.quantile(pool.epistemic, 0.05)),
                "epistemic_q95": float(np.quantile(pool.epistemic, 0.95)),
                "mean_uncertainty_pearson": float(np.corrcoef(pool.means, pool.epistemic)[0, 1]),
                "mean_uncertainty_spearman": float(
                    stats.spearmanr(pool.means, pool.epistemic).statistic
                ),
                "actual_epistemic_median": actual.epistemic_std.median(),
                "actual_penalty_median": actual.cluster_penalty.median(),
                "actual_penalty_max": actual.cluster_penalty.max(),
                "actual_first_in_cluster_fraction": float(
                    np.mean(actual.cluster_count_before == 0)
                ),
                "actual_unique_historical_clusters": actual.clusterid.nunique(),
                "actual_unique_atlas_clusters": actual.atlas_cluster.nunique(),
                "mean_uncertainty_overlap": len(mean_set & uncertainty_set),
                "uncertainty_added_vs_mean": len(uncertainty_set - mean_set),
                "historical_diversity_added_vs_uncertainty": len(actual_set - uncertainty_set),
                "historical_diversity_displaced_from_uncertainty": len(
                    uncertainty_set - actual_set
                ),
                "standard_full_actual_overlap": len(
                    set(map(int, policy_ids["full_lcb_atlas"])) & actual_set
                ),
            }
        )
        sample_rng = np.random.default_rng(args.random_seed + definition.round_id)
        sample_size = min(200_000, len(pool.identifiers))
        sample = sample_rng.choice(len(pool.identifiers), sample_size, replace=False)
        candidate_sample_frames.append(
            pd.DataFrame(
                {
                    "round": definition.round_id,
                    "pred_score": pool.means[sample],
                    "epistemic_std": pool.epistemic[sample],
                }
            )
        )
        del mean_order, lcb_order, mean_rank, lcb_rank, words, popcounts, info, pool

    candidate_summary = pd.DataFrame(candidate_summary_rows)
    overlaps = pd.concat(overlap_frames, ignore_index=True)
    policy_frame = pd.DataFrame(policy_metric_rows)
    actual_groups = pd.concat(actual_group_frames, ignore_index=True)
    actual_groups["hit_contribution_fraction"] = actual_groups.hit_count / actual_groups.groupby(
        "round"
    ).hit_count.transform("sum")
    actual_members = pd.concat(actual_member_frames, ignore_index=True)
    uncertainty_bins = pd.concat(uncertainty_frames, ignore_index=True)
    penalty_bins = pd.concat(penalty_frames, ignore_index=True)
    cluster_counts = pd.concat(cluster_count_frames, ignore_index=True)
    cluster_occupancy = pd.DataFrame(cluster_occupancy_rows)
    mean_control = pd.concat(mean_control_frames, ignore_index=True)
    calibration = pd.DataFrame(calibration_rows)
    policy_selected_ids = pd.DataFrame(selected_id_rows)
    candidate_samples = pd.concat(candidate_sample_frames, ignore_index=True)
    factorial = factorial_attribution(policy_frame)

    write_csv(output / "round_candidate_summary.csv", candidate_summary)
    write_csv(output / "policy_selection_overlap.csv", overlaps)
    write_csv(output / "actual_selection_groups.csv", actual_groups)
    actual_members.to_csv(
        output / "actual_selection_membership.csv.gz", index=False, compression="gzip"
    )
    write_csv(output / "uncertainty_quantile_yield.csv", uncertainty_bins)
    write_csv(output / "uncertainty_calibration.csv", calibration)
    write_csv(output / "mean_decile_uncertainty_control.csv", mean_control)
    write_csv(output / "penalty_quantile_yield.csv", penalty_bins)
    write_csv(output / "cluster_count_yield.csv", cluster_counts)
    write_csv(output / "cluster_occupancy_summary.csv", cluster_occupancy)
    write_csv(output / "counterfactual_policy_metrics.csv", policy_frame)
    write_csv(output / "factorial_attribution.csv", factorial)
    decomposition = candidate_summary.merge(calibration, on="round", suffixes=("", "_calibration"))
    actual_round = (
        actual_members.groupby("round")
        .agg(
            actual_observed=("dock_score", "count"),
            actual_hit_rate=(
                "dock_score",
                lambda values: float(np.mean(values.dropna() <= args.hit_cutoff)),
            ),
            actual_mean_dock_score=("dock_score", "mean"),
            actual_epistemic_mean=("epistemic_std", "mean"),
            actual_penalty_mean=("cluster_penalty", "mean"),
            actual_nearest_seed_mean=("nearest_seed", "mean"),
        )
        .reset_index()
    )
    group_wide = actual_groups.pivot(
        index="round", columns="selection_group", values=["count", "hit_rate"]
    )
    group_wide.columns = [f"{metric}_{group}" for metric, group in group_wide.columns]
    decomposition = decomposition.merge(actual_round, on="round").merge(
        group_wide.reset_index(), on="round"
    )
    round1_groups = actual_groups[actual_groups["round"] == 1].set_index("selection_group")
    round2_groups = actual_groups[actual_groups["round"] == 2].set_index("selection_group")
    weights1 = round1_groups["batch_fraction"]
    weights2 = round2_groups["batch_fraction"]
    rates1 = round1_groups["hit_rate"]
    rates2 = round2_groups["hit_rate"]
    composition_forward = float(np.sum((weights2 - weights1) * rates1))
    within_forward = float(np.sum(weights2 * (rates2 - rates1)))
    composition_reverse = float(np.sum((weights2 - weights1) * rates2))
    within_reverse = float(np.sum(weights1 * (rates2 - rates1)))
    composition_effect = 0.5 * (composition_forward + composition_reverse)
    within_group_effect = 0.5 * (within_forward + within_reverse)
    total_yield_change = float(np.sum(weights2 * rates2) - np.sum(weights1 * rates1))
    yield_decomposition = pd.DataFrame(
        [
            {
                "round1_hit_rate": float(np.sum(weights1 * rates1)),
                "round2_hit_rate": float(np.sum(weights2 * rates2)),
                "total_change": total_yield_change,
                "composition_effect_shapley": composition_effect,
                "within_group_yield_effect_shapley": within_group_effect,
                "composition_share_of_change": composition_effect / total_yield_change,
                "within_group_share_of_change": within_group_effect / total_yield_change,
            }
        ]
    )
    write_csv(output / "round1_vs_round2_decomposition.csv", decomposition)
    write_csv(output / "round_yield_composition_decomposition.csv", yield_decomposition)
    policy_selected_ids.to_csv(
        output / "policy_selected_ids.csv.gz", index=False, compression="gzip"
    )
    make_figures(
        output,
        candidate_samples,
        actual_members,
        actual_groups,
        uncertainty_bins,
        penalty_bins,
        policy_frame,
        factorial,
        args.dpi,
    )
    summary: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "definition": {
            "rounds": [definition.__dict__ for definition in definitions],
            "batch_size": args.batch_size,
            "lcb_beta": 1.0,
            "cluster_lambda": args.cluster_lambda,
            "hit_cutoff": args.hit_cutoff,
            "standardized_policy_clusters": "persistent morgan-r2-1024-t040 atlas",
            "actual_policy_clusters": (
                "historical full-space clustering recorded in acquisition CSV"
            ),
            "observed_counterfactual_outcomes": (
                "LCB docking score when present, otherwise greedy docking score by "
                "reghash; unobserved compounds are not imputed"
            ),
        },
        "validation": {
            "actual_acquisition_arithmetic": True,
            "round1_candidates": int(candidate_summary.iloc[0].candidate_count),
            "round2_candidates": int(candidate_summary.iloc[1].candidate_count),
            "actual_round1_selected": int(
                actual_members[actual_members["round"] == 1].spacehastenid.nunique()
            ),
            "actual_round2_selected": int(
                actual_members[actual_members["round"] == 2].spacehastenid.nunique()
            ),
        },
        "candidate_summary": candidate_summary.to_dict(orient="records"),
        "calibration": calibration.to_dict(orient="records"),
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    required = [
        "round_candidate_summary.csv",
        "policy_selection_overlap.csv",
        "actual_selection_groups.csv",
        "actual_selection_membership.csv.gz",
        "uncertainty_quantile_yield.csv",
        "uncertainty_calibration.csv",
        "mean_decile_uncertainty_control.csv",
        "penalty_quantile_yield.csv",
        "cluster_occupancy_summary.csv",
        "cluster_count_yield.csv",
        "counterfactual_policy_metrics.csv",
        "factorial_attribution.csv",
        "round1_vs_round2_decomposition.csv",
        "round_yield_composition_decomposition.csv",
        "policy_selected_ids.csv.gz",
        "analysis_summary.json",
        *[
            f"{index:02d}_{stem}.{suffix}"
            for index, stem in (
                (1, "candidate_mean_uncertainty_selection"),
                (2, "actual_yield_by_selection_mechanism"),
                (3, "uncertainty_yield_calibration"),
                (4, "penalty_quantile_yield"),
                (5, "four_policy_potency_diversity"),
                (6, "factorial_mechanism_decomposition"),
                (7, "selection_mechanism_umap"),
            )
            for suffix in ("png", "pdf")
        ],
    ]
    missing = [
        name
        for name in required
        if not (output / name).is_file() or (output / name).stat().st_size == 0
    ]
    if missing:
        raise ValueError(f"missing attribution outputs: {missing}")
    LOGGER.info("Acquisition attribution completed in %.1fs", time.monotonic() - started)
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        calculate(parse_args())
    except Exception:
        LOGGER.exception("LCB acquisition attribution failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
