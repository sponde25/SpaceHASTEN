#!/usr/bin/env python3
# ruff: noqa: B007, B023, E501, E701, E702
"""Compare diversity of two completed acquisition runs on common definitions."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters import rdSimDivPickers
from tqdm import tqdm

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#555555"
ACyclic = "[ACYCLIC]"
PAIR_SAMPLES = 1_000_000
GRID_BINS = 60
BYTE_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)
FP_WORDS: np.ndarray | None = None
FP_POPCOUNTS: np.ndarray | None = None
METRIC_DIRECTIONS = {
    "sampled_pair_internal_diversity": "higher",
    "typed_scaffold_richness": "higher",
    "typed_scaffold_richness_per_molecule": "higher",
    "typed_scaffold_largest_family_fraction": "lower",
    "typed_scaffold_entropy": "higher",
    "typed_scaffold_normalized_entropy": "higher",
    "generic_framework_richness": "higher",
    "generic_framework_richness_per_molecule": "higher",
    "generic_framework_largest_family_fraction": "lower",
    "generic_framework_entropy": "higher",
    "generic_framework_normalized_entropy": "higher",
    "atlas_occupied_clusters": "higher",
    "atlas_richness_per_molecule": "higher",
    "atlas_largest_cluster_fraction": "lower",
    "atlas_entropy": "higher",
    "atlas_normalized_entropy": "higher",
    "nearest_seed_mean": "lower",
    "nearest_seed_median": "lower",
    "nearest_seed_fraction_lt_0.4": "higher",
    "nearest_seed_fraction_lt_0.5": "higher",
    "umap_occupied_grid_cells": "higher",
    "umap_grid_coverage": "higher",
    "umap_cells_per_molecule": "higher",
    "umap_max_cell_fraction": "lower",
}


def connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def load_hits(path: Path, cutoff: float, label: str) -> pd.DataFrame:
    with connect(path) as con:
        frame = pd.read_sql_query(
            "SELECT spacehastenid, reghash, smiles, dock_score, dock_iteration "
            "FROM data WHERE dock_iteration > 0 AND dock_score <= ? "
            "ORDER BY reghash, spacehastenid",
            con,
            params=(cutoff,),
        )
    if frame.reghash.isna().any() or frame.reghash.duplicated().any():
        raise ValueError(f"{label}: hit reghashes must be present and unique")
    if frame.smiles.isna().any() or frame.dock_score.isna().any():
        raise ValueError(f"{label}: hit structures and scores must be present")
    frame["run"] = label
    return frame.reset_index(drop=True)


def load_npz(path: Path, value: str) -> dict[int, float]:
    with np.load(path) as data:
        ids, values = data["spacehastenid"], data[value]
    if len(ids) != len(np.unique(ids)) or not np.isfinite(values).all():
        raise ValueError(f"invalid {value} artifact: {path}")
    return dict(zip(ids.astype(int), values.astype(float), strict=True))


def scaffold(smiles: str) -> tuple[str, str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"unparseable SMILES: {smiles}")
    core = MurckoScaffold.GetScaffoldForMol(molecule)
    if core.GetNumAtoms() == 0:
        return ACyclic, ACyclic
    typed = Chem.MolToSmiles(core, canonical=True, isomericSmiles=False)
    generic = Chem.MolToSmiles(
        MurckoScaffold.MakeScaffoldGeneric(core), canonical=True, isomericSmiles=False
    )
    return typed, generic


def add_structure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    values = [
        scaffold(smiles)
        for smiles in tqdm(frame.smiles, total=len(frame), desc=f"{frame.run.iloc[0]} scaffolds")
    ]
    result = frame.copy()
    result[["typed_scaffold", "generic_framework"]] = pd.DataFrame(values, index=result.index)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    result["fingerprint"] = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in result.smiles]
    return result


def family_summary(values: pd.Series, prefix: str) -> dict[str, float | int]:
    counts = values.value_counts()
    total = len(values)
    probabilities = counts.to_numpy() / total
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return {
        f"{prefix}_richness": int(len(counts)),
        f"{prefix}_richness_per_molecule": float(len(counts) / total),
        f"{prefix}_largest_family_fraction": float(counts.iloc[0] / total),
        f"{prefix}_entropy": entropy,
        f"{prefix}_normalized_entropy": float(entropy / math.log(len(counts)))
        if len(counts) > 1
        else 0.0,
    }


def pair_diversity(
    indices: np.ndarray, rng: np.random.Generator, samples: int
) -> dict[str, float | int]:
    if FP_WORDS is None or FP_POPCOUNTS is None:
        raise RuntimeError("fingerprint arrays are not initialized")
    total, squared, completed = 0.0, 0.0, 0
    while completed < samples:
        size = min(250_000, samples - completed)
        left, right = (
            rng.integers(0, len(indices), size=size),
            rng.integers(0, len(indices), size=size),
        )
        equal = left == right
        while equal.any():
            right[equal] = rng.integers(0, len(indices), size=int(equal.sum()))
            equal = left == right
        left_rows, right_rows = indices[left], indices[right]
        intersection = BYTE_POPCOUNT[
            np.bitwise_and(FP_WORDS[left_rows], FP_WORDS[right_rows]).view(np.uint8)
        ].sum(axis=1)
        values = intersection / (FP_POPCOUNTS[left_rows] + FP_POPCOUNTS[right_rows] - intersection)
        total += float(values.sum())
        squared += float(np.square(values).sum())
        completed += size
    mean = total / samples
    error = math.sqrt(max((squared - samples * mean * mean) / (samples - 1), 0.0) / samples)
    return {
        "sampled_pair_internal_diversity": 1 - mean,
        "pair_monte_carlo_se": error,
        "pair_monte_carlo_ci95_low": 1 - mean - 1.96 * error,
        "pair_monte_carlo_ci95_high": 1 - mean + 1.96 * error,
        "pair_samples": samples,
    }


def make_joint_atlas(
    greedy: pd.DataFrame, lcb: pd.DataFrame
) -> tuple[dict[str, int], dict[str, object]]:
    union = (
        pd.concat([greedy, lcb])
        .drop_duplicates("reghash", keep="first")
        .sort_values("reghash")
        .reset_index(drop=True)
    )
    fingerprints = union.fingerprint.tolist()
    picker = rdSimDivPickers.LeaderPicker()
    centroids = list(picker.LazyBitVectorPick(fingerprints, len(fingerprints), 0.6))
    centroid_fps = [fingerprints[index] for index in centroids]
    assignments: dict[str, int] = {}
    minimum_similarity = 1.0
    for row in tqdm(union.itertuples(), total=len(union), desc="Assigning common hit atlas"):
        similarities = DataStructs.BulkTanimotoSimilarity(row.fingerprint, centroid_fps)
        best = int(np.argmax(similarities))
        minimum_similarity = min(minimum_similarity, float(similarities[best]))
        assignments[row.reghash] = best
    if minimum_similarity < 0.4:
        raise ValueError(f"common atlas assignment below 0.4: {minimum_similarity}")
    return assignments, {
        "atlas_definition": "reghash-sorted, deduplicated union of both virtual-hit sets",
        "atlas_similarity_threshold": 0.4,
        "atlas_union_size": len(union),
        "atlas_cluster_count": len(centroids),
        "minimum_assignment_similarity": minimum_similarity,
    }


def atlas_summary(codes: np.ndarray) -> dict[str, float | int]:
    counts = np.bincount(codes)
    counts = counts[counts > 0]
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return {
        "atlas_occupied_clusters": int(len(counts)),
        "atlas_richness_per_molecule": float(len(counts) / counts.sum()),
        "atlas_largest_cluster_fraction": float(counts.max() / counts.sum()),
        "atlas_entropy": entropy,
        "atlas_normalized_entropy": float(entropy / math.log(len(counts)))
        if len(counts) > 1
        else 0.0,
    }


def nearest_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "nearest_seed_mean": float(np.mean(values)),
        "nearest_seed_median": float(np.median(values)),
        "nearest_seed_q05": float(np.quantile(values, 0.05)),
        "nearest_seed_q25": float(np.quantile(values, 0.25)),
        "nearest_seed_q75": float(np.quantile(values, 0.75)),
        "nearest_seed_q95": float(np.quantile(values, 0.95)),
        "nearest_seed_fraction_lt_0.3": float(np.mean(values < 0.3)),
        "nearest_seed_fraction_lt_0.4": float(np.mean(values < 0.4)),
        "nearest_seed_fraction_lt_0.5": float(np.mean(values < 0.5)),
    }


def grid_distribution(
    coordinates: np.ndarray, limits: tuple[float, float, float, float]
) -> tuple[np.ndarray, dict[str, float]]:
    x0, x1, y0, y1 = limits
    grid, _, _ = np.histogram2d(
        coordinates[:, 0], coordinates[:, 1], bins=GRID_BINS, range=[[x0, x1], [y0, y1]]
    )
    nonzero = grid[grid > 0]
    return grid, {
        "umap_occupied_grid_cells": int(len(nonzero)),
        "umap_grid_coverage": float(len(nonzero) / grid.size),
        "umap_cells_per_molecule": float(len(nonzero) / len(coordinates)),
        "umap_max_cell_fraction": float(grid.max() / grid.sum()),
    }


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    p, q = left.ravel() / left.sum(), right.ravel() / right.sum()
    middle = (p + q) / 2
    valid_p, valid_q = p > 0, q > 0
    return float(
        math.sqrt(
            0.5
            * (
                (p[valid_p] * np.log(p[valid_p] / middle[valid_p])).sum()
                + (q[valid_q] * np.log(q[valid_q] / middle[valid_q])).sum()
            )
        )
    )


def cohort_metrics(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    pair_samples: int,
    limits: tuple[float, float, float, float],
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "cohort_size": len(frame),
        **pair_diversity(frame.fp_index.to_numpy(dtype=np.int64), rng, pair_samples),
    }
    metrics.update(family_summary(frame.typed_scaffold, "typed_scaffold"))
    metrics.update(family_summary(frame.generic_framework, "generic_framework"))
    metrics.update(atlas_summary(frame.atlas_code.to_numpy(dtype=int)))
    metrics.update(nearest_summary(frame.nearest_seed.to_numpy(dtype=float)))
    _, grid_metrics = grid_distribution(frame[["umap_x", "umap_y"]].to_numpy(), limits)
    metrics.update(grid_metrics)
    return metrics


def empirical_intervals(
    replicates: pd.DataFrame,
    observed: dict[str, float | int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, direction in METRIC_DIRECTIONS.items():
        values = replicates[metric].to_numpy(dtype=float)
        value = float(observed[metric])
        q025, median, q975 = np.quantile(values, [0.025, 0.5, 0.975])
        lower_tail = (np.count_nonzero(values <= value) + 1) / (len(values) + 1)
        upper_tail = (np.count_nonzero(values >= value) + 1) / (len(values) + 1)
        favorable = value > median if direction == "higher" else value < median
        rows.append(
            {
                "metric": metric,
                "direction_favoring_diversity": direction,
                "reference_q025": q025,
                "reference_median": median,
                "reference_q975": q975,
                "observed_value": value,
                "observed_minus_reference_median": value - median,
                "percent_difference_from_reference_median": 100 * (value - median) / median,
                "outside_reference_95_interval": value < q025 or value > q975,
                "observed_diversity_favorable": favorable,
                "two_sided_empirical_p": min(1.0, 2 * min(lower_tail, upper_tail)),
            }
        )
    return pd.DataFrame(rows)


def save_figure(figure: plt.Figure, output: Path) -> None:
    figure.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plots(
    output: Path,
    full: pd.DataFrame,
    matched: pd.DataFrame,
    greedy: pd.DataFrame,
    lcb: pd.DataFrame,
    limits: tuple[float, float, float, float],
) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for metric, axis in zip(
        ("sampled_pair_internal_diversity", "atlas_occupied_clusters"), axes, strict=True
    ):
        samples = matched[matched.run == "Greedy"][metric]
        violin = axis.violinplot(samples, positions=[0], showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(BLUE)
            body.set_alpha(0.35)
        axis.scatter(
            0,
            full.loc[full.cohort == "SVDKL-LCB", metric].iloc[0],
            color=ORANGE,
            marker="s",
            s=45,
            zorder=3,
            label="LCB full",
        )
        axis.scatter(
            1,
            full.loc[full.cohort == "Greedy", metric].iloc[0],
            color=BLUE,
            marker="o",
            s=45,
            label="Greedy full",
        )
        axis.set_xticks([0, 1], ["Greedy matched\nreplicates + LCB", "Greedy full"])
        axis.set_ylabel(metric.replace("_", " "))
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_figure(fig, output / "01_full_vs_count_matched_diversity")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for metric, axis in zip(
        ("typed_scaffold_richness", "generic_framework_richness"), axes, strict=True
    ):
        violin = axis.violinplot(matched[metric], positions=[0], showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(BLUE)
            body.set_alpha(0.35)
        axis.scatter(
            0,
            full.loc[full.cohort == "SVDKL-LCB", metric].iloc[0],
            color=ORANGE,
            marker="s",
            s=45,
            label="LCB n=95,745",
        )
        axis.scatter(
            1,
            full.loc[full.cohort == "Greedy", metric].iloc[0],
            color=BLUE,
            marker="o",
            s=45,
            label="Greedy full",
        )
        axis.set_xticks([0, 1], ["Greedy matched\nreplicates + LCB", "Greedy full"])
        axis.set_ylabel(metric.replace("_", " "))
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_figure(fig, output / "02_scaffold_framework_richness")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for frame, label, color, style in (
        (greedy, "Greedy", BLUE, "-"),
        (lcb, "SVDKL-LCB", ORANGE, "--"),
    ):
        values = np.sort(frame.nearest_seed.to_numpy())
        ax.step(
            values,
            np.arange(1, len(values) + 1) / len(values),
            where="post",
            label=label,
            color=color,
            linestyle=style,
        )
    ax.set(xlabel="Exact nearest-seed Tanimoto similarity", ylabel="ECDF")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output / "03_nearest_seed_ecdf")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharex=True, sharey=True)
    for axis, frame, title, color in zip(
        axes,
        (greedy, lcb, lcb[lcb.dock_iteration == 2]),
        ("Greedy hits", "LCB hits", "LCB round 2 hits"),
        (BLUE, ORANGE, ORANGE),
        strict=True,
    ):
        axis.hexbin(frame.umap_x, frame.umap_y, gridsize=55, cmap="cividis", mincnt=1)
        axis.set(title=title, xlabel="Fixed baseline UMAP-1")
    axes[0].set_ylabel("Fixed baseline UMAP-2")
    fig.tight_layout()
    save_figure(fig, output / "04_fixed_baseline_umap_coverage")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for frame, label, color, marker in (
        (greedy, "Greedy", BLUE, "o"),
        (lcb, "SVDKL-LCB", ORANGE, "s"),
    ):
        grouped = frame.assign(bin=pd.qcut(frame.dock_score, 8, duplicates="drop")).groupby(
            "bin", observed=True
        )
        ax.plot(
            grouped.dock_score.mean(),
            grouped.nearest_seed.mean(),
            marker=marker,
            color=color,
            label=label,
        )
    ax.set(
        xlabel="Mean docking score within score octile (lower is better)",
        ylabel="Mean nearest-seed similarity",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output / "05_score_stratified_novelty")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--greedy-db", type=Path, required=True)
    parser.add_argument("--lcb-db", type=Path, required=True)
    parser.add_argument("--greedy-nearest", type=Path, required=True)
    parser.add_argument("--lcb-nearest", type=Path, required=True)
    parser.add_argument("--greedy-umap", type=Path, required=True)
    parser.add_argument("--lcb-umap", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=-9.7)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--pair-samples", type=int, default=1_000_000)
    parser.add_argument("--top-k", type=int, default=50_000)
    parser.add_argument("--expected-greedy-hits", type=int)
    parser.add_argument("--expected-lcb-hits", type=int)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    greedy, lcb = (
        add_structure_columns(load_hits(args.greedy_db, args.cutoff, "Greedy")),
        add_structure_columns(load_hits(args.lcb_db, args.cutoff, "SVDKL-LCB")),
    )
    if args.expected_greedy_hits is not None and len(greedy) != args.expected_greedy_hits:
        raise ValueError("unexpected greedy hit count")
    if args.expected_lcb_hits is not None and len(lcb) != args.expected_lcb_hits:
        raise ValueError("unexpected LCB hit count")
    nearest = {
        "Greedy": load_npz(args.greedy_nearest, "tanimoto"),
        "SVDKL-LCB": load_npz(args.lcb_nearest, "tanimoto"),
    }
    # Coordinates require two columns; reload separately while retaining ID validation.
    for label, frame, path in (
        ("Greedy", greedy, args.greedy_umap),
        ("SVDKL-LCB", lcb, args.lcb_umap),
    ):
        with np.load(path) as data:
            ids, xy = data["spacehastenid"].astype(int), data["umap"]
        if (
            xy.shape != (len(ids), 2)
            or not np.isfinite(xy).all()
            or len(ids) != len(np.unique(ids))
        ):
            raise ValueError(f"invalid UMAP: {path}")
        mapping = dict(zip(ids, xy, strict=True))
        frame["nearest_seed"] = frame.spacehastenid.map(nearest[label])
        frame["umap_x"] = frame.spacehastenid.map(lambda i: mapping.get(i, [np.nan, np.nan])[0])
        frame["umap_y"] = frame.spacehastenid.map(lambda i: mapping.get(i, [np.nan, np.nan])[1])
        if (
            frame[["nearest_seed", "umap_x", "umap_y"]].isna().any().any()
            or not ((frame.nearest_seed >= 0) & (frame.nearest_seed <= 1)).all()
        ):
            raise ValueError(f"{label}: missing or invalid joined artifact values")
    fingerprint_union = pd.concat([greedy, lcb]).drop_duplicates("reghash").reset_index(drop=True)
    index_by_hash = dict(zip(fingerprint_union.reghash, range(len(fingerprint_union)), strict=True))
    global FP_WORDS, FP_POPCOUNTS
    packed = b"".join(DataStructs.BitVectToBinaryText(fp) for fp in fingerprint_union.fingerprint)
    FP_WORDS = np.frombuffer(packed, dtype=np.uint64).reshape(len(fingerprint_union), 16)
    FP_POPCOUNTS = np.asarray(
        [fp.GetNumOnBits() for fp in fingerprint_union.fingerprint], dtype=np.int16
    )
    for frame in (greedy, lcb):
        frame["fp_index"] = frame.reghash.map(index_by_hash)
    atlas, atlas_metadata = make_joint_atlas(greedy, lcb)
    for frame in (greedy, lcb):
        frame["atlas_code"] = frame.reghash.map(atlas)
    limits = (
        min(greedy.umap_x.min(), lcb.umap_x.min()),
        max(greedy.umap_x.max(), lcb.umap_x.max()),
        min(greedy.umap_y.min(), lcb.umap_y.min()),
        max(greedy.umap_y.max(), lcb.umap_y.max()),
    )
    rng = np.random.default_rng(args.random_seed)
    full_rows = [
        {
            "cohort": "Greedy",
            "cohort_type": "full_natural_sample_size_confounded",
            **cohort_metrics(greedy, rng, args.pair_samples, limits),
        },
        {
            "cohort": "SVDKL-LCB",
            "cohort_type": "full_natural_sample_size_confounded",
            **cohort_metrics(lcb, rng, args.pair_samples, limits),
        },
    ]
    matched_rows = []
    sequences = np.random.SeedSequence(args.random_seed).spawn(args.replicates)
    lcb_metrics = cohort_metrics(
        lcb, np.random.default_rng(sequences[0]), args.pair_samples, limits
    )
    for index, sequence in enumerate(tqdm(sequences, desc="Count-matched replicates"), 1):
        sample = greedy.iloc[
            np.random.default_rng(sequence).choice(len(greedy), len(lcb), replace=False)
        ]
        row = {
            "replicate": index,
            "run": "Greedy",
            **cohort_metrics(sample, np.random.default_rng(sequence), args.pair_samples, limits),
        }
        lcb_grid, _ = grid_distribution(lcb[["umap_x", "umap_y"]].to_numpy(), limits)
        sample_grid, _ = grid_distribution(sample[["umap_x", "umap_y"]].to_numpy(), limits)
        row["umap_js_distance_to_lcb"] = jensen_shannon(sample_grid, lcb_grid)
        matched_rows.append(row)
    matched = pd.DataFrame(matched_rows)
    full = pd.DataFrame(full_rows)
    full["umap_js_distance_between_runs"] = jensen_shannon(
        *[
            grid_distribution(frame[["umap_x", "umap_y"]].to_numpy(), limits)[0]
            for frame in (greedy, lcb)
        ]
    )
    rounds = pd.DataFrame(
        [
            {
                "run": label,
                "round": int(round_id),
                **cohort_metrics(
                    group,
                    np.random.default_rng(args.random_seed + int(round_id)),
                    args.pair_samples,
                    limits,
                ),
            }
            for label, frame in (("Greedy", greedy), ("SVDKL-LCB", lcb))
            for round_id, group in frame.groupby("dock_iteration")
        ]
    )
    potency = pd.DataFrame(
        [
            {
                "run": label,
                "cohort": f"top_{args.top_k}_by_docking_score",
                **cohort_metrics(
                    frame.nsmallest(args.top_k, "dock_score"),
                    np.random.default_rng(args.random_seed),
                    args.pair_samples,
                    limits,
                ),
            }
            for label, frame in (("Greedy", greedy), ("SVDKL-LCB", lcb))
        ]
    )
    greedy_hashes, lcb_hashes = set(greedy.reghash), set(lcb.reghash)
    shared_hashes = greedy_hashes & lcb_hashes
    exclusive_frames = {
        "greedy_only": greedy[greedy.reghash.isin(greedy_hashes - lcb_hashes)],
        "lcb_only": lcb[lcb.reghash.isin(lcb_hashes - greedy_hashes)],
        "shared": greedy[greedy.reghash.isin(shared_hashes)],
    }
    overlap = pd.DataFrame(
        [{"cohort": name, "count": len(frame)} for name, frame in exclusive_frames.items()]
    )
    exclusive_metrics = pd.DataFrame(
        [
            {
                "cohort": name,
                **cohort_metrics(
                    frame,
                    np.random.default_rng(args.random_seed + index),
                    args.pair_samples,
                    limits,
                ),
            }
            for index, (name, frame) in enumerate(exclusive_frames.items(), start=1)
        ]
    )
    summaries = empirical_intervals(matched, lcb_metrics).rename(
        columns={
            "reference_q025": "greedy_q025",
            "reference_median": "greedy_median",
            "reference_q975": "greedy_q975",
            "observed_value": "lcb_value",
        }
    )
    round_matched_rows: list[dict[str, object]] = []
    round_interval_rows: list[pd.DataFrame] = []
    round_sequences = np.random.SeedSequence(args.random_seed + 10_000).spawn(args.replicates * 2)
    for round_id in (1, 2):
        greedy_round = greedy[greedy.dock_iteration == round_id]
        lcb_round = lcb[lcb.dock_iteration == round_id]
        if len(greedy_round) >= len(lcb_round):
            sampled_run, sampled_frame = "Greedy", greedy_round
            fixed_run, fixed_frame = "SVDKL-LCB", lcb_round
        else:
            sampled_run, sampled_frame = "SVDKL-LCB", lcb_round
            fixed_run, fixed_frame = "Greedy", greedy_round
        target_n = len(fixed_frame)
        fixed_metrics = cohort_metrics(
            fixed_frame,
            np.random.default_rng(args.random_seed + round_id),
            args.pair_samples,
            limits,
        )
        rows: list[dict[str, object]] = []
        start = (round_id - 1) * args.replicates
        for replicate, sequence in enumerate(
            tqdm(
                round_sequences[start : start + args.replicates],
                desc=f"Round {round_id} matched replicates",
            ),
            start=1,
        ):
            sample = sampled_frame.iloc[
                np.random.default_rng(sequence).choice(len(sampled_frame), target_n, replace=False)
            ]
            rows.append(
                {
                    "round": round_id,
                    "replicate": replicate,
                    "sampled_run": sampled_run,
                    "fixed_run": fixed_run,
                    **cohort_metrics(
                        sample,
                        np.random.default_rng(sequence),
                        args.pair_samples,
                        limits,
                    ),
                }
            )
        replicate_frame = pd.DataFrame(rows)
        round_matched_rows.extend(rows)
        interval = empirical_intervals(replicate_frame, fixed_metrics)
        interval.insert(0, "round", round_id)
        interval.insert(1, "sampled_run", sampled_run)
        interval.insert(2, "fixed_run", fixed_run)
        interval.insert(3, "matched_n", target_n)
        round_interval_rows.append(interval)
    round_matched = pd.DataFrame(round_matched_rows)
    round_intervals = pd.concat(round_interval_rows, ignore_index=True)
    score_strata_rows: list[dict[str, object]] = []
    for label, frame in (("Greedy", greedy), ("SVDKL-LCB", lcb)):
        strata = pd.qcut(frame.dock_score, 8, labels=False, duplicates="drop")
        for stratum, group in frame.assign(score_octile=strata).groupby("score_octile"):
            score_strata_rows.append(
                {
                    "run": label,
                    "score_octile": int(stratum) + 1,
                    "count": len(group),
                    "score_min": group.dock_score.min(),
                    "score_mean": group.dock_score.mean(),
                    "score_max": group.dock_score.max(),
                    "nearest_seed_mean": group.nearest_seed.mean(),
                    "nearest_seed_median": group.nearest_seed.median(),
                    "typed_scaffold_richness": group.typed_scaffold.nunique(),
                    "generic_framework_richness": group.generic_framework.nunique(),
                    "atlas_occupied_clusters": group.atlas_code.nunique(),
                }
            )
    score_strata = pd.DataFrame(score_strata_rows)
    shared = greedy[greedy.reghash.isin(shared_hashes)].merge(
        lcb[lcb.reghash.isin(shared_hashes)], on="reghash", suffixes=("_greedy", "_lcb")
    )
    if not (shared.smiles_greedy == shared.smiles_lcb).all():
        raise ValueError("shared reghashes have different SMILES")
    shared_umap_abs = np.abs(
        shared[["umap_x_greedy", "umap_y_greedy"]].to_numpy()
        - shared[["umap_x_lcb", "umap_y_lcb"]].to_numpy()
    )
    shared_artifact_agreement = {
        "shared_hits": len(shared),
        "nearest_seed_max_abs_difference": float(
            np.max(np.abs(shared.nearest_seed_greedy - shared.nearest_seed_lcb))
        ),
        "umap_coordinate_mean_abs_difference": float(shared_umap_abs.mean()),
        "umap_coordinate_max_abs_difference": float(shared_umap_abs.max()),
    }
    summaries.to_csv(output / "count_matched_empirical_intervals.csv", index=False)
    full.to_csv(output / "full_set_comparison.csv", index=False)
    matched.to_csv(output / "count_matched_greedy_replicates.csv", index=False)
    rounds.to_csv(output / "round_stratified_comparison.csv", index=False)
    round_matched.to_csv(output / "round_matched_replicates.csv", index=False)
    round_intervals.to_csv(output / "round_matched_empirical_intervals.csv", index=False)
    potency.to_csv(output / "potency_matched_topk_comparison.csv", index=False)
    overlap.to_csv(output / "hit_overlap_comparison.csv", index=False)
    exclusive_metrics.to_csv(output / "exclusive_shared_diversity.csv", index=False)
    score_strata.to_csv(output / "score_stratified_novelty.csv", index=False)
    pd.DataFrame(
        [nearest_summary(greedy.nearest_seed), nearest_summary(lcb.nearest_seed)],
        index=["Greedy", "SVDKL-LCB"],
    ).rename_axis("run").reset_index().to_csv(output / "nearest_seed_comparison.csv", index=False)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "definition": {
            "cutoff": args.cutoff,
            "fingerprint": "binary Morgan radius 2, 1024 bits",
            "matched_n": len(lcb),
            "replicates": args.replicates,
            "seed": args.random_seed,
            "pair_samples_per_cohort": args.pair_samples,
            "fixed_umap_note": "Coverage/density visualization only; 2D distances are not quantitative chemical distances.",
        },
        "atlas": atlas_metadata,
        "validation": {
            "greedy_hits": len(greedy),
            "lcb_hits": len(lcb),
            "greedy_replicates_exact_n": bool((matched.cohort_size == len(lcb)).all()),
            "all_artifact_joins_complete": True,
            "shared_artifact_agreement": shared_artifact_agreement,
        },
        "lcb_count_matched_metrics": lcb_metrics,
    }
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plots(output, full, matched, greedy, lcb, limits)
    required = [
        *output.glob("*.csv"),
        *output.glob("*.png"),
        *output.glob("*.pdf"),
        output / "analysis_summary.json",
    ]
    if not required or any(not path.exists() or path.stat().st_size == 0 for path in required):
        raise ValueError("missing or empty output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
