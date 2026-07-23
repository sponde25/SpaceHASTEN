#!/usr/bin/env python3
"""Run a fast three-way structural-diversity and yield-mechanism comparison."""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters import rdSimDivPickers
from tqdm import tqdm

plt.switch_backend("Agg")

LOGGER = logging.getLogger("compare_core_diversity")
ACYCLIC = "[ACYCLIC]"
COLORS = {
    "Greedy": "#0072B2",
    "SVDKL-LCB": "#E69F00",
    "SVDKL-EI": "#009E73",
}
MARKERS = {"Greedy": "o", "SVDKL-LCB": "s", "SVDKL-EI": "^"}
BYTE_POPCOUNT = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
    .sum(axis=1)
    .astype(np.uint8)
)
FINGERPRINT_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
CORE_METRICS = (
    "sampled_pair_internal_diversity",
    "typed_scaffold_richness",
    "typed_scaffold_richness_per_molecule",
    "typed_scaffold_largest_family_fraction",
    "typed_scaffold_entropy",
    "typed_scaffold_normalized_entropy",
    "generic_framework_richness",
    "generic_framework_richness_per_molecule",
    "generic_framework_largest_family_fraction",
    "generic_framework_entropy",
    "generic_framework_normalized_entropy",
    "atlas_occupied_clusters",
    "atlas_richness_per_molecule",
    "atlas_largest_cluster_fraction",
    "atlas_entropy",
    "atlas_normalized_entropy",
)
METRIC_DIRECTIONS = {
    metric: "lower" if "largest" in metric else "higher" for metric in CORE_METRICS
}


@dataclass
class RunCohort:
    label: str
    database: Path
    frame: pd.DataFrame


@dataclass
class StructureData:
    hashes: np.ndarray
    words: np.ndarray
    popcounts: np.ndarray
    typed_scaffolds: np.ndarray
    generic_frameworks: np.ndarray
    atlas_codes: np.ndarray


def parse_assignments(
    values: list[str], converter: Callable[[str], Any]
) -> dict[str, Any]:
    assignments: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=VALUE, got {value!r}")
        label, raw = value.split("=", 1)
        if not label or not raw or label in assignments:
            raise ValueError(f"Missing or duplicate assignment label in {value!r}")
        assignments[label] = converter(raw)
    return assignments


def connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def load_scored_virtual(label: str, database: Path, cutoff: float) -> RunCohort:
    started = time.monotonic()
    with connect(database) as connection:
        frame = pd.read_sql_query(
            "SELECT spacehastenid, reghash, smiles, dock_score, dock_iteration, "
            "pred_score FROM data WHERE dock_iteration > 0 AND dock_score IS NOT NULL "
            "ORDER BY reghash, spacehastenid",
            connection,
        )
    if frame.empty:
        raise ValueError(f"{label}: no scored virtual compounds")
    if frame[["spacehastenid", "reghash", "smiles", "dock_score"]].isna().any().any():
        raise ValueError(f"{label}: required scored-virtual values are missing")
    if frame["spacehastenid"].duplicated().any() or frame["reghash"].duplicated().any():
        raise ValueError(f"{label}: scored virtual identifiers are not unique")
    if not np.isfinite(frame["dock_score"].to_numpy(dtype=float)).all():
        raise ValueError(f"{label}: docking scores are not finite")
    frame["is_hit"] = frame["dock_score"] <= cutoff
    LOGGER.info(
        "Loaded %s: %d scored selections, %d hits in %.1f s",
        label,
        len(frame),
        int(frame["is_hit"].sum()),
        time.monotonic() - started,
    )
    return RunCohort(label=label, database=database.resolve(), frame=frame)


def structure_worker(smiles: str) -> tuple[str, str, bytes, int]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"Unparseable SMILES: {smiles}")
    core = MurckoScaffold.GetScaffoldForMol(molecule)
    if core.GetNumAtoms() == 0:
        typed = generic = ACYCLIC
    else:
        typed = Chem.MolToSmiles(core, canonical=True, isomericSmiles=False)
        generic = Chem.MolToSmiles(
            MurckoScaffold.MakeScaffoldGeneric(core),
            canonical=True,
            isomericSmiles=False,
        )
    fingerprint = FINGERPRINT_GENERATOR.GetFingerprint(molecule)
    return (
        typed,
        generic,
        DataStructs.BitVectToBinaryText(fingerprint),
        fingerprint.GetNumOnBits(),
    )


def build_structure_data(runs: list[RunCohort], processes: int) -> StructureData:
    combined = pd.concat(
        [run.frame[["reghash", "smiles"]] for run in runs], ignore_index=True
    ).drop_duplicates()
    conflicts = combined.groupby("reghash", sort=False)["smiles"].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"{int((conflicts > 1).sum())} reghashes map to multiple SMILES")
    structures = (
        combined.drop_duplicates("reghash").sort_values("reghash").reset_index(drop=True)
    )
    chunksize = max(1, len(structures) // (processes * 16))
    typed: list[str] = []
    generic: list[str] = []
    binaries: list[bytes] = []
    popcounts: list[int] = []
    with mp.get_context("fork").Pool(processes) as pool:
        results = pool.imap(structure_worker, structures["smiles"], chunksize=chunksize)
        for typed_value, generic_value, binary, popcount in tqdm(
            results,
            total=len(structures),
            desc="Morgan fingerprints and Murcko scaffolds",
            unit="mol",
        ):
            typed.append(typed_value)
            generic.append(generic_value)
            binaries.append(binary)
            popcounts.append(popcount)
    packed = b"".join(binaries)
    words = np.frombuffer(packed, dtype=np.uint64).reshape(len(structures), 16).copy()
    data = StructureData(
        hashes=structures["reghash"].to_numpy(dtype=object),
        words=words,
        popcounts=np.asarray(popcounts, dtype=np.int16),
        typed_scaffolds=np.asarray(typed, dtype=object),
        generic_frameworks=np.asarray(generic, dtype=object),
        atlas_codes=np.full(len(structures), -1, dtype=np.int32),
    )
    index_by_hash = pd.Series(np.arange(len(structures), dtype=np.int64), index=data.hashes)
    for run in runs:
        run.frame["structure_index"] = run.frame["reghash"].map(index_by_hash)
        if run.frame["structure_index"].isna().any():
            raise ValueError(f"{run.label}: structure-index join is incomplete")
        run.frame["structure_index"] = run.frame["structure_index"].astype(np.int64)
    return data


def fingerprint_from_words(words: np.ndarray) -> Any:
    return DataStructs.CreateFromBinaryText(words.tobytes())


def build_common_hit_atlas(
    runs: list[RunCohort], structures: StructureData
) -> dict[str, Any]:
    hit_hashes = sorted(
        set().union(
            *[set(run.frame.loc[run.frame["is_hit"], "reghash"]) for run in runs]
        )
    )
    index_by_hash = {value: index for index, value in enumerate(structures.hashes)}
    structure_indices = np.asarray([index_by_hash[value] for value in hit_hashes], dtype=np.int64)
    fingerprints = [
        fingerprint_from_words(structures.words[index]) for index in structure_indices
    ]
    picker = rdSimDivPickers.LeaderPicker()
    centroid_rows = list(
        picker.LazyBitVectorPick(fingerprints, len(fingerprints), 0.6)
    )
    centroid_fingerprints = [fingerprints[index] for index in centroid_rows]
    minimum_similarity = 1.0
    assignments = np.empty(len(fingerprints), dtype=np.int32)
    for index, fingerprint in enumerate(
        tqdm(fingerprints, desc="Assigning common Tanimoto-0.4 hit atlas", unit="mol")
    ):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint, centroid_fingerprints
        )
        best = int(np.argmax(similarities))
        assignments[index] = best
        minimum_similarity = min(minimum_similarity, float(similarities[best]))
    if minimum_similarity < 0.4 - 1e-12:
        raise ValueError(f"Common-atlas similarity below 0.4: {minimum_similarity}")
    structures.atlas_codes[structure_indices] = assignments
    return {
        "definition": "reghash-sorted deduplicated union of Greedy, SVDKL-LCB, and SVDKL-EI hits",
        "similarity_threshold": 0.4,
        "union_size": len(hit_hashes),
        "cluster_count": len(centroid_rows),
        "minimum_assignment_similarity": minimum_similarity,
    }


def pair_diversity(
    indices: np.ndarray,
    structures: StructureData,
    rng: np.random.Generator,
    samples: int,
    batch_size: int = 100_000,
) -> dict[str, float | int]:
    if len(indices) < 2:
        raise ValueError("Pair diversity requires at least two compounds")
    total = squared = 0.0
    completed = 0
    while completed < samples:
        size = min(batch_size, samples - completed)
        left = rng.integers(0, len(indices), size=size)
        right = rng.integers(0, len(indices), size=size)
        equal = left == right
        while equal.any():
            right[equal] = rng.integers(0, len(indices), size=int(equal.sum()))
            equal = left == right
        left_rows = indices[left]
        right_rows = indices[right]
        intersections = BYTE_POPCOUNT[
            np.bitwise_and(
                structures.words[left_rows], structures.words[right_rows]
            ).view(np.uint8)
        ].sum(axis=1)
        similarities = intersections / (
            structures.popcounts[left_rows]
            + structures.popcounts[right_rows]
            - intersections
        )
        total += float(similarities.sum())
        squared += float(np.square(similarities).sum())
        completed += size
    mean = total / samples
    variance = max((squared - samples * mean * mean) / (samples - 1), 0.0)
    standard_error = math.sqrt(variance / samples)
    return {
        "sampled_pair_internal_diversity": 1 - mean,
        "pair_monte_carlo_se": standard_error,
        "pair_samples": samples,
    }


def family_summary(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return {
        f"{prefix}_richness": int(len(counts)),
        f"{prefix}_richness_per_molecule": float(len(counts) / counts.sum()),
        f"{prefix}_largest_family_fraction": float(counts.max() / counts.sum()),
        f"{prefix}_entropy": entropy,
        f"{prefix}_normalized_entropy": (
            float(entropy / math.log(len(counts))) if len(counts) > 1 else 0.0
        ),
    }


def atlas_summary(codes: np.ndarray) -> dict[str, float | int]:
    if (codes < 0).any():
        raise ValueError("Hit cohort contains missing common-atlas assignments")
    counts = np.bincount(codes)
    counts = counts[counts > 0]
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return {
        "atlas_occupied_clusters": int(len(counts)),
        "atlas_richness_per_molecule": float(len(counts) / counts.sum()),
        "atlas_largest_cluster_fraction": float(counts.max() / counts.sum()),
        "atlas_entropy": entropy,
        "atlas_normalized_entropy": (
            float(entropy / math.log(len(counts))) if len(counts) > 1 else 0.0
        ),
    }


def cohort_metrics(
    indices: np.ndarray,
    structures: StructureData,
    rng: np.random.Generator,
    pair_samples: int,
    include_atlas: bool,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "cohort_size": len(indices),
        **pair_diversity(indices, structures, rng, pair_samples),
        **family_summary(structures.typed_scaffolds[indices], "typed_scaffold"),
        **family_summary(structures.generic_frameworks[indices], "generic_framework"),
    }
    if include_atlas:
        result.update(atlas_summary(structures.atlas_codes[indices]))
    return result


def empirical_intervals(
    replicates: pd.DataFrame,
    observed: dict[str, float | int],
    sampled_run: str,
    fixed_run: str,
    context: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in CORE_METRICS:
        values = replicates[metric].to_numpy(dtype=float)
        value = float(observed[metric])
        q025, median, q975 = np.quantile(values, [0.025, 0.5, 0.975])
        lower_tail = (np.count_nonzero(values <= value) + 1) / (len(values) + 1)
        upper_tail = (np.count_nonzero(values >= value) + 1) / (len(values) + 1)
        direction = METRIC_DIRECTIONS[metric]
        rows.append(
            {
                **context,
                "sampled_run": sampled_run,
                "fixed_run": fixed_run,
                "metric": metric,
                "direction_favoring_diversity": direction,
                "sampled_q025": q025,
                "sampled_median": median,
                "sampled_q975": q975,
                "fixed_value": value,
                "fixed_minus_sampled_median": value - median,
                "percent_difference_from_sampled_median": (
                    100 * (value - median) / median if median else math.nan
                ),
                "outside_sampled_95_interval": value < q025 or value > q975,
                "fixed_diversity_favorable": (
                    value > median if direction == "higher" else value < median
                ),
                "two_sided_empirical_p": min(
                    1.0, 2 * min(lower_tail, upper_tail)
                ),
            }
        )
    return pd.DataFrame(rows)


def matched_analysis(
    runs: list[RunCohort],
    structures: StructureData,
    fixed_label: str,
    fixed_metrics: dict[str, float | int],
    replicates: int,
    pair_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_label = {run.label: run for run in runs}
    fixed_frame = by_label[fixed_label].frame
    fixed_hits = fixed_frame[fixed_frame["is_hit"]]
    fixed_indices = fixed_hits["structure_index"].to_numpy(dtype=np.int64)
    if int(fixed_metrics["cohort_size"]) != len(fixed_indices):
        raise ValueError("Fixed count-matched metrics have the wrong cohort size")
    records: list[dict[str, Any]] = []
    intervals: list[pd.DataFrame] = []
    sampled_labels = [label for label in by_label if label != fixed_label]
    sequences = np.random.SeedSequence(seed).spawn(replicates * len(sampled_labels))
    for label_index, label in enumerate(sampled_labels):
        source = by_label[label].frame
        source = source[source["is_hit"]]
        source_indices = source["structure_index"].to_numpy(dtype=np.int64)
        start = label_index * replicates
        label_records: list[dict[str, Any]] = []
        for replicate, sequence in enumerate(
            tqdm(
                sequences[start : start + replicates],
                desc=f"Count-matched {label} samples",
            ),
            start=1,
        ):
            rng = np.random.default_rng(sequence)
            sample = rng.choice(source_indices, len(fixed_indices), replace=False)
            row = {
                "replicate": replicate,
                "sampled_run": label,
                "fixed_run": fixed_label,
                **cohort_metrics(sample, structures, rng, pair_samples, True),
            }
            records.append(row)
            label_records.append(row)
        intervals.append(
            empirical_intervals(
                pd.DataFrame(label_records),
                fixed_metrics,
                label,
                fixed_label,
                {"comparison_type": "all_hits_count_matched", "round": "overall"},
            )
        )
    return pd.DataFrame(records), pd.concat(intervals, ignore_index=True)


def round_matched_analysis(
    runs: list[RunCohort],
    structures: StructureData,
    fixed_label: str,
    fixed_metrics_by_round: dict[int, dict[str, float | int]],
    replicates: int,
    pair_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_label = {run.label: run for run in runs}
    sampled_labels = [label for label in by_label if label != fixed_label]
    records: list[dict[str, Any]] = []
    intervals: list[pd.DataFrame] = []
    sequence_count = replicates * len(sampled_labels) * 2
    sequences = np.random.SeedSequence(seed + 10_000).spawn(sequence_count)
    sequence_index = 0
    for round_id in (1, 2):
        fixed = by_label[fixed_label].frame
        fixed = fixed[(fixed["is_hit"]) & (fixed["dock_iteration"] == round_id)]
        fixed_indices = fixed["structure_index"].to_numpy(dtype=np.int64)
        fixed_metrics = fixed_metrics_by_round[round_id]
        if int(fixed_metrics["cohort_size"]) != len(fixed_indices):
            raise ValueError(f"Fixed round-{round_id} metrics have the wrong cohort size")
        for label in sampled_labels:
            source = by_label[label].frame
            source = source[(source["is_hit"]) & (source["dock_iteration"] == round_id)]
            source_indices = source["structure_index"].to_numpy(dtype=np.int64)
            if len(source_indices) < len(fixed_indices):
                raise ValueError(
                    f"{label} round {round_id} has fewer hits than {fixed_label}"
                )
            label_records: list[dict[str, Any]] = []
            for replicate in tqdm(
                range(1, replicates + 1),
                desc=f"Round {round_id} matched {label} samples",
            ):
                sequence = sequences[sequence_index]
                sequence_index += 1
                rng = np.random.default_rng(sequence)
                sample = rng.choice(source_indices, len(fixed_indices), replace=False)
                row = {
                    "round": round_id,
                    "replicate": replicate,
                    "sampled_run": label,
                    "fixed_run": fixed_label,
                    **cohort_metrics(sample, structures, rng, pair_samples, True),
                }
                records.append(row)
                label_records.append(row)
            intervals.append(
                empirical_intervals(
                    pd.DataFrame(label_records),
                    fixed_metrics,
                    label,
                    fixed_label,
                    {"comparison_type": "round_count_matched", "round": round_id},
                )
            )
    return pd.DataFrame(records), pd.concat(intervals, ignore_index=True)


def scaffold_conversion(runs: list[RunCohort], structures: StructureData) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        indices = run.frame["structure_index"].to_numpy(dtype=np.int64)
        hits = run.frame["is_hit"].to_numpy(dtype=bool)
        for family_type, labels in (
            ("typed_scaffold", structures.typed_scaffolds[indices]),
            ("generic_framework", structures.generic_frameworks[indices]),
        ):
            _, inverse, counts = np.unique(
                labels, return_inverse=True, return_counts=True
            )
            hit_counts = np.bincount(inverse, weights=hits.astype(np.int64))
            rates = hit_counts / counts
            rows.append(
                {
                    "run": run.label,
                    "family_type": family_type,
                    "scope": "overall",
                    "selection_count_band": "all",
                    "selected_compounds": len(indices),
                    "hit_compounds": int(hits.sum()),
                    "compound_hit_rate": float(hits.mean()),
                    "selected_families": len(counts),
                    "hit_bearing_families": int(np.count_nonzero(hit_counts)),
                    "zero_hit_families": int(np.count_nonzero(hit_counts == 0)),
                    "hit_bearing_family_fraction": float(np.mean(hit_counts > 0)),
                    "median_family_hit_rate": float(np.median(rates)),
                }
            )
            bands = {
                "1": counts == 1,
                "2-4": (counts >= 2) & (counts <= 4),
                "5-9": (counts >= 5) & (counts <= 9),
                "10-24": (counts >= 10) & (counts <= 24),
                ">=25": counts >= 25,
            }
            for band, mask in bands.items():
                if not mask.any():
                    continue
                rows.append(
                    {
                        "run": run.label,
                        "family_type": family_type,
                        "scope": "selection_count_band",
                        "selection_count_band": band,
                        "selected_compounds": int(counts[mask].sum()),
                        "hit_compounds": int(hit_counts[mask].sum()),
                        "compound_hit_rate": float(
                            hit_counts[mask].sum() / counts[mask].sum()
                        ),
                        "selected_families": int(mask.sum()),
                        "hit_bearing_families": int(
                            np.count_nonzero(hit_counts[mask])
                        ),
                        "zero_hit_families": int(
                            np.count_nonzero(hit_counts[mask] == 0)
                        ),
                        "hit_bearing_family_fraction": float(
                            np.mean(hit_counts[mask] > 0)
                        ),
                        "median_family_hit_rate": float(np.median(rates[mask])),
                    }
                )
    return pd.DataFrame(rows)


def diagnostic_row(
    frame: pd.DataFrame,
    label: str,
    round_id: int,
    scope: str,
    bin_value: str | int,
    cutoff: float,
) -> dict[str, Any]:
    scored = frame["dock_score"].notna()
    hits = frame["dock_score"] <= cutoff
    return {
        "run": label,
        "round": round_id,
        "scope": scope,
        "bin": bin_value,
        "selected": len(frame),
        "scored": int(scored.sum()),
        "hits": int(hits.sum()),
        "hit_rate_scored": float(hits.sum() / scored.sum()) if scored.any() else math.nan,
        "hit_rate_selected": float(hits.mean()),
        "pred_score_mean": float(frame["pred_score"].mean()),
        "pred_score_median": float(frame["pred_score"].median()),
        "epistemic_std_mean": float(frame["epistemic_std"].mean()),
        "epistemic_std_median": float(frame["epistemic_std"].median()),
        "base_score_mean": float(frame["base_score"].mean()),
        "cluster_penalty_mean": float(frame["cluster_penalty"].mean()),
        "dock_score_mean": float(frame["dock_score"].mean()),
        "unique_clusters": int(frame["clusterid"].nunique()),
        "first_cluster_selection_fraction": float(
            np.mean(frame["cluster_count_before"] == 0)
        ),
    }


def acquisition_diagnostics(
    runs: list[RunCohort], acquisition_dirs: dict[str, Path], cutoff: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_label = {run.label: run for run in runs}
    rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    for label, directory in acquisition_dirs.items():
        outcome = by_label[label].frame.set_index("spacehastenid")[
            ["dock_score", "dock_iteration"]
        ]
        files = sorted(
            directory.glob("iter*/acquisition.csv"),
            key=lambda path: int(path.parent.name.removeprefix("iter")),
        )
        for path in files:
            round_id = int(path.parent.name.removeprefix("iter"))
            frame = pd.read_csv(path)
            if len(frame) != 100_000 or frame["spacehastenid"].duplicated().any():
                raise ValueError(f"Invalid acquisition file: {path}")
            base_q05, base_q95 = frame["base_score"].quantile([0.05, 0.95])
            penalty_q05, penalty_q95 = frame["cluster_penalty"].quantile(
                [0.05, 0.95]
            )
            base_span = float(base_q95 - base_q05)
            penalty_span = float(penalty_q95 - penalty_q05)
            scale_rows.append(
                {
                    "run": label,
                    "round": round_id,
                    "base_score_mean": float(frame["base_score"].mean()),
                    "base_score_sd": float(frame["base_score"].std()),
                    "base_score_q05": float(base_q05),
                    "base_score_q95": float(base_q95),
                    "base_score_q90_span": base_span,
                    "cluster_penalty_mean": float(frame["cluster_penalty"].mean()),
                    "cluster_penalty_sd": float(frame["cluster_penalty"].std()),
                    "cluster_penalty_q05": float(penalty_q05),
                    "cluster_penalty_q95": float(penalty_q95),
                    "cluster_penalty_q90_span": penalty_span,
                    "penalty_span_over_base_span": penalty_span / base_span,
                    "unique_clusters": int(frame["clusterid"].nunique()),
                    "first_cluster_selection_fraction": float(
                        np.mean(frame["cluster_count_before"] == 0)
                    ),
                }
            )
            frame = frame.join(outcome, on="spacehastenid", rsuffix="_observed")
            wrong_round = frame["dock_iteration"].notna() & (
                frame["dock_iteration"] != round_id
            )
            frame.loc[wrong_round, ["dock_score", "dock_iteration"]] = np.nan
            rows.append(
                diagnostic_row(frame, label, round_id, "overall", "all", cutoff)
            )
            for metric in ("pred_score", "epistemic_std", "cluster_penalty"):
                deciles = pd.qcut(
                    frame[metric].rank(method="first"), 10, labels=False
                )
                for decile, group in frame.assign(decile=deciles).groupby("decile"):
                    rows.append(
                        diagnostic_row(
                            group,
                            label,
                            round_id,
                            f"{metric}_decile",
                            int(decile) + 1,
                            cutoff,
                        )
                    )
    return pd.DataFrame(rows), pd.DataFrame(scale_rows)


def save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def configure_plots() -> None:
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


def make_plots(
    output: Path,
    full: pd.DataFrame,
    count_replicates: pd.DataFrame,
    round_replicates: pd.DataFrame,
    potency: pd.DataFrame,
    selected: pd.DataFrame,
    dpi: int,
) -> None:
    configure_plots()
    fixed_label = "SVDKL-EI"
    fixed_size = int(
        full.loc[full["run"] == fixed_label, "cohort_size"].iloc[0]
    )
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for axis, metric, title in zip(
        axes,
        ("sampled_pair_internal_diversity", "atlas_occupied_clusters"),
        ("Internal diversity", "Common-atlas occupancy"),
        strict=True,
    ):
        for position, label in enumerate(("Greedy", "SVDKL-LCB")):
            values = count_replicates.loc[
                count_replicates["sampled_run"] == label, metric
            ]
            violin = axis.violinplot(values, positions=[position], showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(COLORS[label])
                body.set_alpha(0.35)
        fixed_value = full.loc[full["run"] == fixed_label, metric].iloc[0]
        axis.axhline(
            fixed_value,
            color=COLORS[fixed_label],
            linestyle="--",
            label=f"EI full (n={fixed_size:,})",
        )
        axis.set_xticks([0, 1], ["Greedy\nmatched", "LCB\nmatched"])
        axis.set_title(title)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, output / "01_full_count_matched_diversity", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for axis, metric, title in zip(
        axes,
        ("typed_scaffold_richness", "generic_framework_richness"),
        ("Typed Bemis-Murcko scaffolds", "Generic Murcko frameworks"),
        strict=True,
    ):
        for position, label in enumerate(("Greedy", "SVDKL-LCB")):
            values = count_replicates.loc[
                count_replicates["sampled_run"] == label, metric
            ]
            violin = axis.violinplot(values, positions=[position], showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(COLORS[label])
                body.set_alpha(0.35)
        fixed_value = full.loc[full["run"] == fixed_label, metric].iloc[0]
        axis.axhline(fixed_value, color=COLORS[fixed_label], linestyle="--")
        axis.set_xticks([0, 1], ["Greedy\nmatched", "LCB\nmatched"])
        axis.set_title(title)
        axis.set_ylabel("Richness at matched hit count")
    figure.tight_layout()
    save_figure(figure, output / "02_scaffold_framework_richness", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for axis, metric, title in zip(
        axes,
        ("sampled_pair_internal_diversity", "generic_framework_richness"),
        ("Internal diversity", "Generic-framework richness"),
        strict=True,
    ):
        positions = []
        labels = []
        position = 0
        for round_id in (1, 2):
            for label in ("Greedy", "SVDKL-LCB"):
                values = round_replicates.loc[
                    (round_replicates["round"] == round_id)
                    & (round_replicates["sampled_run"] == label),
                    metric,
                ]
                violin = axis.violinplot(values, positions=[position], showextrema=False)
                for body in violin["bodies"]:
                    body.set_facecolor(COLORS[label])
                    body.set_alpha(0.35)
                positions.append(position)
                labels.append(f"R{round_id}\n{label.replace('SVDKL-', '')}")
                position += 1
            ei_row = full[
                (full["run"] == fixed_label)
                & (full["cohort_type"] == f"round_{round_id}_hits")
            ]
            axis.hlines(
                ei_row[metric].iloc[0],
                position - 2.3,
                position - 0.7,
                color=COLORS[fixed_label],
                linestyle="--",
                label=("EI fixed cohort" if round_id == 1 else None),
            )
            position += 0.6
        axis.set_xticks(positions, labels)
        axis.set_title(title)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, output / "03_round_matched_diversity", dpi)

    metrics = (
        "sampled_pair_internal_diversity",
        "typed_scaffold_richness",
        "generic_framework_richness",
        "atlas_occupied_clusters",
    )
    figure, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for axis, metric in zip(axes.flat, metrics, strict=True):
        for top_k, group in potency.groupby("top_k"):
            x = np.arange(len(group)) + (-0.12 if top_k == 25_000 else 0.12)
            axis.scatter(
                x,
                group[metric],
                c=[COLORS[label] for label in group["run"]],
                marker="o" if top_k == 25_000 else "s",
                label=f"top {top_k:,}",
            )
        axis.set_xticks(np.arange(3), ["Greedy", "LCB", "EI"])
        axis.set_title(metric.replace("_", " "))
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, output / "04_potency_matched_diversity", dpi)

    overall = selected[selected["cohort_type"] == "all_scored_selections"]
    rounds = selected[selected["cohort_type"].str.startswith("round_")]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for run, group in rounds.groupby("run"):
        group = group.sort_values("round")
        axes[0].plot(
            group["sampled_pair_internal_diversity"],
            100 * group["hit_rate_scored"],
            color=COLORS[run],
            marker=MARKERS[run],
            alpha=0.55,
        )
    for _, row in overall.iterrows():
        axes[0].scatter(
            row["sampled_pair_internal_diversity"],
            100 * row["hit_rate_scored"],
            color=COLORS[row["run"]],
            marker=MARKERS[row["run"]],
            s=65,
            label=row["run"],
        )
        axes[1].scatter(
            1000 * row["generic_framework_richness_per_molecule"],
            100 * row["hit_rate_scored"],
            color=COLORS[row["run"]],
            marker=MARKERS[row["run"]],
            s=65,
            label=row["run"],
        )
    axes[0].set(
        xlabel="Selected-cohort internal diversity",
        ylabel="Observed hit rate (%)",
        title="Selection breadth versus yield",
    )
    axes[1].set(
        xlabel="Generic frameworks per 1,000 scored selections",
        ylabel="Observed hit rate (%)",
        title="Framework breadth versus yield",
    )
    axes[0].legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, output / "05_selected_breadth_vs_yield", dpi)


def calculate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    run_paths = parse_assignments(args.run, Path)
    acquisition_dirs = parse_assignments(args.acquisition_dir, Path)
    expected_hits = parse_assignments(args.expected_hits, int)
    if list(run_paths) != ["Greedy", "SVDKL-LCB", "SVDKL-EI"]:
        raise ValueError("Runs must be ordered as Greedy, SVDKL-LCB, SVDKL-EI")
    if set(acquisition_dirs) != {"SVDKL-LCB", "SVDKL-EI"}:
        raise ValueError("Acquisition directories are required for LCB and EI")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        output / name
        for name in (
            "full_set_comparison.csv",
            "selected_cohort_diversity.csv",
            "count_matched_replicates.csv",
            "count_matched_empirical_intervals.csv",
            "round_matched_replicates.csv",
            "round_matched_comparison.csv",
            "potency_matched_comparison.csv",
            "scaffold_hit_conversion.csv",
            "acquisition_diversity_diagnostics.csv",
            "acquisition_score_scale.csv",
            "analysis_summary.json",
        )
    ]
    expected_outputs.extend(
        output / f"{stem}.{suffix}"
        for stem in (
            "01_full_count_matched_diversity",
            "02_scaffold_framework_richness",
            "03_round_matched_diversity",
            "04_potency_matched_diversity",
            "05_selected_breadth_vs_yield",
        )
        for suffix in ("png", "pdf")
    )
    if any(path.exists() for path in expected_outputs) and not args.force:
        raise FileExistsError("Output artifacts already exist; use --force to replace")

    runs = [
        load_scored_virtual(label, path, args.cutoff)
        for label, path in run_paths.items()
    ]
    observed_hits = {run.label: int(run.frame["is_hit"].sum()) for run in runs}
    if observed_hits != expected_hits:
        raise ValueError(f"Unexpected hit counts: {observed_hits}")
    structures = build_structure_data(runs, args.processes)
    atlas_metadata = build_common_hit_atlas(runs, structures)

    full_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for run_index, run in enumerate(runs):
        hit_frame = run.frame[run.frame["is_hit"]]
        hit_indices = hit_frame["structure_index"].to_numpy(dtype=np.int64)
        full_rows.append(
            {
                "run": run.label,
                "cohort_type": "all_hits",
                "round": "overall",
                **cohort_metrics(
                    hit_indices,
                    structures,
                    np.random.default_rng(args.random_seed + run_index),
                    args.pair_samples,
                    True,
                ),
            }
        )
        for round_id in (1, 2):
            round_hits = hit_frame[hit_frame["dock_iteration"] == round_id]
            full_rows.append(
                {
                    "run": run.label,
                    "cohort_type": f"round_{round_id}_hits",
                    "round": round_id,
                    **cohort_metrics(
                        round_hits["structure_index"].to_numpy(dtype=np.int64),
                        structures,
                        np.random.default_rng(
                            args.random_seed + 100 * run_index + round_id
                        ),
                        args.pair_samples,
                        True,
                    ),
                }
            )
        selected_rows.append(
            {
                "run": run.label,
                "cohort_type": "all_scored_selections",
                "round": "overall",
                "hit_count": int(run.frame["is_hit"].sum()),
                "hit_rate_scored": float(run.frame["is_hit"].mean()),
                **cohort_metrics(
                    run.frame["structure_index"].to_numpy(dtype=np.int64),
                    structures,
                    np.random.default_rng(args.random_seed + 1_000 + run_index),
                    args.pair_samples,
                    False,
                ),
            }
        )
        for round_id in (1, 2):
            round_frame = run.frame[run.frame["dock_iteration"] == round_id]
            selected_rows.append(
                {
                    "run": run.label,
                    "cohort_type": f"round_{round_id}_scored_selections",
                    "round": round_id,
                    "hit_count": int(round_frame["is_hit"].sum()),
                    "hit_rate_scored": float(round_frame["is_hit"].mean()),
                    **cohort_metrics(
                        round_frame["structure_index"].to_numpy(dtype=np.int64),
                        structures,
                        np.random.default_rng(
                            args.random_seed + 2_000 + 100 * run_index + round_id
                        ),
                        args.pair_samples,
                        False,
                    ),
                }
            )
    full = pd.DataFrame(full_rows)
    selected = pd.DataFrame(selected_rows)

    ei_full_metrics = (
        full[(full["run"] == "SVDKL-EI") & (full["cohort_type"] == "all_hits")]
        .iloc[0]
        .drop(["run", "cohort_type", "round"])
        .to_dict()
    )
    ei_round_metrics = {
        round_id: (
            full[
                (full["run"] == "SVDKL-EI")
                & (full["cohort_type"] == f"round_{round_id}_hits")
            ]
            .iloc[0]
            .drop(["run", "cohort_type", "round"])
            .to_dict()
        )
        for round_id in (1, 2)
    }

    count_replicates, count_intervals = matched_analysis(
        runs,
        structures,
        "SVDKL-EI",
        ei_full_metrics,
        args.replicates,
        args.replicate_pair_samples,
        args.random_seed,
    )
    round_replicates, round_intervals = round_matched_analysis(
        runs,
        structures,
        "SVDKL-EI",
        ei_round_metrics,
        args.replicates,
        args.replicate_pair_samples,
        args.random_seed,
    )

    potency_rows: list[dict[str, Any]] = []
    for top_k in args.top_k:
        for run_index, run in enumerate(runs):
            frame = run.frame[run.frame["is_hit"]].nsmallest(top_k, "dock_score")
            if len(frame) != top_k:
                raise ValueError(f"{run.label} has fewer than {top_k} hits")
            potency_rows.append(
                {
                    "run": run.label,
                    "top_k": top_k,
                    **cohort_metrics(
                        frame["structure_index"].to_numpy(dtype=np.int64),
                        structures,
                        np.random.default_rng(
                            args.random_seed + top_k + run_index
                        ),
                        args.pair_samples,
                        True,
                    ),
                }
            )
    potency = pd.DataFrame(potency_rows)
    conversion = scaffold_conversion(runs, structures)
    diagnostics, score_scale = acquisition_diagnostics(
        runs, acquisition_dirs, args.cutoff
    )

    tables = {
        "full_set_comparison.csv": full,
        "selected_cohort_diversity.csv": selected,
        "count_matched_replicates.csv": count_replicates,
        "count_matched_empirical_intervals.csv": count_intervals,
        "round_matched_replicates.csv": round_replicates,
        "round_matched_comparison.csv": round_intervals,
        "potency_matched_comparison.csv": potency,
        "scaffold_hit_conversion.csv": conversion,
        "acquisition_diversity_diagnostics.csv": diagnostics,
        "acquisition_score_scale.csv": score_scale,
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False)
    make_plots(
        output,
        full,
        count_replicates,
        round_replicates,
        potency,
        selected,
        args.dpi,
    )
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "definition": {
            "cutoff": args.cutoff,
            "fingerprint": "binary Morgan radius 2, 1024 bits",
            "fixed_count_matched_run": "SVDKL-EI",
            "matched_n": expected_hits["SVDKL-EI"],
            "replicates": args.replicates,
            "pair_samples_full": args.pair_samples,
            "pair_samples_per_replicate": args.replicate_pair_samples,
            "random_seed": args.random_seed,
            "top_k": args.top_k,
        },
        "inputs": {
            "runs": {label: str(path.resolve()) for label, path in run_paths.items()},
            "acquisition_directories": {
                label: str(path.resolve()) for label, path in acquisition_dirs.items()
            },
        },
        "atlas": atlas_metadata,
        "validation": {
            "observed_hits": observed_hits,
            "unique_structure_count": len(structures.hashes),
            "count_replicates": len(count_replicates),
            "round_replicates": len(round_replicates),
            "count_replicates_exact_n": bool(
                (count_replicates["cohort_size"] == expected_hits["SVDKL-EI"]).all()
            ),
            "all_outputs_nonempty": True,
        },
        "full_hit_metrics": full[full["cohort_type"] == "all_hits"].to_dict(
            orient="records"
        ),
        "selected_cohort_metrics": selected[
            selected["cohort_type"] == "all_scored_selections"
        ].to_dict(orient="records"),
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    for path in expected_outputs:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty output: {path}")
    LOGGER.info(
        "Core diversity comparison complete in %.1f minutes",
        (time.monotonic() - started) / 60,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DB")
    parser.add_argument(
        "--acquisition-dir", action="append", required=True, metavar="LABEL=DIR"
    )
    parser.add_argument(
        "--expected-hits", action="append", required=True, metavar="LABEL=COUNT"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=-9.7)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--pair-samples", type=int, default=1_000_000)
    parser.add_argument("--replicate-pair-samples", type=int, default=100_000)
    parser.add_argument("--top-k", type=int, action="append", default=[])
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.top_k:
        args.top_k = [25_000, 50_000]
    if min(
        args.replicates,
        args.pair_samples,
        args.replicate_pair_samples,
        args.processes,
        args.dpi,
    ) < 1:
        parser.error("Replicates, samples, processes, and DPI must be positive")
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    try:
        calculate(parse_args())
    except Exception:
        LOGGER.exception("Core diversity comparison failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
