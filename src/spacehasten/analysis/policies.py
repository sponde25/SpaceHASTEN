"""Deterministic diversity selectors and reusable structure metrics."""

from __future__ import annotations

import heapq
import math
import multiprocessing as mp
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from spacehasten.analysis.chemistry import family_labels

FloatArray = npt.NDArray[np.float64]
Int16Array = npt.NDArray[np.int16]
Int32Array = npt.NDArray[np.int32]
Int64Array = npt.NDArray[np.int64]
UInt64Array = npt.NDArray[np.uint64]
ObjectArray = npt.NDArray[np.object_]

BYTE_POPCOUNT = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)
)


@dataclass(frozen=True, slots=True)
class PolicySelection:
    name: str
    indices: Int64Array
    cluster_count_before: Int32Array
    penalties: FloatArray


@dataclass(slots=True)
class RunCohort:
    label: str
    database: Path
    frame: pd.DataFrame


@dataclass(frozen=True, slots=True)
class StructureData:
    hashes: ObjectArray
    words: UInt64Array
    popcounts: Int16Array
    typed_scaffolds: ObjectArray
    generic_frameworks: ObjectArray
    atlas_codes: Int32Array


def _validated_selector_inputs(
    scores: FloatArray,
    identifiers: Int64Array,
    clusters: Int64Array | None,
    count: int,
    cluster_lambda: float | None = None,
    cap: int | None = None,
) -> tuple[FloatArray, Int64Array, Int64Array | None]:
    scores_array = np.asarray(scores, dtype=np.float64)
    identifiers_array = np.asarray(identifiers, dtype=np.int64)
    clusters_array = None if clusters is None else np.asarray(clusters, dtype=np.int64)
    if scores_array.ndim != 1 or identifiers_array.ndim != 1:
        raise ValueError("scores and identifiers must be one-dimensional")
    if len(scores_array) != len(identifiers_array):
        raise ValueError("scores and identifiers must have equal length")
    if clusters_array is not None and (
        clusters_array.ndim != 1 or len(clusters_array) != len(scores_array)
    ):
        raise ValueError("clusters must be one-dimensional and match scores")
    if not np.isfinite(scores_array).all():
        raise ValueError("scores must be finite")
    if count < 0 or count > len(scores_array):
        raise ValueError("count must be between zero and the number of candidates")
    if cluster_lambda is not None and not math.isfinite(cluster_lambda):
        raise ValueError("cluster_lambda must be finite")
    if cap is not None and cap < 1:
        raise ValueError("cap must be at least one")
    return scores_array, identifiers_array, clusters_array


def deterministic_top_k(
    scores: FloatArray,
    identifiers: Int64Array,
    count: int,
) -> Int64Array:
    """Return the lowest scores, breaking ties by identifier."""
    scores_array, identifiers_array, _ = _validated_selector_inputs(
        scores, identifiers, None, count
    )
    return cast(Int64Array, np.lexsort((identifiers_array, scores_array))[:count])


def penalized_top_k(
    scores: FloatArray,
    identifiers: Int64Array,
    clusters: Int64Array,
    count: int,
    cluster_lambda: float,
    cap: int | None = None,
) -> tuple[Int64Array, Int32Array, FloatArray]:
    """Select with historical score/identifier/cluster tie-breaking."""
    scores_array, identifiers_array, clusters_array = _validated_selector_inputs(
        scores, identifiers, clusters, count, cluster_lambda, cap
    )
    assert clusters_array is not None
    if cap is not None:
        _, group_sizes = np.unique(clusters_array, return_counts=True)
        if int(np.minimum(group_sizes, cap).sum()) < count:
            raise ValueError("cap capacity cannot fill count")

    order = np.lexsort((identifiers_array, scores_array, clusters_array))
    group_starts = np.flatnonzero(
        np.r_[True, clusters_array[order][1:] != clusters_array[order][:-1]]
    )
    group_stops = np.r_[group_starts[1:], len(order)]
    heap: list[tuple[float, float, int, int, int, int]] = []
    for start, stop in zip(group_starts, group_stops, strict=True):
        index = int(order[start])
        heapq.heappush(
            heap,
            (
                float(scores_array[index]),
                float(scores_array[index]),
                int(identifiers_array[index]),
                int(start),
                int(stop),
                0,
            ),
        )

    selected = np.empty(count, dtype=np.int64)
    counts = np.empty(count, dtype=np.int32)
    penalties = np.empty(count, dtype=np.float64)
    for position in range(count):
        penalized, raw, _, row, stop, previous = heapq.heappop(heap)
        selected[position] = order[row]
        counts[position] = previous
        penalties[position] = penalized - raw
        next_row = row + 1
        next_count = previous + 1
        if next_row < stop and (cap is None or next_count < cap):
            index = int(order[next_row])
            value = float(scores_array[index])
            heapq.heappush(
                heap,
                (
                    value + cluster_lambda * math.log1p(next_count),
                    value,
                    int(identifiers_array[index]),
                    next_row,
                    int(stop),
                    next_count,
                ),
            )
    return selected, counts, penalties


def capped_top_k(
    scores: FloatArray,
    identifiers: Int64Array,
    clusters: Int64Array,
    count: int,
    cluster_lambda: float,
    cap: int,
) -> tuple[Int64Array, Int32Array, FloatArray]:
    return penalized_top_k(scores, identifiers, clusters, count, cluster_lambda, cap)


def load_compounds(database: Path, identifiers: set[int]) -> pd.DataFrame:
    workspace = sqlite3.connect(":memory:", uri=True)
    try:
        source_uri = f"file:{database.resolve()}?mode=ro"
        workspace.execute("ATTACH DATABASE ? AS source", (source_uri,))
        workspace.execute("CREATE TABLE selected(spacehastenid INTEGER PRIMARY KEY)")
        workspace.executemany(
            "INSERT INTO selected VALUES (?)", ((value,) for value in sorted(identifiers))
        )
        return pd.read_sql_query(
            "SELECT d.spacehastenid,d.reghash,d.smiles,d.dock_score "
            "FROM source.data d JOIN selected s USING(spacehastenid)",
            workspace,
        )
    finally:
        workspace.close()


def load_greedy_score_map(path: Path) -> dict[str, float]:
    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        return {
            str(reghash): float(score)
            for reghash, score in connection.execute(
                "SELECT reghash,dock_score FROM data WHERE dock_iteration > 0 "
                "AND dock_score IS NOT NULL"
            )
        }


def _molecule_from_smiles(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"unparseable SMILES: {smiles}")
    return molecule


def _scaffold(smiles: str) -> tuple[str, str]:
    return family_labels(_molecule_from_smiles(smiles))


def _fingerprint_binary(smiles: str) -> tuple[bytes, int]:
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(
        _molecule_from_smiles(smiles)
    )
    return DataStructs.BitVectToBinaryText(fingerprint), fingerprint.GetNumOnBits()


def add_structures(frame: pd.DataFrame) -> tuple[pd.DataFrame, UInt64Array, Int16Array]:
    result = frame.copy()
    scaffolds = [_scaffold(str(smiles)) for smiles in result.smiles]
    result[["typed_scaffold", "generic_framework"]] = pd.DataFrame(scaffolds, index=result.index)
    fingerprints = [_fingerprint_binary(str(smiles)) for smiles in result.smiles]
    binaries, counts = zip(*fingerprints, strict=True)
    words = np.frombuffer(b"".join(binaries), dtype=np.uint64).reshape(len(result), 16)
    result["structure_index"] = np.arange(len(result), dtype=np.int64)
    return result, words, np.asarray(counts, dtype=np.int16)


def structure_worker(smiles: str) -> tuple[str, str, bytes, int]:
    typed, generic = _scaffold(smiles)
    binary, count = _fingerprint_binary(smiles)
    return typed, generic, binary, count


def build_structure_data(runs: list[RunCohort], processes: int) -> StructureData:
    if processes < 1:
        raise ValueError("processes must be at least one")
    combined = pd.concat(
        [run.frame[["reghash", "smiles"]] for run in runs], ignore_index=True
    ).drop_duplicates()
    if (combined.groupby("reghash", sort=False).smiles.nunique() > 1).any():
        raise ValueError("reghashes map to multiple SMILES")
    structures = combined.drop_duplicates("reghash").sort_values("reghash").reset_index(drop=True)
    if structures.empty:
        raise ValueError("no structures are available")
    chunksize = max(1, len(structures) // (processes * 16))
    with mp.get_context("fork").Pool(processes) as pool:
        values = list(pool.imap(structure_worker, structures.smiles, chunksize=chunksize))
    typed, generic, binaries, counts = zip(*values, strict=True)
    data = StructureData(
        hashes=structures.reghash.to_numpy(dtype=object),
        words=np.frombuffer(b"".join(binaries), dtype=np.uint64)
        .reshape(len(structures), 16)
        .copy(),
        popcounts=np.asarray(counts, dtype=np.int16),
        typed_scaffolds=np.asarray(typed, dtype=object),
        generic_frameworks=np.asarray(generic, dtype=object),
        atlas_codes=np.full(len(structures), -1, dtype=np.int32),
    )
    index = pd.Series(np.arange(len(structures), dtype=np.int64), index=data.hashes)
    for run in runs:
        run.frame["structure_index"] = run.frame.reghash.map(index)
        if run.frame.structure_index.isna().any():
            raise ValueError(f"{run.label}: incomplete structure join")
        run.frame["structure_index"] = run.frame.structure_index.astype(np.int64)
    return data


def effective_numbers(row: pd.Series) -> dict[str, float]:
    return {
        "typed_scaffold_effective_number": math.exp(row["typed_scaffold_entropy"]),
        "generic_framework_effective_number": math.exp(row["generic_framework_entropy"]),
        "atlas_effective_clusters": math.exp(row["atlas_entropy"]),
    }


def _distribution_metrics(counts: npt.NDArray[np.int64], prefix: str) -> dict[str, float | int]:
    total = int(counts.sum())
    probabilities = counts / total
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = entropy / math.log(len(counts)) if len(counts) > 1 else 0.0
    return {
        f"{prefix}_richness": len(counts),
        f"{prefix}_richness_per_compound": len(counts) / total,
        f"{prefix}_largest_fraction": int(counts.max()) / total,
        f"{prefix}_entropy": entropy,
        f"{prefix}_normalized_entropy": normalized_entropy,
    }


def _family_metrics(values: pd.Series, prefix: str) -> dict[str, float | int]:
    return _distribution_metrics(values.value_counts().to_numpy(dtype=np.int64), prefix)


def _cluster_metrics(clusters: Int64Array) -> dict[str, float | int]:
    _, counts = np.unique(np.asarray(clusters, dtype=np.int64), return_counts=True)
    metrics = _distribution_metrics(counts.astype(np.int64), "atlas")
    return {
        "atlas_occupied_clusters": metrics["atlas_richness"],
        "atlas_clusters_per_compound": metrics["atlas_richness_per_compound"],
        "atlas_largest_fraction": metrics["atlas_largest_fraction"],
        "atlas_entropy": metrics["atlas_entropy"],
        "atlas_normalized_entropy": metrics["atlas_normalized_entropy"],
    }


family_metrics = _family_metrics
cluster_metrics = _cluster_metrics


def internal_diversity(
    indices: Int64Array,
    words: UInt64Array,
    popcounts: Int16Array,
    pair_samples: int,
    rng: np.random.Generator,
) -> float:
    if len(indices) < 2:
        raise ValueError("at least two selected structures are required")
    if pair_samples < 1:
        raise ValueError("pair_samples must be positive")
    total = 0.0
    index_array = np.asarray(indices, dtype=np.int64)
    popcount_array = np.asarray(popcounts, dtype=np.int64)
    for start in range(0, pair_samples, 250_000):
        size = min(250_000, pair_samples - start)
        left = rng.integers(0, len(index_array), size=size)
        right = rng.integers(0, len(index_array), size=size)
        while (same := left == right).any():
            right[same] = rng.integers(0, len(index_array), size=int(same.sum()))
        left_rows = index_array[left]
        right_rows = index_array[right]
        intersection = BYTE_POPCOUNT[
            np.bitwise_and(words[left_rows], words[right_rows]).view(np.uint8)
        ].sum(axis=1)
        union = popcount_array[left_rows] + popcount_array[right_rows] - intersection
        total += float(np.sum(intersection / union))
    return 1.0 - total / pair_samples


def policy_metrics(
    round_id: int,
    name: str,
    selected_ids: Int64Array,
    selected_clusters: Int64Array,
    info: pd.DataFrame,
    words: UInt64Array,
    popcounts: Int16Array,
    cutoff: float,
    pair_samples: int,
    random_seed: int,
) -> dict[str, object]:
    selected = info.set_index("spacehastenid").loc[selected_ids]
    observed = selected.observed_score.notna().to_numpy()
    scores = selected.observed_score.to_numpy(dtype=np.float64)[observed]
    return {
        "round": round_id,
        "policy": name,
        "selected": len(selected),
        "observed_outcomes": int(observed.sum()),
        "outcome_coverage": float(observed.mean()),
        "observed_hit_count": int(np.sum(scores <= cutoff)),
        "observed_hit_rate": float(np.mean(scores <= cutoff)) if len(scores) else math.nan,
        "observed_mean_dock_score": float(np.mean(scores)) if len(scores) else math.nan,
        "internal_diversity": internal_diversity(
            selected.structure_index.to_numpy(dtype=np.int64),
            words,
            popcounts,
            pair_samples,
            np.random.default_rng(random_seed),
        ),
        **_family_metrics(selected.typed_scaffold, "typed_scaffold"),
        **_family_metrics(selected.generic_framework, "generic_framework"),
        **_cluster_metrics(selected_clusters),
    }


def pairwise_policy_overlap(round_id: int, selections: dict[str, Int64Array]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    names = list(selections)
    for position, left in enumerate(names):
        left_set = set(map(int, selections[left]))
        for right in names[position:]:
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
