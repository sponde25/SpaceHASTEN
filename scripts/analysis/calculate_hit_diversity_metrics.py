#!/usr/bin/env python3
"""Calculate reusable structural-diversity metrics for virtual hits."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from FPSim2 import FPSim2Engine
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.SimDivFilters import rdSimDivPickers
from tqdm import tqdm

from spacehasten.remote.cluster import _build_fpsim2_index, _generate_fingerprints

LOGGER = logging.getLogger("calculate_hit_diversity_metrics")
ACYCLIC = "[ACYCLIC]"
BYTE_POPCOUNT = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)
)


def load_unique_hits(
    database: Path,
    cutoff: float,
    limit: int | None,
) -> tuple[list[tuple[int, str, str]], dict[str, int]]:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    query = (
        "SELECT spacehastenid, reghash, smiles FROM data "
        "WHERE dock_iteration > 0 AND dock_score <= ? "
        "ORDER BY reghash, spacehastenid"
    )
    rows = connection.execute(query, (cutoff,))
    unique: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    raw_count = 0
    null_hash_count = 0
    duplicate_count = 0
    for spacehastenid, reghash, smiles in tqdm(rows, desc="Loading virtual hits"):
        raw_count += 1
        if reghash is None or not str(reghash).strip():
            null_hash_count += 1
            continue
        normalized_hash = str(reghash)
        if normalized_hash in seen:
            duplicate_count += 1
            continue
        seen.add(normalized_hash)
        unique.append((int(spacehastenid), normalized_hash, str(smiles)))
        if limit is not None and len(unique) >= limit:
            break
    connection.close()
    if not unique:
        raise ValueError("no virtual hits matched the requested cutoff")
    diagnostics = {
        "raw_hit_rows_examined": raw_count,
        "unique_reghashes": len(unique),
        "duplicate_reghashes_removed": duplicate_count,
        "missing_reghashes_excluded": null_hash_count,
    }
    return unique, diagnostics


def scaffold_worker(smiles: str) -> tuple[str | None, str | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return ACYCLIC, ACYCLIC
    typed = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)
    generic_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    generic = Chem.MolToSmiles(generic_mol, canonical=True, isomericSmiles=False)
    return typed, generic


def calculate_scaffolds(
    smiles: list[str],
    processes: int,
) -> tuple[Counter[str], Counter[str], int]:
    typed: Counter[str] = Counter()
    generic: Counter[str] = Counter()
    parse_failures = 0
    chunksize = max(1, len(smiles) // (processes * 8))
    with mp.Pool(processes) as pool:
        results = pool.imap(scaffold_worker, smiles, chunksize=chunksize)
        for typed_scaffold, generic_scaffold in tqdm(
            results,
            total=len(smiles),
            desc="Calculating Murcko scaffolds",
            unit="mol",
        ):
            if typed_scaffold is None or generic_scaffold is None:
                parse_failures += 1
                continue
            typed[typed_scaffold] += 1
            generic[generic_scaffold] += 1
    return typed, generic, parse_failures


def fingerprints_to_words(fingerprints: list) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    packed = b"".join(DataStructs.BitVectToBinaryText(fp) for fp in fingerprints)
    words = np.frombuffer(packed, dtype=np.uint64).reshape(len(fingerprints), 16)
    popcounts = np.fromiter(
        (fp.GetNumOnBits() for fp in fingerprints),
        dtype=np.int16,
        count=len(fingerprints),
    )
    return words, popcounts


def sample_internal_diversity(
    words: np.ndarray,
    popcounts: np.ndarray,
    sample_count: int,
    random_seed: int,
    batch_size: int,
) -> dict[str, float | int]:
    if len(words) < 2:
        raise ValueError("internal diversity requires at least two compounds")
    rng = np.random.default_rng(random_seed)
    similarity_sum = 0.0
    similarity_sq_sum = 0.0
    processed = 0
    with tqdm(total=sample_count, desc="Sampling distinct fingerprint pairs", unit="pair") as bar:
        while processed < sample_count:
            size = min(batch_size, sample_count - processed)
            left = rng.integers(0, len(words), size=size, dtype=np.int64)
            right = rng.integers(0, len(words), size=size, dtype=np.int64)
            equal = left == right
            while np.any(equal):
                right[equal] = rng.integers(0, len(words), size=int(equal.sum()), dtype=np.int64)
                equal = left == right
            intersection_words = np.bitwise_and(words[left], words[right])
            intersection = BYTE_POPCOUNT[intersection_words.view(np.uint8).reshape(size, -1)].sum(
                axis=1, dtype=np.int32
            )
            union = popcounts[left] + popcounts[right] - intersection
            similarity = intersection / union
            similarity_sum += float(similarity.sum(dtype=np.float64))
            similarity_sq_sum += float(np.square(similarity).sum(dtype=np.float64))
            processed += size
            bar.update(size)

    mean = similarity_sum / sample_count
    variance = max(
        (similarity_sq_sum - sample_count * mean * mean) / (sample_count - 1),
        0.0,
    )
    standard_error = math.sqrt(variance / sample_count)
    mean_low = max(0.0, mean - 1.96 * standard_error)
    mean_high = min(1.0, mean + 1.96 * standard_error)
    return {
        "sampled_distinct_pairs": sample_count,
        "random_seed": random_seed,
        "mean_pairwise_tanimoto": mean,
        "mean_pairwise_tanimoto_standard_error": standard_error,
        "mean_pairwise_tanimoto_ci95_low": mean_low,
        "mean_pairwise_tanimoto_ci95_high": mean_high,
        "internal_diversity": 1.0 - mean,
        "internal_diversity_ci95_low": 1.0 - mean_high,
        "internal_diversity_ci95_high": 1.0 - mean_low,
    }


def sphere_exclusion_clusters(
    rows: list[tuple[int, str, str]],
    fingerprints: list,  # type: ignore[type-arg]
    output_dir: Path,
    similarity_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    distance_threshold = 1.0 - similarity_threshold
    picker = rdSimDivPickers.LeaderPicker()
    LOGGER.info(
        "Selecting sphere-exclusion centroids at Tanimoto >= %.2f",
        similarity_threshold,
    )
    started = time.monotonic()
    centroid_indices = list(
        picker.LazyBitVectorPick(
            fingerprints,
            len(fingerprints),
            distance_threshold,
        )
    )
    LOGGER.info(
        "Selected %d centroids in %.1f minutes",
        len(centroid_indices),
        (time.monotonic() - started) / 60.0,
    )

    index_path = output_dir / "virtual_hits_morgan_r2_1024.h5"
    index_path.unlink(missing_ok=True)
    _build_fpsim2_index([(smiles, sid) for sid, _, smiles in rows], index_path)
    engine = FPSim2Engine(str(index_path))

    max_id = max(sid for sid, _, _ in rows)
    row_by_id = np.full(max_id + 1, -1, dtype=np.int32)
    for row_index, (spacehastenid, _, _) in enumerate(rows):
        row_by_id[spacehastenid] = row_index
    assignments = np.full(len(rows), -1, dtype=np.int32)
    best_similarity = np.zeros(len(rows), dtype=np.float32)
    centroid_ids = [rows[index][0] for index in centroid_indices]

    for cluster_index, fingerprint in enumerate(
        tqdm(
            (fingerprints[index] for index in centroid_indices),
            total=len(centroid_indices),
            desc="Assigning hits to centroids",
            unit="centroid",
        )
    ):
        matches = engine.similarity(
            fingerprint,
            threshold=similarity_threshold,
            metric="tanimoto",
            n_workers=1,
        )
        matched_rows = row_by_id[np.asarray(matches["mol_id"], dtype=np.int64)]
        similarities = np.asarray(matches["coeff"], dtype=np.float32)
        better = similarities > best_similarity[matched_rows]
        chosen_rows = matched_rows[better]
        best_similarity[chosen_rows] = similarities[better]
        assignments[chosen_rows] = cluster_index

    missing = np.flatnonzero(assignments < 0)
    if len(missing):
        raise RuntimeError(f"{len(missing)} hits were not assigned to a cluster")
    return assignments, best_similarity, centroid_ids


def family_metrics(counter: Counter[str], total_hits: int) -> dict[str, float | int | str]:
    if not counter:
        raise ValueError("no valid scaffolds were generated")
    largest_name, largest_count = counter.most_common(1)[0]
    acyclic_count = int(counter.get(ACYCLIC, 0))
    return {
        "unique_families": len(counter),
        "family_hit_ratio": len(counter) / total_hits,
        "largest_family": largest_name,
        "largest_family_count": largest_count,
        "largest_family_fraction": largest_count / total_hits,
        "acyclic_count": acyclic_count,
        "acyclic_fraction": acyclic_count / total_hits,
    }


def cluster_metrics(assignments: np.ndarray) -> tuple[dict[str, float | int], np.ndarray]:
    cluster_sizes = np.bincount(assignments)
    probabilities = cluster_sizes / cluster_sizes.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    cluster_count = len(cluster_sizes)
    normalized_entropy = entropy / math.log(cluster_count) if cluster_count > 1 else 0.0
    largest_count = int(cluster_sizes.max())
    total = int(cluster_sizes.sum())
    return (
        {
            "number_of_clusters": cluster_count,
            "cluster_hit_ratio": cluster_count / total,
            "largest_cluster_count": largest_count,
            "largest_cluster_fraction": largest_count / total,
            "cluster_entropy": entropy,
            "normalized_cluster_entropy": normalized_entropy,
        },
        cluster_sizes,
    )


def write_family_table(
    path: Path,
    typed: Counter[str],
    generic: Counter[str],
    total_hits: int,
) -> None:
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family_type", "scaffold", "hit_count", "hit_fraction"])
        for family_type, counter in (("typed_murcko", typed), ("generic_murcko", generic)):
            for scaffold, count in counter.most_common():
                writer.writerow([family_type, scaffold, count, count / total_hits])


def write_cluster_tables(
    assignments_path: Path,
    summary_path: Path,
    rows: list[tuple[int, str, str]],
    assignments: np.ndarray,
    best_similarity: np.ndarray,
    centroid_ids: list[int],
    cluster_sizes: np.ndarray,
) -> None:
    with assignments_path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "spacehastenid",
                "reghash",
                "clusterid",
                "centroid_spacehastenid",
                "centroid_similarity",
            ]
        )
        for (spacehastenid, reghash, _), cluster_index, similarity in zip(
            rows, assignments, best_similarity, strict=True
        ):
            writer.writerow(
                [
                    spacehastenid,
                    reghash,
                    int(cluster_index),
                    centroid_ids[int(cluster_index)],
                    float(similarity),
                ]
            )
    with summary_path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clusterid", "centroid_spacehastenid", "hit_count", "hit_fraction"])
        total = int(cluster_sizes.sum())
        for cluster_index in np.argsort(cluster_sizes)[::-1]:
            writer.writerow(
                [
                    int(cluster_index),
                    centroid_ids[int(cluster_index)],
                    int(cluster_sizes[cluster_index]),
                    float(cluster_sizes[cluster_index] / total),
                ]
            )


def calculate(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "diversity_metrics.json"
    output_paths = [
        metrics_path,
        output_dir / "scaffold_families.csv",
        output_dir / "cluster_assignments.csv",
        output_dir / "cluster_summary.csv",
        output_dir / "virtual_hits_morgan_r2_1024.h5",
    ]
    if any(path.exists() for path in output_paths) and not args.force:
        raise FileExistsError("metric outputs already exist; use --force to replace them")
    if args.force:
        for path in output_paths:
            path.unlink(missing_ok=True)

    started = time.monotonic()
    rows, diagnostics = load_unique_hits(args.database, args.cutoff, args.limit)
    with sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True) as connection:
        database_rows, max_spacehastenid = connection.execute(
            "SELECT COUNT(*), MAX(spacehastenid) FROM data"
        ).fetchone()
    identifiers = [row[0] for row in rows]
    smiles = [row[2] for row in rows]
    LOGGER.info("Generating Morgan-2 1024-bit fingerprints for %d unique hits", len(rows))
    fingerprints = _generate_fingerprints(smiles, args.processes)

    typed, generic, scaffold_failures = calculate_scaffolds(smiles, args.processes)
    words, popcounts = fingerprints_to_words(fingerprints)
    internal_diversity = sample_internal_diversity(
        words,
        popcounts,
        args.pair_samples,
        args.random_seed,
        args.pair_batch_size,
    )
    assignments, best_similarity, centroid_ids = sphere_exclusion_clusters(
        rows,
        fingerprints,
        output_dir,
        args.cluster_similarity,
    )
    clustering, cluster_sizes = cluster_metrics(assignments)

    typed_metrics = family_metrics(typed, len(rows))
    generic_metrics = family_metrics(generic, len(rows))
    write_family_table(output_dir / "scaffold_families.csv", typed, generic, len(rows))
    write_cluster_tables(
        output_dir / "cluster_assignments.csv",
        output_dir / "cluster_summary.csv",
        rows,
        assignments,
        best_similarity,
        centroid_ids,
        cluster_sizes,
    )

    elapsed = time.monotonic() - started
    metrics: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "source_database": args.source_label or str(args.database.resolve()),
        "analysis_database": str(args.database.resolve()),
        "definition": {
            "virtual_hit_filter": f"dock_iteration > 0 AND dock_score <= {args.cutoff}",
            "dock_score_cutoff": args.cutoff,
            "uniqueness_key": "SpaceHASTEN data.reghash (RDKit tautomer hash)",
            "fingerprint": "binary Morgan radius 2, 1024 bits",
            "cluster_method": "sphere-exclusion leader clustering",
            "cluster_similarity_threshold": args.cluster_similarity,
        },
        "diagnostics": {
            **diagnostics,
            "scaffold_parse_failures": scaffold_failures,
            "fingerprint_count": len(fingerprints),
            "identifier_count": len(identifiers),
            "database_rows": int(database_rows),
            "max_spacehastenid": int(max_spacehastenid),
            "limit": args.limit,
        },
        "metrics": {
            "total_virtual_hits": len(rows),
            "internal_diversity": internal_diversity,
            "typed_bemis_murcko": typed_metrics,
            "generic_murcko": generic_metrics,
            "fingerprint_clusters": clustering,
        },
        "elapsed_seconds": elapsed,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Diversity metrics complete in %.1f minutes", elapsed / 60.0)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source-label")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, required=True)
    parser.add_argument("--cluster-similarity", type=float, default=0.4)
    parser.add_argument("--pair-samples", type=int, default=10_000_000)
    parser.add_argument("--pair-batch-size", type=int, default=250_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0 < args.cluster_similarity <= 1:
        parser.error("cluster similarity must be in (0, 1]")
    if args.pair_samples < 2:
        parser.error("pair samples must be at least 2")
    if min(args.pair_batch_size, args.processes) < 1:
        parser.error("pair batch size and processes must be positive")
    if args.limit is not None and args.limit < 2:
        parser.error("limit must be at least 2")
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        calculate(parse_args())
    except Exception:
        LOGGER.exception("Diversity metric calculation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
