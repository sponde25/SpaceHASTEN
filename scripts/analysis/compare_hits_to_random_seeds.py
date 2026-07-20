#!/usr/bin/env python3
"""Compare virtual-hit diversity with matched random samples of unique seeds."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import multiprocessing as mp
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from calculate_hit_diversity_metrics import BYTE_POPCOUNT, scaffold_worker
from FPSim2 import FPSim2Engine
from tqdm import tqdm

LOGGER = logging.getLogger("compare_hits_to_random_seeds")

METRIC_DIRECTIONS = {
    "internal_diversity": "higher",
    "typed_scaffold_count": "higher",
    "typed_scaffold_hit_ratio": "higher",
    "largest_typed_scaffold_fraction": "lower",
    "generic_framework_count": "higher",
    "generic_framework_hit_ratio": "higher",
    "largest_generic_framework_fraction": "lower",
    "atlas_occupied_cluster_count": "higher",
    "atlas_cluster_hit_ratio": "higher",
    "atlas_largest_cluster_fraction": "lower",
    "atlas_cluster_entropy": "higher",
    "atlas_normalized_cluster_entropy": "higher",
}


def load_seed_scaffolds(
    connection: sqlite3.Connection,
    max_id: int,
    processes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str], dict[int, str], int]:
    seed_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM data WHERE dock_iteration = 0 AND dock_score IS NOT NULL"
        ).fetchone()[0]
    )
    identifiers = np.empty(seed_count, dtype=np.int64)
    smiles: list[str] = []
    rows = connection.execute(
        "SELECT spacehastenid, smiles FROM data WHERE dock_iteration = 0 "
        "AND dock_score IS NOT NULL "
        "ORDER BY spacehastenid"
    )
    for index, (identifier, smile) in enumerate(
        tqdm(rows, total=seed_count, desc="Loading seed structures", unit="seed")
    ):
        identifiers[index] = int(identifier)
        smiles.append(str(smile))
    typed_by_id = np.full(max_id + 1, -1, dtype=np.int32)
    generic_by_id = np.full(max_id + 1, -1, dtype=np.int32)
    typed_to_code: dict[str, int] = {}
    generic_to_code: dict[str, int] = {}
    typed_values: dict[int, str] = {}
    generic_values: dict[int, str] = {}
    failures = 0
    chunksize = max(1, len(smiles) // (processes * 8))
    with mp.Pool(processes) as pool:
        results = pool.imap(scaffold_worker, smiles, chunksize=chunksize)
        for identifier, (typed, generic) in zip(
            identifiers,
            tqdm(
                results,
                total=len(smiles),
                desc="Caching seed scaffolds",
                unit="seed",
            ),
            strict=True,
        ):
            if typed is None or generic is None:
                failures += 1
                continue
            if typed not in typed_to_code:
                code = len(typed_to_code)
                typed_to_code[typed] = code
                typed_values[code] = typed
            if generic not in generic_to_code:
                code = len(generic_to_code)
                generic_to_code[generic] = code
                generic_values[code] = generic
            typed_by_id[identifier] = typed_to_code[typed]
            generic_by_id[identifier] = generic_to_code[generic]
    return (
        identifiers,
        typed_by_id,
        generic_by_id,
        typed_values,
        generic_values,
        failures,
    )


def load_atlas(path: Path, max_id: int) -> tuple[np.ndarray, np.ndarray]:
    spacehastenids: list[int] = []
    raw_clusters: list[int] = []
    with path.open("rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in tqdm(reader, desc="Loading common cluster atlas", unit="mol"):
            spacehastenids.append(int(row["spacehastenid"]))
            raw_clusters.append(int(row["clusterid"]))
    identifiers = np.asarray(spacehastenids, dtype=np.int64)
    if len(identifiers) == 0:
        raise ValueError("cluster atlas is empty")
    if np.any((identifiers < 0) | (identifiers > max_id)):
        raise ValueError("cluster atlas contains out-of-range compound identifiers")
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("cluster atlas contains duplicate compound identifiers")
    cluster_ids, cluster_codes = np.unique(
        np.asarray(raw_clusters, dtype=np.int64), return_inverse=True
    )
    code_by_id = np.full(max_id + 1, -1, dtype=np.int32)
    code_by_id[identifiers] = cluster_codes.astype(np.int32, copy=False)
    return code_by_id, cluster_ids


def count_metrics(codes: np.ndarray, total: int, prefix: str) -> dict[str, float | int]:
    counts = np.bincount(codes)
    counts = counts[counts > 0]
    unique_count = len(counts)
    return {
        f"{prefix}_count": unique_count,
        f"{prefix}_hit_ratio": unique_count / total,
        f"largest_{prefix}_fraction": int(counts.max()) / total,
    }


def atlas_metrics(codes: np.ndarray, total: int) -> dict[str, float | int]:
    counts = np.bincount(codes)
    counts = counts[counts > 0]
    probabilities = counts / total
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    occupied = len(counts)
    normalized = entropy / math.log(occupied) if occupied > 1 else 0.0
    return {
        "atlas_occupied_cluster_count": occupied,
        "atlas_cluster_hit_ratio": occupied / total,
        "atlas_largest_cluster_fraction": int(counts.max()) / total,
        "atlas_cluster_entropy": entropy,
        "atlas_normalized_cluster_entropy": normalized,
    }


def sample_pairwise_diversity(
    sample_rows: np.ndarray,
    words: np.ndarray,
    popcounts: np.ndarray,
    rng: np.random.Generator,
    pair_samples: int,
    batch_size: int,
) -> float:
    similarity_sum = 0.0
    processed = 0
    while processed < pair_samples:
        size = min(batch_size, pair_samples - processed)
        left = rng.integers(0, len(sample_rows), size=size, dtype=np.int64)
        right = rng.integers(0, len(sample_rows), size=size, dtype=np.int64)
        equal = left == right
        while np.any(equal):
            right[equal] = rng.integers(0, len(sample_rows), size=int(equal.sum()), dtype=np.int64)
            equal = left == right
        left_rows = sample_rows[left]
        right_rows = sample_rows[right]
        intersections = BYTE_POPCOUNT[
            np.bitwise_and(words[left_rows], words[right_rows]).view(np.uint8).reshape(size, -1)
        ].sum(axis=1, dtype=np.int32)
        unions = popcounts[left_rows] + popcounts[right_rows] - intersections
        similarity_sum += float(np.sum(intersections / unions, dtype=np.float64))
        processed += size
    return 1.0 - similarity_sum / pair_samples


def hit_ids(connection: sqlite3.Connection, cutoff: float) -> np.ndarray:
    rows = connection.execute(
        "SELECT MIN(spacehastenid) FROM data WHERE dock_iteration > 0 "
        "AND dock_score <= ? AND reghash IS NOT NULL AND TRIM(reghash) != '' "
        "GROUP BY reghash ORDER BY reghash",
        (cutoff,),
    )
    return np.fromiter((int(row[0]) for row in rows), dtype=np.int64)


def summarize_distribution(
    values: np.ndarray,
    observed: float,
    direction: str,
) -> dict[str, float | bool]:
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    q025, median, q975 = np.quantile(values, [0.025, 0.5, 0.975])
    lower_rank = (np.sum(values < observed) + 0.5 * np.sum(values == observed) + 1) / (
        len(values) + 1
    )
    favorable_percentile = lower_rank if direction == "higher" else 1.0 - lower_rank
    favorable_change = (
        (observed - mean) / mean if direction == "higher" else (mean - observed) / mean
    )
    lower_tail = (np.sum(values <= observed) + 1) / (len(values) + 1)
    upper_tail = (np.sum(values >= observed) + 1) / (len(values) + 1)
    return {
        "observed": observed,
        "random_mean": mean,
        "random_standard_deviation": standard_deviation,
        "random_median": float(median),
        "random_ci95_low": float(q025),
        "random_ci95_high": float(q975),
        "absolute_difference": observed - mean,
        "percent_change_from_random_mean": 100.0 * (observed - mean) / mean,
        "diversity_favorable_percent_change": 100.0 * favorable_change,
        "diversity_favorable_percentile": 100.0 * favorable_percentile,
        "outside_random_ci95": bool(observed < q025 or observed > q975),
        "two_sided_empirical_p": min(1.0, 2.0 * min(lower_tail, upper_tail)),
        "diversity_direction": direction,
    }


def write_replicates(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_table(path: Path, comparison: dict[str, dict[str, float | bool]]) -> None:
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "metric",
                "observed_hits",
                "random_seed_mean",
                "random_ci95_low",
                "random_ci95_high",
                "percent_change",
                "diversity_favorable_percentile",
                "outside_random_ci95",
                "empirical_p_two_sided",
            ]
        )
        for metric, values in comparison.items():
            writer.writerow(
                [
                    metric,
                    values["observed"],
                    values["random_mean"],
                    values["random_ci95_low"],
                    values["random_ci95_high"],
                    values["percent_change_from_random_mean"],
                    values["diversity_favorable_percentile"],
                    values["outside_random_ci95"],
                    values["two_sided_empirical_p"],
                ]
            )


def write_scaffold_categories(
    path: Path,
    typed_values: dict[int, str],
    generic_values: dict[int, str],
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family_type", "code", "scaffold"])
        for family_type, values in (
            ("typed_murcko", typed_values),
            ("generic_murcko", generic_values),
        ):
            for code, scaffold in values.items():
                writer.writerow([family_type, code, scaffold])


def calculate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "random_seed_comparison.json"
    outputs = [
        result_path,
        output_dir / "random_seed_replicates.csv",
        output_dir / "random_seed_comparison_table.csv",
        output_dir / "seed_reference_cache.npz",
        output_dir / "seed_scaffold_categories.csv.gz",
    ]
    if any(path.exists() for path in outputs) and not args.force:
        raise FileExistsError("comparison outputs already exist; use --force to replace")
    if args.force:
        for path in outputs:
            path.unlink(missing_ok=True)

    started = time.monotonic()
    connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    max_id = int(connection.execute("SELECT MAX(spacehastenid) FROM data").fetchone()[0])
    database_rows, max_id = connection.execute(
        "SELECT COUNT(*), MAX(spacehastenid) FROM data"
    ).fetchone()
    max_id = int(max_id)
    seed_count, unique_seed_hashes = connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT reghash) FROM data "
        "WHERE dock_iteration = 0 AND dock_score IS NOT NULL"
    ).fetchone()
    if seed_count != unique_seed_hashes:
        raise ValueError(
            f"seed cohort has {seed_count} rows but {unique_seed_hashes} unique reghashes"
        )
    observed_hit_ids = hit_ids(connection, args.cutoff)
    observed_sample_size = len(observed_hit_ids)
    if not 2 <= observed_sample_size <= seed_count:
        raise ValueError(
            "matched comparison requires between 2 and the seed-population size "
            f"hits; found {observed_sample_size} hits and {seed_count} seeds"
        )

    (
        database_seed_ids,
        typed_by_id,
        generic_by_id,
        typed_values,
        generic_values,
        scaffold_failures,
    ) = load_seed_scaffolds(connection, max_id, args.processes)
    atlas_by_id, atlas_cluster_ids = load_atlas(args.atlas_clustering, max_id)
    connection.close()

    engine = FPSim2Engine(str(args.seed_index.resolve()))
    if engine.fp_type != "Morgan" or engine.fp_params != {"radius": 2, "fpSize": 1024}:
        raise ValueError("seed fingerprint index is not Morgan-2 1024-bit")
    fps = engine.fps
    seed_ids = np.asarray(fps[:, 0], dtype=np.int64)
    if len(seed_ids) != seed_count or len(np.unique(seed_ids)) != seed_count:
        raise ValueError("seed fingerprint index does not match unique seed count")
    if not np.array_equal(np.sort(seed_ids), database_seed_ids):
        raise ValueError("seed fingerprint index identifiers do not match the database")
    typed_codes = typed_by_id[seed_ids]
    generic_codes = generic_by_id[seed_ids]
    atlas_codes = atlas_by_id[seed_ids]
    if np.any(typed_codes < 0) or np.any(generic_codes < 0) or np.any(atlas_codes < 0):
        raise ValueError("seed scaffold or atlas assignments are incomplete")
    hit_atlas_codes = atlas_by_id[observed_hit_ids]
    if np.any(hit_atlas_codes < 0):
        raise ValueError("hit atlas assignments are incomplete")

    with args.hit_metrics.open("rt", encoding="utf-8") as handle:
        hit_result = json.load(handle)
    definition = hit_result.get("definition", {})
    diagnostics = hit_result.get("diagnostics", {})
    hit_metrics = hit_result.get("metrics", {})
    if not math.isclose(
        float(definition.get("dock_score_cutoff", math.nan)),
        args.cutoff,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("hit-metrics cutoff does not match --cutoff")
    if diagnostics.get("limit") is not None:
        raise ValueError("hit metrics were generated from a limited cohort")
    if (
        int(diagnostics.get("database_rows", -1)) != int(database_rows)
        or int(diagnostics.get("max_spacehastenid", -1)) != max_id
    ):
        raise ValueError("hit metrics do not match the analysis database")
    if int(hit_metrics.get("total_virtual_hits", -1)) != observed_sample_size:
        raise ValueError("hit-metrics cohort does not match unique database hits")
    observed: dict[str, float] = {
        "internal_diversity": float(hit_metrics["internal_diversity"]["internal_diversity"]),
        "typed_scaffold_count": float(hit_metrics["typed_bemis_murcko"]["unique_families"]),
        "typed_scaffold_hit_ratio": float(hit_metrics["typed_bemis_murcko"]["family_hit_ratio"]),
        "largest_typed_scaffold_fraction": float(
            hit_metrics["typed_bemis_murcko"]["largest_family_fraction"]
        ),
        "generic_framework_count": float(hit_metrics["generic_murcko"]["unique_families"]),
        "generic_framework_hit_ratio": float(hit_metrics["generic_murcko"]["family_hit_ratio"]),
        "largest_generic_framework_fraction": float(
            hit_metrics["generic_murcko"]["largest_family_fraction"]
        ),
        **{
            key: float(value)
            for key, value in atlas_metrics(hit_atlas_codes, len(observed_hit_ids)).items()
        },
    }

    words = np.ascontiguousarray(fps[:, 1:-1], dtype=np.uint64)
    popcounts = np.asarray(fps[:, -1], dtype=np.int32)
    replicate_rows: list[dict[str, float | int]] = []
    base_sequence = np.random.SeedSequence(args.random_seed)
    child_sequences = base_sequence.spawn(args.replicates)
    for replicate, sequence in enumerate(
        tqdm(child_sequences, desc="Matched random-seed replicates", unit="rep"),
        start=1,
    ):
        rng = np.random.default_rng(sequence)
        sample = rng.choice(seed_count, size=observed_sample_size, replace=False)
        typed = count_metrics(typed_codes[sample], observed_sample_size, "typed_scaffold")
        generic = count_metrics(generic_codes[sample], observed_sample_size, "generic_framework")
        atlas = atlas_metrics(atlas_codes[sample], observed_sample_size)
        diversity = sample_pairwise_diversity(
            sample,
            words,
            popcounts,
            rng,
            args.pair_samples,
            args.pair_batch_size,
        )
        replicate_rows.append(
            {
                "replicate": replicate,
                "random_seed": int(sequence.generate_state(1)[0]),
                "internal_diversity": diversity,
                **typed,
                **generic,
                **atlas,
            }
        )

    comparison: dict[str, dict[str, float | bool]] = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        values = np.asarray([float(row[metric]) for row in replicate_rows])
        comparison[metric] = summarize_distribution(values, observed[metric], direction)

    write_replicates(output_dir / "random_seed_replicates.csv", replicate_rows)
    write_comparison_table(output_dir / "random_seed_comparison_table.csv", comparison)
    write_scaffold_categories(
        output_dir / "seed_scaffold_categories.csv.gz", typed_values, generic_values
    )
    with (output_dir / "seed_reference_cache.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            seed_spacehastenid=seed_ids,
            typed_scaffold_code=typed_codes,
            generic_framework_code=generic_codes,
            atlas_cluster_code=atlas_codes,
            atlas_clusterid=atlas_cluster_ids,
        )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "source_database": args.source_label or str(args.database.resolve()),
        "analysis_database": str(args.database.resolve()),
        "inputs": {
            "seed_index": str(args.seed_index.resolve()),
            "atlas_clustering": str(args.atlas_clustering.resolve()),
            "hit_metrics": str(args.hit_metrics.resolve()),
        },
        "design": {
            "seed_population": int(seed_count),
            "matched_sample_size": int(observed_sample_size),
            "random_replicates": args.replicates,
            "pair_samples_per_replicate": args.pair_samples,
            "base_random_seed": args.random_seed,
            "sampling": "unique seeds sampled without replacement",
            "atlas": (
                f"common seed-first sphere-exclusion atlas at Tanimoto >= {args.atlas_similarity:g}"
            ),
        },
        "diagnostics": {
            "unique_seed_reghashes": int(unique_seed_hashes),
            "seed_scaffold_parse_failures": scaffold_failures,
            "atlas_cluster_count": int(len(atlas_cluster_ids)),
            "observed_virtual_hits": int(len(observed_hit_ids)),
        },
        "observed": observed,
        "comparison": comparison,
        "elapsed_seconds": time.monotonic() - started,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "Matched random-seed comparison complete in %.1f minutes",
        result["elapsed_seconds"] / 60.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source-label")
    parser.add_argument("--seed-index", type=Path, required=True)
    parser.add_argument("--atlas-clustering", type=Path, required=True)
    parser.add_argument("--hit-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, required=True)
    parser.add_argument("--atlas-similarity", type=float, default=0.4)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--pair-samples", type=int, default=1_000_000)
    parser.add_argument("--pair-batch-size", type=int, default=250_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.replicates < 2:
        parser.error("replicates must be at least 2")
    if args.pair_samples < 2:
        parser.error("pair samples must be at least 2")
    if min(args.pair_batch_size, args.processes) < 1:
        parser.error("pair batch size and processes must be positive")
    if not 0 < args.atlas_similarity <= 1:
        parser.error("atlas similarity must be in (0, 1]")
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        calculate(parse_args())
    except Exception:
        LOGGER.exception("Matched random-seed comparison failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
