#!/usr/bin/env python3
"""Translate seed reference labels and coordinates across run-local ID namespaces."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from FPSim2 import FPSim2Engine
from tqdm import tqdm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def infer_atlas_id(connection: sqlite3.Connection, requested: str | None) -> str:
    if requested:
        return requested
    values = {
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT atlas_id FROM acquisition_batches WHERE strategy='portfolio'"
        )
    }
    if len(values) != 1:
        raise ValueError(f"target portfolio atlas is ambiguous: {sorted(values)}")
    return next(iter(values))


def seed_counts(connection: sqlite3.Connection, schema: str = "main") -> tuple[int, int]:
    count, unique = connection.execute(
        f"SELECT COUNT(*),COUNT(DISTINCT reghash) FROM {schema}.data "
        "WHERE dock_iteration=0 AND dock_score IS NOT NULL"
    ).fetchone()
    return int(count), int(unique)


def build_identity_map(
    source_database: Path,
    target_database: Path,
    atlas_id: str | None,
    output: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    connection = read_only(target_database)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        source_uri = f"file:{source_database.resolve()}?mode=ro"
        connection.execute("ATTACH DATABASE ? AS source", (source_uri,))
        target_count, target_unique = seed_counts(connection)
        source_count, source_unique = seed_counts(connection, "source")
        if target_count != target_unique or source_count != source_unique:
            raise ValueError("source or target seed reghashes are not unique")
        if target_count != source_count:
            raise ValueError(
                f"source and target seed counts differ: {source_count} != {target_count}"
            )
        resolved_atlas = infer_atlas_id(connection, atlas_id)
        query = (
            "SELECT t.spacehastenid,s.spacehastenid,t.reghash,t.smiles,s.smiles,a.clusterid "
            "FROM data t JOIN source.data s ON s.reghash=t.reghash "
            "JOIN cluster_atlas_assignments a ON a.spacehastenid=t.spacehastenid "
            "AND a.atlas_id=? WHERE t.dock_iteration=0 AND t.dock_score IS NOT NULL "
            "AND s.dock_iteration=0 AND s.dock_score IS NOT NULL ORDER BY t.spacehastenid"
        )
        target_ids = np.empty(target_count, dtype=np.int64)
        source_ids = np.empty(target_count, dtype=np.int64)
        atlas_clusters = np.empty(target_count, dtype=np.int64)
        offset = 0
        mismatch_count = 0
        with (
            temporary.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
            io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text,
            tqdm(total=target_count, desc="translate seed identities", unit="seed") as progress,
        ):
            writer = csv.writer(text)
            writer.writerow(["target_spacehastenid", "source_spacehastenid", "reghash"])
            cursor = connection.execute(query, (resolved_atlas,))
            while rows := cursor.fetchmany(100_000):
                stop = offset + len(rows)
                if stop > target_count:
                    raise ValueError("seed identity join produced too many rows")
                target_ids[offset:stop] = [int(row[0]) for row in rows]
                source_ids[offset:stop] = [int(row[1]) for row in rows]
                atlas_clusters[offset:stop] = [int(row[5]) for row in rows]
                mismatch_count += sum(str(row[3]) != str(row[4]) for row in rows)
                writer.writerows((int(row[0]), int(row[1]), str(row[2])) for row in rows)
                offset = stop
                progress.update(len(rows))
        if offset != target_count:
            raise ValueError(f"seed identity join produced {offset} rows; expected {target_count}")
        if mismatch_count:
            raise ValueError(f"{mismatch_count} reghash-matched seeds have different SMILES")
        if len(np.unique(target_ids)) != target_count or len(np.unique(source_ids)) != source_count:
            raise ValueError("seed identity translation is not one-to-one")
        temporary.replace(output)
        return target_ids, source_ids, atlas_clusters, resolved_atlas
    finally:
        connection.close()
        temporary.unlink(missing_ok=True)


def translate_reference_arrays(
    source_reference: Path,
    target_index: Path,
    target_ids: np.ndarray,
    source_ids: np.ndarray,
    atlas_clusters: np.ndarray,
) -> dict[str, np.ndarray]:
    with np.load(source_reference, allow_pickle=False) as data:
        reference_ids = data["seed_spacehastenid"].astype(np.int64, copy=True)
        typed = data["typed_scaffold_code"].astype(np.int32, copy=True)
        generic = data["generic_framework_code"].astype(np.int32, copy=True)
    if len({len(reference_ids), len(typed), len(generic)}) != 1:
        raise ValueError("source seed reference arrays differ in length")
    order = np.argsort(reference_ids)
    sorted_ids = reference_ids[order]
    positions = np.searchsorted(sorted_ids, source_ids)
    if np.any(positions == len(sorted_ids)) or not np.array_equal(
        sorted_ids[positions], source_ids
    ):
        raise ValueError("source seed reference does not cover translated source IDs")

    engine = FPSim2Engine(str(target_index.resolve()))
    if engine.fp_type != "Morgan" or engine.fp_params != {"radius": 2, "fpSize": 1024}:
        raise ValueError("target seed index does not use Morgan-2 1024-bit fingerprints")
    index_ids = np.asarray(engine.fps[:, 0], dtype=np.int64)
    if not np.array_equal(np.sort(index_ids), target_ids):
        raise ValueError("target seed index IDs differ from translated target IDs")

    atlas_labels, atlas_codes = np.unique(atlas_clusters, return_inverse=True)
    return {
        "seed_spacehastenid": target_ids,
        "reference_spacehastenid": source_ids,
        "typed_scaffold_code": typed[order][positions],
        "generic_framework_code": generic[order][positions],
        "atlas_cluster_code": atlas_codes.astype(np.int32),
        "atlas_clusterid": atlas_labels.astype(np.int64),
    }


def translate_coordinates(
    source_coordinates: Path,
    target_ids: np.ndarray,
    source_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    with np.load(source_coordinates, allow_pickle=False) as data:
        coordinate_ids = data["spacehastenid"].astype(np.int64, copy=True)
        coordinates = data["umap"].astype(np.float32, copy=True)
    if coordinates.shape != (len(coordinate_ids), 2) or not np.isfinite(coordinates).all():
        raise ValueError("source seed coordinates are invalid")
    order = np.argsort(coordinate_ids)
    sorted_ids = coordinate_ids[order]
    positions = np.searchsorted(sorted_ids, source_ids)
    if np.any(positions == len(sorted_ids)) or not np.array_equal(
        sorted_ids[positions], source_ids
    ):
        raise ValueError("source coordinates do not cover translated source seed IDs")
    return {"spacehastenid": target_ids, "umap": coordinates[order][positions]}


def copy_family_categories(source: Path, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with (
        gzip.open(source, "rt", encoding="utf-8", newline="") as input_handle,
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="") as output_handle,
    ):
        shutil.copyfileobj(input_handle, output_handle)
    temporary.replace(output)


def translate(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output root is non-empty; pass --overwrite: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / "seed_identity_map.csv.gz"
    target_ids, source_ids, atlas_clusters, atlas_id = build_identity_map(
        args.source_database,
        args.target_database,
        args.atlas_id,
        identity_path,
    )
    reference_arrays = translate_reference_arrays(
        args.source_reference,
        args.target_seed_index,
        target_ids,
        source_ids,
        atlas_clusters,
    )
    coordinate_arrays = translate_coordinates(args.source_coordinates, target_ids, source_ids)
    reference_path = root / "seed_reference_cache.npz"
    coordinate_path = root / "seed_coordinates.npz"
    family_path = root / "seed_scaffold_categories.csv.gz"
    write_npz(reference_path, **reference_arrays)
    write_npz(coordinate_path, **coordinate_arrays)
    copy_family_categories(args.source_families, family_path)
    outputs = [identity_path, reference_path, coordinate_path, family_path]
    inputs = [
        args.source_reference,
        args.source_families,
        args.source_coordinates,
        args.target_seed_index,
    ]
    receipt = {
        "status": "complete",
        "source_database": {
            "path": str(args.source_database.resolve()),
            "bytes": args.source_database.stat().st_size,
        },
        "target_database": {
            "path": str(args.target_database.resolve()),
            "bytes": args.target_database.stat().st_size,
        },
        "atlas_id": atlas_id,
        "seed_count": len(target_ids),
        "unique_target_ids": int(len(np.unique(target_ids))),
        "unique_source_ids": int(len(np.unique(source_ids))),
        "same_numeric_id_mapping": int(np.sum(target_ids == source_ids)),
        "fingerprint": {"type": "Morgan", "radius": 2, "n_bits": 1024},
        "inputs": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in inputs
        ],
        "outputs": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in outputs
        ],
    }
    write_json(root / "_SUCCESS.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--target-database", type=Path, required=True)
    parser.add_argument("--source-reference", type=Path, required=True)
    parser.add_argument("--source-families", type=Path, required=True)
    parser.add_argument("--source-coordinates", type=Path, required=True)
    parser.add_argument("--target-seed-index", type=Path, required=True)
    parser.add_argument("--atlas-id")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in (
        args.source_database,
        args.target_database,
        args.source_reference,
        args.source_families,
        args.source_coordinates,
        args.target_seed_index,
    ):
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
    translate(args)


if __name__ == "__main__":
    main()
