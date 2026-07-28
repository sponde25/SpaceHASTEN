#!/usr/bin/env python3
"""Build a packed fingerprint cache for occupied persistent-atlas centroids."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from tqdm import tqdm

from spacehasten.analysis.database import ReadOnlyDatabase
from spacehasten.analysis.discovery import discover_run


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> tuple[list[int], str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("selected manifest is empty")
    cluster_column = "clusterid" if "clusterid" in rows[0] else "atlas_cluster"
    if cluster_column not in rows[0]:
        raise ValueError("selected manifest lacks persistent atlas clusters")
    atlas_ids = {row.get("atlas_id", "") for row in rows} - {""}
    atlas_id = next(iter(atlas_ids)) if len(atlas_ids) == 1 else ""
    return sorted({int(row[cluster_column]) for row in rows}), atlas_id


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def build(args: argparse.Namespace) -> None:
    context = discover_run(args.run_or_db, database_path=args.database)
    clusters, manifest_atlas = read_manifest(args.manifest)
    atlas_id = args.atlas_id or manifest_atlas
    if not atlas_id:
        raise ValueError("atlas ID is ambiguous; pass --atlas-id")
    output = args.output.resolve()
    receipt = output.with_suffix(output.suffix + ".json")
    if (output.exists() or receipt.exists()) and not args.overwrite:
        raise FileExistsError("centroid cache exists; pass --overwrite")
    centroids: dict[int, int] = {}
    smiles: dict[int, str] = {}
    with ReadOnlyDatabase(context.database_path) as database:
        for start in range(0, len(clusters), 900):
            batch = clusters[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            query = (
                "SELECT clusterid,centroid_spacehastenid FROM cluster_atlas_centroids "
                f"WHERE atlas_id=? AND clusterid IN ({placeholders})"
            )
            centroids.update(
                {
                    int(cluster): int(identifier)
                    for cluster, identifier in database.connection.execute(
                        query, (atlas_id, *batch)
                    )
                }
            )
        if set(centroids) != set(clusters):
            raise ValueError("one or more occupied clusters lack registered centroids")
        identifiers = sorted(set(centroids.values()))
        for start in range(0, len(identifiers), 900):
            batch = identifiers[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            query = f"SELECT spacehastenid,smiles FROM data WHERE spacehastenid IN ({placeholders})"
            smiles.update(
                {
                    int(identifier): str(value)
                    for identifier, value in database.connection.execute(query, batch)
                }
            )
    if set(smiles) != set(identifiers):
        raise ValueError("one or more atlas centroids lack structures")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    words = np.empty((len(identifiers), 16), dtype=np.uint64)
    popcounts = np.empty(len(identifiers), dtype=np.uint16)
    for index, identifier in enumerate(
        tqdm(identifiers, desc="atlas centroid fingerprints", unit="centroid")
    ):
        molecule = Chem.MolFromSmiles(smiles[identifier])
        if molecule is None:
            raise ValueError(f"unparseable centroid SMILES for {identifier}")
        fingerprint = generator.GetFingerprint(molecule)
        words[index] = np.frombuffer(
            DataStructs.BitVectToBinaryText(fingerprint), dtype=np.dtype("<u8")
        )
        popcounts[index] = fingerprint.GetNumOnBits()
    write_npz(
        output,
        spacehastenid=np.asarray(identifiers, dtype=np.int64),
        words=words,
        popcounts=popcounts,
    )
    metadata = {
        "status": "complete",
        "database": str(context.database_path),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "atlas_id": atlas_id,
        "occupied_clusters": len(clusters),
        "unique_centroids": len(identifiers),
        "output": str(output),
        "output_sha256": sha256(output),
        "fingerprint": {"type": "Morgan", "radius": 2, "n_bits": 1024},
    }
    receipt.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_or_db")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--atlas-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    build(args)


if __name__ == "__main__":
    main()
