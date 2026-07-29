#!/usr/bin/env python3
"""Build reusable seed scaffold/framework/atlas reference caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from compare_hits_to_random_seeds import (
    load_atlas,
    load_seed_scaffolds,
    write_scaffold_categories,
)
from FPSim2 import FPSim2Engine


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--seed-index", type=Path, required=True)
    parser.add_argument("--atlas-clustering", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.processes < 1:
        parser.error("processes must be positive")

    root = args.output_root.resolve()
    reference_path = root / "seed_reference_cache.npz"
    family_path = root / "seed_scaffold_categories.csv.gz"
    receipt_path = root / "_SUCCESS.json"
    existing = [path for path in (reference_path, family_path, receipt_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"seed reference outputs exist; pass --overwrite: {existing}")
    root.mkdir(parents=True, exist_ok=True)

    database = args.database.resolve()
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        seed_count, unique_reghashes, max_id = connection.execute(
            "SELECT COUNT(*),COUNT(DISTINCT reghash),MAX(spacehastenid) "
            "FROM data WHERE dock_iteration=0 AND dock_score IS NOT NULL"
        ).fetchone()
        if quick_check != "ok" or seed_count != unique_reghashes:
            raise ValueError("seed database integrity or reghash uniqueness failed")
        (
            database_seed_ids,
            typed_by_id,
            generic_by_id,
            typed_values,
            generic_values,
            scaffold_failures,
        ) = load_seed_scaffolds(connection, int(max_id), args.processes)

    atlas_by_id, atlas_cluster_ids = load_atlas(args.atlas_clustering.resolve(), int(max_id))
    engine = FPSim2Engine(str(args.seed_index.resolve()))
    if engine.fp_type != "Morgan" or engine.fp_params != {"radius": 2, "fpSize": 1024}:
        raise ValueError("seed index is not Morgan radius-2/1024-bit")
    seed_ids = np.asarray(engine.fps[:, 0], dtype=np.int64)
    if len(seed_ids) != seed_count or not np.array_equal(np.sort(seed_ids), database_seed_ids):
        raise ValueError("seed index identifiers do not match database seeds")
    typed_codes = typed_by_id[seed_ids]
    generic_codes = generic_by_id[seed_ids]
    atlas_codes = atlas_by_id[seed_ids]
    if np.any(typed_codes < 0) or np.any(generic_codes < 0) or np.any(atlas_codes < 0):
        raise ValueError("seed scaffold or atlas assignments are incomplete")

    write_scaffold_categories(family_path, typed_values, generic_values)
    with reference_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            seed_spacehastenid=seed_ids,
            typed_scaffold_code=typed_codes,
            generic_framework_code=generic_codes,
            atlas_cluster_code=atlas_codes,
            atlas_clusterid=atlas_cluster_ids,
        )
    receipt = {
        "status": "complete",
        "seed_count": int(seed_count),
        "unique_seed_reghashes": int(unique_reghashes),
        "scaffold_failures": scaffold_failures,
        "typed_families": len(typed_values),
        "generic_families": len(generic_values),
        "atlas_clusters": len(atlas_cluster_ids),
        "inputs": {
            "database": str(database),
            "seed_index": str(args.seed_index.resolve()),
            "seed_index_sha256": sha256(args.seed_index.resolve()),
            "atlas_clustering": str(args.atlas_clustering.resolve()),
            "atlas_clustering_sha256": sha256(args.atlas_clustering.resolve()),
        },
        "outputs": [
            {
                "path": reference_path.name,
                "bytes": reference_path.stat().st_size,
                "sha256": sha256(reference_path),
            },
            {
                "path": family_path.name,
                "bytes": family_path.stat().st_size,
                "sha256": sha256(family_path),
            },
        ],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
