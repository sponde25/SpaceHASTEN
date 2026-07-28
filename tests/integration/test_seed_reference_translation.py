from __future__ import annotations

import csv
import gzip
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("FPSim2")

from spacehasten.remote.cluster import _build_fpsim2_index

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "translate_seed_reference.py"
)


def database(path: Path, rows: list[tuple[int, str, str]], *, target: bool) -> Path:
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE data(spacehastenid INTEGER PRIMARY KEY,reghash TEXT,smiles TEXT,"
            "dock_score REAL,dock_iteration INTEGER)"
        )
        connection.executemany(
            "INSERT INTO data VALUES (?,?,?,0.0,0)",
            rows,
        )
        if target:
            connection.execute("CREATE TABLE acquisition_batches(strategy TEXT,atlas_id TEXT)")
            connection.execute("INSERT INTO acquisition_batches VALUES ('portfolio','atlas')")
            connection.execute(
                "CREATE TABLE cluster_atlas_assignments("
                "atlas_id TEXT,spacehastenid INTEGER,clusterid INTEGER)"
            )
            connection.executemany(
                "INSERT INTO cluster_atlas_assignments VALUES ('atlas',?,?)",
                [(1, 101), (2, 102)],
            )
    return path


def test_translate_seed_reference_across_permuted_ids(tmp_path: Path) -> None:
    source_database = database(
        tmp_path / "source.dbsh",
        [(1, "hash-a", "CCO"), (2, "hash-b", "CCN")],
        target=False,
    )
    target_database = database(
        tmp_path / "target.dbsh",
        [(1, "hash-b", "CCN"), (2, "hash-a", "CCO")],
        target=True,
    )
    source_reference = tmp_path / "source_reference.npz"
    np.savez_compressed(
        source_reference,
        seed_spacehastenid=np.asarray([2, 1]),
        typed_scaffold_code=np.asarray([20, 10], dtype=np.int32),
        generic_framework_code=np.asarray([200, 100], dtype=np.int32),
        atlas_cluster_code=np.asarray([1, 0], dtype=np.int32),
    )
    source_coordinates = tmp_path / "source_coordinates.npz"
    np.savez_compressed(
        source_coordinates,
        spacehastenid=np.asarray([1, 2, 99]),
        umap=np.asarray([[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]], dtype=np.float32),
    )
    source_families = tmp_path / "source_families.csv.gz"
    with gzip.open(source_families, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["family_type", "code", "scaffold"])
        writer.writerow(["typed_murcko", 10, "typed-a"])
        writer.writerow(["typed_murcko", 20, "typed-b"])
        writer.writerow(["generic_murcko", 100, "generic-a"])
        writer.writerow(["generic_murcko", 200, "generic-b"])
    target_index = tmp_path / "target_seeds.h5"
    _build_fpsim2_index([("CCN", 1), ("CCO", 2)], target_index)
    output = tmp_path / "translated"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-database",
            str(source_database),
            "--target-database",
            str(target_database),
            "--source-reference",
            str(source_reference),
            "--source-families",
            str(source_families),
            "--source-coordinates",
            str(source_coordinates),
            "--target-seed-index",
            str(target_index),
            "--output-root",
            str(output),
        ],
        check=True,
    )

    with np.load(output / "seed_reference_cache.npz", allow_pickle=False) as data:
        assert data["seed_spacehastenid"].tolist() == [1, 2]
        assert data["reference_spacehastenid"].tolist() == [2, 1]
        assert data["typed_scaffold_code"].tolist() == [20, 10]
        assert data["generic_framework_code"].tolist() == [200, 100]
        assert data["atlas_clusterid"].tolist() == [101, 102]
    with np.load(output / "seed_coordinates.npz", allow_pickle=False) as data:
        assert data["spacehastenid"].tolist() == [1, 2]
        assert data["umap"].tolist() == [[2.0, 2.0], [1.0, 1.0]]
    receipt = json.loads((output / "_SUCCESS.json").read_text())
    assert receipt["seed_count"] == 2
    assert receipt["same_numeric_id_mapping"] == 0
