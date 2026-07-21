"""End-to-end test of resumable map/reduce atlas components."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("FPSim2")
pytest.importorskip("rdkit")

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from spacehasten.remote.atlas import (
    apply_repair,
    assign_centroid_shard,
    combine_centroids,
    map_partition,
    merge_assignment_shards,
    partition_smiles,
    pool_centroids,
    read_smi,
    reduce_centroid_group,
    reduce_centroids,
    repair_uncovered,
)
from spacehasten.remote.cluster import _build_fpsim2_index


def _molecules() -> list[tuple[str, int]]:
    smiles = [
        "CCO",
        "CCCO",
        "CCCCO",
        "CCN",
        "CCCN",
        "CCCCN",
        "c1ccccc1",
        "Cc1ccccc1",
        "CCc1ccccc1",
        "c1ccncc1",
        "Cc1ccncc1",
        "C1CCCCC1",
        "CC1CCCCC1",
        "C1CCNCC1",
        "CC1CCNCC1",
        "CC(=O)O",
        "CCC(=O)O",
        "CCOC(=O)C",
        "CCS",
        "CCCS",
    ]
    return [(smile, index) for index, smile in enumerate(smiles, start=1)]


def _write_input(path: Path, rows: list[tuple[str, int]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for smiles, identifier in rows:
            handle.write(f"{smiles} {identifier}\n")


def test_resumable_map_reduce_assignment_and_repair(tmp_path: Path) -> None:
    rows = _molecules()
    input_path = tmp_path / "molecules.smi.gz"
    _write_input(input_path, rows)

    partitions = tmp_path / "partitions"
    first_partition = partition_smiles(input_path, partitions, 4)
    partition_mtime = (partitions / "complete.json").stat().st_mtime_ns
    assert partition_smiles(input_path, partitions, 4) == first_partition
    assert (partitions / "complete.json").stat().st_mtime_ns == partition_mtime

    mapper_root = tmp_path / "mappers"
    for index in range(4):
        part = partitions / f"part_{index:04d}_of_0004.smi.gz"
        map_partition(part, mapper_root / f"map_{index:04d}", 0.4, 1)

    intermediate_root = tmp_path / "intermediate"
    for index in range(2):
        reduce_centroid_group(
            mapper_root,
            intermediate_root / f"reducer_{index:04d}",
            index,
            2,
            0.4,
            1,
        )
    pool = tmp_path / "pool"
    pool_centroids(intermediate_root, pool)
    reduced = tmp_path / "reduced"
    reduce_centroids(pool / "pooled_centroids.smi.gz", reduced, 0.4, 1)

    molecule_index = tmp_path / "molecules.h5"
    _build_fpsim2_index(rows, molecule_index)
    shard_root = tmp_path / "assignment_shards"
    for index in range(2):
        assign_centroid_shard(
            molecule_index,
            reduced / "centroids.smi.gz",
            shard_root / f"shard_{index:04d}",
            index,
            2,
            0.4,
        )

    merged = tmp_path / "merged"
    merge_assignment_shards(molecule_index, input_path, shard_root, merged, 0.4)
    repair = tmp_path / "repair"
    repair_uncovered(merged / "uncovered.smi.gz", repair, 0.4, 1)
    final = tmp_path / "final"
    apply_repair(
        merged / "assignments.npz",
        repair / "repair_assignments.npz",
        final,
        0.4,
    )
    combined = tmp_path / "combined_centroids"
    combine_centroids(
        reduced / "centroids.smi.gz",
        repair / "repair_centroids.smi.gz",
        combined,
    )
    assert (combined / "centroids_fp.h5").is_file()

    with np.load(final / "assignments.npz") as assignments:
        assert assignments["spacehastenid"].tolist() == list(range(1, 21))
        assert np.all(assignments["clusterid"] >= 0)
        assert np.all(assignments["centroid_similarity"] >= 0.4)

    centroids = read_smi(reduced / "centroids.smi.gz") + read_smi(
        repair / "repair_centroids.smi.gz"
    )
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(smiles)) for smiles, _ in centroids]
    for index, fingerprint in enumerate(fingerprints):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints[:index])
        assert all(similarity < 0.4 for similarity in similarities)
