from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from spacehasten.analysis.comparison import (
    EVENT_FIELDS,
    analyze_comparison,
    prepare_comparison,
    semantic_digest,
    sha256,
)

ROOT = Path(__file__).resolve().parents[2]
VALIDATE = ROOT / "scripts/analysis/validate_workflow_comparison.py"


def fingerprint_cache(path: Path, structures: list[tuple[int, str]]) -> None:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    words = np.empty((len(structures), 16), dtype=np.uint64)
    popcounts = np.empty(len(structures), dtype=np.uint16)
    for index, (_, smiles) in enumerate(structures):
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None
        fingerprint = generator.GetFingerprint(molecule)
        words[index] = np.frombuffer(
            DataStructs.BitVectToBinaryText(fingerprint), dtype=np.dtype("<u8")
        )
        popcounts[index] = fingerprint.GetNumOnBits()
    np.savez_compressed(
        path,
        spacehastenid=np.asarray([row[0] for row in structures], dtype=np.int64),
        words=words,
        popcounts=popcounts,
    )


def database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE data(
                spacehastenid INTEGER PRIMARY KEY,
                reghash TEXT,
                dock_score REAL,
                dock_iteration INTEGER
            );
            CREATE TABLE docking_param(dock_param BLOB);
            CREATE TABLE docking_grid(dock_grid BLOB);
            CREATE TABLE properties(
                property TEXT,
                is_double INTEGER,
                min_limit REAL,
                max_limit REAL
            );
            """
        )
        connection.executemany(
            "INSERT INTO data VALUES(?,?,?,0)",
            [(1, "seed-a", -8.0), (2, "seed-b", -9.0)],
        )
        connection.execute("INSERT INTO docking_param VALUES(?)", (b"parameters",))
        connection.execute("INSERT INTO docking_grid VALUES(?)", (b"grid",))
        connection.execute("INSERT INTO properties VALUES('mw',1,0,500)")


def analysis_inputs(
    root: Path,
    workflow: str,
    structures: list[tuple[int, str, str, float | None]],
) -> None:
    cache = root / "structure_cache"
    nearest = root / "nearest_seed"
    cache.mkdir(parents=True)
    nearest.mkdir()
    manifest = pd.DataFrame(
        {
            "spacehastenid": [row[0] for row in structures],
            "reghash": [row[1] for row in structures],
            "smiles": [row[2] for row in structures],
            "dock_score": [row[3] for row in structures],
            "round": 1,
            "rank": np.arange(1, len(structures) + 1),
            "rank_source": f"{workflow}_rank",
        }
    )
    manifest.to_csv(cache / "selected_manifest.csv.gz", index=False)
    descriptors = manifest[["spacehastenid", "reghash"]].copy()
    descriptors["typed_scaffold"] = [f"typed-{row[1]}" for row in structures]
    descriptors["generic_framework"] = [f"generic-{row[1]}" for row in structures]
    for name, value in (
        ("MW", 300.0),
        ("cLogP", 2.0),
        ("TPSA", 60.0),
        ("HBD", 1),
        ("HBA", 3),
        ("rotatable", 2),
        ("rings", 1),
        ("Fsp3", 0.2),
    ):
        descriptors[name] = value
    descriptors.to_csv(cache / "structure_cache.csv.gz", index=False)
    fingerprint_cache(cache / "fingerprints.npz", [(row[0], row[2]) for row in structures])
    np.savez_compressed(
        nearest / "nearest_seed_similarity.npz",
        spacehastenid=np.asarray([row[0] for row in structures], dtype=np.int64),
        nearest_seed_id=np.ones(len(structures), dtype=np.int64),
        tanimoto=np.linspace(0.4, 0.6, len(structures), dtype=np.float32),
    )


def test_prepare_workflow_comparison(tmp_path: Path) -> None:
    left_database = tmp_path / "left.dbsh"
    right_database = tmp_path / "right.dbsh"
    database(left_database)
    database(right_database)
    left = tmp_path / "left"
    right = tmp_path / "right"
    analysis_inputs(
        left,
        "left",
        [
            (10, "shared", "c1ccccc1", -13.8),
            (11, "left", "C1CCCCC1", -13.7),
            (12, "shared-2", "c1ccoc1", -12.8),
        ],
    )
    analysis_inputs(
        right,
        "right",
        [
            (20, "shared", "c1ccccc1", -13.9),
            (21, "shared-2", "c1ccoc1", -12.7),
            (22, "right", "c1ccncc1", None),
        ],
    )
    left_validation = tmp_path / "left-validation.json"
    right_validation = tmp_path / "right-validation.json"
    left_validation.write_text('{"status":"ok"}\n', encoding="utf-8")
    right_validation.write_text('{"status":"ok"}\n', encoding="utf-8")
    definition = tmp_path / "atlas-definition.json"
    definition.write_text(
        json.dumps(
            {
                "seed_count": 2,
                "fingerprint_type": "Morgan",
                "fingerprint_parameters": {"radius": 2, "fpSize": 1024},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "comparison"
    result = prepare_comparison(
        left_name="left",
        left_label="Left workflow",
        left_database=left_database,
        left_analysis=left,
        left_validation=left_validation,
        right_name="right",
        right_label="Right workflow",
        right_database=right_database,
        right_analysis=right,
        right_validation=right_validation,
        seed_atlas_definition=definition,
        output_root=output,
        hit_cutoff=-12.5,
        strict_cutoff=-13.5,
    )
    assert result["status"] == "compatible"
    assert result["union_counts"] == {
        "union": 4,
        "shared": 2,
        "left_only": 1,
        "right_only": 1,
        "structure_mismatches": 0,
        "fingerprint_mismatches": 0,
    }
    union = pd.read_csv(output / "cache/common_union_manifest.csv.gz")
    assert union["common_id"].tolist() == [1, 2, 3, 4]
    assert union["atlas_query_id"].tolist() == [3, 4, 5, 6]
    right_events = pd.read_csv(output / "cache/right_events.csv.gz")
    assert int(right_events["scored"].sum()) == 2
    assert semantic_digest(right_events, EVENT_FIELDS) == result["event_semantic_sha256"]["right"]
    assert (output / "cache/common_atlas/molecule_index/molecules_fp.h5").is_file()

    assignments = output / "cache/common_atlas/selected_atlas_assignments.npz"
    np.savez_compressed(
        assignments,
        spacehastenid=np.asarray([3, 4, 5, 6], dtype=np.int64),
        clusterid=np.asarray([1, 3, 1, 6], dtype=np.int64),
        centroid_similarity=np.asarray([0.5, 1.0, 0.6, 1.0], dtype=np.float32),
    )
    (assignments.parent / "_SUCCESS.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "selected_compounds": 4,
                "similarity_threshold": 0.4,
                "seed_centroids": 2,
            }
        ),
        encoding="utf-8",
    )
    coordinates = output / "cache/common_union_umap/landmark_umap_coordinates.npz"
    coordinates.parent.mkdir(parents=True)
    np.savez_compressed(
        coordinates,
        spacehastenid=np.arange(1, 5, dtype=np.int64),
        umap=np.asarray([[0.0, 0.0], [0.2, 0.1], [0.4, 0.2], [0.6, 0.3]], dtype=np.float32),
    )
    seed_coordinates = tmp_path / "seed-coordinates.npz"
    np.savez_compressed(
        seed_coordinates,
        spacehastenid=np.asarray([1, 2], dtype=np.int64),
        umap=np.asarray([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32),
    )
    model = tmp_path / "model.joblib"
    model.write_bytes(b"validated model placeholder")
    timing_paths = {}
    for name in ("left", "right"):
        timing = tmp_path / f"{name}-timing.csv"
        pd.DataFrame(
            {
                "round": [1],
                "end_to_end_wall_seconds": [10.0],
            }
        ).to_csv(timing, index=False)
        timing_paths[name] = timing
    receipt = analyze_comparison(
        root=output,
        common_coordinates=coordinates,
        common_atlas_assignments_path=assignments,
        seed_coordinates=seed_coordinates,
        umap_model=model,
        timing_paths=timing_paths,
        replicates=3,
        pair_samples=100,
        random_seed=42,
        dpi=50,
    )
    assert receipt["status"] == "complete"
    assert receipt["count_matching"]["fixed_workflow"] == "right"
    assert receipt["count_matching"]["sampled_workflow"] == "left"
    replicates = pd.read_csv(output / "tables/count_matched_replicates.csv")
    assert replicates["replicate"].tolist() == [0, 1, 2]
    assert set(replicates["sample_size"]) == {2}
    assert len(list((output / "figures").glob("*.png"))) == 8

    for name, hits in (("left", 3), ("right", 2)):
        paper_root = output / f"paper_diversity/{name}"
        paper_root.mkdir(parents=True)
        assignment = paper_root / "paper_aligned_assignments.csv.gz"
        pd.DataFrame(
            {
                "spacehastenid": np.arange(1, hits + 1),
                "scaffoldtree_status": "assigned",
            }
        ).to_csv(assignment, index=False)
        (paper_root / "_SUCCESS.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "cluster_similarity": 0.55,
                    "virtual_hits": hits,
                    "fpsim2_index_matches_cached_fingerprints": True,
                    "level1_assignment_status": {"assigned": hits},
                    "sphere_exclusion_clusters": hits,
                    "outputs": [
                        {
                            "path": assignment.name,
                            "bytes": assignment.stat().st_size,
                            "sha256": sha256(assignment),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    paper_comparison = output / "paper_diversity/comparison"
    paper_comparison.mkdir()
    paper_replicates = paper_comparison / "paper_count_matched_replicates.csv"
    pd.DataFrame(
        {
            "replicate": [0, 1, 2],
            "sample_size": [2, 2, 2],
            "t055_assigned_hits": [2, 2, 2],
            "t055_minimum_similarity": [0.55, 0.55, 0.55],
        }
    ).to_csv(paper_replicates, index=False)
    paper_samples = paper_comparison / "paper_count_matched_sample_ids.npz"
    paper_assignments = paper_comparison / "paper_count_matched_cluster_assignments.npz"
    np.savez_compressed(
        paper_samples,
        replicate=np.arange(3),
        common_id=np.asarray([[1, 2], [1, 3], [2, 3]], dtype=np.int64),
    )
    np.savez_compressed(
        paper_assignments,
        replicate=np.arange(3),
        clusterid=np.asarray([[0, 1], [0, 1], [0, 1]], dtype=np.int64),
    )
    (paper_comparison / "_SUCCESS.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "cluster_similarity": 0.55,
                "replicates": 3,
                "fixed_workflow": "right",
                "sampled_workflow": "left",
                "outputs": [
                    {
                        "path": paper_replicates.name,
                        "bytes": paper_replicates.stat().st_size,
                        "sha256": sha256(paper_replicates),
                    },
                    {
                        "path": paper_samples.name,
                        "bytes": paper_samples.stat().st_size,
                        "sha256": sha256(paper_samples),
                    },
                    {
                        "path": paper_assignments.name,
                        "bytes": paper_assignments.stat().st_size,
                        "sha256": sha256(paper_assignments),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (output / "COMPARATIVE_REPORT.md").write_text(
        "# Comparison\n\n**Table 1. Test table.**\n\n| A |\n|---|\n| 1 |\n\n"
        "*Table 1 note. Test note.*\n\n"
        "![Figure 1. Test figure](figures/01_budget_yield.png)\n\n"
        "*Figure 1. Test figure caption.*\n",
        encoding="utf-8",
    )
    (output / "COMPARATIVE_REPORT.html").write_text(
        '<img src="data:image/png;base64,cG5n">', encoding="utf-8"
    )
    subprocess.run(
        [
            sys.executable,
            str(VALIDATE),
            "--comparison-root",
            str(output),
            "--expected-replicates",
            "3",
        ],
        check=True,
    )
    validation = json.loads((output / "validation_summary.json").read_text())
    assert validation["status"] == "ok"
    assert validation["checks"]["union"]["rows"] == 4
