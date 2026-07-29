from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from spacehasten.remote.cluster import _build_fpsim2_index

ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "scripts/analysis/prepare_paper_comparison.py"
ANALYZE = ROOT / "scripts/analysis/analyze_paper_diversity.py"
COMPARE = ROOT / "scripts/analysis/compare_paper_diversity.py"


def run(script: Path, *arguments: str) -> None:
    subprocess.run([sys.executable, str(script), *arguments], check=True)


def write_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    structures = [
        (1, "c1ccccc1"),
        (2, "Cc1ccccc1"),
        (3, "c1ccncc1"),
        (4, "C1CCCCC1"),
        (5, "c1ccc(-c2ccccc2)cc1"),
        (6, "c1ccoc1"),
    ]
    union = pd.DataFrame(
        {
            "common_id": [row[0] for row in structures],
            "reghash": [f"hash-{row[0]}" for row in structures],
            "smiles": [row[1] for row in structures],
        }
    )
    union_path = root / "union.csv.gz"
    union.to_csv(union_path, index=False)

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
    fingerprints = root / "fingerprints.npz"
    np.savez_compressed(
        fingerprints,
        spacehastenid=union["common_id"].to_numpy(np.int64),
        words=words,
        popcounts=popcounts,
    )
    index_path = root / "fingerprints.h5"
    _build_fpsim2_index([(smiles, identifier) for identifier, smiles in structures], index_path)

    event_paths = []
    for workflow, hit_count in (("left", 4), ("right", 5)):
        selected = union.iloc[: 5 if workflow == "left" else 6].copy()
        selected["workflow"] = workflow
        selected["workflow_label"] = workflow.title()
        selected["local_id"] = selected["common_id"] + (100 if workflow == "left" else 200)
        selected["dock_score"] = (
            [-10.5, -10.2, -9.9, -9.8, -8.0]
            if workflow == "left"
            else [-10.5, -10.2, -9.9, -9.8, -10.1, np.nan]
        )
        selected["round"] = 1
        selected["rank"] = np.arange(1, len(selected) + 1)
        selected["block_50k"] = 1
        selected["cumulative_budget"] = selected["rank"]
        selected["rank_source"] = "test"
        selected["scored"] = selected["dock_score"].notna()
        selected["hit"] = selected["dock_score"].le(-9.7)
        selected["strict_hit"] = selected["dock_score"].le(-11.0)
        selected["nearest_seed_tanimoto"] = 0.5
        assert int(selected["hit"].sum()) == hit_count
        path = root / f"{workflow}_events.csv.gz"
        selected.drop(columns=["common_id"]).to_csv(path, index=False)
        event_paths.append(path)
    return union_path, fingerprints, index_path, event_paths[0], event_paths[1]


@pytest.mark.parametrize("swap", [False, True])
def test_paper_comparison_end_to_end(tmp_path: Path, swap: bool) -> None:
    union, fingerprints, index_path, left_events, right_events = write_inputs(tmp_path)
    if swap:
        left_events, right_events = right_events, left_events
    prepared = tmp_path / "prepared"
    run(
        PREPARE,
        "--workflow",
        "left",
        str(left_events),
        "--workflow",
        "right",
        str(right_events),
        "--union",
        str(union),
        "--fingerprints",
        str(fingerprints),
        "--output-root",
        str(prepared),
    )
    prepared_receipt = json.loads((prepared / "_SUCCESS.json").read_text())
    assert prepared_receipt["workflows"]["left"]["hits"] == (5 if swap else 4)
    assert prepared_receipt["workflows"]["right"]["hits"] == (4 if swap else 5)

    paper_roots = {}
    for workflow in ("left", "right"):
        output = tmp_path / f"paper-{workflow}"
        run(
            ANALYZE,
            "analyze",
            "--manifest",
            str(prepared / f"{workflow}_manifest.csv.gz"),
            "--fingerprints",
            str(fingerprints),
            "--output-root",
            str(output),
            "--hit-threshold",
            "-9.7",
            "--processes",
            "1",
            "--dpi",
            "50",
        )
        paper_roots[workflow] = output

    comparison = tmp_path / "comparison"
    run(
        COMPARE,
        "analyze",
        "--left-root",
        str(paper_roots["left"]),
        "--right-root",
        str(paper_roots["right"]),
        "--left-name",
        "left",
        "--right-name",
        "right",
        "--left-label",
        "Left",
        "--right-label",
        "Right",
        "--fingerprints",
        str(fingerprints),
        "--fp-index",
        str(index_path),
        "--output-root",
        str(comparison),
        "--replicates",
        "3",
        "--processes",
        "1",
    )
    receipt = json.loads((comparison / "_SUCCESS.json").read_text())
    assert receipt["replicates"] == 3
    assert receipt["fixed_workflow"] == ("right" if swap else "left")
    assert receipt["sampled_workflow"] == ("left" if swap else "right")
    assert receipt["fixed_complete_hits"] == receipt["sample_size"] == 4
    replicates = pd.read_csv(comparison / "paper_count_matched_replicates.csv")
    assert set(replicates["replicate"]) == {0, 1, 2}
    assert set(replicates["sample_size"]) == {4}
    assert replicates["t055_q0"].ge(1).all()
    assert replicates["t055_assigned_hits"].eq(4).all()
    assert replicates["t055_minimum_similarity"].ge(0.55).all()
    with np.load(comparison / "paper_count_matched_sample_ids.npz") as data:
        assert data["common_id"].shape == (3, 4)
    with np.load(comparison / "paper_count_matched_cluster_assignments.npz") as data:
        assert data["clusterid"].shape == (3, 4)
