from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analysis" / "analyze_paper_diversity.py"


def _inputs(root: Path) -> tuple[Path, Path]:
    rows = [
        (11, "c1ccccc1", 1, -10.0, True),
        (12, "Cc1ccccc1", 1, -10.2, True),
        (13, "c1ccncc1", 2, -10.5, True),
        (14, "C1CCCCC1", 2, -9.8, True),
        (15, "c1ccc(-c2ccccc2)cc1", 2, -11.2, True),
        (16, "CCO", 2, -7.0, False),
    ]
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "spacehastenid",
                "reghash",
                "smiles",
                "round",
                "dock_score",
                "is_hit",
                "is_strict_hit",
            ],
        )
        writer.writeheader()
        for identifier, smiles, round_id, score, hit in rows:
            writer.writerow(
                {
                    "spacehastenid": identifier,
                    "reghash": f"hash-{identifier}",
                    "smiles": smiles,
                    "round": round_id,
                    "dock_score": score,
                    "is_hit": hit,
                    "is_strict_hit": score <= -11.0,
                }
            )

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    words = np.empty((len(rows), 16), dtype=np.uint64)
    popcounts = np.empty(len(rows), dtype=np.uint16)
    for index, (_, smiles, *_rest) in enumerate(rows):
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
        spacehastenid=np.asarray([row[0] for row in rows], dtype=np.int64),
        words=words,
        popcounts=popcounts,
    )
    return manifest, fingerprints


def _coordinates(path: Path, identifiers: list[int]) -> None:
    values = np.column_stack(
        (
            np.linspace(-1, 1, len(identifiers)),
            np.linspace(1, -1, len(identifiers)),
        )
    ).astype(np.float32)
    np.savez_compressed(path, spacehastenid=np.asarray(identifiers), umap=values)


def test_paper_aligned_diversity_and_fixed_umap(tmp_path: Path) -> None:
    manifest, fingerprints = _inputs(tmp_path)
    output = tmp_path / "paper"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "analyze",
            "--manifest",
            str(manifest),
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
        ],
        check=True,
    )
    receipt = json.loads((output / "_SUCCESS.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["virtual_hits"] == 5
    assert receipt["level1_assigned_hits"] == 5
    assert receipt["cluster_similarity"] == 0.55

    with gzip.open(output / "paper_aligned_assignments.csv.gz", "rt") as handle:
        assignments = list(csv.DictReader(handle))
    biphenyl = next(row for row in assignments if row["spacehastenid"] == "15")
    assert biphenyl["scaffoldtree_level1"] == "c1ccccc1"
    assert biphenyl["scaffoldtree_status"] == "assigned"
    with np.load(output / "t055_centroid_fingerprints.npz") as data:
        centroid_ids = data["spacehastenid"].astype(int).tolist()
        assert data["words"].shape == (receipt["sphere_exclusion_clusters"], 16)

    selected_coordinates = tmp_path / "selected_coordinates.npz"
    seed_coordinates = tmp_path / "seed_coordinates.npz"
    centroid_coordinates = tmp_path / "centroid_coordinates.npz"
    _coordinates(selected_coordinates, [11, 12, 13, 14, 15, 16])
    _coordinates(seed_coordinates, [1, 2, 3, 4])
    _coordinates(centroid_coordinates, centroid_ids)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "plot-umap",
            "--output-root",
            str(output),
            "--selected-coordinates",
            str(selected_coordinates),
            "--seed-coordinates",
            str(seed_coordinates),
            "--centroid-coordinates",
            str(centroid_coordinates),
            "--dpi",
            "50",
        ],
        check=True,
    )
    assert (output / "figures" / "01_paper_aligned_rank_size.png").stat().st_size > 0
    assert (output / "figures" / "02_paper_aligned_effective_diversity.png").stat().st_size > 0
    assert (output / "figures" / "03_t055_centroids_fixed_umap.png").stat().st_size > 0
