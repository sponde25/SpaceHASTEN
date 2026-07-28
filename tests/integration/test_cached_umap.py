from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from FPSim2.io.chem import process_fp
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

pytest.importorskip("sklearn")
from sklearn.decomposition import PCA

from spacehasten.analysis.umap import rdkit_words_to_fpsim2_words, unpack_fingerprints

ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "scripts" / "analysis" / "prepare_cached_umap.py"
WORKER = ROOT / "scripts" / "analysis" / "transform_landmark_umap_chunk.py"
COMBINE = ROOT / "scripts" / "analysis" / "combine_landmark_umap_chunks.py"


def test_rdkit_words_convert_to_fpsim2_order() -> None:
    molecule = Chem.MolFromSmiles("CCOC(=O)N1CCCCC1")
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(
        molecule
    )
    rdkit_words = np.frombuffer(
        DataStructs.BitVectToBinaryText(fingerprint), dtype=np.dtype("<u8")
    ).reshape(1, 16)
    fpsim2_words = np.asarray(process_fp(fingerprint, 7)[1:-1], dtype=np.uint64)
    converted = rdkit_words_to_fpsim2_words(rdkit_words)
    assert np.array_equal(converted[0], fpsim2_words)


def test_cached_umap_pipeline(tmp_path: Path) -> None:
    identifiers = np.asarray([11, 12, 13, 14], dtype=np.int64)
    words = np.zeros((4, 16), dtype=np.uint64)
    words[:, 0] = [3, 5, 15, 9]
    fingerprints = tmp_path / "fingerprints.npz"
    np.savez_compressed(
        fingerprints,
        spacehastenid=identifiers,
        words=words,
        popcounts=np.asarray([2, 2, 4, 2], dtype=np.uint16),
    )
    model = tmp_path / "model.joblib"
    reducer = PCA(n_components=2).fit(unpack_fingerprints(rdkit_words_to_fpsim2_words(words)))
    joblib.dump({"reducer": reducer}, model)
    output = tmp_path / "umap"
    subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            "--fingerprints",
            str(fingerprints),
            "--model",
            str(model),
            "--output-root",
            str(output),
            "--task-count",
            "2",
            "--batch-size",
            "2",
        ],
        check=True,
    )
    preparation = json.loads((output / "preparation.json").read_text())
    assert preparation["compound_count"] == 4
    assert "export PYTHONPATH=" in (output / "submit.sh").read_text()
    for task in (1, 2):
        subprocess.run(
            [
                sys.executable,
                str(WORKER),
                "--fingerprints",
                str(fingerprints),
                "--model",
                str(model),
                "--chunks-dir",
                str(output / "chunks"),
                "--task-index",
                str(task),
                "--task-count",
                "2",
                "--batch-size",
                "2",
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(COMBINE),
            "--chunks-dir",
            str(output / "chunks"),
            "--model",
            str(model),
            "--output-dir",
            str(output),
            "--task-count",
            "2",
            "--expected-count",
            "4",
            "--skip-landmark-overwrite",
        ],
        check=True,
    )
    with np.load(output / "landmark_umap_coordinates.npz") as data:
        assert data["spacehastenid"].tolist() == [11, 12, 13, 14]
        assert data["umap"].shape == (4, 2)
        assert np.isfinite(data["umap"]).all()
