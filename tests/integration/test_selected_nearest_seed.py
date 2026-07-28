from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("FPSim2")

from spacehasten.remote.cluster import _build_fpsim2_index

ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "scripts" / "analysis" / "prepare_selected_nearest_seed.py"
WORKER = ROOT / "scripts" / "analysis" / "nearest_seed_similarity_chunk.py"
COMBINE = ROOT / "scripts" / "analysis" / "combine_nearest_seed_chunks.py"


def test_selected_nearest_seed_pipeline(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv.gz"
    pd.DataFrame(
        {
            "spacehastenid": [11, 12, 11],
            "reghash": ["h11", "h12", "h11"],
            "smiles": ["CCO", "CCCC", "CCO"],
        }
    ).to_csv(manifest_path, index=False)
    index_path = tmp_path / "seeds.h5"
    _build_fpsim2_index([("CCO", 1), ("CCN", 2)], index_path)
    output = tmp_path / "nearest"
    subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            "--manifest",
            str(manifest_path),
            "--seed-index",
            str(index_path),
            "--output-root",
            str(output),
            "--task-count",
            "2",
        ],
        check=True,
    )
    preparation = json.loads((output / "preparation.json").read_text())
    assert preparation["selected_compounds"] == 2
    assert preparation["seed_count"] == 2
    for task in (1, 2):
        token = f"{task:04d}_of_0002"
        source = output / "inputs" / f"selected_{token}.smi.gz"
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            assert len(handle.readlines()) == 1
        subprocess.run(
            [
                sys.executable,
                str(WORKER),
                "--input",
                str(source),
                "--seed-index",
                str(index_path),
                "--output",
                str(output / "chunks" / f"nearest_{token}.npz"),
            ],
            check=True,
        )
    combined = output / "nearest_seed_similarity.npz"
    subprocess.run(
        [
            sys.executable,
            str(COMBINE),
            "--chunks-dir",
            str(output / "chunks"),
            "--output",
            str(combined),
            "--task-count",
            "2",
            "--expected-count",
            "2",
        ],
        check=True,
    )
    with np.load(combined) as data:
        assert data["spacehastenid"].tolist() == [11, 12]
        assert data["nearest_seed_id"][0] == 1
        assert data["tanimoto"][0] == pytest.approx(1.0)
