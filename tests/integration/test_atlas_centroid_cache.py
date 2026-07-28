from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from spacehasten.core.db import ClusterAtlasCentroidRow, Database

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "prepare_atlas_centroid_cache.py"
)


def test_prepare_atlas_centroid_cache(tmp_path: Path) -> None:
    database = tmp_path / "run.dbsh"
    with Database(database) as source:
        source.create_schema()
        first = source.insert_seed_undocked("h1", "CCO", "one")
        second = source.insert_seed_undocked("h2", "CCN", "two")
        source.append_cluster_atlas_centroids(
            [
                ClusterAtlasCentroidRow("atlas", 101, first, 1),
                ClusterAtlasCentroidRow("atlas", 102, second, 1),
            ]
        )
    manifest = tmp_path / "manifest.csv.gz"
    pd.DataFrame(
        {
            "spacehastenid": [11, 12],
            "clusterid": [101, 102],
            "atlas_id": ["atlas", "atlas"],
        }
    ).to_csv(manifest, index=False)
    output = tmp_path / "centroids.npz"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(database),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        check=True,
    )
    with np.load(output) as data:
        assert data["spacehastenid"].tolist() == [first, second]
        assert data["words"].shape == (2, 16)
    receipt = json.loads(output.with_suffix(".npz.json").read_text())
    assert receipt["occupied_clusters"] == 2
    assert receipt["unique_centroids"] == 2
