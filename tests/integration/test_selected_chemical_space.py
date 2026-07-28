from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "plot_selected_chemical_space.py"
)


def test_selected_chemical_space_figures(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv.gz"
    pd.DataFrame(
        {
            "spacehastenid": [11, 12, 13, 14],
            "round": [1, 1, 2, 2],
            "is_hit": [True, False, True, False],
        }
    ).to_csv(manifest, index=False)
    selected_coordinates = tmp_path / "selected_coordinates.npz"
    np.savez_compressed(
        selected_coordinates,
        spacehastenid=np.asarray([11, 12, 13, 14]),
        umap=np.asarray([[0.0, 0.0], [0.1, 0.2], [1.0, 1.0], [1.2, 1.1]]),
    )
    seed_coordinates = tmp_path / "seed_coordinates.npz"
    np.savez_compressed(
        seed_coordinates,
        spacehastenid=np.asarray([1, 2]),
        umap=np.asarray([[-1.0, -1.0], [-0.5, -0.7]]),
    )
    seed_reference = tmp_path / "seed_reference.npz"
    np.savez_compressed(seed_reference, seed_spacehastenid=np.asarray([1, 2]))
    enrichment = tmp_path / "enrichment.csv"
    pd.DataFrame(
        {
            "clusterid": [101, 102],
            "selected_count": [2, 2],
            "scored_count": [2, 2],
            "hit_count": [1, 1],
            "centroid_spacehastenid": [1, 11],
            "centroid_source": ["seed", "virtual"],
        }
    ).to_csv(enrichment, index=False)
    nearest = tmp_path / "nearest.csv.gz"
    pd.DataFrame(
        {
            "round": [1, 1, 2, 2],
            "nearest_seed_tanimoto": [0.8, 0.7, 0.5, 0.4],
        }
    ).to_csv(nearest, index=False)
    output = tmp_path / "figures"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--selected-coordinates",
            str(selected_coordinates),
            "--seed-coordinates",
            str(seed_coordinates),
            "--seed-reference-cache",
            str(seed_reference),
            "--portfolio-enrichment",
            str(enrichment),
            "--selected-nearest",
            str(nearest),
            "--output-root",
            str(output),
            "--label",
            "Test run",
            "--dpi",
            "50",
        ],
        check=True,
    )
    metadata = json.loads((output / "figure_metadata.json").read_text())
    assert metadata["selected"] == 4
    assert metadata["hits"] == 2
    for index in range(1, 5):
        assert list(output.glob(f"0{index}_*.png"))[0].stat().st_size > 0
        assert list(output.glob(f"0{index}_*.pdf"))[0].stat().st_size > 0
