from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "selected_resampling.py"


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, str(SCRIPT), *arguments], check=True)


def test_selected_resampling_end_to_end(tmp_path: Path) -> None:
    identifiers = np.asarray([1, 2, 3, 4], dtype=np.int64)
    manifest = pd.DataFrame(
        {
            "spacehastenid": [1, 2, 3, 4, 1],
            "reghash": ["h1", "h2", "h3", "h4", "h1"],
            "round": [1, 1, 2, 2, 3],
            "is_hit": [True, False, True, False, False],
            "clusterid": [10, 10, 20, 30, 10],
        }
    )
    structures = pd.DataFrame(
        {
            "spacehastenid": identifiers,
            "reghash": ["h1", "h2", "h3", "h4"],
            "typed_scaffold": ["t1", "t1", "t2", "t3"],
            "generic_framework": ["g1", "g1", "g2", "g3"],
        }
    )
    manifest_path = tmp_path / "manifest.csv.gz"
    structure_path = tmp_path / "structures.csv.gz"
    fingerprint_path = tmp_path / "fingerprints.npz"
    manifest.to_csv(manifest_path, index=False)
    structures.to_csv(structure_path, index=False)
    words = np.zeros((4, 16), dtype=np.uint64)
    words[:, 0] = [3, 5, 15, 9]
    np.savez_compressed(
        fingerprint_path,
        spacehastenid=identifiers,
        words=words,
        popcounts=np.asarray([2, 2, 4, 2], dtype=np.uint16),
    )
    output = tmp_path / "resampling"
    run(
        "prepare",
        "--manifest",
        str(manifest_path),
        "--structure-cache",
        str(structure_path),
        "--fingerprints",
        str(fingerprint_path),
        "--output-root",
        str(output),
        "--task-count",
        "2",
        "--replicates",
        "4",
        "--pair-samples",
        "10",
    )
    run("worker", "--output-root", str(output), "--task-index", "1")
    run("worker", "--output-root", str(output), "--task-index", "2")
    run("combine", "--output-root", str(output), "--dpi", "50")

    assert "export PYTHONPATH=" in (output / "submit.sh").read_text()
    receipt = json.loads((output / "_SUCCESS.json").read_text())
    assert receipt["replicates"] == 4
    assert receipt["replicate_rows"] == 12
    replicates = pd.read_csv(output / "resampling_replicates.csv")
    assert set(replicates["replicate"]) == {0, 1, 2, 3}
    assert set(replicates["round"]) == {1, 2, 3}
    assert set(replicates["sample_size"]) == {0, 1}
    assert (output / "count_matched_rarefaction.png").stat().st_size > 0
