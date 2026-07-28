from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "validate_run_analysis.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_run_analysis_validation_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "analysis"
    standard = root / "standard"
    structure = root / "structure_cache"
    selected = root / "selected"
    figures = root / "figures"
    for path in (standard, structure, selected, figures):
        path.mkdir(parents=True)
    database = tmp_path / "snapshot.dbsh"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE data(spacehastenid INTEGER PRIMARY KEY)")
    snapshot_receipt = tmp_path / "snapshot.dbsh.json"
    write_json(
        snapshot_receipt,
        {
            "source_quick_check": "ok",
            "snapshot_quick_check": "ok",
            "size_bytes": database.stat().st_size,
            "counts": {"data": 0},
        },
    )
    (standard / "round_metrics.csv").write_text(
        "round,selected,cumulative_hits\n1,2,1\n2,1,1\n", encoding="utf-8"
    )
    for name in (
        "coverage.csv",
        "cutoff_curve.csv",
        "family_metrics.csv",
        "calibration_metrics.csv",
    ):
        (standard / name).write_text("round,value\n1,1\n", encoding="utf-8")
    write_json(standard / "analysis_manifest.json", {})
    write_json(standard / "_SUCCESS.json", {"status": "ok", "rounds": [1]})

    manifest = structure / "selected_manifest.csv.gz"
    structures = structure / "structure_cache.csv.gz"
    with gzip.open(manifest, "wt", encoding="utf-8") as handle:
        handle.write("spacehastenid,reghash\n1,h1\n2,h2\n1,h1\n")
    with gzip.open(structures, "wt", encoding="utf-8") as handle:
        handle.write("spacehastenid,reghash\n1,h1\n2,h2\n")
    fingerprints = structure / "fingerprints.npz"
    np.savez_compressed(
        fingerprints,
        spacehastenid=np.asarray([1, 2]),
        words=np.zeros((2, 16), dtype=np.uint64),
        popcounts=np.zeros(2, dtype=np.uint16),
    )
    write_json(
        structure / "_SUCCESS.json",
        {
            "status": "complete",
            "rows": 2,
            "selected_attempts": 3,
            "manifest_sha256": digest(manifest),
            "structure_sha256": digest(structures),
            "fingerprint_sha256": digest(fingerprints),
        },
    )
    selected_table = selected / "diversity_metrics.csv"
    selected_table.write_text("round,cohort\n1,selected\n", encoding="utf-8")
    write_json(
        selected / "_SUCCESS.json",
        {
            "status": "complete",
            "selected_attempts": 3,
            "hits": 1,
            "outputs": [
                {
                    "path": selected_table.name,
                    "bytes": selected_table.stat().st_size,
                    "sha256": digest(selected_table),
                }
            ],
        },
    )
    nearest = root / "nearest.npz"
    np.savez_compressed(
        nearest,
        spacehastenid=np.asarray([1, 2]),
        nearest_seed_id=np.asarray([10, 10]),
        tanimoto=np.asarray([0.5, 0.7]),
    )
    umap = root / "umap.npz"
    np.savez_compressed(
        umap,
        spacehastenid=np.asarray([1, 2]),
        umap=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
    )
    (figures / "figure.png").write_bytes(b"png")
    (figures / "figure.pdf").write_bytes(b"pdf")
    markdown = root / "REPORT.md"
    markdown.write_text("# Report\n\n![Figure](figures/figure.png)\n", encoding="utf-8")
    html = root / "REPORT.html"
    html.write_text('<img src="data:image/png;base64,cG5n">', encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--analysis-root",
            str(root),
            "--snapshot-receipt",
            str(snapshot_receipt),
            "--database",
            str(database),
            "--standard-root",
            str(standard),
            "--structure-root",
            str(structure),
            "--selected-root",
            str(selected),
            "--nearest-seed",
            str(nearest),
            "--umap",
            str(umap),
            "--figures-root",
            str(figures),
            "--markdown",
            str(markdown),
            "--html",
            str(html),
        ],
        check=True,
    )
    result = json.loads((root / "FINAL_VALIDATION.json").read_text())
    assert result["status"] == "ok"
    assert result["checks"]["nearest_seed"]["rows"] == 2
    manifest_result = json.loads((root / "artifact_manifest.json").read_text())
    assert manifest_result["status"] == "complete"
    assert manifest_result["artifacts"]
