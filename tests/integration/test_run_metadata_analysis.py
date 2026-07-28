from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "analyze_run_metadata.py"


def test_run_metadata_analysis_end_to_end(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    database = run / "run.dbsh"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE data(spacehastenid INTEGER PRIMARY KEY,smiles TEXT,dock_score REAL,"
            "dock_iteration INTEGER)"
        )
        connection.execute(
            "CREATE TABLE predictions(spacehastenid INTEGER,model_version INTEGER,pred_score REAL,"
            "epistemic_std REAL,total_std REAL)"
        )
        connection.executemany(
            "INSERT INTO data VALUES (?,?,?,?)",
            [(1, "CCO", -10.0, 1), (2, "CCN", -8.0, 2)],
        )
        connection.executemany(
            "INSERT INTO predictions VALUES (?,?,?,?,?)",
            [(1, 0, -9.5, 0.2, 0.3), (2, 1, -8.5, 0.3, 0.4)],
        )
    model_root = run / "run_shared" / "models"
    for version in (0, 1):
        directory = model_root / f"v{version}"
        directory.mkdir(parents=True)
        (directory / "training_metadata.json").write_text(
            json.dumps({"train_rows": 10 + version, "best_epoch": 3}), encoding="utf-8"
        )
        pd.DataFrame({"smiles": ["CCC"]}).to_csv(directory / "train.csv", index=False)
    manifest = tmp_path / "manifest.csv.gz"
    pd.DataFrame(
        {
            "spacehastenid": [1, 2],
            "round": [1, 2],
            "smiles": ["CCO", "CCN"],
            "model_version": [0, 1],
            "is_scored": [True, True],
            "is_hit": [True, False],
        }
    ).to_csv(manifest, index=False)
    (run / "run.log").write_text(
        "\n".join(
            [
                "[07/01/26 10:00:00] Screening round 1/2",
                "[07/01/26 10:01:00] Submitted docking job 100",
                "[07/01/26 10:04:00] Job 100 (dock_iter1) completed successfully",
                "[07/01/26 10:05:00] Updated dock_score for 1 rows (iter=1)",
                "[07/01/26 10:05:01] Screening round 2/2",
                "[07/01/26 10:06:00] Submitted docking job 200",
                "[07/01/26 10:09:00] Job 200 (dock_iter2) completed successfully",
                "[07/01/26 10:10:00] Updated dock_score for 1 rows (iter=2)",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "metadata"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(run),
            "--manifest",
            str(manifest),
            "--output-root",
            str(output),
            "--hit-threshold",
            "-9.7",
            "--dpi",
            "50",
        ],
        check=True,
    )
    receipt = json.loads((output / "_SUCCESS.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["models_with_metadata"] == 2
    assert receipt["stage_jobs"] == 2
    timing = pd.read_csv(output / "round_timing.csv")
    assert timing["selected"].tolist() == [1, 1]
    assert timing["hits"].tolist() == [1, 0]
    leakage = pd.read_csv(output / "training_leakage_validation.csv")
    assert set(leakage["status"]) == {"passed"}
