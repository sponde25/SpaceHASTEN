"""Integration test for the prediction stage with the local scheduler.

Uses a stub ``remote/predict.py`` (a tiny shell script) that emits a
fixed ``pred_score=-7.5`` for every input row, skipping real chemprop
inference. The real chemprop run is exercised by the Session 15 verify
smoke test.
"""

from __future__ import annotations

import stat
from pathlib import Path

from spacehasten.config.settings import Settings
from spacehasten.core.db import Database
from spacehasten.scheduler import LocalScheduler
from spacehasten.stages.prediction import predict_undocked
from spacehasten.workspace.layout import WorkDir


def _make_stub_predict(tmp_path: Path) -> Path:
    """Bash stub that mimics ``remote.predict``.

    Reads the input CSV at $1, writes ``smilesid,docking_score`` with a
    constant score of ``-7.5`` to $3. Argument $2 (model_dir) is checked
    for existence so the test catches a missing model.
    """
    stub = tmp_path / "stub_predict.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'in_csv="$1"\n'
        'model_dir="$2"\n'
        'out_csv="$3"\n'
        'if [ ! -f "$model_dir/model_0/pytorch_model.bin" ]; then\n'
        '  echo "model bin missing under $model_dir" >&2\n'
        '  exit 2\n'
        'fi\n'
        'mkdir -p "$(dirname "$out_csv")"\n'
        'echo "smilesid,docking_score" > "$out_csv"\n'
        # Skip header row, emit each smilesid with a constant score.
        'tail -n +2 "$in_csv" | while IFS=, read -r smi sid; do\n'
        '  echo "${sid},-7.5" >> "$out_csv"\n'
        'done\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return stub


def _seed_db_with_undocked_rows(db: Database, n: int) -> None:
    db.create_schema()
    # A couple of docked rows that must NOT be re-predicted.
    db.insert_seed_docked("hd1", "CCO", "ethanol-1", -8.0)
    db.insert_seed_docked("hd2", "c1ccccc1", "benzene-1", -7.5)
    # n undocked rows.
    for i in range(n):
        db.insert_seed_undocked(f"hu{i}", f"CC{'C' * (i + 1)}", f"ud-{i}")
    db.commit()


def _materialise_fake_model(workdir: WorkDir, version: int) -> Path:
    model_dir = workdir.model_dir(version)
    (model_dir / "model_0").mkdir(parents=True, exist_ok=True)
    bin_path = model_dir / "model_0" / "pytorch_model.bin"
    bin_path.write_bytes(b"fake-checkpoint")
    return model_dir


def test_prediction_stage_local(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="predws")
    db = Database(workdir.dbsh())
    _seed_db_with_undocked_rows(db, n=7)

    # Pretend version 1 is already on disk and recorded in the legacy table.
    _materialise_fake_model(workdir, version=1)
    db.store_model_blob(1, b"")
    db.commit()

    stub = _make_stub_predict(tmp_path)
    settings = Settings()  # defaults
    scheduler = LocalScheduler()

    # chunk_size=3 → expect 3 chunks: [3,3,1]
    n_updated = predict_undocked(
        db,
        workdir,
        scheduler,
        settings,
        model_version=1,
        chunk_size=3,
        predict_command_prefix=("bash", str(stub)),
    )
    db.close()

    assert n_updated == 7

    # Cycle defaults to 1 when no simsearch has run yet.
    predict_dir = workdir.simsearch_dir(1) / "PREDICT"
    assert (predict_dir / "predict_1.csv").exists()
    assert (predict_dir / "predict_2.csv").exists()
    assert (predict_dir / "predict_3.csv").exists()
    assert not (predict_dir / "predict_4.csv").exists()
    assert (predict_dir / "predicted_predict_3.csv").exists()

    # All undocked rows now have pred_score=-7.5 and pred_version=1.
    db2 = Database(workdir.dbsh())
    rows = db2.connection.execute(
        "SELECT smilesid, dock_score, pred_score, pred_version FROM data"
        " WHERE smilesid LIKE 'ud-%'"
    ).fetchall()
    assert len(rows) == 7
    for _sid, dock_score, pred_score, pred_version in rows:
        assert dock_score is None
        assert pred_score == -7.5
        assert pred_version == 1

    # Docked rows untouched.
    docked = db2.connection.execute(
        "SELECT pred_score FROM data WHERE smilesid IN ('ethanol-1','benzene-1')"
    ).fetchall()
    assert all(r[0] is None for r in docked)
    db2.close()


def test_prediction_stage_no_undocked_returns_zero(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="empty")
    db = Database(workdir.dbsh())
    db.create_schema()
    db.insert_seed_docked("h", "CCO", "ethanol-1", -8.0)
    db.commit()

    _materialise_fake_model(workdir, version=1)
    db.store_model_blob(1, b"")
    db.commit()

    settings = Settings()
    scheduler = LocalScheduler()
    n_updated = predict_undocked(
        db,
        workdir,
        scheduler,
        settings,
        model_version=1,
        chunk_size=10,
        # Won't be invoked, but must exist as a path.
        predict_command_prefix=("bash", "-c", "exit 1"),
    )
    db.close()
    assert n_updated == 0
