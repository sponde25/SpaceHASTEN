"""Tests for :mod:`spacehasten.stages.export`."""

from __future__ import annotations

import csv
import os
import tarfile
from pathlib import Path

import pytest

from spacehasten.config.settings import Settings
from spacehasten.core.db import ClusterRow, Database
from spacehasten.stages.export import export_csv, export_poses
from spacehasten.workspace.layout import WorkDir


def _seed_db(db: Database) -> None:
    db.create_schema()
    # 3 docked rows, 1 above cutoff.
    db.insert_seed_docked("h1", "CCO", "ethanol-1", -8.5)
    db.insert_seed_docked("h2", "c1ccccc1", "benzene-1", -7.0)
    db.insert_seed_docked("h3", "Cc1ccccc1", "toluene-1", 1.0)  # above cutoff
    # Need cluster rows because export joins data ⨝ clusters.
    db.replace_clusters([
        ClusterRow(spacehastenid=1, clusterid=1),
        ClusterRow(spacehastenid=2, clusterid=2),
        ClusterRow(spacehastenid=3, clusterid=3),
    ])
    db.commit()


def test_export_csv_filters_and_formats(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="exp")
    db = Database(workdir.dbsh())
    _seed_db(db)
    out = tmp_path / "results.csv"

    n = export_csv(db, out, cutoff=0.0)
    db.close()

    assert n == 2
    with out.open("rt", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "smiles", "smilesid", "dock_score", "pred_score",
        "spacelight", "ftrees", "dock_iteration", "clusterid",
    ]
    body = rows[1:]
    assert len(body) == 2
    # Sorted by dock_score ascending — ethanol first.
    assert body[0][0] == "CCO"
    assert body[0][1] == "ethanol-1/1"
    assert float(body[0][2]) == -8.5
    assert body[0][6] == "0"  # dock_iteration


def test_export_poses_requires_docking_dir(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="exp")
    db = Database(workdir.dbsh())
    _seed_db(db)

    settings = Settings()
    settings.paths.export_poses_script = "/nonexistent/export_poses.py"
    try:
        with pytest.raises(FileNotFoundError, match="no docking iteration"):
            export_poses(
                db, workdir, tmp_path / "out.mae",
                cutoff=0.0, iteration=1, settings=settings,
            )
    finally:
        db.close()


def test_export_poses_no_iterations(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="exp")
    db = Database(workdir.dbsh())
    db.create_schema()
    db.commit()

    settings = Settings()
    settings.paths.export_poses_script = "/nonexistent/export_poses.py"
    try:
        with pytest.raises(ValueError, match="no dock iterations"):
            export_poses(
                db, workdir, tmp_path / "out.mae",
                cutoff=0.0, settings=settings,
            )
    finally:
        db.close()


def test_export_poses_invokes_script_per_pv(tmp_path: Path) -> None:
    """End-to-end smoke test using a stub ``export_poses.py`` script.

    The stub reads its argv, writes a marker file under the same dir as
    the input ``*_pv.maegz``, and creates a fake
    ``spacehasten_virtual_hits_<...>.mae`` in that dir for concatenation.
    """
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="exp")
    db = Database(workdir.dbsh())
    _seed_db(db)
    db.insert_seed_docked("h4", "CC", "ethane-1", -9.0)
    db.connection.execute(
        "UPDATE data SET dock_iteration = 1 WHERE dock_iteration = 0"
    )
    db.commit()

    # Build a fake docking dir with a tar containing two *_pv.maegz files.
    dock_dir = workdir.docking_dir(1)
    dock_dir.mkdir(parents=True, exist_ok=True)
    results_dir = dock_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pv_a = tmp_path / "stage" / "chunk_1_pv.maegz"
    pv_b = tmp_path / "stage" / "chunk_2_pv.maegz"
    pv_a.parent.mkdir(parents=True, exist_ok=True)
    pv_a.write_bytes(b"FAKE-A")
    pv_b.write_bytes(b"FAKE-B")
    tar_path = results_dir / "results-chunk_1.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(pv_a, arcname="chunk_1_pv.maegz")
        tf.add(pv_b, arcname="chunk_2_pv.maegz")

    # Build a stub script that emits a hits file per pv, with content
    # derived from the pv basename so concatenation order is checkable.
    stub = tmp_path / "stub_export_poses.py"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "pv = sys.argv[1]\n"
        "cutoff = sys.argv[2]\n"
        "dbname = sys.argv[3]\n"
        "base = os.path.basename(pv).replace('.maegz', '')\n"
        "outdir = os.path.dirname(pv)\n"
        "out = os.path.join(outdir, 'spacehasten_virtual_hits_' + base + '.mae')\n"
        "open(out, 'wb').write(('HIT:' + base + '\\n').encode())\n"
    )
    stub.chmod(0o755)

    settings = Settings()
    # Use plain python3 in lieu of $SCHRODINGER/run for the stub.
    settings.paths.schrodinger_run = "python3"
    settings.paths.export_poses_script = str(stub)
    # Use a private scratch under tmp_path so we don't pollute /wrk.
    settings.paths.scratch_default = str(tmp_path / "scratch")
    os.makedirs(tmp_path / "scratch", exist_ok=True)

    out_path = tmp_path / "hits.maegz"
    result = export_poses(
        db, workdir, out_path,
        cutoff=0.0, iteration=1, settings=settings,
    )
    db.close()

    assert result == out_path
    assert out_path.exists()
    # The output is gzipped (pigz); decompress and check contents.
    import gzip
    body = gzip.decompress(out_path.read_bytes())
    # Both pv hits concatenated.
    assert b"HIT:chunk_1_pv\n" in body
    assert b"HIT:chunk_2_pv\n" in body
