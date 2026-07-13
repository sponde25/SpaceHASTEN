"""Tests for :mod:`spacehasten.stages.export`."""

from __future__ import annotations

import csv
import os
import tarfile
from pathlib import Path

import pytest

from spacehasten.config.settings import Settings
from spacehasten.core.db import ClusterRow, Database
from spacehasten.stages.export import export_csv, export_poses, export_seeds
from spacehasten.workspace.layout import WorkDir


def _seed_db(db: Database) -> None:
    db.create_schema()
    # 3 docked rows, 1 above cutoff.
    db.insert_seed_docked("h1", "CCO", "ethanol-1", -8.5)
    db.insert_seed_docked("h2", "c1ccccc1", "benzene-1", -7.0)
    db.insert_seed_docked("h3", "Cc1ccccc1", "toluene-1", 1.0)  # above cutoff
    # Populate cluster rows for the happy-path tests (export left-joins
    # against clusters, so this isn't strictly required, but most tests
    # want a non-empty clusterid column to assert against).
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


def test_export_csv_includes_rows_without_cluster_assignment(tmp_path: Path) -> None:
    """Regression test: hits with no ``clusters`` row (e.g. workspace ran
    with ``--strategy greedy`` and ``spacehasten cluster`` was never
    invoked, or the compound was docked after the last clustering pass)
    must still be exported, with an empty ``clusterid`` — not silently
    dropped by an inner join."""
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="exp")
    db = Database(workdir.dbsh())
    db.create_schema()
    db.insert_seed_docked("h1", "CCO", "ethanol-1", -8.5)
    db.insert_seed_docked("h2", "c1ccccc1", "benzene-1", -7.0)
    # Note: clusters table left empty entirely (no `cluster` run yet).
    db.commit()
    out = tmp_path / "results.csv"

    n = export_csv(db, out, cutoff=0.0)
    db.close()

    assert n == 2
    with out.open("rt", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    body = rows[1:]
    assert len(body) == 2
    assert body[0][0] == "CCO"
    assert body[0][7] == ""  # clusterid empty, row still present


def test_export_seeds_formats_for_reimport(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="exp")
    db = Database(workdir.dbsh())
    _seed_db(db)
    out = tmp_path / "seeds.csv"

    n = export_seeds(db, out)

    # All 3 seed rows — seeds are not filtered by dock_score.
    assert n == 3
    with out.open("rt", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["SMILES", "title", "r_i_docking_score"]
    body = rows[1:]
    assert len(body) == 3
    # Ordered by insertion (spacehastenid), not by score.
    assert body[0] == ["CCO", "ethanol-1", "-8.5"]
    assert body[1] == ["c1ccccc1", "benzene-1", "-7.0"]
    assert body[2] == ["Cc1ccccc1", "toluene-1", "1.0"]

    # Round-trips through import_seeds using the default CSV column names.
    from spacehasten.config.properties import PropertyRanges
    from spacehasten.stages.seeds import import_seeds

    workdir2 = WorkDir.bootstrap(tmp_path / "ws2", name="exp2")
    db2 = Database(workdir2.dbsh())
    db2.create_schema()
    n_imported = import_seeds(db2, csv_path=out, props=PropertyRanges())
    db2.close()
    db.close()

    assert n_imported == 3


def test_export_seeds_excludes_later_screening_cycles(tmp_path: Path) -> None:
    """Compounds docked in later screening cycles (``dock_iteration >= 1``)
    must not appear in the seed export, even if their score would pass a
    typical cutoff."""
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="exp")
    db = Database(workdir.dbsh())
    _seed_db(db)  # 3 seed rows at dock_iteration == 0

    # Simulate a compound discovered via simsearch and docked in round 1.
    sid = db.insert_simsearch_hit("h4", "CCN", "later-hit-1", None, None, None, 1)
    db.apply_dock_scores([(-9.5, 1, sid)])
    db.commit()

    out = tmp_path / "seeds.csv"
    n = export_seeds(db, out)
    db.close()

    assert n == 3
    with out.open("rt", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    titles = [r[1] for r in rows[1:]]
    assert titles == ["ethanol-1", "benzene-1", "toluene-1"]
    assert "later-hit-1" not in titles


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
