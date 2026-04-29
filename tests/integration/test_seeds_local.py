"""Tests for :mod:`spacehasten.stages.seeds`."""

from __future__ import annotations

from pathlib import Path

import pytest

from spacehasten.config.properties import PropertyRanges
from spacehasten.core.db import Database
from spacehasten.stages.seeds import import_seeds
from spacehasten.workspace.layout import WorkDir


def _write_smi(path: Path) -> None:
    path.write_text(
        "CCO ethanol-1\n"
        "c1ccccc1 benzene-1\n"
        "Cc1ccccc1 toluene-1\n"
        "  \n"  # blank line should be skipped
        "Oc1ccccc1 phenol-1\n"
        "not_a_smiles_at_all garbage-1\n"  # parse failure → dropped
    )


def _write_csv(path: Path) -> None:
    path.write_text(
        "SMILES,title,r_i_docking_score\n"
        "CCO,ethanol-1,-8.5\n"
        "c1ccccc1,benzene-1,-7.0\n"
        "Cc1ccccc1,toluene-1,-6.0\n"
        "Nc1ccccc1,aniline-1,bogus_score\n"  # non-numeric → dropped
    )


def _make_blobs(tmp_path: Path) -> tuple[Path, Path]:
    dock_param = tmp_path / "test_dock.in"
    dock_param.write_bytes(b"FORCEFIELD OPLS_2005\nPRECISION SP\n")
    dock_grid = tmp_path / "grid.zip"
    dock_grid.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    return dock_param, dock_grid


def test_import_smi_seeds_undocked(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="seedws")
    smi_path = tmp_path / "seeds.smi"
    _write_smi(smi_path)
    dp, dg = _make_blobs(tmp_path)

    db = Database(workdir.dbsh())
    n = import_seeds(
        db,
        workdir,
        smi_path=smi_path,
        dock_params_path=dp,
        dock_grid_path=dg,
        props=PropertyRanges(),
        processes=1,
        auto_train=False,
    )
    db.close()

    # 4 valid SMILES (1 garbage line dropped, 1 blank dropped)
    assert n == 4

    db2 = Database(workdir.dbsh())
    rows = db2.connection.execute(
        "SELECT smiles, smilesid, dock_score, dock_iteration FROM data ORDER BY spacehastenid"
    ).fetchall()
    assert len(rows) == 4
    assert all(r[2] is None and r[3] is None for r in rows)
    smilesids = {r[1] for r in rows}
    assert {"ethanol-1", "benzene-1", "toluene-1", "phenol-1"} == smilesids

    # Blobs and properties persisted.
    assert db2.load_dock_param() == dp.read_bytes()
    assert db2.load_dock_grid() == dg.read_bytes()
    props = db2.load_properties()
    assert props is not None
    assert props.mw == ("0.0", "500.0")

    # Index exists (from create_schema).
    has_idx = db2.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_reghash'"
    ).fetchone()
    assert has_idx is not None
    db2.close()


def test_import_csv_seeds_docked(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="seedws")
    csv_path = tmp_path / "seeds.csv"
    _write_csv(csv_path)
    dp, dg = _make_blobs(tmp_path)

    db = Database(workdir.dbsh())
    n = import_seeds(
        db,
        workdir,
        csv_path=csv_path,
        dock_params_path=dp,
        dock_grid_path=dg,
        props=PropertyRanges(),
        processes=1,
        auto_train=False,
    )
    db.close()

    # 3 numeric scores, 1 row dropped (non-numeric score).
    assert n == 3

    db2 = Database(workdir.dbsh())
    rows = db2.connection.execute(
        "SELECT smilesid, dock_score, dock_iteration FROM data ORDER BY spacehastenid"
    ).fetchall()
    db2.close()
    assert len(rows) == 3
    for _sid, score, it in rows:
        assert score is not None
        assert it == 0


def test_import_seeds_requires_exactly_one_input(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="seedws")
    dp, dg = _make_blobs(tmp_path)
    db = Database(workdir.dbsh())
    try:
        with pytest.raises(ValueError, match="exactly one"):
            import_seeds(
                db,
                workdir,
                dock_params_path=dp,
                dock_grid_path=dg,
                props=PropertyRanges(),
            )
    finally:
        db.close()


def test_import_seeds_csv_missing_columns(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="seedws")
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("foo,bar\n1,2\n")
    dp, dg = _make_blobs(tmp_path)
    db = Database(workdir.dbsh())
    try:
        with pytest.raises(ValueError, match="missing columns"):
            import_seeds(
                db,
                workdir,
                csv_path=csv_path,
                dock_params_path=dp,
                dock_grid_path=dg,
                props=PropertyRanges(),
                processes=1,
            )
    finally:
        db.close()


def test_import_seeds_smi_auto_train_requires_scheduler(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="seedws")
    smi = tmp_path / "seeds.smi"
    smi.write_text("CCO ethanol-1\n")
    dp, dg = _make_blobs(tmp_path)
    db = Database(workdir.dbsh())
    try:
        with pytest.raises(ValueError, match="scheduler"):
            import_seeds(
                db,
                workdir,
                smi_path=smi,
                dock_params_path=dp,
                dock_grid_path=dg,
                props=PropertyRanges(),
                processes=1,
                auto_train=True,
            )
    finally:
        db.close()
