"""Tests for :mod:`spacehasten.stages.archive`."""

from __future__ import annotations

from pathlib import Path

import pytest

from spacehasten.core.db import Database
from spacehasten.stages.archive import (
    archive_clean,
    archive_create,
    archive_extract,
    archive_restore,
)
from spacehasten.workspace.layout import WorkDir


def _populate_workdir(tmp_path: Path, name: str = "wsa") -> WorkDir:
    workdir = WorkDir.bootstrap(tmp_path / name, name=name)
    db = Database(workdir.dbsh())
    db.create_schema()
    db.insert_seed_docked("h1", "CCO", "ethanol-1", -8.0)
    db.commit()
    db.close()
    # Create a few regenerable subdirs with content.
    (workdir.simsearch_dir(1)).mkdir(parents=True, exist_ok=True)
    (workdir.simsearch_dir(1) / "sentinel.txt").write_text("ss1")
    (workdir.docking_dir(1)).mkdir(parents=True, exist_ok=True)
    (workdir.docking_dir(1) / "sentinel.txt").write_text("dock1")
    (workdir.clustering_dir()).mkdir(parents=True, exist_ok=True)
    (workdir.clustering_dir() / "sentinel.txt").write_text("clu")
    return workdir


@pytest.mark.parametrize("bundle", [False, True])
def test_archive_create_roundtrip(tmp_path: Path, bundle: bool) -> None:
    workdir = _populate_workdir(tmp_path)
    archive_path = archive_create(workdir, bundle=bundle)

    expected_suffix = ".archived-spacehasten.tgz" if bundle else ".archived-spacehasten"
    assert archive_path.name.endswith(expected_suffix)
    assert archive_path.exists()
    assert archive_path.stat().st_size > 0

    # Restore (or extract) into a fresh location.
    target = WorkDir(root=tmp_path / "restored")
    if bundle:
        archive_extract(archive_path, target)
    else:
        archive_restore(archive_path, target)

    # All sentinel files round-tripped under the restored root.
    assert (target.root / f"{workdir.name}.dbsh").exists()
    assert (target.simsearch_dir(1) / "sentinel.txt").read_text() == "ss1"
    assert (target.docking_dir(1) / "sentinel.txt").read_text() == "dock1"
    assert (target.clustering_dir() / "sentinel.txt").read_text() == "clu"

    # The DB inside the restored archive is queryable.
    db = Database(target.root / f"{workdir.name}.dbsh")
    rows = db.connection.execute("SELECT smilesid FROM data").fetchall()
    db.close()
    assert rows == [("ethanol-1",)]


def test_archive_extract_refuses_nonempty_target(tmp_path: Path) -> None:
    workdir = _populate_workdir(tmp_path)
    archive_path = archive_create(workdir, bundle=True)

    target_root = tmp_path / "target"
    target_root.mkdir()
    (target_root / "stuff.txt").write_text("existing")
    target = WorkDir(root=target_root)
    with pytest.raises(FileExistsError):
        archive_extract(archive_path, target)


def test_archive_clean_removes_regenerable_only(tmp_path: Path) -> None:
    workdir = _populate_workdir(tmp_path)
    n = archive_clean(workdir)
    assert n == 3

    # Regenerable scratch is gone.
    assert not (workdir.root / "simsearch").exists()
    assert not (workdir.root / "docking").exists()
    assert not (workdir.root / "clustering").exists()
    # Preserved artefacts still present.
    assert workdir.dbsh().exists()
    assert workdir.manifest_path().exists()
    assert (workdir.root / "models").exists()
    assert workdir.logs_dir().exists()


def test_archive_clean_idempotent(tmp_path: Path) -> None:
    workdir = _populate_workdir(tmp_path)
    archive_clean(workdir)
    # Second call removes nothing more.
    assert archive_clean(workdir) == 0
