"""Tests for ``spacehasten.workspace.layout.WorkDir``."""

from __future__ import annotations

import logging
from pathlib import Path

from spacehasten.workspace import Manifest, WorkDir


def test_paths_are_relative_to_root(tmp_path: Path) -> None:
    wd = WorkDir(root=tmp_path / "myrun")
    assert wd.name == "myrun"
    assert wd.dbsh() == tmp_path / "myrun" / "myrun.dbsh"
    assert wd.simsearch_dir(3) == tmp_path / "myrun" / "simsearch" / "cycle3"
    assert wd.docking_dir(2) == tmp_path / "myrun" / "docking" / "iter2"
    assert wd.model_dir(1) == tmp_path / "myrun" / "models" / "v1"
    assert wd.clustering_dir() == tmp_path / "myrun" / "clustering"
    assert wd.archive_dir() == tmp_path / "myrun" / "archive"
    assert wd.logs_dir() == tmp_path / "myrun" / "logs"
    assert (
        wd.slurm_logs_dir("dock_iter1")
        == tmp_path / "myrun" / "logs" / "slurm" / "dock_iter1"
    )
    assert wd.manifest_path() == tmp_path / "myrun" / "manifest.json"
    assert wd.props_path() == tmp_path / "myrun" / "props.toml"


def test_constructor_does_no_io(tmp_path: Path) -> None:
    target = tmp_path / "ghost"
    WorkDir(root=target)
    assert not target.exists()


def test_bootstrap_creates_skeleton(tmp_path: Path) -> None:
    root = tmp_path / "run1"
    wd = WorkDir.bootstrap(root)
    assert root.is_dir()
    assert wd.logs_dir().is_dir()
    assert (root / "models").is_dir()
    assert wd.manifest_path().is_file()

    manifest = Manifest.load(wd.manifest_path())
    assert manifest.name == "run1"
    assert manifest.schema_version == 1


def test_bootstrap_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "run2"
    wd1 = WorkDir.bootstrap(root)
    # Mutate the manifest so we can confirm a second bootstrap does not
    # overwrite an existing manifest.
    manifest = Manifest.load(wd1.manifest_path())
    manifest.record_stage_start("training", {"version": 1})
    manifest.save(wd1.manifest_path())

    wd2 = WorkDir.bootstrap(root)
    assert wd2 == wd1
    reloaded = Manifest.load(wd2.manifest_path())
    assert "training" in reloaded.stages


def test_bootstrap_custom_name(tmp_path: Path) -> None:
    root = tmp_path / "physical_dir"
    wd = WorkDir.bootstrap(root, name="logical_name")
    manifest = Manifest.load(wd.manifest_path())
    assert manifest.name == "logical_name"


def test_warn_if_wrong_disk_warns_on_wrk(monkeypatch, caplog) -> None:
    fake_root = Path("/wrk/someuser/SPACEHASTEN/myrun")
    monkeypatch.setenv("USER", "someuser")
    wd = WorkDir(root=fake_root)
    with caplog.at_level(logging.WARNING, logger="spacehasten.workspace.layout"):
        suggested = wd.warn_if_wrong_disk()
    assert suggested == "/data/someuser/SPACEHASTEN/myrun/"
    assert any("/wrk" in r.message for r in caplog.records)


def test_warn_if_wrong_disk_silent_on_data(tmp_path: Path) -> None:
    # tmp_path is not on /wrk; expect None and no exception.
    wd = WorkDir(root=tmp_path / "ok")
    assert wd.warn_if_wrong_disk() is None
