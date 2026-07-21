"""Tests for ``spacehasten.workspace.layout.WorkDir``."""

from __future__ import annotations

import logging
from pathlib import Path

from spacehasten.workspace import Manifest, WorkDir

# --------------------------------------------------------------------------- #
# Single-root (backward compat: shared_root defaults to root)                 #
# --------------------------------------------------------------------------- #


def test_paths_are_relative_to_root(tmp_path: Path) -> None:
    wd = WorkDir(root=tmp_path / "myrun")
    assert wd.name == "myrun"
    assert wd.dbsh() == tmp_path / "myrun" / "myrun.dbsh"
    assert wd.simsearch_dir(3) == tmp_path / "myrun" / "simsearch" / "cycle3"
    assert wd.docking_dir(2) == tmp_path / "myrun" / "docking" / "iter2"
    assert wd.model_dir(1) == tmp_path / "myrun" / "models" / "v1"
    assert wd.clustering_dir() == tmp_path / "myrun" / "clustering"
    assert wd.atlas_dir() == tmp_path / "myrun" / "clustering" / "atlas"
    assert wd.archive_dir() == tmp_path / "myrun" / "archive"
    assert wd.logs_dir() == tmp_path / "myrun" / "logs"
    assert wd.slurm_logs_dir("dock_iter1") == tmp_path / "myrun" / "logs" / "slurm" / "dock_iter1"
    assert wd.manifest_path() == tmp_path / "myrun" / "manifest.json"
    assert wd.props_path() == tmp_path / "myrun" / "props.toml"


def test_single_root_shared_defaults_to_root(tmp_path: Path) -> None:
    wd = WorkDir(root=tmp_path / "single")
    assert wd.shared_root == tmp_path / "single"


# --------------------------------------------------------------------------- #
# Dual-root (local fast + shared NFS)                                         #
# --------------------------------------------------------------------------- #


def test_dual_root_local_paths(tmp_path: Path) -> None:
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    wd = WorkDir(root=local, shared_root=shared)

    # Local: DB, manifest, props, app logs
    assert wd.dbsh() == local / "local.dbsh"
    assert wd.manifest_path() == local / "manifest.json"
    assert wd.props_path() == local / "props.toml"
    assert wd.logs_dir() == local / "logs"


def test_dual_root_shared_paths(tmp_path: Path) -> None:
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    wd = WorkDir(root=local, shared_root=shared)

    # Shared: stage artefacts, models, SLURM logs
    assert wd.simsearch_dir(1) == shared / "simsearch" / "cycle1"
    assert wd.docking_dir(2) == shared / "docking" / "iter2"
    assert wd.model_dir(3) == shared / "models" / "v3"
    assert wd.clustering_dir() == shared / "clustering"
    assert wd.archive_dir() == shared / "archive"
    assert wd.export_dir() == shared / "export"
    assert wd.slurm_logs_dir("dock_iter1") == shared / "logs" / "slurm" / "dock_iter1"


# --------------------------------------------------------------------------- #
# Constructor                                                                 #
# --------------------------------------------------------------------------- #


def test_constructor_does_no_io(tmp_path: Path) -> None:
    target = tmp_path / "ghost"
    WorkDir(root=target)
    assert not target.exists()


# --------------------------------------------------------------------------- #
# Bootstrap                                                                   #
# --------------------------------------------------------------------------- #


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
    # Single-root: shared_root is None in manifest
    assert manifest.shared_root is None


def test_bootstrap_dual_root(tmp_path: Path) -> None:
    local = tmp_path / "fast"
    shared = tmp_path / "nfs"
    wd = WorkDir.bootstrap(local, name="proj", shared_root=shared)

    # Local root
    assert local.is_dir()
    assert wd.logs_dir().is_dir()
    assert wd.manifest_path().is_file()

    # Shared root
    assert shared.is_dir()
    assert (shared / "models").is_dir()

    # Manifest records shared_root
    manifest = Manifest.load(wd.manifest_path())
    assert manifest.name == "proj"
    assert manifest.shared_root == str(shared)


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


# --------------------------------------------------------------------------- #
# Disk policy                                                                 #
# --------------------------------------------------------------------------- #


def test_warn_if_wrong_disk_warns_on_data(monkeypatch, caplog) -> None:
    fake_root = Path("/data/someuser/SPACEHASTEN/myrun")
    monkeypatch.setenv("USER", "someuser")
    wd = WorkDir(root=fake_root)
    with caplog.at_level(logging.WARNING, logger="spacehasten.workspace.layout"):
        suggested = wd.warn_if_wrong_disk()
    assert suggested == "/wrk/someuser/SPACEHASTEN/myrun/"
    assert any("/data" in r.message for r in caplog.records)


def test_warn_if_wrong_disk_silent_on_wrk(tmp_path: Path) -> None:
    # tmp_path is not on /data; expect None and no exception.
    wd = WorkDir(root=tmp_path / "ok")
    assert wd.warn_if_wrong_disk() is None
