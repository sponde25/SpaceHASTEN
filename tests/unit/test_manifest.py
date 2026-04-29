"""Tests for ``spacehasten.workspace.manifest``."""

from __future__ import annotations

from pathlib import Path

from spacehasten.workspace import Manifest


def test_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    original = Manifest(name="run1")
    original.record_stage_start("training", {"version": 1, "cutoff": 10.0})
    original.record_stage_finish("training", "completed", scheduler_job_id="12345")
    original.record_run_start("train", ["--db", "x.dbsh"])
    original.record_run_finish("completed")
    original.save(path)

    loaded = Manifest.load(path)
    assert loaded.name == "run1"
    assert loaded.schema_version == 1
    assert loaded.stages["training"].status == "completed"
    assert loaded.stages["training"].scheduler_job_id == "12345"
    assert loaded.stages["training"].params == {"version": 1, "cutoff": 10.0}
    assert loaded.stages["training"].started_at is not None
    assert loaded.stages["training"].ended_at is not None
    assert len(loaded.runs) == 1
    assert loaded.runs[0].command == "train"
    assert loaded.runs[0].args == ["--db", "x.dbsh"]
    assert loaded.runs[0].status == "completed"


def test_save_is_atomic_no_tempfile_left(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    Manifest(name="r").save(path)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "manifest.json"]
    assert leftovers == []


def test_record_stage_start_overwrites_running_record() -> None:
    m = Manifest(name="r")
    m.record_stage_start("simsearch", {"cycle": 1})
    m.record_stage_finish("simsearch", "completed")
    m.record_stage_start("simsearch", {"cycle": 2})
    assert m.stages["simsearch"].status == "running"
    assert m.stages["simsearch"].params == {"cycle": 2}
    assert m.stages["simsearch"].ended_at is None


def test_record_stage_finish_without_start_creates_record() -> None:
    m = Manifest(name="r")
    m.record_stage_finish("docking", "failed")
    assert m.stages["docking"].status == "failed"
    assert m.stages["docking"].ended_at is not None


def test_record_run_finish_without_runs_returns_none() -> None:
    m = Manifest(name="r")
    assert m.record_run_finish("completed") is None


def test_runs_appended_in_order() -> None:
    m = Manifest(name="r")
    m.record_run_start("a", [])
    m.record_run_finish("completed")
    m.record_run_start("b", [])
    assert [r.command for r in m.runs] == ["a", "b"]
    assert m.runs[0].status == "completed"
    assert m.runs[1].status == "running"


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "manifest.json"
    Manifest(name="r").save(path)
    assert path.is_file()


def test_default_created_at_is_utc() -> None:
    m = Manifest(name="r")
    assert m.created_at.tzinfo is not None
