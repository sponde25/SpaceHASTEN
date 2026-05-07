"""Integration test for the ``spacehasten screening-cycle`` workflow command.

Verifies the ordering — [train] → search(docked) →
search(predicted) ×2 → dock per round — by monkeypatching
each stage with a recorder. The CLI is exercised end-to-end (argparse →
``_common`` → stage dispatch); only the stages themselves are stubbed
out so the test runs in milliseconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from spacehasten.cli import main as cli_main
from spacehasten.workspace.layout import WorkDir


@pytest.fixture
def stub_workspace(tmp_path: Path) -> Path:
    """Bootstrap a workspace and return the workspace root."""
    root = tmp_path / "ws"
    WorkDir.bootstrap(root, name="ws")
    db_path = root / "ws.dbsh"
    db_path.write_bytes(b"")  # empty so Database(...) opens
    return root


def _install_stage_stubs(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, dict[str, Any]]],
    *,
    dock_iteration: int = 0,
    model_version: int = 1,
) -> None:
    def _record(name: str):  # type: ignore[no-untyped-def]
        def _fn(db, workdir, scheduler, settings, **kwargs):  # type: ignore[no-untyped-def]
            calls.append((name, kwargs))
            return 1
        return _fn

    monkeypatch.setattr(cli_main.training, "train", _record("train"))
    monkeypatch.setattr(cli_main.simsearch, "simsearch", _record("simsearch"))
    monkeypatch.setattr(cli_main.docking, "dock", _record("dock"))
    monkeypatch.setattr(cli_main.prediction, "predict_undocked", _record("predict"))

    # Stub DB methods used to decide whether to train.
    import spacehasten.core.db as db_mod
    monkeypatch.setattr(
        db_mod.Database, "latest_dock_iteration", lambda self: dock_iteration,
    )
    monkeypatch.setattr(
        db_mod.Database, "latest_model_version", lambda self: model_version,
    )


def test_screening_cycle_first_round_no_train(
    stub_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First screening cycle (dock_iteration=0) should NOT train."""
    calls: list[tuple[str, dict[str, Any]]] = []
    _install_stage_stubs(monkeypatch, calls, dock_iteration=0)

    rc = cli_main.main([
        "-w", str(stub_workspace),
        "--scheduler", "local",
        "screening-cycle",
        "--rounds", "1",
        "--simsearch-top-n", "5",
        "--simsearch-jobs", "2",
        "--dock-top-n", "10",
        "--dock-cpus", "2",
    ])
    assert rc == 0
    names = [c[0] for c in calls]
    # No train; then search(docked) → search(predicted) ×2 → dock
    assert names == [
        "simsearch",
        "simsearch",
        "simsearch",
        "dock",
    ]
    sources = [c[1].get("source") for c in calls if c[0] == "simsearch"]
    assert sources == ["docked", "predicted", "predicted"]
    dock_call = next(c for c in calls if c[0] == "dock")
    assert dock_call[1]["top_n"] == 10
    assert dock_call[1]["cpus"] == 2


def test_screening_cycle_trains_after_first(
    stub_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After first cycle (dock_iteration>0), screening-cycle trains first."""
    calls: list[tuple[str, dict[str, Any]]] = []
    _install_stage_stubs(monkeypatch, calls, dock_iteration=1)

    rc = cli_main.main([
        "-w", str(stub_workspace),
        "--scheduler", "local",
        "screening-cycle",
        "--rounds", "1",
        "--simsearch-top-n", "5",
        "--simsearch-jobs", "2",
        "--dock-top-n", "10",
        "--dock-cpus", "2",
    ])
    assert rc == 0
    names = [c[0] for c in calls]
    # Train first, then the search pipeline
    assert names == [
        "train",
        "simsearch",
        "simsearch",
        "simsearch",
        "dock",
    ]
