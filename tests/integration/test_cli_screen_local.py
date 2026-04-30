"""Integration test for the ``spacehasten screen`` macro command.

Verifies the legacy ordering — train? → simsearch(docked) →
simsearch(predicted) ×2 → dock per round — by monkeypatching each
stage with a recorder. The CLI is exercised end-to-end (argparse →
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
    monkeypatch: pytest.MonkeyPatch, calls: list[tuple[str, dict[str, Any]]]
) -> None:
    def _record(name: str):  # type: ignore[no-untyped-def]
        def _fn(db, workdir, scheduler, settings, **kwargs):  # type: ignore[no-untyped-def]
            calls.append((name, kwargs))
            return 1
        return _fn

    monkeypatch.setattr(cli_main.training, "train", _record("train"))
    monkeypatch.setattr(cli_main.simsearch, "simsearch", _record("simsearch"))
    monkeypatch.setattr(cli_main.docking, "dock", _record("dock"))


def test_screen_order_one_round(
    stub_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    _install_stage_stubs(monkeypatch, calls)

    rc = cli_main.main([
        "-w", str(stub_workspace),
        "--scheduler", "local",
        "screen",
        "--rounds", "1",
        "--simsearch-top-n", "5",
        "--simsearch-cpu", "2",
        "--dock-top-n", "10",
        "--dock-cpus", "2",
    ])
    assert rc == 0
    names = [c[0] for c in calls]
    # No train, then 1 docked + 2 predicted simsearch + 1 dock.
    assert names == ["simsearch", "simsearch", "simsearch", "dock"]
    sources = [c[1].get("source") for c in calls if c[0] == "simsearch"]
    assert sources == ["docked", "predicted", "predicted"]
    dock_call = next(c for c in calls if c[0] == "dock")
    assert dock_call[1]["top_n"] == 10
    assert dock_call[1]["cpus"] == 2


def test_screen_train_first_two_rounds(
    stub_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    _install_stage_stubs(monkeypatch, calls)

    rc = cli_main.main([
        "-w", str(stub_workspace),
        "--scheduler", "local",
        "screen",
        "--rounds", "2",
        "--train-first",
    ])
    assert rc == 0
    names = [c[0] for c in calls]
    one_round = ["train", "simsearch", "simsearch", "simsearch", "dock"]
    assert names == one_round + one_round
