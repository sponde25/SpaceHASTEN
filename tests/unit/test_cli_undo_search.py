"""CLI-level tests for ``spacehasten undo search``.

Builds a minimal workspace (just a ``<name>.dbsh`` file — no full
``WorkDir.bootstrap``/manifest needed, since ``undo search`` never touches
the scheduler or shared root) and drives the command through
:func:`spacehasten.cli.main.main` to exercise argument parsing, the
confirmation gate, and the DB update end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacehasten.cli.main import main
from spacehasten.core.db import Database


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    with Database(root / "ws.dbsh") as db:
        db.create_schema()
    return root


def _seed_stuck_query(root: Path) -> int:
    """Simulate a failed cycle 1: a query marked and committed, no hits."""
    with Database(root / "ws.dbsh") as db:
        seed_id = db.insert_seed_docked("h1", "CCO", "SEED1", dock_score=-7.0)
        db.mark_as_query(seed_id, cycle=1)
        db.commit()
    return seed_id


def test_undo_search_nothing_to_undo(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["--quiet", "-w", str(workspace), "undo", "search", "--yes"])
    assert rc == 0
    assert "nothing to undo" in capsys.readouterr().out.lower()


def test_undo_search_refuses_non_interactive_without_yes(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_stuck_query(workspace)
    rc = main(["--quiet", "-w", str(workspace), "undo", "search"])
    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "--yes" in err

    # Nothing was changed.
    with Database(workspace / "ws.dbsh") as db:
        (q,) = db.connection.execute("SELECT query FROM data").fetchone()
        assert q == 1


def test_undo_search_yes_releases_stuck_query(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    seed_id = _seed_stuck_query(workspace)
    rc = main(["--quiet", "-w", str(workspace), "undo", "search", "--yes"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Reverted cycle 1" in out
    assert "removed 0 hit compound(s)" in out
    assert "released 1 query mark(s)" in out

    with Database(workspace / "ws.dbsh") as db:
        (q,) = db.connection.execute(
            "SELECT query FROM data WHERE spacehastenid = ?", (seed_id,)
        ).fetchone()
        assert q is None
