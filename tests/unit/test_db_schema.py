"""Schema roundtrip: ``Database.create_schema()`` → reference fixture."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from spacehasten.core.db import Database

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
LEGACY_SCHEMA = FIXTURES / "legacy_schema.sql"
LEGACY_BASELINE = FIXTURES / "legacy_baseline.dbsh"


def _sqlite_master(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    return [
        (t, n, _normalise(sql))
        for t, n, sql in conn.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            " ORDER BY type, name"
        ).fetchall()
    ]


def _normalise(sql: str | None) -> str:
    s = " ".join((sql or "").split())
    # Strip IF NOT EXISTS so we can compare against the frozen legacy schema.
    return s.replace(" IF NOT EXISTS", "")


def _column_info(conn: sqlite3.Connection, table: str) -> list[tuple]:
    return conn.execute(f"PRAGMA table_info({table})").fetchall()


def test_create_schema_matches_legacy_baseline(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh.dbsh"
    with Database(fresh) as db:
        db.create_schema()

    fresh_conn = sqlite3.connect(fresh)
    legacy_conn = sqlite3.connect(LEGACY_BASELINE)
    try:
        # sqlite_master entries (type, name, normalised SQL).
        assert _sqlite_master(fresh_conn) == _sqlite_master(legacy_conn)
        # Per-table column info matches column-for-column.
        for table in ("data", "docking_param", "docking_grid", "models", "clusters"):
            assert _column_info(fresh_conn, table) == _column_info(legacy_conn, table)
    finally:
        fresh_conn.close()
        legacy_conn.close()


def test_schema_statements_match_legacy_schema_sql() -> None:
    """The hard-coded SCHEMA_STATEMENTS must agree (modulo whitespace) with
    the frozen ``legacy_schema.sql`` reference."""
    from spacehasten.core.db import SCHEMA_STATEMENTS

    text = LEGACY_SCHEMA.read_text()
    fixture_stmts = [
        s.strip()
        for s in "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("--")
        ).split(";")
        if s.strip()
    ]
    assert [_normalise(s) for s in SCHEMA_STATEMENTS] == [
        _normalise(s) for s in fixture_stmts
    ]
