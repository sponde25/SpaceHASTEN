"""Lock-in tests for tests/fixtures/legacy_baseline.dbsh.

Verifies the fixture matches the frozen schema reference (tests/fixtures/legacy_schema.sql)
and contains exactly the 5 data rows + singleton support rows specified by
docs/SESSIONS.md Session 2.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"
DBSH = FIXTURE_DIR / "legacy_baseline.dbsh"
SCHEMA_SQL = FIXTURE_DIR / "legacy_schema.sql"


EXPECTED_TABLES = {"data", "docking_param", "docking_grid", "models", "properties", "clusters"}
EXPECTED_INDEXES = {"idx_reghash"}


# Parsed once from legacy_schema.sql at collection time so that any drift in the
# frozen schema is caught immediately.
def _parse_schema_columns() -> dict[str, list[tuple[str, str]]]:
    """Return {table_name: [(col_name, col_type), ...]} from legacy_schema.sql."""
    text = SCHEMA_SQL.read_text()
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("--"))
    out: dict[str, list[tuple[str, str]]] = {}
    for stmt in (s.strip() for s in text.split(";") if s.strip()):
        m = re.match(r"CREATE\s+TABLE\s+(\w+)\s*\((.*)\)$", stmt, re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        name, body = m.group(1), m.group(2)
        cols: list[tuple[str, str]] = []
        # Split top-level commas (no nested parens in our schema).
        for raw in body.split(","):
            parts = raw.strip().split(None, 1)
            if not parts:
                continue
            col_name = parts[0]
            col_type = ""
            if len(parts) > 1:
                # Strip "PRIMARY KEY", "UNIQUE", etc. for type comparison.
                rest = parts[1].upper()
                col_type = rest.split()[0]
            cols.append((col_name, col_type))
        out[name] = cols
    return out


SCHEMA_COLUMNS = _parse_schema_columns()


@pytest.fixture(scope="module")
def conn() -> sqlite3.Connection:
    assert DBSH.exists(), f"Fixture missing: {DBSH}. Run scripts/build_fixture_dbsh.py."
    c = sqlite3.connect(DBSH)
    yield c
    c.close()


def test_sqlite_master_tables_and_index(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT type, name FROM sqlite_master ORDER BY type, name").fetchall()
    tables = {name for typ, name in rows if typ == "table" and not name.startswith("sqlite_")}
    indexes = {name for typ, name in rows if typ == "index" and not name.startswith("sqlite_")}
    assert tables == EXPECTED_TABLES
    assert indexes == EXPECTED_INDEXES


@pytest.mark.parametrize("table", sorted(EXPECTED_TABLES))
def test_table_columns_match_schema(conn: sqlite3.Connection, table: str) -> None:
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    actual = [(row[1], row[2].upper().split()[0] if row[2] else "") for row in info]
    expected = SCHEMA_COLUMNS[table]
    assert actual == expected, f"{table}: actual={actual} expected={expected}"


def test_data_invariants(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT spacehastenid, smilesid, dock_score, pred_score, spacelight, ftrees, "
        "query, dock_iteration, pred_version, simsearch_cycle "
        "FROM data ORDER BY spacehastenid"
    ).fetchall()
    assert len(rows) == 5

    # Row 1: ethanol — docked seed.
    assert rows[0] == (1, "ETHANOL", -7.5, None, None, None, None, 0, None, None)
    # Row 2: benzene — docked seed.
    assert rows[1] == (2, "BENZENE", -8.2, None, None, None, None, 0, None, None)
    # Row 3: toluene — simsearch hit with pred_score only.
    assert rows[2] == (3, "TOLUENE", None, -6.7, None, None, None, None, 1, 1)
    # Row 4: phenol — simsearch hit with spacelight + ftrees + pred_score.
    assert rows[3] == (4, "PHENOL", None, -7.1, 0.85, 0.72, None, None, 1, 1)
    # Row 5: aniline — query.
    assert rows[4] == (5, "ANILINE", None, None, None, None, 1, None, None, 1)


def test_singleton_support_rows(conn: sqlite3.Connection) -> None:
    (param_count,) = conn.execute("SELECT COUNT(*) FROM docking_param").fetchone()
    (grid_count,) = conn.execute("SELECT COUNT(*) FROM docking_grid").fetchone()
    (model_count,) = conn.execute("SELECT COUNT(*) FROM models").fetchone()
    assert (param_count, grid_count, model_count) == (1, 1, 1)

    (param_blob,) = conn.execute("SELECT dock_param FROM docking_param").fetchone()
    assert isinstance(param_blob, bytes) and len(param_blob) > 0

    (grid_blob,) = conn.execute("SELECT dock_grid FROM docking_grid").fetchone()
    assert grid_blob == b"PK\x05\x06" + b"\x00" * 18

    model_row = conn.execute("SELECT model_version, model_tar FROM models").fetchone()
    assert model_row == (1, b"dummytar")


def test_properties_default_six_rows(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT property, is_double, min_limit, max_limit FROM properties "
        "ORDER BY rowid"
    ).fetchall()
    assert rows == [
        ("mw", 1, "0.0", "500.0"),
        ("slogp", 1, "-10.0", "5.0"),
        ("hba", 0, "0", "10"),
        ("hbd", 0, "0", "5"),
        ("rotbonds", 0, "0", "10"),
        ("tpsa", 1, "0.0", "140.0"),
    ]


def test_clusters_one_per_data_row(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT spacehastenid, clusterid FROM clusters ORDER BY spacehastenid"
    ).fetchall()
    assert rows == [(i, 1) for i in range(1, 6)]
