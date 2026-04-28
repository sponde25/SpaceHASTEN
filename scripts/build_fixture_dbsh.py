"""Build tests/fixtures/legacy_baseline.dbsh.

Standalone: imports only the stdlib (sqlite3, pathlib). Reproduces the legacy
SQLite schema byte-for-byte (see tests/fixtures/legacy_schema.sql) and inserts
five hand-chosen rows plus the singleton blob/properties/clusters/models rows
described in docs/SESSIONS.md (Session 2).

Run from the repo root:

    python3 scripts/build_fixture_dbsh.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
SCHEMA_SQL = FIXTURE_DIR / "legacy_schema.sql"
DBSH_PATH = FIXTURE_DIR / "legacy_baseline.dbsh"
EXAMPLES_SMI = REPO_ROOT / "examples.smi"


# Synthetic deterministic reghash placeholders. The legacy `mol2hash` produces
# RDKit RegistrationHash strings; for fixture purposes any stable distinct
# string is sufficient — Session 4 covers real hash equality.
#
# Columns: reghash, smiles, smilesid, dock_score, pred_score, spacelight,
#          ftrees, query, dock_iteration, pred_version, simsearch_cycle.
DataRow = tuple[
    str, str, str,
    float | None, float | None, float | None, float | None,
    int | None, int | None, int | None, int | None,
]
ROWS: list[DataRow] = [
    # 1: docked seed (ethanol).
    ("hash_ethanol_seed", "CCO", "ETHANOL",
     -7.5, None, None, None, None, 0, None, None),
    # 2: docked seed (benzene).
    ("hash_benzene_seed", "c1ccccc1", "BENZENE",
     -8.2, None, None, None, None, 0, None, None),
    # 3: simsearch hit with pred_score only (toluene).
    ("hash_toluene_pred", "Cc1ccccc1", "TOLUENE",
     None, -6.7, None, None, None, None, 1, 1),
    # 4: simsearch hit with spacelight + ftrees + pred_score (phenol).
    ("hash_phenol_full", "Oc1ccccc1", "PHENOL",
     None, -7.1, 0.85, 0.72, None, None, 1, 1),
    # 5: query (aniline).
    ("hash_aniline_query", "Nc1ccccc1", "ANILINE",
     None, None, None, None, 1, None, None, 1),
]


def _read_schema_statements(path: Path) -> list[str]:
    text = path.read_text()
    # Strip SQL line comments and split on semicolons. Keep statement order.
    cleaned = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("--")
    )
    return [s.strip() for s in cleaned.split(";") if s.strip()]


def build(dbsh_path: Path = DBSH_PATH) -> None:
    if dbsh_path.exists():
        dbsh_path.unlink()
    dbsh_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(dbsh_path)
    try:
        c = conn.cursor()

        # 1. Create schema using the frozen reference text.
        for stmt in _read_schema_statements(SCHEMA_SQL):
            c.execute(stmt)

        # 2. data: 5 hand-chosen rows.
        c.executemany(
            "INSERT INTO data("
            "reghash,smiles,smilesid,dock_score,pred_score,"
            "spacelight,ftrees,query,dock_iteration,pred_version,simsearch_cycle"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ROWS,
        )

        # 3. docking_param: blob from examples.smi (just to verify roundtrip).
        param_blob = EXAMPLES_SMI.read_bytes() if EXAMPLES_SMI.exists() else b"placeholder"
        c.execute("INSERT INTO docking_param VALUES (?)", [memoryview(param_blob)])

        # 4. docking_grid: minimum legal empty zip end-of-central-dir record.
        empty_zip_eocd = b"PK\x05\x06" + b"\x00" * 18
        c.execute("INSERT INTO docking_grid VALUES (?)", [memoryview(empty_zip_eocd)])

        # 5. models: one row.
        c.execute("INSERT INTO models VALUES (?,?)", (1, memoryview(b"dummytar")))

        # 6. properties: six default rows (cfg.py defaults).
        properties = [
            ("mw",       1, "0.0",  "500.0"),
            ("slogp",    1, "-10.0", "5.0"),
            ("hba",      0, "0",    "10"),
            ("hbd",      0, "0",    "5"),
            ("rotbonds", 0, "0",    "10"),
            ("tpsa",     1, "0.0",  "140.0"),
        ]
        c.executemany(
            "INSERT INTO properties (property,is_double,min_limit,max_limit) VALUES (?,?,?,?)",
            properties,
        )

        # 7. clusters: one row per data row, single cluster id.
        c.executemany(
            "INSERT INTO clusters (spacehastenid,clusterid) VALUES (?,?)",
            [(i, 1) for i in range(1, len(ROWS) + 1)],
        )

        conn.commit()
    finally:
        conn.close()
    print(f"Wrote {dbsh_path}")


if __name__ == "__main__":
    build()
