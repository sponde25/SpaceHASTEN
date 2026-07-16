#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

EXPECTED_COLUMNS = {
    "spacehastenid",
    "model_version",
    "pred_score",
    "epistemic_std",
    "aleatoric_std",
    "total_std",
    "created_at",
}


def scalar(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return int(row[0]) if row else 0


def verify_database(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        if columns != EXPECTED_COLUMNS:
            raise SystemExit(f"unexpected predictions schema: {sorted(columns)}")

        rows = conn.execute(
            "SELECT spacehastenid, model_version, pred_score, epistemic_std, "
            "aleatoric_std, total_std FROM predictions"
        ).fetchall()
        if not rows:
            raise SystemExit("predictions table is empty")
        for sid, version, score, epistemic, aleatoric, total in rows:
            values = (score, epistemic, aleatoric, total)
            if not all(value is not None and math.isfinite(value) for value in values):
                raise SystemExit(f"invalid prediction values for id={sid}, model={version}")
            if min(epistemic, aleatoric, total) < 0:
                raise SystemExit(f"negative uncertainty for id={sid}, model={version}")
            if not math.isclose(total**2, epistemic**2 + aleatoric**2, rel_tol=1e-5):
                raise SystemExit(f"uncertainty decomposition mismatch for id={sid}")

        checks = {
            "latest prediction cache mismatches": (
                "SELECT COUNT(*) FROM data AS d JOIN predictions AS p "
                "ON p.spacehastenid = d.spacehastenid "
                "AND p.model_version = d.pred_version "
                "WHERE ABS(p.pred_score - d.pred_score) > 1e-10"
            ),
            "rows missing latest prediction history": (
                "SELECT COUNT(*) FROM data AS d WHERE d.pred_version IS NOT NULL "
                "AND NOT EXISTS (SELECT 1 FROM predictions AS p "
                "WHERE p.spacehastenid = d.spacehastenid "
                "AND p.model_version = d.pred_version)"
            ),
            "orphan prediction compounds": (
                "SELECT COUNT(*) FROM predictions AS p "
                "LEFT JOIN data AS d ON d.spacehastenid = p.spacehastenid "
                "WHERE d.spacehastenid IS NULL"
            ),
            "orphan prediction model versions": (
                "SELECT COUNT(*) FROM predictions AS p "
                "LEFT JOIN models AS m ON m.model_version = p.model_version "
                "WHERE m.model_version IS NULL"
            ),
        }
        for description, sql in checks.items():
            count = scalar(conn, sql)
            if count:
                raise SystemExit(f"{count} {description}")

        print(f"Verified uncertainty database: {path}")
        print(
            "Compounds: total={} docked={} predicted={}".format(
                scalar(conn, "SELECT COUNT(*) FROM data"),
                scalar(conn, "SELECT COUNT(*) FROM data WHERE dock_score IS NOT NULL"),
                scalar(conn, "SELECT COUNT(*) FROM data WHERE pred_version IS NOT NULL"),
            )
        )
        print(
            "Models={} simsearch_cycles={} docking_iterations={}".format(
                scalar(conn, "SELECT COUNT(*) FROM models"),
                scalar(conn, "SELECT COUNT(DISTINCT simsearch_cycle) FROM data"),
                scalar(conn, "SELECT COUNT(DISTINCT dock_iteration) FROM data"),
            )
        )
        for version, count in conn.execute(
            "SELECT model_version, COUNT(*) FROM predictions "
            "GROUP BY model_version ORDER BY model_version"
        ):
            print(f"Prediction history: model={version} rows={count}")

        uncertainty = conn.execute(
            "SELECT MIN(epistemic_std), AVG(epistemic_std), MAX(epistemic_std), "
            "MIN(aleatoric_std), AVG(aleatoric_std), MAX(aleatoric_std), "
            "MIN(total_std), AVG(total_std), MAX(total_std) FROM predictions"
        ).fetchone()
        print(
            "Uncertainty min/mean/max: epistemic={:.6g}/{:.6g}/{:.6g} "
            "aleatoric={:.6g}/{:.6g}/{:.6g} total={:.6g}/{:.6g}/{:.6g}".format(
                *uncertainty
            )
        )
        print(f"All {len(rows)} versioned predictions passed consistency checks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify persisted SpaceHASTEN uncertainty.")
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    if not args.database.exists():
        raise FileNotFoundError(args.database)
    verify_database(args.database)


if __name__ == "__main__":
    main()
