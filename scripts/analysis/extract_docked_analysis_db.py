#!/usr/bin/env python3
"""Create a compact immutable analysis DB containing only scored docked rows."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

LOGGER = logging.getLogger("extract_docked_analysis_db")
BATCH_SIZE = 10_000
COLUMNS = (
    "spacehastenid",
    "reghash",
    "smiles",
    "smilesid",
    "dock_score",
    "pred_score",
    "spacelight",
    "ftrees",
    "query",
    "dock_iteration",
    "pred_version",
    "simsearch_cycle",
)
SCHEMA = (
    "CREATE TABLE data("
    "spacehastenid INTEGER PRIMARY KEY,"
    "reghash TEXT,smiles TEXT,smilesid TEXT,dock_score REAL,pred_score REAL,"
    "spacelight REAL,ftrees REAL,query INTEGER,dock_iteration INTEGER,"
    "pred_version INTEGER,simsearch_cycle INTEGER)"
)


def readonly_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, required=True)
    parser.add_argument("--expected-virtual", type=int, required=True)
    parser.add_argument("--expected-hits", type=int, required=True)
    parser.add_argument("--cutoff", type=float, default=-9.7)
    return parser.parse_args()


def extract(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    source_path = args.source.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.unlink(missing_ok=True)

    with readonly_connection(source_path) as source:
        source_columns = {str(row[1]) for row in source.execute("PRAGMA table_info(data)")}
        missing = set(COLUMNS) - source_columns
        if missing:
            raise ValueError(f"source data table lacks columns: {sorted(missing)}")
        expected_total = int(
            source.execute(
                "SELECT COUNT(*) FROM data WHERE dock_score IS NOT NULL "
                "AND dock_iteration IS NOT NULL"
            ).fetchone()[0]
        )
        expected = args.expected_seeds + args.expected_virtual
        if expected_total != expected:
            raise ValueError(f"source has {expected_total} scored docked rows; expected {expected}")

        target = sqlite3.connect(temporary)
        try:
            target.execute("PRAGMA journal_mode=OFF")
            target.execute("PRAGMA synchronous=OFF")
            target.execute("PRAGMA temp_store=MEMORY")
            target.execute(SCHEMA)
            placeholders = ",".join("?" for _ in COLUMNS)
            insert = f"INSERT INTO data({','.join(COLUMNS)}) VALUES ({placeholders})"
            query = (
                f"SELECT {','.join(COLUMNS)} FROM data "
                "WHERE dock_score IS NOT NULL AND dock_iteration IS NOT NULL "
                "ORDER BY spacehastenid"
            )
            cursor = source.execute(query)
            copied = 0
            with tqdm(total=expected_total, desc=source_path.stem, unit="mol") as progress:
                while True:
                    batch = cursor.fetchmany(BATCH_SIZE)
                    if not batch:
                        break
                    target.executemany(insert, batch)
                    copied += len(batch)
                    progress.update(len(batch))
            if copied != expected_total:
                raise ValueError(f"copied {copied} rows; expected {expected_total}")
            target.execute("CREATE INDEX idx_reghash ON data(reghash)")
            target.execute("CREATE INDEX idx_dock_iteration ON data(dock_iteration)")
            target.commit()

            validation = {
                "quick_check": target.execute("PRAGMA quick_check").fetchone()[0],
                "rows": int(target.execute("SELECT COUNT(*) FROM data").fetchone()[0]),
                "seeds": int(
                    target.execute("SELECT COUNT(*) FROM data WHERE dock_iteration = 0").fetchone()[
                        0
                    ]
                ),
                "virtual": int(
                    target.execute("SELECT COUNT(*) FROM data WHERE dock_iteration > 0").fetchone()[
                        0
                    ]
                ),
                "hits": int(
                    target.execute(
                        "SELECT COUNT(*) FROM data WHERE dock_iteration > 0 AND dock_score <= ?",
                        (args.cutoff,),
                    ).fetchone()[0]
                ),
                "missing_reghashes": int(
                    target.execute(
                        "SELECT COUNT(*) FROM data WHERE reghash IS NULL OR TRIM(reghash) = ''"
                    ).fetchone()[0]
                ),
                "duplicate_reghashes": int(
                    target.execute(
                        "SELECT COUNT(*) - COUNT(DISTINCT reghash) FROM data"
                    ).fetchone()[0]
                ),
            }
        finally:
            target.close()

    expected_validation = {
        "quick_check": "ok",
        "rows": args.expected_seeds + args.expected_virtual,
        "seeds": args.expected_seeds,
        "virtual": args.expected_virtual,
        "hits": args.expected_hits,
        "missing_reghashes": 0,
        "duplicate_reghashes": 0,
    }
    if validation != expected_validation:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"analysis DB validation failed: {validation}")
    temporary.replace(output_path)
    metadata: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": str(source_path),
        "output": str(output_path),
        "filter": "dock_score IS NOT NULL AND dock_iteration IS NOT NULL",
        "cutoff": args.cutoff,
        "validation": validation,
        "elapsed_seconds": time.monotonic() - started,
        "size_bytes": output_path.stat().st_size,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(extract(parse_args()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
