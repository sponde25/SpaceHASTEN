#!/usr/bin/env python3
"""Create and validate a read-consistent SQLite analysis snapshot."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

LOGGER = logging.getLogger("snapshot_sqlite_database")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages-per-step", type=int, default=16_384)
    args = parser.parse_args()
    if args.pages_per_step < 1:
        parser.error("--pages-per-step must be positive")
    return args


def database_counts(connection: sqlite3.Connection) -> dict[str, object]:
    tables = {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }

    def count(table: str) -> int | None:
        if table not in tables:
            return None
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    dock_iterations = (
        {
            str(row[0]) if row[0] is not None else "null": int(row[1])
            for row in connection.execute(
                "SELECT dock_iteration,COUNT(*) FROM data GROUP BY dock_iteration"
            )
        }
        if "data" in tables
        else {}
    )
    scored_iterations = (
        {
            str(row[0]): int(row[1])
            for row in connection.execute(
                "SELECT dock_iteration,COUNT(*) FROM data "
                "WHERE dock_iteration IS NOT NULL AND dock_score IS NOT NULL "
                "GROUP BY dock_iteration"
            )
        }
        if "data" in tables
        else {}
    )
    return {
        "data": count("data"),
        "predictions": count("predictions"),
        "models": count("models"),
        "cluster_atlas_assignments": count("cluster_atlas_assignments"),
        "cluster_atlas_versions": count("cluster_atlas_versions"),
        "dock_iterations": dock_iterations,
        "scored_iterations": scored_iterations,
    }


def snapshot(source_path: Path, output_path: Path, pages_per_step: int) -> dict[str, object]:
    started = time.monotonic()
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.unlink(missing_ok=True)

    with sqlite3.connect(f"file:{source_path}?mode=ro", uri=True) as source:
        source.execute("PRAGMA query_only=ON")
        source.execute("BEGIN")
        source_quick_check = str(source.execute("PRAGMA quick_check").fetchone()[0])
        if source_quick_check != "ok":
            raise ValueError(f"source quick_check failed: {source_quick_check}")
        page_count = int(source.execute("PRAGMA page_count").fetchone()[0])
        page_size = int(source.execute("PRAGMA page_size").fetchone()[0])
        expected_bytes = page_count * page_size
        free_bytes = shutil.disk_usage(output_path.parent).free
        if free_bytes < int(expected_bytes * 1.1):
            raise OSError(
                f"insufficient free space: need about {expected_bytes:,} bytes, have {free_bytes:,}"
            )
        source_counts = database_counts(source)

        target = sqlite3.connect(temporary)
        progress = tqdm(total=page_count, desc="SQLite online backup", unit="page")
        previous_remaining = page_count

        def report(_status: int, remaining: int, total: int) -> None:
            nonlocal previous_remaining
            completed = previous_remaining - remaining
            if completed > 0:
                progress.update(completed)
            previous_remaining = remaining
            if progress.total != total:
                progress.total = total
                progress.refresh()

        try:
            source.backup(
                target,
                pages=pages_per_step,
                progress=report,
                sleep=0.05,
            )
            target.commit()
        finally:
            progress.close()
            target.close()

    with sqlite3.connect(temporary) as copied:
        copied_quick_check = str(copied.execute("PRAGMA quick_check").fetchone()[0])
        copied_counts = database_counts(copied)
    if copied_quick_check != "ok" or copied_counts != source_counts:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            "snapshot validation failed: "
            f"quick_check={copied_quick_check}, counts_match={copied_counts == source_counts}"
        )
    temporary.replace(output_path)

    metadata: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": str(source_path),
        "output": str(output_path),
        "method": "sqlite3.Connection.backup",
        "source_quick_check": source_quick_check,
        "snapshot_quick_check": copied_quick_check,
        "page_count": page_count,
        "page_size": page_size,
        "size_bytes": output_path.stat().st_size,
        "counts": copied_counts,
        "elapsed_seconds": time.monotonic() - started,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    print(json.dumps(snapshot(args.source, args.output, args.pages_per_step), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
