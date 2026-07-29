#!/usr/bin/env python3
"""Export a normalized selected-attempt manifest from a completed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spacehasten.analysis.discovery import discover_run
from spacehasten.analysis.selected import selected_manifest, write_selected_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_or_db", metavar="RUN_OR_DB")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hit-threshold", type=float, required=True)
    parser.add_argument("--strict-threshold", type=float, default=-11.0)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists() and not arguments.overwrite:
        parser.error(f"output exists; pass --overwrite to replace it: {arguments.output}")
    context = discover_run(arguments.run_or_db, database_path=arguments.database)
    rows = selected_manifest(
        context,
        hit_threshold=arguments.hit_threshold,
        strict_threshold=arguments.strict_threshold,
    )
    write_selected_manifest(arguments.output, rows)
    sources = sorted({str(row.get("selection_source", "unknown")) for row in rows})
    summary = {
        "output": str(arguments.output.resolve()),
        "selected_attempts": len(rows),
        "selected_unique_compounds": len({int(row["spacehastenid"]) for row in rows}),
        "scored": sum(bool(row["is_scored"]) for row in rows),
        "hits": sum(bool(row["is_hit"]) for row in rows),
        "strict_hits": sum(bool(row["is_strict_hit"]) for row in rows),
        "source": (
            sources[0]
            if len(sources) == 1
            else "none"
            if not sources
            else "mixed:" + ",".join(sources)
        ),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
