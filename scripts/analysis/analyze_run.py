#!/usr/bin/env python3
"""Run the reusable, read-only single-run analysis."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from spacehasten.analysis import AnalysisConfig, analyze_run, discover_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_or_db", metavar="RUN_OR_DB")
    parser.add_argument(
        "--database",
        type=Path,
        help=(
            "Optional immutable database snapshot. Run layout and acquisition files are still "
            "discovered from RUN_OR_DB."
        ),
    )
    parser.add_argument("--analysis-root", type=Path)
    parser.add_argument("--hit-threshold", type=float, required=True)
    parser.add_argument("--cutoff", type=float, action="append", default=[])
    parser.add_argument("--cutoff-range", nargs=3, type=float, metavar=("START", "STOP", "STEP"))
    parser.add_argument("--pair-samples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cutoffs = list(arguments.cutoff)
    if arguments.cutoff_range:
        start, stop, step = arguments.cutoff_range
        if step == 0 or (stop - start) * step < 0:
            parser.error("cutoff range step must move from start toward stop")
        while (step > 0 and start <= stop) or (step < 0 and start >= stop):
            cutoffs.append(start)
            start += step
    context = discover_run(arguments.run_or_db, database_path=arguments.database)
    root = arguments.analysis_root or (context.database_path.parent / "analysis")
    summary = analyze_run(
        context,
        AnalysisConfig(
            arguments.hit_threshold,
            tuple(cutoffs or [arguments.hit_threshold]),
            arguments.pair_samples,
            arguments.random_seed,
            arguments.dpi,
        ),
        root,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
