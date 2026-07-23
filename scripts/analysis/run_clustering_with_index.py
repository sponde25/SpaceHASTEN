#!/usr/bin/env python3
"""Run SpaceHASTEN clustering while reusing an existing FPSim2 index."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from spacehasten.remote import cluster as cluster_module

LOGGER = logging.getLogger("run_clustering_with_index")
EXPECTED_FP_TYPE = "Morgan"
EXPECTED_FP_PARAMS = {"radius": 2, "fpSize": 1024}


def run_with_existing_index(
    input_path: Path,
    index_path: Path,
    output_path: Path,
    *,
    processes: int,
    similarity_threshold: float,
) -> int:
    import tables

    input_path = input_path.resolve()
    index_path = index_path.resolve()
    output_path = output_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    working_index = output_path.parent / "fp.h5"
    if index_path == working_index:
        raise ValueError("--fp-index cannot be the clustering work file fp.h5")

    def reuse_index(rows: list[tuple[str, int]], destination: Path) -> None:
        import numpy as np

        with tables.open_file(index_path, mode="r") as handle:
            row_count = int(handle.root.fps.nrows)
            fp_type = str(handle.root.config[0])
            fp_params = dict(handle.root.config[1])
            index_ids = np.asarray(handle.root.fps.col("fp_id"), dtype=np.int64)
        if row_count != len(rows):
            raise ValueError(f"index contains {row_count} rows but input contains {len(rows)}")
        if fp_type != EXPECTED_FP_TYPE or fp_params != EXPECTED_FP_PARAMS:
            raise ValueError(
                f"index uses {fp_type} {fp_params}; expected "
                f"{EXPECTED_FP_TYPE} {EXPECTED_FP_PARAMS}"
            )
        input_ids = np.fromiter((identifier for _, identifier in rows), dtype=np.int64)
        if len(np.unique(input_ids)) != len(input_ids):
            raise ValueError("input contains duplicate compound identifiers")
        if not np.array_equal(np.sort(index_ids), np.sort(input_ids)):
            raise ValueError("index identifiers do not match the clustering input")
        destination.symlink_to(index_path)
        LOGGER.info("Reusing validated FPSim2 index %s", index_path)

    original_builder = cluster_module._build_fpsim2_index
    try:
        cluster_module._build_fpsim2_index = reuse_index
        return cluster_module.run_clustering(
            input_path,
            output_path,
            processes=processes,
            similarity_threshold=similarity_threshold,
        )
    finally:
        cluster_module._build_fpsim2_index = original_builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run seed-first sphere-exclusion clustering with an existing index."
    )
    parser.add_argument("input", type=Path, help="ordered .smi or .smi.gz input")
    parser.add_argument("--fp-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--similarity-threshold", type=float, default=0.3)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    if args.processes < 1:
        parser.error("--processes must be at least 1")
    if not 0.0 < args.similarity_threshold <= 1.0:
        parser.error("--similarity-threshold must be in (0, 1]")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        cluster_count = run_with_existing_index(
            args.input,
            args.fp_index,
            args.output,
            processes=args.processes,
            similarity_threshold=args.similarity_threshold,
        )
    except Exception:
        LOGGER.exception("Clustering failed")
        return 1
    LOGGER.info("Clustering complete: %d centroids", cluster_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
