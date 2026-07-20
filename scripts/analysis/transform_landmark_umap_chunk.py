#!/usr/bin/env python3
"""Transform one deterministic fingerprint-index slice with a saved UMAP model."""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
from FPSim2 import FPSim2Engine
from run_landmark_umap import EXPECTED_FP_PARAMS, EXPECTED_FP_TYPE, _unpack_fingerprints
from tqdm import tqdm

LOGGER = logging.getLogger("transform_landmark_umap_chunk")


def transform_chunk(args: argparse.Namespace) -> Path:
    index_path = args.fp_index.resolve()
    model_path = args.model.resolve()
    chunks_dir = args.chunks_dir.resolve()
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    output = chunks_dir / f"chunk_{args.task_index:04d}_of_{args.task_count:04d}.npz"
    if output.exists() and not args.force:
        raise FileExistsError(f"chunk already exists; use --force: {output}")

    bundle = joblib.load(model_path)
    reducer = bundle["reducer"]
    reducer.verbose = False
    engine = FPSim2Engine(str(index_path))
    if engine.fp_type != EXPECTED_FP_TYPE or engine.fp_params != EXPECTED_FP_PARAMS:
        raise ValueError("fingerprint index does not use Morgan-2 1024-bit fingerprints")

    fps = engine.fps
    total = len(fps)
    start = total * (args.task_index - 1) // args.task_count
    stop = total * args.task_index // args.task_count
    coordinates = np.empty((stop - start, 2), dtype=np.float32)
    warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)

    with tqdm(
        total=stop - start,
        desc=f"UMAP chunk {args.task_index}/{args.task_count}",
        unit="mol",
        dynamic_ncols=True,
    ) as progress:
        for offset in range(start, stop, args.batch_size):
            batch_stop = min(offset + args.batch_size, stop)
            batch = _unpack_fingerprints(fps[offset:batch_stop, 1:-1])
            coordinates[offset - start : batch_stop - start] = reducer.transform(batch).astype(
                np.float32, copy=False
            )
            progress.update(batch_stop - offset)

    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            spacehastenid=np.asarray(fps[start:stop, 0], dtype=np.uint64),
            umap=coordinates,
            start=np.asarray(start),
            stop=np.asarray(stop),
        )
    temporary.replace(output)
    LOGGER.info("Saved %d coordinates to %s", stop - start, output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-index", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.task_count < 1:
        parser.error("task-count must be at least 1")
    if not 1 <= args.task_index <= args.task_count:
        parser.error("task-index must be between 1 and task-count")
    if args.batch_size < 1:
        parser.error("batch-size must be at least 1")
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        transform_chunk(parse_args())
    except Exception:
        LOGGER.exception("UMAP chunk transform failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
