#!/usr/bin/env python3
"""Fit and save a centroid-landmark Jaccard UMAP model."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import umap
from FPSim2 import FPSim2Engine
from pynndescent import NNDescent

from spacehasten.analysis import umap as landmark_umap

LOGGER = logging.getLogger("fit_landmark_umap")


def fit_model(args: argparse.Namespace) -> None:
    index_path = args.fp_index.resolve()
    clustering_path = args.clustering.resolve()
    output_dir = args.output_dir.resolve()
    model_path = output_dir / "landmark_umap_model.joblib"
    metadata_path = output_dir / "landmark_umap_fit_metadata.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if not clustering_path.is_file():
        raise FileNotFoundError(clustering_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in (model_path, metadata_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError("model already exists; use --force to replace it")
    for path in existing:
        path.unlink()

    started = time.monotonic()
    centroid_ids = np.asarray(landmark_umap.load_centroid_ids(clustering_path), dtype=np.uint64)
    engine = FPSim2Engine(str(index_path))
    if (
        engine.fp_type != landmark_umap.EXPECTED_FP_TYPE
        or engine.fp_params != landmark_umap.EXPECTED_FP_PARAMS
    ):
        raise ValueError("fingerprint index does not use Morgan-2 1024-bit fingerprints")

    fps = engine.fps
    compound_ids = np.asarray(fps[:, 0], dtype=np.uint64)
    sorted_rows = np.argsort(compound_ids)
    sorted_ids = compound_ids[sorted_rows]
    positions = np.searchsorted(sorted_ids, centroid_ids)
    if np.any(positions == len(sorted_ids)) or not np.array_equal(
        sorted_ids[positions], centroid_ids
    ):
        raise ValueError("one or more centroids are absent from the fingerprint index")
    centroid_rows = sorted_rows[positions]
    centroid_fingerprints = landmark_umap.unpack_fingerprints(fps[centroid_rows, 1:-1])

    n_neighbors = min(args.n_neighbors, len(centroid_ids) - 1)
    if n_neighbors < 2:
        raise ValueError("at least three centroids are required for UMAP")
    LOGGER.info(
        "Building Jaccard graph for %d landmarks with %d workers",
        len(centroid_ids),
        args.processes,
    )
    search_index = NNDescent(
        centroid_fingerprints,
        metric="jaccard",
        n_neighbors=n_neighbors,
        random_state=args.random_seed,
        n_jobs=args.processes,
        low_memory=True,
        verbose=True,
    )
    knn_indices, knn_distances = search_index.neighbor_graph
    disconnected = int(np.sum(np.all(knn_distances[:, 1:] >= 1.0, axis=1)))

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        metric="jaccard",
        min_dist=args.min_dist,
        random_state=args.random_seed,
        low_memory=True,
        n_jobs=args.processes,
        precomputed_knn=(knn_indices, knn_distances, search_index),
        verbose=True,
    )
    centroid_coordinates = np.asarray(
        reducer.fit_transform(centroid_fingerprints), dtype=np.float32
    )
    if not np.isfinite(centroid_coordinates).all():
        raise RuntimeError("landmark fit produced non-finite coordinates")

    model_temp = model_path.with_name(f".{model_path.name}.tmp")
    joblib.dump(
        {
            "reducer": reducer,
            "centroid_ids": centroid_ids,
            "centroid_coordinates": centroid_coordinates,
        },
        model_temp,
    )
    model_temp.replace(model_path)

    elapsed = time.monotonic() - started
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "inputs": {"fp_index": str(index_path), "clustering": str(clustering_path)},
        "model": str(model_path),
        "counts": {
            "compounds": int(len(compound_ids)),
            "landmark_centroids": int(len(centroid_ids)),
            "disconnected_landmarks": disconnected,
        },
        "fingerprint": {
            "type": landmark_umap.EXPECTED_FP_TYPE,
            **landmark_umap.EXPECTED_FP_PARAMS,
        },
        "umap": {
            "metric": "jaccard",
            "n_neighbors": n_neighbors,
            "min_dist": args.min_dist,
            "random_seed": args.random_seed,
            "processes": args.processes,
        },
        "software": {
            "umap-learn": version("umap-learn"),
            "pynndescent": version("pynndescent"),
            "FPSim2": version("FPSim2"),
        },
        "elapsed_seconds": elapsed,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "Saved model with %d landmarks (%d disconnected) in %.1f minutes",
        len(centroid_ids),
        disconnected,
        elapsed / 60.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-index", type=Path, required=True)
    parser.add_argument("--clustering", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.processes < 1:
        parser.error("processes must be at least 1")
    if args.n_neighbors < 2:
        parser.error("n-neighbors must be at least 2")
    if not 0 <= args.min_dist <= 1:
        parser.error("min-dist must be between 0 and 1")
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        fit_model(args)
    except Exception:
        LOGGER.exception("Landmark UMAP fit failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
