#!/usr/bin/env python3
"""Fit landmark Jaccard UMAP and transform every indexed compound."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

LOGGER = logging.getLogger("run_landmark_umap")
EXPECTED_FP_TYPE = "Morgan"
EXPECTED_FP_PARAMS = {"radius": 2, "fpSize": 1024}


def _load_centroid_ids(path: Path) -> list[int]:
    centroids: set[int] = set()
    with path.open("rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["spacehastenid", "clusterid"]:
            raise ValueError(f"unexpected clustering columns: {reader.fieldnames}")
        for row in reader:
            centroids.add(int(row["clusterid"]))
    if not centroids:
        raise ValueError(f"no centroids found in {path}")
    return sorted(centroids)


def _unpack_fingerprints(words):  # type: ignore[no-untyped-def]
    import numpy as np

    packed = np.ascontiguousarray(words, dtype=np.uint64)
    byte_view = packed.view(np.uint8).reshape(len(packed), -1)
    return np.unpackbits(byte_view, axis=1, bitorder="little")


def _artifact_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / "landmark_umap_coordinates.npz",
        output_dir / "landmark_umap_model.joblib",
        output_dir / "landmark_umap_metadata.json",
    )


def run_umap(args: argparse.Namespace) -> dict[str, object]:
    import joblib
    import numpy as np
    import umap
    from FPSim2 import FPSim2Engine
    from pynndescent import NNDescent
    from tqdm import tqdm

    index_path = args.fp_index.resolve()
    clustering_path = args.clustering.resolve()
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if not clustering_path.is_file():
        raise FileNotFoundError(clustering_path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path, model_path, metadata_path = _artifact_paths(output_dir)
    existing = [path for path in (coordinates_path, model_path, metadata_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "refusing to overwrite existing UMAP artifacts; use --force: "
            + ", ".join(str(path) for path in existing)
        )
    for path in existing:
        path.unlink()

    started = time.monotonic()
    centroid_ids = np.asarray(_load_centroid_ids(clustering_path), dtype=np.uint64)
    LOGGER.info("Loading FPSim2 index %s", index_path)
    engine = FPSim2Engine(str(index_path))
    if engine.fp_type != EXPECTED_FP_TYPE or engine.fp_params != EXPECTED_FP_PARAMS:
        raise ValueError(
            f"index uses {engine.fp_type} {engine.fp_params}; expected "
            f"{EXPECTED_FP_TYPE} {EXPECTED_FP_PARAMS}"
        )

    fps = engine.fps
    compound_ids = np.asarray(fps[:, 0], dtype=np.uint64)
    fingerprint_words = fps[:, 1:-1]
    sorted_rows = np.argsort(compound_ids)
    sorted_ids = compound_ids[sorted_rows]
    centroid_positions = np.searchsorted(sorted_ids, centroid_ids)
    if np.any(centroid_positions == len(sorted_ids)) or not np.array_equal(
        sorted_ids[centroid_positions], centroid_ids
    ):
        raise ValueError("one or more cluster centroids are absent from the index")
    centroid_rows = sorted_rows[centroid_positions]
    centroid_fingerprints = _unpack_fingerprints(fingerprint_words[centroid_rows])

    n_neighbors = min(args.n_neighbors, len(centroid_ids) - 1)
    if n_neighbors < 2:
        raise ValueError("at least three centroids are required for UMAP")
    LOGGER.info(
        "Building Jaccard neighbor graph for %d centroids with %d workers",
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
    LOGGER.info("Centroids with no nonzero-overlap neighbor: %d", disconnected)

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
    LOGGER.info("Fitting landmark UMAP")
    centroid_coordinates = np.asarray(
        reducer.fit_transform(centroid_fingerprints), dtype=np.float32
    )
    if not np.isfinite(centroid_coordinates).all():
        raise RuntimeError("landmark UMAP produced non-finite coordinates")

    LOGGER.info("Transforming %d indexed compounds", len(compound_ids))
    coordinates = np.empty((len(compound_ids), 2), dtype=np.float32)
    with tqdm(
        total=len(compound_ids),
        desc="Transforming compounds",
        unit="mol",
        dynamic_ncols=True,
    ) as progress:
        for start in range(0, len(compound_ids), args.batch_size):
            stop = min(start + args.batch_size, len(compound_ids))
            batch = _unpack_fingerprints(fingerprint_words[start:stop])
            coordinates[start:stop] = reducer.transform(batch).astype(np.float32, copy=False)
            progress.update(stop - start)

    coordinates[centroid_rows] = centroid_coordinates
    if not np.isfinite(coordinates).all():
        raise RuntimeError("UMAP transform produced non-finite coordinates")

    output_order = np.argsort(compound_ids)
    coordinates_temp = coordinates_path.with_name(f".{coordinates_path.name}.tmp")
    with coordinates_temp.open("wb") as handle:
        np.savez_compressed(
            handle,
            spacehastenid=compound_ids[output_order],
            umap=coordinates[output_order],
        )
    coordinates_temp.replace(coordinates_path)

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

    elapsed_seconds = time.monotonic() - started
    metadata: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "inputs": {
            "fp_index": str(index_path),
            "clustering": str(clustering_path),
        },
        "outputs": {
            "coordinates": str(coordinates_path),
            "model": str(model_path),
        },
        "counts": {
            "compounds": int(len(compound_ids)),
            "landmark_centroids": int(len(centroid_ids)),
            "disconnected_landmarks": disconnected,
        },
        "fingerprint": {
            "type": EXPECTED_FP_TYPE,
            **EXPECTED_FP_PARAMS,
            "distance": "Jaccard",
        },
        "umap": {
            "n_neighbors": n_neighbors,
            "min_dist": args.min_dist,
            "random_seed": args.random_seed,
            "batch_size": args.batch_size,
            "processes": args.processes,
        },
        "software": {
            "umap-learn": version("umap-learn"),
            "pynndescent": version("pynndescent"),
            "FPSim2": version("FPSim2"),
        },
        "elapsed_seconds": elapsed_seconds,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Landmark UMAP complete in %.1f minutes", elapsed_seconds / 60.0)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit centroid-landmark Jaccard UMAP and transform all compounds."
    )
    parser.add_argument("--fp-index", type=Path, required=True)
    parser.add_argument("--clustering", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    if args.n_neighbors < 2:
        parser.error("--n-neighbors must be at least 2")
    if not 0.0 <= args.min_dist <= 1.0:
        parser.error("--min-dist must be between 0 and 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.processes < 1:
        parser.error("--processes must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        run_umap(args)
    except Exception:
        LOGGER.exception("Landmark UMAP failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
