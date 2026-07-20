#!/usr/bin/env python3
"""Combine and validate independent landmark-UMAP transform chunks."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm

LOGGER = logging.getLogger("combine_landmark_umap_chunks")


def combine(args: argparse.Namespace) -> None:
    chunks_dir = args.chunks_dir.resolve()
    model_path = args.model.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinates_path = output_dir / "landmark_umap_coordinates.npz"
    metadata_path = output_dir / "landmark_umap_metadata.json"
    if (coordinates_path.exists() or metadata_path.exists()) and not args.force:
        raise FileExistsError("combined output already exists; use --force to replace it")

    identifiers: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    expected_start = 0
    with tqdm(total=args.task_count, desc="Combining UMAP chunks", unit="chunk") as bar:
        for task_index in range(1, args.task_count + 1):
            path = chunks_dir / f"chunk_{task_index:04d}_of_{args.task_count:04d}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path) as chunk:
                start = int(chunk["start"])
                stop = int(chunk["stop"])
                if start != expected_start or stop - start != len(chunk["spacehastenid"]):
                    raise ValueError(f"non-contiguous or invalid chunk: {path}")
                chunk_ids = chunk["spacehastenid"].copy()
                chunk_coordinates = chunk["umap"].copy()
                if chunk_coordinates.shape != (len(chunk_ids), 2):
                    raise ValueError(f"invalid coordinate shape in chunk: {path}")
                identifiers.append(chunk_ids)
                coordinates.append(chunk_coordinates)
                expected_start = stop
            bar.update(1)

    if expected_start != args.expected_count:
        raise ValueError(f"chunks contain {expected_start} rows; expected {args.expected_count}")

    compound_ids = np.concatenate(identifiers)
    embedding = np.concatenate(coordinates)
    if len(np.unique(compound_ids)) != len(compound_ids):
        raise ValueError("duplicate compound identifiers found across chunks")
    if not np.isfinite(embedding).all():
        raise ValueError("non-finite UMAP coordinates found in chunks")

    bundle = joblib.load(model_path)
    centroid_ids = np.asarray(bundle["centroid_ids"], dtype=np.uint64)
    centroid_coordinates = np.asarray(bundle["centroid_coordinates"], dtype=np.float32)
    if centroid_coordinates.shape != (len(centroid_ids), 2):
        raise ValueError("model centroid identifiers and coordinates are inconsistent")
    order = np.argsort(compound_ids)
    sorted_ids = compound_ids[order]
    positions = np.searchsorted(sorted_ids, centroid_ids)
    if np.any(positions == len(sorted_ids)) or not np.array_equal(
        sorted_ids[positions], centroid_ids
    ):
        raise ValueError("centroid IDs are missing from transformed chunks")
    embedding[order[positions]] = centroid_coordinates

    temporary = coordinates_path.with_name(f".{coordinates_path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            spacehastenid=sorted_ids,
            umap=embedding[order],
        )
    temporary.replace(coordinates_path)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "model": str(model_path),
        "chunks_directory": str(chunks_dir),
        "coordinates": str(coordinates_path),
        "task_count": args.task_count,
        "compound_count": int(len(compound_ids)),
        "landmark_count": int(len(centroid_ids)),
        "coordinate_min": embedding.min(axis=0).tolist(),
        "coordinate_max": embedding.max(axis=0).tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Combined %d coordinates into %s", len(compound_ids), coordinates_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if min(args.task_count, args.expected_count) < 1:
        parser.error("task-count and expected-count must be positive")
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        combine(parse_args())
    except Exception:
        LOGGER.exception("Combining UMAP chunks failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
