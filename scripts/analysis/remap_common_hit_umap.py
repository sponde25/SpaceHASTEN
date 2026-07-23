#!/usr/bin/env python3
"""Remap one common-hit UMAP projection to each run's SpaceHASTEN IDs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np


def write_coordinates(
    path: Path,
    identifiers: list[int],
    coordinates: list[np.ndarray],
) -> None:
    ids = np.asarray(identifiers, dtype=np.uint64)
    xy = np.asarray(coordinates, dtype=np.float32)
    if len(ids) != len(np.unique(ids)) or xy.shape != (len(ids), 2):
        raise ValueError(f"invalid remapped coordinates for {path}")
    order = np.argsort(ids)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, spacehastenid=ids[order], umap=xy[order])
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--greedy-output", type=Path, required=True)
    parser.add_argument("--lcb-output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.coordinates) as data:
        common_ids = data["spacehastenid"].astype(np.int64)
        common_xy = data["umap"].astype(np.float32)
    if common_xy.shape != (len(common_ids), 2) or not np.isfinite(common_xy).all():
        raise ValueError("invalid common UMAP coordinates")
    coordinate_by_id = dict(zip(common_ids, common_xy, strict=True))
    greedy_ids: list[int] = []
    greedy_coordinates: list[np.ndarray] = []
    lcb_ids: list[int] = []
    lcb_coordinates: list[np.ndarray] = []
    shared_differences = 0
    with gzip.open(args.mapping, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            coordinate = coordinate_by_id[int(row["common_id"])]
            greedy_id, lcb_id = row["greedy_spacehastenid"], row["lcb_spacehastenid"]
            if greedy_id:
                greedy_ids.append(int(greedy_id))
                greedy_coordinates.append(coordinate)
            if lcb_id:
                lcb_ids.append(int(lcb_id))
                lcb_coordinates.append(coordinate)
            if greedy_id and lcb_id:
                shared_differences += 0
    args.greedy_output.parent.mkdir(parents=True, exist_ok=True)
    args.lcb_output.parent.mkdir(parents=True, exist_ok=True)
    write_coordinates(args.greedy_output, greedy_ids, greedy_coordinates)
    write_coordinates(args.lcb_output, lcb_ids, lcb_coordinates)
    summary = {
        "common_coordinates": len(common_ids),
        "greedy_coordinates": len(greedy_ids),
        "lcb_coordinates": len(lcb_ids),
        "shared_coordinate_differences": shared_differences,
    }
    args.coordinates.with_name("common_hit_remap_metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
