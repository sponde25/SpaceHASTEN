#!/usr/bin/env python3
"""Combine and validate distributed nearest-seed similarity results."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

LOGGER = logging.getLogger("combine_nearest_seed_chunks")


def combine(args: argparse.Namespace) -> None:
    chunks_dir = args.chunks_dir.resolve()
    output_path = args.output.resolve()
    metadata_path = output_path.with_name("nearest_seed_similarity_metadata.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if (output_path.exists() or metadata_path.exists()) and not args.force:
        raise FileExistsError("combined nearest-seed output already exists")

    arrays: dict[str, list[np.ndarray]] = {
        "spacehastenid": [],
        "nearest_seed_id": [],
        "tanimoto": [],
        "threshold_used": [],
    }
    with tqdm(total=args.task_count, desc="Combining similarity chunks", unit="chunk") as bar:
        for index in range(1, args.task_count + 1):
            path = chunks_dir / f"nearest_{index:04d}_of_{args.task_count:04d}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path) as chunk:
                lengths = {name: len(chunk[name]) for name in arrays}
                if len(set(lengths.values())) != 1:
                    raise ValueError(f"chunk arrays differ in length: {path}: {lengths}")
                for name in arrays:
                    arrays[name].append(chunk[name].copy())
            bar.update(1)

    merged = {name: np.concatenate(parts) for name, parts in arrays.items()}
    order = np.argsort(merged["spacehastenid"])
    merged = {name: values[order] for name, values in merged.items()}
    count = len(merged["spacehastenid"])
    if count != args.expected_count:
        raise ValueError(f"combined {count} rows; expected {args.expected_count}")
    if not np.all(merged["spacehastenid"][1:] > merged["spacehastenid"][:-1]):
        raise ValueError("compound identifiers are duplicated")
    if not np.isfinite(merged["tanimoto"]).all() or np.any(
        (merged["tanimoto"] < 0) | (merged["tanimoto"] > 1)
    ):
        raise ValueError("invalid Tanimoto values")
    if not np.isfinite(merged["threshold_used"]).all():
        raise ValueError("invalid threshold-tier values")

    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **merged)
    temporary.replace(output_path)

    thresholds, threshold_counts = np.unique(merged["threshold_used"], return_counts=True)
    metadata = {
        "output": str(output_path),
        "compound_count": count,
        "task_count": args.task_count,
        "tanimoto_min": float(merged["tanimoto"].min()),
        "tanimoto_median": float(np.median(merged["tanimoto"])),
        "tanimoto_max": float(merged["tanimoto"].max()),
        "threshold_tiers": {
            str(float(threshold)): int(tier_count)
            for threshold, tier_count in zip(thresholds, threshold_counts, strict=True)
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Combined %d exact nearest-seed similarities", count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if min(args.task_count, args.expected_count) < 1:
        parser.error("task count and expected count must be positive")
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        combine(parse_args())
    except Exception:
        LOGGER.exception("Combining nearest-seed chunks failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
