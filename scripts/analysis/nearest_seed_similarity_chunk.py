#!/usr/bin/env python3
"""Calculate exact top-1 seed Tanimoto similarity for one SMILES chunk."""

from __future__ import annotations

import argparse
import gzip
import logging
from pathlib import Path

import numpy as np
from FPSim2 import FPSim2Engine
from tqdm import tqdm

LOGGER = logging.getLogger("nearest_seed_similarity_chunk")
THRESHOLDS = (0.3, 0.1, 0.0)
EXPECTED_FP_TYPE = "Morgan"
EXPECTED_FP_PARAMS = {"radius": 2, "fpSize": 1024}


def calculate(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    index_path = args.seed_index.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"output already exists; use --force: {output_path}")

    with gzip.open(input_path, "rt", encoding="utf-8") as handle:
        count = sum(1 for _ in handle)
    if count == 0:
        raise ValueError("input chunk contains no compounds")
    engine = FPSim2Engine(str(index_path))
    if engine.fp_type != EXPECTED_FP_TYPE or engine.fp_params != EXPECTED_FP_PARAMS:
        raise ValueError("seed index does not use Morgan-2 1024-bit fingerprints")
    if len(engine.fps) == 0:
        raise ValueError("seed index contains no fingerprints")
    compound_ids = np.empty(count, dtype=np.uint64)
    seed_ids = np.empty(count, dtype=np.uint64)
    similarities = np.empty(count, dtype=np.float32)
    threshold_used = np.empty(count, dtype=np.float32)

    with gzip.open(input_path, "rt", encoding="utf-8") as handle:
        for row_index, line in enumerate(
            tqdm(handle, total=count, desc=input_path.stem, unit="mol")
        ):
            smiles, compound_id = line.rstrip().rsplit(None, 1)
            result = None
            used = 0.0
            for threshold in THRESHOLDS:
                matches = engine.top_k(
                    smiles,
                    k=1,
                    threshold=threshold,
                    metric="tanimoto",
                    n_workers=1,
                )
                if len(matches):
                    result = matches[0]
                    used = threshold
                    break
            compound_ids[row_index] = int(compound_id)
            if result is None:
                seed_ids[row_index] = 0
                similarities[row_index] = 0.0
            else:
                seed_ids[row_index] = int(result["mol_id"])
                similarities[row_index] = float(result["coeff"])
            threshold_used[row_index] = used

    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            spacehastenid=compound_ids,
            nearest_seed_id=seed_ids,
            tanimoto=similarities,
            threshold_used=threshold_used,
        )
    temporary.replace(output_path)
    LOGGER.info("Saved %d nearest-seed results to %s", count, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        calculate(parse_args())
    except Exception:
        LOGGER.exception("Nearest-seed calculation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
