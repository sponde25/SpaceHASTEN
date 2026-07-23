#!/usr/bin/env python3
"""Build a reghash-sorted common-hit fingerprint index for fixed UMAP projection."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from FPSim2.io import create_db_file


def connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def load_hits(path: Path, cutoff: float) -> dict[str, tuple[str, int]]:
    with connect(path) as connection:
        rows = connection.execute(
            "SELECT reghash, smiles, spacehastenid FROM data "
            "WHERE dock_iteration > 0 AND dock_score <= ? ORDER BY reghash",
            (cutoff,),
        )
        hits = {
            str(reghash): (str(smiles), int(identifier)) for reghash, smiles, identifier in rows
        }
    if not hits:
        raise ValueError(f"no virtual hits found in {path}")
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--greedy-db", type=Path, required=True)
    parser.add_argument("--lcb-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=-9.7)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    smiles_path = output / "common_hits.smi.gz"
    mapping_path = output / "common_hit_mapping.csv.gz"
    index_path = output / "common_hits_morgan_r2_1024.h5"
    metadata_path = output / "common_hit_metadata.json"
    for path in (smiles_path, mapping_path, index_path, metadata_path):
        if path.exists():
            raise FileExistsError(path)

    greedy, lcb = load_hits(args.greedy_db, args.cutoff), load_hits(args.lcb_db, args.cutoff)
    hashes = sorted(set(greedy) | set(lcb))
    temporary_smiles = smiles_path.with_name(f".{smiles_path.name}.tmp")
    temporary_mapping = mapping_path.with_name(f".{mapping_path.name}.tmp")
    with (
        gzip.open(temporary_smiles, "wt", encoding="utf-8") as smiles_handle,
        gzip.open(temporary_mapping, "wt", encoding="utf-8", newline="") as mapping_handle,
    ):
        writer = csv.writer(mapping_handle)
        writer.writerow(("common_id", "reghash", "greedy_spacehastenid", "lcb_spacehastenid"))
        for common_id, reghash in enumerate(hashes, start=1):
            greedy_value, lcb_value = greedy.get(reghash), lcb.get(reghash)
            smiles = greedy_value[0] if greedy_value is not None else lcb_value[0]  # type: ignore[index]
            if lcb_value is not None and greedy_value is not None and lcb_value[0] != smiles:
                raise ValueError(f"shared reghash has different SMILES: {reghash}")
            smiles_handle.write(f"{smiles} {common_id}\n")
            writer.writerow(
                (
                    common_id,
                    reghash,
                    greedy_value[1] if greedy_value is not None else "",
                    lcb_value[1] if lcb_value is not None else "",
                )
            )
    temporary_smiles.replace(smiles_path)
    temporary_mapping.replace(mapping_path)

    def molecules():  # type: ignore[no-untyped-def]
        with gzip.open(smiles_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                smiles, common_id = line.rstrip().rsplit(None, 1)
                yield smiles, int(common_id)

    temporary_index = index_path.with_name(f".{index_path.name}.tmp")
    create_db_file(
        molecules(),
        str(temporary_index),
        "smiles",
        "Morgan",
        {"radius": 2, "fpSize": 1024},
    )
    temporary_index.replace(index_path)
    shared = len(set(greedy) & set(lcb))
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "cutoff": args.cutoff,
        "greedy_hits": len(greedy),
        "lcb_hits": len(lcb),
        "shared_hits": shared,
        "union_hits": len(hashes),
        "mapping": str(mapping_path),
        "smiles": str(smiles_path),
        "fingerprint_index": str(index_path),
        "fingerprint": "binary Morgan radius 2, 1024 bits",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
