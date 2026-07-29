#!/usr/bin/env python3
"""Prepare common-ID manifests for paper-aligned comparison diversity."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output root is non-empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize_events(
    path: Path,
    workflow: str,
    common_by_hash: pd.Series,
) -> pd.DataFrame:
    events = pd.read_csv(path)
    required = {
        "reghash",
        "smiles",
        "dock_score",
        "round",
        "scored",
        "hit",
        "strict_hit",
    }
    if missing := required - set(events.columns):
        raise ValueError(f"{workflow} events lack columns: {sorted(missing)}")
    if events["reghash"].isna().any() or not events["reghash"].is_unique:
        raise ValueError(f"{workflow} event reghashes must be non-null and unique")
    common_ids = events["reghash"].map(common_by_hash)
    if common_ids.isna().any():
        raise ValueError(f"{workflow} events are not fully covered by the common union")
    manifest = pd.DataFrame(
        {
            "spacehastenid": common_ids.astype(np.int64),
            "reghash": events["reghash"].astype(str),
            "smiles": events["smiles"].astype(str),
            "round": events["round"].astype(int),
            "dock_score": events["dock_score"].astype(float),
            "is_scored": events["scored"].astype(bool),
            "is_hit": events["hit"].astype(bool),
            "is_strict_hit": events["strict_hit"].astype(bool),
            "workflow": workflow,
        }
    ).sort_values(["reghash", "spacehastenid"], kind="stable")
    return manifest.reset_index(drop=True)


def write_manifest(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow",
        nargs=2,
        action="append",
        metavar=("NAME", "EVENTS"),
        required=True,
    )
    parser.add_argument("--union", type=Path, required=True)
    parser.add_argument("--fingerprints", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if len(args.workflow) != 2 or len({item[0] for item in args.workflow}) != 2:
        parser.error("provide exactly two workflows with unique names")
    inputs = [args.union, args.fingerprints, *(Path(item[1]) for item in args.workflow)]
    for path in inputs:
        if not path.is_file():
            parser.error(f"input does not exist: {path}")

    root = args.output_root.resolve()
    prepare_output(root, args.overwrite)
    union = pd.read_csv(args.union, usecols=["common_id", "reghash"])
    if not union["common_id"].is_unique or not union["reghash"].is_unique:
        raise ValueError("common union IDs and reghashes must be unique")
    expected_ids = union["common_id"].to_numpy(np.int64)
    with np.load(args.fingerprints, allow_pickle=False) as data:
        fingerprint_ids = data["spacehastenid"].astype(np.int64)
        words_shape = data["words"].shape
    if not np.array_equal(fingerprint_ids, expected_ids) or words_shape != (len(union), 16):
        raise ValueError("common union and fingerprint cache differ")

    common_by_hash = pd.Series(union["common_id"].to_numpy(), index=union["reghash"])
    outputs = []
    workflows = {}
    for name, event_path_text in args.workflow:
        event_path = Path(event_path_text).resolve()
        manifest = normalize_events(event_path, name, common_by_hash)
        output = root / f"{name}_manifest.csv.gz"
        write_manifest(output, manifest)
        outputs.append(output)
        workflows[name] = {
            "selected": len(manifest),
            "hits": int(manifest["is_hit"].sum()),
            "strict_hits": int(manifest["is_strict_hit"].sum()),
            "manifest": str(output),
        }
    receipt = {
        "status": "complete",
        "union_rows": len(union),
        "fingerprints": str(args.fingerprints.resolve()),
        "fingerprints_sha256": sha256(args.fingerprints),
        "workflows": workflows,
        "inputs": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in inputs
        ],
        "outputs": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
        ],
    }
    write_json(root / "_SUCCESS.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
