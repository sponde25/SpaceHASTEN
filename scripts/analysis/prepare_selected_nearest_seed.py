#!/usr/bin/env python3
"""Prepare deterministic selected-compound chunks for exact nearest-seed search."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Any

from FPSim2 import FPSim2Engine
from tqdm import tqdm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"spacehastenid", "reghash", "smiles"}
    missing = required - set(rows[0] if rows else ())
    if not rows or missing:
        raise ValueError(f"selected manifest is empty or lacks columns: {sorted(missing)}")
    if any(not row["smiles"] for row in rows):
        raise ValueError("selected manifest contains missing SMILES")
    compounds: list[dict[str, str]] = []
    by_identifier: dict[int, tuple[str, str]] = {}
    reghashes: set[str] = set()
    for row in rows:
        identifier = int(row["spacehastenid"])
        identity = (row["reghash"], row["smiles"])
        if identifier in by_identifier:
            if by_identifier[identifier] != identity:
                raise ValueError(f"selected ID maps to multiple structures: {identifier}")
            continue
        if identity[0] in reghashes:
            raise ValueError(f"selected reghash maps to multiple IDs: {identity[0]}")
        by_identifier[identifier] = identity
        reghashes.add(identity[0])
        compounds.append(row)
    return compounds


def write_chunk(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text,
    ):
        for row in rows:
            text.write(f"{row['smiles']} {int(row['spacehastenid'])}\n")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    index_path = args.seed_index.resolve()
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output root is non-empty; pass --overwrite: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("inputs", "chunks", "logs"):
        (root / name).mkdir()
    rows = read_manifest(manifest_path)
    task_count = min(args.task_count, len(rows))
    engine = FPSim2Engine(str(index_path))
    if engine.fp_type != "Morgan" or engine.fp_params != {"radius": 2, "fpSize": 1024}:
        raise ValueError("seed index does not use Morgan-2 1024-bit fingerprints")
    if len(engine.fps) == 0:
        raise ValueError("seed index contains no fingerprints")

    chunks = []
    for task in tqdm(range(1, task_count + 1), desc="nearest-seed inputs", unit="chunk"):
        start = len(rows) * (task - 1) // task_count
        stop = len(rows) * task // task_count
        path = root / "inputs" / f"selected_{task:04d}_of_{task_count:04d}.smi.gz"
        write_chunk(path, rows[start:stop])
        chunks.append(
            {
                "task_index": task,
                "start": start,
                "stop": stop,
                "rows": stop - start,
                "input": str(path),
                "input_sha256": sha256(path),
            }
        )
    worker = Path(__file__).with_name("nearest_seed_similarity_chunk.py").resolve()
    submit = root / "submit.sh"
    submit.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=selected_nearest_seed
#SBATCH --partition=jobs
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --array=1-{task_count}%{task_count}
#SBATCH --output={root}/logs/task-%A_%a.out
#SBATCH --error={root}/logs/task-%A_%a.err
set -euo pipefail
source /data/programs/oce/actoce
conda activate fpsim2-0.7.3
TASK=$(printf "%04d" "$SLURM_ARRAY_TASK_ID")
python3 -u {shlex.quote(str(worker))} \
  --input {shlex.quote(str(root))}/inputs/selected_${{TASK}}_of_{task_count:04d}.smi.gz \
  --seed-index {shlex.quote(str(index_path))} \
  --output {shlex.quote(str(root))}/chunks/nearest_${{TASK}}_of_{task_count:04d}.npz
""",
        encoding="utf-8",
    )
    submit.chmod(0o755)
    metadata = {
        "status": "ready_not_submitted",
        "selected_compounds": len(rows),
        "task_count": task_count,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "seed_index": str(index_path),
        "seed_index_sha256": sha256(index_path),
        "seed_count": len(engine.fps),
        "fingerprint": {"type": "Morgan", "radius": 2, "n_bits": 1024},
        "chunks": chunks,
        "submit_script": str(submit),
    }
    write_json(root / "preparation.json", metadata)
    print(json.dumps({"status": "ready", "tasks": task_count, "rows": len(rows)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.task_count < 1:
        parser.error("task-count must be positive")
    for path in (args.manifest, args.seed_index):
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
    prepare(args)


if __name__ == "__main__":
    main()
