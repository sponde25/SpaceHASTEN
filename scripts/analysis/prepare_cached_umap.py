#!/usr/bin/env python3
"""Prepare a fixed-model UMAP array for a selected fingerprint cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
from pathlib import Path

import joblib
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(args: argparse.Namespace) -> None:
    fingerprints = args.fingerprints.resolve()
    model = args.model.resolve()
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output root is non-empty; pass --overwrite: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("chunks", "logs"):
        (root / name).mkdir()
    with np.load(fingerprints, allow_pickle=False) as data:
        identifiers = data["spacehastenid"].astype(np.int64)
        words = data["words"]
    if identifiers.ndim != 1 or words.shape != (len(identifiers), 16):
        raise ValueError("fingerprint cache must contain IDs and N x 16 words")
    if len(identifiers) == 0 or len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("fingerprint cache IDs must be non-empty and unique")
    bundle = joblib.load(model)
    if not isinstance(bundle, dict) or "reducer" not in bundle:
        raise ValueError("UMAP model bundle lacks a reducer")
    task_count = min(args.task_count, len(identifiers))
    write_submit(root, fingerprints, model, task_count, args.batch_size)
    metadata = {
        "status": "ready_not_submitted",
        "fingerprints": str(fingerprints),
        "fingerprints_sha256": sha256(fingerprints),
        "model": str(model),
        "model_sha256": sha256(model),
        "compound_count": len(identifiers),
        "task_count": task_count,
        "batch_size": args.batch_size,
        "fingerprint": {"type": "Morgan", "radius": 2, "n_bits": 1024},
        "submit_script": str(root / "submit.sh"),
    }
    (root / "preparation.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ready", "tasks": task_count, "rows": len(identifiers)}))


def write_submit(
    root: Path,
    fingerprints: Path,
    model: Path,
    task_count: int,
    batch_size: int,
) -> None:
    worker = Path(__file__).with_name("transform_landmark_umap_chunk.py").resolve()
    source_root = worker.parents[2] / "src"
    submit = root / "submit.sh"
    submit.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=selected_umap
#SBATCH --partition=jobs
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --array=1-{task_count}%{task_count}
#SBATCH --output={root}/logs/task-%A_%a.out
#SBATCH --error={root}/logs/task-%A_%a.err
set -euo pipefail
source /data/programs/oce/actoce
conda activate fpsim2-0.7.3
source /data/$USER/venvs/spacehasten-umap/bin/activate
export PYTHONPATH={source_root}:${{PYTHONPATH:-}}
python3 -u {shlex.quote(str(worker))} \
  --fingerprints {shlex.quote(str(fingerprints))} \
  --model {shlex.quote(str(model))} \
  --chunks-dir {shlex.quote(str(root / 'chunks'))} \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --task-count {task_count} \
  --batch-size {batch_size}
""",
        encoding="utf-8",
    )
    submit.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fingerprints", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-count", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.task_count, args.batch_size) < 1:
        parser.error("task-count and batch-size must be positive")
    for path in (args.fingerprints, args.model):
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
    prepare(args)


if __name__ == "__main__":
    main()
