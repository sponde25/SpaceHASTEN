#!/usr/bin/env python3
"""Prepare, run, and combine selected-cohort diversity resampling."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from spacehasten.analysis.cached import family_distribution, sampled_diversity


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def codes(values: list[str]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(sorted(set(values)))}
    return np.asarray([lookup[value] for value in values], dtype=np.int32)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false", "1", "0"}:
        raise ValueError(f"invalid boolean value: {value}")
    return normalized in {"true", "1"}


def prepare(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output root is non-empty; pass --overwrite: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "chunks").mkdir()
    (root / "logs").mkdir()
    manifest = read_rows(args.manifest)
    structures = read_rows(args.structure_cache)
    if not manifest or not structures:
        raise ValueError("manifest and structure cache must be non-empty")
    manifest_ids = np.asarray([int(row["spacehastenid"]) for row in manifest], dtype=np.int64)
    structure_ids = np.asarray([int(row["spacehastenid"]) for row in structures], dtype=np.int64)
    if len(set(structure_ids.tolist())) != len(structure_ids):
        raise ValueError("structure cache IDs must be unique")
    structure_lookup = {identifier: index for index, identifier in enumerate(structure_ids)}
    if not set(manifest_ids).issubset(structure_lookup):
        raise ValueError("manifest contains IDs absent from the structure cache")
    event_indices = np.asarray([structure_lookup[value] for value in manifest_ids], dtype=np.int64)
    structure_hashes = {int(row["spacehastenid"]): row["reghash"] for row in structures}
    if any(structure_hashes[int(row["spacehastenid"])] != row["reghash"] for row in manifest):
        raise ValueError("manifest and structure cache reghashes differ")
    with np.load(args.fingerprints, allow_pickle=False) as data:
        fingerprint_ids = data["spacehastenid"].astype(np.int64, copy=True)
        words = data["words"].astype(np.uint64, copy=True)
        popcounts = data["popcounts"].astype(np.int64, copy=True)
    if not np.array_equal(structure_ids, fingerprint_ids) or words.shape != (len(structures), 16):
        raise ValueError("fingerprint cache does not match the selected manifest")
    atlas_name = "clusterid" if "clusterid" in manifest[0] else "atlas_cluster"
    if atlas_name not in manifest[0]:
        raise ValueError("selected manifest lacks persistent atlas cluster labels")
    arrays: dict[str, np.ndarray] = {
        "spacehastenid": manifest_ids,
        "round": np.asarray([int(row["round"]) for row in manifest], dtype=np.int32),
        "is_hit": np.asarray([parse_bool(row["is_hit"]) for row in manifest], dtype=bool),
        "typed_code": codes([structures[index]["typed_scaffold"] for index in event_indices]),
        "generic_code": codes([structures[index]["generic_framework"] for index in event_indices]),
        "atlas_code": codes([row[atlas_name] for row in manifest]),
        "words": words[event_indices],
        "popcounts": popcounts[event_indices],
    }
    seed_count = 0
    if args.seed_reference_cache or args.seed_index:
        if not args.seed_reference_cache or not args.seed_index:
            raise ValueError("seed reference cache and seed index must be supplied together")
        arrays.update(load_seed_cache(args.seed_reference_cache, args.seed_index))
        seed_count = len(arrays["seed_spacehastenid"])
    cache_path = root / "resampling_cache.npz"
    write_npz(cache_path, **arrays)
    write_preparation(args, root, cache_path, manifest, seed_count)


def load_seed_cache(reference_path: Path, index_path: Path) -> dict[str, np.ndarray]:
    from FPSim2 import FPSim2Engine

    with np.load(reference_path, allow_pickle=False) as data:
        seed_ids = data["seed_spacehastenid"].astype(np.int64, copy=True)
        seed_typed = data["typed_scaffold_code"].astype(np.int32, copy=True)
        seed_generic = data["generic_framework_code"].astype(np.int32, copy=True)
        seed_atlas = data["atlas_cluster_code"].astype(np.int32, copy=True)
    if len({len(seed_ids), len(seed_typed), len(seed_generic), len(seed_atlas)}) != 1:
        raise ValueError("seed reference arrays differ in length")
    order = np.argsort(seed_ids)
    seed_ids = seed_ids[order]
    engine = FPSim2Engine(str(index_path.resolve()))
    if engine.fp_type != "Morgan" or engine.fp_params != {"radius": 2, "fpSize": 1024}:
        raise ValueError("seed index does not use Morgan-2 1024-bit fingerprints")
    fingerprints = engine.fps
    fingerprint_order = np.argsort(np.asarray(fingerprints[:, 0], dtype=np.int64))
    if not np.array_equal(
        np.asarray(fingerprints[fingerprint_order, 0], dtype=np.int64), seed_ids
    ):
        raise ValueError("seed index and seed reference IDs differ")
    return {
        "seed_spacehastenid": seed_ids,
        "seed_typed_code": seed_typed[order],
        "seed_generic_code": seed_generic[order],
        "seed_atlas_code": seed_atlas[order],
        "seed_words": np.asarray(fingerprints[fingerprint_order, 1:-1], dtype=np.uint64),
        "seed_popcounts": np.asarray(fingerprints[fingerprint_order, -1], dtype=np.int64),
    }


def write_preparation(
    args: argparse.Namespace,
    root: Path,
    cache_path: Path,
    manifest: list[dict[str, str]],
    seed_count: int,
) -> None:
    rounds = sorted({int(row["round"]) for row in manifest})
    hit_count = sum(parse_bool(row["is_hit"]) for row in manifest)
    worker = Path(__file__).resolve()
    source_root = worker.parents[2] / "src"
    submit = root / "submit.sh"
    submit.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=selected_resampling
#SBATCH --partition=jobs
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --array=1-{args.task_count}%{args.task_count}
#SBATCH --output={root}/logs/task-%A_%a.out
#SBATCH --error={root}/logs/task-%A_%a.err
set -euo pipefail
source /data/programs/oce/actoce
conda activate fpsim2-0.7.3
export PYTHONPATH={source_root}:${{PYTHONPATH:-}}
python3 -u {worker} worker --output-root {root} --task-index "$SLURM_ARRAY_TASK_ID"
""",
        encoding="utf-8",
    )
    submit.chmod(0o755)
    inputs = [args.manifest, args.structure_cache, args.fingerprints]
    if args.seed_reference_cache:
        inputs.extend([args.seed_reference_cache, args.seed_index])
    metadata = {
        "status": "ready_not_submitted",
        "rows": len(manifest),
        "rounds": rounds,
        "hits": hit_count,
        "seed_rows": seed_count,
        "task_count": args.task_count,
        "replicates": args.replicates,
        "pair_samples": args.pair_samples,
        "random_seed": args.random_seed,
        "cache": str(cache_path),
        "cache_sha256": sha256(cache_path),
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in inputs
        ],
        "submit_script": str(submit),
    }
    write_json(root / "preparation.json", metadata)
    print(json.dumps({"status": "ready", "tasks": args.task_count, "rows": len(manifest)}))


def metric_record(
    indices: np.ndarray,
    words: np.ndarray,
    popcounts: np.ndarray,
    typed: np.ndarray,
    generic: np.ndarray,
    atlas: np.ndarray,
    *,
    seed: int,
    pair_samples: int,
) -> dict[str, Any]:
    diversity, standard_error, pairs = sampled_diversity(
        indices.astype(np.int64, copy=False),
        words,
        popcounts,
        seed=seed,
        samples=pair_samples,
    )
    result: dict[str, Any] = {
        "sample_size": len(indices),
        "internal_diversity": diversity,
        "internal_diversity_mc_se": standard_error,
        "pair_samples": pairs,
    }
    for prefix, values in (
        ("typed", typed[indices]),
        ("generic", generic[indices]),
        ("atlas", atlas[indices]),
    ):
        result.update(
            {
                f"{prefix}_{name}": value
                for name, value in family_distribution(values).items()
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty resampling task")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fields = sorted({field for row in rows for field in row})
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def worker(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    preparation_path = root / "preparation.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    task_count = int(preparation["task_count"])
    if not 1 <= args.task_index <= task_count:
        raise ValueError(f"task index must be in [1,{task_count}]")
    cache_path = root / "resampling_cache.npz"
    if sha256(cache_path) != preparation["cache_sha256"]:
        raise ValueError("resampling cache digest differs from preparation")
    token = f"{args.task_index:04d}_of_{task_count:04d}"
    output = root / "chunks" / f"resampling_{token}.csv"
    receipt_path = root / "chunks" / f"resampling_{token}.json"
    if output.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("output_sha256") == sha256(output):
            print(json.dumps({"status": "already_complete", "task": args.task_index}))
            return
    with np.load(cache_path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    rounds = arrays["round"].astype(np.int32)
    hits = arrays["is_hit"].astype(bool)
    total_replicates = int(preparation["replicates"])
    first = total_replicates * (args.task_index - 1) // task_count
    stop = total_replicates * args.task_index // task_count
    base_seed = int(preparation["random_seed"])
    pair_samples = int(preparation["pair_samples"])
    rows: list[dict[str, Any]] = []
    for replicate in tqdm(range(first, stop), desc=f"resampling {token}", unit="replicate"):
        for round_id in sorted(np.unique(rounds).astype(int).tolist()):
            eligible = np.flatnonzero(rounds == round_id)
            sample_size = int(hits[eligible].sum())
            random = np.random.default_rng(base_seed + round_id * 1000 + replicate)
            selected = random.choice(eligible, size=sample_size, replace=False)
            rows.append(
                {
                    "design": "round_selected_matched_to_hits",
                    "task": args.task_index,
                    "replicate": replicate,
                    "round": round_id,
                    **metric_record(
                        selected,
                        arrays["words"],
                        arrays["popcounts"],
                        arrays["typed_code"],
                        arrays["generic_code"],
                        arrays["atlas_code"],
                        seed=base_seed + round_id * 10_000 + replicate,
                        pair_samples=pair_samples,
                    ),
                }
            )
        if "seed_words" in arrays:
            sample_size = int(hits.sum())
            if sample_size > len(arrays["seed_words"]):
                raise ValueError("final hit count exceeds the seed reference population")
            random = np.random.default_rng(base_seed + 100_000 + replicate)
            selected = random.choice(len(arrays["seed_words"]), size=sample_size, replace=False)
            rows.append(
                {
                    "design": "starting_seed_matched_to_final_hits",
                    "task": args.task_index,
                    "replicate": replicate,
                    "round": "",
                    **metric_record(
                        selected,
                        arrays["seed_words"],
                        arrays["seed_popcounts"],
                        arrays["seed_typed_code"],
                        arrays["seed_generic_code"],
                        arrays["seed_atlas_code"],
                        seed=base_seed + 200_000 + replicate,
                        pair_samples=pair_samples,
                    ),
                }
            )
    write_csv(output, rows)
    write_json(
        receipt_path,
        {
            "status": "complete",
            "task_index": args.task_index,
            "first_replicate": first,
            "stop_replicate": stop,
            "rows": len(rows),
            "cache_sha256": preparation["cache_sha256"],
            "output_sha256": sha256(output),
        },
    )
    print(json.dumps({"status": "complete", "task": args.task_index, "rows": len(rows)}))


def combine(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    root = args.output_root.resolve()
    success = root / "_SUCCESS.json"
    if success.exists() and not args.overwrite:
        raise FileExistsError(f"combined output exists; pass --overwrite: {success}")
    preparation = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
    frames = []
    for task in tqdm(
        range(1, int(preparation["task_count"]) + 1),
        desc="combine resampling",
        unit="task",
    ):
        token = f"{task:04d}_of_{int(preparation['task_count']):04d}"
        output = root / "chunks" / f"resampling_{token}.csv"
        receipt_path = root / "chunks" / f"resampling_{token}.json"
        if not output.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(f"resampling task is incomplete: {token}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("cache_sha256") != preparation["cache_sha256"]
            or receipt.get("output_sha256") != sha256(output)
        ):
            raise ValueError(f"resampling task receipt mismatch: {token}")
        frames.append(pd.read_csv(output))
    frame = pd.concat(frames, ignore_index=True)
    frame["round"] = pd.to_numeric(frame["round"], errors="coerce").astype("Int64")
    replicates = int(preparation["replicates"])
    rounds = [int(value) for value in preparation["rounds"]]
    has_seed = int(preparation["seed_rows"]) > 0
    expected_rows = replicates * (len(rounds) + int(has_seed))
    if len(frame) != expected_rows:
        raise ValueError(f"combined {len(frame)} rows; expected {expected_rows}")
    if frame.duplicated(["design", "replicate", "round"]).any():
        raise ValueError("resampling contains duplicate design cells")
    expected_replicates = set(range(replicates))
    if set(frame["replicate"].astype(int)) != expected_replicates:
        raise ValueError("resampling replicate coverage is incomplete")
    for round_id in rounds:
        observed = frame[
            (frame["design"] == "round_selected_matched_to_hits")
            & (frame["round"] == round_id)
        ]
        if set(observed["replicate"].astype(int)) != expected_replicates:
            raise ValueError(f"round {round_id} replicate coverage is incomplete")
    if has_seed:
        observed = frame[frame["design"] == "starting_seed_matched_to_final_hits"]
        if set(observed["replicate"].astype(int)) != expected_replicates:
            raise ValueError("starting-seed replicate coverage is incomplete")

    replicate_path = root / "resampling_replicates.csv"
    frame.to_csv(replicate_path, index=False)
    intervals = summarize_intervals(frame)
    interval_path = root / "resampling_intervals.csv"
    intervals.to_csv(interval_path, index=False)
    save_resampling_figure(intervals, root / "count_matched_rarefaction", rounds, args.dpi)
    outputs = [replicate_path, interval_path]
    outputs.extend(sorted(root.glob("count_matched_rarefaction.*")))
    receipt = {
        "status": "complete",
        "task_count": int(preparation["task_count"]),
        "replicates": replicates,
        "rounds": rounds,
        "seed_reference_available": has_seed,
        "pair_samples_per_replicate": int(preparation["pair_samples"]),
        "replicate_rows": len(frame),
        "cache_sha256": preparation["cache_sha256"],
        "outputs": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in outputs
        ],
    }
    write_json(success, receipt)
    print(json.dumps(receipt, sort_keys=True))


def summarize_intervals(frame: Any) -> Any:
    import pandas as pd

    excluded = {"design", "task", "replicate", "round", "sample_size", "pair_samples"}
    metrics = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    rows: list[dict[str, Any]] = []
    for (design, round_id), group in frame.groupby(["design", "round"], dropna=False, sort=True):
        for metric in metrics:
            values = group[metric].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "design": design,
                    "round": round_id,
                    "metric": metric,
                    "mean": values.mean(),
                    "median": values.median(),
                    "ci95_low": values.quantile(0.025),
                    "ci95_high": values.quantile(0.975),
                    "replicates": len(values),
                }
            )
    return pd.DataFrame(rows)


def save_resampling_figure(intervals: Any, stem: Path, rounds: list[int], dpi: int) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    selected = intervals[intervals["design"] == "round_selected_matched_to_hits"]
    for axis, metric, label in (
        (axes[0], "internal_diversity", "Internal diversity"),
        (axes[1], "generic_q0", "Generic framework richness"),
    ):
        subset = selected[selected["metric"] == metric]
        axis.errorbar(
            subset["round"],
            subset["median"],
            yerr=[
                subset["median"] - subset["ci95_low"],
                subset["ci95_high"] - subset["median"],
            ],
            marker="o",
        )
        axis.set(xlabel="Round", ylabel=label, xticks=rounds)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--manifest", type=Path, required=True)
    prepare_parser.add_argument("--structure-cache", type=Path, required=True)
    prepare_parser.add_argument("--fingerprints", type=Path, required=True)
    prepare_parser.add_argument("--seed-reference-cache", type=Path)
    prepare_parser.add_argument("--seed-index", type=Path)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--task-count", type=int, default=40)
    prepare_parser.add_argument("--replicates", type=int, default=200)
    prepare_parser.add_argument("--pair-samples", type=int, default=100_000)
    prepare_parser.add_argument("--random-seed", type=int, default=42)
    prepare_parser.add_argument("--overwrite", action="store_true")
    worker_parser = commands.add_parser("worker")
    worker_parser.add_argument("--output-root", type=Path, required=True)
    worker_parser.add_argument("--task-index", type=int, required=True)
    combine_parser = commands.add_parser("combine")
    combine_parser.add_argument("--output-root", type=Path, required=True)
    combine_parser.add_argument("--dpi", type=int, default=600)
    combine_parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        if min(args.task_count, args.replicates, args.pair_samples) < 1:
            parser.error("task-count, replicates, and pair-samples must be positive")
        if args.task_count > args.replicates:
            parser.error("task-count cannot exceed replicates")
        for path in (args.manifest, args.structure_cache, args.fingerprints):
            if not path.is_file():
                parser.error(f"input does not exist: {path}")
        prepare(args)
    elif args.command == "worker":
        worker(args)
    else:
        if args.dpi < 1:
            parser.error("dpi must be positive")
        combine(args)


if __name__ == "__main__":
    main()
