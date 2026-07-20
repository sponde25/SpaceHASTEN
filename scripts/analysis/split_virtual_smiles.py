#!/usr/bin/env python3
"""Split the virtual portion of a seed-first SMILES file for SLURM arrays."""

from __future__ import annotations

import argparse
import gzip
import json
from contextlib import ExitStack
from pathlib import Path

from tqdm import tqdm


def split_input(args: argparse.Namespace) -> None:
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / f"virtual_{index:04d}_of_{args.task_count:04d}.smi.gz"
        for index in range(1, args.task_count + 1)
    ]
    metadata_path = output_dir / "split_metadata.json"
    if (any(path.exists() for path in paths) or metadata_path.exists()) and not args.force:
        raise FileExistsError("split outputs already exist; use --force to replace them")
    if args.force:
        for path in [*paths, metadata_path]:
            path.unlink(missing_ok=True)

    counts = [0] * args.task_count
    temporary = [path.with_name(f".{path.name}.tmp") for path in paths]
    virtual_index = 0
    total_lines = 0
    with ExitStack() as stack:
        handles = [
            stack.enter_context(gzip.open(path, "wt", encoding="utf-8")) for path in temporary
        ]
        with gzip.open(input_path, "rt", encoding="utf-8") as source:
            progress = tqdm(
                total=args.virtual_count,
                desc="Splitting virtual compounds",
                unit="mol",
            )
            for total_lines, line in enumerate(source, start=1):
                if total_lines <= args.seed_count:
                    continue
                if virtual_index >= args.virtual_count:
                    raise ValueError("input contains more virtual compounds than expected")
                task = virtual_index % args.task_count
                handles[task].write(line)
                counts[task] += 1
                virtual_index += 1
                progress.update(1)
            progress.close()

    if total_lines != args.seed_count + args.virtual_count:
        raise ValueError(
            f"input contains {total_lines} rows; expected {args.seed_count + args.virtual_count}"
        )
    for source, destination in zip(temporary, paths, strict=True):
        source.replace(destination)

    metadata_path.write_text(
        json.dumps(
            {
                "input": str(input_path),
                "seed_count": args.seed_count,
                "virtual_count": args.virtual_count,
                "task_count": args.task_count,
                "chunk_counts": counts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--virtual-count", type=int, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if min(args.seed_count, args.virtual_count, args.task_count) < 1:
        parser.error("counts must be positive")
    return args


if __name__ == "__main__":
    split_input(parse_args())
