#!/usr/bin/env python3
"""Prepare, analyze, plot, and validate two-workflow comparisons."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from spacehasten.analysis.comparison import (
    analyze_comparison,
    prepare_comparison,
    refresh_comparison_semantics,
)


def prepare(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output root is non-empty; pass --overwrite: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    result = prepare_comparison(
        left_name=args.left_name,
        left_label=args.left_label,
        left_database=args.left_database,
        left_analysis=args.left_analysis,
        left_validation=args.left_validation,
        right_name=args.right_name,
        right_label=args.right_label,
        right_database=args.right_database,
        right_analysis=args.right_analysis,
        right_validation=args.right_validation,
        seed_atlas_definition=args.seed_atlas_definition,
        output_root=root,
        hit_cutoff=args.hit_cutoff,
        strict_cutoff=args.strict_cutoff,
    )
    print(json.dumps(result, sort_keys=True))


def analyze(args: argparse.Namespace) -> None:
    result = analyze_comparison(
        root=args.comparison_root.resolve(),
        common_coordinates=args.common_coordinates,
        common_atlas_assignments_path=args.common_atlas_assignments,
        seed_coordinates=args.seed_coordinates,
        umap_model=args.umap_model,
        timing_paths={
            args.left_name: args.left_timing,
            args.right_name: args.right_timing,
        },
        replicates=args.replicates,
        pair_samples=args.pair_samples,
        random_seed=args.random_seed,
        dpi=args.dpi,
    )
    print(json.dumps(result, sort_keys=True))


def refresh(args: argparse.Namespace) -> None:
    result = refresh_comparison_semantics(args.comparison_root.resolve())
    print(json.dumps(result, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    for side in ("left", "right"):
        prepare_parser.add_argument(f"--{side}-name", required=True)
        prepare_parser.add_argument(f"--{side}-label", required=True)
        prepare_parser.add_argument(f"--{side}-database", type=Path, required=True)
        prepare_parser.add_argument(f"--{side}-analysis", type=Path, required=True)
        prepare_parser.add_argument(f"--{side}-validation", type=Path, required=True)
    prepare_parser.add_argument("--seed-atlas-definition", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--hit-cutoff", type=float, required=True)
    prepare_parser.add_argument("--strict-cutoff", type=float, required=True)
    prepare_parser.add_argument("--overwrite", action="store_true")
    prepare_parser.set_defaults(function=prepare)

    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--comparison-root", type=Path, required=True)
    analyze_parser.add_argument("--common-coordinates", type=Path, required=True)
    analyze_parser.add_argument("--common-atlas-assignments", type=Path, required=True)
    analyze_parser.add_argument("--seed-coordinates", type=Path, required=True)
    analyze_parser.add_argument("--umap-model", type=Path, required=True)
    analyze_parser.add_argument("--left-name", required=True)
    analyze_parser.add_argument("--left-timing", type=Path, required=True)
    analyze_parser.add_argument("--right-name", required=True)
    analyze_parser.add_argument("--right-timing", type=Path, required=True)
    analyze_parser.add_argument("--replicates", type=int, default=200)
    analyze_parser.add_argument("--pair-samples", type=int, default=100_000)
    analyze_parser.add_argument("--random-seed", type=int, default=42)
    analyze_parser.add_argument("--dpi", type=int, default=600)
    analyze_parser.set_defaults(function=analyze)

    refresh_parser = commands.add_parser("refresh")
    refresh_parser.add_argument("--comparison-root", type=Path, required=True)
    refresh_parser.set_defaults(function=refresh)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "prepare":
        required = (
            args.left_database,
            args.left_analysis,
            args.left_validation,
            args.right_database,
            args.right_analysis,
            args.right_validation,
            args.seed_atlas_definition,
        )
        for path in required:
            if not path.exists():
                parser.error(f"required input does not exist: {path}")
        if args.left_name == args.right_name:
            parser.error("workflow names must differ")
    elif args.command == "analyze":
        required = (
            args.comparison_root / "comparison_compatibility.json",
            args.common_coordinates,
            args.common_atlas_assignments,
            args.seed_coordinates,
            args.umap_model,
            args.left_timing,
            args.right_timing,
        )
        for path in required:
            if not path.exists():
                parser.error(f"required input does not exist: {path}")
        if args.left_name == args.right_name:
            parser.error("workflow names must differ")
        if min(args.replicates, args.pair_samples, args.random_seed, args.dpi) < 1:
            parser.error("replicates, pair-samples, random-seed, and dpi must be positive")
    elif not (args.comparison_root / "comparison_compatibility.json").is_file():
        parser.error("comparison compatibility receipt does not exist")
    args.function(args)


if __name__ == "__main__":
    main()
