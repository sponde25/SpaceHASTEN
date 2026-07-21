#!/usr/bin/env python3
"""Verify uncertainty-aware acquisition artifacts from a quick-test run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from spacehasten.core.acquisition import expected_improvement, lower_confidence_bound


def verify(args: argparse.Namespace) -> None:
    workspace = args.workspace.resolve()
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    shared_root = Path(manifest.get("shared_root") or workspace)
    acquisition_files = sorted(
        (shared_root / "docking").glob("iter*/acquisition.csv"),
        key=lambda path: int(path.parent.name.removeprefix("iter")),
    )
    if len(acquisition_files) != args.expected_rounds:
        raise SystemExit(
            f"found {len(acquisition_files)} acquisition files; expected {args.expected_rounds}"
        )

    control_scripts = sorted(
        (shared_root / "simsearch").glob("cycle*/CONTROL/submit_control_cycle*.sh")
    )
    expected_control_cycles = args.expected_rounds * 3
    if len(control_scripts) != expected_control_cycles:
        raise SystemExit(
            f"found {len(control_scripts)} control scripts; expected {expected_control_cycles}"
        )
    for path in control_scripts:
        script = path.read_text(encoding="utf-8")
        if "#SBATCH --gpus" in script or "#SBATCH --gres=gpu" in script:
            raise SystemExit(f"{path}: control prediction still requests a GPU")
        if 'export CUDA_VISIBLE_DEVICES=""' not in script:
            raise SystemExit(f"{path}: CUDA is not disabled")
        if "--accelerator cpu" not in script:
            raise SystemExit(f"{path}: prediction accelerator is not CPU")
    print(f"Verified CPU scheduling in {len(control_scripts)} control scripts")

    for path in acquisition_files:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != args.expected_batch_size:
            raise SystemExit(
                f"{path}: found {len(rows)} selections; expected {args.expected_batch_size}"
            )

        cluster_counts: dict[int, int] = defaultdict(int)
        selected_ids: set[int] = set()
        for expected_rank, row in enumerate(rows, start=1):
            sid = int(row["spacehastenid"])
            clusterid = int(row["clusterid"])
            mean = float(row["pred_score"])
            epistemic = float(row["epistemic_std"])
            base_score = float(row["base_score"])
            count_before = int(row["cluster_count_before"])
            penalty = float(row["cluster_penalty"])
            penalized_score = float(row["penalized_score"])

            if int(row["rank"]) != expected_rank:
                raise SystemExit(f"{path}: non-contiguous acquisition ranks")
            if row["method"] != args.method:
                raise SystemExit(f"{path}: unexpected method {row['method']!r}")
            if sid in selected_ids:
                raise SystemExit(f"{path}: duplicate selected compound {sid}")
            selected_ids.add(sid)
            if count_before != cluster_counts[clusterid]:
                raise SystemExit(f"{path}: invalid cluster count for compound {sid}")

            expected_penalty = args.cluster_lambda * math.log1p(count_before)
            if not math.isclose(penalty, expected_penalty, rel_tol=1e-12, abs_tol=1e-12):
                raise SystemExit(f"{path}: invalid cluster penalty for compound {sid}")
            if not math.isclose(
                penalized_score,
                base_score + penalty,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise SystemExit(f"{path}: invalid penalized score for compound {sid}")

            if args.method == "lcb":
                expected_base = lower_confidence_bound(mean, epistemic, args.lcb_beta)
            else:
                expected_base = -expected_improvement(
                    mean,
                    epistemic,
                    args.ei_hit_threshold,
                    args.ei_xi,
                )
            if not math.isclose(base_score, expected_base, rel_tol=1e-12, abs_tol=1e-12):
                raise SystemExit(f"{path}: invalid base score for compound {sid}")
            if not math.isclose(
                float(row["cluster_lambda"]), args.cluster_lambda, rel_tol=0, abs_tol=1e-12
            ):
                raise SystemExit(f"{path}: unexpected cluster lambda")
            if not math.isclose(
                float(row["cluster_similarity_threshold"]),
                0.4,
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise SystemExit(f"{path}: clustering threshold is not 0.4")
            cluster_counts[clusterid] += 1

        print(
            f"Verified {path}: method={args.method} selections={len(rows)} "
            f"clusters={len(cluster_counts)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--method", choices=("lcb", "ei"), required=True)
    parser.add_argument("--expected-rounds", type=int, required=True)
    parser.add_argument("--expected-batch-size", type=int, required=True)
    parser.add_argument("--cluster-lambda", type=float, required=True)
    parser.add_argument("--lcb-beta", type=float, required=True)
    parser.add_argument("--ei-hit-threshold", type=float, required=True)
    parser.add_argument("--ei-xi", type=float, required=True)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
