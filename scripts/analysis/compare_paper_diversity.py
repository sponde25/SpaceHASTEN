#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Compare paper-aligned diversity for two validated virtual-hit cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from analyze_paper_diversity import validate_fpsim2_index
from FPSim2 import FPSim2Engine
from rdkit import DataStructs
from rdkit.SimDivFilters import rdSimDivPickers
from tqdm import tqdm

from spacehasten.analysis.cached import family_distribution

_WORDS: np.ndarray | None = None
_ROW_BY_ID: np.ndarray | None = None
_INDEX_PATH: str | None = None
_THRESHOLD = 0.55
_RIGHT_IDS: np.ndarray | None = None
_RIGHT_LEVEL1: np.ndarray | None = None
_MATCHED_SIZE = 0
_BASE_SEED = 42


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(root.resolve()) if root else resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def save_figure(figure: plt.Figure, output_dir: Path, name: str, dpi: int) -> None:
    figure.savefig(output_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def marker_sizes(counts: np.ndarray, maximum_count: float) -> np.ndarray:
    return 4.0 + 64.0 * np.sqrt(np.asarray(counts, dtype=float)) / math.sqrt(maximum_count)


def natural_rows(root: Path, workflow: str, label: str) -> list[dict[str, Any]]:
    metrics = pd.read_csv(root / "paper_aligned_metrics.csv")
    final_round = int(metrics["round"].max())
    selected = metrics[(metrics["scope"] == "cumulative") & (metrics["round"] == final_round)]
    if set(selected["representation"]) != {
        "scaffoldtree_level1",
        "sphere_exclusion_tanimoto_0.55",
    }:
        raise ValueError(f"{workflow} paper metrics are incomplete")
    return [
        {"workflow": workflow, "workflow_label": label, **row}
        for row in selected.to_dict(orient="records")
    ]


def plot_rank_size(args: argparse.Namespace, output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), sharex="row", sharey="row")
    definitions = (
        ("scaffoldtree_level1_families.csv", "hit_count", "ScaffoldTree Level 1"),
        ("sphere_exclusion_t055_clusters.csv", "hit_count", "Tanimoto-0.55 clusters"),
    )
    for row_index, (filename, count_column, row_title) in enumerate(definitions):
        for column_index, (root, label) in enumerate(
            ((args.left_root, args.left_label), (args.right_root, args.right_label))
        ):
            counts = pd.read_csv(root / filename)[count_column].to_numpy(np.int64)
            counts = np.sort(counts)[::-1]
            axis = axes[row_index, column_index]
            axis.plot(np.arange(1, len(counts) + 1), counts, linewidth=1.2)
            axis.set(
                xscale="log",
                yscale="log",
                xlabel="Family or cluster rank",
                ylabel="Virtual hits",
                title=f"{label}\n{row_title}",
            )
            axis.grid(color="#E6E6E6", linewidth=0.6)
    figure.tight_layout()
    save_figure(figure, output_dir, "09_paper_aligned_rank_size", args.dpi)


def plot_effective_diversity(args: argparse.Namespace, output_dir: Path) -> None:
    natural = pd.read_csv(args.comparison_root / "natural_paper_diversity.csv")
    matched = pd.read_csv(args.comparison_root / "paper_count_matched_summary.csv")
    figure, axes = plt.subplots(2, 3, figsize=(10.0, 6.8))
    colors = ("#4D4D4D", "#D55E00", "#E69F00")
    representations = (
        ("scaffoldtree_level1", "ScaffoldTree Level 1"),
        ("sphere_exclusion_tanimoto_0.55", "Tanimoto-0.55 clusters"),
    )
    for row_index, (representation, title) in enumerate(representations):
        current = natural[natural["representation"] == representation].set_index("workflow")
        matched_current = matched[matched["representation"] == representation].set_index("metric")
        for column_index, (metric, natural_column, ylabel) in enumerate(
            (
                ("q0", "q0_richness", "q0 richness"),
                ("q1", "q1_effective_families", "q1 effective families"),
                ("q2", "q2_effective_families", "q2 effective families"),
            )
        ):
            row = matched_current.loc[metric]
            fixed_name = str(row["fixed_workflow"])
            sampled_name = str(row["sampled_workflow"])
            labels = {
                args.left_name: args.left_name.title(),
                args.right_name: args.right_name.title(),
            }
            values = [
                current.loc[fixed_name, natural_column],
                current.loc[sampled_name, natural_column],
                row["sampled_median"],
            ]
            axis = axes[row_index, column_index]
            axis.bar([0, 1, 2], values, color=colors, width=0.68)
            axis.errorbar(
                2,
                row["sampled_median"],
                yerr=[
                    [row["sampled_median"] - row["sampled_interval_low"]],
                    [row["sampled_interval_high"] - row["sampled_median"]],
                ],
                fmt="none",
                color="black",
                capsize=3,
            )
            axis.set(
                xticks=[0, 1, 2],
                xticklabels=[
                    f"{labels[fixed_name]}\nfixed",
                    f"{labels[sampled_name]}\nnatural",
                    f"{labels[sampled_name]}\nmatched",
                ],
                ylabel=ylabel,
                title=title if column_index == 1 else None,
            )
            axis.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    figure.tight_layout()
    save_figure(figure, output_dir, "10_paper_aligned_effective_diversity", args.dpi)


def plot_centroids(args: argparse.Namespace, output_dir: Path) -> None:
    with np.load(args.common_coordinates, allow_pickle=False) as data:
        identifiers = data["spacehastenid"].astype(np.int64)
        coordinates = data["umap"].astype(np.float32)
    if coordinates.shape != (len(identifiers), 2) or not np.isfinite(coordinates).all():
        raise ValueError("common fixed-model coordinates are invalid")
    coordinate_map = pd.DataFrame(
        {"spacehastenid": identifiers, "umap_x": coordinates[:, 0], "umap_y": coordinates[:, 1]}
    )
    definition = json.loads(args.fixed_umap_definition.read_text())
    extent = tuple(float(value) for value in definition["extent"])
    workflow_data = []
    maximum_count = 1.0
    for root, label in ((args.left_root, args.left_label), (args.right_root, args.right_label)):
        assignments = pd.read_csv(
            root / "paper_aligned_assignments.csv.gz", usecols=["spacehastenid"]
        ).merge(coordinate_map, on="spacehastenid", validate="one_to_one")
        clusters = pd.read_csv(root / "sphere_exclusion_t055_clusters.csv").merge(
            coordinate_map,
            left_on="centroid_spacehastenid",
            right_on="spacehastenid",
            validate="one_to_one",
        )
        maximum_count = max(maximum_count, float(clusters["hit_count"].max()))
        workflow_data.append((assignments, clusters, label))

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    norm = LogNorm(vmin=1, vmax=maximum_count)
    scatter = None
    for axis, (assignments, clusters, label) in zip(axes, workflow_data, strict=True):
        axis.hexbin(
            assignments["umap_x"],
            assignments["umap_y"],
            gridsize=150,
            bins="log",
            extent=extent,
            mincnt=1,
            cmap="Greys",
            alpha=0.28,
        )
        clusters = clusters.sort_values("hit_count")
        scatter = axis.scatter(
            clusters["umap_x"],
            clusters["umap_y"],
            c=clusters["hit_count"],
            s=marker_sizes(clusters["hit_count"].to_numpy(), maximum_count),
            cmap="viridis",
            norm=norm,
            linewidths=0,
            alpha=0.82,
            rasterized=True,
        )
        axis.set(
            xlim=extent[:2],
            ylim=extent[2:],
            xlabel="Fixed-reference UMAP 1",
            ylabel="Fixed-reference UMAP 2",
            title=f"{label}\n{len(clusters):,} Tanimoto-0.55 centroids",
        )
    assert scatter is not None
    figure.colorbar(
        scatter,
        ax=axes.ravel().tolist(),
        label="Virtual hits per Tanimoto-0.55 cluster",
        shrink=0.88,
    )
    legend_counts = [value for value in (10, 100, 1_000) if value <= maximum_count]
    for value in legend_counts:
        axes[1].scatter(
            [],
            [],
            s=marker_sizes(np.asarray([value]), maximum_count)[0],
            color="#BDBDBD",
            label=f"{value:,}",
        )
    axes[1].legend(title="Hits per cluster", frameon=False, loc="upper right")
    figure.suptitle("Paper-aligned cluster centroids in the common fixed UMAP reference")
    save_figure(figure, output_dir, "11_paper_aligned_centroids", args.dpi)


def load_fingerprint_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        identifiers = data["spacehastenid"].astype(np.int64)
        words = data["words"].astype(np.dtype("<u8"), copy=True)
        popcounts = data["popcounts"].astype(np.int64, copy=True)
    if identifiers.ndim != 1 or words.shape != (len(identifiers), 16):
        raise ValueError("fingerprint cache must contain IDs and N x 16 words")
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("fingerprint cache IDs must be unique")
    maximum = int(identifiers.max())
    row_by_id = np.full(maximum + 1, -1, dtype=np.int64)
    row_by_id[identifiers] = np.arange(len(identifiers), dtype=np.int64)
    return words, popcounts, row_by_id


def cluster_distribution(
    common_ids: np.ndarray,
) -> tuple[dict[str, float | int | None], np.ndarray, float]:
    if _WORDS is None or _ROW_BY_ID is None or _INDEX_PATH is None:
        raise RuntimeError("paper comparison worker is not initialized")
    rows = _ROW_BY_ID[common_ids]
    if np.any(rows < 0):
        raise ValueError("sample IDs are absent from the fingerprint cache")
    fingerprints = [DataStructs.CreateFromBinaryText(row.tobytes()) for row in _WORDS[rows]]
    picker = rdSimDivPickers.LeaderPicker()
    centroid_rows = list(
        picker.LazyBitVectorPick(
            fingerprints,
            len(fingerprints),
            1.0 - _THRESHOLD,
        )
    )
    engine = FPSim2Engine(_INDEX_PATH)
    sample_position = np.full(len(_ROW_BY_ID), -1, dtype=np.int64)
    sample_position[common_ids] = np.arange(len(common_ids), dtype=np.int64)
    assignments = np.full(len(common_ids), -1, dtype=np.int32)
    best_similarity = np.zeros(len(common_ids), dtype=np.float32)
    for cluster_index, centroid_row in enumerate(centroid_rows):
        matches = engine.similarity(
            fingerprints[int(centroid_row)],
            threshold=_THRESHOLD,
            metric="tanimoto",
            n_workers=1,
        )
        match_ids = np.asarray(matches["mol_id"], dtype=np.int64)
        positions = sample_position[match_ids]
        included = positions >= 0
        positions = positions[included]
        similarities = np.asarray(matches["coeff"], dtype=np.float32)[included]
        better = similarities > best_similarity[positions]
        selected = positions[better]
        assignments[selected] = cluster_index
        best_similarity[selected] = similarities[better]
    if np.any(assignments < 0) or np.any(best_similarity < _THRESHOLD):
        raise RuntimeError("one or more sampled hits lack a Tanimoto-0.55 cluster assignment")
    return family_distribution(assignments), assignments, float(best_similarity.min())


def matched_worker(replicate: int) -> dict[str, Any]:
    if _RIGHT_IDS is None or _RIGHT_LEVEL1 is None:
        raise RuntimeError("matched worker inputs are not initialized")
    random = np.random.default_rng(_BASE_SEED + replicate)
    selected_rows = np.sort(random.choice(len(_RIGHT_IDS), size=_MATCHED_SIZE, replace=False))
    common_ids = _RIGHT_IDS[selected_rows]
    level1 = _RIGHT_LEVEL1[selected_rows]
    t055_metrics, t055_assignments, minimum_similarity = cluster_distribution(common_ids)
    assigned_level1 = level1[level1 != ""]
    row: dict[str, Any] = {
        "replicate": replicate,
        "sample_size": len(common_ids),
        "level1_assigned_hits": len(assigned_level1),
        "level1_unassigned_hits": int(np.count_nonzero(level1 == "")),
        "t055_assigned_hits": len(t055_assignments),
        "t055_minimum_similarity": minimum_similarity,
        "_common_ids": common_ids,
        "_t055_assignments": t055_assignments,
    }
    row.update(
        {f"level1_{name}": value for name, value in family_distribution(assigned_level1).items()}
    )
    row.update({f"t055_{name}": value for name, value in t055_metrics.items()})
    return row


def summarize_matched(
    replicates: pd.DataFrame,
    fixed_natural: pd.DataFrame,
    fixed_workflow: str,
    sampled_workflow: str,
) -> pd.DataFrame:
    representation = {
        "level1": "scaffoldtree_level1",
        "t055": "sphere_exclusion_tanimoto_0.55",
    }
    metrics = ("q0", "q1", "q2", "hhi", "largest_fraction", "top10_fraction")
    rows = []
    for prefix, representation_name in representation.items():
        fixed = fixed_natural[fixed_natural["representation"] == representation_name].iloc[0]
        for metric in metrics:
            values = replicates[f"{prefix}_{metric}"].dropna()
            rows.append(
                {
                    "representation": representation_name,
                    "metric": metric,
                    "fixed_workflow": fixed_workflow,
                    "fixed_complete_value": fixed[f"{metric}_richness"]
                    if metric == "q0"
                    else fixed[f"{metric}_effective_families"]
                    if metric in {"q1", "q2"}
                    else fixed[metric],
                    "sampled_workflow": sampled_workflow,
                    "sampled_median": values.median(),
                    "sampled_interval_low": values.quantile(0.025),
                    "sampled_interval_high": values.quantile(0.975),
                    "replicates": len(values),
                }
            )
    return pd.DataFrame(rows)


def analyze(args: argparse.Namespace) -> None:
    global _BASE_SEED, _INDEX_PATH, _MATCHED_SIZE
    global _RIGHT_IDS, _RIGHT_LEVEL1, _ROW_BY_ID, _THRESHOLD, _WORDS

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    natural = pd.DataFrame(
        [
            *natural_rows(args.left_root, args.left_name, args.left_label),
            *natural_rows(args.right_root, args.right_name, args.right_label),
        ]
    )
    natural.to_csv(root / "natural_paper_diversity.csv", index=False)

    words, popcounts, row_by_id = load_fingerprint_cache(args.fingerprints)
    identifiers = np.flatnonzero(row_by_id >= 0).astype(np.int64)
    validate_fpsim2_index(
        args.fp_index, identifiers, words[row_by_id[identifiers]], popcounts[row_by_id[identifiers]]
    )
    hit_counts = {
        name: int(
            natural[
                (natural["workflow"] == name)
                & (natural["representation"] == "sphere_exclusion_tanimoto_0.55")
            ].iloc[0]["total_hits"]
        )
        for name in (args.left_name, args.right_name)
    }
    fixed_workflow = min(hit_counts, key=hit_counts.get)
    sampled_workflow = max(hit_counts, key=hit_counts.get)
    if fixed_workflow == sampled_workflow:
        raise ValueError("paper count matching requires different natural hit counts")
    roots = {args.left_name: args.left_root, args.right_name: args.right_root}
    sampled_assignments = pd.read_csv(roots[sampled_workflow] / "paper_aligned_assignments.csv.gz")
    if not sampled_assignments["spacehastenid"].is_unique:
        raise ValueError("sampled paper assignments contain duplicate IDs")
    sampled_assignments = sampled_assignments.sort_values(
        ["reghash", "spacehastenid"], kind="stable"
    )
    fixed_hits = hit_counts[fixed_workflow]
    if fixed_hits > len(sampled_assignments):
        raise ValueError("fixed hit count exceeds the sampled hit cohort")

    _WORDS = words
    _ROW_BY_ID = row_by_id
    _INDEX_PATH = str(args.fp_index.resolve())
    _THRESHOLD = args.cluster_similarity
    _RIGHT_IDS = sampled_assignments["spacehastenid"].to_numpy(np.int64)
    _RIGHT_LEVEL1 = sampled_assignments["scaffoldtree_level1"].fillna("").to_numpy(str)
    _MATCHED_SIZE = fixed_hits
    _BASE_SEED = args.random_seed

    started = time.monotonic()
    context = mp.get_context("fork")
    rows = []
    sample_ids: dict[int, np.ndarray] = {}
    cluster_assignments: dict[int, np.ndarray] = {}
    with context.Pool(args.processes) as pool:
        iterator = pool.imap_unordered(matched_worker, range(args.replicates), chunksize=1)
        for row in tqdm(
            iterator,
            total=args.replicates,
            desc="paper-aligned count matching",
            unit="replicate",
            dynamic_ncols=True,
        ):
            replicate = int(row["replicate"])
            sample_ids[replicate] = row.pop("_common_ids")
            cluster_assignments[replicate] = row.pop("_t055_assignments")
            rows.append(row)
    replicates = pd.DataFrame(rows).sort_values("replicate")
    replicate_path = root / "paper_count_matched_replicates.csv"
    replicates.to_csv(replicate_path, index=False)
    sample_path = root / "paper_count_matched_sample_ids.npz"
    assignment_path = root / "paper_count_matched_cluster_assignments.npz"
    ordered_replicates = np.arange(args.replicates, dtype=np.int32)
    write_npz(
        sample_path,
        replicate=ordered_replicates,
        common_id=np.stack([sample_ids[int(index)] for index in ordered_replicates]),
    )
    write_npz(
        assignment_path,
        replicate=ordered_replicates,
        clusterid=np.stack([cluster_assignments[int(index)] for index in ordered_replicates]),
    )
    fixed_natural = natural[natural["workflow"] == fixed_workflow]
    summary = summarize_matched(
        replicates,
        fixed_natural,
        fixed_workflow,
        sampled_workflow,
    )
    summary_path = root / "paper_count_matched_summary.csv"
    summary.to_csv(summary_path, index=False)
    natural_path = root / "natural_paper_diversity.csv"
    receipt = {
        "status": "complete",
        "left_workflow": args.left_name,
        "right_workflow": args.right_name,
        "fixed_workflow": fixed_workflow,
        "sampled_workflow": sampled_workflow,
        "fixed_complete_hits": fixed_hits,
        "sample_size": fixed_hits,
        "replicates": args.replicates,
        "processes": args.processes,
        "random_seed": args.random_seed,
        "cluster_similarity": args.cluster_similarity,
        "elapsed_seconds": time.monotonic() - started,
        "sampled_level1_status": sampled_assignments["scaffoldtree_status"]
        .value_counts()
        .to_dict(),
        "inputs": [
            file_record(args.left_root / "_SUCCESS.json"),
            file_record(args.right_root / "_SUCCESS.json"),
            file_record(roots[sampled_workflow] / "paper_aligned_assignments.csv.gz"),
            file_record(args.fingerprints),
            file_record(args.fp_index),
        ],
        "outputs": [
            file_record(natural_path, root=root),
            file_record(replicate_path, root=root),
            file_record(sample_path, root=root),
            file_record(assignment_path, root=root),
            file_record(summary_path, root=root),
        ],
    }
    write_json(root / "_SUCCESS.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


def plot(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_rank_size(args, output_dir)
    plot_effective_diversity(args, output_dir)
    plot_centroids(args, output_dir)
    print(json.dumps({"status": "complete", "figures": 3}, sort_keys=True))


def finalize(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    receipt_path = root / "_SUCCESS.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "complete":
        raise ValueError("paper comparison receipt is not complete")
    roots = {
        str(receipt["left_workflow"]): args.left_root,
        str(receipt["right_workflow"]): args.right_root,
    }
    sampled_workflow = str(receipt["sampled_workflow"])
    outputs = [
        root / "natural_paper_diversity.csv",
        root / "paper_count_matched_replicates.csv",
        root / "paper_count_matched_sample_ids.npz",
        root / "paper_count_matched_cluster_assignments.npz",
        root / "paper_count_matched_summary.csv",
    ]
    receipt["inputs"] = [
        file_record(args.left_root / "_SUCCESS.json"),
        file_record(args.right_root / "_SUCCESS.json"),
        file_record(roots[sampled_workflow] / "paper_aligned_assignments.csv.gz"),
        file_record(args.fingerprints),
        file_record(args.fp_index),
    ]
    receipt["outputs"] = [file_record(path, root=root) for path in outputs]
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--left-root", type=Path, required=True)
    analyze_parser.add_argument("--right-root", type=Path, required=True)
    analyze_parser.add_argument("--left-name", required=True)
    analyze_parser.add_argument("--right-name", required=True)
    analyze_parser.add_argument("--left-label", required=True)
    analyze_parser.add_argument("--right-label", required=True)
    analyze_parser.add_argument("--fingerprints", type=Path, required=True)
    analyze_parser.add_argument("--fp-index", type=Path, required=True)
    analyze_parser.add_argument("--output-root", type=Path, required=True)
    analyze_parser.add_argument("--replicates", type=int, default=200)
    analyze_parser.add_argument("--processes", type=int, default=8)
    analyze_parser.add_argument("--random-seed", type=int, default=42)
    analyze_parser.add_argument("--cluster-similarity", type=float, default=0.55)
    analyze_parser.set_defaults(function=analyze)

    plot_parser = commands.add_parser("plot")
    plot_parser.add_argument("--left-root", type=Path, required=True)
    plot_parser.add_argument("--right-root", type=Path, required=True)
    plot_parser.add_argument("--left-name", required=True)
    plot_parser.add_argument("--right-name", required=True)
    plot_parser.add_argument("--left-label", required=True)
    plot_parser.add_argument("--right-label", required=True)
    plot_parser.add_argument("--comparison-root", type=Path, required=True)
    plot_parser.add_argument("--common-coordinates", type=Path, required=True)
    plot_parser.add_argument("--fixed-umap-definition", type=Path, required=True)
    plot_parser.add_argument("--output-dir", type=Path, required=True)
    plot_parser.add_argument("--dpi", type=int, default=600)
    plot_parser.set_defaults(function=plot)

    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--left-root", type=Path, required=True)
    finalize_parser.add_argument("--right-root", type=Path, required=True)
    finalize_parser.add_argument("--fingerprints", type=Path, required=True)
    finalize_parser.add_argument("--fp-index", type=Path, required=True)
    finalize_parser.add_argument("--output-root", type=Path, required=True)
    finalize_parser.set_defaults(function=finalize)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for path in (args.left_root, args.right_root):
        if not (path / "paper_aligned_metrics.csv").is_file():
            parser.error(f"paper diversity root is incomplete: {path}")
    if args.command == "analyze":
        if min(args.replicates, args.processes) < 1:
            parser.error("replicates and processes must be positive")
        if not 0 < args.cluster_similarity <= 1:
            parser.error("cluster similarity must be in (0, 1]")
        inputs = (args.fingerprints, args.fp_index)
    elif args.command == "plot":
        if args.dpi < 1:
            parser.error("dpi must be positive")
        inputs = (
            args.comparison_root / "natural_paper_diversity.csv",
            args.comparison_root / "paper_count_matched_summary.csv",
            args.common_coordinates,
            args.fixed_umap_definition,
        )
    else:
        inputs = (
            args.output_root / "_SUCCESS.json",
            args.left_root / "_SUCCESS.json",
            args.right_root / "_SUCCESS.json",
            args.fingerprints,
            args.fp_index,
        )
    for path in inputs:
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
    args.function(args)


if __name__ == "__main__":
    main()
