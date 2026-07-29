#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Reproduce the original SpaceHASTEN paper's virtual-hit diversity endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import rdkit
import scaffoldgraph
from FPSim2 import FPSim2Engine
from matplotlib.colors import LogNorm
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from scaffoldgraph import tree_frags_from_mol
from tqdm import tqdm

from calculate_hit_diversity_metrics import sphere_exclusion_clusters
from spacehasten.analysis.umap import rdkit_words_to_fpsim2_words

REPRESENTATIONS = ("scaffoldtree_level1", "sphere_exclusion_tanimoto_0.55")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_boolean(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false", "1", "0"}).all():
        raise ValueError("boolean manifest columns contain unsupported values")
    return normalized.isin({"true", "1"})


def load_hits(manifest_path: Path, hit_threshold: float) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {
        "spacehastenid",
        "reghash",
        "smiles",
        "round",
        "dock_score",
        "is_hit",
        "is_strict_hit",
    }
    if missing := required - set(manifest.columns):
        raise ValueError(f"selected manifest lacks columns: {sorted(missing)}")
    manifest["is_hit"] = as_boolean(manifest["is_hit"])
    manifest["is_strict_hit"] = as_boolean(manifest["is_strict_hit"])
    if not manifest["spacehastenid"].is_unique:
        raise ValueError("paper-aligned analysis requires unique selected compounds")
    if manifest["reghash"].isna().any() or not manifest["reghash"].is_unique:
        raise ValueError("selected manifest reghashes must be non-null and unique")
    hits = manifest[manifest["is_hit"]].copy()
    if hits.empty:
        raise ValueError("selected manifest contains no virtual hits")
    if hits["dock_score"].isna().any() or not hits["dock_score"].le(hit_threshold).all():
        raise ValueError("manifest hit labels disagree with the requested docking cutoff")
    return hits.sort_values(["reghash", "spacehastenid"], kind="stable").reset_index(drop=True)


def load_fingerprints(path: Path, hits: pd.DataFrame) -> tuple[list[Any], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        identifiers = data["spacehastenid"].astype(np.int64)
        words = data["words"].astype(np.dtype("<u8"), copy=True)
        popcounts = data["popcounts"].astype(np.uint16, copy=True)
    if identifiers.ndim != 1 or words.shape != (len(identifiers), 16):
        raise ValueError("fingerprint cache must contain IDs and N x 16 Morgan words")
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("fingerprint cache contains duplicate identifiers")
    row_by_id = pd.Series(np.arange(len(identifiers)), index=identifiers)
    hit_ids = hits["spacehastenid"].to_numpy(np.int64)
    if not np.isin(hit_ids, identifiers).all():
        raise ValueError("fingerprint cache does not cover every virtual hit")
    positions = row_by_id.loc[hit_ids].to_numpy(np.int64)
    hit_words = words[positions]
    hit_popcounts = popcounts[positions]
    fingerprints = [
        DataStructs.CreateFromBinaryText(row.tobytes())
        for row in tqdm(hit_words, desc="loading cached hit fingerprints", unit="hit")
    ]
    if any(
        fp.GetNumOnBits() != int(count)
        for fp, count in zip(fingerprints, hit_popcounts, strict=True)
    ):
        raise ValueError("cached fingerprint words and popcounts disagree")
    return fingerprints, hit_words, hit_popcounts


def scaffoldtree_level1(smiles: str) -> tuple[str | None, str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit failed to parse selected SMILES: {smiles!r}")
    if rdMolDescriptors.CalcNumRings(molecule) == 0:
        return None, "acyclic"
    fragments = tree_frags_from_mol(molecule)
    level_one = [fragment for fragment in fragments if rdMolDescriptors.CalcNumRings(fragment) == 1]
    if len(level_one) > 1:
        raise ValueError(f"ScaffoldGraph returned multiple Level 1 parents for {smiles!r}")
    if not level_one:
        return None, "no_level1_parent"
    return (
        Chem.MolToSmiles(level_one[0], canonical=True, isomericSmiles=False),
        "assigned",
    )


def scaffoldtree_labels(smiles: list[str], processes: int) -> list[tuple[str | None, str]]:
    if processes == 1:
        return [
            scaffoldtree_level1(value)
            for value in tqdm(smiles, desc="ScaffoldTree Level 1", unit="hit")
        ]
    chunksize = max(1, len(smiles) // (processes * 16))
    with mp.Pool(processes) as pool:
        return list(
            tqdm(
                pool.imap(scaffoldtree_level1, smiles, chunksize=chunksize),
                total=len(smiles),
                desc="ScaffoldTree Level 1",
                unit="hit",
            )
        )


def diversity(values: pd.Series) -> dict[str, float | int]:
    counts = values.dropna().value_counts().to_numpy(np.int64)
    assigned = int(counts.sum())
    if assigned == 0:
        return {
            "assigned_hits": 0,
            "q0_richness": 0,
            "q1_effective_families": 0.0,
            "q2_effective_families": 0.0,
            "hhi": math.nan,
            "largest_fraction": math.nan,
            "top10_fraction": math.nan,
            "shannon_entropy": math.nan,
        }
    proportions = counts / assigned
    entropy = float(-np.sum(proportions * np.log(proportions)))
    hhi = float(np.sum(proportions**2))
    return {
        "assigned_hits": assigned,
        "q0_richness": len(counts),
        "q1_effective_families": math.exp(entropy),
        "q2_effective_families": 1 / hhi,
        "hhi": hhi,
        "largest_fraction": float(proportions.max()),
        "top10_fraction": float(np.sort(proportions)[-10:].sum()),
        "shannon_entropy": entropy,
    }


def metric_rows(frame: pd.DataFrame) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    round_ids = sorted(frame["round"].astype(int).unique())
    for scope in ("round", "cumulative"):
        for round_id in round_ids:
            current = frame[frame["round"] == round_id]
            if scope == "cumulative":
                current = frame[frame["round"] <= round_id]
            for representation, column in zip(
                REPRESENTATIONS,
                ("scaffoldtree_level1", "sphere_exclusion_cluster"),
                strict=True,
            ):
                metrics = diversity(current[column])
                rows.append(
                    {
                        "scope": scope,
                        "round": round_id,
                        "representation": representation,
                        "total_hits": len(current),
                        "unassigned_hits": len(current) - int(metrics["assigned_hits"]),
                        **metrics,
                    }
                )
    return rows


def family_summary(frame: pd.DataFrame, column: str, name: str) -> pd.DataFrame:
    assigned = frame[frame[column].notna()]
    result = (
        assigned.groupby(column, sort=False)
        .agg(
            hit_count=("spacehastenid", "size"),
            strict_hits=("is_strict_hit", "sum"),
            first_round=("round", "min"),
            rounds=("round", "nunique"),
            median_dock_score=("dock_score", "median"),
        )
        .reset_index()
        .rename(columns={column: name})
    )
    result["hit_fraction"] = result["hit_count"] / len(assigned)
    return result.sort_values(["hit_count", name], ascending=[False, True], kind="stable")


def save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_rank_size(
    scaffold_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    output_root: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    panels = (
        (scaffold_summary["hit_count"], "ScaffoldTree Level 1 families"),
        (cluster_summary["hit_count"], "Tanimoto-0.55 sphere-exclusion clusters"),
    )
    for axis, (counts, title) in zip(axes, panels, strict=True):
        ordered = np.sort(counts.to_numpy(np.int64))[::-1]
        axis.plot(np.arange(1, len(ordered) + 1), ordered, linewidth=1.2)
        axis.set(
            xscale="log",
            yscale="log",
            xlabel="Family or cluster rank",
            ylabel="Virtual hits",
            title=title,
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#E6E6E6", linewidth=0.6)
    figure.tight_layout()
    save_figure(figure, output_root / "figures" / "01_paper_aligned_rank_size", dpi)


def plot_effective_diversity(metrics: pd.DataFrame, output_root: Path, dpi: int) -> None:
    cumulative = metrics[metrics["scope"] == "cumulative"]
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    for axis, representation in zip(axes, REPRESENTATIONS, strict=True):
        current = cumulative[cumulative["representation"] == representation]
        for column, label, marker in (
            ("q0_richness", "q0 richness", "o"),
            ("q1_effective_families", "q1 effective families", "s"),
            ("q2_effective_families", "q2 effective families", "^"),
        ):
            axis.plot(current["round"], current[column], marker=marker, label=label)
        title = (
            "ScaffoldTree Level 1"
            if representation == "scaffoldtree_level1"
            else "Tanimoto-0.55 clusters"
        )
        axis.set(
            yscale="log",
            xlabel="Cumulative round",
            ylabel="Number of families or clusters",
            title=title,
            xticks=current["round"],
        )
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="#E6E6E6", linewidth=0.6)
    figure.tight_layout()
    save_figure(figure, output_root / "figures" / "02_paper_aligned_effective_diversity", dpi)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def validate_fpsim2_index(
    path: Path,
    identifiers: np.ndarray,
    words: np.ndarray,
    popcounts: np.ndarray,
) -> None:
    engine = FPSim2Engine(str(path))
    index_ids = np.asarray(engine.fps[:, 0], dtype=np.int64)
    index_order = np.argsort(index_ids)
    cache_order = np.argsort(identifiers)
    if not np.array_equal(index_ids[index_order], identifiers[cache_order]):
        raise ValueError("FPSim2 index IDs differ from the cached virtual-hit IDs")
    expected_words = rdkit_words_to_fpsim2_words(words[cache_order])
    observed_words = np.asarray(engine.fps[index_order, 1:-1], dtype=np.uint64)
    observed_popcounts = np.asarray(engine.fps[index_order, -1], dtype=np.uint16)
    if not np.array_equal(observed_words, expected_words) or not np.array_equal(
        observed_popcounts, popcounts[cache_order]
    ):
        raise ValueError("FPSim2 index fingerprints differ from the validated cache")


def canonical_outputs(root: Path) -> list[dict[str, Any]]:
    outputs = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "_SUCCESS.json" or "paper_hits_fp.h5" in path.name:
            continue
        outputs.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return outputs


def write_receipt(root: Path, metadata: dict[str, Any]) -> None:
    receipt = {**metadata, "status": "complete", "outputs": canonical_outputs(root)}
    (root / "_SUCCESS.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def analyze(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output root is non-empty; pass --overwrite: {root}")
        shutil.rmtree(root)
    (root / "figures").mkdir(parents=True)
    hits = load_hits(args.manifest, args.hit_threshold)
    fingerprints, words, popcounts = load_fingerprints(args.fingerprints, hits)
    scaffold_results = scaffoldtree_labels(hits["smiles"].astype(str).tolist(), args.processes)
    hits["scaffoldtree_level1"] = [result[0] for result in scaffold_results]
    hits["scaffoldtree_status"] = [result[1] for result in scaffold_results]

    cluster_rows = list(
        zip(
            hits["spacehastenid"].astype(int),
            hits["reghash"].astype(str),
            hits["smiles"].astype(str),
            strict=True,
        )
    )
    assignments, similarities, centroid_ids = sphere_exclusion_clusters(
        cluster_rows,
        fingerprints,
        root,
        args.cluster_similarity,
    )
    index_path = root / "virtual_hits_morgan_r2_1024.h5"
    validate_fpsim2_index(
        index_path,
        hits["spacehastenid"].to_numpy(np.int64),
        words,
        popcounts,
    )
    index_path.unlink(missing_ok=True)
    centroid_array = np.asarray(centroid_ids, dtype=np.int64)
    hits["sphere_exclusion_cluster"] = centroid_array[assignments]
    hits["sphere_exclusion_centroid"] = centroid_array[assignments]
    hits["centroid_tanimoto"] = similarities

    scaffold_summary = family_summary(hits, "scaffoldtree_level1", "scaffoldtree_level1")
    cluster_summary = family_summary(hits, "sphere_exclusion_cluster", "centroid_spacehastenid")
    centroid_positions = pd.Series(
        np.arange(len(hits)), index=hits["spacehastenid"].to_numpy(np.int64)
    ).loc[centroid_array]
    write_npz(
        root / "t055_centroid_fingerprints.npz",
        spacehastenid=centroid_array,
        words=words[centroid_positions.to_numpy(np.int64)],
        popcounts=popcounts[centroid_positions.to_numpy(np.int64)],
    )

    assignment_columns = [
        "spacehastenid",
        "reghash",
        "round",
        "dock_score",
        "is_strict_hit",
        "scaffoldtree_level1",
        "scaffoldtree_status",
        "sphere_exclusion_cluster",
        "sphere_exclusion_centroid",
        "centroid_tanimoto",
    ]
    hits[assignment_columns].to_csv(root / "paper_aligned_assignments.csv.gz", index=False)
    scaffold_summary.to_csv(root / "scaffoldtree_level1_families.csv", index=False)
    cluster_summary.to_csv(root / "sphere_exclusion_t055_clusters.csv", index=False)
    metrics = pd.DataFrame(metric_rows(hits))
    metrics.to_csv(root / "paper_aligned_metrics.csv", index=False)
    plot_rank_size(scaffold_summary, cluster_summary, root, args.dpi)
    plot_effective_diversity(metrics, root, args.dpi)

    final_round = int(hits["round"].max())
    final_metrics = metrics[(metrics["scope"] == "cumulative") & (metrics["round"] == final_round)]
    metadata = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "fingerprints": str(args.fingerprints.resolve()),
        "fingerprints_sha256": sha256(args.fingerprints),
        "hit_threshold": args.hit_threshold,
        "cluster_similarity": args.cluster_similarity,
        "cluster_method": "RDKit LeaderPicker sphere exclusion with FPSim2 assignment",
        "cluster_order": "reghash then spacehastenid",
        "fingerprint": {"type": "Morgan", "radius": 2, "n_bits": 1024},
        "fpsim2_index_matches_cached_fingerprints": True,
        "scaffold_method": "ScaffoldGraph ScaffoldTree Level 1",
        "scaffoldgraph_version": scaffoldgraph.__version__,
        "rdkit_version": rdkit.__version__,
        "virtual_hits": len(hits),
        "rounds": sorted(hits["round"].astype(int).unique().tolist()),
        "level1_assigned_hits": int(hits["scaffoldtree_level1"].notna().sum()),
        "level1_assignment_status": {
            str(status): int(count)
            for status, count in hits["scaffoldtree_status"].value_counts().items()
        },
        "sphere_exclusion_clusters": len(centroid_ids),
        "final_metrics": final_metrics.to_dict(orient="records"),
    }
    write_receipt(root, metadata)
    print(json.dumps(metadata, sort_keys=True))


def coordinates(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as data:
        identifiers = data["spacehastenid"].astype(np.int64)
        values = data["umap"].astype(np.float32)
    if values.shape != (len(identifiers), 2) or not np.isfinite(values).all():
        raise ValueError(f"invalid fixed-model UMAP coordinates: {path}")
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate UMAP identifiers: {path}")
    return pd.DataFrame(
        {"spacehastenid": identifiers, "umap_x": values[:, 0], "umap_y": values[:, 1]}
    )


def marker_sizes(counts: np.ndarray, maximum: float = 70.0) -> np.ndarray:
    transformed = np.sqrt(np.asarray(counts, dtype=float))
    return 5.0 + maximum * transformed / transformed.max()


def plot_umap(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    receipt_path = root / "_SUCCESS.json"
    if not receipt_path.is_file():
        raise FileNotFoundError("paper-aligned metrics must be complete before UMAP plotting")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assignments = pd.read_csv(root / "paper_aligned_assignments.csv.gz")
    clusters = pd.read_csv(root / "sphere_exclusion_t055_clusters.csv")
    selected = coordinates(args.selected_coordinates)
    seeds = coordinates(args.seed_coordinates)
    centroids = coordinates(args.centroid_coordinates)
    hit_coordinates = assignments[["spacehastenid"]].merge(
        selected, on="spacehastenid", validate="one_to_one"
    )
    if len(hit_coordinates) != len(assignments):
        raise ValueError("selected fixed-model coordinates do not cover every virtual hit")
    centroid_frame = clusters.merge(
        centroids,
        left_on="centroid_spacehastenid",
        right_on="spacehastenid",
        validate="one_to_one",
    )
    if len(centroid_frame) != len(clusters):
        raise ValueError("fixed-model coordinates do not cover every Tanimoto-0.55 centroid")
    extent = (
        min(float(seeds["umap_x"].min()), float(hit_coordinates["umap_x"].min())),
        max(float(seeds["umap_x"].max()), float(hit_coordinates["umap_x"].max())),
        min(float(seeds["umap_y"].min()), float(hit_coordinates["umap_y"].min())),
        max(float(seeds["umap_y"].max()), float(hit_coordinates["umap_y"].max())),
    )
    counts = centroid_frame["hit_count"].to_numpy(float)
    sizes = marker_sizes(counts)
    norm = LogNorm(vmin=1, vmax=max(float(counts.max()), 2.0))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    backgrounds = (
        (seeds, "Starting seeds"),
        (hit_coordinates, "Virtual hits"),
    )
    scatter = None
    for axis, (background, title) in zip(axes, backgrounds, strict=True):
        axis.hexbin(
            background["umap_x"],
            background["umap_y"],
            gridsize=140,
            bins="log",
            extent=extent,
            mincnt=1,
            cmap="Greys",
            alpha=0.55,
        )
        scatter = axis.scatter(
            centroid_frame["umap_x"],
            centroid_frame["umap_y"],
            c=counts,
            s=sizes,
            cmap="viridis",
            norm=norm,
            linewidths=0,
            alpha=0.82,
            rasterized=True,
        )
        axis.set(
            xlabel="Fixed-reference UMAP 1",
            ylabel="Fixed-reference UMAP 2",
            title=title,
        )
    assert scatter is not None
    figure.colorbar(
        scatter,
        ax=axes.ravel().tolist(),
        label="Virtual hits per Tanimoto-0.55 cluster",
        shrink=0.9,
    )
    figure.suptitle("Paper-aligned cluster centroids in the fixed UMAP reference")
    save_figure(figure, root / "figures" / "03_t055_centroids_fixed_umap", args.dpi)

    receipt.update(
        {
            "selected_coordinates": str(args.selected_coordinates.resolve()),
            "selected_coordinates_sha256": sha256(args.selected_coordinates),
            "seed_coordinates": str(args.seed_coordinates.resolve()),
            "seed_coordinates_sha256": sha256(args.seed_coordinates),
            "centroid_coordinates": str(args.centroid_coordinates.resolve()),
            "centroid_coordinates_sha256": sha256(args.centroid_coordinates),
            "umap_interpretation": (
                "Existing fixed-model visualization only; UMAP distance does not define the "
                "Tanimoto-0.55 clusters."
            ),
        }
    )
    receipt.pop("outputs", None)
    write_receipt(root, receipt)
    print(json.dumps({"status": "complete", "centroids": len(centroid_frame)}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--manifest", type=Path, required=True)
    analyze_parser.add_argument("--fingerprints", type=Path, required=True)
    analyze_parser.add_argument("--output-root", type=Path, required=True)
    analyze_parser.add_argument("--hit-threshold", type=float, required=True)
    analyze_parser.add_argument("--cluster-similarity", type=float, default=0.55)
    analyze_parser.add_argument("--processes", type=int, default=1)
    analyze_parser.add_argument("--dpi", type=int, default=600)
    analyze_parser.add_argument("--overwrite", action="store_true")
    analyze_parser.set_defaults(function=analyze)

    plot_parser = subparsers.add_parser("plot-umap")
    plot_parser.add_argument("--output-root", type=Path, required=True)
    plot_parser.add_argument("--selected-coordinates", type=Path, required=True)
    plot_parser.add_argument("--seed-coordinates", type=Path, required=True)
    plot_parser.add_argument("--centroid-coordinates", type=Path, required=True)
    plot_parser.add_argument("--dpi", type=int, default=600)
    plot_parser.set_defaults(function=plot_umap)
    return root


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args()
    if args.dpi < 1:
        argument_parser.error("dpi must be positive")
    if args.command == "analyze":
        if args.processes < 1 or not 0 < args.cluster_similarity <= 1:
            argument_parser.error("processes must be positive and cluster similarity in (0, 1]")
        for path in (args.manifest, args.fingerprints):
            if not path.is_file():
                argument_parser.error(f"input does not exist: {path}")
    else:
        for path in (
            args.selected_coordinates,
            args.seed_coordinates,
            args.centroid_coordinates,
        ):
            if not path.is_file():
                argument_parser.error(f"input does not exist: {path}")
    args.function(args)


if __name__ == "__main__":
    main()
