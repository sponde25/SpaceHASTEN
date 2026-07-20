#!/usr/bin/env python3
# ruff: noqa: E501
"""Create reusable chemical-space figures from SpaceHASTEN analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from tqdm import tqdm

BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#777777"
DIVERGING = LinearSegmentedColormap.from_list("spacehasten_blue_orange", [BLUE, "#F7F7F7", ORANGE])


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_database(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    count = int(
        connection.execute(
            "SELECT COUNT(*) FROM data WHERE dock_score IS NOT NULL AND dock_iteration IS NOT NULL"
        ).fetchone()[0]
    )
    identifiers = np.empty(count, dtype=np.uint64)
    scores = np.empty(count, dtype=np.float32)
    iterations = np.empty(count, dtype=np.int16)
    query = (
        "SELECT spacehastenid, dock_score, dock_iteration FROM data "
        "WHERE dock_score IS NOT NULL AND dock_iteration IS NOT NULL "
        "ORDER BY spacehastenid"
    )
    for index, (identifier, score, iteration) in enumerate(
        tqdm(connection.execute(query), total=count, desc="Loading docking metadata")
    ):
        identifiers[index] = int(identifier)
        scores[index] = float(score)
        iterations[index] = int(iteration)
    connection.close()
    return identifiers, scores, iterations


def load_clusters(path: Path, expected_count: int) -> tuple[np.ndarray, np.ndarray]:
    identifiers = np.empty(expected_count, dtype=np.uint64)
    cluster_ids = np.empty(expected_count, dtype=np.uint64)
    with path.open("rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row_count = 0
        for index, row in enumerate(
            tqdm(reader, total=expected_count, desc="Loading cluster assignments")
        ):
            if index >= expected_count:
                raise ValueError(f"cluster assignments contain more than {expected_count} rows")
            identifiers[index] = int(row["spacehastenid"])
            cluster_ids[index] = int(row["clusterid"])
            row_count = index + 1
    if row_count != expected_count:
        raise ValueError(f"cluster assignments contain {row_count} rows; expected {expected_count}")
    order = np.argsort(identifiers)
    return identifiers[order], cluster_ids[order]


def load_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["spacehastenid"].copy(), data["umap"].copy()


def shared_extent(coordinates: np.ndarray) -> tuple[float, float, float, float]:
    x_min, y_min = coordinates.min(axis=0)
    x_max, y_max = coordinates.max(axis=0)
    x_pad = max((x_max - x_min) * 0.03, 0.1)
    y_pad = max((y_max - y_min) * 0.03, 0.1)
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_density(
    coordinates: np.ndarray,
    is_seed: np.ndarray,
    is_hit: np.ndarray,
    extent: tuple[float, float, float, float],
    output_dir: Path,
    label: str,
    dpi: int,
    bins: int,
) -> None:
    x_range = (extent[0], extent[1])
    y_range = (extent[2], extent[3])
    seed_density, _, _ = np.histogram2d(
        coordinates[is_seed, 0],
        coordinates[is_seed, 1],
        bins=bins,
        range=(x_range, y_range),
    )
    hit_density, _, _ = np.histogram2d(
        coordinates[is_hit, 0],
        coordinates[is_hit, 1],
        bins=bins,
        range=(x_range, y_range),
    )
    seed_layer = np.log1p(seed_density.T)
    hit_layer = np.log1p(hit_density.T)
    seed_layer /= max(seed_layer.max(), 1.0)
    hit_layer /= max(hit_layer.max(), 1.0)

    image = np.ones((*seed_layer.shape, 3), dtype=np.float32)
    for layer, color, strength in (
        (seed_layer, matplotlib.colors.to_rgb(BLUE), 0.85),
        (hit_layer, matplotlib.colors.to_rgb(ORANGE), 0.95),
    ):
        alpha = (strength * layer)[..., None]
        image = image * (1.0 - alpha) + np.asarray(color) * alpha

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.imshow(image, origin="lower", extent=extent, aspect="auto", interpolation="nearest")
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title=f"{label}: seeds and virtual hits")
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, label=f"Seeds (n={is_seed.sum():,})"),
            Patch(facecolor=ORANGE, label=f"Virtual hits (n={is_hit.sum():,})"),
        ],
        frameon=False,
        loc="best",
    )
    ax.text(
        0.01,
        0.01,
        "All compounds shown; log density normalized independently by cohort",
        transform=ax.transAxes,
        fontsize=7,
        color="#444444",
    )
    save_figure(fig, output_dir, "01_seed_virtual_hit_density", dpi)


def marker_sizes(counts: np.ndarray, maximum: float = 80.0) -> np.ndarray:
    transformed = np.sqrt(np.asarray(counts, dtype=float))
    if transformed.max() == 0:
        return np.full(len(counts), 5.0)
    return 4.0 + maximum * transformed / transformed.max()


def cluster_statistics(
    compound_ids: np.ndarray,
    cluster_for_compound: np.ndarray,
    coordinates: np.ndarray,
    is_seed: np.ndarray,
    is_hit: np.ndarray,
) -> dict[str, np.ndarray]:
    cluster_ids, inverse = np.unique(cluster_for_compound, return_inverse=True)
    seed_count = np.bincount(inverse, weights=is_seed.astype(int)).astype(int)
    virtual_count = np.bincount(inverse, weights=(~is_seed).astype(int)).astype(int)
    hit_count = np.bincount(inverse, weights=is_hit.astype(int)).astype(int)
    centroid_positions = np.searchsorted(compound_ids, cluster_ids)
    if np.any(centroid_positions == len(compound_ids)) or not np.array_equal(
        compound_ids[centroid_positions], cluster_ids
    ):
        raise ValueError("one or more cluster centroids are absent from the compounds")
    centroid_is_seed = is_seed[centroid_positions]
    return {
        "clusterid": cluster_ids,
        "seed_count": seed_count,
        "virtual_count": virtual_count,
        "hit_count": hit_count,
        "centroid_is_seed": centroid_is_seed,
        "x": coordinates[centroid_positions, 0],
        "y": coordinates[centroid_positions, 1],
    }


def plot_acquisition_shift(
    stats: dict[str, np.ndarray],
    extent: tuple[float, float, float, float],
    output_dir: Path,
    label: str,
    dpi: int,
) -> np.ndarray:
    seed_count = stats["seed_count"]
    virtual_count = stats["virtual_count"]
    cluster_count = len(seed_count)
    alpha = 0.5
    seed_probability = (seed_count + alpha) / (seed_count.sum() + alpha * cluster_count)
    virtual_probability = (virtual_count + alpha) / (virtual_count.sum() + alpha * cluster_count)
    shift = np.log2(virtual_probability / seed_probability)
    limit = max(float(np.quantile(np.abs(shift), 0.98)), 1.0)
    novel = ~stats["centroid_is_seed"]

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    scatter = ax.scatter(
        stats["x"],
        stats["y"],
        c=shift,
        s=marker_sizes(seed_count + virtual_count),
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        linewidths=0,
        alpha=0.85,
        rasterized=True,
    )
    ax.scatter(
        stats["x"][novel],
        stats["y"][novel],
        s=marker_sizes((seed_count + virtual_count)[novel]),
        facecolors="none",
        edgecolors="black",
        linewidths=0.6,
        rasterized=True,
    )
    ax.set(
        xlim=extent[:2],
        ylim=extent[2:],
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        title=f"{label}: acquisition shift relative to seeds",
    )
    fig.colorbar(scatter, ax=ax, label="log2 normalized virtual-docked / seed representation")
    ax.text(
        0.01,
        0.01,
        "Marker size: cluster population; black outline: virtual-derived centroid",
        transform=ax.transAxes,
        fontsize=7,
    )
    save_figure(fig, output_dir, "02_acquisition_shift", dpi)
    return shift


def plot_hit_enrichment(
    stats: dict[str, np.ndarray],
    extent: tuple[float, float, float, float],
    output_dir: Path,
    label: str,
    dpi: int,
    prior_strength: float,
) -> np.ndarray:
    virtual_count = stats["virtual_count"]
    hit_count = stats["hit_count"]
    global_rate = hit_count.sum() / virtual_count.sum()
    posterior_rate = (hit_count + prior_strength * global_rate) / (virtual_count + prior_strength)
    difference = 100.0 * (posterior_rate - global_rate)
    present = virtual_count > 0
    novel = (~stats["centroid_is_seed"]) & present
    limit = max(float(np.quantile(np.abs(difference[present]), 0.98)), 1.0)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    scatter = ax.scatter(
        stats["x"][present],
        stats["y"][present],
        c=difference[present],
        s=marker_sizes(virtual_count[present]),
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        linewidths=0,
        alpha=0.85,
        rasterized=True,
    )
    ax.scatter(
        stats["x"][novel],
        stats["y"][novel],
        s=marker_sizes(virtual_count[novel]),
        facecolors="none",
        edgecolors="black",
        linewidths=0.6,
        rasterized=True,
    )
    ax.set(
        xlim=extent[:2],
        ylim=extent[2:],
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        title=f"{label}: local virtual-hit enrichment",
    )
    fig.colorbar(
        scatter, ax=ax, label="Posterior hit-rate difference from global (percentage points)"
    )
    ax.text(
        0.01,
        0.01,
        f"Global virtual hit rate: {100 * global_rate:.1f}%; marker size: virtual docked count",
        transform=ax.transAxes,
        fontsize=7,
    )
    save_figure(fig, output_dir, "03_cluster_hit_enrichment", dpi)
    return posterior_rate


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    return x, np.arange(1, len(x) + 1) / len(x)


def plot_nearest_seed(
    similarities: np.ndarray,
    is_hit: np.ndarray,
    iterations: np.ndarray,
    output_dir: Path,
    label: str,
    dpi: int,
) -> None:
    groups: list[tuple[str, np.ndarray, object, str]] = [("Virtual non-hits", ~is_hit, GRAY, "--")]
    hit_iterations = np.unique(iterations[is_hit])
    color_map = plt.get_cmap("autumn")
    for index, iteration in enumerate(hit_iterations):
        color = color_map((index + 0.35) / max(len(hit_iterations), 1))
        groups.append(
            (
                f"Iteration {int(iteration)} hits",
                is_hit & (iterations == iteration),
                color,
                "-",
            )
        )
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for name, mask, color, linestyle in groups:
        values = similarities[mask]
        if len(values) == 0:
            continue
        x, y = ecdf(values)
        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=f"{name} (n={len(values):,}, median={np.median(values):.3f})",
        )
    for threshold in (0.3, 0.5, 0.7):
        ax.axvline(threshold, color="#AAAAAA", linewidth=0.8, linestyle=":")
    ax.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        xlabel="Maximum Tanimoto similarity to any seed",
        ylabel="Cumulative fraction",
        title=f"{label}: nearest-seed structural similarity",
    )
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.01,
        0.98,
        "Curves farther left indicate greater structural novelty",
        transform=ax.transAxes,
        va="top",
        fontsize=7,
    )
    save_figure(fig, output_dir, "04_nearest_seed_similarity_ecdf", dpi)


def write_cluster_table(
    path: Path,
    stats: dict[str, np.ndarray],
    shift: np.ndarray,
    posterior_rate: np.ndarray,
) -> None:
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "clusterid",
                "centroid_origin",
                "seed_count",
                "virtual_docked_count",
                "hit_count",
                "raw_hit_rate",
                "posterior_hit_rate",
                "acquisition_log2_shift",
                "umap1",
                "umap2",
            ]
        )
        for index in range(len(stats["clusterid"])):
            virtual_count = int(stats["virtual_count"][index])
            writer.writerow(
                [
                    int(stats["clusterid"][index]),
                    "seed" if stats["centroid_is_seed"][index] else "virtual",
                    int(stats["seed_count"][index]),
                    virtual_count,
                    int(stats["hit_count"][index]),
                    (float(stats["hit_count"][index]) / virtual_count if virtual_count else ""),
                    float(posterior_rate[index]),
                    float(shift[index]),
                    float(stats["x"][index]),
                    float(stats["y"][index]),
                ]
            )


def percentage_below(values: np.ndarray, threshold: float) -> float:
    return 100.0 * float(np.mean(values < threshold))


def write_summary(
    path: Path,
    figure_dir: Path,
    stats: dict[str, np.ndarray],
    similarities: np.ndarray,
    virtual_is_hit: np.ndarray,
    virtual_iterations: np.ndarray,
    seed_count: int,
    seed_hit_count: int,
    virtual_count: int,
    virtual_hit_count: int,
    cutoff: float,
    label: str,
    cluster_similarity: float,
    prior_strength: float,
    umap_neighbors: int,
    umap_min_dist: float,
    umap_random_seed: int,
) -> None:
    seed_rate = seed_hit_count / seed_count
    virtual_rate = virtual_hit_count / virtual_count
    novel = ~stats["centroid_is_seed"]
    novel_virtual = int(stats["virtual_count"][novel].sum())
    novel_hits = int(stats["hit_count"][novel].sum())
    hit_similarities = similarities[virtual_is_hit]
    nonhit_similarities = similarities[~virtual_is_hit]
    relative_figure_dir = Path(os.path.relpath(figure_dir, start=path.parent))
    text = f"""# {label} Chemical-Space Analysis

## Overview

This analysis compares **{seed_count:,} pre-docked seeds** with **{virtual_count:,} compounds acquired and docked by SpaceHASTEN**. A virtual hit is defined by docking score `<= {cutoff}`. The seed set contains **{seed_hit_count:,} hits ({100 * seed_rate:.3f}%)**, whereas the virtual set contains **{virtual_hit_count:,} hits ({100 * virtual_rate:.2f}%)**, corresponding to **{virtual_rate / seed_rate:.1f}-fold enrichment**.

Molecules use binary Morgan fingerprints (radius 2, 1024 bits). Seed-first sphere-exclusion clustering at Tanimoto {cluster_similarity:g} produced **{len(stats["clusterid"]):,} centroids**, including **{novel.sum():,} virtual-derived centroids**. Jaccard landmark UMAP was fit on these centroids, and all docked molecules were transformed into the same fixed coordinate system.

## Figure 1: Seed And Virtual-Hit Density

![Seed and virtual-hit density]({relative_figure_dir}/01_seed_virtual_hit_density.png)

The blue layer represents projected seed density and the orange foreground represents virtual hits. Every molecule contributes to the raster; layer-wise log normalization prevents the much larger seed cohort from obscuring hit structure. Scientifically, this figure shows whether successful virtual compounds occupy the dominant seed regions or concentrate in distinct portions of the seed-defined chemical landscape. UMAP density is descriptive and should not be interpreted as preserved high-dimensional volume.

## Figure 2: Acquisition Shift

![Acquisition shift]({relative_figure_dir}/02_acquisition_shift.png)

Each point is a sphere-exclusion cluster positioned at its centroid UMAP coordinate. Color is the Jeffreys-smoothed log2 ratio of normalized virtual-docked representation to normalized seed representation. Orange clusters were sampled more heavily by SpaceHASTEN than expected from seed prevalence; blue clusters were sampled less heavily. Marker size represents total cluster population, and black outlines mark virtual-derived centroids. This separates where the acquisition policy moved its docking budget from whether that movement produced hits.

## Figure 3: Cluster Hit Enrichment

![Cluster hit enrichment]({relative_figure_dir}/03_cluster_hit_enrichment.png)

Color represents the cluster's beta-binomial-shrunk hit-rate difference from the global virtual hit rate ({100 * virtual_rate:.2f}% global baseline), using prior strength {prior_strength:g}. Marker size represents the number of virtual docked compounds. The {novel.sum():,} virtual-derived clusters contain **{novel_virtual:,} virtual docked compounds** and **{novel_hits:,} hits**. This figure identifies chemical regions where the acquired compounds were more or less productive than the overall acquisition set without allowing tiny clusters to dominate through unstable raw ratios.

## Figure 4: Nearest-Seed Similarity

![Nearest-seed similarity]({relative_figure_dir}/04_nearest_seed_similarity_ecdf.png)

The ECDF reports each virtual compound's exact maximum Tanimoto similarity to the complete seed set. Virtual-hit curves are separated by docking iteration and compared with virtual non-hits. The median nearest-seed similarity is **{np.median(hit_similarities):.3f} for hits** and **{np.median(nonhit_similarities):.3f} for non-hits**. Among hits, **{percentage_below(hit_similarities, 0.5):.1f}%** have nearest-seed similarity below 0.5 and **{percentage_below(hit_similarities, 0.7):.1f}%** are below 0.7. This is the quantitative measure of structural movement; unlike UMAP separation, it is computed directly in the original fingerprint space.

## Scientific Interpretation

Together, the figures decompose SpaceHASTEN performance into three distinct effects: redistribution away from the original seed composition, local productivity among the compounds selected for docking, and structural novelty relative to the nearest seed. The maps describe this completed adaptive screening run and do not imply uniform sampling of an underlying chemical universe. Cluster effects are descriptive; molecular analogs are correlated and should not be treated as independent statistical replicates.

## Reproducibility

- Hit cutoff: `{cutoff}`
- Fingerprint: binary Morgan radius 2, 1024 bits
- Similarity: Tanimoto/Jaccard
- Sphere-exclusion similarity threshold: {cluster_similarity:g}
- UMAP landmarks: sphere-exclusion centroids
- UMAP neighbors: {umap_neighbors}
- UMAP minimum distance: {umap_min_dist:g}
- UMAP random seed: {umap_random_seed}
- Cluster statistics: `{relative_figure_dir}/cluster_statistics.csv`
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--clustering", type=Path, required=True)
    parser.add_argument("--nearest-seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--label", default="SpaceHASTEN run")
    parser.add_argument("--cutoff", type=float, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--density-bins", type=int, default=1200)
    parser.add_argument("--prior-strength", type=float, default=20.0)
    parser.add_argument("--cluster-similarity", type=float, default=0.3)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-random-seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.dpi, args.density_bins, args.umap_neighbors) < 1:
        parser.error("dpi, density bins, and UMAP neighbors must be positive")
    if args.prior_strength <= 0:
        parser.error("prior strength must be positive")
    if not 0 < args.cluster_similarity <= 1:
        parser.error("cluster similarity must be in (0, 1]")
    if not 0 <= args.umap_min_dist <= 1:
        parser.error("UMAP minimum distance must be between 0 and 1")
    return args


def main() -> int:
    args = parse_args()
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    compound_ids, scores, iterations = load_database(args.database)
    coordinate_ids, coordinates = load_coordinates(args.coordinates)
    if not np.array_equal(compound_ids, coordinate_ids):
        raise ValueError("coordinate IDs do not match docked database IDs")
    assignment_ids, cluster_ids = load_clusters(args.clustering, len(compound_ids))
    if not np.array_equal(compound_ids, assignment_ids):
        raise ValueError("cluster IDs do not match docked database IDs")

    is_seed = iterations == 0
    is_hit = (~is_seed) & (scores <= args.cutoff)
    extent = shared_extent(coordinates)
    stats = cluster_statistics(compound_ids, cluster_ids, coordinates, is_seed, is_hit)

    plot_density(
        coordinates,
        is_seed,
        is_hit,
        extent,
        args.output_dir,
        args.label,
        args.dpi,
        args.density_bins,
    )
    shift = plot_acquisition_shift(stats, extent, args.output_dir, args.label, args.dpi)
    posterior_rate = plot_hit_enrichment(
        stats,
        extent,
        args.output_dir,
        args.label,
        args.dpi,
        args.prior_strength,
    )
    write_cluster_table(args.output_dir / "cluster_statistics.csv", stats, shift, posterior_rate)

    with np.load(args.nearest_seed) as nearest:
        nearest_ids = nearest["spacehastenid"].copy()
        similarities = nearest["tanimoto"].copy()
    if len(nearest_ids) != len(similarities):
        raise ValueError("nearest-seed identifiers and similarities differ in length")
    virtual_positions = np.flatnonzero(~is_seed)
    if not np.array_equal(compound_ids[virtual_positions], nearest_ids):
        raise ValueError("nearest-seed IDs do not match all virtual database IDs")
    virtual_is_hit = is_hit[virtual_positions]
    virtual_iterations = iterations[virtual_positions]
    plot_nearest_seed(
        similarities,
        virtual_is_hit,
        virtual_iterations,
        args.output_dir,
        args.label,
        args.dpi,
    )

    seed_hit_count = int(np.sum(is_seed & (scores <= args.cutoff)))
    write_summary(
        args.summary,
        args.output_dir,
        stats,
        similarities,
        virtual_is_hit,
        virtual_iterations,
        int(is_seed.sum()),
        seed_hit_count,
        int((~is_seed).sum()),
        int(is_hit.sum()),
        args.cutoff,
        args.label,
        args.cluster_similarity,
        args.prior_strength,
        args.umap_neighbors,
        args.umap_min_dist,
        args.umap_random_seed,
    )
    metadata = {
        "seed_count": int(is_seed.sum()),
        "virtual_count": int((~is_seed).sum()),
        "virtual_hit_count": int(is_hit.sum()),
        "cluster_count": int(len(stats["clusterid"])),
        "figure_directory": str(args.output_dir.resolve()),
        "summary": str(args.summary.resolve()),
    }
    (args.output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
