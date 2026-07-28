#!/usr/bin/env python3
"""Plot fixed-reference chemical space for a selected-compound cache."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

BLUE = "#0072B2"
ORANGE = "#D55E00"
DIVERGING = LinearSegmentedColormap.from_list(
    "spacehasten_blue_orange", [BLUE, "#F7F7F7", ORANGE]
)


def coordinates(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as data:
        identifiers = data["spacehastenid"].astype(np.int64)
        values = data["umap"].astype(np.float32)
    if values.shape != (len(identifiers), 2) or not np.isfinite(values).all():
        raise ValueError(f"invalid UMAP coordinates: {path}")
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate UMAP identifiers: {path}")
    return pd.DataFrame(
        {"spacehastenid": identifiers, "umap_x": values[:, 0], "umap_y": values[:, 1]}
    )


def seed_coordinates(coordinate_path: Path, reference_path: Path) -> pd.DataFrame:
    with np.load(reference_path, allow_pickle=False) as data:
        identifiers = data["seed_spacehastenid"].astype(np.int64)
    frame = coordinates(coordinate_path)
    selected = frame[frame["spacehastenid"].isin(identifiers)]
    if len(selected) != len(identifiers):
        raise ValueError("fixed-reference coordinates do not cover every starting seed")
    return selected


def save(figure: plt.Figure, root: Path, name: str, dpi: int) -> None:
    figure.savefig(root / f"{name}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def marker_sizes(counts: np.ndarray, maximum: float = 80.0) -> np.ndarray:
    transformed = np.sqrt(np.asarray(counts, dtype=float))
    return 4.0 + maximum * transformed / transformed.max()


def plot_density_and_shift(
    frame: pd.DataFrame,
    seeds: pd.DataFrame,
    root: Path,
    dpi: int,
) -> tuple[float, float, float, float]:
    extent = (
        min(float(seeds["umap_x"].min()), float(frame["umap_x"].min())),
        max(float(seeds["umap_x"].max()), float(frame["umap_x"].max())),
        min(float(seeds["umap_y"].min()), float(frame["umap_y"].min())),
        max(float(seeds["umap_y"].max()), float(frame["umap_y"].max())),
    )
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hexbin(
        seeds["umap_x"],
        seeds["umap_y"],
        gridsize=160,
        bins="log",
        extent=extent,
        mincnt=1,
        cmap="Greys",
    )
    hits = frame[frame["is_hit"]]
    axes[1].hexbin(
        hits["umap_x"],
        hits["umap_y"],
        gridsize=160,
        bins="log",
        extent=extent,
        mincnt=1,
        cmap="viridis",
    )
    axes[0].set_title("Starting seeds")
    axes[1].set_title("Selected virtual hits")
    for axis in axes:
        axis.set(xlabel="Fixed-reference UMAP 1", ylabel="Fixed-reference UMAP 2")
    figure.tight_layout()
    save(figure, root, "01_seed_virtual_hit_density", dpi)

    rounds = sorted(frame["round"].unique().astype(int).tolist())
    columns = min(3, len(rounds))
    rows = (len(rounds) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.5 * rows), squeeze=False)
    for axis, round_id in zip(axes.flat, rounds, strict=False):
        current = frame[frame["round"] == round_id]
        axis.hexbin(
            current["umap_x"],
            current["umap_y"],
            gridsize=120,
            bins="log",
            extent=extent,
            mincnt=1,
            cmap="cividis",
        )
        axis.set(title=f"Round {round_id}", xlabel="UMAP 1", ylabel="UMAP 2")
    for axis in axes.flat[len(rounds) :]:
        axis.set_visible(False)
    figure.tight_layout()
    save(figure, root, "02_acquisition_shift", dpi)
    return extent


def plot_cluster_enrichment(
    enrichment_path: Path,
    coordinate_frames: list[pd.DataFrame],
    root: Path,
    label: str,
    prior_strength: float,
    dpi: int,
) -> dict[str, object]:
    enrichment = pd.read_csv(enrichment_path)
    required = {
        "clusterid",
        "selected_count",
        "scored_count",
        "hit_count",
        "centroid_spacehastenid",
        "centroid_source",
    }
    if missing := required - set(enrichment.columns):
        raise ValueError(f"portfolio enrichment lacks columns: {sorted(missing)}")
    grouped = (
        enrichment.groupby("clusterid", sort=True)
        .agg(
            selected_count=("selected_count", "sum"),
            scored_count=("scored_count", "sum"),
            hit_count=("hit_count", "sum"),
            centroid_spacehastenid=("centroid_spacehastenid", "first"),
            centroid_source=("centroid_source", "first"),
        )
        .reset_index()
    )
    coordinate_map = (
        pd.concat(coordinate_frames, ignore_index=True)
        .drop_duplicates("spacehastenid", keep="first")
        .set_index("spacehastenid")[["umap_x", "umap_y"]]
    )
    centroid_ids = grouped["centroid_spacehastenid"].astype(np.int64)
    missing_ids = sorted(set(centroid_ids) - set(coordinate_map.index))
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} occupied atlas centroids lack fixed-reference coordinates"
        )
    grouped[["umap_x", "umap_y"]] = coordinate_map.loc[centroid_ids].to_numpy()
    total_scored = int(grouped["scored_count"].sum())
    total_hits = int(grouped["hit_count"].sum())
    if total_scored == 0:
        raise ValueError("cluster enrichment has no scored selections")
    global_rate = total_hits / total_scored
    grouped["posterior_hit_rate"] = (
        grouped["hit_count"] + prior_strength * global_rate
    ) / (grouped["scored_count"] + prior_strength)
    grouped["posterior_difference_pp"] = 100 * (
        grouped["posterior_hit_rate"] - global_rate
    )
    grouped.to_csv(root / "cluster_statistics.csv", index=False)
    difference = grouped["posterior_difference_pp"].to_numpy(float)
    limit = max(float(np.quantile(np.abs(difference), 0.98)), 1.0)
    sizes = marker_sizes(grouped["scored_count"].to_numpy())
    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    scatter = axis.scatter(
        grouped["umap_x"],
        grouped["umap_y"],
        c=difference,
        s=sizes,
        cmap=DIVERGING,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        linewidths=0,
        alpha=0.85,
        rasterized=True,
    )
    virtual = grouped["centroid_source"] == "virtual"
    axis.scatter(
        grouped.loc[virtual, "umap_x"],
        grouped.loc[virtual, "umap_y"],
        s=sizes[virtual.to_numpy()],
        facecolors="none",
        edgecolors="black",
        linewidths=0.6,
        rasterized=True,
    )
    axis.set(xlabel="UMAP 1", ylabel="UMAP 2", title=f"{label}: local virtual-hit enrichment")
    figure.colorbar(
        scatter,
        ax=axis,
        label="Posterior hit-rate difference from global (percentage points)",
    )
    figure.tight_layout()
    save(figure, root, "03_cluster_hit_enrichment", dpi)
    return {
        "occupied_clusters": len(grouped),
        "scored": total_scored,
        "hits": total_hits,
        "global_hit_rate": global_rate,
        "prior_strength": prior_strength,
        "color_limit_percentage_points": limit,
    }


def plot_nearest(path: Path, root: Path, dpi: int) -> None:
    frame = pd.read_csv(path)
    required = {"round", "nearest_seed_tanimoto"}
    if missing := required - set(frame.columns):
        raise ValueError(f"nearest-seed table lacks columns: {sorted(missing)}")
    figure, axis = plt.subplots(figsize=(5.8, 4.0))
    for round_id, current in frame.groupby("round", sort=True):
        values = np.sort(current["nearest_seed_tanimoto"].to_numpy(float))
        axis.plot(values, np.arange(1, len(values) + 1) / len(values), label=f"round {round_id}")
    axis.set(xlabel="Nearest starting-seed Tanimoto", ylabel="Empirical cumulative fraction")
    axis.legend(frameon=False)
    figure.tight_layout()
    save(figure, root, "04_nearest_seed_similarity_ecdf", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selected-coordinates", type=Path, required=True)
    parser.add_argument("--seed-coordinates", type=Path, required=True)
    parser.add_argument("--seed-reference-cache", type=Path, required=True)
    parser.add_argument("--centroid-coordinates", type=Path, action="append", default=[])
    parser.add_argument("--portfolio-enrichment", type=Path, required=True)
    parser.add_argument("--selected-nearest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prior-strength", type=float, default=20.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dpi < 1 or args.prior_strength <= 0:
        parser.error("dpi and prior-strength must be positive")
    inputs = [
        args.manifest,
        args.selected_coordinates,
        args.seed_coordinates,
        args.seed_reference_cache,
        args.portfolio_enrichment,
        args.selected_nearest,
        *args.centroid_coordinates,
    ]
    for path in inputs:
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output root is non-empty; pass --overwrite: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    required = {"spacehastenid", "round", "is_hit"}
    if missing := required - set(manifest.columns):
        raise ValueError(f"selected manifest lacks columns: {sorted(missing)}")
    selected_coordinates = coordinates(args.selected_coordinates)
    selected = manifest.merge(selected_coordinates, on="spacehastenid", validate="many_to_one")
    if len(selected) != len(manifest):
        raise ValueError("selected coordinates do not cover the complete manifest")
    seeds = seed_coordinates(args.seed_coordinates, args.seed_reference_cache)
    plt.style.use("tableau-colorblind10")
    extent = plot_density_and_shift(selected, seeds, root, args.dpi)
    coordinate_frames = [seeds, selected_coordinates]
    coordinate_frames.extend(coordinates(path) for path in args.centroid_coordinates)
    cluster_metadata = plot_cluster_enrichment(
        args.portfolio_enrichment,
        coordinate_frames,
        root,
        args.label,
        args.prior_strength,
        args.dpi,
    )
    plot_nearest(args.selected_nearest, root, args.dpi)
    metadata = {
        "status": "complete",
        "selected": len(selected),
        "hits": int(selected["is_hit"].sum()),
        "rounds": sorted(selected["round"].unique().astype(int).tolist()),
        "seeds": len(seeds),
        "extent": extent,
        "cluster_hit_enrichment": cluster_metadata,
        "interpretation": (
            "Fixed-reference visualization only; UMAP distance is not chemical distance."
        ),
    }
    (root / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
