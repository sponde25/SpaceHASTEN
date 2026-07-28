#!/usr/bin/env python3
"""Analyze exact candidate, selection, and hit enrichment for portfolio batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from tqdm import tqdm

from spacehasten.analysis.artifacts import write_json
from spacehasten.analysis.discovery import discover_run
from spacehasten.core.portfolio_acquisition import CandidatePool, candidate_pool_digest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output root is non-empty; pass --overwrite: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def reconstruct_pool(connection: sqlite3.Connection, batch: sqlite3.Row) -> CandidatePool:
    exclusion = ""
    parameters: list[Any] = [
        int(batch["model_version"]),
        str(batch["atlas_id"]),
        int(batch["atlas_version"]),
        int(batch["candidate_watermark"]),
        int(batch["dock_iteration"]),
    ]
    if batch["history_attempt_policy"] == "once_per_campaign":
        exclusion = (
            "AND NOT EXISTS (SELECT 1 FROM acquisition_selections old_s "
            "JOIN acquisition_batches old_b ON old_b.batch_id=old_s.batch_id "
            "WHERE old_s.spacehastenid=d.spacehastenid AND old_b.dock_iteration < ?) "
        )
        parameters.append(int(batch["dock_iteration"]))
    where = (
        "p.model_version=? AND a.atlas_id=? AND a.assigned_version<=? "
        "AND d.spacehastenid<=? AND (d.dock_score IS NULL OR d.dock_iteration>=?) "
        + exclusion
    )
    count = int(batch["candidate_count"])
    query = (
        "SELECT d.spacehastenid,p.pred_score,p.epistemic_std,a.clusterid,p.model_version "
        "FROM data d JOIN predictions p ON p.spacehastenid=d.spacehastenid "
        "JOIN cluster_atlas_assignments a ON a.spacehastenid=d.spacehastenid "
        f"WHERE {where} ORDER BY d.spacehastenid"
    )
    arrays = [np.empty(count, dtype=np.int64), np.empty(count), np.empty(count),
              np.empty(count, dtype=np.int64), np.empty(count, dtype=np.int64)]
    cursor = connection.execute(query, parameters)
    offset = 0
    with tqdm(
        total=count,
        desc=f"round {batch['dock_iteration']} candidates",
        unit="candidate",
    ) as bar:
        while rows := cursor.fetchmany(100_000):
            stop = offset + len(rows)
            if stop > count:
                raise ValueError(
                    f"round {batch['dock_iteration']}: reconstructed more than {count} candidates"
                )
            columns = tuple(zip(*rows, strict=True))
            for index, values in enumerate(columns):
                arrays[index][offset:stop] = values
            offset = stop
            bar.update(len(rows))
    if offset != count:
        raise ValueError(
            f"round {batch['dock_iteration']}: reconstructed {offset} candidates; "
            f"history records {count}"
        )
    pool = CandidatePool(
        ids=arrays[0],
        raw_means=arrays[1],
        raw_epistemic_stds=arrays[2],
        cluster_ids=arrays[3],
        model_versions=arrays[4],
    )
    digest = candidate_pool_digest(pool)
    if digest != batch["candidate_digest"]:
        raise ValueError(f"round {batch['dock_iteration']}: candidate digest mismatch")
    return pool


def load_selections(
    connection: sqlite3.Connection,
    batch: sqlite3.Row,
    hit_threshold: float,
    strict_threshold: float,
) -> pd.DataFrame:
    frame = pd.read_sql_query(
        "SELECT s.selection_rank,s.spacehastenid,s.clusterid,s.cluster_count_before,"
        "s.p_hit,s.expected_improvement,s.quality,s.support_before,s.support_after,"
        "s.marginal_reward,s.crowding_penalty,s.final_utility,s.cap_reached_after,"
        "o.status AS outcome_status,o.dock_score "
        "FROM acquisition_selections s LEFT JOIN acquisition_outcomes o "
        "ON o.batch_id=s.batch_id AND o.spacehastenid=s.spacehastenid "
        "WHERE s.batch_id=? ORDER BY s.selection_rank",
        connection,
        params=(batch["batch_id"],),
    )
    if len(frame) != int(batch["selected_count"]):
        raise ValueError(f"round {batch['dock_iteration']}: selected-count mismatch")
    if frame["spacehastenid"].duplicated().any() or frame["selection_rank"].tolist() != list(
        range(1, len(frame) + 1)
    ):
        raise ValueError(f"round {batch['dock_iteration']}: invalid selection history")
    frame["round"] = int(batch["dock_iteration"])
    frame["is_scored"] = frame["dock_score"].notna()
    frame["is_hit"] = frame["is_scored"] & (frame["dock_score"] <= hit_threshold)
    frame["is_strict_hit"] = frame["is_scored"] & (frame["dock_score"] <= strict_threshold)
    frame["within_cluster_order"] = frame["cluster_count_before"] + 1
    frame["within_cluster_order_bin"] = pd.cut(
        frame["within_cluster_order"],
        [0, 1, 2, 5, 10, 25, np.inf],
        labels=["1", "2", "3-5", "6-10", "11-25", "26+"],
    )
    return frame


def concentration(counts: np.ndarray) -> dict[str, float | int | None]:
    positive = counts[counts > 0].astype(float)
    if len(positive) == 0:
        return {"regions": 0, "hhi": None, "effective_regions": 0.0, "top10_share": None}
    shares = positive / positive.sum()
    hhi = float(np.sum(shares**2))
    return {
        "regions": len(positive),
        "hhi": hhi,
        "effective_regions": 1.0 / hhi,
        "top10_share": float(np.sort(shares)[-10:].sum()),
    }


def centroid_table(connection: sqlite3.Connection, atlas_id: str) -> pd.DataFrame:
    frame = pd.read_sql_query(
        "SELECT c.clusterid,c.centroid_spacehastenid,c.created_version,d.dock_iteration "
        "FROM cluster_atlas_centroids c LEFT JOIN data d "
        "ON d.spacehastenid=c.centroid_spacehastenid WHERE c.atlas_id=?",
        connection,
        params=(atlas_id,),
    )
    frame["centroid_source"] = np.where(frame["dock_iteration"] == 0, "seed", "virtual")
    return frame


def cluster_metrics(
    pool: CandidatePool,
    selected: pd.DataFrame,
    round_id: int,
    centroids: pd.DataFrame,
    previous: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate = pd.Series(pool.cluster_ids).value_counts(sort=False).rename("candidate_count")
    observed = selected.groupby("clusterid", sort=False).agg(
        selected_count=("spacehastenid", "size"),
        scored_count=("is_scored", "sum"),
        hit_count=("is_hit", "sum"),
        strict_hit_count=("is_strict_hit", "sum"),
        expected_hit_mass=("p_hit", "sum"),
    )
    frame = (
        pd.concat([candidate, observed], axis=1)
        .fillna(0)
        .rename_axis("clusterid")
        .reset_index()
    )
    integer_columns = (
        "candidate_count",
        "selected_count",
        "scored_count",
        "hit_count",
        "strict_hit_count",
    )
    for column in integer_columns:
        frame[column] = frame[column].astype(np.int64)
    frame["round"] = round_id
    frame["candidate_share"] = frame["candidate_count"] / len(pool.ids)
    frame["selected_share"] = frame["selected_count"] / len(selected)
    total_hits = int(frame["hit_count"].sum())
    frame["hit_share"] = frame["hit_count"] / total_hits if total_hits else 0.0
    frame["selection_enrichment"] = frame["selected_share"] / frame["candidate_share"]
    frame["hit_enrichment"] = (
        frame["hit_share"] / frame["candidate_share"] if total_hits else math.nan
    )
    frame["observed_hit_rate"] = frame["hit_count"] / frame["scored_count"].replace(0, np.nan)
    current = frame[["clusterid", "candidate_share", "selected_share"]].copy()
    if previous is None:
        frame["candidate_share_growth"] = math.nan
        frame["selected_share_growth"] = math.nan
    else:
        growth = current.merge(previous, on="clusterid", how="left", suffixes=("", "_previous"))
        growth[["candidate_share_previous", "selected_share_previous"]] = growth[
            ["candidate_share_previous", "selected_share_previous"]
        ].fillna(0.0)
        frame = frame.merge(
            growth[
                [
                    "clusterid",
                    "candidate_share_previous",
                    "selected_share_previous",
                ]
            ],
            on="clusterid",
        )
        frame["candidate_share_growth"] = (
            frame["candidate_share"] - frame["candidate_share_previous"]
        )
        frame["selected_share_growth"] = (
            frame["selected_share"] - frame["selected_share_previous"]
        )
        frame = frame.drop(columns=["candidate_share_previous", "selected_share_previous"])
    frame = frame.merge(centroids, on="clusterid", how="left", validate="one_to_one")
    return frame, current


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_figures(
    root: Path,
    clusters: pd.DataFrame,
    concentration_rows: pd.DataFrame,
    dpi: int,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    plt.style.use("tableau-colorblind10")
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for round_id, frame in clusters.groupby("round", sort=True):
        observed = frame[frame["scored_count"] > 0]
        axes[0].scatter(
            observed["selection_enrichment"],
            observed["observed_hit_rate"],
            s=np.sqrt(observed["selected_count"]) * 1.5,
            alpha=0.45,
            label=f"round {round_id}",
        )
    axes[0].set(
        xlabel="Selection enrichment (selected share / candidate share)",
        ylabel="Observed hit rate",
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(
        concentration_rows["round"],
        concentration_rows["selected_hhi"],
        marker="o",
        label="selected HHI",
    )
    axes[1].plot(
        concentration_rows["round"],
        concentration_rows["hit_hhi"],
        marker="s",
        label="hit HHI",
    )
    axes[1].set(
        xlabel="Round",
        ylabel="Region concentration",
        xticks=sorted(concentration_rows["round"].unique()),
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(root / "portfolio_enrichment.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(root / "portfolio_enrichment.pdf", bbox_inches="tight")
    plt.close(figure)


def analyze(args: argparse.Namespace) -> None:
    context = discover_run(args.run_or_db, database_path=args.database)
    root = args.output_root.resolve()
    prepare_output(root, args.overwrite)
    connection = sqlite3.connect(f"file:{context.database_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        required = {
            "acquisition_batches",
            "acquisition_selections",
            "acquisition_outcomes",
            "acquisition_region_summaries",
            "cluster_atlas_assignments",
            "cluster_atlas_centroids",
            "predictions",
            "data",
        }
        if missing := required - tables:
            raise ValueError(f"portfolio enrichment requires tables: {sorted(missing)}")
        batches = connection.execute(
            "SELECT * FROM acquisition_batches WHERE strategy='portfolio' "
            "ORDER BY dock_iteration"
        ).fetchall()
        if not batches:
            raise ValueError("portfolio acquisition history is empty")
        cluster_frames: list[pd.DataFrame] = []
        selection_frames: list[pd.DataFrame] = []
        concentration_rows: list[dict[str, Any]] = []
        reconstruction_rows: list[dict[str, Any]] = []
        previous_by_atlas: dict[str, pd.DataFrame] = {}
        centroids: dict[str, pd.DataFrame] = {}
        for batch in batches:
            round_id = int(batch["dock_iteration"])
            pool = reconstruct_pool(connection, batch)
            selected = load_selections(
                connection,
                batch,
                args.hit_threshold,
                args.strict_threshold,
            )
            atlas_id = str(batch["atlas_id"])
            centroids.setdefault(atlas_id, centroid_table(connection, atlas_id))
            cluster, current_shares = cluster_metrics(
                pool,
                selected,
                round_id,
                centroids[atlas_id],
                previous_by_atlas.get(atlas_id),
            )
            previous_by_atlas[atlas_id] = current_shares
            validate_region_summaries(connection, str(batch["batch_id"]), cluster)
            cluster_frames.append(cluster)
            selection_frames.append(selected)
            candidate_metrics = concentration(cluster["candidate_count"].to_numpy())
            selected_metrics = concentration(cluster["selected_count"].to_numpy())
            hit_metrics = concentration(cluster["hit_count"].to_numpy())
            concentration_rows.append(
                {
                    "round": round_id,
                    **{f"candidate_{name}": value for name, value in candidate_metrics.items()},
                    **{f"selected_{name}": value for name, value in selected_metrics.items()},
                    **{f"hit_{name}": value for name, value in hit_metrics.items()},
                }
            )
            reconstruction_rows.append(
                {
                    "round": round_id,
                    "candidate_count": len(pool.ids),
                    "candidate_digest": candidate_pool_digest(pool),
                    "recorded_candidate_count": int(batch["candidate_count"]),
                    "recorded_candidate_digest": str(batch["candidate_digest"]),
                    "exact": True,
                }
            )
        clusters = pd.concat(cluster_frames, ignore_index=True)
        selections = pd.concat(selection_frames, ignore_index=True)
    finally:
        connection.close()

    concentration_frame = pd.DataFrame(concentration_rows)
    order_bins = (
        selections.groupby(["round", "within_cluster_order_bin"], observed=True)
        .agg(
            selected_count=("spacehastenid", "size"),
            scored_count=("is_scored", "sum"),
            hit_count=("is_hit", "sum"),
        )
        .reset_index()
    )
    order_bins["marginal_observed_hit_rate"] = order_bins["hit_count"] / order_bins[
        "scored_count"
    ].replace(0, np.nan)
    top_rows = []
    for round_id, frame in clusters.groupby("round", sort=True):
        hit_share = np.sort(frame["hit_share"].to_numpy())
        for top_n in (1, 5, 10, 50):
            top_rows.append(
                {
                    "round": int(round_id),
                    "top_n_regions": top_n,
                    "hit_contribution": float(hit_share[-top_n:].sum()),
                }
            )
    outputs = {
        "cluster_round_enrichment.csv": clusters,
        "within_cluster_selection_order.csv": selections,
        "within_cluster_order_hit_rates.csv": order_bins,
        "cluster_concentration.csv": concentration_frame,
        "top_region_hit_contribution.csv": pd.DataFrame(top_rows),
        "candidate_reconstruction.csv": pd.DataFrame(reconstruction_rows),
    }
    for name, frame in outputs.items():
        write_frame(frame, root / name)
    save_figures(root / "figures", clusters, concentration_frame, args.dpi)
    artifact_paths = sorted(path for path in root.rglob("*") if path.is_file())
    receipt = {
        "status": "complete",
        "database": str(context.database_path),
        "rounds": [int(batch["dock_iteration"]) for batch in batches],
        "candidate_reconstruction_exact": True,
        "selected": len(selections),
        "scored": int(selections["is_scored"].sum()),
        "hits": int(selections["is_hit"].sum()),
        "hit_threshold": args.hit_threshold,
        "strict_threshold": args.strict_threshold,
        "outputs": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifact_paths
        ],
    }
    write_json(root / "_SUCCESS.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


def validate_region_summaries(
    connection: sqlite3.Connection,
    batch_id: str,
    clusters: pd.DataFrame,
) -> None:
    persisted = pd.read_sql_query(
        "SELECT clusterid,selected_count,scored_count,observed_hits,unresolved_count "
        "FROM acquisition_region_summaries WHERE batch_id=? ORDER BY clusterid",
        connection,
        params=(batch_id,),
    )
    observed = clusters[clusters["selected_count"] > 0][
        ["clusterid", "selected_count", "scored_count", "hit_count"]
    ].sort_values("clusterid")
    if len(persisted) != len(observed):
        raise ValueError(f"batch {batch_id}: persisted region coverage differs from selections")
    if not np.array_equal(persisted["clusterid"].to_numpy(), observed["clusterid"].to_numpy()):
        raise ValueError(f"batch {batch_id}: persisted region IDs differ from selections")
    if not np.array_equal(
        persisted["selected_count"].to_numpy(), observed["selected_count"].to_numpy()
    ):
        raise ValueError(f"batch {batch_id}: selected region counts differ")
    if not np.array_equal(
        persisted["scored_count"].to_numpy(), observed["scored_count"].to_numpy()
    ) or not np.array_equal(
        persisted["observed_hits"].to_numpy(), observed["hit_count"].to_numpy()
    ):
        raise ValueError(f"batch {batch_id}: persisted region outcomes differ")
    if not np.array_equal(
        persisted["unresolved_count"].to_numpy(),
        observed["selected_count"].to_numpy() - observed["scored_count"].to_numpy(),
    ):
        raise ValueError(f"batch {batch_id}: unresolved region counts differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_or_db")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hit-threshold", type=float, required=True)
    parser.add_argument("--strict-threshold", type=float, default=-11.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("dpi must be positive")
    analyze(args)


if __name__ == "__main__":
    main()
