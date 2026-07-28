#!/usr/bin/env python3
"""Analyze diversity, descriptors, and seed coverage from one selected cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from spacehasten.analysis.artifacts import write_json
from spacehasten.analysis.cached import family_distribution, sampled_diversity

DESCRIPTORS = ("MW", "cLogP", "TPSA", "HBD", "HBA", "rotatable", "rings", "Fsp3")
FAMILY_COLUMNS = (("typed", "typed_scaffold"), ("generic", "generic_framework"))


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


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    temporary.replace(path)


def load_cache(
    manifest_path: Path,
    structure_path: Path,
    fingerprint_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    manifest = pd.read_csv(manifest_path)
    structures = pd.read_csv(structure_path)
    require_columns(
        manifest,
        {
            "spacehastenid",
            "reghash",
            "smiles",
            "round",
            "rank",
            "dock_score",
            "is_scored",
            "is_hit",
        },
        "selected manifest",
    )
    require_columns(
        structures,
        {"spacehastenid", "reghash", "typed_scaffold", "generic_framework", *DESCRIPTORS},
        "structure cache",
    )
    if manifest.empty:
        raise ValueError("selected manifest is empty")
    if (manifest.groupby("spacehastenid")["reghash"].nunique() > 1).any() or (
        manifest.groupby("spacehastenid")["smiles"].nunique() > 1
    ).any():
        raise ValueError("selected IDs map to multiple structures")
    if structures["spacehastenid"].duplicated().any() or structures["reghash"].duplicated().any():
        raise ValueError("structure cache contains duplicate identities")
    attempt_ids = manifest["spacehastenid"].to_numpy(np.int64)
    manifest_ids = np.asarray(list(dict.fromkeys(attempt_ids.tolist())), dtype=np.int64)
    structure_ids = structures["spacehastenid"].to_numpy(np.int64)
    if not np.array_equal(manifest_ids, structure_ids):
        raise ValueError("manifest and structure cache ID order differs")
    manifest_identity = manifest.drop_duplicates("spacehastenid", keep="first")
    if not np.array_equal(
        manifest_identity["reghash"].to_numpy(), structures["reghash"].to_numpy()
    ):
        raise ValueError("manifest and structure cache reghash order differs")
    with np.load(fingerprint_path, allow_pickle=False) as data:
        fingerprint_ids = data["spacehastenid"].astype(np.int64, copy=True)
        words = data["words"].astype(np.uint64, copy=True)
        popcounts = data["popcounts"].astype(np.int64, copy=True)
    if not np.array_equal(manifest_ids, fingerprint_ids):
        raise ValueError("manifest and fingerprint cache ID order differs")
    if words.shape != (len(structures), 16) or popcounts.shape != (len(structures),):
        raise ValueError("fingerprint cache dimensions are invalid")
    manifest = manifest.copy()
    manifest["_attempt_order"] = np.arange(len(manifest), dtype=np.int64)
    frame = (
        manifest.merge(structures, on=["spacehastenid", "reghash"], validate="many_to_one")
        .sort_values("_attempt_order")
        .drop(columns="_attempt_order")
        .reset_index(drop=True)
    )
    structure_index = pd.Series(np.arange(len(structures), dtype=np.int64), index=structure_ids)
    frame["structure_index"] = frame["spacehastenid"].map(structure_index)
    frame["round"] = frame["round"].astype(np.int64)
    if (frame["round"] < 1).any():
        raise ValueError("selection rounds must be positive")
    for column in DESCRIPTORS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(float)).all():
            raise ValueError(f"descriptor contains non-finite values: {column}")
    return frame, words, popcounts


def cohort_record(
    frame: pd.DataFrame,
    words: np.ndarray,
    popcounts: np.ndarray,
    *,
    round_id: int,
    cohort: str,
    seed: int,
    pair_samples: int,
    atlas_column: str | None,
) -> dict[str, Any]:
    unique = frame.drop_duplicates("reghash")
    diversity, standard_error, sample_count = sampled_diversity(
        unique["structure_index"].to_numpy(np.int64),
        words,
        popcounts,
        seed=seed,
        samples=pair_samples,
    )
    record: dict[str, Any] = {
        "round": round_id,
        "cohort": cohort,
        "events": len(frame),
        "unique_compounds": len(unique),
        "internal_diversity": diversity,
        "internal_diversity_mc_se": standard_error,
        "pair_samples": sample_count,
    }
    families = list(FAMILY_COLUMNS)
    if atlas_column is not None:
        families.append(("atlas", atlas_column))
    for prefix, column in families:
        metrics = family_distribution(unique[column].to_numpy())
        record.update({f"{prefix}_{name}": value for name, value in metrics.items()})
        record[f"{prefix}_richness_per_10k"] = (
            10_000 * int(metrics["q0"]) / len(unique) if len(unique) else None
        )
    return record


def descriptor_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (round_id, cohort), group in frame.groupby(["round", "cohort"], sort=True):
        for descriptor in DESCRIPTORS:
            values = group[descriptor]
            rows.append(
                {
                    "round": int(round_id),
                    "cohort": str(cohort),
                    "descriptor": descriptor,
                    "n": len(values),
                    "mean": values.mean(),
                    "sd": values.std(),
                    "q05": values.quantile(0.05),
                    "q25": values.quantile(0.25),
                    "median": values.median(),
                    "q75": values.quantile(0.75),
                    "q95": values.quantile(0.95),
                }
            )
    return pd.DataFrame(rows)


def add_nearest_seed(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as data:
        nearest = pd.DataFrame(
            {
                "spacehastenid": data["spacehastenid"].astype(np.int64),
                "nearest_seed_id": data["nearest_seed_id"].astype(np.int64),
                "nearest_seed_tanimoto": data["tanimoto"].astype(float),
            }
        )
    if nearest["spacehastenid"].duplicated().any() or set(nearest["spacehastenid"]) != set(
        frame["spacehastenid"]
    ):
        raise ValueError("nearest-seed IDs do not exactly match the selected manifest")
    if not np.isfinite(nearest["nearest_seed_tanimoto"]).all() or not nearest[
        "nearest_seed_tanimoto"
    ].between(0, 1).all():
        raise ValueError("nearest-seed similarities must be finite and within [0,1]")
    return frame.merge(nearest, on="spacehastenid", validate="many_to_one")


def seed_family_sets(path: Path | None) -> tuple[set[str], set[str]]:
    if path is None:
        return set(), set()
    families = pd.read_csv(path)
    require_columns(families, {"family_type", "scaffold"}, "seed family table")
    typed_labels = {"typed", "typed_murcko", "typed_scaffold"}
    generic_labels = {"generic", "generic_murcko", "generic_framework"}
    typed = set(families.loc[families["family_type"].isin(typed_labels), "scaffold"].astype(str))
    generic = set(
        families.loc[families["family_type"].isin(generic_labels), "scaffold"].astype(str)
    )
    if not typed or not generic:
        raise ValueError("seed family table lacks typed or generic families")
    return typed, generic


def add_centroid_origin(
    frame: pd.DataFrame,
    path: Path,
    atlas_column: str,
) -> pd.DataFrame:
    enrichment = pd.read_csv(path, usecols=["clusterid", "centroid_source"])
    sources = enrichment.drop_duplicates()
    if sources.groupby("clusterid")["centroid_source"].nunique().max() != 1:
        raise ValueError("portfolio enrichment maps a cluster to multiple centroid sources")
    source_map = sources.drop_duplicates("clusterid").set_index("clusterid")["centroid_source"]
    result = frame.copy()
    selected_sources = result[atlas_column].map(source_map)
    if selected_sources.isna().any():
        raise ValueError("portfolio enrichment does not cover every selected atlas cluster")
    result["atlas_seed_centred"] = selected_sources.eq("seed")
    return result


def seed_coverage(
    frame: pd.DataFrame,
    rounds: list[int],
    seed_typed: set[str],
    seed_generic: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for round_id in rounds:
        selected = frame[frame["round"] == round_id]
        for cohort_name, cohort in (
            ("selected", selected),
            ("hit_only", selected[selected["is_hit"]]),
        ):
            similarity = cohort["nearest_seed_tanimoto"]
            typed = set(cohort["typed_scaffold"].astype(str))
            generic = set(cohort["generic_framework"].astype(str))
            row: dict[str, Any] = {
                "round": round_id,
                "cohort": cohort_name,
                "n": len(cohort),
                "nearest_seed_mean": similarity.mean(),
                "nearest_seed_median": similarity.median(),
                "nearest_seed_q05": similarity.quantile(0.05),
                "nearest_seed_q95": similarity.quantile(0.95),
            }
            row.update(
                {
                    f"nearest_seed_fraction_below_{threshold:g}": float(
                        (similarity < threshold).mean()
                    )
                    if len(similarity)
                    else None
                    for threshold in (0.3, 0.4, 0.5, 0.7)
                }
            )
            if seed_typed:
                row["typed_seed_novel_count"] = len(typed - seed_typed)
                row["typed_seed_novel_fraction"] = (
                    len(typed - seed_typed) / len(typed) if typed else None
                )
                row["generic_seed_novel_count"] = len(generic - seed_generic)
                row["generic_seed_novel_fraction"] = (
                    len(generic - seed_generic) / len(generic) if generic else None
                )
            if "atlas_seed_centred" in cohort:
                row["seed_centred_atlas_fraction"] = cohort["atlas_seed_centred"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def save_figures(
    root: Path,
    metrics: pd.DataFrame,
    descriptors: pd.DataFrame,
    nearest: pd.DataFrame | None,
    rounds: list[int],
    dpi: int,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    plt.style.use("tableau-colorblind10")
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    for marker, cohort in (("o", "selected"), ("s", "hit_only")):
        subset = metrics[metrics["cohort"] == cohort]
        axes[0].errorbar(
            subset["round"],
            subset["internal_diversity"],
            yerr=subset["internal_diversity_mc_se"],
            marker=marker,
            label=cohort.replace("_", " "),
        )
        axes[1].plot(subset["round"], subset["generic_q1"], marker=marker, label=cohort)
    axes[0].set(xlabel="Round", ylabel="Internal diversity", xticks=rounds)
    axes[1].set(xlabel="Round", ylabel="Effective generic frameworks", xticks=rounds)
    for axis in axes:
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    save_figure(figure, root / "diversity_overview", dpi)

    figure, axes = plt.subplots(2, 4, figsize=(11, 5.8))
    for axis, descriptor in zip(axes.flat, DESCRIPTORS, strict=True):
        for marker, cohort in (("o", "selected"), ("s", "hit_only")):
            subset = descriptors[
                (descriptors["descriptor"] == descriptor) & (descriptors["cohort"] == cohort)
            ]
            axis.plot(subset["round"], subset["median"], marker=marker, label=cohort)
        axis.set(title=descriptor, xlabel="Round", xticks=rounds)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, root / "descriptor_shift", dpi)

    if nearest is not None:
        figure, axis = plt.subplots(figsize=(5.8, 4.0))
        for round_id in rounds:
            values = np.sort(
                nearest.loc[nearest["round"] == round_id, "nearest_seed_tanimoto"].to_numpy()
            )
            axis.plot(
                values,
                np.arange(1, len(values) + 1) / len(values),
                label=f"round {round_id}",
            )
        axis.set(xlabel="Nearest starting-seed Tanimoto", ylabel="Empirical cumulative fraction")
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        save_figure(figure, root / "nearest_seed_ecdf", dpi)


def save_figure(figure: plt.Figure, stem: Path, dpi: int) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--structure-cache", type=Path, required=True)
    parser.add_argument("--fingerprints", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nearest-seed", type=Path)
    parser.add_argument("--seed-families", type=Path)
    parser.add_argument("--portfolio-enrichment", type=Path)
    parser.add_argument("--pair-samples", type=int, default=1_000_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if min(arguments.pair_samples, arguments.dpi) < 1:
        parser.error("pair-samples and dpi must be positive")
    if arguments.seed_families and not arguments.nearest_seed:
        parser.error("--seed-families requires --nearest-seed")
    return arguments


def main() -> None:
    arguments = parse_args()
    inputs = [arguments.manifest, arguments.structure_cache, arguments.fingerprints]
    if arguments.nearest_seed:
        inputs.append(arguments.nearest_seed)
    if arguments.seed_families:
        inputs.append(arguments.seed_families)
    if arguments.portfolio_enrichment:
        inputs.append(arguments.portfolio_enrichment)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    root = arguments.output_root.resolve()
    prepare_output(root, arguments.overwrite)
    frame, words, popcounts = load_cache(
        arguments.manifest, arguments.structure_cache, arguments.fingerprints
    )
    rounds = sorted(frame["round"].unique().astype(int).tolist())
    atlas_column = next(
        (column for column in ("clusterid", "atlas_cluster") if column in frame.columns), None
    )
    if arguments.portfolio_enrichment:
        if atlas_column is None:
            raise ValueError("portfolio enrichment requires atlas clusters in the manifest")
        frame = add_centroid_origin(frame, arguments.portfolio_enrichment, atlas_column)

    records: list[dict[str, Any]] = []
    novelty_rows: list[dict[str, Any]] = []
    descriptor_frames: list[pd.DataFrame] = []
    cumulative_selected: list[pd.DataFrame] = []
    cumulative_hits: list[pd.DataFrame] = []
    prior_families: dict[str, set[Any]] = {"typed": set(), "generic": set(), "atlas": set()}
    for round_id in rounds:
        selected = frame[frame["round"] == round_id].copy()
        hits = selected[selected["is_hit"]].copy()
        cumulative_selected.append(selected)
        cumulative_hits.append(hits)
        cohorts = (
            ("selected", selected),
            ("hit_only", hits),
            ("cumulative_selected", pd.concat(cumulative_selected, ignore_index=True)),
            ("cumulative_hit_only", pd.concat(cumulative_hits, ignore_index=True)),
        )
        for offset, (name, cohort) in enumerate(cohorts):
            records.append(
                cohort_record(
                    cohort,
                    words,
                    popcounts,
                    round_id=round_id,
                    cohort=name,
                    seed=arguments.random_seed + round_id * 10 + offset,
                    pair_samples=arguments.pair_samples,
                    atlas_column=atlas_column,
                )
            )
        current: dict[str, set[Any]] = {
            "typed": set(hits["typed_scaffold"]),
            "generic": set(hits["generic_framework"]),
        }
        if atlas_column is not None:
            current["atlas"] = set(hits[atlas_column])
        novelty_rows.append(
            {
                "round": round_id,
                **{
                    f"new_hit_{name}": len(labels - prior_families[name])
                    for name, labels in current.items()
                },
            }
        )
        for name, labels in current.items():
            prior_families[name].update(labels)
        for name, cohort in (("selected", selected), ("hit_only", hits)):
            values = cohort[["round", *DESCRIPTORS]].copy()
            values["cohort"] = name
            descriptor_frames.append(values)

    metrics = pd.DataFrame(records)
    descriptor_values = pd.concat(descriptor_frames, ignore_index=True)
    descriptors = descriptor_summary(descriptor_values)
    write_frame(metrics, root / "diversity_metrics.csv")
    write_frame(pd.DataFrame(novelty_rows), root / "new_productive_families.csv")
    write_frame(descriptor_values, root / "descriptor_values.csv.gz")
    write_frame(descriptors, root / "descriptor_summary.csv")

    nearest_frame: pd.DataFrame | None = None
    if arguments.nearest_seed:
        nearest_frame = add_nearest_seed(frame, arguments.nearest_seed)
        seed_typed, seed_generic = seed_family_sets(arguments.seed_families)
        coverage = seed_coverage(nearest_frame, rounds, seed_typed, seed_generic)
        write_frame(coverage, root / "seed_coverage_metrics.csv")
        write_frame(
            nearest_frame.drop(columns=["structure_index"]),
            root / "selected_nearest_seed.csv.gz",
        )
    save_figures(root / "figures", metrics, descriptors, nearest_frame, rounds, arguments.dpi)

    artifacts = sorted(path for path in root.rglob("*") if path.is_file())
    receipt = {
        "status": "complete",
        "selected_attempts": len(frame),
        "selected_unique_compounds": frame["reghash"].nunique(),
        "scored": int(frame["is_scored"].sum()),
        "hits": int(frame["is_hit"].sum()),
        "rounds": rounds,
        "pair_samples": arguments.pair_samples,
        "random_seed": arguments.random_seed,
        "atlas_column": atlas_column,
        "nearest_seed_available": nearest_frame is not None,
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in sorted(inputs)
        ],
        "outputs": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        ],
    }
    write_json(root / "_SUCCESS.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
