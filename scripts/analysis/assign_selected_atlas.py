#!/usr/bin/env python3
"""Assign selected compounds to a reusable seed atlas with native parallel repair."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from spacehasten.analysis.selected import write_selected_manifest
from spacehasten.config.settings import Settings
from spacehasten.scheduler.slurm import SlurmScheduler
from spacehasten.stages.atlas import build_atlas_version

LOGGER = logging.getLogger("assign-selected-atlas")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("selected manifest is empty")
    required = {"spacehastenid", "smiles", "reghash"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"selected manifest lacks fields: {sorted(missing)}")
    identifiers = [int(row["spacehastenid"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("selected atlas assignment requires unique compound IDs")
    return rows


def write_selected_smiles(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as text,
    ):
        for row in rows:
            smiles = str(row["smiles"]).strip()
            if not smiles or any(character.isspace() for character in smiles):
                raise ValueError(f"invalid whitespace-delimited SMILES for {row['spacehastenid']}")
            text.write(f"{smiles} {int(row['spacehastenid'])}\n")
    temporary.replace(path)


def centroid_ids(path: Path) -> set[int]:
    result: set[int] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                identifier = int(line.rstrip().rsplit(None, 1)[1])
            except (IndexError, ValueError) as error:
                raise ValueError(f"invalid centroid row {path}:{line_number}") from error
            if identifier in result:
                raise ValueError(f"duplicate centroid ID {identifier} in {path}")
            result.add(identifier)
    if not result:
        raise ValueError(f"centroid file is empty: {path}")
    return result


def write_npz(path: Path, **arrays: npt.NDArray[Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_combined_atlas(
    path: Path,
    seed_ids: npt.NDArray[Any],
    seed_clusters: npt.NDArray[Any],
    selected_ids: npt.NDArray[Any],
    selected_clusters: npt.NDArray[Any],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["spacehastenid", "clusterid"])
        writer.writerows(zip(seed_ids, seed_clusters, strict=True))
        writer.writerows(zip(selected_ids, selected_clusters, strict=True))
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed-atlas", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manifest = args.manifest.resolve()
    seed_atlas = args.seed_atlas.resolve()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    success = root / "_SUCCESS.json"
    augmented_manifest = root / "selected_manifest_with_atlas.csv.gz"
    ordered_assignments = root / "selected_atlas_assignments.npz"
    combined_atlas = root / "combined_seed_selected_atlas.csv"
    if success.is_file() and not args.overwrite:
        print(success.read_text(encoding="utf-8"), end="")
        return 0
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    definition_path = seed_atlas / "atlas_definition.json"
    seed_centroids = seed_atlas / "final/centroids/centroids.smi.gz"
    for path in (definition_path, seed_centroids):
        if not path.is_file():
            raise FileNotFoundError(path)
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    threshold = float(definition["similarity_threshold"])
    if definition.get("fingerprint_type") != "Morgan" or definition.get(
        "fingerprint_parameters"
    ) != {"radius": 2, "fpSize": 1024}:
        raise ValueError("seed atlas fingerprint definition is incompatible")

    settings = Settings.load(ini_path=args.config.resolve())
    if settings.general.atlas_similarity_threshold != threshold:
        raise ValueError("config and seed-atlas similarity thresholds differ")
    rows = read_manifest(manifest)
    selected_ids = np.asarray([int(row["spacehastenid"]) for row in rows], dtype=np.int64)
    input_smiles = root / "inputs/selected.smi.gz"
    write_selected_smiles(input_smiles, rows)

    scheduler = SlurmScheduler(
        partition=settings.slurm.slurm_partition,
        gpu_parameter=settings.slurm.slurm_gpu_parameter,
        log_dir=root / "logs/slurm",
    )
    native_root = root / "native"
    artifacts = build_atlas_version(
        scheduler,
        settings,
        input_smiles=input_smiles,
        output_root=native_root,
        command_prefix=("python3", str(settings.remote_script_path("atlas"))),
        job_suffix="selected",
        existing_centroids=seed_centroids,
    )

    with np.load(artifacts.assignments, allow_pickle=False) as data:
        assignment_ids = data["spacehastenid"].astype(np.int64)
        cluster_ids = data["clusterid"].astype(np.int64)
        similarities = data["centroid_similarity"].astype(np.float64)
    if len(assignment_ids) != len(selected_ids) or len(np.unique(assignment_ids)) != len(
        assignment_ids
    ):
        raise ValueError("selected atlas assignments are incomplete or non-unique")
    positions = {int(identifier): index for index, identifier in enumerate(assignment_ids)}
    if set(positions) != set(selected_ids.tolist()):
        raise ValueError("selected atlas assignment IDs differ from the manifest")
    order = np.asarray([positions[int(identifier)] for identifier in selected_ids], dtype=np.int64)
    cluster_ids = cluster_ids[order]
    similarities = similarities[order]
    if not np.isfinite(similarities).all() or np.any(similarities < threshold - 1e-7):
        raise ValueError("selected atlas assignments violate the similarity threshold")

    seed_cluster_ids = centroid_ids(seed_centroids)
    repair_cluster_ids = set(cluster_ids.tolist()) - seed_cluster_ids
    if not repair_cluster_ids <= set(selected_ids.tolist()):
        raise ValueError("repair centroid IDs are not selected-compound IDs")
    derived_rows: list[dict[str, Any]] = []
    for row, cluster, similarity in zip(rows, cluster_ids, similarities, strict=True):
        derived = dict(row)
        derived["atlas_cluster"] = int(cluster)
        derived["atlas_similarity"] = float(similarity)
        derived["atlas_centroid_source"] = (
            "seed" if int(cluster) in seed_cluster_ids else "selected_repair"
        )
        derived_rows.append(derived)
    write_selected_manifest(augmented_manifest, derived_rows)
    write_npz(
        ordered_assignments,
        spacehastenid=selected_ids,
        clusterid=cluster_ids,
        centroid_similarity=similarities.astype(np.float32),
    )
    seed_assignment_path = seed_atlas / "final/assignments.npz"
    with np.load(seed_assignment_path, allow_pickle=False) as data:
        seed_ids = data["spacehastenid"].astype(np.int64)
        seed_assignments = data["clusterid"].astype(np.int64)
    if len(seed_ids) != int(definition["seed_count"]):
        raise ValueError("seed atlas assignment count differs from its definition")
    if set(seed_ids.tolist()) & set(selected_ids.tolist()):
        raise ValueError("seed and selected atlas namespaces overlap")
    write_combined_atlas(
        combined_atlas,
        seed_ids,
        seed_assignments,
        selected_ids,
        cluster_ids,
    )

    outputs = [
        augmented_manifest,
        ordered_assignments,
        combined_atlas,
        artifacts.assignments,
    ]
    outputs.extend(artifacts.new_centroid_files)
    receipt = {
        "status": "complete",
        "selected_compounds": len(selected_ids),
        "seed_atlas": str(seed_atlas),
        "seed_atlas_definition_sha256": sha256(definition_path),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "similarity_threshold": threshold,
        "seed_centroids": len(seed_cluster_ids),
        "repair_centroids": len(repair_cluster_ids),
        "combined_atlas_compounds": len(seed_ids) + len(selected_ids),
        "minimum_similarity": float(similarities.min()),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
