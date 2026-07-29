#!/usr/bin/env python3
"""Validate a complete two-workflow comparison and write its artifact manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spacehasten.analysis.comparison import EVENT_FIELDS, semantic_digest, sha256, write_json

EXCLUDED_DIRECTORIES = {"chunks", "inputs", "logs", "native"}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def verify_record(record: dict[str, Any], *, root: Path | None = None) -> None:
    path = Path(str(record["path"]))
    if root is not None and not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"artifact size changed: {path}")
    if "sha256" in record and sha256(path) != str(record["sha256"]):
        raise ValueError(f"artifact digest changed: {path}")


def validate_receipt(path: Path, *, output_root: Path | None = None) -> dict[str, Any]:
    receipt = read_json(path)
    if receipt.get("status") not in {"complete", "ok", "compatible"}:
        raise ValueError(f"receipt is not complete: {path}")
    inputs = receipt.get("inputs", [])
    if isinstance(inputs, list):
        for record in inputs:
            verify_record(record)
    for record in receipt.get("outputs", []):
        verify_record(record, root=output_root or path.parent)
    return receipt


def validate_events(root: Path, compatibility: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name in (compatibility["left_workflow"], compatibility["right_workflow"]):
        path = root / f"cache/{name}_events.csv.gz"
        events = pd.read_csv(path)
        if semantic_digest(events, EVENT_FIELDS) != compatibility["event_semantic_sha256"][name]:
            raise ValueError(f"normalized event semantic digest mismatch: {name}")
        if events["reghash"].isna().any() or not events["reghash"].is_unique:
            raise ValueError(f"normalized event identities are invalid: {name}")
        result[name] = {
            "rows": len(events),
            "scored": int(events["scored"].sum()),
            "unresolved": int((~events["scored"].astype(bool)).sum()),
            "hits": int(events["hit"].sum()),
            "strict_hits": int(events["strict_hit"].sum()),
            "semantic_sha256": compatibility["event_semantic_sha256"][name],
        }
    return result


def validate_union(root: Path, compatibility: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    union = pd.read_csv(root / "cache/common_union_manifest.csv.gz")
    expected_ids = np.arange(1, len(union) + 1, dtype=np.int64)
    if not np.array_equal(union["common_id"].to_numpy(np.int64), expected_ids):
        raise ValueError("common union IDs are not consecutive and reghash-sorted")
    if not union["reghash"].is_unique:
        raise ValueError("common union reghashes are duplicated")
    if (
        semantic_digest(
            union,
            (
                "common_id",
                "atlas_query_id",
                "reghash",
                "smiles",
                "left_local_id",
                "right_local_id",
                "left_present",
                "right_present",
                "typed_scaffold",
                "generic_framework",
                "MW",
                "cLogP",
                "TPSA",
                "HBD",
                "HBA",
                "rotatable",
                "rings",
                "Fsp3",
            ),
        )
        != compatibility["union_semantic_sha256"]
    ):
        raise ValueError("common union semantic digest changed")
    counts = compatibility["union_counts"]
    if int(counts["union"]) != len(union):
        raise ValueError("common union count differs from compatibility receipt")
    if int(counts["structure_mismatches"]) or int(counts["fingerprint_mismatches"]):
        raise ValueError("common union records one or more shared mismatches")
    with np.load(root / "cache/common_union_fingerprints.npz", allow_pickle=False) as data:
        identifiers = data["spacehastenid"].astype(np.int64)
        words = data["words"]
        popcounts = data["popcounts"]
    if not np.array_equal(identifiers, expected_ids) or words.shape != (len(union), 16):
        raise ValueError("common union fingerprint cache is invalid")
    if popcounts.shape != (len(union),):
        raise ValueError("common union fingerprint popcounts are invalid")
    return expected_ids, {
        "rows": len(union),
        "shared": int(counts["shared"]),
        "left_only": int(counts["left_only"]),
        "right_only": int(counts["right_only"]),
        "fingerprint_shape": list(words.shape),
        "semantic_sha256": compatibility["union_semantic_sha256"],
    }


def validate_coordinates(root: Path, expected_ids: np.ndarray) -> dict[str, Any]:
    path = root / "cache/common_union_umap/landmark_umap_coordinates.npz"
    with np.load(path, allow_pickle=False) as data:
        identifiers = data["spacehastenid"].astype(np.int64)
        coordinates = data["umap"].astype(np.float64)
    if not np.array_equal(identifiers, expected_ids):
        raise ValueError("common UMAP IDs differ from the selected union")
    if coordinates.shape != (len(expected_ids), 2) or not np.isfinite(coordinates).all():
        raise ValueError("common UMAP coordinates are invalid")
    return {"rows": len(identifiers), "shape": list(coordinates.shape), "finite": True}


def validate_atlas(root: Path, expected_ids: np.ndarray) -> dict[str, Any]:
    path = root / "cache/common_atlas/common_union_assignments.npz"
    with np.load(path, allow_pickle=False) as data:
        identifiers = data["spacehastenid"].astype(np.int64)
        clusters = data["clusterid"].astype(np.int64)
        similarities = data["centroid_similarity"].astype(np.float64)
    if not np.array_equal(identifiers, expected_ids) or len(np.unique(identifiers)) != len(
        identifiers
    ):
        raise ValueError("common-atlas IDs differ from the selected union")
    if not np.isfinite(similarities).all() or np.any(similarities < 0.4 - 1e-7):
        raise ValueError("common-atlas similarities violate threshold 0.40")
    if np.any(clusters < 1):
        raise ValueError("common-atlas cluster IDs are invalid")
    return {
        "rows": len(identifiers),
        "unique_clusters": int(np.unique(clusters).size),
        "minimum_similarity": float(similarities.min()),
        "uncovered": 0,
    }


def validate_natural_and_matched(
    root: Path,
    event_counts: dict[str, Any],
    *,
    expected_replicates: int,
) -> dict[str, Any]:
    endpoints = pd.read_csv(root / "tables/endpoint_summary.csv").set_index("workflow")
    for name, counts in event_counts.items():
        for field in ("scored", "unresolved", "hits", "strict_hits"):
            if int(endpoints.loc[name, field]) != int(counts[field]):
                raise ValueError(f"endpoint reconciliation failed: {name} {field}")
        if int(endpoints.loc[name, "selected"]) != int(counts["rows"]):
            raise ValueError(f"selected endpoint reconciliation failed: {name}")
    replicates = pd.read_csv(root / "tables/count_matched_replicates.csv")
    if len(replicates) != expected_replicates or set(replicates["replicate"]) != set(
        range(expected_replicates)
    ):
        raise ValueError("standard count-matched replicate IDs are incomplete")
    if replicates["sample_size"].nunique() != 1:
        raise ValueError("standard count-matched sample sizes differ")
    sample_size = int(replicates["sample_size"].iloc[0])
    with np.load(root / "tables/count_matched_sample_ids.npz", allow_pickle=False) as data:
        replicate_ids = data["replicate"].astype(np.int64)
        sampled_ids = data["common_id"].astype(np.int64)
    if not np.array_equal(replicate_ids, np.arange(expected_replicates)):
        raise ValueError("saved standard matched replicate IDs differ")
    if sampled_ids.shape != (expected_replicates, sample_size):
        raise ValueError("saved standard matched samples have an invalid shape")
    if any(len(np.unique(row)) != sample_size for row in sampled_ids):
        raise ValueError("a standard matched sample contains duplicate compounds")
    required = {
        "budget_yield.csv",
        "cutoff_sensitivity.csv",
        "top_k_score_quality.csv",
        "natural_diversity.csv",
        "common_atlas_coverage.csv",
        "common_atlas_depth.csv",
        "common_atlas_region_contrasts.csv",
        "common_atlas_region_contrast_summary.csv",
        "nearest_seed_summary.csv",
        "nearest_seed_productivity.csv",
        "overlap_summary.csv",
        "shared_score_summary.csv",
        "shared_score_pairs.csv.gz",
        "workflow_exclusive_hit_chemistry.csv",
        "fixed_grid_umap_summary.csv",
        "operational_efficiency.csv",
        "count_matched_summary.csv",
    }
    missing = sorted(name for name in required if not (root / "tables" / name).is_file())
    if missing:
        raise ValueError(f"comparison tables are missing: {missing}")
    return {
        "endpoint_rows": len(endpoints),
        "matched_replicates": len(replicates),
        "matched_sample_size": sample_size,
        "saved_sample_shape": list(sampled_ids.shape),
    }


def validate_paper(
    root: Path,
    event_counts: dict[str, Any],
    *,
    expected_replicates: int,
) -> dict[str, Any]:
    results = {}
    for name in event_counts:
        paper_root = root / f"paper_diversity/{name}"
        receipt = validate_receipt(paper_root / "_SUCCESS.json", output_root=paper_root)
        if float(receipt.get("cluster_similarity", -1)) != 0.55:
            raise ValueError(f"paper diversity threshold differs from 0.55: {name}")
        if int(receipt.get("virtual_hits", -1)) != int(event_counts[name]["hits"]):
            raise ValueError(f"paper hit count differs from normalized events: {name}")
        if not receipt.get("fpsim2_index_matches_cached_fingerprints"):
            raise ValueError(f"paper fingerprints were not validated against FPSim2: {name}")
        level1_status = receipt.get("level1_assignment_status", {})
        if sum(int(value) for value in level1_status.values()) != int(receipt["virtual_hits"]):
            raise ValueError(f"paper Level 1 status does not cover every hit: {name}")
        assignments = pd.read_csv(paper_root / "paper_aligned_assignments.csv.gz")
        if len(assignments) != int(receipt["virtual_hits"]):
            raise ValueError(f"paper assignment count differs from hit count: {name}")
        results[name] = {
            "virtual_hits": int(receipt["virtual_hits"]),
            "level1_status": level1_status,
            "t055_clusters": int(receipt["sphere_exclusion_clusters"]),
        }
    comparison_root = root / "paper_diversity/comparison"
    receipt = validate_receipt(comparison_root / "_SUCCESS.json", output_root=comparison_root)
    if float(receipt.get("cluster_similarity", -1)) != 0.55:
        raise ValueError("paper comparison threshold differs from 0.55")
    if int(receipt.get("replicates", -1)) != expected_replicates:
        raise ValueError("paper comparison replicate count differs from the requirement")
    replicates = pd.read_csv(comparison_root / "paper_count_matched_replicates.csv")
    if len(replicates) != expected_replicates or set(replicates["replicate"]) != set(
        range(expected_replicates)
    ):
        raise ValueError("paper count-matched replicate IDs are incomplete")
    if replicates["sample_size"].nunique() != 1:
        raise ValueError("paper count-matched sample sizes differ")
    sample_size = int(replicates["sample_size"].iloc[0])
    if not replicates["t055_assigned_hits"].eq(sample_size).all():
        raise ValueError("a paper replicate lacks complete Tanimoto-0.55 assignments")
    if replicates["t055_minimum_similarity"].lt(0.55 - 1e-7).any():
        raise ValueError("a paper replicate assignment falls below Tanimoto 0.55")
    with np.load(
        comparison_root / "paper_count_matched_sample_ids.npz", allow_pickle=False
    ) as data:
        sample_replicates = data["replicate"].astype(np.int64)
        sample_ids = data["common_id"].astype(np.int64)
    with np.load(
        comparison_root / "paper_count_matched_cluster_assignments.npz",
        allow_pickle=False,
    ) as data:
        assignment_replicates = data["replicate"].astype(np.int64)
        assignments = data["clusterid"].astype(np.int64)
    expected = np.arange(expected_replicates)
    if not np.array_equal(sample_replicates, expected) or not np.array_equal(
        assignment_replicates, expected
    ):
        raise ValueError("saved paper replicate IDs are incomplete")
    expected_shape = (expected_replicates, sample_size)
    if sample_ids.shape != expected_shape or assignments.shape != expected_shape:
        raise ValueError("saved paper sample or assignment matrices have invalid shapes")
    if np.any(assignments < 0) or any(len(np.unique(row)) != sample_size for row in sample_ids):
        raise ValueError("saved paper samples or cluster assignments are incomplete")
    results["comparison"] = {
        "replicates": len(replicates),
        "sample_size": sample_size,
        "fixed_workflow": receipt["fixed_workflow"],
        "sampled_workflow": receipt["sampled_workflow"],
        "saved_sample_shape": list(sample_ids.shape),
        "saved_assignment_shape": list(assignments.shape),
    }
    return results


def validate_report(root: Path) -> dict[str, Any]:
    markdown = root / "COMPARATIVE_REPORT.md"
    html = root / "COMPARATIVE_REPORT.html"
    if not markdown.is_file() or not html.is_file():
        raise FileNotFoundError("comparison Markdown or HTML report is missing")
    source = markdown.read_text(encoding="utf-8")
    table_titles = re.findall(r"\*\*Table (\d+)\.[^\n]*\*\*", source)
    table_notes = re.findall(r"\*Table (\d+) note\.", source)
    if table_titles != table_notes or len(set(table_titles)) != len(table_titles):
        raise ValueError("numbered comparison table titles and notes do not have parity")
    images = re.findall(r"!\[Figure (\d+)\.[^]]*\]\(([^)]+)\)", source)
    captions = re.findall(r"\*Figure (\d+)\.[^\n]*", source)
    image_numbers = [number for number, _ in images]
    if image_numbers != captions or len(set(image_numbers)) != len(image_numbers):
        raise ValueError("numbered comparison figure links and captions do not have parity")
    for _, link in images:
        if not (root / link).is_file():
            raise FileNotFoundError(root / link)
    rendered = html.read_text(encoding="utf-8")
    embedded = len(re.findall(r"src=[\"']data:image/", rendered))
    if embedded < len(images):
        raise ValueError("standalone comparison HTML does not embed every report image")
    return {
        "table_count": len(table_titles),
        "figure_count": len(images),
        "embedded_images": embedded,
    }


def validate_figures(root: Path) -> dict[str, Any]:
    figure_root = root / "figures"
    png = {path.stem: path for path in figure_root.glob("*.png")}
    pdf = {path.stem: path for path in figure_root.glob("*.pdf")}
    if not png or set(png) != set(pdf):
        raise ValueError("comparison PNG/PDF figure stems do not match")
    if any(path.stat().st_size == 0 for path in (*png.values(), *pdf.values())):
        raise ValueError("comparison contains an empty figure")
    return {"png_count": len(png), "pdf_count": len(pdf), "paired_stems": sorted(png)}


def artifact_manifest(root: Path, excluded: set[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path in excluded:
            continue
        relative = path.relative_to(root)
        if EXCLUDED_DIRECTORIES.intersection(relative.parts):
            continue
        rows.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def validate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.comparison_root.resolve()
    validation_path = root / "validation_summary.json"
    manifest_path = root / "artifact_manifest.json"
    if (validation_path.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError("comparison validation exists; pass --overwrite")
    compatibility = validate_receipt(root / "comparison_compatibility.json")
    if compatibility.get("status") != "compatible":
        raise ValueError("comparison compatibility status is not compatible")
    for name, record in compatibility["individual_validations"].items():
        path = Path(record["path"])
        if read_json(path).get("status") != "ok" or sha256(path) != record["sha256"]:
            raise ValueError(f"individual validation receipt changed: {name}")
    verify_record(compatibility["seed_atlas_definition"])
    for name, records in compatibility["inputs"].items():
        for record in records.values():
            try:
                verify_record(record)
            except (FileNotFoundError, ValueError) as error:
                raise ValueError(f"comparison compatibility input changed: {name}") from error
    event_counts = validate_events(root, compatibility)
    expected_ids, union = validate_union(root, compatibility)
    analysis = validate_receipt(root / "comparison_analysis_receipt.json", output_root=root)
    atlas_receipt = validate_receipt(root / "cache/common_atlas/_SUCCESS.json")
    if (
        int(atlas_receipt.get("selected_compounds", -1)) != len(expected_ids)
        or float(atlas_receipt.get("similarity_threshold", -1)) != 0.4
        or int(atlas_receipt.get("seed_centroids", -1)) < 1
    ):
        raise ValueError("common-atlas receipt does not match the selected union")
    coordinates = validate_coordinates(root, expected_ids)
    atlas = validate_atlas(root, expected_ids)
    natural = validate_natural_and_matched(
        root,
        event_counts,
        expected_replicates=args.expected_replicates,
    )
    paper = validate_paper(
        root,
        event_counts,
        expected_replicates=args.expected_replicates,
    )
    report = validate_report(root)
    figures = validate_figures(root)
    excluded = {validation_path, manifest_path}
    artifacts = artifact_manifest(root, excluded)
    if not artifacts:
        raise ValueError("comparison artifact manifest would be empty")
    write_json(manifest_path, {"status": "complete", "artifacts": artifacts})
    result = {
        "status": "ok",
        "artifact_count": len(artifacts),
        "checks": {
            "compatibility": {
                "status": compatibility["status"],
                "individual_validations": compatibility["individual_validations"],
            },
            "events": event_counts,
            "union": union,
            "analysis": {
                "status": analysis["status"],
                "common_atlas_threshold": analysis["common_atlas_threshold"],
            },
            "coordinates": coordinates,
            "common_atlas": atlas,
            "natural_and_matched": natural,
            "paper_diversity": paper,
            "figures": figures,
            "report": report,
        },
    }
    write_json(validation_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--expected-replicates", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.comparison_root.is_dir():
        parser.error(f"comparison root does not exist: {args.comparison_root}")
    if args.expected_replicates < 1:
        parser.error("expected-replicates must be positive")
    print(json.dumps(validate(args), sort_keys=True))


if __name__ == "__main__":
    main()
