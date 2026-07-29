#!/usr/bin/env python3
"""Validate canonical run-analysis artifacts and write a checksummed manifest."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from spacehasten.analysis.artifacts import write_json

INTERMEDIATE_DIRECTORIES = {"chunks", "inputs", "logs", "resampling_chunks"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_ids(path: Path) -> npt.NDArray[np.int64]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        if "spacehastenid" not in (rows.fieldnames or ()):
            raise ValueError(f"manifest lacks spacehastenid: {path}")
        identifiers = np.asarray([int(row["spacehastenid"]) for row in rows], dtype=np.int64)
    if len(identifiers) == 0:
        raise ValueError(f"manifest IDs are empty: {path}")
    return np.asarray(list(dict.fromkeys(identifiers.tolist())), dtype=np.int64)


def validate_receipt(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    receipt = read_json(path)
    if receipt.get("status") not in {"complete", "ok"}:
        raise ValueError(f"stage receipt is not complete: {path}")
    for output in receipt.get("outputs", receipt.get("artifacts", [])):
        artifact = root / str(output["path"])
        if not artifact.is_file() or artifact.stat().st_size != int(output["bytes"]):
            raise ValueError(f"receipt output is missing or has changed size: {artifact}")
        if sha256(artifact) != output["sha256"]:
            raise ValueError(f"receipt output digest mismatch: {artifact}")
    return receipt


def validate_npz_ids(
    path: Path,
    expected_ids: npt.NDArray[np.int64],
    *,
    coordinate_field: str | None = None,
    similarity_field: str | None = None,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        identifiers = data["spacehastenid"].astype(np.int64)
        if len(np.unique(identifiers)) != len(identifiers) or set(identifiers) != set(expected_ids):
            raise ValueError(f"artifact IDs do not match selected manifest: {path}")
        result: dict[str, Any] = {"rows": len(identifiers)}
        if coordinate_field:
            coordinates = data[coordinate_field]
            if coordinates.shape != (len(identifiers), 2) or not np.isfinite(coordinates).all():
                raise ValueError(f"coordinates are invalid: {path}")
            result["coordinates_finite"] = True
        if similarity_field:
            similarities = data[similarity_field]
            if not np.isfinite(similarities).all() or np.any(
                (similarities < 0) | (similarities > 1)
            ):
                raise ValueError(f"similarities are invalid: {path}")
            result["similarities_in_range"] = True
    return result


def validate_snapshot(
    receipt_path: Path,
    database: Path | None,
    quick_check: bool,
) -> dict[str, Any]:
    receipt = read_json(receipt_path)
    if receipt.get("source_quick_check") != "ok" or receipt.get("snapshot_quick_check") != "ok":
        raise ValueError("snapshot receipt does not record successful integrity checks")
    result = {
        "receipt": str(receipt_path.resolve()),
        "snapshot_quick_check": receipt["snapshot_quick_check"],
        "size_bytes": int(receipt["size_bytes"]),
        "counts": receipt.get("counts", {}),
    }
    if database is not None:
        database = database.resolve()
        if not database.is_file() or database.stat().st_size != int(receipt["size_bytes"]):
            raise ValueError("snapshot database is missing or differs in size from its receipt")
        result["database"] = str(database)
        if quick_check:
            with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
                observed = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            if observed != "ok":
                raise ValueError(f"snapshot quick_check failed: {observed}")
            result["rechecked_quick_check"] = observed
    return result


def validate_report(markdown: Path | None, html: Path | None) -> dict[str, Any]:
    if markdown is None and html is None:
        return {"status": "not_requested"}
    if markdown is None or html is None:
        raise ValueError("Markdown and HTML report paths must be supplied together")
    if not markdown.is_file() or not html.is_file() or html.stat().st_size == 0:
        raise FileNotFoundError("Markdown or HTML report is missing or empty")
    source = markdown.read_text(encoding="utf-8")
    links = re.findall(r"!\[[^]]*\]\(([^)]+)\)", source)
    missing = [link for link in links if not (markdown.parent / link).is_file()]
    if missing:
        raise ValueError(f"Markdown report has missing image links: {missing}")
    rendered = html.read_text(encoding="utf-8")
    embedded = len(re.findall(r"src=[\"']data:image/", rendered))
    if embedded < len(links):
        raise ValueError("standalone HTML does not embed every Markdown image")
    return {
        "status": "complete",
        "markdown": str(markdown.resolve()),
        "html": str(html.resolve()),
        "image_links": len(links),
        "embedded_images": embedded,
    }


def artifact_manifest(root: Path, excluded: set[Path]) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path in excluded:
            continue
        relative = path.relative_to(root)
        if INTERMEDIATE_DIRECTORIES.intersection(relative.parts):
            continue
        artifacts.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return artifacts


def validate_structure(root: Path) -> tuple[npt.NDArray[np.int64], dict[str, Any]]:
    receipt = read_json(root / "_SUCCESS.json")
    if receipt.get("status") != "complete":
        raise ValueError("selected structure cache is not complete")
    manifest = root / "selected_manifest.csv.gz"
    structures = root / "structure_cache.csv.gz"
    fingerprints = root / "fingerprints.npz"
    for path, field in (
        (manifest, "manifest_sha256"),
        (structures, "structure_sha256"),
        (fingerprints, "fingerprint_sha256"),
    ):
        if not path.is_file() or sha256(path) != receipt.get(field):
            raise ValueError(f"selected structure cache digest mismatch: {path}")
    identifiers = read_ids(manifest)
    with np.load(fingerprints, allow_pickle=False) as data:
        fingerprint_ids = data["spacehastenid"].astype(np.int64)
        words = data["words"]
    if not np.array_equal(identifiers, fingerprint_ids) or words.shape != (len(identifiers), 16):
        raise ValueError("selected manifest and fingerprint cache differ")
    return identifiers, receipt


def validate_standard(root: Path) -> dict[str, Any]:
    receipt = validate_receipt(root, "_SUCCESS.json")
    required = {
        "round_metrics.csv",
        "coverage.csv",
        "cutoff_curve.csv",
        "family_metrics.csv",
        "calibration_metrics.csv",
        "analysis_manifest.json",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise ValueError(f"standard analysis lacks required artifacts: {missing}")
    with (root / "round_metrics.csv").open("rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "selected" not in rows[0]:
        raise ValueError("standard round metrics lack selected counts")
    receipt["selected_attempts"] = sum(int(row["selected"]) for row in rows)
    receipt["cumulative_hits"] = int(rows[-1].get("cumulative_hits", 0))
    return receipt


def validate_figures(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "not_requested"}
    figures = [
        item
        for item in path.rglob("*")
        if item.is_file()
        and item.suffix.lower() in {".png", ".pdf"}
        and not INTERMEDIATE_DIRECTORIES.intersection(item.relative_to(path).parts)
    ]
    if not figures or any(item.stat().st_size == 0 for item in figures):
        raise ValueError("figure set is empty or contains empty files")
    return {
        "status": "complete",
        "png_count": sum(item.suffix.lower() == ".png" for item in figures),
        "pdf_count": sum(item.suffix.lower() == ".pdf" for item in figures),
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.analysis_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    final_path = root / "FINAL_VALIDATION.json"
    manifest_path = root / "artifact_manifest.json"
    if (final_path.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError("final validation outputs exist; pass --overwrite")
    checks: dict[str, Any] = {
        "snapshot": validate_snapshot(args.snapshot_receipt, args.database, args.quick_check),
        "standard": validate_standard(args.standard_root),
    }
    if args.reference_root:
        receipt = validate_receipt(args.reference_root, "_SUCCESS.json")
        snapshot_seed_count = (
            checks["snapshot"].get("counts", {}).get("dock_iterations", {}).get("0")
        )
        if snapshot_seed_count is not None and int(receipt.get("seed_count", -1)) != int(
            snapshot_seed_count
        ):
            raise ValueError("translated seed count differs from snapshot seed count")
        checks["seed_reference"] = receipt
    selected_ids, structure_receipt = validate_structure(args.structure_root)
    selected_attempts = int(
        structure_receipt.get("selected_attempts", structure_receipt.get("rows", -1))
    )
    checks["structure_cache"] = {
        "status": "complete",
        "rows": len(selected_ids),
        "selected_attempts": selected_attempts,
        "receipt_rows": structure_receipt.get("rows"),
    }
    if int(structure_receipt.get("rows", -1)) != len(selected_ids):
        raise ValueError("structure receipt row count differs from selected manifest")
    if int(checks["standard"].get("selected_attempts", -1)) != selected_attempts:
        raise ValueError("standard selected count differs from selected manifest")
    if args.selected_root:
        receipt = validate_receipt(args.selected_root, "_SUCCESS.json")
        if int(receipt.get("selected_attempts", -1)) != selected_attempts:
            raise ValueError("selected-cache analysis count differs from manifest")
        if int(receipt.get("hits", -1)) != int(checks["standard"].get("cumulative_hits", -2)):
            raise ValueError("selected-cache and standard cumulative hit counts differ")
        checks["selected_analysis"] = receipt
    if args.run_metadata_root:
        checks["run_metadata"] = validate_receipt(args.run_metadata_root, "_SUCCESS.json")
    if args.resampling_root:
        checks["resampling"] = validate_receipt(args.resampling_root, "_SUCCESS.json")
    if args.selected_atlas_root:
        receipt = validate_receipt(args.selected_atlas_root, "_SUCCESS.json")
        if int(receipt.get("selected_compounds", -1)) != len(selected_ids):
            raise ValueError("selected-atlas count differs from selected manifest")
        if float(receipt.get("minimum_similarity", -1)) < float(
            receipt.get("similarity_threshold", 0)
        ):
            raise ValueError("selected-atlas assignments violate the similarity threshold")
        checks["selected_atlas"] = receipt
    if args.random_seed_root:
        receipt = validate_receipt(args.random_seed_root, "_SUCCESS.json")
        if int(receipt.get("observed_virtual_hits", -1)) != int(
            checks["standard"].get("cumulative_hits", -2)
        ):
            raise ValueError("matched-random hit count differs from standard analysis")
        snapshot_seed_count = (
            checks["snapshot"].get("counts", {}).get("dock_iterations", {}).get("0")
        )
        if snapshot_seed_count is not None and int(receipt.get("seed_count", -1)) != int(
            snapshot_seed_count
        ):
            raise ValueError("matched-random seed count differs from snapshot")
        checks["matched_random_seed"] = receipt
    if args.fixed_reference_root:
        receipt = validate_receipt(args.fixed_reference_root, "_SUCCESS.json")
        snapshot_seed_count = (
            checks["snapshot"].get("counts", {}).get("dock_iterations", {}).get("0")
        )
        if snapshot_seed_count is not None and int(receipt.get("seed_count", -1)) != int(
            snapshot_seed_count
        ):
            raise ValueError("fixed-reference seed count differs from snapshot")
        umap = receipt.get("umap", {})
        if umap.get("coordinate_shape") != [int(receipt["seed_count"]), 2] or not umap.get(
            "coordinates_finite"
        ):
            raise ValueError("fixed-reference seed coordinates are invalid")
        checks["fixed_reference"] = receipt
    if args.portfolio_history_root:
        receipt = validate_receipt(args.portfolio_history_root, "_SUCCESS.json")
        if int(receipt.get("selected", -1)) != selected_attempts:
            raise ValueError("portfolio-history selected count differs from manifest")
        checks["portfolio_history"] = receipt
    if args.portfolio_enrichment_root:
        receipt = validate_receipt(args.portfolio_enrichment_root, "_SUCCESS.json")
        if int(receipt.get("selected", -1)) != selected_attempts:
            raise ValueError("portfolio-enrichment selected count differs from manifest")
        if int(receipt.get("hits", -1)) != int(checks["standard"].get("cumulative_hits", -2)):
            raise ValueError("portfolio-enrichment and standard hit counts differ")
        checks["portfolio_enrichment"] = receipt
    if args.paper_diversity_root:
        receipt = validate_receipt(args.paper_diversity_root, "_SUCCESS.json")
        if int(receipt.get("virtual_hits", -1)) != int(
            checks["standard"].get("cumulative_hits", -2)
        ):
            raise ValueError("paper-aligned diversity and standard hit counts differ")
        if float(receipt.get("cluster_similarity", -1)) != 0.55:
            raise ValueError("paper-aligned diversity does not use Tanimoto 0.55")
        checks["paper_aligned_diversity"] = receipt
    if args.nearest_seed:
        checks["nearest_seed"] = validate_npz_ids(
            args.nearest_seed, selected_ids, similarity_field="tanimoto"
        )
    if args.umap:
        checks["umap"] = validate_npz_ids(args.umap, selected_ids, coordinate_field="umap")
    checks["figures"] = validate_figures(args.figures_root)
    checks["report"] = validate_report(args.markdown, args.html)
    excluded = {final_path, manifest_path}
    artifacts = artifact_manifest(root, excluded)
    if not artifacts:
        raise ValueError("analysis root contains no canonical artifacts")
    write_json(manifest_path, {"status": "complete", "artifacts": artifacts})
    result = {"status": "ok", "checks": checks, "artifact_count": len(artifacts)}
    write_json(final_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--snapshot-receipt", type=Path, required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--quick-check", action="store_true")
    parser.add_argument("--standard-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--structure-root", type=Path, required=True)
    parser.add_argument("--selected-root", type=Path)
    parser.add_argument("--run-metadata-root", type=Path)
    parser.add_argument("--resampling-root", type=Path)
    parser.add_argument("--selected-atlas-root", type=Path)
    parser.add_argument("--random-seed-root", type=Path)
    parser.add_argument("--fixed-reference-root", type=Path)
    parser.add_argument("--portfolio-history-root", type=Path)
    parser.add_argument("--portfolio-enrichment-root", type=Path)
    parser.add_argument("--paper-diversity-root", type=Path)
    parser.add_argument("--nearest-seed", type=Path)
    parser.add_argument("--umap", type=Path)
    parser.add_argument("--figures-root", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    required_paths = [args.snapshot_receipt, args.standard_root, args.structure_root]
    if args.database:
        required_paths.append(args.database)
    for path in required_paths:
        if not path.exists():
            parser.error(f"required input does not exist: {path}")
    result = validate(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
