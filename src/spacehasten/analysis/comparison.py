"""Reusable preparation and metrics for two-workflow comparisons."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from .cached import family_distribution, sampled_diversity
from .coverage import coverage_depth, coverage_summary
from .umap import rdkit_words_to_fpsim2_words

EVENT_FIELDS = (
    "workflow",
    "workflow_label",
    "local_id",
    "reghash",
    "smiles",
    "dock_score",
    "round",
    "rank",
    "block_50k",
    "cumulative_budget",
    "rank_source",
    "scored",
    "hit",
    "strict_hit",
    "nearest_seed_tanimoto",
    "adaptive_boundary",
    "block_boundary_kind",
)
DESCRIPTOR_FIELDS = (
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
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(root.resolve()) if root else resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    compression: dict[str, Any] | None = (
        {"method": "gzip", "mtime": 0} if path.suffix == ".gz" else None
    )
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_smi(path: Path, frame: pd.DataFrame, id_column: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as text,
    ):
        for row in frame[["smiles", id_column]].itertuples(index=False):
            smiles = str(row.smiles)
            if not smiles or any(character.isspace() for character in smiles):
                raise ValueError(
                    f"invalid whitespace-delimited SMILES for {getattr(row, id_column)}"
                )
            text.write(f"{smiles} {int(getattr(row, id_column))}\n")
    temporary.replace(path)


def semantic_digest(frame: pd.DataFrame, fields: tuple[str, ...]) -> str:
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ValueError(f"semantic digest lacks fields: {sorted(missing)}")
    digest = hashlib.sha256()
    for values in frame.loc[:, fields].itertuples(index=False, name=None):
        normalized = [_canonical_scalar(value) for value in values]
        digest.update(json.dumps(normalized, separators=(",", ":"), ensure_ascii=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_scalar(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".15g")
    return str(value)


def validate_individual_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "ok":
        raise ValueError(f"individual validation is not current and successful: {path}")
    return receipt


def database_protocol(path: Path) -> dict[str, Any]:
    seed_identity = hashlib.sha256()
    seed_scores = hashlib.sha256()
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check != "ok":
            raise ValueError(f"database quick check failed: {path}: {quick_check}")
        seed_count = 0
        previous: str | None = None
        for reghash, score in connection.execute(
            "SELECT reghash,dock_score FROM data WHERE dock_iteration=0 ORDER BY reghash"
        ):
            identity = str(reghash)
            if not identity or identity == previous:
                raise ValueError(f"seed reghashes are empty or duplicated: {path}")
            previous = identity
            seed_identity.update(identity.encode())
            seed_identity.update(b"\n")
            seed_scores.update(identity.encode())
            seed_scores.update(b"\t")
            seed_scores.update(("" if score is None else float(score).hex()).encode())
            seed_scores.update(b"\n")
            seed_count += 1
        parameter = connection.execute("SELECT dock_param FROM docking_param").fetchone()
        grid = connection.execute("SELECT dock_grid FROM docking_grid").fetchone()
        if parameter is None or grid is None:
            raise ValueError(f"docking protocol blobs are missing: {path}")
        property_rows = [
            {
                "property": str(name),
                "is_double": int(is_double),
                "min_limit": _canonical_scalar(minimum),
                "max_limit": _canonical_scalar(maximum),
            }
            for name, is_double, minimum, maximum in connection.execute(
                "SELECT property,is_double,min_limit,max_limit FROM properties ORDER BY property"
            )
        ]
    parameter_bytes = bytes(parameter[0])
    grid_bytes = bytes(grid[0])
    properties_payload = json.dumps(property_rows, separators=(",", ":"), sort_keys=True).encode()
    return {
        "database": str(path.resolve()),
        "quick_check": quick_check,
        "seed_count": seed_count,
        "seed_reghash_digest": seed_identity.hexdigest(),
        "seed_score_map_digest": seed_scores.hexdigest(),
        "docking_parameter_sha256": hashlib.sha256(parameter_bytes).hexdigest(),
        "docking_grid_sha256": hashlib.sha256(grid_bytes).hexdigest(),
        "properties_semantic_sha256": hashlib.sha256(properties_payload).hexdigest(),
        "properties": property_rows,
    }


def load_fingerprints(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        identifiers = data["spacehastenid"].astype(np.int64)
        words = data["words"].astype(np.uint64)
        popcounts = data["popcounts"].astype(np.uint16)
    if identifiers.ndim != 1 or words.shape != (len(identifiers), 16):
        raise ValueError(f"fingerprint cache has an invalid shape: {path}")
    if len(np.unique(identifiers)) != len(identifiers) or popcounts.shape != (len(identifiers),):
        raise ValueError(f"fingerprint IDs or popcounts are invalid: {path}")
    return identifiers, words, popcounts


def load_nearest(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as data:
        identifiers = data["spacehastenid"].astype(np.int64)
        tanimoto = data["tanimoto"].astype(np.float64)
    if len(np.unique(identifiers)) != len(identifiers):
        raise ValueError(f"nearest-seed IDs are duplicated: {path}")
    if not np.isfinite(tanimoto).all() or np.any((tanimoto < 0) | (tanimoto > 1)):
        raise ValueError(f"nearest-seed coefficients are outside [0, 1]: {path}")
    return pd.DataFrame({"local_id": identifiers, "nearest_seed_tanimoto": tanimoto})


def normalize_events(
    manifest_path: Path,
    nearest_path: Path,
    *,
    workflow: str,
    label: str,
    hit_cutoff: float,
    strict_cutoff: float,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {"spacehastenid", "reghash", "smiles", "dock_score", "round", "rank", "rank_source"}
    if missing := required - set(manifest.columns):
        raise ValueError(f"{workflow} manifest lacks fields: {sorted(missing)}")
    manifest = manifest.sort_values(["round", "rank"], kind="stable").reset_index(drop=True)
    if manifest["spacehastenid"].isna().any() or not manifest["spacehastenid"].is_unique:
        raise ValueError(f"{workflow} local IDs must be non-null and unique")
    if manifest["reghash"].isna().any() or not manifest["reghash"].is_unique:
        raise ValueError(f"{workflow} reghashes must be non-null and unique")
    nearest = load_nearest(nearest_path)
    events = pd.DataFrame(
        {
            "workflow": workflow,
            "workflow_label": label,
            "local_id": manifest["spacehastenid"].astype(np.int64),
            "reghash": manifest["reghash"].astype(str),
            "smiles": manifest["smiles"].astype(str),
            "dock_score": pd.to_numeric(manifest["dock_score"], errors="coerce"),
            "round": manifest["round"].astype(np.int64),
            "rank": manifest["rank"].astype(np.int64),
            "rank_source": manifest["rank_source"].fillna("unavailable").astype(str),
        }
    )
    events["cumulative_budget"] = np.arange(1, len(events) + 1, dtype=np.int64)
    events["block_50k"] = ((events["cumulative_budget"] - 1) // 50_000 + 1).astype(np.int64)
    events["scored"] = events["dock_score"].notna()
    events["hit"] = events["scored"] & events["dock_score"].le(hit_cutoff)
    events["strict_hit"] = events["scored"] & events["dock_score"].le(strict_cutoff)
    round_boundaries = set(
        events.groupby("round", sort=True)["cumulative_budget"].max().astype(int)
    )
    events["adaptive_boundary"] = events["cumulative_budget"].isin(round_boundaries)
    block_ends = events["cumulative_budget"].mod(50_000).eq(0)
    events["block_boundary_kind"] = "within_block"
    events.loc[block_ends & events["adaptive_boundary"], "block_boundary_kind"] = (
        "adaptive_boundary"
    )
    events.loc[block_ends & ~events["adaptive_boundary"], "block_boundary_kind"] = (
        "reconstructed_prefix"
    )
    events = events.merge(nearest, on="local_id", how="left", validate="one_to_one")
    if events["nearest_seed_tanimoto"].isna().any():
        raise ValueError(f"{workflow} nearest-seed result does not cover every selected compound")
    return events.loc[:, EVENT_FIELDS]


def _canonical_smiles(value: str) -> str:
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise ValueError(f"invalid selected SMILES: {value!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _cache_by_reghash(
    events: pd.DataFrame,
    structure_path: Path,
    fingerprint_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[int, int]]:
    structures = pd.read_csv(structure_path)
    required = {"spacehastenid", "reghash", *DESCRIPTOR_FIELDS}
    if missing := required - set(structures.columns):
        raise ValueError(f"structure cache lacks fields: {sorted(missing)}")
    if not structures["spacehastenid"].is_unique or not structures["reghash"].is_unique:
        raise ValueError("structure cache identifiers must be unique")
    joined = events.merge(
        structures,
        left_on=["local_id", "reghash"],
        right_on=["spacehastenid", "reghash"],
        how="left",
        validate="one_to_one",
    )
    if joined["spacehastenid"].isna().any():
        raise ValueError("structure cache does not match normalized events")
    identifiers, words, popcounts = load_fingerprints(fingerprint_path)
    row_by_id = {int(identifier): index for index, identifier in enumerate(identifiers)}
    if set(row_by_id) != set(events["local_id"].astype(int)):
        raise ValueError("fingerprint IDs do not match normalized events")
    return joined.set_index("reghash", drop=False), words, popcounts, row_by_id


def build_union(
    left_events: pd.DataFrame,
    right_events: pd.DataFrame,
    *,
    left_structure: Path,
    right_structure: Path,
    left_fingerprints: Path,
    right_fingerprints: Path,
    seed_count: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, int]]:
    left, left_words, left_popcounts, left_rows = _cache_by_reghash(
        left_events, left_structure, left_fingerprints
    )
    right, right_words, right_popcounts, right_rows = _cache_by_reghash(
        right_events, right_structure, right_fingerprints
    )
    shared_hashes = sorted(set(left.index).intersection(right.index))
    structure_mismatches = 0
    fingerprint_mismatches = 0
    for reghash in shared_hashes:
        left_row = left.loc[reghash]
        right_row = right.loc[reghash]
        if str(left_row["smiles"]) != str(right_row["smiles"]):
            structure_mismatches += _canonical_smiles(str(left_row["smiles"])) != _canonical_smiles(
                str(right_row["smiles"])
            )
        left_position = left_rows[int(left_row["local_id"])]
        right_position = right_rows[int(right_row["local_id"])]
        fingerprint_mismatches += not (
            np.array_equal(left_words[left_position], right_words[right_position])
            and left_popcounts[left_position] == right_popcounts[right_position]
        )
        for field in ("typed_scaffold", "generic_framework"):
            structure_mismatches += str(left_row[field]) != str(right_row[field])
    if structure_mismatches or fingerprint_mismatches:
        raise ValueError(
            f"shared structure validation failed: structures={structure_mismatches}, "
            f"fingerprints={fingerprint_mismatches}"
        )

    union_hashes = sorted(set(left.index).union(right.index))
    union_words = np.empty((len(union_hashes), 16), dtype=np.uint64)
    union_popcounts = np.empty(len(union_hashes), dtype=np.uint16)
    rows: list[dict[str, Any]] = []
    for common_id, reghash in enumerate(union_hashes, 1):
        left_row = left.loc[reghash] if reghash in left.index else None
        right_row = right.loc[reghash] if reghash in right.index else None
        source = left_row if left_row is not None else right_row
        assert source is not None
        if left_row is not None:
            position = left_rows[int(left_row["local_id"])]
            union_words[common_id - 1] = left_words[position]
            union_popcounts[common_id - 1] = left_popcounts[position]
        else:
            assert right_row is not None
            position = right_rows[int(right_row["local_id"])]
            union_words[common_id - 1] = right_words[position]
            union_popcounts[common_id - 1] = right_popcounts[position]
        row: dict[str, Any] = {
            "common_id": common_id,
            "atlas_query_id": seed_count + common_id,
            "reghash": reghash,
            "smiles": str(source["smiles"]),
            "left_local_id": int(left_row["local_id"]) if left_row is not None else None,
            "right_local_id": int(right_row["local_id"]) if right_row is not None else None,
            "left_present": left_row is not None,
            "right_present": right_row is not None,
        }
        for field in DESCRIPTOR_FIELDS:
            row[field] = source[field]
        rows.append(row)
    union = pd.DataFrame(rows)
    counts = {
        "union": len(union),
        "shared": len(shared_hashes),
        "left_only": len(left) - len(shared_hashes),
        "right_only": len(right) - len(shared_hashes),
        "structure_mismatches": structure_mismatches,
        "fingerprint_mismatches": fingerprint_mismatches,
    }
    return union, union_words, union_popcounts, counts


def build_fpsim2_index(
    union: pd.DataFrame,
    words: np.ndarray,
    popcounts: np.ndarray,
    output: Path,
) -> None:
    from FPSim2 import FPSim2Engine

    from spacehasten.remote.cluster import _build_fpsim2_index

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    _build_fpsim2_index(
        [(str(row.smiles), int(row.common_id)) for row in union.itertuples(index=False)],
        temporary,
    )
    temporary.replace(output)
    engine = FPSim2Engine(str(output))
    index_ids = np.asarray(engine.fps[:, 0], dtype=np.int64)
    order = np.argsort(index_ids)
    expected_ids = union["common_id"].to_numpy(np.int64)
    if not np.array_equal(index_ids[order], expected_ids):
        raise ValueError("common FPSim2 index IDs differ from union IDs")
    observed_words = np.asarray(engine.fps[order, 1:-1], dtype=np.uint64)
    observed_popcounts = np.asarray(engine.fps[order, -1], dtype=np.uint16)
    expected_words = rdkit_words_to_fpsim2_words(words)
    if not np.array_equal(observed_words, expected_words) or not np.array_equal(
        observed_popcounts, popcounts
    ):
        raise ValueError("common FPSim2 index fingerprints differ from the union cache")


def prepare_comparison(
    *,
    left_name: str,
    left_label: str,
    left_database: Path,
    left_analysis: Path,
    left_validation: Path,
    right_name: str,
    right_label: str,
    right_database: Path,
    right_analysis: Path,
    right_validation: Path,
    seed_atlas_definition: Path,
    output_root: Path,
    hit_cutoff: float,
    strict_cutoff: float,
) -> dict[str, Any]:
    if strict_cutoff >= hit_cutoff:
        raise ValueError("strict cutoff must be numerically lower than the primary cutoff")
    left_receipt = validate_individual_receipt(left_validation)
    right_receipt = validate_individual_receipt(right_validation)
    definition = json.loads(seed_atlas_definition.read_text(encoding="utf-8"))
    seed_count = int(definition["seed_count"])
    if definition.get("fingerprint_type") != "Morgan" or definition.get(
        "fingerprint_parameters"
    ) != {"radius": 2, "fpSize": 1024}:
        raise ValueError("seed atlas does not use Morgan radius-2/1024 fingerprints")

    protocol = {
        left_name: database_protocol(left_database),
        right_name: database_protocol(right_database),
    }
    required_equal = (
        "seed_count",
        "seed_reghash_digest",
        "seed_score_map_digest",
        "docking_parameter_sha256",
        "docking_grid_sha256",
        "properties_semantic_sha256",
    )
    differences = [
        field
        for field in required_equal
        if protocol[left_name][field] != protocol[right_name][field]
    ]
    if differences:
        raise ValueError(f"workflow protocols are incompatible: {differences}")
    if protocol[left_name]["seed_count"] != seed_count:
        raise ValueError("database seed count differs from the reusable seed atlas")

    cache = output_root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    workflow_paths: dict[str, dict[str, Path]] = {}
    events_by_name: dict[str, pd.DataFrame] = {}
    for name, label, analysis in (
        (left_name, left_label, left_analysis),
        (right_name, right_label, right_analysis),
    ):
        paths = {
            "manifest": analysis / "structure_cache/selected_manifest.csv.gz",
            "structures": analysis / "structure_cache/structure_cache.csv.gz",
            "fingerprints": analysis / "structure_cache/fingerprints.npz",
            "nearest": analysis / "nearest_seed/nearest_seed_similarity.npz",
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        workflow_paths[name] = paths
        events = normalize_events(
            paths["manifest"],
            paths["nearest"],
            workflow=name,
            label=label,
            hit_cutoff=hit_cutoff,
            strict_cutoff=strict_cutoff,
        )
        events_by_name[name] = events
        write_csv(cache / f"{name}_events.csv.gz", events)

    union, words, popcounts, counts = build_union(
        events_by_name[left_name],
        events_by_name[right_name],
        left_structure=workflow_paths[left_name]["structures"],
        right_structure=workflow_paths[right_name]["structures"],
        left_fingerprints=workflow_paths[left_name]["fingerprints"],
        right_fingerprints=workflow_paths[right_name]["fingerprints"],
        seed_count=seed_count,
    )
    union_path = cache / "common_union_manifest.csv.gz"
    fingerprints_path = cache / "common_union_fingerprints.npz"
    smiles_path = cache / "common_union.smi.gz"
    atlas_manifest_path = cache / "common_atlas_input_manifest.csv.gz"
    write_csv(union_path, union)
    write_npz(
        fingerprints_path,
        spacehastenid=union["common_id"].to_numpy(np.int64),
        words=words,
        popcounts=popcounts,
    )
    write_smi(smiles_path, union, "common_id")
    atlas_manifest = union[["atlas_query_id", "common_id", "reghash", "smiles"]].rename(
        columns={"atlas_query_id": "spacehastenid"}
    )
    write_csv(atlas_manifest_path, atlas_manifest)
    index_path = cache / "common_atlas/molecule_index/molecules_fp.h5"
    build_fpsim2_index(union, words, popcounts, index_path)

    event_semantics = {
        name: semantic_digest(events, EVENT_FIELDS) for name, events in events_by_name.items()
    }
    union_fields = (
        "common_id",
        "atlas_query_id",
        "reghash",
        "smiles",
        "left_local_id",
        "right_local_id",
        "left_present",
        "right_present",
        *DESCRIPTOR_FIELDS,
    )
    union_semantic = semantic_digest(union, union_fields)
    compatibility = {
        "status": "compatible",
        "left_workflow": left_name,
        "right_workflow": right_name,
        "labels": {left_name: left_label, right_name: right_label},
        "hit_cutoff": hit_cutoff,
        "strict_cutoff": strict_cutoff,
        "fingerprint": {
            "type": "Morgan binary",
            "radius": 2,
            "n_bits": 1024,
            "use_chirality": False,
            "word_count": 16,
            "word_dtype": "uint64",
        },
        "protocol": protocol,
        "individual_validations": {
            left_name: {
                "path": str(left_validation.resolve()),
                "sha256": sha256(left_validation),
                "status": left_receipt["status"],
            },
            right_name: {
                "path": str(right_validation.resolve()),
                "sha256": sha256(right_validation),
                "status": right_receipt["status"],
            },
        },
        "event_semantic_sha256": event_semantics,
        "union_semantic_sha256": union_semantic,
        "union_counts": counts,
        "seed_atlas_definition": file_record(seed_atlas_definition),
        "inputs": {
            name: {key: file_record(path) for key, path in paths.items()}
            for name, paths in workflow_paths.items()
        },
        "outputs": [
            file_record(cache / f"{left_name}_events.csv.gz", root=output_root),
            file_record(cache / f"{right_name}_events.csv.gz", root=output_root),
            file_record(union_path, root=output_root),
            file_record(fingerprints_path, root=output_root),
            file_record(smiles_path, root=output_root),
            file_record(atlas_manifest_path, root=output_root),
            file_record(index_path, root=output_root),
        ],
    }
    write_json(output_root / "comparison_compatibility.json", compatibility)
    write_json(cache / "_SUCCESS.json", {**compatibility, "status": "complete"})
    return compatibility


def refresh_comparison_semantics(root: Path) -> dict[str, Any]:
    """Refresh semantic digests and byte hashes without rebuilding comparison caches."""
    compatibility_path = root / "comparison_compatibility.json"
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    if compatibility.get("status") != "compatible":
        raise ValueError("comparison compatibility gate did not pass")
    cache = root / "cache"
    names = (str(compatibility["left_workflow"]), str(compatibility["right_workflow"]))
    compatibility["event_semantic_sha256"] = {
        name: semantic_digest(pd.read_csv(cache / f"{name}_events.csv.gz"), EVENT_FIELDS)
        for name in names
    }
    union = pd.read_csv(cache / "common_union_manifest.csv.gz")
    compatibility["union_semantic_sha256"] = semantic_digest(
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
            *DESCRIPTOR_FIELDS,
        ),
    )
    compatibility["outputs"] = [
        file_record(root / str(record["path"]), root=root) for record in compatibility["outputs"]
    ]
    write_json(compatibility_path, compatibility)
    write_json(cache / "_SUCCESS.json", {**compatibility, "status": "complete"})
    return compatibility


def common_atlas_assignments(
    union: pd.DataFrame,
    assignments_path: Path,
    *,
    threshold: float,
) -> pd.DataFrame:
    with np.load(assignments_path, allow_pickle=False) as data:
        query_ids = data["spacehastenid"].astype(np.int64)
        clusters = data["clusterid"].astype(np.int64)
        similarities = data["centroid_similarity"].astype(np.float64)
    if len(np.unique(query_ids)) != len(query_ids) or len(query_ids) != len(union):
        raise ValueError("common-atlas assignments are incomplete or duplicated")
    mapping = pd.DataFrame(
        {"atlas_query_id": query_ids, "clusterid": clusters, "centroid_similarity": similarities}
    )
    result = union[["common_id", "atlas_query_id", "reghash"]].merge(
        mapping, on="atlas_query_id", how="left", validate="one_to_one"
    )
    if result["clusterid"].isna().any() or not np.isfinite(result["centroid_similarity"]).all():
        raise ValueError("common-atlas assignment join is incomplete")
    if result["centroid_similarity"].lt(threshold - 1e-7).any():
        raise ValueError("common-atlas assignment falls below the similarity threshold")
    result["clusterid"] = result["clusterid"].astype(np.int64)
    seed_count = int(union["atlas_query_id"].min() - union["common_id"].min())
    result["centroid_source"] = np.where(result["clusterid"].le(seed_count), "seed", "repair")
    return result


def natural_diversity_row(
    events: pd.DataFrame,
    union: pd.DataFrame,
    words: np.ndarray,
    popcounts: np.ndarray,
    atlas_by_common_id: np.ndarray,
    *,
    workflow: str,
    cohort: str,
    seed: int,
    pair_samples: int,
) -> dict[str, Any]:
    if cohort == "selected":
        selected = events
    elif cohort == "hit":
        selected = events[events["hit"]]
    elif cohort == "strict_hit":
        selected = events[events["strict_hit"]]
    else:
        raise ValueError(f"unknown comparison cohort: {cohort}")
    common_by_hash = pd.Series(union["common_id"].to_numpy(), index=union["reghash"])
    common_ids = selected["reghash"].map(common_by_hash).to_numpy(np.int64)
    rows = common_ids - 1
    internal, mc_se, completed = sampled_diversity(
        rows,
        words,
        popcounts.astype(np.int64),
        seed=seed,
        samples=pair_samples,
    )
    typed = family_distribution(union.iloc[rows]["typed_scaffold"].astype(str).to_numpy())
    generic = family_distribution(union.iloc[rows]["generic_framework"].astype(str).to_numpy())
    atlas = family_distribution(atlas_by_common_id[rows])
    return {
        "workflow": workflow,
        "cohort": cohort,
        "n": len(rows),
        "internal_diversity": internal,
        "internal_diversity_mc_se": mc_se,
        "pair_samples": completed,
        **{f"typed_{key}": value for key, value in typed.items()},
        **{f"generic_{key}": value for key, value in generic.items()},
        **{f"atlas_{key}": value for key, value in atlas.items()},
    }


def _score_summary(values: pd.Series) -> dict[str, float | int | None]:
    scores = pd.to_numeric(values, errors="coerce").dropna()
    if scores.empty:
        return {
            "score_mean": None,
            "score_median": None,
            "score_q05": None,
            "score_q25": None,
            "score_q75": None,
            "score_q95": None,
        }
    return {
        "score_mean": float(scores.mean()),
        "score_median": float(scores.median()),
        "score_q05": float(scores.quantile(0.05)),
        "score_q25": float(scores.quantile(0.25)),
        "score_q75": float(scores.quantile(0.75)),
        "score_q95": float(scores.quantile(0.95)),
    }


def endpoint_table(events_by_name: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, events in events_by_name.items():
        selected = len(events)
        scored = int(events["scored"].sum())
        hits = int(events["hit"].sum())
        strict_hits = int(events["strict_hit"].sum())
        rows.append(
            {
                "workflow": name,
                "workflow_label": str(events["workflow_label"].iloc[0]),
                "selected": selected,
                "scored": scored,
                "unresolved": selected - scored,
                "hits": hits,
                "strict_hits": strict_hits,
                "hit_rate_selected": hits / selected,
                "hit_rate_scored": hits / scored,
                "strict_hit_rate_selected": strict_hits / selected,
                "strict_hit_rate_scored": strict_hits / scored,
                **_score_summary(events.loc[events["scored"], "dock_score"]),
            }
        )
    return pd.DataFrame(rows)


def budget_table(events_by_name: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, events in events_by_name.items():
        for block, current in events.groupby("block_50k", sort=True):
            budget = int(current["cumulative_budget"].max())
            cumulative = events[events["cumulative_budget"].le(budget)]
            boundary = str(current.iloc[-1]["block_boundary_kind"])
            rows.append(
                {
                    "workflow": name,
                    "workflow_label": str(events["workflow_label"].iloc[0]),
                    "block_50k": int(block),
                    "cumulative_budget": budget,
                    "block_selected": len(current),
                    "block_scored": int(current["scored"].sum()),
                    "block_hits": int(current["hit"].sum()),
                    "block_strict_hits": int(current["strict_hit"].sum()),
                    "block_hit_rate_selected": float(current["hit"].mean()),
                    "cumulative_scored": int(cumulative["scored"].sum()),
                    "cumulative_hits": int(cumulative["hit"].sum()),
                    "cumulative_strict_hits": int(cumulative["strict_hit"].sum()),
                    "cumulative_hit_rate_selected": float(cumulative["hit"].mean()),
                    "boundary_kind": boundary,
                    "adaptive_boundary": boundary == "adaptive_boundary",
                    "rank_source": ";".join(sorted(set(current["rank_source"].astype(str)))),
                    **_score_summary(current.loc[current["scored"], "dock_score"]),
                }
            )
    return pd.DataFrame(rows)


def cutoff_table(
    events_by_name: dict[str, pd.DataFrame],
    *,
    minimum: float = -14.5,
    maximum: float = -10.0,
    step: float = 0.25,
) -> pd.DataFrame:
    rows = []
    cutoffs = np.arange(minimum, maximum + step / 2, step)
    for name, events in events_by_name.items():
        scores = events["dock_score"].to_numpy(float)
        scored = np.isfinite(scores)
        for cutoff in cutoffs:
            hits = int(np.count_nonzero(scored & (scores <= cutoff)))
            rows.append(
                {
                    "workflow": name,
                    "cutoff": float(cutoff),
                    "selected": len(events),
                    "scored": int(scored.sum()),
                    "hits": hits,
                    "hit_rate_selected": hits / len(events),
                    "hit_rate_scored": hits / scored.sum(),
                }
            )
    return pd.DataFrame(rows)


def top_k_table(events_by_name: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, events in events_by_name.items():
        scores = events.loc[events["scored"], "dock_score"].sort_values(kind="stable")
        for requested in (100, 500, 1_000, 5_000, 10_000, 25_000, 50_000, 100_000):
            count = min(requested, len(scores))
            selected = scores.iloc[:count]
            rows.append(
                {
                    "workflow": name,
                    "requested_top_k": requested,
                    "observed_top_k": count,
                    "best_score": float(selected.iloc[0]),
                    "median_score": float(selected.median()),
                    "worst_score": float(selected.iloc[-1]),
                    "mean_score": float(selected.mean()),
                }
            )
    return pd.DataFrame(rows)


def _event_common_ids(events: pd.DataFrame, common_by_hash: pd.Series) -> np.ndarray:
    identifiers = events["reghash"].map(common_by_hash)
    if identifiers.isna().any():
        raise ValueError("normalized events are not fully covered by the common union")
    return identifiers.to_numpy(np.int64)


def regional_tables(
    events_by_name: dict[str, pd.DataFrame],
    common_by_hash: pd.Series,
    atlas_by_common_id: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    depths = []
    cluster_frames: dict[str, pd.DataFrame] = {}
    for name, events in events_by_name.items():
        current = events.copy()
        common_ids = _event_common_ids(current, common_by_hash)
        current["clusterid"] = atlas_by_common_id[common_ids - 1]
        scored = current[current["scored"]]
        grouped = scored.groupby("clusterid", as_index=False).agg(
            scored=("scored", "size"), hits=("hit", "sum"), strict_hits=("strict_hit", "sum")
        )
        selected_counts = current.groupby("clusterid").size().rename("selected")
        grouped = grouped.merge(selected_counts, on="clusterid", how="outer").fillna(0)
        cluster_frames[name] = grouped
        hit_counts = grouped.loc[grouped["hits"].gt(0), "hits"].to_numpy(np.int64)
        summary = coverage_summary(hit_counts)
        summaries.append({"workflow": name, **summary})
        depths.extend(
            {"workflow": name, **row} for row in coverage_depth(hit_counts, max_threshold=100)
        )
    left_name, right_name = tuple(events_by_name)
    left = cluster_frames[left_name].rename(
        columns={
            column: f"{left_name}_{column}"
            for column in ("selected", "scored", "hits", "strict_hits")
        }
    )
    right = cluster_frames[right_name].rename(
        columns={
            column: f"{right_name}_{column}"
            for column in ("selected", "scored", "hits", "strict_hits")
        }
    )
    contrasts = left.merge(right, on="clusterid", how="outer").fillna(0)
    for name in (left_name, right_name):
        global_rate = float(
            events_by_name[name]["hit"].sum() / events_by_name[name]["scored"].sum()
        )
        contrasts[f"{name}_posterior_hit_rate"] = (contrasts[f"{name}_hits"] + 20 * global_rate) / (
            contrasts[f"{name}_scored"] + 20
        )
    contrasts["posterior_difference_pp"] = 100 * (
        contrasts[f"{right_name}_posterior_hit_rate"] - contrasts[f"{left_name}_posterior_hit_rate"]
    )
    contrasts["selected_by_both"] = contrasts[f"{left_name}_selected"].gt(0) & contrasts[
        f"{right_name}_selected"
    ].gt(0)
    return pd.DataFrame(summaries), pd.DataFrame(depths), contrasts


def nearest_seed_tables(
    events_by_name: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    bins = [-np.inf, 0.3, 0.4, 0.5, 0.7, np.inf]
    labels = ["<0.3", "0.3-<0.4", "0.4-<0.5", "0.5-<0.7", ">=0.7"]
    productivity = []
    for name, events in events_by_name.items():
        for cohort, current in (
            ("selected", events),
            ("hit", events[events["hit"]]),
            ("strict_hit", events[events["strict_hit"]]),
        ):
            values = current["nearest_seed_tanimoto"]
            summaries.append(
                {
                    "workflow": name,
                    "cohort": cohort,
                    "n": len(current),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "q05": float(values.quantile(0.05)),
                    "q95": float(values.quantile(0.95)),
                    **{
                        f"fraction_below_{threshold:g}": float(values.lt(threshold).mean())
                        for threshold in (0.3, 0.4, 0.5, 0.7)
                    },
                }
            )
        assigned = pd.cut(events["nearest_seed_tanimoto"], bins=bins, labels=labels, right=False)
        for label in labels:
            current = events[assigned == label]
            scored = int(current["scored"].sum())
            hits = int(current["hit"].sum())
            productivity.append(
                {
                    "workflow": name,
                    "nearest_seed_bin": label,
                    "selected": len(current),
                    "scored": scored,
                    "hits": hits,
                    "hit_rate_selected": hits / len(current) if len(current) else None,
                    "hit_rate_scored": hits / scored if scored else None,
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(productivity)


def overlap_table(
    events_by_name: dict[str, pd.DataFrame],
    common_by_hash: pd.Series,
    atlas_by_common_id: np.ndarray,
) -> pd.DataFrame:
    left_name, right_name = tuple(events_by_name)
    sets: dict[str, dict[str, set[Any]]] = {}
    for name, events in events_by_name.items():
        common_ids = _event_common_ids(events, common_by_hash)
        hit_ids = common_ids[events["hit"].to_numpy(bool)]
        sets[name] = {
            "selected_compounds": set(events["reghash"]),
            "hits": set(events.loc[events["hit"], "reghash"]),
            "strict_hits": set(events.loc[events["strict_hit"], "reghash"]),
            "seed_distant_selected": set(
                events.loc[events["nearest_seed_tanimoto"].lt(0.4), "reghash"]
            ),
            "seed_distant_hits": set(
                events.loc[events["hit"] & events["nearest_seed_tanimoto"].lt(0.4), "reghash"]
            ),
            "productive_regions": set(atlas_by_common_id[hit_ids - 1].tolist()),
        }
    rows = []
    for cohort in sets[left_name]:
        left = sets[left_name][cohort]
        right = sets[right_name][cohort]
        intersection = left & right
        union = left | right
        rows.append(
            {
                "cohort": cohort,
                f"{left_name}_count": len(left),
                f"{right_name}_count": len(right),
                "intersection": len(intersection),
                "union": len(union),
                "jaccard": len(intersection) / len(union) if union else None,
                f"{left_name}_directional_capture": len(intersection) / len(left) if left else None,
                f"{right_name}_directional_capture": len(intersection) / len(right)
                if right
                else None,
            }
        )
    return pd.DataFrame(rows)


def shared_score_tables(
    events_by_name: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left_name, right_name = tuple(events_by_name)
    left = (
        events_by_name[left_name]
        .loc[events_by_name[left_name]["scored"], ["reghash", "dock_score", "round"]]
        .rename(columns={"dock_score": f"{left_name}_score", "round": f"{left_name}_round"})
    )
    right = (
        events_by_name[right_name]
        .loc[events_by_name[right_name]["scored"], ["reghash", "dock_score", "round"]]
        .rename(columns={"dock_score": f"{right_name}_score", "round": f"{right_name}_round"})
    )
    pairs = left.merge(right, on="reghash", how="inner", validate="one_to_one")
    difference = pairs[f"{right_name}_score"] - pairs[f"{left_name}_score"]
    summary = pd.DataFrame(
        [
            {
                "shared_scored_compounds": len(pairs),
                "pearson": float(
                    pairs[[f"{left_name}_score", f"{right_name}_score"]].corr("pearson").iloc[0, 1]
                )
                if len(pairs) > 1
                else None,
                "spearman": float(
                    pairs[[f"{left_name}_score", f"{right_name}_score"]].corr("spearman").iloc[0, 1]
                )
                if len(pairs) > 1
                else None,
                f"mean_{right_name}_minus_{left_name}": float(difference.mean())
                if len(pairs)
                else None,
                "mean_absolute_difference": float(difference.abs().mean()) if len(pairs) else None,
                "maximum_absolute_difference": float(difference.abs().max())
                if len(pairs)
                else None,
                "bit_identical_scores": int(difference.eq(0).sum()),
                "absolute_difference_le_0_001": int(difference.abs().le(0.001).sum()),
                "absolute_difference_gt_0_1": int(difference.abs().gt(0.1).sum()),
                "absolute_difference_gt_1": int(difference.abs().gt(1.0).sum()),
            }
        ]
    )
    return summary, pairs


def exclusive_hit_table(
    events_by_name: dict[str, pd.DataFrame], union: pd.DataFrame
) -> pd.DataFrame:
    hit_sets = {
        name: set(events.loc[events["hit"], "reghash"]) for name, events in events_by_name.items()
    }
    rows = []
    union_by_hash = union.set_index("reghash")
    for name, events in events_by_name.items():
        other_name = next(value for value in events_by_name if value != name)
        exclusive = hit_sets[name] - hit_sets[other_name]
        current = events[events["reghash"].isin(exclusive)]
        structures = union_by_hash.loc[current["reghash"]]
        typed = family_distribution(structures["typed_scaffold"].astype(str).to_numpy())
        generic = family_distribution(structures["generic_framework"].astype(str).to_numpy())
        rows.append(
            {
                "workflow": name,
                "exclusive_hits": len(current),
                "score_median": float(current["dock_score"].median()) if len(current) else None,
                "nearest_seed_median": float(current["nearest_seed_tanimoto"].median())
                if len(current)
                else None,
                "mw_median": float(structures["MW"].median()) if len(current) else None,
                "clogp_median": float(structures["cLogP"].median()) if len(current) else None,
                "tpsa_median": float(structures["TPSA"].median()) if len(current) else None,
                "typed_q0": typed["q0"],
                "typed_q1": typed["q1"],
                "typed_q2": typed["q2"],
                "generic_q0": generic["q0"],
                "generic_q1": generic["q1"],
                "generic_q2": generic["q2"],
            }
        )
    return pd.DataFrame(rows)


def fixed_grid_tables(
    events_by_name: dict[str, pd.DataFrame],
    common_by_hash: pd.Series,
    coordinates: pd.DataFrame,
    seed_coordinates: Path,
    *,
    bins: int = 150,
) -> tuple[pd.DataFrame, dict[str, Any], dict[tuple[str, str], np.ndarray]]:
    with np.load(seed_coordinates, allow_pickle=False) as data:
        seed_xy = data["umap"].astype(np.float64)
    if seed_xy.ndim != 2 or seed_xy.shape[1] != 2 or not np.isfinite(seed_xy).all():
        raise ValueError("fixed seed coordinates are invalid")
    extent = (
        float(seed_xy[:, 0].min()),
        float(seed_xy[:, 0].max()),
        float(seed_xy[:, 1].min()),
        float(seed_xy[:, 1].max()),
    )
    coordinate_map = coordinates.set_index("common_id")[["umap_x", "umap_y"]]
    seed_hist, x_edges, y_edges = np.histogram2d(
        seed_xy[:, 0], seed_xy[:, 1], bins=bins, range=(extent[:2], extent[2:])
    )
    seed_occupied = seed_hist > 0
    histograms: dict[tuple[str, str], np.ndarray] = {}
    rows = []
    for name, events in events_by_name.items():
        common_ids = _event_common_ids(events, common_by_hash)
        for cohort, selected_ids in (
            ("selected", common_ids),
            ("hit", common_ids[events["hit"].to_numpy(bool)]),
            ("strict_hit", common_ids[events["strict_hit"].to_numpy(bool)]),
        ):
            xy = coordinate_map.loc[selected_ids].to_numpy(float)
            histogram, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(x_edges, y_edges))
            histogram /= len(xy)
            histograms[(name, cohort)] = histogram
            occupied = histogram > 0
            rows.append(
                {
                    "workflow": name,
                    "cohort": cohort,
                    "n": len(xy),
                    "occupied_grid_cells": int(occupied.sum()),
                    "fraction_of_seed_occupied_cells": float(
                        np.count_nonzero(occupied & seed_occupied) / np.count_nonzero(seed_occupied)
                    ),
                    "fraction_outside_seed_occupied_cells": float(
                        np.count_nonzero(occupied & ~seed_occupied) / np.count_nonzero(occupied)
                    ),
                }
            )
    left_name, right_name = tuple(events_by_name)
    for cohort in ("selected", "hit", "strict_hit"):
        distance = float(
            jensenshannon(
                histograms[(left_name, cohort)].ravel(),
                histograms[(right_name, cohort)].ravel(),
                base=2,
            )
        )
        for row in rows:
            if row["cohort"] == cohort:
                row["cross_workflow_jensen_shannon_distance"] = distance
    definition = {
        "status": "complete",
        "extent": list(extent),
        "bins": bins,
        "normalization": "within-cohort fraction per fixed grid cell",
        "distance": "Jensen-Shannon distance base 2",
        "seed_rows": len(seed_xy),
        "seed_occupied_grid_cells": int(seed_occupied.sum()),
    }
    return pd.DataFrame(rows), definition, histograms


def operational_table(
    events_by_name: dict[str, pd.DataFrame], timing_paths: dict[str, Path]
) -> pd.DataFrame:
    rows = []
    for name, events in events_by_name.items():
        timing = pd.read_csv(timing_paths[name])
        wall_seconds = float(timing["end_to_end_wall_seconds"].sum())
        rows.append(
            {
                "workflow": name,
                "rounds": len(timing),
                "selected": len(events),
                "scored": int(events["scored"].sum()),
                "hits": int(events["hit"].sum()),
                "wall_seconds": wall_seconds,
                "wall_hours": wall_seconds / 3600,
                "selected_per_wall_hour": len(events) / (wall_seconds / 3600),
                "scored_per_wall_hour": events["scored"].sum() / (wall_seconds / 3600),
                "hits_per_wall_hour": events["hit"].sum() / (wall_seconds / 3600),
            }
        )
    return pd.DataFrame(rows)


def _sample_metrics(
    common_ids: np.ndarray,
    union: pd.DataFrame,
    words: np.ndarray,
    popcounts: np.ndarray,
    atlas_by_common_id: np.ndarray,
    *,
    seed: int,
    pair_samples: int,
) -> dict[str, Any]:
    rows = common_ids - 1
    internal, mc_se, completed = sampled_diversity(
        rows,
        words,
        popcounts.astype(np.int64),
        seed=seed,
        samples=pair_samples,
    )
    metrics: dict[str, Any] = {
        "internal_diversity": internal,
        "internal_diversity_mc_se": mc_se,
        "pair_samples": completed,
    }
    for prefix, values in (
        ("typed", union.iloc[rows]["typed_scaffold"].astype(str).to_numpy()),
        ("generic", union.iloc[rows]["generic_framework"].astype(str).to_numpy()),
        ("atlas", atlas_by_common_id[rows]),
    ):
        metrics.update(
            {f"{prefix}_{key}": value for key, value in family_distribution(values).items()}
        )
    return metrics


def count_matched_tables(
    events_by_name: dict[str, pd.DataFrame],
    union: pd.DataFrame,
    words: np.ndarray,
    popcounts: np.ndarray,
    atlas_by_common_id: np.ndarray,
    natural: pd.DataFrame,
    *,
    replicates: int,
    pair_samples: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, Any]]:
    common_by_hash = pd.Series(union["common_id"].to_numpy(), index=union["reghash"])
    hit_ids = {
        name: _event_common_ids(events[events["hit"]], common_by_hash)
        for name, events in events_by_name.items()
    }
    fixed_name = min(hit_ids, key=lambda name: len(hit_ids[name]))
    sampled_name = max(hit_ids, key=lambda name: len(hit_ids[name]))
    if fixed_name == sampled_name:
        raise ValueError("count matching requires different natural hit counts")
    sample_size = len(hit_ids[fixed_name])
    sampled_ids = np.empty((replicates, sample_size), dtype=np.int64)
    rows = []
    for replicate in tqdm(
        range(replicates),
        desc="comparison hit-count matching",
        unit="replicate",
        dynamic_ncols=True,
    ):
        random = np.random.default_rng(random_seed + replicate)
        selected = np.sort(random.choice(hit_ids[sampled_name], size=sample_size, replace=False))
        sampled_ids[replicate] = selected
        rows.append(
            {
                "replicate": replicate,
                "sampled_workflow": sampled_name,
                "fixed_workflow": fixed_name,
                "sample_size": sample_size,
                **_sample_metrics(
                    selected,
                    union,
                    words,
                    popcounts,
                    atlas_by_common_id,
                    seed=random_seed + 10_000 + replicate,
                    pair_samples=pair_samples,
                ),
            }
        )
    replicate_frame = pd.DataFrame(rows).sort_values("replicate")
    fixed = natural[(natural["workflow"] == fixed_name) & (natural["cohort"] == "hit")].iloc[0]
    metrics = (
        "internal_diversity",
        "typed_q0",
        "typed_q1",
        "typed_q2",
        "typed_hhi",
        "generic_q0",
        "generic_q1",
        "generic_q2",
        "generic_hhi",
        "atlas_q0",
        "atlas_q1",
        "atlas_q2",
        "atlas_hhi",
    )
    summary_rows = []
    for metric in metrics:
        values = replicate_frame[metric].dropna()
        summary_rows.append(
            {
                "metric": metric,
                "fixed_workflow": fixed_name,
                "fixed_complete_value": fixed[metric],
                "sampled_workflow": sampled_name,
                "sample_size": sample_size,
                "sampled_median": float(values.median()),
                "sampled_interval_low": float(values.quantile(0.025)),
                "sampled_interval_high": float(values.quantile(0.975)),
                "replicates": len(values),
            }
        )
    design = {
        "fixed_workflow": fixed_name,
        "fixed_complete_hits": sample_size,
        "sampled_workflow": sampled_name,
        "sampled_natural_hits": len(hit_ids[sampled_name]),
        "sample_size": sample_size,
        "replicates": replicates,
        "pair_samples_per_replicate": pair_samples,
        "random_seed": random_seed,
    }
    return replicate_frame, pd.DataFrame(summary_rows), sampled_ids, design


def _save_figure(figure: Any, root: Path, name: str, dpi: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    figure.savefig(root / f"{name}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(root / f"{name}.pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)


def comparison_figures(
    *,
    output: Path,
    events_by_name: dict[str, pd.DataFrame],
    labels: dict[str, str],
    budget: pd.DataFrame,
    cutoffs: pd.DataFrame,
    natural: pd.DataFrame,
    matched_summary: pd.DataFrame,
    regional: pd.DataFrame,
    depth: pd.DataFrame,
    nearest_productivity: pd.DataFrame,
    overlap: pd.DataFrame,
    score_pairs: pd.DataFrame,
    exclusive: pd.DataFrame,
    grid_histograms: dict[tuple[str, str], np.ndarray],
    fixed_definition: dict[str, Any],
    dpi: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    plt.style.use("tableau-colorblind10")
    names = tuple(events_by_name)
    colors = {names[0]: "#0072B2", names[1]: "#D55E00"}
    generated: list[Path] = []

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for name in names:
        current = budget[budget["workflow"] == name]
        axes[0].plot(
            current["cumulative_budget"],
            current["cumulative_hits"],
            color=colors[name],
            label=labels[name],
        )
        for exact, marker in ((True, "o"), (False, "o")):
            points = current[current["adaptive_boundary"] == exact]
            axes[0].scatter(
                points["cumulative_budget"],
                points["cumulative_hits"],
                marker=marker,
                facecolors=colors[name] if exact else "white",
                edgecolors=colors[name],
                zorder=3,
            )
        axes[1].plot(
            current["block_50k"],
            current["block_hit_rate_selected"],
            marker="o",
            color=colors[name],
            label=labels[name],
        )
    axes[0].set(xlabel="Selected budget", ylabel="Cumulative hits", title="Cumulative discovery")
    axes[1].set(
        xlabel="50,000-compound budget block",
        ylabel="Selected-denominator hit rate",
        title="Block yield",
        xticks=sorted(budget["block_50k"].unique()),
    )
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save_figure(figure, output, "01_budget_yield", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for name in names:
        scores = np.sort(events_by_name[name].loc[events_by_name[name]["scored"], "dock_score"])
        axes[0].plot(
            scores,
            np.arange(1, len(scores) + 1) / len(scores),
            color=colors[name],
            label=labels[name],
        )
        current = cutoffs[cutoffs["workflow"] == name]
        axes[1].plot(
            current["cutoff"],
            current["hit_rate_selected"],
            marker="o",
            color=colors[name],
            label=labels[name],
        )
    axes[0].set(
        xlabel="Docking score (lower is better)",
        ylabel="Empirical cumulative fraction",
        title="Score distributions",
    )
    axes[1].set(
        xlabel="Docking-score cutoff",
        ylabel="Selected-denominator hit rate",
        title="Cutoff sensitivity",
    )
    for axis in axes:
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save_figure(figure, output, "02_score_quality", dpi)

    fixed_name = str(matched_summary["fixed_workflow"].iloc[0])
    sampled_name = str(matched_summary["sampled_workflow"].iloc[0])
    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.8))
    concise = {name: name.title() for name in names}
    for axis, metric, title in zip(
        axes,
        ("internal_diversity", "generic_q0", "atlas_q0"),
        ("Internal diversity", "Generic-framework q0", "Common-atlas q0"),
        strict=True,
    ):
        natural_values = natural[(natural["cohort"] == "hit")].set_index("workflow")
        match = matched_summary.set_index("metric").loc[metric]
        values = [
            natural_values.loc[fixed_name, metric],
            natural_values.loc[sampled_name, metric],
            match["sampled_median"],
        ]
        axis.bar([0, 1, 2], values, color=[colors[fixed_name], colors[sampled_name], "#E69F00"])
        axis.errorbar(
            2,
            match["sampled_median"],
            yerr=[
                [match["sampled_median"] - match["sampled_interval_low"]],
                [match["sampled_interval_high"] - match["sampled_median"]],
            ],
            fmt="none",
            color="black",
            capsize=3,
        )
        axis.set(
            xticks=[0, 1, 2],
            xticklabels=[
                f"{concise[fixed_name]}\nfixed",
                f"{concise[sampled_name]}\nnatural",
                f"{concise[sampled_name]}\nmatched",
            ],
            ylabel=title,
        )
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save_figure(figure, output, "03_natural_count_matched_diversity", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    x = np.arange(3)
    width = 0.36
    for index, name in enumerate(names):
        row = natural[(natural["workflow"] == name) & (natural["cohort"] == "hit")].iloc[0]
        axes[0].bar(
            x + (index - 0.5) * width,
            [row["atlas_q0"], row["atlas_q1"], row["atlas_q2"]],
            width,
            color=colors[name],
            label=labels[name],
        )
        current = depth[depth["workflow"] == name]
        axes[1].plot(
            current["threshold"], current["regions"], color=colors[name], label=labels[name]
        )
    axes[0].set(
        xticks=x,
        xticklabels=["q0", "q1", "q2"],
        ylabel="Effective common-atlas regions",
        title="Natural hit diversity",
    )
    axes[1].set(xlabel="At least k hits", ylabel="Common-atlas regions", title="Productive depth")
    for axis in axes:
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save_figure(figure, output, "04_common_atlas", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    for name in names:
        for cohort, linestyle in (("selected", "-"), ("hit", "--")):
            current = (
                events_by_name[name]
                if cohort == "selected"
                else events_by_name[name][events_by_name[name]["hit"]]
            )
            values = np.sort(current["nearest_seed_tanimoto"].to_numpy())
            axes[0].plot(
                values,
                np.arange(1, len(values) + 1) / len(values),
                color=colors[name],
                linestyle=linestyle,
                label=f"{labels[name]} {cohort}",
            )
        current_bins = nearest_productivity[nearest_productivity["workflow"] == name]
        axes[1].plot(
            np.arange(len(current_bins)),
            current_bins["hit_rate_selected"],
            marker="o",
            color=colors[name],
            label=labels[name],
        )
    axes[0].set(
        xlabel="Exact nearest-seed Tanimoto",
        ylabel="Empirical cumulative fraction",
        title="Seed distance",
    )
    axes[1].set(
        xlabel="Nearest-seed Tanimoto bin",
        ylabel="Selected-denominator hit rate",
        title="Productivity by seed distance",
        xticks=np.arange(5),
        xticklabels=["<0.3", ".3-.4", ".4-.5", ".5-.7", ">=.7"],
    )
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save_figure(figure, output, "05_seed_distance", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    current_overlap = overlap[
        overlap["cohort"].isin(["selected_compounds", "hits", "strict_hits", "seed_distant_hits"])
    ]
    x = np.arange(len(current_overlap))
    axes[0].bar(x, current_overlap["jaccard"], color="#56B4E9")
    overlap_labels = [
        str(value).replace("selected_compounds", "selected").replace("_", " ")
        for value in current_overlap["cohort"]
    ]
    axes[0].set(
        xticks=x,
        xticklabels=overlap_labels,
        ylabel="Jaccard overlap",
        title="Set overlap",
    )
    width = 0.36
    axes[1].bar(
        x - width / 2,
        current_overlap[f"{names[0]}_directional_capture"],
        width,
        color=colors[names[0]],
        label=labels[names[0]],
    )
    axes[1].bar(
        x + width / 2,
        current_overlap[f"{names[1]}_directional_capture"],
        width,
        color=colors[names[1]],
        label=labels[names[1]],
    )
    axes[1].set(
        xticks=x,
        xticklabels=overlap_labels,
        ylabel="Directional capture",
        title="Intersection / source set",
    )
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save_figure(figure, output, "06_overlap", dpi)

    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    if len(score_pairs):
        axes[0].hexbin(
            score_pairs[f"{names[0]}_score"],
            score_pairs[f"{names[1]}_score"],
            gridsize=70,
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        limits = [
            min(score_pairs[f"{names[0]}_score"].min(), score_pairs[f"{names[1]}_score"].min()),
            max(score_pairs[f"{names[0]}_score"].max(), score_pairs[f"{names[1]}_score"].max()),
        ]
        axes[0].plot(limits, limits, linestyle="--", color="black", linewidth=1)
    axes[0].set(
        xlabel=f"{labels[names[0]]} docking score",
        ylabel=f"{labels[names[1]]} docking score",
        title="Shared selected compounds",
    )
    axes[1].bar(
        np.arange(len(exclusive)),
        exclusive["exclusive_hits"],
        color=[colors[name] for name in exclusive["workflow"]],
    )
    axes[1].set(
        xticks=np.arange(len(exclusive)),
        xticklabels=[labels[name] for name in exclusive["workflow"]],
        ylabel="Workflow-exclusive hits",
        title="Exclusive hit counts",
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save_figure(figure, output, "07_shared_scores_exclusive_hits", dpi)

    figure, axes = plt.subplots(
        2, 2, figsize=(9.2, 7.0), sharex=True, sharey=True, constrained_layout=True
    )
    positive = np.concatenate([hist[hist > 0] for hist in grid_histograms.values()])
    norm = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
    image = None
    extent = fixed_definition["extent"]
    for row_index, cohort in enumerate(("selected", "hit")):
        for column_index, name in enumerate(names):
            image = axes[row_index, column_index].imshow(
                grid_histograms[(name, cohort)].T,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="viridis",
                norm=norm,
                rasterized=True,
            )
            n = (
                len(events_by_name[name])
                if cohort == "selected"
                else int(events_by_name[name]["hit"].sum())
            )
            axes[row_index, column_index].set(
                xlabel="Fixed-reference UMAP 1",
                ylabel="Fixed-reference UMAP 2",
                title=f"{labels[name]} {cohort} (n={n:,})",
            )
    assert image is not None
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Within-cohort fraction per fixed grid cell",
        shrink=0.88,
    )
    _save_figure(figure, output, "08_common_fixed_umap", dpi)

    for stem in (
        "01_budget_yield",
        "02_score_quality",
        "03_natural_count_matched_diversity",
        "04_common_atlas",
        "05_seed_distance",
        "06_overlap",
        "07_shared_scores_exclusive_hits",
        "08_common_fixed_umap",
    ):
        generated.extend([output / f"{stem}.png", output / f"{stem}.pdf"])
    return generated


def analyze_comparison(
    *,
    root: Path,
    common_coordinates: Path,
    common_atlas_assignments_path: Path,
    seed_coordinates: Path,
    umap_model: Path,
    timing_paths: dict[str, Path],
    replicates: int,
    pair_samples: int,
    random_seed: int,
    dpi: int,
) -> dict[str, Any]:
    compatibility_path = root / "comparison_compatibility.json"
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    if compatibility.get("status") != "compatible":
        raise ValueError("comparison compatibility gate did not pass")
    names = (str(compatibility["left_workflow"]), str(compatibility["right_workflow"]))
    labels = {str(key): str(value) for key, value in compatibility["labels"].items()}
    cache = root / "cache"
    events_by_name = {name: pd.read_csv(cache / f"{name}_events.csv.gz") for name in names}
    for name, events in events_by_name.items():
        if semantic_digest(events, EVENT_FIELDS) != compatibility["event_semantic_sha256"][name]:
            raise ValueError(f"normalized event semantic digest changed: {name}")
    union = pd.read_csv(cache / "common_union_manifest.csv.gz")
    with np.load(cache / "common_union_fingerprints.npz", allow_pickle=False) as data:
        identifiers = data["spacehastenid"].astype(np.int64)
        words = data["words"].astype(np.uint64)
        popcounts = data["popcounts"].astype(np.uint16)
    if not np.array_equal(identifiers, union["common_id"].to_numpy(np.int64)):
        raise ValueError("common union and fingerprint IDs differ")
    atlas = common_atlas_assignments(
        union,
        common_atlas_assignments_path,
        threshold=0.4,
    )
    common_atlas_root = cache / "common_atlas"
    write_csv(common_atlas_root / "common_union_assignments.csv.gz", atlas)
    write_npz(
        common_atlas_root / "common_union_assignments.npz",
        spacehastenid=atlas["common_id"].to_numpy(np.int64),
        clusterid=atlas["clusterid"].to_numpy(np.int64),
        centroid_similarity=atlas["centroid_similarity"].to_numpy(np.float32),
    )
    atlas_by_common_id = atlas.sort_values("common_id")["clusterid"].to_numpy(np.int64)
    common_by_hash = pd.Series(union["common_id"].to_numpy(), index=union["reghash"])

    with np.load(common_coordinates, allow_pickle=False) as data:
        coordinate_ids = data["spacehastenid"].astype(np.int64)
        xy = data["umap"].astype(np.float64)
    if not np.array_equal(coordinate_ids, identifiers) or xy.shape != (len(union), 2):
        raise ValueError("common fixed-reference coordinates differ from the selected union")
    if not np.isfinite(xy).all():
        raise ValueError("common fixed-reference coordinates are non-finite")
    coordinates = pd.DataFrame(
        {"common_id": coordinate_ids, "umap_x": xy[:, 0], "umap_y": xy[:, 1]}
    )

    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    endpoints = endpoint_table(events_by_name)
    budget = budget_table(events_by_name)
    cutoffs = cutoff_table(events_by_name)
    top_k = top_k_table(events_by_name)
    natural = pd.DataFrame(
        [
            natural_diversity_row(
                events,
                union,
                words,
                popcounts,
                atlas_by_common_id,
                workflow=name,
                cohort=cohort,
                seed=random_seed + workflow_index * 100 + cohort_index,
                pair_samples=1_000_000,
            )
            for workflow_index, (name, events) in enumerate(events_by_name.items())
            for cohort_index, cohort in enumerate(("selected", "hit", "strict_hit"))
        ]
    )
    regional, depth, region_contrasts = regional_tables(
        events_by_name, common_by_hash, atlas_by_common_id
    )
    shared_region_contrasts = region_contrasts[region_contrasts["selected_by_both"]]
    scored_by_both = shared_region_contrasts[
        shared_region_contrasts[f"{names[0]}_scored"].gt(0)
        & shared_region_contrasts[f"{names[1]}_scored"].gt(0)
    ]
    region_contrast_summary = pd.DataFrame(
        [
            {
                "selected_by_both_regions": len(shared_region_contrasts),
                "scored_by_both_regions": len(scored_by_both),
                "portfolio_minus_greedy_posterior_pp_median": float(
                    scored_by_both["posterior_difference_pp"].median()
                ),
                "portfolio_minus_greedy_posterior_pp_q05": float(
                    scored_by_both["posterior_difference_pp"].quantile(0.05)
                ),
                "portfolio_minus_greedy_posterior_pp_q95": float(
                    scored_by_both["posterior_difference_pp"].quantile(0.95)
                ),
                "regions_portfolio_posterior_higher": int(
                    scored_by_both["posterior_difference_pp"].gt(0).sum()
                ),
                "regions_greedy_posterior_higher": int(
                    scored_by_both["posterior_difference_pp"].lt(0).sum()
                ),
            }
        ]
    )
    nearest_summary, nearest_productivity = nearest_seed_tables(events_by_name)
    overlap = overlap_table(events_by_name, common_by_hash, atlas_by_common_id)
    shared_summary, shared_pairs = shared_score_tables(events_by_name)
    exclusive = exclusive_hit_table(events_by_name, union)
    fixed_grid, fixed_definition, histograms = fixed_grid_tables(
        events_by_name,
        common_by_hash,
        coordinates,
        seed_coordinates,
    )
    fixed_definition.update(
        {
            "model": str(umap_model.resolve()),
            "model_sha256": sha256(umap_model),
            "coordinates": str(common_coordinates.resolve()),
            "coordinates_sha256": sha256(common_coordinates),
        }
    )
    write_json(root / "fixed_umap_definition.json", fixed_definition)
    operations = operational_table(events_by_name, timing_paths)
    matched_replicates, matched_summary, matched_ids, matched_design = count_matched_tables(
        events_by_name,
        union,
        words,
        popcounts,
        atlas_by_common_id,
        natural,
        replicates=replicates,
        pair_samples=pair_samples,
        random_seed=random_seed,
    )
    table_frames = {
        "endpoint_summary.csv": endpoints,
        "budget_yield.csv": budget,
        "cutoff_sensitivity.csv": cutoffs,
        "top_k_score_quality.csv": top_k,
        "natural_diversity.csv": natural,
        "common_atlas_coverage.csv": regional,
        "common_atlas_depth.csv": depth,
        "common_atlas_region_contrasts.csv": region_contrasts,
        "common_atlas_region_contrast_summary.csv": region_contrast_summary,
        "nearest_seed_summary.csv": nearest_summary,
        "nearest_seed_productivity.csv": nearest_productivity,
        "overlap_summary.csv": overlap,
        "shared_score_summary.csv": shared_summary,
        "workflow_exclusive_hit_chemistry.csv": exclusive,
        "fixed_grid_umap_summary.csv": fixed_grid,
        "operational_efficiency.csv": operations,
        "count_matched_replicates.csv": matched_replicates,
        "count_matched_summary.csv": matched_summary,
    }
    for filename, frame in table_frames.items():
        write_csv(tables / filename, frame)
    write_csv(tables / "shared_score_pairs.csv.gz", shared_pairs)
    write_npz(
        tables / "count_matched_sample_ids.npz",
        replicate=np.arange(replicates, dtype=np.int32),
        common_id=matched_ids,
    )
    write_json(tables / "count_matched_design.json", matched_design)
    figure_paths = comparison_figures(
        output=figures,
        events_by_name=events_by_name,
        labels=labels,
        budget=budget,
        cutoffs=cutoffs,
        natural=natural,
        matched_summary=matched_summary,
        regional=regional,
        depth=depth,
        nearest_productivity=nearest_productivity,
        overlap=overlap,
        score_pairs=shared_pairs,
        exclusive=exclusive,
        grid_histograms=histograms,
        fixed_definition=fixed_definition,
        dpi=dpi,
    )
    outputs = [
        *(tables / name for name in table_frames),
        tables / "shared_score_pairs.csv.gz",
        tables / "count_matched_sample_ids.npz",
        tables / "count_matched_design.json",
        root / "fixed_umap_definition.json",
        common_atlas_root / "common_union_assignments.csv.gz",
        common_atlas_root / "common_union_assignments.npz",
        *figure_paths,
    ]
    receipt = {
        "status": "complete",
        "workflows": list(names),
        "labels": labels,
        "hit_cutoff": float(compatibility["hit_cutoff"]),
        "strict_cutoff": float(compatibility["strict_cutoff"]),
        "union_rows": len(union),
        "common_atlas_threshold": 0.4,
        "common_atlas_minimum_similarity": float(atlas["centroid_similarity"].min()),
        "common_atlas_assignments": len(atlas),
        "count_matching": matched_design,
        "inputs": [
            file_record(compatibility_path),
            file_record(cache / "common_union_manifest.csv.gz"),
            file_record(cache / "common_union_fingerprints.npz"),
            file_record(common_coordinates),
            file_record(common_atlas_assignments_path),
            file_record(common_atlas_assignments_path.parent / "_SUCCESS.json"),
            file_record(seed_coordinates),
            file_record(umap_model),
            *(file_record(path) for path in timing_paths.values()),
        ],
        "outputs": [file_record(path, root=root) for path in outputs],
    }
    write_json(root / "comparison_analysis_receipt.json", receipt)
    return receipt


__all__ = [
    "DESCRIPTOR_FIELDS",
    "EVENT_FIELDS",
    "build_fpsim2_index",
    "build_union",
    "analyze_comparison",
    "common_atlas_assignments",
    "database_protocol",
    "file_record",
    "load_fingerprints",
    "natural_diversity_row",
    "normalize_events",
    "prepare_comparison",
    "refresh_comparison_semantics",
    "semantic_digest",
    "sha256",
    "write_csv",
    "write_json",
    "write_npz",
]
