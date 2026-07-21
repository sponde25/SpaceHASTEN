#!/usr/bin/env python3
"""Resumable map/reduce components for a persistent sphere-exclusion atlas."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

try:
    from spacehasten.remote.cluster import _build_fpsim2_index, _generate_fingerprints
except ImportError:  # pragma: no cover - direct compute-node script execution
    from cluster import _build_fpsim2_index, _generate_fingerprints  # type: ignore[no-redef]

LOGGER = logging.getLogger("spacehasten.remote.atlas")
FP_TYPE = "Morgan"
FP_PARAMS = {"radius": 2, "fpSize": 1024}
COMPLETE_FILE = "complete.json"


def _open_smi(path: Path):  # type: ignore[no-untyped-def]
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, "rt", encoding="utf-8")


def read_smi(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with _open_smi(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                smiles, identifier = stripped.rsplit(None, 1)
                rows.append((smiles, int(identifier)))
            except ValueError as exc:
                raise ValueError(f"malformed SMI row {path}:{line_number}") from exc
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _complete(output_dir: Path, expected: dict[str, Any], required: tuple[str, ...]) -> bool:
    marker = output_dir / COMPLETE_FILE
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("inputs") != expected:
        return False
    outputs = payload.get("outputs", {})
    for name in required:
        path = output_dir / name
        details = outputs.get(name)
        if not path.is_file() or not isinstance(details, dict):
            return False
        if details.get("sha256") != _sha256(path):
            return False
    return True


def _new_temp_dir(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))


def _commit_dir(temp_dir: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace incomplete output: {output_dir}")
    temp_dir.replace(output_dir)


def _finish(
    temp_dir: Path,
    *,
    inputs: dict[str, Any],
    outputs: tuple[str, ...],
    metrics: dict[str, Any],
) -> None:
    payload = {
        "inputs": inputs,
        "outputs": {
            name: {
                "size": (temp_dir / name).stat().st_size,
                "sha256": _sha256(temp_dir / name),
            }
            for name in outputs
        },
        "metrics": metrics,
    }
    _atomic_json(temp_dir / COMPLETE_FILE, payload)


def partition_smiles(input_path: Path, output_dir: Path, partition_count: int) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if partition_count < 1:
        raise ValueError("partition_count must be positive")
    inputs = {
        "input": {**_identity(input_path), "sha256": _sha256(input_path)},
        "partition_count": partition_count,
        "partition_rule": "spacehastenid_modulo",
    }
    names = tuple(
        f"part_{index:04d}_of_{partition_count:04d}.smi.gz" for index in range(partition_count)
    )
    if _complete(output_dir, inputs, names):
        LOGGER.info("Reusing completed partition output: %s", output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())

    temp_dir = _new_temp_dir(output_dir)
    counts = [0] * partition_count
    started = time.monotonic()
    try:
        with ExitStack() as stack:
            handles = [
                stack.enter_context(gzip.open(temp_dir / name, "wt", encoding="utf-8"))
                for name in names
            ]
            with _open_smi(input_path) as source:
                for row_count, line in enumerate(source, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        identifier = int(stripped.rsplit(None, 1)[1])
                    except ValueError as exc:
                        raise ValueError(f"malformed SMI row {input_path}:{row_count}") from exc
                    partition = identifier % partition_count
                    handles[partition].write(stripped + "\n")
                    counts[partition] += 1
                    if row_count % 500_000 == 0:
                        elapsed = time.monotonic() - started
                        LOGGER.info(
                            "Partitioned %d molecules (%.0f mol/s)",
                            row_count,
                            row_count / max(elapsed, 1e-9),
                        )
        metrics = {
            "molecule_count": sum(counts),
            "partition_counts": counts,
            "elapsed_seconds": time.monotonic() - started,
        }
        _finish(temp_dir, inputs=inputs, outputs=names, metrics=metrics)
        _commit_dir(temp_dir, output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _leader_centroids(
    rows: list[tuple[str, int]], similarity_threshold: float, processes: int
) -> tuple[list[tuple[str, int]], int]:
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0, 1]")
    if not rows:
        return [], 0
    from rdkit.SimDivFilters import rdSimDivPickers

    rows = sorted(rows, key=lambda row: row[1])
    fingerprints = _generate_fingerprints([smiles for smiles, _ in rows], processes)
    picker = rdSimDivPickers.LeaderPicker()
    indices = picker.LazyBitVectorPick(
        fingerprints,
        len(fingerprints),
        1.0 - similarity_threshold,
    )
    return [rows[int(index)] for index in indices], len(fingerprints)


def _write_smi(path: Path, rows: list[tuple[str, int]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for smiles, identifier in rows:
            handle.write(f"{smiles} {identifier}\n")


def map_partition(
    input_path: Path,
    output_dir: Path,
    similarity_threshold: float,
    processes: int,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    inputs = {
        "input": {**_identity(input_path), "sha256": _sha256(input_path)},
        "similarity_threshold": similarity_threshold,
        "fingerprint_type": FP_TYPE,
        "fingerprint_parameters": FP_PARAMS,
    }
    outputs = ("centroids.smi.gz",)
    if _complete(output_dir, inputs, outputs):
        LOGGER.info("Reusing completed mapper output: %s", output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    temp_dir = _new_temp_dir(output_dir)
    started = time.monotonic()
    try:
        rows = read_smi(input_path)
        centroids, fingerprint_count = _leader_centroids(rows, similarity_threshold, processes)
        _write_smi(temp_dir / outputs[0], centroids)
        metrics = {
            "molecule_count": len(rows),
            "fingerprint_count": fingerprint_count,
            "centroid_count": len(centroids),
            "elapsed_seconds": time.monotonic() - started,
        }
        _finish(temp_dir, inputs=inputs, outputs=outputs, metrics=metrics)
        _commit_dir(temp_dir, output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def pool_centroids(map_root: Path, output_dir: Path) -> dict[str, Any]:
    map_root = map_root.resolve()
    output_dir = output_dir.resolve()
    centroid_files = sorted(map_root.glob("*/centroids.smi.gz"))
    if not centroid_files:
        raise FileNotFoundError(f"no mapper centroids under {map_root}")
    inputs = {
        "centroids": [{**_identity(path), "sha256": _sha256(path)} for path in centroid_files]
    }
    outputs = ("pooled_centroids.smi.gz",)
    if _complete(output_dir, inputs, outputs):
        LOGGER.info("Reusing completed centroid pool: %s", output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    temp_dir = _new_temp_dir(output_dir)
    started = time.monotonic()
    try:
        pooled: dict[int, str] = {}
        for path in centroid_files:
            marker = path.parent / COMPLETE_FILE
            if not marker.is_file():
                raise ValueError(f"mapper output has no completion marker: {path.parent}")
            for smiles, identifier in read_smi(path):
                if identifier in pooled and pooled[identifier] != smiles:
                    raise ValueError(f"conflicting centroid SMILES for {identifier}")
                pooled[identifier] = smiles
        rows = [(smiles, identifier) for identifier, smiles in sorted(pooled.items())]
        _write_smi(temp_dir / outputs[0], rows)
        metrics = {
            "mapper_count": len(centroid_files),
            "pooled_centroid_count": len(rows),
            "elapsed_seconds": time.monotonic() - started,
        }
        _finish(temp_dir, inputs=inputs, outputs=outputs, metrics=metrics)
        _commit_dir(temp_dir, output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def reduce_centroid_group(
    map_root: Path,
    output_dir: Path,
    group_index: int,
    group_count: int,
    similarity_threshold: float,
    processes: int,
) -> dict[str, Any]:
    map_root = map_root.resolve()
    output_dir = output_dir.resolve()
    all_centroid_files = sorted(map_root.glob("*/centroids.smi.gz"))
    if not 0 <= group_index < group_count:
        raise ValueError("group_index must be in [0, group_count)")
    centroid_files = [
        path for index, path in enumerate(all_centroid_files) if index % group_count == group_index
    ]
    if not centroid_files:
        raise ValueError(f"intermediate reducer {group_index} has no mapper inputs")
    inputs = {
        "centroids": [{**_identity(path), "sha256": _sha256(path)} for path in centroid_files],
        "group_index": group_index,
        "group_count": group_count,
        "similarity_threshold": similarity_threshold,
        "fingerprint_type": FP_TYPE,
        "fingerprint_parameters": FP_PARAMS,
    }
    outputs = ("centroids.smi.gz",)
    if _complete(output_dir, inputs, outputs):
        LOGGER.info("Reusing intermediate centroid reducer: %s", output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    temp_dir = _new_temp_dir(output_dir)
    started = time.monotonic()
    try:
        pooled: dict[int, str] = {}
        for path in centroid_files:
            if not (path.parent / COMPLETE_FILE).is_file():
                raise ValueError(f"mapper output has no completion marker: {path.parent}")
            for smiles, identifier in read_smi(path):
                if identifier in pooled and pooled[identifier] != smiles:
                    raise ValueError(f"conflicting centroid SMILES for {identifier}")
                pooled[identifier] = smiles
        rows = [(smiles, identifier) for identifier, smiles in sorted(pooled.items())]
        centroids, _ = _leader_centroids(rows, similarity_threshold, processes)
        _write_smi(temp_dir / outputs[0], centroids)
        metrics = {
            "mapper_count": len(centroid_files),
            "pooled_centroid_count": len(rows),
            "reduced_centroid_count": len(centroids),
            "elapsed_seconds": time.monotonic() - started,
        }
        _finish(temp_dir, inputs=inputs, outputs=outputs, metrics=metrics)
        _commit_dir(temp_dir, output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def reduce_centroids(
    pooled_path: Path,
    output_dir: Path,
    similarity_threshold: float,
    processes: int,
) -> dict[str, Any]:
    pooled_path = pooled_path.resolve()
    output_dir = output_dir.resolve()
    inputs = {
        "input": {**_identity(pooled_path), "sha256": _sha256(pooled_path)},
        "similarity_threshold": similarity_threshold,
        "fingerprint_type": FP_TYPE,
        "fingerprint_parameters": FP_PARAMS,
    }
    outputs = ("centroids.smi.gz", "centroids_fp.h5")
    if _complete(output_dir, inputs, outputs):
        LOGGER.info("Reusing completed centroid reduction: %s", output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    temp_dir = _new_temp_dir(output_dir)
    started = time.monotonic()
    try:
        rows = read_smi(pooled_path)
        centroids, _ = _leader_centroids(rows, similarity_threshold, processes)
        _write_smi(temp_dir / outputs[0], centroids)
        _build_fpsim2_index(centroids, temp_dir / outputs[1])
        metrics = {
            "pooled_centroid_count": len(rows),
            "final_centroid_count": len(centroids),
            "elapsed_seconds": time.monotonic() - started,
        }
        _finish(temp_dir, inputs=inputs, outputs=outputs, metrics=metrics)
        _commit_dir(temp_dir, output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def build_molecule_index(input_path: Path, output_dir: Path) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    inputs = {
        "input": {**_identity(input_path), "sha256": _sha256(input_path)},
        "fingerprint_type": FP_TYPE,
        "fingerprint_parameters": FP_PARAMS,
    }
    outputs = ("molecules_fp.h5",)
    if _complete(output_dir, inputs, outputs):
        LOGGER.info("Reusing completed molecule index: %s", output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    temp_dir = _new_temp_dir(output_dir)
    started = time.monotonic()
    try:
        rows = read_smi(input_path)
        _build_fpsim2_index(rows, temp_dir / outputs[0])
        metrics = {
            "molecule_count": len(rows),
            "elapsed_seconds": time.monotonic() - started,
        }
        _finish(temp_dir, inputs=inputs, outputs=outputs, metrics=metrics)
        _commit_dir(temp_dir, output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _validate_index(engine: Any) -> None:
    if engine.fp_type != FP_TYPE or engine.fp_params != FP_PARAMS:
        raise ValueError(
            f"index uses {engine.fp_type} {engine.fp_params}; expected {FP_TYPE} {FP_PARAMS}"
        )


def _dense_row_lookup(molecule_ids: np.ndarray) -> np.ndarray:
    if len(molecule_ids) == 0:
        return np.empty(0, dtype=np.int32)
    maximum = int(molecule_ids.max())
    if maximum > max(1_000_000, 4 * len(molecule_ids)):
        raise ValueError("molecule identifiers are too sparse for dense atlas assignment lookup")
    lookup = np.full(maximum + 1, -1, dtype=np.int32)
    lookup[molecule_ids] = np.arange(len(molecule_ids), dtype=np.int32)
    return lookup


def assign_centroid_shard(
    molecule_index: Path,
    centroids_path: Path,
    output_dir: Path,
    shard_index: int,
    shard_count: int,
    similarity_threshold: float,
) -> dict[str, Any]:
    from FPSim2 import FPSim2Engine

    molecule_index = molecule_index.resolve()
    centroids_path = centroids_path.resolve()
    output_dir = output_dir.resolve()
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    inputs = {
        "molecule_index": _identity(molecule_index),
        "centroids": {**_identity(centroids_path), "sha256": _sha256(centroids_path)},
        "shard_index": shard_index,
        "shard_count": shard_count,
        "similarity_threshold": similarity_threshold,
    }
    outputs = ("best_cluster.npy", "best_similarity.npy")
    if _complete(output_dir, inputs, outputs):
        LOGGER.info("Reusing completed assignment shard: %s", output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())

    temp_dir = _new_temp_dir(output_dir)
    started = time.monotonic()
    try:
        engine = FPSim2Engine(str(molecule_index))
        _validate_index(engine)
        molecule_ids = np.asarray(engine.fps[:, 0], dtype=np.int64)
        row_by_id = _dense_row_lookup(molecule_ids)
        centroids = read_smi(centroids_path)
        start = len(centroids) * shard_index // shard_count
        stop = len(centroids) * (shard_index + 1) // shard_count
        shard_centroids = centroids[start:stop]
        best_cluster = np.full(len(molecule_ids), -1, dtype=np.int64)
        best_similarity = np.zeros(len(molecule_ids), dtype=np.float32)

        for offset, (smiles, clusterid) in enumerate(shard_centroids, start=1):
            matches = engine.similarity(
                smiles,
                threshold=similarity_threshold,
                metric="tanimoto",
                n_workers=1,
            )
            match_ids = np.asarray(matches["mol_id"], dtype=np.int64)
            rows = row_by_id[match_ids]
            if np.any(rows < 0):
                raise RuntimeError("FPSim2 returned an unknown molecule identifier")
            similarities = np.asarray(matches["coeff"], dtype=np.float32)
            current = best_similarity[rows]
            current_clusters = best_cluster[rows]
            better = (similarities > current) | (
                (similarities == current)
                & ((current_clusters < 0) | (clusterid < current_clusters))
            )
            chosen = rows[better]
            best_similarity[chosen] = similarities[better]
            best_cluster[chosen] = clusterid
            if offset % 1_000 == 0:
                elapsed = time.monotonic() - started
                rate = offset / max(elapsed, 1e-9)
                remaining = len(shard_centroids) - offset
                LOGGER.info(
                    "Assignment shard %d/%d: %d/%d centroids (%.1f/s, ETA %.1fs)",
                    shard_index + 1,
                    shard_count,
                    offset,
                    len(shard_centroids),
                    rate,
                    remaining / max(rate, 1e-9),
                )

        np.save(temp_dir / outputs[0], best_cluster, allow_pickle=False)
        np.save(temp_dir / outputs[1], best_similarity, allow_pickle=False)
        metrics = {
            "molecule_count": len(molecule_ids),
            "centroid_start": start,
            "centroid_stop": stop,
            "centroid_count": len(shard_centroids),
            "assigned_count": int(np.count_nonzero(best_cluster >= 0)),
            "elapsed_seconds": time.monotonic() - started,
        }
        _finish(temp_dir, inputs=inputs, outputs=outputs, metrics=metrics)
        _commit_dir(temp_dir, output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _write_uncovered(
    input_smiles: Path,
    output_path: Path,
    uncovered_ids: np.ndarray,
) -> int:
    if len(uncovered_ids) == 0:
        with gzip.open(output_path, "wt", encoding="utf-8"):
            pass
        return 0
    maximum = int(uncovered_ids.max())
    selected = np.zeros(maximum + 1, dtype=np.bool_)
    selected[uncovered_ids] = True
    written = 0
    with (
        _open_smi(input_smiles) as source,
        gzip.open(output_path, "wt", encoding="utf-8") as output,
    ):
        for line in source:
            stripped = line.strip()
            if not stripped:
                continue
            identifier = int(stripped.rsplit(None, 1)[1])
            if identifier <= maximum and selected[identifier]:
                output.write(stripped + "\n")
                written += 1
    if written != len(uncovered_ids):
        raise ValueError(f"wrote {written} uncovered rows; expected {len(uncovered_ids)}")
    return written


def merge_assignment_shards(
    molecule_index: Path,
    input_smiles: Path,
    shard_root: Path,
    output_dir: Path,
    similarity_threshold: float,
) -> dict[str, Any]:
    from FPSim2 import FPSim2Engine

    molecule_index = molecule_index.resolve()
    input_smiles = input_smiles.resolve()
    shard_root = shard_root.resolve()
    output_dir = output_dir.resolve()
    shard_dirs = sorted(path.parent for path in shard_root.glob("*/best_cluster.npy"))
    if not shard_dirs:
        raise FileNotFoundError(f"no assignment shards under {shard_root}")
    inputs = {
        "molecule_index": _identity(molecule_index),
        "input_smiles": {**_identity(input_smiles), "sha256": _sha256(input_smiles)},
        "shards": [
            {
                "path": str(path),
                "marker_sha256": _sha256(path / COMPLETE_FILE),
            }
            for path in shard_dirs
        ],
        "similarity_threshold": similarity_threshold,
    }
    outputs = ("assignments.npz", "uncovered.smi.gz")
    if _complete(output_dir, inputs, outputs):
        LOGGER.info("Reusing completed assignment merge: %s", output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())

    temp_dir = _new_temp_dir(output_dir)
    started = time.monotonic()
    try:
        engine = FPSim2Engine(str(molecule_index))
        _validate_index(engine)
        molecule_ids = np.asarray(engine.fps[:, 0], dtype=np.int64)
        best_cluster = np.full(len(molecule_ids), -1, dtype=np.int64)
        best_similarity = np.zeros(len(molecule_ids), dtype=np.float32)
        for shard_dir in shard_dirs:
            if not (shard_dir / COMPLETE_FILE).is_file():
                raise ValueError(f"assignment shard is incomplete: {shard_dir}")
            clusters = np.load(shard_dir / "best_cluster.npy", mmap_mode="r")
            similarities = np.load(shard_dir / "best_similarity.npy", mmap_mode="r")
            if clusters.shape != best_cluster.shape or similarities.shape != best_similarity.shape:
                raise ValueError(f"assignment shard shape mismatch: {shard_dir}")
            better = (similarities > best_similarity) | (
                (similarities == best_similarity)
                & (clusters >= 0)
                & ((best_cluster < 0) | (clusters < best_cluster))
            )
            best_cluster[better] = clusters[better]
            best_similarity[better] = similarities[better]

        uncovered_rows = np.flatnonzero(best_cluster < 0)
        uncovered_ids = molecule_ids[uncovered_rows]
        order = np.argsort(molecule_ids)
        with (temp_dir / outputs[0]).open("wb") as handle:
            np.savez(
                handle,
                spacehastenid=molecule_ids[order],
                clusterid=best_cluster[order],
                centroid_similarity=best_similarity[order],
            )
        _write_uncovered(input_smiles, temp_dir / outputs[1], uncovered_ids)
        covered = best_cluster >= 0
        if np.any(best_similarity[covered] < similarity_threshold):
            raise RuntimeError("covered assignments fall below the threshold")
        metrics = {
            "molecule_count": len(molecule_ids),
            "shard_count": len(shard_dirs),
            "covered_count": int(np.count_nonzero(covered)),
            "uncovered_count": int(len(uncovered_ids)),
            "elapsed_seconds": time.monotonic() - started,
        }
        _finish(temp_dir, inputs=inputs, outputs=outputs, metrics=metrics)
        _commit_dir(temp_dir, output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def repair_uncovered(
    uncovered_path: Path,
    output_dir: Path,
    similarity_threshold: float,
    processes: int,
) -> dict[str, Any]:
    from FPSim2 import FPSim2Engine

    uncovered_path = uncovered_path.resolve()
    output_dir = output_dir.resolve()
    inputs = {
        "uncovered": {
            **_identity(uncovered_path),
            "sha256": _sha256(uncovered_path),
        },
        "similarity_threshold": similarity_threshold,
        "fingerprint_type": FP_TYPE,
        "fingerprint_parameters": FP_PARAMS,
    }
    outputs = ("repair_centroids.smi.gz", "repair_assignments.npz")
    if _complete(output_dir, inputs, outputs):
        LOGGER.info("Reusing completed coverage repair: %s", output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())

    temp_dir = _new_temp_dir(output_dir)
    started = time.monotonic()
    try:
        rows = read_smi(uncovered_path)
        if not rows:
            _write_smi(temp_dir / outputs[0], [])
            with (temp_dir / outputs[1]).open("wb") as handle:
                np.savez(
                    handle,
                    spacehastenid=np.empty(0, dtype=np.int64),
                    clusterid=np.empty(0, dtype=np.int64),
                    centroid_similarity=np.empty(0, dtype=np.float32),
                )
            metrics = {
                "molecule_count": 0,
                "repair_centroid_count": 0,
                "elapsed_seconds": time.monotonic() - started,
            }
        else:
            centroids, _ = _leader_centroids(rows, similarity_threshold, processes)
            _write_smi(temp_dir / outputs[0], centroids)
            index_path = temp_dir / "uncovered_fp.h5"
            _build_fpsim2_index(rows, index_path)
            engine = FPSim2Engine(str(index_path))
            molecule_ids = np.asarray(engine.fps[:, 0], dtype=np.int64)
            row_by_id = _dense_row_lookup(molecule_ids)
            best_cluster = np.full(len(molecule_ids), -1, dtype=np.int64)
            best_similarity = np.zeros(len(molecule_ids), dtype=np.float32)
            for smiles, clusterid in centroids:
                matches = engine.similarity(
                    smiles,
                    threshold=similarity_threshold,
                    metric="tanimoto",
                    n_workers=1,
                )
                match_ids = np.asarray(matches["mol_id"], dtype=np.int64)
                match_rows = row_by_id[match_ids]
                similarities = np.asarray(matches["coeff"], dtype=np.float32)
                current = best_similarity[match_rows]
                current_clusters = best_cluster[match_rows]
                better = (similarities > current) | (
                    (similarities == current)
                    & ((current_clusters < 0) | (clusterid < current_clusters))
                )
                chosen = match_rows[better]
                best_cluster[chosen] = clusterid
                best_similarity[chosen] = similarities[better]
            if np.any(best_cluster < 0):
                raise RuntimeError("repair LeaderPicker did not cover every molecule")
            order = np.argsort(molecule_ids)
            with (temp_dir / outputs[1]).open("wb") as handle:
                np.savez(
                    handle,
                    spacehastenid=molecule_ids[order],
                    clusterid=best_cluster[order],
                    centroid_similarity=best_similarity[order],
                )
            index_path.unlink(missing_ok=True)
            metrics = {
                "molecule_count": len(rows),
                "repair_centroid_count": len(centroids),
                "elapsed_seconds": time.monotonic() - started,
            }
        _finish(temp_dir, inputs=inputs, outputs=outputs, metrics=metrics)
        _commit_dir(temp_dir, output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def apply_repair(
    base_assignments: Path,
    repair_assignments: Path,
    output_dir: Path,
    similarity_threshold: float,
) -> dict[str, Any]:
    base_assignments = base_assignments.resolve()
    repair_assignments = repair_assignments.resolve()
    output_dir = output_dir.resolve()
    inputs = {
        "base": {**_identity(base_assignments), "sha256": _sha256(base_assignments)},
        "repair": {
            **_identity(repair_assignments),
            "sha256": _sha256(repair_assignments),
        },
        "similarity_threshold": similarity_threshold,
    }
    outputs = ("assignments.npz",)
    if _complete(output_dir, inputs, outputs):
        LOGGER.info("Reusing completed repaired assignments: %s", output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    temp_dir = _new_temp_dir(output_dir)
    started = time.monotonic()
    try:
        with np.load(base_assignments) as base:
            identifiers = base["spacehastenid"].copy()
            clusters = base["clusterid"].copy()
            similarities = base["centroid_similarity"].copy()
        with np.load(repair_assignments) as repair:
            repair_ids = repair["spacehastenid"].copy()
            repair_clusters = repair["clusterid"].copy()
            repair_similarities = repair["centroid_similarity"].copy()
        positions = np.searchsorted(identifiers, repair_ids)
        if np.any(positions == len(identifiers)) or not np.array_equal(
            identifiers[positions], repair_ids
        ):
            raise ValueError("repair assignments contain unknown molecule identifiers")
        if np.any(clusters[positions] >= 0):
            raise ValueError("repair assignments overlap already covered molecules")
        clusters[positions] = repair_clusters
        similarities[positions] = repair_similarities
        if np.any(clusters < 0):
            raise RuntimeError("coverage repair left uncovered molecules")
        if np.any(similarities < similarity_threshold):
            raise RuntimeError("coverage repair produced below-threshold assignments")
        with (temp_dir / outputs[0]).open("wb") as handle:
            np.savez(
                handle,
                spacehastenid=identifiers,
                clusterid=clusters,
                centroid_similarity=similarities,
            )
        metrics = {
            "molecule_count": len(identifiers),
            "repaired_count": len(repair_ids),
            "uncovered_count": 0,
            "elapsed_seconds": time.monotonic() - started,
        }
        _finish(temp_dir, inputs=inputs, outputs=outputs, metrics=metrics)
        _commit_dir(temp_dir, output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def combine_centroids(
    base_centroids: Path,
    repair_centroids: Path,
    output_dir: Path,
) -> dict[str, Any]:
    base_centroids = base_centroids.resolve()
    repair_centroids = repair_centroids.resolve()
    output_dir = output_dir.resolve()
    inputs = {
        "base": {**_identity(base_centroids), "sha256": _sha256(base_centroids)},
        "repair": {
            **_identity(repair_centroids),
            "sha256": _sha256(repair_centroids),
        },
        "fingerprint_type": FP_TYPE,
        "fingerprint_parameters": FP_PARAMS,
    }
    outputs = ("centroids.smi.gz", "centroids_fp.h5")
    if _complete(output_dir, inputs, outputs):
        LOGGER.info("Reusing combined centroid atlas: %s", output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    temp_dir = _new_temp_dir(output_dir)
    started = time.monotonic()
    try:
        combined: dict[int, str] = {}
        for path in (base_centroids, repair_centroids):
            for smiles, identifier in read_smi(path):
                if identifier in combined and combined[identifier] != smiles:
                    raise ValueError(f"conflicting centroid SMILES for {identifier}")
                combined[identifier] = smiles
        rows = [(smiles, identifier) for identifier, smiles in sorted(combined.items())]
        _write_smi(temp_dir / outputs[0], rows)
        _build_fpsim2_index(rows, temp_dir / outputs[1])
        metrics = {
            "centroid_count": len(rows),
            "elapsed_seconds": time.monotonic() - started,
        }
        _finish(temp_dir, inputs=inputs, outputs=outputs, metrics=metrics)
        _commit_dir(temp_dir, output_dir)
        return json.loads((output_dir / COMPLETE_FILE).read_text())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    partition = subparsers.add_parser("partition")
    partition.add_argument("--input", type=Path, required=True)
    partition.add_argument("--output-dir", type=Path, required=True)
    partition.add_argument("--partition-count", type=int, default=64)

    mapper = subparsers.add_parser("map")
    mapper.add_argument("--input", type=Path, required=True)
    mapper.add_argument("--output-dir", type=Path, required=True)
    mapper.add_argument("--similarity-threshold", type=float, default=0.4)
    mapper.add_argument("--processes", type=int, default=1)

    pool = subparsers.add_parser("pool")
    pool.add_argument("--map-root", type=Path, required=True)
    pool.add_argument("--output-dir", type=Path, required=True)

    intermediate = subparsers.add_parser("intermediate-reduce")
    intermediate.add_argument("--map-root", type=Path, required=True)
    intermediate.add_argument("--output-dir", type=Path, required=True)
    intermediate.add_argument("--group-index", type=int, required=True)
    intermediate.add_argument("--group-count", type=int, required=True)
    intermediate.add_argument("--similarity-threshold", type=float, default=0.4)
    intermediate.add_argument("--processes", type=int, default=1)

    reducer = subparsers.add_parser("reduce")
    reducer.add_argument("--input", type=Path, required=True)
    reducer.add_argument("--output-dir", type=Path, required=True)
    reducer.add_argument("--similarity-threshold", type=float, default=0.4)
    reducer.add_argument("--processes", type=int, default=1)

    index = subparsers.add_parser("build-index")
    index.add_argument("--input", type=Path, required=True)
    index.add_argument("--output-dir", type=Path, required=True)

    assign = subparsers.add_parser("assign-shard")
    assign.add_argument("--molecule-index", type=Path, required=True)
    assign.add_argument("--centroids", type=Path, required=True)
    assign.add_argument("--output-dir", type=Path, required=True)
    assign.add_argument("--shard-index", type=int, required=True)
    assign.add_argument("--shard-count", type=int, required=True)
    assign.add_argument("--similarity-threshold", type=float, default=0.4)

    merge = subparsers.add_parser("merge-assignments")
    merge.add_argument("--molecule-index", type=Path, required=True)
    merge.add_argument("--input-smiles", type=Path, required=True)
    merge.add_argument("--shard-root", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--similarity-threshold", type=float, default=0.4)

    repair = subparsers.add_parser("repair")
    repair.add_argument("--uncovered", type=Path, required=True)
    repair.add_argument("--output-dir", type=Path, required=True)
    repair.add_argument("--similarity-threshold", type=float, default=0.4)
    repair.add_argument("--processes", type=int, default=1)

    apply = subparsers.add_parser("apply-repair")
    apply.add_argument("--base-assignments", type=Path, required=True)
    apply.add_argument("--repair-assignments", type=Path, required=True)
    apply.add_argument("--output-dir", type=Path, required=True)
    apply.add_argument("--similarity-threshold", type=float, default=0.4)

    combine = subparsers.add_parser("combine-centroids")
    combine.add_argument("--base-centroids", type=Path, required=True)
    combine.add_argument("--repair-centroids", type=Path, required=True)
    combine.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.command == "partition":
        partition_smiles(args.input, args.output_dir, args.partition_count)
    elif args.command == "map":
        map_partition(args.input, args.output_dir, args.similarity_threshold, args.processes)
    elif args.command == "pool":
        pool_centroids(args.map_root, args.output_dir)
    elif args.command == "intermediate-reduce":
        reduce_centroid_group(
            args.map_root,
            args.output_dir,
            args.group_index,
            args.group_count,
            args.similarity_threshold,
            args.processes,
        )
    elif args.command == "reduce":
        reduce_centroids(args.input, args.output_dir, args.similarity_threshold, args.processes)
    elif args.command == "build-index":
        build_molecule_index(args.input, args.output_dir)
    elif args.command == "assign-shard":
        assign_centroid_shard(
            args.molecule_index,
            args.centroids,
            args.output_dir,
            args.shard_index,
            args.shard_count,
            args.similarity_threshold,
        )
    elif args.command == "merge-assignments":
        merge_assignment_shards(
            args.molecule_index,
            args.input_smiles,
            args.shard_root,
            args.output_dir,
            args.similarity_threshold,
        )
    elif args.command == "repair":
        repair_uncovered(
            args.uncovered,
            args.output_dir,
            args.similarity_threshold,
            args.processes,
        )
    elif args.command == "apply-repair":
        apply_repair(
            args.base_assignments,
            args.repair_assignments,
            args.output_dir,
            args.similarity_threshold,
        )
    elif args.command == "combine-centroids":
        combine_centroids(
            args.base_centroids,
            args.repair_centroids,
            args.output_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
