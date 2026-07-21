"""Resumable SLURM orchestration for the initial persistent seed atlas."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shlex
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from spacehasten.config.settings import Settings
from spacehasten.core.db import (
    ClusterAtlasAssignmentRow,
    ClusterAtlasCentroidRow,
    ClusterAtlasRow,
    ClusterAtlasVersionRow,
    Database,
)
from spacehasten.remote.atlas import FP_PARAMS, FP_TYPE, read_smi
from spacehasten.scheduler.base import ArrayHandle, ArrayJob, Scheduler
from spacehasten.workspace.layout import WorkDir

LOGGER = logging.getLogger(__name__)
DEFAULT_ATLAS_ID = "morgan-r2-1024-t040"


def _default_atlas_command(settings: Settings) -> tuple[str, ...]:
    try:
        return ("python3", str(settings.remote_script_path("atlas")))
    except ValueError:
        return ("python3", "-m", "spacehasten.remote.atlas")


def _quote(parts: Sequence[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _database_seed_digest(db: Database) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    rows = db.connection.execute(
        "SELECT smiles, spacehastenid FROM data WHERE dock_iteration = 0 ORDER BY spacehastenid"
    )
    for smiles, identifier in rows:
        digest.update(f"{str(smiles).strip()} {int(identifier)}\n".encode())
        count += 1
    return count, digest.hexdigest()


def _export_seed_smiles(db: Database, path: Path) -> int:
    metadata_path = path.with_suffix(path.suffix + ".json")
    expected_count = int(
        db.connection.execute("SELECT COUNT(*) FROM data WHERE dock_iteration = 0").fetchone()[0]
    )
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        database_count, database_hash = _database_seed_digest(db)
        if (
            metadata.get("seed_count") == expected_count
            and database_count == expected_count
            and metadata.get("content_sha256") == database_hash
            and metadata.get("file_sha256") == _sha256(path)
        ):
            LOGGER.info("Reusing exported seed atlas input: %s", path)
            return expected_count
        raise ValueError(f"reusable atlas seed input does not match database seeds: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    count = 0
    content_digest = hashlib.sha256()
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        rows = db.connection.execute(
            "SELECT smiles, spacehastenid FROM data WHERE dock_iteration = 0 ORDER BY spacehastenid"
        )
        for smiles, identifier in rows:
            line = f"{str(smiles).strip()} {int(identifier)}\n"
            handle.write(line)
            content_digest.update(line.encode())
            count += 1
    if count != expected_count:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"exported {count} seeds; expected {expected_count}")
    temporary.replace(path)
    metadata_path.write_text(
        json.dumps(
            {
                "seed_count": count,
                "content_sha256": content_digest.hexdigest(),
                "file_sha256": _sha256(path),
            },
            indent=2,
        )
        + "\n"
    )
    return count


def _wait_job(scheduler: Scheduler, handle: ArrayHandle) -> None:
    result = scheduler.wait(handle)
    if result.success:
        return
    from spacehasten.scheduler.diagnostics import tail_logs

    raise RuntimeError(
        f"atlas job {handle.job_id} ({handle.name}) failed; "
        f"failed tasks: {result.failed_indices}\n{tail_logs(handle)}"
    )


def _submit(
    scheduler: Scheduler,
    *,
    name: str,
    workdir: Path,
    array_size: int,
    max_concurrent: int,
    cpus: int,
    command: str,
    env_setup: list[str],
    exclusive: bool = False,
) -> None:
    job = ArrayJob(
        name=name,
        workdir=workdir,
        array_size=array_size,
        max_concurrent=max_concurrent,
        cpus_per_task=cpus,
        exclusive=exclusive,
        env_setup=env_setup,
        command_template=command,
    )
    handle = scheduler.submit_array(job)
    LOGGER.info("Submitted atlas job %s (%s)", handle.job_id, name)
    _wait_job(scheduler, handle)


def _atlas_environment(settings: Settings) -> list[str]:
    return [
        line
        for line in (
            settings.general.prepare_anaconda,
            settings.general.activate_clustering,
        )
        if line
    ]


def _mapper_command(
    prefix: Sequence[str],
    partitions: Path,
    mapper_root: Path,
    partition_count: int,
    threshold: float,
) -> str:
    suffix = f"of_{partition_count:04d}"
    return "\n".join(
        [
            'INDEX=$(printf "%04d" $(( ${TASK_ID} - 1 )))',
            f'INPUT="{partitions}/part_${{INDEX}}_{suffix}.smi.gz"',
            f'OUTPUT="{mapper_root}/map_${{INDEX}}"',
            _quote(
                [
                    *prefix,
                    "map",
                    "--input",
                    "${INPUT}",
                    "--output-dir",
                    "${OUTPUT}",
                    "--similarity-threshold",
                    str(threshold),
                    "--processes",
                    "1",
                ]
            )
            .replace("'${INPUT}'", '"${INPUT}"')
            .replace("'${OUTPUT}'", '"${OUTPUT}"'),
        ]
    )


def _assignment_command(
    prefix: Sequence[str],
    molecule_index: Path,
    centroids: Path,
    shard_root: Path,
    shard_count: int,
    threshold: float,
) -> str:
    return "\n".join(
        [
            "INDEX=$(( ${TASK_ID} - 1 ))",
            'PADDED=$(printf "%04d" "${INDEX}")',
            'SCRATCH="/fastwrk/${USER}/spacehasten-atlas/${SLURM_JOB_ID:-local}_${TASK_ID}"',
            'mkdir -p "${SCRATCH}"',
            f'cp "{molecule_index}" "${{SCRATCH}}/molecules_fp.h5"',
            f'OUTPUT="{shard_root}/shard_${{PADDED}}"',
            _quote(
                [
                    *prefix,
                    "assign-shard",
                    "--molecule-index",
                    "${SCRATCH}/molecules_fp.h5",
                    "--centroids",
                    centroids,
                    "--output-dir",
                    "${OUTPUT}",
                    "--shard-index",
                    "${INDEX}",
                    "--shard-count",
                    str(shard_count),
                    "--similarity-threshold",
                    str(threshold),
                ]
            )
            .replace("'${SCRATCH}/molecules_fp.h5'", '"${SCRATCH}/molecules_fp.h5"')
            .replace("'${OUTPUT}'", '"${OUTPUT}"')
            .replace("'${INDEX}'", '"${INDEX}"'),
            'rm -rf "${SCRATCH}"',
        ]
    )


def _batched(values: Iterable, size: int):  # type: ignore[no-untyped-def]
    batch = []
    for value in values:
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _ingest_initial_atlas(
    db: Database,
    *,
    atlas_id: str,
    atlas_root: Path,
    partition_count: int,
    threshold: float,
) -> ClusterAtlasVersionRow:
    final_assignments = atlas_root / "final" / "assignments.npz"
    reduced_centroids = atlas_root / "reduced" / "centroids.smi.gz"
    repair_centroids = atlas_root / "repair" / "repair_centroids.smi.gz"
    if not all(path.is_file() for path in (final_assignments, reduced_centroids, repair_centroids)):
        raise FileNotFoundError("initial atlas outputs are incomplete")

    centroids = {
        identifier for _, identifier in read_smi(reduced_centroids) + read_smi(repair_centroids)
    }
    with np.load(final_assignments) as assignments:
        identifiers = assignments["spacehastenid"].copy()
        cluster_ids = assignments["clusterid"].copy()
        similarities = assignments["centroid_similarity"].copy()
    if len(identifiers) == 0 or np.any(cluster_ids < 0):
        raise ValueError("initial atlas assignments are empty or incomplete")
    if np.any(similarities < threshold):
        raise ValueError("initial atlas contains below-threshold assignments")
    if not set(np.unique(cluster_ids)).issubset(centroids):
        raise ValueError("assignments reference unknown atlas centroids")

    db.upsert_cluster_atlas(
        ClusterAtlasRow(
            atlas_id=atlas_id,
            similarity_threshold=threshold,
            fingerprint_type=FP_TYPE,
            fingerprint_parameters=json.dumps(FP_PARAMS, sort_keys=True),
            partition_count=partition_count,
        )
    )
    try:
        for batch in _batched(sorted(centroids), 10_000):
            db.append_cluster_atlas_centroids(
                ClusterAtlasCentroidRow(atlas_id, sid, sid, 0) for sid in batch
            )
        rows = (
            ClusterAtlasAssignmentRow(
                atlas_id,
                int(identifier),
                int(cluster_id),
                float(similarity),
                0,
            )
            for identifier, cluster_id, similarity in zip(
                identifiers, cluster_ids, similarities, strict=True
            )
        )
        for batch in _batched(rows, 50_000):
            db.append_cluster_atlas_assignments(batch)
        version = ClusterAtlasVersionRow(
            atlas_id=atlas_id,
            version=0,
            last_spacehastenid=int(identifiers.max()),
            compound_count=len(identifiers),
            centroid_count=len(centroids),
            metadata_path=str(atlas_root / "final" / "complete.json"),
        )
        db.record_cluster_atlas_version(version)
        db.materialize_cluster_atlas(atlas_id)
        db.commit()
        return version
    except Exception:
        db.connection.rollback()
        raise


def build_initial_seed_atlas(
    db: Database,
    workdir: WorkDir,
    scheduler: Scheduler,
    settings: Settings,
    *,
    atlas_id: str = DEFAULT_ATLAS_ID,
    atlas_root: Path | None = None,
    command_prefix: Sequence[str] | None = None,
) -> ClusterAtlasVersionRow:
    """Build or resume atlas version 0 from all ``dock_iteration=0`` seeds."""
    existing = db.latest_cluster_atlas_version(atlas_id)
    if existing is not None:
        if existing.version != 0:
            raise ValueError(f"atlas {atlas_id!r} is already beyond version 0")
        db.materialize_cluster_atlas(atlas_id)
        db.commit()
        LOGGER.info("Reusing persisted initial atlas %s", atlas_id)
        return existing

    atlas_root = (atlas_root or workdir.atlas_dir()).resolve()
    atlas_root.mkdir(parents=True, exist_ok=True)
    source = atlas_root / "source" / "seeds.smi.gz"
    seed_count = _export_seed_smiles(db, source)
    if seed_count == 0:
        raise ValueError("cannot build a seed atlas without docked seeds")

    g = settings.general
    partition_count = g.atlas_partition_count
    shard_count = g.atlas_assignment_shards
    threshold = g.atlas_similarity_threshold
    clustering_cpus = max(1, int(g.cpu_count_clustering or 1))
    prefix = command_prefix or _default_atlas_command(settings)
    env_setup = _atlas_environment(settings)

    seed_metadata = json.loads(source.with_suffix(source.suffix + ".json").read_text())
    definition = {
        "atlas_id": atlas_id,
        "seed_count": seed_count,
        "seed_sha256": seed_metadata["content_sha256"],
        "similarity_threshold": threshold,
        "fingerprint_type": FP_TYPE,
        "fingerprint_parameters": FP_PARAMS,
        "partition_count": partition_count,
        "partition_rule": "spacehastenid_modulo",
    }
    definition_path = atlas_root / "atlas_definition.json"
    if definition_path.is_file():
        if json.loads(definition_path.read_text()) != definition:
            raise ValueError(f"reusable atlas definition does not match this run: {atlas_root}")
    else:
        temporary = definition_path.with_name(f".{definition_path.name}.tmp")
        temporary.write_text(json.dumps(definition, indent=2, sort_keys=True) + "\n")
        temporary.replace(definition_path)

    reusable_outputs = (
        atlas_root / "final" / "assignments.npz",
        atlas_root / "final" / "complete.json",
        atlas_root / "reduced" / "centroids.smi.gz",
        atlas_root / "repair" / "repair_centroids.smi.gz",
    )
    if all(path.is_file() for path in reusable_outputs):
        LOGGER.info("Importing completed reusable seed atlas: %s", atlas_root)
        return _ingest_initial_atlas(
            db,
            atlas_id=atlas_id,
            atlas_root=atlas_root,
            partition_count=partition_count,
            threshold=threshold,
        )

    partitions = atlas_root / "partitions"
    _submit(
        scheduler,
        name="atlas_partition_v0",
        workdir=atlas_root,
        array_size=1,
        max_concurrent=1,
        cpus=1,
        env_setup=env_setup,
        command=_quote(
            [
                *prefix,
                "partition",
                "--input",
                source,
                "--output-dir",
                partitions,
                "--partition-count",
                str(partition_count),
            ]
        ),
    )

    mapper_root = atlas_root / "mappers"
    _submit(
        scheduler,
        name="atlas_map_v0",
        workdir=atlas_root,
        array_size=partition_count,
        max_concurrent=partition_count,
        cpus=1,
        env_setup=env_setup,
        command=_mapper_command(prefix, partitions, mapper_root, partition_count, threshold),
    )

    pool = atlas_root / "pool"
    reduced = atlas_root / "reduced"
    molecule_index = atlas_root / "molecule_index"
    reducer_command = "\n".join(
        [
            _quote([*prefix, "pool", "--map-root", mapper_root, "--output-dir", pool]),
            _quote(
                [
                    *prefix,
                    "reduce",
                    "--input",
                    pool / "pooled_centroids.smi.gz",
                    "--output-dir",
                    reduced,
                    "--similarity-threshold",
                    str(threshold),
                    "--processes",
                    str(clustering_cpus),
                ]
            ),
            _quote(
                [
                    *prefix,
                    "build-index",
                    "--input",
                    source,
                    "--output-dir",
                    molecule_index,
                ]
            ),
        ]
    )
    _submit(
        scheduler,
        name="atlas_reduce_v0",
        workdir=atlas_root,
        array_size=1,
        max_concurrent=1,
        cpus=clustering_cpus,
        exclusive=True,
        env_setup=env_setup,
        command=reducer_command,
    )

    shard_root = atlas_root / "assignment_shards"
    _submit(
        scheduler,
        name="atlas_assign_v0",
        workdir=atlas_root,
        array_size=shard_count,
        max_concurrent=shard_count,
        cpus=1,
        env_setup=env_setup,
        command=_assignment_command(
            prefix,
            molecule_index / "molecules_fp.h5",
            reduced / "centroids.smi.gz",
            shard_root,
            shard_count,
            threshold,
        ),
    )

    merged = atlas_root / "merged"
    repair = atlas_root / "repair"
    final = atlas_root / "final"
    finalize_command = "\n".join(
        [
            _quote(
                [
                    *prefix,
                    "merge-assignments",
                    "--molecule-index",
                    molecule_index / "molecules_fp.h5",
                    "--input-smiles",
                    source,
                    "--shard-root",
                    shard_root,
                    "--output-dir",
                    merged,
                    "--similarity-threshold",
                    str(threshold),
                ]
            ),
            _quote(
                [
                    *prefix,
                    "repair",
                    "--uncovered",
                    merged / "uncovered.smi.gz",
                    "--output-dir",
                    repair,
                    "--similarity-threshold",
                    str(threshold),
                    "--processes",
                    str(clustering_cpus),
                ]
            ),
            _quote(
                [
                    *prefix,
                    "apply-repair",
                    "--base-assignments",
                    merged / "assignments.npz",
                    "--repair-assignments",
                    repair / "repair_assignments.npz",
                    "--output-dir",
                    final,
                    "--similarity-threshold",
                    str(threshold),
                ]
            ),
        ]
    )
    _submit(
        scheduler,
        name="atlas_finalize_v0",
        workdir=atlas_root,
        array_size=1,
        max_concurrent=1,
        cpus=clustering_cpus,
        exclusive=True,
        env_setup=env_setup,
        command=finalize_command,
    )
    version = _ingest_initial_atlas(
        db,
        atlas_id=atlas_id,
        atlas_root=atlas_root,
        partition_count=partition_count,
        threshold=threshold,
    )
    LOGGER.info(
        "Built initial atlas %s: compounds=%d centroids=%d",
        atlas_id,
        version.compound_count,
        version.centroid_count,
    )
    return version


__all__ = ["DEFAULT_ATLAS_ID", "build_initial_seed_atlas"]
