"""Resumable SLURM orchestration for the initial persistent seed atlas."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shlex
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

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
T = TypeVar("T")


@dataclass(frozen=True)
class CentroidDiscoveryArtifacts:
    centroids: Path
    centroid_index: Path
    molecule_index: Path


@dataclass(frozen=True)
class AssignmentArtifacts:
    assignments: Path
    uncovered: Path


@dataclass(frozen=True)
class AtlasVersionArtifacts:
    assignments: Path
    new_centroid_files: tuple[Path, ...]
    metadata_path: Path


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


def _database_range_digest(
    db: Database, lower_exclusive: int, upper_inclusive: int
) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    rows = db.connection.execute(
        "SELECT smiles, spacehastenid FROM data "
        "WHERE spacehastenid > ? AND spacehastenid <= ? ORDER BY spacehastenid",
        (lower_exclusive, upper_inclusive),
    )
    for smiles, identifier in rows:
        digest.update(f"{str(smiles).strip()} {int(identifier)}\n".encode())
        count += 1
    return count, digest.hexdigest()


def _export_incremental_smiles(
    db: Database,
    path: Path,
    *,
    lower_exclusive: int,
    upper_inclusive: int,
) -> int:
    metadata_path = path.with_suffix(path.suffix + ".json")
    expected_count, expected_hash = _database_range_digest(db, lower_exclusive, upper_inclusive)
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("lower_exclusive") == lower_exclusive
            and metadata.get("upper_inclusive") == upper_inclusive
            and metadata.get("compound_count") == expected_count
            and metadata.get("content_sha256") == expected_hash
            and metadata.get("file_sha256") == _sha256(path)
        ):
            LOGGER.info("Reusing exported incremental atlas input: %s", path)
            return expected_count
        raise ValueError(f"incremental atlas input does not match database: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    digest = hashlib.sha256()
    count = 0
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        rows = db.connection.execute(
            "SELECT smiles, spacehastenid FROM data "
            "WHERE spacehastenid > ? AND spacehastenid <= ? "
            "ORDER BY spacehastenid",
            (lower_exclusive, upper_inclusive),
        )
        for smiles, identifier in rows:
            line = f"{str(smiles).strip()} {int(identifier)}\n"
            handle.write(line)
            digest.update(line.encode())
            count += 1
    if count != expected_count or digest.hexdigest() != expected_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("incremental atlas export changed during streaming")
    temporary.replace(path)
    metadata_path.write_text(
        json.dumps(
            {
                "lower_exclusive": lower_exclusive,
                "upper_inclusive": upper_inclusive,
                "compound_count": count,
                "content_sha256": expected_hash,
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
                    "--identity-molecule-index",
                    molecule_index,
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


def _intermediate_reducer_command(
    prefix: Sequence[str],
    mapper_root: Path,
    output_root: Path,
    reducer_count: int,
    threshold: float,
) -> str:
    return "\n".join(
        [
            "INDEX=$(( ${TASK_ID} - 1 ))",
            'PADDED=$(printf "%04d" "${INDEX}")',
            f'OUTPUT="{output_root}/reducer_${{PADDED}}"',
            _quote(
                [
                    *prefix,
                    "intermediate-reduce",
                    "--map-root",
                    mapper_root,
                    "--output-dir",
                    "${OUTPUT}",
                    "--group-index",
                    "${INDEX}",
                    "--group-count",
                    str(reducer_count),
                    "--similarity-threshold",
                    str(threshold),
                    "--processes",
                    "1",
                ]
            )
            .replace("'${OUTPUT}'", '"${OUTPUT}"')
            .replace("'${INDEX}'", '"${INDEX}"'),
        ]
    )


def run_centroid_discovery(
    scheduler: Scheduler,
    settings: Settings,
    *,
    input_smiles: Path,
    output_root: Path,
    command_prefix: Sequence[str],
    job_suffix: str,
) -> CentroidDiscoveryArtifacts:
    """Run the shared partition/map/hierarchical-reduce discovery pipeline."""
    g = settings.general
    partition_count = g.atlas_partition_count
    intermediate_reducers = g.atlas_intermediate_reducers
    threshold = g.atlas_similarity_threshold
    clustering_cpus = max(1, int(g.cpu_count_clustering or 1))
    if not 1 <= intermediate_reducers <= partition_count:
        raise ValueError("atlas_intermediate_reducers must be between 1 and atlas_partition_count")
    env_setup = _atlas_environment(settings)
    output_root.mkdir(parents=True, exist_ok=True)

    partitions = output_root / "partitions"
    _submit(
        scheduler,
        name=f"atlas_partition_{job_suffix}",
        workdir=output_root,
        array_size=1,
        max_concurrent=1,
        cpus=1,
        env_setup=env_setup,
        command=_quote(
            [
                *command_prefix,
                "partition",
                "--input",
                input_smiles,
                "--output-dir",
                partitions,
                "--partition-count",
                str(partition_count),
            ]
        ),
    )

    mapper_root = output_root / "mappers"
    _submit(
        scheduler,
        name=f"atlas_map_{job_suffix}",
        workdir=output_root,
        array_size=partition_count,
        max_concurrent=partition_count,
        cpus=1,
        env_setup=env_setup,
        command=_mapper_command(
            command_prefix,
            partitions,
            mapper_root,
            partition_count,
            threshold,
        ),
    )

    intermediate_root = output_root / "intermediate_reducers"
    _submit(
        scheduler,
        name=f"atlas_intermediate_{job_suffix}",
        workdir=output_root,
        array_size=intermediate_reducers,
        max_concurrent=intermediate_reducers,
        cpus=1,
        env_setup=env_setup,
        command=_intermediate_reducer_command(
            command_prefix,
            mapper_root,
            intermediate_root,
            intermediate_reducers,
            threshold,
        ),
    )

    pool = output_root / "pool"
    reduced = output_root / "reduced"
    molecule_index = output_root / "molecule_index"
    reducer_command = "\n".join(
        [
            _quote(
                [
                    *command_prefix,
                    "pool",
                    "--map-root",
                    intermediate_root,
                    "--output-dir",
                    pool,
                ]
            ),
            _quote(
                [
                    *command_prefix,
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
                    *command_prefix,
                    "build-index",
                    "--input",
                    input_smiles,
                    "--output-dir",
                    molecule_index,
                ]
            ),
        ]
    )
    _submit(
        scheduler,
        name=f"atlas_reduce_{job_suffix}",
        workdir=output_root,
        array_size=1,
        max_concurrent=1,
        cpus=clustering_cpus,
        exclusive=True,
        env_setup=env_setup,
        command=reducer_command,
    )
    return CentroidDiscoveryArtifacts(
        centroids=reduced / "centroids.smi.gz",
        centroid_index=reduced / "centroids_fp.h5",
        molecule_index=molecule_index / "molecules_fp.h5",
    )


def run_parallel_assignment(
    scheduler: Scheduler,
    settings: Settings,
    *,
    input_smiles: Path,
    molecule_index: Path,
    centroids: Path,
    output_root: Path,
    command_prefix: Sequence[str],
    job_suffix: str,
) -> AssignmentArtifacts:
    """Run the shared centroid-sharded assignment and deterministic merge."""
    shard_count = settings.general.atlas_assignment_shards
    threshold = settings.general.atlas_similarity_threshold
    env_setup = _atlas_environment(settings)
    shard_root = output_root / "assignment_shards"
    _submit(
        scheduler,
        name=f"atlas_assign_{job_suffix}",
        workdir=output_root,
        array_size=shard_count,
        max_concurrent=shard_count,
        cpus=1,
        env_setup=env_setup,
        command=_assignment_command(
            command_prefix,
            molecule_index,
            centroids,
            shard_root,
            shard_count,
            threshold,
        ),
    )
    merged = output_root / "merged"
    _submit(
        scheduler,
        name=f"atlas_merge_{job_suffix}",
        workdir=output_root,
        array_size=1,
        max_concurrent=1,
        cpus=1,
        env_setup=env_setup,
        command=_quote(
            [
                *command_prefix,
                "merge-assignments",
                "--molecule-index",
                molecule_index,
                "--input-smiles",
                input_smiles,
                "--shard-root",
                shard_root,
                "--output-dir",
                merged,
                "--similarity-threshold",
                str(threshold),
            ]
        ),
    )
    return AssignmentArtifacts(
        assignments=merged / "assignments.npz",
        uncovered=merged / "uncovered.smi.gz",
    )


def run_coverage_repair(
    scheduler: Scheduler,
    settings: Settings,
    *,
    assignments: AssignmentArtifacts,
    base_centroids: Path,
    output_root: Path,
    command_prefix: Sequence[str],
    job_suffix: str,
) -> AtlasVersionArtifacts:
    """Run exact uncovered repair and publish combined final centroids."""
    threshold = settings.general.atlas_similarity_threshold
    clustering_cpus = max(1, int(settings.general.cpu_count_clustering or 1))
    env_setup = _atlas_environment(settings)
    repair = output_root / "repair"
    final = output_root / "final"
    command = "\n".join(
        [
            _quote(
                [
                    *command_prefix,
                    "repair",
                    "--uncovered",
                    assignments.uncovered,
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
                    *command_prefix,
                    "apply-repair",
                    "--base-assignments",
                    assignments.assignments,
                    "--repair-assignments",
                    repair / "repair_assignments.npz",
                    "--output-dir",
                    final,
                    "--similarity-threshold",
                    str(threshold),
                ]
            ),
            _quote(
                [
                    *command_prefix,
                    "combine-centroids",
                    "--base-centroids",
                    base_centroids,
                    "--repair-centroids",
                    repair / "repair_centroids.smi.gz",
                    "--output-dir",
                    final / "centroids",
                ]
            ),
        ]
    )
    _submit(
        scheduler,
        name=f"atlas_repair_{job_suffix}",
        workdir=output_root,
        array_size=1,
        max_concurrent=1,
        cpus=clustering_cpus,
        exclusive=True,
        env_setup=env_setup,
        command=command,
    )
    return AtlasVersionArtifacts(
        assignments=final / "assignments.npz",
        new_centroid_files=(final / "centroids" / "centroids.smi.gz",),
        metadata_path=final / "complete.json",
    )


def build_atlas_version(
    scheduler: Scheduler,
    settings: Settings,
    *,
    input_smiles: Path,
    output_root: Path,
    command_prefix: Sequence[str],
    job_suffix: str,
    existing_centroids: Path | None = None,
) -> AtlasVersionArtifacts:
    """Build one atlas version using shared discovery/assignment/repair steps."""
    if existing_centroids is None:
        discovery = run_centroid_discovery(
            scheduler,
            settings,
            input_smiles=input_smiles,
            output_root=output_root,
            command_prefix=command_prefix,
            job_suffix=job_suffix,
        )
        assignment = run_parallel_assignment(
            scheduler,
            settings,
            input_smiles=input_smiles,
            molecule_index=discovery.molecule_index,
            centroids=discovery.centroids,
            output_root=output_root,
            command_prefix=command_prefix,
            job_suffix=job_suffix,
        )
        return run_coverage_repair(
            scheduler,
            settings,
            assignments=assignment,
            base_centroids=discovery.centroids,
            output_root=output_root,
            command_prefix=command_prefix,
            job_suffix=job_suffix,
        )

    env_setup = _atlas_environment(settings)
    new_index_root = output_root / "new_molecule_index"
    _submit(
        scheduler,
        name=f"atlas_index_{job_suffix}",
        workdir=output_root,
        array_size=1,
        max_concurrent=1,
        cpus=1,
        env_setup=env_setup,
        command=_quote(
            [
                *command_prefix,
                "build-index",
                "--input",
                input_smiles,
                "--output-dir",
                new_index_root,
            ]
        ),
    )
    existing_assignment = run_parallel_assignment(
        scheduler,
        settings,
        input_smiles=input_smiles,
        molecule_index=new_index_root / "molecules_fp.h5",
        centroids=existing_centroids,
        output_root=output_root / "existing_assignment",
        command_prefix=command_prefix,
        job_suffix=f"{job_suffix}_existing",
    )
    merge_metadata = json.loads(
        (existing_assignment.assignments.parent / "complete.json").read_text()
    )
    if int(merge_metadata["metrics"]["uncovered_count"]) == 0:
        return AtlasVersionArtifacts(
            assignments=existing_assignment.assignments,
            new_centroid_files=(),
            metadata_path=existing_assignment.assignments.parent / "complete.json",
        )

    novel_root = output_root / "novel"
    novel = build_atlas_version(
        scheduler,
        settings,
        input_smiles=existing_assignment.uncovered,
        output_root=novel_root,
        command_prefix=command_prefix,
        job_suffix=f"{job_suffix}_novel",
    )
    final = output_root / "final"
    _submit(
        scheduler,
        name=f"atlas_combine_{job_suffix}",
        workdir=output_root,
        array_size=1,
        max_concurrent=1,
        cpus=1,
        env_setup=env_setup,
        command=_quote(
            [
                *command_prefix,
                "apply-repair",
                "--base-assignments",
                existing_assignment.assignments,
                "--repair-assignments",
                novel.assignments,
                "--output-dir",
                final,
                "--similarity-threshold",
                str(settings.general.atlas_similarity_threshold),
            ]
        ),
    )
    return AtlasVersionArtifacts(
        assignments=final / "assignments.npz",
        new_centroid_files=novel.new_centroid_files,
        metadata_path=final / "complete.json",
    )


def _batched(values: Iterable[T], size: int) -> Iterator[list[T]]:
    batch: list[T] = []
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
    combined_centroids = atlas_root / "final" / "centroids" / "centroids.smi.gz"
    centroid_files: tuple[Path, ...]
    if combined_centroids.is_file():
        centroid_files = (combined_centroids,)
    else:
        centroid_files = (
            atlas_root / "reduced" / "centroids.smi.gz",
            atlas_root / "repair" / "repair_centroids.smi.gz",
        )
    if not final_assignments.is_file() or not all(path.is_file() for path in centroid_files):
        raise FileNotFoundError("initial atlas outputs are incomplete")

    centroids = {identifier for path in centroid_files for _, identifier in read_smi(path)}
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
        for centroid_batch in _batched(sorted(centroids), 10_000):
            db.append_cluster_atlas_centroids(
                ClusterAtlasCentroidRow(atlas_id, sid, sid, 0) for sid in centroid_batch
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
        for assignment_batch in _batched(rows, 50_000):
            db.append_cluster_atlas_assignments(assignment_batch)
        version = ClusterAtlasVersionRow(
            atlas_id=atlas_id,
            version=0,
            last_spacehastenid=int(identifiers.max()),
            compound_count=len(identifiers),
            centroid_count=len(centroids),
            metadata_path=str(atlas_root / "final" / "complete.json"),
        )
        db.record_cluster_atlas_version(version)
        db.commit()
        return version
    except Exception:
        db.connection.rollback()
        raise


def _version_root(version: ClusterAtlasVersionRow) -> Path:
    return Path(version.metadata_path).resolve().parent.parent


def _existing_centroids_path(version: ClusterAtlasVersionRow) -> Path | None:
    root = _version_root(version)
    for candidate in (
        root / "centroids" / "centroids.smi.gz",
        root / "final" / "centroids" / "centroids.smi.gz",
    ):
        if candidate.is_file():
            return candidate
    return None


def _combine_centroid_versions(
    scheduler: Scheduler,
    settings: Settings,
    *,
    existing_centroids: Path,
    new_centroids: Path,
    output_root: Path,
    command_prefix: Sequence[str],
    job_suffix: str,
) -> Path:
    env_setup = _atlas_environment(settings)
    _submit(
        scheduler,
        name=f"atlas_centroids_{job_suffix}",
        workdir=output_root,
        array_size=1,
        max_concurrent=1,
        cpus=1,
        env_setup=env_setup,
        command=_quote(
            [
                *command_prefix,
                "combine-centroids",
                "--base-centroids",
                existing_centroids,
                "--repair-centroids",
                new_centroids,
                "--output-dir",
                output_root / "centroids",
            ]
        ),
    )
    return output_root / "centroids" / "centroids.smi.gz"


def _ensure_initial_centroid_index(
    scheduler: Scheduler,
    settings: Settings,
    *,
    version: ClusterAtlasVersionRow,
    command_prefix: Sequence[str],
) -> Path:
    existing = _existing_centroids_path(version)
    if existing is not None:
        return existing
    root = _version_root(version)
    reduced = root / "reduced" / "centroids.smi.gz"
    repair = root / "repair" / "repair_centroids.smi.gz"
    if not reduced.is_file() or not repair.is_file():
        raise FileNotFoundError("atlas version has no reusable centroid artifacts")
    return _combine_centroid_versions(
        scheduler,
        settings,
        existing_centroids=reduced,
        new_centroids=repair,
        output_root=root / "final",
        command_prefix=command_prefix,
        job_suffix=f"v{version.version}_upgrade",
    )


def _ingest_incremental_atlas(
    db: Database,
    *,
    atlas_id: str,
    previous: ClusterAtlasVersionRow,
    artifacts: AtlasVersionArtifacts,
    combined_centroids: Path,
    version_root: Path,
    upper_spacehastenid: int,
    similarity_threshold: float,
) -> ClusterAtlasVersionRow:
    with np.load(artifacts.assignments) as assignments:
        identifiers = assignments["spacehastenid"].copy()
        cluster_ids = assignments["clusterid"].copy()
        similarities = assignments["centroid_similarity"].copy()
    if len(identifiers) == 0 or np.any(cluster_ids < 0):
        raise ValueError("incremental atlas assignments are empty or incomplete")
    if np.any(similarities < similarity_threshold):
        raise ValueError("incremental atlas contains below-threshold assignments")
    new_centroids = {
        identifier for path in artifacts.new_centroid_files for _, identifier in read_smi(path)
    }
    all_centroids = {identifier for _, identifier in read_smi(combined_centroids)}
    if not set(np.unique(cluster_ids)).issubset(all_centroids):
        raise ValueError("incremental assignments reference unknown centroids")
    version_number = previous.version + 1
    version_metadata = version_root / "final" / "atlas_version.json"
    version_metadata.parent.mkdir(parents=True, exist_ok=True)
    temporary_metadata = version_metadata.with_name(f".{version_metadata.name}.tmp")
    temporary_metadata.write_text(
        json.dumps(
            {
                "atlas_id": atlas_id,
                "version": version_number,
                "previous_version": previous.version,
                "lower_exclusive": previous.last_spacehastenid,
                "upper_inclusive": upper_spacehastenid,
                "assignment_count": len(identifiers),
                "new_centroid_count": len(new_centroids),
                "assignments": {
                    "path": str(artifacts.assignments.resolve()),
                    "sha256": _sha256(artifacts.assignments),
                },
                "centroids": {
                    "path": str(combined_centroids.resolve()),
                    "sha256": _sha256(combined_centroids),
                },
                "component_metadata_path": str(artifacts.metadata_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary_metadata.replace(version_metadata)
    try:
        for centroid_batch in _batched(sorted(new_centroids), 10_000):
            db.append_cluster_atlas_centroids(
                ClusterAtlasCentroidRow(
                    atlas_id,
                    identifier,
                    identifier,
                    version_number,
                )
                for identifier in centroid_batch
            )
        rows = (
            ClusterAtlasAssignmentRow(
                atlas_id,
                int(identifier),
                int(cluster_id),
                float(similarity),
                version_number,
            )
            for identifier, cluster_id, similarity in zip(
                identifiers, cluster_ids, similarities, strict=True
            )
        )
        for assignment_batch in _batched(rows, 50_000):
            db.append_cluster_atlas_assignments(assignment_batch)
        version = ClusterAtlasVersionRow(
            atlas_id=atlas_id,
            version=version_number,
            last_spacehastenid=upper_spacehastenid,
            compound_count=previous.compound_count + len(identifiers),
            centroid_count=len(all_centroids),
            metadata_path=str(version_metadata),
        )
        db.record_cluster_atlas_version(version)
        db.commit()
        return version
    except Exception:
        db.connection.rollback()
        raise


def _initial_atlas_outputs(atlas_root: Path) -> tuple[Path, ...]:
    source = atlas_root / "source" / "seeds.smi.gz"
    return (
        atlas_root / "atlas_definition.json",
        source,
        source.with_suffix(source.suffix + ".json"),
        atlas_root / "final" / "assignments.npz",
        atlas_root / "final" / "complete.json",
        atlas_root / "reduced" / "centroids.smi.gz",
        atlas_root / "repair" / "repair_centroids.smi.gz",
    )


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
    intermediate_reducers = g.atlas_intermediate_reducers
    threshold = g.atlas_similarity_threshold
    prefix = command_prefix or _default_atlas_command(settings)
    if not 1 <= intermediate_reducers <= partition_count:
        raise ValueError("atlas_intermediate_reducers must be between 1 and atlas_partition_count")

    seed_metadata = json.loads(source.with_suffix(source.suffix + ".json").read_text())
    definition = {
        "atlas_id": atlas_id,
        "seed_count": seed_count,
        "seed_sha256": seed_metadata["content_sha256"],
        "similarity_threshold": threshold,
        "fingerprint_type": FP_TYPE,
        "fingerprint_parameters": FP_PARAMS,
        "partition_count": partition_count,
        "intermediate_reducers": intermediate_reducers,
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

    reusable_outputs = _initial_atlas_outputs(atlas_root)[3:]
    if all(path.is_file() for path in reusable_outputs):
        LOGGER.info("Importing completed reusable seed atlas: %s", atlas_root)
        return _ingest_initial_atlas(
            db,
            atlas_id=atlas_id,
            atlas_root=atlas_root,
            partition_count=partition_count,
            threshold=threshold,
        )

    build_atlas_version(
        scheduler,
        settings,
        input_smiles=source,
        output_root=atlas_root,
        command_prefix=prefix,
        job_suffix="v0",
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


def import_initial_seed_atlas(
    db: Database,
    workdir: WorkDir,
    scheduler: Scheduler,
    settings: Settings,
    *,
    atlas_root: Path,
    atlas_id: str = DEFAULT_ATLAS_ID,
    command_prefix: Sequence[str] | None = None,
) -> ClusterAtlasVersionRow:
    """Import a completed seed atlas without allowing an implicit build."""
    existing = db.latest_cluster_atlas_version(atlas_id)
    if existing is not None:
        return existing
    atlas_root = atlas_root.resolve()
    missing = [path for path in _initial_atlas_outputs(atlas_root) if not path.is_file()]
    if missing:
        missing_names = ", ".join(str(path.relative_to(atlas_root)) for path in missing)
        raise FileNotFoundError(
            f"no completed seed atlas at {atlas_root}; missing: {missing_names}"
        )
    return build_initial_seed_atlas(
        db,
        workdir,
        scheduler,
        settings,
        atlas_id=atlas_id,
        atlas_root=atlas_root,
        command_prefix=command_prefix,
    )


def update_cluster_atlas(
    db: Database,
    workdir: WorkDir,
    scheduler: Scheduler,
    settings: Settings,
    *,
    atlas_id: str = DEFAULT_ATLAS_ID,
    through_spacehastenid: int | None = None,
    command_prefix: Sequence[str] | None = None,
) -> ClusterAtlasVersionRow:
    """Append one atlas version for compounds beyond the persisted watermark."""
    previous = db.latest_cluster_atlas_version(atlas_id)
    if previous is None:
        raise ValueError(f"atlas {atlas_id!r} is not initialized")
    atlas_config = db.connection.execute(
        "SELECT similarity_threshold, fingerprint_type, fingerprint_parameters, "
        "partition_count FROM cluster_atlases WHERE atlas_id = ?",
        (atlas_id,),
    ).fetchone()
    expected_config = (
        settings.general.atlas_similarity_threshold,
        FP_TYPE,
        json.dumps(FP_PARAMS, sort_keys=True),
        settings.general.atlas_partition_count,
    )
    if atlas_config is None or tuple(atlas_config) != expected_config:
        raise ValueError("atlas configuration does not match current settings")

    database_max = int(
        db.connection.execute("SELECT COALESCE(MAX(spacehastenid), 0) FROM data").fetchone()[0]
    )
    upper = database_max if through_spacehastenid is None else through_spacehastenid
    if upper > database_max:
        raise ValueError(f"through_spacehastenid {upper} exceeds database maximum {database_max}")

    database_existing = int(
        db.connection.execute(
            "SELECT COUNT(*) FROM data WHERE spacehastenid <= ?",
            (previous.last_spacehastenid,),
        ).fetchone()[0]
    )
    if database_existing != previous.compound_count:
        raise ValueError("database compound count does not match the latest atlas version")
    missing_existing = int(
        db.connection.execute(
            "SELECT COUNT(*) FROM data AS d "
            "LEFT JOIN cluster_atlas_assignments AS a "
            "ON a.atlas_id = ? AND a.spacehastenid = d.spacehastenid "
            "WHERE d.spacehastenid <= ? AND a.spacehastenid IS NULL",
            (atlas_id, previous.last_spacehastenid),
        ).fetchone()[0]
    )
    if missing_existing:
        raise ValueError(
            f"{missing_existing} compounds at or below the atlas watermark are unassigned"
        )
    assigned_existing = int(
        db.connection.execute(
            "SELECT COUNT(*) FROM cluster_atlas_assignments "
            "WHERE atlas_id = ? AND spacehastenid <= ?",
            (atlas_id, previous.last_spacehastenid),
        ).fetchone()[0]
    )
    if assigned_existing != previous.compound_count:
        raise ValueError("atlas assignment count does not match the latest version")
    if upper <= previous.last_spacehastenid:
        LOGGER.info("Atlas %s is already current through %d", atlas_id, upper)
        return previous

    version_number = previous.version + 1
    update_root = workdir.atlas_dir() / "updates" / f"version_{version_number:03d}"
    source = update_root / "source" / "new_compounds.smi.gz"
    new_count = _export_incremental_smiles(
        db,
        source,
        lower_exclusive=previous.last_spacehastenid,
        upper_inclusive=upper,
    )
    if new_count == 0:
        LOGGER.info("No new compounds require atlas assignment")
        return previous

    prefix = command_prefix or _default_atlas_command(settings)
    existing_centroids = _ensure_initial_centroid_index(
        scheduler,
        settings,
        version=previous,
        command_prefix=prefix,
    )
    artifacts = build_atlas_version(
        scheduler,
        settings,
        input_smiles=source,
        output_root=update_root,
        command_prefix=prefix,
        job_suffix=f"v{version_number}",
        existing_centroids=existing_centroids,
    )

    if artifacts.new_centroid_files:
        if len(artifacts.new_centroid_files) != 1:
            raise ValueError("incremental atlas produced multiple centroid bundles")
        new_centroids = artifacts.new_centroid_files[0]
    else:
        new_centroids = update_root / "empty_centroids.smi.gz"
        if not new_centroids.exists():
            with gzip.open(new_centroids, "wt", encoding="utf-8"):
                pass
    combined_centroids = _combine_centroid_versions(
        scheduler,
        settings,
        existing_centroids=existing_centroids,
        new_centroids=new_centroids,
        output_root=update_root,
        command_prefix=prefix,
        job_suffix=f"v{version_number}",
    )
    version = _ingest_incremental_atlas(
        db,
        atlas_id=atlas_id,
        previous=previous,
        artifacts=artifacts,
        combined_centroids=combined_centroids,
        version_root=update_root,
        upper_spacehastenid=upper,
        similarity_threshold=settings.general.atlas_similarity_threshold,
    )
    LOGGER.info(
        "Updated atlas %s to v%d: +%d compounds, total centroids=%d, watermark=%d",
        atlas_id,
        version.version,
        new_count,
        version.centroid_count,
        version.last_spacehastenid,
    )
    return version


__all__ = [
    "DEFAULT_ATLAS_ID",
    "build_atlas_version",
    "build_initial_seed_atlas",
    "import_initial_seed_atlas",
    "run_centroid_discovery",
    "run_coverage_repair",
    "run_parallel_assignment",
    "update_cluster_atlas",
]
