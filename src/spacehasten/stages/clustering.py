"""Clustering stage — orchestrates one sphere-exclusion clustering pass.

Replaces legacy ``cluster_functions.cluster_dbsh``. The orchestration owns
the SMILES export, scheduler submission, and result ingestion; the actual
RDKit + FPSim2 algorithm runs on the compute node via
:mod:`spacehasten.remote.cluster`.

Layout (matches CODEBASE_REFERENCE.md §A.4 row 6, but rooted in the new
single-root workspace rather than ``$HOME/SPACEHASTEN/``)::

    <workdir>/clustering/
        clustering_input.smi.gz   # streamed from the data table
        clustering.csv            # spacehastenid,clusterid (ingested into DB)
        fp.h5                     # FPSim2 index (left in place for diagnostics)
"""

from __future__ import annotations

import csv
import gzip
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from spacehasten.config.settings import Settings
from spacehasten.core.db import ClusterRow, Database
from spacehasten.scheduler.base import ArrayJob, Scheduler
from spacehasten.workspace.layout import WorkDir

logger = logging.getLogger(__name__)
_OUTPUT_VISIBILITY_TIMEOUT_SECONDS = 60.0


_DEFAULT_CLUSTER_COMMAND: tuple[str, ...] = (
    "python3",
    "-m",
    "spacehasten.remote.cluster",
)
"""Legacy fallback prefix; production uses ``Settings.remote_script_path``."""


def _default_cluster_command(settings: Settings) -> tuple[str, ...]:
    try:
        return ("python3", str(settings.remote_script_path("cluster")))
    except ValueError:
        return _DEFAULT_CLUSTER_COMMAND


def _stream_smiles_to_gzip(
    db: Database,
    out_path: Path,
    *,
    docked_only: bool = False,
    cutoff: float | None = None,
) -> int:
    """Write ``<smiles> <spacehastenid>`` lines to a gzipped file.

    :param docked_only: restrict to compounds with a non-NULL ``dock_score``.
    :param cutoff: restrict to ``dock_score <= cutoff`` (requires
        ``docked_only=True``; validated by the caller).
    :returns: number of compounds written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    query = "SELECT smiles, spacehastenid FROM data"
    params: tuple[float, ...] = ()
    if cutoff is not None:
        query += " WHERE dock_score IS NOT NULL AND dock_score <= ?"
        params = (cutoff,)
    elif docked_only:
        query += " WHERE dock_score IS NOT NULL"
    n = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as w:
        for smiles, sid in db.connection.execute(query, params):
            w.write(f"{smiles} {sid}\n")
            n += 1
    return n


def _ingest_clustering_csv(csv_path: Path) -> list[ClusterRow]:
    """Parse ``clustering.csv`` (header: ``spacehastenid,clusterid``)."""
    rows: list[ClusterRow] = []
    with csv_path.open("rt", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["spacehastenid", "clusterid"]:
            raise ValueError(f"unexpected columns in {csv_path}: {reader.fieldnames!r}")
        for row in reader:
            rows.append(
                ClusterRow(
                    spacehastenid=int(row["spacehastenid"]),
                    clusterid=int(row["clusterid"]),
                )
            )
    return rows


def _wait_for_output(
    path: Path,
    *,
    timeout: float = _OUTPUT_VISIBILITY_TIMEOUT_SECONDS,
    poll_interval: float = 0.5,
) -> bool:
    if path.exists():
        return True
    logger.info("Waiting up to %.0fs for output visibility: %s", timeout, path)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(poll_interval)
        if path.exists():
            return True
    return False


def _build_cluster_command(
    input_smi: Path,
    output_csv: Path,
    cpus: int,
    command_prefix: Sequence[str],
    similarity_threshold: float,
) -> str:
    parts: list[str] = [
        *command_prefix,
        str(input_smi),
        "--output",
        str(output_csv),
        "--processes",
        str(cpus),
        "--similarity-threshold",
        str(similarity_threshold),
    ]
    cmd = " ".join(parts)
    return (
        f'echo "Starting clustering ({input_smi.name}, cpus={cpus})"\n'
        f"{cmd}\n"
        f'echo "Clustering complete"'
    )


def cluster(
    db: Database,
    workdir: WorkDir,
    scheduler: Scheduler,
    settings: Settings,
    *,
    cluster_command_prefix: Sequence[str] | None = None,
    docked_only: bool = False,
    cutoff: float | None = None,
    similarity_threshold: float = 0.3,
) -> int:
    """Run one sphere-exclusion clustering round.

    Streams ``(smiles, spacehastenid)`` rows out of the DB to a gzipped
    SMI file, submits a single CPU-heavy task that invokes
    :mod:`spacehasten.remote.cluster`, and on success replaces the
    ``clusters`` table contents in a single transaction.

    By default every compound in ``data`` is clustered (required for the
    ``--strategy clustering`` acquisition mode, which needs diversity
    information across the whole space). Pass ``docked_only``/``cutoff``
    to restrict clustering to hits only — much faster on large libraries
    when the only goal is populating ``clusterid`` for ``export csv``.

    :param docked_only: cluster only compounds with a non-NULL
        ``dock_score`` (i.e. already docked).
    :param cutoff: further restrict to ``dock_score <= cutoff``. Requires
        ``docked_only=True``.
    :param similarity_threshold: minimum within-cluster Tanimoto similarity.
    :param cluster_command_prefix: command to launch
        ``remote.cluster``. Override in tests with a stub.
    :returns: number of cluster rows ingested.
    :raises ValueError: if ``cutoff`` is given without ``docked_only``.
    :raises RuntimeError: if the clustering job fails on the scheduler.
    """
    if cutoff is not None and not docked_only:
        raise ValueError("cutoff requires docked_only=True")
    if not 0.0 < similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0, 1]")
    if docked_only:
        logger.warning(
            "Clustering a filtered subset (docked_only=%s, cutoff=%s); this "
            "REPLACES the entire clusters table, so any previous full-space "
            "clustering is discarded. Do not use a filtered cluster run if "
            "you rely on `--strategy clustering` for search/dock acquisition "
            "— that strategy needs cluster assignments for the whole space.",
            docked_only,
            cutoff,
        )

    cluster_dir = workdir.clustering_dir()
    input_smi = cluster_dir / "clustering_input.smi.gz"
    output_csv = cluster_dir / "clustering.csv"

    # Clean stale outputs from a previous (failed) run before submitting.
    if output_csv.exists():
        output_csv.unlink()

    n_compounds = _stream_smiles_to_gzip(db, input_smi, docked_only=docked_only, cutoff=cutoff)
    if n_compounds == 0:
        logger.info("no matching compounds; skipping clustering")
        return 0
    logger.info("Wrote %d compounds to %s", n_compounds, input_smi)

    env_setup = [
        line
        for line in (
            settings.general.prepare_anaconda,
            settings.general.activate_clustering,
        )
        if line
    ]

    cpus = max(1, int(settings.general.cpu_count_clustering or 1))
    prefix = (
        cluster_command_prefix
        if cluster_command_prefix is not None
        else _default_cluster_command(settings)
    )
    command = _build_cluster_command(input_smi, output_csv, cpus, prefix, similarity_threshold)

    job = ArrayJob(
        name="clustering",
        workdir=cluster_dir,
        array_size=1,
        max_concurrent=1,
        cpus_per_task=cpus,
        exclusive=True,
        env_setup=env_setup,
        command_template=command,
    )
    handle = scheduler.submit_array(job)
    logger.info("Submitted clustering job %s (%d compounds)", handle.job_id, n_compounds)
    result = scheduler.wait(handle)
    if not result.success:
        from spacehasten.scheduler.diagnostics import tail_logs

        raise RuntimeError(
            f"clustering job {handle.job_id} failed; failed task indices: "
            f"{result.failed_indices}\n"
            f"--- tail of task logs ---\n{tail_logs(handle)}"
        )

    if not _wait_for_output(output_csv):
        from spacehasten.scheduler.diagnostics import tail_logs

        raise FileNotFoundError(
            f"clustering job did not produce {output_csv}\n"
            f"--- tail of task logs ---\n{tail_logs(handle)}"
        )

    rows = _ingest_clustering_csv(output_csv)
    if not rows:
        raise RuntimeError(f"{output_csv} contains no cluster assignments")

    db.replace_clusters(rows)
    db.commit()
    logger.info("Ingested %d cluster assignments", len(rows))
    return len(rows)


__all__ = ["cluster"]
