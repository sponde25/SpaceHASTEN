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
from collections.abc import Sequence
from pathlib import Path

from spacehasten.config.settings import Settings
from spacehasten.core.db import ClusterRow, Database
from spacehasten.scheduler.base import ArrayJob, Scheduler
from spacehasten.workspace.layout import WorkDir

logger = logging.getLogger(__name__)


_DEFAULT_CLUSTER_COMMAND: tuple[str, ...] = (
    "python3",
    "-m",
    "spacehasten.remote.cluster",
)


def _stream_smiles_to_gzip(db: Database, out_path: Path) -> int:
    """Write ``<smiles> <spacehastenid>`` lines to a gzipped file.

    Returns the number of compounds written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as w:
        for smiles, sid in db.connection.execute(
            "SELECT smiles, spacehastenid FROM data"
        ):
            w.write(f"{smiles} {sid}\n")
            n += 1
    return n


def _ingest_clustering_csv(csv_path: Path) -> list[ClusterRow]:
    """Parse ``clustering.csv`` (header: ``spacehastenid,clusterid``)."""
    rows: list[ClusterRow] = []
    with csv_path.open("rt", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["spacehastenid", "clusterid"]:
            raise ValueError(
                f"unexpected columns in {csv_path}: {reader.fieldnames!r}"
            )
        for row in reader:
            rows.append(
                ClusterRow(
                    spacehastenid=int(row["spacehastenid"]),
                    clusterid=int(row["clusterid"]),
                )
            )
    return rows


def _build_cluster_command(
    input_smi: Path,
    output_csv: Path,
    cpus: int,
    command_prefix: Sequence[str],
) -> str:
    parts: list[str] = [
        *command_prefix,
        str(input_smi),
        "--output",
        str(output_csv),
        "--processes",
        str(cpus),
    ]
    return " ".join(parts)


def cluster(
    db: Database,
    workdir: WorkDir,
    scheduler: Scheduler,
    settings: Settings,
    *,
    cluster_command_prefix: Sequence[str] = _DEFAULT_CLUSTER_COMMAND,
) -> int:
    """Run one sphere-exclusion clustering round.

    Streams every ``(smiles, spacehastenid)`` row out of the DB to a
    gzipped SMI file, submits a single CPU-heavy task that invokes
    :mod:`spacehasten.remote.cluster`, and on success replaces the
    ``clusters`` table contents in a single transaction.

    :param cluster_command_prefix: command to launch
        ``remote.cluster``. Override in tests with a stub.
    :returns: number of cluster rows ingested.
    :raises RuntimeError: if the clustering job fails on the scheduler.
    """
    cluster_dir = workdir.clustering_dir()
    input_smi = cluster_dir / "clustering_input.smi.gz"
    output_csv = cluster_dir / "clustering.csv"

    # Clean stale outputs from a previous (failed) run before submitting.
    if output_csv.exists():
        output_csv.unlink()

    n_compounds = _stream_smiles_to_gzip(db, input_smi)
    if n_compounds == 0:
        logger.info("data table is empty; skipping clustering")
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
    command = _build_cluster_command(
        input_smi, output_csv, cpus, cluster_command_prefix
    )

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
        raise RuntimeError(
            f"clustering job {handle.job_id} failed; failed task indices: "
            f"{result.failed_indices}"
        )

    if not output_csv.exists():
        raise FileNotFoundError(f"clustering job did not produce {output_csv}")

    rows = _ingest_clustering_csv(output_csv)
    if not rows:
        raise RuntimeError(f"{output_csv} contains no cluster assignments")

    db.replace_clusters(rows)
    db.commit()
    logger.info("Ingested %d cluster assignments", len(rows))
    return len(rows)


__all__ = ["cluster"]
