"""Export stage — CSV and pose export.

Replaces legacy ``export_functions.export_results`` and
``export_functions.export_poses``.

The CSV exporter is pure-Python: it pulls rows via
:meth:`Database.select_export_rows` and writes them to the requested
output path. Pose export submits an exclusive-node SLURM job that
untars docking results on local scratch, runs ``$SCHRODINGER/run
export_poses.py`` in parallel (all cores), concatenates the output,
compresses with pigz, and copies the final ``.maegz`` back to NFS.
"""

from __future__ import annotations

import csv
import logging
import textwrap
from pathlib import Path

from spacehasten.config.settings import Settings
from spacehasten.core.db import Database
from spacehasten.scheduler.base import ArrayJob, Scheduler
from spacehasten.workspace.layout import WorkDir

logger = logging.getLogger(__name__)


_EXPORT_HEADER: tuple[str, ...] = (
    "smiles",
    "smilesid",
    "dock_score",
    "pred_score",
    "spacelight",
    "ftrees",
    "dock_iteration",
    "clusterid",
)


def export_csv(db: Database, output: Path, *, cutoff: float) -> int:
    """Export ``data ⨝ clusters`` rows with ``dock_score <= cutoff``.

    Output columns mirror the legacy CSV produced by
    ``export_functions.export_results``::

        smiles,smilesid,dock_score,pred_score,spacelight,ftrees,
        dock_iteration,clusterid

    The ``smilesid`` column packs the legacy
    ``<smilesid_stripped>/<spacehastenid>`` form so downstream tooling
    keeps its 1:1 mapping back to the database row.

    :param cutoff: ``dock_score <= cutoff`` filter (NULL scores excluded).
    :returns: number of rows written.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = db.select_export_rows(cutoff)
    with output.open("wt", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(_EXPORT_HEADER)
        for r in rows:
            writer.writerow([
                (r.smiles or "").strip(),
                f"{(r.smilesid or '').strip()}/{r.spacehastenid}",
                r.dock_score,
                r.pred_score,
                r.spacelight,
                r.ftrees,
                r.dock_iteration,
                r.clusterid,
            ])
    logger.info("Exported %d rows to %s", len(rows), output)
    return len(rows)


def export_poses(
    db: Database,
    workdir: WorkDir,
    output: Path,
    *,
    cutoff: float,
    iteration: int | None = None,
    settings: Settings | None = None,
    scheduler: Scheduler | None = None,
) -> Path:
    """Export poses via an exclusive-node SLURM job.

    The job runs on a single compute node using all cores:
      1. Untars all ``results-*.tar.gz`` in parallel on local scratch.
      2. Runs ``$SCHRODINGER/run export_poses.py`` in parallel for each
         ``*_pv.maegz``.
      3. Concatenates all ``spacehasten_virtual_hits_*.mae`` on scratch.
      4. Compresses with ``pigz`` → ``.maegz``.
      5. Copies the final file back to NFS ``output``.

    All heavy I/O happens on the compute node's fast local ``/wrk``
    scratch — no intermediate data crosses NFS.

    :param iteration: dock iteration to export; defaults to the latest.
    :param settings: required for paths (schrodinger, scratch, script).
    :param scheduler: the scheduler to submit the job to.
    :returns: path to the output ``.maegz`` file.
    """
    if settings is None:
        raise ValueError("export_poses requires Settings")
    if scheduler is None:
        raise ValueError("export_poses requires a Scheduler")
    script = settings.paths.export_poses_script
    if not script:
        raise ValueError(
            "settings.paths.export_poses_script is unset; point it at the"
            " legacy export_poses.py file"
        )

    iter_n = iteration if iteration is not None else db.latest_dock_iteration()
    if iter_n < 1:
        raise ValueError("no dock iterations recorded; nothing to export")
    dock_dir = workdir.docking_dir(iter_n)
    if not dock_dir.is_dir():
        raise FileNotFoundError(f"docking iteration directory missing: {dock_dir}")

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    scratch_root = settings.paths.scratch_default or "/wrk"
    schrodinger_run = settings.paths.schrodinger_run or "$SCHRODINGER/run"
    dbsh_path = workdir.dbsh().resolve()

    # Build a self-contained bash script that runs on the compute node.
    command_body = textwrap.dedent(f"""\
        set -e
        SCRATCH="{scratch_root}/$USER/COLLECT_{workdir.name}_iter{iter_n}"
        DOCK_DIR="{dock_dir}"
        OUTPUT="{output}"
        DBSH="{dbsh_path}"
        CUTOFF="{cutoff}"
        SCHRODINGER_RUN="{schrodinger_run}"
        EXPORT_SCRIPT="{script}"

        rm -fr "$SCRATCH"
        mkdir -p "$SCRATCH"

        echo "Decompressing results to $SCRATCH ..."
        ls "$DOCK_DIR"/results-*.tar.gz | xargs -P $(nproc) -I {{}} tar xzf {{}} -C "$SCRATCH"

        echo "Extracting poses ..."
        find "$SCRATCH" -name '*_pv.maegz' | xargs -P $(nproc) -I {{}} \\
            $SCHRODINGER_RUN "$EXPORT_SCRIPT" {{}} $CUTOFF "$DBSH"

        echo "Concatenating results ..."
        cat "$SCRATCH"/spacehasten_virtual_hits_*.mae > "$SCRATCH/all_hits.mae"

        echo "Compressing with pigz ..."
        pigz -c "$SCRATCH/all_hits.mae" > "$SCRATCH/all_hits.maegz"

        echo "Copying to output ..."
        cp "$SCRATCH/all_hits.maegz" "$OUTPUT"

        rm -fr "$SCRATCH"
        echo "Done: $OUTPUT"
    """)

    export_dir = workdir.root / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    job = ArrayJob(
        name=f"export_poses_iter{iter_n}",
        workdir=export_dir,
        array_size=1,
        max_concurrent=1,
        cpus_per_task=1,  # irrelevant with exclusive
        exclusive=True,
        env_setup=[],
        command_template=command_body,
    )
    handle = scheduler.submit_array(job)
    logger.info(
        "Submitted export_poses job %s (iteration %d, cutoff %s)",
        handle.job_id, iter_n, cutoff,
    )
    result = scheduler.wait(handle)
    if not result.success:
        from spacehasten.scheduler.diagnostics import tail_logs

        raise RuntimeError(
            f"export_poses job {handle.job_id} failed\n"
            f"--- tail of task logs ---\n{tail_logs(handle)}"
        )

    logger.info("Pose export complete: %s", output)
    return output


__all__ = ["export_csv", "export_poses"]
