"""Export stage — CSV and pose export.

Replaces legacy ``export_functions.export_results`` and
``export_functions.export_poses``.

The CSV exporter is pure-Python: it pulls rows via
:meth:`Database.select_export_rows` and writes them to the requested
output path. Pose export still relies on Schrödinger's ``$SCHRODINGER/run``
to drive the legacy ``export_poses.py`` over each ``*_pv.maegz`` produced
by the docking stage.
"""

from __future__ import annotations

import csv
import glob
import logging
import os
import shutil
import subprocess
from pathlib import Path

from spacehasten.config.settings import Settings
from spacehasten.core.db import Database
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
) -> Path:
    """Export the Schrödinger Maestro pose file for compounds with
    ``dock_score <= cutoff``.

    Walks the relevant docking iteration directory, untars each
    ``results-*.tar.gz``, then invokes ``$SCHRODINGER/run
    <export_poses.py>`` over every ``*_pv.maegz`` to write
    ``spacehasten_virtual_hits_*.mae`` files. They are concatenated to
    ``output``.

    :param iteration: dock iteration to export; defaults to the latest.
    :param settings: required (``settings.paths.export_poses_script``
        and ``settings.paths.schrodinger_run``).
    :returns: path to the concatenated ``output`` Maestro file.
    :raises ValueError: if ``settings.paths.export_poses_script`` is unset.
    """
    if settings is None:
        raise ValueError("export_poses requires Settings")
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

    # Stage extracted poses under a scratch subdirectory inside workdir.
    scratch_root = Path(settings.paths.scratch_default or "/tmp")
    user = os.environ.get("USER", "spacehasten")
    resdir = scratch_root / user / f"COLLECT_{workdir.name}_iter{iter_n}"
    if resdir.exists():
        shutil.rmtree(resdir)
    resdir.mkdir(parents=True, exist_ok=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Untar all chunk results.
    tarballs = sorted(glob.glob(str(dock_dir / "results-*.tar.gz")))
    if not tarballs:
        raise FileNotFoundError(f"no results-*.tar.gz under {dock_dir}")
    for tar_path in tarballs:
        subprocess.run(
            ["tar", "xzf", tar_path, "-C", str(resdir)],
            check=True,
        )

    # Invoke export_poses.py for each *_pv.maegz.
    pv_files = sorted(glob.glob(str(resdir / "*_pv.maegz")))
    if not pv_files:
        shutil.rmtree(resdir, ignore_errors=True)
        raise FileNotFoundError(
            f"no *_pv.maegz produced from {dock_dir}; nothing to export"
        )

    schrodinger_run = settings.paths.schrodinger_run or "$SCHRODINGER/run"
    schrodinger_run_argv = schrodinger_run.split()
    for pv in pv_files:
        argv = [
            *schrodinger_run_argv,
            str(script),
            pv,
            str(cutoff),
            str(workdir.dbsh()),
        ]
        subprocess.run(argv, check=True)

    # Concatenate per-pv hits into ``output``.
    hit_files = sorted(glob.glob(str(resdir / "spacehasten_virtual_hits_*.mae")))
    with output.open("wb") as out_fh:
        for hit in hit_files:
            with open(hit, "rb") as src:
                shutil.copyfileobj(src, out_fh)

    shutil.rmtree(resdir, ignore_errors=True)
    logger.info("Exported %d pose files to %s", len(hit_files), output)
    return output


__all__ = ["export_csv", "export_poses"]
