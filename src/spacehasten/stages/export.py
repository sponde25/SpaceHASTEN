"""Export stage — CSV and pose export.

Replaces legacy ``export_functions.export_results`` and
``export_functions.export_poses``.

The CSV exporter is pure-Python: it pulls rows via
:meth:`Database.select_export_rows` and writes them to the requested
output path. Pose export writes a self-contained bash script and
executes it locally in a clean shell (``env -i bash``) so that
``$SCHRODINGER/run`` works correctly. The script untars docking results
on local scratch, runs ``export_poses.py`` in parallel, concatenates the
output, compresses with pigz, and produces the final ``.maegz``.
"""

from __future__ import annotations

import csv
import logging
import os
import subprocess
import textwrap
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

# Matches ``seeds.import_seeds`` CSV defaults (smiles_col, title_col, score_col).
_SEEDS_HEADER: tuple[str, ...] = ("SMILES", "title", "r_i_docking_score")


def export_csv(db: Database, output: Path, *, cutoff: float) -> int:
    """Export ``data`` rows (left-joined with ``clusters``) with ``dock_score <= cutoff``.

    Output columns mirror the legacy CSV produced by
    ``export_functions.export_results``::

        smiles,smilesid,dock_score,pred_score,spacelight,ftrees,
        dock_iteration,clusterid

    The ``smilesid`` column packs the legacy
    ``<smilesid_stripped>/<spacehastenid>`` form so downstream tooling
    keeps its 1:1 mapping back to the database row.

    ``clusters`` is joined with a ``LEFT JOIN``, not an inner join: rows
    are exported even if the compound has never been assigned a
    ``clusterid`` (e.g. the workspace ran with ``--strategy greedy`` and
    ``spacehasten cluster`` was never invoked, or a hit was discovered
    after the last clustering pass). ``clusterid`` is empty in that case
    rather than silently dropping the row.

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


def export_seeds(db: Database, output: Path) -> int:
    """Export the original seed batch (``dock_iteration == 0``) as a CSV,
    ready to feed straight back into ``spacehasten import-seeds --csv``
    (e.g. to seed a new workspace)::

        SMILES,title,r_i_docking_score

    These are exactly the column names/defaults ``seeds.import_seeds``
    expects, so no ``--smiles-col``/``--title-col``/``--score-col``
    overrides are needed on import.

    Unlike :func:`export_csv`, seeds are identified structurally by
    ``dock_iteration == 0`` rather than by a docking-score cutoff — a
    compound is either part of the original seed batch or it isn't,
    regardless of its score. Compounds discovered in later screening
    cycles (``dock_iteration >= 1``) are never included.

    :returns: number of rows written.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = db.select_seed_rows()
    with output.open("wt", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(_SEEDS_HEADER)
        for smiles, smilesid, dock_score in rows:
            writer.writerow([
                (smiles or "").strip(),
                (smilesid or "").strip(),
                dock_score,
            ])
    logger.info("Exported %d seed rows to %s", len(rows), output)
    return len(rows)


def export_poses(
    db: Database,
    workdir: WorkDir,
    output: Path,
    *,
    cutoff: float,
    iteration: int | None = None,
    settings: Settings | None = None,
    scheduler: None = None,  # kept for API compat; unused
) -> Path:
    """Export poses by running a script locally in a clean shell.

    The script uses a clean environment (``env -i bash``) so that
    ``$SCHRODINGER/run`` works correctly without interference from the
    current conda environment.

    Steps performed by the script:
      1. Untars all ``results-*.tar.gz`` in parallel on local scratch.
      2. Runs ``$SCHRODINGER/run export_poses.py`` in parallel for each
         ``*_pv.maegz``.
      3. Concatenates all ``spacehasten_virtual_hits_*.mae`` on scratch.
      4. Compresses with ``pigz`` → ``.maegz``.
      5. Moves the final file to ``output``.

    :param iteration: if given, export only that iteration; otherwise
        exports all iterations.
    :param settings: required for paths (schrodinger, scratch, script).
    :returns: path to the output ``.maegz`` file.
    """
    if settings is None:
        raise ValueError("export_poses requires Settings")
    script = (
        settings.paths.export_poses_script
        or str(settings.remote_script_path("export_poses"))
    )

    # Determine which iterations to export.
    if iteration is not None:
        iterations = [iteration]
    else:
        latest = db.latest_dock_iteration()
        if latest is None:
            raise ValueError("no dock iterations recorded; nothing to export")
        iterations = list(range(0, latest + 1))

    # Collect all docking directories.
    dock_dirs: list[Path] = []
    for it in iterations:
        d = workdir.docking_dir(it)
        if d.is_dir():
            dock_dirs.append(d)
    if not dock_dirs:
        raise FileNotFoundError(
            f"no docking iteration directories found for iterations {iterations}"
        )

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    scratch_root = settings.paths.scratch_default or "/wrk"
    schrodinger_run = settings.paths.schrodinger_run or "$SCHRODINGER/run"
    dbsh_path = workdir.dbsh().resolve()

    # Space-separated list of docking dirs for the bash script.
    dock_dirs_str = " ".join(str(d) for d in dock_dirs)
    iter_label = (
        f"iter{iterations[0]}"
        if len(iterations) == 1
        else f"iter1-{iterations[-1]}"
    )

    # Build a self-contained bash script.
    script_content = textwrap.dedent(f"""\
        #!/bin/bash
        set -e
        SCRATCH="{scratch_root}/$USER/COLLECT_{workdir.name}_{iter_label}"
        DOCK_DIRS=({dock_dirs_str})
        OUTPUT="{output}"
        DBSH="{dbsh_path}"
        CUTOFF="{cutoff}"
        SCHRODINGER_RUN="{schrodinger_run}"
        EXPORT_SCRIPT="{script}"

        rm -fr "$SCRATCH"
        mkdir -p "$SCRATCH"

        echo "Decompressing results to $SCRATCH ..."
        for DOCK_DIR in "${{DOCK_DIRS[@]}}"; do
            ITER_SUBDIR="$SCRATCH/$(basename "$DOCK_DIR")"
            mkdir -p "$ITER_SUBDIR"
            for TAR in "$DOCK_DIR"/results/results-*.tar.gz; do
                [ -f "$TAR" ] && echo "$TAR $ITER_SUBDIR"
            done
        done | xargs -P $(nproc) -I {{}} bash -c 'TAR="${{0%% *}}"; DIR="${{0#* }}"; tar xzf "$TAR" -C "$DIR"' {{}}

        echo "Extracting poses ..."
        find "$SCRATCH" -name '*_pv.maegz' | xargs -P $(nproc) -I {{}} \\
            $SCHRODINGER_RUN "$EXPORT_SCRIPT" {{}} $CUTOFF "$DBSH"

        echo "Concatenating results ..."
        find "$SCRATCH" -name 'spacehasten_virtual_hits_*.mae' | sort | xargs cat > "$SCRATCH/all_hits.mae"

        echo "Compressing with pigz ..."
        pigz -c "$SCRATCH/all_hits.mae" > "$SCRATCH/all_hits.maegz"

        echo "Moving to output ..."
        mv "$SCRATCH/all_hits.maegz" "$OUTPUT"

        rm -fr "$SCRATCH"
        echo "Done: $OUTPUT"
    """)

    export_dir = workdir.export_dir()
    export_dir.mkdir(parents=True, exist_ok=True)
    script_path = export_dir / f"export_poses_{iter_label}.sh"
    script_path.write_text(script_content, encoding="utf-8")
    os.chmod(script_path, 0o755)

    logger.info("Running export_poses script: %s", script_path)

    # Build environment stripped of conda/virtualenv vars so $SCHRODINGER/run
    # works correctly, but keep system PATH, HOME, SCHRODINGER, etc.
    clean_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("CONDA_")
        and k not in ("VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "CONDA_PREFIX")
    }

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(export_dir),
        env=clean_env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("export_poses stderr:\n%s", result.stderr)
        raise RuntimeError(
            f"export_poses script failed (rc={result.returncode}):\n{result.stderr}"
        )

    logger.info("Pose export complete: %s", output)
    return output


__all__ = ["export_csv", "export_poses", "export_seeds"]
