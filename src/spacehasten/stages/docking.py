"""Docking stage — Schrödinger Phase/LigPrep + Glide via the scheduler.

Replaces legacy ``docking_functions.dock`` and
``docking_functions.process_docking_results``. The orchestrator owns
chunk planning, file generation, scheduler submission, and result
ingestion; the Schrödinger pipeline itself runs on the compute node via
a per-chunk bash body that mirrors CODEBASE_REFERENCE.md §A.4 row 2.

Layout (rooted in the new single-root workspace, replacing
``$HOME/SPACEHASTEN/DOCKING_<name>_iter<N>/``)::

    <workdir>/docking/iter<N>/
        glide_grid.zip                # extracted once from docking_grid blob
        inputs/
            chunk_<i>.smi             # SMILES + spacehastenid title per line
            chunk_<i>.inp             # Phase/LigPrep input
            glide_chunk_<i>.in        # Glide input
        results/
            results-chunk_<i>.tar.gz  # produced by the compute-node task
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal

from spacehasten.config.settings import Settings
from spacehasten.core.db import Database
from spacehasten.scheduler.base import ArrayJob, Scheduler
from spacehasten.tools.glide import parse_glide_csv, write_glide_in, write_phase_inp
from spacehasten.workspace.layout import WorkDir

logger = logging.getLogger(__name__)


#: Maximum SMILES per docking task (cf. legacy ``cfg.DOCKING_CHUNK``).
DOCKING_CHUNK: Final[int] = 1000


def _chunked(
    rows: Sequence[tuple[str, int]], chunk_size: int
) -> list[list[tuple[str, int]]]:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    return [list(rows[i : i + chunk_size]) for i in range(0, len(rows), chunk_size)]


def _write_chunk_smi(path: Path, rows: Sequence[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as w:
        for smiles, sid in rows:
            w.write(f"{smiles.strip()} {sid}\n")


def _build_dock_command_body(settings: Settings, dock_dir: Path) -> str:
    """Render the per-task bash body for one docking chunk.

    Mirrors CODEBASE_REFERENCE.md §A.4 row 2. ``${TASK_ID}`` is
    substituted by the scheduler at run time (1-based chunk index).

    Scratch is taken from ``settings.paths.scratch_default`` and a
    per-task subdirectory under ``$USER/dock_<basename>_<task>``; on
    success ``results-chunk_<task>.tar.gz`` is moved back to ``dock_dir``.
    """
    scratch_root = settings.paths.scratch_default or "/tmp"
    feature_flags = settings.general.schrodinger_feature_flags
    iter_token = dock_dir.name  # e.g. "iter3"
    user = os.environ.get("USER", "spacehasten")
    scratch_path = (
        f"{scratch_root}/{user}/dock_{iter_token}_chunk_${{TASK_ID}}"
    )
    lines: list[str] = [
        'echo "[task ${TASK_ID}] Starting docking chunk_${TASK_ID}"',
        'check_and_start_jobserver() {',
        '    status=$($SCHRODINGER/jsc local-server-status 2>&1)',
        '    if echo \"$status\" | grep -q \"STOPPED\"; then',
        '        echo \"Job server is not running. Starting it...\"',
        '        $SCHRODINGER/jsc local-server-start',
        '    else',
        '        echo \"Job server is already running.\"',
        '    fi',
        '}',
        'check_and_start_jobserver',
        'curdir=$(pwd)',
        f'scratch_dir="{scratch_path}"',
        'rm -fr "$scratch_dir"',
        'mkdir -p "$scratch_dir"',
        'cp inputs/chunk_${TASK_ID}.smi "$scratch_dir/"',
        'cp inputs/chunk_${TASK_ID}.inp "$scratch_dir/"',
        'cp inputs/glide_chunk_${TASK_ID}.in "$scratch_dir/"',
        'cp glide_grid.zip "$scratch_dir/"',
        'cd "$scratch_dir"',
    ]
    if feature_flags:
        lines.append(f'export SCHRODINGER_FEATURE_FLAGS="{feature_flags}"')
    lines += [
        'echo "[task ${TASK_ID}] Building phase database"',
        '$SCHRODINGER/pipeline -prog phase_db chunk_${TASK_ID}.inp'
        ' -OVERWRITE -WAIT -NOJOBID -NJOBS 1',
        'echo "[task ${TASK_ID}] Exporting structures"',
        '$SCHRODINGER/phase_database $(pwd)/chunk_${TASK_ID}.phdb export'
        ' -omae $(pwd)/chunk_${TASK_ID} -get 1 -limit 99999999 -WAIT',
        'rm -fr $(pwd)/chunk_${TASK_ID}.phdb',
        'echo "[task ${TASK_ID}] Running Glide docking"',
        '$SCHRODINGER/glide -new -OVERWRITE -WAIT -NJOBS 1'
        ' -HOST localhost:1 glide_chunk_${TASK_ID}.in',
        'echo "[task ${TASK_ID}] Packaging results"',
        'rm -f glide_grid.zip',
        'mkdir -p "$curdir/results"',
        'tar -czf "$curdir/results/results-chunk_${TASK_ID}.tar.gz" .',
        'cd "$curdir"',
        'rm -fr "$scratch_dir"',
        'echo "[task ${TASK_ID}] Done"',
    ]
    return "\n".join(lines)


def _extract_results(dock_dir: Path, scratch_root: str) -> Path:
    """Untar each ``results-chunk_*.tar.gz`` to scratch in parallel.

    Mirrors legacy ``process_docking_results``: extracts to a fast local
    drive (``/wrk``) instead of NFS, and cleans up after ingestion.
    """
    user = os.environ.get("USER", "spacehasten")
    extract_root = Path(scratch_root) / user / f"COLLECTdock_{dock_dir.parent.parent.name}_{dock_dir.name}"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    tars = sorted((dock_dir / "results").glob("results-chunk_*.tar.gz"))
    if not tars:
        raise FileNotFoundError(
            f"no results-chunk_*.tar.gz under {dock_dir}; the docking job"
            " produced no output"
        )
    # Parallel extraction via xargs, like the legacy multiprocessing approach.
    tar_list = "\n".join(str(t) for t in tars)
    subprocess.run(
        f"echo '{tar_list}' | xargs -P $(nproc) -I {{}} tar xzf {{}} -C {extract_root}",
        shell=True,
        check=True,
    )
    return extract_root


def _ingest_dock_results(extract_root: Path) -> dict[str, float]:
    """Aggregate Glide CSVs from every extracted chunk into ``{title: min}``."""
    results: dict[str, float] = {}
    csvs = list(extract_root.rglob("glide_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"no glide_*.csv under {extract_root}; cannot ingest dock scores"
        )
    for csv_path in csvs:
        if csv_path.name.endswith("_skip.csv"):
            continue
        for title, score in parse_glide_csv(csv_path).items():
            prev = results.get(title)
            if prev is None or score < prev:
                results[title] = score
    return results


def dock(
    db: Database,
    workdir: WorkDir,
    scheduler: Scheduler,
    settings: Settings,
    *,
    top_n: int,
    strategy: Literal["greedy", "clustering"],
    cpus: int,
    dock_command_template: str | None = None,
    seed: int | None = None,
) -> int:
    """Dock the next ``top_n`` compounds and write back ``dock_score``.

    Pulls compounds via :meth:`Database.select_compounds_to_dock`,
    shuffles, chunks by ``min(N/cpus, DOCKING_CHUNK)``, writes per-chunk
    SMI/Phase/Glide files, extracts the grid blob once, and submits an
    array job whose body matches CODEBASE_REFERENCE.md §A.4 row 2. On
    success, parallel-extracts the result tarballs, parses the Glide
    CSVs, and applies ``UPDATE data SET dock_score = ?, dock_iteration =
    ? WHERE spacehastenid = ?`` in a single transaction.

    :param top_n: number of candidates to acquire from the DB.
    :param strategy: acquisition strategy passed straight to
        :meth:`Database.select_compounds_to_dock`.
    :param cpus: maximum concurrent docking tasks (also the chunk-count
        cap — chunks scale down toward the CPU count when N is small).
    :param dock_command_template: per-task bash body. If ``None``, the
        canonical Schrödinger pipeline (§A.4 row 2) is used. Tests pass
        a stub that emits a synthetic ``results-chunk_<i>.tar.gz``.
    :param seed: optional RNG seed for the pre-chunk shuffle (tests).
    :returns: the new ``dock_iteration`` value.
    :raises RuntimeError: on scheduler failure or empty results.
    :raises ValueError: when no compounds match the acquisition query.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    if cpus < 1:
        raise ValueError(f"cpus must be >= 1, got {cpus}")

    rows = db.select_compounds_to_dock(strategy, top_n)
    if not rows:
        raise ValueError(
            f"no compounds match the {strategy!r} acquisition query;"
            " run prediction first"
        )

    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    chunk_size = max(1, round(len(shuffled) / cpus))
    if chunk_size > DOCKING_CHUNK:
        chunk_size = DOCKING_CHUNK
    chunks = _chunked(shuffled, chunk_size)
    n_chunks = len(chunks)

    latest = db.latest_dock_iteration()
    iteration = 0 if latest is None else latest + 1
    dock_dir = workdir.docking_dir(iteration)
    dock_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Docking iter%d: %d compounds, %d chunks of <=%d (cpus=%d)",
        iteration, len(shuffled), n_chunks, chunk_size, cpus,
    )

    # Extract the grid + load the dock_param template once.
    grid_blob = db.load_dock_grid()
    (dock_dir / "glide_grid.zip").write_bytes(grid_blob)
    dock_param_blob = db.load_dock_param()

    # Per-chunk files.
    inputs_dir = dock_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks, start=1):
        stem = f"chunk_{i}"
        _write_chunk_smi(inputs_dir / f"{stem}.smi", chunk)
        write_phase_inp(inputs_dir / f"{stem}.inp")
        write_glide_in(
            inputs_dir / f"glide_{stem}.in",
            dock_param_blob,
            ligand_stem=stem,
        )

    body = (
        dock_command_template
        if dock_command_template is not None
        else _build_dock_command_body(settings, dock_dir)
    )

    job = ArrayJob(
        name=f"dock_iter{iteration}",
        workdir=dock_dir,
        array_size=n_chunks,
        max_concurrent=min(n_chunks, cpus),
        cpus_per_task=int(settings.general.cpu_count_dock or 1),
        export_none=False,
        env_setup=[],
        command_template=body,
    )
    handle = scheduler.submit_array(job)
    logger.info("Submitted docking job %s (%d tasks)", handle.job_id, n_chunks)
    result = scheduler.wait(handle)
    if not result.success:
        from spacehasten.scheduler.diagnostics import tail_logs
        raise RuntimeError(
            f"docking job {handle.job_id} failed; failed task indices: "
            f"{result.failed_indices}\n"
            f"--- tail of task logs ---\n{tail_logs(handle)}"
        )

    extract_root = _extract_results(dock_dir, settings.paths.scratch_default or "/wrk")
    title_to_score = _ingest_dock_results(extract_root)
    shutil.rmtree(extract_root, ignore_errors=True)
    if not title_to_score:
        raise RuntimeError(
            f"docking iter{iteration}: no scores parsed from {extract_root}"
        )

    update_rows: list[tuple[float, int, int]] = []
    for title, score in title_to_score.items():
        try:
            sid = int(title)
        except ValueError:
            logger.warning("skipping non-integer Glide title %r", title)
            continue
        update_rows.append((float(score), iteration, sid))
    if not update_rows:
        raise RuntimeError(
            f"docking iter{iteration}: no integer titles in result CSVs"
        )

    db.apply_dock_scores(update_rows)
    db.commit()
    logger.info(
        "Updated dock_score for %d rows (iter=%d)", len(update_rows), iteration
    )
    return iteration


__all__ = ["DOCKING_CHUNK", "dock"]
