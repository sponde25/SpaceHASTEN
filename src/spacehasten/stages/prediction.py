"""Prediction stage — chunked chemprop prediction over undocked rows.

Replaces legacy ``prediction_functions.update_predicted_scores``. The
orchestration owns chunking, scheduler submission, and result ingestion;
the actual chemprop inference runs on the compute node via
:mod:`spacehasten.remote.predict`.

Layout (matches CODEBASE_REFERENCE.md §A.4 row 3, but rooted in the new
single-root workspace rather than ``$HOME/SPACEHASTEN/``)::

    <workdir>/simsearch/cycle<N>/PREDICT/
        predict_<i>.csv             # input chunks, columns: smiles, smilesid
        predicted_predict_<i>.csv   # output chunks, columns: smilesid, docking_score
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from spacehasten.config.settings import Settings
from spacehasten.core.db import Database
from spacehasten.scheduler.base import ArrayJob, Scheduler
from spacehasten.workspace.layout import WorkDir

logger = logging.getLogger(__name__)


_DEFAULT_PREDICT_COMMAND: tuple[str, ...] = (
    "python3",
    "-m",
    "spacehasten.remote.predict",
)
"""Legacy fallback prefix; production uses ``Settings.remote_script_path``."""


def _default_predict_command(settings: Settings) -> tuple[str, ...]:
    try:
        return ("python3", str(settings.remote_script_path("predict")))
    except ValueError:
        return _DEFAULT_PREDICT_COMMAND


def _chunk_index_filename(prefix: str, index: int) -> str:
    return f"{prefix}_{index}.csv"


def _write_chunks(
    rows: Iterable[tuple[str, int]],
    predict_dir: Path,
    chunk_size: int,
) -> int:
    """Stream ``(smiles, spacehastenid)`` rows into ``predict_<i>.csv`` chunks.

    Returns the number of chunks written. The first chunk is index 1 to
    match the 1-based ``${TASK_ID}`` convention used by the scheduler.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    predict_dir.mkdir(parents=True, exist_ok=True)
    # Clean any stale chunks from a previous (failed) run.
    for stale in predict_dir.glob("predict_*.csv"):
        stale.unlink()
    for stale in predict_dir.glob("predicted_predict_*.csv"):
        stale.unlink()

    chunk_index = 0
    in_chunk = 0
    fh = None
    for smiles, spacehastenid in rows:
        if in_chunk == 0:
            chunk_index += 1
            csv_path = predict_dir / _chunk_index_filename("predict", chunk_index)
            fh = csv_path.open("wt", encoding="utf-8")
            fh.write("smiles,smilesid\n")
        assert fh is not None
        fh.write(f"{smiles.strip()},{spacehastenid}\n")
        in_chunk += 1
        if in_chunk >= chunk_size:
            fh.close()
            fh = None
            in_chunk = 0
    if fh is not None:
        fh.close()
    return chunk_index


def _build_predict_command(
    predict_dir: Path,
    model_dir: Path,
    settings: Settings,
    command_prefix: Sequence[str],
) -> str:
    """Build the bash command for ``remote.predict``.

    Mirrors ``scheduler_functions.write_predict_scheduler``: each task
    selects its chunk via ``${TASK_ID}`` and writes
    ``predicted_predict_<i>.csv`` next to the input.
    """
    g = settings.general
    in_csv = predict_dir / "predict_${TASK_ID}.csv"
    out_csv = predict_dir / "predicted_predict_${TASK_ID}.csv"
    parts: list[str] = [*command_prefix, str(in_csv), str(model_dir), str(out_csv)]
    parts += [
        "--batch-size", str(g.pred_batch_size),
        "--num-workers", str(g.pred_num_workers),
        "--accelerator", g.pred_accelerator,
        "--devices", g.pred_devices,
    ]
    cmd = " ".join(parts)
    return (
        f'echo "[task ${{TASK_ID}}] Predicting chunk ${{TASK_ID}}"\n'
        f'export OMP_NUM_THREADS=1\n'
        f'{cmd}\n'
        f'echo "[task ${{TASK_ID}}] Prediction done"'
    )


def _ingest_predictions(
    predict_dir: Path, n_chunks: int
) -> list[tuple[float, int]]:
    """Read ``predicted_predict_<i>.csv`` files into ``(score, spacehastenid)`` pairs."""
    rows: list[tuple[float, int]] = []
    for i in range(1, n_chunks + 1):
        out_csv = predict_dir / _chunk_index_filename("predicted_predict", i)
        if not out_csv.exists():
            raise FileNotFoundError(
                f"missing prediction output for chunk {i}: {out_csv}"
            )
        df = pd.read_csv(out_csv)
        if "smilesid" not in df.columns or "docking_score" not in df.columns:
            raise ValueError(
                f"{out_csv}: expected columns 'smilesid' and 'docking_score', "
                f"got {list(df.columns)}"
            )
        for sid, score in zip(df["smilesid"], df["docking_score"], strict=False):
            rows.append((float(score), int(sid)))
    return rows


def predict_undocked(
    db: Database,
    workdir: WorkDir,
    scheduler: Scheduler,
    settings: Settings,
    *,
    model_version: int,
    chunk_size: int | None = None,
    jobs: int | None = None,
    predict_command_prefix: Sequence[str] | None = None,
) -> int:
    """Predict ``pred_score`` for every undocked row and update the DB.

    :param model_version: model version to use; resolved on disk via
        :meth:`Database.load_model_path` (BLOB fallback for legacy
        databases).
    :param chunk_size: rows per CSV chunk; one scheduler array task per
        chunk. Ignored when ``jobs`` is given.
    :param jobs: number of scheduler array tasks to spread the undocked
        rows across; ``chunk_size`` is derived as
        ``ceil(count_undocked_for_prediction() / jobs)``. Takes precedence
        over ``chunk_size`` when set.
    :param predict_command_prefix: command to launch ``remote.predict``.
        Override in tests with a stub script.
    :returns: number of rows whose ``pred_score`` was updated.
    :raises RuntimeError: if the prediction job fails on the scheduler.
    """
    if jobs is not None:
        if jobs < 1:
            raise ValueError(f"jobs must be >= 1, got {jobs}")
        total = db.count_undocked_for_prediction()
        if total == 0:
            logger.info("no undocked rows; skipping prediction")
            return 0
        chunk_size = -(-total // jobs)  # ceil division
    cycle = db.latest_simsearch_cycle()
    if cycle == 0:
        cycle = 1

    # Materialise the model on disk (handles legacy BLOB fallback).
    bin_path = db.load_model_path(model_version, workdir)
    model_dir = bin_path.parent.parent  # <model_dir>/model_0/pytorch_model.bin
    if not (model_dir / "model_0" / "pytorch_model.bin").exists():
        raise RuntimeError(
            f"model resolution mismatch: {bin_path} not under {model_dir}/model_0/"
        )

    if chunk_size is None:
        chunk_size = settings.general.pred_chunk_size

    predict_dir = workdir.simsearch_dir(cycle) / "PREDICT"
    n_chunks = _write_chunks(db.select_undocked_for_prediction(), predict_dir, chunk_size)
    if n_chunks == 0:
        logger.info("no undocked rows; skipping prediction")
        return 0

    logger.info(
        "Wrote %d prediction chunks (chunk_size=%d) to %s",
        n_chunks,
        chunk_size,
        predict_dir,
    )

    env_setup = [
        line
        for line in (
            settings.general.prepare_anaconda,
            settings.general.activate_chemprop,
        )
        if line
    ]

    command = _build_predict_command(
        predict_dir, model_dir, settings,
        predict_command_prefix
        if predict_command_prefix is not None
        else _default_predict_command(settings),
    )

    job = ArrayJob(
        name=f"predict_v{model_version}_cycle{cycle}",
        workdir=predict_dir,
        array_size=n_chunks,
        max_concurrent=n_chunks,
        cpus_per_task=int(settings.general.cpu_count_predict or 1),
        gpus=1,
        env_setup=env_setup,
        command_template=command,
    )
    handle = scheduler.submit_array(job)
    logger.info(
        "Submitted prediction job %s (%d tasks, model v%d)",
        handle.job_id,
        n_chunks,
        model_version,
    )
    result = scheduler.wait(handle)
    if not result.success:
        from spacehasten.scheduler.diagnostics import tail_logs
        raise RuntimeError(
            f"prediction job {handle.job_id} failed; failed task indices: "
            f"{result.failed_indices}\n"
            f"--- tail of task logs ---\n{tail_logs(handle)}"
        )

    pairs = _ingest_predictions(predict_dir, n_chunks)
    if not pairs:
        return 0

    # Single-transaction bulk update.
    update_rows = [(score, model_version, sid) for score, sid in pairs]
    db.apply_pred_scores(update_rows)
    db.commit()
    logger.info("Updated pred_score for %d rows (version=%d)", len(update_rows), model_version)
    return len(update_rows)


__all__ = ["predict_undocked"]
