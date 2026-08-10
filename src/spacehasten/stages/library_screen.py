"""Library-screen stage — property-filter + chemprop-score a library store.

Runs one screening campaign over an existing Parquet library store built
by :func:`spacehasten.stages.library_build.library_build`: every chunk is
property-filtered (vectorized, precomputed columns) and chemprop-scored
on a compute node via :mod:`spacehasten.remote.library_infer`; surviving
compounds are selected (top-N, explicit score cutoff, or a default cutoff
derived from the seed docking-score distribution) and inserted into the
existing ``.dbsh`` database with no schema change (see
``docs/plan-library-screening.md`` §D1, §D2, §5.4).

Layout (shared/NFS storage, one directory per invocation)::

    <shared_root>/library_screen/run<K>/
        inputs/  control.param  infer_<i>.parquet (dense symlinks)
        results/ predicted_<chunk_stem>.parquet
        report.json  [report.survivors.csv if --dry-run]
"""

from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from spacehasten.config.properties import PropertyRanges
from spacehasten.config.settings import Settings
from spacehasten.core.db import Database
from spacehasten.scheduler.base import ArrayJob, Scheduler
from spacehasten.stages.library_build import LibraryManifest
from spacehasten.workspace.layout import WorkDir

logger = logging.getLogger(__name__)


_DEFAULT_INFER_COMMAND: tuple[str, ...] = (
    "python3", "-m", "spacehasten.remote.library_infer",
)


def _default_infer_command(settings: Settings) -> tuple[str, ...]:
    try:
        return ("python3", str(settings.remote_script_path("library_infer")))
    except ValueError:
        return _DEFAULT_INFER_COMMAND


def _next_run(workdir: WorkDir) -> int:
    """Pick the first unused ``run<N>`` (1-based) under ``library_screen/``."""
    n = 1
    while workdir.library_screen_dir(n).exists():
        n += 1
    return n


def _write_control_param(path: Path, props: PropertyRanges) -> None:
    """Write the 12-line PC-bounds file directly from a typed ``PropertyRanges``.

    Values are written in the canonical order shared with
    ``stages.simsearch._write_control_param`` / ``remote.prop_filter``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    bounds: list[tuple[float, float]] = [
        (props.mw.min, props.mw.max),
        (props.slogp.min, props.slogp.max),
        (props.hba.min, props.hba.max),
        (props.hbd.min, props.hbd.max),
        (props.rotbonds.min, props.rotbonds.max),
        (props.tpsa.min, props.tpsa.max),
    ]
    with path.open("wt", encoding="utf-8") as w:
        for lo, hi in bounds:
            w.write(f"{lo}\n{hi}\n")


def _prepare_missing_chunks(
    chunk_paths: Sequence[Path], inputs_dir: Path, results_dir: Path,
) -> list[Path]:
    """Symlink chunks lacking a predicted output as dense ``infer_<i>.parquet``.

    A chunk is considered done when ``results/predicted_<chunk_stem>.parquet``
    already exists (resumable — see plan §5.4 step 6). Returns the dense,
    1-based symlink paths (``array_size = len(result)``).
    """
    inputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    for stale in inputs_dir.glob("infer_*.parquet"):
        stale.unlink()

    missing = [
        p for p in chunk_paths
        if not (results_dir / f"predicted_{p.stem}.parquet").exists()
    ]
    links: list[Path] = []
    for i, src in enumerate(missing, start=1):
        link = inputs_dir / f"infer_{i}.parquet"
        link.symlink_to(Path(src).resolve())
        links.append(link)
    return links


def _build_infer_command(
    results_dir: Path,
    model_dir: Path,
    params_path: Path,
    settings: Settings,
    command_prefix: Sequence[str],
    *,
    top_n: int | None,
    cutoff: float | None,
) -> str:
    """Build the per-task bash body: filter + predict one dense chunk.

    The output filename is derived from the *resolved* symlink target's
    stem (not the dense task index) so a re-run of the same directory
    remains resumable even if the missing-chunk set (and therefore the
    dense numbering) changes between invocations.
    """
    g = settings.general
    in_link = Path("inputs") / "infer_${TASK_ID}.parquet"
    selection = (
        f"--top-n-per-chunk {top_n}" if top_n is not None else f"--score-cutoff {cutoff}"
    )
    parts: list[str] = [
        *command_prefix, str(in_link), str(model_dir), "$OUT_PATH",
        "--params", str(params_path), selection,
        "--batch-size", str(g.library_infer_batch_size),
        "--num-workers", str(g.library_infer_num_workers),
        "--accelerator", g.library_infer_accelerator,
        "--devices", g.library_infer_devices,
    ]
    cmd = " ".join(parts)
    return (
        f'CHUNK_STEM=$(basename "$(readlink -f "{in_link}")")\n'
        'CHUNK_STEM="${CHUNK_STEM%.parquet}"\n'
        f'OUT_PATH="{results_dir}/predicted_${{CHUNK_STEM}}.parquet"\n'
        'if [ -f "$OUT_PATH" ]; then\n'
        '  echo "[task ${TASK_ID}] $OUT_PATH already exists, skipping"\n'
        '  exit 0\n'
        'fi\n'
        f'{cmd}\n'
        'echo "[task ${TASK_ID}] wrote $OUT_PATH"\n'
    )


def _ingest_predictions(
    chunk_paths: Sequence[Path], results_dir: Path,
) -> tuple[pd.DataFrame, int]:
    """Concatenate every ``predicted_<chunk_stem>.parquet`` and dedup by reghash.

    :returns: ``(deduped_df, n_predicted)`` where ``n_predicted`` is the
        total row count across all chunk outputs before dedup (i.e. the
        property-filter survivor count, summed).
    """
    frames: list[pd.DataFrame] = []
    for p in chunk_paths:
        out_path = results_dir / f"predicted_{p.stem}.parquet"
        if not out_path.exists():
            raise FileNotFoundError(f"missing predicted output for chunk {p}: {out_path}")
        frames.append(pd.read_parquet(out_path))
    n_predicted = sum(len(f) for f in frames)
    if not frames:
        return pd.DataFrame(columns=["reghash", "smiles", "compound_id", "pred_score"]), 0
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return combined, n_predicted
    combined = (
        combined.sort_values("pred_score")
        .drop_duplicates(subset="reghash", keep="first")
        .reset_index(drop=True)
    )
    return combined, n_predicted


def library_screen(
    db: Database,
    workdir: WorkDir,
    scheduler: Scheduler,
    settings: Settings,
    *,
    library_dir: Path,
    model_version: int,
    props: PropertyRanges,
    top_n: int | None = None,
    score_cutoff: float | None = None,
    top_pct: float = 1.0,
    max_concurrent: int | None = None,
    dry_run: bool = False,
    report_path: Path | None = None,
    infer_command_prefix: Sequence[str] | None = None,
) -> int:
    """Screen a library store and insert high-scoring survivors into ``db``.

    Selection precedence (plan §D2): ``top_n`` > ``score_cutoff`` >
    default (``db.seed_dock_score_percentile(top_pct)``).

    :param library_dir: directory containing ``manifest.json`` + chunks,
        produced by :func:`spacehasten.stages.library_build.library_build`.
    :param model_version: chemprop model version to score with; resolved
        on disk via :meth:`Database.load_model_path`.
    :param props: resolved property-filter ranges (CLI precedence:
        ``--props-toml`` > DB ``properties`` table > ``PropertyRanges()``
        defaults).
    :param top_n: keep the global top-N by ``pred_score`` (ascending —
        lower is better). Mutually exclusive with ``score_cutoff``.
    :param score_cutoff: keep every survivor with ``pred_score <= cutoff``.
        Mutually exclusive with ``top_n``.
    :param top_pct: used only when neither ``top_n`` nor ``score_cutoff``
        is given: cutoff = the top ``top_pct`` percent of seed dock scores.
    :param dry_run: compute the selection but do not insert into ``db``;
        writes the ranked survivor list next to ``report_path`` instead.
    :param report_path: optional JSON report destination.
    :param infer_command_prefix: override the command used to launch
        ``remote.library_infer`` (used by tests with stub scripts).
    :returns: number of compounds inserted (or that would be inserted, if
        ``dry_run``).
    :raises ValueError: if both ``top_n`` and ``score_cutoff`` are given.
    :raises RuntimeError: if neither selector is given and there is no
        seed docking yet, or if the scheduler job fails.
    """
    if top_n is not None and score_cutoff is not None:
        raise ValueError("top_n and score_cutoff are mutually exclusive")

    library_dir = Path(library_dir)
    manifest = LibraryManifest.load(library_dir / "manifest.json")
    manifest.validate()
    chunk_paths = manifest.chunk_paths(library_dir)
    if not chunk_paths:
        raise RuntimeError(f"library store {library_dir} has no chunk files")

    cutoff: float | None
    if top_n is not None:
        cutoff = None
    elif score_cutoff is not None:
        cutoff = score_cutoff
    else:
        cutoff = db.seed_dock_score_percentile(top_pct)
        if cutoff is None:
            raise RuntimeError(
                "no seed docking scores; run seed-training first, or pass"
                " --score-cutoff/--top-n"
            )

    # Materialise the model on disk (handles legacy BLOB fallback).
    bin_path = db.load_model_path(model_version, workdir)
    model_dir = bin_path.parent.parent
    if not (model_dir / "model_0" / "pytorch_model.bin").exists():
        raise RuntimeError(
            f"model resolution mismatch: {bin_path} not under {model_dir}/model_0/"
        )

    run = _next_run(workdir)
    run_dir = workdir.library_screen_dir(run)
    inputs_dir = run_dir / "inputs"
    results_dir = run_dir / "results"
    params_path = inputs_dir / "control.param"
    _write_control_param(params_path, props)

    links = _prepare_missing_chunks(chunk_paths, inputs_dir, results_dir)
    if links:
        logger.info(
            "library-screen run%d: %d/%d chunks pending", run, len(links), len(chunk_paths),
        )
        env_setup = [
            line for line in (
                settings.general.prepare_anaconda,
                settings.general.activate_chemprop,
            ) if line
        ]
        command = _build_infer_command(
            results_dir, model_dir, params_path, settings,
            infer_command_prefix
            if infer_command_prefix is not None
            else _default_infer_command(settings),
            top_n=top_n, cutoff=cutoff,
        )
        job = ArrayJob(
            name=f"library_screen_run{run}",
            workdir=run_dir,
            array_size=len(links),
            max_concurrent=max_concurrent if max_concurrent is not None else len(links),
            cpus_per_task=max(1, int(settings.general.cpu_count_library or 1)),
            env_setup=env_setup,
            command_template=command,
        )
        handle = scheduler.submit_array(job)
        logger.info(
            "Submitted library-screen job %s (%d tasks, model v%d)",
            handle.job_id, len(links), model_version,
        )
        result = scheduler.wait(handle)
        if not result.success:
            from spacehasten.scheduler.diagnostics import tail_logs
            raise RuntimeError(
                f"library-screen job {handle.job_id} failed; failed task indices: "
                f"{result.failed_indices}\n"
                f"--- tail of task logs ---\n{tail_logs(handle)}"
            )
    else:
        logger.info("library-screen run%d: all %d chunks already predicted", run, len(chunk_paths))

    combined, n_predicted = _ingest_predictions(chunk_paths, results_dir)

    if top_n is not None:
        selected = combined.nsmallest(top_n, "pred_score")
    else:
        selected = combined.loc[combined["pred_score"] <= cutoff]
    selected = selected.reset_index(drop=True)
    n_selected = len(selected)

    existing = db.filter_existing_reghashes(selected["reghash"].tolist())
    survivors = selected.loc[~selected["reghash"].isin(existing)].reset_index(drop=True)
    n_deduped_existing = n_selected - len(survivors)

    n_inserted = 0
    if not dry_run:
        for row in survivors.itertuples(index=False):
            db.insert_library_hit(
                reghash=row.reghash,
                smiles=row.smiles,
                smilesid=row.compound_id,
                pred_score=float(row.pred_score),
                pred_version=model_version,
            )
        db.commit()
        n_inserted = len(survivors)
    else:
        n_inserted = len(survivors)  # "would insert" count

    inserted_scores = survivors["pred_score"].astype(float).tolist()
    report = {
        "run": run,
        "n_chunks": len(chunk_paths),
        "n_predicted": n_predicted,
        "n_selected": n_selected,
        "n_deduped_existing": n_deduped_existing,
        "n_inserted": n_inserted,
        "cutoff_used": cutoff,
        "top_n": top_n,
        "model_version": model_version,
        "dry_run": dry_run,
    }
    if inserted_scores:
        report["pred_score_min"] = min(inserted_scores)
        report["pred_score_median"] = statistics.median(inserted_scores)
        report["pred_score_max"] = max(inserted_scores)
    logger.info("library-screen run%d report: %s", run, report)
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if dry_run:
            csv_path = report_path.with_name(report_path.stem + ".survivors.csv")
            survivors.to_csv(csv_path, index=False)

    return n_inserted


__all__ = ["library_screen"]
