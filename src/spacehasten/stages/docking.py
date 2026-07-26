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

import csv
import hashlib
import logging
import math
import os
import random
import shutil
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final, Literal

from tqdm import tqdm  # type: ignore[import-untyped]

from spacehasten.config.acquisition import (
    CalibrationConfig,
    PerClusterCapConstraintConfig,
    PortfolioAcquisitionPolicy,
)
from spacehasten.config.settings import Settings
from spacehasten.core.acquisition import (
    AcquisitionSelection,
    DockAcquisition,
    NormalizedPenalty,
    select_normalized_penalized_batch,
    select_penalized_batch,
)
from spacehasten.core.db import (
    AcquisitionBatchRow,
    AcquisitionSelectionRow,
    Database,
    ModelCalibrationRow,
    acquisition_selection_digest,
    canonical_json,
    sha256_hex,
)
from spacehasten.core.portfolio_acquisition import (
    candidate_pool_digest,
    select_portfolio_batch,
)
from spacehasten.scheduler.base import ArrayHandle, ArrayJob, Scheduler
from spacehasten.tools.glide import parse_glide_csv, write_glide_in, write_phase_inp
from spacehasten.workspace.layout import WorkDir

logger = logging.getLogger(__name__)


#: Maximum SMILES per docking task (cf. legacy ``cfg.DOCKING_CHUNK``).
DOCKING_CHUNK: Final[int] = 1000
PENALTY_CLUSTER_SIMILARITY: Final[float] = 0.4
DockStrategy = Literal["greedy", "clustering", "lcb", "ei", "portfolio"]


def _chunked(rows: Sequence[tuple[str, int]], chunk_size: int) -> list[list[tuple[str, int]]]:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    return [list(rows[i : i + chunk_size]) for i in range(0, len(rows), chunk_size)]


def _write_chunk_smi(path: Path, rows: Sequence[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as w:
        for smiles, sid in rows:
            w.write(f"{smiles.strip()} {sid}\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wt", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_portfolio_artifacts(
    dock_dir: Path,
    selections: Sequence[AcquisitionSelectionRow],
    smiles: Sequence[tuple[str, int]],
    batch: AcquisitionBatchRow,
    policy: PortfolioAcquisitionPolicy,
    calibrations: dict[int, ModelCalibrationRow],
) -> None:
    smiles_by_id = {identifier: smiles_value for smiles_value, identifier in smiles}
    fields = [
        "rank",
        "method",
        "batch_id",
        "spacehastenid",
        "smiles",
        "clusterid",
        "model_version",
        "raw_mean",
        "raw_epistemic_std",
        "calibrated_mean",
        "calibrated_std",
        "p_hit",
        "expected_improvement",
        "quality",
        "support_before",
        "support_after",
        "marginal_reward",
        "crowding_penalty",
        "final_utility",
        "cluster_count_before",
        "cap_reached_after",
        "policy_schema_version",
        "policy_sha256",
        "history_attempt_policy",
        "candidate_count",
        "candidate_watermark",
        "candidate_digest",
        "batch_size",
        "selected_count",
        "selection_digest",
        "cluster_atlas_id",
        "cluster_atlas_version",
        "cap_scope",
        "cap_limit",
        "cluster_cap",
        "ei_hit_threshold",
        "ei_xi",
        "calibration_kind",
        "calibration_uncertainty_source",
        "calibration_mean_shift",
        "calibration_std_scale",
        "calibration_std_floor",
        "calibration_artifact_sha256",
    ]
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in selections:
        calibration = calibrations[row.model_version]
        writer.writerow(
            {
                "rank": row.selection_rank,
                "method": "portfolio",
                "batch_id": batch.batch_id,
                "spacehastenid": row.spacehastenid,
                "smiles": smiles_by_id[row.spacehastenid],
                "clusterid": row.clusterid,
                "model_version": row.model_version,
                "raw_mean": row.raw_mean,
                "raw_epistemic_std": row.raw_epistemic_std,
                "calibrated_mean": row.calibrated_mean,
                "calibrated_std": row.calibrated_std,
                "p_hit": row.p_hit,
                "expected_improvement": row.expected_improvement,
                "quality": row.quality,
                "support_before": row.support_before,
                "support_after": row.support_after,
                "marginal_reward": row.marginal_reward,
                "crowding_penalty": row.crowding_penalty,
                "final_utility": row.final_utility,
                "cluster_count_before": row.cluster_count_before,
                "cap_reached_after": int(row.cap_reached_after),
                "policy_schema_version": batch.policy_schema_version,
                "policy_sha256": batch.policy_sha256,
                "history_attempt_policy": batch.history_attempt_policy,
                "candidate_count": batch.candidate_count,
                "candidate_watermark": batch.candidate_watermark,
                "candidate_digest": batch.candidate_digest,
                "batch_size": batch.requested_count,
                "selected_count": batch.selected_count,
                "selection_digest": batch.selection_digest,
                "cluster_atlas_id": batch.atlas_id,
                "cluster_atlas_version": batch.atlas_version,
                "cap_scope": batch.cap_scope or "",
                "cap_limit": batch.cap_limit or "",
                "cluster_cap": batch.cap_limit or "",
                "ei_hit_threshold": policy.quality.hit_threshold,
                "ei_xi": policy.quality.xi,
                "calibration_kind": calibration.calibration_kind,
                "calibration_uncertainty_source": calibration.uncertainty_source,
                "calibration_mean_shift": calibration.mean_shift,
                "calibration_std_scale": calibration.std_scale,
                "calibration_std_floor": calibration.std_floor,
                "calibration_artifact_sha256": calibration.artifact_sha256 or "",
            }
        )
    _atomic_write_text(dock_dir / "acquisition.csv", buffer.getvalue())
    _atomic_write_text(
        dock_dir / "acquisition_policy.json",
        canonical_json(
            {
                "batch": {
                    "batch_id": batch.batch_id,
                    "dock_iteration": batch.dock_iteration,
                    "policy_schema_version": batch.policy_schema_version,
                    "policy_sha256": batch.policy_sha256,
                    "history_attempt_policy": batch.history_attempt_policy,
                    "model_version": batch.model_version,
                    "atlas_id": batch.atlas_id,
                    "atlas_version": batch.atlas_version,
                    "candidate_count": batch.candidate_count,
                    "candidate_watermark": batch.candidate_watermark,
                    "candidate_digest": batch.candidate_digest,
                    "requested_count": batch.requested_count,
                    "selected_count": batch.selected_count,
                    "selection_digest": batch.selection_digest,
                    "cap_scope": batch.cap_scope,
                    "cap_limit": batch.cap_limit,
                },
                "policy": policy.model_dump(mode="json"),
                "calibrations": {
                    str(version): calibration.__dict__
                    for version, calibration in calibrations.items()
                },
            },
        )
        + "\n",
    )


def _write_acquisition_csv(
    path: Path,
    selections: Sequence[AcquisitionSelection],
    *,
    method: DockAcquisition,
    lcb_beta: float,
    ei_hit_threshold: float | None,
    ei_xi: float,
    cluster_lambda: float,
    candidate_count: int,
    batch_size: int,
    atlas_id: str | None,
    atlas_version: int | None,
    cluster_cap: int | None,
    normalized_penalty: NormalizedPenalty | None,
) -> None:
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "method",
                "spacehastenid",
                "smiles",
                "model_version",
                "pred_score",
                "epistemic_std",
                "base_score",
                "clusterid",
                "cluster_count_before",
                "cluster_penalty",
                "penalized_score",
                "lcb_beta",
                "ei_hit_threshold",
                "ei_xi",
                "cluster_lambda",
                "cluster_similarity_threshold",
                "candidate_count",
                "batch_size",
                "cluster_atlas_id",
                "cluster_atlas_version",
                "cluster_cap",
                "cluster_alpha",
                "frontier_start_rank",
                "frontier_stop_rank",
                "frontier_q10",
                "frontier_q90",
                "frontier_scale",
            ]
        )
        for rank, selection in enumerate(selections, start=1):
            candidate = selection.candidate
            writer.writerow(
                [
                    rank,
                    method,
                    candidate.spacehastenid,
                    candidate.smiles,
                    candidate.model_version,
                    candidate.pred_score,
                    candidate.epistemic_std,
                    selection.base_score,
                    candidate.clusterid if candidate.clusterid is not None else "",
                    selection.cluster_count_before,
                    selection.cluster_penalty,
                    selection.penalized_score,
                    lcb_beta,
                    ei_hit_threshold if ei_hit_threshold is not None else "",
                    ei_xi,
                    cluster_lambda,
                    (
                        PENALTY_CLUSTER_SIMILARITY
                        if cluster_lambda > 0 or cluster_cap is not None
                        else ""
                    ),
                    candidate_count,
                    batch_size,
                    atlas_id if atlas_id is not None else "",
                    atlas_version if atlas_version is not None else "",
                    cluster_cap if cluster_cap is not None else "",
                    normalized_penalty.cluster_alpha if normalized_penalty is not None else "",
                    (
                        normalized_penalty.frontier_start_rank
                        if normalized_penalty is not None
                        else ""
                    ),
                    (
                        normalized_penalty.frontier_stop_rank
                        if normalized_penalty is not None
                        else ""
                    ),
                    normalized_penalty.frontier_q10 if normalized_penalty is not None else "",
                    normalized_penalty.frontier_q90 if normalized_penalty is not None else "",
                    normalized_penalty.frontier_scale if normalized_penalty is not None else "",
                ]
            )


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
    run_token = dock_dir.parent.parent.name
    user = os.environ.get("USER", "spacehasten")
    scratch_path = f"{scratch_root}/{user}/dock_{run_token}_{iter_token}_chunk_${{TASK_ID}}"
    lines: list[str] = [
        "set -euo pipefail",
        'echo "[task ${TASK_ID}] Starting docking chunk_${TASK_ID}"',
        "check_and_start_jobserver() {",
        "    status=$($SCHRODINGER/jsc local-server-status 2>&1)",
        '    if echo "$status" | grep -q "STOPPED"; then',
        '        echo "Job server is not running. Starting it..."',
        "        $SCHRODINGER/jsc local-server-start",
        "    else",
        '        echo "Job server is already running."',
        "    fi",
        "}",
        "check_and_start_jobserver",
        "curdir=$(pwd)",
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
        "$SCHRODINGER/pipeline -prog phase_db chunk_${TASK_ID}.inp"
        " -OVERWRITE -WAIT -NOJOBID -NJOBS 1",
        'echo "[task ${TASK_ID}] Exporting structures"',
        "$SCHRODINGER/phase_database $(pwd)/chunk_${TASK_ID}.phdb export"
        " -omae $(pwd)/chunk_${TASK_ID} -get 1 -limit 99999999 -WAIT",
        "rm -fr $(pwd)/chunk_${TASK_ID}.phdb",
        'echo "[task ${TASK_ID}] Running Glide docking"',
        "$SCHRODINGER/glide -new -OVERWRITE -WAIT -NJOBS 1"
        " -HOST localhost:1 glide_chunk_${TASK_ID}.in",
        'echo "[task ${TASK_ID}] Packaging results"',
        "rm -f glide_grid.zip",
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
    extract_root = (
        Path(scratch_root) / user / f"COLLECTdock_{dock_dir.parent.parent.name}_{dock_dir.name}"
    )
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    tars = sorted((dock_dir / "results").glob("results-chunk_*.tar.gz"))
    if not tars:
        raise FileNotFoundError(
            f"no results-chunk_*.tar.gz under {dock_dir}; the docking job produced no output"
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
        raise FileNotFoundError(f"no glide_*.csv under {extract_root}; cannot ingest dock scores")
    for csv_path in csvs:
        if csv_path.name.endswith("_skip.csv"):
            continue
        for title, score in parse_glide_csv(csv_path).items():
            prev = results.get(title)
            if prev is None or score < prev:
                results[title] = score
    return results


def _require_current_cluster_atlas(db: Database, atlas_id: str) -> int:
    atlas = db.cluster_atlas(atlas_id)
    if atlas is None:
        raise ValueError(
            f"atlas {atlas_id!r} is not initialized; run `spacehasten atlas init "
            f"--atlas-id {atlas_id} --atlas-root PATH`"
        )
    if not math.isclose(
        atlas.similarity_threshold,
        PENALTY_CLUSTER_SIMILARITY,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"atlas {atlas_id!r} uses similarity threshold "
            f"{atlas.similarity_threshold}, expected {PENALTY_CLUSTER_SIMILARITY}"
        )
    version = db.latest_cluster_atlas_version(atlas_id)
    if version is None:
        raise ValueError(f"atlas {atlas_id!r} has no completed version")
    database_max = db.latest_spacehastenid()
    database_count = db.count_total()
    if version.last_spacehastenid != database_max or version.compound_count != database_count:
        raise ValueError(
            f"atlas {atlas_id!r} is stale at watermark {version.last_spacehastenid} "
            f"for database maximum {database_max}; run `spacehasten atlas update "
            f"--atlas-id {atlas_id}`"
        )
    missing = db.count_missing_cluster_atlas_assignments(atlas_id)
    if missing:
        raise ValueError(
            f"atlas {atlas_id!r} is incomplete: {missing} database compounds "
            "lack assignments; restore or rebuild the atlas"
        )
    return version.version


def dock(
    db: Database,
    workdir: WorkDir,
    scheduler: Scheduler,
    settings: Settings,
    *,
    top_n: int,
    strategy: DockStrategy,
    cpus: int,
    lcb_beta: float = 1.0,
    ei_hit_threshold: float | None = None,
    ei_xi: float = 0.0,
    cluster_lambda: float = 0.0,
    cluster_alpha: float | None = None,
    cluster_cap: int | None = None,
    atlas_id: str | None = None,
    portfolio_policy: PortfolioAcquisitionPolicy | None = None,
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
    :param strategy: docking acquisition method. ``lcb`` and ``ei`` use
        version-matched epistemic uncertainty and optionally apply a dynamic
        within-batch cluster penalty.
    :param cluster_alpha: optional dimensionless EI cluster penalty. The
        corresponding lambda is derived from the live acquisition-score frontier.
    :param cluster_cap: optional maximum selections from any persistent-atlas
        cluster within this batch.
    :param atlas_id: persistent cluster atlas used by cluster penalties or caps.
        The atlas must cover the current database exactly.
    :param cpus: maximum concurrent docking tasks (also the chunk-count
        cap — chunks scale down toward the CPU count when N is small).
    :param dock_command_template: per-task bash body. If ``None``, the
        canonical Schrödinger pipeline (§A.4 row 2) is used. Tests pass
        a stub that emits a synthetic ``results-chunk_<i>.tar.gz``.
    :param seed: optional RNG seed for the pre-chunk shuffle (tests).
    :returns: the new ``dock_iteration`` value.
    :raises RuntimeError: on scheduler failure or empty results.
    :raises ValueError: when no compounds match the acquisition query, or
        when ``strategy='clustering'`` but no cluster assignments exist.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    if cpus < 1:
        raise ValueError(f"cpus must be >= 1, got {cpus}")
    if cluster_alpha is not None and strategy != "ei":
        raise ValueError("cluster_alpha requires EI acquisition")
    if cluster_alpha is not None and cluster_lambda != 0:
        raise ValueError("cluster_alpha and cluster_lambda cannot be used together")
    if cluster_alpha is not None and (not math.isfinite(cluster_alpha) or cluster_alpha < 0):
        raise ValueError(f"cluster_alpha must be finite and non-negative, got {cluster_alpha}")
    if cluster_cap is not None and cluster_cap < 1:
        raise ValueError(f"cluster_cap must be at least 1, got {cluster_cap}")
    if strategy == "clustering" and not db.has_clusters():
        raise ValueError(
            "strategy='clustering' requires cluster assignments, but none exist yet;"
            " run `spacehasten cluster` first (or use"
            " `screening-cycle --strategy clustering`, which clusters automatically)"
        )
    if strategy == "portfolio":
        if portfolio_policy is None:
            raise ValueError("strategy='portfolio' requires a portfolio policy")
        if atlas_id is None:
            raise ValueError("strategy='portfolio' requires a current atlas ID")
        if cluster_lambda != 0 or cluster_alpha is not None or cluster_cap is not None:
            raise ValueError("portfolio policy owns cluster reward, crowding, and cap settings")

    selections: list[AcquisitionSelection] = []
    acquisition_method: DockAcquisition | None = None
    normalized_penalty: NormalizedPenalty | None = None
    effective_cluster_lambda = cluster_lambda
    acquisition_candidate_count = 0
    penalty_atlas_version: int | None = None
    portfolio_batch: AcquisitionBatchRow | None = None
    portfolio_selections: list[AcquisitionSelectionRow] = []
    portfolio_calibrations: dict[int, ModelCalibrationRow] = {}
    portfolio_resume_submitted = False
    if strategy == "portfolio":
        assert portfolio_policy is not None and atlas_id is not None
        latest_data = db.latest_dock_iteration()
        latest_batch = db.latest_acquisition_iteration()
        existing = (
            db.get_acquisition_batch_by_dock_iteration(latest_batch)
            if latest_batch is not None
            else None
        )
        policy_json = canonical_json(portfolio_policy.model_dump(mode="json"))
        if (
            existing is not None
            and existing.strategy == "portfolio"
            and existing.status in {"planned", "submitted", "failed"}
            and (latest_data is None or existing.dock_iteration > latest_data)
        ):
            if existing.atlas_id != atlas_id or existing.policy_sha256 != sha256_hex(policy_json):
                raise ValueError("resume batch atlas or effective policy hash does not match")
            if existing.requested_count != top_n:
                raise ValueError("resume batch requested count does not match top_n")
            iteration = existing.dock_iteration
            portfolio_batch = existing
            portfolio_selections = db.load_acquisition_selections(existing.batch_id)
            if len(portfolio_selections) != top_n:
                raise ValueError("resume batch persisted selection count does not match top_n")
            for version in {row.model_version for row in portfolio_selections}:
                calibration = db.load_model_calibration(version)
                if calibration is None:
                    raise ValueError(f"resume batch has no calibration for model version {version}")
                if calibration.uncertainty_source != portfolio_policy.quality.uncertainty_source:
                    raise ValueError("resume batch calibration uncertainty_source is incompatible")
                portfolio_calibrations[version] = calibration
            rows = db.select_smiles_by_ids([row.spacehastenid for row in portfolio_selections])
            acquisition_candidate_count = existing.candidate_count
            penalty_atlas_version = existing.atlas_version
            portfolio_resume_submitted = existing.status == "submitted"
        else:
            penalty_atlas_version = _require_current_cluster_atlas(db, atlas_id)
            iteration = (
                max(value for value in (latest_data, latest_batch, 0) if value is not None) + 1
            )
            pool = db.select_portfolio_candidate_pool(
                atlas_id,
                exclude_selected_attempts=(
                    portfolio_policy.history.attempt_policy == "once_per_campaign"
                ),
            )
            candidate_digest = candidate_pool_digest(pool)
            versions = {int(version) for version in pool.model_versions}
            if len(versions) != 1:
                raise ValueError("production portfolio batches require exactly one model version")
            for version in versions:
                calibration = db.load_model_calibration(version)
                if calibration is None:
                    raise ValueError(
                        f"portfolio requires a registered calibration for model version {version}"
                    )
                if calibration.uncertainty_source != portfolio_policy.quality.uncertainty_source:
                    raise ValueError(
                        "portfolio calibration uncertainty_source is incompatible with policy"
                    )
                portfolio_calibrations[version] = calibration
            core_calibrations = {
                version: CalibrationConfig(
                    mean_shift=calibration.mean_shift,
                    std_scale=calibration.std_scale,
                    std_floor=calibration.std_floor,
                )
                for version, calibration in portfolio_calibrations.items()
            }
            prior = db.prior_observed_hit_counts(
                atlas_id,
                before_dock_iteration=iteration,
                hit_threshold=portfolio_policy.quality.hit_threshold,
            )
            with tqdm(total=top_n, desc="portfolio selection", unit="compound") as progress:
                selected = select_portfolio_batch(
                    pool,
                    portfolio_policy,
                    core_calibrations,
                    batch_size=top_n,
                    prior_observed_hits=prior,
                    progress=lambda done, total: progress.update(1),
                ).selections
            portfolio_selections = [
                AcquisitionSelectionRow(
                    batch_id="pending",
                    selection_rank=rank,
                    spacehastenid=item.candidate_id,
                    clusterid=item.cluster_id,
                    model_version=item.model_version,
                    raw_mean=item.raw_mean,
                    raw_epistemic_std=item.raw_epistemic_std,
                    calibrated_mean=item.calibrated_mean,
                    calibrated_std=item.calibrated_std,
                    p_hit=item.p_hit,
                    expected_improvement=item.expected_improvement,
                    quality=item.quality,
                    support_before=item.support_before,
                    support_after=item.support_after,
                    marginal_reward=item.marginal_reward,
                    crowding_penalty=item.crowding_penalty,
                    final_utility=item.final_utility,
                    cluster_count_before=item.cluster_count_before,
                    cap_reached_after=bool(item.cap_reached_after),
                    contributions_json=canonical_json(
                        {
                            "quality": item.quality,
                            "marginal_reward": item.marginal_reward,
                            "crowding_penalty": item.crowding_penalty,
                            "final_utility": item.final_utility,
                        }
                    ),
                )
                for rank, item in enumerate(selected, start=1)
            ]
            selection_digest = acquisition_selection_digest(portfolio_selections)
            batch_id = (
                "portfolio-"
                + hashlib.sha256(
                    f"{iteration}:{sha256_hex(policy_json)}:{candidate_digest}:{selection_digest}".encode()
                ).hexdigest()[:24]
            )
            portfolio_selections = [replace(row, batch_id=batch_id) for row in portfolio_selections]
            selection_digest = acquisition_selection_digest(portfolio_selections)
            constraint = portfolio_policy.constraint
            portfolio_batch = db.plan_acquisition_batch(
                AcquisitionBatchRow(
                    batch_id=batch_id,
                    dock_iteration=iteration,
                    strategy="portfolio",
                    status="planned",
                    policy_schema_version=portfolio_policy.schema_version,
                    policy_json=policy_json,
                    policy_sha256=sha256_hex(policy_json),
                    history_attempt_policy=portfolio_policy.history.attempt_policy,
                    model_version=next(iter(versions)),
                    atlas_id=atlas_id,
                    atlas_version=penalty_atlas_version,
                    candidate_count=len(pool.ids),
                    candidate_watermark=db.latest_spacehastenid(),
                    candidate_digest=candidate_digest,
                    requested_count=top_n,
                    selected_count=len(portfolio_selections),
                    selection_digest=selection_digest,
                    cap_scope=(
                        constraint.scope
                        if isinstance(constraint, PerClusterCapConstraintConfig)
                        else None
                    ),
                    cap_limit=(
                        constraint.limit
                        if isinstance(constraint, PerClusterCapConstraintConfig)
                        else None
                    ),
                ),
                portfolio_selections,
            )
            db.commit()
            rows = db.select_smiles_by_ids([row.spacehastenid for row in portfolio_selections])
            acquisition_candidate_count = len(pool.ids)
    elif strategy in {"lcb", "ei"}:
        acquisition_method = strategy
        use_cluster_penalty = (
            cluster_lambda > 0
            or cluster_cap is not None
            or (cluster_alpha is not None and cluster_alpha > 0)
        )
        if use_cluster_penalty:
            if atlas_id is None:
                raise ValueError("a cluster atlas ID is required for a cluster constraint")
            penalty_atlas_version = _require_current_cluster_atlas(db, atlas_id)
        candidates = db.select_uncertainty_docking_candidates(
            atlas_id=atlas_id if use_cluster_penalty else None
        )
        acquisition_candidate_count = len(candidates)
        if cluster_alpha is None:
            selections = select_penalized_batch(
                candidates,
                method=acquisition_method,
                batch_size=top_n,
                cluster_lambda=cluster_lambda,
                cluster_cap=cluster_cap,
                beta=lcb_beta,
                hit_threshold=ei_hit_threshold,
                xi=ei_xi,
            )
        else:
            selections, normalized_penalty = select_normalized_penalized_batch(
                candidates,
                method=acquisition_method,
                batch_size=top_n,
                cluster_alpha=cluster_alpha,
                cluster_cap=cluster_cap,
                beta=lcb_beta,
                hit_threshold=ei_hit_threshold,
                xi=ei_xi,
            )
            effective_cluster_lambda = normalized_penalty.cluster_lambda
            logger.info(
                "EI normalized cluster penalty: alpha=%.6g, frontier ranks=%d-%d, "
                "q10=%.9g, q90=%.9g, scale=%.9g, lambda=%.9g",
                normalized_penalty.cluster_alpha,
                normalized_penalty.frontier_start_rank,
                normalized_penalty.frontier_stop_rank,
                normalized_penalty.frontier_q10,
                normalized_penalty.frontier_q90,
                normalized_penalty.frontier_scale,
                normalized_penalty.cluster_lambda,
            )
        del candidates
        rows = [
            (selection.candidate.smiles, selection.candidate.spacehastenid)
            for selection in selections
        ]
    elif strategy in {"greedy", "clustering"}:
        rows = db.select_compounds_to_dock(strategy, top_n)
    else:
        raise ValueError(f"unknown docking acquisition strategy: {strategy!r}")
    if not rows:
        raise ValueError(
            f"no compounds match the {strategy!r} acquisition query; run prediction first"
        )

    if strategy == "portfolio" and seed is None:
        assert portfolio_batch is not None
        seed = int(portfolio_batch.selection_digest[:16], 16)
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    chunk_size = max(1, round(len(shuffled) / cpus))
    if chunk_size > DOCKING_CHUNK:
        chunk_size = DOCKING_CHUNK
    chunks = _chunked(shuffled, chunk_size)
    n_chunks = len(chunks)

    if strategy != "portfolio":
        latest = db.latest_dock_iteration()
        iteration = 0 if latest is None else latest + 1
    dock_dir = workdir.docking_dir(iteration)
    dock_dir.mkdir(parents=True, exist_ok=True)
    if (
        portfolio_batch is not None
        and portfolio_policy is not None
        and not portfolio_resume_submitted
    ):
        _write_portfolio_artifacts(
            dock_dir,
            portfolio_selections,
            rows,
            portfolio_batch,
            portfolio_policy,
            portfolio_calibrations,
        )
    if selections:
        assert acquisition_method is not None
        _write_acquisition_csv(
            dock_dir / "acquisition.csv",
            selections,
            method=acquisition_method,
            lcb_beta=lcb_beta,
            ei_hit_threshold=ei_hit_threshold,
            ei_xi=ei_xi,
            cluster_lambda=effective_cluster_lambda,
            candidate_count=acquisition_candidate_count,
            batch_size=top_n,
            atlas_id=atlas_id if use_cluster_penalty else None,
            atlas_version=penalty_atlas_version,
            cluster_cap=cluster_cap,
            normalized_penalty=normalized_penalty,
        )
        selected_clusters = Counter(
            selection.candidate.clusterid
            for selection in selections
            if selection.candidate.clusterid is not None
        )
        if selected_clusters:
            logger.info(
                "%s acquisition selected %d compounds across %d clusters "
                "(largest cluster contribution=%d)",
                strategy.upper(),
                len(selections),
                len(selected_clusters),
                max(selected_clusters.values()),
            )
        else:
            logger.info("%s acquisition selected %d compounds", strategy.upper(), len(selections))
    logger.info(
        "Docking iter%d: %d compounds, %d chunks of <=%d (cpus=%d)",
        iteration,
        len(shuffled),
        n_chunks,
        chunk_size,
        cpus,
    )

    # A submitted portfolio batch may be actively using these immutable inputs.
    if portfolio_resume_submitted:
        required = [dock_dir / "glide_grid.zip", dock_dir / "inputs"]
        if not all(path.exists() for path in required):
            raise RuntimeError("submitted portfolio batch is missing persisted docking inputs")
    else:
        grid_blob = db.load_dock_grid()
        (dock_dir / "glide_grid.zip").write_bytes(grid_blob)
        dock_param_blob = db.load_dock_param()

    # Per-chunk files.
    inputs_dir = dock_dir / "inputs"
    if not portfolio_resume_submitted:
        inputs_dir.mkdir(parents=True, exist_ok=True)
        for i, chunk in enumerate(chunks, start=1):
            stem = f"chunk_{i}"
            _write_chunk_smi(inputs_dir / f"{stem}.smi", chunk)
            write_phase_inp(inputs_dir / f"{stem}.inp")
            write_glide_in(inputs_dir / f"glide_{stem}.in", dock_param_blob, ligand_stem=stem)

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
    if portfolio_resume_submitted:
        assert portfolio_batch is not None and portfolio_batch.scheduler_job_id is not None
        handle = ArrayHandle(portfolio_batch.scheduler_job_id, job.name, n_chunks, dock_dir)
    else:
        handle = scheduler.submit_array(job)
    if portfolio_batch is not None and not portfolio_resume_submitted:
        db.update_acquisition_submitted(portfolio_batch.batch_id, handle.job_id)
        db.commit()
    logger.info("Submitted docking job %s (%d tasks)", handle.job_id, n_chunks)
    result = scheduler.wait(handle)
    if not result.success:
        if portfolio_batch is not None:
            db.mark_acquisition_batch_failed(portfolio_batch.batch_id)
            db.commit()
        from spacehasten.scheduler.diagnostics import tail_logs

        raise RuntimeError(
            f"docking job {handle.job_id} failed; failed task indices: "
            f"{result.failed_indices}\n"
            f"--- tail of task logs ---\n{tail_logs(handle)}"
        )

    try:
        extract_root = _extract_results(dock_dir, settings.paths.scratch_default or "/wrk")
        title_to_score = _ingest_dock_results(extract_root)
        shutil.rmtree(extract_root, ignore_errors=True)
    except BaseException:
        if portfolio_batch is not None:
            db.mark_acquisition_batch_failed(portfolio_batch.batch_id)
            db.commit()
        raise
    if not title_to_score:
        if portfolio_batch is not None:
            assert portfolio_policy is not None
            db.finalize_acquisition_outcomes(
                portfolio_batch.batch_id, {}, hit_threshold=portfolio_policy.quality.hit_threshold
            )
            db.commit()
            return iteration
        raise RuntimeError(f"docking iter{iteration}: no scores parsed from {extract_root}")

    update_rows: list[tuple[float, int, int]] = []
    planned_ids = {row.spacehastenid for row in portfolio_selections}
    for title, score in title_to_score.items():
        try:
            sid = int(title)
        except ValueError as exc:
            if portfolio_batch is not None:
                db.mark_acquisition_batch_failed(portfolio_batch.batch_id)
                db.commit()
                raise RuntimeError(
                    f"portfolio docking result has non-integer title {title!r}"
                ) from exc
            logger.warning("skipping non-integer Glide title %r", title)
            continue
        if portfolio_batch is not None and sid not in planned_ids:
            db.mark_acquisition_batch_failed(portfolio_batch.batch_id)
            db.commit()
            raise RuntimeError(f"portfolio docking result has unexpected title {title!r}")
        update_rows.append((float(score), iteration, sid))
    if not update_rows:
        raise RuntimeError(f"docking iter{iteration}: no integer titles in result CSVs")

    db.connection.execute("SAVEPOINT portfolio_ingest")
    try:
        db.apply_dock_scores(update_rows)
        if portfolio_batch is not None and portfolio_policy is not None:
            db.finalize_acquisition_outcomes(
                portfolio_batch.batch_id,
                {sid: (score, "dock") for score, _iteration, sid in update_rows},
                hit_threshold=portfolio_policy.quality.hit_threshold,
            )
        db.connection.execute("RELEASE SAVEPOINT portfolio_ingest")
    except BaseException:
        db.connection.execute("ROLLBACK TO SAVEPOINT portfolio_ingest")
        db.connection.execute("RELEASE SAVEPOINT portfolio_ingest")
        if portfolio_batch is not None:
            db.mark_acquisition_batch_failed(portfolio_batch.batch_id)
        db.commit()
        raise
    db.commit()
    logger.info("Updated dock_score for %d rows (iter=%d)", len(update_rows), iteration)
    return iteration


__all__ = [
    "DOCKING_CHUNK",
    "PENALTY_CLUSTER_SIMILARITY",
    "DockStrategy",
    "dock",
]
