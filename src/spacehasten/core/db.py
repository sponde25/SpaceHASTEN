"""Typed sqlite3 wrapper for the SpaceHASTEN ``.dbsh`` schema.

This module is a thin layer over :mod:`sqlite3`. It performs *no* business
logic; every method maps to a single SQL statement (or a tightly-scoped
group such as DROP+CREATE+INSERT). The legacy schema (see
``tests/fixtures/legacy_schema.sql`` and §A.1 of the codebase reference) is
preserved byte-for-byte.

Acquisition SQL strings (see §A.6 of the codebase reference) are stored as
class-level ``_SQL_*`` constants. They are the regression lock — tests in
``tests/unit/test_db_sql_locked.py`` assert byte-equality with §A.6.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Literal

import numpy as np

from spacehasten.core.acquisition import AcquisitionCandidate
from spacehasten.core.portfolio_acquisition import CandidatePool

# ---------------------------------------------------------------------------
# Schema (frozen — must stay byte-identical to tests/fixtures/legacy_schema.sql)
# ---------------------------------------------------------------------------

SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    (
        "CREATE TABLE IF NOT EXISTS data ("
        "spacehastenid INTEGER PRIMARY KEY,"
        "reghash TEXT,"
        "smiles TEXT,"
        "smilesid TEXT,"
        "dock_score REAL,"
        "pred_score REAL,"
        "spacelight REAL,"
        "ftrees REAL,"
        "query INTEGER,"
        "dock_iteration INTEGER,"
        "pred_version INTEGER,"
        "simsearch_cycle INTEGER"
        ")"
    ),
    "CREATE TABLE IF NOT EXISTS docking_param (dock_param BLOB)",
    "CREATE TABLE IF NOT EXISTS docking_grid (dock_grid BLOB)",
    "CREATE TABLE IF NOT EXISTS models (model_version INTEGER UNIQUE,model_tar BLOB)",
    (
        "CREATE TABLE IF NOT EXISTS properties ("
        "property TEXT,is_double INTEGER,min_limit TEXT,max_limit TEXT)"
    ),
    "CREATE TABLE IF NOT EXISTS clusters(spacehastenid INTEGER PRIMARY KEY,clusterid INTEGER)",
    "CREATE INDEX IF NOT EXISTS idx_reghash ON data(reghash)",
)

EXTENSION_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    (
        "CREATE TABLE IF NOT EXISTS predictions ("
        "spacehastenid INTEGER NOT NULL,"
        "model_version INTEGER NOT NULL,"
        "pred_score REAL NOT NULL,"
        "epistemic_std REAL,"
        "aleatoric_std REAL,"
        "total_std REAL,"
        "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "PRIMARY KEY(spacehastenid, model_version)"
        ")"
    ),
    "CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version)",
    (
        "CREATE TABLE IF NOT EXISTS cluster_atlases ("
        "atlas_id TEXT PRIMARY KEY,"
        "similarity_threshold REAL NOT NULL,"
        "fingerprint_type TEXT NOT NULL,"
        "fingerprint_parameters TEXT NOT NULL,"
        "partition_count INTEGER NOT NULL,"
        "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
        ")"
    ),
    (
        "CREATE TABLE IF NOT EXISTS cluster_atlas_versions ("
        "atlas_id TEXT NOT NULL,"
        "version INTEGER NOT NULL,"
        "last_spacehastenid INTEGER NOT NULL,"
        "compound_count INTEGER NOT NULL,"
        "centroid_count INTEGER NOT NULL,"
        "metadata_path TEXT NOT NULL,"
        "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "PRIMARY KEY(atlas_id, version)"
        ")"
    ),
    (
        "CREATE TABLE IF NOT EXISTS cluster_atlas_centroids ("
        "atlas_id TEXT NOT NULL,"
        "clusterid INTEGER NOT NULL,"
        "centroid_spacehastenid INTEGER NOT NULL,"
        "created_version INTEGER NOT NULL,"
        "PRIMARY KEY(atlas_id, clusterid)"
        ")"
    ),
    (
        "CREATE TABLE IF NOT EXISTS cluster_atlas_assignments ("
        "atlas_id TEXT NOT NULL,"
        "spacehastenid INTEGER NOT NULL,"
        "clusterid INTEGER NOT NULL,"
        "centroid_similarity REAL NOT NULL,"
        "assigned_version INTEGER NOT NULL,"
        "PRIMARY KEY(atlas_id, spacehastenid)"
        ")"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_cluster_atlas_assignments_cluster "
        "ON cluster_atlas_assignments(atlas_id, clusterid)"
    ),
    (
        "CREATE TABLE IF NOT EXISTS model_calibrations ("
        "model_version INTEGER PRIMARY KEY,"
        "calibration_kind TEXT NOT NULL,"
        "uncertainty_source TEXT NOT NULL,"
        "mean_shift REAL NOT NULL,"
        "std_scale REAL NOT NULL,"
        "std_floor REAL NOT NULL,"
        "fit_source TEXT,"
        "fit_split_name TEXT,"
        "fit_row_count INTEGER,"
        "split_sha256 TEXT,"
        "artifact_path TEXT,"
        "artifact_sha256 TEXT,"
        "metadata_json TEXT NOT NULL,"
        "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
        ")"
    ),
    (
        "CREATE TABLE IF NOT EXISTS acquisition_batches ("
        "batch_id TEXT PRIMARY KEY,"
        "dock_iteration INTEGER NOT NULL UNIQUE,"
        "strategy TEXT NOT NULL,"
        "status TEXT NOT NULL,"
        "policy_schema_version INTEGER NOT NULL,"
        "policy_json TEXT NOT NULL,"
        "policy_sha256 TEXT NOT NULL,"
        "history_attempt_policy TEXT NOT NULL,"
        "model_version INTEGER NOT NULL,"
        "atlas_id TEXT NOT NULL,"
        "atlas_version INTEGER NOT NULL,"
        "candidate_count INTEGER NOT NULL,"
        "candidate_watermark INTEGER NOT NULL,"
        "candidate_digest TEXT NOT NULL,"
        "requested_count INTEGER NOT NULL,"
        "selected_count INTEGER NOT NULL,"
        "selection_digest TEXT NOT NULL,"
        "cap_scope TEXT,"
        "cap_limit INTEGER,"
        "scheduler_job_id TEXT,"
        "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "submitted_at TEXT,"
        "completed_at TEXT"
        ")"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_acquisition_batches_iteration "
        "ON acquisition_batches(dock_iteration)"
    ),
    (
        "CREATE TABLE IF NOT EXISTS acquisition_selections ("
        "batch_id TEXT NOT NULL,"
        "selection_rank INTEGER NOT NULL,"
        "spacehastenid INTEGER NOT NULL,"
        "clusterid INTEGER NOT NULL,"
        "model_version INTEGER NOT NULL,"
        "raw_mean REAL NOT NULL,"
        "raw_epistemic_std REAL NOT NULL,"
        "calibrated_mean REAL NOT NULL,"
        "calibrated_std REAL NOT NULL,"
        "p_hit REAL NOT NULL,"
        "expected_improvement REAL NOT NULL,"
        "quality REAL NOT NULL,"
        "support_before REAL NOT NULL,"
        "support_after REAL NOT NULL,"
        "marginal_reward REAL NOT NULL,"
        "crowding_penalty REAL NOT NULL,"
        "final_utility REAL NOT NULL,"
        "cluster_count_before INTEGER NOT NULL,"
        "cap_reached_after INTEGER NOT NULL,"
        "contributions_json TEXT NOT NULL,"
        "PRIMARY KEY(batch_id, selection_rank),"
        "UNIQUE(batch_id, spacehastenid),"
        "FOREIGN KEY(batch_id) REFERENCES acquisition_batches(batch_id)"
        ")"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_acquisition_selections_spacehastenid "
        "ON acquisition_selections(spacehastenid)"
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_acquisition_selections_batch_cluster "
        "ON acquisition_selections(batch_id, clusterid)"
    ),
    (
        "CREATE TABLE IF NOT EXISTS acquisition_outcomes ("
        "batch_id TEXT NOT NULL,"
        "spacehastenid INTEGER NOT NULL,"
        "status TEXT NOT NULL,"
        "dock_score REAL,"
        "source TEXT,"
        "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "PRIMARY KEY(batch_id, spacehastenid),"
        "FOREIGN KEY(batch_id, spacehastenid) REFERENCES "
        "acquisition_selections(batch_id, spacehastenid)"
        ")"
    ),
    (
        "CREATE TABLE IF NOT EXISTS acquisition_region_summaries ("
        "batch_id TEXT NOT NULL,"
        "clusterid INTEGER NOT NULL,"
        "prior_observed_hits INTEGER NOT NULL,"
        "selected_count INTEGER NOT NULL,"
        "expected_hit_mass REAL NOT NULL,"
        "scored_count INTEGER NOT NULL,"
        "observed_hits INTEGER NOT NULL,"
        "unresolved_count INTEGER NOT NULL,"
        "cap_reached INTEGER NOT NULL,"
        "PRIMARY KEY(batch_id, clusterid),"
        "FOREIGN KEY(batch_id) REFERENCES acquisition_batches(batch_id)"
        ")"
    ),
)


# ---------------------------------------------------------------------------
# Row dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataRow:
    spacehastenid: int
    reghash: str | None
    smiles: str | None
    smilesid: str | None
    dock_score: float | None
    pred_score: float | None
    spacelight: float | None
    ftrees: float | None
    query: int | None
    dock_iteration: int | None
    pred_version: int | None
    simsearch_cycle: int | None


@dataclass(frozen=True)
class ClusterRow:
    spacehastenid: int
    clusterid: int


@dataclass(frozen=True)
class ClusterAtlasRow:
    atlas_id: str
    similarity_threshold: float
    fingerprint_type: str
    fingerprint_parameters: str
    partition_count: int


@dataclass(frozen=True)
class ClusterAtlasVersionRow:
    atlas_id: str
    version: int
    last_spacehastenid: int
    compound_count: int
    centroid_count: int
    metadata_path: str


@dataclass(frozen=True)
class ClusterAtlasCentroidRow:
    atlas_id: str
    clusterid: int
    centroid_spacehastenid: int
    created_version: int


@dataclass(frozen=True)
class ClusterAtlasAssignmentRow:
    atlas_id: str
    spacehastenid: int
    clusterid: int
    centroid_similarity: float
    assigned_version: int


@dataclass(frozen=True)
class PropertyRow:
    """One row of the ``properties`` table.

    ``min_limit`` / ``max_limit`` are stored as TEXT in the legacy schema and
    cast at read time; we keep them as ``str`` here to round-trip exactly.
    """

    property: str
    is_double: int  # 1 == float, 0 == int
    min_limit: str
    max_limit: str


@dataclass(frozen=True)
class ModelRow:
    model_version: int
    model_tar: bytes


@dataclass(frozen=True)
class PredictionRow:
    spacehastenid: int
    model_version: int
    pred_score: float
    epistemic_std: float | None
    aleatoric_std: float | None
    total_std: float | None
    created_at: str


@dataclass(frozen=True)
class ModelCalibrationRow:
    model_version: int
    calibration_kind: str
    uncertainty_source: str
    mean_shift: float
    std_scale: float
    std_floor: float
    fit_source: str | None
    fit_split_name: str | None
    fit_row_count: int | None
    split_sha256: str | None
    artifact_path: str | None
    artifact_sha256: str | None
    metadata_json: str
    created_at: str | None = None


@dataclass(frozen=True)
class AcquisitionBatchRow:
    batch_id: str
    dock_iteration: int
    strategy: str
    status: Literal["planned", "submitted", "completed", "partial", "failed"]
    policy_schema_version: int
    policy_json: str
    policy_sha256: str
    history_attempt_policy: str
    model_version: int
    atlas_id: str
    atlas_version: int
    candidate_count: int
    candidate_watermark: int
    candidate_digest: str
    requested_count: int
    selected_count: int
    selection_digest: str
    cap_scope: str | None
    cap_limit: int | None
    scheduler_job_id: str | None = None
    created_at: str | None = None
    submitted_at: str | None = None
    completed_at: str | None = None


@dataclass(frozen=True)
class AcquisitionSelectionRow:
    batch_id: str
    selection_rank: int
    spacehastenid: int
    clusterid: int
    model_version: int
    raw_mean: float
    raw_epistemic_std: float
    calibrated_mean: float
    calibrated_std: float
    p_hit: float
    expected_improvement: float
    quality: float
    support_before: float
    support_after: float
    marginal_reward: float
    crowding_penalty: float
    final_utility: float
    cluster_count_before: int
    cap_reached_after: bool
    contributions_json: str


@dataclass(frozen=True)
class AcquisitionOutcomeRow:
    batch_id: str
    spacehastenid: int
    status: Literal["pending", "scored", "unresolved"]
    dock_score: float | None
    source: str | None
    updated_at: str | None = None


@dataclass(frozen=True)
class AcquisitionRegionSummaryRow:
    batch_id: str
    clusterid: int
    prior_observed_hits: int
    selected_count: int
    expected_hit_mass: float
    scored_count: int
    observed_hits: int
    unresolved_count: int
    cap_reached: bool


@dataclass(frozen=True)
class AcquisitionOutcomeUpdate:
    spacehastenid: int
    dock_score: float
    source: str


def canonical_json(value: object) -> str:
    """Serialize JSON deterministically for persisted provenance records."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(text: str) -> str:
    """Return the SHA256 digest for UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def acquisition_selection_digest(selections: Sequence[AcquisitionSelectionRow]) -> str:
    """Hash complete selection diagnostics in deterministic rank order."""
    digest = hashlib.sha256()
    for selection in sorted(selections, key=lambda item: item.selection_rank):
        record = {
            "selection_rank": selection.selection_rank,
            "spacehastenid": selection.spacehastenid,
            "clusterid": selection.clusterid,
            "model_version": selection.model_version,
            "raw_mean": selection.raw_mean,
            "raw_epistemic_std": selection.raw_epistemic_std,
            "calibrated_mean": selection.calibrated_mean,
            "calibrated_std": selection.calibrated_std,
            "p_hit": selection.p_hit,
            "expected_improvement": selection.expected_improvement,
            "quality": selection.quality,
            "support_before": selection.support_before,
            "support_after": selection.support_after,
            "marginal_reward": selection.marginal_reward,
            "crowding_penalty": selection.crowding_penalty,
            "final_utility": selection.final_utility,
            "cluster_count_before": selection.cluster_count_before,
            "cap_reached_after": selection.cap_reached_after,
            "contributions_json": selection.contributions_json,
        }
        digest.update(canonical_json(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class SimsearchCycleStats:
    """Impact summary for :meth:`Database.undo_simsearch_cycle`.

    ``n_hits_docked`` and ``n_hits_used_as_query`` are the guardrail
    counts: if either is non-zero, ``undo_simsearch_cycle`` refuses to
    proceed (see its docstring).
    """

    cycle: int
    n_hits: int
    n_queries: int
    n_hits_docked: int
    n_hits_used_as_query: int


@dataclass(frozen=True)
class ExportRow:
    smiles: str
    spacehastenid: int
    smilesid: str
    dock_score: float
    pred_score: float | None
    spacelight: float | None
    ftrees: float | None
    dock_iteration: int | None
    clusterid: int | None


@dataclass(frozen=True)
class PropertyRanges:
    """Six property ranges, mirroring legacy cfg.py keys.

    Limits are stored as ``str`` to preserve the legacy TEXT storage format
    byte-for-byte. Session 4 introduces a typed pydantic equivalent that
    converts to/from this representation.
    """

    mw: tuple[str, str]
    slogp: tuple[str, str]
    hba: tuple[str, str]
    hbd: tuple[str, str]
    rotbonds: tuple[str, str]
    tpsa: tuple[str, str]

    def to_rows(self) -> list[PropertyRow]:
        return [
            PropertyRow("mw", 1, self.mw[0], self.mw[1]),
            PropertyRow("slogp", 1, self.slogp[0], self.slogp[1]),
            PropertyRow("hba", 0, self.hba[0], self.hba[1]),
            PropertyRow("hbd", 0, self.hbd[0], self.hbd[1]),
            PropertyRow("rotbonds", 0, self.rotbonds[0], self.rotbonds[1]),
            PropertyRow("tpsa", 1, self.tpsa[0], self.tpsa[1]),
        ]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class Database:
    """Long-lived sqlite3 connection wrapper.

    Unlike the legacy code (which opens/closes a connection per call), this
    class keeps a single connection for the lifetime of the instance. Use as
    a context manager or call :meth:`close` explicitly.
    """

    # ----- §A.6 acquisition SQL (preserve verbatim) -----
    _SQL_DOCK_GREEDY: Final[str] = (
        "SELECT smiles, spacehastenid FROM data\n"
        " WHERE dock_score IS NULL\n"
        " ORDER BY pred_score LIMIT ?"
    )
    _SQL_DOCK_CLUSTERING: Final[str] = (
        "SELECT smiles, data.spacehastenid FROM data, clusters\n"
        " WHERE data.spacehastenid = clusters.spacehastenid\n"
        "   AND dock_score IS NULL\n"
        " GROUP BY clusterid ORDER BY MIN(pred_score) LIMIT ?"
    )
    _SQL_SIMSEARCH_DOCKED_GREEDY: Final[str] = (
        "SELECT smiles, spacehastenid FROM data\n"
        " WHERE query IS NULL AND dock_score IS NOT NULL\n"
        " ORDER BY dock_score LIMIT ?"
    )
    _SQL_SIMSEARCH_DOCKED_CLUSTERING: Final[str] = (
        "SELECT smiles, data.spacehastenid FROM data, clusters\n"
        " WHERE data.spacehastenid = clusters.spacehastenid\n"
        "   AND query IS NULL AND dock_score IS NOT NULL\n"
        " GROUP BY clusterid ORDER BY MIN(dock_score) LIMIT ?"
    )
    _SQL_SIMSEARCH_PREDICTED_GREEDY: Final[str] = (
        "SELECT smiles, spacehastenid FROM data\n"
        " WHERE query IS NULL AND pred_score IS NOT NULL AND dock_score IS NULL\n"
        " ORDER BY pred_score LIMIT ?"
    )
    _SQL_SIMSEARCH_PREDICTED_CLUSTERING: Final[str] = (
        "SELECT smiles, data.spacehastenid FROM data, clusters\n"
        " WHERE data.spacehastenid = clusters.spacehastenid\n"
        "   AND query IS NULL AND pred_score IS NOT NULL AND dock_score IS NULL\n"
        " GROUP BY clusterid ORDER BY MIN(pred_score) LIMIT ?"
    )
    _SQL_DOCK_UNCERTAINTY_CANDIDATES: Final[str] = (
        "SELECT d.smiles, d.spacehastenid, p.pred_score, p.epistemic_std,"
        " d.pred_version, NULL AS clusterid\n"
        " FROM data AS d\n"
        " LEFT JOIN predictions AS p"
        " ON p.spacehastenid = d.spacehastenid"
        " AND p.model_version = d.pred_version\n"
        " WHERE d.dock_score IS NULL"
        " AND d.pred_score IS NOT NULL"
        " AND d.pred_version IS NOT NULL\n"
        " ORDER BY d.spacehastenid"
    )
    _SQL_DOCK_ATLAS_UNCERTAINTY_CANDIDATES: Final[str] = (
        "SELECT d.smiles, d.spacehastenid, p.pred_score, p.epistemic_std,"
        " d.pred_version, a.clusterid\n"
        " FROM data AS d\n"
        " LEFT JOIN predictions AS p"
        " ON p.spacehastenid = d.spacehastenid"
        " AND p.model_version = d.pred_version\n"
        " LEFT JOIN cluster_atlas_assignments AS a"
        " ON a.spacehastenid = d.spacehastenid AND a.atlas_id = ?\n"
        " WHERE d.dock_score IS NULL"
        " AND d.pred_score IS NOT NULL"
        " AND d.pred_version IS NOT NULL\n"
        " ORDER BY d.spacehastenid"
    )

    # ----- update SQL (parameterised) -----
    _SQL_UPDATE_DOCK_SCORE: Final[str] = (
        "UPDATE data SET dock_score = ?, dock_iteration = ? WHERE spacehastenid = ?"
    )
    _SQL_UPDATE_PRED_SCORE: Final[str] = (
        "UPDATE data SET pred_score = ?, pred_version = ? WHERE spacehastenid = ?"
    )
    _SQL_UPSERT_PREDICTION: Final[str] = (
        "INSERT INTO predictions("
        "spacehastenid, model_version, pred_score, epistemic_std, aleatoric_std, total_std"
        ") VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(spacehastenid, model_version) DO UPDATE SET "
        "pred_score = excluded.pred_score, "
        "epistemic_std = excluded.epistemic_std, "
        "aleatoric_std = excluded.aleatoric_std, "
        "total_std = excluded.total_std, "
        "created_at = CURRENT_TIMESTAMP"
    )
    _SQL_MARK_AS_QUERY: Final[str] = "UPDATE data SET query = ? WHERE spacehastenid = ?"

    # ----- supporting select SQL -----
    _SQL_SELECT_UNDOCKED: Final[str] = (
        "SELECT smiles, spacehastenid FROM data WHERE dock_score IS NULL"
    )
    _SQL_SELECT_TRAINING: Final[str] = (
        "SELECT smiles, dock_score FROM data\n WHERE dock_score IS NOT NULL AND dock_score < ?"
    )
    _SQL_SELECT_EXPORT: Final[str] = (
        "SELECT smiles, data.spacehastenid, smilesid, dock_score, pred_score,"
        " spacelight, ftrees, dock_iteration, clusterid\n"
        " FROM data LEFT JOIN clusters ON data.spacehastenid = clusters.spacehastenid\n"
        " WHERE dock_score <= ?\n"
        " ORDER BY dock_score"
    )
    _SQL_SELECT_SEEDS: Final[str] = (
        "SELECT smiles, smilesid, dock_score FROM data\n"
        " WHERE dock_iteration = 0\n"
        " ORDER BY spacehastenid"
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection = sqlite3.connect(self.path)

    # ----- context manager -----

    def __enter__(self) -> Database:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def commit(self) -> None:
        self._conn.commit()

    # ----- schema -----

    def create_schema(self) -> None:
        c = self._conn.cursor()
        for stmt in (*SCHEMA_STATEMENTS, *EXTENSION_SCHEMA_STATEMENTS):
            c.execute(stmt)
        self._conn.commit()

    def ensure_extension_schema(self) -> None:
        """Create additive SpaceHASTEN extension tables for an existing database."""
        for stmt in EXTENSION_SCHEMA_STATEMENTS:
            self._conn.execute(stmt)

    # ----- immutable acquisition and calibration history -----

    def register_model_calibration(self, calibration: ModelCalibrationRow) -> ModelCalibrationRow:
        """Store one immutable calibrator, allowing only an exact repeat registration."""
        self.ensure_extension_schema()
        self._validate_model_calibration(calibration)
        columns = (
            "model_version, calibration_kind, uncertainty_source, mean_shift, std_scale, "
            "std_floor, fit_source, fit_split_name, fit_row_count, split_sha256, "
            "artifact_path, artifact_sha256, metadata_json"
        )
        values = (
            calibration.model_version,
            calibration.calibration_kind,
            calibration.uncertainty_source,
            calibration.mean_shift,
            calibration.std_scale,
            calibration.std_floor,
            calibration.fit_source,
            calibration.fit_split_name,
            calibration.fit_row_count,
            calibration.split_sha256,
            calibration.artifact_path,
            calibration.artifact_sha256,
            calibration.metadata_json,
        )
        existing = self._conn.execute(
            f"SELECT {columns} FROM model_calibrations WHERE model_version = ?",
            (calibration.model_version,),
        ).fetchone()
        if existing is not None:
            if tuple(existing) != values:
                raise ValueError(
                    f"model version {calibration.model_version} already has a different calibration"
                )
            loaded = self.load_model_calibration(calibration.model_version)
            assert loaded is not None
            return loaded
        self._conn.execute(
            f"INSERT INTO model_calibrations({columns}) VALUES ({','.join('?' * len(values))})",
            values,
        )
        loaded = self.load_model_calibration(calibration.model_version)
        assert loaded is not None
        return loaded

    def load_model_calibration(self, model_version: int) -> ModelCalibrationRow | None:
        self.ensure_extension_schema()
        row = self._conn.execute(
            "SELECT model_version, calibration_kind, uncertainty_source, mean_shift, std_scale, "
            "std_floor, fit_source, fit_split_name, fit_row_count, split_sha256, artifact_path, "
            "artifact_sha256, metadata_json, created_at FROM model_calibrations "
            "WHERE model_version = ?",
            (model_version,),
        ).fetchone()
        return ModelCalibrationRow(*row) if row is not None else None

    def plan_acquisition_batch(
        self,
        batch: AcquisitionBatchRow,
        selections: Sequence[AcquisitionSelectionRow],
    ) -> AcquisitionBatchRow:
        """Atomically persist an immutable planned batch and its pending outcomes."""
        self.ensure_extension_schema()
        self._validate_acquisition_plan(batch, selections)
        self._conn.execute("SAVEPOINT acquisition_plan")
        try:
            existing = self.get_acquisition_batch_by_dock_iteration(batch.dock_iteration)
            if existing is not None:
                if self._immutable_batch_fields(existing) != self._immutable_batch_fields(batch):
                    raise ValueError(
                        f"dock iteration {batch.dock_iteration} already has a different "
                        "acquisition plan"
                    )
                self._conn.execute("RELEASE SAVEPOINT acquisition_plan")
                return existing
            self._conn.execute(
                "INSERT INTO acquisition_batches("
                "batch_id, dock_iteration, strategy, status, policy_schema_version, policy_json, "
                "policy_sha256, history_attempt_policy, model_version, atlas_id, atlas_version, "
                "candidate_count, candidate_watermark, candidate_digest, requested_count, "
                "selected_count, selection_digest, cap_scope, cap_limit, scheduler_job_id"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    batch.batch_id,
                    batch.dock_iteration,
                    batch.strategy,
                    batch.status,
                    batch.policy_schema_version,
                    batch.policy_json,
                    batch.policy_sha256,
                    batch.history_attempt_policy,
                    batch.model_version,
                    batch.atlas_id,
                    batch.atlas_version,
                    batch.candidate_count,
                    batch.candidate_watermark,
                    batch.candidate_digest,
                    batch.requested_count,
                    batch.selected_count,
                    batch.selection_digest,
                    batch.cap_scope,
                    batch.cap_limit,
                    batch.scheduler_job_id,
                ),
            )
            self._conn.executemany(
                "INSERT INTO acquisition_selections("
                "batch_id, selection_rank, spacehastenid, clusterid, model_version, raw_mean, "
                "raw_epistemic_std, calibrated_mean, calibrated_std, p_hit, "
                "expected_improvement, quality, "
                "support_before, support_after, marginal_reward, crowding_penalty, final_utility, "
                "cluster_count_before, cap_reached_after, contributions_json"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        selection.batch_id,
                        selection.selection_rank,
                        selection.spacehastenid,
                        selection.clusterid,
                        selection.model_version,
                        selection.raw_mean,
                        selection.raw_epistemic_std,
                        selection.calibrated_mean,
                        selection.calibrated_std,
                        selection.p_hit,
                        selection.expected_improvement,
                        selection.quality,
                        selection.support_before,
                        selection.support_after,
                        selection.marginal_reward,
                        selection.crowding_penalty,
                        selection.final_utility,
                        selection.cluster_count_before,
                        int(selection.cap_reached_after),
                        selection.contributions_json,
                    )
                    for selection in selections
                ],
            )
            self._conn.executemany(
                "INSERT INTO acquisition_outcomes(batch_id, spacehastenid, status) "
                "VALUES (?, ?, 'pending')",
                [(batch.batch_id, selection.spacehastenid) for selection in selections],
            )
            self._conn.execute("RELEASE SAVEPOINT acquisition_plan")
        except BaseException:
            self._conn.execute("ROLLBACK TO SAVEPOINT acquisition_plan")
            self._conn.execute("RELEASE SAVEPOINT acquisition_plan")
            raise
        loaded = self.get_acquisition_batch(batch.batch_id)
        assert loaded is not None
        return loaded

    def get_acquisition_batch(self, batch_id: str) -> AcquisitionBatchRow | None:
        self.ensure_extension_schema()
        row = self._conn.execute(
            "SELECT batch_id, dock_iteration, strategy, status, policy_schema_version, "
            "policy_json, policy_sha256, history_attempt_policy, model_version, atlas_id, "
            "atlas_version, candidate_count, candidate_watermark, candidate_digest, "
            "requested_count, selected_count, "
            "selection_digest, cap_scope, cap_limit, scheduler_job_id, created_at, submitted_at, "
            "completed_at FROM acquisition_batches WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        return AcquisitionBatchRow(*row) if row is not None else None

    def get_acquisition_batch_by_dock_iteration(
        self, dock_iteration: int
    ) -> AcquisitionBatchRow | None:
        self.ensure_extension_schema()
        row = self._conn.execute(
            "SELECT batch_id FROM acquisition_batches WHERE dock_iteration = ?", (dock_iteration,)
        ).fetchone()
        return self.get_acquisition_batch(str(row[0])) if row is not None else None

    def load_acquisition_selections(self, batch_id: str) -> list[AcquisitionSelectionRow]:
        self.ensure_extension_schema()
        rows = self._conn.execute(
            "SELECT batch_id, selection_rank, spacehastenid, clusterid, model_version, raw_mean, "
            "raw_epistemic_std, calibrated_mean, calibrated_std, p_hit, "
            "expected_improvement, quality, "
            "support_before, support_after, marginal_reward, crowding_penalty, final_utility, "
            "cluster_count_before, cap_reached_after, contributions_json "
            "FROM acquisition_selections "
            "WHERE batch_id = ? ORDER BY selection_rank",
            (batch_id,),
        ).fetchall()
        return [
            AcquisitionSelectionRow(
                batch_id=row[0],
                selection_rank=row[1],
                spacehastenid=row[2],
                clusterid=row[3],
                model_version=row[4],
                raw_mean=row[5],
                raw_epistemic_std=row[6],
                calibrated_mean=row[7],
                calibrated_std=row[8],
                p_hit=row[9],
                expected_improvement=row[10],
                quality=row[11],
                support_before=row[12],
                support_after=row[13],
                marginal_reward=row[14],
                crowding_penalty=row[15],
                final_utility=row[16],
                cluster_count_before=row[17],
                cap_reached_after=bool(row[18]),
                contributions_json=row[19],
            )
            for row in rows
        ]

    def load_acquisition_outcomes(self, batch_id: str) -> list[AcquisitionOutcomeRow]:
        self.ensure_extension_schema()
        rows = self._conn.execute(
            "SELECT batch_id, spacehastenid, status, dock_score, source, updated_at "
            "FROM acquisition_outcomes WHERE batch_id = ? ORDER BY spacehastenid",
            (batch_id,),
        ).fetchall()
        return [AcquisitionOutcomeRow(*row) for row in rows]

    def update_acquisition_submitted(self, batch_id: str, scheduler_job_id: str) -> None:
        self.ensure_extension_schema()
        batch = self.get_acquisition_batch(batch_id)
        if batch is None:
            raise KeyError(f"no acquisition batch {batch_id!r}")
        if batch.status == "submitted":
            if batch.scheduler_job_id == scheduler_job_id:
                return
            raise ValueError(f"batch {batch_id!r} is already submitted with a different job ID")
        if batch.status not in ("planned", "failed"):
            raise ValueError(f"batch {batch_id!r} cannot be submitted from status {batch.status!r}")
        cursor = self._conn.execute(
            "UPDATE acquisition_batches SET status = 'submitted', scheduler_job_id = ?, "
            "submitted_at = CURRENT_TIMESTAMP, completed_at = NULL WHERE batch_id = ?",
            (scheduler_job_id, batch_id),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"batch {batch_id!r} is not a planned acquisition batch")

    def mark_acquisition_batch_failed(self, batch_id: str) -> None:
        self.ensure_extension_schema()
        batch = self.get_acquisition_batch(batch_id)
        if batch is None:
            raise KeyError(f"no acquisition batch {batch_id!r}")
        if batch.status == "failed":
            return
        cursor = self._conn.execute(
            "UPDATE acquisition_batches SET status = 'failed', completed_at = CURRENT_TIMESTAMP "
            "WHERE batch_id = ? AND status IN ('planned', 'submitted')",
            (batch_id,),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"batch {batch_id!r} cannot be marked failed")

    def finalize_acquisition_outcomes(
        self,
        batch_id: str,
        outcomes: Mapping[int, tuple[float, str]] | Sequence[AcquisitionOutcomeUpdate],
        *,
        hit_threshold: float,
    ) -> list[AcquisitionRegionSummaryRow]:
        """Persist immutable scored outcomes, resolve missing selections, and summarize regions."""
        self.ensure_extension_schema()
        if not math.isfinite(hit_threshold):
            raise ValueError("hit_threshold must be finite")
        updates = self._normalize_outcome_updates(outcomes)
        self._conn.execute("SAVEPOINT acquisition_finalize")
        try:
            batch = self.get_acquisition_batch(batch_id)
            if batch is None:
                raise KeyError(f"no acquisition batch {batch_id!r}")
            planned = {
                int(row[0]): (str(row[1]), row[2], row[3])
                for row in self._conn.execute(
                    "SELECT spacehastenid, status, dock_score, source FROM acquisition_outcomes "
                    "WHERE batch_id = ?",
                    (batch_id,),
                )
            }
            unexpected = set(updates).difference(planned)
            if unexpected:
                raise ValueError(
                    f"outcomes include IDs not planned for batch {batch_id!r}: {sorted(unexpected)}"
                )
            for identifier, update in updates.items():
                status, score, source = planned[identifier]
                if status in ("pending", "unresolved"):
                    self._conn.execute(
                        "UPDATE acquisition_outcomes SET status = 'scored', dock_score = ?, "
                        "source = ?, "
                        "updated_at = CURRENT_TIMESTAMP WHERE batch_id = ? AND spacehastenid = ?",
                        (update.dock_score, update.source, batch_id, identifier),
                    )
                elif status != "scored" or score != update.dock_score or source != update.source:
                    raise ValueError(
                        f"conflicting immutable outcome for spacehastenid {identifier}"
                    )
            absent = set(planned).difference(updates)
            if absent:
                placeholders = ",".join("?" * len(absent))
                self._conn.execute(
                    "UPDATE acquisition_outcomes SET status = 'unresolved', "
                    "updated_at = CURRENT_TIMESTAMP "
                    f"WHERE batch_id = ? AND status = 'pending' "
                    f"AND spacehastenid IN ({placeholders})",
                    (batch_id, *sorted(absent)),
                )
            summaries = self._derive_acquisition_region_summaries(batch, hit_threshold)
            self._conn.execute(
                "UPDATE acquisition_batches SET status = ?, completed_at = CURRENT_TIMESTAMP "
                "WHERE batch_id = ?",
                (
                    "completed"
                    if all(summary.unresolved_count == 0 for summary in summaries)
                    else "partial",
                    batch_id,
                ),
            )
            self._conn.execute("RELEASE SAVEPOINT acquisition_finalize")
        except BaseException:
            self._conn.execute("ROLLBACK TO SAVEPOINT acquisition_finalize")
            self._conn.execute("RELEASE SAVEPOINT acquisition_finalize")
            raise
        return summaries

    def selected_attempt_ids(self, *, before_dock_iteration: int | None = None) -> set[int]:
        self.ensure_extension_schema()
        sql = (
            "SELECT DISTINCT s.spacehastenid FROM acquisition_selections AS s "
            "JOIN acquisition_batches AS b ON b.batch_id = s.batch_id"
        )
        params: tuple[int, ...] = ()
        if before_dock_iteration is not None:
            sql += " WHERE b.dock_iteration < ?"
            params = (before_dock_iteration,)
        return {int(row[0]) for row in self._conn.execute(sql, params)}

    def latest_acquisition_iteration(self) -> int | None:
        """Return the latest planned acquisition iteration, including unresolved batches."""
        self.ensure_extension_schema()
        row = self._conn.execute("SELECT MAX(dock_iteration) FROM acquisition_batches").fetchone()
        return None if row is None or row[0] is None else int(row[0])

    def select_portfolio_candidate_pool(
        self, atlas_id: str, *, exclude_selected_attempts: bool
    ) -> CandidatePool:
        """Load a compact, strict portfolio pool without materializing SMILES objects."""
        self.ensure_extension_schema()
        exclusion = (
            "AND NOT EXISTS (SELECT 1 FROM acquisition_selections s "
            "WHERE s.spacehastenid = d.spacehastenid)"
            if exclude_selected_attempts
            else ""
        )
        coverage_sql = (
            "SELECT COUNT(*), SUM(CASE WHEN p.spacehastenid IS NOT NULL "
            "AND p.epistemic_std IS NOT NULL AND a.spacehastenid IS NOT NULL "
            "AND d.smiles IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM data d LEFT JOIN predictions p ON p.spacehastenid=d.spacehastenid "
            "AND p.model_version=d.pred_version LEFT JOIN cluster_atlas_assignments a "
            "ON a.spacehastenid=d.spacehastenid AND a.atlas_id=? "
            "WHERE d.dock_score IS NULL AND d.pred_version IS NOT NULL " + exclusion
        )
        count, covered = self._conn.execute(coverage_sql, (atlas_id,)).fetchone()
        if not count:
            raise ValueError("portfolio candidate pool is empty")
        if int(covered or 0) != int(count):
            raise ValueError(
                "portfolio candidate pool has missing SMILES, prediction, uncertainty, "
                "or atlas assignment"
            )
        sql = (
            "SELECT d.spacehastenid, p.pred_score, p.epistemic_std, a.clusterid, p.model_version "
            "FROM data d JOIN predictions p ON p.spacehastenid=d.spacehastenid "
            "AND p.model_version=d.pred_version JOIN cluster_atlas_assignments a "
            "ON a.spacehastenid=d.spacehastenid AND a.atlas_id=? "
            "WHERE d.dock_score IS NULL AND d.pred_version IS NOT NULL " + exclusion +
            " ORDER BY d.spacehastenid"
        )
        chunks: list[list[np.ndarray[Any, Any]]] = [[], [], [], [], []]
        cursor = self._conn.execute(sql, (atlas_id,))
        while rows := cursor.fetchmany(100_000):
            for index, values in enumerate(zip(*rows, strict=True)):
                chunks[index].append(
                    np.asarray(values, dtype=(np.float64 if index in (1, 2) else np.int64))
                )
        return CandidatePool(
            ids=np.concatenate(chunks[0]),
            raw_means=np.concatenate(chunks[1]),
            raw_epistemic_stds=np.concatenate(chunks[2]),
            cluster_ids=np.concatenate(chunks[3]),
            model_versions=np.concatenate(chunks[4]),
        )

    def select_smiles_by_ids(self, ids: Sequence[int]) -> list[tuple[str, int]]:
        """Fetch selected SMILES in input order and reject incomplete lookups."""
        if not ids:
            return []
        normalized = [int(identifier) for identifier in ids]
        if len(set(normalized)) != len(normalized):
            raise ValueError("selected IDs must be unique")
        found: dict[int, str] = {}
        for start in range(0, len(normalized), 900):
            requested = normalized[start : start + 900]
            placeholders = ",".join("?" * len(requested))
            rows = self._conn.execute(
                f"SELECT spacehastenid, smiles FROM data WHERE spacehastenid IN ({placeholders})",
                requested,
            ).fetchall()
            for identifier, smiles in rows:
                if smiles is None:
                    raise ValueError(f"selected ID {identifier} has NULL SMILES")
                found[int(identifier)] = str(smiles)
        missing = [identifier for identifier in normalized if identifier not in found]
        if missing:
            raise ValueError(f"selected IDs are missing SMILES: {missing}")
        return [(found[identifier], identifier) for identifier in normalized]

    def prior_observed_hit_counts(
        self, atlas_id: str, *, before_dock_iteration: int, hit_threshold: float
    ) -> dict[int, int]:
        """Count scored historical hits by the cluster stored at selection time."""
        self.ensure_extension_schema()
        rows = self._conn.execute(
            "SELECT s.clusterid, COUNT(*) FROM acquisition_outcomes AS o "
            "JOIN acquisition_selections AS s ON s.batch_id = o.batch_id "
            "AND s.spacehastenid = o.spacehastenid "
            "JOIN acquisition_batches AS b ON b.batch_id = o.batch_id "
            "WHERE b.atlas_id = ? AND b.dock_iteration < ? "
            "AND b.status IN ('completed', 'partial') AND o.status = 'scored' "
            "AND o.dock_score <= ? GROUP BY s.clusterid",
            (atlas_id, before_dock_iteration, hit_threshold),
        ).fetchall()
        return {int(clusterid): int(count) for clusterid, count in rows}

    def load_acquisition_region_summaries(self, batch_id: str) -> list[AcquisitionRegionSummaryRow]:
        self.ensure_extension_schema()
        rows = self._conn.execute(
            "SELECT batch_id, clusterid, prior_observed_hits, selected_count, expected_hit_mass, "
            "scored_count, observed_hits, unresolved_count, cap_reached "
            "FROM acquisition_region_summaries WHERE batch_id = ? ORDER BY clusterid",
            (batch_id,),
        ).fetchall()
        return [
            AcquisitionRegionSummaryRow(
                batch_id=row[0],
                clusterid=row[1],
                prior_observed_hits=row[2],
                selected_count=row[3],
                expected_hit_mass=row[4],
                scored_count=row[5],
                observed_hits=row[6],
                unresolved_count=row[7],
                cap_reached=bool(row[8]),
            )
            for row in rows
        ]

    @staticmethod
    def _normalize_outcome_updates(
        outcomes: Mapping[int, tuple[float, str]] | Sequence[AcquisitionOutcomeUpdate],
    ) -> dict[int, AcquisitionOutcomeUpdate]:
        items = (
            (
                AcquisitionOutcomeUpdate(identifier, score, source)
                for identifier, (score, source) in outcomes.items()
            )
            if isinstance(outcomes, Mapping)
            else iter(outcomes)
        )
        normalized: dict[int, AcquisitionOutcomeUpdate] = {}
        for update in items:
            if update.spacehastenid in normalized:
                raise ValueError(f"duplicate outcome for spacehastenid {update.spacehastenid}")
            if not math.isfinite(update.dock_score):
                raise ValueError("outcome dock scores must be finite")
            normalized[update.spacehastenid] = update
        return normalized

    @staticmethod
    def _validate_acquisition_plan(
        batch: AcquisitionBatchRow, selections: Sequence[AcquisitionSelectionRow]
    ) -> None:
        if batch.status != "planned":
            raise ValueError("new acquisition batches must have planned status")
        if not batch.batch_id or not batch.strategy:
            raise ValueError("batch_id and strategy must be non-empty")
        if batch.dock_iteration < 0:
            raise ValueError("dock_iteration must be non-negative")
        if batch.policy_schema_version < 1:
            raise ValueError("policy_schema_version must be positive")
        if batch.history_attempt_policy not in ("once_per_campaign", "unscored_eligible"):
            raise ValueError("history_attempt_policy is not recognized")
        if batch.model_version < 0 or batch.atlas_version < 0:
            raise ValueError("model_version and atlas_version must be non-negative")
        if not batch.atlas_id or batch.candidate_count < 1 or batch.candidate_watermark < 1:
            raise ValueError("atlas_id, candidate_count, and candidate_watermark must be positive")
        if not batch.candidate_digest:
            raise ValueError("candidate_digest must be non-empty")
        if batch.selected_count != len(selections):
            raise ValueError("selected_count does not match the selections provided")
        if batch.requested_count != batch.selected_count:
            raise ValueError(
                "requested_count must equal selected_count for a complete portfolio plan"
            )
        if batch.requested_count < 1:
            raise ValueError("requested_count must be positive")
        if batch.candidate_count < batch.selected_count:
            raise ValueError("candidate_count cannot be below selected_count")
        if batch.cap_scope is None:
            if batch.cap_limit is not None:
                raise ValueError("cap_limit requires a cap_scope")
        elif batch.cap_scope == "batch":
            if batch.cap_limit is None or batch.cap_limit < 1:
                raise ValueError("batch cap_scope requires a positive cap_limit")
        else:
            raise ValueError("cap_scope is not recognized")
        if sha256_hex(batch.policy_json) != batch.policy_sha256:
            raise ValueError("policy_sha256 does not match policy_json")
        try:
            if canonical_json(json.loads(batch.policy_json)) != batch.policy_json:
                raise ValueError("policy_json must be canonical JSON")
        except json.JSONDecodeError as exc:
            raise ValueError("policy_json must be valid JSON") from exc
        ranks = [selection.selection_rank for selection in selections]
        if sorted(ranks) != list(range(1, len(selections) + 1)):
            raise ValueError("selection ranks must be contiguous and start at one")
        identifiers = [selection.spacehastenid for selection in selections]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("selection spacehastenids must be unique within a batch")
        if any(selection.batch_id != batch.batch_id for selection in selections):
            raise ValueError("all selections must belong to the planned batch")
        for selection in selections:
            try:
                if (
                    canonical_json(json.loads(selection.contributions_json))
                    != selection.contributions_json
                ):
                    raise ValueError("contributions_json must be canonical JSON")
            except json.JSONDecodeError as exc:
                raise ValueError("contributions_json must be valid JSON") from exc
            if (
                selection.spacehastenid < 0
                or selection.clusterid < 0
                or selection.model_version < 0
                or selection.cluster_count_before < 0
            ):
                raise ValueError("selection IDs, model version, and cluster count are invalid")
            values = (
                selection.raw_mean,
                selection.raw_epistemic_std,
                selection.calibrated_mean,
                selection.calibrated_std,
                selection.p_hit,
                selection.expected_improvement,
                selection.quality,
                selection.support_before,
                selection.support_after,
                selection.marginal_reward,
                selection.crowding_penalty,
                selection.final_utility,
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError("selection diagnostics must be finite")
            if selection.raw_epistemic_std < 0 or selection.calibrated_std <= 0:
                raise ValueError("selection uncertainty standard deviations are invalid")
            if not 0.0 <= selection.p_hit <= 1.0:
                raise ValueError("selection p_hit must be in [0, 1]")
            if selection.support_after < selection.support_before:
                raise ValueError("selection support_after cannot be below support_before")
            if not isinstance(selection.cap_reached_after, bool):
                raise ValueError("selection cap_reached_after must be a bool")
        if acquisition_selection_digest(selections) != batch.selection_digest:
            raise ValueError("selection_digest does not match the selections provided")

    @staticmethod
    def _validate_model_calibration(calibration: ModelCalibrationRow) -> None:
        if calibration.model_version < 0:
            raise ValueError("calibration model_version must be non-negative")
        if not calibration.calibration_kind or not calibration.uncertainty_source:
            raise ValueError("calibration kind and uncertainty source must be non-empty")
        if not all(
            math.isfinite(value)
            for value in (calibration.mean_shift, calibration.std_scale, calibration.std_floor)
        ):
            raise ValueError("calibration numeric values must be finite")
        if calibration.std_scale <= 0 or calibration.std_floor < 0:
            raise ValueError("calibration std_scale must be positive and std_floor non-negative")
        if calibration.fit_row_count is not None and calibration.fit_row_count < 0:
            raise ValueError("calibration fit_row_count must be non-negative")
        try:
            if canonical_json(json.loads(calibration.metadata_json)) != calibration.metadata_json:
                raise ValueError("calibration metadata_json must be canonical JSON")
        except json.JSONDecodeError as exc:
            raise ValueError("calibration metadata_json must be valid JSON") from exc

    @staticmethod
    def _immutable_batch_fields(batch: AcquisitionBatchRow) -> tuple[object, ...]:
        return (
            batch.batch_id,
            batch.dock_iteration,
            batch.strategy,
            batch.policy_schema_version,
            batch.policy_json,
            batch.policy_sha256,
            batch.history_attempt_policy,
            batch.model_version,
            batch.atlas_id,
            batch.atlas_version,
            batch.candidate_count,
            batch.candidate_watermark,
            batch.candidate_digest,
            batch.requested_count,
            batch.selected_count,
            batch.selection_digest,
            batch.cap_scope,
            batch.cap_limit,
        )

    def _derive_acquisition_region_summaries(
        self, batch: AcquisitionBatchRow, hit_threshold: float
    ) -> list[AcquisitionRegionSummaryRow]:
        prior = self.prior_observed_hit_counts(
            batch.atlas_id,
            before_dock_iteration=batch.dock_iteration,
            hit_threshold=hit_threshold,
        )
        rows = self._conn.execute(
            "SELECT s.clusterid, COUNT(*), SUM(s.p_hit), "
            "SUM(CASE WHEN o.status = 'scored' THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN o.status = 'scored' AND o.dock_score <= ? THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN o.status = 'unresolved' THEN 1 ELSE 0 END), MAX(s.cap_reached_after) "
            "FROM acquisition_selections AS s JOIN acquisition_outcomes AS o "
            "ON o.batch_id = s.batch_id AND o.spacehastenid = s.spacehastenid "
            "WHERE s.batch_id = ? GROUP BY s.clusterid ORDER BY s.clusterid",
            (hit_threshold, batch.batch_id),
        ).fetchall()
        summaries = [
            AcquisitionRegionSummaryRow(
                batch_id=batch.batch_id,
                clusterid=int(clusterid),
                prior_observed_hits=prior.get(int(clusterid), 0),
                selected_count=int(selected_count),
                expected_hit_mass=float(expected_hit_mass),
                scored_count=int(scored_count),
                observed_hits=int(observed_hits),
                unresolved_count=int(unresolved_count),
                cap_reached=bool(cap_reached),
            )
            for (
                clusterid,
                selected_count,
                expected_hit_mass,
                scored_count,
                observed_hits,
                unresolved_count,
                cap_reached,
            ) in rows
        ]
        self._conn.execute(
            "DELETE FROM acquisition_region_summaries WHERE batch_id = ?", (batch.batch_id,)
        )
        self._conn.executemany(
            "INSERT INTO acquisition_region_summaries("
            "batch_id, clusterid, prior_observed_hits, selected_count, expected_hit_mass, "
            "scored_count, observed_hits, unresolved_count, cap_reached"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    summary.batch_id,
                    summary.clusterid,
                    summary.prior_observed_hits,
                    summary.selected_count,
                    summary.expected_hit_mass,
                    summary.scored_count,
                    summary.observed_hits,
                    summary.unresolved_count,
                    int(summary.cap_reached),
                )
                for summary in summaries
            ],
        )
        return summaries

    # ----- lookups -----

    def reghash_exists(self, reghash: str) -> bool:
        """Return True if a row with this reghash is already in the data table."""
        row = self._conn.execute(
            "SELECT 1 FROM data WHERE reghash = ? LIMIT 1", (reghash,)
        ).fetchone()
        return row is not None

    # ----- inserts -----

    def insert_seed_undocked(self, reghash: str, smiles: str, smilesid: str) -> int:
        c = self._conn.execute(
            "INSERT INTO data(reghash, smiles, smilesid) VALUES (?, ?, ?)",
            (reghash, smiles, smilesid),
        )
        assert c.lastrowid is not None
        return c.lastrowid

    def insert_seed_docked(
        self, reghash: str, smiles: str, smilesid: str, dock_score: float
    ) -> int:
        c = self._conn.execute(
            "INSERT INTO data(reghash, smiles, smilesid, dock_score, dock_iteration)"
            " VALUES (?, ?, ?, ?, 0)",
            (reghash, smiles, smilesid, dock_score),
        )
        assert c.lastrowid is not None
        return c.lastrowid

    def insert_simsearch_hit(
        self,
        reghash: str,
        smiles: str,
        smilesid: str,
        spacelight: float | None,
        ftrees: float | None,
        pred_score: float | None,
        simsearch_cycle: int,
        pred_version: int | None = None,
    ) -> int:
        c = self._conn.execute(
            "INSERT INTO data("
            "reghash, smiles, smilesid, spacelight, ftrees, pred_score, "
            "simsearch_cycle, pred_version"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                reghash,
                smiles,
                smilesid,
                spacelight,
                ftrees,
                pred_score,
                simsearch_cycle,
                pred_version,
            ),
        )
        assert c.lastrowid is not None
        return c.lastrowid

    # ----- updates -----

    def update_dock_score(self, spacehastenid: int, dock_score: float, dock_iteration: int) -> None:
        self._conn.execute(self._SQL_UPDATE_DOCK_SCORE, (dock_score, dock_iteration, spacehastenid))

    def update_pred_score(self, spacehastenid: int, pred_score: float, pred_version: int) -> None:
        self._conn.execute(self._SQL_UPDATE_PRED_SCORE, (pred_score, pred_version, spacehastenid))

    def store_prediction(
        self,
        spacehastenid: int,
        model_version: int,
        pred_score: float,
        epistemic_std: float | None,
        aleatoric_std: float | None,
        total_std: float | None,
    ) -> None:
        """Store one versioned prediction and its uncertainty decomposition."""
        self.ensure_extension_schema()
        self._conn.execute(
            self._SQL_UPSERT_PREDICTION,
            (
                spacehastenid,
                model_version,
                pred_score,
                epistemic_std,
                aleatoric_std,
                total_std,
            ),
        )

    def mark_as_query(self, spacehastenid: int, cycle: int) -> None:
        self._conn.execute(self._SQL_MARK_AS_QUERY, (cycle, spacehastenid))

    # ----- maxima -----

    def latest_model_version(self) -> int | None:
        """Return the highest model version, or None if no models exist."""
        row = self._conn.execute("SELECT MAX(model_version) FROM models").fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def latest_simsearch_cycle(self) -> int:
        row = self._conn.execute("SELECT MAX(simsearch_cycle) FROM data").fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def latest_dock_iteration(self) -> int | None:
        """Return the highest dock_iteration, or None if no rows have been docked."""
        row = self._conn.execute("SELECT MAX(dock_iteration) FROM data").fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def latest_spacehastenid(self) -> int:
        row = self._conn.execute("SELECT COALESCE(MAX(spacehastenid), 0) FROM data").fetchone()
        return int(row[0])

    def latest_search_attempt_cycle(self) -> int | None:
        """Return the highest simsearch cycle number *attempted* so far.

        Unlike :meth:`latest_simsearch_cycle` (which only looks at
        ``simsearch_cycle``, i.e. cycles that actually produced hits),
        this also considers the ``query`` column. ``simsearch()`` marks
        and commits queries *before* running the search job, so a cycle
        whose search job then fails leaves ``query = cycle`` rows but no
        ``simsearch_cycle = cycle`` rows. Taking the max of both columns
        recovers that failed attempt as "the latest cycle" so ``undo
        search`` can target it. Returns ``None`` if no simsearch cycle has
        ever been attempted.
        """
        row = self._conn.execute(
            "SELECT MAX(x) FROM ("
            "SELECT MAX(query) AS x FROM data "
            "UNION ALL SELECT MAX(simsearch_cycle) AS x FROM data"
            ")"
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def simsearch_cycle_stats(self, cycle: int) -> SimsearchCycleStats:
        """Return impact counts for reverting ``cycle`` (see :meth:`undo_simsearch_cycle`)."""
        n_hits = self._conn.execute(
            "SELECT COUNT(*) FROM data WHERE simsearch_cycle = ?", (cycle,)
        ).fetchone()[0]
        n_queries = self._conn.execute(
            "SELECT COUNT(*) FROM data WHERE query = ?", (cycle,)
        ).fetchone()[0]
        n_hits_docked = self._conn.execute(
            "SELECT COUNT(*) FROM data WHERE simsearch_cycle = ? AND dock_score IS NOT NULL",
            (cycle,),
        ).fetchone()[0]
        n_hits_used_as_query = self._conn.execute(
            "SELECT COUNT(*) FROM data WHERE simsearch_cycle = ? AND query IS NOT NULL",
            (cycle,),
        ).fetchone()[0]
        return SimsearchCycleStats(
            cycle=cycle,
            n_hits=int(n_hits),
            n_queries=int(n_queries),
            n_hits_docked=int(n_hits_docked),
            n_hits_used_as_query=int(n_hits_used_as_query),
        )

    def undo_simsearch_cycle(self, cycle: int) -> tuple[int, int]:
        """Revert simsearch ``cycle``: delete its hit compounds, release its query marks.

        Intended for the case where ``simsearch()`` marked queries (and
        committed) but then failed before inserting any hits, permanently
        stranding those compounds behind ``query IS NOT NULL``. Deletes
        every row with ``simsearch_cycle = cycle`` (plus any stale
        ``clusters`` rows for those ids) and resets ``query = NULL`` for
        every row with ``query = cycle``, in a single transaction.

        :raises ValueError: if any hit compound from ``cycle`` has
            already been docked (real docking data is never silently
            discarded — inspect manually instead), or has already been
            used as a query for a later cycle (which is structurally
            impossible if ``cycle`` is genuinely the latest search
            attempt; if this fires, undo the later cycle first).
        :returns: ``(hits_removed, queries_released)``.
        """
        stats = self.simsearch_cycle_stats(cycle)
        if stats.n_hits_docked:
            raise ValueError(
                f"cannot undo simsearch cycle {cycle}: {stats.n_hits_docked} of its "
                "hit compound(s) have already been docked; undoing would discard "
                "real docking results. Inspect the database manually before proceeding."
            )
        if stats.n_hits_used_as_query:
            raise ValueError(
                f"cannot undo simsearch cycle {cycle}: {stats.n_hits_used_as_query} of "
                "its hit compound(s) have already been used as queries for a later "
                f"cycle, so cycle {cycle} is not actually the latest search attempt. "
                "Undo the later cycle first."
            )
        self.ensure_extension_schema()
        n_atlas_hits = int(
            self._conn.execute(
                "SELECT COUNT(DISTINCT d.spacehastenid) FROM data AS d "
                "JOIN cluster_atlas_assignments AS a "
                "ON a.spacehastenid = d.spacehastenid "
                "WHERE d.simsearch_cycle = ?",
                (cycle,),
            ).fetchone()[0]
        )
        if n_atlas_hits:
            raise ValueError(
                f"cannot undo simsearch cycle {cycle}: {n_atlas_hits} hit compound(s) "
                "are assigned to an append-only cluster atlas"
            )
        hit_ids = [
            row[0]
            for row in self._conn.execute(
                "SELECT spacehastenid FROM data WHERE simsearch_cycle = ?", (cycle,)
            ).fetchall()
        ]
        if hit_ids:
            placeholders = ",".join("?" * len(hit_ids))
            self.ensure_extension_schema()
            self._conn.execute(
                f"DELETE FROM predictions WHERE spacehastenid IN ({placeholders})",
                hit_ids,
            )
            self._conn.execute(
                f"DELETE FROM clusters WHERE spacehastenid IN ({placeholders})", hit_ids
            )
            self._conn.execute("DELETE FROM data WHERE simsearch_cycle = ?", (cycle,))
        self._conn.execute("UPDATE data SET query = NULL WHERE query = ?", (cycle,))
        self.commit()
        return stats.n_hits, stats.n_queries

    # ----- counts -----

    def count_total(self) -> int:
        """Total number of compounds in the database."""
        row = self._conn.execute("SELECT COUNT(*) FROM data").fetchone()
        return int(row[0])

    def count_docked(self) -> int:
        """Number of compounds with a dock_score."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM data WHERE dock_score IS NOT NULL"
        ).fetchone()
        return int(row[0])

    def count_actives(self, threshold: float) -> int:
        """Number of docked compounds with dock_score < threshold."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM data WHERE dock_score IS NOT NULL AND dock_score < ?",
            (threshold,),
        ).fetchone()
        return int(row[0])

    # ----- model blob storage -----

    def store_model_blob(self, version: int, blob: bytes) -> None:
        self._conn.execute(
            "INSERT INTO models(model_version, model_tar) VALUES (?, ?)",
            (version, memoryview(blob)),
        )

    def load_model_blob(self, version: int) -> bytes:
        row = self._conn.execute(
            "SELECT model_tar FROM models WHERE model_version = ?", (version,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no model with version {version}")
        return bytes(row[0])

    def load_model_path(self, version: int, workdir: object) -> Path:
        """Resolve the on-disk path to a trained model checkpoint.

        Prefers the new on-disk registry layout
        (``workdir.model_dir(version)/model_0/pytorch_model.bin``). Falls
        back to extracting the legacy ``models.model_tar`` BLOB to that
        same location for back-compat with pre-Session-8 databases.

        ``workdir`` is typed as :class:`object` to avoid an import cycle
        with :mod:`spacehasten.workspace`; only its ``model_dir(version)``
        method is used.
        """
        model_dir_method = getattr(workdir, "model_dir", None)
        if not callable(model_dir_method):
            raise TypeError("workdir must expose a model_dir(version) method")
        model_dir = Path(model_dir_method(version))
        bin_path = model_dir / "model_0" / "pytorch_model.bin"
        if bin_path.exists():
            return bin_path

        # Back-compat: extract legacy BLOB.
        try:
            blob = self.load_model_blob(version)
        except KeyError as exc:
            raise FileNotFoundError(
                f"model version {version} not found on disk and no legacy BLOB available"
            ) from exc
        if not blob:
            raise FileNotFoundError(
                f"model version {version} has no on-disk checkpoint and the legacy"
                " BLOB is empty (likely written by Session-8+ training)"
            )
        import io
        import tarfile

        bin_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            tar.extractall(path=model_dir.parent)
        if not bin_path.exists():
            # Some legacy tars contain a top-level model_<name>_ver<N>/model_0/...
            # directory rather than model_0/ directly. Find the checkpoint.
            for candidate in model_dir.parent.rglob("pytorch_model.bin"):
                bin_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil as _shutil

                _shutil.copy(candidate, bin_path)
                break
        if not bin_path.exists():
            raise FileNotFoundError(
                "could not locate pytorch_model.bin after extracting legacy BLOB "
                f"for version {version}"
            )
        return bin_path

    # ----- docking blobs -----

    def store_dock_param(self, blob: bytes) -> None:
        self._conn.execute("INSERT INTO docking_param VALUES (?)", (memoryview(blob),))

    def store_dock_grid(self, blob: bytes) -> None:
        self._conn.execute("INSERT INTO docking_grid VALUES (?)", (memoryview(blob),))

    def load_dock_param(self) -> bytes:
        row = self._conn.execute("SELECT dock_param FROM docking_param").fetchone()
        if row is None:
            raise LookupError("docking_param is empty")
        return bytes(row[0])

    def load_dock_grid(self) -> bytes:
        row = self._conn.execute("SELECT dock_grid FROM docking_grid").fetchone()
        if row is None:
            raise LookupError("docking_grid is empty")
        return bytes(row[0])

    # ----- clusters -----

    def replace_clusters(self, rows: Iterable[ClusterRow]) -> None:
        c = self._conn.cursor()
        c.execute("DROP TABLE IF EXISTS clusters")
        c.execute("CREATE TABLE clusters(spacehastenid INTEGER PRIMARY KEY,clusterid INTEGER)")
        c.executemany(
            "INSERT INTO clusters(spacehastenid, clusterid) VALUES (?, ?)",
            ((r.spacehastenid, r.clusterid) for r in rows),
        )

    def upsert_cluster_atlas(self, row: ClusterAtlasRow) -> None:
        """Create an atlas definition or verify its immutable configuration."""
        self.ensure_extension_schema()
        existing = self._conn.execute(
            "SELECT similarity_threshold, fingerprint_type, "
            "fingerprint_parameters, partition_count FROM cluster_atlases "
            "WHERE atlas_id = ?",
            (row.atlas_id,),
        ).fetchone()
        values = (
            row.similarity_threshold,
            row.fingerprint_type,
            row.fingerprint_parameters,
            row.partition_count,
        )
        if existing is not None:
            if tuple(existing) != values:
                raise ValueError(
                    f"atlas {row.atlas_id!r} already exists with different configuration"
                )
            return
        self._conn.execute(
            "INSERT INTO cluster_atlases("
            "atlas_id, similarity_threshold, fingerprint_type, "
            "fingerprint_parameters, partition_count"
            ") VALUES (?, ?, ?, ?, ?)",
            (row.atlas_id, *values),
        )

    def cluster_atlas(self, atlas_id: str) -> ClusterAtlasRow | None:
        self.ensure_extension_schema()
        row = self._conn.execute(
            "SELECT atlas_id, similarity_threshold, fingerprint_type, "
            "fingerprint_parameters, partition_count FROM cluster_atlases "
            "WHERE atlas_id = ?",
            (atlas_id,),
        ).fetchone()
        return ClusterAtlasRow(*row) if row is not None else None

    def record_cluster_atlas_version(self, row: ClusterAtlasVersionRow) -> None:
        self.ensure_extension_schema()
        self._conn.execute(
            "INSERT INTO cluster_atlas_versions("
            "atlas_id, version, last_spacehastenid, compound_count, "
            "centroid_count, metadata_path"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (
                row.atlas_id,
                row.version,
                row.last_spacehastenid,
                row.compound_count,
                row.centroid_count,
                row.metadata_path,
            ),
        )

    def append_cluster_atlas_centroids(self, rows: Iterable[ClusterAtlasCentroidRow]) -> None:
        self.ensure_extension_schema()
        self._conn.executemany(
            "INSERT INTO cluster_atlas_centroids("
            "atlas_id, clusterid, centroid_spacehastenid, created_version"
            ") VALUES (?, ?, ?, ?)",
            (
                (
                    row.atlas_id,
                    row.clusterid,
                    row.centroid_spacehastenid,
                    row.created_version,
                )
                for row in rows
            ),
        )

    def append_cluster_atlas_assignments(self, rows: Iterable[ClusterAtlasAssignmentRow]) -> None:
        self.ensure_extension_schema()
        self._conn.executemany(
            "INSERT INTO cluster_atlas_assignments("
            "atlas_id, spacehastenid, clusterid, centroid_similarity, "
            "assigned_version"
            ") VALUES (?, ?, ?, ?, ?)",
            (
                (
                    row.atlas_id,
                    row.spacehastenid,
                    row.clusterid,
                    row.centroid_similarity,
                    row.assigned_version,
                )
                for row in rows
            ),
        )

    def latest_cluster_atlas_version(self, atlas_id: str) -> ClusterAtlasVersionRow | None:
        self.ensure_extension_schema()
        row = self._conn.execute(
            "SELECT atlas_id, version, last_spacehastenid, compound_count, "
            "centroid_count, metadata_path FROM cluster_atlas_versions "
            "WHERE atlas_id = ? ORDER BY version DESC LIMIT 1",
            (atlas_id,),
        ).fetchone()
        return ClusterAtlasVersionRow(*row) if row is not None else None

    def count_missing_cluster_atlas_assignments(self, atlas_id: str) -> int:
        self.ensure_extension_schema()
        row = self._conn.execute(
            "SELECT COUNT(*) FROM data AS d "
            "LEFT JOIN cluster_atlas_assignments AS a "
            "ON a.atlas_id = ? AND a.spacehastenid = d.spacehastenid "
            "WHERE a.spacehastenid IS NULL",
            (atlas_id,),
        ).fetchone()
        return int(row[0])

    def materialize_cluster_atlas(self, atlas_id: str) -> int:
        """Replace the legacy clusters table from a persistent atlas."""
        self.ensure_extension_schema()
        rows = self._conn.execute(
            "SELECT spacehastenid, clusterid FROM cluster_atlas_assignments "
            "WHERE atlas_id = ? ORDER BY spacehastenid",
            (atlas_id,),
        ).fetchall()
        self.replace_clusters(ClusterRow(int(sid), int(cid)) for sid, cid in rows)
        count = int(self._conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0])
        return count

    # ----- properties -----

    def replace_properties(self, props: PropertyRanges) -> None:
        c = self._conn.cursor()
        c.execute("DROP TABLE IF EXISTS properties")
        c.execute(
            "CREATE TABLE properties (property TEXT,is_double INTEGER,"
            "min_limit TEXT,max_limit TEXT)"
        )
        c.executemany(
            "INSERT INTO properties (property, is_double, min_limit, max_limit)"
            " VALUES (?, ?, ?, ?)",
            ((r.property, r.is_double, r.min_limit, r.max_limit) for r in props.to_rows()),
        )

    def load_properties(self) -> PropertyRanges | None:
        try:
            rows = self._conn.execute(
                "SELECT property, is_double, min_limit, max_limit FROM properties"
            ).fetchall()
        except sqlite3.OperationalError:
            return None
        if not rows:
            return None
        index: dict[str, tuple[str, str]] = {str(r[0]): (str(r[2]), str(r[3])) for r in rows}
        required = ("mw", "slogp", "hba", "hbd", "rotbonds", "tpsa")
        if not all(k in index for k in required):
            return None
        return PropertyRanges(
            mw=index["mw"],
            slogp=index["slogp"],
            hba=index["hba"],
            hbd=index["hbd"],
            rotbonds=index["rotbonds"],
            tpsa=index["tpsa"],
        )

    # ----- smarts filters -----

    def replace_smarts_filters(self, patterns: list[tuple[str, str]]) -> None:
        """Persist SMARTS include/exclude patterns.

        ``patterns`` is a list of ``(mode, smarts)`` pairs where *mode* is
        either ``'include'`` or ``'exclude'``.  Calling with an empty list
        clears any previously stored patterns.
        """
        c = self._conn.cursor()
        c.execute("DROP TABLE IF EXISTS smarts_filters")
        c.execute("CREATE TABLE smarts_filters (mode TEXT, pattern TEXT)")
        if patterns:
            c.executemany(
                "INSERT INTO smarts_filters (mode, pattern) VALUES (?, ?)",
                patterns,
            )

    def load_smarts_filters(self) -> list[tuple[str, str]]:
        """Return stored ``(mode, smarts)`` pairs, or ``[]`` if none stored."""
        try:
            rows = self._conn.execute("SELECT mode, pattern FROM smarts_filters").fetchall()
        except sqlite3.OperationalError:
            return []
        return [(str(r[0]), str(r[1])) for r in rows]

    def select_queries_for_simsearch(
        self,
        source: Literal["docked", "predicted"],
        strategy: Literal["greedy", "clustering"],
        limit: int,
    ) -> list[tuple[str, int]]:
        sql = self._simsearch_sql(source, strategy)
        return [(s, i) for s, i in self._conn.execute(sql, (limit,)).fetchall()]

    @classmethod
    def _simsearch_sql(
        cls,
        source: Literal["docked", "predicted"],
        strategy: Literal["greedy", "clustering"],
    ) -> str:
        if source == "docked" and strategy == "greedy":
            return cls._SQL_SIMSEARCH_DOCKED_GREEDY
        if source == "docked" and strategy == "clustering":
            return cls._SQL_SIMSEARCH_DOCKED_CLUSTERING
        if source == "predicted" and strategy == "greedy":
            return cls._SQL_SIMSEARCH_PREDICTED_GREEDY
        if source == "predicted" and strategy == "clustering":
            return cls._SQL_SIMSEARCH_PREDICTED_CLUSTERING
        raise ValueError(f"unknown simsearch SQL: source={source!r} strategy={strategy!r}")

    def select_compounds_to_dock(
        self, strategy: Literal["greedy", "clustering"], limit: int
    ) -> list[tuple[str, int]]:
        sql = self._SQL_DOCK_GREEDY if strategy == "greedy" else self._SQL_DOCK_CLUSTERING
        return [(s, i) for s, i in self._conn.execute(sql, (limit,)).fetchall()]

    def select_uncertainty_docking_candidates(
        self, *, atlas_id: str | None = None
    ) -> list[AcquisitionCandidate]:
        """Return undocked candidates with version-matched epistemic uncertainty."""
        self.ensure_extension_schema()
        if atlas_id is None:
            rows = self._conn.execute(self._SQL_DOCK_UNCERTAINTY_CANDIDATES).fetchall()
        else:
            rows = self._conn.execute(
                self._SQL_DOCK_ATLAS_UNCERTAINTY_CANDIDATES,
                (atlas_id,),
            ).fetchall()
        missing: list[int] = []
        missing_atlas: list[int] = []
        candidates: list[AcquisitionCandidate] = []
        for smiles, sid, mean, epistemic_std, version, clusterid in rows:
            if mean is None or epistemic_std is None:
                missing.append(int(sid))
                continue
            if smiles is None:
                raise ValueError(f"acquisition candidate {sid} has no SMILES")
            if atlas_id is not None and clusterid is None:
                missing_atlas.append(int(sid))
                continue
            candidates.append(
                AcquisitionCandidate(
                    smiles=str(smiles),
                    spacehastenid=int(sid),
                    pred_score=float(mean),
                    epistemic_std=float(epistemic_std),
                    model_version=int(version),
                    clusterid=int(clusterid) if clusterid is not None else None,
                )
            )
        if missing:
            preview = ", ".join(str(identifier) for identifier in missing[:5])
            raise ValueError(
                f"{len(missing)} predicted undocked compounds lack version-matched "
                f"epistemic uncertainty (first IDs: {preview})"
            )
        if missing_atlas:
            preview = ", ".join(str(identifier) for identifier in missing_atlas[:5])
            raise ValueError(
                f"{len(missing_atlas)} predicted undocked compounds lack assignments "
                f"in atlas {atlas_id!r} (first IDs: {preview})"
            )
        return candidates

    def has_clusters(self) -> bool:
        """Whether ``clusters`` has any rows (i.e. ``cluster`` has run)."""
        return self._conn.execute("SELECT 1 FROM clusters LIMIT 1").fetchone() is not None

    def select_undocked_for_prediction(self, batch_size: int = 10000) -> Iterator[tuple[str, int]]:
        cur = self._conn.execute(self._SQL_SELECT_UNDOCKED)
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            yield from rows

    def count_undocked_for_prediction(self) -> int:
        """Number of rows :meth:`select_undocked_for_prediction` would yield."""
        sql = self._SQL_SELECT_UNDOCKED.replace(
            "SELECT smiles, spacehastenid", "SELECT COUNT(*)", 1
        )
        row = self._conn.execute(sql).fetchone()
        return int(row[0]) if row else 0

    def select_training_data(self, cutoff: float = 10.0) -> list[tuple[str, float]]:
        return [
            (s, d) for s, d in self._conn.execute(self._SQL_SELECT_TRAINING, (cutoff,)).fetchall()
        ]

    def select_predictions(self, model_version: int | None = None) -> list[PredictionRow]:
        """Return persisted prediction history, optionally for one model version."""
        self.ensure_extension_schema()
        sql = (
            "SELECT spacehastenid, model_version, pred_score, epistemic_std, "
            "aleatoric_std, total_std, created_at FROM predictions"
        )
        params: tuple[int, ...] = ()
        if model_version is not None:
            sql += " WHERE model_version = ?"
            params = (model_version,)
        sql += " ORDER BY model_version, spacehastenid"
        return [PredictionRow(*row) for row in self._conn.execute(sql, params).fetchall()]

    def select_export_rows(self, cutoff: float) -> list[ExportRow]:
        return [
            ExportRow(
                smiles=row[0],
                spacehastenid=row[1],
                smilesid=row[2],
                dock_score=row[3],
                pred_score=row[4],
                spacelight=row[5],
                ftrees=row[6],
                dock_iteration=row[7],
                clusterid=row[8],
            )
            for row in self._conn.execute(self._SQL_SELECT_EXPORT, (cutoff,)).fetchall()
        ]

    def select_seed_rows(self) -> list[tuple[str, str, float]]:
        """Docked rows from the original seed batch (``dock_iteration == 0``).

        ``dock_iteration == 0`` uniquely identifies the seed round: it is
        set on pre-docked CSV seed imports (:meth:`insert_seed_docked`) and
        is also the iteration number assigned to the first ever ``dock``
        call (which docks previously-undocked ``.smi`` seeds). Every later
        ``dock`` call gets ``iteration = latest + 1 >= 1``, so compounds
        discovered in subsequent screening cycles never carry
        ``dock_iteration == 0``.
        """
        return [
            (row[0], row[1], row[2])
            for row in self._conn.execute(self._SQL_SELECT_SEEDS).fetchall()
        ]

    # ----- bulk applies (used by stages, not specified above but useful) -----

    def apply_dock_scores(self, rows: Sequence[tuple[float, int, int]]) -> None:
        """Apply many ``(dock_score, dock_iteration, spacehastenid)`` updates."""
        self._conn.executemany(self._SQL_UPDATE_DOCK_SCORE, rows)

    def apply_pred_scores(self, rows: Sequence[tuple[float, int, int]]) -> None:
        """Apply many ``(pred_score, pred_version, spacehastenid)`` updates."""
        self._conn.executemany(self._SQL_UPDATE_PRED_SCORE, rows)

    def apply_predictions(
        self,
        rows: Sequence[tuple[int, int, float, float | None, float | None, float | None]],
    ) -> None:
        """Persist versioned predictions and update the legacy latest-score cache."""
        if not rows:
            return
        self.ensure_extension_schema()
        self._conn.executemany(
            self._SQL_UPDATE_PRED_SCORE,
            [(score, version, sid) for sid, version, score, _, _, _ in rows],
        )
        self._conn.executemany(self._SQL_UPSERT_PREDICTION, rows)


__all__ = [
    "AcquisitionBatchRow",
    "AcquisitionOutcomeRow",
    "AcquisitionOutcomeUpdate",
    "AcquisitionRegionSummaryRow",
    "AcquisitionSelectionRow",
    "acquisition_selection_digest",
    "ClusterAtlasAssignmentRow",
    "ClusterAtlasCentroidRow",
    "ClusterAtlasRow",
    "ClusterAtlasVersionRow",
    "ClusterRow",
    "Database",
    "DataRow",
    "EXTENSION_SCHEMA_STATEMENTS",
    "ExportRow",
    "canonical_json",
    "ModelRow",
    "ModelCalibrationRow",
    "PredictionRow",
    "PropertyRanges",
    "PropertyRow",
    "SCHEMA_STATEMENTS",
    "sha256_hex",
    "SimsearchCycleStats",
]
