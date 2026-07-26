"""Reusable candidate-pool loading, EI scoring, and replay validation."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

from spacehasten.core.acquisition import expected_improvement

FloatArray = npt.NDArray[np.float64]
Int8Array = npt.NDArray[np.int8]
Int64Array = npt.NDArray[np.int64]


class ReplaySelection(Protocol):
    """Minimal replay result required for historical validation."""

    indices: Int64Array


@dataclass(frozen=True, slots=True)
class RoundDefinition:
    """A database-independent definition of one acquisition candidate pool."""

    round_id: int
    model_version: int
    atlas_version: int
    upper_id: int
    candidate_filter: str


@dataclass(frozen=True, slots=True)
class CandidatePool:
    identifiers: Int64Array
    means: FloatArray
    epistemic: FloatArray
    atlas_clusters: Int64Array
    dock_scores: FloatArray
    dock_iterations: Int8Array


def readonly_connection(path: Path) -> sqlite3.Connection:
    """Open a connection that cannot modify the source database."""
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _validate_score_inputs(
    means: FloatArray,
    epistemic: FloatArray,
    threshold: float,
    xi: float,
) -> tuple[FloatArray, FloatArray]:
    means_array = np.asarray(means, dtype=np.float64)
    epistemic_array = np.asarray(epistemic, dtype=np.float64)
    if means_array.ndim != 1 or epistemic_array.ndim != 1:
        raise ValueError("means and epistemic must be one-dimensional")
    if len(means_array) != len(epistemic_array):
        raise ValueError("means and epistemic must have equal length")
    if not math.isfinite(threshold) or not math.isfinite(xi):
        raise ValueError("threshold and xi must be finite")
    if not np.isfinite(means_array).all() or not np.isfinite(epistemic_array).all():
        raise ValueError("means and epistemic must be finite")
    if np.any(epistemic_array < 0):
        raise ValueError("epistemic values must be non-negative")
    return means_array, epistemic_array


def expected_improvement_scores(
    means: FloatArray,
    epistemic: FloatArray,
    threshold: float,
    xi: float,
) -> FloatArray:
    """Return production-exact negative EI scores for minimization."""
    means_array, epistemic_array = _validate_score_inputs(means, epistemic, threshold, xi)
    return np.fromiter(
        (
            -expected_improvement(float(mean), float(std), threshold, xi)
            for mean, std in zip(means_array, epistemic_array, strict=True)
        ),
        dtype=np.float64,
        count=len(means_array),
    )


def deterministic_improvement_scores(means: FloatArray, threshold: float, xi: float) -> FloatArray:
    """Return deterministic EI scores using the production minimization sign."""
    means_array, _ = _validate_score_inputs(means, np.zeros(len(means)), threshold, xi)
    return -np.maximum(threshold - means_array - xi, 0.0)


def load_candidate_pool(
    database: Path,
    definition: RoundDefinition,
    *,
    atlas_id: str,
    seed_boundary: int | None = None,
) -> CandidatePool:
    """Load a sorted pool, excluding an explicit ID boundary or iteration-zero seeds."""
    if not atlas_id:
        raise ValueError("atlas_id must be non-empty")
    if definition.round_id < 1 or definition.upper_id < 1:
        raise ValueError("round_id and upper_id must be positive")
    if seed_boundary is not None and seed_boundary < 0:
        raise ValueError("seed_boundary must be non-negative")

    seed_clause = (
        "d.spacehastenid > ?"
        if seed_boundary is not None
        else "(d.dock_iteration IS NULL OR d.dock_iteration != 0)"
    )
    where = f"{seed_clause} AND d.spacehastenid <= ? AND ({definition.candidate_filter})"
    parameters: tuple[object, ...] = (
        (definition.model_version, atlas_id, seed_boundary, definition.upper_id)
        if seed_boundary is not None
        else (definition.model_version, atlas_id, definition.upper_id)
    )
    count_query = (
        "SELECT COUNT(*) FROM data AS d "
        "JOIN predictions AS p ON p.spacehastenid=d.spacehastenid AND p.model_version=? "
        "JOIN cluster_atlas_assignments AS a "
        "ON a.atlas_id=? AND a.spacehastenid=d.spacehastenid "
        f"WHERE {where}"
    )
    data_query = (
        "SELECT d.spacehastenid,p.pred_score,p.epistemic_std,a.clusterid,d.dock_score,"
        "COALESCE(d.dock_iteration,-1) FROM data AS d "
        "JOIN predictions AS p ON p.spacehastenid=d.spacehastenid AND p.model_version=? "
        "JOIN cluster_atlas_assignments AS a "
        "ON a.atlas_id=? AND a.spacehastenid=d.spacehastenid "
        f"WHERE {where} ORDER BY d.spacehastenid"
    )
    with readonly_connection(database) as connection:
        row_count = int(connection.execute(count_query, parameters).fetchone()[0])
        if row_count == 0:
            raise ValueError(f"round {definition.round_id}: candidate pool is empty")
        values = np.empty((row_count, 6), dtype=np.float64)
        for position, row in enumerate(connection.execute(data_query, parameters)):
            values[position] = row

    pool = CandidatePool(
        identifiers=values[:, 0].astype(np.int64),
        means=values[:, 1],
        epistemic=values[:, 2],
        atlas_clusters=values[:, 3].astype(np.int64),
        dock_scores=values[:, 4],
        dock_iterations=values[:, 5].astype(np.int8),
    )
    if len(np.unique(pool.identifiers)) != len(pool.identifiers) or not np.all(
        np.diff(pool.identifiers) > 0
    ):
        raise ValueError(f"round {definition.round_id}: candidate IDs are not unique and sorted")
    _validate_score_inputs(pool.means, pool.epistemic, 0.0, 0.0)
    return pool


def load_score_map(path: Path) -> dict[str, float]:
    with readonly_connection(path) as connection:
        return {
            str(key): float(value)
            for key, value in connection.execute(
                "SELECT reghash,dock_score FROM data "
                "WHERE dock_iteration > 0 AND dock_score IS NOT NULL"
            )
        }


def discover_round_definitions(
    database: Path, acquisition_root: Path, *, atlas_id: str
) -> tuple[RoundDefinition, ...]:
    """Discover any contiguous positive set of acquisition-round definitions."""
    if not atlas_id:
        raise ValueError("atlas_id must be non-empty")
    paths = sorted(
        acquisition_root.glob("iter*/acquisition.csv"),
        key=lambda path: int(path.parent.name[4:]),
    )
    round_ids = [int(path.parent.name[4:]) for path in paths]
    if not round_ids or any(round_id < 1 for round_id in round_ids):
        raise ValueError("at least one positive acquisition round is required")
    if len(set(round_ids)) != len(round_ids):
        raise ValueError("acquisition round definitions must be unique")
    if round_ids != list(range(1, max(round_ids) + 1)):
        raise ValueError("acquisition rounds must be contiguous and start at one")
    with readonly_connection(database) as connection:
        watermarks = {
            int(version): int(last_identifier)
            for version, last_identifier in connection.execute(
                "SELECT version,last_spacehastenid FROM cluster_atlas_versions WHERE atlas_id=?",
                (atlas_id,),
            )
        }

    definitions: list[RoundDefinition] = []
    for path, round_id in zip(paths, round_ids, strict=True):
        metadata = pd.read_csv(path, nrows=1)
        if metadata.empty or "model_version" not in metadata.columns:
            raise ValueError(f"round {round_id}: acquisition metadata lacks model_version")
        row = metadata.iloc[0]
        atlas_version = int(row.get("cluster_atlas_version", round_id))
        if atlas_version not in watermarks:
            raise ValueError(f"round {round_id}: atlas version has no registered watermark")
        candidate_filter = (
            "1=1" if round_id == 1 else f"d.dock_iteration IS NULL OR d.dock_iteration={round_id}"
        )
        definitions.append(
            RoundDefinition(
                round_id=round_id,
                model_version=int(row["model_version"]),
                atlas_version=atlas_version,
                upper_id=watermarks[atlas_version],
                candidate_filter=candidate_filter,
            )
        )
    return tuple(definitions)


def load_actual_acquisition(
    path: Path,
    definition: RoundDefinition,
    batch_size: int,
    threshold: float,
    xi: float,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "rank",
        "method",
        "spacehastenid",
        "model_version",
        "pred_score",
        "epistemic_std",
        "base_score",
        "clusterid",
        "cluster_count_before",
        "cluster_penalty",
        "penalized_score",
        "ei_hit_threshold",
        "ei_xi",
        "cluster_lambda",
    }
    if (
        not required.issubset(frame.columns)
        or len(frame) != batch_size
        or frame.spacehastenid.duplicated().any()
    ):
        raise ValueError(f"round {definition.round_id}: invalid historical acquisition")
    expected = expected_improvement_scores(
        frame.pred_score.to_numpy(dtype=np.float64),
        frame.epistemic_std.to_numpy(dtype=np.float64),
        threshold,
        xi,
    )
    if not np.allclose(frame.base_score, expected, atol=1e-10, rtol=1e-9):
        raise ValueError(f"round {definition.round_id}: invalid EI base scores")
    return frame


def validate_historical_replay(
    definition: RoundDefinition,
    pool: CandidatePool,
    actual: pd.DataFrame,
    replay: ReplaySelection,
    scores: FloatArray,
) -> None:
    indices = np.asarray(replay.indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(pool.identifiers)):
        raise ValueError(f"round {definition.round_id}: replay indices are invalid")
    replay_ids = pool.identifiers[indices]
    if not np.array_equal(replay_ids, actual.spacehastenid.to_numpy(dtype=np.int64)):
        raise ValueError(f"round {definition.round_id}: replay differs from history")
    if not np.allclose(scores[indices], actual.base_score, atol=1e-10, rtol=1e-9):
        raise ValueError(f"round {definition.round_id}: replayed EI scores differ from history")


def load_selected_compounds_with_outcomes(
    database: Path,
    identifiers: Int64Array,
    lcb_scores: dict[str, float],
    greedy_scores: dict[str, float],
) -> pd.DataFrame:
    workspace = sqlite3.connect(":memory:", uri=True)
    try:
        source_uri = f"file:{database.resolve()}?mode=ro"
        workspace.execute("ATTACH DATABASE ? AS source", (source_uri,))
        workspace.execute("CREATE TABLE selected(spacehastenid INTEGER PRIMARY KEY)")
        workspace.executemany(
            "INSERT INTO selected VALUES (?)", ((int(value),) for value in identifiers)
        )
        frame = pd.read_sql_query(
            "SELECT d.spacehastenid,d.reghash,d.smiles,d.dock_score,d.dock_iteration "
            "FROM source.data d JOIN selected s USING(spacehastenid) "
            "ORDER BY d.spacehastenid",
            workspace,
        )
    finally:
        workspace.close()
    if len(frame) != len(identifiers):
        raise ValueError("failed to load every selected compound")
    lcb = frame.reghash.map(lcb_scores)
    greedy = frame.reghash.map(greedy_scores)
    frame["observed_score"] = frame.dock_score.fillna(lcb).fillna(greedy)
    frame["outcome_source"] = np.select(
        [frame.dock_score.notna(), lcb.notna(), greedy.notna()],
        ["ei", "lcb", "greedy"],
        default="unobserved",
    )
    return frame
