"""Strictly read-only SQLite access for analysis."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path
from urllib.parse import quote

from .models import AtlasResolution, Capabilities, Row, SelectionRound

SQLITE_BATCH_SIZE = 900


class ReadOnlyDatabase:
    """SQLite adapter that opens an immutable source in URI read-only mode."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.connection = sqlite3.connect(f"file:{quote(str(self.path))}?mode=ro", uri=True)
        self.connection.execute("PRAGMA query_only = ON")

    def __enter__(self) -> ReadOnlyDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.connection.close()

    def capabilities(self) -> Capabilities:
        tables = tuple(
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        )
        columns = (
            tuple(str(row[1]) for row in self.connection.execute("PRAGMA table_info(data)"))
            if "data" in tables
            else ()
        )
        predictions_columns = (
            tuple(str(row[1]) for row in self.connection.execute("PRAGMA table_info(predictions)"))
            if "predictions" in tables
            else ()
        )
        return Capabilities(
            tables,
            columns,
            predictions_columns,
            "data" in tables,
            "predictions" in tables,
            "cluster_atlas_assignments" in tables,
            "clusters" in tables,
        )

    def quick_check(self) -> str:
        return str(self.connection.execute("PRAGMA quick_check").fetchone()[0])


def fetch_rows(
    database: ReadOnlyDatabase, capabilities: Capabilities, selections: dict[int, SelectionRound]
) -> dict[int, Row]:
    required = {"spacehastenid", "dock_score", "dock_iteration", "smiles"}
    missing = required - set(capabilities.data_columns)
    if missing:
        raise ValueError(f"data table lacks required columns: {sorted(missing)}")
    fields = ("spacehastenid", "dock_score", "dock_iteration", "smiles")
    if not selections:
        query = (
            "SELECT "
            + ", ".join(fields)
            + " FROM data WHERE dock_iteration > 0 ORDER BY dock_iteration, spacehastenid"
        )
        return {
            int(row[0]): dict(zip(fields, row, strict=True))
            for row in database.connection.execute(query)
        }
    identifiers = sorted(
        {identifier for selection in selections.values() for identifier in selection.attempts}
    )
    result: dict[int, Row] = {}
    for batch in batches(identifiers, SQLITE_BATCH_SIZE):
        placeholders = ",".join("?" for _ in batch)
        scored_query = (
            f"SELECT {', '.join(fields)} FROM data WHERE dock_iteration > 0 "
            f"AND dock_score IS NOT NULL AND spacehastenid IN ({placeholders})"
        )
        for row in database.connection.execute(scored_query, batch):
            result[int(row[0])] = dict(zip(fields, row, strict=True))
        unresolved = tuple(identifier for identifier in batch if identifier not in result)
        if unresolved:
            missing_placeholders = ",".join("?" for _ in unresolved)
            missing_query = (
                f"SELECT {', '.join(fields)} FROM data WHERE spacehastenid "
                f"IN ({missing_placeholders})"
            )
            for row in database.connection.execute(missing_query, unresolved):
                result[int(row[0])] = dict(zip(fields, row, strict=True))
    return result


def fetch_predictions(
    database: ReadOnlyDatabase,
    capabilities: Capabilities,
    identifiers: Sequence[int],
    model_version: str,
) -> dict[int, Row]:
    """Fetch selected prediction rows for exactly one acquisition model version."""
    required = {"spacehastenid", "model_version", "pred_score"}
    missing = required - set(capabilities.predictions_columns)
    if not capabilities.has_predictions or missing:
        return {}
    fields = ["spacehastenid", "model_version", "pred_score"]
    fields.extend(
        column
        for column in ("epistemic_std", "aleatoric_std", "total_std")
        if column in capabilities.predictions_columns
    )
    result: dict[int, Row] = {}
    for batch in batches(identifiers, SQLITE_BATCH_SIZE):
        placeholders = ",".join("?" for _ in batch)
        query = (
            f"SELECT {', '.join(fields)} FROM predictions "
            f"WHERE model_version = ? AND spacehastenid IN ({placeholders})"
        )
        for row in database.connection.execute(query, (model_version, *batch)):
            result[int(row[0])] = dict(zip(fields, row, strict=True))
    return result


def resolve_atlas(
    database: ReadOnlyDatabase, selections: dict[int, SelectionRound]
) -> AtlasResolution:
    if not database.capabilities().has_atlas_assignments:
        return AtlasResolution(None, "unavailable", "cluster_atlas_assignments table absent")
    provenance = {atlas for selection in selections.values() for atlas in selection.atlas_ids}
    if len(provenance) == 1:
        return AtlasResolution(next(iter(provenance)), "available", None)
    if len(provenance) > 1:
        return AtlasResolution(None, "ambiguous", "acquisition provenance names multiple atlas IDs")
    atlas_ids = {
        str(row[0])
        for row in database.connection.execute(
            "SELECT DISTINCT atlas_id FROM cluster_atlas_assignments"
        )
    }
    if len(atlas_ids) == 1:
        return AtlasResolution(next(iter(atlas_ids)), "available", None)
    reason = "no atlas assignments" if not atlas_ids else "database contains multiple atlas IDs"
    return AtlasResolution(None, "unavailable" if not atlas_ids else "ambiguous", reason)


def fetch_atlas(
    database: ReadOnlyDatabase, atlas: AtlasResolution, identifiers: Sequence[int]
) -> dict[int, str]:
    if atlas.atlas_id is None:
        return {}
    result: dict[int, str] = {}
    for batch in batches(identifiers, SQLITE_BATCH_SIZE):
        placeholders = ",".join("?" for _ in batch)
        query = (
            "SELECT spacehastenid, clusterid FROM cluster_atlas_assignments "
            "WHERE atlas_id = ? AND spacehastenid IN (" + placeholders + ")"
        )
        for identifier, cluster in database.connection.execute(query, (atlas.atlas_id, *batch)):
            result[int(identifier)] = str(cluster)
    return result


def batches(values: Sequence[int], size: int) -> Iterator[tuple[int, ...]]:
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])
