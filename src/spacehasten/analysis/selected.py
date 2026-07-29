"""Normalized selected-attempt manifests for modern and legacy runs."""

from __future__ import annotations

import csv
import gzip
import io
from collections.abc import Sequence
from pathlib import Path

from .artifacts import AtomicPath
from .database import ReadOnlyDatabase, batches
from .models import Row, RunContext
from .recovery import has_modern_history, recover_docking_input_acquisitions

ID_COLUMNS = ("spacehastenid", "id", "compound_id")


def selected_manifest(
    context: RunContext,
    *,
    hit_threshold: float,
    strict_threshold: float,
) -> list[Row]:
    """Return one normalized row per selected attempt in acquisition order."""
    with ReadOnlyDatabase(context.database_path) as database:
        if has_modern_history(database):
            rows = _modern_manifest(database)
        elif context.acquisition_paths:
            rows = _legacy_manifest(database, context.acquisition_paths)
        elif context.docking_input_paths:
            _, _, rows = recover_docking_input_acquisitions(
                database,
                context.capabilities,
                context.docking_input_paths,
            )
        else:
            rows = []
    for row in rows:
        score = row["dock_score"]
        row["is_scored"] = score is not None
        row["is_hit"] = score is not None and float(score) <= hit_threshold
        row["is_strict_hit"] = score is not None and float(score) <= strict_threshold
    _validate_manifest(rows)
    return rows


def write_selected_manifest(path: Path, rows: Sequence[Row]) -> None:
    """Write a deterministic CSV or gzip-compressed CSV atomically."""
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with AtomicPath(path) as temporary:
        if path.suffix == ".gz":
            with (
                temporary.open("wb") as raw,
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
                io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text,
            ):
                writer = csv.DictWriter(text, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        else:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)


def _modern_manifest(database: ReadOnlyDatabase) -> list[Row]:
    calibration_join = (
        "LEFT JOIN model_calibrations AS c ON c.model_version=s.model_version "
        if "model_calibrations" in database.capabilities().tables
        else ""
    )
    calibration_fields = (
        "c.calibration_kind,c.uncertainty_source,c.mean_shift,c.std_scale,c.std_floor,"
        "c.fit_source,c.fit_split_name,c.fit_row_count,c.split_sha256,c.artifact_sha256,"
        if calibration_join
        else "NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,"
    )
    query = (
        "SELECT b.dock_iteration,s.selection_rank,s.spacehastenid,d.reghash,d.smiles,"
        "o.status,o.dock_score,o.source,b.batch_id,b.strategy,b.status,b.policy_schema_version,"
        "b.policy_sha256,b.history_attempt_policy,b.model_version,b.atlas_id,b.atlas_version,"
        "b.candidate_count,b.candidate_watermark,b.candidate_digest,b.requested_count,"
        "b.selected_count,b.selection_digest,b.cap_scope,b.cap_limit,s.clusterid,"
        "s.model_version,s.raw_mean,s.raw_epistemic_std,s.calibrated_mean,s.calibrated_std,"
        "s.p_hit,s.expected_improvement,s.quality,s.support_before,s.support_after,"
        "s.marginal_reward,s.crowding_penalty,s.final_utility,s.cluster_count_before,"
        "s.cap_reached_after,s.contributions_json,"
        + calibration_fields
        + "d.dock_iteration,d.dock_score "
        "FROM acquisition_batches AS b "
        "JOIN acquisition_selections AS s ON s.batch_id=b.batch_id "
        "JOIN acquisition_outcomes AS o ON o.batch_id=s.batch_id "
        "AND o.spacehastenid=s.spacehastenid "
        "LEFT JOIN data AS d ON d.spacehastenid=s.spacehastenid "
        + calibration_join
        + "ORDER BY b.dock_iteration,s.selection_rank"
    )
    fields = (
        "round",
        "rank",
        "spacehastenid",
        "reghash",
        "smiles",
        "outcome_status",
        "dock_score",
        "outcome_source",
        "batch_id",
        "strategy",
        "batch_status",
        "policy_schema_version",
        "policy_sha256",
        "history_attempt_policy",
        "batch_model_version",
        "atlas_id",
        "atlas_version",
        "candidate_count",
        "candidate_watermark",
        "candidate_digest",
        "requested_count",
        "selected_count",
        "selection_digest",
        "cap_scope",
        "cap_limit",
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
        "contributions_json",
        "calibration_kind",
        "calibration_uncertainty_source",
        "calibration_mean_shift",
        "calibration_std_scale",
        "calibration_std_floor",
        "calibration_fit_source",
        "calibration_fit_split_name",
        "calibration_fit_row_count",
        "calibration_split_sha256",
        "calibration_artifact_sha256",
        "data_dock_iteration",
        "data_dock_score",
    )
    rows = [dict(zip(fields, values, strict=True)) for values in database.connection.execute(query)]
    for row in rows:
        row["selection_id"] = f"{row['batch_id']}:{row['rank']}"
        row["cap_reached_after"] = bool(row["cap_reached_after"])
        row["selection_source"] = "database_history"
        row["rank_source"] = "persisted_selection_rank"
    return rows


def _legacy_manifest(
    database: ReadOnlyDatabase,
    paths: tuple[tuple[int, Path], ...],
) -> list[Row]:
    attempts: list[Row] = []
    for round_id, path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            id_column = next((column for column in ID_COLUMNS if column in fields), None)
            if id_column is None:
                raise ValueError(f"acquisition file lacks an identifier column: {path}")
            for offset, acquisition in enumerate(reader, 1):
                identifier = int(acquisition[id_column])
                rank = int(acquisition.get("rank") or offset)
                row: Row = {
                    "selection_id": f"round-{round_id}:{rank}",
                    "round": round_id,
                    "rank": rank,
                    "spacehastenid": identifier,
                    "selection_source": "acquisition_csv",
                    "rank_source": "acquisition_csv",
                }
                for name, value in acquisition.items():
                    if name not in ID_COLUMNS and name != "rank":
                        row[f"acquisition_{name}"] = value
                attempts.append(row)
    details = _selected_data(database, [int(row["spacehastenid"]) for row in attempts])
    for row in attempts:
        detail = details.get(int(row["spacehastenid"]))
        if detail is None:
            raise ValueError(f"selected ID is absent from data: {row['spacehastenid']}")
        row.update(detail)
        score = row["data_dock_score"]
        row["dock_score"] = score if row["data_dock_iteration"] == row["round"] else None
        row["outcome_status"] = "scored" if row["dock_score"] is not None else "unresolved"
        row["outcome_source"] = "data"
    return attempts


def _selected_data(database: ReadOnlyDatabase, identifiers: Sequence[int]) -> dict[int, Row]:
    result: dict[int, Row] = {}
    unique = sorted(set(identifiers))
    for batch in batches(unique, 900):
        placeholders = ",".join("?" for _ in batch)
        query = (
            "SELECT spacehastenid,reghash,smiles,dock_iteration,dock_score FROM data "
            f"WHERE spacehastenid IN ({placeholders})"
        )
        for identifier, reghash, smiles, iteration, score in database.connection.execute(
            query, batch
        ):
            result[int(identifier)] = {
                "reghash": reghash,
                "smiles": smiles,
                "data_dock_iteration": iteration,
                "data_dock_score": score,
            }
    return result


def _validate_manifest(rows: Sequence[Row]) -> None:
    selection_ids = [str(row["selection_id"]) for row in rows]
    if len(set(selection_ids)) != len(selection_ids):
        raise ValueError("selection IDs are not unique")
    previous: tuple[int, int] | None = None
    for row in rows:
        round_id, rank = int(row["round"]), int(row["rank"])
        if round_id < 1 or rank < 1:
            raise ValueError("selection rounds and ranks must be positive")
        if previous is not None and (round_id, rank) <= previous:
            raise ValueError("selected manifest is not strictly ordered by round and rank")
        if row.get("smiles") in (None, "") or row.get("reghash") in (None, ""):
            raise ValueError(f"selected ID lacks structure identity: {row['spacehastenid']}")
        previous = (round_id, rank)


__all__ = ["selected_manifest", "write_selected_manifest"]
