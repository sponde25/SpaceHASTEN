"""Recovery of selected attempts from immutable docking input chunks."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from .database import ReadOnlyDatabase, batches, fetch_predictions
from .models import Capabilities, Row, SelectionRound

MODERN_HISTORY_TABLES = frozenset(
    {
        "acquisition_batches",
        "acquisition_selections",
        "acquisition_outcomes",
    }
)


def has_modern_history(database: ReadOnlyDatabase) -> bool:
    """Return whether immutable acquisition tables exist and contain a batch."""
    if not set(database.capabilities().tables) >= MODERN_HISTORY_TABLES:
        return False
    return (
        database.connection.execute("SELECT 1 FROM acquisition_batches LIMIT 1").fetchone()
        is not None
    )


def recover_docking_input_acquisitions(
    database: ReadOnlyDatabase,
    capabilities: Capabilities,
    paths: tuple[tuple[int, tuple[Path, ...]], ...],
) -> tuple[dict[int, SelectionRound], list[dict[str, object]], list[Row]]:
    """Recover attempts from chunk SMILES and rank by version-matched prediction."""
    required = {"spacehastenid", "reghash", "smiles", "dock_score", "dock_iteration"}
    missing = required - set(capabilities.data_columns)
    if missing:
        raise ValueError(f"data table lacks recovery columns: {sorted(missing)}")

    selections: dict[int, SelectionRound] = {}
    acquisition_rows: list[dict[str, object]] = []
    manifest_rows: list[Row] = []
    prior_ids: set[int] = set()
    for round_id, chunk_paths in paths:
        input_smiles: dict[int, str] = {}
        for path in chunk_paths:
            with path.open("rt", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        smiles, token = stripped.rsplit(maxsplit=1)
                        identifier = int(token)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"invalid docking input row {path}:{line_number}: {stripped!r}"
                        ) from error
                    if identifier in input_smiles:
                        raise ValueError(
                            f"duplicate selected ID {identifier} in round {round_id} docking inputs"
                        )
                    input_smiles[identifier] = smiles
        if not input_smiles:
            raise ValueError(f"round {round_id} docking inputs are empty")
        repeated = prior_ids & set(input_smiles)
        if repeated:
            preview = ", ".join(str(value) for value in sorted(repeated)[:5])
            raise ValueError(f"round {round_id} repeats previously selected IDs: {preview}")
        prior_ids.update(input_smiles)

        details = _selected_details(database, capabilities, tuple(sorted(input_smiles)))
        absent = sorted(set(input_smiles) - set(details))
        if absent:
            raise ValueError(f"round {round_id} selected IDs are absent from data: {absent[:5]}")
        for identifier, smiles in input_smiles.items():
            if str(details[identifier]["smiles"]).strip() != smiles.strip():
                raise ValueError(f"round {round_id} selected SMILES differs for ID {identifier}")

        versions = {
            int(value)
            for detail in details.values()
            if (value := detail.get("pred_version")) is not None
        }
        prediction_version_counts = (
            _prediction_version_counts(database, tuple(sorted(input_smiles)))
            if capabilities.has_predictions
            else {}
        )
        scored_count = sum(
            detail["dock_iteration"] == round_id and detail["dock_score"] is not None
            for detail in details.values()
        )
        inferred_versions = {
            version for version, count in prediction_version_counts.items() if count >= scored_count
        }
        if len(versions) == 1:
            model_version = str(next(iter(versions)))
            version_reason = "model version recovered from data.pred_version"
        elif len(inferred_versions) == 1:
            model_version = str(next(iter(inferred_versions)))
            version_reason = "model version inferred from dominant complete prediction coverage"
        else:
            model_version = None
            version_reason = "model version is absent or ambiguous"
        prediction_map = (
            fetch_predictions(database, capabilities, tuple(sorted(input_smiles)), model_version)
            if model_version is not None
            else {}
        )
        if len(prediction_map) == len(input_smiles):
            scores = {
                identifier: float(prediction_map[identifier]["pred_score"])
                for identifier in input_smiles
            }
            rank_source = "versioned_prediction_score_then_id"
            model_status = "reconstructed"
            model_reason = version_reason
        else:
            scores = {
                identifier: float(detail["pred_score"])
                for identifier, detail in details.items()
                if detail.get("pred_score") is not None
                and math.isfinite(float(detail["pred_score"]))
            }
            rank_source = "data_prediction_score_then_id" if len(scores) == len(details) else "id"
            model_status = "reconstructed" if model_version is not None else "unavailable"
            model_reason = (
                version_reason + "; rank recovered from data.pred_score"
                if model_version is not None
                else "complete version-matched prediction history unavailable"
            )
        ordered_ids = sorted(input_smiles, key=lambda value: (scores.get(value, math.inf), value))
        fields = (
            "spacehastenid",
            "model_version",
            "pred_score",
            "selection_source",
            "rank_source",
        )
        selections[round_id] = SelectionRound(
            round_id=round_id,
            attempts=tuple(ordered_ids),
            clusters=Counter(),
            atlas_ids=frozenset(),
            csv_columns=fields,
            model_version=model_version,
            model_version_status=model_status,
            model_version_reason=model_reason,
            selection_source="docking_input_chunks",
            rank_source=rank_source,
        )
        finite_scores = [scores[identifier] for identifier in ordered_ids if identifier in scores]
        acquisition_rows.append(
            {
                "round": round_id,
                "selected_attempts": len(ordered_ids),
                "selected_unique": len(set(ordered_ids)),
                "within_round_duplicate_attempts": 0,
                "selection_source": "docking_input_chunks",
                "rank_source": rank_source,
                "input_chunk_count": len(chunk_paths),
                "model_version": model_version,
                "model_version_status": model_status,
                "model_version_reason": model_reason,
                "pred_score_count": len(finite_scores),
                "pred_score_min": min(finite_scores) if finite_scores else None,
                "pred_score_max": max(finite_scores) if finite_scores else None,
                "pred_score_mean": (
                    sum(finite_scores) / len(finite_scores) if finite_scores else None
                ),
            }
        )
        for rank, identifier in enumerate(ordered_ids, 1):
            detail = details[identifier]
            score = detail["dock_score"]
            scored = detail["dock_iteration"] == round_id and score is not None
            if detail["dock_iteration"] not in (None, round_id):
                raise ValueError(
                    f"round {round_id} selected ID {identifier} has dock_iteration "
                    f"{detail['dock_iteration']}"
                )
            manifest_rows.append(
                {
                    "selection_id": f"recovered-round-{round_id}:{rank}",
                    "round": round_id,
                    "rank": rank,
                    "spacehastenid": identifier,
                    "reghash": detail["reghash"],
                    "smiles": detail["smiles"],
                    "outcome_status": "scored" if scored else "unresolved",
                    "dock_score": float(score) if scored else None,
                    "outcome_source": "data",
                    "data_dock_iteration": detail["dock_iteration"],
                    "data_dock_score": score,
                    "model_version": model_version,
                    "raw_mean": scores.get(identifier),
                    "selection_source": "docking_input_chunks",
                    "rank_source": rank_source,
                }
            )
    return selections, acquisition_rows, manifest_rows


def _selected_details(
    database: ReadOnlyDatabase,
    capabilities: Capabilities,
    identifiers: tuple[int, ...],
) -> dict[int, Row]:
    optional = [
        column for column in ("pred_score", "pred_version") if column in capabilities.data_columns
    ]
    fields = [
        "spacehastenid",
        "reghash",
        "smiles",
        "dock_score",
        "dock_iteration",
        *optional,
    ]
    result: dict[int, Row] = {}
    for batch in batches(identifiers, 900):
        placeholders = ",".join("?" for _ in batch)
        query = f"SELECT {','.join(fields)} FROM data WHERE spacehastenid IN ({placeholders})"
        for values in database.connection.execute(query, batch):
            row = dict(zip(fields, values, strict=True))
            result[int(row["spacehastenid"])] = row
    return result


def _prediction_version_counts(
    database: ReadOnlyDatabase,
    identifiers: tuple[int, ...],
) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for batch in batches(identifiers, 900):
        placeholders = ",".join("?" for _ in batch)
        query = (
            "SELECT model_version,COUNT(DISTINCT spacehastenid) FROM predictions "
            f"WHERE spacehastenid IN ({placeholders}) GROUP BY model_version"
        )
        counts.update(
            {
                int(version): int(count)
                for version, count in database.connection.execute(query, batch)
            }
        )
    return dict(counts)


__all__ = [
    "MODERN_HISTORY_TABLES",
    "has_modern_history",
    "recover_docking_input_acquisitions",
]
