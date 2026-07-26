"""Acquisition normalization and run-level metric tables."""

from __future__ import annotations

import csv
import logging
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem
from tqdm import tqdm  # type: ignore[import-untyped]

from .chemistry import distribution, diversity, family_labels
from .models import AnalysisConfig, AtlasResolution, Row, SelectionRound

LOG = logging.getLogger(__name__)
ID_COLUMNS = ("spacehastenid", "id", "compound_id")
ATLAS_COLUMNS = ("cluster_atlas_id", "atlas_id", "atlasid")
INVARIANT_ACQUISITION_FIELDS = (
    "method",
    "model_version",
    "candidate_count",
    "batch_size",
    "cluster_atlas_id",
    "cluster_atlas_version",
    "cluster_alpha",
    "cluster_lambda",
    "cluster_cap",
    "ei_hit_threshold",
    "ei_xi",
    "frontier_start_rank",
    "frontier_stop_rank",
    "frontier_q10",
    "frontier_q90",
    "frontier_scale",
)
SERIES_ACQUISITION_FIELDS = (
    "pred_score",
    "epistemic_std",
    "base_score",
    "cluster_count_before",
    "cluster_penalty",
    "penalized_score",
)


def normalize_acquisitions(
    paths: tuple[tuple[int, Path], ...],
) -> tuple[dict[int, SelectionRound], list[dict[str, Any]]]:
    selections: dict[int, SelectionRound] = {}
    output: list[dict[str, Any]] = []
    for round_id, path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            id_column = next((column for column in ID_COLUMNS if column in fields), None)
            if id_column is None:
                LOG.warning("Skipping %s: no recognized identifier column", path)
                continue
            attempts: list[int] = []
            clusters: Counter[str] = Counter()
            atlas_ids: set[str] = set()
            invariant_values: dict[str, set[str]] = {
                field: set() for field in INVARIANT_ACQUISITION_FIELDS
            }
            series_values: dict[str, list[float]] = {
                field: [] for field in SERIES_ACQUISITION_FIELDS
            }
            for row in reader:
                try:
                    identifier = int(str(row[id_column]))
                except (TypeError, ValueError):
                    continue
                attempts.append(identifier)
                cluster = row.get("clusterid") or row.get("cluster_id")
                if cluster:
                    clusters[str(cluster)] += 1
                atlas_id = next(
                    (row.get(column) for column in ATLAS_COLUMNS if row.get(column)), None
                )
                if atlas_id:
                    atlas_ids.add(str(atlas_id))
                for field, invariant_set in invariant_values.items():
                    value = row.get(field)
                    if value not in (None, ""):
                        invariant_set.add(str(value))
                for field, series in series_values.items():
                    value = row.get(field)
                    if value in (None, ""):
                        continue
                    try:
                        series.append(float(str(value)))
                    except ValueError:
                        LOG.warning("Ignoring nonnumeric %s=%r in %s", field, value, path)
        model_versions = invariant_values["model_version"]
        if len(model_versions) == 1:
            model_version = next(iter(model_versions))
            model_version_status, model_version_reason = "available", None
        elif not model_versions:
            model_version = None
            model_version_status, model_version_reason = "unavailable", "model_version absent"
        else:
            model_version = None
            model_version_status = "inconsistent"
            model_version_reason = "acquisition rows contain multiple model_version values"
        selection = SelectionRound(
            round_id,
            tuple(attempts),
            clusters,
            frozenset(atlas_ids),
            fields,
            model_version,
            model_version_status,
            model_version_reason,
        )
        selections[round_id] = selection
        record: dict[str, Any] = {
            "round": round_id,
            "selected_attempts": len(attempts),
            "selected_unique": len(set(attempts)),
            "within_round_duplicate_attempts": len(attempts) - len(set(attempts)),
            "csv_columns": ";".join(fields),
            **prefixed("selected_cluster", distribution(clusters)),
        }
        for field, invariant_set in invariant_values.items():
            if len(invariant_set) == 1:
                record[field] = next(iter(invariant_set))
            elif len(invariant_set) > 1:
                record[field] = None
                record[f"{field}_status"] = "inconsistent"
        for field, series in series_values.items():
            if not series:
                continue
            summary = score_summary(series)
            record.update(
                {f"{field}_{name.removeprefix('score_')}": value for name, value in summary.items()}
            )
            record[f"{field}_min"] = min(series)
            record[f"{field}_max"] = max(series)
        penalties = series_values["cluster_penalty"]
        if penalties:
            record["cluster_penalty_nonzero_fraction"] = rate(
                sum(value != 0 for value in penalties), len(penalties)
            )
        candidate_values = invariant_values["candidate_count"]
        if len(candidate_values) == 1:
            try:
                candidate_count = int(float(next(iter(candidate_values))))
            except ValueError:
                candidate_count = 0
            record["selected_candidate_fraction"] = rate(len(attempts), candidate_count)
        output.append(record)
    return selections, output


def rounds(rows: dict[int, Row], selections: dict[int, SelectionRound]) -> list[int]:
    if selections:
        return sorted(selections)
    return sorted(
        {int(row["dock_iteration"]) for row in rows.values() if int(row["dock_iteration"]) > 0}
    )


def analysis_ids(rows: dict[int, Row], selections: dict[int, SelectionRound]) -> tuple[int, ...]:
    if selections:
        return tuple(
            sorted(
                {
                    identifier
                    for selection in selections.values()
                    for identifier in selection.attempts
                }
            )
        )
    return tuple(sorted(rows))


def round_metrics(
    rows: dict[int, Row],
    selections: dict[int, SelectionRound],
    round_ids: Sequence[int],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    selected_attempts = scored_attempts = hit_attempts = 0
    selected_unique: set[int] = set()
    scored_unique: set[int] = set()
    hit_unique: set[int] = set()
    prior: set[int] = set()
    for round_id in round_ids:
        selected = attempts(round_id, rows, selections)
        scored = [
            identifier
            for identifier in selected
            if is_scored_in_round(rows.get(identifier), round_id)
        ]
        hits = [
            identifier
            for identifier in scored
            if float(rows[identifier]["dock_score"]) <= threshold
        ]
        selected_attempts += len(selected)
        scored_attempts += len(scored)
        hit_attempts += len(hits)
        selected_unique.update(selected)
        scored_unique.update(scored)
        hit_unique.update(hits)
        within_duplicates = len(selected) - len(set(selected))
        cross_reused = sum(identifier in prior for identifier in selected)
        prior.update(selected)
        selected_ci, scored_ci = wilson(len(hits), len(selected)), wilson(len(hits), len(scored))
        scores = [float(rows[identifier]["dock_score"]) for identifier in scored]
        common = {
            "round": round_id,
            "selected": len(selected),
            "scored": len(scored),
            "missing": len(selected) - len(scored),
            "hits": len(hits),
            "within_round_duplicate_attempts": within_duplicates,
            "cross_round_reused_attempts": cross_reused,
            "cumulative_selected": selected_attempts,
            "cumulative_scored": scored_attempts,
            "cumulative_hits": hit_attempts,
            "cumulative_selected_unique": len(selected_unique),
            "cumulative_scored_unique": len(scored_unique),
            "cumulative_hits_unique": len(hit_unique),
        }
        metrics.append(
            {
                **common,
                "hit_rate_selected": rate(len(hits), len(selected)),
                "hit_rate_selected_ci_low": selected_ci[0],
                "hit_rate_selected_ci_high": selected_ci[1],
                "hit_rate_scored": rate(len(hits), len(scored)),
                "hit_rate_scored_ci_low": scored_ci[0],
                "hit_rate_scored_ci_high": scored_ci[1],
                "docking_success": rate(len(scored), len(selected)),
                **score_summary(scores),
            }
        )
        coverage.append(
            {
                **common,
                "source": "acquisition_csv" if round_id in selections else "database",
                "status": "complete" if len(scored) == len(selected) else "partial",
                "reason": None
                if len(scored) == len(selected)
                else "selected IDs missing a positive-round docking outcome",
            }
        )
    return metrics, coverage


def cutoff_metrics(
    rows: dict[int, Row],
    selections: dict[int, SelectionRound],
    round_ids: Sequence[int],
    cutoffs: Sequence[float],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for cutoff in sorted(set(cutoffs)):
        selected = scored = 0
        for round_id in round_ids:
            attempted = attempts(round_id, rows, selections)
            scored_ids = [
                identifier
                for identifier in attempted
                if is_scored_in_round(rows.get(identifier), round_id)
            ]
            selected += len(attempted)
            scored += len(scored_ids)
            hits = sum(float(rows[identifier]["dock_score"]) <= cutoff for identifier in scored_ids)
            result.append(
                {
                    "round": round_id,
                    "cutoff": cutoff,
                    "selected": selected,
                    "scored": scored,
                    "hits": hits,
                    "hit_rate_selected": rate(hits, selected),
                    "hit_rate_scored": rate(hits, scored),
                }
            )
    return result


def budget_curve(
    rows: dict[int, Row],
    selections: dict[int, SelectionRound],
    round_ids: Sequence[int],
    threshold: float,
    points: int = 200,
) -> list[dict[str, Any]]:
    total_attempts = sum(len(attempts(round_id, rows, selections)) for round_id in round_ids)
    step = max(1, total_attempts // points)
    selected = scored = hits = 0
    result: list[dict[str, Any]] = []
    for round_id in round_ids:
        round_attempts = attempts(round_id, rows, selections)
        for round_rank, identifier in enumerate(round_attempts, start=1):
            selected += 1
            row = rows.get(identifier)
            if is_scored_in_round(row, round_id):
                assert row is not None
                scored += 1
                hits += float(row["dock_score"]) <= threshold
            if selected % step == 0 or round_rank == len(round_attempts):
                result.append(
                    {
                        "round": round_id,
                        "round_rank": round_rank,
                        "selected_budget": selected,
                        "cumulative_scored": scored,
                        "cumulative_hits": hits,
                        "hit_rate_selected": rate(hits, selected),
                        "hit_rate_scored": rate(hits, scored),
                    }
                )
    return result


def score_distribution(
    rows: dict[int, Row],
    selections: dict[int, SelectionRound],
    round_ids: Sequence[int],
    quantile_points: int = 101,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for round_id in round_ids:
        scores = sorted(
            float(rows[identifier]["dock_score"])
            for identifier in attempts(round_id, rows, selections)
            if is_scored_in_round(rows.get(identifier), round_id)
        )
        if not scores:
            continue
        for index in range(quantile_points):
            quantile_value = index / (quantile_points - 1)
            result.append(
                {
                    "round": round_id,
                    "quantile": quantile_value,
                    "score": quantile(scores, quantile_value),
                    "scored": len(scores),
                }
            )
    return result


def family_metrics(
    rows: dict[int, Row],
    selections: dict[int, SelectionRound],
    round_ids: Sequence[int],
    atlas: AtlasResolution,
    atlas_map: dict[int, str],
    config: AnalysisConfig,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for round_id in tqdm(round_ids, desc="chemistry metrics", unit="round"):
        selected = attempts(round_id, rows, selections)
        valid: list[Chem.Mol] = []
        typed: Counter[str] = Counter()
        generic: Counter[str] = Counter()
        for identifier in selected:
            row = rows.get(identifier)
            molecule = Chem.MolFromSmiles(str(row["smiles"])) if row and row["smiles"] else None
            if molecule is None:
                continue
            valid.append(molecule)
            typed_label, generic_label = family_labels(molecule)
            typed[typed_label] += 1
            generic[generic_label] += 1
        record: dict[str, Any] = {
            "round": round_id,
            "chemistry_status": "available" if valid else "unavailable",
            "chemistry_reason": None if valid else "no valid selected SMILES",
            "chemistry_coverage": rate(len(valid), len(selected)),
            "valid_smiles": len(valid),
            "invalid_or_missing_smiles": len(selected) - len(valid),
            "atlas_status": atlas.status,
            "atlas_reason": atlas.reason,
        }
        record.update(prefixed("typed", distribution(typed)))
        record.update(prefixed("generic", distribution(generic)))
        record.update(diversity(valid, config))
        if atlas.atlas_id is not None:
            assignments = Counter(
                atlas_map[identifier] for identifier in selected if identifier in atlas_map
            )
            record.update(
                prefixed(
                    "atlas",
                    {
                        "assignment_coverage": rate(sum(assignments.values()), len(selected)),
                        **distribution(assignments),
                    },
                )
            )
        result.append(record)
    return result


def attempts(
    round_id: int, rows: dict[int, Row], selections: dict[int, SelectionRound]
) -> tuple[int, ...]:
    if round_id in selections:
        return selections[round_id].attempts
    return tuple(
        sorted(identifier for identifier, row in rows.items() if row["dock_iteration"] == round_id)
    )


def is_scored(row: Row | None) -> bool:
    return (
        row is not None
        and row["dock_iteration"] is not None
        and int(row["dock_iteration"]) > 0
        and row["dock_score"] is not None
    )


def is_scored_in_round(row: Row | None, round_id: int) -> bool:
    if not is_scored(row):
        return False
    assert row is not None
    return int(row["dock_iteration"]) == round_id


def score_summary(scores: Sequence[float]) -> dict[str, float | None]:
    if not scores:
        return {f"score_{name}": None for name in ("mean", "median", "q05", "q25", "q75", "q95")}
    ordered = sorted(scores)
    return {
        "score_mean": sum(scores) / len(scores),
        "score_median": quantile(ordered, 0.5),
        "score_q05": quantile(ordered, 0.05),
        "score_q25": quantile(ordered, 0.25),
        "score_q75": quantile(ordered, 0.75),
        "score_q95": quantile(ordered, 0.95),
    }


def quantile(values: Sequence[float], quantile_value: float) -> float:
    position = (len(values) - 1) * quantile_value
    low, high = math.floor(position), math.ceil(position)
    return (
        values[low]
        if low == high
        else values[low] + (values[high] - values[low]) * (position - low)
    )


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def wilson(hits: int, total: int) -> tuple[float | None, float | None]:
    if not total:
        return None, None
    z = 1.959963984540054
    proportion = hits / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return center - margin, center + margin


def prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}
