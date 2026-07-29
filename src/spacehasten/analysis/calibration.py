"""Capability-conditional prediction calibration diagnostics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .database import ReadOnlyDatabase, fetch_predictions
from .metrics import is_scored_in_round
from .models import Capabilities, Row, SelectionRound

PROBABILITY_BINS = 10
INTERVAL_Z = (
    (50, 0.6744897501960817),
    (80, 1.2815515655446004),
    (90, 1.6448536269514722),
    (95, 1.959963984540054),
)


def calibration_metrics(
    database: ReadOnlyDatabase,
    capabilities: Capabilities,
    rows: dict[int, Row],
    selections: dict[int, SelectionRound],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calculate selected-compound diagnostics and reliability-curve records by round."""
    metrics: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    for round_id, selection in sorted(selections.items()):
        selected = set(selection.attempts)
        observed = {
            identifier: float(rows[identifier]["dock_score"])
            for identifier in selected
            if is_scored_in_round(rows.get(identifier), round_id)
        }
        record = _base_record(round_id, selection, len(selected), len(observed))
        if not capabilities.has_predictions:
            record.update(_unavailable("predictions table absent"))
            metrics.append(record)
            continue
        required = {"spacehastenid", "model_version", "pred_score"}
        missing = sorted(required - set(capabilities.predictions_columns))
        if missing:
            record.update(_unavailable(f"predictions table lacks required columns: {missing}"))
            metrics.append(record)
            continue
        if selection.model_version is None:
            record.update(
                _unavailable(selection.model_version_reason or "model version unavailable")
            )
            metrics.append(record)
            continue
        predictions = fetch_predictions(
            database, capabilities, tuple(sorted(selected)), selection.model_version
        )
        matched = sorted(set(observed) & set(predictions))
        record["matched_predictions"] = len(matched)
        record["coverage"] = _rate(len(matched), len(selected))
        if not matched:
            record.update(
                _unavailable("no selected observed outcomes match predictions for model_version")
            )
            metrics.append(record)
            continue
        predicted = [float(predictions[identifier]["pred_score"]) for identifier in matched]
        actual = [observed[identifier] for identifier in matched]
        errors = [
            prediction - outcome for prediction, outcome in zip(predicted, actual, strict=True)
        ]
        record.update(
            {
                "status": "available",
                "reason": None,
                "bias": sum(errors) / len(errors),
                "mae": sum(abs(error) for error in errors) / len(errors),
                "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
                "pearson_correlation": _pearson(predicted, actual),
                "spearman_correlation": _pearson(_ranks(predicted), _ranks(actual)),
                "observed_hit_rate": sum(value <= threshold for value in actual) / len(actual),
            }
        )
        absolute_errors = [abs(error) for error in errors]
        epistemic = _numeric_column(predictions, matched, "epistemic_std")
        record["epistemic_uncertainty_abs_error_correlation"] = (
            _pearson(epistemic, absolute_errors) if epistemic is not None else None
        )
        total_std, total_reason = _total_std(predictions, matched)
        if total_std is None:
            record.update(_probability_unavailable(total_reason))
            metrics.append(record)
            continue
        probabilities = [
            _hit_probability(mean, std, threshold)
            for mean, std in zip(predicted, total_std, strict=True)
        ]
        hits = [float(value <= threshold) for value in actual]
        record.update(_probability_metrics(probabilities, hits))
        record.update(_interval_metrics(predicted, actual, total_std))
        record["total_uncertainty_abs_error_correlation"] = _pearson(total_std, absolute_errors)
        curve.extend(_curve_rows(round_id, selection.model_version, probabilities, hits))
        metrics.append(record)
    return metrics, curve


def _base_record(
    round_id: int, selection: SelectionRound, selected: int, observed: int
) -> dict[str, Any]:
    return {
        "round": round_id,
        "model_version": selection.model_version,
        "mean_source": "predictions.pred_score",
        "probability_source": "raw_gaussian_pred_score_total_std",
        "selected_attempts": len(selection.attempts),
        "selected_unique": selected,
        "observed_outcomes": observed,
        "matched_predictions": 0,
        "coverage": 0.0 if selected else None,
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "bias": None,
        "mae": None,
        "rmse": None,
        "pearson_correlation": None,
        "spearman_correlation": None,
        "observed_hit_rate": None,
        "epistemic_uncertainty_abs_error_correlation": None,
        **_probability_unavailable(reason),
    }


def _probability_unavailable(reason: str | None) -> dict[str, Any]:
    return {
        "probability_status": "unavailable",
        "probability_reason": reason,
        "interval_status": "unavailable",
        "interval_reason": reason,
        "predicted_hit_rate": None,
        "brier_score": None,
        "clipped_log_loss": None,
        "ece": None,
        "total_uncertainty_abs_error_correlation": None,
        **{f"interval_{level}_coverage": None for level, _ in INTERVAL_Z},
    }


def _total_std(
    predictions: dict[int, Row], matched: Sequence[int]
) -> tuple[list[float] | None, str | None]:
    direct = _numeric_column(predictions, matched, "total_std")
    if direct is not None:
        return direct, None
    epistemic = _numeric_column(predictions, matched, "epistemic_std")
    aleatoric = _numeric_column(predictions, matched, "aleatoric_std")
    if epistemic is not None and aleatoric is not None:
        return [math.hypot(epi, ale) for epi, ale in zip(epistemic, aleatoric, strict=True)], None
    return None, "total_std absent and epistemic_std plus aleatoric_std are not both available"


def _numeric_column(
    predictions: dict[int, Row], identifiers: Sequence[int], column: str
) -> list[float] | None:
    values = [predictions[identifier].get(column) for identifier in identifiers]
    if any(value is None for value in values):
        return None
    try:
        numeric = [float(str(value)) for value in values]
    except (TypeError, ValueError):
        return None
    return numeric if all(value >= 0 and math.isfinite(value) for value in numeric) else None


def _hit_probability(mean: float, std: float, threshold: float) -> float:
    if std == 0:
        return float(mean <= threshold)
    return 0.5 * (1 + math.erf((threshold - mean) / (std * math.sqrt(2))))


def _probability_metrics(probabilities: Sequence[float], hits: Sequence[float]) -> dict[str, Any]:
    clipped = [min(max(probability, 1e-15), 1 - 1e-15) for probability in probabilities]
    return {
        "probability_status": "available",
        "probability_reason": None,
        "interval_status": "available",
        "interval_reason": None,
        "predicted_hit_rate": sum(probabilities) / len(probabilities),
        "brier_score": sum(
            (probability - hit) ** 2 for probability, hit in zip(probabilities, hits, strict=True)
        )
        / len(hits),
        "clipped_log_loss": -sum(
            hit * math.log(probability) + (1 - hit) * math.log(1 - probability)
            for probability, hit in zip(clipped, hits, strict=True)
        )
        / len(hits),
        "ece": _ece(probabilities, hits),
    }


def _interval_metrics(
    means: Sequence[float], actual: Sequence[float], std: Sequence[float]
) -> dict[str, float]:
    return {
        f"interval_{level}_coverage": (
            sum(
                mean - z * uncertainty <= outcome <= mean + z * uncertainty
                for mean, outcome, uncertainty in zip(means, actual, std, strict=True)
            )
            / len(actual)
        )
        for level, z in INTERVAL_Z
    }


def _curve_rows(
    round_id: int,
    model_version: str,
    probabilities: Sequence[float],
    hits: Sequence[float],
) -> list[dict[str, Any]]:
    bins: dict[int, list[tuple[float, float]]] = {}
    for probability, hit in zip(probabilities, hits, strict=True):
        index = min(int(probability * PROBABILITY_BINS), PROBABILITY_BINS - 1)
        bins.setdefault(index, []).append((probability, hit))
    return [
        {
            "round": round_id,
            "model_version": model_version,
            "probability_source": "raw_gaussian_pred_score_total_std",
            "bin_lower": index / PROBABILITY_BINS,
            "bin_upper": (index + 1) / PROBABILITY_BINS,
            "count": len(values),
            "mean_predicted_probability": sum(value[0] for value in values) / len(values),
            "observed_hit_fraction": sum(value[1] for value in values) / len(values),
        }
        for index, values in sorted(bins.items())
    ]


def _ece(probabilities: Sequence[float], hits: Sequence[float]) -> float:
    return sum(
        len(values)
        / len(probabilities)
        * abs(
            sum(value[0] for value in values) / len(values)
            - sum(value[1] for value in values) / len(values)
        )
        for values in _curve_values(probabilities, hits).values()
    )


def _curve_values(
    probabilities: Sequence[float], hits: Sequence[float]
) -> dict[int, list[tuple[float, float]]]:
    bins: dict[int, list[tuple[float, float]]] = {}
    for probability, hit in zip(probabilities, hits, strict=True):
        index = min(int(probability * PROBABILITY_BINS), PROBABILITY_BINS - 1)
        bins.setdefault(index, []).append((probability, hit))
    return bins


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_sum = sum((x - left_mean) ** 2 for x in left)
    right_sum = sum((y - right_mean) ** 2 for y in right)
    return numerator / math.sqrt(left_sum * right_sum) if left_sum and right_sum else None


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[start][1]:
            stop += 1
        rank = (start + stop - 1) / 2 + 1
        for index, _ in ordered[start:stop]:
            ranks[index] = rank
        start = stop
    return ranks


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
