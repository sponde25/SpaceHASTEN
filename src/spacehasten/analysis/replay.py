"""Scale normalization used by reusable EI replay studies."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Int64Array = npt.NDArray[np.int64]


def _validate_frontier_inputs(
    base_scores: FloatArray,
    identifiers: Int64Array,
    batch_size: int,
) -> tuple[FloatArray, Int64Array]:
    scores = np.asarray(base_scores, dtype=np.float64)
    ids = np.asarray(identifiers, dtype=np.int64)
    if scores.ndim != 1 or ids.ndim != 1:
        raise ValueError("base_scores and identifiers must be one-dimensional")
    if len(scores) != len(ids):
        raise ValueError("base_scores and identifiers must have equal length")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not np.isfinite(scores).all():
        raise ValueError("base_scores must be finite")
    return scores, ids


def frontier_scale(
    base_scores: FloatArray,
    identifiers: Int64Array,
    batch_size: int,
) -> dict[str, float | int]:
    """Summarize the historical primary and sensitivity EI frontier windows."""
    scores, ids = _validate_frontier_inputs(base_scores, identifiers, batch_size)
    order = np.lexsort((ids, scores))

    def summarize(start_multiplier: float, stop_multiplier: float) -> dict[str, float | int]:
        start = int(start_multiplier * batch_size)
        stop = min(int(stop_multiplier * batch_size), len(order))
        if start >= stop:
            raise ValueError(
                f"candidate pool cannot fill the requested EI frontier window [{start + 1}, {stop}]"
            )
        q10, median, q90 = np.quantile(scores[order[start:stop]], [0.10, 0.50, 0.90])
        return {
            "start_rank": start + 1,
            "stop_rank": stop,
            "q10": float(q10),
            "median": float(median),
            "q90": float(q90),
            "scale": float(q90 - q10),
        }

    primary = summarize(0.5, 2.0)
    sensitivity = summarize(1.0, 3.0)
    primary_scale = float(primary["scale"])
    sensitivity_scale = float(sensitivity["scale"])
    if not math.isfinite(primary_scale) or primary_scale <= 0:
        raise ValueError("primary EI frontier scale is not positive")
    return {
        "primary_start_rank": primary["start_rank"],
        "primary_stop_rank": primary["stop_rank"],
        "primary_q10": primary["q10"],
        "primary_median": primary["median"],
        "primary_q90": primary["q90"],
        "primary_scale": primary_scale,
        "sensitivity_start_rank": sensitivity["start_rank"],
        "sensitivity_stop_rank": sensitivity["stop_rank"],
        "sensitivity_q10": sensitivity["q10"],
        "sensitivity_median": sensitivity["median"],
        "sensitivity_q90": sensitivity["q90"],
        "sensitivity_scale": sensitivity_scale,
        "scale_ratio_sensitivity_over_primary": sensitivity_scale / primary_scale,
    }
