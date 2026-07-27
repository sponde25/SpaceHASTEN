"""Reusable productive-coverage and concentration metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]


def coverage_summary(
    counts: Sequence[int] | IntArray, *, useful_target: int = 20
) -> dict[str, Any]:
    """Summarize positive hit counts across regions."""
    values = _counts(counts)
    if len(values) == 0:
        return {
            "hits": 0,
            "occupied_regions": 0,
            "broad_q0": 0,
            "broad_q1": 0.0,
            "broad_q2": 0.0,
            "hhi": 0.0,
            "gini": 0.0,
            "largest_region_share": 0.0,
            "top10_hit_share": 0.0,
            "u20": 0,
            "o20": 0,
            "max_region_hits": 0,
            **{f"regions_ge_{threshold}": 0 for threshold in (5, 10, 20, 40)},
        }
    probabilities = values / values.sum()
    return {
        "hits": int(values.sum()),
        "occupied_regions": len(values),
        "broad_q0": len(values),
        "broad_q1": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
        "broad_q2": float(1.0 / np.sum(probabilities**2)),
        "hhi": float(np.sum(probabilities**2)),
        "gini": gini(values),
        "largest_region_share": float(probabilities.max()),
        "top10_hit_share": float(np.sort(probabilities)[-10:].sum()),
        "u20": int(np.minimum(values, useful_target).sum()),
        "o20": int(np.maximum(values - useful_target, 0).sum()),
        "max_region_hits": int(values.max()),
        **{
            f"regions_ge_{threshold}": int((values >= threshold).sum())
            for threshold in (5, 10, 20, 40)
        },
    }


def coverage_depth(
    counts: Sequence[int] | IntArray,
    *,
    max_threshold: int = 100,
) -> list[dict[str, int]]:
    values = _counts(counts)
    if max_threshold < 1:
        raise ValueError("max_threshold must be positive")
    return [
        {"threshold": threshold, "regions": int((values >= threshold).sum())}
        for threshold in range(1, max_threshold + 1)
    ]


def hit_depth_bins(counts: Sequence[int] | IntArray) -> list[dict[str, int | str]]:
    values = _counts(counts)
    bins = (
        ("1-4", 1, 4),
        ("5-9", 5, 9),
        ("10-19", 10, 19),
        ("20-39", 20, 39),
        ("40-99", 40, 99),
        (">=100", 100, math.inf),
    )
    result: list[dict[str, int | str]] = []
    for label, lower, upper in bins:
        selected = values[(values >= lower) & (values <= upper)]
        result.append(
            {
                "depth_bin": label,
                "regions": len(selected),
                "hits": int(selected.sum()),
            }
        )
    return result


def gini(counts: Sequence[int] | IntArray) -> float:
    values = np.sort(_counts(counts).astype(np.float64))
    if len(values) == 0:
        return 0.0
    ranks = np.arange(1, len(values) + 1, dtype=np.float64)
    return float(np.sum((2 * ranks - len(values) - 1) * values) / (len(values) * values.sum()))


def _counts(counts: Sequence[int] | IntArray) -> IntArray:
    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("region counts must be a one-dimensional sequence")
    if np.any(values <= 0):
        raise ValueError("region counts must be positive")
    return values


__all__ = ["coverage_depth", "coverage_summary", "gini", "hit_depth_bins"]
