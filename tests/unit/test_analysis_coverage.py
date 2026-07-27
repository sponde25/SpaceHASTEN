from __future__ import annotations

import math

import pytest

from spacehasten.analysis.coverage import coverage_depth, coverage_summary, gini, hit_depth_bins


def test_productive_coverage_metrics() -> None:
    counts = [1, 10, 20, 30]
    summary = coverage_summary(counts)
    assert summary["hits"] == 61
    assert summary["occupied_regions"] == 4
    assert summary["regions_ge_10"] == 3
    assert summary["regions_ge_20"] == 2
    assert summary["u20"] == 51
    assert summary["o20"] == 10
    assert summary["broad_q2"] == pytest.approx(61**2 / (1 + 100 + 400 + 900))
    assert summary["hhi"] == pytest.approx(1 / summary["broad_q2"])
    assert summary["gini"] == pytest.approx(gini(counts))
    assert math.isclose(summary["top10_hit_share"], 1.0)

    assert coverage_depth(counts, max_threshold=3) == [
        {"threshold": 1, "regions": 4},
        {"threshold": 2, "regions": 3},
        {"threshold": 3, "regions": 3},
    ]
    assert hit_depth_bins(counts) == [
        {"depth_bin": "1-4", "regions": 1, "hits": 1},
        {"depth_bin": "5-9", "regions": 0, "hits": 0},
        {"depth_bin": "10-19", "regions": 1, "hits": 10},
        {"depth_bin": "20-39", "regions": 2, "hits": 50},
        {"depth_bin": "40-99", "regions": 0, "hits": 0},
        {"depth_bin": ">=100", "regions": 0, "hits": 0},
    ]


def test_empty_productive_coverage() -> None:
    assert coverage_summary([]) == {
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
        "regions_ge_5": 0,
        "regions_ge_10": 0,
        "regions_ge_20": 0,
        "regions_ge_40": 0,
    }
    assert coverage_depth([], max_threshold=2) == [
        {"threshold": 1, "regions": 0},
        {"threshold": 2, "regions": 0},
    ]
    assert gini([]) == 0.0


@pytest.mark.parametrize("counts", [[0, 1], [-1, 2]])
def test_coverage_rejects_invalid_counts(counts: list[int]) -> None:
    with pytest.raises(ValueError):
        coverage_summary(counts)
