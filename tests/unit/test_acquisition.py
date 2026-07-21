"""Unit tests for uncertainty-aware docking acquisition."""

from __future__ import annotations

import math

import pytest

from spacehasten.core.acquisition import (
    AcquisitionCandidate,
    expected_improvement,
    lower_confidence_bound,
    select_penalized_batch,
)


def _candidate(
    sid: int,
    mean: float,
    uncertainty: float,
    clusterid: int | None,
) -> AcquisitionCandidate:
    return AcquisitionCandidate(
        smiles=f"C{'C' * sid}",
        spacehastenid=sid,
        pred_score=mean,
        epistemic_std=uncertainty,
        model_version=1,
        clusterid=clusterid,
    )


def test_lower_confidence_bound_minimizes_score() -> None:
    assert lower_confidence_bound(-8.0, 1.5, beta=2.0) == pytest.approx(-11.0)


def test_expected_improvement_uses_target_specific_threshold() -> None:
    threshold = -9.7
    assert expected_improvement(-10.0, 0.0, threshold) == pytest.approx(0.3)
    assert expected_improvement(-9.0, 0.0, threshold) == 0.0
    assert expected_improvement(threshold, 1.0, threshold) == pytest.approx(
        1.0 / math.sqrt(2.0 * math.pi)
    )


def test_zero_penalty_reproduces_base_lcb_ranking() -> None:
    candidates = [
        _candidate(1, -10.0, 0.0, 10),
        _candidate(2, -9.9, 0.0, 10),
        _candidate(3, -9.8, 0.0, 20),
    ]
    selected = select_penalized_batch(
        candidates,
        method="lcb",
        batch_size=3,
        cluster_lambda=0.0,
        beta=0.0,
    )
    assert [row.candidate.spacehastenid for row in selected] == [1, 2, 3]


def test_dynamic_cluster_penalty_changes_batch_order() -> None:
    candidates = [
        _candidate(1, -10.0, 0.0, 10),
        _candidate(2, -9.9, 0.0, 10),
        _candidate(3, -9.8, 0.0, 20),
    ]
    selected = select_penalized_batch(
        candidates,
        method="lcb",
        batch_size=3,
        cluster_lambda=1.0,
        beta=0.0,
    )

    assert [row.candidate.spacehastenid for row in selected] == [1, 3, 2]
    assert [row.cluster_count_before for row in selected] == [0, 0, 1]
    assert selected[2].cluster_penalty == pytest.approx(math.log(2.0))
    assert selected[2].penalized_score == pytest.approx(-9.9 + math.log(2.0))


def test_positive_penalty_requires_complete_cluster_assignments() -> None:
    with pytest.raises(ValueError, match="lack cluster assignments"):
        select_penalized_batch(
            [_candidate(1, -10.0, 0.2, None)],
            method="lcb",
            batch_size=1,
            cluster_lambda=1.0,
        )


def test_expected_improvement_requires_threshold() -> None:
    with pytest.raises(ValueError, match="hit_threshold"):
        select_penalized_batch(
            [_candidate(1, -10.0, 0.2, 10)],
            method="ei",
            batch_size=1,
            cluster_lambda=0.0,
        )
