"""Unit tests for uncertainty-aware docking acquisition."""

from __future__ import annotations

import math

import pytest

from spacehasten.core.acquisition import (
    AcquisitionCandidate,
    expected_improvement,
    lower_confidence_bound,
    select_normalized_penalized_batch,
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


def test_hard_cluster_cap_limits_contribution() -> None:
    candidates = [
        _candidate(1, -10.0, 0.0, 10),
        _candidate(2, -9.9, 0.0, 10),
        _candidate(3, -9.8, 0.0, 10),
        _candidate(4, -9.7, 0.0, 20),
        _candidate(5, -9.6, 0.0, 20),
    ]
    selected = select_penalized_batch(
        candidates,
        method="lcb",
        batch_size=4,
        cluster_lambda=0.0,
        cluster_cap=2,
        beta=0.0,
    )
    assert [row.candidate.spacehastenid for row in selected] == [1, 2, 4, 5]
    assert [row.cluster_count_before for row in selected] == [0, 1, 0, 1]


def test_hard_cluster_cap_rejects_insufficient_capacity() -> None:
    with pytest.raises(ValueError, match="permits only 2 of 3"):
        select_penalized_batch(
            [
                _candidate(1, -10.0, 0.0, 10),
                _candidate(2, -9.9, 0.0, 10),
                _candidate(3, -9.8, 0.0, 10),
            ],
            method="lcb",
            batch_size=3,
            cluster_lambda=0.0,
            cluster_cap=2,
            beta=0.0,
        )


def test_normalized_penalty_uses_live_frontier_scale() -> None:
    candidates = [
        _candidate(1, -10.2, 0.0, 10),
        _candidate(2, -10.1, 0.0, 10),
        _candidate(3, -10.0, 0.0, 20),
        _candidate(4, -9.9, 0.0, 20),
        _candidate(5, -9.8, 0.0, 30),
    ]

    selected, normalization = select_normalized_penalized_batch(
        candidates,
        method="ei",
        batch_size=2,
        cluster_alpha=1.0,
        hit_threshold=-9.7,
    )

    assert normalization.frontier_start_rank == 2
    assert normalization.frontier_stop_rank == 4
    assert normalization.candidate_count == 5
    assert normalization.batch_size == 2
    assert normalization.frontier_q10 == pytest.approx(-0.38)
    assert normalization.frontier_q90 == pytest.approx(-0.22)
    assert normalization.frontier_scale == pytest.approx(0.16)
    assert normalization.cluster_lambda == pytest.approx(0.16 / math.log(2.0))
    assert normalization.cluster_lambda * math.log(2.0) == pytest.approx(0.16)
    assert [row.candidate.spacehastenid for row in selected] == [1, 3]

    reversed_selection, reversed_normalization = select_normalized_penalized_batch(
        list(reversed(candidates)),
        method="ei",
        batch_size=2,
        cluster_alpha=1.0,
        hit_threshold=-9.7,
    )
    assert reversed_normalization == normalization
    assert [row.candidate.spacehastenid for row in reversed_selection] == [1, 3]


def test_normalized_penalty_rejects_an_empty_frontier() -> None:
    with pytest.raises(ValueError, match="requires more than 5 acquisition candidates"):
        select_normalized_penalized_batch(
            [_candidate(1, -10.0, 0.0, 10)],
            method="ei",
            batch_size=10,
            cluster_alpha=0.1,
            hit_threshold=-9.7,
        )


def test_normalized_penalty_rejects_a_flat_frontier() -> None:
    with pytest.raises(ValueError, match="positive finite acquisition frontier scale"):
        select_normalized_penalized_batch(
            [_candidate(sid, -10.0, 0.0, sid) for sid in range(1, 6)],
            method="ei",
            batch_size=2,
            cluster_alpha=0.1,
            hit_threshold=-9.7,
        )


def test_normalized_penalty_rejects_lambda_overflow() -> None:
    candidates = [
        _candidate(1, -1.0e308, 0.0, 1),
        _candidate(2, -8.0e307, 0.0, 2),
        _candidate(3, -6.0e307, 0.0, 3),
        _candidate(4, -4.0e307, 0.0, 4),
        _candidate(5, -2.0e307, 0.0, 5),
    ]
    with pytest.raises(ValueError, match="non-finite lambda"):
        select_normalized_penalized_batch(
            candidates,
            method="ei",
            batch_size=2,
            cluster_alpha=1.0e308,
            hit_threshold=0.0,
        )


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
