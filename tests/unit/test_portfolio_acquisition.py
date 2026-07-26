from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from spacehasten.config.acquisition import CalibrationConfig, PortfolioAcquisitionPolicy
from spacehasten.core.portfolio_acquisition import (
    CandidatePool,
    candidate_pool_digest,
    cumulative_reward,
    gaussian_expected_improvement,
    select_portfolio_batch,
)


def _policy(**extra: object) -> PortfolioAcquisitionPolicy:
    data: dict[str, object] = {
        "quality": {"kind": "gaussian_hit_ei", "hit_threshold": -9.7},
        "reward": {
            "kind": "piecewise_linear",
            "breakpoints": [1, 5, 20],
            "slopes": [0.25, 1, 2],
            "weight": 0.1,
        },
    }
    data.update(extra)
    return PortfolioAcquisitionPolicy.model_validate(data)


def _pool(
    ids=(1, 2, 3),
    means=(-10.0, -9.9, -9.8),
    clusters=(1, 1, 2),
    versions=(1, 1, 1),
) -> CandidatePool:
    size = len(ids)
    if len(means) != size:
        means = (-10.0,) * size
    if len(versions) != size:
        versions = (1,) * size
    return CandidatePool(
        np.array(ids),
        np.array(means),
        np.ones(size),
        np.array(clusters),
        np.array(versions),
    )


def test_exact_tiers_and_xi() -> None:
    policy = _policy()
    assert cumulative_reward(np.array([1.0, 5.0, 20.0, 30.0]), policy.reward).tolist() == [
        0.25,
        4.25,
        34.25,
        34.25,
    ]
    assert cumulative_reward(5.0, policy.reward) - cumulative_reward(1.0, policy.reward) == 4.0
    assert (
        gaussian_expected_improvement(np.array([-10.0]), np.array([1.0]), -9.7, 0.5)[0]
        < gaussian_expected_improvement(np.array([-10.0]), np.array([1.0]), -9.7)[0]
    )


def test_candidate_pool_digest_is_layout_independent_and_input_sensitive() -> None:
    pool = _pool(ids=(1, 2), means=(-10.0, -9.0), clusters=(1, 2))
    layout_variant = CandidatePool(
        np.asfortranarray(pool.ids.reshape(1, -1)).reshape(-1),
        np.asfortranarray(pool.raw_means.reshape(1, -1)).reshape(-1),
        np.asfortranarray(pool.raw_epistemic_stds.reshape(1, -1)).reshape(-1),
        np.asfortranarray(pool.cluster_ids.reshape(1, -1)).reshape(-1),
        np.asfortranarray(pool.model_versions.reshape(1, -1)).reshape(-1),
    )
    assert candidate_pool_digest(pool) == candidate_pool_digest(layout_variant)
    changed = CandidatePool(
        pool.ids,
        pool.raw_means + 0.1,
        pool.raw_epistemic_stds,
        pool.cluster_ids,
        pool.model_versions,
    )
    assert candidate_pool_digest(pool) != candidate_pool_digest(changed)


def test_nontrivial_model_calibration() -> None:
    result = select_portfolio_batch(
        _pool(ids=(1,), means=(-10.0,), clusters=(1,), versions=(2,)),
        _policy(),
        {2: CalibrationConfig(mean_shift=1.0, std_scale=0.5, std_floor=0.3)},
        batch_size=1,
    )
    selection = result.selections[0]
    assert selection.calibrated_mean == -9.0
    assert selection.calibrated_std == pytest.approx(math.sqrt(0.34))


def test_crowding_before_and_cap_semantics() -> None:
    policy = _policy(
        crowding={
            "kind": "logarithmic_post_target",
            "target": 1,
            "weight": 10,
            "scale": 1,
        },
        constraint={"kind": "per_cluster_cap", "limit": 1, "scope": "batch"},
    )
    result = select_portfolio_batch(
        _pool(ids=(1, 2), means=(-10, -9.9), clusters=(1, 2)),
        policy,
        {1: CalibrationConfig()},
        batch_size=2,
        prior_observed_hits={1: 2, 2: 0},
    )
    assert [row.candidate_id for row in result.selections] == [2, 1]
    assert result.selections[1].crowding_penalty == pytest.approx(10 * math.log(2))
    assert all(row.cap_reached_after for row in result.selections)
    again = select_portfolio_batch(
        _pool(ids=(1, 2), clusters=(1, 2)),
        policy,
        {1: CalibrationConfig()},
        batch_size=2,
        prior_observed_hits={1: 100, 2: 0},
    )
    assert len(again.selections) == 2


def test_ties_calibration_and_fail_closed() -> None:
    pool = _pool(
        ids=(3, 1, 2),
        means=(-10, -10, -10),
        clusters=(2, 1, 1),
        versions=(1, 2, 1),
    )
    result = select_portfolio_batch(
        pool,
        _policy(),
        {1: CalibrationConfig(), 2: CalibrationConfig()},
        batch_size=3,
    )
    assert [row.candidate_id for row in result.selections] == [1, 2, 3]
    with pytest.raises(ValueError, match="missing calibration"):
        select_portfolio_batch(pool, _policy(), {1: CalibrationConfig()}, batch_size=1)
    with pytest.raises(ValueError, match="negative cluster"):
        _pool(clusters=(-1, 1, 2))
    with pytest.raises(ValueError, match="capped capacity"):
        select_portfolio_batch(
            _pool(clusters=(1, 1, 1)),
            _policy(constraint={"kind": "per_cluster_cap", "limit": 1}),
            {1: CalibrationConfig()},
            batch_size=2,
        )


def test_equal_utility_is_permutation_invariant() -> None:
    expected = [1, 2, 3, 4]
    for permutation in itertools.permutations(range(4)):
        ids = np.array([4, 2, 3, 1])[list(permutation)]
        pool = CandidatePool(
            ids=ids,
            raw_means=np.full(4, -10.0),
            raw_epistemic_stds=np.ones(4),
            cluster_ids=np.array([2, 1, 2, 1])[list(permutation)],
            model_versions=np.ones(4, dtype=np.int64),
        )
        result = select_portfolio_batch(
            pool,
            _policy(),
            {1: CalibrationConfig()},
            batch_size=4,
        )
        assert [row.candidate_id for row in result.selections] == expected


def test_unknown_prior_clusters_are_ignored() -> None:
    result = select_portfolio_batch(
        _pool(),
        _policy(),
        {1: CalibrationConfig()},
        batch_size=1,
        prior_observed_hits={1: 2, 999: 20},
    )
    assert result.selections[0].support_before >= 0


def test_randomized_small_pool_matches_naive_oracle() -> None:
    rng = np.random.default_rng(112)
    policy = _policy(
        crowding={
            "kind": "logarithmic_post_target",
            "target": 2,
            "weight": 0.04,
            "scale": 2,
        },
        constraint={"kind": "per_cluster_cap", "limit": 3},
    )
    for _ in range(20):
        size = 15
        pool = CandidatePool(
            ids=rng.permutation(np.arange(1, size + 1, dtype=np.int64)),
            raw_means=rng.normal(-9.8, 0.5, size=size),
            raw_epistemic_stds=rng.uniform(0.1, 1.2, size=size),
            cluster_ids=rng.integers(0, 5, size=size, dtype=np.int64),
            model_versions=np.zeros(size, dtype=np.int64),
        )
        prior = {cluster: float(rng.integers(0, 5)) for cluster in range(5)}
        actual = select_portfolio_batch(
            pool,
            policy,
            {0: CalibrationConfig()},
            batch_size=8,
            prior_observed_hits=prior,
        )
        expected = _naive_selection(pool, policy, prior, batch_size=8)
        assert [row.candidate_id for row in actual.selections] == expected


def _naive_selection(
    pool: CandidatePool,
    policy: PortfolioAcquisitionPolicy,
    prior: dict[int, float],
    *,
    batch_size: int,
) -> list[int]:
    means = pool.raw_means.astype(np.float64)
    stds = np.maximum(pool.raw_epistemic_stds.astype(np.float64), 1e-8)
    quality_config = policy.quality
    z = (quality_config.hit_threshold - means) / stds
    p_hit = np.array([0.5 * math.erfc(-value / math.sqrt(2)) for value in z])
    threshold = quality_config.hit_threshold - quality_config.xi
    ei_z = (threshold - means) / stds
    ei = (threshold - means) * np.array(
        [0.5 * math.erfc(-value / math.sqrt(2)) for value in ei_z]
    ) + stds * np.exp(-0.5 * ei_z**2) / math.sqrt(2 * math.pi)
    quality = (
        quality_config.probability_weight * p_hit + quality_config.expected_improvement_weight * ei
    )
    support = dict(prior)
    counts: dict[int, int] = {}
    available = set(range(len(pool.ids)))
    result: list[int] = []
    while len(result) < batch_size:
        scored: list[tuple[float, int, int]] = []
        for index in available:
            cluster = int(pool.cluster_ids[index])
            if counts.get(cluster, 0) >= 3:
                continue
            before = support.get(cluster, 0.0)
            after = before + p_hit[index]
            reward = policy.reward.weight * float(
                cumulative_reward(after, policy.reward) - cumulative_reward(before, policy.reward)
            )
            penalty = policy.crowding.weight * math.log1p(
                max(before - policy.crowding.target, 0.0) / policy.crowding.scale
            )
            scored.append((-(quality[index] + reward - penalty), int(pool.ids[index]), index))
        _, identifier, index = min(scored)
        cluster = int(pool.cluster_ids[index])
        support[cluster] = support.get(cluster, 0.0) + p_hit[index]
        counts[cluster] = counts.get(cluster, 0) + 1
        available.remove(index)
        result.append(identifier)
    return result
