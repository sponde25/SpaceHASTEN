"""Array-oriented greedy portfolio acquisition for minimization campaigns."""

from __future__ import annotations

import heapq
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

from spacehasten.config.acquisition import (
    CalibrationConfig,
    LogarithmicPostTargetCrowdingConfig,
    NoCrowdingConfig,
    PerClusterCapConstraintConfig,
    PiecewiseLinearRewardConfig,
    PortfolioAcquisitionPolicy,
)

ProgressCallback: TypeAlias = Callable[[int, int], None]
FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CandidatePool:
    ids: IntArray
    raw_means: FloatArray
    raw_epistemic_stds: FloatArray
    cluster_ids: IntArray
    model_versions: IntArray

    def __post_init__(self) -> None:
        arrays = (
            self.ids,
            self.raw_means,
            self.raw_epistemic_stds,
            self.cluster_ids,
            self.model_versions,
        )
        if not all(isinstance(array, np.ndarray) and array.ndim == 1 for array in arrays):
            raise ValueError("candidate arrays must be one-dimensional NumPy arrays")
        if len(self.ids) == 0 or any(len(array) != len(self.ids) for array in arrays):
            raise ValueError("candidate arrays must be non-empty and have equal lengths")
        if self.ids.dtype.kind not in "iu" or self.cluster_ids.dtype.kind not in "iu":
            raise ValueError("IDs and cluster IDs must use integral NumPy dtypes")
        if np.any(self.ids < 0):
            raise ValueError("candidate IDs must be non-negative")
        if np.any(self.cluster_ids < 0):
            raise ValueError("negative cluster IDs are missing assignments")
        if self.model_versions.dtype.kind not in "iu":
            raise ValueError("model_versions must use integral NumPy dtypes")
        if len(np.unique(self.ids)) != len(self.ids):
            raise ValueError("candidate IDs must be unique")
        if not np.all(np.isfinite(self.raw_means)) or not np.all(
            np.isfinite(self.raw_epistemic_stds)
        ):
            raise ValueError("candidate predictions must be finite")
        if np.any(self.raw_epistemic_stds < 0):
            raise ValueError("candidate epistemic standard deviations must be non-negative")


@dataclass(frozen=True, slots=True)
class PortfolioSelection:
    candidate_id: int
    cluster_id: int
    model_version: int
    raw_mean: float
    raw_epistemic_std: float
    calibrated_mean: float
    calibrated_std: float
    p_hit: float
    expected_improvement: float
    quality: float
    support_before: float
    support_after: float
    marginal_reward: float
    crowding_penalty: float
    final_utility: float
    cluster_count_before: int
    cap_reached_after: bool


@dataclass(frozen=True, slots=True)
class PortfolioSelectionResult:
    selections: tuple[PortfolioSelection, ...]


class CrowdingEvaluator(Protocol):
    def __call__(self, support_before: FloatArray | float) -> FloatArray: ...


class RewardEvaluator(Protocol):
    def __call__(
        self,
        support_before: FloatArray | float,
        support_after: FloatArray | float,
    ) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class CandidateScores:
    calibrated_means: FloatArray
    calibrated_stds: FloatArray
    hit_probabilities: FloatArray
    expected_improvements: FloatArray
    qualities: FloatArray


def gaussian_hit_probability(
    means: FloatArray,
    stds: FloatArray,
    threshold: float,
) -> FloatArray:
    return np.asarray(ndtr((threshold - means) / stds), dtype=np.float64)


def gaussian_expected_improvement(
    means: FloatArray,
    stds: FloatArray,
    threshold: float,
    xi: float = 0.0,
) -> FloatArray:
    effective_threshold = threshold - xi
    z = (effective_threshold - means) / stds
    return np.asarray(
        (effective_threshold - means) * ndtr(z)
        + stds * np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi),
        dtype=np.float64,
    )


def cumulative_reward(
    support: FloatArray | float,
    config: PiecewiseLinearRewardConfig,
) -> FloatArray:
    value = np.asarray(support, dtype=np.float64)
    lower = np.asarray((0.0, *config.breakpoints[:-1]), dtype=np.float64)
    upper = np.asarray(config.breakpoints, dtype=np.float64)
    widths = np.maximum(np.minimum(value[..., None], upper) - lower, 0.0)
    return np.asarray(
        (widths * np.asarray(config.slopes, dtype=np.float64)).sum(axis=-1),
        dtype=np.float64,
    )


def _crowding_evaluator(policy: PortfolioAcquisitionPolicy) -> CrowdingEvaluator:
    config = policy.crowding
    if isinstance(config, NoCrowdingConfig):
        return lambda support: np.zeros_like(np.asarray(support, dtype=np.float64))
    assert isinstance(config, LogarithmicPostTargetCrowdingConfig)
    return lambda support: (
        config.weight
        * np.log1p(
            np.maximum(np.asarray(support, dtype=np.float64) - config.target, 0.0) / config.scale
        )
    )


def _reward_evaluator(policy: PortfolioAcquisitionPolicy) -> RewardEvaluator:
    config = policy.reward

    def evaluate(
        support_before: FloatArray | float,
        support_after: FloatArray | float,
    ) -> FloatArray:
        return np.asarray(
            config.weight
            * (
                cumulative_reward(support_after, config) - cumulative_reward(support_before, config)
            ),
            dtype=np.float64,
        )

    return evaluate


def _score_candidates(
    pool: CandidatePool,
    policy: PortfolioAcquisitionPolicy,
    calibrations: Mapping[int, CalibrationConfig],
) -> CandidateScores:
    means, stds = _calibrate(pool, calibrations)
    config = policy.quality
    p_hit = gaussian_hit_probability(means, stds, config.hit_threshold)
    expected_improvement = gaussian_expected_improvement(
        means,
        stds,
        config.hit_threshold,
        config.xi,
    )
    quality = (
        config.probability_weight * p_hit
        + config.expected_improvement_weight * expected_improvement
    )
    return CandidateScores(means, stds, p_hit, expected_improvement, quality)


def select_portfolio_batch(
    pool: CandidatePool,
    policy: PortfolioAcquisitionPolicy,
    calibrations: Mapping[int, CalibrationConfig],
    *,
    batch_size: int,
    prior_observed_hits: Mapping[int, float] | None = None,
    progress: ProgressCallback | None = None,
) -> PortfolioSelectionResult:
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    order = np.argsort(pool.cluster_ids, kind="stable")
    sorted_clusters = pool.cluster_ids[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_clusters)) + 1]
    stops = np.r_[starts[1:], len(order)]
    clusters = sorted_clusters[starts]
    cap = (
        policy.constraint.limit
        if isinstance(policy.constraint, PerClusterCapConstraintConfig)
        else None
    )
    capacity = int(np.minimum(stops - starts, cap).sum()) if cap is not None else len(pool.ids)
    if batch_size > capacity:
        raise ValueError(f"capped capacity is {capacity}, below requested batch size {batch_size}")
    support = _prior_support(clusters, prior_observed_hits)
    scores = _score_candidates(pool, policy, calibrations)
    means = scores.calibrated_means
    stds = scores.calibrated_stds
    p_hit = scores.hit_probabilities
    ei = scores.expected_improvements
    quality = scores.qualities
    selected = np.zeros(len(pool.ids), dtype=bool)
    counts = np.zeros(len(clusters), dtype=np.int64)
    reward_evaluator = _reward_evaluator(policy)
    crowding = _crowding_evaluator(policy)
    generations = np.zeros(len(clusters), dtype=np.int64)
    heap: list[tuple[float, int, int, int, int]] = []

    def refresh(group: int) -> None:
        if cap is not None and counts[group] >= cap:
            return
        members = order[starts[group] : stops[group]]
        remaining = members[~selected[members]]
        if len(remaining) == 0:
            return
        before = support[group]
        after = before + p_hit[remaining]
        reward = reward_evaluator(before, after)
        utilities = quality[remaining] + reward - crowding(before)
        best_utility = np.max(utilities)
        best = remaining[np.flatnonzero(utilities == best_utility)]
        candidate = best[np.argmin(pool.ids[best])]
        generations[group] += 1
        heapq.heappush(
            heap,
            (
                -float(best_utility),
                int(pool.ids[candidate]),
                group,
                candidate,
                int(generations[group]),
            ),
        )

    for group in range(len(clusters)):
        refresh(group)
    result: list[PortfolioSelection] = []
    while len(result) < batch_size:
        if not heap:
            raise RuntimeError("portfolio heap exhausted before requested batch capacity")
        _, _, group, candidate, generation = heapq.heappop(heap)
        if generation != generations[group] or selected[candidate]:
            continue
        before = support[group]
        after = before + p_hit[candidate]
        reward = float(reward_evaluator(before, after))
        penalty = float(crowding(before))
        utility = float(quality[candidate] + reward - penalty)
        selected[candidate] = True
        count_before = int(counts[group])
        counts[group] += 1
        support[group] = after
        version = int(pool.model_versions[candidate])
        result.append(
            PortfolioSelection(
                candidate_id=int(pool.ids[candidate]),
                cluster_id=int(clusters[group]),
                model_version=version,
                raw_mean=float(pool.raw_means[candidate]),
                raw_epistemic_std=float(pool.raw_epistemic_stds[candidate]),
                calibrated_mean=float(means[candidate]),
                calibrated_std=float(stds[candidate]),
                p_hit=float(p_hit[candidate]),
                expected_improvement=float(ei[candidate]),
                quality=float(quality[candidate]),
                support_before=float(before),
                support_after=float(after),
                marginal_reward=reward,
                crowding_penalty=penalty,
                final_utility=utility,
                cluster_count_before=count_before,
                cap_reached_after=cap is not None and counts[group] == cap,
            )
        )
        refresh(group)
        if progress is not None:
            progress(len(result), batch_size)
    return PortfolioSelectionResult(tuple(result))


def _prior_support(
    clusters: IntArray,
    prior: Mapping[int, float] | None,
) -> FloatArray:
    support = np.zeros(len(clusters), dtype=np.float64)
    if prior is None:
        return support
    positions = {int(cluster): index for index, cluster in enumerate(clusters)}
    for cluster, value in prior.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError("prior observed hits must be finite and non-negative")
        position = positions.get(cluster)
        if position is not None:
            support[position] = value
    return support


def _calibrate(
    pool: CandidatePool,
    calibrations: Mapping[int, CalibrationConfig],
) -> tuple[FloatArray, FloatArray]:
    means = np.empty(len(pool.ids), dtype=np.float64)
    stds = np.empty(len(pool.ids), dtype=np.float64)
    for version in np.unique(pool.model_versions):
        key = version.item()
        if key not in calibrations:
            raise ValueError(f"missing calibration for model version {key!r}")
        config = calibrations[key]
        indices = np.flatnonzero(pool.model_versions == version)
        means[indices] = pool.raw_means[indices] + config.mean_shift
        raw = np.maximum(pool.raw_epistemic_stds[indices], 1e-8)
        stds[indices] = np.sqrt((config.std_scale * raw) ** 2 + config.std_floor**2)
    return means, stds


__all__ = [
    "CandidatePool",
    "PortfolioSelection",
    "PortfolioSelectionResult",
    "cumulative_reward",
    "gaussian_expected_improvement",
    "gaussian_hit_probability",
    "select_portfolio_batch",
]
