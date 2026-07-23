"""Uncertainty-aware acquisition functions for docking batch selection."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

DockAcquisition = Literal["lcb", "ei"]


@dataclass(frozen=True, slots=True)
class AcquisitionCandidate:
    smiles: str
    spacehastenid: int
    pred_score: float
    epistemic_std: float
    model_version: int
    clusterid: int | None


@dataclass(frozen=True, slots=True)
class AcquisitionSelection:
    candidate: AcquisitionCandidate
    base_score: float
    cluster_count_before: int
    cluster_penalty: float
    penalized_score: float


@dataclass(frozen=True, slots=True)
class NormalizedPenalty:
    cluster_alpha: float
    cluster_lambda: float
    candidate_count: int
    batch_size: int
    frontier_start_rank: int
    frontier_stop_rank: int
    frontier_q10: float
    frontier_q90: float
    frontier_scale: float


def lower_confidence_bound(mean: float, epistemic_std: float, beta: float) -> float:
    """Return a lower confidence bound for a minimization objective."""
    _validate_prediction(mean, epistemic_std)
    if not math.isfinite(beta) or beta < 0:
        raise ValueError(f"beta must be finite and non-negative, got {beta}")
    return mean - beta * epistemic_std


def expected_improvement(
    mean: float,
    epistemic_std: float,
    hit_threshold: float,
    xi: float = 0.0,
) -> float:
    """Return expected improvement below a fixed target-specific threshold."""
    _validate_prediction(mean, epistemic_std)
    if not math.isfinite(hit_threshold):
        raise ValueError(f"hit_threshold must be finite, got {hit_threshold}")
    if not math.isfinite(xi) or xi < 0:
        raise ValueError(f"xi must be finite and non-negative, got {xi}")

    improvement = hit_threshold - mean - xi
    if epistemic_std == 0:
        return max(improvement, 0.0)

    z = improvement / epistemic_std
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return max(improvement * cdf + epistemic_std * pdf, 0.0)


def select_penalized_batch(
    candidates: Sequence[AcquisitionCandidate],
    *,
    method: DockAcquisition,
    batch_size: int,
    cluster_lambda: float,
    beta: float = 1.0,
    hit_threshold: float | None = None,
    xi: float = 0.0,
) -> list[AcquisitionSelection]:
    """Greedily select a batch with a dynamic within-cluster log penalty."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if not math.isfinite(cluster_lambda) or cluster_lambda < 0:
        raise ValueError(f"cluster_lambda must be finite and non-negative, got {cluster_lambda}")
    scored = _score_candidates(
        candidates,
        method=method,
        beta=beta,
        hit_threshold=hit_threshold,
        xi=xi,
    )
    return _select_scored_batch(
        scored,
        batch_size=batch_size,
        cluster_lambda=cluster_lambda,
    )


def select_normalized_penalized_batch(
    candidates: Sequence[AcquisitionCandidate],
    *,
    method: DockAcquisition,
    batch_size: int,
    cluster_alpha: float,
    beta: float = 1.0,
    hit_threshold: float | None = None,
    xi: float = 0.0,
) -> tuple[list[AcquisitionSelection], NormalizedPenalty]:
    """Select a batch after scaling the cluster penalty to the score frontier."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if not math.isfinite(cluster_alpha) or cluster_alpha < 0:
        raise ValueError(f"cluster_alpha must be finite and non-negative, got {cluster_alpha}")

    scored = _score_candidates(
        candidates,
        method=method,
        beta=beta,
        hit_threshold=hit_threshold,
        xi=xi,
    )
    frontier_start = batch_size // 2
    frontier_stop = min(2 * batch_size, len(scored))
    if frontier_stop <= frontier_start:
        raise ValueError(
            "normalized cluster penalty requires more than "
            f"{frontier_start} acquisition candidates, got {len(scored)}"
        )
    frontier = heapq.nsmallest(
        frontier_stop,
        scored,
        key=lambda item: (item[0], item[1].spacehastenid),
    )
    frontier_q10 = _scored_quantile(frontier, frontier_start, frontier_stop, 0.10)
    frontier_q90 = _scored_quantile(frontier, frontier_start, frontier_stop, 0.90)
    del frontier
    frontier_scale = frontier_q90 - frontier_q10
    if not math.isfinite(frontier_scale) or frontier_scale <= 0:
        raise ValueError(
            "normalized cluster penalty requires a positive finite acquisition frontier scale, "
            f"got {frontier_scale}"
        )
    cluster_lambda = cluster_alpha * frontier_scale / math.log(2.0)
    if not math.isfinite(cluster_lambda):
        raise ValueError(f"normalized cluster penalty produced non-finite lambda {cluster_lambda}")
    normalization = NormalizedPenalty(
        cluster_alpha=cluster_alpha,
        cluster_lambda=cluster_lambda,
        candidate_count=len(scored),
        batch_size=batch_size,
        frontier_start_rank=frontier_start + 1,
        frontier_stop_rank=frontier_stop,
        frontier_q10=frontier_q10,
        frontier_q90=frontier_q90,
        frontier_scale=frontier_scale,
    )
    selections = _select_scored_batch(
        scored,
        batch_size=batch_size,
        cluster_lambda=cluster_lambda,
    )
    return selections, normalization


def _score_candidates(
    candidates: Sequence[AcquisitionCandidate],
    *,
    method: DockAcquisition,
    beta: float,
    hit_threshold: float | None,
    xi: float,
) -> list[tuple[float, AcquisitionCandidate]]:
    if method == "ei" and hit_threshold is None:
        raise ValueError("hit_threshold is required for expected improvement")
    if method not in {"lcb", "ei"}:
        raise ValueError(f"unknown docking acquisition method: {method!r}")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError(f"beta must be finite and non-negative, got {beta}")
    if not math.isfinite(xi) or xi < 0:
        raise ValueError(f"xi must be finite and non-negative, got {xi}")
    if hit_threshold is not None and not math.isfinite(hit_threshold):
        raise ValueError(f"hit_threshold must be finite, got {hit_threshold}")

    seen_ids: set[int] = set()
    scored: list[tuple[float, AcquisitionCandidate]] = []
    for candidate in candidates:
        if candidate.spacehastenid in seen_ids:
            raise ValueError(f"duplicate acquisition candidate {candidate.spacehastenid}")
        seen_ids.add(candidate.spacehastenid)
        if method == "lcb":
            base_score = lower_confidence_bound(candidate.pred_score, candidate.epistemic_std, beta)
        else:
            assert hit_threshold is not None
            base_score = -expected_improvement(
                candidate.pred_score,
                candidate.epistemic_std,
                hit_threshold,
                xi,
            )
        scored.append((base_score, candidate))
    return scored


def _scored_quantile(
    scored: Sequence[tuple[float, AcquisitionCandidate]],
    start: int,
    stop: int,
    quantile: float,
) -> float:
    position = (stop - start - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    lower_value = scored[start + lower][0]
    if lower == upper:
        return lower_value
    fraction = position - lower
    upper_value = scored[start + upper][0]
    return lower_value + fraction * (upper_value - lower_value)


def _select_scored_batch(
    scored: list[tuple[float, AcquisitionCandidate]],
    *,
    batch_size: int,
    cluster_lambda: float,
    scores_are_sorted: bool = False,
) -> list[AcquisitionSelection]:
    selection_count = min(batch_size, len(scored))

    if cluster_lambda == 0:
        if not scores_are_sorted:
            scored.sort(key=lambda item: (item[0], item[1].spacehastenid))
        unpenalized_selections: list[AcquisitionSelection] = []
        unpenalized_cluster_counts: dict[int, int] = defaultdict(int)
        for base_score, candidate in scored[:selection_count]:
            count_before = (
                unpenalized_cluster_counts[candidate.clusterid]
                if candidate.clusterid is not None
                else 0
            )
            unpenalized_selections.append(
                AcquisitionSelection(
                    candidate=candidate,
                    base_score=base_score,
                    cluster_count_before=count_before,
                    cluster_penalty=0.0,
                    penalized_score=base_score,
                )
            )
            if candidate.clusterid is not None:
                unpenalized_cluster_counts[candidate.clusterid] += 1
        return unpenalized_selections

    missing_clusters = [
        candidate.spacehastenid for _, candidate in scored if candidate.clusterid is None
    ]
    if missing_clusters:
        preview = ", ".join(str(identifier) for identifier in missing_clusters[:5])
        raise ValueError(
            f"{len(missing_clusters)} acquisition candidates lack cluster assignments "
            f"(first IDs: {preview})"
        )

    by_cluster: dict[int, list[tuple[float, AcquisitionCandidate]]] = defaultdict(list)
    for item in scored:
        candidate = item[1]
        assert candidate.clusterid is not None
        by_cluster[candidate.clusterid].append(item)
    scored.clear()
    if not scores_are_sorted:
        for members in by_cluster.values():
            members.sort(key=lambda item: (item[0], item[1].spacehastenid))

    heap: list[tuple[float, float, int, int, int]] = []
    for clusterid, members in by_cluster.items():
        base_score, candidate = members[0]
        heapq.heappush(
            heap,
            (base_score, base_score, candidate.spacehastenid, clusterid, 0),
        )

    selections: list[AcquisitionSelection] = []
    cluster_counts: dict[int, int] = defaultdict(int)
    while heap and len(selections) < selection_count:
        penalized_score, base_score, _, clusterid, member_index = heapq.heappop(heap)
        candidate = by_cluster[clusterid][member_index][1]
        count_before = cluster_counts[clusterid]
        penalty = cluster_lambda * math.log1p(count_before)
        selections.append(
            AcquisitionSelection(
                candidate=candidate,
                base_score=base_score,
                cluster_count_before=count_before,
                cluster_penalty=penalty,
                penalized_score=penalized_score,
            )
        )

        cluster_counts[clusterid] += 1
        next_index = member_index + 1
        members = by_cluster[clusterid]
        if next_index < len(members):
            next_base, next_candidate = members[next_index]
            next_penalty = cluster_lambda * math.log1p(cluster_counts[clusterid])
            heapq.heappush(
                heap,
                (
                    next_base + next_penalty,
                    next_base,
                    next_candidate.spacehastenid,
                    clusterid,
                    next_index,
                ),
            )

    return selections


def _validate_prediction(mean: float, epistemic_std: float) -> None:
    if not math.isfinite(mean):
        raise ValueError(f"prediction mean must be finite, got {mean}")
    if not math.isfinite(epistemic_std) or epistemic_std < 0:
        raise ValueError(
            f"epistemic uncertainty must be finite and non-negative, got {epistemic_std}"
        )


__all__ = [
    "AcquisitionCandidate",
    "AcquisitionSelection",
    "DockAcquisition",
    "NormalizedPenalty",
    "expected_improvement",
    "lower_confidence_bound",
    "select_normalized_penalized_batch",
    "select_penalized_batch",
]
