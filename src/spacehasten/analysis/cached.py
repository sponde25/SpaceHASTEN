"""Metrics for packed binary fingerprints and categorical families."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int64]
UInt64Array = npt.NDArray[np.uint64]

BYTE_POPCOUNT = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)
)


def sampled_diversity(
    indices: IntArray,
    words: UInt64Array,
    popcounts: IntArray,
    *,
    seed: int,
    samples: int,
    batch_size: int = 250_000,
) -> tuple[float | None, float | None, int]:
    """Estimate mean Tanimoto distance from cached Morgan fingerprints."""
    selected = np.asarray(indices, dtype=np.int64)
    if len(selected) < 2:
        return None, None, 0
    if samples < 1 or batch_size < 1:
        raise ValueError("samples and batch_size must be positive")
    if words.ndim != 2 or words.shape[1] != 16 or len(words) != len(popcounts):
        raise ValueError("fingerprint cache must contain N x 16 words and N popcounts")
    if selected.min() < 0 or selected.max() >= len(words):
        raise ValueError("fingerprint indices are out of range")

    count = min(samples, len(selected) * (len(selected) - 1) // 2)
    random = np.random.default_rng(seed)
    total = total_square = 0.0
    completed = 0
    while completed < count:
        size = min(batch_size, count - completed)
        left = random.integers(0, len(selected), size=size)
        right = random.integers(0, len(selected) - 1, size=size)
        right += right >= left
        left_rows, right_rows = selected[left], selected[right]
        intersection = BYTE_POPCOUNT[
            np.bitwise_and(words[left_rows], words[right_rows]).view(np.uint8)
        ].sum(axis=1)
        union = popcounts[left_rows] + popcounts[right_rows] - intersection
        distances = np.divide(
            union - intersection,
            union,
            out=np.zeros(size, dtype=np.float64),
            where=union != 0,
        )
        total += float(distances.sum())
        total_square += float(np.square(distances).sum())
        completed += size
    mean = total / count
    variance = max(0.0, (total_square - count * mean * mean) / max(1, count - 1))
    return mean, math.sqrt(variance / count), count


def family_distribution(values: Sequence[Any]) -> dict[str, float | int | None]:
    """Return Hill-number and concentration metrics for categorical labels."""
    labels = np.asarray(values)
    if len(labels) == 0:
        return {
            "q0": 0,
            "q1": 0.0,
            "q2": 0.0,
            "hhi": None,
            "largest_fraction": None,
            "top10_fraction": None,
            "entropy": None,
        }
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    hhi = float(np.sum(probabilities**2))
    return {
        "q0": len(counts),
        "q1": math.exp(entropy),
        "q2": 1.0 / hhi,
        "hhi": hhi,
        "largest_fraction": float(probabilities.max()),
        "top10_fraction": float(np.sort(probabilities)[-10:].sum()),
        "entropy": entropy,
    }


__all__ = ["family_distribution", "sampled_diversity"]
