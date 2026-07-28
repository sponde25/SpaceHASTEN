from __future__ import annotations

import numpy as np
import pytest

from spacehasten.analysis.cached import family_distribution, sampled_diversity


def test_family_distribution() -> None:
    metrics = family_distribution(["a", "a", "b"])
    assert metrics["q0"] == 2
    assert metrics["q2"] == pytest.approx(1.8)
    assert metrics["largest_fraction"] == pytest.approx(2 / 3)
    assert metrics["top10_fraction"] == pytest.approx(1.0)
    assert family_distribution([])["q0"] == 0


def test_sampled_diversity_is_deterministic() -> None:
    words = np.zeros((3, 16), dtype=np.uint64)
    words[0, 0] = 0b0011
    words[1, 0] = 0b0101
    words[2, 0] = 0b1111
    popcounts = np.asarray([2, 2, 4], dtype=np.int64)
    indices = np.arange(3, dtype=np.int64)

    first = sampled_diversity(indices, words, popcounts, seed=42, samples=100)
    second = sampled_diversity(indices, words, popcounts, seed=42, samples=100)

    assert first == second
    assert first[0] == pytest.approx(5 / 9)
    assert first[2] == 3


def test_sampled_diversity_handles_singleton() -> None:
    words = np.zeros((1, 16), dtype=np.uint64)
    assert sampled_diversity(
        np.asarray([0]), words, np.asarray([0]), seed=1, samples=10
    ) == (None, None, 0)
