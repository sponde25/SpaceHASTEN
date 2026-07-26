"""Tests for dependency-light Gaussian calibration."""

from __future__ import annotations

import numpy as np
import pytest

from spacehasten.core.calibration import (
    GaussianCalibrationResult,
    apply_gaussian_calibration,
    fit_gaussian_calibration,
)


def test_apply_gaussian_calibration_uses_affine_mean_and_std_floor() -> None:
    result = GaussianCalibrationResult(1.5, 2.0, 3.0, 0.0)
    mean, std = apply_gaussian_calibration([1.0, 2.0], [0.0, 4.0], result)

    assert np.allclose(mean, [2.5, 3.5])
    assert np.allclose(std, [3.0, np.sqrt(73.0)])


def test_fit_gaussian_calibration_recovers_shift_and_scale() -> None:
    raw_mean = np.linspace(-2.0, 2.0, 400)
    raw_std = np.linspace(0.2, 1.2, 400)
    rng = np.random.default_rng(12)
    targets = raw_mean + 0.7 + rng.normal(scale=1.4 * raw_std)

    fitted = fit_gaussian_calibration(raw_mean, raw_std, targets)

    assert fitted.mean_shift == pytest.approx(0.7, abs=0.15)
    assert fitted.std_scale == pytest.approx(1.4, abs=0.2)
    assert fitted.std_floor == pytest.approx(0.0, abs=0.1)
    assert np.isfinite(fitted.objective)


@pytest.mark.parametrize(
    ("mean", "std", "targets"),
    [([], [], []), ([1.0], [-1.0], [1.0]), ([1.0], [1.0], [np.nan]), ([1.0], [1.0, 2.0], [1.0])],
)
def test_fit_gaussian_calibration_rejects_invalid_arrays(
    mean: list[float], std: list[float], targets: list[float]
) -> None:
    with pytest.raises(ValueError):
        fit_gaussian_calibration(mean, std, targets)
