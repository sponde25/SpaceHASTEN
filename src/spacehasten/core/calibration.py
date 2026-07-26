"""Post-hoc Gaussian calibration for predictive means and uncertainties."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

_STD_EPSILON = 1e-8


@dataclass(frozen=True)
class GaussianCalibrationResult:
    """Parameters and optimized objective for affine Gaussian calibration."""

    mean_shift: float
    std_scale: float
    std_floor: float
    objective: float


def apply_gaussian_calibration(
    raw_mean: ArrayLike,
    epistemic_std: ArrayLike,
    calibration: GaussianCalibrationResult,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply a fitted affine-mean, scaled-and-floored-std calibration."""

    mean, std = _validated_inputs(raw_mean, epistemic_std)
    parameters = np.asarray(
        [calibration.mean_shift, calibration.std_scale, calibration.std_floor], dtype=np.float64
    )
    if not np.isfinite(parameters).all() or calibration.std_scale <= 0 or calibration.std_floor < 0:
        raise ValueError(
            "calibration parameters must be finite with positive scale and non-negative floor"
        )
    calibrated_mean = mean + calibration.mean_shift
    calibrated_std = np.sqrt(
        (calibration.std_scale * np.maximum(std, _STD_EPSILON)) ** 2 + calibration.std_floor**2
    )
    return calibrated_mean, calibrated_std


def fit_gaussian_calibration(
    raw_mean: ArrayLike,
    epistemic_std: ArrayLike,
    targets: ArrayLike,
) -> GaussianCalibrationResult:
    """Fit a Gaussian NLL calibrator with bounded L-BFGS-B optimization."""

    mean, std = _validated_inputs(raw_mean, epistemic_std)
    target = _validated_targets(targets, len(mean))
    std = np.maximum(std, _STD_EPSILON)
    residual = target - mean
    initial = np.array(
        [
            float(np.median(residual)),
            0.0,
            np.log(max(0.05, float(np.std(residual)) * 0.25)),
        ],
        dtype=np.float64,
    )

    def objective(parameters: NDArray[np.float64]) -> float:
        mean_shift, log_std_scale, log_std_floor = parameters
        std_scale = np.exp(log_std_scale)
        std_floor = np.exp(log_std_floor)
        calibrated_std = np.sqrt((std_scale * std) ** 2 + std_floor**2)
        calibrated_residual = (target - (mean + mean_shift)) / calibrated_std
        return float(np.mean(np.log(calibrated_std) + 0.5 * calibrated_residual**2))

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(-3.0, 3.0), (-5.0, 5.0), (-5.0, 2.0)],
    )
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f"Gaussian calibration optimization failed: {result.message}")
    return GaussianCalibrationResult(
        mean_shift=float(result.x[0]),
        std_scale=float(np.exp(result.x[1])),
        std_floor=float(np.exp(result.x[2])),
        objective=float(result.fun),
    )


def _validated_inputs(
    raw_mean: ArrayLike, epistemic_std: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mean = np.asarray(raw_mean, dtype=np.float64).reshape(-1)
    std = np.asarray(epistemic_std, dtype=np.float64).reshape(-1)
    if len(mean) == 0 or len(mean) != len(std):
        raise ValueError("raw_mean and epistemic_std must be aligned non-empty arrays")
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("raw_mean and epistemic_std must contain only finite values")
    if np.any(std < 0):
        raise ValueError("epistemic_std must be non-negative")
    return mean, std


def _validated_targets(targets: ArrayLike, expected_length: int) -> NDArray[np.float64]:
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    if len(target) != expected_length or len(target) == 0:
        raise ValueError("targets must be a non-empty array aligned with raw_mean")
    if not np.isfinite(target).all():
        raise ValueError("targets must contain only finite values")
    return target
