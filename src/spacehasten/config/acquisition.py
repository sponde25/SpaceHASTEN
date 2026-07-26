"""Immutable, versioned TOML policy models for portfolio acquisition."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _PolicyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GaussianHitEIQualityConfig(_PolicyModel):
    kind: Literal["gaussian_hit_ei"] = "gaussian_hit_ei"
    hit_threshold: float
    probability_weight: float = 1.0
    expected_improvement_weight: float = 1.0
    xi: float = 0.0
    uncertainty_source: Literal["epistemic"] = "epistemic"

    @field_validator(
        "hit_threshold",
        "probability_weight",
        "expected_improvement_weight",
        "xi",
    )
    @classmethod
    def finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("quality values must be finite")
        return value

    @model_validator(mode="after")
    def non_negative(self) -> GaussianHitEIQualityConfig:
        if self.probability_weight < 0 or self.expected_improvement_weight < 0 or self.xi < 0:
            raise ValueError("quality weights and xi must be non-negative")
        return self


class ObservedHitSupportConfig(_PolicyModel):
    prior: Literal["observed_hits"] = "observed_hits"
    current_batch_increment: Literal["hit_probability"] = "hit_probability"


class PiecewiseLinearRewardConfig(_PolicyModel):
    kind: Literal["piecewise_linear"] = "piecewise_linear"
    breakpoints: tuple[float, ...]
    slopes: tuple[float, ...]
    weight: float = 1.0

    @model_validator(mode="after")
    def valid_tiers(self) -> PiecewiseLinearRewardConfig:
        if not self.breakpoints or len(self.breakpoints) != len(self.slopes):
            raise ValueError("breakpoints and slopes must be non-empty and have equal length")
        if not all(math.isfinite(item) and item > 0 for item in self.breakpoints):
            raise ValueError("breakpoints must be finite and positive")
        if any(
            self.breakpoints[index] >= self.breakpoints[index + 1]
            for index in range(len(self.breakpoints) - 1)
        ):
            raise ValueError("breakpoints must be strictly increasing")
        if not all(math.isfinite(item) and item >= 0 for item in self.slopes):
            raise ValueError("slopes must be finite and non-negative")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("reward weight must be finite and non-negative")
        return self


class NoCrowdingConfig(_PolicyModel):
    kind: Literal["none"] = "none"


class LogarithmicPostTargetCrowdingConfig(_PolicyModel):
    kind: Literal["logarithmic_post_target"] = "logarithmic_post_target"
    target: float
    weight: float
    scale: float

    @field_validator("target", "scale")
    @classmethod
    def positive(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("crowding target and scale must be finite and positive")
        return value

    @field_validator("weight")
    @classmethod
    def non_negative_weight(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("crowding weight must be finite and non-negative")
        return value


CrowdingConfig = Annotated[
    NoCrowdingConfig | LogarithmicPostTargetCrowdingConfig,
    Field(discriminator="kind"),
]


class NoConstraintConfig(_PolicyModel):
    kind: Literal["none"] = "none"


class PerClusterCapConstraintConfig(_PolicyModel):
    kind: Literal["per_cluster_cap"] = "per_cluster_cap"
    limit: int
    scope: Literal["batch"] = "batch"

    @field_validator("limit")
    @classmethod
    def positive_limit(cls, value: int) -> int:
        if value < 1:
            raise ValueError("per-cluster cap limit must be at least one")
        return value


ConstraintConfig = Annotated[
    NoConstraintConfig | PerClusterCapConstraintConfig,
    Field(discriminator="kind"),
]


class HistoryConfig(_PolicyModel):
    attempt_policy: Literal["once_per_campaign", "unscored_eligible"] = "once_per_campaign"


class CalibrationConfig(_PolicyModel):
    mean_shift: float = 0.0
    std_scale: float = 1.0
    std_floor: float = 0.0
    provenance: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_calibration(self) -> CalibrationConfig:
        if not all(
            math.isfinite(item) for item in (self.mean_shift, self.std_scale, self.std_floor)
        ):
            raise ValueError("calibration values must be finite")
        if self.std_scale <= 0 or self.std_floor < 0:
            raise ValueError("std_scale must be positive and std_floor non-negative")
        return self


class PortfolioAcquisitionPolicy(_PolicyModel):
    schema_version: Literal[1] = 1
    name: str | None = None
    quality: GaussianHitEIQualityConfig
    support: ObservedHitSupportConfig = Field(default_factory=ObservedHitSupportConfig)
    reward: PiecewiseLinearRewardConfig
    crowding: CrowdingConfig = Field(default_factory=NoCrowdingConfig)
    constraint: ConstraintConfig = Field(default_factory=NoConstraintConfig)
    history: HistoryConfig = Field(default_factory=HistoryConfig)

    @classmethod
    def from_toml(cls, path: Path) -> PortfolioAcquisitionPolicy:
        with Path(path).open("rb") as handle:
            return cls.model_validate(tomllib.load(handle))


def load_acquisition_policy(path: Path) -> PortfolioAcquisitionPolicy:
    return PortfolioAcquisitionPolicy.from_toml(path)


__all__ = [
    "CalibrationConfig",
    "ConstraintConfig",
    "CrowdingConfig",
    "GaussianHitEIQualityConfig",
    "HistoryConfig",
    "LogarithmicPostTargetCrowdingConfig",
    "NoConstraintConfig",
    "NoCrowdingConfig",
    "ObservedHitSupportConfig",
    "PerClusterCapConstraintConfig",
    "PiecewiseLinearRewardConfig",
    "PortfolioAcquisitionPolicy",
    "load_acquisition_policy",
]
