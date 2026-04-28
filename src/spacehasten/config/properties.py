"""Pydantic models for molecular property ranges (replaces legacy properties table)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FloatRange(BaseModel):
    """Closed [min, max] interval over floats."""

    model_config = ConfigDict(frozen=True)

    min: float
    max: float

    @model_validator(mode="after")
    def _check_order(self) -> FloatRange:
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        return self


class IntRange(BaseModel):
    """Closed [min, max] interval over ints."""

    model_config = ConfigDict(frozen=True)

    min: int
    max: int

    @model_validator(mode="after")
    def _check_order(self) -> IntRange:
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        return self


class PropertyRanges(BaseModel):
    """Property-filter ranges for ligand acquisition.

    Defaults match the legacy ``cfg.SpaceHASTENConfiguration`` defaults.
    """

    model_config = ConfigDict(frozen=True)

    mw: FloatRange = Field(default_factory=lambda: FloatRange(min=0.0, max=500.0))
    slogp: FloatRange = Field(default_factory=lambda: FloatRange(min=-10.0, max=5.0))
    hba: IntRange = Field(default_factory=lambda: IntRange(min=0, max=10))
    hbd: IntRange = Field(default_factory=lambda: IntRange(min=0, max=5))
    rotbonds: IntRange = Field(default_factory=lambda: IntRange(min=0, max=10))
    tpsa: FloatRange = Field(default_factory=lambda: FloatRange(min=0.0, max=140.0))

    @classmethod
    def from_toml(cls, path: Path) -> PropertyRanges:
        with Path(path).open("rb") as fh:
            data = tomllib.load(fh)
        section = data.get("properties", data)
        return cls.model_validate(section)

    def to_toml(self, path: Path) -> None:
        payload: dict[str, Any] = {"properties": self.model_dump()}
        with Path(path).open("wb") as fh:
            tomli_w.dump(payload, fh)
