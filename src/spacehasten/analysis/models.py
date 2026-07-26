"""Immutable data models for single-run analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Capabilities:
    tables: tuple[str, ...]
    data_columns: tuple[str, ...]
    predictions_columns: tuple[str, ...]
    has_data: bool
    has_predictions: bool
    has_atlas_assignments: bool
    has_clusters: bool


@dataclass(frozen=True, slots=True)
class RunContext:
    input_path: Path
    database_path: Path
    shared_root: Path | None
    acquisition_paths: tuple[tuple[int, Path], ...]
    capabilities: Capabilities


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    hit_threshold: float
    cutoffs: tuple[float, ...]
    pair_samples: int = 10_000
    random_seed: int = 0
    dpi: int = 600


@dataclass(frozen=True, slots=True)
class SelectionRound:
    round_id: int
    attempts: tuple[int, ...]
    clusters: Counter[str]
    atlas_ids: frozenset[str]
    csv_columns: tuple[str, ...]
    model_version: str | None
    model_version_status: str
    model_version_reason: str | None


@dataclass(frozen=True, slots=True)
class AtlasResolution:
    atlas_id: str | None
    status: str
    reason: str | None
