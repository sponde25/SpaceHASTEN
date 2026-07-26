"""Orchestration for read-only, bounded-memory analysis of one run."""

from __future__ import annotations

import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .artifacts import checksums, context_json, package_versions, write_csv, write_json
from .calibration import calibration_metrics
from .database import ReadOnlyDatabase, fetch_atlas, fetch_rows, resolve_atlas
from .discovery import discover_run
from .metrics import (
    analysis_ids,
    budget_curve,
    cutoff_metrics,
    family_metrics,
    normalize_acquisitions,
    round_metrics,
    rounds,
    score_distribution,
)
from .models import AnalysisConfig, RunContext
from .plots import render_plots


def analyze_run(
    context: RunContext, config: AnalysisConfig, analysis_root: Path, *, overwrite: bool
) -> dict[str, Any]:
    """Write normalized analysis artifacts without mutating the source database."""
    if config.pair_samples < 1 or config.dpi < 1 or not math.isfinite(config.hit_threshold):
        raise ValueError("threshold must be finite and pair samples/dpi must be positive")
    analysis_root = analysis_root.resolve()
    if analysis_root.exists() and any(analysis_root.iterdir()) and not overwrite:
        raise FileExistsError(f"analysis root exists and is non-empty: {analysis_root}")
    analysis_root.mkdir(parents=True, exist_ok=True)
    selections, acquisition_rows = normalize_acquisitions(context.acquisition_paths)
    with ReadOnlyDatabase(context.database_path) as database:
        check = database.quick_check()
        if check != "ok":
            raise ValueError(f"sqlite quick_check failed: {check}")
        rows = fetch_rows(database, context.capabilities, selections)
        round_ids = rounds(rows, selections)
        atlas = resolve_atlas(database, selections)
        atlas_map = fetch_atlas(database, atlas, analysis_ids(rows, selections))
        metrics, coverage = round_metrics(rows, selections, round_ids, config.hit_threshold)
        cutoff = cutoff_metrics(rows, selections, round_ids, config.cutoffs)
        discovery_curve = budget_curve(rows, selections, round_ids, config.hit_threshold)
        score_curve = score_distribution(rows, selections, round_ids)
        family = family_metrics(rows, selections, round_ids, atlas, atlas_map, config)
        calibration, calibration_curve = calibration_metrics(
            database, context.capabilities, rows, selections, config.hit_threshold
        )
    write_csv(analysis_root / "coverage.csv", coverage)
    write_csv(analysis_root / "round_metrics.csv", metrics)
    write_csv(analysis_root / "cutoff_curve.csv", cutoff)
    write_csv(analysis_root / "budget_curve.csv", discovery_curve)
    write_csv(analysis_root / "score_distribution.csv", score_curve)
    write_csv(analysis_root / "family_metrics.csv", family)
    write_csv(analysis_root / "acquisition_metrics.csv", acquisition_rows)
    write_csv(analysis_root / "calibration_metrics.csv", calibration)
    write_csv(analysis_root / "calibration_curve.csv", calibration_curve)
    write_json(analysis_root / "run_context.json", context_json(context))
    artifacts = render_plots(analysis_root, config.dpi)
    manifest = {
        "database_quick_check": check,
        "config": asdict(config),
        "inputs": checksums(
            [context.database_path, *(path for _, path in context.acquisition_paths)]
        ),
        "checksum_note": "Database SHA-256 is one additional sequential source-file pass.",
        "packages": package_versions(),
        "command": sys.argv,
        "artifacts": artifacts,
    }
    write_json(analysis_root / "analysis_manifest.json", manifest)
    write_json(analysis_root / "plot_catalog.json", {"plots": artifacts})
    write_json(analysis_root / "_SUCCESS.json", {"status": "ok", "rounds": round_ids})
    return {
        "analysis_root": str(analysis_root),
        "rounds": round_ids,
        "cumulative_hits": metrics[-1]["cumulative_hits"] if metrics else 0,
    }


__all__ = ["analyze_run", "discover_run"]
