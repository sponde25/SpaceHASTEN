from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from spacehasten.analysis.acquisition import (
    RoundDefinition,
    discover_round_definitions,
    expected_improvement_scores,
    load_candidate_pool,
)
from spacehasten.analysis.policies import (
    capped_top_k,
    deterministic_top_k,
    load_compounds,
    penalized_top_k,
)
from spacehasten.analysis.replay import frontier_scale
from spacehasten.core.acquisition import expected_improvement


def _round_database(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE cluster_atlas_versions("
        "atlas_id TEXT, version INTEGER, last_spacehastenid INTEGER)"
    )
    connection.executemany(
        "INSERT INTO cluster_atlas_versions VALUES (?, ?, ?)",
        [("atlas-a", 1, 100), ("atlas-a", 2, 200), ("atlas-a", 3, 300)],
    )
    connection.commit()
    connection.close()
    return path


def _acquisition_metadata(path: Path, *, model_version: int, atlas_version: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_version", "cluster_atlas_version"])
        writer.writeheader()
        writer.writerow({"model_version": model_version, "cluster_atlas_version": atlas_version})


def test_selectors_preserve_deterministic_identifier_order_and_cap_failure() -> None:
    scores = np.array([1.0, 1.0, 1.0, 2.0], dtype=np.float64)
    identifiers = np.array([20, 10, 30, 40], dtype=np.int64)
    assert deterministic_top_k(scores, identifiers, 3).tolist() == [1, 0, 2]

    penalized_scores = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    penalized_ids = np.array([10, 11, 12], dtype=np.int64)
    clusters = np.array([1, 1, 2], dtype=np.int64)
    selected, counts, penalties = penalized_top_k(
        penalized_scores, penalized_ids, clusters, 3, cluster_lambda=1.0
    )
    assert penalized_ids[selected].tolist() == [10, 12, 11]
    assert counts.tolist() == [0, 0, 1]
    assert penalties.tolist() == pytest.approx([0.0, 0.0, np.log(2.0)])
    with pytest.raises(ValueError, match="cap capacity cannot fill count"):
        capped_top_k(penalized_scores, penalized_ids, clusters, 3, 1.0, cap=1)


def test_discovery_supports_arbitrary_positive_rounds_and_requires_atlas_id(tmp_path: Path) -> None:
    database = _round_database(tmp_path / "rounds.db")
    acquisition_root = tmp_path / "acquisitions"
    _acquisition_metadata(
        acquisition_root / "iter1" / "acquisition.csv", model_version=11, atlas_version=1
    )
    _acquisition_metadata(
        acquisition_root / "iter2" / "acquisition.csv", model_version=12, atlas_version=2
    )
    _acquisition_metadata(
        acquisition_root / "iter3" / "acquisition.csv", model_version=13, atlas_version=3
    )

    definitions = discover_round_definitions(database, acquisition_root, atlas_id="atlas-a")
    assert [(item.round_id, item.upper_id) for item in definitions] == [
        (1, 100),
        (2, 200),
        (3, 300),
    ]
    with pytest.raises(TypeError, match="atlas_id"):
        discover_round_definitions(database, acquisition_root)  # type: ignore[call-arg]

    (acquisition_root / "iter2" / "acquisition.csv").unlink()
    with pytest.raises(ValueError, match="contiguous"):
        discover_round_definitions(database, acquisition_root, atlas_id="atlas-a")


def test_candidate_pool_requires_explicit_atlas_id() -> None:
    definition = RoundDefinition(1, 1, 1, 1, "1=1")
    with pytest.raises(TypeError, match="atlas_id"):
        load_candidate_pool(Path("unused.db"), definition)  # type: ignore[call-arg]


def test_compound_staging_keeps_source_read_only(tmp_path: Path) -> None:
    database = tmp_path / "source.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE data(spacehastenid INTEGER PRIMARY KEY, reghash TEXT, "
            "smiles TEXT, dock_score REAL)"
        )
        connection.executemany(
            "INSERT INTO data VALUES (?, ?, ?, ?)",
            [(1, "a", "CC", -7.0), (2, "b", "CCC", None)],
        )
    before = database.read_bytes()
    frame = load_compounds(database, {1, 2})
    assert set(frame["spacehastenid"]) == {1, 2}
    assert database.read_bytes() == before


def test_ei_scores_match_production_implementation() -> None:
    means = np.array([-8.0, -7.0, -6.5], dtype=np.float64)
    epistemic = np.array([0.0, 0.25, 1.5], dtype=np.float64)
    threshold = -7.0
    xi = 0.01
    expected = np.array(
        [
            -expected_improvement(float(mean), float(std), threshold, xi)
            for mean, std in zip(means, epistemic, strict=True)
        ],
        dtype=np.float64,
    )
    assert expected_improvement_scores(means, epistemic, threshold, xi) == pytest.approx(expected)


def test_frontier_scale_uses_historical_windows() -> None:
    scores = np.arange(12, dtype=np.float64)
    identifiers = np.arange(100, 112, dtype=np.int64)
    scale = frontier_scale(scores, identifiers, batch_size=6)
    assert scale["primary_start_rank"] == 4
    assert scale["primary_stop_rank"] == 12
    assert scale["primary_scale"] == pytest.approx(6.4)
    assert scale["sensitivity_start_rank"] == 7
    assert scale["sensitivity_stop_rank"] == 12
    assert scale["sensitivity_scale"] == pytest.approx(4.0)
    assert scale["scale_ratio_sensitivity_over_primary"] == pytest.approx(4.0 / 6.4)
