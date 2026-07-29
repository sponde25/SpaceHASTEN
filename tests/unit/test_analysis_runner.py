from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest
from rdkit import Chem

from spacehasten.analysis import AnalysisConfig, analyze_run, discover_run
from spacehasten.analysis.chemistry import ACYCLIC, diversity, family_labels
from spacehasten.core.db import Database


def _database(path: Path, *, atlas_ids: tuple[str, ...] = ("atlas-a",)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE data(spacehastenid INTEGER PRIMARY KEY, smiles TEXT, "
        "dock_score REAL, dock_iteration INTEGER)"
    )
    connection.executemany(
        "INSERT INTO data VALUES (?, ?, ?, ?)",
        [
            (1, "CCO", -12.0, 0),
            (2, "CCN", -7.0, 1),
            (3, "not-smiles", None, 1),
            (4, "c1ccccc1", -6.0, 2),
            (5, "CCCl", -8.0, 3),
        ],
    )
    if atlas_ids:
        connection.execute(
            "CREATE TABLE cluster_atlas_assignments(atlas_id TEXT, "
            "spacehastenid INTEGER, clusterid INTEGER)"
        )
        for atlas_id in atlas_ids:
            connection.executemany(
                "INSERT INTO cluster_atlas_assignments VALUES (?, ?, ?)",
                [(atlas_id, identifier, identifier % 2) for identifier in range(1, 6)],
            )
    connection.commit()
    connection.close()
    return path


def _acquisition(path: Path, identifiers: list[int], *, atlas_id: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "spacehastenid",
        "clusterid",
        "optional",
        "candidate_count",
        "cluster_alpha",
        "cluster_lambda",
        "cluster_penalty",
        "cluster_count_before",
        "pred_score",
        "epistemic_std",
    ]
    if atlas_id is not None:
        fields.append("atlas_id")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, identifier in enumerate(identifiers, 1):
            row = {
                "rank": rank,
                "spacehastenid": identifier,
                "clusterid": identifier % 2,
                "optional": "x",
                "candidate_count": 100,
                "cluster_alpha": 0.2,
                "cluster_lambda": 0.1,
                "cluster_penalty": 0.0 if rank == 1 else 0.1,
                "cluster_count_before": rank - 1,
                "pred_score": -8.0 + rank,
                "epistemic_std": 0.5,
            }
            if atlas_id is not None:
                row["atlas_id"] = atlas_id
            writer.writerow(row)


def _calibration_acquisition(path: Path, identifiers: list[int], model_version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["spacehastenid", "model_version"])
        writer.writeheader()
        writer.writerows(
            {"spacehastenid": identifier, "model_version": model_version}
            for identifier in identifiers
        )


def _predictions_database(path: Path) -> Path:
    database = _database(path, atlas_ids=())
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE data SET dock_score = -6.0 WHERE spacehastenid = 3")
        connection.execute(
            "CREATE TABLE predictions(spacehastenid INTEGER, model_version TEXT, pred_score REAL, "
            "epistemic_std REAL, aleatoric_std REAL)"
        )
        connection.executemany(
            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
            [
                (2, "v1", -7.0, 0.0, 0.0),
                (3, "v1", -6.0, 0.0, 0.0),
                (2, "v2", -100.0, 0.0, 0.0),
                (4, "v2", -8.0, 0.0, 0.0),
            ],
        )
    return database


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _run(path: Path, root: Path) -> Path:
    context = discover_run(path)
    analyze_run(
        context, AnalysisConfig(-7.0, (-8.0, -7.0), pair_samples=2, dpi=50), root, overwrite=False
    )
    return root


def test_real_layout_outer_run_local_and_docking_acquisitions(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    database = _database(workspace / "run.dbsh")
    outer = tmp_path / "experiment"
    outer.mkdir()
    (outer / "run_local").symlink_to(workspace, target_is_directory=True)
    _acquisition(
        outer / "run_shared" / "docking" / "iter1" / "acquisition.csv",
        [2, 3, 99],
        atlas_id="atlas-a",
    )
    _acquisition(
        outer / "run_shared" / "docking" / "iter2" / "acquisition.csv", [4, 2], atlas_id="atlas-a"
    )
    _acquisition(outer / "run_shared" / "iter9" / "acquisition.csv", [5])
    context = discover_run(outer)
    assert context.database_path == database.resolve()
    assert [round_id for round_id, _ in context.acquisition_paths] == [1, 2]
    root = _run(outer, tmp_path / "analysis")
    metrics = _read_csv(root / "round_metrics.csv")
    coverage = _read_csv(root / "coverage.csv")
    family = _read_csv(root / "family_metrics.csv")
    acquisition = _read_csv(root / "acquisition_metrics.csv")
    assert [row["round"] for row in metrics] == ["1", "2"]
    assert metrics[0]["selected"] == "3"
    assert metrics[0]["scored"] == "1"
    assert metrics[0]["hits"] == "1"  # equality at -7 is a hit
    assert metrics[1]["cross_round_reused_attempts"] == "1"
    assert metrics[1]["scored"] == "1"
    assert metrics[1]["missing"] == "1"
    assert metrics[1]["cumulative_selected"] == "5"
    assert coverage[0]["status"] == "partial"
    assert family[0]["atlas_status"] == "available"
    assert family[0]["generic_richness"] == "1"
    assert acquisition[0]["candidate_count"] == "100"
    assert acquisition[0]["selected_candidate_fraction"] == "0.03"
    assert acquisition[0]["cluster_penalty_nonzero_fraction"] == str(2 / 3)
    assert acquisition[0]["cluster_count_before_max"] == "2.0"
    assert _read_csv(root / "budget_curve.csv")[-1]["selected_budget"] == "5"
    cutoff = _read_csv(root / "cutoff_curve.csv")
    final_cutoffs = {float(row["cutoff"]): row for row in cutoff if row["round"] == "2"}
    assert final_cutoffs[-8.0]["hits"] == "0"
    assert final_cutoffs[-8.0]["scored"] == "2"
    assert final_cutoffs[-8.0]["hit_rate_scored"] == "0.0"
    assert final_cutoffs[-7.0]["hits"] == "1"
    assert final_cutoffs[-7.0]["hit_rate_scored"] == "0.5"
    assert len(_read_csv(root / "score_distribution.csv")) == 202
    assert (root / "cumulative_hits.png").stat().st_size > 0
    assert (root / "score_ecdf.png").stat().st_size > 0
    assert (root / "acquisition_diagnostics.png").stat().st_size > 0
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM data").fetchone()[0] == 5


def test_outer_discovery_prefers_canonical_run_local_database(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    database = _database(workspace / "run.dbsh", atlas_ids=())
    outer = tmp_path / "experiment"
    outer.mkdir()
    (outer / "run_local").symlink_to(workspace, target_is_directory=True)
    _database(outer / "analysis" / "interim.dbsh", atlas_ids=())
    assert discover_run(outer).database_path == database.resolve()


def test_database_override_preserves_run_acquisition_discovery(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outer = tmp_path / "experiment"
    outer.mkdir()
    (outer / "run_local").symlink_to(workspace, target_is_directory=True)
    acquisition = outer / "run_shared" / "docking" / "iter1" / "acquisition.csv"
    _acquisition(acquisition, [2, 3], atlas_id="atlas-a")
    snapshot = _database(tmp_path / "snapshots" / "final.dbsh")

    context = discover_run(outer, database_path=snapshot)

    assert context.database_path == snapshot.resolve()
    assert context.input_path == outer.absolute()
    assert context.acquisition_paths == ((1, acquisition.absolute()),)
    assert context.capabilities.has_data


def test_database_override_must_exist_and_contain_data(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _database(run / "run.dbsh")
    with pytest.raises(FileNotFoundError):
        discover_run(run, database_path=tmp_path / "missing.dbsh")

    invalid = tmp_path / "invalid.dbsh"
    sqlite3.connect(invalid).close()
    with pytest.raises(ValueError, match="does not contain a data table"):
        discover_run(run, database_path=invalid)


def test_analysis_recovers_attempt_denominators_from_docking_inputs(tmp_path: Path) -> None:
    run = tmp_path / "recovered"
    run.mkdir()
    database_path = run / "run.dbsh"
    with Database(database_path) as database:
        database.create_schema()
        database.connection.executemany(
            "INSERT INTO data(spacehastenid,reghash,smiles,dock_score,pred_score,"
            "dock_iteration,pred_version) VALUES (?,?,?,?,?,?,?)",
            [
                (1, "seed", "C", -5.0, None, 0, None),
                (2, "h2", "CC", -9.0, -8.0, 1, None),
                (3, "h3", "CCC", None, -9.0, None, None),
                (4, "h4", "CCCC", -10.0, -7.0, 2, 1),
            ],
        )
        database.connection.executemany(
            "INSERT INTO predictions(spacehastenid,model_version,pred_score,epistemic_std) "
            "VALUES (?,?,?,?)",
            [(2, 0, -8.0, 0.2), (4, 1, -7.0, 0.4)],
        )
        database.commit()
    for round_id, rows in ((1, ("CC 2", "CCC 3")), (2, ("CCCC 4",))):
        path = run / f"run_shared/docking/iter{round_id}/inputs/chunk_1.smi"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    root = _run(run, tmp_path / "recovered-analysis")
    metrics = _read_csv(root / "round_metrics.csv")
    coverage = _read_csv(root / "coverage.csv")
    acquisition = _read_csv(root / "acquisition_metrics.csv")

    assert metrics[0]["selected"] == "2"
    assert metrics[0]["scored"] == "1"
    assert metrics[0]["missing"] == "1"
    assert metrics[1]["selected"] == "1"
    assert metrics[1]["scored"] == "1"
    assert coverage[0]["status"] == "partial"
    assert coverage[0]["source"] == "docking_input_chunks"
    assert acquisition[0]["selection_source"] == "docking_input_chunks"
    assert acquisition[0]["rank_source"] == "data_prediction_score_then_id"


def test_seed_exclusion_threshold_equality_and_legacy_layout(tmp_path: Path) -> None:
    database = _database(tmp_path / "run" / "run.dbsh", atlas_ids=())
    _acquisition(tmp_path / "run" / "run_shared" / "iter1" / "acquisition.csv", [1, 2, 2])
    root = _run(database, tmp_path / "analysis")
    metrics = _read_csv(root / "round_metrics.csv")
    assert metrics[0]["selected"] == "3"
    assert metrics[0]["scored"] == "2"  # seed ID 1 is deliberately excluded
    assert metrics[0]["hits"] == "2"
    assert metrics[0]["within_round_duplicate_attempts"] == "1"
    assert metrics[0]["cumulative_selected_unique"] == "2"
    assert _read_csv(root / "cutoff_curve.csv")[1]["hits"] == "2"


def test_ambiguous_database_and_atlas_are_reported(tmp_path: Path) -> None:
    ambiguous = tmp_path / "ambiguous"
    _database(ambiguous / "a.dbsh")
    _database(ambiguous / "b.dbsh")
    with pytest.raises(ValueError, match="exactly one database"):
        discover_run(ambiguous)
    database = _database(tmp_path / "atlas" / "run.dbsh", atlas_ids=("a", "b"))
    _acquisition(tmp_path / "atlas" / "run_shared" / "docking" / "iter1" / "acquisition.csv", [2])
    root = _run(database, tmp_path / "atlas-analysis")
    family = _read_csv(root / "family_metrics.csv")
    assert family[0]["atlas_status"] == "ambiguous"
    assert family[0]["atlas_reason"] == "database contains multiple atlas IDs"


def test_large_n_diversity_sampling_does_not_enumerate_pairs() -> None:
    molecule = Chem.MolFromSmiles("CCO")
    assert molecule is not None
    assert family_labels(molecule) == (ACYCLIC, ACYCLIC)
    metrics = diversity([molecule] * 100_000, AnalysisConfig(-7.0, (-7.0,), pair_samples=3))
    assert metrics["pair_samples_used"] == 3
    assert metrics["internal_diversity"] == 0.0


def test_calibration_uses_exact_model_version_and_zero_std_is_deterministic(tmp_path: Path) -> None:
    database = _predictions_database(tmp_path / "run" / "run.dbsh")
    _calibration_acquisition(
        tmp_path / "run" / "run_shared" / "iter1" / "acquisition.csv", [2, 3, 99], "v1"
    )
    _calibration_acquisition(
        tmp_path / "run" / "run_shared" / "iter2" / "acquisition.csv", [2, 4], "v2"
    )
    root = _run(database, tmp_path / "analysis")
    metrics = _read_csv(root / "calibration_metrics.csv")
    curve = _read_csv(root / "calibration_curve.csv")
    first, second = metrics
    assert first["model_version"] == "v1"
    assert first["mean_source"] == "predictions.pred_score"
    assert first["probability_source"] == "raw_gaussian_pred_score_total_std"
    assert first["observed_outcomes"] == "2"
    assert first["matched_predictions"] == "2"
    assert first["coverage"] == str(2 / 3)
    assert first["bias"] == "0.0"
    assert first["mae"] == "0.0"
    assert first["rmse"] == "0.0"
    assert first["predicted_hit_rate"] == "0.5"
    assert first["observed_hit_rate"] == "0.5"
    assert first["brier_score"] == "0.0"
    assert first["clipped_log_loss"] != "0.0"
    assert first["interval_95_coverage"] == "1.0"
    assert second["model_version"] == "v2"
    assert second["matched_predictions"] == "1"
    assert second["predicted_hit_rate"] == "1.0"  # v2 value, not v1 or latest-by-ID behavior
    assert second["coverage"] == "0.5"  # ID 4 has an outcome but no v2 prediction
    assert [(row["bin_lower"], row["count"]) for row in curve if row["round"] == "1"] == [
        ("0.0", "1"),
        ("0.9", "1"),
    ]
    assert {row["probability_source"] for row in curve} == {"raw_gaussian_pred_score_total_std"}
    assert (root / "calibration_reliability.png").stat().st_size > 0
    assert (root / "calibration_reliability.pdf").stat().st_size > 0


def test_calibration_skips_explicitly_without_predictions_table(tmp_path: Path) -> None:
    database = _database(tmp_path / "run" / "run.dbsh", atlas_ids=())
    _calibration_acquisition(
        tmp_path / "run" / "run_shared" / "iter1" / "acquisition.csv", [2], "v1"
    )
    root = _run(database, tmp_path / "analysis")
    metrics = _read_csv(root / "calibration_metrics.csv")
    assert len(metrics) == 1
    assert metrics[0]["status"] == "unavailable"
    assert metrics[0]["reason"] == "predictions table absent"
    assert metrics[0]["matched_predictions"] == "0"
    assert metrics[0]["brier_score"] == ""
    assert not (root / "calibration_reliability.png").exists()
