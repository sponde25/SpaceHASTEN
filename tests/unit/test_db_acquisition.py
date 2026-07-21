"""Acquisition methods exercised against the legacy baseline fixture.

Fixture rows (see ``scripts/build_fixture_dbsh.py``):
  1 ethanol     dock_score=-7.5, dock_iter=0
  2 benzene     dock_score=-8.2, dock_iter=0
  3 toluene     pred_score=-6.7, simsearch_cycle=1
  4 phenol      pred_score=-7.1, spacelight=0.85, ftrees=0.72, simsearch_cycle=1
  5 aniline     query=1, simsearch_cycle=1
All clusters share clusterid=1.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from spacehasten.core.db import ClusterRow, Database

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
LEGACY_BASELINE = FIXTURES / "legacy_baseline.dbsh"


@pytest.fixture
def db(tmp_path: Path) -> Database:
    # Copy fixture into tmp_path so update tests don't mutate it.
    target = tmp_path / "baseline.dbsh"
    shutil.copy(LEGACY_BASELINE, target)
    return Database(target)


def test_simsearch_docked_greedy(db: Database) -> None:
    rows = db.select_queries_for_simsearch("docked", "greedy", limit=10)
    # Both docked rows are query=NULL; ORDER BY dock_score puts -8.2 before -7.5.
    assert rows == [("c1ccccc1", 2), ("CCO", 1)]


def test_simsearch_predicted_greedy(db: Database) -> None:
    rows = db.select_queries_for_simsearch("predicted", "greedy", limit=10)
    # Rows with pred_score, dock_score=NULL, query=NULL: phenol (-7.1), toluene (-6.7).
    assert rows == [("Oc1ccccc1", 4), ("Cc1ccccc1", 3)]


def test_simsearch_clustering_returns_one_per_cluster(db: Database) -> None:
    rows = db.select_queries_for_simsearch("docked", "clustering", limit=10)
    assert len(rows) == 1  # single cluster id in fixture


def test_dock_greedy(db: Database) -> None:
    rows = db.select_compounds_to_dock("greedy", limit=10)
    # Rows with dock_score IS NULL: 3, 4, 5. ORDER BY pred_score (NULLs first).
    sids = [sid for _, sid in rows]
    assert set(sids) == {3, 4, 5}
    # Aniline (5) has pred_score=NULL so it sorts first under SQLite default.
    assert sids[0] == 5
    # Phenol (-7.1) before toluene (-6.7).
    assert sids[1:] == [4, 3]


def test_dock_clustering_groups_by_cluster(db: Database) -> None:
    rows = db.select_compounds_to_dock("clustering", limit=10)
    assert len(rows) == 1


def test_select_undocked_for_prediction(db: Database) -> None:
    rows = list(db.select_undocked_for_prediction())
    sids = sorted(sid for _, sid in rows)
    assert sids == [3, 4, 5]


def test_uncertainty_candidates_use_each_rows_current_prediction_version(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "uncertainty.dbsh")
    database.create_schema()
    first = database.insert_seed_undocked("h1", "CC", "first")
    second = database.insert_seed_undocked("h2", "CCC", "second")
    database.apply_predictions(
        [
            (first, 1, -100.0, 50.0, 0.1, 50.0),
            (second, 1, -100.0, 50.0, 0.1, 50.0),
        ]
    )
    database.apply_predictions(
        [
            (first, 2, -8.0, 0.2, 0.1, 0.3),
            (second, 2, -7.0, 0.4, 0.1, 0.5),
        ]
    )
    database.replace_clusters(
        [
            ClusterRow(spacehastenid=first, clusterid=10),
            ClusterRow(spacehastenid=second, clusterid=20),
        ]
    )
    database.commit()

    candidates = database.select_uncertainty_docking_candidates()
    database.close()

    assert [candidate.model_version for candidate in candidates] == [2, 2]
    assert [candidate.pred_score for candidate in candidates] == [-8.0, -7.0]
    assert [candidate.epistemic_std for candidate in candidates] == [0.2, 0.4]
    assert [candidate.clusterid for candidate in candidates] == [10, 20]


def test_uncertainty_candidates_reject_missing_epistemic_uncertainty(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "missing-uncertainty.dbsh")
    database.create_schema()
    sid = database.insert_seed_undocked("h1", "CC", "first")
    database.apply_predictions([(sid, 1, -8.0, None, None, None)])
    database.commit()

    with pytest.raises(ValueError, match="epistemic uncertainty"):
        database.select_uncertainty_docking_candidates()
    database.close()


def test_select_training_data(db: Database) -> None:
    rows = db.select_training_data(cutoff=10.0)
    rows_sorted = sorted(rows, key=lambda r: r[1])
    assert rows_sorted == [("c1ccccc1", -8.2), ("CCO", -7.5)]


def test_select_export_rows_empty_when_cutoff_excludes_all(db: Database) -> None:
    rows = db.select_export_rows(cutoff=-100.0)
    assert rows == []


def test_select_export_rows_includes_docked(db: Database) -> None:
    rows = db.select_export_rows(cutoff=0.0)
    sids = sorted(r.spacehastenid for r in rows)
    # Only docked rows (1, 2); rows with dock_score NULL are excluded by `<= ?`.
    assert sids == [1, 2]
    assert rows[0].dock_score == -8.2  # ORDER BY dock_score


def test_latest_helpers(db: Database) -> None:
    assert db.latest_model_version() == 1
    assert db.latest_simsearch_cycle() == 1
    assert db.latest_dock_iteration() == 0


def test_load_blobs_and_model(db: Database) -> None:
    assert db.load_dock_grid().startswith(b"PK\x05\x06")
    assert db.load_dock_param()  # non-empty
    assert db.load_model_blob(1) == b"dummytar"


def test_load_properties(db: Database) -> None:
    props = db.load_properties()
    assert props is not None
    assert props.mw == ("0.0", "500.0")
    assert props.hba == ("0", "10")


def test_update_dock_score_roundtrip(db: Database) -> None:
    db.update_dock_score(spacehastenid=3, dock_score=-9.9, dock_iteration=2)
    db.commit()
    row = db.connection.execute(
        "SELECT dock_score, dock_iteration FROM data WHERE spacehastenid = 3"
    ).fetchone()
    assert row == (-9.9, 2)


def test_update_pred_score_roundtrip(db: Database) -> None:
    db.update_pred_score(spacehastenid=1, pred_score=-1.23, pred_version=2)
    db.commit()
    row = db.connection.execute(
        "SELECT pred_score, pred_version FROM data WHERE spacehastenid = 1"
    ).fetchone()
    assert row == (-1.23, 2)


def test_apply_predictions_persists_uncertainty_history(db: Database) -> None:
    db.apply_predictions(
        [
            (1, 2, -7.25, 0.20, 0.15, 0.25),
            (2, 2, -6.50, 0.30, 0.40, 0.50),
        ]
    )
    db.commit()

    rows = db.select_predictions(model_version=2)
    assert [(row.spacehastenid, row.pred_score) for row in rows] == [
        (1, -7.25),
        (2, -6.50),
    ]
    assert rows[0].epistemic_std == pytest.approx(0.20)
    assert rows[0].aleatoric_std == pytest.approx(0.15)
    assert rows[0].total_std == pytest.approx(0.25)
    latest = db.connection.execute(
        "SELECT pred_score, pred_version FROM data WHERE spacehastenid = 1"
    ).fetchone()
    assert latest == (-7.25, 2)


def test_mark_as_query(db: Database) -> None:
    db.mark_as_query(spacehastenid=3, cycle=7)
    db.commit()
    (q,) = db.connection.execute("SELECT query FROM data WHERE spacehastenid = 3").fetchone()
    assert q == 7
