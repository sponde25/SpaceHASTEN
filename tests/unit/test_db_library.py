"""Tests for the library-screening DB additions.

Covers :meth:`Database.insert_library_hit`,
:meth:`Database.seed_dock_score_percentile`, and
:meth:`Database.filter_existing_reghashes` (plan §5.6).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacehasten.core.db import Database


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "fresh.dbsh")
    database.create_schema()
    return database


# --------------------------------------------------------------------------- #
# insert_library_hit                                                          #
# --------------------------------------------------------------------------- #


def test_insert_library_hit_sets_pred_score_and_leaves_simsearch_cycle_null(
    db: Database,
) -> None:
    rowid = db.insert_library_hit(
        "hlib1", "CCO", "ENA-1", pred_score=-8.3, pred_version=2,
    )
    db.commit()

    row = db.connection.execute(
        "SELECT reghash, smiles, smilesid, pred_score, pred_version,"
        " simsearch_cycle, dock_score, query, dock_iteration"
        " FROM data WHERE spacehastenid = ?",
        (rowid,),
    ).fetchone()
    assert row == ("hlib1", "CCO", "ENA-1", -8.3, 2, None, None, None, None)


def test_insert_library_hit_is_distinguishable_from_seed_and_simsearch(
    db: Database,
) -> None:
    db.insert_seed_undocked("hseed", "CCO", "SEED-1")
    db.insert_simsearch_hit(
        "hsim", "c1ccccc1", "SIM-1",
        spacelight=0.9, ftrees=0.8, pred_score=-6.0, simsearch_cycle=1,
    )
    db.insert_library_hit("hlib", "c1ccccn1", "ENA-2", pred_score=-9.0, pred_version=1)
    db.commit()

    # library-screened undocked = pred_score IS NOT NULL AND simsearch_cycle
    # IS NULL AND dock_score IS NULL (plan §D1 table).
    rows = db.connection.execute(
        "SELECT reghash FROM data"
        " WHERE pred_score IS NOT NULL AND simsearch_cycle IS NULL"
        " AND dock_score IS NULL"
    ).fetchall()
    assert [r[0] for r in rows] == ["hlib"]


# --------------------------------------------------------------------------- #
# seed_dock_score_percentile                                                  #
# --------------------------------------------------------------------------- #


def test_seed_dock_score_percentile_none_when_no_seeds(db: Database) -> None:
    assert db.seed_dock_score_percentile(1.0) is None


def test_seed_dock_score_percentile_first_percentile(db: Database) -> None:
    # 100 seed dock scores: -100.0 .. -1.0 (best = -100.0).
    for i in range(1, 101):
        db.insert_seed_docked(f"h{i}", "CCO", f"SEED-{i}", dock_score=float(-i))
    db.commit()

    # 1st percentile of 100 sorted-ascending values -> index 0 -> -100.0.
    assert db.seed_dock_score_percentile(1.0) == -100.0


def test_seed_dock_score_percentile_50th_percentile(db: Database) -> None:
    for i in range(1, 101):
        db.insert_seed_docked(f"h{i}", "CCO", f"SEED-{i}", dock_score=float(-i))
    db.commit()

    # 50th percentile -> ceil(100 * 0.5) = 50th smallest value -> -51.0.
    assert db.seed_dock_score_percentile(50.0) == -51.0


def test_seed_dock_score_percentile_ignores_non_seed_dock_scores(db: Database) -> None:
    db.insert_seed_docked("h1", "CCO", "SEED-1", dock_score=-5.0)
    # Later docking round (dock_iteration=1) must not affect the seed percentile.
    other = db.insert_seed_undocked("h2", "c1ccccc1", "OTHER-1")
    db.update_dock_score(other, dock_score=-99.0, dock_iteration=1)
    db.commit()

    assert db.seed_dock_score_percentile(100.0) == -5.0


# --------------------------------------------------------------------------- #
# filter_existing_reghashes                                                   #
# --------------------------------------------------------------------------- #


def test_filter_existing_reghashes_empty_candidates(db: Database) -> None:
    assert db.filter_existing_reghashes([]) == set()


def test_filter_existing_reghashes_returns_only_present(db: Database) -> None:
    db.insert_seed_undocked("h1", "CCO", "SEED-1")
    db.insert_seed_undocked("h2", "c1ccccc1", "SEED-2")
    db.commit()

    found = db.filter_existing_reghashes(["h1", "h2", "h3"])
    assert found == {"h1", "h2"}


def test_filter_existing_reghashes_chunks_large_batches(db: Database) -> None:
    # Insert more than one chunk's worth (chunk size is 500) of reghashes.
    n = 1200
    for i in range(n):
        db.insert_seed_undocked(f"h{i}", "CCO", f"SEED-{i}")
    db.commit()

    candidates = [f"h{i}" for i in range(n)] + ["missing1", "missing2"]
    found = db.filter_existing_reghashes(candidates)
    assert len(found) == n
    assert "missing1" not in found
    assert "missing2" not in found
