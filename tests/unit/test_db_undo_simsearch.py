"""Tests for :meth:`Database.undo_simsearch_cycle` and its helpers.

Builds small, realistic scenarios from scratch (rather than reusing the
shared ``legacy_baseline.dbsh`` fixture) so that the invariant "a compound's
``query`` value is always strictly greater than its own ``simsearch_cycle``"
holds — that invariant is what makes ``undo_simsearch_cycle`` safe to call
without a ``--cycle`` argument.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacehasten.core.db import ClusterRow, Database


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "fresh.dbsh")
    database.create_schema()
    return database


def test_latest_search_attempt_cycle_none_when_empty(db: Database) -> None:
    assert db.latest_search_attempt_cycle() is None


def test_latest_search_attempt_cycle_matches_successful_cycle(db: Database) -> None:
    seed_id = db.insert_seed_docked("h1", "CCO", "SEED1", dock_score=-7.0)
    db.mark_as_query(seed_id, cycle=1)
    db.insert_simsearch_hit(
        "h2", "c1ccccc1", "HIT1",
        spacelight=0.9, ftrees=0.8, pred_score=-6.0, simsearch_cycle=1,
    )
    db.commit()

    assert db.latest_simsearch_cycle() == 1
    assert db.latest_search_attempt_cycle() == 1


def test_latest_search_attempt_cycle_detects_failed_cycle(db: Database) -> None:
    seed_id = db.insert_seed_docked("h1", "CCO", "SEED1", dock_score=-7.0)
    # Cycle 1 completes successfully.
    db.mark_as_query(seed_id, cycle=1)
    db.insert_simsearch_hit(
        "h2", "c1ccccc1", "HIT1",
        spacelight=0.9, ftrees=0.8, pred_score=-6.0, simsearch_cycle=1,
    )
    db.commit()

    # Cycle 2 is attempted: queries marked and committed, then the search
    # job dies before any simsearch_cycle=2 hits are inserted.
    other_seed = db.insert_seed_docked("h3", "CCN", "SEED2", dock_score=-8.0)
    db.mark_as_query(other_seed, cycle=2)
    db.commit()

    assert db.latest_simsearch_cycle() == 1  # no hits recorded for cycle 2
    assert db.latest_search_attempt_cycle() == 2  # but it *was* attempted


def test_simsearch_cycle_stats(db: Database) -> None:
    seed_id = db.insert_seed_docked("h1", "CCO", "SEED1", dock_score=-7.0)
    db.mark_as_query(seed_id, cycle=1)
    hit_id = db.insert_simsearch_hit(
        "h2", "c1ccccc1", "HIT1",
        spacelight=0.9, ftrees=0.8, pred_score=-6.0, simsearch_cycle=1,
    )
    db.commit()

    stats = db.simsearch_cycle_stats(1)
    assert stats.cycle == 1
    assert stats.n_hits == 1
    assert stats.n_queries == 1
    assert stats.n_hits_docked == 0
    assert stats.n_hits_used_as_query == 0
    assert hit_id > seed_id  # sanity: hit inserted after the seed/query


def test_undo_simsearch_cycle_removes_hits_and_releases_queries(db: Database) -> None:
    seed_id = db.insert_seed_docked("h1", "CCO", "SEED1", dock_score=-7.0)
    db.mark_as_query(seed_id, cycle=1)
    hit1 = db.insert_simsearch_hit(
        "h2", "c1ccccc1", "HIT1",
        spacelight=0.9, ftrees=0.8, pred_score=-6.0, simsearch_cycle=1,
    )
    hit2 = db.insert_simsearch_hit(
        "h3", "Cc1ccccc1", "HIT2",
        spacelight=0.7, ftrees=0.6, pred_score=-5.5, simsearch_cycle=1,
    )
    db.replace_clusters(
        [ClusterRow(seed_id, 1), ClusterRow(hit1, 1), ClusterRow(hit2, 1)]
    )
    db.commit()

    n_hits, n_queries = db.undo_simsearch_cycle(1)
    assert (n_hits, n_queries) == (2, 1)

    remaining_ids = {
        row[0] for row in db.connection.execute("SELECT spacehastenid FROM data")
    }
    assert remaining_ids == {seed_id}

    (q,) = db.connection.execute(
        "SELECT query FROM data WHERE spacehastenid = ?", (seed_id,)
    ).fetchone()
    assert q is None  # query mark released

    cluster_ids = {
        row[0] for row in db.connection.execute("SELECT spacehastenid FROM clusters")
    }
    assert cluster_ids.isdisjoint({hit1, hit2})  # stale cluster rows cleaned up


def test_undo_simsearch_cycle_handles_failed_attempt_with_no_hits(db: Database) -> None:
    seed_id = db.insert_seed_docked("h1", "CCO", "SEED1", dock_score=-7.0)
    db.mark_as_query(seed_id, cycle=1)
    db.commit()

    n_hits, n_queries = db.undo_simsearch_cycle(1)
    assert (n_hits, n_queries) == (0, 1)

    (q,) = db.connection.execute(
        "SELECT query FROM data WHERE spacehastenid = ?", (seed_id,)
    ).fetchone()
    assert q is None


def test_undo_simsearch_cycle_refuses_when_hit_already_docked(db: Database) -> None:
    seed_id = db.insert_seed_docked("h1", "CCO", "SEED1", dock_score=-7.0)
    db.mark_as_query(seed_id, cycle=1)
    hit_id = db.insert_simsearch_hit(
        "h2", "c1ccccc1", "HIT1",
        spacelight=0.9, ftrees=0.8, pred_score=-6.0, simsearch_cycle=1,
    )
    db.commit()

    # The hit got docked before anyone noticed the cycle should be undone.
    db.update_dock_score(hit_id, dock_score=-9.9, dock_iteration=0)
    db.commit()

    with pytest.raises(ValueError, match="already been docked"):
        db.undo_simsearch_cycle(1)
    # Nothing was removed.
    assert db.simsearch_cycle_stats(1).n_hits == 1


def test_undo_simsearch_cycle_refuses_when_hit_used_as_later_query(db: Database) -> None:
    seed_id = db.insert_seed_docked("h1", "CCO", "SEED1", dock_score=-7.0)
    db.mark_as_query(seed_id, cycle=1)
    hit_id = db.insert_simsearch_hit(
        "h2", "c1ccccc1", "HIT1",
        spacelight=0.9, ftrees=0.8, pred_score=-6.0, simsearch_cycle=1,
    )
    db.commit()

    # cycle 1's own hit was later selected as a query for cycle 2 — cycle 1
    # is therefore not really "the latest search attempt" anymore.
    db.mark_as_query(hit_id, cycle=2)
    db.commit()

    with pytest.raises(ValueError, match="not actually the latest search attempt"):
        db.undo_simsearch_cycle(1)
    assert db.simsearch_cycle_stats(1).n_hits == 1
