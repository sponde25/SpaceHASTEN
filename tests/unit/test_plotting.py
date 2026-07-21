"""Tests for :mod:`spacehasten.stages.plotting`."""

from __future__ import annotations

from pathlib import Path

import pytest

from spacehasten.core.db import Database
from spacehasten.stages.plotting import (
    plot_dock_score_distribution,
    plot_pred_score_distribution,
    plot_pred_vs_dock_accuracy,
)
from spacehasten.workspace.layout import WorkDir


def _seed_db(db: Database) -> None:
    """Build a small but representative multi-round database:

    - dock_iteration 0 (seed): 20 docked compounds, scores -10..-1 (lower better)
    - query=1 marks the top 5 best (most negative) seed scores as cycle-1 queries
    - simsearch_cycle 1: 10 hits with pred_score, later docked at iteration 1
    - simsearch_cycle 2: 10 hits with pred_score, later docked at iteration 2
    """
    db.create_schema()
    seed_scores = [-1.0 * (i + 1) for i in range(20)]  # -1 .. -20
    for i, score in enumerate(seed_scores):
        db.insert_seed_docked(f"seed{i}", "CCO", f"seed-{i}", score)
    # Mark the 5 best (most negative) seed scores as cycle-1 queries.
    # spacehastenid 1..20 map 1:1 to seed_scores index+1 (insertion order).
    best_ids = sorted(range(1, 21), key=lambda sid: seed_scores[sid - 1])[:5]
    for sid in best_ids:
        db.mark_as_query(sid, 1)

    # Cycle 1 hits: predicted scores, later docked at iteration 1.
    cycle1_ids = []
    for i in range(10):
        sid = db.insert_simsearch_hit(
            f"c1hit{i}", "CCN", f"c1-{i}", None, None, -5.0 - i, 1
        )
        cycle1_ids.append(sid)
    db.apply_dock_scores([(-6.0 - i, 1, sid) for i, sid in enumerate(cycle1_ids)])

    # Cycle 2 hits: predicted scores, later docked at iteration 2.
    cycle2_ids = []
    for i in range(10):
        sid = db.insert_simsearch_hit(
            f"c2hit{i}", "CCC", f"c2-{i}", None, None, -7.0 - i, 2
        )
        cycle2_ids.append(sid)
    db.apply_dock_scores([(-7.5 - i, 2, sid) for i, sid in enumerate(cycle2_ids)])

    db.commit()


def _make_db(tmp_path: Path) -> tuple[WorkDir, Database]:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="plt")
    db = Database(workdir.dbsh())
    return workdir, db


def test_query_cycle1_score_cutoff(tmp_path: Path) -> None:
    workdir, db = _make_db(tmp_path)
    _seed_db(db)
    # Best 5 seed scores are -16..-20 (most negative); worst of those is -16.
    assert db.query_cycle1_score_cutoff() == -16.0
    db.close()


def test_query_cycle1_score_cutoff_none_when_no_queries(tmp_path: Path) -> None:
    workdir, db = _make_db(tmp_path)
    db.create_schema()
    db.insert_seed_docked("h1", "CCO", "ethanol-1", -8.5)
    db.commit()
    assert db.query_cycle1_score_cutoff() is None
    db.close()


def test_dock_scores_by_iteration(tmp_path: Path) -> None:
    workdir, db = _make_db(tmp_path)
    _seed_db(db)
    by_iter = db.dock_scores_by_iteration()
    assert set(by_iter) == {0, 1, 2}
    assert len(by_iter[0]) == 20
    assert len(by_iter[1]) == 10
    assert len(by_iter[2]) == 10
    db.close()


def test_pred_scores_by_simsearch_cycle(tmp_path: Path) -> None:
    workdir, db = _make_db(tmp_path)
    _seed_db(db)
    by_cycle = db.pred_scores_by_simsearch_cycle()
    assert set(by_cycle) == {1, 2}
    assert len(by_cycle[1]) == 10
    assert len(by_cycle[2]) == 10
    db.close()


def test_pred_vs_dock_pairs(tmp_path: Path) -> None:
    workdir, db = _make_db(tmp_path)
    _seed_db(db)
    pairs = db.pred_vs_dock_pairs(1)
    assert len(pairs) == 10
    pred, dock = pairs[0]
    assert pred == -5.0
    assert dock == -6.0
    assert db.pred_vs_dock_pairs(99) == []
    db.close()


def test_plot_dock_score_distribution_writes_png(tmp_path: Path) -> None:
    workdir, db = _make_db(tmp_path)
    _seed_db(db)
    out = plot_dock_score_distribution(db, workdir)
    db.close()
    assert out == workdir.plots_dir() / "dock_score_distribution.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_dock_score_distribution_caps_xlim_at_zero_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt

    workdir, db = _make_db(tmp_path)
    db.create_schema()
    # Mostly negative scores with a few positive outliers so the natural
    # buffered xlim would exceed 0.
    for i in range(10):
        db.insert_seed_docked(f"h{i}", "CCO", f"h-{i}", -10.0 + i * 1.5)
    db.commit()

    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    try:
        plot_dock_score_distribution(db, workdir)
        fig = plt.gcf()
        xlim_max = fig.axes[0].get_xlim()[1]
    finally:
        real_close("all")
        db.close()

    assert xlim_max == 0.0


def test_plot_dock_score_distribution_show_all_scores(tmp_path: Path) -> None:
    workdir, db = _make_db(tmp_path)
    db.create_schema()
    for i in range(10):
        db.insert_seed_docked(f"h{i}", "CCO", f"h-{i}", 5.0 + i)  # all positive
    db.commit()
    out = plot_dock_score_distribution(db, workdir, max_dock_score=None)
    db.close()
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_dock_score_distribution_requires_seed_data(tmp_path: Path) -> None:
    workdir, db = _make_db(tmp_path)
    db.create_schema()
    db.commit()
    try:
        with pytest.raises(ValueError, match="no seed docking scores"):
            plot_dock_score_distribution(db, workdir)
    finally:
        db.close()


def test_plot_pred_score_distribution_writes_png(tmp_path: Path) -> None:
    workdir, db = _make_db(tmp_path)
    _seed_db(db)
    out = plot_pred_score_distribution(db, workdir)
    db.close()
    assert out == workdir.plots_dir() / "pred_score_distribution.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_pred_score_distribution_caps_xlim_at_zero_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt

    workdir, db = _make_db(tmp_path)
    db.create_schema()
    db.insert_seed_docked("seed0", "CCO", "seed-0", -5.0)
    # Predicted scores with a few positive outliers so the natural
    # buffered xlim would exceed 0.
    for i in range(10):
        sid = db.insert_simsearch_hit(
            f"c1hit{i}", "CCN", f"c1-{i}", None, None, -10.0 + i * 1.5, 1
        )
    db.commit()

    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)
    try:
        plot_pred_score_distribution(db, workdir)
        fig = plt.gcf()
        xlim_max = fig.axes[0].get_xlim()[1]
    finally:
        real_close("all")
        db.close()

    assert xlim_max == 0.0


def test_plot_pred_score_distribution_show_all_scores(tmp_path: Path) -> None:
    workdir, db = _make_db(tmp_path)
    db.create_schema()
    db.insert_seed_docked("seed0", "CCO", "seed-0", -5.0)
    for i in range(10):
        db.insert_simsearch_hit(
            f"c1hit{i}", "CCN", f"c1-{i}", None, None, 5.0 + i, 1
        )
    db.commit()
    out = plot_pred_score_distribution(db, workdir, max_dock_score=None)
    db.close()
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_pred_score_distribution_requires_predictions(tmp_path: Path) -> None:
    workdir, db = _make_db(tmp_path)
    db.create_schema()
    db.insert_seed_docked("h1", "CCO", "ethanol-1", -8.5)
    db.commit()
    try:
        with pytest.raises(ValueError, match="no predicted scores"):
            plot_pred_score_distribution(db, workdir)
    finally:
        db.close()


def test_plot_pred_vs_dock_accuracy_writes_png(tmp_path: Path) -> None:
    workdir, db = _make_db(tmp_path)
    _seed_db(db)
    out = plot_pred_vs_dock_accuracy(db, workdir)
    db.close()
    assert out == workdir.plots_dir() / "pred_vs_dock_accuracy.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_pred_vs_dock_accuracy_custom_iterations(tmp_path: Path) -> None:
    workdir, db = _make_db(tmp_path)
    _seed_db(db)
    out = plot_pred_vs_dock_accuracy(db, workdir, dock_iterations=(1,))
    db.close()
    assert out.exists()


def test_plot_pred_vs_dock_accuracy_requires_pairs(tmp_path: Path) -> None:
    workdir, db = _make_db(tmp_path)
    db.create_schema()
    db.insert_seed_docked("h1", "CCO", "ethanol-1", -8.5)
    db.commit()
    try:
        with pytest.raises(ValueError, match="no pred_score/dock_score pairs"):
            plot_pred_vs_dock_accuracy(db, workdir, dock_iterations=(1, 2))
    finally:
        db.close()
