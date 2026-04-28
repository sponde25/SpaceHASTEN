"""Regression lock for the acquisition SQL strings (§A.6).

Any change to these strings must be a deliberate, reviewed event: adjust this
file *and* docs/CODEBASE_REFERENCE.md §A.6 in the same commit.
"""

from __future__ import annotations

from spacehasten.core.db import Database

EXPECTED_DOCK_GREEDY = (
    "SELECT smiles, spacehastenid FROM data\n"
    " WHERE dock_score IS NULL\n"
    " ORDER BY pred_score LIMIT ?"
)
EXPECTED_DOCK_CLUSTERING = (
    "SELECT smiles, data.spacehastenid FROM data, clusters\n"
    " WHERE data.spacehastenid = clusters.spacehastenid\n"
    "   AND dock_score IS NULL\n"
    " GROUP BY clusterid ORDER BY MIN(pred_score) LIMIT ?"
)
EXPECTED_SIM_DOCKED_GREEDY = (
    "SELECT smiles, spacehastenid FROM data\n"
    " WHERE query IS NULL AND dock_score IS NOT NULL\n"
    " ORDER BY dock_score LIMIT ?"
)
EXPECTED_SIM_DOCKED_CLUSTERING = (
    "SELECT smiles, data.spacehastenid FROM data, clusters\n"
    " WHERE data.spacehastenid = clusters.spacehastenid\n"
    "   AND query IS NULL AND dock_score IS NOT NULL\n"
    " GROUP BY clusterid ORDER BY MIN(dock_score) LIMIT ?"
)
EXPECTED_SIM_PREDICTED_GREEDY = (
    "SELECT smiles, spacehastenid FROM data\n"
    " WHERE query IS NULL AND pred_score IS NOT NULL AND dock_score IS NULL\n"
    " ORDER BY pred_score LIMIT ?"
)
EXPECTED_SIM_PREDICTED_CLUSTERING = (
    "SELECT smiles, data.spacehastenid FROM data, clusters\n"
    " WHERE data.spacehastenid = clusters.spacehastenid\n"
    "   AND query IS NULL AND pred_score IS NOT NULL AND dock_score IS NULL\n"
    " GROUP BY clusterid ORDER BY MIN(pred_score) LIMIT ?"
)


def test_dock_greedy_sql_locked() -> None:
    assert Database._SQL_DOCK_GREEDY == EXPECTED_DOCK_GREEDY


def test_dock_clustering_sql_locked() -> None:
    assert Database._SQL_DOCK_CLUSTERING == EXPECTED_DOCK_CLUSTERING


def test_sim_docked_greedy_sql_locked() -> None:
    assert Database._SQL_SIMSEARCH_DOCKED_GREEDY == EXPECTED_SIM_DOCKED_GREEDY


def test_sim_docked_clustering_sql_locked() -> None:
    assert Database._SQL_SIMSEARCH_DOCKED_CLUSTERING == EXPECTED_SIM_DOCKED_CLUSTERING


def test_sim_predicted_greedy_sql_locked() -> None:
    assert Database._SQL_SIMSEARCH_PREDICTED_GREEDY == EXPECTED_SIM_PREDICTED_GREEDY


def test_sim_predicted_clustering_sql_locked() -> None:
    assert (
        Database._SQL_SIMSEARCH_PREDICTED_CLUSTERING == EXPECTED_SIM_PREDICTED_CLUSTERING
    )


def test_update_sql_uses_placeholders() -> None:
    # Defensive: no f-strings in update SQL.
    for sql in (
        Database._SQL_UPDATE_DOCK_SCORE,
        Database._SQL_UPDATE_PRED_SCORE,
        Database._SQL_MARK_AS_QUERY,
    ):
        assert "?" in sql
        assert "+" not in sql
