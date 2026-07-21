"""Persistent cluster-atlas schema and database operations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from spacehasten.core.db import (
    ClusterAtlasAssignmentRow,
    ClusterAtlasCentroidRow,
    ClusterAtlasRow,
    ClusterAtlasVersionRow,
    Database,
)


def test_cluster_atlas_roundtrip_and_legacy_materialization(tmp_path: Path) -> None:
    path = tmp_path / "atlas.dbsh"
    with Database(path) as db:
        db.create_schema()
        db.upsert_cluster_atlas(
            ClusterAtlasRow(
                atlas_id="morgan-r2-t040",
                similarity_threshold=0.4,
                fingerprint_type="Morgan",
                fingerprint_parameters='{"fpSize":1024,"radius":2}',
                partition_count=64,
            )
        )
        db.append_cluster_atlas_centroids(
            [
                ClusterAtlasCentroidRow("morgan-r2-t040", 1, 1, 0),
                ClusterAtlasCentroidRow("morgan-r2-t040", 3, 3, 0),
            ]
        )
        db.append_cluster_atlas_assignments(
            [
                ClusterAtlasAssignmentRow("morgan-r2-t040", 1, 1, 1.0, 0),
                ClusterAtlasAssignmentRow("morgan-r2-t040", 2, 1, 0.8, 0),
                ClusterAtlasAssignmentRow("morgan-r2-t040", 3, 3, 1.0, 0),
            ]
        )
        version = ClusterAtlasVersionRow(
            atlas_id="morgan-r2-t040",
            version=0,
            last_spacehastenid=3,
            compound_count=3,
            centroid_count=2,
            metadata_path="atlas/version_000/metadata.json",
        )
        db.record_cluster_atlas_version(version)
        assert db.latest_cluster_atlas_version("morgan-r2-t040") == version
        assert db.materialize_cluster_atlas("morgan-r2-t040") == 3
        assert db.connection.execute(
            "SELECT spacehastenid, clusterid FROM clusters ORDER BY spacehastenid"
        ).fetchall() == [(1, 1), (2, 1), (3, 3)]

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {
        "cluster_atlases",
        "cluster_atlas_versions",
        "cluster_atlas_centroids",
        "cluster_atlas_assignments",
    }.issubset(tables)


def test_cluster_atlas_configuration_is_immutable(tmp_path: Path) -> None:
    with Database(tmp_path / "atlas.dbsh") as db:
        db.create_schema()
        db.upsert_cluster_atlas(ClusterAtlasRow("atlas", 0.4, "Morgan", "{}", 64))
        with pytest.raises(ValueError, match="different configuration"):
            db.upsert_cluster_atlas(ClusterAtlasRow("atlas", 0.3, "Morgan", "{}", 64))
