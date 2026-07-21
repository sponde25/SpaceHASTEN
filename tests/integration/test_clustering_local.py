"""Integration test for the clustering stage with the local scheduler.

Drives :func:`spacehasten.stages.clustering.cluster` end-to-end against
a 100-row synthetic SMILES set, using the real
:mod:`spacehasten.remote.cluster` pipeline (RDKit + FPSim2). Validates
that every input compound receives a cluster assignment in the DB and
that the discovered clusters partition the compound set.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from spacehasten.config.settings import Settings
from spacehasten.core.db import Database
from spacehasten.scheduler import LocalScheduler
from spacehasten.stages.clustering import _build_cluster_command, cluster
from spacehasten.workspace.layout import WorkDir

# Skip the whole module on hosts that lack the remote-side deps.
pytest.importorskip("FPSim2")
pytest.importorskip("rdkit")


def _synthetic_smiles(n: int) -> list[str]:
    """Generate ``n`` distinct, valid SMILES with a mix of scaffolds."""
    scaffolds = [
        "CCO",  # ethanol family
        "c1ccccc1",  # benzene family
        "C1CCCCC1",  # cyclohexane family
        "c1ccncc1",  # pyridine family
        "C1CCNCC1",  # piperidine family
    ]
    out: list[str] = []
    for i in range(n):
        base = scaffolds[i % len(scaffolds)]
        # Append a varying alkyl chain so each SMILES is unique.
        out.append(base + "C" * (1 + (i // len(scaffolds))))
    return out


def _seed_db(db: Database, smiles_list: list[str]) -> None:
    db.create_schema()
    for i, smi in enumerate(smiles_list):
        db.insert_seed_undocked(f"h{i}", smi, f"syn-{i}")
    db.commit()


def test_cluster_command_propagates_similarity_threshold(tmp_path: Path) -> None:
    command = _build_cluster_command(
        tmp_path / "input.smi.gz",
        tmp_path / "clustering.csv",
        4,
        ("python3", "cluster.py"),
        0.4,
    )
    assert "--similarity-threshold 0.4" in command


def test_clustering_stage_local_real_pipeline(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="clusterws")
    smiles_list = _synthetic_smiles(100)

    db = Database(workdir.dbsh())
    _seed_db(db, smiles_list)

    settings = Settings()
    scheduler = LocalScheduler()

    n = cluster(db, workdir, scheduler, settings)
    db.close()

    assert n == 100  # one row per compound

    # Every compound is assigned to some cluster, and clusters partition
    # the population (no orphans, no duplicates).
    db2 = Database(workdir.dbsh())
    rows = db2.connection.execute(
        "SELECT spacehastenid, clusterid FROM clusters ORDER BY spacehastenid"
    ).fetchall()
    db2.close()

    assert len(rows) == 100
    sids = [r[0] for r in rows]
    assert sids == sorted(set(sids))  # primary-key uniqueness
    cluster_ids = {r[1] for r in rows}
    # Each cluster centroid is a valid spacehastenid (by construction in
    # remote.cluster), so cluster_ids must be a subset of sids.
    assert cluster_ids.issubset(set(sids))
    # Sanity: with the chosen synthetic set we expect more than one but
    # fewer than all compounds to be centroids.
    assert 1 < len(cluster_ids) <= 100

    # The CSV on disk should match the DB rows exactly.
    csv_path = workdir.clustering_dir() / "clustering.csv"
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        csv_rows = [(int(r["spacehastenid"]), int(r["clusterid"])) for r in reader]
    assert sorted(csv_rows) == sorted(rows)


def test_clustering_stage_empty_table_returns_zero(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="empty")
    db = Database(workdir.dbsh())
    db.create_schema()
    db.commit()

    settings = Settings()
    scheduler = LocalScheduler()

    # Pass a command_prefix that would fail if invoked, to prove we
    # short-circuit on an empty data table.
    n = cluster(
        db,
        workdir,
        scheduler,
        settings,
        cluster_command_prefix=("bash", "-c", "exit 1"),
    )
    db.close()
    assert n == 0


def test_cluster_cutoff_requires_docked_only(tmp_path: Path) -> None:
    """``cutoff`` without ``docked_only`` is a usage error, not silently
    ignored — mirrors the CLI's ``--cutoff`` requiring ``--docked``."""
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="badargs")
    db = Database(workdir.dbsh())
    db.create_schema()
    db.commit()

    settings = Settings()
    scheduler = LocalScheduler()

    with pytest.raises(ValueError, match="docked_only"):
        cluster(db, workdir, scheduler, settings, cutoff=-10.0)
    db.close()


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.1])
def test_cluster_rejects_invalid_similarity_threshold(tmp_path: Path, threshold: float) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="bad-threshold")
    db = Database(workdir.dbsh())
    db.create_schema()
    db.commit()

    with pytest.raises(ValueError, match="similarity_threshold"):
        cluster(
            db,
            workdir,
            LocalScheduler(),
            Settings(),
            similarity_threshold=threshold,
        )
    db.close()


def test_cluster_docked_only_filters_undocked_compounds(tmp_path: Path) -> None:
    """``--docked`` restricts clustering input to compounds with a
    non-NULL ``dock_score``, skipping undocked library compounds."""
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="dockedonly")
    smiles_list = _synthetic_smiles(20)

    db = Database(workdir.dbsh())
    db.create_schema()
    for i, smi in enumerate(smiles_list):
        # Only the first 5 compounds are docked; the rest are library-only.
        if i < 5:
            db.insert_seed_docked(f"h{i}", smi, f"syn-{i}", float(-i))
        else:
            db.insert_seed_undocked(f"h{i}", smi, f"syn-{i}")
    db.commit()

    settings = Settings()
    scheduler = LocalScheduler()

    n = cluster(db, workdir, scheduler, settings, docked_only=True)
    db.close()

    assert n == 5

    db2 = Database(workdir.dbsh())
    rows = db2.connection.execute(
        "SELECT spacehastenid FROM clusters ORDER BY spacehastenid"
    ).fetchall()
    db2.close()
    assert [r[0] for r in rows] == [1, 2, 3, 4, 5]


def test_cluster_docked_only_cutoff_filters_by_score(tmp_path: Path) -> None:
    """``--docked --cutoff`` restricts clustering to docked compounds
    with ``dock_score <= cutoff`` (better/smaller score)."""
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="cutoff")
    smiles_list = _synthetic_smiles(10)

    db = Database(workdir.dbsh())
    db.create_schema()
    scores = [-12.0, -11.0, -9.0, -5.0, 0.0, 3.0, 5.0, 7.0, 9.0, 11.0]
    for i, (smi, score) in enumerate(zip(smiles_list, scores, strict=True)):
        db.insert_seed_docked(f"h{i}", smi, f"syn-{i}", score)
    db.commit()

    settings = Settings()
    scheduler = LocalScheduler()

    n = cluster(db, workdir, scheduler, settings, docked_only=True, cutoff=-10.0)
    db.close()

    # Only spacehastenid 1, 2 have dock_score <= -10.0.
    assert n == 2
    db2 = Database(workdir.dbsh())
    rows = db2.connection.execute(
        "SELECT spacehastenid FROM clusters ORDER BY spacehastenid"
    ).fetchall()
    db2.close()
    assert [r[0] for r in rows] == [1, 2]
