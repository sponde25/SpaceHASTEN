"""Local-scheduler integration test for initial seed-atlas orchestration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("FPSim2")
pytest.importorskip("rdkit")

from spacehasten.config.settings import GeneralSettings, Settings
from spacehasten.core.db import Database
from spacehasten.scheduler import LocalScheduler
from spacehasten.stages.atlas import DEFAULT_ATLAS_ID, build_initial_seed_atlas
from spacehasten.workspace.layout import WorkDir


def _seed_database(db: Database) -> int:
    db.create_schema()
    smiles = [
        "CCO",
        "CCCO",
        "CCCCO",
        "CCN",
        "CCCN",
        "CCCCN",
        "c1ccccc1",
        "Cc1ccccc1",
        "CCc1ccccc1",
        "c1ccncc1",
        "Cc1ccncc1",
        "C1CCCCC1",
        "CC1CCCCC1",
        "C1CCNCC1",
        "CC1CCNCC1",
        "CC(=O)O",
        "CCC(=O)O",
        "CCOC(=O)C",
        "CCS",
        "CCCS",
    ]
    for index, smile in enumerate(smiles, start=1):
        db.insert_seed_docked(f"hash-{index}", smile, f"seed-{index}", -7.0)
    db.commit()
    return len(smiles)


def test_initial_seed_atlas_stage_is_resumable(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "local", shared_root=tmp_path / "shared")
    db = Database(workdir.dbsh())
    seed_count = _seed_database(db)
    settings = Settings(
        general=GeneralSettings(
            cpu_count_clustering="1",
            atlas_partition_count=4,
            atlas_intermediate_reducers=2,
            atlas_assignment_shards=2,
            atlas_similarity_threshold=0.4,
        )
    )
    scheduler = LocalScheduler()
    command_prefix = (
        sys.executable,
        str(Path("src/spacehasten/remote/atlas.py").resolve()),
    )

    version = build_initial_seed_atlas(
        db,
        workdir,
        scheduler,
        settings,
        command_prefix=command_prefix,
    )
    assert version.version == 0
    assert version.compound_count == seed_count
    assert 1 <= version.centroid_count <= seed_count
    assert db.connection.execute("SELECT COUNT(*) FROM clusters").fetchone()[0] == seed_count
    first_job_count = len(scheduler._jobs)  # noqa: SLF001
    assert first_job_count == 6

    resumed = build_initial_seed_atlas(
        db,
        workdir,
        scheduler,
        settings,
        command_prefix=command_prefix,
    )
    assert resumed == version
    assert len(scheduler._jobs) == first_job_count  # noqa: SLF001
    assert (workdir.atlas_dir() / "final" / "assignments.npz").is_file()
    assert db.latest_cluster_atlas_version(DEFAULT_ATLAS_ID) == version
    db.close()

    second_workdir = WorkDir.bootstrap(
        tmp_path / "local-second", shared_root=tmp_path / "shared-second"
    )
    second_db = Database(second_workdir.dbsh())
    _seed_database(second_db)
    second_scheduler = LocalScheduler()
    reused = build_initial_seed_atlas(
        second_db,
        second_workdir,
        second_scheduler,
        settings,
        atlas_root=workdir.atlas_dir(),
        command_prefix=command_prefix,
    )
    assert reused.compound_count == seed_count
    assert len(second_scheduler._jobs) == 0  # noqa: SLF001
    assert second_db.connection.execute("SELECT COUNT(*) FROM clusters").fetchone()[0] == seed_count
    second_db.close()
