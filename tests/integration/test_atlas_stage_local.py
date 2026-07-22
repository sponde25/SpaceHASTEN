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
from spacehasten.stages.atlas import (
    DEFAULT_ATLAS_ID,
    build_initial_seed_atlas,
    update_cluster_atlas,
)
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
    assert db.connection.execute("SELECT COUNT(*) FROM clusters").fetchone()[0] == 0
    first_job_count = len(scheduler._jobs)  # noqa: SLF001
    assert first_job_count == 7

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
    assert (workdir.atlas_dir() / "final" / "centroids" / "centroids_fp.h5").is_file()
    assert db.latest_cluster_atlas_version(DEFAULT_ATLAS_ID) == version

    original_assignments = db.connection.execute(
        "SELECT spacehastenid, clusterid FROM cluster_atlas_assignments "
        "WHERE atlas_id = ? ORDER BY spacehastenid",
        (DEFAULT_ATLAS_ID,),
    ).fetchall()
    for index, smile in enumerate(("N#N", "OP(=O)(O)O", "Cl[Si](Cl)(Cl)Cl"), start=1):
        db.insert_simsearch_hit(
            f"new-hash-{index}",
            smile,
            f"new-{index}",
            None,
            None,
            -7.0,
            1,
            pred_version=0,
        )
    db.commit()
    updated = update_cluster_atlas(
        db,
        workdir,
        scheduler,
        settings,
        through_spacehastenid=23,
        command_prefix=command_prefix,
    )
    assert updated.version == 1
    assert updated.last_spacehastenid == 23
    assert updated.compound_count == 23
    assert (
        db.connection.execute(
            "SELECT COUNT(*) FROM cluster_atlas_assignments WHERE atlas_id = ?",
            (DEFAULT_ATLAS_ID,),
        ).fetchone()[0]
        == 23
    )
    assert db.connection.execute("SELECT COUNT(*) FROM clusters").fetchone()[0] == 0
    assert (
        db.connection.execute(
            "SELECT spacehastenid, clusterid FROM cluster_atlas_assignments "
            "WHERE atlas_id = ? AND spacehastenid <= 20 ORDER BY spacehastenid",
            (DEFAULT_ATLAS_ID,),
        ).fetchall()
        == original_assignments
    )
    previous_centroid_count = updated.centroid_count
    for expected_version, smile in ((2, "CCO"), (3, "CCN")):
        identifier = db.insert_simsearch_hit(
            f"covered-hash-{expected_version}",
            smile,
            f"covered-{expected_version}",
            None,
            None,
            -7.0,
            expected_version,
            pred_version=0,
        )
        db.commit()
        updated = update_cluster_atlas(
            db,
            workdir,
            scheduler,
            settings,
            through_spacehastenid=identifier,
            command_prefix=command_prefix,
        )
        assert updated.version == expected_version
        assert updated.last_spacehastenid == identifier
        assert updated.centroid_count == previous_centroid_count
        assert Path(updated.metadata_path).name == "atlas_version.json"

    assert (
        db.connection.execute(
            "SELECT COUNT(*) FROM cluster_atlas_assignments WHERE atlas_id = ?",
            (DEFAULT_ATLAS_ID,),
        ).fetchone()[0]
        == 25
    )
    assert db.connection.execute("SELECT COUNT(*) FROM clusters").fetchone()[0] == 0
    update_job_count = len(scheduler._jobs)  # noqa: SLF001
    assert (
        update_cluster_atlas(
            db,
            workdir,
            scheduler,
            settings,
            through_spacehastenid=25,
            command_prefix=command_prefix,
        )
        == updated
    )
    assert len(scheduler._jobs) == update_job_count  # noqa: SLF001
    db.connection.execute("DELETE FROM data WHERE spacehastenid = 25")
    db.commit()
    with pytest.raises(ValueError, match="compound count"):
        update_cluster_atlas(
            db,
            workdir,
            scheduler,
            settings,
            through_spacehastenid=24,
            command_prefix=command_prefix,
        )
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
    assert second_db.connection.execute("SELECT COUNT(*) FROM clusters").fetchone()[0] == 0
    second_db.close()
