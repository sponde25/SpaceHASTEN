"""Integration test for the docking stage with the local scheduler.

Uses a stub bash body that mimics the Schrödinger pipeline by emitting a
synthetic ``glide_chunk_<i>.csv`` for the chunk's SMILES file and
tarring the result back to the dock directory.
"""

from __future__ import annotations

import csv
import sqlite3
import tarfile
from pathlib import Path

import pytest

from spacehasten.config.settings import Settings
from spacehasten.core.db import ClusterRow, Database
from spacehasten.scheduler import LocalScheduler
from spacehasten.stages.docking import _build_dock_command_body, dock
from spacehasten.workspace.layout import WorkDir

# A bash body that replaces the Schrödinger pipeline. It reads
# ``chunk_${TASK_ID}.smi`` (each line: ``<smiles> <spacehastenid>``),
# emits a Glide CSV with columns ``title,r_i_docking_score`` (one row
# per spacehastenid plus a duplicate row to exercise the min-reducer),
# and tars it back to the dock directory under the legacy filename
# ``results-chunk_<i>.tar.gz``.
_STUB_BODY = r"""
set -eu
chunk="chunk_${TASK_ID}"
smi="inputs/${chunk}.smi"
csv="glide_${chunk}.csv"
echo "title,r_i_docking_score" > "$csv"
i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  sid="${line##* }"
  # Two rows per compound: -7.0 and -8.0 → min reducer must pick -8.0.
  echo "${sid},-7.0" >> "$csv"
  echo "${sid},-8.0" >> "$csv"
  i=$((i+1))
done < "$smi"
mkdir -p results
tar -czf "results/results-${chunk}.tar.gz" "$csv"
"""


def _seed_db(db: Database, n_predicted: int = 12) -> None:
    db.create_schema()
    # A docked seed that must NOT be re-docked.
    db.insert_seed_docked("hd0", "CCO", "ethanol-1", -8.0)
    # n undocked rows with predictions, eligible for greedy docking.
    for i in range(n_predicted):
        sid = db.insert_seed_undocked(f"hu{i}", f"CC{'C' * (i + 1)}", f"ud-{i}")
        # pred_score gradient so greedy ORDER BY pred_score is well-defined.
        db.connection.execute(
            "UPDATE data SET pred_score = ?, pred_version = 1 WHERE spacehastenid = ?",
            (-9.0 + 0.1 * i, sid),
        )
    # Required blobs for write_glide_in / extracted grid.
    glide_template = (
        b"GRIDFILE     /old.zip\nLIGANDFILE   /old.maegz\nFORCEFIELD   OPLS_2005\nPRECISION    SP\n"
    )
    db.store_dock_param(glide_template)
    db.store_dock_grid(b"PK\x05\x06" + b"\x00" * 18)  # minimum legal empty zip
    db.commit()


def _seed_uncertainty_db(db: Database) -> list[int]:
    db.create_schema()
    db.insert_seed_docked("hd0", "CCO", "docked-seed", -8.0)
    candidates = [
        ("hu0", "CCCC", "candidate-0", -9.0, 2.0, 10),
        ("hu1", "CCCCC", "candidate-1", -10.0, 0.0, 10),
        ("hu2", "CCCCCC", "candidate-2", -9.8, 0.0, 20),
    ]
    identifiers: list[int] = []
    predictions: list[tuple[int, int, float, float, float, float]] = []
    clusters: list[ClusterRow] = []
    for reghash, smiles, smilesid, mean, epistemic, clusterid in candidates:
        sid = db.insert_seed_undocked(reghash, smiles, smilesid)
        identifiers.append(sid)
        predictions.append((sid, 1, mean, epistemic, 0.1, epistemic + 0.1))
        clusters.append(ClusterRow(spacehastenid=sid, clusterid=clusterid))
    db.apply_predictions(predictions)
    db.replace_clusters(clusters)
    db.store_dock_param(b"GRIDFILE /old.zip\nLIGANDFILE /old.maegz\nPRECISION SP\n")
    db.store_dock_grid(b"PK\x05\x06" + b"\x00" * 18)
    db.commit()
    return identifiers


def test_dock_command_uses_run_specific_scratch_and_fails_fast(
    tmp_path: Path,
) -> None:
    settings = Settings()
    settings.paths.scratch_default = "/wrk"
    dock_dir = tmp_path / "shared" / "run-a" / "docking" / "iter2"

    command = _build_dock_command_body(settings, dock_dir)

    assert command.startswith("set -euo pipefail\n")
    assert "/dock_run-a_iter2_chunk_${TASK_ID}" in command


def test_dock_stage_local_stub(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="dockws")
    db = Database(workdir.dbsh())
    _seed_db(db, n_predicted=12)

    settings = Settings()  # defaults
    settings.paths.scratch_default = str(tmp_path / "scratch")
    scheduler = LocalScheduler()

    iteration = dock(
        db,
        workdir,
        scheduler,
        settings,
        top_n=10,
        strategy="greedy",
        cpus=4,
        dock_command_template=_STUB_BODY,
        seed=0,
    )
    db.close()

    assert iteration == 1

    # Per-chunk artefacts exist on disk.
    dock_dir = workdir.docking_dir(1)
    assert (dock_dir / "glide_grid.zip").exists()
    inputs_dir = dock_dir / "inputs"
    smi_files = sorted(inputs_dir.glob("chunk_*.smi"))
    inp_files = sorted(inputs_dir.glob("chunk_*.inp"))
    glide_in_files = sorted(inputs_dir.glob("glide_chunk_*.in"))
    tarballs = sorted((dock_dir / "results").glob("results-chunk_*.tar.gz"))
    assert smi_files, "no chunk SMI files produced"
    assert len(smi_files) == len(inp_files) == len(glide_in_files) == len(tarballs)
    # cpus=4, top_n=10 → chunk_size = round(10/4) = 2 (< 1000), 5 chunks.
    assert len(smi_files) == 5

    # The Glide .in files reference the per-chunk LigPrep output and the
    # local grid zip — not the originals from the template blob.
    sample = (inputs_dir / "glide_chunk_1.in").read_text()
    assert sample.startswith("LIGANDFILE   chunk_1_1.maegz\n")
    assert "GRIDFILE     glide_grid.zip\n" in sample
    assert "/old.zip" not in sample and "/old.maegz" not in sample

    # All 10 acquired compounds are docked with score = -8.0 and iter=1.
    db2 = Database(workdir.dbsh())
    docked = db2.connection.execute(
        "SELECT smilesid, dock_score, dock_iteration FROM data"
        " WHERE smilesid LIKE 'ud-%' AND dock_score IS NOT NULL"
        " ORDER BY smilesid"
    ).fetchall()
    assert len(docked) == 10
    for _sid, score, it in docked:
        assert score == -8.0
        assert it == 1

    # The pre-existing seed is untouched.
    seed = db2.connection.execute(
        "SELECT dock_score, dock_iteration FROM data WHERE smilesid = 'ethanol-1'"
    ).fetchone()
    assert seed == (-8.0, 0)
    db2.close()


def test_dock_stage_clustering_strategy(tmp_path: Path) -> None:
    """Clustering acquisition must be honoured by ``select_compounds_to_dock``."""
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="dockclu")
    db = Database(workdir.dbsh())
    _seed_db(db, n_predicted=6)
    # Assign all undocked rows to two clusters so GROUP BY clusterid
    # collapses 6 candidates into 2.
    sids = [
        r[0]
        for r in db.connection.execute(
            "SELECT spacehastenid FROM data WHERE pred_score IS NOT NULL"
        ).fetchall()
    ]
    db.replace_clusters([ClusterRow(spacehastenid=sid, clusterid=(sid % 2)) for sid in sids])
    db.commit()

    settings = Settings()
    settings.paths.scratch_default = str(tmp_path / "scratch")
    scheduler = LocalScheduler()
    iteration = dock(
        db,
        workdir,
        scheduler,
        settings,
        top_n=10,
        strategy="clustering",
        cpus=2,
        dock_command_template=_STUB_BODY,
        seed=0,
    )
    db.close()
    assert iteration == 1

    db2 = Database(workdir.dbsh())
    n_docked = db2.connection.execute(
        "SELECT COUNT(*) FROM data WHERE smilesid LIKE 'ud-%' AND dock_score IS NOT NULL"
    ).fetchone()[0]
    db2.close()
    # GROUP BY clusterid LIMIT 10 over 2 clusters → exactly 2 representatives.
    assert n_docked == 2


@pytest.mark.parametrize(
    ("strategy", "method_args"),
    [
        ("lcb", {"lcb_beta": 1.0}),
        ("ei", {"ei_hit_threshold": -9.7}),
    ],
)
def test_uncertainty_acquisition_with_dynamic_cluster_penalty(
    tmp_path: Path,
    strategy: str,
    method_args: dict[str, float],
) -> None:
    workdir = WorkDir.bootstrap(tmp_path / strategy, name=f"dock-{strategy}")
    db = Database(workdir.dbsh())
    first, _same_cluster, other_cluster = _seed_uncertainty_db(db)

    settings = Settings()
    settings.paths.scratch_default = str(tmp_path / "scratch")
    scheduler = LocalScheduler()
    iteration = dock(
        db,
        workdir,
        scheduler,
        settings,
        top_n=2,
        strategy=strategy,  # type: ignore[arg-type]
        cpus=1,
        cluster_lambda=1.0,
        dock_command_template=_STUB_BODY,
        seed=0,
        **method_args,
    )
    db.close()

    assert iteration == 1
    with Database(workdir.dbsh()) as result_db:
        selected_ids = [
            int(row[0])
            for row in result_db.connection.execute(
                "SELECT spacehastenid FROM data WHERE dock_iteration = 1 ORDER BY spacehastenid"
            ).fetchall()
        ]
    assert selected_ids == sorted([first, other_cluster])

    with (workdir.docking_dir(1) / "acquisition.csv").open(newline="") as handle:
        acquisition_rows = list(csv.DictReader(handle))
    assert [int(row["spacehastenid"]) for row in acquisition_rows] == [
        first,
        other_cluster,
    ]
    assert {row["method"] for row in acquisition_rows} == {strategy}
    assert {float(row["cluster_lambda"]) for row in acquisition_rows} == {1.0}
    assert {float(row["cluster_similarity_threshold"]) for row in acquisition_rows} == {0.4}


def test_dock_stage_no_candidates_raises(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="empty")
    db = Database(workdir.dbsh())
    db.create_schema()
    # All rows are docked → greedy query returns nothing.
    db.insert_seed_docked("h", "CCO", "ethanol-1", -8.0)
    db.store_dock_param(b"FORCEFIELD OPLS_2005\n")
    db.store_dock_grid(b"PK\x05\x06" + b"\x00" * 18)
    db.commit()

    settings = Settings()
    settings.paths.scratch_default = str(tmp_path / "scratch")
    scheduler = LocalScheduler()
    try:
        dock(
            db,
            workdir,
            scheduler,
            settings,
            top_n=5,
            strategy="greedy",
            cpus=1,
            dock_command_template="exit 0",
        )
    except ValueError as exc:
        assert "no compounds" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
    finally:
        db.close()


def test_dock_stage_clustering_without_clusters_raises(tmp_path: Path) -> None:
    """``strategy='clustering'`` must fail fast with a clear message when
    ``spacehasten cluster`` has never been run (``clusters`` is empty)."""
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="noclu")
    db = Database(workdir.dbsh())
    _seed_db(db, n_predicted=6)

    settings = Settings()
    settings.paths.scratch_default = str(tmp_path / "scratch")
    scheduler = LocalScheduler()
    try:
        dock(
            db,
            workdir,
            scheduler,
            settings,
            top_n=5,
            strategy="clustering",
            cpus=1,
            dock_command_template="exit 0",
        )
    except ValueError as exc:
        assert "cluster" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
    finally:
        db.close()


def test_dock_stage_persists_a_valid_results_tar(tmp_path: Path) -> None:
    """Smoke check: the stub produces a tar that ``tarfile`` can re-open."""
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="tarcheck")
    db = Database(workdir.dbsh())
    _seed_db(db, n_predicted=2)

    settings = Settings()
    settings.paths.scratch_default = str(tmp_path / "scratch")
    scheduler = LocalScheduler()
    dock(
        db,
        workdir,
        scheduler,
        settings,
        top_n=2,
        strategy="greedy",
        cpus=1,
        dock_command_template=_STUB_BODY,
        seed=0,
    )
    db.close()

    tarballs = sorted((workdir.docking_dir(1) / "results").glob("results-chunk_*.tar.gz"))
    assert tarballs
    for tar_path in tarballs:
        with tarfile.open(tar_path) as tf:
            names = tf.getnames()
        assert any(n.startswith("glide_chunk_") and n.endswith(".csv") for n in names)

    # Schema stayed intact (no accidental DDL).
    with sqlite3.connect(workdir.dbsh()) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"data", "docking_param", "docking_grid", "models", "properties", "clusters"}.issubset(
        tables
    )
