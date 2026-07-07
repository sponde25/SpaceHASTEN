"""Integration test for the simsearch stage with the local scheduler.

Uses fully-stubbed search and control bash bodies so the test does not
depend on real BiosolveIT binaries or chemprop:

* The search stub emits a fixed-shape CSV per task with two hits per
  query, including one duplicate hit shared across queries to exercise
  the SMILES dedup path.
* The control stub runs the *real* :mod:`spacehasten.remote.prop_filter`
  to produce ``propoutput_control_<i>.csv`` (so the reghash packing and
  RDKit property check are exercised), then writes a synthetic
  ``predicted_propoutput_control_<i>.csv`` with a constant
  ``docking_score`` per row.

The test asserts:

* Exactly the unique non-duplicate-of-existing reghashes are inserted
  (no duplicate-reghash invariant violation).
* Per-row ``spacelight``, ``ftrees``, ``pred_score``, and
  ``simsearch_cycle`` are populated correctly.
* The query rows have their ``query`` column set to the new cycle.
"""

from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from spacehasten.config.settings import Settings
from spacehasten.core.db import Database, PropertyRanges
from spacehasten.scheduler import LocalScheduler
from spacehasten.stages.simsearch import simsearch
from spacehasten.workspace.layout import WorkDir

_PERMISSIVE_PROPS = PropertyRanges(
    mw=("0", "10000"),
    slogp=("-10", "10"),
    hba=("0", "20"),
    hbd=("0", "20"),
    rotbonds=("0", "30"),
    tpsa=("0", "500"),
)


def _real_reghash(smiles: str) -> str:
    """Compute the same tautomer hash the prop filter writes."""
    from rdkit import Chem
    from rdkit.Chem import RegistrationHash

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return RegistrationHash.GetMolLayers(mol)[RegistrationHash.HashLayer.TAUTOMER_HASH]


def _seed_db(db: Database) -> list[int]:
    """Seed the DB with two docked rows + an existing simsearch hit.

    The two docked rows have ``dock_score`` set so they qualify for
    ``select_queries_for_simsearch(source="docked", ...)``. A third row
    pre-exists with the *real* reghash of ``CC(=O)O`` (which the search
    stub will emit), exercising the dedup-against-existing-reghash path.
    """
    db.create_schema()
    sid_a = db.insert_seed_docked("rh-A", "CCO", "ethanol-1", -8.5)
    sid_b = db.insert_seed_docked("rh-B", "c1ccccc1", "benzene-1", -7.5)
    db.insert_simsearch_hit(
        reghash=_real_reghash("CC(=O)O"),
        smiles="CC(=O)O",
        smilesid="acetic-pre-existing",
        spacelight=0.6,
        ftrees=0.7,
        pred_score=-6.5,
        simsearch_cycle=0,  # legacy 0 == undefined cycle
    )
    db.replace_properties(_PERMISSIVE_PROPS)
    db.commit()
    return [sid_a, sid_b]


def _materialise_fake_model(workdir: WorkDir, version: int) -> Path:
    model_dir = workdir.model_dir(version)
    (model_dir / "model_0").mkdir(parents=True, exist_ok=True)
    bin_path = model_dir / "model_0" / "pytorch_model.bin"
    bin_path.write_bytes(b"fake-checkpoint")
    return model_dir


# Search stub: emit one CSV per method per task. The first task emits
# three hits — one of which collides with the pre-existing rh-DUP — and
# the second task emits two hits (one shared SMILES with task 1 to
# exercise per-method dedup-by-SMILES with max similarity).
_SEARCH_STUB = dedent(r"""
    set -eu
    mkdir -p results
    case "${TASK_ID}" in
      1)
        cat > "results/spacelightresult_${TASK_ID}.csv" <<EOF
    #result-smiles,result-name,fingerprint-similarity
    CCN,enamine-1,0.91
    CCC,enamine-2,0.80
    CC(=O)O,enamine-dup,0.70
    EOF
        cat > "results/ftreesresult_${TASK_ID}.csv" <<EOF
    #result-smiles,result-name,pharmacophore-similarity
    CCN,enamine-1,0.95
    CCC,enamine-2,0.85
    CC(=O)O,enamine-dup,0.60
    EOF
        ;;
      2)
        cat > "results/spacelightresult_${TASK_ID}.csv" <<EOF
    #result-smiles,result-name,fingerprint-similarity
    CCN,enamine-1,0.55
    CCO,extra-1,0.50
    EOF
        cat > "results/ftreesresult_${TASK_ID}.csv" <<EOF
    #result-smiles,result-name,pharmacophore-similarity
    CCN,enamine-1,0.50
    CCO,extra-1,0.40
    EOF
        ;;
    esac
""").lstrip()


def _build_control_stub(prop_filter_argv: list[str]) -> str:
    """Render a control-task body that uses the real prop_filter and
    a fake predict step."""
    pf = " ".join(prop_filter_argv)
    return dedent(rf"""
        set -eu
        mkdir -p results_propfilter results_prediction
        # 1) Real prop_filter: produces results_propfilter/propoutput_control_${{TASK_ID}}.csv.
        {pf} inputs/control_${{TASK_ID}}.smi.gz inputs/control.param \
            --output results_propfilter/propoutput_control_${{TASK_ID}}.csv

        # 2) Fake predict: write a constant docking_score per row,
        #    keeping the smilesid column verbatim so the ingest step can
        #    recover (reghash, smiles, title).
        out="results_prediction/predicted_propoutput_control_${{TASK_ID}}.csv"
        echo "smilesid,docking_score" > "$out"
        # Skip CSV header.
        tail -n +2 "results_propfilter/propoutput_control_${{TASK_ID}}.csv" \
            | while IFS=, read -r smi sid; do
                printf '%s,%s\n' "$sid" "-7.0" >> "$out"
            done
    """).lstrip()


def test_simsearch_stage_local(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="simws")
    db = Database(workdir.dbsh())
    seed_sids = _seed_db(db)

    _materialise_fake_model(workdir, version=1)
    db.store_model_blob(1, b"")
    db.commit()

    settings = Settings()  # defaults; field names align with stub CSVs
    scheduler = LocalScheduler()

    prop_filter_argv = [sys.executable, "-m", "spacehasten.remote.prop_filter"]
    control_body = _build_control_stub(prop_filter_argv)

    new_cycle = simsearch(
        db,
        workdir,
        scheduler,
        settings,
        source="docked",
        strategy="greedy",
        top_n=2,
        cpu=2,
        search_command_template=_SEARCH_STUB,
        control_command_template=control_body,
    )
    db.close()

    assert new_cycle == 1

    # ---- Query bookkeeping --------------------------------------------- #
    db2 = Database(workdir.dbsh())
    queries = db2.connection.execute(
        "SELECT spacehastenid, query FROM data WHERE query IS NOT NULL"
        " ORDER BY spacehastenid"
    ).fetchall()
    assert {sid for sid, _ in queries} == set(seed_sids)
    assert all(q == new_cycle for _, q in queries)

    # ---- Inserted simsearch hits --------------------------------------- #
    rows = db2.connection.execute(
        "SELECT reghash, smiles, smilesid, spacelight, ftrees, pred_score,"
        " simsearch_cycle FROM data WHERE simsearch_cycle = ?",
        (new_cycle,),
    ).fetchall()
    db2.close()

    # We expect three unique-by-reghash rows ingested:
    # CCN (enamine-1), CCC (enamine-2), CCO (extra-1). The acetic-acid
    # SMILES collides with the pre-existing rh-DUP and must be skipped.
    smiles_set = {r[1] for r in rows}
    assert smiles_set == {"CCN", "CCC", "CCO"}

    # Reghash uniqueness invariant.
    reghashes = [r[0] for r in rows]
    assert len(set(reghashes)) == len(reghashes), "duplicate reghash inserted"

    # Per-row similarity & cycle bookkeeping.
    by_smiles = {r[1]: r for r in rows}
    # CCN appears in both tasks; per-method best similarity is taken.
    assert by_smiles["CCN"][3] == pytest.approx(0.91)  # spacelight (max)
    assert by_smiles["CCN"][4] == pytest.approx(0.95)  # ftrees   (max)
    assert by_smiles["CCC"][3] == pytest.approx(0.80)
    assert by_smiles["CCC"][4] == pytest.approx(0.85)
    assert by_smiles["CCO"][3] == pytest.approx(0.50)
    assert by_smiles["CCO"][4] == pytest.approx(0.40)
    for r in rows:
        assert r[5] == pytest.approx(-7.0)
        assert r[6] == new_cycle


def test_simsearch_writes_control_artefacts(tmp_path: Path) -> None:
    """Smoke check that on-disk artefacts match the documented layout."""
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="simart")
    db = Database(workdir.dbsh())
    _seed_db(db)
    _materialise_fake_model(workdir, version=1)
    db.store_model_blob(1, b"")
    db.commit()

    settings = Settings()
    scheduler = LocalScheduler()
    prop_filter_argv = [sys.executable, "-m", "spacehasten.remote.prop_filter"]
    simsearch(
        db,
        workdir,
        scheduler,
        settings,
        source="docked",
        strategy="greedy",
        top_n=2,
        cpu=2,
        search_command_template=_SEARCH_STUB,
        control_command_template=_build_control_stub(prop_filter_argv),
    )
    db.close()

    cycle_dir = workdir.simsearch_dir(1)
    queries = cycle_dir / f"queries_{workdir.name}.smi"
    assert queries.exists()
    assert len(queries.read_text().strip().splitlines()) == 2
    control_dir = cycle_dir / "CONTROL"
    assert (control_dir / "inputs" / "control.param").exists()
    chunks = sorted((control_dir / "inputs").glob("control_*.smi.gz"))
    assert chunks, "no control chunks written"
    # Each chunk is a real gzip we can read back.
    with gzip.open(chunks[0], "rt") as fh:
        first = fh.readline()
    assert "§" in first
    # The model is NOT copied; the control command uses the absolute path.
    assert not (control_dir / "v1").exists()
    # Predicted files exist for each chunk.
    preds = sorted((control_dir / "results_prediction").glob("predicted_propoutput_control_*.csv"))
    assert len(preds) == len(chunks)
    rows = list(csv.DictReader(preds[0].open()))
    assert rows and rows[0]["docking_score"] == "-7.0"


def test_simsearch_no_queries_raises(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="empty")
    db = Database(workdir.dbsh())
    db.create_schema()
    db.replace_properties(_PERMISSIVE_PROPS)
    db.commit()
    _materialise_fake_model(workdir, version=1)
    db.store_model_blob(1, b"")
    db.commit()

    settings = Settings()
    scheduler = LocalScheduler()
    with pytest.raises(ValueError, match="no candidate queries"):
        simsearch(
            db,
            workdir,
            scheduler,
            settings,
            source="docked",
            strategy="greedy",
            top_n=5,
            cpu=1,
            search_command_template="exit 0",
            control_command_template="exit 0",
        )
    db.close()


def test_simsearch_clustering_without_clusters_raises(tmp_path: Path) -> None:
    """``strategy='clustering'`` must fail fast with a clear message when
    ``spacehasten cluster`` has never been run (``clusters`` is empty)."""
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="noclu")
    db = Database(workdir.dbsh())
    _seed_db(db)
    db.commit()

    settings = Settings()
    scheduler = LocalScheduler()
    with pytest.raises(ValueError, match="(?i)cluster"):
        simsearch(
            db,
            workdir,
            scheduler,
            settings,
            source="docked",
            strategy="clustering",
            top_n=2,
            cpu=1,
            search_command_template="exit 0",
            control_command_template="exit 0",
        )
    db.close()


def test_simsearch_requires_trained_model(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="nomodel")
    db = Database(workdir.dbsh())
    _seed_db(db)
    db.commit()

    settings = Settings()
    scheduler = LocalScheduler()
    with pytest.raises(RuntimeError, match="no trained model"):
        simsearch(
            db,
            workdir,
            scheduler,
            settings,
            source="docked",
            strategy="greedy",
            top_n=2,
            cpu=2,
            search_command_template=_SEARCH_STUB,
            control_command_template="exit 0",
        )
    db.close()
