"""Integration test for the library-screen stage with the local scheduler.

Uses a stub ``remote/library_infer.py`` (tiny python script) that applies
a real mw-bound property filter (reading the same 12-line control.param
the stage writes) and a deterministic ``pred_score = -mw`` in place of
real chemprop inference (the real chemprop path is unit-tested in
tests/unit/test_library_infer.py and exercised by the Session 15 verify
smoke test).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from spacehasten.config.properties import PropertyRanges
from spacehasten.config.settings import Settings
from spacehasten.core.db import Database
from spacehasten.scheduler import LocalScheduler
from spacehasten.stages.library_build import REQUIRED_COLUMNS, LibraryManifest
from spacehasten.stages.library_screen import library_screen
from spacehasten.workspace.layout import WorkDir

_STUB_SCRIPT = '''\
import sys
import pathlib
import pandas as pd

chunk_path, model_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
args = sys.argv[4:]


def _get(flag):
    return args[args.index(flag) + 1] if flag in args else None


params_path = _get("--params")
score_cutoff = _get("--score-cutoff")
top_n_per_chunk = _get("--top-n-per-chunk")

model_bin = pathlib.Path(model_dir) / "model_0" / "pytorch_model.bin"
if not model_bin.exists():
    print(f"model bin missing under {model_dir}", file=sys.stderr)
    sys.exit(2)

bounds = [float(x) for x in pathlib.Path(params_path).read_text().split()]
mw_min, mw_max = bounds[0], bounds[1]

df = pd.read_parquet(chunk_path)
df = df[df["mw"].between(mw_min, mw_max)].copy()
df["pred_score"] = -df["mw"]

if score_cutoff is not None:
    df = df[df["pred_score"] <= float(score_cutoff)]
elif top_n_per_chunk is not None:
    df = df.nsmallest(int(top_n_per_chunk), "pred_score")

pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
df[["reghash", "smiles", "compound_id", "pred_score"]].to_parquet(out_path, index=False)
'''

_PERMISSIVE_PROPS = PropertyRanges.model_validate({
    "mw": {"min": 0.0, "max": 500.0},
    "slogp": {"min": -10.0, "max": 10.0},
    "hba": {"min": 0, "max": 20},
    "hbd": {"min": 0, "max": 20},
    "rotbonds": {"min": 0, "max": 30},
    "tpsa": {"min": 0.0, "max": 500.0},
})


def _make_stub(tmp_path: Path) -> tuple[str, str]:
    stub = tmp_path / "stub_library_infer.py"
    stub.write_text(_STUB_SCRIPT)
    return (sys.executable, str(stub))


def _row(compound_id: str, smiles: str, reghash: str, mw: float) -> dict:
    return {
        "compound_id": compound_id,
        "smiles": smiles,
        "reghash": reghash,
        "mw": mw,
        "slogp": 1.0,
        "hba": 1,
        "hbd": 1,
        "rotbonds": 0,
        "tpsa": 20.0,
    }


def _make_store(store_dir: Path) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    chunk0 = pd.DataFrame([
        _row("ENA-1", "CCO", "h1", 46.07),
        _row("ENA-2", "c1ccccc1", "h2", 78.11),
        _row("ENA-3", "CCCl", "h3", 600.0),   # dropped by mw property bound
    ])
    chunk1 = pd.DataFrame([
        _row("ENA-4", "CCN", "h4", 45.08),
        _row("ENA-5", "CCC", "h5", 44.1),
    ])
    chunk0.to_parquet(store_dir / "chunk_00000.parquet", index=False)
    chunk1.to_parquet(store_dir / "chunk_00001.parquet", index=False)
    manifest = LibraryManifest(
        format_version=1,
        source_files=["dummy.cxsmiles"],
        n_compounds=5,
        n_chunks=2,
        chunk_glob="chunk_*.parquet",
        chunk_rows=[3, 2],
        columns=list(REQUIRED_COLUMNS),
        optional_columns=[],
    )
    manifest.save(store_dir / "manifest.json")


def _materialise_fake_model(workdir: WorkDir, version: int) -> None:
    model_dir = workdir.model_dir(version)
    (model_dir / "model_0").mkdir(parents=True, exist_ok=True)
    (model_dir / "model_0" / "pytorch_model.bin").write_bytes(b"fake-checkpoint")


def _bootstrap(tmp_path: Path, name: str) -> tuple[WorkDir, Database, Path]:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name=name)
    db = Database(workdir.dbsh())
    db.create_schema()
    _materialise_fake_model(workdir, version=1)
    db.store_model_blob(1, b"")
    db.commit()
    store = tmp_path / "store"
    _make_store(store)
    return workdir, db, store


# Ordering by pred_score (=-mw) ascending, property-filter survivors only
# (ENA-3 dropped, mw=600 > 500 bound):
#   ENA-2 (-78.11) < ENA-1 (-46.07) < ENA-4 (-45.08) < ENA-5 (-44.1)


def test_library_screen_score_cutoff_selects_and_inserts(tmp_path: Path) -> None:
    workdir, db, store = _bootstrap(tmp_path, "cutoff")
    scheduler = LocalScheduler()
    settings = Settings()

    n_inserted = library_screen(
        db, workdir, scheduler, settings,
        library_dir=store, model_version=1, props=_PERMISSIVE_PROPS,
        score_cutoff=-46.0,
        infer_command_prefix=_make_stub(tmp_path),
    )
    db.close()

    # score_cutoff=-46.0 keeps pred_score <= -46.0: ENA-2(-78.11), ENA-1(-46.07).
    assert n_inserted == 2

    db2 = Database(workdir.dbsh())
    rows = db2.connection.execute(
        "SELECT smilesid, pred_score, pred_version, simsearch_cycle, dock_score"
        " FROM data ORDER BY smilesid"
    ).fetchall()
    db2.close()
    by_id = {r[0]: r for r in rows}
    assert set(by_id) == {"ENA-1", "ENA-2"}
    for _sid, _pred_score, pred_version, simsearch_cycle, dock_score in rows:
        assert pred_version == 1
        assert simsearch_cycle is None
        assert dock_score is None
    assert by_id["ENA-2"][1] == pytest.approx(-78.11)


def test_library_screen_top_n_global_selection(tmp_path: Path) -> None:
    workdir, db, store = _bootstrap(tmp_path, "topn")
    scheduler = LocalScheduler()
    settings = Settings()

    n_inserted = library_screen(
        db, workdir, scheduler, settings,
        library_dir=store, model_version=1, props=_PERMISSIVE_PROPS,
        top_n=3,
        infer_command_prefix=_make_stub(tmp_path),
    )
    db.close()

    # Global top-3 across both chunks: ENA-2, ENA-1, ENA-4.
    assert n_inserted == 3
    db2 = Database(workdir.dbsh())
    ids = {
        r[0] for r in db2.connection.execute(
            "SELECT smilesid FROM data WHERE smilesid LIKE 'ENA-%'"
        ).fetchall()
    }
    db2.close()
    assert ids == {"ENA-1", "ENA-2", "ENA-4"}


def test_library_screen_dedup_against_existing_db_rows(tmp_path: Path) -> None:
    workdir, db, store = _bootstrap(tmp_path, "dedup")
    # ENA-1's reghash is already present (e.g. from a prior simsearch cycle).
    db.insert_simsearch_hit("h1", "CCO", "already-here", None, None, -1.0, 1)
    db.commit()

    scheduler = LocalScheduler()
    settings = Settings()
    n_inserted = library_screen(
        db, workdir, scheduler, settings,
        library_dir=store, model_version=1, props=_PERMISSIVE_PROPS,
        score_cutoff=-46.0,  # would otherwise select ENA-2 and ENA-1 (h1)
        infer_command_prefix=_make_stub(tmp_path),
    )
    db.close()

    # h1 (ENA-1) already exists -> deduped, only ENA-2 newly inserted.
    assert n_inserted == 1
    db2 = Database(workdir.dbsh())
    n_h1_rows = db2.connection.execute(
        "SELECT COUNT(*) FROM data WHERE reghash = 'h1'"
    ).fetchone()[0]
    db2.close()
    assert n_h1_rows == 1  # not duplicated


def test_library_screen_dry_run_inserts_nothing(tmp_path: Path) -> None:
    workdir, db, store = _bootstrap(tmp_path, "dryrun")
    scheduler = LocalScheduler()
    settings = Settings()
    report_path = tmp_path / "report.json"

    n_would_insert = library_screen(
        db, workdir, scheduler, settings,
        library_dir=store, model_version=1, props=_PERMISSIVE_PROPS,
        score_cutoff=-46.0,
        dry_run=True,
        report_path=report_path,
        infer_command_prefix=_make_stub(tmp_path),
    )

    n_rows = db.connection.execute(
        "SELECT COUNT(*) FROM data WHERE smilesid LIKE 'ENA-%'"
    ).fetchone()[0]
    db.close()

    assert n_would_insert == 2
    assert n_rows == 0  # nothing actually inserted

    report = json.loads(report_path.read_text())
    assert report["dry_run"] is True
    assert report["n_inserted"] == 2

    csv_path = report_path.with_name("report.survivors.csv")
    assert csv_path.exists()
    survivors = pd.read_csv(csv_path)
    assert set(survivors["compound_id"]) == {"ENA-1", "ENA-2"}


def test_library_screen_no_seed_docking_and_no_selector_raises(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="noselector")
    db = Database(workdir.dbsh())
    db.create_schema()
    _materialise_fake_model(workdir, version=1)
    db.store_model_blob(1, b"")
    db.commit()
    store = tmp_path / "store"
    _make_store(store)

    scheduler = LocalScheduler()
    settings = Settings()
    with pytest.raises(RuntimeError, match="no seed docking"):
        library_screen(
            db, workdir, scheduler, settings,
            library_dir=store, model_version=1, props=_PERMISSIVE_PROPS,
            infer_command_prefix=_make_stub(tmp_path),
        )
    db.close()


def test_library_screen_top_n_and_score_cutoff_mutually_exclusive(tmp_path: Path) -> None:
    workdir, db, store = _bootstrap(tmp_path, "mutex")
    scheduler = LocalScheduler()
    settings = Settings()
    with pytest.raises(ValueError, match="mutually exclusive"):
        library_screen(
            db, workdir, scheduler, settings,
            library_dir=store, model_version=1, props=_PERMISSIVE_PROPS,
            top_n=1, score_cutoff=-10.0,
            infer_command_prefix=_make_stub(tmp_path),
        )
    db.close()
