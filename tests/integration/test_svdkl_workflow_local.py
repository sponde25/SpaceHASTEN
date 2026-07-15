"""Local workflow smoke for the SVDKL replacement with greedy acquisition."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd
import pytest

from spacehasten.config.settings import GeneralSettings, PathsSettings, Settings
from spacehasten.core.db import Database, PropertyRanges
from spacehasten.core.molecules import tautomer_hash
from spacehasten.scheduler import LocalScheduler
from spacehasten.stages.simsearch import simsearch
from spacehasten.stages.training import train
from spacehasten.workspace.layout import WorkDir

HAS_SVDKL_DEPS = all(
    importlib.util.find_spec(name) is not None
    for name in ("chemprop", "gpytorch", "pandas")
)

requires_svdkl_deps = pytest.mark.skipif(
    not HAS_SVDKL_DEPS,
    reason="chemprop, gpytorch, and pandas are required for local SVDKL workflow smoke",
)

EXAMPLE_CSV = Path(__file__).resolve().parents[2] / "example.csv"

_PERMISSIVE_PROPS = PropertyRanges(
    mw=("0", "10000"),
    slogp=("-10", "10"),
    hba=("0", "20"),
    hbd=("0", "20"),
    rotbonds=("0", "30"),
    tpsa=("0", "500"),
)

_SEARCH_STUB = dedent(
    r"""
    set -eu
    mkdir -p results
    cat > "results/spacelightresult_${TASK_ID}.csv" <<EOF
#result-smiles,result-name,fingerprint-similarity
CCN,svdkl-hit-1,0.91
CCC,svdkl-hit-2,0.80
CCOC,svdkl-hit-3,0.70
EOF
    cat > "results/ftreesresult_${TASK_ID}.csv" <<EOF
#result-smiles,result-name,pharmacophore-similarity
CCN,svdkl-hit-1,0.95
CCC,svdkl-hit-2,0.85
CCOC,svdkl-hit-3,0.75
EOF
"""
).lstrip()


@requires_svdkl_deps
def test_svdkl_replacement_supports_greedy_simsearch_acquisition(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="svdklws")
    db = Database(workdir.dbsh())
    db.create_schema()
    db.replace_properties(_PERMISSIVE_PROPS)
    _seed_training_rows(db, rows=32)
    db.commit()

    settings = Settings(
        paths=PathsSettings(spacehasten_src_dir=str(Path("src/spacehasten").resolve())),
        general=GeneralSettings(
            prepare_anaconda="source /wrk/setup_conda.sh",
            activate_chemprop="conda activate spacehasten-quick",
            train_batch_size=8,
            train_epochs=1,
            train_mp_hidden_size=16,
            train_mp_depth=2,
            train_ffn_hidden_size=16,
            train_ffn_layers=1,
            train_dropout=0.0,
            train_svdkl_grid_size=16,
            pred_batch_size=4,
        ),
    )
    scheduler = LocalScheduler()

    model_version = train(
        db,
        workdir,
        scheduler,
        settings,
        cutoff=10.0,
        train_command_prefix=(sys.executable, str(settings.remote_script_path("train"))),
    )
    assert model_version == 0

    cycle = simsearch(
        db,
        workdir,
        scheduler,
        settings,
        source="docked",
        strategy="greedy",
        top_n=1,
        cpu=1,
        search_command_template=_SEARCH_STUB,
        prop_filter_command_prefix=(
            sys.executable,
            str(settings.remote_script_path("prop_filter")),
        ),
        predict_command_prefix=(sys.executable, str(settings.remote_script_path("predict"))),
    )
    assert cycle == 1

    pred_files = sorted(
        (workdir.simsearch_dir(cycle) / "CONTROL" / "results_prediction").glob(
            "predicted_propoutput_control_*.csv"
        )
    )
    assert pred_files
    pred_df = pd.read_csv(pred_files[0])
    assert "docking_score_std" in pred_df.columns
    assert len(pred_df) >= 1

    selectable = db.select_queries_for_simsearch(
        source="predicted",
        strategy="greedy",
        limit=2,
    )
    assert selectable
    for _smiles, spacehastenid in selectable:
        pred_score = db.connection.execute(
            "SELECT pred_score FROM data WHERE spacehastenid = ?",
            (spacehastenid,),
        ).fetchone()[0]
        assert pred_score is not None
    db.close()


def _seed_training_rows(db: Database, rows: int) -> None:
    df = pd.read_csv(EXAMPLE_CSV).head(rows)
    for row in df.itertuples(index=False):
        reghash = tautomer_hash(row.smiles)
        if reghash is None:
            continue
        db.insert_seed_docked(reghash, row.smiles, row.smilesid, float(row.docking_score))
