"""Unit tests for the pure helper functions in spacehasten.stages.library_screen.

The full scheduler orchestration is exercised end-to-end in
tests/integration/test_library_screen_local.py; these tests cover the
resumability, control.param writing, and ingest/dedup logic in isolation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from spacehasten.config.properties import PropertyRanges
from spacehasten.stages.library_screen import (
    _build_infer_command,
    _ingest_predictions,
    _prepare_missing_chunks,
    _write_control_param,
    _write_smarts_param,
)


def test_write_control_param_canonical_order(tmp_path: Path) -> None:
    props = PropertyRanges.model_validate({
        "mw": {"min": 10.0, "max": 500.0},
        "slogp": {"min": -2.0, "max": 5.0},
        "hba": {"min": 0, "max": 10},
        "hbd": {"min": 0, "max": 5},
        "rotbonds": {"min": 0, "max": 10},
        "tpsa": {"min": 0.0, "max": 140.0},
    })
    out = tmp_path / "inputs" / "control.param"
    _write_control_param(out, props)
    lines = out.read_text().splitlines()
    assert lines == [
        "10.0", "500.0",   # mw
        "-2.0", "5.0",     # slogp
        "0", "10",         # hba
        "0", "5",          # hbd
        "0", "10",         # rotbonds
        "0.0", "140.0",    # tpsa
    ]


def test_write_smarts_param_empty_returns_false(tmp_path: Path) -> None:
    props = PropertyRanges()  # no SMARTS by default
    out = tmp_path / "inputs" / "smarts.txt"
    has_smarts = _write_smarts_param(out, props)
    assert has_smarts is False
    assert out.read_text() == ""


def test_write_smarts_param_writes_mode_prefixed_lines(tmp_path: Path) -> None:
    props = PropertyRanges.model_validate({
        "smarts_include": ["c1ccccc1"],
        "smarts_exclude": ["C(=O)[OH]", "[N+]"],
    })
    out = tmp_path / "inputs" / "smarts.txt"
    has_smarts = _write_smarts_param(out, props)
    assert has_smarts is True
    assert out.read_text().splitlines() == [
        "include:c1ccccc1",
        "exclude:C(=O)[OH]",
        "exclude:[N+]",
    ]


def test_build_infer_command_omits_smarts_flag_by_default(tmp_path: Path) -> None:
    from spacehasten.config.settings import Settings

    cmd = _build_infer_command(
        tmp_path / "results", tmp_path / "model", tmp_path / "control.param",
        Settings(), ["python", "-m", "spacehasten.remote.library_infer"],
        top_n=None, cutoff=-8.0,
    )
    assert "--smarts" not in cmd


def test_build_infer_command_includes_smarts_flag_when_given(tmp_path: Path) -> None:
    from spacehasten.config.settings import Settings

    smarts_path = tmp_path / "inputs" / "smarts.txt"
    cmd = _build_infer_command(
        tmp_path / "results", tmp_path / "model", tmp_path / "control.param",
        Settings(), ["python", "-m", "spacehasten.remote.library_infer"],
        top_n=None, cutoff=-8.0, smarts_path=smarts_path,
    )
    assert f"--smarts {smarts_path}" in cmd


def test_prepare_missing_chunks_skips_existing_outputs(tmp_path: Path) -> None:
    chunk_a = tmp_path / "chunk_00000.parquet"
    chunk_b = tmp_path / "chunk_00001.parquet"
    chunk_a.write_bytes(b"a")
    chunk_b.write_bytes(b"b")

    inputs_dir = tmp_path / "inputs"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    # chunk_00000 already has a predicted output -> should be skipped.
    (results_dir / "predicted_chunk_00000.parquet").write_bytes(b"done")

    links = _prepare_missing_chunks([chunk_a, chunk_b], inputs_dir, results_dir)

    assert len(links) == 1
    assert links[0].name == "infer_1.parquet"
    assert links[0].resolve() == chunk_b.resolve()


def test_prepare_missing_chunks_all_pending(tmp_path: Path) -> None:
    chunk_a = tmp_path / "chunk_00000.parquet"
    chunk_b = tmp_path / "chunk_00001.parquet"
    chunk_a.write_bytes(b"a")
    chunk_b.write_bytes(b"b")

    inputs_dir = tmp_path / "inputs"
    results_dir = tmp_path / "results"

    links = _prepare_missing_chunks([chunk_a, chunk_b], inputs_dir, results_dir)
    assert [link.name for link in links] == ["infer_1.parquet", "infer_2.parquet"]
    assert links[0].resolve() == chunk_a.resolve()
    assert links[1].resolve() == chunk_b.resolve()


def test_prepare_missing_chunks_clears_stale_symlinks(tmp_path: Path) -> None:
    chunk_a = tmp_path / "chunk_00000.parquet"
    chunk_a.write_bytes(b"a")
    inputs_dir = tmp_path / "inputs"
    results_dir = tmp_path / "results"
    inputs_dir.mkdir()
    stale = inputs_dir / "infer_1.parquet"
    stale.symlink_to(tmp_path / "does-not-exist.parquet")

    links = _prepare_missing_chunks([chunk_a], inputs_dir, results_dir)
    assert len(links) == 1
    assert links[0].resolve() == chunk_a.resolve()


def _write_pred_chunk(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_ingest_predictions_dedups_keeping_min_score(tmp_path: Path) -> None:
    chunk_a = tmp_path / "chunk_00000.parquet"
    chunk_b = tmp_path / "chunk_00001.parquet"
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    _write_pred_chunk(results_dir / "predicted_chunk_00000.parquet", [
        {"reghash": "h1", "smiles": "CCO", "compound_id": "ENA-1", "pred_score": -5.0},
        {"reghash": "h2", "smiles": "CCN", "compound_id": "ENA-2", "pred_score": -3.0},
    ])
    # h1 reappears with a better (more negative) score in the second chunk.
    _write_pred_chunk(results_dir / "predicted_chunk_00001.parquet", [
        {"reghash": "h1", "smiles": "CCO", "compound_id": "ENA-1-dup", "pred_score": -9.0},
        {"reghash": "h3", "smiles": "CCC", "compound_id": "ENA-3", "pred_score": -1.0},
    ])

    combined, n_predicted = _ingest_predictions([chunk_a, chunk_b], results_dir)
    assert n_predicted == 4
    assert len(combined) == 3  # h1 deduped
    h1_row = combined.loc[combined["reghash"] == "h1"].iloc[0]
    assert h1_row["pred_score"] == -9.0


def test_ingest_predictions_missing_output_raises(tmp_path: Path) -> None:
    chunk_a = tmp_path / "chunk_00000.parquet"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        _ingest_predictions([chunk_a], results_dir)
