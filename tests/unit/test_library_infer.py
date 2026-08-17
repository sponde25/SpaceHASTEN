"""Unit tests for spacehasten.remote.library_infer.

The vectorized property filter (apply_property_filter) and the parquet
I/O paths are exercised directly; the chemprop-dependent predict_scores
is monkeypatched (mirrors the existing convention of not exercising real
chemprop in unit tests - see test_prop_filter.py / test_prediction_local.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacehasten.remote import library_infer
from spacehasten.remote.library_infer import (
    apply_property_filter,
    apply_smarts_filter,
    infer_chunk,
)
from spacehasten.remote.prop_filter import _Bounds

_PERMISSIVE_PARAM_LINES = [
    "0", "10000",   # mw
    "-10", "10",    # slogp
    "0", "20",      # hba
    "0", "20",      # hbd
    "0", "30",      # rotbonds
    "0", "500",     # tpsa
]

_STRICT_PARAM_LINES = [
    "0", "60",      # mw: only ethanol-like passes
    "-10", "10",
    "0", "20",
    "0", "20",
    "0", "30",
    "0", "500",
]


def _write_bounds(tmp_path: Path, lines: list[str]) -> _Bounds:
    path = tmp_path / "control.param"
    path.write_text("\n".join(lines) + "\n")
    return _Bounds.read(path)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


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


# --------------------------------------------------------------------------- #
# apply_property_filter                                                       #
# --------------------------------------------------------------------------- #


def test_apply_property_filter_permissive_keeps_all(tmp_path: Path) -> None:
    bounds = _write_bounds(tmp_path, _PERMISSIVE_PARAM_LINES)
    df = _make_df([
        _row("ENA-1", "CCO", "h1", 46.07),
        _row("ENA-2", "c1ccccc1", "h2", 78.11),
    ])
    survivors = apply_property_filter(df, bounds)
    assert len(survivors) == 2


def test_apply_property_filter_strict_drops_heavy(tmp_path: Path) -> None:
    bounds = _write_bounds(tmp_path, _STRICT_PARAM_LINES)
    df = _make_df([
        _row("ENA-1", "CCO", "h1", 46.07),
        _row("ENA-2", "c1ccccc1", "h2", 78.11),  # too heavy for strict bounds
    ])
    survivors = apply_property_filter(df, bounds)
    assert survivors["compound_id"].tolist() == ["ENA-1"]


# --------------------------------------------------------------------------- #
# infer_chunk (predict_scores monkeypatched)                                  #
# --------------------------------------------------------------------------- #


def test_infer_chunk_writes_filtered_and_predicted_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = _make_df([
        _row("ENA-1", "CCO", "h1", 46.07),
        _row("ENA-2", "c1ccccc1", "h2", 78.11),
        _row("ENA-3", "CCN", "h3", 500.0),  # dropped by strict bounds
    ])
    chunk_path = tmp_path / "chunk.parquet"
    df.to_parquet(chunk_path, index=False)
    bounds = _write_bounds(tmp_path, _STRICT_PARAM_LINES)

    def fake_predict_scores(smiles, model_dir, batch_size, num_workers, accelerator, devices):
        return np.array([-5.0] * len(smiles))

    monkeypatch.setattr(library_infer, "predict_scores", fake_predict_scores)

    model_dir = tmp_path / "model"
    (model_dir / "model_0").mkdir(parents=True)
    (model_dir / "model_0" / "pytorch_model.bin").write_bytes(b"fake")

    out_path = tmp_path / "out.csv"
    n = infer_chunk(chunk_path, model_dir, out_path, bounds=bounds)
    assert n == 1  # only ENA-1 passes strict bounds

    out_df = pd.read_csv(out_path)
    assert list(out_df.columns) == ["reghash", "smiles", "compound_id", "pred_score"]
    assert out_df["compound_id"].tolist() == ["ENA-1"]
    assert out_df["pred_score"].tolist() == [-5.0]


def test_infer_chunk_score_cutoff_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = _make_df([
        _row("ENA-1", "CCO", "h1", 46.07),
        _row("ENA-2", "CCN", "h2", 45.08),
        _row("ENA-3", "CCC", "h3", 44.1),
    ])
    chunk_path = tmp_path / "chunk.parquet"
    df.to_parquet(chunk_path, index=False)
    bounds = _write_bounds(tmp_path, _PERMISSIVE_PARAM_LINES)

    def fake_predict_scores(smiles, model_dir, batch_size, num_workers, accelerator, devices):
        return np.array([-9.0, -3.0, -8.5])

    monkeypatch.setattr(library_infer, "predict_scores", fake_predict_scores)

    model_dir = tmp_path / "model"
    (model_dir / "model_0").mkdir(parents=True)
    (model_dir / "model_0" / "pytorch_model.bin").write_bytes(b"fake")

    out_path = tmp_path / "out.csv"
    n = infer_chunk(chunk_path, model_dir, out_path, bounds=bounds, score_cutoff=-8.0)
    assert n == 2  # ENA-1 (-9.0) and ENA-3 (-8.5) pass; ENA-2 (-3.0) doesn't

    out_df = pd.read_csv(out_path)
    assert set(out_df["compound_id"]) == {"ENA-1", "ENA-3"}


def test_infer_chunk_top_n_per_chunk_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = _make_df([
        _row("ENA-1", "CCO", "h1", 46.07),
        _row("ENA-2", "CCN", "h2", 45.08),
        _row("ENA-3", "CCC", "h3", 44.1),
    ])
    chunk_path = tmp_path / "chunk.parquet"
    df.to_parquet(chunk_path, index=False)
    bounds = _write_bounds(tmp_path, _PERMISSIVE_PARAM_LINES)

    def fake_predict_scores(smiles, model_dir, batch_size, num_workers, accelerator, devices):
        return np.array([-9.0, -3.0, -8.5])

    monkeypatch.setattr(library_infer, "predict_scores", fake_predict_scores)

    model_dir = tmp_path / "model"
    (model_dir / "model_0").mkdir(parents=True)
    (model_dir / "model_0" / "pytorch_model.bin").write_bytes(b"fake")

    out_path = tmp_path / "out.csv"
    n = infer_chunk(chunk_path, model_dir, out_path, bounds=bounds, top_n_per_chunk=1)
    assert n == 1
    out_df = pd.read_csv(out_path)
    assert out_df["compound_id"].tolist() == ["ENA-1"]  # best (lowest) score


def test_infer_chunk_top_n_per_chunk_spans_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # block_size=3 -> two blocks of [s1,s2,s3] and [s4,s5,s6]. The per-block
    # trim (nsmallest(top_n) within each block) plus the final trim must still
    # yield the exact global top-N, including winners that live in different
    # blocks (s2 in block 1, s4 in block 2).
    score_by_smiles = {
        "s1": -1.0, "s2": -9.0, "s3": -2.0,   # block 1 -> keep s2(-9), s3(-2)
        "s4": -8.0, "s5": -3.0, "s6": -7.0,   # block 2 -> keep s4(-8), s6(-7)
    }
    df = _make_df([
        _row("ENA-1", "s1", "h1", 46.0),
        _row("ENA-2", "s2", "h2", 46.0),
        _row("ENA-3", "s3", "h3", 46.0),
        _row("ENA-4", "s4", "h4", 46.0),
        _row("ENA-5", "s5", "h5", 46.0),
        _row("ENA-6", "s6", "h6", 46.0),
    ])
    chunk_path = tmp_path / "chunk.parquet"
    df.to_parquet(chunk_path, index=False)
    bounds = _write_bounds(tmp_path, _PERMISSIVE_PARAM_LINES)

    calls: list[int] = []

    def fake_predict_scores(smiles, model_dir, batch_size, num_workers, accelerator, devices):
        calls.append(len(smiles))
        return np.array([score_by_smiles[s] for s in smiles])

    monkeypatch.setattr(library_infer, "predict_scores", fake_predict_scores)

    model_dir = tmp_path / "model"
    (model_dir / "model_0").mkdir(parents=True)
    (model_dir / "model_0" / "pytorch_model.bin").write_bytes(b"fake")

    out_path = tmp_path / "out.csv"
    n = infer_chunk(
        chunk_path, model_dir, out_path,
        bounds=bounds, top_n_per_chunk=2, block_size=3,
    )
    # Streamed in two blocks of 3 (proves per-block trimming is exercised).
    assert calls == [3, 3]
    assert n == 2
    out_df = pd.read_csv(out_path).sort_values("pred_score").reset_index(drop=True)
    # Global top-2 by score: s2(-9.0) then s4(-8.0).
    assert out_df["compound_id"].tolist() == ["ENA-2", "ENA-4"]
    assert out_df["pred_score"].tolist() == [-9.0, -8.0]


def test_infer_chunk_no_survivors_writes_empty_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = _make_df([_row("ENA-1", "CCO", "h1", 500.0)])  # too heavy
    chunk_path = tmp_path / "chunk.parquet"
    df.to_parquet(chunk_path, index=False)
    bounds = _write_bounds(tmp_path, _STRICT_PARAM_LINES)

    def fail_predict(*args, **kwargs):
        raise AssertionError("predict_scores must not be called with 0 survivors")

    monkeypatch.setattr(library_infer, "predict_scores", fail_predict)

    out_path = tmp_path / "out.csv"
    n = infer_chunk(chunk_path, tmp_path / "model", out_path, bounds=bounds)
    assert n == 0

    out_df = pd.read_csv(out_path)
    assert list(out_df.columns) == ["reghash", "smiles", "compound_id", "pred_score"]
    assert len(out_df) == 0


def test_infer_chunk_missing_required_columns_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"smiles": ["CCO"], "compound_id": ["ENA-1"]})
    chunk_path = tmp_path / "chunk.parquet"
    df.to_parquet(chunk_path, index=False)
    bounds = _write_bounds(tmp_path, _PERMISSIVE_PARAM_LINES)

    with pytest.raises(ValueError, match="missing required columns"):
        infer_chunk(chunk_path, tmp_path / "model", tmp_path / "out.csv", bounds=bounds)


# --------------------------------------------------------------------------- #
# SMARTS filtering                                                            #
# --------------------------------------------------------------------------- #


def _write_smarts(tmp_path: Path, lines: list[str]) -> library_infer._SmartsBounds:
    path = tmp_path / "smarts.txt"
    path.write_text("\n".join(lines) + "\n")
    return library_infer._SmartsBounds.read(path)


def test_smarts_bounds_inactive_when_empty(tmp_path: Path) -> None:
    smarts = _write_smarts(tmp_path, ["# just a comment", ""])
    assert smarts.active is False


def test_smarts_bounds_bad_mode_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        _write_smarts(tmp_path, ["banana:CCO"])


def test_smarts_bounds_bad_pattern_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="could not parse SMARTS"):
        _write_smarts(tmp_path, ["exclude:[[[not-smarts"])


def test_apply_smarts_filter_exclude_drops_carboxylic_acid(tmp_path: Path) -> None:
    # Carboxylic acid SMARTS excludes acetic acid but keeps ethanol/benzene.
    smarts = _write_smarts(tmp_path, ["exclude:C(=O)[OH]"])
    df = _make_df([
        _row("ENA-1", "CCO", "h1", 46.07),       # ethanol -> keep
        _row("ENA-2", "CC(=O)O", "h2", 60.05),   # acetic acid -> drop
        _row("ENA-3", "c1ccccc1", "h3", 78.11),  # benzene -> keep
    ])
    out = apply_smarts_filter(df, smarts)
    assert out["compound_id"].tolist() == ["ENA-1", "ENA-3"]


def test_apply_smarts_filter_include_requires_match(tmp_path: Path) -> None:
    # Require an aromatic ring: only benzene survives.
    smarts = _write_smarts(tmp_path, ["include:c1ccccc1"])
    df = _make_df([
        _row("ENA-1", "CCO", "h1", 46.07),        # no ring -> drop
        _row("ENA-2", "c1ccccc1", "h2", 78.11),   # benzene -> keep
    ])
    out = apply_smarts_filter(df, smarts)
    assert out["compound_id"].tolist() == ["ENA-2"]


def test_apply_smarts_filter_drops_unparseable_smiles(tmp_path: Path) -> None:
    smarts = _write_smarts(tmp_path, ["exclude:C(=O)[OH]"])
    df = _make_df([
        _row("ENA-1", "CCO", "h1", 46.07),
        _row("ENA-2", "NOTASMILES", "h2", 50.0),  # unparseable -> drop
    ])
    out = apply_smarts_filter(df, smarts)
    assert out["compound_id"].tolist() == ["ENA-1"]


def test_apply_smarts_filter_inactive_is_noop(tmp_path: Path) -> None:
    smarts = _write_smarts(tmp_path, ["# nothing"])
    df = _make_df([_row("ENA-1", "CC(=O)O", "h1", 60.05)])
    out = apply_smarts_filter(df, smarts)
    assert out["compound_id"].tolist() == ["ENA-1"]


def test_infer_chunk_applies_smarts_after_property_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = _make_df([
        _row("ENA-1", "CCO", "h1", 46.07),        # keep
        _row("ENA-2", "CC(=O)O", "h2", 60.05),    # dropped by SMARTS exclude
        _row("ENA-3", "c1ccccc1", "h3", 78.11),   # keep
    ])
    chunk_path = tmp_path / "chunk.parquet"
    df.to_parquet(chunk_path, index=False)
    bounds = _write_bounds(tmp_path, _PERMISSIVE_PARAM_LINES)
    smarts = _write_smarts(tmp_path, ["exclude:C(=O)[OH]"])

    captured: dict[str, list[str]] = {}

    def fake_predict_scores(smiles, model_dir, batch_size, num_workers, accelerator, devices):
        captured["smiles"] = list(smiles)
        return np.array([-5.0] * len(smiles))

    monkeypatch.setattr(library_infer, "predict_scores", fake_predict_scores)

    model_dir = tmp_path / "model"
    (model_dir / "model_0").mkdir(parents=True)
    (model_dir / "model_0" / "pytorch_model.bin").write_bytes(b"fake")

    out_path = tmp_path / "out.csv"
    n = infer_chunk(chunk_path, model_dir, out_path, bounds=bounds, smarts=smarts)
    assert n == 2
    # The acid must be removed *before* prediction (predict only sees survivors).
    assert captured["smiles"] == ["CCO", "c1ccccc1"]

    out_df = pd.read_csv(out_path)
    assert out_df["compound_id"].tolist() == ["ENA-1", "ENA-3"]


def test_infer_chunk_all_dropped_by_smarts_writes_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = _make_df([_row("ENA-1", "CC(=O)O", "h1", 60.05)])  # acid, excluded
    chunk_path = tmp_path / "chunk.parquet"
    df.to_parquet(chunk_path, index=False)
    bounds = _write_bounds(tmp_path, _PERMISSIVE_PARAM_LINES)
    smarts = _write_smarts(tmp_path, ["exclude:C(=O)[OH]"])

    def fail_predict(*args, **kwargs):
        raise AssertionError("predict_scores must not be called with 0 survivors")

    monkeypatch.setattr(library_infer, "predict_scores", fail_predict)

    out_path = tmp_path / "out.csv"
    n = infer_chunk(chunk_path, tmp_path / "model", out_path, bounds=bounds, smarts=smarts)
    assert n == 0
    out_df = pd.read_csv(out_path)
    assert list(out_df.columns) == ["reghash", "smiles", "compound_id", "pred_score"]
    assert len(out_df) == 0
