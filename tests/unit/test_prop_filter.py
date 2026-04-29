"""Unit tests for the remote prop-filter (legacy ``control.py`` port)."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from spacehasten.remote.prop_filter import _Bounds, filter_smiles, main


def _write_param(path: Path) -> None:
    # Permissive bounds: every reasonable drug-like molecule passes.
    path.write_text("\n".join([
        "0", "10000",      # mw
        "-10", "10",       # slogp
        "0", "20",         # hba
        "0", "20",         # hbd
        "0", "30",         # rotbonds
        "0", "500",        # tpsa
    ]) + "\n")


def _write_strict_param(path: Path) -> None:
    # Bounds tight enough to filter out anything heavier than ethanol.
    path.write_text("\n".join([
        "0", "60",         # mw  (ethanol: ~46.07)
        "-10", "10",
        "0", "20",
        "0", "20",
        "0", "30",
        "0", "500",
    ]) + "\n")


def test_prop_filter_passes_drug_like(tmp_path: Path) -> None:
    inp = tmp_path / "input.smi"
    inp.write_text("CCO§ethanol-1\nc1ccccc1§benzene-1\nCc1ccccc1§toluene-1\n")
    params = tmp_path / "control.param"
    _write_param(params)

    out = tmp_path / "out.csv"
    n = filter_smiles(inp, _Bounds.read(params), out)
    assert n == 3

    rows = list(csv.DictReader(out.open()))
    assert [r["smiles"] for r in rows] == ["CCO", "c1ccccc1", "Cc1ccccc1"]
    # smilesid carries the legacy reghash§smiles§title packed string.
    for r in rows:
        parts = r["smilesid"].split("§")
        assert len(parts) == 3
        assert parts[1] == r["smiles"]
        assert parts[2].endswith("-1")


def test_prop_filter_strict_bounds_drop_heavies(tmp_path: Path) -> None:
    inp = tmp_path / "input.smi"
    inp.write_text("CCO§ethanol\nc1ccccc1§benzene\n")
    params = tmp_path / "control.param"
    _write_strict_param(params)

    out = tmp_path / "out.csv"
    n = filter_smiles(inp, _Bounds.read(params), out)
    assert n == 1

    smiles = [r["smiles"] for r in csv.DictReader(out.open())]
    assert smiles == ["CCO"]


def test_prop_filter_reads_gzip_input(tmp_path: Path) -> None:
    inp = tmp_path / "input.smi.gz"
    with gzip.open(inp, "wt") as w:
        w.write("CCO§ethanol\n")
    params = tmp_path / "control.param"
    _write_param(params)

    out = tmp_path / "out.csv"
    n = filter_smiles(inp, _Bounds.read(params), out)
    assert n == 1


def test_prop_filter_skips_invalid_smiles(tmp_path: Path) -> None:
    inp = tmp_path / "input.smi"
    inp.write_text("CCO§ok\nNOTASMILES§bad\n\n")
    params = tmp_path / "control.param"
    _write_param(params)

    out = tmp_path / "out.csv"
    n = filter_smiles(inp, _Bounds.read(params), out)
    assert n == 1


def test_prop_filter_main_default_output_name(tmp_path: Path) -> None:
    inp = tmp_path / "control_x_cpu1.smi.gz"
    with gzip.open(inp, "wt") as w:
        w.write("CCO§e\n")
    params = tmp_path / "control.param"
    _write_param(params)

    rc = main([str(inp), str(params)])
    assert rc == 0
    expected = tmp_path / "propoutput_control_x_cpu1.csv"
    assert expected.exists()


def test_prop_filter_main_missing_input(tmp_path: Path) -> None:
    params = tmp_path / "control.param"
    _write_param(params)
    rc = main([str(tmp_path / "nope.smi"), str(params)])
    assert rc == 1


def test_bounds_short_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "short.param"
    p.write_text("0\n100\n")
    with pytest.raises(ValueError):
        _Bounds.read(p)
