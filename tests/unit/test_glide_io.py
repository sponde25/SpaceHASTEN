"""Unit tests for :mod:`spacehasten.tools.glide`."""

from __future__ import annotations

from pathlib import Path

import pytest

from spacehasten.tools.glide import parse_glide_csv, write_glide_in, write_phase_inp


def test_write_phase_inp_references_basenames(tmp_path: Path) -> None:
    inp = tmp_path / "chunk_3.inp"
    write_phase_inp(inp)
    text = inp.read_text()
    # FILES line points at <stem>.smi (basename only).
    assert "FILES   chunk_3.smi," in text
    # DATABASE line points at <stem>.phdb (basename only).
    assert "DATABASE  chunk_3.phdb" in text
    # Sentinel header so we know we wrote the right template.
    assert text.startswith("[SET:ORIGINAL_LIGANDS]\n")
    assert "[USEROUTS]" in text


def test_write_glide_in_strips_and_rewrites_ligand_and_grid(tmp_path: Path) -> None:
    template = (
        b"FORCEFIELD   OPLS_2005\n"
        b"GRIDFILE     /old/training/grid.zip\n"
        b"LIGANDFILE   /old/ligands.maegz\n"
        b"PRECISION    SP\n"
        b"POSE_OUTTYPE poseviewer\n"
    )
    out = tmp_path / "glide_chunk_2.in"
    write_glide_in(out, template, ligand_stem="chunk_2")
    text = out.read_text()
    lines = text.splitlines()
    # Rewritten lines come first, in the documented order.
    assert lines[0] == "LIGANDFILE   chunk_2_1.maegz"
    assert lines[1] == "GRIDFILE     glide_grid.zip"
    # The original LIGANDFILE/GRIDFILE lines were dropped.
    assert "/old/training/grid.zip" not in text
    assert "/old/ligands.maegz" not in text
    # Other template lines survive.
    assert "FORCEFIELD   OPLS_2005" in text
    assert "PRECISION    SP" in text
    assert "POSE_OUTTYPE poseviewer" in text


def test_parse_glide_csv_keeps_min_score_per_title(tmp_path: Path) -> None:
    csv_path = tmp_path / "glide_chunk_1.csv"
    csv_path.write_text(
        "title,r_i_docking_score,r_i_glide_emodel\n"
        "100,-7.5,-50\n"
        "100,-8.2,-55\n"   # better pose for the same compound — must win
        "101,-6.0,-40\n"
        "100,-7.9,-52\n"   # worse than -8.2 — must be ignored
        "102,,\n"           # no score — skipped
        "103,not-a-number,-30\n"  # unparseable — skipped
        "104,-5.1,-25\n"
    )
    result = parse_glide_csv(csv_path)
    assert result == {
        "100": -8.2,
        "101": -6.0,
        "104": -5.1,
    }


def test_parse_glide_csv_rejects_missing_columns(tmp_path: Path) -> None:
    bad = tmp_path / "no_title.csv"
    bad.write_text("name,r_i_docking_score\nfoo,-7\n")
    with pytest.raises(ValueError, match="title"):
        parse_glide_csv(bad)
    bad2 = tmp_path / "no_score.csv"
    bad2.write_text("title,other\nfoo,1\n")
    with pytest.raises(ValueError, match="r_i_docking_score"):
        parse_glide_csv(bad2)


def test_glide_io_roundtrip(tmp_path: Path) -> None:
    # Write a Glide .in via the tool, then re-read it and assert it can
    # be re-stripped to the same kept lines (idempotency).
    template = (
        b"GRIDFILE     /old.zip\n"
        b"LIGANDFILE   /old.maegz\n"
        b"FORCEFIELD   OPLS_2005\n"
    )
    out1 = tmp_path / "glide_chunk_1.in"
    write_glide_in(out1, template, ligand_stem="chunk_1")
    # Roundtrip: feed the produced .in back through the writer with a
    # different stem; the previous LIGANDFILE/GRIDFILE must be replaced
    # again rather than duplicated.
    out2 = tmp_path / "glide_chunk_2.in"
    write_glide_in(out2, out1.read_bytes(), ligand_stem="chunk_2")
    text = out2.read_text()
    assert text.count("LIGANDFILE") == 1
    assert text.count("GRIDFILE") == 1
    assert text.startswith("LIGANDFILE   chunk_2_1.maegz\n")
    assert "FORCEFIELD   OPLS_2005" in text
