"""Tests for ``spacehasten.core.molecules``."""

from __future__ import annotations

import pytest

from spacehasten.core.molecules import (
    canonical_smiles,
    parse_cxsmiles,
    tautomer_hash,
)


@pytest.mark.parametrize(
    ("smi_a", "smi_b"),
    [
        ("O=C(O)C", "OC(=O)C"),  # acetic acid: carbonyl/hydroxyl swap
        ("Oc1ccccc1", "OC1=CC=CC=C1"),  # phenol: aromatic vs Kekulé
        ("CC(=O)CC(=O)C", "CC(=O)CC(=O)C"),  # identical input
    ],
)
def test_tautomer_hash_equality(smi_a: str, smi_b: str) -> None:
    h1 = tautomer_hash(smi_a)
    h2 = tautomer_hash(smi_b)
    assert h1 is not None
    assert h2 is not None
    assert h1 == h2


def test_tautomer_hash_distinguishes_unrelated() -> None:
    assert tautomer_hash("CCO") != tautomer_hash("c1ccccc1")


def test_tautomer_hash_invalid_returns_none() -> None:
    assert tautomer_hash("not_a_smiles!!") is None
    assert tautomer_hash("") is None


def test_canonical_smiles_roundtrip() -> None:
    assert canonical_smiles("c1ccccc1") == canonical_smiles("C1=CC=CC=C1")


def test_canonical_smiles_invalid() -> None:
    assert canonical_smiles("zzz") is None


def test_parse_cxsmiles_basic() -> None:
    out = parse_cxsmiles("CCO\tethanol\n")
    assert out is not None
    smi, mol_id = out
    assert mol_id == "ethanol"
    assert smi == canonical_smiles("CCO")


def test_parse_cxsmiles_space_separated() -> None:
    out = parse_cxsmiles("c1ccccc1 benzene")
    assert out is not None
    assert out[1] == "benzene"


def test_parse_cxsmiles_missing_id() -> None:
    assert parse_cxsmiles("CCO") is None


def test_parse_cxsmiles_invalid_smiles() -> None:
    assert parse_cxsmiles("xx_invalid xx 1") is None


def test_parse_cxsmiles_empty() -> None:
    assert parse_cxsmiles("") is None
    assert parse_cxsmiles("\n") is None
