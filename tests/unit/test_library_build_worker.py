"""Tests for column resolution (stages/library_build) and the
remote/library_build worker's descriptor parsing (plan sections 5.2, 8).

Uses the confirmed real Enamine REAL header:
``smiles id MW HAC sLogP HBA HBD RotBonds FSP3 TPSA QED lead-like
350/3_lead-like fragments strict_fragments PPI_modulators
natural_product-like Type InChiKey``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacehasten.remote.library_build import build_chunk
from spacehasten.stages.library_build import _resolve_columns

REAL_HEADER = (
    "smiles\tid\tMW\tHAC\tsLogP\tHBA\tHBD\tRotBonds\tFSP3\tTPSA\tQED\t"
    "lead-like\t350/3_lead-like\tfragments\tstrict_fragments\t"
    "PPI_modulators\tnatural_product-like\tType\tInChiKey"
)


# --------------------------------------------------------------------------- #
# Header-by-name column resolution                                            #
# --------------------------------------------------------------------------- #


def test_resolve_columns_by_name_real_header() -> None:
    fields = REAL_HEADER.split("\t")
    resolved = _resolve_columns(fields)
    assert resolved["smiles"] == 0
    assert resolved["id"] == 1
    assert resolved["mw"] == 2
    assert resolved["slogp"] == 4
    assert resolved["hba"] == 5
    assert resolved["hbd"] == 6
    assert resolved["rotbonds"] == 7
    assert resolved["fsp3"] == 8
    assert resolved["tpsa"] == 9
    assert resolved["qed"] == 10
    assert resolved["inchikey"] == 18


def test_resolve_columns_is_case_insensitive() -> None:
    fields = ["SMILES", "ID", "mw", "SLOGP", "Hba", "hbd", "ROTBONDS", "Tpsa"]
    resolved = _resolve_columns(fields)
    assert resolved["smiles"] == 0
    assert resolved["id"] == 1


def test_resolve_columns_raises_on_missing_required_field() -> None:
    fields = ["smiles", "id", "MW"]  # missing slogp/hba/hbd/rotbonds/tpsa
    with pytest.raises(ValueError, match="required columns"):
        _resolve_columns(fields)


def test_resolve_columns_override_by_index() -> None:
    fields = ["a", "b", "c", "d", "e", "f", "g", "h"]
    overrides = {
        "smiles": "0", "id": "1", "mw": "2", "slogp": "3",
        "hba": "4", "hbd": "5", "rotbonds": "6", "tpsa": "7",
    }
    resolved = _resolve_columns(fields, overrides)
    assert resolved == {
        "smiles": 0, "id": 1, "mw": 2, "slogp": 3,
        "hba": 4, "hbd": 5, "rotbonds": 6, "tpsa": 7,
    }


def test_resolve_columns_override_by_name() -> None:
    fields = ["mySmiles", "myId", "MW", "sLogP", "HBA", "HBD", "RotBonds", "TPSA"]
    overrides = {"smiles": "mySmiles", "id": "myId"}
    resolved = _resolve_columns(fields, overrides)
    assert resolved["smiles"] == 0
    assert resolved["id"] == 1


# --------------------------------------------------------------------------- #
# remote/library_build.build_chunk descriptor parsing                        #
# --------------------------------------------------------------------------- #


def _write_shard(tmp_path: Path) -> Path:
    """A headerless shard matching the real Enamine column layout."""
    shard = tmp_path / "shard_1.smi"
    shard.write_text(
        "CCO\tENA-1\t46.07\t3\t-0.31\t1\t1\t0\t1.0\t20.23\t0.45\t1\t0\t0\t0\t0\t0\tsmall\tKEY1\n"
        "c1ccccc1\tENA-2\t78.11\t6\t1.9\t0\t0\t0\t0.0\t0.0\t0.44\t1\t0\t0\t0\t0\t0\tsmall\tKEY2\n"
    )
    return shard


def test_build_chunk_uses_source_descriptors_by_default(tmp_path: Path) -> None:
    shard = _write_shard(tmp_path)
    out = tmp_path / "chunk.parquet"

    n = build_chunk(
        shard, out,
        smiles_col=0, id_col=1,
        mw_col=2, slogp_col=4, hba_col=5, hbd_col=6, rotbonds_col=7, tpsa_col=9,
        fsp3_col=8, qed_col=10, inchikey_col=18,
        recompute_props=False,
    )
    assert n == 2

    import pyarrow.parquet as pq

    table = pq.read_table(out).to_pydict()
    assert table["compound_id"] == ["ENA-1", "ENA-2"]
    assert table["mw"] == pytest.approx([46.07, 78.11], abs=1e-2)
    assert table["slogp"] == pytest.approx([-0.31, 1.9], abs=1e-2)
    assert table["hba"] == [1, 0]
    assert table["hbd"] == [1, 0]
    assert table["rotbonds"] == [0, 0]
    assert table["tpsa"] == pytest.approx([20.23, 0.0], abs=1e-2)
    assert table["fsp3"] == pytest.approx([1.0, 0.0], abs=1e-2)
    assert table["qed"] == pytest.approx([0.45, 0.44], abs=1e-2)
    assert table["inchikey"] == ["KEY1", "KEY2"]
    # reghash + canonical smiles are always RDKit-derived.
    assert all(table["reghash"])
    assert table["smiles"] == ["CCO", "c1ccccc1"]


def test_build_chunk_recompute_props_uses_rdkit_values(tmp_path: Path) -> None:
    shard = _write_shard(tmp_path)
    out = tmp_path / "chunk.parquet"

    build_chunk(
        shard, out,
        smiles_col=0, id_col=1,
        recompute_props=True,
    )

    import pyarrow.parquet as pq
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    table = pq.read_table(out).to_pydict()
    mol_ccc = Chem.MolFromSmiles("CCO")
    assert table["mw"][0] == pytest.approx(Descriptors.MolWt(mol_ccc), abs=1e-2)
    assert table["slogp"][0] == pytest.approx(Crippen.MolLogP(mol_ccc), abs=1e-2)
    assert table["hba"][0] == rdMolDescriptors.CalcNumHBA(mol_ccc)
    assert table["tpsa"][0] == pytest.approx(rdMolDescriptors.CalcTPSA(mol_ccc), abs=1e-2)
    # The source's slogp (-0.31) must NOT leak through in recompute mode;
    # RDKit's Crippen LogP for ethanol is a different (positive) value.
    assert table["slogp"][0] != pytest.approx(-0.31, abs=1e-4)


def test_build_chunk_drops_unparseable_smiles(tmp_path: Path) -> None:
    shard = tmp_path / "shard.smi"
    shard.write_text(
        "CCO\tENA-1\t46.07\t3\t-0.31\t1\t1\t0\t1.0\t20.23\t0.45\n"
        "not-a-smiles###\tENA-2\t50.0\t3\t1.0\t1\t1\t1\t1.0\t20.0\t0.4\n"
    )
    out = tmp_path / "chunk.parquet"
    n = build_chunk(
        shard, out,
        smiles_col=0, id_col=1,
        mw_col=2, slogp_col=4, hba_col=5, hbd_col=6, rotbonds_col=7, tpsa_col=9,
        recompute_props=False,
    )
    assert n == 1


def test_build_chunk_writes_empty_parquet_when_all_rows_dropped(tmp_path: Path) -> None:
    shard = tmp_path / "shard.smi"
    shard.write_text("not-a-smiles###\tENA-2\t50.0\t3\t1.0\t1\t1\t1\t1.0\t20.0\t0.4\n")
    out = tmp_path / "chunk.parquet"
    n = build_chunk(
        shard, out,
        smiles_col=0, id_col=1,
        mw_col=2, slogp_col=4, hba_col=5, hbd_col=6, rotbonds_col=7, tpsa_col=9,
        recompute_props=False,
    )
    assert n == 0

    import pyarrow.parquet as pq

    table = pq.read_table(out)
    assert table.num_rows == 0
    assert set(table.column_names) == {
        "compound_id", "smiles", "reghash", "mw", "slogp", "hba", "hbd",
        "rotbonds", "tpsa", "fsp3", "qed", "inchikey",
    }
