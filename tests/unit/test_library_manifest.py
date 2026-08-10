"""Tests for :class:`spacehasten.stages.library_build.LibraryManifest`."""

from __future__ import annotations

from pathlib import Path

import pytest

from spacehasten.stages.library_build import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    LibraryManifest,
)


def _make_manifest(**overrides: object) -> LibraryManifest:
    defaults: dict[str, object] = dict(
        format_version=1,
        source_files=["Enamine_Diverse_REAL.cxsmiles.bz2"],
        n_compounds=4,
        n_chunks=2,
        chunk_glob="chunk_*.parquet",
        chunk_rows=[2, 2],
        columns=list(REQUIRED_COLUMNS),
        optional_columns=list(OPTIONAL_COLUMNS),
        chunk_size=2,
        compression="zstd",
        props_source="enamine",
        rdkit_version="2024.03.5",
        reghash_algo="RegistrationHash.TAUTOMER_HASH",
        canonicalization="rdkit-canonical",
        built_at="2026-08-03T15:50:00Z",
    )
    defaults.update(overrides)
    return LibraryManifest(**defaults)  # type: ignore[arg-type]


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    manifest = _make_manifest()
    path = tmp_path / "manifest.json"
    manifest.save(path)

    loaded = LibraryManifest.load(path)
    assert loaded == manifest


def test_validate_passes_for_well_formed_manifest() -> None:
    manifest = _make_manifest()
    manifest.validate()  # must not raise


def test_validate_rejects_wrong_format_version() -> None:
    manifest = _make_manifest(format_version=2)
    with pytest.raises(ValueError, match="format_version"):
        manifest.validate()


def test_validate_rejects_missing_required_column() -> None:
    manifest = _make_manifest(columns=["compound_id", "smiles", "reghash"])
    with pytest.raises(ValueError, match="missing required columns"):
        manifest.validate()


def test_chunk_paths_enumerates_and_sorts(tmp_path: Path) -> None:
    manifest = _make_manifest(chunk_glob="chunk_*.parquet")
    (tmp_path / "chunk_00001.parquet").write_bytes(b"")
    (tmp_path / "chunk_00000.parquet").write_bytes(b"")
    (tmp_path / "unrelated.txt").write_bytes(b"")

    paths = manifest.chunk_paths(tmp_path)
    assert [p.name for p in paths] == ["chunk_00000.parquet", "chunk_00001.parquet"]
