"""Integration test for the library-build stage with the local scheduler.

Runs the real spacehasten.remote.library_build worker (fast, pure RDKit +
pyarrow, no chemprop) via the local scheduler to exercise the full
split -> array-job -> chunk -> manifest pipeline end to end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from spacehasten.config.settings import Settings
from spacehasten.scheduler import LocalScheduler
from spacehasten.stages.library_build import LibraryManifest, library_build

_REAL_HEADER = (
    "smiles\tid\tMW\tHAC\tsLogP\tHBA\tHBD\tRotBonds\tFSP3\tTPSA\tQED\tType\tInChiKey"
)

# 5 valid rows + 1 unparseable SMILES (dropped).
_ROWS = [
    "CCO\tENA-1\t46.07\t3\t-0.31\t1\t1\t0\t1.0\t20.23\t0.45\tsmall\tKEY1",
    "c1ccccc1\tENA-2\t78.11\t6\t1.9\t0\t0\t0\t0.0\t0.0\t0.44\tsmall\tKEY2",
    "CCN\tENA-3\t45.08\t3\t-0.28\t1\t2\t0\t1.0\t26.02\t0.42\tsmall\tKEY3",
    "CCC\tENA-4\t44.1\t3\t1.4\t0\t0\t0\t1.0\t0.0\t0.4\tsmall\tKEY4",
    "CCCl\tENA-5\t64.51\t3\t1.3\t0\t0\t0\t1.0\t0.0\t0.4\tsmall\tKEY5",
    "not-a-smiles###\tENA-6\t50.0\t3\t1.0\t1\t1\t1\t1.0\t20.0\t0.4\tsmall\tKEY6",
]

_LIBRARY_BUILD_PREFIX = (sys.executable, "-m", "spacehasten.remote.library_build")


def test_library_build_stage_local(tmp_path: Path) -> None:
    src = tmp_path / "diverse.cxsmiles"
    src.write_text(_REAL_HEADER + "\n" + "\n".join(_ROWS) + "\n")

    store = tmp_path / "store"
    settings = Settings()
    scheduler = LocalScheduler()

    manifest = library_build(
        scheduler,
        settings,
        source_files=[src],
        store_dir=store,
        chunk_size=2,  # 3 shards: [2,2,1(+1 dropped)]
        build_command_prefix=_LIBRARY_BUILD_PREFIX,
    )

    assert manifest.format_version == 1
    assert manifest.n_compounds == 5  # 6 rows - 1 unparseable
    assert manifest.n_chunks == 3
    assert manifest.chunk_rows == [2, 2, 1]
    assert manifest.props_source == "enamine"
    assert manifest.columns == [
        "compound_id", "smiles", "reghash", "mw", "slogp", "hba", "hbd",
        "rotbonds", "tpsa",
    ]
    assert manifest.optional_columns == ["fsp3", "qed", "inchikey"]

    # manifest.json persisted and round-trips.
    on_disk = LibraryManifest.load(store / "manifest.json")
    assert on_disk == manifest
    on_disk.validate()  # must not raise

    # _raw/ cleaned up on success.
    assert not (store / "_raw").exists()

    # Chunks are zero-padded, 5-digit, and readable.
    chunk_paths = manifest.chunk_paths(store)
    assert [p.name for p in chunk_paths] == [
        "chunk_00000.parquet", "chunk_00001.parquet", "chunk_00002.parquet",
    ]
    all_ids = []
    for p in chunk_paths:
        table = pq.read_table(p)
        all_ids.extend(table.column("compound_id").to_pylist())
    assert sorted(all_ids) == ["ENA-1", "ENA-2", "ENA-3", "ENA-4", "ENA-5"]


def test_library_build_stage_is_resumable(tmp_path: Path) -> None:
    """Re-running with pre-existing chunks skips their (re-)build."""
    src = tmp_path / "diverse.cxsmiles"
    src.write_text(_REAL_HEADER + "\n" + "\n".join(_ROWS) + "\n")

    store = tmp_path / "store"
    settings = Settings()
    scheduler = LocalScheduler()

    library_build(
        scheduler, settings,
        source_files=[src], store_dir=store, chunk_size=2,
        build_command_prefix=_LIBRARY_BUILD_PREFIX,
    )
    first_mtime = (store / "chunk_00000.parquet").stat().st_mtime_ns

    # Re-run: the per-task command body should skip already-built chunks
    # (the _raw/ dir was deleted, so a real re-run would fail the split
    # step for missing shards were it not skipped first).
    manifest2 = library_build(
        scheduler, settings,
        source_files=[src], store_dir=store, chunk_size=2,
        build_command_prefix=_LIBRARY_BUILD_PREFIX,
    )
    second_mtime = (store / "chunk_00000.parquet").stat().st_mtime_ns
    assert first_mtime == second_mtime  # untouched: skipped as already-built
    assert manifest2.n_compounds == 5


def test_library_build_raises_on_empty_sources(tmp_path: Path) -> None:
    src = tmp_path / "empty.cxsmiles"
    src.write_text(_REAL_HEADER + "\n")  # header only, no data rows

    settings = Settings()
    scheduler = LocalScheduler()
    with pytest.raises(ValueError, match="no data rows"):
        library_build(
            scheduler, settings,
            source_files=[src], store_dir=tmp_path / "store", chunk_size=2,
            build_command_prefix=_LIBRARY_BUILD_PREFIX,
        )


def test_library_build_resolves_relative_store_dir_to_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relative --output must not leave ArrayJob.workdir relative.

    SlurmScheduler.submit_array() runs ``sbatch`` with ``cwd=workdir``,
    which re-resolves any relative script path argument against that *new*
    cwd -- silently producing a doubly-nested, nonexistent path and a bare
    sbatch exit 1 (library-build runs independently of any workspace, so
    there is no earlier `.resolve()` on a workspace root to save us).
    """
    class _RecordingScheduler(LocalScheduler):
        def __init__(self) -> None:
            super().__init__()
            self.submitted_workdirs: list[Path] = []

        def submit_array(self, job):  # type: ignore[override]
            self.submitted_workdirs.append(job.workdir)
            return super().submit_array(job)

    src = tmp_path / "diverse.cxsmiles"
    src.write_text(_REAL_HEADER + "\n" + "\n".join(_ROWS) + "\n")

    monkeypatch.chdir(tmp_path)
    settings = Settings()
    scheduler = _RecordingScheduler()

    manifest = library_build(
        scheduler, settings,
        source_files=[Path("diverse.cxsmiles")],
        store_dir=Path("relative_store"),  # relative, like a bare CLI --output
        chunk_size=2,
        build_command_prefix=_LIBRARY_BUILD_PREFIX,
    )

    assert scheduler.submitted_workdirs, "submit_array was never called"
    assert all(p.is_absolute() for p in scheduler.submitted_workdirs)
    assert manifest.n_compounds == 5
    assert (tmp_path / "relative_store" / "manifest.json").exists()
