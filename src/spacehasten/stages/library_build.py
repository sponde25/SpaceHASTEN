"""Library-build stage — Enamine ``.cxsmiles`` diverse subset → Parquet store.

One-time (per library) conversion of an Enamine REAL diverse ``.cxsmiles``
(optionally ``.bz2``/``.gz``) subset into a chunked, canonicalized Parquet
store with precomputed PC properties + tautomer reghash, plus a
self-describing ``manifest.json``. The store is target-agnostic (SMILES +
properties only) so it is built once and reused across every campaign
(see ``docs/plan-library-screening.md`` §3, §5.1).

Layout::

    <store_dir>/
        manifest.json
        chunk_00000.parquet
        chunk_00001.parquet
        ...
        _raw/                       # deleted on success
            shard_1.smi ...

Algorithm:

1. Resolve the smiles/id/property column positions from the header of the
   *first* source file, by name (case-insensitive); ``column_map``
   overrides.
2. Stream-split every source (headerless output) into
   ``_raw/shard_<i>.smi`` chunks of ``chunk_size`` lines.
3. Submit one array task per shard running
   :mod:`spacehasten.remote.library_build`, writing
   ``chunk_<i-1 zero-padded>.parquet``. Resumable: each task skips its
   output chunk if it already exists.
4. On success, delete ``_raw/``, count rows per chunk via Parquet
   metadata (no full load), and write ``manifest.json``.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Final, Literal, cast

from spacehasten.config.settings import Settings
from spacehasten.scheduler.base import ArrayJob, Scheduler

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Schema constants (shared with library_screen.py)                           #
# --------------------------------------------------------------------------- #

#: The nine columns every library chunk must carry (plan §3.2).
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "compound_id",
    "smiles",
    "reghash",
    "mw",
    "slogp",
    "hba",
    "hbd",
    "rotbonds",
    "tpsa",
)
#: Best-effort extra columns, present only when the source provides them.
OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ("fsp3", "qed", "inchikey")

#: Header column names (case-insensitive) recognised for each canonical field.
_HEADER_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "smiles": ("smiles",),
    "id": ("id",),
    "mw": ("mw",),
    "slogp": ("slogp",),
    "hba": ("hba",),
    "hbd": ("hbd",),
    "rotbonds": ("rotbonds",),
    "tpsa": ("tpsa",),
    "fsp3": ("fsp3",),
    "qed": ("qed",),
    "inchikey": ("inchikey", "inchi_key"),
}
_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "smiles", "id", "mw", "slogp", "hba", "hbd", "rotbonds", "tpsa",
)


# --------------------------------------------------------------------------- #
# LibraryManifest                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class LibraryManifest:
    """Self-describing metadata for a library store (plan §3.3)."""

    format_version: int
    source_files: list[str]
    n_compounds: int
    n_chunks: int
    chunk_glob: str
    chunk_rows: list[int]
    columns: list[str]
    optional_columns: list[str] = field(default_factory=list)
    chunk_size: int = 2_000_000
    compression: str = "zstd"
    props_source: Literal["enamine", "rdkit"] = "enamine"
    rdkit_version: str = "unknown"
    reghash_algo: str = "RegistrationHash.TAUTOMER_HASH"
    canonicalization: str = "rdkit-canonical"
    built_at: str = ""

    def save(self, path: Path) -> None:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LibraryManifest:
        import json

        with Path(path).open("rt", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**data)

    def validate(self) -> None:
        """Raise :class:`ValueError` if the manifest is unusable.

        Checks ``format_version == 1`` and that all :data:`REQUIRED_COLUMNS`
        are present in ``self.columns``.
        """
        if self.format_version != 1:
            raise ValueError(
                f"unsupported library manifest format_version {self.format_version}"
                " (expected 1)"
            )
        missing = [c for c in REQUIRED_COLUMNS if c not in self.columns]
        if missing:
            raise ValueError(f"library manifest missing required columns: {missing}")

    def chunk_paths(self, store_dir: Path) -> list[Path]:
        """Enumerate this library's chunk files, sorted."""
        return sorted(Path(store_dir).glob(self.chunk_glob))


# --------------------------------------------------------------------------- #
# Header resolution & source splitting                                       #
# --------------------------------------------------------------------------- #


def _split_header_line(line: str) -> list[str]:
    stripped = line.rstrip("\n").rstrip("\r")
    return stripped.split("\t") if "\t" in stripped else stripped.split()


def _resolve_columns(
    header_fields: Sequence[str] | None,
    overrides: dict[str, str] | None = None,
) -> dict[str, int]:
    """Resolve canonical field names to 0-based column indices.

    ``overrides`` values may be a numeric string (explicit 0-based index)
    or a column name to look up case-insensitively in ``header_fields``.
    Raises :class:`ValueError` if a required field cannot be resolved.
    """
    lower_fields = [f.strip().lower() for f in header_fields] if header_fields else None
    overrides = overrides or {}
    resolved: dict[str, int] = {}

    for canonical, aliases in _HEADER_ALIASES.items():
        if canonical in overrides:
            value = overrides[canonical]
            if value.lstrip("-").isdigit():
                resolved[canonical] = int(value)
                continue
            if lower_fields and value.lower() in lower_fields:
                resolved[canonical] = lower_fields.index(value.lower())
                continue
            raise ValueError(f"cannot resolve column override {canonical}={value!r}")
        elif lower_fields:
            for alias in aliases:
                if alias in lower_fields:
                    resolved[canonical] = lower_fields.index(alias)
                    break

    missing = [f for f in _REQUIRED_FIELDS if f not in resolved]
    if missing:
        raise ValueError(
            f"could not resolve required columns {missing} from header"
            f" {header_fields!r}; pass --smiles-col/--id-col/--prop-cols overrides"
        )
    return resolved


def _open_source(path: Path) -> IO[str]:
    """Open a ``.cxsmiles``/``.smi`` source, transparently decompressing."""
    if path.suffix == ".bz2":
        import bz2

        return cast(IO[str], bz2.open(path, "rt", encoding="utf-8"))
    if path.suffix == ".gz":
        import gzip

        return cast(IO[str], gzip.open(path, "rt", encoding="utf-8"))
    return path.open("rt", encoding="utf-8")


def _split_sources(
    source_files: Sequence[Path],
    raw_dir: Path,
    chunk_size: int,
    overrides: dict[str, str] | None = None,
) -> tuple[int, dict[str, int]]:
    """Stream every source into headerless ``shard_<i>.smi`` files.

    Column mapping is resolved from the header of the *first* source file
    (or entirely from ``overrides`` when given); every source's own header
    line is stripped before its data rows are written out.

    :returns: ``(n_shards, resolved_columns)``.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    for stale in raw_dir.glob("shard_*.smi"):
        stale.unlink()

    resolved_columns: dict[str, int] | None = None
    shard_idx = 0
    in_shard = 0
    out_fh: IO[str] | None = None

    def _next_shard() -> IO[str]:
        nonlocal shard_idx, in_shard
        shard_idx += 1
        in_shard = 0
        return (raw_dir / f"shard_{shard_idx}.smi").open("wt", encoding="utf-8")

    for i, src in enumerate(source_files):
        with _open_source(src) as fh:
            header_line = fh.readline()
            if i == 0:
                header_fields = _split_header_line(header_line)
                resolved_columns = _resolve_columns(header_fields, overrides)
            for line in fh:
                if not line.strip():
                    continue
                if out_fh is None or in_shard >= chunk_size:
                    if out_fh is not None:
                        out_fh.close()
                    out_fh = _next_shard()
                out_fh.write(line)
                in_shard += 1
    if out_fh is not None:
        out_fh.close()

    if resolved_columns is None:
        # No source files had data to establish a header; still try to
        # resolve purely from overrides (all-numeric case).
        resolved_columns = _resolve_columns(None, overrides)
    return shard_idx, resolved_columns


# --------------------------------------------------------------------------- #
# Scheduler command                                                           #
# --------------------------------------------------------------------------- #


_DEFAULT_BUILD_COMMAND: tuple[str, ...] = ("python3", "-m", "spacehasten.remote.library_build")


def _default_build_command(settings: Settings) -> tuple[str, ...]:
    try:
        return ("python3", str(settings.remote_script_path("library_build")))
    except ValueError:
        return _DEFAULT_BUILD_COMMAND


def _build_build_command(
    store_dir: Path,
    column_map: dict[str, int],
    recompute_props: bool,
    command_prefix: Sequence[str],
) -> str:
    """Render the per-task bash body: shard → chunk, resumable & idempotent."""
    prefix = " ".join(command_prefix)
    flag_by_field = {
        "mw": "--mw-col",
        "slogp": "--slogp-col",
        "hba": "--hba-col",
        "hbd": "--hbd-col",
        "rotbonds": "--rotbonds-col",
        "tpsa": "--tpsa-col",
        "fsp3": "--fsp3-col",
        "qed": "--qed-col",
        "inchikey": "--inchikey-col",
    }
    prop_flags = " ".join(
        f"{flag} {column_map[field]}"
        for field, flag in flag_by_field.items()
        if field in column_map
    )
    recompute = " --recompute-props" if recompute_props else ""
    return (
        'printf -v CHUNK_IDX "%05d" $((TASK_ID - 1))\n'
        f'OUT_PATH="{store_dir}/chunk_${{CHUNK_IDX}}.parquet"\n'
        'if [ -f "$OUT_PATH" ]; then\n'
        '  echo "[task ${TASK_ID}] $OUT_PATH already exists, skipping"\n'
        '  exit 0\n'
        'fi\n'
        f'{prefix} "{store_dir}/_raw/shard_${{TASK_ID}}.smi" "$OUT_PATH" '
        f'--smiles-col {column_map["smiles"]} --id-col {column_map["id"]}'
        f'{" " + prop_flags if prop_flags else ""}{recompute}\n'
        'echo "[task ${TASK_ID}] wrote $OUT_PATH"\n'
    )


# --------------------------------------------------------------------------- #
# Top-level orchestration                                                     #
# --------------------------------------------------------------------------- #


def library_build(
    scheduler: Scheduler,
    settings: Settings,
    *,
    source_files: Sequence[Path],
    store_dir: Path,
    chunk_size: int = 2_000_000,
    recompute_props: bool = False,
    column_map: dict[str, str] | None = None,
    max_concurrent: int | None = None,
    build_command_prefix: Sequence[str] | None = None,
) -> LibraryManifest:
    """Convert ``source_files`` into a chunked Parquet library store.

    :param source_files: one or more Enamine ``.cxsmiles``/``.smi``
        (optionally ``.bz2``/``.gz``) files sharing the same header.
    :param store_dir: output directory (``manifest.json`` + chunks).
    :param chunk_size: rows per shard/chunk; one array task per shard.
    :param recompute_props: force RDKit computation of the six PC
        descriptors instead of parsing them from the source columns
        (plan §D5).
    :param column_map: optional overrides ``{canonical_field: name_or_index}``
        for headerless inputs or nonstandard headers.
    :param max_concurrent: max concurrent array tasks. Default: all shards
        at once.
    :param build_command_prefix: override the command used to launch
        ``remote.library_build`` (used by tests with stub scripts).
    :returns: the written :class:`LibraryManifest`.
    :raises ValueError: on empty ``source_files``/no data rows, or
        unresolvable columns.
    :raises RuntimeError: on scheduler failure, or if no chunks resulted.
    """
    if not source_files:
        raise ValueError("source_files must be non-empty")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    # Resolve to an absolute path: library-build runs independently of any
    # workspace, so a relative --output would leave ArrayJob.workdir relative
    # too. SlurmScheduler.submit_array() runs sbatch with cwd=workdir, which
    # re-resolves any relative script path against that *new* cwd -- silently
    # producing a doubly-nested, nonexistent path and a bare sbatch exit 1.
    store_dir = Path(store_dir).resolve()
    store_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = store_dir / "_raw"

    n_shards, resolved_columns = _split_sources(
        [Path(p) for p in source_files], raw_dir, chunk_size, overrides=column_map,
    )
    if n_shards == 0:
        raise ValueError(f"no data rows found in source_files: {list(source_files)}")

    logger.info(
        "Split %d source file(s) into %d shard(s) of <=%d lines under %s",
        len(source_files), n_shards, chunk_size, raw_dir,
    )

    # Uses the chemprop env (RDKit + pyarrow) rather than the clustering env:
    # RDKit is already present there for training/prediction, and pyarrow is
    # the only extra dependency needed for parquet chunk output.
    env_setup = [
        line
        for line in (
            settings.general.prepare_anaconda,
            settings.general.activate_chemprop,
        )
        if line
    ]
    command_prefix = (
        build_command_prefix
        if build_command_prefix is not None
        else _default_build_command(settings)
    )
    command = _build_build_command(store_dir, resolved_columns, recompute_props, command_prefix)

    cpus = int(settings.general.cpu_count_library or 1)
    job = ArrayJob(
        name="library_build",
        workdir=raw_dir,
        array_size=n_shards,
        max_concurrent=max_concurrent if max_concurrent is not None else n_shards,
        cpus_per_task=max(1, cpus),
        env_setup=env_setup,
        command_template=command,
    )
    handle = scheduler.submit_array(job)
    logger.info("Submitted library-build job %s (%d shards)", handle.job_id, n_shards)
    result = scheduler.wait(handle)
    if not result.success:
        from spacehasten.scheduler.diagnostics import tail_logs
        raise RuntimeError(
            f"library-build job {handle.job_id} failed; failed task indices: "
            f"{result.failed_indices}\n"
            f"--- tail of task logs ---\n{tail_logs(handle)}"
        )

    import pyarrow.parquet as pq

    chunk_paths = sorted(store_dir.glob("chunk_*.parquet"))
    if not chunk_paths:
        raise RuntimeError(f"library-build produced no chunk files under {store_dir}")
    chunk_rows = [pq.ParquetFile(p).metadata.num_rows for p in chunk_paths]
    n_compounds = sum(chunk_rows)

    shutil.rmtree(raw_dir, ignore_errors=True)

    try:
        from rdkit import rdBase

        rdkit_version = rdBase.rdkitVersion
    except Exception:  # pragma: no cover - defensive
        rdkit_version = "unknown"

    manifest = LibraryManifest(
        format_version=1,
        source_files=[str(Path(p)) for p in source_files],
        n_compounds=n_compounds,
        n_chunks=len(chunk_rows),
        chunk_glob="chunk_*.parquet",
        chunk_rows=chunk_rows,
        columns=list(REQUIRED_COLUMNS),
        optional_columns=list(OPTIONAL_COLUMNS),
        chunk_size=chunk_size,
        compression="zstd",
        props_source="rdkit" if recompute_props else "enamine",
        rdkit_version=rdkit_version,
        reghash_algo="RegistrationHash.TAUTOMER_HASH",
        canonicalization="rdkit-canonical",
        built_at=datetime.now(tz=UTC).isoformat(),
    )
    manifest.save(store_dir / "manifest.json")
    logger.info(
        "Wrote manifest: %d compounds across %d chunks -> %s",
        n_compounds, len(chunk_rows), store_dir / "manifest.json",
    )
    return manifest


__all__ = [
    "LibraryManifest",
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "library_build",
]
