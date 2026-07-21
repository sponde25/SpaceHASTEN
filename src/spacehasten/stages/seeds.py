"""Seeds stage — import the initial set of compounds into a fresh ``.dbsh``.

Replaces legacy ``importseeds_functions.import_seeds``. Mirrors the legacy
flow:

    1. Store property ranges (legacy TEXT format).
    2. Read seeds from a ``.smi`` (undocked) or ``.csv`` (docked) file.
    3. Compute tautomer hashes in parallel via ``mp.Pool`` over
       :func:`core.molecules.tautomer_hash`.
    4. Insert rows via ``db.insert_seed_*``.

The database and schema are created at ``init`` time. Docking parameters
(Glide ``.in`` template and grid ``.zip``) are stored at ``init`` and are
**not** required here.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Iterable, Iterator
from pathlib import Path

from spacehasten.config.properties import PropertyRanges as TypedPropertyRanges
from spacehasten.core.db import Database
from spacehasten.core.db import PropertyRanges as DbPropertyRanges
from spacehasten.core.molecules import tautomer_hash

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def typed_to_db_props(props: TypedPropertyRanges) -> DbPropertyRanges:
    """Convert the typed pydantic :class:`PropertyRanges` to the DB form."""
    return DbPropertyRanges(
        mw=(str(props.mw.min), str(props.mw.max)),
        slogp=(str(props.slogp.min), str(props.slogp.max)),
        hba=(str(props.hba.min), str(props.hba.max)),
        hbd=(str(props.hbd.min), str(props.hbd.max)),
        rotbonds=(str(props.rotbonds.min), str(props.rotbonds.max)),
        tpsa=(str(props.tpsa.min), str(props.tpsa.max)),
    )


def typed_smarts_to_db(props: TypedPropertyRanges) -> list[tuple[str, str]]:
    """Extract SMARTS include/exclude pairs from a typed :class:`PropertyRanges`."""
    return [("include", s) for s in props.smarts_include] + [
        ("exclude", s) for s in props.smarts_exclude
    ]


# Worker function for ``mp.Pool``. Must be top-level (picklable).
SeedRow = tuple[str, str, str | None]  # (smiles, smilesid, dock_score_str)
HashedSeed = tuple[str, str, str, str | None]  # (reghash, smiles, smilesid, score)


def _hash_seed(row: SeedRow) -> HashedSeed | None:
    smiles, smilesid, score = row
    h = tautomer_hash(smiles)
    if h is None:
        return None
    return (h, smiles, smilesid, score)


def _read_smi(path: Path) -> Iterator[SeedRow]:
    """Yield ``(smiles, smilesid, None)`` from a whitespace-separated SMI."""
    with path.open("rt", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            yield (parts[0], parts[1].strip(), None)


def _read_csv(
    path: Path,
    *,
    smiles_col: str,
    title_col: str,
    score_col: str,
) -> Iterator[SeedRow]:
    """Yield ``(smiles, smilesid, dock_score_str)`` from a docked CSV."""
    import csv

    with path.open("rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV")
        missing = [c for c in (smiles_col, title_col, score_col) if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing columns {missing!r} (have {reader.fieldnames!r})")
        for row in reader:
            smi = (row.get(smiles_col) or "").strip()
            title = (row.get(title_col) or "").strip()
            score = (row.get(score_col) or "").strip()
            if not smi or not title or not score:
                continue
            yield (smi, title, score)


def _hash_in_parallel(rows: Iterable[SeedRow], *, processes: int) -> list[HashedSeed]:
    """Map :func:`_hash_seed` over ``rows`` using a process pool."""
    rows_list = list(rows)
    if not rows_list:
        return []
    if processes <= 1:
        results = [_hash_seed(r) for r in rows_list]
    else:
        with mp.Pool(processes) as pool:
            results = list(pool.imap(_hash_seed, rows_list))
    return [r for r in results if r is not None]


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #


def import_seeds(
    db: Database,
    *,
    smi_path: Path | None = None,
    csv_path: Path | None = None,
    props: TypedPropertyRanges,
    smiles_col: str = "SMILES",
    title_col: str = "title",
    score_col: str = "r_i_docking_score",
    processes: int | None = None,
) -> int:
    """Import seeds into an existing database.

    Exactly one of ``smi_path`` or ``csv_path`` must be provided. The CSV
    path is the *docked seeds* mode (legacy: pre-existing dock scores);
    the SMI path is the *undocked seeds* mode.

    The database, schema, and docking parameters (Glide ``.in`` and grid
    ``.zip``) must already exist — created by ``spacehasten init``.

    :param props: property-filter ranges to write into the ``properties``
        table.
    :param smiles_col, title_col, score_col: CSV column names (defaults
        match legacy ``cfg.FIELD_*_DEFAULT``).
    :param processes: worker pool size for tautomer hashing. ``None``
        defaults to ``mp.cpu_count()``.
    :returns: number of rows successfully inserted.
    :raises ValueError: on bad/missing arguments or if no seeds could be
        parsed.
    """
    if (smi_path is None) == (csv_path is None):
        raise ValueError("provide exactly one of smi_path or csv_path")

    seed_path = smi_path if smi_path is not None else csv_path
    assert seed_path is not None  # for type checker
    is_csv = csv_path is not None

    # 1. Properties (numerical ranges + SMARTS patterns).
    db.replace_properties(typed_to_db_props(props))
    db.replace_smarts_filters(typed_smarts_to_db(props))

    # 2. Read seeds.
    if is_csv:
        rows: Iterable[SeedRow] = _read_csv(
            seed_path,
            smiles_col=smiles_col,
            title_col=title_col,
            score_col=score_col,
        )
    else:
        rows = _read_smi(seed_path)

    # 3. Parallel hash.
    n_workers = processes if processes is not None else mp.cpu_count()
    hashed = _hash_in_parallel(rows, processes=max(1, n_workers))
    if not hashed:
        db.commit()
        raise ValueError(f"no parseable seeds in {seed_path}")

    # 4. Insert (skip duplicates by reghash).
    n_inserted = 0
    n_skipped = 0
    for reghash, smiles, smilesid, score in hashed:
        if db.reghash_exists(reghash):
            n_skipped += 1
            continue
        if is_csv:
            assert score is not None
            try:
                score_f = float(score)
            except ValueError:
                logger.warning("skipping seed with non-numeric score: %s", score)
                continue
            db.insert_seed_docked(reghash, smiles, smilesid, score_f)
        else:
            db.insert_seed_undocked(reghash, smiles, smilesid)
        n_inserted += 1
    db.commit()
    if n_skipped:
        logger.info("Skipped %d seeds already in database", n_skipped)
    logger.info("Imported %d seed rows from %s", n_inserted, seed_path)

    return n_inserted


__all__ = ["import_seeds", "typed_to_db_props", "typed_smarts_to_db"]
