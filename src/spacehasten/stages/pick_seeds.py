"""Pick seeds — sample and canonicalize molecules from a large seed collection.

Replaces legacy ``gui_thread_pickseeds`` in ``gui.py``. Reads a
bz2-compressed Enamine REAL or FreedomSpace cxsmiles file, randomly
samples N entries, canonicalizes each SMILES via RDKit in a parallel
worker pool, and writes the result as a ``.smi`` file
(one ``<smiles> <id>`` per line).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)


def _cxsmi2smi(cxsmiles_with_id: str) -> str | None:
    """Canonicalize a ``<cxsmiles>§<id>`` string to ``<smiles> <id>\\n``.

    Returns ``None`` if RDKit cannot parse the SMILES.
    """
    parts = cxsmiles_with_id.split("§")
    if len(parts) < 2:
        return None
    mol = Chem.MolFromSmiles(parts[0])
    if mol is None:
        return None
    return Chem.MolToSmiles(mol) + " " + parts[1] + "\n"


def _load_seeds_dataframe(path: Path) -> pd.Series:
    """Load a seed collection and return a Series of ``<smiles>§<id>`` strings.

    Supports Enamine REAL (bz2, tab-separated, columns: smiles, idnumber, Type)
    and FreedomSpace (tab-separated, columns: SMILES, ID, ...) formats.
    Detection is based on filename heuristics matching the legacy logic.
    """
    name = path.name.upper()
    if "REAL" in name:
        logger.info("Loading seeds from Enamine REAL tab-separated cxsmiles...")
        df = pd.read_csv(path, compression="bz2", sep="\t")
        return df["smiles"] + "§" + df["idnumber"]
    elif "FREEDOM" in name:
        logger.info("Loading seeds from FreedomSpace smiles...")
        df = pd.read_csv(path, sep="\t")
        return df["SMILES"] + "§" + df["ID"]
    else:
        # Try generic two-column format: first col = SMILES, second = ID
        logger.info("Unknown format, trying generic tab-separated (smiles, id)...")
        df = pd.read_csv(path, sep="\t")
        cols = df.columns.tolist()
        if len(cols) < 2:
            raise ValueError(
                f"Cannot parse seed file {path}: expected at least 2 tab-separated columns"
            )
        return df.iloc[:, 0].astype(str) + "§" + df.iloc[:, 1].astype(str)


def pick_seeds(
    seeds_file: Path,
    output: Path,
    n_seeds: int = 1_000_000,
    n_cores: int = 4,
) -> int:
    """Sample and canonicalize seeds from a large collection.

    :param seeds_file: Path to the seed collection (bz2/tsv).
    :param output: Path to write the output ``.smi`` file.
    :param n_seeds: Number of seeds to randomly sample.
    :param n_cores: Number of parallel worker processes for RDKit.
    :returns: Number of valid seeds written.
    :raises FileExistsError: if ``output`` already exists.
    """
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[pick-seeds] Loading seed collection: {seeds_file}", flush=True)
    all_entries = _load_seeds_dataframe(seeds_file)
    actual_n = min(n_seeds, len(all_entries))
    print(f"[pick-seeds] Loaded {len(all_entries)} entries, sampling {actual_n}", flush=True)
    logger.info("Sampling %d seeds from %d total entries", actual_n, len(all_entries))

    sampled = all_entries.sample(n=actual_n).tolist()

    print(f"[pick-seeds] Canonicalizing {actual_n} SMILES with {n_cores} cores...", flush=True)
    logger.info("Canonicalizing %d SMILES with %d cores...", actual_n, n_cores)
    with mp.Pool(n_cores) as pool:
        results = pool.map(_cxsmi2smi, sampled)

    valid = [r for r in results if r is not None]
    n_failed = actual_n - len(valid)
    print(f"[pick-seeds] Writing {len(valid)} valid seeds to {output} "
          f"({n_failed} failed RDKit parse)", flush=True)
    logger.info("Writing %d valid seeds to %s (%d failed RDKit parse)",
                len(valid), output, n_failed)

    with output.open("wt", encoding="utf-8") as fh:
        for line in valid:
            fh.write(line)

    return len(valid)


__all__ = ["pick_seeds"]
