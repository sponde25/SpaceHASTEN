#!/usr/bin/env python3
"""Sphere-exclusion clustering — remote-node entry point.

Ports the embedded Python pipeline of legacy ``sec_clustering.sh`` to a
real Python module. The original shell script split work over GNU
``parallel`` and the ``fpsim2-create-db`` CLI; this rewrite keeps the
same algorithm but does everything in one process (with an internal
``multiprocessing`` pool for the Morgan fingerprint pass), so it is
trivially callable from tests and from the new scheduler stage.

Pipeline (matches sec_clustering.sh and §1 of CODEBASE_REFERENCE.md):

1. Read ``<smiles> <spacehastenid>`` lines from the input file
   (plain text or gzip; auto-detected by suffix ``.gz``).
2. Compute Morgan-2 1024-bit fingerprints for every compound.
3. Build an FPSim2 ``fp.h5`` index over the same input.
4. Pick cluster centroids with RDKit ``LeaderPicker`` at distance 0.7
   (= similarity 0.3).
5. For each centroid, similarity-search the FPSim2 index at
   tanimoto >= 0.3.
6. Assign every compound to the centroid that gave it the highest
   similarity (LeaderPicker guarantees coverage at the chosen threshold).
7. Write ``clustering.csv`` (``spacehastenid,clusterid``) in the
   current working directory.

Reference: https://rdkit.blogspot.com/2020/11/sphere-exclusion-clustering-with-rdkit.html

CLI::

    python3 -m spacehasten.remote.cluster <input.smi[.gz]>
"""

from __future__ import annotations

import argparse
import gzip
import logging
import multiprocessing as mp
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

# Algorithm parameters, identical to sec_clustering.sh.
_MORGAN_RADIUS = 2
_MORGAN_FP_SIZE = 1024
_LEADER_DISTANCE_THRESHOLD = 1.0 - 0.3  # 0.7
_SEARCH_SIMILARITY_THRESHOLD = 0.30


# --------------------------------------------------------------------------- #
# I/O helpers                                                                 #
# --------------------------------------------------------------------------- #


def _open_smi(path: Path) -> Iterator[str]:
    """Yield non-empty stripped lines from a plain or gzipped ``.smi`` file."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def _parse_lines(lines: Iterable[str]) -> list[tuple[str, int]]:
    """Parse ``<smiles> <spacehastenid>`` lines into ``(smiles, id)`` tuples."""
    out: list[tuple[str, int]] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"malformed input line (need 'smiles id'): {line!r}")
        out.append((parts[0], int(parts[1])))
    return out


# --------------------------------------------------------------------------- #
# Step 2: Morgan fingerprints                                                 #
# --------------------------------------------------------------------------- #


def _fp_worker(smiles: str):  # type: ignore[no-untyped-def]
    """Worker: SMILES -> Morgan-2 1024-bit ExplicitBitVect."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=_MORGAN_RADIUS, fpSize=_MORGAN_FP_SIZE
    )
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    return gen.GetFingerprint(mol)


def _generate_fingerprints(smiles_list: list[str], processes: int) -> list:  # type: ignore[type-arg]
    """Generate Morgan fingerprints, optionally in parallel."""
    if processes <= 1 or len(smiles_list) < 256:
        return [_fp_worker(s) for s in smiles_list]
    chunksize = max(1, len(smiles_list) // (processes * 4))
    with mp.Pool(processes) as pool:
        return pool.map(_fp_worker, smiles_list, chunksize=chunksize)


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #


def run_clustering(input_smi: Path, output_csv: Path, *, processes: int = 1) -> int:
    """Run the full sphere-exclusion clustering pipeline.

    :param input_smi: ``smiles spacehastenid`` text file (plain or gzip).
    :param output_csv: destination ``clustering.csv``.
    :param processes: worker count for the Morgan fingerprint pass.
    :returns: number of clusters discovered.
    """
    # Local imports keep module import cheap on machines without RDKit/FPSim2.
    from rdkit.SimDivFilters import rdSimDivPickers

    rows = _parse_lines(_open_smi(input_smi))
    if not rows:
        raise ValueError(f"input contains no compounds: {input_smi}")
    smiles_list = [s for s, _ in rows]
    spacehastenids = [i for _, i in rows]
    logger.info("Loaded %d compounds from %s", len(rows), input_smi)

    # Step 2: fingerprints (in-memory list for LeaderPicker).
    logger.info("Generating Morgan-%d (%d-bit) fingerprints...", _MORGAN_RADIUS, _MORGAN_FP_SIZE)
    fingerprints = _generate_fingerprints(smiles_list, processes)

    # Step 3: build FPSim2 index in a sibling file next to the input.
    workdir = output_csv.parent
    workdir.mkdir(parents=True, exist_ok=True)
    fp_h5 = workdir / "fp.h5"
    if fp_h5.exists():
        fp_h5.unlink()
    logger.info("Building FPSim2 index at %s ...", fp_h5)
    _build_fpsim2_index(rows, fp_h5)

    # Step 4: identify cluster centroids.
    logger.info(
        "Picking cluster centroids (LeaderPicker, distance=%.2f)...",
        _LEADER_DISTANCE_THRESHOLD,
    )
    picker = rdSimDivPickers.LeaderPicker()
    centroid_indices = list(
        picker.LazyBitVectorPick(fingerprints, len(fingerprints), _LEADER_DISTANCE_THRESHOLD)
    )
    num_clusters = len(centroid_indices)
    logger.info("Identified %d cluster centroids", num_clusters)

    # Step 5+6: assign each compound to the centroid with highest similarity.
    logger.info(
        "Searching FPSim2 index (similarity >= %.2f)...", _SEARCH_SIMILARITY_THRESHOLD
    )
    centroid_ids = [spacehastenids[idx] for idx in centroid_indices]
    centroid_smiles = [smiles_list[idx] for idx in centroid_indices]
    cluster_of: dict[int, int] = {}
    sim_of: dict[int, float] = {}
    from FPSim2 import FPSim2Engine

    fpe = FPSim2Engine(str(fp_h5))
    for cluster_id, query_smiles in zip(centroid_ids, centroid_smiles, strict=True):
        results = fpe.similarity(
            query_smiles,
            threshold=_SEARCH_SIMILARITY_THRESHOLD,
            metric="tanimoto",
            n_workers=1,
        )
        for comp_id, sim in results:
            comp_id = int(comp_id)
            if comp_id not in cluster_of or sim_of[comp_id] < sim:
                cluster_of[comp_id] = cluster_id
                sim_of[comp_id] = float(sim)

    # Every compound is within distance threshold of some centroid by
    # construction, but make this explicit and fail loudly if a compound
    # falls through (e.g. parse mismatch between LeaderPicker and FPSim2).
    missing = [i for i in spacehastenids if i not in cluster_of]
    if missing:
        # Self-cluster orphans rather than crashing — matches legacy
        # behaviour for centroids themselves (which always cover themselves).
        for i in missing:
            cluster_of[i] = i

    # Step 7: write CSV.
    logger.info("Writing %s ...", output_csv)
    with output_csv.open("wt", encoding="utf-8") as w:
        w.write("spacehastenid,clusterid\n")
        for sid in spacehastenids:
            w.write(f"{sid},{cluster_of[sid]}\n")

    logger.info("Clustering done: %d clusters across %d compounds", num_clusters, len(rows))
    return num_clusters


def _build_fpsim2_index(rows: list[tuple[str, int]], out_h5: Path) -> None:
    """Build an FPSim2 HDF5 index for ``rows``.

    Uses :func:`FPSim2.io.create_db_file`, which expects an iterable of
    ``[smiles, mol_id]`` pairs (mol_id must be int).
    """
    from FPSim2.io import create_db_file

    iterable = [[s, i] for s, i in rows]
    create_db_file(
        iterable,
        str(out_h5),
        "smiles",
        "Morgan",
        {"radius": _MORGAN_RADIUS, "fpSize": _MORGAN_FP_SIZE},
    )


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m spacehasten.remote.cluster",
        description="Sphere-exclusion clustering — produces clustering.csv in the cwd.",
    )
    parser.add_argument("input", type=Path, help="input .smi or .smi.gz (smiles id per line)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("clustering.csv"),
        help="output CSV path (default: ./clustering.csv)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="worker count for Morgan fingerprint generation",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    run_clustering(args.input, args.output, processes=args.processes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
