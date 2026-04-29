"""Property filter — RDKit prop control over similarity-search hits.

Replaces legacy ``control.py``. Reads a SMI-like input file (one
``<smiles>§<title>`` per line, optionally gzipped) and a parameter file
listing the six property bounds in canonical order; writes a CSV
suitable for direct consumption by :mod:`spacehasten.remote.predict`.

The legacy script wrote a CSV with columns ``smiles,rawmol`` whose
filename ended in ``.csv.gz`` but which was written as plain text — and
the matching scheduler step then ran ``gunzip`` on it. This port fixes
that latent inconsistency:

- Input is auto-detected by ``.gz`` extension.
- Output is plain CSV with columns ``smiles,smilesid`` (where
  ``smilesid`` is the legacy ``<reghash>§<smiles>§<title>`` packed
  string), so :mod:`spacehasten.remote.predict` accepts it directly.
- Output filename is ``propoutput_<input-stem>.csv`` next to the input.

Property file format (12 lines, mirroring legacy)::

    mw_min
    mw_max
    slogp_min
    slogp_max
    hba_min
    hba_max
    hbd_min
    hbd_max
    rotbonds_min
    rotbonds_max
    tpsa_min
    tpsa_max
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO, cast

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Bounds:
    mw_min: float
    mw_max: float
    slogp_min: float
    slogp_max: float
    hba_min: int
    hba_max: int
    hbd_min: int
    hbd_max: int
    rotbonds_min: int
    rotbonds_max: int
    tpsa_min: float
    tpsa_max: float

    @classmethod
    def read(cls, path: Path) -> _Bounds:
        with path.open("rt", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh.readlines() if line.strip()]
        if len(lines) < 12:
            raise ValueError(
                f"{path}: expected 12 property bound lines, got {len(lines)}"
            )
        return cls(
            mw_min=float(lines[0]),
            mw_max=float(lines[1]),
            slogp_min=float(lines[2]),
            slogp_max=float(lines[3]),
            hba_min=int(lines[4]),
            hba_max=int(lines[5]),
            hbd_min=int(lines[6]),
            hbd_max=int(lines[7]),
            rotbonds_min=int(lines[8]),
            rotbonds_max=int(lines[9]),
            tpsa_min=float(lines[10]),
            tpsa_max=float(lines[11]),
        )


def _open_input(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return cast(IO[str], gzip.open(path, "rt", encoding="utf-8"))
    return path.open("rt", encoding="utf-8")


def _output_path_for(input_path: Path) -> Path:
    stem = input_path.name
    # Strip ``.gz`` then the next suffix (typically ``.smi``); whatever
    # remains becomes the propoutput stem so e.g.
    # ``control_x_cpu1.smi.gz`` -> ``propoutput_control_x_cpu1.csv``.
    if stem.endswith(".gz"):
        stem = stem[:-3]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    return input_path.parent / f"propoutput_{stem}.csv"


def filter_smiles(
    input_path: Path,
    bounds: _Bounds,
    output_path: Path,
) -> int:
    """Filter the input file by RDKit-computed properties.

    :returns: number of rows that passed the filter.
    """
    # Imports kept inside the function so the unit test can monkeypatch.
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, RegistrationHash, rdMolDescriptors

    n_pass = 0
    with _open_input(input_path) as r, output_path.open(
        "wt", encoding="utf-8", newline=""
    ) as w:
        writer = csv.writer(w, lineterminator="\n")
        writer.writerow(["smiles", "smilesid"])
        for raw in r:
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("§", 1)
            smiles = parts[0].strip()
            title_tail = parts[1] if len(parts) > 1 else ""
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            molwt = Descriptors.MolWt(mol)  # type: ignore[attr-defined]
            if molwt < bounds.mw_min or molwt > bounds.mw_max:
                continue
            logp = Crippen.MolLogP(mol)  # type: ignore[attr-defined]
            if logp < bounds.slogp_min or logp > bounds.slogp_max:
                continue
            hba = rdMolDescriptors.CalcNumHBA(mol)
            if hba < bounds.hba_min or hba > bounds.hba_max:
                continue
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            if hbd < bounds.hbd_min or hbd > bounds.hbd_max:
                continue
            rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            if rotbonds < bounds.rotbonds_min or rotbonds > bounds.rotbonds_max:
                continue
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            if tpsa < bounds.tpsa_min or tpsa > bounds.tpsa_max:
                continue
            reghash = RegistrationHash.GetMolLayers(mol)[
                RegistrationHash.HashLayer.TAUTOMER_HASH
            ]
            # smilesid packs reghash + original line so downstream
            # ingestion can recover all three fields with a single split.
            smilesid = f"{reghash}§{smiles}§{title_tail}" if title_tail else f"{reghash}§{smiles}"
            writer.writerow([smiles, smilesid])
            n_pass += 1
    return n_pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a SMI(.gz) file by RDKit properties; output a CSV with"
            " columns smiles,smilesid suitable for chemprop prediction."
        )
    )
    parser.add_argument("input", help="Input SMI file (optionally gzipped)")
    parser.add_argument("params", help="Property bounds file (12 lines)")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. Defaults to propoutput_<input-stem>.csv next"
            " to the input."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("input not found: %s", input_path)
        return 1
    params_path = Path(args.params)
    if not params_path.exists():
        logger.error("params file not found: %s", params_path)
        return 1
    output_path = (
        Path(args.output) if args.output is not None else _output_path_for(input_path)
    )
    bounds = _Bounds.read(params_path)
    n = filter_smiles(input_path, bounds, output_path)
    logger.info("filtered %d rows -> %s", n, output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    sys.exit(main())
