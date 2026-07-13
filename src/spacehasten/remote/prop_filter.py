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

Optional SMARTS file format (``--smarts``)::

    include:<SMARTS>   # molecule must match at least one include pattern
    exclude:<SMARTS>   # molecule must not match any exclude pattern

Lines starting with ``#`` are treated as comments and ignored.  If no
``--smarts`` file is provided (or the file is empty), SMARTS filtering
is skipped entirely.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import sys
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class _SmartsBounds:
    """Compiled SMARTS patterns for substructure filtering.

    ``include``: molecule must match at least one pattern (empty = no constraint).
    ``exclude``: molecule must not match any pattern (empty = no constraint).
    """

    include: list[object] = field(default_factory=list)
    exclude: list[object] = field(default_factory=list)

    @classmethod
    def read(cls, path: Path) -> _SmartsBounds:
        """Load and compile SMARTS from *path*.

        File format: one ``<mode>:<smarts>`` per line.  Lines starting with
        ``#`` or blank lines are ignored.  Raises ``ValueError`` for
        unrecognised modes or patterns that RDKit cannot parse.
        """
        from rdkit import Chem

        include: list[object] = []
        exclude: list[object] = []
        with path.open("rt", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    raise ValueError(
                        f"{path}:{lineno}: expected 'include:<smarts>' or "
                        f"'exclude:<smarts>', got {line!r}"
                    )
                mode, _, pattern = line.partition(":")
                mode = mode.strip().lower()
                if mode not in ("include", "exclude"):
                    raise ValueError(
                        f"{path}:{lineno}: unknown mode {mode!r} "
                        "(expected 'include' or 'exclude')"
                    )
                mol = Chem.MolFromSmarts(pattern)
                if mol is None:
                    raise ValueError(
                        f"{path}:{lineno}: RDKit could not parse SMARTS {pattern!r}"
                    )
                if mode == "include":
                    include.append(mol)
                else:
                    exclude.append(mol)
        return cls(include=include, exclude=exclude)

    @property
    def active(self) -> bool:
        return bool(self.include or self.exclude)


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
    smarts: _SmartsBounds | None = None,
) -> int:
    """Filter the input file by RDKit-computed properties and optional SMARTS.

    Property filtering is applied first; SMARTS filtering is applied only to
    molecules that pass the property gates (saves SMARTS matching work).

    :returns: number of rows that passed all filters.
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
            # SMARTS filtering (only when patterns are configured).
            if smarts is not None and smarts.active:
                if smarts.include and not any(
                    mol.HasSubstructMatch(q) for q in smarts.include
                ):
                    continue
                if any(mol.HasSubstructMatch(q) for q in smarts.exclude):
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
            "Filter a SMI(.gz) file by RDKit properties and optional SMARTS;"
            " output a CSV with columns smiles,smilesid suitable for chemprop prediction."
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
    parser.add_argument(
        "--smarts",
        default=None,
        metavar="FILE",
        help=(
            "Optional SMARTS filter file. Each non-blank, non-comment line "
            "must be 'include:<SMARTS>' or 'exclude:<SMARTS>'. "
            "Molecules must match at least one include pattern (if any) and "
            "must not match any exclude pattern."
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
    smarts: _SmartsBounds | None = None
    if args.smarts is not None:
        smarts_path = Path(args.smarts)
        if smarts_path.exists():
            smarts = _SmartsBounds.read(smarts_path)
            if smarts.active:
                logger.info(
                    "SMARTS filter: %d include, %d exclude patterns",
                    len(smarts.include), len(smarts.exclude),
                )
        else:
            logger.debug("--smarts file not found (%s), skipping SMARTS filter", smarts_path)
    n = filter_smiles(input_path, bounds, output_path, smarts=smarts)
    logger.info("filtered %d rows -> %s", n, output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    sys.exit(main())
