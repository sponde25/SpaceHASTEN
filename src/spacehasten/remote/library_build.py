#!/usr/bin/env python3
"""Remote library-build worker - cxsmiles shard to Parquet chunk.

Invoked once per shard (produced by
:func:`spacehasten.stages.library_build.library_build`) on a compute node
with RDKit available. Reads a headerless, tab/whitespace-separated shard of
an Enamine ``.cxsmiles`` file, canonicalizes each SMILES with RDKit,
computes the tautomer-insensitive registration hash, and either takes the
six PC-filter descriptors from the source columns (default; plan D5) or
recomputes them with RDKit (``--recompute-props``, using the exact calls
from :mod:`spacehasten.remote.prop_filter`). Rows that fail
``Chem.MolFromSmiles`` are dropped; rows missing a required source
descriptor are dropped too (unless ``--recompute-props``). Writes a single
zstd-compressed Parquet file matching the schema in
``docs/plan-library-screening.md`` section 3.2.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:  # pragma: no cover - import guarded for remote node only
    print(f"Error: Failed to import pyarrow: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Schema per docs/plan-library-screening.md section 3.2.
SCHEMA = pa.schema(
    [
        ("compound_id", pa.string()),
        ("smiles", pa.string()),
        ("reghash", pa.string()),
        ("mw", pa.float32()),
        ("slogp", pa.float32()),
        ("hba", pa.int16()),
        ("hbd", pa.int16()),
        ("rotbonds", pa.int16()),
        ("tpsa", pa.float32()),
        ("fsp3", pa.float32()),
        ("qed", pa.float32()),
        ("inchikey", pa.string()),
    ]
)

_FLUSH_EVERY = 200_000


def _split_line(line: str) -> list[str] | None:
    stripped = line.rstrip("\n").rstrip("\r")
    if not stripped:
        return None
    return stripped.split("\t") if "\t" in stripped else stripped.split()


def _field(parts: list[str], col: int | None) -> str | None:
    if col is None or col >= len(parts) or col < 0:
        return None
    value = parts[col].strip()
    return value or None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    f = _to_float(value)
    return None if f is None else int(round(f))


def build_chunk(
    shard_path: Path,
    output_path: Path,
    *,
    smiles_col: int,
    id_col: int,
    mw_col: int | None = None,
    slogp_col: int | None = None,
    hba_col: int | None = None,
    hbd_col: int | None = None,
    rotbonds_col: int | None = None,
    tpsa_col: int | None = None,
    fsp3_col: int | None = None,
    qed_col: int | None = None,
    inchikey_col: int | None = None,
    recompute_props: bool = False,
) -> int:
    """Convert one headerless shard into a single Parquet chunk.

    :returns: number of rows written.
    """
    from rdkit import Chem
    from rdkit.Chem import RegistrationHash

    if recompute_props:
        from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns: dict[str, list] = {name: [] for name in SCHEMA.names}
    n_written = 0
    writer: pq.ParquetWriter | None = None

    def _flush() -> None:
        nonlocal writer
        if not columns["smiles"]:
            return
        table = pa.table(columns, schema=SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(output_path, SCHEMA, compression="zstd")
        writer.write_table(table)
        for lst in columns.values():
            lst.clear()

    with shard_path.open("rt", encoding="utf-8") as fh:
        for raw in fh:
            parts = _split_line(raw)
            if parts is None:
                continue
            smiles = _field(parts, smiles_col)
            compound_id = _field(parts, id_col)
            if smiles is None or compound_id is None:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            canon = Chem.MolToSmiles(mol)
            reghash = RegistrationHash.GetMolLayers(mol)[
                RegistrationHash.HashLayer.TAUTOMER_HASH
            ]

            if recompute_props:
                mw: float | None = Descriptors.MolWt(mol)  # type: ignore[attr-defined]
                slogp: float | None = Crippen.MolLogP(mol)  # type: ignore[attr-defined]
                hba: int | None = rdMolDescriptors.CalcNumHBA(mol)
                hbd: int | None = rdMolDescriptors.CalcNumHBD(mol)
                rotbonds: int | None = rdMolDescriptors.CalcNumRotatableBonds(mol)
                tpsa: float | None = rdMolDescriptors.CalcTPSA(mol)
            else:
                mw = _to_float(_field(parts, mw_col))
                slogp = _to_float(_field(parts, slogp_col))
                hba = _to_int(_field(parts, hba_col))
                hbd = _to_int(_field(parts, hbd_col))
                rotbonds = _to_int(_field(parts, rotbonds_col))
                tpsa = _to_float(_field(parts, tpsa_col))
                if None in (mw, slogp, hba, hbd, rotbonds, tpsa):
                    continue

            fsp3 = _to_float(_field(parts, fsp3_col))
            qed = _to_float(_field(parts, qed_col))
            inchikey = _field(parts, inchikey_col)

            columns["compound_id"].append(compound_id)
            columns["smiles"].append(canon)
            columns["reghash"].append(reghash)
            columns["mw"].append(mw)
            columns["slogp"].append(slogp)
            columns["hba"].append(hba)
            columns["hbd"].append(hbd)
            columns["rotbonds"].append(rotbonds)
            columns["tpsa"].append(tpsa)
            columns["fsp3"].append(fsp3)
            columns["qed"].append(qed)
            columns["inchikey"].append(inchikey)
            n_written += 1

            if len(columns["smiles"]) >= _FLUSH_EVERY:
                _flush()

    _flush()
    if writer is None:
        # No rows survived; still emit an empty parquet file with the
        # correct schema so downstream globbing/reads never see a hole.
        pq.write_table(
            pa.table({name: [] for name in SCHEMA.names}, schema=SCHEMA),
            output_path,
            compression="zstd",
        )
    else:
        writer.close()
    return n_written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a headerless Enamine cxsmiles shard into a canonicalized,"
            " zstd-compressed Parquet library chunk."
        )
    )
    parser.add_argument("shard", help="Headerless input shard (plain text)")
    parser.add_argument("output", help="Output Parquet chunk path")
    parser.add_argument("--smiles-col", type=int, required=True)
    parser.add_argument("--id-col", type=int, required=True)
    parser.add_argument("--mw-col", type=int, default=None)
    parser.add_argument("--slogp-col", type=int, default=None)
    parser.add_argument("--hba-col", type=int, default=None)
    parser.add_argument("--hbd-col", type=int, default=None)
    parser.add_argument("--rotbonds-col", type=int, default=None)
    parser.add_argument("--tpsa-col", type=int, default=None)
    parser.add_argument("--fsp3-col", type=int, default=None)
    parser.add_argument("--qed-col", type=int, default=None)
    parser.add_argument("--inchikey-col", type=int, default=None)
    parser.add_argument(
        "--recompute-props",
        action="store_true",
        help="Force RDKit computation of mw/slogp/hba/hbd/rotbonds/tpsa"
        " instead of parsing them from the source columns.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shard_path = Path(args.shard)
    if not shard_path.exists():
        logger.error("shard not found: %s", shard_path)
        return 1

    if not args.recompute_props:
        required = {
            "--mw-col": args.mw_col,
            "--slogp-col": args.slogp_col,
            "--hba-col": args.hba_col,
            "--hbd-col": args.hbd_col,
            "--rotbonds-col": args.rotbonds_col,
            "--tpsa-col": args.tpsa_col,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            logger.error(
                "missing required column flags (or pass --recompute-props): %s",
                missing,
            )
            return 1

    n = build_chunk(
        shard_path,
        Path(args.output),
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        mw_col=args.mw_col,
        slogp_col=args.slogp_col,
        hba_col=args.hba_col,
        hbd_col=args.hbd_col,
        rotbonds_col=args.rotbonds_col,
        tpsa_col=args.tpsa_col,
        fsp3_col=args.fsp3_col,
        qed_col=args.qed_col,
        inchikey_col=args.inchikey_col,
        recompute_props=args.recompute_props,
    )
    logger.info("wrote %d rows -> %s", n, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
