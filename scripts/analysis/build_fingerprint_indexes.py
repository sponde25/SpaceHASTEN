#!/usr/bin/env python3
"""Export ordered SpaceHASTEN compounds and build reusable FPSim2 indexes."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

LOGGER = logging.getLogger("build_fingerprint_indexes")
REQUIRED_COLUMNS = {
    "spacehastenid",
    "smiles",
    "dock_score",
    "dock_iteration",
}


def _connect_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def _validate_schema(connection: sqlite3.Connection) -> None:
    columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(data)")}
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"data table is missing columns: {sorted(missing)}")


def _iter_compounds(
    connection: sqlite3.Connection,
    *,
    seeds: bool,
    limit: int | None,
) -> Iterator[tuple[str, int]]:
    iteration_filter = "dock_iteration = 0" if seeds else "dock_iteration > 0"
    query = (
        "SELECT smiles, spacehastenid FROM data "
        f"WHERE {iteration_filter} AND dock_score IS NOT NULL "
        "ORDER BY spacehastenid"
    )
    parameters: tuple[int, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        parameters = (limit,)

    for smiles, spacehastenid in connection.execute(query, parameters):
        normalized = str(smiles).strip()
        if not normalized or any(character.isspace() for character in normalized):
            raise ValueError(
                f"spacehastenid {spacehastenid} has an invalid whitespace-delimited SMILES"
            )
        yield normalized, int(spacehastenid)


def _write_ordered_inputs(
    database: Path,
    seeds_path: Path,
    all_docked_path: Path,
    *,
    seed_limit: int | None,
    virtual_limit: int | None,
) -> tuple[int, int]:
    seeds_temp = seeds_path.with_name(f".{seeds_path.name}.tmp")
    all_temp = all_docked_path.with_name(f".{all_docked_path.name}.tmp")

    seed_count = 0
    virtual_count = 0
    with _connect_read_only(database) as connection:
        _validate_schema(connection)
        with (
            gzip.open(seeds_temp, "wt", encoding="utf-8") as seed_handle,
            gzip.open(all_temp, "wt", encoding="utf-8") as all_handle,
        ):
            for smiles, spacehastenid in _iter_compounds(connection, seeds=True, limit=seed_limit):
                line = f"{smiles} {spacehastenid}\n"
                seed_handle.write(line)
                all_handle.write(line)
                seed_count += 1
                if seed_count % 100_000 == 0:
                    LOGGER.info("Exported %d seeds", seed_count)

            for smiles, spacehastenid in _iter_compounds(
                connection, seeds=False, limit=virtual_limit
            ):
                all_handle.write(f"{smiles} {spacehastenid}\n")
                virtual_count += 1
                if virtual_count % 100_000 == 0:
                    LOGGER.info("Exported %d virtual docked compounds", virtual_count)

    if seed_count == 0:
        raise ValueError("no docked seed compounds were found")
    if virtual_count == 0:
        raise ValueError("no virtual docked compounds were found")

    seeds_temp.replace(seeds_path)
    all_temp.replace(all_docked_path)
    return seed_count, virtual_count


def _build_index(
    input_path: Path,
    output_path: Path,
    *,
    radius: int,
    fp_size: int,
) -> None:
    from FPSim2.io import create_db_file

    def molecule_rows() -> Iterator[tuple[str, int]]:
        with gzip.open(input_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                smiles, molecule_id = line.rstrip().rsplit(None, 1)
                yield smiles, int(molecule_id)

    temporary = output_path.with_name(f".{output_path.name}.tmp")
    LOGGER.info("Building FPSim2 index %s", output_path)
    create_db_file(
        molecule_rows(),
        str(temporary),
        "smiles",
        "Morgan",
        {"radius": radius, "fpSize": fp_size},
    )
    temporary.replace(output_path)


def _inspect_index(path: Path) -> dict[str, object]:
    import tables

    with tables.open_file(path, mode="r") as handle:
        return {
            "rows": int(handle.root.fps.nrows),
            "fp_type": str(handle.root.config[0]),
            "fp_params": dict(handle.root.config[1]),
            "rdkit_version": str(handle.root.config[2]),
            "fpsim2_version": str(handle.root.config[3]),
            "size_bytes": path.stat().st_size,
        }


def _remove_existing(paths: list[Path], *, force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "refusing to overwrite existing fingerprint artifacts:\n"
            f"{formatted}\nUse --force to replace them."
        )
    for path in existing:
        path.unlink()


def build_indexes(args: argparse.Namespace) -> dict[str, object]:
    database = args.database.resolve()
    if not database.is_file():
        raise FileNotFoundError(database)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"morgan_r{args.radius}_{args.fp_size}"
    seeds_path = output_dir / "seeds.smi.gz"
    all_docked_path = output_dir / "all_docked.smi.gz"
    seed_index = output_dir / f"seeds_{suffix}.h5"
    all_docked_index = output_dir / f"all_docked_{suffix}.h5"
    metadata_path = output_dir / "fingerprint_metadata.json"

    artifacts = [
        seeds_path,
        all_docked_path,
        seed_index,
        all_docked_index,
        metadata_path,
    ]
    _remove_existing(artifacts, force=args.force)

    LOGGER.info("Exporting deterministic seed-first inputs from %s", database)
    seed_count, virtual_count = _write_ordered_inputs(
        database,
        seeds_path,
        all_docked_path,
        seed_limit=args.seed_limit,
        virtual_limit=args.virtual_limit,
    )

    _build_index(
        seeds_path,
        seed_index,
        radius=args.radius,
        fp_size=args.fp_size,
    )
    _build_index(
        all_docked_path,
        all_docked_index,
        radius=args.radius,
        fp_size=args.fp_size,
    )

    seed_details = _inspect_index(seed_index)
    all_details = _inspect_index(all_docked_index)
    expected_all = seed_count + virtual_count
    if seed_details["rows"] != seed_count:
        raise RuntimeError(
            f"seed index row count {seed_details['rows']} != exported count {seed_count}"
        )
    if all_details["rows"] != expected_all:
        raise RuntimeError(
            f"all-docked index row count {all_details['rows']} != exported count {expected_all}"
        )

    metadata: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "source_database": str(database),
        "output_directory": str(output_dir),
        "fingerprint": {
            "type": "Morgan",
            "radius": args.radius,
            "fp_size": args.fp_size,
            "binary": True,
            "similarity": "Tanimoto/Jaccard",
        },
        "counts": {
            "seeds": seed_count,
            "virtual_docked": virtual_count,
            "all_docked": expected_all,
        },
        "inputs": {
            "seeds": str(seeds_path),
            "all_docked_seed_first": str(all_docked_path),
        },
        "indexes": {
            "seeds": {"path": str(seed_index), **seed_details},
            "all_docked": {"path": str(all_docked_index), **all_details},
        },
        "software": {"fpsim2": version("FPSim2")},
        "limits": {
            "seeds": args.seed_limit,
            "virtual_docked": args.virtual_limit,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export docked SpaceHASTEN compounds in deterministic seed-first "
            "order and build Morgan FPSim2 HDF5 indexes."
        )
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--fp-size", type=int, default=1024)
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--virtual-limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()
    if args.radius < 1:
        parser.error("--radius must be at least 1")
    if args.fp_size < 64 or args.fp_size % 64 != 0:
        parser.error("--fp-size must be a positive multiple of 64")
    if args.seed_limit is not None and args.seed_limit < 1:
        parser.error("--seed-limit must be at least 1")
    if args.virtual_limit is not None and args.virtual_limit < 1:
        parser.error("--virtual-limit must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        metadata = build_indexes(args)
    except Exception:
        LOGGER.exception("Fingerprint index generation failed")
        return 1

    counts = metadata["counts"]
    LOGGER.info(
        "Fingerprint indexes complete: seeds=%s virtual_docked=%s all=%s",
        counts["seeds"],
        counts["virtual_docked"],
        counts["all_docked"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
