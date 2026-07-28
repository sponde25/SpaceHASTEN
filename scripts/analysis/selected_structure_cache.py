#!/usr/bin/env python3
"""Prepare, build, and combine a selected-compound structure cache."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from spacehasten.analysis.discovery import discover_run
from spacehasten.analysis.selected import selected_manifest, write_selected_manifest

STRUCTURE_FIELDS = (
    "spacehastenid",
    "reghash",
    "typed_scaffold",
    "generic_framework",
    "MW",
    "cLogP",
    "TPSA",
    "HBD",
    "HBA",
    "rotatable",
    "rings",
    "Fsp3",
)
FINGERPRINT = {
    "type": "Morgan binary",
    "radius": 2,
    "n_bits": 1024,
    "use_chirality": False,
    "word_dtype": "uint64",
    "word_count": 16,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_gzip_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text,
    ):
        writer = csv.DictWriter(text, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def command_prepare(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    manifest_path = root / "selected_manifest.csv.gz"
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output root is non-empty; pass --overwrite: {root}")
    root.mkdir(parents=True, exist_ok=True)
    context = discover_run(args.run_or_db, database_path=args.database)
    rows = selected_manifest(
        context,
        hit_threshold=args.hit_threshold,
        strict_threshold=args.strict_threshold,
    )
    if not rows:
        raise ValueError("selected manifest is empty")
    if args.task_count < 1:
        raise ValueError("task count must be positive")
    write_selected_manifest(manifest_path, rows)
    compounds: list[dict[str, Any]] = []
    by_identifier: dict[int, tuple[str, str]] = {}
    reghashes: set[str] = set()
    for row in rows:
        identifier = int(row["spacehastenid"])
        identity = (str(row["reghash"]), str(row["smiles"]))
        if identifier in by_identifier:
            if by_identifier[identifier] != identity:
                raise ValueError(f"selected ID maps to multiple structures: {identifier}")
            continue
        if identity[0] in reghashes:
            raise ValueError(f"selected reghash maps to multiple IDs: {identity[0]}")
        by_identifier[identifier] = identity
        reghashes.add(identity[0])
        compounds.append(row)
    task_count = min(args.task_count, len(compounds))
    (root / "logs").mkdir(exist_ok=True)
    chunks = []
    input_fields = ("spacehastenid", "reghash", "smiles")
    for task in tqdm(range(1, task_count + 1), desc="structure inputs", unit="chunk"):
        start = len(compounds) * (task - 1) // task_count
        stop = len(compounds) * task // task_count
        subset = [{name: row[name] for name in input_fields} for row in compounds[start:stop]]
        path = root / "inputs" / f"selected_{task:04d}_of_{task_count:04d}.csv.gz"
        write_gzip_csv(path, input_fields, subset)
        chunks.append(
            {
                "task_index": task,
                "start": start,
                "stop": stop,
                "rows": stop - start,
                "input": str(path),
                "input_sha256": sha256(path),
            }
        )
    worker = Path(__file__).resolve()
    source_root = worker.parents[2] / "src"
    submit = root / "submit.sh"
    submit.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=selected_structures
#SBATCH --partition=jobs
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --array=1-{task_count}%{task_count}
#SBATCH --output={root}/logs/task-%A_%a.out
#SBATCH --error={root}/logs/task-%A_%a.err
set -euo pipefail
source /data/programs/oce/actoce
conda activate fpsim2-0.7.3
export PYTHONPATH={source_root}:${{PYTHONPATH:-}}
python3 -u {worker} worker --output-root {root} --task-index "$SLURM_ARRAY_TASK_ID"
""",
        encoding="utf-8",
    )
    submit.chmod(0o755)
    write_json(
        root / "preparation.json",
        {
            "status": "ready_not_submitted",
            "database": str(context.database_path),
            "run_input": str(context.input_path),
            "selected_attempts": len(rows),
            "selected_unique_compounds": len({int(row["spacehastenid"]) for row in rows}),
            "scored": sum(bool(row["is_scored"]) for row in rows),
            "hits": sum(bool(row["is_hit"]) for row in rows),
            "strict_hits": sum(bool(row["is_strict_hit"]) for row in rows),
            "hit_threshold": args.hit_threshold,
            "strict_threshold": args.strict_threshold,
            "task_count": task_count,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "fingerprint": FINGERPRINT,
            "chunks": chunks,
            "submit_script": str(submit),
        },
    )
    print(json.dumps({"status": "ready", "tasks": task_count, "rows": len(rows)}))


def command_worker(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    preparation = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
    task_count = int(preparation["task_count"])
    if not 1 <= args.task_index <= task_count:
        raise ValueError(f"task index must be in [1,{task_count}]")
    token = f"{args.task_index:04d}_of_{task_count:04d}"
    source = root / "inputs" / f"selected_{token}.csv.gz"
    output_csv = root / "chunks" / f"structure_{token}.csv.gz"
    output_npz = root / "chunks" / f"fingerprint_{token}.npz"
    receipt = root / "receipts" / f"structure_{token}.json"
    expected = preparation["chunks"][args.task_index - 1]
    if receipt.is_file() and output_csv.is_file() and output_npz.is_file():
        existing = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            existing.get("input_sha256") == sha256(source)
            and existing.get("csv_sha256") == sha256(output_csv)
            and existing.get("npz_sha256") == sha256(output_npz)
        ):
            print(json.dumps({"status": "already_complete", "task": args.task_index}))
            return
    rows = read_gzip_csv(source)
    if len(rows) != int(expected["rows"]) or sha256(source) != expected["input_sha256"]:
        raise ValueError("chunk input does not match preparation receipt")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    records: list[dict[str, Any]] = []
    identifiers = np.empty(len(rows), dtype=np.int64)
    words = np.empty((len(rows), 16), dtype=np.uint64)
    popcounts = np.empty(len(rows), dtype=np.uint16)
    for index, row in enumerate(tqdm(rows, desc=f"structures {token}", unit="molecule")):
        identifier = int(row["spacehastenid"])
        molecule = Chem.MolFromSmiles(row["smiles"])
        if molecule is None:
            raise ValueError(f"unparseable selected SMILES for {identifier}")
        scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
        if scaffold.GetNumAtoms() == 0:
            typed = generic = "[ACYCLIC]"
        else:
            typed = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)
            generic = Chem.MolToSmiles(
                MurckoScaffold.MakeScaffoldGeneric(scaffold),
                canonical=True,
                isomericSmiles=False,
            )
        fingerprint = generator.GetFingerprint(molecule)
        packed = np.frombuffer(DataStructs.BitVectToBinaryText(fingerprint), dtype=np.dtype("<u8"))
        if packed.shape != (16,):
            raise ValueError(f"unexpected fingerprint shape for {identifier}: {packed.shape}")
        descriptors = {
            "MW": Descriptors.MolWt(molecule),
            "cLogP": Descriptors.MolLogP(molecule),
            "TPSA": Descriptors.TPSA(molecule),
            "HBD": Lipinski.NumHDonors(molecule),
            "HBA": Lipinski.NumHAcceptors(molecule),
            "rotatable": Lipinski.NumRotatableBonds(molecule),
            "rings": Lipinski.RingCount(molecule),
            "Fsp3": Lipinski.FractionCSP3(molecule),
        }
        records.append(
            {
                "spacehastenid": identifier,
                "reghash": row["reghash"],
                "typed_scaffold": typed,
                "generic_framework": generic,
                **descriptors,
            }
        )
        identifiers[index] = identifier
        words[index] = packed
        popcounts[index] = fingerprint.GetNumOnBits()
    write_gzip_csv(output_csv, STRUCTURE_FIELDS, records)
    write_npz(output_npz, spacehastenid=identifiers, words=words, popcounts=popcounts)
    write_json(
        receipt,
        {
            "status": "complete",
            "task_index": args.task_index,
            "rows": len(rows),
            "input_sha256": sha256(source),
            "csv_sha256": sha256(output_csv),
            "npz_sha256": sha256(output_npz),
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "rdkit": rdBase.rdkitVersion,
            },
        },
    )
    print(json.dumps({"status": "complete", "task": args.task_index, "rows": len(rows)}))


def command_combine(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    success = root / "_SUCCESS.json"
    if success.is_file() and not args.overwrite:
        print(success.read_text(encoding="utf-8"), end="")
        return
    preparation = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
    manifest_rows = read_gzip_csv(root / "selected_manifest.csv.gz")
    expected_ids = np.asarray(
        list(dict.fromkeys(int(row["spacehastenid"]) for row in manifest_rows)),
        dtype=np.int64,
    )
    task_count = int(preparation["task_count"])
    structures: list[dict[str, str]] = []
    id_parts: list[np.ndarray] = []
    word_parts: list[np.ndarray] = []
    popcount_parts: list[np.ndarray] = []
    software = set()
    for task in tqdm(range(1, task_count + 1), desc="combine structures", unit="chunk"):
        token = f"{task:04d}_of_{task_count:04d}"
        csv_path = root / "chunks" / f"structure_{token}.csv.gz"
        npz_path = root / "chunks" / f"fingerprint_{token}.npz"
        receipt_path = root / "receipts" / f"structure_{token}.json"
        if not all(path.is_file() for path in (csv_path, npz_path, receipt_path)):
            raise FileNotFoundError(f"incomplete chunk {token}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("csv_sha256") != sha256(csv_path) or receipt.get("npz_sha256") != sha256(
            npz_path
        ):
            raise ValueError(f"chunk receipt digest mismatch: {token}")
        structures.extend(read_gzip_csv(csv_path))
        with np.load(npz_path, allow_pickle=False) as data:
            id_parts.append(data["spacehastenid"].astype(np.int64, copy=True))
            word_parts.append(data["words"].astype(np.uint64, copy=True))
            popcount_parts.append(data["popcounts"].astype(np.uint16, copy=True))
        software.add(tuple(sorted(receipt["software"].items())))
    ids = np.concatenate(id_parts)
    words = np.concatenate(word_parts)
    popcounts = np.concatenate(popcount_parts)
    if not np.array_equal(ids, expected_ids):
        raise ValueError("combined fingerprint IDs do not match selected manifest order")
    if words.shape != (len(expected_ids), 16) or len(structures) != len(expected_ids):
        raise ValueError("combined structure/fingerprint dimensions are invalid")
    if len(set(ids.tolist())) != len(ids) or len({row["reghash"] for row in structures}) != len(
        ids
    ):
        raise ValueError("combined structure cache contains duplicate IDs or reghashes")
    structure_path = root / "structure_cache.csv.gz"
    fingerprint_path = root / "fingerprints.npz"
    write_gzip_csv(structure_path, STRUCTURE_FIELDS, structures)
    write_npz(
        fingerprint_path,
        spacehastenid=ids,
        words=words,
        popcounts=popcounts,
    )
    metadata = {
        "status": "complete",
        "rows": len(ids),
        "selected_attempts": len(manifest_rows),
        "task_count": task_count,
        "manifest_sha256": sha256(root / "selected_manifest.csv.gz"),
        "structure_sha256": sha256(structure_path),
        "fingerprint_sha256": sha256(fingerprint_path),
        "fingerprint": FINGERPRINT,
        "software": [dict(values) for values in sorted(software)],
    }
    write_json(success, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("run_or_db")
    prepare.add_argument("--database", type=Path)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--hit-threshold", type=float, required=True)
    prepare.add_argument("--strict-threshold", type=float, default=-11.0)
    prepare.add_argument("--task-count", type=int, default=80)
    prepare.add_argument("--overwrite", action="store_true")
    worker = commands.add_parser("worker")
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--task-index", type=int, required=True)
    combine = commands.add_parser("combine")
    combine.add_argument("--output-root", type=Path, required=True)
    combine.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    {"prepare": command_prepare, "worker": command_worker, "combine": command_combine}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
