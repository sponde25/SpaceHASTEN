"""Atomic artifact writes and provenance serialization."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .models import RunContext


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with AtomicPath(path) as temporary, temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    with AtomicPath(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


class AtomicPath:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary: Path | None = None

    def __enter__(self) -> Path:
        descriptor, name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        os.close(descriptor)
        self.temporary = Path(name)
        return self.temporary

    def __exit__(self, exc_type: object, *_: object) -> None:
        assert self.temporary is not None
        if exc_type is None:
            self.temporary.replace(self.path)
        elif self.temporary.exists():
            self.temporary.unlink()


def context_json(context: RunContext) -> dict[str, Any]:
    return {
        "input_path": str(context.input_path),
        "database_path": str(context.database_path),
        "shared_root": str(context.shared_root) if context.shared_root else None,
        "acquisition_paths": [
            {"round": round_id, "path": str(path)} for round_id, path in context.acquisition_paths
        ],
        "docking_input_paths": [
            {
                "round": round_id,
                "paths": [str(path) for path in paths],
            }
            for round_id, paths in context.docking_input_paths
        ],
        "capabilities": asdict(context.capabilities),
    }


def checksums(paths: Iterable[Path]) -> list[dict[str, str]]:
    result = []
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result.append({"path": str(path), "sha256": digest.hexdigest()})
    return result


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("spacehasten", "rdkit", "matplotlib", "numpy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unknown"
    return result
