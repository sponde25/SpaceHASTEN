"""Run-layout discovery with deliberate ambiguity failures."""

from __future__ import annotations

import json
from pathlib import Path

from .database import ReadOnlyDatabase
from .models import RunContext


def discover_run(run_or_db: str | Path) -> RunContext:
    """Discover a run from its outer directory, local/shared roots, manifest, or DB."""
    supplied = Path(run_or_db).expanduser()
    if not supplied.exists():
        raise FileNotFoundError(supplied)
    input_path = supplied.absolute()
    database_path = supplied.resolve() if supplied.is_file() else _discover_database(input_path)
    shared_root = _discover_shared_root(input_path)
    acquisition_paths = _discover_acquisitions(shared_root) if shared_root else ()
    with ReadOnlyDatabase(database_path) as database:
        capabilities = database.capabilities()
    if not capabilities.has_data:
        raise ValueError(f"{database_path} does not contain a data table")
    return RunContext(input_path, database_path, shared_root, acquisition_paths, capabilities)


def _candidate_roots(root: Path) -> tuple[Path, ...]:
    outer = root.parent if root.name in {"run_local", "run_shared"} else root
    candidates = (root, outer, outer / "run_local")
    return tuple(path for path in candidates if path.is_dir())


def _discover_database(root: Path) -> Path:
    outer = root.parent if root.name in {"run_local", "run_shared"} else root
    local_root = root if root.name == "run_local" else outer / "run_local"
    if local_root.is_dir():
        local_candidates = {
            path.resolve()
            for suffix in ("*.dbsh", "*.sqlite", "*.sqlite3", "*.db")
            for path in local_root.glob(suffix)
        }
        if len(local_candidates) == 1:
            return next(iter(local_candidates))
        if len(local_candidates) > 1:
            paths = ", ".join(str(path) for path in sorted(local_candidates))
            raise ValueError(
                f"expected exactly one database in canonical run_local {local_root}; "
                f"found {len(local_candidates)}: {paths}"
            )

    candidates: set[Path] = set()
    for candidate_root in _candidate_roots(root):
        for manifest in candidate_root.rglob("*.json"):
            try:
                payload = json.loads(manifest.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                for key in ("database", "database_path", "db_path"):
                    value = payload.get(key)
                    if isinstance(value, str) and (manifest.parent / value).is_file():
                        candidates.add((manifest.parent / value).resolve())
        for suffix in ("*.dbsh", "*.sqlite", "*.sqlite3", "*.db"):
            candidates.update(path.resolve() for path in candidate_root.rglob(suffix))
    if len(candidates) != 1:
        paths = ", ".join(str(path) for path in sorted(candidates))
        raise ValueError(
            f"expected exactly one database below {root}; found {len(candidates)}: {paths}"
        )
    return next(iter(candidates))


def _discover_shared_root(root: Path) -> Path | None:
    outer = root.parent if root.is_file() or root.name in {"run_local", "run_shared"} else root
    candidates = {
        path.absolute()
        for path in (root, outer / "run_shared")
        if path.name == "run_shared" or path.is_dir() and path.name == "run_shared"
    }
    if root.name == "run_shared":
        candidates.add(root)
    if len(candidates) > 1:
        raise ValueError(f"ambiguous run_shared roots: {sorted(map(str, candidates))}")
    return next(iter(candidates)) if candidates else None


def _discover_acquisitions(shared_root: Path) -> tuple[tuple[int, Path], ...]:
    primary = tuple(sorted(shared_root.glob("docking/iter*/acquisition.csv")))
    paths = primary or tuple(sorted(shared_root.glob("iter*/acquisition.csv")))
    found: dict[int, Path] = {}
    for path in paths:
        suffix = path.parent.name.removeprefix("iter")
        if suffix.isdigit():
            round_id = int(suffix)
            if round_id <= 0:
                continue
            if round_id in found:
                raise ValueError(f"ambiguous acquisition CSV for round {round_id}")
            found[round_id] = path
    return tuple(sorted(found.items()))
