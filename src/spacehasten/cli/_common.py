"""Shared helpers for argparse subcommands.

Builds :class:`Database`, :class:`WorkDir`, :class:`Scheduler`, and
:class:`Settings` instances from the parsed top-level CLI namespace, and
configures workspace logging.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from spacehasten.config.settings import Settings
from spacehasten.core.db import Database
from spacehasten.scheduler import Scheduler, make_scheduler
from spacehasten.scheduler.factory import SchedulerKind
from spacehasten.workspace.layout import WorkDir
from spacehasten.workspace.logging_setup import configure_logging


def add_global_options(parser: argparse.ArgumentParser) -> None:
    """Add the global options shared by every subcommand."""
    parser.add_argument(
        "-w",
        "--workspace",
        type=Path,
        default=None,
        help="Path to the SpaceHASTEN workspace directory. "
        "If omitted, the current working directory is used.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a TOML or INI config file (auto-detected by suffix).",
    )
    parser.add_argument(
        "--scheduler",
        choices=("auto", "slurm", "local"),
        default="auto",
        help="Scheduler backend (default: auto).",
    )
    parser.add_argument(
        "--partition",
        default=None,
        help="SLURM partition (overrides the config file).",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default=None,
        help="Override the scratch directory (paths.scratch_default).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON to stdout where supported.",
    )


def _find_dbsh(root: Path) -> Path:
    """Locate the .dbsh file inside a workspace directory.

    Looks first for ``<dirname>.dbsh`` (canonical name), then falls back to
    any single ``.dbsh`` file in the directory.
    """
    canonical = root / f"{root.name}.dbsh"
    if canonical.exists():
        return canonical
    # Fallback: any .dbsh file
    dbsh_files = list(root.glob("*.dbsh"))
    if len(dbsh_files) == 1:
        return dbsh_files[0]
    if len(dbsh_files) > 1:
        raise SystemExit(
            f"error: multiple .dbsh files in {root}; rename to {canonical.name} "
            "or specify -w with the correct workspace"
        )
    raise SystemExit(
        f"error: no .dbsh file found in {root}\n"
        "Hint: run `spacehasten init` to create the workspace and database, or "
        "check that you are in the correct workspace directory."
    )


def _resolve_workspace(workspace_arg: Path | None) -> Path:
    """Resolve the workspace root directory from the CLI argument or cwd."""
    if workspace_arg is not None:
        root = Path(workspace_arg).resolve()
    else:
        root = Path.cwd()
    if not root.is_dir():
        raise SystemExit(f"error: workspace path is not a directory: {root}")
    return root


def workdir_from_args(args: argparse.Namespace) -> WorkDir:
    """Resolve ``-w`` / cwd to a :class:`WorkDir`."""
    root = _resolve_workspace(args.workspace)
    # Validate that it looks like a workspace (has .dbsh or manifest)
    manifest = root / "manifest.json"
    has_dbsh = any(root.glob("*.dbsh"))
    if not manifest.exists() and not has_dbsh:
        raise SystemExit(
            f"error: {root} does not look like a SpaceHASTEN workspace "
            "(no .dbsh file or manifest.json found).\n"
            "Hint: run `spacehasten init {root}` to create one, or use -w "
            "to point to an existing workspace."
        )
    return WorkDir(root=root)


def open_db(args: argparse.Namespace) -> Database:
    """Open the database inside the resolved workspace."""
    root = _resolve_workspace(args.workspace)
    db_path = _find_dbsh(root)
    return Database(db_path)


def _find_site_config() -> Path | None:
    """Locate the site-wide config installed alongside the package.

    Search order:
    1. ``spacehasten.toml`` or ``spacehasten.ini`` next to the package source
       root (works for editable installs and standard ``pip install``).
    2. Same files in the directory containing the running entry-point script.
    """
    import spacehasten as _pkg

    # Package root: e.g. .../src/spacehasten/__init__.py → .../src/spacehasten
    pkg_dir = Path(_pkg.__file__).resolve().parent
    # Walk up two levels (src/spacehasten → src → repo root).
    for ancestor in (pkg_dir.parent.parent, pkg_dir.parent):
        for name in ("spacehasten.toml", "spacehasten.ini"):
            candidate = ancestor / name
            if candidate.is_file():
                return candidate

    # Fallback: check next to the entry-point script (e.g. .../bin/spacehasten).
    script_dir = Path(sys.argv[0]).resolve().parent
    for name in ("spacehasten.toml", "spacehasten.ini"):
        candidate = script_dir / name
        if candidate.is_file():
            return candidate

    return None


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Build :class:`Settings`, layering ``--config`` and CLI overrides.

    Config resolution order (first found wins):
    1. Explicit ``--config`` path from the command line.
    2. ``spacehasten.toml`` / ``spacehasten.ini`` in the workspace directory.
    3. Site-wide config installed alongside the package.
    """
    ini_path: Path | None = None
    toml_path: Path | None = None
    if args.config is not None:
        cfg = Path(args.config)
        suffix = cfg.suffix.lower()
        if suffix == ".toml":
            toml_path = cfg
        else:
            ini_path = cfg
    else:
        # Auto-discover: workspace first, then site-wide.
        root = _resolve_workspace(args.workspace)
        candidate_toml = root / "spacehasten.toml"
        candidate_ini = root / "spacehasten.ini"
        if candidate_toml.is_file():
            toml_path = candidate_toml
        elif candidate_ini.is_file():
            ini_path = candidate_ini
        else:
            site_cfg = _find_site_config()
            if site_cfg is not None:
                if site_cfg.suffix.lower() == ".toml":
                    toml_path = site_cfg
                else:
                    ini_path = site_cfg

    overrides: dict[str, dict[str, object]] = {}
    if args.partition is not None:
        overrides.setdefault("slurm", {})["slurm_partition"] = args.partition
    if args.scratch is not None:
        overrides.setdefault("paths", {})["scratch_default"] = str(args.scratch)

    return Settings.load(
        ini_path=ini_path,
        toml_path=toml_path,
        cli_overrides=overrides or None,
    )


def scheduler_from_args(args: argparse.Namespace, settings: Settings) -> Scheduler:
    """Build the configured :class:`Scheduler`."""
    kind: SchedulerKind = args.scheduler
    return make_scheduler(kind, settings)


def _log_invocation() -> None:
    """Log the full CLI command line."""
    import shlex

    cmd = " ".join(shlex.quote(a) for a in sys.argv)
    logging.getLogger("spacehasten.cli").info("$ %s", cmd)


def setup_logging(workdir: WorkDir, args: argparse.Namespace) -> None:
    """Bootstrap the workspace skeleton and configure logging."""
    workdir.logs_dir().mkdir(parents=True, exist_ok=True)
    level = getattr(logging, args.log_level)
    configure_logging(workdir, level=level)
    _log_invocation()
