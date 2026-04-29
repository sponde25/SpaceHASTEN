"""Shared helpers for argparse subcommands.

Builds :class:`Database`, :class:`WorkDir`, :class:`Scheduler`, and
:class:`Settings` instances from the parsed top-level CLI namespace, and
configures workspace logging.
"""

from __future__ import annotations

import argparse
import logging
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
        "--db",
        type=Path,
        default=None,
        help="Path to the SpaceHASTEN .dbsh file. The workspace root is "
        "the file's parent directory.",
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


def workdir_from_args(args: argparse.Namespace) -> WorkDir:
    """Resolve ``--db`` to a :class:`WorkDir` rooted at the file's parent."""
    if args.db is None:
        raise SystemExit("error: --db is required")
    db_path = Path(args.db)
    return WorkDir(root=db_path.parent)


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Build :class:`Settings`, layering ``--config`` and CLI overrides."""
    ini_path: Path | None = None
    toml_path: Path | None = None
    if args.config is not None:
        cfg = Path(args.config)
        suffix = cfg.suffix.lower()
        if suffix == ".toml":
            toml_path = cfg
        else:
            # Treat .ini and any other suffix as INI (legacy default).
            ini_path = cfg

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


def open_db(args: argparse.Namespace) -> Database:
    """Open the database referenced by ``--db``."""
    if args.db is None:
        raise SystemExit("error: --db is required")
    return Database(args.db)


def setup_logging(workdir: WorkDir, args: argparse.Namespace) -> None:
    """Bootstrap the workspace skeleton and configure logging."""
    workdir.logs_dir().mkdir(parents=True, exist_ok=True)
    level = getattr(logging, args.log_level)
    configure_logging(workdir, level=level)
