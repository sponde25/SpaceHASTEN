"""Three-tier logging setup for a SpaceHASTEN workspace.

Tier 1 — *master log* (``logs/spacehasten.log``): rotating, append-only
audit trail of every CLI invocation, stage transition, and error.

Tier 2 — *per-stage logs* (``logs/<stage>-<n>.log``): orchestrator-side log
of one stage's activity, attached for the duration of
:func:`stage_log_context`.

Tier 3 — *scheduler logs* (``logs/slurm/<jobname>/``): produced by the
scheduler itself; not configured here.
"""

from __future__ import annotations

import contextlib
import logging
import logging.handlers
from collections.abc import Iterator
from pathlib import Path

from .layout import WorkDir

_MASTER_LOG_NAME = "spacehasten.log"
_MASTER_MAX_BYTES = 20 * 1024 * 1024  # 20 MiB
_MASTER_BACKUP_COUNT = 5

_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

_MASTER_HANDLER_ATTR = "_spacehasten_master"
_CONSOLE_HANDLER_ATTR = "_spacehasten_console"


def _formatter() -> logging.Formatter:
    return logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)


def configure_logging(
    workdir: WorkDir,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Attach master + console handlers to the root logger.

    Idempotent: calling twice for the same workdir does not duplicate
    handlers. The master handler is a :class:`RotatingFileHandler`; the
    console handler is a :class:`rich.logging.RichHandler` when ``rich`` is
    importable, falling back to :class:`logging.StreamHandler` otherwise.
    """
    workdir.logs_dir().mkdir(parents=True, exist_ok=True)
    master_path = workdir.logs_dir() / _MASTER_LOG_NAME

    root = logging.getLogger()
    root.setLevel(level)

    # Master file handler (rotating).
    master = _existing_handler(root, _MASTER_HANDLER_ATTR)
    if master is None or getattr(master, "baseFilename", None) != str(master_path):
        if master is not None:
            root.removeHandler(master)
            master.close()
        master = logging.handlers.RotatingFileHandler(
            master_path,
            maxBytes=_MASTER_MAX_BYTES,
            backupCount=_MASTER_BACKUP_COUNT,
            encoding="utf-8",
        )
        master.setFormatter(_formatter())
        master.setLevel(level)
        setattr(master, _MASTER_HANDLER_ATTR, True)
        root.addHandler(master)
    else:
        master.setLevel(level)

    # Console handler.
    existing_console = _existing_handler(root, _CONSOLE_HANDLER_ATTR)
    if console:
        if existing_console is None:
            handler = _make_console_handler()
            handler.setLevel(level)
            setattr(handler, _CONSOLE_HANDLER_ATTR, True)
            root.addHandler(handler)
        else:
            existing_console.setLevel(level)
    elif existing_console is not None:
        root.removeHandler(existing_console)
        existing_console.close()

    return root


def _existing_handler(
    logger: logging.Logger, marker: str
) -> logging.Handler | None:
    for handler in logger.handlers:
        if getattr(handler, marker, False):
            return handler
    return None


def _make_console_handler() -> logging.Handler:
    try:
        from rich.logging import RichHandler
    except ImportError:
        handler: logging.Handler = logging.StreamHandler()
        handler.setFormatter(_formatter())
        return handler
    rich_handler = RichHandler(rich_tracebacks=True, show_path=False)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    return rich_handler


@contextlib.contextmanager
def stage_log_context(
    workdir: WorkDir, stage_name: str, level: int = logging.INFO
) -> Iterator[Path]:
    """Attach a per-stage :class:`FileHandler` for the duration of the block.

    Picks the next free ``logs/<stage>-<n>.log`` (1-indexed). Yields the
    chosen log path so callers can reference it (e.g. in the manifest).
    """
    logs_dir = workdir.logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = _next_stage_log_path(logs_dir, stage_name)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(_formatter())

    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield log_path
    finally:
        root.removeHandler(handler)
        handler.close()


def _next_stage_log_path(logs_dir: Path, stage_name: str) -> Path:
    n = 1
    while (logs_dir / f"{stage_name}-{n}.log").exists():
        n += 1
    return logs_dir / f"{stage_name}-{n}.log"
