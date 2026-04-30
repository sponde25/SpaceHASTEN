"""Tail per-task scheduler logs for diagnostics.

Used by stage modules to attach the last few lines of a failed task's
``.out`` / ``.err`` to the exception they raise. Keeps "job finished but
no output" failures debuggable without manually digging into SLURM log
directories.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .base import ArrayHandle


def _candidate_logs(handle: ArrayHandle) -> Iterable[Path]:
    """Yield candidate ``.out`` / ``.err`` paths for every task.

    Covers both the SLURM layout (``slurm-<jobid>_<task>.{out,err}`` in
    the job workdir) and the :class:`LocalScheduler` layout
    (``logs/local/<jobname>/task-NNNN.{out,err}`` rooted in the local
    scheduler's logs root, which is the same workdir).
    """
    wd = handle.workdir
    for task in range(1, handle.array_size + 1):
        # SLURM layout
        yield wd / f"slurm-{handle.job_id}_{task}.out"
        yield wd / f"slurm-{handle.job_id}_{task}.err"
        # LocalScheduler layout (logs sit under the workdir's logs/local)
        yield wd / "logs" / "local" / handle.name / f"task-{task:04d}.out"
        yield wd / "logs" / "local" / handle.name / f"task-{task:04d}.err"


def _tail(path: Path, n_lines: int) -> str:
    try:
        with path.open("rt", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        return f"<read failed: {exc}>"
    return "".join(lines[-n_lines:])


def tail_logs(handle: ArrayHandle, n_lines: int = 50) -> str:
    """Return a formatted snippet of the last ``n_lines`` from each log.

    Skips paths that do not exist. Returns ``"<no logs found>"`` when no
    candidate file is present (typical for the LocalScheduler when the
    array task never started).
    """
    chunks: list[str] = []
    for path in _candidate_logs(handle):
        if not path.exists():
            continue
        snippet = _tail(path, n_lines)
        if not snippet.strip():
            continue
        rel = path
        chunks.append(f"--- {rel} (last {n_lines} lines) ---\n{snippet}")
    if not chunks:
        return "<no logs found>"
    return "\n".join(chunks)


__all__ = ["tail_logs"]
