"""Scheduler factory: pick a backend based on settings or environment."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from spacehasten.config.settings import Settings

from .base import Scheduler
from .local import LocalScheduler
from .slurm import SlurmScheduler

SchedulerKind = Literal["auto", "slurm", "local"]


def make_scheduler(
    kind: SchedulerKind,
    settings: Settings,
    *,
    log_dir: Path | None = None,
) -> Scheduler:
    """Build a :class:`Scheduler` instance.

    ``auto`` picks :class:`SlurmScheduler` if ``sbatch`` is on ``$PATH``,
    otherwise :class:`LocalScheduler`.
    """
    if kind == "auto":
        kind = "slurm" if shutil.which("sbatch") is not None else "local"

    if kind == "local":
        return LocalScheduler()
    if kind == "slurm":
        return SlurmScheduler(
            partition=settings.slurm.slurm_partition,
            gpu_parameter=settings.slurm.slurm_gpu_parameter,
            log_dir=log_dir,
        )
    raise ValueError(f"unknown scheduler kind: {kind!r}")
