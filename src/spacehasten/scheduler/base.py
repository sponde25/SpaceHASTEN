"""Scheduler abstraction for SpaceHASTEN array jobs.

This module defines the public surface that every concrete scheduler
backend (local subprocess, SLURM, SGE, ...) must implement. It models
each unit of work as an *array job* — N independent tasks sharing a
command template, where ``${TASK_ID}`` (1-based, matching SLURM
conventions) is substituted per task.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """State of a single array task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"

    @property
    def is_terminal(self) -> bool:
        return self in {
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
            TaskState.TIMEOUT,
        }

    @property
    def is_success(self) -> bool:
        return self is TaskState.COMPLETED


@dataclass(frozen=True)
class ArrayHandle:
    """Opaque handle to a submitted array job."""

    job_id: str
    name: str
    array_size: int
    workdir: Path


@dataclass(frozen=True)
class ArrayJob:
    """Specification for an array job to submit.

    ``command_template`` is an arbitrary bash snippet; the literal token
    ``${TASK_ID}`` is substituted with the 1-based task index by the
    scheduler before execution. ``env_setup`` lines are emitted before the
    command (e.g. ``source /data/programs/oce/actoce``).
    """

    name: str
    workdir: Path
    array_size: int
    max_concurrent: int
    cpus_per_task: int
    command_template: str
    gpus: int = 0
    exclusive: bool = False
    export_none: bool = True
    env_setup: list[str] = field(default_factory=list)
    depends_on: list[ArrayHandle] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.array_size < 1:
            raise ValueError(f"array_size must be >= 1, got {self.array_size}")
        if self.max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {self.max_concurrent}")
        if self.cpus_per_task < 1:
            raise ValueError(f"cpus_per_task must be >= 1, got {self.cpus_per_task}")
        if self.gpus < 0:
            raise ValueError(f"gpus must be >= 0, got {self.gpus}")


@dataclass(frozen=True)
class ArrayStatus:
    """Snapshot of per-task states for a submitted array job."""

    handle: ArrayHandle
    task_states: tuple[TaskState, ...]

    def __post_init__(self) -> None:
        if len(self.task_states) != self.handle.array_size:
            raise ValueError(
                "task_states length "
                f"({len(self.task_states)}) does not match array_size "
                f"({self.handle.array_size})"
            )

    @property
    def all_terminal(self) -> bool:
        return all(s.is_terminal for s in self.task_states)

    @property
    def failed_indices(self) -> tuple[int, ...]:
        """1-based indices of tasks in any terminal non-success state."""
        return tuple(
            i + 1
            for i, s in enumerate(self.task_states)
            if s.is_terminal and not s.is_success
        )

    @property
    def running_count(self) -> int:
        return sum(1 for s in self.task_states if s is TaskState.RUNNING)

    @property
    def pending_count(self) -> int:
        return sum(1 for s in self.task_states if s is TaskState.PENDING)

    @property
    def completed_count(self) -> int:
        return sum(1 for s in self.task_states if s is TaskState.COMPLETED)


@dataclass(frozen=True)
class ArrayResult:
    """Final outcome of an array job after :meth:`Scheduler.wait`."""

    handle: ArrayHandle
    task_states: tuple[TaskState, ...]
    failed_indices: tuple[int, ...]

    @property
    def success(self) -> bool:
        return len(self.failed_indices) == 0


ProgressCallback = Callable[[ArrayStatus], None]


class Scheduler(ABC):
    """Abstract array-job scheduler."""

    #: Initial poll interval for :meth:`wait` (seconds).
    initial_poll_interval: float = 1.0
    #: Maximum poll interval after exponential backoff (seconds).
    max_poll_interval: float = 30.0
    #: Multiplicative backoff factor between polls.
    backoff_factor: float = 1.5

    @abstractmethod
    def submit_array(self, job: ArrayJob) -> ArrayHandle:
        """Submit an array job and return a handle."""

    @abstractmethod
    def status(self, handle: ArrayHandle) -> ArrayStatus:
        """Return the current per-task status snapshot."""

    @abstractmethod
    def cancel(self, handle: ArrayHandle) -> None:
        """Cancel all tasks of the array job."""

    def wait(
        self,
        handle: ArrayHandle,
        on_progress: ProgressCallback | None = None,
    ) -> ArrayResult:
        """Poll :meth:`status` with exponential backoff until terminal."""
        interval = self.initial_poll_interval
        start_time = time.time()
        prev_completed = -1
        while True:
            snap = self.status(handle)
            if on_progress is not None:
                on_progress(snap)
            # Log progress when completed count changes.
            if snap.completed_count != prev_completed:
                elapsed = time.time() - start_time
                hours, _rem = divmod(int(elapsed), 3600)
                mins, secs = divmod(_rem, 60)
                elapsed_str = (
                    f"{hours}h {mins:02d}m {secs:02d}s"
                    if hours
                    else f"{mins}m {secs:02d}s"
                )
                logger.info(
                    "[%s] %d/%d tasks completed (elapsed: %s)",
                    handle.name,
                    snap.completed_count,
                    handle.array_size,
                    elapsed_str,
                )
                prev_completed = snap.completed_count
            if snap.all_terminal:
                result = ArrayResult(
                    handle=handle,
                    task_states=snap.task_states,
                    failed_indices=snap.failed_indices,
                )
                if result.success:
                    elapsed = time.time() - start_time
                    hours, _rem = divmod(int(elapsed), 3600)
                    mins, secs = divmod(_rem, 60)
                    elapsed_str = (
                        f"{hours}h {mins:02d}m {secs:02d}s"
                        if hours
                        else f"{mins}m {secs:02d}s"
                    )
                    logger.info(
                        "Job %s (%s) completed successfully (%d/%d tasks, %s)",
                        handle.job_id, handle.name,
                        snap.completed_count, handle.array_size,
                        elapsed_str,
                    )
                else:
                    logger.warning(
                        "Job %s (%s) finished with failures: %d/%d tasks failed",
                        handle.job_id, handle.name,
                        len(result.failed_indices), handle.array_size,
                    )
                return result
            time.sleep(interval)
            interval = 5.0
