"""Local subprocess-based scheduler.

Runs each array task as a ``bash -c`` child process on the current host,
honouring ``max_concurrent`` via a small worker pool. Designed for fast
integration tests and single-host runs; not intended for cluster use.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .base import (
    ArrayHandle,
    ArrayJob,
    ArrayStatus,
    Scheduler,
    TaskState,
)

logger = logging.getLogger(__name__)


@dataclass
class _Task:
    index: int  # 1-based, matching SLURM ${SLURM_ARRAY_TASK_ID}
    state: TaskState = TaskState.PENDING
    process: subprocess.Popen[bytes] | None = None
    out_path: Path | None = None
    err_path: Path | None = None
    out_handle: object | None = None  # IO[bytes]; typed as object to avoid generics fuss
    err_handle: object | None = None


class LocalScheduler(Scheduler):
    """Run array jobs locally as subprocesses.

    Tasks are dispatched in 1..N order. At most ``max_concurrent`` are
    running concurrently. ``status()`` polls each ``Popen`` via
    ``poll()`` and advances ready-to-run tasks, keeping the pool full.

    Per-task stdout/stderr is captured to
    ``<workdir>/logs/local/<job_name>/task-NNNN.{out,err}``.
    """

    # Local execution returns quickly; tighten the polling envelope.
    initial_poll_interval = 0.05
    max_poll_interval = 1.0
    backoff_factor = 1.5

    def __init__(self) -> None:
        self._jobs: dict[str, _LocalJob] = {}
        self._lock = threading.Lock()

    def submit_array(self, job: ArrayJob) -> ArrayHandle:
        # Honour dependencies: wait for predecessors to complete.
        for dep in job.depends_on:
            if dep.job_id in self._jobs:
                self.wait(dep)

        job_id = f"local-{uuid.uuid4().hex[:12]}"
        log_dir = job.workdir / "logs" / "local" / job.name
        log_dir.mkdir(parents=True, exist_ok=True)
        job.workdir.mkdir(parents=True, exist_ok=True)

        handle = ArrayHandle(
            job_id=job_id,
            name=job.name,
            array_size=job.array_size,
            workdir=job.workdir,
        )
        local_job = _LocalJob(
            handle=handle,
            spec=job,
            log_dir=log_dir,
            tasks=[_Task(index=i) for i in range(1, job.array_size + 1)],
            pending=deque(range(1, job.array_size + 1)),
            lock=threading.Lock(),
        )
        with self._lock:
            self._jobs[job_id] = local_job
        local_job.fill_pool()
        return handle

    def status(self, handle: ArrayHandle) -> ArrayStatus:
        with self._lock:
            local_job = self._jobs.get(handle.job_id)
        if local_job is None:
            raise KeyError(f"unknown job_id: {handle.job_id}")
        local_job.refresh()
        states = tuple(t.state for t in local_job.tasks)
        return ArrayStatus(handle=handle, task_states=states)

    def cancel(self, handle: ArrayHandle) -> None:
        with self._lock:
            local_job = self._jobs.get(handle.job_id)
        if local_job is None:
            raise KeyError(f"unknown job_id: {handle.job_id}")
        local_job.cancel()


@dataclass
class _LocalJob:
    handle: ArrayHandle
    spec: ArrayJob
    log_dir: Path
    tasks: list[_Task]
    pending: deque[int]
    lock: threading.Lock

    def fill_pool(self) -> None:
        with self.lock:
            running = sum(1 for t in self.tasks if t.state is TaskState.RUNNING)
            slots = self.spec.max_concurrent - running
            while slots > 0 and self.pending:
                idx = self.pending.popleft()
                self._launch(self.tasks[idx - 1])
                slots -= 1

    def refresh(self) -> None:
        with self.lock:
            for t in self.tasks:
                if t.state is TaskState.RUNNING and t.process is not None:
                    rc = t.process.poll()
                    if rc is not None:
                        self._finalize(t, rc)
        self.fill_pool()

    def cancel(self) -> None:
        with self.lock:
            self.pending.clear()
            for t in self.tasks:
                if t.state is TaskState.RUNNING and t.process is not None:
                    with contextlib.suppress(ProcessLookupError):
                        t.process.terminate()
                    try:
                        t.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        t.process.kill()
                        t.process.wait()
                    self._close_handles(t)
                    t.state = TaskState.CANCELLED
                elif t.state is TaskState.PENDING:
                    t.state = TaskState.CANCELLED

    def _launch(self, task: _Task) -> None:
        body = self._render(task.index)
        out_path = self.log_dir / f"task-{task.index:04d}.out"
        err_path = self.log_dir / f"task-{task.index:04d}.err"
        # Truncate prior runs.
        out_handle = out_path.open("wb")
        err_handle = err_path.open("wb")
        env = os.environ.copy()
        env["TASK_ID"] = str(task.index)
        env["SLURM_ARRAY_TASK_ID"] = str(task.index)
        bash = shutil.which("bash") or "/bin/bash"
        try:
            proc = subprocess.Popen(
                [bash, "-c", body],
                cwd=str(self.spec.workdir),
                env=env,
                stdout=out_handle,
                stderr=err_handle,
                stdin=subprocess.DEVNULL,
            )
        except OSError:
            out_handle.close()
            err_handle.close()
            task.state = TaskState.FAILED
            return
        task.process = proc
        task.out_path = out_path
        task.err_path = err_path
        task.out_handle = out_handle
        task.err_handle = err_handle
        task.state = TaskState.RUNNING

    def _finalize(self, task: _Task, returncode: int) -> None:
        self._close_handles(task)
        if returncode == 0:
            task.state = TaskState.COMPLETED
        elif returncode < 0:
            task.state = TaskState.CANCELLED
        else:
            task.state = TaskState.FAILED

    def _close_handles(self, task: _Task) -> None:
        for h in (task.out_handle, task.err_handle):
            if h is not None:
                with contextlib.suppress(Exception):
                    h.close()  # type: ignore[attr-defined]
        task.out_handle = None
        task.err_handle = None

    def _render(self, task_id: int) -> str:
        lines: list[str] = ["set -u"]
        lines.extend(self.spec.env_setup)
        # Substitute ${TASK_ID} (and the bare $TASK_ID via the exported env).
        # We do the literal ${TASK_ID} substitution here so callers do not
        # need a real shell variable; the env export covers cases where the
        # template uses $TASK_ID directly.
        body = self.spec.command_template.replace("${TASK_ID}", str(task_id))
        lines.append(body)
        return "\n".join(lines) + "\n"
