"""SLURM scheduler backend.

Submits array jobs via ``sbatch --parsable``, polls them with
``sacct -P --noheader``, and supports ``afterok`` dependency chains.

Replaces the six near-duplicate writers in legacy ``scheduler_functions.py``
by rendering a single shared Jinja template.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from .base import (
    ArrayHandle,
    ArrayJob,
    ArrayStatus,
    Scheduler,
    TaskState,
)

logger = logging.getLogger(__name__)

# Map sacct State strings to our TaskState enum. sacct may decorate cancelled
# states with " by <uid>" — we strip the suffix before lookup.
_SACCT_STATE_MAP: dict[str, TaskState] = {
    "PENDING": TaskState.PENDING,
    "REQUEUED": TaskState.PENDING,
    "RESIZING": TaskState.PENDING,
    "SUSPENDED": TaskState.PENDING,
    "CONFIGURING": TaskState.PENDING,
    "RUNNING": TaskState.RUNNING,
    "COMPLETING": TaskState.RUNNING,
    "COMPLETED": TaskState.COMPLETED,
    "FAILED": TaskState.FAILED,
    "NODE_FAIL": TaskState.FAILED,
    "OUT_OF_MEMORY": TaskState.FAILED,
    "BOOT_FAIL": TaskState.FAILED,
    "DEADLINE": TaskState.FAILED,
    "PREEMPTED": TaskState.FAILED,
    "TIMEOUT": TaskState.TIMEOUT,
    "CANCELLED": TaskState.CANCELLED,
}

# JobID format produced by `sacct -X`: "<jobid>_<task>" (or "<jobid>_[<range>]"
# for not-yet-expanded array tasks). We accept both forms.
_JOBID_RE = re.compile(r"^(?P<job>\d+)_(?P<task>\d+|\[.*\])$")


def _load_template() -> str:
    """Load the bash template bundled with the package."""
    resource = files("spacehasten.scheduler").joinpath("_template.sh.j2")
    return resource.read_text(encoding="utf-8")


@dataclass(frozen=True)
class _RenderedScript:
    text: str
    path: Path


class SlurmScheduler(Scheduler):
    """Scheduler backend for SLURM clusters.

    Parameters
    ----------
    partition:
        Optional SLURM partition; rendered as ``#SBATCH -p <partition>``.
    gpu_parameter:
        Verbatim extra ``#SBATCH`` directive for GPU jobs (e.g.
        ``--gres=gpu:1``). Only emitted when the job requests GPUs.
    log_dir:
        Where stdout/stderr go. Defaults to ``<workdir>/logs/slurm/<name>``.
    """

    initial_poll_interval = 5.0
    max_poll_interval = 30.0
    backoff_factor = 1.5

    def __init__(
        self,
        *,
        partition: str | None = None,
        gpu_parameter: str | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self.partition = partition
        self.gpu_parameter = gpu_parameter
        self._log_dir_override = log_dir
        self._env = Environment(
            keep_trailing_newline=True,
            undefined=StrictUndefined,
            autoescape=False,
        )
        self._template = self._env.from_string(_load_template())
        # Track submitted handles so wait()/status() can locate them.
        self._handles: dict[str, ArrayHandle] = {}

    # ------------------------------------------------------------------ #
    # Submission                                                         #
    # ------------------------------------------------------------------ #

    def submit_array(self, job: ArrayJob) -> ArrayHandle:
        rendered = self._render_script(job)
        rendered.path.parent.mkdir(parents=True, exist_ok=True)
        rendered.path.write_text(rendered.text, encoding="utf-8")
        rendered.path.chmod(0o755)

        cmd = ["sbatch", "--parsable"]
        if job.export_none:
            cmd.append("--export=NONE")
        cmd.append(str(rendered.path))
        logger.debug("submitting slurm job: %s", " ".join(cmd))
        completed = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            text=True,
            cwd=str(job.workdir),
        )
        # `sbatch --parsable` prints "<jobid>" or "<jobid>;<cluster>".
        first_line = completed.stdout.strip().splitlines()[0]
        job_id = first_line.split(";", 1)[0].strip()
        if not job_id.isdigit():
            raise RuntimeError(
                f"unexpected sbatch output: {completed.stdout!r}"
            )

        handle = ArrayHandle(
            job_id=job_id,
            name=job.name,
            array_size=job.array_size,
            workdir=job.workdir,
        )
        self._handles[job_id] = handle
        return handle

    # ------------------------------------------------------------------ #
    # Polling                                                            #
    # ------------------------------------------------------------------ #

    def status(self, handle: ArrayHandle) -> ArrayStatus:
        cmd = [
            "sacct",
            "-j",
            handle.job_id,
            "--format=JobID,State,ExitCode",
            "-P",
            "--noheader",
        ]
        completed = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            text=True,
        )
        per_task = self._parse_sacct(
            completed.stdout, handle.job_id, handle.array_size
        )
        return ArrayStatus(handle=handle, task_states=per_task)

    @staticmethod
    def _parse_sacct(
        sacct_output: str, job_id: str, array_size: int
    ) -> tuple[TaskState, ...]:
        """Parse ``sacct -P -n`` output into a per-task state tuple.

        Tasks not yet seen in sacct (e.g. still queued before SLURM has
        expanded the array) are treated as ``PENDING``.
        """
        states: dict[int, TaskState] = {}
        # Track a "bulk state" for ranges that haven't been expanded yet.
        bulk_state: TaskState | None = None
        for raw in sacct_output.splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            this_jobid, raw_state = parts[0], parts[1]
            match = _JOBID_RE.match(this_jobid)
            if match is None or match["job"] != job_id:
                continue
            # Strip "CANCELLED by 1234" → "CANCELLED".
            base_state = raw_state.split(" ", 1)[0].strip().upper()
            resolved = _SACCT_STATE_MAP.get(base_state, TaskState.FAILED)

            task_field = match["task"]
            if task_field.startswith("["):
                # Range like "12345_[1-100]" — applies to all tasks in range.
                # If it's a terminal state (e.g. CANCELLED), record it as bulk.
                if resolved.is_terminal:
                    bulk_state = resolved
                continue
            task_idx = int(task_field)
            states[task_idx] = resolved

        # Fill in missing tasks: use bulk_state if set (e.g. whole array
        # cancelled before expansion), otherwise PENDING.
        default = bulk_state if bulk_state is not None else TaskState.PENDING
        return tuple(
            states.get(i, default) for i in range(1, array_size + 1)
        )

    def cancel(self, handle: ArrayHandle) -> None:
        subprocess.run(
            ["scancel", handle.job_id],
            check=True,
            capture_output=True,
            text=True,
        )

    # ------------------------------------------------------------------ #
    # Rendering                                                          #
    # ------------------------------------------------------------------ #

    def _log_dir(self, job: ArrayJob) -> Path:
        if self._log_dir_override is not None:
            return self._log_dir_override / job.name
        return job.workdir / "logs" / "slurm" / job.name

    def _render_script(self, job: ArrayJob) -> _RenderedScript:
        log_dir = self._log_dir(job)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Substitute literal ${TASK_ID} for callers that prefer it; the
        # template also exports TASK_ID as an env var so $TASK_ID and
        # ${SLURM_ARRAY_TASK_ID} both work in command bodies.
        command_body = job.command_template.replace(
            "${TASK_ID}", "${SLURM_ARRAY_TASK_ID}"
        )

        gpu_parameter: str | None = None
        if job.gpus > 0:
            gpu_parameter = self.gpu_parameter or f"--gres=gpu:{job.gpus}"

        depends_on = [dep.job_id for dep in job.depends_on]

        text = self._template.render(
            name=job.name,
            partition=self.partition,
            cpus_per_task=job.cpus_per_task,
            array_size=job.array_size,
            max_concurrent=job.max_concurrent,
            gpu_parameter=gpu_parameter,
            exclusive=job.exclusive,
            depends_on=depends_on,
            env_setup=list(job.env_setup),
            command_body=command_body,
            workdir=str(job.workdir),
            log_out=str(log_dir / "task-%a.out"),
            log_err=str(log_dir / "task-%a.err"),
        )
        path = job.workdir / f"submit_{job.name}.sh"
        return _RenderedScript(text=text, path=path)
