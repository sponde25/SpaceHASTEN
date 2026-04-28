"""Tests for the local subprocess scheduler."""

from __future__ import annotations

from pathlib import Path

import pytest

from spacehasten.scheduler import (
    ArrayJob,
    ArrayResult,
    LocalScheduler,
    TaskState,
)


def _run(scheduler: LocalScheduler, job: ArrayJob) -> ArrayResult:
    handle = scheduler.submit_array(job)
    return scheduler.wait(handle)


def test_local_scheduler_runs_array_tasks(tmp_path: Path) -> None:
    scheduler = LocalScheduler()
    job = ArrayJob(
        name="echo",
        workdir=tmp_path,
        array_size=4,
        max_concurrent=2,
        cpus_per_task=1,
        command_template='echo "$TASK_ID" > out_${TASK_ID}.txt',
    )
    result = _run(scheduler, job)

    assert result.success
    assert result.failed_indices == ()
    assert all(s is TaskState.COMPLETED for s in result.task_states)
    for i in range(1, 5):
        out = tmp_path / f"out_{i}.txt"
        assert out.exists(), f"missing {out}"
        assert out.read_text().strip() == str(i)


def test_local_scheduler_reports_failed_tasks(tmp_path: Path) -> None:
    scheduler = LocalScheduler()
    # Task 2 fails; tasks 1 and 3 succeed.
    template = (
        'if [ "${TASK_ID}" = "2" ]; then exit 1; fi; '
        'echo "${TASK_ID}" > ok_${TASK_ID}.txt'
    )
    job = ArrayJob(
        name="mixed",
        workdir=tmp_path,
        array_size=3,
        max_concurrent=3,
        cpus_per_task=1,
        command_template=template,
    )
    result = _run(scheduler, job)

    assert not result.success
    assert result.failed_indices == (2,)
    assert result.task_states[0] is TaskState.COMPLETED
    assert result.task_states[1] is TaskState.FAILED
    assert result.task_states[2] is TaskState.COMPLETED
    assert (tmp_path / "ok_1.txt").exists()
    assert not (tmp_path / "ok_2.txt").exists()
    assert (tmp_path / "ok_3.txt").exists()


def test_local_scheduler_writes_per_task_logs(tmp_path: Path) -> None:
    scheduler = LocalScheduler()
    job = ArrayJob(
        name="loggy",
        workdir=tmp_path,
        array_size=2,
        max_concurrent=2,
        cpus_per_task=1,
        command_template='echo "stdout-${TASK_ID}"; echo "stderr-${TASK_ID}" >&2',
    )
    result = _run(scheduler, job)
    assert result.success

    log_dir = tmp_path / "logs" / "local" / "loggy"
    assert (log_dir / "task-0001.out").read_text().strip() == "stdout-1"
    assert (log_dir / "task-0001.err").read_text().strip() == "stderr-1"
    assert (log_dir / "task-0002.out").read_text().strip() == "stdout-2"
    assert (log_dir / "task-0002.err").read_text().strip() == "stderr-2"


def test_local_scheduler_honours_max_concurrent(tmp_path: Path) -> None:
    scheduler = LocalScheduler()
    # Each task appends its index to a shared counter file while sleeping
    # briefly, so we can observe concurrency. We just verify the snapshot
    # never exceeds max_concurrent running tasks.
    template = "sleep 0.2"
    job = ArrayJob(
        name="cap",
        workdir=tmp_path,
        array_size=6,
        max_concurrent=2,
        cpus_per_task=1,
        command_template=template,
    )
    handle = scheduler.submit_array(job)

    observed_max = 0
    while True:
        snap = scheduler.status(handle)
        observed_max = max(observed_max, snap.running_count)
        if snap.all_terminal:
            break

    result = scheduler.wait(handle)
    assert result.success
    assert observed_max <= 2


def test_local_scheduler_env_setup_runs_first(tmp_path: Path) -> None:
    scheduler = LocalScheduler()
    job = ArrayJob(
        name="envsetup",
        workdir=tmp_path,
        array_size=1,
        max_concurrent=1,
        cpus_per_task=1,
        env_setup=["export GREETING=hello"],
        command_template='echo "$GREETING $TASK_ID" > out.txt',
    )
    result = _run(scheduler, job)
    assert result.success
    assert (tmp_path / "out.txt").read_text().strip() == "hello 1"


def test_local_scheduler_cancel_pending_tasks(tmp_path: Path) -> None:
    scheduler = LocalScheduler()
    job = ArrayJob(
        name="cancel",
        workdir=tmp_path,
        array_size=4,
        max_concurrent=1,
        cpus_per_task=1,
        command_template="sleep 5",
    )
    handle = scheduler.submit_array(job)
    scheduler.cancel(handle)
    snap = scheduler.status(handle)
    assert snap.all_terminal
    assert all(s is TaskState.CANCELLED for s in snap.task_states)


def test_array_job_validation() -> None:
    with pytest.raises(ValueError):
        ArrayJob(
            name="bad",
            workdir=Path("."),
            array_size=0,
            max_concurrent=1,
            cpus_per_task=1,
            command_template="true",
        )
    with pytest.raises(ValueError):
        ArrayJob(
            name="bad",
            workdir=Path("."),
            array_size=1,
            max_concurrent=0,
            cpus_per_task=1,
            command_template="true",
        )
    with pytest.raises(ValueError):
        ArrayJob(
            name="bad",
            workdir=Path("."),
            array_size=1,
            max_concurrent=1,
            cpus_per_task=0,
            command_template="true",
        )
