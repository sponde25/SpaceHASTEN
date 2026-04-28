"""Tests for the SLURM scheduler backend.

Network/cluster-free: ``subprocess.run`` is monkeypatched to a recorder.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from spacehasten.config.settings import Settings
from spacehasten.scheduler import (
    ArrayHandle,
    ArrayJob,
    SlurmScheduler,
    TaskState,
    make_scheduler,
)
from spacehasten.scheduler.local import LocalScheduler
from spacehasten.scheduler.slurm import _JOBID_RE  # noqa: F401  (sanity import)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


# --------------------------------------------------------------------------- #
# subprocess.run recorder                                                     #
# --------------------------------------------------------------------------- #


@dataclass
class FakeRun:
    """Records subprocess.run() calls and replays canned outputs."""

    responses: dict[str, subprocess.CompletedProcess[str]] = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)

    def add(self, prog: str, stdout: str = "", returncode: int = 0) -> None:
        self.responses[prog] = subprocess.CompletedProcess(
            args=[prog], returncode=returncode, stdout=stdout, stderr=""
        )

    def __call__(
        self,
        cmd: list[str],
        *args: Any,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        prog = cmd[0]
        if prog not in self.responses:
            raise AssertionError(f"unexpected subprocess call: {cmd!r}")
        resp = self.responses[prog]
        if kwargs.get("check") and resp.returncode != 0:
            raise subprocess.CalledProcessError(
                resp.returncode, cmd, output=resp.stdout, stderr=resp.stderr
            )
        return resp

    def calls_for(self, prog: str) -> Iterable[list[str]]:
        return [c for c in self.calls if c and c[0] == prog]


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> FakeRun:
    fake = FakeRun()
    monkeypatch.setattr("spacehasten.scheduler.slurm.subprocess.run", fake)
    return fake


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _basic_job(workdir: Path, **overrides: Any) -> ArrayJob:
    kwargs: dict[str, Any] = dict(
        name="search",
        workdir=workdir,
        array_size=4,
        max_concurrent=2,
        cpus_per_task=2,
        command_template='echo "task ${TASK_ID}" > out_${TASK_ID}.txt',
        env_setup=[
            "source /data/programs/oce/actoce",
            "conda activate chemprop-2.1.2",
        ],
    )
    kwargs.update(overrides)
    return ArrayJob(**kwargs)


# --------------------------------------------------------------------------- #
# Render snapshot                                                             #
# --------------------------------------------------------------------------- #


def test_render_matches_snapshot(tmp_path: Path, fake_run: FakeRun) -> None:
    scheduler = SlurmScheduler(partition="jobs")
    fake_run.add("sbatch", stdout="1234567\n")
    scheduler.submit_array(_basic_job(tmp_path))

    rendered = (tmp_path / "submit_search.sh").read_text(encoding="utf-8")
    normalised = rendered.replace(str(tmp_path), "/WORKDIR")

    expected_path = FIXTURES / "expected_submit_search.sh"
    if not expected_path.exists():
        expected_path.write_text(normalised, encoding="utf-8")
    assert normalised == expected_path.read_text(encoding="utf-8")


def test_render_train_gpu_snapshot(tmp_path: Path, fake_run: FakeRun) -> None:
    scheduler = SlurmScheduler(
        partition="gpu", gpu_parameter="--gres=gpu:1"
    )
    fake_run.add("sbatch", stdout="9999;cluster\n")
    job = _basic_job(
        tmp_path,
        name="train",
        array_size=1,
        max_concurrent=1,
        gpus=1,
        exclusive=True,
        command_template="python3 -m spacehasten.remote.train data.csv model_v1",
    )
    scheduler.submit_array(job)

    rendered = (tmp_path / "submit_train.sh").read_text(encoding="utf-8")
    normalised = rendered.replace(str(tmp_path), "/WORKDIR")

    expected_path = FIXTURES / "expected_submit_train.sh"
    if not expected_path.exists():
        expected_path.write_text(normalised, encoding="utf-8")
    assert normalised == expected_path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Submission                                                                  #
# --------------------------------------------------------------------------- #


def test_submit_calls_sbatch_parsable(tmp_path: Path, fake_run: FakeRun) -> None:
    scheduler = SlurmScheduler(partition="jobs")
    fake_run.add("sbatch", stdout="1234567\n")

    handle = scheduler.submit_array(_basic_job(tmp_path))

    assert handle.job_id == "1234567"
    assert handle.array_size == 4
    submit_path = tmp_path / "submit_search.sh"
    assert submit_path.exists()
    sbatch_calls = list(fake_run.calls_for("sbatch"))
    assert sbatch_calls == [["sbatch", "--parsable", str(submit_path)]]


def test_submit_parses_clustered_jobid(tmp_path: Path, fake_run: FakeRun) -> None:
    """sbatch may print "<jobid>;<cluster>"; we keep only the jobid."""
    scheduler = SlurmScheduler()
    fake_run.add("sbatch", stdout="42;cluster\n")
    handle = scheduler.submit_array(_basic_job(tmp_path))
    assert handle.job_id == "42"


def test_submit_dependency_chain_renders_afterok(
    tmp_path: Path, fake_run: FakeRun
) -> None:
    scheduler = SlurmScheduler()
    fake_run.add("sbatch", stdout="555\n")
    parent_a = ArrayHandle(
        job_id="111", name="prev_a", array_size=1, workdir=tmp_path
    )
    parent_b = ArrayHandle(
        job_id="222", name="prev_b", array_size=1, workdir=tmp_path
    )
    job = _basic_job(tmp_path, depends_on=[parent_a, parent_b])
    scheduler.submit_array(job)

    text = (tmp_path / "submit_search.sh").read_text(encoding="utf-8")
    assert "#SBATCH --dependency=afterok:111:222" in text


# --------------------------------------------------------------------------- #
# Status parsing                                                              #
# --------------------------------------------------------------------------- #


def test_status_parses_sacct_output(tmp_path: Path, fake_run: FakeRun) -> None:
    scheduler = SlurmScheduler()
    fake_run.add("sbatch", stdout="1234567\n")
    handle = scheduler.submit_array(_basic_job(tmp_path))

    fake_run.add(
        "sacct",
        stdout=(
            "1234567_1|COMPLETED|0:0\n"
            "1234567_2|FAILED|1:0\n"
            "1234567_3|RUNNING|0:0\n"
            "1234567_4|TIMEOUT|0:0\n"
        ),
    )

    snap = scheduler.status(handle)
    assert snap.task_states == (
        TaskState.COMPLETED,
        TaskState.FAILED,
        TaskState.RUNNING,
        TaskState.TIMEOUT,
    )
    assert snap.failed_indices == (2, 4)
    assert not snap.all_terminal  # task 3 still running


def test_status_treats_missing_tasks_as_pending(
    tmp_path: Path, fake_run: FakeRun
) -> None:
    scheduler = SlurmScheduler()
    fake_run.add("sbatch", stdout="100\n")
    handle = scheduler.submit_array(_basic_job(tmp_path))

    # sacct may report only the array head before tasks expand.
    fake_run.add("sacct", stdout="100_[1-4]|PENDING|0:0\n")
    snap = scheduler.status(handle)
    assert all(s is TaskState.PENDING for s in snap.task_states)


def test_status_handles_cancelled_with_uid_suffix(
    tmp_path: Path, fake_run: FakeRun
) -> None:
    scheduler = SlurmScheduler()
    fake_run.add("sbatch", stdout="100\n")
    handle = scheduler.submit_array(_basic_job(tmp_path, array_size=1, max_concurrent=1))

    fake_run.add("sacct", stdout="100_1|CANCELLED by 1001|0:0\n")
    snap = scheduler.status(handle)
    assert snap.task_states == (TaskState.CANCELLED,)


# --------------------------------------------------------------------------- #
# Cancel                                                                      #
# --------------------------------------------------------------------------- #


def test_cancel_runs_scancel(tmp_path: Path, fake_run: FakeRun) -> None:
    scheduler = SlurmScheduler()
    fake_run.add("sbatch", stdout="777\n")
    fake_run.add("scancel", stdout="")
    handle = scheduler.submit_array(_basic_job(tmp_path))
    scheduler.cancel(handle)
    assert ["scancel", "777"] in fake_run.calls


# --------------------------------------------------------------------------- #
# wait()                                                                      #
# --------------------------------------------------------------------------- #


def test_wait_polls_until_terminal(
    tmp_path: Path, fake_run: FakeRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduler = SlurmScheduler()
    scheduler.initial_poll_interval = 0.0  # type: ignore[misc]
    scheduler.max_poll_interval = 0.0  # type: ignore[misc]
    monkeypatch.setattr("spacehasten.scheduler.base.time.sleep", lambda _s: None)

    fake_run.add("sbatch", stdout="42\n")
    handle = scheduler.submit_array(_basic_job(tmp_path, array_size=2, max_concurrent=2))

    # Cycle through canned sacct responses.
    sacct_responses = iter(
        [
            "42_1|RUNNING|0:0\n42_2|PENDING|0:0\n",
            "42_1|COMPLETED|0:0\n42_2|RUNNING|0:0\n",
            "42_1|COMPLETED|0:0\n42_2|COMPLETED|0:0\n",
        ]
    )

    def sacct_run(cmd: list[str], *a: Any, **kw: Any) -> subprocess.CompletedProcess[str]:
        if cmd[0] == "sacct":
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=next(sacct_responses), stderr=""
            )
        return fake_run(cmd, *a, **kw)

    monkeypatch.setattr("spacehasten.scheduler.slurm.subprocess.run", sacct_run)

    progress: list[int] = []
    result = scheduler.wait(handle, on_progress=lambda s: progress.append(s.completed_count))
    assert result.success
    assert result.task_states == (TaskState.COMPLETED, TaskState.COMPLETED)
    assert progress == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


def test_factory_local() -> None:
    s = make_scheduler("local", Settings())
    assert isinstance(s, LocalScheduler)


def test_factory_slurm() -> None:
    settings = Settings.load(
        cli_overrides={
            "slurm": {"slurm_partition": "jobs", "slurm_gpu_parameter": "--gres=gpu:1"}
        }
    )
    s = make_scheduler("slurm", settings)
    assert isinstance(s, SlurmScheduler)
    assert s.partition == "jobs"
    assert s.gpu_parameter == "--gres=gpu:1"


def test_factory_auto_local_without_sbatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("spacehasten.scheduler.factory.shutil.which", lambda _: None)
    s = make_scheduler("auto", Settings())
    assert isinstance(s, LocalScheduler)


def test_factory_auto_slurm_with_sbatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "spacehasten.scheduler.factory.shutil.which", lambda _: "/usr/bin/sbatch"
    )
    s = make_scheduler("auto", Settings())
    assert isinstance(s, SlurmScheduler)
