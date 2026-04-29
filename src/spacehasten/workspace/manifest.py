"""Workspace JSON manifest.

The manifest is the source of truth for a workspace's stage history. It
records every CLI run (command line, args, timing, status) and the latest
state of each stage (cycle/iteration/version, scheduler job id).

Atomic save: write to a sibling tempfile, ``fsync``, then ``os.replace``.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class StageRecord(BaseModel):
    """Latest state for one stage (e.g. ``training``, ``simsearch``)."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    status: str = "pending"  # pending|running|completed|failed
    started_at: datetime | None = None
    ended_at: datetime | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    scheduler_job_id: str | None = None


class RunRecord(BaseModel):
    """One CLI invocation."""

    model_config = ConfigDict(extra="forbid")

    command: str
    args: list[str] = Field(default_factory=list)
    started_at: datetime
    ended_at: datetime | None = None
    status: str = "running"  # running|completed|failed


class ModelRecord(BaseModel):
    """On-disk registry entry for a trained model.

    The manifest is the source of truth for the model registry; the
    legacy ``models`` SQL table is kept only as a (now-empty) compatibility
    blob for older code paths.
    """

    model_config = ConfigDict(extra="forbid")

    version: int
    model_dir: str  # workspace-relative or absolute path
    recorded_at: datetime = Field(default_factory=_utcnow)


class Manifest(BaseModel):
    """Workspace manifest persisted as ``manifest.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    name: str
    created_at: datetime = Field(default_factory=_utcnow)
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    runs: list[RunRecord] = Field(default_factory=list)
    models: dict[str, ModelRecord] = Field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # I/O                                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: Path) -> Manifest:
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.model_validate(data)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        # Atomic write: tempfile in same dir + os.replace.
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, path)
        except Exception:
            # Best-effort cleanup of the tempfile on failure.
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

    # ------------------------------------------------------------------ #
    # Stage record helpers                                                #
    # ------------------------------------------------------------------ #

    def record_stage_start(
        self, stage: str, params: dict[str, Any] | None = None
    ) -> StageRecord:
        record = StageRecord(
            stage=stage,
            status="running",
            started_at=_utcnow(),
            params=dict(params or {}),
        )
        self.stages[stage] = record
        return record

    def record_stage_finish(
        self,
        stage: str,
        status: str,
        scheduler_job_id: str | None = None,
    ) -> StageRecord:
        record = self.stages.get(stage) or StageRecord(stage=stage)
        record.status = status
        record.ended_at = _utcnow()
        if scheduler_job_id is not None:
            record.scheduler_job_id = scheduler_job_id
        self.stages[stage] = record
        return record

    # ------------------------------------------------------------------ #
    # Run record helpers                                                  #
    # ------------------------------------------------------------------ #

    def record_run_start(self, command: str, args: list[str]) -> RunRecord:
        record = RunRecord(command=command, args=list(args), started_at=_utcnow())
        self.runs.append(record)
        return record

    def record_run_finish(self, status: str) -> RunRecord | None:
        if not self.runs:
            return None
        record = self.runs[-1]
        record.status = status
        record.ended_at = _utcnow()
        return record

    # ------------------------------------------------------------------ #
    # Model registry                                                      #
    # ------------------------------------------------------------------ #

    def record_model(self, version: int, model_dir: Path | str) -> ModelRecord:
        """Add or replace the registry entry for a trained model version."""
        record = ModelRecord(version=version, model_dir=str(model_dir))
        self.models[str(version)] = record
        return record

    def get_model(self, version: int) -> ModelRecord | None:
        return self.models.get(str(version))
