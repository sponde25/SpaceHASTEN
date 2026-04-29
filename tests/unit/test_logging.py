"""Tests for ``spacehasten.workspace.logging_setup``."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import pytest

from spacehasten.workspace import WorkDir, configure_logging, stage_log_context
from spacehasten.workspace.logging_setup import (
    _CONSOLE_HANDLER_ATTR,
    _MASTER_HANDLER_ATTR,
)


@pytest.fixture(autouse=True)
def _clean_root_logger() -> None:
    """Snapshot/restore root handlers around each test so the tests don't
    bleed handlers into the rest of the suite."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    for handler in list(root.handlers):
        if handler not in saved_handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(saved_level)


def test_configure_logging_attaches_master_and_console(tmp_path: Path) -> None:
    wd = WorkDir.bootstrap(tmp_path / "run")
    configure_logging(wd)
    root = logging.getLogger()

    master = [h for h in root.handlers if getattr(h, _MASTER_HANDLER_ATTR, False)]
    console = [h for h in root.handlers if getattr(h, _CONSOLE_HANDLER_ATTR, False)]
    assert len(master) == 1
    assert len(console) == 1
    assert isinstance(master[0], logging.handlers.RotatingFileHandler)
    assert master[0].baseFilename == str(wd.logs_dir() / "spacehasten.log")


def test_configure_logging_writes_to_master(tmp_path: Path) -> None:
    wd = WorkDir.bootstrap(tmp_path / "run")
    configure_logging(wd, console=False)
    logging.getLogger("spacehasten.test").info("hello-master")
    for h in logging.getLogger().handlers:
        h.flush()
    log_text = (wd.logs_dir() / "spacehasten.log").read_text()
    assert "hello-master" in log_text


def test_configure_logging_idempotent(tmp_path: Path) -> None:
    wd = WorkDir.bootstrap(tmp_path / "run")
    configure_logging(wd)
    configure_logging(wd)
    root = logging.getLogger()
    master = [h for h in root.handlers if getattr(h, _MASTER_HANDLER_ATTR, False)]
    console = [h for h in root.handlers if getattr(h, _CONSOLE_HANDLER_ATTR, False)]
    assert len(master) == 1
    assert len(console) == 1


def test_configure_logging_console_false_removes_console(tmp_path: Path) -> None:
    wd = WorkDir.bootstrap(tmp_path / "run")
    configure_logging(wd, console=True)
    configure_logging(wd, console=False)
    root = logging.getLogger()
    console = [h for h in root.handlers if getattr(h, _CONSOLE_HANDLER_ATTR, False)]
    assert console == []


def test_stage_log_context_writes_and_detaches(tmp_path: Path) -> None:
    wd = WorkDir.bootstrap(tmp_path / "run")
    configure_logging(wd, console=False)
    root = logging.getLogger()
    handlers_before = list(root.handlers)

    with stage_log_context(wd, "training") as log_path:
        assert log_path == wd.logs_dir() / "training-1.log"
        assert log_path.exists()
        assert len(root.handlers) == len(handlers_before) + 1
        logging.getLogger("spacehasten.stages.training").info("stage-msg-1")
        for h in root.handlers:
            h.flush()
        text = log_path.read_text()
        assert "stage-msg-1" in text

    # Handler removed after context exits.
    assert root.handlers == handlers_before


def test_stage_log_context_increments_index(tmp_path: Path) -> None:
    wd = WorkDir.bootstrap(tmp_path / "run")
    configure_logging(wd, console=False)
    with stage_log_context(wd, "simsearch") as p1:
        pass
    with stage_log_context(wd, "simsearch") as p2:
        pass
    assert p1.name == "simsearch-1.log"
    assert p2.name == "simsearch-2.log"


def test_stage_log_does_not_pollute_master_after_exit(tmp_path: Path) -> None:
    wd = WorkDir.bootstrap(tmp_path / "run")
    configure_logging(wd, console=False)
    with stage_log_context(wd, "docking"):
        logging.getLogger("spacehasten").info("inside-stage")
    logging.getLogger("spacehasten").info("after-stage")
    for h in logging.getLogger().handlers:
        h.flush()
    stage_text = (wd.logs_dir() / "docking-1.log").read_text()
    master_text = (wd.logs_dir() / "spacehasten.log").read_text()
    assert "inside-stage" in stage_text
    assert "after-stage" not in stage_text
    assert "inside-stage" in master_text
    assert "after-stage" in master_text
