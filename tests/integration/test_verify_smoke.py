"""Fast integration smoke for ``cli.verify.run_verify``.

Exercises the structural parts of verify (argument parsing, workdir
bootstrap, summary printing) using only the cheap checks ``pigz`` and
``scheduler``, both of which run against the in-process
:class:`LocalScheduler`. Heavier checks (``clustering``, ``docking``,
``training``, ``biosolveit``) require external binaries / GPUs and are
covered by the full on-cluster verify run.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pytest

from spacehasten.cli._common import add_global_options
from spacehasten.cli.verify import add_verify_arguments, run_verify


def _make_args(tmp_path: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_global_options(parser)
    add_verify_arguments(parser)
    return parser.parse_args(
        [
            "--scheduler", "local",
            "--workdir", str(tmp_path / "verify"),
            "--only", "pigz", "scheduler",
            "--keep-workdir",
        ]
    )


def test_verify_smoke_pigz_and_scheduler(tmp_path: Path) -> None:
    if shutil.which("pigz") is None:
        pytest.skip("pigz not installed")

    args = _make_args(tmp_path)
    rc = run_verify(args)
    assert rc == 0

    # --keep-workdir: artefacts should still be on disk.
    workdir = Path(args.workdir)
    assert workdir.exists()
    # The scheduler check creates its own subdir under the verify root.
    assert (workdir / "scheduler").exists()
