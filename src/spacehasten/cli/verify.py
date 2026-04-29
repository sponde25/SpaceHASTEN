"""End-to-end ``spacehasten verify`` command (Session 15).

Replaces legacy ``verify_spacehasten.py``. Runs the same six checks via
the new stage APIs against a temp workspace:

1. ``pigz`` available on ``PATH``
2. Scheduler reachable (``sbatch`` / ``LocalScheduler``) — submits a
   trivial 1-task ``echo`` array
3. Clustering — five hand-picked SMILES through
   :func:`spacehasten.stages.clustering.cluster`
4. Docking — :func:`spacehasten.stages.seeds.import_seeds` +
   :func:`spacehasten.stages.docking.dock` over ``examples.smi``,
   ``test_dock.in`` and ``grid-test_dock.zip``
5. Training — :func:`spacehasten.stages.training.train` against the
   docked DB
6. BioSolveIT — local SpaceLight + FTrees invocations against the
   default chemical space using ``example.smi``

Each check is independently selectable / skippable. Failures are
collected and the command exits non-zero only at the end.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from spacehasten.config.properties import PropertyRanges
from spacehasten.config.settings import Settings
from spacehasten.core.db import Database
from spacehasten.scheduler.base import ArrayJob, Scheduler
from spacehasten.stages import clustering as _clustering
from spacehasten.stages import docking as _docking
from spacehasten.stages import seeds as _seeds
from spacehasten.stages import training as _training
from spacehasten.tools.ftrees import FTreesAdapter
from spacehasten.tools.spacelight import SpacelightAdapter
from spacehasten.workspace.layout import WorkDir

from ._common import scheduler_from_args, settings_from_args, setup_logging

logger = logging.getLogger(__name__)


CHECK_NAMES: tuple[str, ...] = (
    "pigz",
    "scheduler",
    "clustering",
    "docking",
    "training",
    "biosolveit",
)


# --------------------------------------------------------------------------- #
# CLI wiring                                                                  #
# --------------------------------------------------------------------------- #


def add_verify_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach verify-specific options to ``parser``."""
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Workspace root for the verify run "
        "(default: $HOME/SPACEHASTEN/VERIFY-<version>).",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=None,
        help="Directory containing examples.smi, example.smi, example.csv, "
        "test_dock.in and grid-test_dock.zip (default: this package's "
        "install root).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=CHECK_NAMES,
        default=None,
        help="Run only these checks.",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=CHECK_NAMES,
        default=(),
        help="Skip these checks.",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Do not delete the verify workdir on success.",
    )


# --------------------------------------------------------------------------- #
# Fixture resolution                                                          #
# --------------------------------------------------------------------------- #


def _resolve_fixture(name: str, fixtures_dir: Path | None) -> Path:
    """Resolve a verify fixture file.

    Searches: explicit ``--fixtures-dir`` → repo root (development) →
    ``site-packages/spacehasten/_fixtures``.
    """
    candidates: list[Path] = []
    if fixtures_dir is not None:
        candidates.append(Path(fixtures_dir) / name)
    # Repo root: walk up from this file until we find a pyproject.toml.
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            candidates.append(parent / name)
            break
    # Installed-package data: src/spacehasten/_fixtures/<name>
    candidates.append(Path(__file__).resolve().parent.parent / "_fixtures" / name)

    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"verify fixture {name!r} not found; searched {[str(c) for c in candidates]!r}. "
        "Pass --fixtures-dir explicitly."
    )


# --------------------------------------------------------------------------- #
# Check infrastructure                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str
    duration_s: float


def _run_checks(
    selected: Iterable[str],
    callbacks: dict[str, Callable[[], str]],
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for name in selected:
        fn = callbacks[name]
        print(f"[verify] {name}: running ...", flush=True)
        t0 = time.monotonic()
        try:
            msg = fn()
            ok = True
        except Exception as exc:  # noqa: BLE001 — we want every failure caught
            msg = f"{type(exc).__name__}: {exc}"
            ok = False
            logger.exception("verify check %s failed", name)
        dur = time.monotonic() - t0
        status = "ok" if ok else "FAIL"
        print(f"[verify] {name}: {status} ({dur:.1f}s) {msg}", flush=True)
        results.append(CheckResult(name=name, ok=ok, message=msg, duration_s=dur))
    return results


# --------------------------------------------------------------------------- #
# Individual checks                                                           #
# --------------------------------------------------------------------------- #


def _check_pigz() -> str:
    path = shutil.which("pigz")
    if path is None:
        raise RuntimeError("pigz not found on PATH")
    return f"pigz at {path}"


def _check_scheduler(workdir: WorkDir, scheduler: Scheduler) -> str:
    """Submit a trivial array of size 1 and wait for completion."""
    smoke_dir = workdir.root / "scheduler_smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    job = ArrayJob(
        name="verify_smoke",
        workdir=smoke_dir,
        array_size=1,
        max_concurrent=1,
        cpus_per_task=1,
        command_template='echo "Hello from SpaceHASTEN verify (task ${TASK_ID})"',
    )
    handle = scheduler.submit_array(job)
    result = scheduler.wait(handle)
    if not result.success:
        raise RuntimeError(
            f"scheduler smoke job {handle.job_id} failed; "
            f"failed indices: {result.failed_indices}"
        )
    return f"scheduler={type(scheduler).__name__} job_id={handle.job_id}"


def _check_clustering(
    workdir: WorkDir, scheduler: Scheduler, settings: Settings
) -> str:
    cdir = workdir.root / "clustering_smoke"
    cdir.mkdir(parents=True, exist_ok=True)
    db_path = cdir / "verify_clustering.dbsh"
    if db_path.exists():
        db_path.unlink()
    db = Database(db_path)
    db.create_schema()
    smiles_pool = [
        ("CCO", "ethanol"),
        ("CCN", "ethylamine"),
        ("CCC", "propane"),
        ("CCF", "fluoroethane"),
        ("CCCl", "chloroethane"),
    ]
    for i, (smi, name) in enumerate(smiles_pool):
        db.insert_seed_undocked(f"hv{i}", smi, name)
    db.commit()
    sub_workdir = WorkDir(root=cdir)
    n = _clustering.cluster(db, sub_workdir, scheduler, settings)
    db.close()
    if n < 1:
        raise RuntimeError("clustering produced no rows")
    return f"clustered {n} compounds"


def _check_docking_and_training(
    workdir: WorkDir,
    scheduler: Scheduler,
    settings: Settings,
    fixtures_dir: Path | None,
    *,
    do_training: bool,
) -> tuple[str, str | None]:
    """Run the docking and (optionally) training stage.

    Returns ``(docking_msg, training_msg_or_None)``. We bundle them into
    one orchestration to share a ``.dbsh`` (training reads dock_score
    from the same DB).
    """
    smi = _resolve_fixture("examples.smi", fixtures_dir)
    inp = _resolve_fixture("test_dock.in", fixtures_dir)
    grid = _resolve_fixture("grid-test_dock.zip", fixtures_dir)

    ddir = workdir.root / "dock_smoke"
    ddir.mkdir(parents=True, exist_ok=True)
    db_path = ddir / "verify_dock.dbsh"
    if db_path.exists():
        db_path.unlink()
    db = Database(db_path)
    sub_workdir = WorkDir(root=ddir)
    sub_workdir.logs_dir().mkdir(parents=True, exist_ok=True)

    n = _seeds.import_seeds(
        db,
        sub_workdir,
        smi_path=smi,
        dock_params_path=inp,
        dock_grid_path=grid,
        props=PropertyRanges(),
        auto_train=False,
    )
    if n < 1:
        db.close()
        raise RuntimeError(f"no seeds parsed from {smi}")

    iteration = _docking.dock(
        db, sub_workdir, scheduler, settings,
        top_n=n, strategy="greedy", cpus=1,
    )
    docked = list(
        db.connection.execute(
            "SELECT spacehastenid FROM data WHERE dock_score IS NOT NULL"
        )
    )
    if not docked:
        db.close()
        raise RuntimeError("docking succeeded but no dock_score values were written")
    dock_msg = f"iter={iteration}; {len(docked)} compounds with dock_score"

    train_msg: str | None = None
    if do_training:
        version = _training.train(
            db, sub_workdir, scheduler, settings, cutoff=10.0,
        )
        model_path = sub_workdir.model_dir(version) / "model_0" / "pytorch_model.bin"
        if not model_path.exists():
            db.close()
            raise RuntimeError(f"training did not produce {model_path}")
        train_msg = f"model_version={version} at {model_path}"

    db.close()
    return dock_msg, train_msg


def _check_biosolveit(
    workdir: WorkDir,
    settings: Settings,
    fixtures_dir: Path | None,
) -> str:
    """Run SpaceLight and FTrees once against the default chemical space."""
    space = settings.paths.spaces_file_default
    if not space or not Path(space).exists():
        raise RuntimeError(f"default chemical space not found: {space!r}")
    query_smi = _resolve_fixture("example.smi", fixtures_dir)

    bdir = workdir.root / "biosolveit_smoke"
    if bdir.exists():
        shutil.rmtree(bdir)
    bdir.mkdir(parents=True)

    sl = SpacelightAdapter(exe=settings.paths.exe_spacelight_default)
    ft = FTreesAdapter(exe=settings.paths.exe_ftrees_default)

    sl_out = bdir / "spacelightresult.csv"
    sl_cmd = sl.command_for(
        str(query_smi),
        space,
        sl_out,
        max_results=100,
        similarity=0.5,
        threads=1,
    )
    sl_proc = subprocess.run(sl_cmd, capture_output=True, text=True)
    # SpaceLight writes <stem>_1.csv per query (legacy behaviour).
    sl_actual = bdir / "spacelightresult_1.csv"
    if sl_proc.returncode != 0 or not sl_actual.exists():
        raise RuntimeError(
            f"SpaceLight failed (rc={sl_proc.returncode}); "
            f"stderr={sl_proc.stderr.strip()!r}"
        )
    sl_lines = sum(1 for _ in sl_actual.open("rt"))

    ft_out = bdir / "ftreesresult.csv"
    ft_cmd = ft.command_for(
        str(query_smi),
        space,
        ft_out,
        max_results=100,
        similarity=0.9,
        threads=1,
    )
    ft_proc = subprocess.run(ft_cmd, capture_output=True, text=True)
    ft_actual = bdir / "ftreesresult_1.csv"
    if ft_proc.returncode != 0 or not ft_actual.exists():
        raise RuntimeError(
            f"FTrees failed (rc={ft_proc.returncode}); "
            f"stderr={ft_proc.stderr.strip()!r}"
        )
    ft_lines = sum(1 for _ in ft_actual.open("rt"))

    return (
        f"spacelight rows={sl_lines} ({sl_actual.name}); "
        f"ftrees rows={ft_lines} ({ft_actual.name})"
    )


# --------------------------------------------------------------------------- #
# Top-level entry point (called from cli.main._cmd_verify)                    #
# --------------------------------------------------------------------------- #


def run_verify(args: argparse.Namespace) -> int:
    """Implement ``spacehasten verify``."""
    settings = settings_from_args(args)

    # Pick a workdir. Default mirrors legacy: $HOME/SPACEHASTEN/VERIFY-<ver>.
    if args.workdir is not None:
        root = Path(args.workdir).expanduser()
    else:
        from spacehasten import __version__

        root = (
            Path.home() / "SPACEHASTEN" / f"VERIFY-{__version__.replace('.', '')}"
        )
    if root.exists():
        shutil.rmtree(root)
    workdir = WorkDir.bootstrap(root, name=root.name)
    setup_logging(workdir, args)
    logger.info("verify workdir: %s", root)
    print(f"[verify] workdir: {root}")

    scheduler = scheduler_from_args(args, settings)
    print(f"[verify] scheduler: {type(scheduler).__name__}")

    selected = list(args.only) if args.only else list(CHECK_NAMES)
    skip = set(args.skip or ())
    selected = [name for name in selected if name not in skip]

    # Build the callback table. Docking and training share a DB so we
    # treat them as a tuple here and split the message back out.
    docking_msg_holder: dict[str, str] = {}

    def _cb_docking() -> str:
        do_train = "training" in selected
        dock_msg, train_msg = _check_docking_and_training(
            workdir, scheduler, settings, args.fixtures_dir, do_training=do_train,
        )
        if train_msg is not None:
            docking_msg_holder["training"] = train_msg
        return dock_msg

    def _cb_training() -> str:
        msg = docking_msg_holder.get("training")
        if msg is None:
            # Training was selected without docking; run docking implicitly.
            dock_msg, train_msg = _check_docking_and_training(
                workdir, scheduler, settings, args.fixtures_dir, do_training=True,
            )
            assert train_msg is not None
            return f"(implicit docking ran first: {dock_msg}) {train_msg}"
        return msg

    callbacks: dict[str, Callable[[], str]] = {
        "pigz": _check_pigz,
        "scheduler": lambda: _check_scheduler(workdir, scheduler),
        "clustering": lambda: _check_clustering(workdir, scheduler, settings),
        "docking": _cb_docking,
        "training": _cb_training,
        "biosolveit": lambda: _check_biosolveit(workdir, settings, args.fixtures_dir),
    }

    # Run docking before training so the shared-DB flow works without surprises.
    order = [c for c in CHECK_NAMES if c in selected]
    results = _run_checks(order, callbacks)

    failures = [r for r in results if not r.ok]
    print()
    print("=" * 60)
    print("verify summary")
    print("=" * 60)
    for r in results:
        marker = "OK  " if r.ok else "FAIL"
        print(f"  [{marker}] {r.name:<12} {r.duration_s:6.1f}s  {r.message}")
    if failures:
        print(f"\n{len(failures)} check(s) failed: "
              f"{', '.join(r.name for r in failures)}")
        return 1

    print("\nAll checks passed.")
    if not args.keep_workdir:
        shutil.rmtree(root, ignore_errors=True)
        print(f"[verify] removed workdir {root}")
    else:
        print(f"[verify] kept workdir {root}")
    return 0


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Standalone entry (mainly for ad-hoc testing)."""
    parser = argparse.ArgumentParser(prog="spacehasten-verify")
    from ._common import add_global_options

    add_global_options(parser)
    add_verify_arguments(parser)
    args = parser.parse_args(argv)
    return run_verify(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
