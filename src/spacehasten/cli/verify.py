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
    smoke_dir = workdir.root / "scheduler"
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
    """Cluster five hand-picked SMILES through the clustering stage."""
    cdir = workdir.root / "clustering"
    cdir.mkdir(parents=True, exist_ok=True)
    db_path = cdir / "clustering.dbsh"
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
    sub_workdir = WorkDir.bootstrap(cdir, name="clustering")
    n = _clustering.cluster(db, sub_workdir, scheduler, settings)
    db.close()
    if n < 1:
        raise RuntimeError("clustering produced no rows")
    return f"clustered {n} compounds"


def _check_docking(
    workdir: WorkDir,
    scheduler: Scheduler,
    settings: Settings,
    fixtures_dir: Path | None,
) -> str:
    """Smoke-test the Glide docking pipeline over ``examples.smi``."""
    smi = _resolve_fixture("examples.smi", fixtures_dir)
    inp = _resolve_fixture("test_dock.in", fixtures_dir)
    grid = _resolve_fixture("grid-test_dock.zip", fixtures_dir)

    ddir = workdir.root / "docking"
    ddir.mkdir(parents=True, exist_ok=True)
    db_path = ddir / "docking.dbsh"
    if db_path.exists():
        db_path.unlink()
    db = Database(db_path)
    sub_workdir = WorkDir.bootstrap(ddir, name="docking")

    db.create_schema()
    db.store_dock_param(inp.read_bytes())
    db.store_dock_grid(grid.read_bytes())

    n = _seeds.import_seeds(
        db,
        smi_path=smi,
        props=PropertyRanges(),
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
    db.close()
    if not docked:
        raise RuntimeError("docking succeeded but no dock_score values were written")
    return f"iter={iteration}; {len(docked)} compounds with dock_score"


def _ingest_example_csv(db: Database, csv_path: Path) -> int:
    """Verify-only CSV-to-DB shim for the training check (Stage B.2.a).

    Reads ``example.csv`` (legacy three-column format
    ``smiles,smilesid,docking_score``, no header) and ingests each row
    via :meth:`Database.insert_seed_docked`. Rows whose tautomer hash
    cannot be computed are silently skipped (matches legacy
    behaviour). Returns the number of rows inserted.
    """
    from spacehasten.core.molecules import tautomer_hash

    inserted = 0
    with csv_path.open("rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            smiles, smilesid, score_str = parts[0], parts[1], parts[2]
            try:
                score = float(score_str)
            except ValueError:
                continue
            reghash = tautomer_hash(smiles)
            if reghash is None:
                continue
            db.insert_seed_docked(reghash, smiles, smilesid, dock_score=score)
            inserted += 1
    db.commit()
    return inserted


def _maybe_warn_no_cuda(model_dir: Path) -> None:
    """Emit a non-fatal CUDA warning by tailing chemprop's ``train.err`` log.

    Mirrors the legacy ``verify_spacehasten.py`` behaviour. The training
    stage writes ``train.err`` next to the model directory (lightning
    logs go to stderr).
    """
    candidates = [
        model_dir / "train.err",
        model_dir / "model_0" / "train.err",
        model_dir.parent / "train.err",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "GPU available: True" in text:
            return
        # Found a log without the GPU line — warn once.
        print(
            "WARNING: CUDA not available, training will be slower "
            f"(checked {path})",
            flush=True,
        )
        return
    # No log found at all — quietly skip; do not fail.


def _check_training(
    workdir: WorkDir,
    scheduler: Scheduler,
    settings: Settings,
    fixtures_dir: Path | None,
) -> str:
    """Train one chemprop model on ``example.csv`` (≈10k rows)."""
    csv_path = _resolve_fixture("example.csv", fixtures_dir)

    tdir = workdir.root / "training"
    tdir.mkdir(parents=True, exist_ok=True)
    db_path = tdir / "training.dbsh"
    if db_path.exists():
        db_path.unlink()
    db = Database(db_path)
    db.create_schema()
    sub_workdir = WorkDir.bootstrap(tdir, name="training")

    n_rows = _ingest_example_csv(db, csv_path)
    if n_rows < 1:
        db.close()
        raise RuntimeError(f"no training rows parsed from {csv_path}")

    version = _training.train(
        db, sub_workdir, scheduler, settings, cutoff=10.0,
    )
    model_dir = sub_workdir.model_dir(version)
    model_bin = model_dir / "model_0" / "pytorch_model.bin"
    db.close()
    if not model_bin.exists():
        raise RuntimeError(f"training did not produce {model_bin}")

    _maybe_warn_no_cuda(model_dir)
    return f"ingested {n_rows} rows; model_version={version} at {model_bin}"


def _check_biosolveit(
    workdir: WorkDir,
    settings: Settings,
    fixtures_dir: Path | None,
    config_path: Path | None,
) -> str:
    """Run SpaceLight and FTrees once against the default chemical space."""
    space = settings.paths.spaces_file_default
    if not space or not Path(space).exists():
        cfg_hint = (
            f"--config {config_path}"
            if config_path is not None
            else "--config <path-to-spacehasten.ini>"
        )
        raise RuntimeError(
            f"default chemical space not found: {space!r}; "
            f"pass {cfg_hint} to point at a site config that defines "
            "[Paths] spaces_file_default."
        )
    query_smi = _resolve_fixture("example.smi", fixtures_dir)

    bdir = workdir.root / "biosolveit"
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


def _autodiscover_config(args: argparse.Namespace) -> Path | None:
    """Resolve a ``spacehasten.ini`` location when ``--config`` is unset.

    Order:

    1. ``$SPACEHASTEN_INI``
    2. ``<fixtures_dir>/spacehasten.ini`` (if ``--fixtures-dir`` set)
    3. The directory above the resolved fixtures dir (an install root
       containing both fixtures and config).
    4. Repo root (walk up from this file to find ``pyproject.toml``).
    5. Current working directory.
    6. ``None`` — caller falls through to defaults.
    """
    import os

    env_val = os.environ.get("SPACEHASTEN_INI")
    if env_val:
        candidate = Path(env_val).expanduser()
        if candidate.exists():
            return candidate

    if args.fixtures_dir is not None:
        fdir = Path(args.fixtures_dir).expanduser().resolve()
        for cand in (fdir / "spacehasten.ini", fdir.parent / "spacehasten.ini"):
            if cand.exists():
                return cand

    # Walk up from this file to the repo root (same heuristic as
    # _resolve_fixture) and check for spacehasten.ini there.
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            cand = parent / "spacehasten.ini"
            if cand.exists():
                return cand
            break

    # Finally, check the current working directory.
    cwd_cand = Path.cwd() / "spacehasten.ini"
    if cwd_cand.exists():
        return cwd_cand

    return None


def run_verify(args: argparse.Namespace) -> int:
    """Implement ``spacehasten verify``."""
    # Auto-discover config when none was passed, before building Settings.
    if getattr(args, "config", None) is None:
        discovered = _autodiscover_config(args)
        if discovered is not None:
            print(f"[verify] auto-discovered config: {discovered}")
            args.config = discovered
        else:
            print(
                "[verify] WARNING: no --config and no spacehasten.ini "
                "found via $SPACEHASTEN_INI or --fixtures-dir; using "
                "package defaults (BioSolveIT defaults will likely be wrong)."
            )

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

    config_path = getattr(args, "config", None)

    callbacks: dict[str, Callable[[], str]] = {
        "pigz": _check_pigz,
        "scheduler": lambda: _check_scheduler(workdir, scheduler),
        "clustering": lambda: _check_clustering(workdir, scheduler, settings),
        "docking": lambda: _check_docking(
            workdir, scheduler, settings, args.fixtures_dir,
        ),
        "training": lambda: _check_training(
            workdir, scheduler, settings, args.fixtures_dir,
        ),
        "biosolveit": lambda: _check_biosolveit(
            workdir, settings, args.fixtures_dir, config_path,
        ),
    }

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
