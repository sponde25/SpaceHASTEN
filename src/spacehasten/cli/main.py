"""SpaceHASTEN command-line entry point.

Stitches the :mod:`spacehasten.stages` API behind argparse subcommands.
The console-script is registered in ``pyproject.toml`` as
``spacehasten = spacehasten.cli.main:main``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal

from spacehasten.config.properties import PropertyRanges
from spacehasten.stages import (
    archive,
    clustering,
    docking,
    export,
    prediction,
    seeds,
    simsearch,
    training,
)
from spacehasten.workspace.layout import WorkDir

from ._banner import banner, print_banner
from ._common import (
    add_global_options,
    open_db,
    scheduler_from_args,
    settings_from_args,
    setup_logging,
    workdir_from_args,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Parser construction                                                         #
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    epilog = """\
command groups:
  setup
    init                Bootstrap workspace, create DB, store docking settings
    pick-seeds          Sample and canonicalize seeds from a large collection

  workflows (recommended)
    seed-training       Import seeds → dock → train → cluster
    screening-cycle     [train] → (search → predict)×3 → dock per round
    export              Export results (csv, poses)

  manual stages (expert)
    import-seeds        Import seed compounds into the database (no training)
    dock                Dock the next batch of compounds
    train               Train one chemprop model
    search              Run one simsearch cycle
    predict             Predict scores for undocked rows
    cluster             Run sphere-exclusion clustering

  utilities
    status              Print workspace manifest summary
    resume              Resume the last interrupted run
    archive             Archive lifecycle (create/extract/restore/clean)
    verify              End-to-end smoke test
"""
    parser = argparse.ArgumentParser(
        prog="spacehasten",
        description=banner() + "\n\nIterative virtual-screening orchestrator.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_global_options(parser)
    parser.add_argument(
        "--quiet", action="store_true",
        help="Optional. Suppress the startup banner.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    _add_init(sub)
    _add_pick_seeds(sub)
    _add_seed_training(sub)
    _add_screening_cycle(sub)
    _add_import_seeds(sub)
    _add_train(sub)
    _add_predict(sub)
    _add_search(sub)
    _add_dock(sub)
    _add_cluster(sub)
    _add_export(sub)
    _add_archive(sub)
    _add_status(sub)
    _add_resume(sub)
    _add_verify(sub)
    return parser


def _add_init(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("init", help="Bootstrap a fresh workspace, create the database, and store docking settings.")
    p.add_argument("path", type=Path, help="Local root directory (should be on fast storage: /wrk or /fastwrk).")
    p.add_argument("--name", default=None,
                   help="Optional. Project name. Default: directory name.")
    p.add_argument(
        "--shared-root", type=Path, default=None,
        help="Optional. NFS directory for stage artefacts visible to compute nodes. "
        "Default: /data/$USER/SPACEHASTEN/<name>/.",
    )
    p.add_argument("--dock-params", type=Path, required=True,
                   help="Glide docking parameter .in file.")
    p.add_argument("--dock-grid", type=Path, required=True,
                   help="Glide grid .zip file.")
    p.set_defaults(func=_cmd_init)


def _add_pick_seeds(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "pick-seeds",
        help="Sample and canonicalize seeds from a large collection file.",
    )
    p.add_argument(
        "--seeds-file", type=Path, default=None,
        help="Optional. Path to seed collection (bz2/tsv). Default: from config.",
    )
    p.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output .smi file path.",
    )
    p.add_argument(
        "--n-seeds", type=int, required=True,
        help="Number of seeds to sample.",
    )
    p.add_argument(
        "--cores", type=int, default=None,
        help="Optional. Local cores for RDKit canonicalization. Default: all available CPUs.",
    )
    p.set_defaults(func=_cmd_pick_seeds)


def _add_import_seeds(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("import-seeds", help="Import seed compounds into the database (no training).")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--smi", type=Path, help="SMI file with undocked seed compounds.")
    grp.add_argument("--csv", type=Path, help="CSV file with pre-docked seed compounds.")
    p.add_argument("--props-toml", type=Path, default=None,
                   help="Optional. PropertyRanges TOML override. Default: built-in ranges.")
    p.add_argument("--processes", type=int, default=None,
                   help="Optional. Worker pool size for hashing. Default: all available CPUs.")
    p.set_defaults(func=_cmd_import_seeds)


def _add_seed_training(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "seed-training",
        help="Workflow: import seeds → dock → train → cluster.",
    )
    p.add_argument("--smi", type=Path, required=True, help="SMI file with undocked seed compounds.")
    p.add_argument("--dock-cpus", type=int, required=True, help="Number of concurrent docking tasks.")
    p.add_argument("--props-toml", type=Path, default=None,
                   help="Optional. PropertyRanges TOML override. Default: built-in ranges.")
    p.add_argument("--processes", type=int, default=None,
                   help="Optional. Worker pool size for hashing. Default: all available CPUs.")
    p.set_defaults(func=_cmd_seed_training)


def _add_train(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("train", help="Run one chemprop training round.")
    p.add_argument("--cutoff", type=float, default=10.0,
                   help="Optional. Docking score cutoff for including compounds in the training set. Default: 10.0.")
    p.set_defaults(func=_cmd_train)


def _add_predict(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("predict", help="Predict pred_score for every undocked row.")
    p.add_argument("--model-version", type=int, default=None,
                   help="Optional. Model version to use. Default: latest.")
    p.set_defaults(func=_cmd_predict)


def _add_search(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("search", help="Run one simsearch cycle.")
    p.add_argument("--source", choices=("docked", "predicted"), required=True,
                   help="Source compound pool: docked or predicted.")
    p.add_argument("--top-n", type=int, required=True,
                   help="Number of query compounds.")
    p.add_argument("--cpus", type=int, required=True,
                   help="Number of CPUs for simsearch tasks. Recommendation: max 250.")
    p.add_argument("--strategy", choices=("greedy", "clustering"), default="greedy",
                   help="Optional. Query acquisition strategy. Default: greedy.")
    p.add_argument("--space", type=Path, default=None,
                   help="Optional. BioSolveIT .space file override. Default: from config.")
    p.add_argument("--nnn", type=int, default=None,
                   help="Optional. Max results per query from chemical space. Default: from config (10000).")
    p.add_argument("--sim-spacelight", type=float, default=None,
                   help="Optional. SpaceLight similarity threshold. Default: from config.")
    p.add_argument("--sim-ftrees", type=float, default=None,
                   help="Optional. FTrees similarity threshold. Default: from config.")
    p.add_argument("--threads-per-task", type=int, default=2,
                   help="Optional. Threads per simsearch task. Default: 2.")
    p.add_argument("--cluster-after", action="store_true",
                   help="Optional. Run clustering after search completes.")
    p.set_defaults(func=_cmd_search)


def _add_dock(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("dock", help="Dock the next batch of compounds.")
    p.add_argument("--top-n", type=int, required=True,
                   help="Number of compounds to dock.")
    p.add_argument("--cpus", type=int, required=True,
                   help="Number of CPUs for docking tasks. Recommendation: max 250.")
    p.add_argument("--strategy", choices=("greedy", "clustering"), default="greedy",
                   help="Optional. Acquisition strategy for choosing which compounds to dock. Default: greedy.")
    p.set_defaults(func=_cmd_dock)


def _add_cluster(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("cluster", help="Run one sphere-exclusion clustering round.")
    p.set_defaults(func=_cmd_cluster)


def _add_screening_cycle(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "screening-cycle",
        help="Workflow: [train] → (search → predict)×3 → dock per round.",
    )
    p.add_argument("--simsearch-top-n", type=int, required=True,
                   help="Number of simsearch queries. Recommendation: 1000.")
    p.add_argument("--simsearch-cpu", type=int, required=True,
                   help="Number of CPUs for simsearch tasks. Recommendation: max 250.")
    p.add_argument("--dock-top-n", type=int, required=True,
                   help="Number of compounds to dock per round. Recommendation: 1000000 (1M).")
    p.add_argument("--dock-cpus", type=int, required=True,
                   help="Number of CPUs for docking tasks. Recommendation: max 250.")
    p.add_argument("--rounds", type=int, default=1,
                   help="Optional. Number of screening cycle rounds. Default: 1.")
    p.add_argument("--strategy", choices=("greedy", "clustering"), default="greedy",
                   help="Optional. Acquisition strategy for choosing which compounds to dock. Default: greedy.")
    p.add_argument("--space", type=Path, default=None,
                   help="Optional. BioSolveIT .space file override. Default: from config.")
    p.set_defaults(func=_cmd_screening_cycle)


def _add_export(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("export", help="Export results.")
    sub2 = p.add_subparsers(dest="export_kind", required=True)

    csv_p = sub2.add_parser("csv", help="Export docking results as CSV.")
    csv_p.add_argument("--cutoff", type=float, required=True,
                       help="Docking score cutoff for export.")
    csv_p.add_argument("--output", type=Path, required=True,
                       help="Output CSV file path.")
    csv_p.set_defaults(func=_cmd_export_csv)

    poses_p = sub2.add_parser("poses", help="Export Maestro pose file.")
    poses_p.add_argument("--cutoff", type=float, required=True,
                         help="Docking score cutoff for export.")
    poses_p.add_argument("--output", type=Path, required=True,
                         help="Output Maestro .mae file path.")
    poses_p.add_argument("--iteration", type=int, default=None,
                         help="Optional. Limit to a specific docking iteration. Default: all iterations.")
    poses_p.set_defaults(func=_cmd_export_poses)


def _add_archive(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("archive", help="Archive lifecycle operations.")
    sub2 = p.add_subparsers(dest="archive_op", required=True)

    cre = sub2.add_parser("create", help="Tar the workspace root.")
    cre.add_argument("--bundle", action="store_true",
                     help="Optional. Produce a single .tgz bundle.")
    cre.set_defaults(func=_cmd_archive_create)

    ext = sub2.add_parser("extract", help="Inverse of `archive create --bundle`.")
    ext.add_argument("--archive", type=Path, required=True,
                     help="Path to the .tgz bundle to extract.")
    ext.add_argument("--target", type=Path, required=True,
                     help="Target directory for extraction.")
    ext.set_defaults(func=_cmd_archive_extract)

    res = sub2.add_parser("restore", help="Inverse of `archive create` (.archived-spacehasten).")
    res.add_argument("--archive", type=Path, required=True,
                     help="Path to .archived-spacehasten directory.")
    res.add_argument("--target", type=Path, required=True,
                     help="Target workspace directory.")
    res.set_defaults(func=_cmd_archive_restore)

    clean = sub2.add_parser("clean", help="Remove regenerable scratch dirs.")
    clean.set_defaults(func=_cmd_archive_clean)


def _add_status(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("status", help="Print workspace manifest summary.")
    p.set_defaults(func=_cmd_status)


def _add_resume(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("resume", help="Resume the last interrupted run (alias of status).")
    p.set_defaults(func=_cmd_resume)


def _add_verify(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    from .verify import add_verify_arguments

    p = sub.add_parser("verify", help="End-to-end smoke test (Session 15).")
    add_verify_arguments(p)
    p.set_defaults(func=_cmd_verify)


# --------------------------------------------------------------------------- #
# Subcommand implementations                                                  #
# --------------------------------------------------------------------------- #


def _cmd_init(args: argparse.Namespace) -> int:
    from spacehasten.core.db import Database

    root = Path(args.path).resolve()
    name = args.name or root.name
    settings = settings_from_args(args)

    shared_root: Path | None = args.shared_root
    if shared_root is None:
        shared_root = settings.compute_shared_root(name)
    shared_root = shared_root.resolve()

    if shared_root.exists():
        logger.error(
            "Shared directory already exists: %s\n"
            "Pick a different project name or specify a different "
            "--shared-root path.",
            shared_root,
        )
        return 1

    workdir = WorkDir.bootstrap(root, name=name, shared_root=shared_root)
    setup_logging(workdir, args)
    workdir.warn_if_wrong_disk()

    with Database(workdir.dbsh()) as db:
        db.create_schema()
        db.store_dock_param(Path(args.dock_params).read_bytes())
        db.store_dock_grid(Path(args.dock_grid).read_bytes())

    logger.info("Workspace initialised: root=%s  shared=%s", root, shared_root)
    print(f"Initialised workspace at {root}")
    print(f"  local root (DB + logs): {root}")
    print(f"  shared root (stages):   {shared_root}")
    return 0


def _cmd_pick_seeds(args: argparse.Namespace) -> int:
    from spacehasten.stages.pick_seeds import pick_seeds

    workdir = workdir_from_args(args)
    setup_logging(workdir, args)
    settings = settings_from_args(args)

    seeds_file = args.seeds_file or Path(settings.paths.seeds_file_default)
    n_seeds = args.n_seeds
    cores = args.cores or os.cpu_count() or 1

    n = pick_seeds(
        seeds_file=seeds_file,
        output=args.output,
        n_seeds=n_seeds,
        n_cores=cores,
    )
    print(f"Wrote {n} seeds to {args.output}")
    return 0


def _cmd_import_seeds(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    workdir.logs_dir().mkdir(parents=True, exist_ok=True)
    setup_logging(workdir, args)

    props = (
        PropertyRanges.from_toml(args.props_toml)
        if args.props_toml is not None
        else PropertyRanges()
    )

    with open_db(args) as db:
        n = seeds.import_seeds(
            db,
            smi_path=args.smi,
            csv_path=args.csv,
            props=props,
            processes=args.processes,
        )
    print(f"Imported {n} seeds")
    return 0


def _cmd_seed_training(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    workdir.logs_dir().mkdir(parents=True, exist_ok=True)
    setup_logging(workdir, args)
    settings = settings_from_args(args)
    scheduler = scheduler_from_args(args, settings)

    props = (
        PropertyRanges.from_toml(args.props_toml)
        if args.props_toml is not None
        else PropertyRanges()
    )

    with open_db(args) as db:
        n = seeds.import_seeds(
            db,
            smi_path=args.smi,
            props=props,
            processes=args.processes,
        )
        logger.info("Imported %d seeds; starting dock → train → cluster", n)
        docking.dock(
            db, workdir, scheduler, settings,
            top_n=n,
            strategy="greedy",
            cpus=args.dock_cpus,
        )
        training.train(
            db, workdir, scheduler, settings,
        )
        clustering.cluster(db, workdir, scheduler, settings)

    print(f"Seed-training complete ({n} seeds imported, docked, trained, clustered)")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    setup_logging(workdir, args)
    settings = settings_from_args(args)
    scheduler = scheduler_from_args(args, settings)
    with open_db(args) as db:
        version = training.train(db, workdir, scheduler, settings, cutoff=args.cutoff)
    print(f"Trained model v{version}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    setup_logging(workdir, args)
    settings = settings_from_args(args)
    scheduler = scheduler_from_args(args, settings)
    with open_db(args) as db:
        version = args.model_version
        if version is None:
            version = db.latest_model_version()
            if version is None:
                raise SystemExit("error: no trained model; run `spacehasten train` first")
        n = prediction.predict_undocked(
            db, workdir, scheduler, settings,
            model_version=version,
        )
    print(f"Updated pred_score for {n} rows (model v{version})")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    setup_logging(workdir, args)
    settings = settings_from_args(args)
    scheduler = scheduler_from_args(args, settings)
    with open_db(args) as db:
        cycle = simsearch.simsearch(
            db, workdir, scheduler, settings,
            source=args.source,
            strategy=args.strategy,
            top_n=args.top_n,
            space=args.space,
            nnn=args.nnn,
            sim_spacelight=args.sim_spacelight,
            sim_ftrees=args.sim_ftrees,
            cpu=args.cpus,
            threads_per_task=args.threads_per_task,
            cluster_after=args.cluster_after,
        )
    print(f"Simsearch cycle {cycle} complete")
    return 0


def _cmd_dock(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    setup_logging(workdir, args)
    settings = settings_from_args(args)
    scheduler = scheduler_from_args(args, settings)
    with open_db(args) as db:
        iteration = docking.dock(
            db, workdir, scheduler, settings,
            top_n=args.top_n,
            strategy=args.strategy,
            cpus=args.cpus,
        )
    print(f"Dock iteration {iteration} complete")
    return 0


def _cmd_cluster(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    setup_logging(workdir, args)
    settings = settings_from_args(args)
    scheduler = scheduler_from_args(args, settings)
    with open_db(args) as db:
        n = clustering.cluster(db, workdir, scheduler, settings)
    print(f"Clustered {n} compounds")
    return 0


def _cmd_screening_cycle(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    setup_logging(workdir, args)
    settings = settings_from_args(args)
    scheduler = scheduler_from_args(args, settings)

    strategy: Literal["greedy", "clustering"] = args.strategy
    with open_db(args) as db:
        for round_n in range(1, args.rounds + 1):
            logger.info("Screening round %d/%d", round_n, args.rounds)

            # Train if there are newly docked compounds from a previous
            # screening cycle (i.e. not the very first round ever run).
            if db.latest_dock_iteration() is not None and db.latest_dock_iteration() > 0:
                logger.info("Training on newly docked data before round %d", round_n)
                training.train(db, workdir, scheduler, settings)

            # search(docked) → predict
            simsearch.simsearch(
                db, workdir, scheduler, settings,
                source="docked", strategy=strategy,
                top_n=args.simsearch_top_n,
                space=args.space, cpu=args.simsearch_cpu,
            )
            prediction.predict_undocked(
                db, workdir, scheduler, settings,
                model_version=db.latest_model_version(),
            )

            # (search(predicted) → predict) × 2
            for _ in range(2):
                simsearch.simsearch(
                    db, workdir, scheduler, settings,
                    source="predicted", strategy=strategy,
                    top_n=args.simsearch_top_n,
                    space=args.space, cpu=args.simsearch_cpu,
                )
                prediction.predict_undocked(
                    db, workdir, scheduler, settings,
                    model_version=db.latest_model_version(),
                )

            # dock
            docking.dock(
                db, workdir, scheduler, settings,
                top_n=args.dock_top_n,
                strategy=strategy,
                cpus=args.dock_cpus,
            )
    print(f"Completed {args.rounds} screening round(s)")
    return 0


def _cmd_export_csv(args: argparse.Namespace) -> int:
    setup_logging(workdir_from_args(args), args)
    with open_db(args) as db:
        n = export.export_csv(db, args.output, cutoff=args.cutoff)
    print(f"Exported {n} rows to {args.output}")
    return 0


def _cmd_export_poses(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    setup_logging(workdir, args)
    settings = settings_from_args(args)
    scheduler = scheduler_from_args(args, settings)
    with open_db(args) as db:
        out = export.export_poses(
            db, workdir, args.output,
            cutoff=args.cutoff,
            iteration=args.iteration,
            settings=settings,
            scheduler=scheduler,
        )
    print(f"Wrote poses to {out}")
    return 0


def _cmd_archive_create(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    setup_logging(workdir, args)
    out = archive.archive_create(workdir, bundle=args.bundle)
    print(f"Created archive {out}")
    return 0


def _cmd_archive_extract(args: argparse.Namespace) -> int:
    target = WorkDir(root=Path(args.target))
    archive.archive_extract(args.archive, target)
    print(f"Extracted {args.archive} to {target.root}")
    return 0


def _cmd_archive_restore(args: argparse.Namespace) -> int:
    target = WorkDir(root=Path(args.target))
    archive.archive_restore(args.archive, target)
    print(f"Restored {args.archive} to {target.root}")
    return 0


def _cmd_archive_clean(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    setup_logging(workdir, args)
    n = archive.archive_clean(workdir)
    print(f"Removed {n} regenerable directories")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    setup_logging(workdir, args)
    manifest_path = workdir.manifest_path()
    if not manifest_path.exists():
        raise SystemExit(f"error: no manifest at {manifest_path}")
    from spacehasten.workspace.manifest import Manifest

    manifest = Manifest.load(manifest_path)
    payload = manifest.model_dump(mode="json")
    if args.json:
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(f"Workspace: {manifest.name}")
        print(f"Created:   {manifest.created_at}")
        print(f"Stages ({len(manifest.stages)}):")
        for name, stage in manifest.stages.items():
            print(f"  {name}: {stage.status}")
        print(f"Runs:      {len(manifest.runs)}")
        print(f"Models:    {len(manifest.models)}")
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    # Resume semantics are not yet defined beyond status; mirror status.
    return _cmd_status(args)


def _cmd_verify(args: argparse.Namespace) -> int:
    from .verify import run_verify

    return run_verify(args)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.quiet and not args.json:
        print_banner()
    func = getattr(args, "func", None)
    if func is None:  # pragma: no cover - argparse enforces required=True
        parser.error("no command specified")
    result = func(args)
    return int(result or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
