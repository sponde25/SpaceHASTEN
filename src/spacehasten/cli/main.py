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
from spacehasten.core.db import Database
from spacehasten.stages import (
    archive,
    clustering,
    docking,
    export,
    plotting,
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
    library-build       Convert an Enamine diverse subset into a reusable Parquet library store

  workflows (recommended)
    seed-training       Import seeds → dock → train
    screening-cycle     [train] → (search → predict)×3 → dock per round
    export              Export results (csv, poses, seeds)
    plot                Plot docking/prediction score diagnostics

  manual stages (expert)
    import-seeds        Import seed compounds into the database (no training)
    dock                Dock the next batch of compounds
    train               Train one chemprop model
    search              Run one simsearch cycle
    predict             Predict scores for undocked rows
    cluster             Run sphere-exclusion clustering
    library-screen      Property-filter + chemprop-score a library store, insert survivors

  utilities
    status              Print workspace manifest summary
    resume              Resume the last interrupted run
    undo                Revert a failed or unwanted search cycle (manual intervention)
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
    _add_library_build(sub)
    _add_seed_training(sub)
    _add_screening_cycle(sub)
    _add_import_seeds(sub)
    _add_train(sub)
    _add_predict(sub)
    _add_search(sub)
    _add_dock(sub)
    _add_cluster(sub)
    _add_library_screen(sub)
    _add_export(sub)
    _add_plot(sub)
    _add_archive(sub)
    _add_status(sub)
    _add_resume(sub)
    _add_undo(sub)
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


def _add_library_build(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "library-build",
        help="Convert an Enamine diverse subset into a reusable chunked Parquet"
             " library store (build once, screen many times).",
    )
    p.add_argument(
        "--source", type=Path, action="append", required=True, metavar="FILE",
        help="Required. Source .cxsmiles/.smi (optionally .bz2/.gz) file. "
             "Repeatable for multiple source files sharing the same header.",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Output store directory (manifest.json + chunk_*.parquet).",
    )
    p.add_argument(
        "--chunk-size", type=int, default=None,
        help="Optional. Rows per shard/chunk. Default: from config (2,000,000).",
    )
    p.add_argument(
        "--recompute-props", action="store_true",
        help="Optional. Force RDKit computation of the six PC descriptors "
             "instead of reusing the Enamine source columns.",
    )
    p.add_argument(
        "--smiles-col", type=str, default=None,
        help="Optional. Column name or 0-based index override. Default: header 'smiles'.",
    )
    p.add_argument(
        "--id-col", type=str, default=None,
        help="Optional. Column name or 0-based index override. Default: header 'id'.",
    )
    p.add_argument(
        "--jobs", type=int, default=250,
        help="Optional. Max number of build tasks (array jobs) to run "
             "simultaneously. Each task uses one core, so this caps peak CPU "
             "usage. The number of tasks is set by --chunk-size. Default: 250.",
    )
    p.set_defaults(func=_cmd_library_build)


def _add_import_seeds(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("import-seeds", help="Import seed compounds into the database (no training).")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--smi", type=Path, help="SMI file with undocked seed compounds.")
    grp.add_argument("--csv", type=Path, help="CSV file with pre-docked seed compounds.")
    p.add_argument("--props-toml", type=Path, default=None,
                   help="Optional. PropertyRanges TOML override. Default: the "
                        "workspace's props.toml (from `init`), else built-in ranges.")
    p.add_argument("--processes", type=int, default=None,
                   help="Optional. Worker pool size for hashing. Default: all available CPUs.")
    p.set_defaults(func=_cmd_import_seeds)


def _add_seed_training(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "seed-training",
        help="Workflow: import seeds → dock → train.",
    )
    p.add_argument("--smi", type=Path, required=True, help="SMI file with undocked seed compounds.")
    p.add_argument("--dock-cpus", type=int, required=True, help="Number of concurrent docking tasks.")
    p.add_argument("--props-toml", type=Path, default=None,
                   help="Optional. PropertyRanges TOML override. Default: the "
                        "workspace's props.toml (from `init`), else built-in ranges.")
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
    p.add_argument("--jobs", type=int, default=None,
                   help="Optional. Number of scheduler array tasks to spread "
                        "undocked rows across (implicitly sets chunk size). "
                        "Default: a fixed chunk size.")
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
                   help="Optional. Query acquisition strategy. Default: greedy."
                        " 'clustering' requires cluster assignments to already exist"
                        " (run `spacehasten cluster` first).")
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
    p.set_defaults(func=_cmd_search)


def _add_dock(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("dock", help="Dock the next batch of compounds.")
    p.add_argument("--top-n", type=int, required=True,
                   help="Number of compounds to dock.")
    p.add_argument("--cpus", type=int, required=True,
                   help="Number of CPUs for docking tasks. Recommendation: max 250.")
    p.add_argument("--strategy", choices=("greedy", "clustering"), default="greedy",
                   help="Optional. Acquisition strategy for choosing which compounds to dock. Default: greedy."
                        " 'clustering' requires cluster assignments to already exist"
                        " (run `spacehasten cluster` first).")
    p.set_defaults(func=_cmd_dock)


def _add_cluster(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("cluster", help="Run one sphere-exclusion clustering round.")
    p.add_argument("--docked", action="store_true",
                   help="Cluster only compounds that have been docked"
                        " (dock_score IS NOT NULL), instead of the whole"
                        " data table. Faster on large libraries when the"
                        " only goal is populating clusterid for `export csv`."
                        " Note: this REPLACES the entire clusters table, so"
                        " any previous full-space clustering is discarded.")
    p.add_argument("--cutoff", type=float, default=None,
                   help="Further restrict to docked compounds with"
                        " dock_score <= CUTOFF (better/smaller score)."
                        " Requires --docked.")
    p.set_defaults(func=_cmd_cluster)


def _add_library_screen(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "library-screen",
        help="Property-filter + chemprop-score a library store (from "
             "`library-build`); insert high-scoring survivors into the DB.",
    )
    p.add_argument(
        "--library", type=Path, default=None,
        help="Optional. Library store directory (manifest.json + chunks). "
             "Default: paths.library_store_default.",
    )
    p.add_argument(
        "--model-version", type=int, default=None,
        help="Optional. Model version to score with. Default: latest.",
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--top-n", type=int, default=None,
        help="Optional. Keep the global top-N survivors by pred_score. "
             "Mutually exclusive with --score-cutoff.",
    )
    grp.add_argument(
        "--score-cutoff", type=float, default=None,
        help="Optional. Keep every survivor with pred_score <= CUTOFF. "
             "Mutually exclusive with --top-n.",
    )
    p.add_argument(
        "--top-pct", type=float, default=None,
        help="Optional. Used only when neither --top-n nor --score-cutoff is"
             " given: cutoff = the top TOP-PCT percent of seed dock scores."
             " Default: from config (1.0).",
    )
    p.add_argument(
        "--props-toml", type=Path, default=None,
        help="Optional. PropertyRanges TOML override. Default: the workspace's"
             " props.toml (from `init`), else the DB properties table, falling"
             " back to built-in ranges.",
    )
    p.add_argument(
        "--jobs", type=int, default=250,
        help="Optional. Max number of screening tasks (array jobs) to run "
             "simultaneously. Default: 250.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Optional. Compute the selection but do not insert into the DB;"
             " writes the ranked survivor list to a CSV next to --report.",
    )
    p.add_argument(
        "--report", type=Path, default=None,
        help="Optional. JSON report output path.",
    )
    p.set_defaults(func=_cmd_library_screen)


def _add_screening_cycle(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "screening-cycle",
        help="Workflow: [train] → (search → predict)×3 → dock per round.",
    )
    p.add_argument("--simsearch-top-n", type=int, required=True,
                   help="Number of simsearch queries. Recommendation: 1000.")
    p.add_argument("--simsearch-jobs", type=int, required=True,
                   help="Number of simsearch jobs (2 CPUs per job). Recommendation: max 250.")
    p.add_argument("--dock-top-n", type=int, required=True,
                   help="Number of compounds to dock per round. Recommendation: 1000000 (1M).")
    p.add_argument("--dock-cpus", type=int, required=True,
                   help="Number of CPUs for docking tasks. Recommendation: max 250.")
    p.add_argument("--rounds", type=int, default=1,
                   help="Optional. Number of screening cycle rounds. Default: 1.")
    p.add_argument("--strategy", choices=("greedy", "clustering"), default="greedy",
                   help="Optional. Acquisition strategy for choosing which compounds to dock. Default: greedy."
                        " 'clustering' auto-clusters before each search/dock step in every round.")
    p.add_argument("--space", type=Path, default=None,
                   help="Optional. BioSolveIT .space file override. Default: from config.")
    p.add_argument("--nnn", type=int, default=None,
                   help="Optional. Max results per query from chemical space. Default: from config.")
    p.add_argument("--props-toml", type=Path, default=None,
                   help="Optional. PropertyRanges TOML to update the stored property filter ranges before running. Default: the workspace's props.toml (from `init`) if present, else use ranges already in the database.")
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

    seeds_p = sub2.add_parser(
        "seeds",
        help="Export the original seed batch as a CSV (for `import-seeds --csv`).",
    )
    seeds_p.add_argument("--output", type=Path, required=True,
                         help="Output CSV file path.")
    seeds_p.set_defaults(func=_cmd_export_seeds)


def _add_plot(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "plot",
        help="Plot docking/prediction score diagnostics from the database so far.",
    )
    p.add_argument(
        "--kind", choices=("dock-scores", "pred-scores", "accuracy", "all"),
        default="all",
        help="Optional. Which plot(s) to generate. Default: all.\n"
             "  dock-scores: dock_score KDE, seed vs. each dock iteration.\n"
             "  pred-scores: pred_score KDE per simsearch cycle vs. seed baseline.\n"
             "  accuracy:    hexbin of predicted vs. actual dock_score.",
    )
    p.add_argument(
        "--dock-iterations", default="1,2", metavar="N,N,...",
        help="Optional. Comma-separated dock_iteration values for the"
             " 'accuracy' plot. Default: 1,2.",
    )
    p.add_argument(
        "--bw-adjust", type=float, default=2.0,
        help="Optional. KDE bandwidth multiplier for the distribution"
             " plots. Default: 2.0.",
    )
    p.add_argument(
        "--max-dock-score", type=float, default=0.0, metavar="SCORE",
        help="Optional. Cap the x-axis at this value on the dock-scores"
             " and pred-scores plots, hiding the long positive-score"
             " tail. Default: 0.0. Ignored if --show-all-scores is set.",
    )
    p.add_argument(
        "--show-all-scores", action="store_true",
        help="Optional. Show the full score range on the dock-scores and"
             " pred-scores plots, including positive (unfavourable)"
             " scores, instead of capping the x-axis at --max-dock-score.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Optional. Directory to write PNGs into."
             " Default: <workspace>/plots (or shared_root/plots).",
    )
    p.set_defaults(func=_cmd_plot)


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
    p = sub.add_parser("status", help="Print workspace status summary.")
    p.add_argument("--actives", type=float, default=None, metavar="THRESHOLD",
                   help="Show count of docked compounds with dock_score < THRESHOLD.")
    p.set_defaults(func=_cmd_status)


def _add_resume(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("resume", help="Resume the last interrupted run (alias of status).")
    p.set_defaults(func=_cmd_resume)


def _add_undo(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "undo",
        help="Revert a failed or unwanted operation. Manual-intervention only.",
    )
    sub2 = p.add_subparsers(dest="undo_kind", required=True)

    search_p = sub2.add_parser(
        "search",
        help="Revert the latest simsearch cycle: delete its hit compounds and "
             "release its query marks so those compounds can be selected as "
             "queries again.",
    )
    search_p.add_argument(
        "--yes", "-y", action="store_true",
        help="Optional. Skip the confirmation prompt. Required when stdin is "
             "not a terminal (e.g. scripted use).",
    )
    search_p.set_defaults(func=_cmd_undo_search)


def _add_verify(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    from .verify import add_verify_arguments

    p = sub.add_parser("verify", help="End-to-end smoke test (Session 15).")
    add_verify_arguments(p)
    p.set_defaults(func=_cmd_verify)


# --------------------------------------------------------------------------- #
# Subcommand implementations                                                  #
# --------------------------------------------------------------------------- #


def _cmd_init(args: argparse.Namespace) -> int:
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

    PropertyRanges().to_toml(workdir.props_path())

    logger.info("Workspace initialised: root=%s  shared=%s", root, shared_root)
    print(f"Initialised workspace at {root}")
    print(f"  local root (DB + logs): {root}")
    print(f"  shared root (stages):   {shared_root}")
    print(f"  property filter template: {workdir.props_path()} (edit before running seed-training or screening-cycle)")
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


def _cmd_library_build(args: argparse.Namespace) -> int:
    from spacehasten.stages.library_build import library_build

    # library-build is independent of any spacehasten workspace: it builds a
    # reusable library store at --output for use across many future runs, so
    # it must not require the cwd (or -w) to already be an initialized
    # workspace. Logs go to <output>/logs instead of a workspace's logs dir.
    # Resolve --output to an absolute path up front (library_build() also
    # resolves it, but doing it here too keeps the printed/logged path and
    # the logs directory consistent with a relative --output argument).
    output_dir = Path(args.output).resolve()
    settings = settings_from_args(args)
    scheduler = scheduler_from_args(args, settings)
    setup_logging(WorkDir(root=output_dir), args)

    if args.jobs is not None and args.jobs < 1:
        raise SystemExit(f"error: --jobs must be >= 1, got {args.jobs}")

    column_map: dict[str, str] = {}
    if args.smiles_col is not None:
        column_map["smiles"] = args.smiles_col
    if args.id_col is not None:
        column_map["id"] = args.id_col

    chunk_size = (
        args.chunk_size if args.chunk_size is not None
        else settings.general.library_build_chunk_size
    )

    manifest = library_build(
        scheduler, settings,
        source_files=args.source,
        store_dir=output_dir,
        chunk_size=chunk_size,
        recompute_props=args.recompute_props,
        column_map=column_map or None,
        max_concurrent=args.jobs,
    )
    print(
        f"Built library store at {output_dir}: "
        f"{manifest.n_compounds} compounds across {manifest.n_chunks} chunks"
    )
    return 0


def _cmd_import_seeds(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    workdir.logs_dir().mkdir(parents=True, exist_ok=True)
    setup_logging(workdir, args)

    props_toml = _effective_props_toml(args, workdir)
    props = (
        PropertyRanges.from_toml(props_toml)
        if props_toml is not None
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

    props_toml = _effective_props_toml(args, workdir)
    props = (
        PropertyRanges.from_toml(props_toml)
        if props_toml is not None
        else PropertyRanges()
    )

    with open_db(args) as db:
        n = seeds.import_seeds(
            db,
            smi_path=args.smi,
            props=props,
            processes=args.processes,
        )
        logger.info("Imported %d seeds; starting dock → train", n)
        docking.dock(
            db, workdir, scheduler, settings,
            top_n=n,
            strategy="greedy",
            cpus=args.dock_cpus,
        )
        training.train(
            db, workdir, scheduler, settings,
        )

    print(f"Seed-training complete ({n} seeds imported, docked, trained)")
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
            jobs=args.jobs,
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
    if args.cutoff is not None and not args.docked:
        print("error: --cutoff requires --docked", file=sys.stderr)
        return 2
    with open_db(args) as db:
        n = clustering.cluster(
            db, workdir, scheduler, settings,
            docked_only=args.docked, cutoff=args.cutoff,
        )
    print(f"Clustered {n} compounds")
    return 0


def _effective_props_toml(args: argparse.Namespace, workdir: WorkDir) -> Path | None:
    """Resolve which property-filter TOML a workspace command should use.

    Precedence: an explicit ``--props-toml`` always wins; otherwise fall back
    to the workspace's own ``props.toml`` (written by ``init`` and meant to be
    edited) when it exists. ``None`` means "no TOML available" — the caller
    then uses built-in :class:`PropertyRanges` defaults.

    Defaulting to ``workdir.props_path()`` makes edits to that file take effect
    without re-passing ``--props-toml`` on every command, and prevents a bare
    ``import-seeds``/``seed-training`` from silently reverting the stored
    ranges back to the built-in defaults.
    """
    if args.props_toml is not None:
        return Path(args.props_toml)
    candidate = workdir.props_path()
    return candidate if candidate.exists() else None


def _resolve_library_screen_props(
    args: argparse.Namespace, workdir: WorkDir, db: Database
) -> PropertyRanges:
    """Resolve --props-toml / workspace props.toml > DB properties table > built-in defaults."""
    props_toml = _effective_props_toml(args, workdir)
    if props_toml is not None:
        return PropertyRanges.from_toml(props_toml)
    db_props = db.load_properties()
    if db_props is not None:
        return PropertyRanges.model_validate({
            "mw": {"min": float(db_props.mw[0]), "max": float(db_props.mw[1])},
            "slogp": {"min": float(db_props.slogp[0]), "max": float(db_props.slogp[1])},
            "hba": {"min": int(db_props.hba[0]), "max": int(db_props.hba[1])},
            "hbd": {"min": int(db_props.hbd[0]), "max": int(db_props.hbd[1])},
            "rotbonds": {"min": int(db_props.rotbonds[0]), "max": int(db_props.rotbonds[1])},
            "tpsa": {"min": float(db_props.tpsa[0]), "max": float(db_props.tpsa[1])},
        })
    return PropertyRanges()


def _cmd_library_screen(args: argparse.Namespace) -> int:
    from spacehasten.stages.library_screen import library_screen

    workdir = workdir_from_args(args)
    setup_logging(workdir, args)
    settings = settings_from_args(args)
    scheduler = scheduler_from_args(args, settings)

    if args.jobs is not None and args.jobs < 1:
        raise SystemExit(f"error: --jobs must be >= 1, got {args.jobs}")

    library_dir = args.library
    if library_dir is None and settings.paths.library_store_default:
        library_dir = Path(settings.paths.library_store_default)
    if library_dir is None:
        raise SystemExit(
            "error: no library store given; pass --library or set "
            "paths.library_store_default in config"
        )

    top_pct = (
        args.top_pct if args.top_pct is not None
        else settings.general.library_default_top_pct
    )

    with open_db(args) as db:
        version = args.model_version
        if version is None:
            version = db.latest_model_version()
            if version is None:
                raise SystemExit("error: no trained model; run `spacehasten train` first")

        props = _resolve_library_screen_props(args, workdir, db)

        n = library_screen(
            db, workdir, scheduler, settings,
            library_dir=library_dir,
            model_version=version,
            props=props,
            top_n=args.top_n,
            score_cutoff=args.score_cutoff,
            top_pct=top_pct,
            max_concurrent=args.jobs,
            dry_run=args.dry_run,
            report_path=args.report,
        )
    verb = "Would insert" if args.dry_run else "Inserted"
    print(f"{verb} {n} compounds from library-screen (model v{version})")
    return 0


def _cmd_screening_cycle(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    setup_logging(workdir, args)
    settings = settings_from_args(args)
    scheduler = scheduler_from_args(args, settings)

    strategy: Literal["greedy", "clustering"] = args.strategy

    def _maybe_cluster(db: Database) -> None:
        """Re-cluster before each search/dock step when using the
        ``clustering`` acquisition strategy, so query/dock selection sees
        up-to-date cluster assignments (including compounds ingested
        earlier in this same round). No-op for ``greedy``."""
        if strategy == "clustering":
            clustering.cluster(db, workdir, scheduler, settings)

    with open_db(args) as db:
        props_toml = _effective_props_toml(args, workdir)
        if props_toml is not None:
            props = PropertyRanges.from_toml(props_toml)
            db.replace_properties(seeds.typed_to_db_props(props))
            db.replace_smarts_filters(seeds.typed_smarts_to_db(props))
            logger.info("Updated property filter ranges from %s", props_toml)
        for round_n in range(1, args.rounds + 1):
            logger.info("Screening round %d/%d", round_n, args.rounds)

            # Train if there are newly docked compounds from a previous
            # screening cycle (i.e. not the very first round ever run).
            if db.latest_dock_iteration() is not None and db.latest_dock_iteration() > 0:
                logger.info("Training on newly docked data before round %d", round_n)
                training.train(db, workdir, scheduler, settings)

            # search(docked) — CONTROL phase handles prop filter + predict
            _maybe_cluster(db)
            simsearch.simsearch(
                db, workdir, scheduler, settings,
                source="docked", strategy=strategy,
                top_n=args.simsearch_top_n,
                space=args.space, cpu=args.simsearch_jobs,
                nnn=args.nnn,
            )

            # search(predicted) × 2 — new hits get pred_score via CONTROL
            for _ in range(2):
                _maybe_cluster(db)
                simsearch.simsearch(
                    db, workdir, scheduler, settings,
                    source="predicted", strategy=strategy,
                    top_n=args.simsearch_top_n,
                    space=args.space, cpu=args.simsearch_jobs,
                    nnn=args.nnn,
                )

            # dock
            _maybe_cluster(db)
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
    with open_db(args) as db:
        out = export.export_poses(
            db, workdir, args.output,
            cutoff=args.cutoff,
            iteration=args.iteration,
            settings=settings,
        )
    print(f"Wrote poses to {out}")
    return 0


def _cmd_export_seeds(args: argparse.Namespace) -> int:
    setup_logging(workdir_from_args(args), args)
    with open_db(args) as db:
        n = export.export_seeds(db, args.output)
    print(f"Exported {n} seed rows to {args.output}")
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    setup_logging(workdir, args)

    try:
        dock_iterations = tuple(int(x) for x in args.dock_iterations.split(","))
    except ValueError:
        raise SystemExit(
            f"error: --dock-iterations must be a comma-separated list of"
            f" integers, got {args.dock_iterations!r}"
        )

    outputs: list[Path] = []
    max_dock_score = None if args.show_all_scores else args.max_dock_score
    with open_db(args) as db:
        if args.kind in ("dock-scores", "all"):
            out = (args.output_dir / "dock_score_distribution.png") if args.output_dir else None
            outputs.append(plotting.plot_dock_score_distribution(
                db, workdir, bw_adjust=args.bw_adjust, output=out,
                max_dock_score=max_dock_score,
            ))
        if args.kind in ("pred-scores", "all"):
            out = (args.output_dir / "pred_score_distribution.png") if args.output_dir else None
            outputs.append(plotting.plot_pred_score_distribution(
                db, workdir, bw_adjust=args.bw_adjust, output=out,
                max_dock_score=max_dock_score,
            ))
        if args.kind in ("accuracy", "all"):
            out = (args.output_dir / "pred_vs_dock_accuracy.png") if args.output_dir else None
            outputs.append(plotting.plot_pred_vs_dock_accuracy(
                db, workdir, dock_iterations=dock_iterations, output=out,
            ))

    for out in outputs:
        print(f"Wrote plot to {out}")
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

    with open_db(args) as db:
        cycles = db.latest_simsearch_cycle()
        iterations = db.latest_dock_iteration() or 0
        model_ver = db.latest_model_version()
        total_compounds = db.count_total()
        total_docked = db.count_docked()

        if args.json:
            payload = {
                "workspace": workdir.name,
                "simsearch_cycles": cycles,
                "dock_iterations": iterations,
                "model_versions": model_ver or 0,
                "total_compounds": total_compounds,
                "docked_compounds": total_docked,
            }
            if args.actives is not None:
                payload["actives_threshold"] = args.actives
                payload["actives_count"] = db.count_actives(args.actives)
            json.dump(payload, sys.stdout, indent=2)
            sys.stdout.write("\n")
        else:
            print(f"Workspace:               {workdir.name}")
            print(f"Similarity search cycles: {cycles}")
            print(f"Docking iterations:       {iterations}")
            print(f"Model versions:           {model_ver or 0}")
            print(f"Total compounds:          {total_compounds}")
            print(f"Docked compounds:         {total_docked}")
            if args.actives is not None:
                n = db.count_actives(args.actives)
                print(f"Actives (dock_score < {args.actives}): {n}")
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    # Resume semantics are not yet defined beyond status; mirror status.
    return _cmd_status(args)


def _cmd_undo_search(args: argparse.Namespace) -> int:
    """Revert the latest simsearch cycle (see :meth:`Database.undo_simsearch_cycle`).

    Always targets the most recent search *attempt* (successful or
    failed) — there is no ``--cycle`` argument, since a cycle whose hits
    were already used as later queries can never be the latest attempt
    (see :meth:`Database.latest_search_attempt_cycle`). This is
    deliberately interactive: it is meant for the rare, manual-
    intervention case where a search cycle failed after marking queries,
    stranding those compounds behind ``query IS NOT NULL``.
    """
    workdir = workdir_from_args(args)
    setup_logging(workdir, args)

    with open_db(args) as db:
        cycle = db.latest_search_attempt_cycle()
        if cycle is None:
            print("No simsearch cycles found; nothing to undo.")
            return 0

        stats = db.simsearch_cycle_stats(cycle)
        outcome = "completed" if stats.n_hits > 0 else "failed or incomplete"
        print(f"Latest search attempt: cycle {cycle} ({outcome})")
        print(f"  hit compounds discovered  : {stats.n_hits}")
        print(f"  compounds marked as query : {stats.n_queries}")

        if not args.yes:
            if not sys.stdin.isatty():
                logger.error(
                    "refusing to undo simsearch cycle %d non-interactively without --yes",
                    cycle,
                )
                print(
                    "error: `undo search` requires manual confirmation; "
                    "re-run with --yes to confirm non-interactively, or run "
                    "interactively to confirm at the prompt.",
                    file=sys.stderr,
                )
                return 1
            answer = input(
                f"Undo simsearch cycle {cycle}? This deletes {stats.n_hits} hit "
                f"compound(s) and releases {stats.n_queries} query mark(s). [y/N] "
            )
            if answer.strip().lower() not in ("y", "yes"):
                print("Aborted; no changes made.")
                return 1

        try:
            n_hits, n_queries = db.undo_simsearch_cycle(cycle)
        except ValueError as exc:
            logger.error("%s", exc)
            print(f"error: {exc}", file=sys.stderr)
            return 1

    print(
        f"Reverted cycle {cycle}: removed {n_hits} hit compound(s), "
        f"released {n_queries} query mark(s)."
    )
    return 0


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
