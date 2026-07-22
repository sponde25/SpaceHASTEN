"""SpaceHASTEN command-line entry point.

Stitches the :mod:`spacehasten.stages` API behind argparse subcommands.
The console-script is registered in ``pyproject.toml`` as
``spacehasten = spacehasten.cli.main:main``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
    prediction,
    seeds,
    simsearch,
    training,
)
from spacehasten.stages import (
    atlas as atlas_stage,
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
    seed-training       Import seeds → dock → train
    screening-cycle     [train] → (search → predict)×3 → dock per round
    export              Export results (csv, poses, seeds)

  manual stages (expert)
    import-seeds        Import seed compounds into the database (no training)
    dock                Dock the next batch of compounds
    train               Train one chemprop model
    search              Run one simsearch cycle
    predict             Predict scores for undocked rows
    cluster             Run sphere-exclusion clustering
    atlas               Persistent cluster-atlas operations

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
        "--quiet",
        action="store_true",
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
    _add_atlas(sub)
    _add_export(sub)
    _add_archive(sub)
    _add_status(sub)
    _add_resume(sub)
    _add_undo(sub)
    _add_verify(sub)
    return parser


def _add_init(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "init", help="Bootstrap a fresh workspace, create the database, and store docking settings."
    )
    p.add_argument(
        "path",
        type=Path,
        help="Local root directory (should be on fast storage: /wrk or /fastwrk).",
    )
    p.add_argument("--name", default=None, help="Optional. Project name. Default: directory name.")
    p.add_argument(
        "--shared-root",
        type=Path,
        default=None,
        help="Optional. NFS directory for stage artefacts visible to compute nodes. "
        "Default: /data/$USER/SPACEHASTEN/<name>/.",
    )
    p.add_argument(
        "--dock-params", type=Path, required=True, help="Glide docking parameter .in file."
    )
    p.add_argument("--dock-grid", type=Path, required=True, help="Glide grid .zip file.")
    p.set_defaults(func=_cmd_init)


def _add_pick_seeds(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "pick-seeds",
        help="Sample and canonicalize seeds from a large collection file.",
    )
    p.add_argument(
        "--seeds-file",
        type=Path,
        default=None,
        help="Optional. Path to seed collection (bz2/tsv). Default: from config.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output .smi file path.",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        required=True,
        help="Number of seeds to sample.",
    )
    p.add_argument(
        "--cores",
        type=int,
        default=None,
        help="Optional. Local cores for RDKit canonicalization. Default: all available CPUs.",
    )
    p.set_defaults(func=_cmd_pick_seeds)


def _add_import_seeds(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "import-seeds", help="Import seed compounds into the database (no training)."
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--smi", type=Path, help="SMI file with undocked seed compounds.")
    grp.add_argument("--csv", type=Path, help="CSV file with pre-docked seed compounds.")
    p.add_argument(
        "--props-toml",
        type=Path,
        default=None,
        help="Optional. PropertyRanges TOML override. Default: built-in ranges.",
    )
    p.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Optional. Worker pool size for hashing. Default: all available CPUs.",
    )
    p.set_defaults(func=_cmd_import_seeds)


def _add_seed_training(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "seed-training",
        help="Workflow: import seeds → dock → train.",
    )
    p.add_argument("--smi", type=Path, required=True, help="SMI file with undocked seed compounds.")
    p.add_argument(
        "--dock-cpus", type=int, required=True, help="Number of concurrent docking tasks."
    )
    p.add_argument(
        "--props-toml",
        type=Path,
        default=None,
        help="Optional. PropertyRanges TOML override. Default: built-in ranges.",
    )
    p.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Optional. Worker pool size for hashing. Default: all available CPUs.",
    )
    p.set_defaults(func=_cmd_seed_training)


def _add_train(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("train", help="Run one chemprop training round.")
    p.add_argument(
        "--cutoff",
        type=float,
        default=10.0,
        help=(
            "Optional. Docking score cutoff for including compounds in the "
            "training set. Default: 10.0."
        ),
    )
    p.set_defaults(func=_cmd_train)


def _add_predict(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("predict", help="Predict pred_score for every undocked row.")
    p.add_argument(
        "--model-version",
        type=int,
        default=None,
        help="Optional. Model version to use. Default: latest.",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Optional. Number of scheduler array tasks to spread "
        "undocked rows across (implicitly sets chunk size). "
        "Default: a fixed chunk size.",
    )
    p.set_defaults(func=_cmd_predict)


def _add_search(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("search", help="Run one simsearch cycle.")
    p.add_argument(
        "--source",
        choices=("docked", "predicted"),
        required=True,
        help="Source compound pool: docked or predicted.",
    )
    p.add_argument("--top-n", type=int, required=True, help="Number of query compounds.")
    p.add_argument(
        "--cpus",
        type=int,
        required=True,
        help="Number of CPUs for simsearch tasks. Recommendation: max 250.",
    )
    p.add_argument(
        "--strategy",
        choices=("greedy", "clustering"),
        default="greedy",
        help="Optional. Query acquisition strategy. Default: greedy."
        " 'clustering' requires cluster assignments to already exist"
        " (run `spacehasten cluster` first).",
    )
    p.add_argument(
        "--space",
        type=Path,
        default=None,
        help="Optional. BioSolveIT .space file override. Default: from config.",
    )
    p.add_argument(
        "--nnn",
        type=int,
        default=None,
        help="Optional. Max results per query from chemical space. Default: from config (10000).",
    )
    p.add_argument(
        "--sim-spacelight",
        type=float,
        default=None,
        help="Optional. SpaceLight similarity threshold. Default: from config.",
    )
    p.add_argument(
        "--sim-ftrees",
        type=float,
        default=None,
        help="Optional. FTrees similarity threshold. Default: from config.",
    )
    p.add_argument(
        "--threads-per-task",
        type=int,
        default=2,
        help="Optional. Threads per simsearch task. Default: 2.",
    )
    p.set_defaults(func=_cmd_search)


def _add_uncertainty_dock_options(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--lcb-beta",
        type=float,
        default=1.0,
        help="Optional. Exploration weight for LCB. Default: 1.0.",
    )
    p.add_argument(
        "--ei-hit-threshold",
        type=float,
        default=None,
        help="Target-specific virtual-hit threshold. Required for EI.",
    )
    p.add_argument(
        "--ei-xi",
        type=float,
        default=0.0,
        help="Optional. Minimum improvement margin for EI. Default: 0.",
    )
    p.add_argument(
        "--cluster-lambda",
        type=float,
        default=0.0,
        help=(
            "Optional. Weight for the dynamic within-batch cluster penalty. "
            "Values above zero require a current Tanimoto-0.4 cluster atlas."
        ),
    )
    p.add_argument(
        "--atlas-id",
        default=atlas_stage.DEFAULT_ATLAS_ID,
        help=(
            f"Persistent atlas used for cluster penalties. Default: {atlas_stage.DEFAULT_ATLAS_ID}."
        ),
    )


def _validate_dock_acquisition_args(strategy: str, args: argparse.Namespace) -> None:
    if not math.isfinite(args.lcb_beta) or args.lcb_beta < 0:
        raise SystemExit("error: --lcb-beta must be finite and non-negative")
    if not math.isfinite(args.ei_xi) or args.ei_xi < 0:
        raise SystemExit("error: --ei-xi must be finite and non-negative")
    if not math.isfinite(args.cluster_lambda) or args.cluster_lambda < 0:
        raise SystemExit("error: --cluster-lambda must be finite and non-negative")
    if strategy not in {"lcb", "ei"} and args.cluster_lambda != 0:
        raise SystemExit("error: --cluster-lambda requires LCB or EI acquisition")
    if args.ei_hit_threshold is not None and not math.isfinite(args.ei_hit_threshold):
        raise SystemExit("error: --ei-hit-threshold must be finite")
    if strategy == "ei" and args.ei_hit_threshold is None:
        raise SystemExit("error: --ei-hit-threshold is required for EI acquisition")


def _add_dock(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("dock", help="Dock the next batch of compounds.")
    p.add_argument("--top-n", type=int, required=True, help="Number of compounds to dock.")
    p.add_argument(
        "--cpus",
        type=int,
        required=True,
        help="Number of CPUs for docking tasks. Recommendation: max 250.",
    )
    p.add_argument(
        "--strategy",
        choices=("greedy", "clustering", "lcb", "ei"),
        default="greedy",
        help=(
            "Optional. Docking acquisition method. Default: greedy. "
            "LCB and EI use epistemic uncertainty."
        ),
    )
    _add_uncertainty_dock_options(p)
    p.set_defaults(func=_cmd_dock)


def _add_cluster(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("cluster", help="Run one sphere-exclusion clustering round.")
    p.add_argument(
        "--docked",
        action="store_true",
        help="Cluster only compounds that have been docked"
        " (dock_score IS NOT NULL), instead of the whole"
        " data table. Faster on large libraries when the"
        " only goal is populating clusterid for `export csv`."
        " Note: this REPLACES the entire clusters table, so"
        " any previous full-space clustering is discarded.",
    )
    p.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Further restrict to docked compounds with"
        " dock_score <= CUTOFF (better/smaller score)."
        " Requires --docked.",
    )
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.3,
        help="Optional. Minimum within-cluster Tanimoto similarity. Default: 0.3.",
    )
    p.set_defaults(func=_cmd_cluster)


def _add_atlas(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = sub.add_parser("atlas", help="Persistent cluster-atlas operations.")
    atlas_sub = parser.add_subparsers(dest="atlas_operation", required=True)

    init = atlas_sub.add_parser("init", help="Build or reuse the initial seed atlas.")
    init.add_argument("--atlas-id", default=atlas_stage.DEFAULT_ATLAS_ID)
    init.add_argument(
        "--atlas-root",
        type=Path,
        default=None,
        help="Optional shared reusable atlas directory.",
    )
    init.set_defaults(func=_cmd_atlas_init)

    update = atlas_sub.add_parser(
        "update", help="Assign compounds beyond the current atlas watermark."
    )
    update.add_argument("--atlas-id", default=atlas_stage.DEFAULT_ATLAS_ID)
    update.add_argument(
        "--through-spacehastenid",
        type=int,
        default=None,
        help="Optional upper ID bound for replay or controlled updates.",
    )
    update.set_defaults(func=_cmd_atlas_update)

    status = atlas_sub.add_parser("status", help="Show persistent atlas status.")
    status.add_argument("--atlas-id", default=atlas_stage.DEFAULT_ATLAS_ID)
    status.set_defaults(func=_cmd_atlas_status)


def _add_screening_cycle(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "screening-cycle",
        help="Workflow: [train] → (search → predict)×3 → dock per round.",
    )
    p.add_argument(
        "--simsearch-top-n",
        type=int,
        required=True,
        help="Number of simsearch queries. Recommendation: 1000.",
    )
    p.add_argument(
        "--simsearch-jobs",
        type=int,
        required=True,
        help="Number of simsearch jobs (2 CPUs per job). Recommendation: max 250.",
    )
    p.add_argument(
        "--prediction-jobs",
        type=int,
        default=None,
        help=(
            "Optional. CPU array tasks used to refresh every undocked prediction "
            "after retraining. Default: derive chunks from config."
        ),
    )
    p.add_argument(
        "--dock-top-n",
        type=int,
        required=True,
        help="Number of compounds to dock per round. Recommendation: 1000000 (1M).",
    )
    p.add_argument(
        "--dock-cpus",
        type=int,
        required=True,
        help="Number of CPUs for docking tasks. Recommendation: max 250.",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Optional. Number of screening cycle rounds. Default: 1.",
    )
    p.add_argument(
        "--strategy",
        choices=("greedy", "clustering"),
        default="greedy",
        help=(
            "Optional. Similarity-search acquisition strategy. It also controls "
            "docking when --dock-acquisition is omitted. Default: greedy."
        ),
    )
    p.add_argument(
        "--dock-acquisition",
        choices=("greedy", "clustering", "lcb", "ei"),
        default=None,
        help=(
            "Optional. Override acquisition only for docking; similarity-search "
            "queries continue to use --strategy."
        ),
    )
    _add_uncertainty_dock_options(p)
    p.add_argument(
        "--atlas-root",
        type=Path,
        default=None,
        help="Existing completed seed-atlas directory used when the run has no registered atlas.",
    )
    p.add_argument(
        "--space",
        type=Path,
        default=None,
        help="Optional. BioSolveIT .space file override. Default: from config.",
    )
    p.add_argument(
        "--nnn",
        type=int,
        default=None,
        help="Optional. Max results per query from chemical space. Default: from config.",
    )
    p.add_argument(
        "--props-toml",
        type=Path,
        default=None,
        help=(
            "Optional. PropertyRanges TOML to update the stored property filter "
            "ranges before running. Default: use ranges already in the database."
        ),
    )
    p.set_defaults(func=_cmd_screening_cycle)


def _add_export(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("export", help="Export results.")
    sub2 = p.add_subparsers(dest="export_kind", required=True)

    csv_p = sub2.add_parser("csv", help="Export docking results as CSV.")
    csv_p.add_argument(
        "--cutoff", type=float, required=True, help="Docking score cutoff for export."
    )
    csv_p.add_argument("--output", type=Path, required=True, help="Output CSV file path.")
    csv_p.set_defaults(func=_cmd_export_csv)

    poses_p = sub2.add_parser("poses", help="Export Maestro pose file.")
    poses_p.add_argument(
        "--cutoff", type=float, required=True, help="Docking score cutoff for export."
    )
    poses_p.add_argument(
        "--output", type=Path, required=True, help="Output Maestro .mae file path."
    )
    poses_p.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Optional. Limit to a specific docking iteration. Default: all iterations.",
    )
    poses_p.set_defaults(func=_cmd_export_poses)

    seeds_p = sub2.add_parser(
        "seeds",
        help="Export the original seed batch as a CSV (for `import-seeds --csv`).",
    )
    seeds_p.add_argument("--output", type=Path, required=True, help="Output CSV file path.")
    seeds_p.set_defaults(func=_cmd_export_seeds)


def _add_archive(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("archive", help="Archive lifecycle operations.")
    sub2 = p.add_subparsers(dest="archive_op", required=True)

    cre = sub2.add_parser("create", help="Tar the workspace root.")
    cre.add_argument(
        "--bundle", action="store_true", help="Optional. Produce a single .tgz bundle."
    )
    cre.set_defaults(func=_cmd_archive_create)

    ext = sub2.add_parser("extract", help="Inverse of `archive create --bundle`.")
    ext.add_argument(
        "--archive", type=Path, required=True, help="Path to the .tgz bundle to extract."
    )
    ext.add_argument("--target", type=Path, required=True, help="Target directory for extraction.")
    ext.set_defaults(func=_cmd_archive_extract)

    res = sub2.add_parser("restore", help="Inverse of `archive create` (.archived-spacehasten).")
    res.add_argument(
        "--archive", type=Path, required=True, help="Path to .archived-spacehasten directory."
    )
    res.add_argument("--target", type=Path, required=True, help="Target workspace directory.")
    res.set_defaults(func=_cmd_archive_restore)

    clean = sub2.add_parser("clean", help="Remove regenerable scratch dirs.")
    clean.set_defaults(func=_cmd_archive_clean)


def _add_status(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("status", help="Print workspace status summary.")
    p.add_argument(
        "--actives",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Show count of docked compounds with dock_score < THRESHOLD.",
    )
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
        "--yes",
        "-y",
        action="store_true",
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
    print(
        f"  property filter template: {workdir.props_path()} "
        "(edit before running seed-training or screening-cycle)"
    )
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
        logger.info("Imported %d seeds; starting dock → train", n)
        docking.dock(
            db,
            workdir,
            scheduler,
            settings,
            top_n=n,
            strategy="greedy",
            cpus=args.dock_cpus,
        )
        training.train(
            db,
            workdir,
            scheduler,
            settings,
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
            db,
            workdir,
            scheduler,
            settings,
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
            db,
            workdir,
            scheduler,
            settings,
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
    _validate_dock_acquisition_args(args.strategy, args)
    with open_db(args) as db:
        try:
            iteration = docking.dock(
                db,
                workdir,
                scheduler,
                settings,
                top_n=args.top_n,
                strategy=args.strategy,
                cpus=args.cpus,
                lcb_beta=args.lcb_beta,
                ei_hit_threshold=args.ei_hit_threshold,
                ei_xi=args.ei_xi,
                cluster_lambda=args.cluster_lambda,
                atlas_id=args.atlas_id,
            )
        except ValueError as exc:
            raise SystemExit(f"error: {exc}") from exc
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
            db,
            workdir,
            scheduler,
            settings,
            docked_only=args.docked,
            cutoff=args.cutoff,
            similarity_threshold=args.similarity_threshold,
        )
    print(f"Clustered {n} compounds")
    return 0


def _cmd_atlas_init(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    setup_logging(workdir, args)
    settings = settings_from_args(args)
    scheduler = scheduler_from_args(args, settings)
    with open_db(args) as db:
        version = atlas_stage.build_initial_seed_atlas(
            db,
            workdir,
            scheduler,
            settings,
            atlas_id=args.atlas_id,
            atlas_root=args.atlas_root,
        )
    print(
        f"Atlas {version.atlas_id} v{version.version}: "
        f"{version.compound_count} compounds, {version.centroid_count} centroids"
    )
    return 0


def _cmd_atlas_status(args: argparse.Namespace) -> int:
    setup_logging(workdir_from_args(args), args)
    with open_db(args) as db:
        version = db.latest_cluster_atlas_version(args.atlas_id)
    if version is None:
        print(f"Atlas {args.atlas_id}: not initialized")
        return 1
    print(
        json.dumps(
            {
                "atlas_id": version.atlas_id,
                "version": version.version,
                "last_spacehastenid": version.last_spacehastenid,
                "compound_count": version.compound_count,
                "centroid_count": version.centroid_count,
                "metadata_path": version.metadata_path,
            },
            indent=2,
        )
    )
    return 0


def _cmd_atlas_update(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    setup_logging(workdir, args)
    settings = settings_from_args(args)
    scheduler = scheduler_from_args(args, settings)
    with open_db(args) as db:
        version = atlas_stage.update_cluster_atlas(
            db,
            workdir,
            scheduler,
            settings,
            atlas_id=args.atlas_id,
            through_spacehastenid=args.through_spacehastenid,
        )
    print(
        f"Atlas {version.atlas_id} v{version.version}: "
        f"{version.compound_count} compounds, {version.centroid_count} centroids, "
        f"watermark={version.last_spacehastenid}"
    )
    return 0


def _cmd_screening_cycle(args: argparse.Namespace) -> int:
    workdir = workdir_from_args(args)
    setup_logging(workdir, args)
    settings = settings_from_args(args)
    scheduler = scheduler_from_args(args, settings)

    strategy: Literal["greedy", "clustering"] = args.strategy
    dock_strategy: docking.DockStrategy = args.dock_acquisition or strategy
    _validate_dock_acquisition_args(dock_strategy, args)
    use_cluster_atlas = dock_strategy in {"lcb", "ei"} and args.cluster_lambda > 0

    def _maybe_cluster_queries(db: Database) -> None:
        """Refresh assignments only for hard-clustered simsearch queries."""
        if strategy == "clustering":
            clustering.cluster(db, workdir, scheduler, settings)

    def _maybe_cluster_docking(db: Database) -> None:
        if dock_strategy == "clustering":
            clustering.cluster(db, workdir, scheduler, settings)
        elif use_cluster_atlas:
            atlas_stage.update_cluster_atlas(
                db,
                workdir,
                scheduler,
                settings,
                atlas_id=args.atlas_id,
            )

    with open_db(args) as db:
        if use_cluster_atlas and db.latest_cluster_atlas_version(args.atlas_id) is None:
            if args.atlas_root is None:
                raise SystemExit(
                    "error: clustered LCB/EI acquisition requires an existing seed atlas; "
                    "pass --atlas-root PATH or run `spacehasten atlas init "
                    "--atlas-root PATH` first"
                )
            try:
                atlas_stage.import_initial_seed_atlas(
                    db,
                    workdir,
                    scheduler,
                    settings,
                    atlas_root=args.atlas_root,
                    atlas_id=args.atlas_id,
                )
            except (FileNotFoundError, ValueError) as exc:
                raise SystemExit(
                    f"error: {exc}; run `spacehasten atlas init --atlas-root "
                    f"{args.atlas_root}` first"
                ) from exc
        if args.props_toml is not None:
            props = PropertyRanges.from_toml(args.props_toml)
            db.replace_properties(seeds.typed_to_db_props(props))
            db.replace_smarts_filters(seeds.typed_smarts_to_db(props))
            logger.info("Updated property filter ranges from %s", args.props_toml)
        for round_n in range(1, args.rounds + 1):
            logger.info("Screening round %d/%d", round_n, args.rounds)

            # Train if there are newly docked compounds from a previous
            # screening cycle (i.e. not the very first round ever run).
            if db.latest_dock_iteration() is not None and db.latest_dock_iteration() > 0:
                logger.info("Training on newly docked data before round %d", round_n)
                model_version = training.train(db, workdir, scheduler, settings)
                refreshed = prediction.predict_undocked(
                    db,
                    workdir,
                    scheduler,
                    settings,
                    model_version=model_version,
                    jobs=args.prediction_jobs,
                )
                logger.info(
                    "Refreshed %d undocked predictions with model v%d before round %d",
                    refreshed,
                    model_version,
                    round_n,
                )

            # search(docked) — CONTROL phase handles prop filter + predict
            _maybe_cluster_queries(db)
            simsearch.simsearch(
                db,
                workdir,
                scheduler,
                settings,
                source="docked",
                strategy=strategy,
                top_n=args.simsearch_top_n,
                space=args.space,
                cpu=args.simsearch_jobs,
                nnn=args.nnn,
            )

            # search(predicted) × 2 — new hits get pred_score via CONTROL
            for _ in range(2):
                _maybe_cluster_queries(db)
                simsearch.simsearch(
                    db,
                    workdir,
                    scheduler,
                    settings,
                    source="predicted",
                    strategy=strategy,
                    top_n=args.simsearch_top_n,
                    space=args.space,
                    cpu=args.simsearch_jobs,
                    nnn=args.nnn,
                )

            # dock
            _maybe_cluster_docking(db)
            docking.dock(
                db,
                workdir,
                scheduler,
                settings,
                top_n=args.dock_top_n,
                strategy=dock_strategy,
                cpus=args.dock_cpus,
                lcb_beta=args.lcb_beta,
                ei_hit_threshold=args.ei_hit_threshold,
                ei_xi=args.ei_xi,
                cluster_lambda=args.cluster_lambda,
                atlas_id=args.atlas_id,
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
            db,
            workdir,
            args.output,
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
