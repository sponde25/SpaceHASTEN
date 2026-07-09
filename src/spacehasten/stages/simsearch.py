"""Simsearch stage — SpaceLight + FTrees + property filter + chemprop predict.

Replaces legacy ``simsearch_functions.simsearch`` /
``process_sim_results``. Two-phase orchestration:

1. **Search** (Phase A) — pick top-N seeds via
   :meth:`Database.select_queries_for_simsearch`, mark them as queries,
   write a ``queries_<name>.smi`` file, submit one array task per query
   that runs SpaceLight + FTrees against the configured space, and copy
   per-task CSVs back to the cycle directory.
2. **Control** (Phase B) — aggregate per-method best similarity per
   SMILES, chunk hits into ``control/control_<i>.smi.gz``, materialise
   the latest model on disk, write ``control.param`` from the DB
   ``properties`` table, write ``smarts.txt`` from the DB
   ``smarts_filters`` table, and submit one array task per chunk that
   filters by RDKit properties and optional SMARTS
   (:mod:`spacehasten.remote.prop_filter`) and predicts via chemprop
   (:mod:`spacehasten.remote.predict`).
3. **Ingest** (Phase C) — read ``predicted_propoutput_control_*.csv``,
   pick the *minimum* predicted ``docking_score`` per ``reghash``, drop
   any reghash already present in ``data``, and insert the survivors
   via :meth:`Database.insert_simsearch_hit`.

Layout::

    <workdir>/simsearch/cycle<N>/
        queries_<name>.smi                  # one query per line
        results/
            spacelightresult_<task>.csv     # per-task search output
            ftreesresult_<task>.csv
        CONTROL/
            inputs/
                control.param              # 12-line property bounds
                smarts.txt                 # optional SMARTS patterns
                control_<i>.smi.gz         # one chunk per task
            results_propfilter/
                propoutput_control_<i>.csv      # post-prop-filter
            results_prediction/
                predicted_propoutput_control_<i>.csv  # post-chemprop predict
"""

from __future__ import annotations

import csv
import gzip
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final, Literal

from spacehasten.config.settings import Settings
from spacehasten.core.db import Database
from spacehasten.scheduler.base import ArrayJob, Scheduler
from spacehasten.tools.ftrees import FTreesAdapter
from spacehasten.tools.spacelight import SpacelightAdapter
from spacehasten.workspace.layout import WorkDir

logger = logging.getLogger(__name__)


_SIM_METHODS: Final[tuple[str, ...]] = ("spacelight", "ftrees")
_PROP_ORDER: Final[tuple[str, ...]] = (
    "mw", "slogp", "hba", "hbd", "rotbonds", "tpsa",
)

_DEFAULT_PROP_FILTER_COMMAND: Final[tuple[str, ...]] = (
    "python3", "-m", "spacehasten.remote.prop_filter",
)
_DEFAULT_PREDICT_COMMAND: Final[tuple[str, ...]] = (
    "python3", "-m", "spacehasten.remote.predict",
)


def _default_prop_filter_command(settings: Settings) -> tuple[str, ...]:
    try:
        return ("python3", str(settings.remote_script_path("prop_filter")))
    except ValueError:
        return _DEFAULT_PROP_FILTER_COMMAND


def _default_predict_command(settings: Settings) -> tuple[str, ...]:
    try:
        return ("python3", str(settings.remote_script_path("predict")))
    except ValueError:
        return _DEFAULT_PREDICT_COMMAND


# --------------------------------------------------------------------------- #
# Phase A — search                                                            #
# --------------------------------------------------------------------------- #


def _write_queries_file(path: Path, queries: Sequence[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as w:
        for smiles, sid in queries:
            w.write(f"{smiles.strip()} {sid}\n")


def _build_search_command(
    queries_file: Path,
    cycle_dir: Path,
    spacelight: SpacelightAdapter,
    ftrees: FTreesAdapter,
    space: str | Path,
    nnn: int,
    sim_spacelight: float,
    sim_ftrees: float,
    threads: int,
) -> str:
    """Render the per-task bash body for one search task.

    Each task picks its query line from ``queries_file`` via
    ``${TASK_ID}`` and runs SpaceLight + FTrees with output paths
    rooted in ``cycle_dir`` (no scratch dir indirection — the local
    scheduler runs in-place; SLURM users can prepend their own
    scratch lines via ``ArrayJob.env_setup``).
    """
    results_dir = cycle_dir / "results"
    sl_cmd = " ".join(
        spacelight.command_for(
            "$query",
            space,
            results_dir / "spacelightresult_${TASK_ID}.csv",
            max_results=nnn,
            similarity=sim_spacelight,
            threads=threads,
        )
    )
    ft_cmd = " ".join(
        ftrees.command_for(
            "$query",
            space,
            results_dir / "ftreesresult_${TASK_ID}.csv",
            max_results=nnn,
            similarity=sim_ftrees,
            threads=threads,
        )
    )
    # Note: $query is a bash variable; it is intentionally NOT formatted
    # by python here so the local scheduler / sbatch sees the literal.
    return (
        f'mkdir -p {results_dir}\n'
        f'query=$(sed -n "${{TASK_ID}}p" {queries_file} | awk \'{{print $1}}\')\n'
        f'echo "[task ${{TASK_ID}}] query: $query"\n'
        f'{sl_cmd}\n'
        f'echo "[task ${{TASK_ID}}] SpaceLight done"\n'
        f'{ft_cmd}\n'
        f'echo "[task ${{TASK_ID}}] FTrees done"\n'
    )


# --------------------------------------------------------------------------- #
# Phase B — aggregate & control                                               #
# --------------------------------------------------------------------------- #


def _aggregate_search_results(
    cycle_dir: Path,
    field_spacelight: str,
    field_ftrees: str,
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Collapse per-method search CSVs into ``(raw_mols, sims)``.

    ``raw_mols[smiles] = "<smiles>§<title>"`` — the legacy "rawmol"
    packed string used downstream as the SMI input to the prop filter.

    ``sims[method][smiles] = max_similarity`` — keeps the *highest*
    similarity per SMILES across multiple result rows / pagination
    files (legacy behaviour: groupby SMILES, max).
    """
    results_dir = cycle_dir / "results"
    raw_mols: dict[str, str] = {}
    sims: dict[str, dict[str, float]] = {m: {} for m in _SIM_METHODS}
    for method in _SIM_METHODS:
        sim_field = field_spacelight if method == "spacelight" else field_ftrees
        # Match either ``<m>result_<task>.csv`` or legacy paginated
        # ``<m>result_<task>_<page>.csv``.
        for resfile in sorted(results_dir.glob(f"{method}result_*.csv")):
            with resfile.open("rt", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                fields = reader.fieldnames or []
                # Legacy column names: "#result-smiles", "result-name",
                # "fingerprint-similarity" or "pharmacophore-similarity".
                smi_col = "#result-smiles" if "#result-smiles" in fields else "result-smiles"
                title_col = "result-name"
                if smi_col not in fields or title_col not in fields or sim_field not in fields:
                    logger.warning(
                        "skipping %s: missing one of %s",
                        resfile, (smi_col, title_col, sim_field),
                    )
                    continue
                for row in reader:
                    smiles = row[smi_col]
                    title = row[title_col]
                    try:
                        similarity = float(row[sim_field])
                    except (TypeError, ValueError):
                        continue
                    cur = sims[method].get(smiles)
                    if cur is None or cur < similarity:
                        sims[method][smiles] = similarity
                    if smiles not in raw_mols:
                        raw_mols[smiles] = f"{smiles}§{title}"
    return raw_mols, sims


def _write_control_chunks(
    raw_mols: dict[str, str], control_dir: Path, n_chunks: int
) -> int:
    """Distribute raw-mol lines across ``n_chunks`` gzipped chunks.

    Returns the number of chunks actually written (may be less than
    ``n_chunks`` when ``len(raw_mols) < n_chunks``).
    """
    inputs_dir = control_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for stale in inputs_dir.glob("control_*.smi.gz"):
        stale.unlink()
    items = list(raw_mols.values())
    if not items:
        return 0
    n_chunks = max(1, min(n_chunks, len(items)))
    chunk_size = (len(items) + n_chunks - 1) // n_chunks
    written = 0
    for i in range(n_chunks):
        slice_ = items[i * chunk_size : (i + 1) * chunk_size]
        if not slice_:
            break
        path = inputs_dir / f"control_{i + 1}.smi.gz"
        with gzip.open(path, "wt", encoding="utf-8") as w:
            for line in slice_:
                w.write(line + "\n")
        written += 1
    return written


def _write_control_param(path: Path, db: Database) -> None:
    props = db.load_properties()
    if props is None:
        raise RuntimeError(
            "properties table is empty; cannot write control.param"
            " (run `import-seeds` first)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs = {
        "mw": props.mw,
        "slogp": props.slogp,
        "hba": props.hba,
        "hbd": props.hbd,
        "rotbonds": props.rotbonds,
        "tpsa": props.tpsa,
    }
    with path.open("wt", encoding="utf-8") as w:
        for key in _PROP_ORDER:
            lo, hi = pairs[key]
            w.write(f"{lo}\n{hi}\n")


def _write_smarts_file(path: Path, db: Database) -> bool:
    """Write SMARTS include/exclude patterns to *path*.

    Each line is ``<mode>:<pattern>`` where *mode* is ``include`` or
    ``exclude``.  Returns ``True`` if any patterns were written, ``False``
    when the DB has none (file is still written but empty).
    """
    patterns = db.load_smarts_filters()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as w:
        for mode, pattern in patterns:
            w.write(f"{mode}:{pattern}\n")
    return bool(patterns)


def _build_control_command(
    control_dir: Path,
    model_dir: Path,
    settings: Settings,
    prop_filter_prefix: Sequence[str],
    predict_prefix: Sequence[str],
    has_smarts: bool = False,
) -> str:
    """Render the per-task bash body for one control task.

    Each task receives ``${TASK_ID}`` and:

    1. Runs ``prop_filter`` on ``inputs/control_${TASK_ID}.smi.gz``, producing
       ``results_propfilter/propoutput_control_${TASK_ID}.csv``.
    2. Runs ``predict`` on that CSV, producing
       ``results_prediction/predicted_propoutput_control_${TASK_ID}.csv``.
    """
    g = settings.general
    in_smi = "inputs/control_${TASK_ID}.smi.gz"
    propout = "results_propfilter/propoutput_control_${TASK_ID}.csv"
    predout = "results_prediction/predicted_propoutput_control_${TASK_ID}.csv"
    param = "inputs/control.param"

    pf_parts = [*prop_filter_prefix, in_smi, param, "--output", propout]
    if has_smarts:
        pf_parts += ["--smarts", "inputs/smarts.txt"]
    pred_parts = [
        *predict_prefix, propout, str(model_dir), predout,
        "--batch-size", str(g.pred_batch_size),
        "--num-workers", str(g.pred_num_workers),
        "--accelerator", g.pred_accelerator,
        "--devices", g.pred_devices,
    ]
    pf_cmd = " ".join(pf_parts)
    pred_cmd = " ".join(pred_parts)
    return (
        f'mkdir -p results_propfilter results_prediction\n'
        f'echo "[task ${{TASK_ID}}] Property filter"\n'
        f'{pf_cmd}\n'
        f'echo "[task ${{TASK_ID}}] Predicting"\n'
        f'export OMP_NUM_THREADS=1\n'
        f'{pred_cmd}\n'
        f'echo "[task ${{TASK_ID}}] Done"\n'
    )


# --------------------------------------------------------------------------- #
# Phase C — ingest                                                            #
# --------------------------------------------------------------------------- #


def _ingest_predictions(
    control_dir: Path,
) -> tuple[list[tuple[str, str, str]], dict[str, float]]:
    """Read ``results_prediction/predicted_propoutput_control_*.csv`` files.

    :returns: ``(rows, scores)`` where ``rows`` is a list of
        ``(reghash, smiles, title)`` triples in first-seen order
        (deduplicated by reghash with min-score retention) and
        ``scores`` is ``{reghash: min_predicted_score}``.
    """
    pred_dir = control_dir / "results_prediction"
    files = sorted(pred_dir.glob("predicted_propoutput_control_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no predicted_propoutput_control_*.csv under {pred_dir}"
        )
    scores: dict[str, float] = {}
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for f in files:
        with f.open("rt", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            if "smilesid" not in cols or "docking_score" not in cols:
                raise ValueError(
                    f"{f}: expected smilesid + docking_score, got {cols}"
                )
            for row in reader:
                smilesid = row["smilesid"]
                try:
                    score = float(row["docking_score"])
                except (TypeError, ValueError):
                    continue
                parts = smilesid.split("§")
                if len(parts) < 2:
                    continue
                reghash = parts[0]
                smiles = parts[1]
                title = parts[2] if len(parts) >= 3 else ""
                cur = scores.get(reghash)
                if cur is None or score < cur:
                    scores[reghash] = score
                if reghash not in seen:
                    seen.add(reghash)
                    rows.append((reghash, smiles, title))
    return rows, scores


def _existing_reghashes(db: Database, candidates: Iterable[str]) -> set[str]:
    """Return the subset of ``candidates`` already present in ``data.reghash``."""
    cands = list(candidates)
    if not cands:
        return set()
    found: set[str] = set()
    # SQLite parameter limit is 999 by default; chunk to be safe.
    chunk = 500
    for i in range(0, len(cands), chunk):
        batch = cands[i : i + chunk]
        placeholders = ",".join("?" * len(batch))
        sql = f"SELECT reghash FROM data WHERE reghash IN ({placeholders})"
        for (rh,) in db.connection.execute(sql, batch).fetchall():
            if rh is not None:
                found.add(str(rh))
    return found


# --------------------------------------------------------------------------- #
# Top-level orchestration                                                     #
# --------------------------------------------------------------------------- #


def simsearch(
    db: Database,
    workdir: WorkDir,
    scheduler: Scheduler,
    settings: Settings,
    *,
    source: Literal["docked", "predicted"],
    strategy: Literal["greedy", "clustering"],
    top_n: int,
    space: str | Path | None = None,
    nnn: int | None = None,
    sim_spacelight: float | None = None,
    sim_ftrees: float | None = None,
    cpu: int = 1,
    threads_per_task: int = 2,
    spacelight_adapter: SpacelightAdapter | None = None,
    ftrees_adapter: FTreesAdapter | None = None,
    search_command_template: str | None = None,
    control_command_template: str | None = None,
    prop_filter_command_prefix: Sequence[str] | None = None,
    predict_command_prefix: Sequence[str] | None = None,
) -> int:
    """Run one full simsearch cycle.

    :param source: ``docked`` or ``predicted`` — which score column to
        ORDER BY when picking queries.
    :param strategy: ``greedy`` or ``clustering`` — query acquisition
        strategy. ``clustering`` requires cluster assignments to already
        exist (run ``spacehasten cluster`` first, or use
        ``screening-cycle --strategy clustering``, which clusters
        automatically before each round).
    :param top_n: number of queries (and hence search-array size).
    :param space: SpaceLight/FTrees ``.space`` file. Defaults to
        ``settings.paths.spaces_file_default``.
    :param nnn: ``--max-nof-results`` for both search tools. Defaults
        to ``settings.general.nnn_default``.
    :param sim_spacelight: SpaceLight similarity threshold. Defaults
        to ``settings.general.sim_spacelight_default``.
    :param sim_ftrees: FTrees similarity threshold. Defaults to
        ``settings.general.sim_ftrees_default``.
    :param cpu: max concurrent control tasks (also chunk count cap).
    :param threads_per_task: ``--thread-count`` for the search tools.
    :param search_command_template: override the canonical search body
        (used by tests with stub binaries).
    :param control_command_template: override the canonical control
        body (used by tests with stubs).
    :returns: the new simsearch cycle number.
    :raises ValueError: when no candidate queries match the strategy.
    :raises RuntimeError: on scheduler failure or empty search results.
    """
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
    if cpu < 1:
        raise ValueError(f"cpu must be >= 1, got {cpu}")
    if strategy == "clustering" and not db.has_clusters():
        raise ValueError(
            "strategy='clustering' requires cluster assignments, but none exist yet;"
            " run `spacehasten cluster` first (or use"
            " `screening-cycle --strategy clustering`, which clusters automatically)"
        )

    # --- Resolve defaults from settings ---------------------------------- #
    sp_exe = settings.paths.exe_spacelight_default
    ft_exe = settings.paths.exe_ftrees_default
    sl_adapter = spacelight_adapter or SpacelightAdapter(exe=sp_exe)
    ft_adapter = ftrees_adapter or FTreesAdapter(exe=ft_exe)

    space_path = (
        Path(space) if space is not None else Path(settings.paths.spaces_file_default)
    )
    nnn_v = nnn if nnn is not None else settings.general.nnn_default
    sim_sl = (
        sim_spacelight if sim_spacelight is not None
        else settings.general.sim_spacelight_default
    )
    sim_ft = (
        sim_ftrees if sim_ftrees is not None
        else settings.general.sim_ftrees_default
    )

    cycle = db.latest_simsearch_cycle() + 1
    cycle_dir = workdir.simsearch_dir(cycle)
    cycle_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Simsearch cycle %d: source=%s strategy=%s top=%d", cycle, source, strategy, top_n,
    )

    # --- Phase A: pick queries ------------------------------------------- #
    queries = db.select_queries_for_simsearch(source, strategy, top_n)
    if not queries:
        raise ValueError(
            f"no candidate queries for source={source!r} strategy={strategy!r}"
        )
    for _smiles, sid in queries:
        db.mark_as_query(sid, cycle)
    db.commit()

    queries_file = cycle_dir / f"queries_{workdir.name}.smi"
    _write_queries_file(queries_file, queries)
    logger.info("Wrote %d queries to %s", len(queries), queries_file)

    env_setup = [
        line for line in (
            settings.general.prepare_anaconda,
            settings.general.activate_chemprop,
        ) if line
    ]

    search_body = (
        search_command_template
        if search_command_template is not None
        else _build_search_command(
            queries_file=queries_file,
            cycle_dir=cycle_dir,
            spacelight=sl_adapter,
            ftrees=ft_adapter,
            space=space_path,
            nnn=nnn_v,
            sim_spacelight=sim_sl,
            sim_ftrees=sim_ft,
            threads=threads_per_task,
        )
    )
    search_cpus = int(settings.general.cpu_count_search or 1)
    search_job = ArrayJob(
        name=f"search_cycle{cycle}",
        workdir=cycle_dir,
        array_size=len(queries),
        max_concurrent=min(len(queries), cpu),
        cpus_per_task=max(1, search_cpus),
        env_setup=[],
        command_template=search_body,
    )
    handle_a = scheduler.submit_array(search_job)
    logger.info("Submitted search job %s (%d queries)", handle_a.job_id, len(queries))
    res_a = scheduler.wait(handle_a)
    if not res_a.success:
        from spacehasten.scheduler.diagnostics import tail_logs
        raise RuntimeError(
            f"search job {handle_a.job_id} failed; failed task indices: "
            f"{res_a.failed_indices}\n"
            f"--- tail of task logs ---\n{tail_logs(handle_a)}"
        )

    # --- Phase B aggregate & control ------------------------------------- #
    raw_mols, sims = _aggregate_search_results(
        cycle_dir,
        field_spacelight=settings.general.field_similarity_spacelight,
        field_ftrees=settings.general.field_similarity_ftrees,
    )
    logger.info(
        "Aggregated %d unique SMILES from %d spacelight + %d ftrees rows",
        len(raw_mols), len(sims["spacelight"]), len(sims["ftrees"]),
    )
    if not raw_mols:
        logger.warning("simsearch cycle %d: no raw mols; nothing to ingest", cycle)
        return cycle

    control_dir = cycle_dir / "CONTROL"
    n_chunks = _write_control_chunks(raw_mols, control_dir, cpu)
    logger.info("Wrote %d control chunks to %s", n_chunks, control_dir)

    _write_control_param(control_dir / "inputs" / "control.param", db)
    has_smarts = _write_smarts_file(control_dir / "inputs" / "smarts.txt", db)
    if has_smarts:
        logger.info("SMARTS filter active — patterns written to %s", control_dir / "inputs" / "smarts.txt")

    # Resolve the model path. Use the absolute path so the control script
    # can reference it directly without copying into CONTROL/.
    model_version = db.latest_model_version()
    if model_version is None:
        raise RuntimeError(
            "no trained model available; train one before running simsearch"
        )
    bin_path = db.load_model_path(model_version, workdir)
    model_dir = bin_path.parent.parent  # <model_dir>/model_0/pytorch_model.bin

    pf_prefix = (
        prop_filter_command_prefix
        if prop_filter_command_prefix is not None
        else _default_prop_filter_command(settings)
    )
    pred_prefix = (
        predict_command_prefix
        if predict_command_prefix is not None
        else _default_predict_command(settings)
    )
    control_body = (
        control_command_template
        if control_command_template is not None
        else _build_control_command(
            control_dir=control_dir,
            model_dir=model_dir,
            settings=settings,
            prop_filter_prefix=pf_prefix,
            predict_prefix=pred_prefix,
            has_smarts=has_smarts,
        )
    )

    control_cpus = int(settings.general.cpu_count_control or 1)
    control_job = ArrayJob(
        name=f"control_cycle{cycle}",
        workdir=control_dir,
        array_size=n_chunks,
        max_concurrent=min(n_chunks, cpu),
        cpus_per_task=max(1, control_cpus),
        gpus=1,
        env_setup=env_setup,
        command_template=control_body,
    )
    handle_b = scheduler.submit_array(control_job)
    logger.info(
        "Submitted control job %s (%d chunks, model v%d)",
        handle_b.job_id, n_chunks, model_version,
    )
    res_b = scheduler.wait(handle_b)
    if not res_b.success:
        from spacehasten.scheduler.diagnostics import tail_logs
        raise RuntimeError(
            f"control job {handle_b.job_id} failed; failed task indices: "
            f"{res_b.failed_indices}\n"
            f"--- tail of task logs ---\n{tail_logs(handle_b)}"
        )

    # --- Phase C: ingest ------------------------------------------------- #
    rows, scores = _ingest_predictions(control_dir)
    if not rows:
        logger.warning("simsearch cycle %d: no rows survived prediction", cycle)
        return cycle

    existing = _existing_reghashes(db, (rh for rh, _, _ in rows))

    inserted = 0
    for reghash, smiles, title in rows:
        if reghash in existing:
            continue
        sl_sim = sims["spacelight"].get(smiles)
        ft_sim = sims["ftrees"].get(smiles)
        pred_score = scores.get(reghash)
        db.insert_simsearch_hit(
            reghash=reghash,
            smiles=smiles,
            smilesid=title,
            spacelight=sl_sim,
            ftrees=ft_sim,
            pred_score=pred_score,
            simsearch_cycle=cycle,
        )
        inserted += 1
    db.commit()
    logger.info(
        "Inserted %d new compounds (deduped %d existing reghashes) into cycle %d",
        inserted, len(rows) - inserted, cycle,
    )

    return cycle


__all__ = ["simsearch"]
