"""Training stage — orchestrates one chemprop training run.

Replaces legacy ``training_functions.train_new_model``. This module owns
the orchestration only; the chemprop training itself runs on the compute
node via :mod:`spacehasten.remote.train`.

Key change from legacy: trained models are stored on-disk under
``workdir.model_dir(version)/model_0/pytorch_model.bin`` and recorded in
the JSON manifest. The legacy ``models`` SQL table is still updated
(``model_tar = b""``) to keep older readers happy until the cutover.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from spacehasten.config.settings import Settings
from spacehasten.core.db import Database
from spacehasten.scheduler.base import ArrayJob, Scheduler
from spacehasten.workspace.layout import WorkDir
from spacehasten.workspace.manifest import Manifest

logger = logging.getLogger(__name__)


_DEFAULT_TRAIN_COMMAND: tuple[str, ...] = (
    "python3",
    "-m",
    "spacehasten.remote.train",
)
"""Legacy fallback command. Production runs derive the prefix from
``Settings.remote_script_path('train')`` so compute nodes do not need
``spacehasten`` importable in the chemprop conda env."""


def _default_train_command(settings: Settings) -> tuple[str, ...]:
    """Resolve the default ``train`` command prefix from ``settings``.

    Falls back to the legacy ``-m spacehasten.remote.train`` form only
    when ``paths.spacehasten_src_dir`` is unset (e.g. in unit tests that
    do not configure it).
    """
    try:
        return ("python3", str(settings.remote_script_path("train")))
    except ValueError:
        return _DEFAULT_TRAIN_COMMAND


def _build_train_command(
    csv_path: Path,
    model_dir: Path,
    settings: Settings,
    command_prefix: Sequence[str],
) -> str:
    """Build the bash command for ``remote.train``.

    Mirrors the hyper-parameter pass-through of legacy
    ``scheduler_functions.write_train_scheduler``.
    """
    g = settings.general
    parts: list[str] = [*command_prefix, str(csv_path), str(model_dir)]
    parts += [
        "--batch-size", str(g.train_batch_size),
        "--epochs", str(g.train_epochs),
        "--num-workers", str(g.train_num_workers),
        "--devices", g.train_devices,
        "--mp-hidden-size", str(g.train_mp_hidden_size),
        "--mp-depth", str(g.train_mp_depth),
        "--ffn-hidden-size", str(g.train_ffn_hidden_size),
        "--ffn-layers", str(g.train_ffn_layers),
        "--dropout", str(g.train_dropout),
        "--activation", g.train_activation,
        "--batch-norm", str(g.train_batch_norm),
        "--warmup-epochs", str(g.train_warmup_epochs),
        "--init-lr", str(g.train_init_lr),
        "--max-lr", str(g.train_max_lr),
        "--final-lr", str(g.train_final_lr),
        "--early-stopping-patience", str(g.train_early_stopping_patience),
        "--early-stopping-min-delta", str(g.train_early_stopping_min_delta),
        "--svdkl-gp-dim", str(g.train_svdkl_gp_dim),
        "--svdkl-grid-size", str(g.train_svdkl_grid_size),
        "--svdkl-grid-lower", str(g.train_svdkl_grid_lower),
        "--svdkl-grid-upper", str(g.train_svdkl_grid_upper),
        "--seed", str(g.train_seed),
    ]
    cmd = " ".join(parts)
    return (
        f'echo "Starting chemprop training (csv={csv_path.name}, model_dir={model_dir.name})"\n'
        f'{cmd}\n'
        f'echo "Training complete"'
    )


def _write_training_csv(rows: Sequence[tuple[str, float]], csv_path: Path) -> None:
    """Write training rows to a CSV with columns ``smiles, docking_score``."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["smiles", "docking_score"])
    df.to_csv(csv_path, index=False)


def train(
    db: Database,
    workdir: WorkDir,
    scheduler: Scheduler,
    settings: Settings,
    *,
    cutoff: float = 10.0,
    train_command_prefix: Sequence[str] | None = None,
) -> int:
    """Run one chemprop training round.

    Pulls ``(smiles, dock_score)`` rows from the database (filtered by
    ``cutoff``), writes them to ``workdir.model_dir(next_version)/train.csv``,
    submits a single-task array job that invokes
    :mod:`spacehasten.remote.train`, waits for completion, and on success
    records the new model version in both the manifest (source of truth)
    and the legacy ``models`` table (compatibility shim).

    :param cutoff: ``dock_score < cutoff`` filter for training rows
        (matches legacy ``TRAIN_DOCKING_CUTOFF``).
    :param train_command_prefix: command to launch ``remote.train``.
        Override in tests to point at a stub script.
    :returns: the new model version.
    :raises RuntimeError: if the training job fails on the scheduler.
    :raises ValueError: if the training set is empty.
    """
    rows = db.select_training_data(cutoff)
    if not rows:
        raise ValueError(
            f"no training rows (dock_score < {cutoff} AND NOT NULL); "
            "run docking before training"
        )

    latest = db.latest_model_version()
    next_version = 0 if latest is None else latest + 1
    model_dir = workdir.model_dir(next_version)
    model_dir.mkdir(parents=True, exist_ok=True)

    csv_path = model_dir / "train.csv"
    _write_training_csv(rows, csv_path)
    logger.info("Wrote %d training rows to %s", len(rows), csv_path)

    env_setup = [
        line
        for line in (
            settings.general.prepare_anaconda,
            settings.general.activate_chemprop,
        )
        if line
    ]

    prefix = (
        train_command_prefix
        if train_command_prefix is not None
        else _default_train_command(settings)
    )
    command = _build_train_command(csv_path, model_dir, settings, prefix)

    job = ArrayJob(
        name=f"train_v{next_version}",
        workdir=model_dir,
        array_size=1,
        max_concurrent=1,
        cpus_per_task=max(1, int(settings.general.cpu_count_train or 1)),
        gpus=1,
        exclusive=settings.general.gpu_exclusive == "1",
        env_setup=env_setup,
        command_template=command,
    )
    handle = scheduler.submit_array(job)
    logger.info("Submitted training job %s (version %d)", handle.job_id, next_version)
    result = scheduler.wait(handle)
    if not result.success:
        from spacehasten.scheduler.diagnostics import tail_logs
        raise RuntimeError(
            f"training job {handle.job_id} failed; failed task indices: "
            f"{result.failed_indices}\n"
            f"--- tail of task logs ---\n{tail_logs(handle)}"
        )

    bin_path = model_dir / "model_0" / "pytorch_model.bin"
    if not bin_path.exists():
        from spacehasten.scheduler.diagnostics import tail_logs
        raise RuntimeError(
            f"training job reported success but {bin_path} is missing\n"
            f"--- tail of task logs ---\n{tail_logs(handle)}"
        )

    # Compatibility shim: leave a row in the legacy ``models`` table with
    # an empty BLOB so older readers see the version. The on-disk file is
    # the source of truth.
    db.store_model_blob(next_version, b"")
    db.commit()

    # Manifest is the source of truth for the model registry.
    manifest_path = workdir.manifest_path()
    if manifest_path.exists():
        manifest = Manifest.load(manifest_path)
    else:
        manifest = Manifest(name=workdir.name)
    manifest.record_model(next_version, model_dir)
    manifest.save(manifest_path)
    logger.info("Recorded model version %d at %s", next_version, model_dir)
    return next_version


__all__ = ["train"]
