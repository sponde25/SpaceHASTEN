"""Integration test for the training stage with the local scheduler.

Uses a stub ``remote/train.py`` (a tiny shell script) that just touches
the expected ``model_0/pytorch_model.bin`` output, skipping real chemprop
training. The real chemprop run is exercised by the Session 15 verify
smoke test.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from spacehasten.config.settings import Settings
from spacehasten.core.db import Database
from spacehasten.scheduler import LocalScheduler
from spacehasten.stages.training import train
from spacehasten.workspace.layout import WorkDir
from spacehasten.workspace.manifest import Manifest


def _make_stub_train(tmp_path: Path) -> Path:
    """Write a bash stub that mimics ``remote.train``: makes
    ``<save_dir>/model_0/pytorch_model.bin`` and exits 0.

    The stub also echoes its arguments so the test can verify
    hyper-parameter pass-through if needed.
    """
    stub = tmp_path / "stub_train.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'data_path="$1"\n'
        'save_dir="$2"\n'
        "shift 2\n"
        'mkdir -p "$save_dir/model_0"\n'
        'touch "$save_dir/model_0/pytorch_model.bin"\n'
        # Record args for assertions.
        'echo "$data_path" > "$save_dir/stub_args.txt"\n'
        'echo "$save_dir" >> "$save_dir/stub_args.txt"\n'
        'for a in "$@"; do echo "$a" >> "$save_dir/stub_args.txt"; done\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return stub


def _seed_db_with_dock_scores(db: Database) -> None:
    """Insert a handful of docked rows so training has data."""
    db.create_schema()
    samples = [
        ("h1", "CCO", "ethanol-1", -8.0),
        ("h2", "c1ccccc1", "benzene-1", -7.5),
        ("h3", "Cc1ccccc1", "toluene-1", -7.0),
        ("h4", "Oc1ccccc1", "phenol-1", -6.5),
        ("h5", "Nc1ccccc1", "aniline-1", -6.0),
        # An over-cutoff row that must be excluded:
        ("h6", "CCN", "ethylamine-1", 99.0),
        # NULL row also excluded:
    ]
    for reghash, smi, sid, score in samples:
        db.insert_seed_docked(reghash, smi, sid, score)
    db.insert_seed_undocked("h7", "CCC", "propane-1")
    db.commit()


def test_training_stage_local(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="trainws")
    db_path = workdir.dbsh()
    db = Database(db_path)
    _seed_db_with_dock_scores(db)

    stub = _make_stub_train(tmp_path)
    settings = Settings()  # all defaults

    scheduler = LocalScheduler()
    version = train(
        db,
        workdir,
        scheduler,
        settings,
        cutoff=10.0,
        train_command_prefix=("bash", str(stub)),
    )
    db.close()

    # New version is 0 (no prior models).
    assert version == 0

    # On-disk artefacts.
    model_dir = workdir.model_dir(0)
    assert (model_dir / "train.csv").exists(), "training CSV must be written"
    bin_path = model_dir / "model_0" / "pytorch_model.bin"
    assert bin_path.exists(), "stub must produce pytorch_model.bin"

    # CSV has the right columns and row count (cutoff filter applied).
    csv_text = (model_dir / "train.csv").read_text().splitlines()
    assert csv_text[0] == "smiles,docking_score"
    assert len(csv_text) == 1 + 5, "5 docked rows below cutoff"

    # Compatibility shim: legacy ``models`` table got an empty BLOB.
    db2 = Database(db_path)
    assert db2.latest_model_version() == 0
    assert db2.load_model_blob(0) == b""

    # On-disk path resolution prefers the new layout.
    resolved = db2.load_model_path(0, workdir)
    assert resolved == bin_path
    db2.close()

    # Manifest is the source of truth and records the model.
    manifest = Manifest.load(workdir.manifest_path())
    record = manifest.get_model(0)
    assert record is not None
    assert Path(record.model_dir) == model_dir

    # Verify hyper-parameter pass-through to the stub.
    args_file = (model_dir / "stub_args.txt").read_text().splitlines()
    # data_path, save_dir, then the --flag value pairs.
    assert args_file[0] == str(model_dir / "train.csv")
    assert args_file[1] == str(model_dir)
    flag_pairs = dict(zip(args_file[2::2], args_file[3::2], strict=False))
    assert flag_pairs["--batch-size"] == str(settings.general.train_batch_size)
    assert flag_pairs["--epochs"] == str(settings.general.train_epochs)
    assert flag_pairs["--final-lr"] == str(settings.general.train_final_lr)
    assert flag_pairs["--early-stopping-patience"] == str(
        settings.general.train_early_stopping_patience
    )
    assert flag_pairs["--early-stopping-min-delta"] == str(
        settings.general.train_early_stopping_min_delta
    )
    assert flag_pairs["--validation-fraction"] == str(settings.general.train_validation_fraction)
    assert flag_pairs["--gradient-clip-val"] == str(settings.general.train_gradient_clip_val)
    assert flag_pairs["--precision"] == settings.general.train_precision
    assert flag_pairs["--svdkl-gp-dim"] == str(settings.general.train_svdkl_gp_dim)
    assert flag_pairs["--svdkl-grid-size"] == str(settings.general.train_svdkl_grid_size)
    assert flag_pairs["--svdkl-cholesky-jitter"] == str(
        settings.general.train_svdkl_cholesky_jitter
    )
    assert flag_pairs["--svdkl-feature-transform"] == settings.general.train_svdkl_feature_transform
    assert flag_pairs["--svdkl-tanh-temperature"] == str(
        settings.general.train_svdkl_tanh_temperature
    )
    assert flag_pairs["--seed"] == str(settings.general.train_seed)

    submitted_job = next(iter(scheduler._jobs.values())).spec  # noqa: SLF001
    assert submitted_job.cpus_per_task == int(settings.general.cpu_count_train)


def test_training_stage_raises_on_empty_dataset(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="empty")
    db = Database(workdir.dbsh())
    db.create_schema()
    db.commit()

    settings = Settings()
    scheduler = LocalScheduler()

    with pytest.raises(ValueError, match="no training rows"):
        train(db, workdir, scheduler, settings, cutoff=10.0)
    db.close()


def test_training_stage_warm_starts_subsequent_version(tmp_path: Path) -> None:
    workdir = WorkDir.bootstrap(tmp_path / "ws", name="warmws")
    db = Database(workdir.dbsh())
    _seed_db_with_dock_scores(db)
    stub = _make_stub_train(tmp_path)
    settings = Settings()
    scheduler = LocalScheduler()

    assert train(
        db,
        workdir,
        scheduler,
        settings,
        cutoff=10.0,
        train_command_prefix=("bash", str(stub)),
    ) == 0
    db.insert_seed_docked("h8", "COC", "dimethyl-ether-1", -8.2)
    db.commit()
    assert train(
        db,
        workdir,
        scheduler,
        settings,
        cutoff=10.0,
        train_command_prefix=("bash", str(stub)),
    ) == 1
    db.close()

    model_v0 = workdir.model_dir(0)
    model_v1 = workdir.model_dir(1)
    args = (model_v1 / "stub_args.txt").read_text().splitlines()[2:]
    flag_pairs = dict(zip(args[::2], args[1::2], strict=False))
    assert flag_pairs["--epochs"] == str(settings.general.train_warm_epochs)
    assert flag_pairs["--warmup-epochs"] == str(
        settings.general.train_warm_warmup_epochs
    )
    assert flag_pairs["--init-lr"] == str(settings.general.train_warm_init_lr)
    assert flag_pairs["--max-lr"] == str(settings.general.train_warm_max_lr)
    assert flag_pairs["--final-lr"] == str(settings.general.train_warm_final_lr)
    assert flag_pairs["--parent-checkpoint"] == str(
        model_v0 / "model_0" / "pytorch_model.bin"
    )
    assert flag_pairs["--previous-data-path"] == str(model_v0 / "train.csv")
    assert flag_pairs["--new-data-repeat"] == "2"
    assert "--pin-memory" in args
    assert "--non-blocking" in args
    assert "--defer-batch-metrics" in args


def test_load_model_path_falls_back_to_blob(tmp_path: Path) -> None:
    """Legacy databases without an on-disk file must still resolve via BLOB."""
    import io
    import tarfile

    workdir = WorkDir.bootstrap(tmp_path / "ws", name="legacy")
    db = Database(workdir.dbsh())
    db.create_schema()

    # Build a legacy-style tar containing model_0/pytorch_model.bin.
    fake_payload = b"legacy-checkpoint-bytes"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="model_0/pytorch_model.bin")
        info.size = len(fake_payload)
        tar.addfile(info, io.BytesIO(fake_payload))
    db.store_model_blob(2, buf.getvalue())
    db.commit()

    # No on-disk model_0/pytorch_model.bin exists yet.
    bin_path = workdir.model_dir(2) / "model_0" / "pytorch_model.bin"
    assert not bin_path.exists()

    resolved = db.load_model_path(2, workdir)
    assert resolved == bin_path
    assert resolved.read_bytes() == fake_payload
    db.close()
