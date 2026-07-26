#!/usr/bin/env python3
"""Remote Chemprop + SVDKL training entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import resource
import sys
import tempfile
import time
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import gpytorch
    import torch
    from chemprop import data, featurizers
    from chemprop.data.collate import collate_batch
    from chemprop.schedulers import build_NoamLike_LRSched
    from torch.utils.data import DataLoader
except ImportError as e:  # pragma: no cover - import guarded for remote node only
    print(
        "Error: Failed to import chemprop, torch, or gpytorch. "
        f"Make sure they are installed in the Chemprop environment: {e}"
    )
    sys.exit(1)

try:  # script path execution on compute nodes does not import the package root
    from spacehasten.remote.svdkl import (
        ChempropConfig,
        ChempropSVDKLModel,
        SVDKLConfig,
        SVDKLHead,
        build_chemprop_mpnn,
        enable_batch_mol_graph_pinning,
        fit_target_scaler,
        load_chemprop_svdkl_checkpoint,
        move_batch_to_device,
        predictive_mean_stds,
        save_chemprop_svdkl_checkpoint,
        scale_targets,
    )

except ImportError:  # pragma: no cover - exercised by file-path remote execution
    from svdkl import (  # type: ignore[no-redef]
        ChempropConfig,
        ChempropSVDKLModel,
        SVDKLConfig,
        SVDKLHead,
        build_chemprop_mpnn,
        enable_batch_mol_graph_pinning,
        fit_target_scaler,
        load_chemprop_svdkl_checkpoint,
        move_batch_to_device,
        predictive_mean_stds,
        save_chemprop_svdkl_checkpoint,
        scale_targets,
    )

try:
    from spacehasten.core.calibration import fit_gaussian_calibration
except ImportError:  # pragma: no cover - exercised by file-path remote execution
    package_root = str(Path(__file__).resolve().parents[2])
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from spacehasten.core.calibration import fit_gaussian_calibration

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def deterministic_split_indices(
    row_count: int,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic random train/validation row indices."""
    if row_count < 2:
        raise ValueError("at least two rows are required")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    validation_count = min(
        row_count - 1,
        max(1, int(round(validation_fraction * row_count))),
    )
    permutation = np.random.default_rng(seed).permutation(row_count)
    validation_indices = np.sort(permutation[:validation_count])
    train_indices = np.sort(permutation[validation_count:])
    return train_indices, validation_indices


def identify_new_row_indices(
    current: pd.DataFrame,
    previous: pd.DataFrame,
) -> np.ndarray:
    """Return current-row indices whose SMILES were absent from the previous dataset."""

    required_columns = {"smiles", "docking_score"}
    if not required_columns <= set(current.columns) or not required_columns <= set(
        previous.columns
    ):
        raise ValueError("current and previous training data require smiles and docking_score")
    if current["smiles"].duplicated().any() or previous["smiles"].duplicated().any():
        raise ValueError("warm-start training requires unique SMILES in both datasets")

    previous_scores = previous.set_index("smiles")["docking_score"]
    previous_mask = current["smiles"].isin(previous_scores.index).to_numpy()
    if int(previous_mask.sum()) != len(previous):
        raise ValueError("current training data does not contain every previous SMILES")
    preserved_scores = current.loc[previous_mask, "smiles"].map(previous_scores).to_numpy()
    if not np.array_equal(
        preserved_scores,
        current.loc[previous_mask, "docking_score"].to_numpy(),
    ):
        raise ValueError("docking scores changed for previously trained SMILES")
    return np.flatnonzero(~previous_mask).astype(np.int64)


def _format_optional(value: object | None) -> str:
    if value is None:
        return "nan"
    if hasattr(value, "detach"):
        value = value.detach().cpu()  # type: ignore[union-attr]
    return f"{float(value):.4g}"


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _max_rss_bytes() -> int:
    """Return peak resident memory for the training process on Linux."""

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _mean_batch_metric(
    *,
    count: int,
    host_sum: float,
    device_sum: torch.Tensor | None,
) -> float:
    if count == 0:
        return float("nan")
    if device_sum is not None:
        return float((device_sum / count).detach().cpu())
    return host_sum / count


def _cuda_event_seconds(event_pairs: list[tuple[Any, Any]]) -> float | None:
    if not event_pairs:
        return None
    torch.cuda.synchronize()
    return sum(start.elapsed_time(stop) for start, stop in event_pairs) / 1000.0


def train_model(
    data_path: str,
    save_dir: str,
    batch_size: int,
    epochs: int,
    num_workers: int,
    devices: str,
    mp_hidden_size: int,
    mp_depth: int,
    ffn_hidden_size: int,
    ffn_layers: int,
    dropout: float,
    activation: str,
    batch_norm: bool,
    warmup_epochs: int,
    init_lr: float,
    max_lr: float,
    final_lr: float,
    svdkl_gp_dim: int = 16,
    svdkl_grid_size: int = 128,
    svdkl_grid_lower: float = -10.0,
    svdkl_grid_upper: float = 10.0,
    svdkl_cholesky_jitter: float = 1e-3,
    svdkl_feature_transform: str = "scale_to_bounds",
    svdkl_tanh_temperature: float = 3.0,
    seed: int = 0,
    early_stopping_patience: int = 8,
    early_stopping_min_delta: float = 0.0,
    validation_fraction: float = 0.1,
    gradient_clip_val: float = 5.0,
    precision: str = "32-true",
    cache_molgraphs: bool = False,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    non_blocking: bool = False,
    defer_batch_metrics: bool = False,
    prefetch_factor: int = 2,
    profile: bool = False,
    parent_checkpoint: str | None = None,
    previous_data_path: str | None = None,
    new_data_repeat: int = 1,
    fit_gaussian_calibrator: bool = False,
) -> int:
    profile_started = time.monotonic()
    timing_profile: dict[str, Any] = {
        "settings": {
            "batch_size": batch_size,
            "epochs": epochs,
            "num_workers": num_workers,
            "cache_molgraphs": cache_molgraphs,
            "persistent_workers": persistent_workers,
            "pin_memory": pin_memory,
            "non_blocking": non_blocking,
            "defer_batch_metrics": defer_batch_metrics,
            "prefetch_factor": prefetch_factor,
            "warm_start": parent_checkpoint is not None,
            "parent_checkpoint": parent_checkpoint,
            "previous_data_path": previous_data_path,
            "new_data_repeat": new_data_repeat,
        },
        "phases": {},
        "epochs": [],
    }
    _seed_everything(seed)
    torch.set_float32_matmul_precision("medium")

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    phase_started = time.monotonic()
    df = pd.read_csv(data_path)
    timing_profile["phases"]["csv_read_s"] = time.monotonic() - phase_started
    if "smiles" not in df.columns or "docking_score" not in df.columns:
        logger.error("Training CSV must have 'smiles' and 'docking_score' columns")
        return 1

    smiles = df["smiles"].astype(str).to_numpy()
    targets = df["docking_score"].astype(float).to_numpy().reshape(-1, 1)

    if len(smiles) < 2:
        logger.error("Need at least 2 training rows to build train/validation splits")
        return 1
    if ffn_layers < 1:
        logger.error("SVDKL requires at least 1 FFN layer to produce its input embedding")
        return 1
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        logger.error("warmup_epochs must satisfy 0 <= warmup_epochs < epochs")
        return 1
    if early_stopping_patience < 1:
        logger.error("early_stopping_patience must be at least 1")
        return 1
    if early_stopping_min_delta < 0:
        logger.error("early_stopping_min_delta must be non-negative")
        return 1
    if not 0.0 < validation_fraction < 1.0:
        logger.error("validation_fraction must be in (0, 1)")
        return 1
    if gradient_clip_val <= 0:
        logger.error("gradient_clip_val must be positive")
        return 1
    if precision not in {"32", "32-true"}:
        logger.error("only float32 training is supported; got precision=%s", precision)
        return 1
    if persistent_workers and num_workers == 0:
        logger.error("persistent_workers requires num_workers > 0")
        return 1
    if prefetch_factor < 1:
        logger.error("prefetch_factor must be at least 1")
        return 1
    if (parent_checkpoint is None) != (previous_data_path is None):
        logger.error("parent_checkpoint and previous_data_path must be provided together")
        return 1
    if new_data_repeat < 1:
        logger.error("new_data_repeat must be at least 1")
        return 1
    if parent_checkpoint is None and new_data_repeat != 1:
        logger.error("new_data_repeat requires warm-start inputs")
        return 1

    logger.info("Loaded %d molecules for training", len(smiles))

    phase_started = time.monotonic()
    train_indices, val_indices = deterministic_split_indices(
        len(smiles), validation_fraction=validation_fraction, seed=seed
    )
    warm_start = parent_checkpoint is not None
    model: ChempropSVDKLModel | None = None
    chemprop_config: ChempropConfig | None = None
    parent_metadata: dict[str, Any] = {}
    new_source_indices = np.array([], dtype=np.int64)
    new_train_indices = np.array([], dtype=np.int64)
    if warm_start:
        assert parent_checkpoint is not None
        assert previous_data_path is not None
        parent_path = Path(parent_checkpoint)
        previous_path = Path(previous_data_path)
        if not parent_path.exists() or not previous_path.exists():
            logger.error(
                "Warm-start input missing: checkpoint=%s previous_data=%s",
                parent_path,
                previous_path,
            )
            return 1
        try:
            model, target_scaler, parent_metadata = load_chemprop_svdkl_checkpoint(parent_path)
        except (KeyError, TypeError, ValueError) as error:
            logger.error("Could not load warm-start checkpoint %s: %s", parent_path, error)
            return 1
        try:
            chemprop_config = ChempropConfig(**parent_metadata["chemprop_config"])
        except (KeyError, TypeError) as error:
            logger.error("Parent checkpoint lacks Chemprop configuration: %s", error)
            return 1
        previous_df = pd.read_csv(previous_path)
        try:
            new_source_indices = identify_new_row_indices(df, previous_df)
        except ValueError as error:
            logger.error("Warm-start dataset validation failed: %s", error)
            return 1
        new_train_indices = np.intersect1d(
            train_indices,
            new_source_indices,
            assume_unique=True,
        )
        effective_train_indices = np.concatenate(
            [train_indices, *([new_train_indices] * (new_data_repeat - 1))]
        )
        logger.info(
            "Warm start: parent_rows=%d new_rows=%d new_train_rows=%d repeat=%d ",
            len(previous_df),
            len(new_source_indices),
            len(new_train_indices),
            new_data_repeat,
        )
    else:
        effective_train_indices = train_indices
        raw_scaler_targets = targets[train_indices].reshape(-1)
        target_scaler = fit_target_scaler(raw_scaler_targets)

    train_smiles = smiles[effective_train_indices]
    raw_train_targets = targets[effective_train_indices]
    val_smiles, raw_val_targets = smiles[val_indices], targets[val_indices]
    train_targets = scale_targets(raw_train_targets.reshape(-1), target_scaler).reshape(-1, 1)
    val_targets = scale_targets(raw_val_targets.reshape(-1), target_scaler).reshape(-1, 1)
    timing_profile["phases"]["split_and_scale_s"] = time.monotonic() - phase_started
    logger.info(
        "Split: %d unique train / %d effective train / %d val",
        len(train_indices),
        len(train_smiles),
        len(val_smiles),
    )

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    phase_started = time.monotonic()
    train_data = [
        data.MoleculeDatapoint.from_smi(smi=smi, y=np.asarray(y, dtype=float))
        for smi, y in zip(train_smiles, train_targets, strict=False)
    ]
    val_data = [
        data.MoleculeDatapoint.from_smi(smi=smi, y=np.asarray(y, dtype=float))
        for smi, y in zip(val_smiles, val_targets, strict=False)
    ]
    timing_profile["phases"]["datapoint_build_s"] = time.monotonic() - phase_started

    phase_started = time.monotonic()
    train_ds = data.MoleculeDataset(train_data, featurizer)
    val_ds = data.MoleculeDataset(val_data, featurizer)
    timing_profile["phases"]["dataset_init_s"] = time.monotonic() - phase_started

    cache_started = time.monotonic()
    if cache_molgraphs:
        logger.info(
            "Caching MolGraphs for %d train and %d validation molecules",
            len(train_ds),
            len(val_ds),
        )
        train_ds.cache = True
        val_ds.cache = True
        logger.info("MolGraph caching completed in %.1f seconds", time.monotonic() - cache_started)
    timing_profile["phases"]["molgraph_cache_s"] = time.monotonic() - cache_started
    timing_profile["phases"]["rss_after_dataset_bytes"] = _max_rss_bytes()

    if pin_memory:
        enable_batch_mol_graph_pinning()

    effective_batch_norm = chemprop_config.batch_norm if chemprop_config is not None else batch_norm
    drop_last_train = effective_batch_norm and len(train_ds) % batch_size == 1
    loader_options: dict[str, Any] = {
        "num_workers": num_workers,
        "collate_fn": collate_batch,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_options["persistent_workers"] = persistent_workers
        loader_options["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
        **loader_options,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    phase_started = time.monotonic()
    if model is None:
        chemprop_config = ChempropConfig(
            mp_hidden_size=mp_hidden_size,
            mp_depth=mp_depth,
            ffn_hidden_size=ffn_hidden_size,
            ffn_layers=ffn_layers,
            dropout=dropout,
            activation=activation,
            batch_norm=batch_norm,
            warmup_epochs=warmup_epochs,
            init_lr=init_lr,
            max_lr=max_lr,
            final_lr=final_lr,
        )
        mpnn = build_chemprop_mpnn(chemprop_config)
        head = SVDKLHead(
            SVDKLConfig(
                input_dim=ffn_hidden_size,
                gp_dim=svdkl_gp_dim,
                grid_size=svdkl_grid_size,
                grid_lower=svdkl_grid_lower,
                grid_upper=svdkl_grid_upper,
                cholesky_jitter=svdkl_cholesky_jitter,
                feature_transform=svdkl_feature_transform,
                tanh_temperature=svdkl_tanh_temperature,
            )
        )
        model = ChempropSVDKLModel(mpnn, head, embedding_i=chemprop_config.embedding_i)
    assert chemprop_config is not None

    try:
        n_devices = int(devices)
    except ValueError:
        n_devices = 1

    device = _select_device(n_devices)
    model.to(device)
    timing_profile["phases"]["model_init_s"] = time.monotonic() - phase_started
    logger.info("Training SVDKL model on %s", device)

    mll = gpytorch.mlls.PredictiveLogLikelihood(
        likelihood=model.likelihood,
        model=model.gp_layer,
        num_data=len(train_smiles) - int(drop_last_train),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    steps_per_epoch = len(train_loader)
    scheduler = build_NoamLike_LRSched(
        optimizer,
        warmup_steps=warmup_epochs * steps_per_epoch,
        cooldown_steps=(epochs - warmup_epochs) * steps_per_epoch,
        init_lr=init_lr,
        max_lr=max_lr,
        final_lr=final_lr,
    )

    timing_profile["phases"]["setup_total_s"] = time.monotonic() - profile_started
    timing_profile["phases"]["rss_before_training_bytes"] = _max_rss_bytes()
    training_started = time.monotonic()
    best_val_nll = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epochs_completed = 0
    stop_reason = "max_epochs"
    for epoch in range(epochs):
        epoch_started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        model.likelihood.train()
        train_loss_host_sum = 0.0
        gradient_norm_host_sum = 0.0
        train_loss_device_sum = torch.zeros((), device=device) if defer_batch_metrics else None
        gradient_norm_device_sum = torch.zeros((), device=device) if defer_batch_metrics else None
        train_batch_count = 0
        train_row_count = 0
        train_data_wait_s = 0.0
        train_transfer_enqueue_s = 0.0
        train_cuda_events: list[tuple[Any, Any]] = []
        train_started = time.monotonic()
        iterator_started = time.monotonic()
        train_iterator = iter(train_loader)
        train_iterator_start_s = time.monotonic() - iterator_started
        while True:
            fetch_started = time.monotonic() if profile else 0.0
            try:
                batch = next(train_iterator)
            except StopIteration:
                break
            if profile:
                train_data_wait_s += time.monotonic() - fetch_started
            cuda_start = cuda_stop = None
            if profile and device.type == "cuda":
                cuda_start = torch.cuda.Event(enable_timing=True)
                cuda_stop = torch.cuda.Event(enable_timing=True)
                cuda_start.record()
            transfer_started = time.monotonic() if profile else 0.0
            batch = move_batch_to_device(batch, device, non_blocking=non_blocking)
            if profile:
                train_transfer_enqueue_s += time.monotonic() - transfer_started
            target = _batch_targets(batch, device)
            train_row_count += int(target.shape[0])
            optimizer.zero_grad(set_to_none=True)
            loss = -mll(model(batch), target)
            if not defer_batch_metrics and not torch.isfinite(loss):
                logger.error("Non-finite training loss at epoch %d", epoch + 1)
                return 1
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            optimizer.step()
            scheduler.step()
            if cuda_stop is not None and cuda_start is not None:
                cuda_stop.record()
                train_cuda_events.append((cuda_start, cuda_stop))
            if defer_batch_metrics:
                assert train_loss_device_sum is not None
                assert gradient_norm_device_sum is not None
                train_loss_device_sum.add_(loss.detach())
                gradient_norm_device_sum.add_(gradient_norm.detach())
            else:
                train_loss_host_sum += float(loss.detach().cpu())
                gradient_norm_host_sum += float(gradient_norm.detach().cpu())
            train_batch_count += 1
        train_gpu_batch_s = _cuda_event_seconds(train_cuda_events) if profile else None
        train_wall_s = time.monotonic() - train_started
        train_loss = _mean_batch_metric(
            count=train_batch_count,
            host_sum=train_loss_host_sum,
            device_sum=train_loss_device_sum,
        )
        mean_gradient_norm = _mean_batch_metric(
            count=train_batch_count,
            host_sum=gradient_norm_host_sum,
            device_sum=gradient_norm_device_sum,
        )
        if not np.isfinite(train_loss):
            logger.error("Non-finite training loss at epoch %d", epoch + 1)
            return 1

        val_loss_host_sum = 0.0
        val_loss_device_sum = torch.zeros((), device=device) if defer_batch_metrics else None
        val_batch_count = 0
        val_row_count = 0
        val_data_wait_s = 0.0
        val_transfer_enqueue_s = 0.0
        val_cuda_events: list[tuple[Any, Any]] = []
        model.eval()
        model.likelihood.eval()
        val_started = time.monotonic()
        iterator_started = time.monotonic()
        val_iterator = iter(val_loader)
        val_iterator_start_s = time.monotonic() - iterator_started
        with torch.no_grad():
            while True:
                fetch_started = time.monotonic() if profile else 0.0
                try:
                    batch = next(val_iterator)
                except StopIteration:
                    break
                if profile:
                    val_data_wait_s += time.monotonic() - fetch_started
                cuda_start = cuda_stop = None
                if profile and device.type == "cuda":
                    cuda_start = torch.cuda.Event(enable_timing=True)
                    cuda_stop = torch.cuda.Event(enable_timing=True)
                    cuda_start.record()
                transfer_started = time.monotonic() if profile else 0.0
                batch = move_batch_to_device(batch, device, non_blocking=non_blocking)
                if profile:
                    val_transfer_enqueue_s += time.monotonic() - transfer_started
                target = _batch_targets(batch, device)
                val_row_count += int(target.shape[0])
                val_loss = -mll(model(batch), target)
                if cuda_stop is not None and cuda_start is not None:
                    cuda_stop.record()
                    val_cuda_events.append((cuda_start, cuda_stop))
                if defer_batch_metrics:
                    assert val_loss_device_sum is not None
                    val_loss_device_sum.add_(val_loss.detach())
                else:
                    val_loss_host_sum += float(val_loss.detach().cpu())
                val_batch_count += 1
        val_gpu_batch_s = _cuda_event_seconds(val_cuda_events) if profile else None
        val_wall_s = time.monotonic() - val_started
        val_nll = _mean_batch_metric(
            count=val_batch_count,
            host_sum=val_loss_host_sum,
            device_sum=val_loss_device_sum,
        )
        if not np.isfinite(val_nll):
            logger.error("Non-finite validation NLL at epoch %d", epoch + 1)
            return 1

        improved = val_nll < best_val_nll - early_stopping_min_delta
        checkpoint_snapshot_started = time.monotonic()
        if improved:
            best_val_nll = val_nll
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        checkpoint_snapshot_s = time.monotonic() - checkpoint_snapshot_started

        epochs_completed = epoch + 1
        elapsed = time.monotonic() - training_started
        mean_epoch_seconds = elapsed / epochs_completed
        remaining_epochs = max(0, epochs - epochs_completed)
        eta_seconds = mean_epoch_seconds * remaining_epochs
        current_lr = float(optimizer.param_groups[0]["lr"])
        noise = float(model.likelihood.base_likelihood.noise.detach().mean().cpu())
        epoch_seconds = time.monotonic() - epoch_started
        epoch_profile = {
            "epoch": epochs_completed,
            "train_batches": train_batch_count,
            "train_rows": train_row_count,
            "train_wall_s": train_wall_s,
            "train_iterator_start_s": train_iterator_start_s,
            "train_data_wait_s": train_data_wait_s,
            "train_transfer_enqueue_s": train_transfer_enqueue_s,
            "train_gpu_batch_s": train_gpu_batch_s,
            "train_rows_per_s": train_row_count / train_wall_s,
            "val_batches": val_batch_count,
            "val_rows": val_row_count,
            "val_wall_s": val_wall_s,
            "val_iterator_start_s": val_iterator_start_s,
            "val_data_wait_s": val_data_wait_s,
            "val_transfer_enqueue_s": val_transfer_enqueue_s,
            "val_gpu_batch_s": val_gpu_batch_s,
            "val_rows_per_s": val_row_count / val_wall_s,
            "checkpoint_snapshot_s": checkpoint_snapshot_s,
            "epoch_wall_s": epoch_seconds,
            "train_loss": train_loss,
            "val_nll": val_nll,
            "mean_gradient_norm": mean_gradient_norm,
            "max_rss_bytes": _max_rss_bytes(),
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
            "cuda_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
            ),
        }
        timing_profile["epochs"].append(epoch_profile)
        logger.info(
            "Epoch %d/%d train_loss=%.4f val_nll=%.4f best_val_nll=%.4f "
            "best_epoch=%d patience=%d/%d lr=%.3g grad_norm=%.3g noise=%.3g "
            "raw_features=[%s,%s] gp_features=[%s,%s] epoch_s=%.1f "
            "elapsed_s=%.1f eta_s=%.1f",
            epoch + 1,
            epochs,
            train_loss,
            val_nll,
            best_val_nll,
            best_epoch,
            epochs_without_improvement,
            early_stopping_patience,
            current_lr,
            mean_gradient_norm,
            noise,
            _format_optional(model.head.last_raw_feature_min),
            _format_optional(model.head.last_raw_feature_max),
            _format_optional(model.head.last_transformed_feature_min),
            _format_optional(model.head.last_transformed_feature_max),
            epoch_seconds,
            elapsed,
            eta_seconds,
        )
        if epochs_without_improvement >= early_stopping_patience:
            stop_reason = "early_stopping"
            logger.info(
                "Early stopping after epoch %d; best epoch=%d val_nll=%.4f",
                epochs_completed,
                best_epoch,
                best_val_nll,
            )
            break

    if best_state is None:
        logger.error("Training completed without a finite best checkpoint")
        return 1
    model.load_state_dict(best_state)

    checkpoint_dir = save_dir_path / "model_0"
    bin_path = checkpoint_dir / "pytorch_model.bin"
    split_hash = hashlib.sha256(val_indices.astype(np.int64).tobytes()).hexdigest()
    calibration_metadata: dict[str, Any] | None = None
    if fit_gaussian_calibrator:
        logger.info("Fitting Gaussian calibrator on %d deterministic validation rows", len(val_ds))
        model.eval()
        model.likelihood.eval()
        validation_embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device, non_blocking=non_blocking)
                validation_embeddings.append(model.encode_batch(batch).detach().cpu())
        if not validation_embeddings:
            logger.error("No validation embeddings available for Gaussian calibration")
            return 1
        embeddings = torch.cat(validation_embeddings, dim=0).to(device)
        mean, epistemic_std, _, _ = predictive_mean_stds(
            model.head, embeddings, target_scaler=target_scaler
        )
        raw_mean = mean.detach().cpu().numpy().reshape(-1)
        raw_std = epistemic_std.detach().cpu().numpy().reshape(-1)
        raw_targets = raw_val_targets.reshape(-1)
        if len(raw_mean) != len(raw_targets):
            logger.error(
                "Calibration prediction row count (%d) differs from validation rows (%d)",
                len(raw_mean),
                len(raw_targets),
            )
            return 1
        try:
            calibration = fit_gaussian_calibration(raw_mean, raw_std, raw_targets)
        except (RuntimeError, ValueError) as error:
            logger.error("Gaussian calibration fit failed: %s", error)
            return 1
        calibration_payload = {
            "schema_version": 1,
            "calibration_kind": "gaussian_affine_std_floor",
            "uncertainty_source": "epistemic",
            "parameters": {
                "mean_shift": calibration.mean_shift,
                "std_scale": calibration.std_scale,
                "std_floor": calibration.std_floor,
            },
            "fit_source": "deterministic_validation_split",
            "fit_split_name": "validation",
            "fit_row_count": len(raw_targets),
            "validation_indices_sha256": split_hash,
            "validation_early_stopping_overlap": True,
            "objective": {
                "name": "mean_gaussian_nll_without_constant",
                "value": calibration.objective,
            },
        }
        calibration_path = save_dir_path / "calibration.json"
        _atomic_write_json(calibration_path, calibration_payload)
        calibration_metadata = {
            "payload": calibration_payload,
            "artifact": "calibration.json",
            "artifact_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        }
        logger.info("Gaussian calibration fit complete: objective=%.6g", calibration.objective)
    model.cpu()
    metadata = {
        "train_rows": int(len(train_smiles)),
        "unique_train_rows": int(len(train_indices)),
        "warm_start": warm_start,
        "parent_checkpoint": parent_checkpoint,
        "previous_data_path": previous_data_path,
        "new_data_repeat": new_data_repeat,
        "new_rows": int(len(new_source_indices)),
        "new_train_rows": int(len(new_train_indices)),
        "val_rows": int(len(val_smiles)),
        "validation_fraction": validation_fraction,
        "validation_indices_sha256": split_hash,
        "split_seed": seed,
        "epochs_requested": epochs,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "best_val_nll": best_val_nll,
        "stop_reason": stop_reason,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "gradient_clip_val": gradient_clip_val,
        "precision": precision,
        "seed": seed,
        "target_mean": float(target_scaler.mean_[0]),
        "target_scale": float(target_scaler.scale_[0]),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "cache_molgraphs": cache_molgraphs,
        "persistent_workers": persistent_workers,
        "pin_memory": pin_memory,
        "non_blocking": non_blocking,
        "defer_batch_metrics": defer_batch_metrics,
        "prefetch_factor": prefetch_factor,
        "optimizer": "Adam",
        "chemprop_config": asdict(chemprop_config),
        "svdkl_config": asdict(model.head.config),
        "software": {
            "chemprop": _package_version("chemprop"),
            "gpytorch": _package_version("gpytorch"),
            "torch": torch.__version__,
        },
    }
    if calibration_metadata is not None:
        metadata["gaussian_calibration"] = calibration_metadata
    save_chemprop_svdkl_checkpoint(
        bin_path,
        model=model,
        target_scaler=target_scaler,
        chemprop_config=chemprop_config,
        metadata=metadata,
    )
    (save_dir_path / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if profile:
        timing_profile["training_total_s"] = time.monotonic() - training_started
        timing_profile["total_s"] = time.monotonic() - profile_started
        timing_profile["final_max_rss_bytes"] = _max_rss_bytes()
        timing_profile["best_epoch"] = best_epoch
        timing_profile["best_val_nll"] = best_val_nll
        (save_dir_path / "training_profile.json").write_text(
            json.dumps(timing_profile, indent=2) + "\n",
            encoding="utf-8",
        )
    logger.info("Model saved to %s", bin_path)
    return 0


def _select_device(n_devices: int) -> torch.device:
    if n_devices > 0 and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as temporary:
        json.dump(payload, temporary, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _batch_targets(batch, device: torch.device) -> torch.Tensor:
    for attr in ("Y", "y", "targets"):
        if hasattr(batch, attr):
            target = getattr(batch, attr)
            if target is not None:
                break
    else:
        raise AttributeError("Could not find targets on Chemprop training batch")

    if not torch.is_tensor(target):
        target = torch.as_tensor(target)
    return target.to(device).float().reshape(target.shape[0], -1)[:, 0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Chemprop model using Python API")
    parser.add_argument("data_path", help="Path to training CSV (columns: smiles, docking_score)")
    parser.add_argument("save_dir", help="Directory to save model checkpoint")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--devices",
        type=str,
        default="1",
        help="Positive values enable the single CUDA device used by this trainer",
    )
    parser.add_argument("--mp-hidden-size", type=int, default=300)
    parser.add_argument("--mp-depth", type=int, default=3)
    parser.add_argument("--ffn-hidden-size", type=int, default=300)
    parser.add_argument("--ffn-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument(
        "--batch-norm",
        type=int,
        choices=[0, 1],
        default=0,
        help="Use batch norm in MPNN (0/1)",
    )
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--init-lr", type=float, default=1e-4)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--final-lr", type=float, default=1e-4)
    parser.add_argument("--svdkl-gp-dim", type=int, default=16)
    parser.add_argument("--svdkl-grid-size", type=int, default=128)
    parser.add_argument("--svdkl-grid-lower", type=float, default=-10.0)
    parser.add_argument("--svdkl-grid-upper", type=float, default=10.0)
    parser.add_argument("--svdkl-cholesky-jitter", type=float, default=1e-3)
    parser.add_argument(
        "--svdkl-feature-transform",
        choices=("scale_to_bounds", "tanh"),
        default="scale_to_bounds",
    )
    parser.add_argument("--svdkl-tanh-temperature", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--gradient-clip-val", type=float, default=5.0)
    parser.add_argument("--precision", choices=("32", "32-true"), default="32-true")
    parser.add_argument("--cache-molgraphs", action="store_true")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--non-blocking", action="store_true")
    parser.add_argument("--defer-batch-metrics", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--previous-data-path")
    parser.add_argument("--new-data-repeat", type=int, default=1)
    parser.add_argument("--fit-gaussian-calibrator", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    data_path = Path(args.data_path).resolve()
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        return 1

    return train_model(
        str(data_path),
        args.save_dir,
        args.batch_size,
        args.epochs,
        args.num_workers,
        args.devices,
        args.mp_hidden_size,
        args.mp_depth,
        args.ffn_hidden_size,
        args.ffn_layers,
        args.dropout,
        args.activation,
        bool(args.batch_norm),
        args.warmup_epochs,
        args.init_lr,
        args.max_lr,
        args.final_lr,
        args.svdkl_gp_dim,
        args.svdkl_grid_size,
        args.svdkl_grid_lower,
        args.svdkl_grid_upper,
        args.svdkl_cholesky_jitter,
        args.svdkl_feature_transform,
        args.svdkl_tanh_temperature,
        args.seed,
        args.early_stopping_patience,
        args.early_stopping_min_delta,
        args.validation_fraction,
        args.gradient_clip_val,
        args.precision,
        cache_molgraphs=args.cache_molgraphs,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        non_blocking=args.non_blocking,
        defer_batch_metrics=args.defer_batch_metrics,
        prefetch_factor=args.prefetch_factor,
        profile=args.profile,
        parent_checkpoint=args.parent_checkpoint,
        previous_data_path=args.previous_data_path,
        new_data_repeat=args.new_data_repeat,
        fit_gaussian_calibrator=args.fit_gaussian_calibrator,
    )


if __name__ == "__main__":
    sys.exit(main())
