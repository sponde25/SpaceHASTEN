#!/usr/bin/env python3
"""Remote Chemprop + SVDKL training entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

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
        fit_target_scaler,
        move_batch_to_device,
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
        fit_target_scaler,
        move_batch_to_device,
        save_chemprop_svdkl_checkpoint,
        scale_targets,
    )

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
) -> int:
    _seed_everything(seed)
    torch.set_float32_matmul_precision("medium")

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
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

    logger.info("Loaded %d molecules for training", len(smiles))

    train_indices, val_indices = deterministic_split_indices(
        len(smiles), validation_fraction=validation_fraction, seed=seed
    )
    train_smiles, raw_train_targets = smiles[train_indices], targets[train_indices]
    val_smiles, raw_val_targets = smiles[val_indices], targets[val_indices]
    target_scaler = fit_target_scaler(raw_train_targets.reshape(-1))
    train_targets = scale_targets(raw_train_targets.reshape(-1), target_scaler).reshape(-1, 1)
    val_targets = scale_targets(raw_val_targets.reshape(-1), target_scaler).reshape(-1, 1)
    logger.info("Split: %d train / %d val", len(train_smiles), len(val_smiles))

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_data = [
        data.MoleculeDatapoint.from_smi(smi=smi, y=np.asarray(y, dtype=float))
        for smi, y in zip(train_smiles, train_targets, strict=False)
    ]
    val_data = [
        data.MoleculeDatapoint.from_smi(smi=smi, y=np.asarray(y, dtype=float))
        for smi, y in zip(val_smiles, val_targets, strict=False)
    ]

    train_ds = data.MoleculeDataset(train_data, featurizer)
    val_ds = data.MoleculeDataset(val_data, featurizer)

    drop_last_train = batch_norm and len(train_ds) % batch_size == 1
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
        drop_last=drop_last_train,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        drop_last=False,
    )

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

    try:
        n_devices = int(devices)
    except ValueError:
        n_devices = 1

    device = _select_device(n_devices)
    model.to(device)
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

    training_started = time.monotonic()
    best_val_nll = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    epochs_completed = 0
    stop_reason = "max_epochs"
    for epoch in range(epochs):
        epoch_started = time.monotonic()
        model.train()
        model.likelihood.train()
        train_losses: list[float] = []
        gradient_norms: list[float] = []
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            target = _batch_targets(batch, device)
            optimizer.zero_grad()
            loss = -mll(model(batch), target)
            if not torch.isfinite(loss):
                logger.error("Non-finite training loss at epoch %d", epoch + 1)
                return 1
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.detach().cpu()))
            gradient_norms.append(float(gradient_norm.detach().cpu()))

        val_losses: list[float] = []
        model.eval()
        model.likelihood.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                target = _batch_targets(batch, device)
                val_loss = -mll(model(batch), target)
                val_losses.append(float(val_loss.detach().cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_nll = float(np.mean(val_losses)) if val_losses else float("nan")
        if not np.isfinite(val_nll):
            logger.error("Non-finite validation NLL at epoch %d", epoch + 1)
            return 1

        improved = val_nll < best_val_nll - early_stopping_min_delta
        if improved:
            best_val_nll = val_nll
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epochs_completed = epoch + 1
        elapsed = time.monotonic() - training_started
        mean_epoch_seconds = elapsed / epochs_completed
        remaining_epochs = max(0, epochs - epochs_completed)
        eta_seconds = mean_epoch_seconds * remaining_epochs
        current_lr = float(optimizer.param_groups[0]["lr"])
        noise = float(model.likelihood.base_likelihood.noise.detach().mean().cpu())
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
            float(np.mean(gradient_norms)) if gradient_norms else float("nan"),
            noise,
            _format_optional(model.head.last_raw_feature_min),
            _format_optional(model.head.last_raw_feature_max),
            _format_optional(model.head.last_transformed_feature_min),
            _format_optional(model.head.last_transformed_feature_max),
            time.monotonic() - epoch_started,
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
    model.cpu()
    split_hash = hashlib.sha256(val_indices.astype(np.int64).tobytes()).hexdigest()
    metadata = {
        "train_rows": int(len(train_smiles)),
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
        "optimizer": "Adam",
        "chemprop_config": asdict(chemprop_config),
        "svdkl_config": asdict(model.head.config),
        "software": {
            "chemprop": _package_version("chemprop"),
            "gpytorch": _package_version("gpytorch"),
            "torch": torch.__version__,
        },
    }
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
    logger.info("Model saved to %s", bin_path)
    return 0


def _select_device(n_devices: int) -> torch.device:
    if n_devices > 0 and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


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
        help="Number of devices for Lightning Trainer (integer)",
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
    )


if __name__ == "__main__":
    sys.exit(main())
