#!/usr/bin/env python3
"""Remote Chemprop + SVDKL training entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import gpytorch
    import torch
    from chemprop import data, featurizers
<<<<<<< HEAD
    from chemprop import models as chemprop_models
    from chemprop import nn as chemprop_nn
    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.callbacks import ModelCheckpoint
=======
>>>>>>> 32a6621 (Wire SVDKL into SpaceHASTEN workflow)
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
svdkl_gp_dim: int = 2,
    svdkl_grid_size: int = 128,
    svdkl_grid_lower: float = -10.0,
    svdkl_grid_upper: float = 10.0,
    seed: int = 0,
) -> int:
    torch.manual_seed(seed)

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

    logger.info("Loaded %d molecules for training", len(smiles))

    # 90/10 train/val split (last 10% as val to keep deterministic ordering)
    n_val = max(1, int(0.1 * len(smiles)))
    target_scaler = fit_target_scaler(targets.reshape(-1))
    scaled_targets = scale_targets(targets.reshape(-1), target_scaler).reshape(-1, 1)
    train_smiles, train_targets = smiles[:-n_val], scaled_targets[:-n_val]
    val_smiles, val_targets = smiles[-n_val:], scaled_targets[-n_val:]
    if len(train_smiles) == 0:
        logger.error("Training split is empty after creating validation split")
        return 1
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

    train_loader = data.build_dataloader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = data.build_dataloader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
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
            input_dim=mp_hidden_size,
            gp_dim=svdkl_gp_dim,
            grid_size=svdkl_grid_size,
            grid_lower=svdkl_grid_lower,
            grid_upper=svdkl_grid_upper,
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
        num_data=len(train_smiles),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)

    for epoch in range(epochs):
        model.train()
        model.likelihood.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            target = _batch_targets(batch, device)
            optimizer.zero_grad()
            loss = -mll(model(batch), target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_losses: list[float] = []
        model.eval()
        model.likelihood.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                target = _batch_targets(batch, device)
                val_loss = -mll(model(batch), target)
                val_losses.append(float(val_loss.detach().cpu()))
        logger.info(
            "Epoch %d/%d train_loss=%.4f val_loss=%.4f",
            epoch + 1,
            epochs,
            float(np.mean(train_losses)) if train_losses else float("nan"),
            float(np.mean(val_losses)) if val_losses else float("nan"),
        )

    checkpoint_dir = save_dir_path / "model_0"
<<<<<<< HEAD
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="pytorch_model",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    callbacks: list = [ckpt_cb]
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
            )
        )

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=True,
        accelerator="auto",
        devices=n_devices,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Copy the best Lightning checkpoint to pytorch_model.bin for downstream loading
    best_ckpt = ckpt_cb.best_model_path
    if not best_ckpt:
        logger.error("No best checkpoint was saved — check for training errors above")
        return 1
=======
>>>>>>> 32a6621 (Wire SVDKL into SpaceHASTEN workflow)
    bin_path = checkpoint_dir / "pytorch_model.bin"
    model.cpu()
    save_chemprop_svdkl_checkpoint(
        bin_path,
        model=model,
        target_scaler=target_scaler,
        chemprop_config=chemprop_config,
        metadata={
            "train_rows": int(len(train_smiles)),
            "val_rows": int(len(val_smiles)),
            "epochs": int(epochs),
            "seed": int(seed),
        },
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
<<<<<<< HEAD
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
=======
    parser.add_argument("--svdkl-gp-dim", type=int, default=2)
    parser.add_argument("--svdkl-grid-size", type=int, default=128)
    parser.add_argument("--svdkl-grid-lower", type=float, default=-10.0)
    parser.add_argument("--svdkl-grid-upper", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
>>>>>>> 32a6621 (Wire SVDKL into SpaceHASTEN workflow)
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
<<<<<<< HEAD
        args.early_stopping_patience,
        args.early_stopping_min_delta,
=======
        args.svdkl_gp_dim,
        args.svdkl_grid_size,
        args.svdkl_grid_lower,
        args.svdkl_grid_upper,
        args.seed,
>>>>>>> 32a6621 (Wire SVDKL into SpaceHASTEN workflow)
    )


if __name__ == "__main__":
    sys.exit(main())
