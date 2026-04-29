#!/usr/bin/env python3
# SpaceHASTEN: Train models using Chemprop Python API
#
# Copyright (c) 2024-2026 Orion Corporation
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, 
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, 
# this list of conditions and the following disclaimer in the documentation 
# and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software 
# without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
# POSSIBILITY OF SUCH DAMAGE.

import sys
import shutil
import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    from chemprop import data, featurizers
    from chemprop import models as chemprop_models
    from chemprop import nn as chemprop_nn
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
except ImportError as e:
    print(f"Error: Failed to import chemprop or lightning. Make sure chemprop 2.x is installed: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_path: str, save_dir: str, batch_size: int, epochs: int,
                num_workers: int, devices: str, mp_hidden_size: int, mp_depth: int,
                ffn_hidden_size: int, ffn_layers: int, dropout: float,
                activation: str, batch_norm: bool, warmup_epochs: int,
                init_lr: float, max_lr: float, final_lr: float) -> int:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if "smiles" not in df.columns or "docking_score" not in df.columns:
        logger.error("Training CSV must have 'smiles' and 'docking_score' columns")
        return 1

    smiles = df["smiles"].astype(str).to_numpy()
    targets = df["docking_score"].astype(float).to_numpy().reshape(-1, 1)

    if len(smiles) < 2:
        logger.error("Need at least 2 training rows to build train/validation splits")
        return 1

    logger.info(f"Loaded {len(smiles)} molecules for training")

    # 90/10 train/val split (last 10% as val to keep deterministic ordering)
    n_val = max(1, int(0.1 * len(smiles)))
    train_smiles, train_targets = smiles[:-n_val], targets[:-n_val]
    val_smiles, val_targets = smiles[-n_val:], targets[-n_val:]
    if len(train_smiles) == 0:
        logger.error("Training split is empty after creating validation split")
        return 1
    logger.info(f"Split: {len(train_smiles)} train / {len(val_smiles)} val")

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_data = [data.MoleculeDatapoint.from_smi(smi=smi, y=y) for smi, y in zip(train_smiles, train_targets)]
    val_data = [data.MoleculeDatapoint.from_smi(smi=smi, y=y) for smi, y in zip(val_smiles, val_targets)]

    train_ds = data.MoleculeDataset(train_data, featurizer)
    target_scaler = train_ds.normalize_targets()

    val_ds = data.MoleculeDataset(val_data, featurizer)
    val_ds.normalize_targets(target_scaler)

    train_loader = data.build_dataloader(train_ds, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers)
    val_loader = data.build_dataloader(val_ds, batch_size=batch_size, shuffle=False,
                                        num_workers=num_workers)

    d_h = mp_hidden_size
    mp = chemprop_nn.BondMessagePassing(d_h=d_h, depth=mp_depth, dropout=dropout, activation=activation)
    agg = chemprop_nn.MeanAggregation()
    output_transform = chemprop_nn.UnscaleTransform.from_standard_scaler(target_scaler)
    ffn = chemprop_nn.RegressionFFN(
        input_dim=d_h,
        hidden_dim=ffn_hidden_size,
        n_layers=ffn_layers,
        dropout=dropout,
        activation=activation,
        output_transform=output_transform,
    )
    model = chemprop_models.MPNN(
        message_passing=mp, agg=agg, predictor=ffn,
        batch_norm=batch_norm,
        warmup_epochs=warmup_epochs,
        init_lr=init_lr,
        max_lr=max_lr,
        final_lr=final_lr,
    )

    try:
        n_devices = int(devices)
    except ValueError:
        n_devices = 1

    checkpoint_dir = save_dir / "model_0"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="pytorch_model",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[ckpt_cb],
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
    bin_path = checkpoint_dir / "pytorch_model.bin"
    shutil.copy(best_ckpt, str(bin_path))
    logger.info(f"Model saved to {bin_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Chemprop model using Python API")
    parser.add_argument("data_path", help="Path to training CSV (columns: smiles, docking_score)")
    parser.add_argument("save_dir", help="Directory to save model checkpoint")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--devices", type=str, default="1",
                        help="Number of devices for Lightning Trainer (integer)")
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
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
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
    )


if __name__ == "__main__":
    sys.exit(main())
