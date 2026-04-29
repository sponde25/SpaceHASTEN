#!/usr/bin/env python3
# SpaceHASTEN: Predict using Chemprop Python API
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
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from chemprop import data, featurizers
    from chemprop import models as chemprop_models
    from chemprop import nn as chemprop_nn
    import lightning.pytorch as pl
except ImportError as e:
    print(f"Error: Failed to import chemprop or lightning. Make sure chemprop 2.x is installed: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def predict(data_path: str, model_path: str, output_path: str, batch_size: int,
            num_workers: int, accelerator: str, devices: str) -> int:
    model_file = Path(model_path) / "model_0" / "pytorch_model.bin"
    if not model_file.exists():
        logger.error(f"Model file not found: {model_file}")
        return 1

    df = pd.read_csv(data_path)
    if "smiles" not in df.columns or "smilesid" not in df.columns:
        logger.error("Input CSV must have 'smiles' and 'smilesid' columns")
        return 1

    smiles = df["smiles"].astype(str).tolist()
    smilesids = df["smilesid"].tolist()
    logger.info(f"Loaded {len(smiles)} molecules for prediction")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(smiles) == 0:
        pd.DataFrame(columns=["smilesid", "docking_score"]).to_csv(out_path, index=False)
        logger.info(f"Saved 0 predictions to {out_path}")
        return 0

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    pred_data = [data.MoleculeDatapoint.from_smi(smi=smi) for smi in smiles]
    pred_ds = data.MoleculeDataset(pred_data, featurizer)
    pred_loader = data.build_dataloader(pred_ds, batch_size=batch_size, shuffle=False,
                                        num_workers=num_workers)

    logger.info(f"Loading model from {model_file}")
    model = chemprop_models.MPNN.load_from_checkpoint(str(model_file), weights_only=False)
    model.eval()
    has_unscale_transform = isinstance(
        getattr(model.predictor, "output_transform", None),
        chemprop_nn.UnscaleTransform,
    )

    try:
        n_devices = int(devices)
    except ValueError:
        n_devices = 1

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=n_devices,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    batch_preds = trainer.predict(model, dataloaders=pred_loader)
    pred_array = np.concatenate([p.cpu().numpy() for p in batch_preds], axis=0)
    if pred_array.ndim == 1:
        pred_array = pred_array.reshape(-1, 1)

    if not has_unscale_transform:
        logger.warning(
            "Model has no output unscale transform; "
            "predictions may remain in scaled target space"
        )

    all_preds = pred_array.reshape(-1)

    if len(all_preds) != len(smilesids):
        logger.warning(
            f"Prediction count ({len(all_preds)}) != input count ({len(smilesids)}); "
            "some SMILES may have been skipped"
        )

    output_df = pd.DataFrame({
        "smilesid": smilesids[:len(all_preds)],
        "docking_score": all_preds,
    })

    output_df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(output_df)} predictions to {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict using Chemprop model via Python API")
    parser.add_argument("data_path", help="Path to input CSV (columns: smiles, smilesid)")
    parser.add_argument("model_path", help="Path to trained model directory")
    parser.add_argument("output_path", help="Path for output predictions CSV")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--devices", type=str, default="1",
                        help="Number of devices for Lightning Trainer (integer)")
    args = parser.parse_args()

    for path, name in [(args.data_path, "data"), (args.model_path, "model")]:
        if not Path(path).exists():
            logger.error(f"{name} path not found: {path}")
            return 1

    return predict(args.data_path, args.model_path, args.output_path,
                   args.batch_size, args.num_workers, args.accelerator, args.devices)


if __name__ == "__main__":
    sys.exit(main())
