#!/usr/bin/env python3
"""Remote Chemprop/SVDKL prediction entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightning.pytorch as pl
    import torch
    from chemprop import data, featurizers
    from chemprop import models as chemprop_models
    from chemprop import nn as chemprop_nn
except ImportError as e:  # pragma: no cover - import guarded for remote node only
    print(
        "Error: Failed to import chemprop or lightning. "
        f"Make sure chemprop 2.x is installed: {e}"
    )
    sys.exit(1)

try:  # script path execution on compute nodes does not import the package root
    from spacehasten.remote.svdkl import (
        load_chemprop_svdkl_checkpoint,
        move_batch_to_device,
        predictive_mean_std,
    )
except ImportError:  # pragma: no cover - exercised by file-path remote execution
    from svdkl import (  # type: ignore[no-redef]
        load_chemprop_svdkl_checkpoint,
        move_batch_to_device,
        predictive_mean_std,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def predict(
    data_path: str,
    model_path: str,
    output_path: str,
    batch_size: int,
    num_workers: int,
    accelerator: str,
    devices: str,
) -> int:
    model_file = Path(model_path) / "model_0" / "pytorch_model.bin"
    if not model_file.exists():
        logger.error("Model file not found: %s", model_file)
        return 1

    df = pd.read_csv(data_path)
    if "smiles" not in df.columns or "smilesid" not in df.columns:
        logger.error("Input CSV must have 'smiles' and 'smilesid' columns")
        return 1

    smiles = df["smiles"].astype(str).tolist()
    smilesids = df["smilesid"].tolist()
    logger.info("Loaded %d molecules for prediction", len(smiles))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(smiles) == 0:
        pd.DataFrame(columns=["smilesid", "docking_score", "docking_score_std"]).to_csv(
            out_path,
            index=False,
        )
        logger.info("Saved 0 predictions to %s", out_path)
        return 0

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    pred_data = [data.MoleculeDatapoint.from_smi(smi=smi) for smi in smiles]
    pred_ds = data.MoleculeDataset(pred_data, featurizer)
    pred_loader = data.build_dataloader(
        pred_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    logger.info("Loading model from %s", model_file)
    try:
        model, target_scaler, _metadata = load_chemprop_svdkl_checkpoint(model_file)
    except ValueError:
        return _predict_legacy_chemprop(
            model_file=model_file,
            pred_loader=pred_loader,
            smilesids=smilesids,
            out_path=out_path,
            accelerator=accelerator,
            devices=devices,
        )

    try:
        n_devices = int(devices)
    except ValueError:
        n_devices = 1
    device = _select_device(accelerator, n_devices)
    model.to(device)
    model.eval()
    logger.info("Running SVDKL prediction on %s", device)

    embeddings = []
    with torch.no_grad():
        for batch in pred_loader:
            batch = move_batch_to_device(batch, device)
            embeddings.append(model.encode_batch(batch))
    if not embeddings:
        pd.DataFrame(columns=["smilesid", "docking_score", "docking_score_std"]).to_csv(
            out_path,
            index=False,
        )
        logger.info("Saved 0 predictions to %s", out_path)
        return 0

    all_embeddings = torch.cat(embeddings, dim=0)
    mean, std = predictive_mean_std(model.head, all_embeddings, target_scaler=target_scaler)
    all_preds = mean.detach().cpu().numpy().reshape(-1)
    all_stds = std.detach().cpu().numpy().reshape(-1)

    if len(all_preds) != len(smilesids):
        logger.warning(
            "Prediction count (%d) != input count (%d); "
            "some SMILES may have been skipped",
            len(all_preds),
            len(smilesids),
        )

    output_df = pd.DataFrame(
        {
            "smilesid": smilesids[: len(all_preds)],
            "docking_score": all_preds,
            "docking_score_std": all_stds,
        }
    )

    output_df.to_csv(out_path, index=False)
    logger.info("Saved %d SVDKL predictions to %s", len(output_df), out_path)
    return 0


def _select_device(accelerator: str, n_devices: int) -> torch.device:
    wants_gpu = accelerator.lower() in {"auto", "cuda", "gpu"}
    if wants_gpu and n_devices > 0 and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _predict_legacy_chemprop(
    *,
    model_file: Path,
    pred_loader,
    smilesids: list,
    out_path: Path,
    accelerator: str,
    devices: str,
) -> int:
    logger.info("Checkpoint is not SVDKL; falling back to legacy Chemprop prediction")
    model = chemprop_models.MPNN.load_from_checkpoint(
        str(model_file), map_location="cpu", weights_only=False
    )
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
            "Prediction count (%d) != input count (%d); "
            "some SMILES may have been skipped",
            len(all_preds),
            len(smilesids),
        )

    output_df = pd.DataFrame(
        {
            "smilesid": smilesids[: len(all_preds)],
            "docking_score": all_preds,
        }
    )

    output_df.to_csv(out_path, index=False)
    logger.info("Saved %d predictions to %s", len(output_df), out_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict using Chemprop model via Python API")
    parser.add_argument("data_path", help="Path to input CSV (columns: smiles, smilesid)")
    parser.add_argument("model_path", help="Path to trained model directory")
    parser.add_argument("output_path", help="Path for output predictions CSV")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument(
        "--devices",
        type=str,
        default="1",
        help="Number of devices for Lightning Trainer (integer)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    for path, name in [(args.data_path, "data"), (args.model_path, "model")]:
        if not Path(path).exists():
            logger.error("%s path not found: %s", name, path)
            return 1

    return predict(
        args.data_path,
        args.model_path,
        args.output_path,
        args.batch_size,
        args.num_workers,
        args.accelerator,
        args.devices,
    )


if __name__ == "__main__":
    sys.exit(main())
