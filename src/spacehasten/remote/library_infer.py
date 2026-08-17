#!/usr/bin/env python3
"""Remote library-infer worker - property filter + chemprop predict.

Invoked once per library Parquet chunk (produced by
spacehasten.stages.library_build.library_build) on a compute node with the
chemprop conda env. The chunk is streamed in sub-blocks (via pyarrow
``iter_batches``) so peak memory is bounded by ``--block-size`` rows rather
than the full chunk — this lets large (1-2M row) chunks be screened, and
many tasks run in parallel per node, without OOM. Each block is
property-filtered over the precomputed PC-property columns (no RDKit call -
this is the columnar speed win over 50M-1B compounds), then chemprop
D-MPNN inference runs on the property-passing survivors, and a small CSV of
just the winning (reghash, smiles, compound_id, pred_score) rows is written.

Output is CSV (not Parquet) so survivor files are easy to inspect (head/
less/grep) in the shared work directory; the library store itself stays
Parquet. The chunk is read directly from wherever the library lives — no
copy or symlink of library data is made into the work directory.

The chemprop inference calls (featurizer, MoleculeDatapoint.from_smi,
MPNN.load_from_checkpoint, pl.Trainer(...).predict, unscale-transform
warning) intentionally mirror spacehasten.remote.predict so predictions
are produced identically to the existing simsearch/predict pipeline.

This module is invoked as ``python3 <abs path>`` on a compute node inside
the chemprop conda env (see ``Settings.remote_script_path``), which does
not have the ``spacehasten`` package installed. It therefore does not
import anything from ``spacehasten`` — the ``_Bounds`` property-bounds
reader is duplicated here from :mod:`spacehasten.remote.prop_filter`
(kept in sync manually; see that module's docstring for the file format)
rather than imported, mirroring how :mod:`spacehasten.remote.predict`
stays self-contained.

SMARTS note: precomputed columns do not encode SMARTS matches, so SMARTS
filtering requires parsing each property-filter survivor's SMILES with
RDKit.  It is applied *after* the vectorized property filter (only to the
survivors) to keep the RDKit cost proportional to the small surviving
fraction.  When no SMARTS patterns are configured the RDKit path is
skipped entirely and the screen stays RDKit-free.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightning.pytorch as pl
    from chemprop import data, featurizers
    from chemprop import models as chemprop_models
    from chemprop import nn as chemprop_nn
except ImportError as e:  # pragma: no cover - import guarded for remote node only
    print(
        "Error: Failed to import chemprop or lightning. "
        f"Make sure chemprop 2.x is installed: {e}"
    )
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Bounds:
    """Property bounds, duplicated from :class:`spacehasten.remote.prop_filter._Bounds`.

    Duplicated (rather than imported) so this script has no ``spacehasten``
    dependency when run standalone on a compute node; see module docstring.
    """

    mw_min: float
    mw_max: float
    slogp_min: float
    slogp_max: float
    hba_min: int
    hba_max: int
    hbd_min: int
    hbd_max: int
    rotbonds_min: int
    rotbonds_max: int
    tpsa_min: float
    tpsa_max: float

    @classmethod
    def read(cls, path: Path) -> _Bounds:
        with path.open("rt", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh.readlines() if line.strip()]
        if len(lines) < 12:
            raise ValueError(
                f"{path}: expected 12 property bound lines, got {len(lines)}"
            )
        return cls(
            mw_min=float(lines[0]),
            mw_max=float(lines[1]),
            slogp_min=float(lines[2]),
            slogp_max=float(lines[3]),
            hba_min=int(lines[4]),
            hba_max=int(lines[5]),
            hbd_min=int(lines[6]),
            hbd_max=int(lines[7]),
            rotbonds_min=int(lines[8]),
            rotbonds_max=int(lines[9]),
            tpsa_min=float(lines[10]),
            tpsa_max=float(lines[11]),
        )


@dataclass(frozen=True)
class _SmartsBounds:
    """Compiled SMARTS patterns for substructure filtering.

    Duplicated (rather than imported) from
    :class:`spacehasten.remote.prop_filter._SmartsBounds` so this script has
    no ``spacehasten`` dependency when run standalone on a compute node; see
    module docstring (mirrors the ``_Bounds`` duplication above).

    ``include``: molecule must match at least one pattern (empty = no constraint).
    ``exclude``: molecule must not match any pattern (empty = no constraint).
    """

    include: list[object] = field(default_factory=list)
    exclude: list[object] = field(default_factory=list)

    @classmethod
    def read(cls, path: Path) -> _SmartsBounds:
        """Load and compile SMARTS from *path*.

        File format: one ``<mode>:<smarts>`` per line.  Lines starting with
        ``#`` or blank lines are ignored.  Raises ``ValueError`` for
        unrecognised modes or patterns that RDKit cannot parse.
        """
        from rdkit import Chem

        include: list[object] = []
        exclude: list[object] = []
        with path.open("rt", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    raise ValueError(
                        f"{path}:{lineno}: expected 'include:<smarts>' or "
                        f"'exclude:<smarts>', got {line!r}"
                    )
                mode, _, pattern = line.partition(":")
                mode = mode.strip().lower()
                if mode not in ("include", "exclude"):
                    raise ValueError(
                        f"{path}:{lineno}: unknown mode {mode!r} "
                        "(expected 'include' or 'exclude')"
                    )
                mol = Chem.MolFromSmarts(pattern)
                if mol is None:
                    raise ValueError(
                        f"{path}:{lineno}: RDKit could not parse SMARTS {pattern!r}"
                    )
                if mode == "include":
                    include.append(mol)
                else:
                    exclude.append(mol)
        return cls(include=include, exclude=exclude)

    @property
    def active(self) -> bool:
        return bool(self.include or self.exclude)


def apply_smarts_filter(df: pd.DataFrame, smarts: _SmartsBounds) -> pd.DataFrame:
    """Drop rows whose ``smiles`` fails the SMARTS include/exclude gates.

    Applied only to property-filter survivors, so the per-molecule RDKit
    parse cost scales with the (small) surviving fraction.  Rows whose
    SMILES RDKit cannot parse are dropped.  A no-op returning ``df``
    unchanged when ``smarts`` has no active patterns.

    :returns: the filtered DataFrame (index reset).
    """
    if not smarts.active:
        return df
    from rdkit import Chem

    keep: list[bool] = []
    for smi in df["smiles"].astype(str):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            keep.append(False)
            continue
        if smarts.include and not any(
            mol.HasSubstructMatch(q) for q in smarts.include
        ):
            keep.append(False)
            continue
        if any(mol.HasSubstructMatch(q) for q in smarts.exclude):
            keep.append(False)
            continue
        keep.append(True)
    return df.loc[keep].reset_index(drop=True)


OUTPUT_COLUMNS: tuple[str, ...] = ("reghash", "smiles", "compound_id", "pred_score")

_REQUIRED_INPUT_COLUMNS = (
    "compound_id", "smiles", "reghash", "mw", "slogp", "hba", "hbd", "rotbonds", "tpsa",
)


def _write_empty_output(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=list(OUTPUT_COLUMNS)).to_csv(output_path, index=False)


def apply_property_filter(df: pd.DataFrame, bounds: _Bounds) -> pd.DataFrame:
    """Vectorized property-filter mask over the precomputed columns.

    No RDKit call: the six bounds are compared directly against the
    columns produced by library-build (plan D5).
    """
    mask = (
        df["mw"].between(bounds.mw_min, bounds.mw_max)
        & df["slogp"].between(bounds.slogp_min, bounds.slogp_max)
        & df["hba"].between(bounds.hba_min, bounds.hba_max)
        & df["hbd"].between(bounds.hbd_min, bounds.hbd_max)
        & df["rotbonds"].between(bounds.rotbonds_min, bounds.rotbonds_max)
        & df["tpsa"].between(bounds.tpsa_min, bounds.tpsa_max)
    )
    return df.loc[mask].reset_index(drop=True)


def predict_scores(
    smiles: list[str],
    model_dir: str,
    batch_size: int,
    num_workers: int,
    accelerator: str,
    devices: str,
) -> np.ndarray:
    """Run chemprop D-MPNN inference on ``smiles``.

    Mirrors spacehasten.remote.predict.predict's chemprop API calls
    exactly (same featurizer, datapoint construction, checkpoint loading,
    and Trainer configuration) so predictions match the rest of the
    pipeline byte-for-byte.
    """
    model_file = Path(model_dir) / "model_0" / "pytorch_model.bin"
    if not model_file.exists():
        raise FileNotFoundError(f"model file not found: {model_file}")

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    pred_data = [data.MoleculeDatapoint.from_smi(smi=smi) for smi in smiles]
    pred_ds = data.MoleculeDataset(pred_data, featurizer)
    pred_loader = data.build_dataloader(
        pred_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    logger.info("Loading model from %s", model_file)
    model = chemprop_models.MPNN.load_from_checkpoint(
        str(model_file), map_location="cpu", weights_only=False
    )
    model.eval()
    has_unscale_transform = isinstance(
        getattr(model.predictor, "output_transform", None),
        chemprop_nn.UnscaleTransform,
    )
    if not has_unscale_transform:
        logger.warning(
            "Model has no output unscale transform; "
            "predictions may remain in scaled target space"
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
    return pred_array.reshape(-1)


def infer_chunk(
    chunk_path: Path,
    model_dir: Path,
    output_path: Path,
    *,
    bounds: _Bounds,
    smarts: _SmartsBounds | None = None,
    score_cutoff: float | None = None,
    top_n_per_chunk: int | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    accelerator: str = "cpu",
    devices: str = "1",
    block_size: int = 100_000,
) -> int:
    """Filter + predict one library chunk in sub-blocks; write surviving rows as CSV.

    The chunk is streamed via :func:`pyarrow.parquet.ParquetFile.iter_batches`
    in slices of ``block_size`` rows so peak memory (the DataFrame plus the
    chemprop featurized graphs) is bounded by the slice, independent of the
    chunk's total size.  Only the small post-selection survivor set is held
    across blocks and written to ``output_path`` as CSV.

    :returns: number of rows written to ``output_path``.
    """
    import pyarrow.parquet as pq

    pqf = pq.ParquetFile(chunk_path)
    schema_names = set(pqf.schema_arrow.names)
    missing = [c for c in _REQUIRED_INPUT_COLUMNS if c not in schema_names]
    if missing:
        raise ValueError(f"{chunk_path}: missing required columns {missing}")

    collected: list[pd.DataFrame] = []
    total_in = 0
    total_prop_survivors = 0
    for batch in pqf.iter_batches(
        batch_size=max(1, block_size), columns=list(_REQUIRED_INPUT_COLUMNS)
    ):
        block = batch.to_pandas()
        total_in += len(block)

        survivors = apply_property_filter(block, bounds)
        del block
        if not survivors.empty and smarts is not None and smarts.active:
            survivors = apply_smarts_filter(survivors, smarts)
        if survivors.empty:
            continue
        total_prop_survivors += len(survivors)

        pred_scores = predict_scores(
            survivors["smiles"].astype(str).tolist(),
            str(model_dir),
            batch_size,
            num_workers,
            accelerator,
            devices,
        )
        n = min(len(pred_scores), len(survivors))
        survivors = survivors.iloc[:n].copy()
        survivors["pred_score"] = pred_scores[:n]

        # score_cutoff is a per-row predicate, so applying it per block keeps
        # the accumulated survivor set small. top_n_per_chunk is bounded the
        # same way: trimming each block to its own best top_n rows can never
        # drop a true global winner (a chunk-wide top-N row has at most N-1
        # better-scoring rows in the whole chunk, so it is always within its
        # block's top-N), while capping accumulation at n_blocks * top_n
        # instead of "every predicted survivor" (see plan §5.4).
        if score_cutoff is not None:
            survivors = survivors.loc[survivors["pred_score"] <= score_cutoff]
        elif top_n_per_chunk is not None:
            survivors = survivors.nsmallest(top_n_per_chunk, "pred_score")

        out_block = survivors[["reghash", "smiles", "compound_id", "pred_score"]]
        if not out_block.empty:
            collected.append(out_block.reset_index(drop=True))
        del survivors

    logger.info(
        "%s: %d rows read, %d property/SMARTS survivors predicted",
        chunk_path, total_in, total_prop_survivors,
    )
    if not collected:
        _write_empty_output(output_path)
        return 0

    out_df = pd.concat(collected, ignore_index=True)
    if score_cutoff is None and top_n_per_chunk is not None:
        out_df = out_df.nsmallest(top_n_per_chunk, "pred_score")
    out_df = out_df.reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    logger.info("wrote %d rows -> %s", len(out_df), output_path)
    return len(out_df)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Property-filter (vectorized, precomputed columns) then chemprop-predict"
            " one library Parquet chunk."
        )
    )
    parser.add_argument("chunk", help="Input library Parquet chunk")
    parser.add_argument("model_dir", help="Trained chemprop model directory")
    parser.add_argument(
        "output", help="Output CSV path (reghash,smiles,compound_id,pred_score)"
    )
    parser.add_argument("--params", required=True, help="Property bounds file (12 lines)")
    parser.add_argument(
        "--smarts",
        default=None,
        metavar="FILE",
        help=(
            "Optional SMARTS filter file. Each non-blank, non-comment line "
            "must be 'include:<SMARTS>' or 'exclude:<SMARTS>'. Applied to "
            "property-filter survivors: molecules must match at least one "
            "include pattern (if any) and must not match any exclude pattern."
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--score-cutoff", type=float, default=None)
    group.add_argument("--top-n-per-chunk", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default="cpu")
    parser.add_argument("--devices", type=str, default="1")
    parser.add_argument(
        "--block-size",
        type=int,
        default=100_000,
        help=(
            "Number of rows read/predicted per streamed sub-block. Bounds peak "
            "memory independent of the chunk's total row count."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chunk_path = Path(args.chunk)
    if not chunk_path.exists():
        logger.error("chunk not found: %s", chunk_path)
        return 1
    params_path = Path(args.params)
    if not params_path.exists():
        logger.error("params file not found: %s", params_path)
        return 1

    bounds = _Bounds.read(params_path)
    smarts: _SmartsBounds | None = None
    if args.smarts is not None:
        smarts_path = Path(args.smarts)
        if smarts_path.exists():
            smarts = _SmartsBounds.read(smarts_path)
            if smarts.active:
                logger.info(
                    "SMARTS filter: %d include, %d exclude patterns",
                    len(smarts.include), len(smarts.exclude),
                )
        else:
            logger.debug(
                "--smarts file not found (%s), skipping SMARTS filter", smarts_path
            )
    infer_chunk(
        chunk_path,
        Path(args.model_dir),
        Path(args.output),
        bounds=bounds,
        smarts=smarts,
        score_cutoff=args.score_cutoff,
        top_n_per_chunk=args.top_n_per_chunk,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        accelerator=args.accelerator,
        devices=args.devices,
        block_size=args.block_size,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
