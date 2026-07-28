"""Shared constants and binary-fingerprint handling for landmark UMAP CLIs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import numpy.typing as npt

EXPECTED_FP_TYPE = "Morgan"
EXPECTED_FP_PARAMS: dict[str, int] = {"radius": 2, "fpSize": 1024}
BIT_REVERSE = np.packbits(
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)[:, ::-1],
    axis=1,
).ravel()


def load_centroid_ids(path: Path) -> list[int]:
    with path.open("rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["spacehastenid", "clusterid"]:
            raise ValueError(f"unexpected clustering columns: {reader.fieldnames}")
        identifiers = sorted({int(row["clusterid"]) for row in reader})
    if not identifiers:
        raise ValueError(f"no centroids found in {path}")
    return identifiers


def unpack_fingerprints(
    words: npt.NDArray[np.uint64],
) -> npt.NDArray[np.uint8]:
    """Unpack little-endian FPSim2 binary words without changing UMAP input."""
    packed = np.ascontiguousarray(words, dtype=np.uint64)
    if packed.ndim != 2:
        raise ValueError("fingerprint words must be a two-dimensional array")
    return np.unpackbits(packed.view(np.uint8).reshape(len(packed), -1), axis=1, bitorder="little")


def rdkit_words_to_fpsim2_words(
    words: npt.NDArray[np.uint64],
) -> npt.NDArray[np.uint64]:
    """Reverse each 64-bit block from RDKit binary-text order to FPSim2 order."""
    packed = np.ascontiguousarray(words, dtype=np.uint64)
    if packed.ndim != 2:
        raise ValueError("fingerprint words must be a two-dimensional array")
    bytes_view = packed.view(np.uint8).reshape(*packed.shape, 8)
    reversed_bytes = BIT_REVERSE[bytes_view][..., ::-1]
    return np.ascontiguousarray(reversed_bytes).view(np.uint64).reshape(packed.shape)
