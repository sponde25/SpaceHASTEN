"""Chemical-family and sampled fingerprint-diversity metrics."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from .models import AnalysisConfig

ACYCLIC = "[ACYCLIC]"
_POPCOUNT_LOOKUP = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def family_labels(mol: Chem.Mol) -> tuple[str, str]:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)  # type: ignore[no-untyped-call]
    if scaffold.GetNumAtoms() == 0:
        return ACYCLIC, ACYCLIC
    typed = Chem.MolToSmiles(scaffold, isomericSmiles=False)
    generic_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold)  # type: ignore[no-untyped-call]
    return typed, Chem.MolToSmiles(generic_mol, isomericSmiles=False)


def distribution(counts: Counter[str]) -> dict[str, float | int | None]:
    total = sum(counts.values())
    if not total:
        return {
            "richness": 0,
            "largest_fraction": None,
            "hhi": None,
            "shannon_entropy": None,
            "effective_q1": None,
            "inverse_simpson_q2": None,
        }
    proportions = [count / total for count in counts.values()]
    entropy = -sum(value * math.log(value) for value in proportions)
    hhi = sum(value * value for value in proportions)
    return {
        "richness": len(counts),
        "largest_fraction": max(proportions),
        "hhi": hhi,
        "shannon_entropy": entropy,
        "effective_q1": math.exp(entropy),
        "inverse_simpson_q2": 1 / hhi,
    }


def diversity(
    molecules: Sequence[Chem.Mol], config: AnalysisConfig
) -> dict[str, float | int | None]:
    if len(molecules) < 2:
        return {
            "internal_diversity": None,
            "internal_diversity_mc_se": None,
            "pair_samples_used": 0,
        }
    fingerprints = _packed_fingerprints(molecules)
    count = min(config.pair_samples, len(molecules) * (len(molecules) - 1) // 2)
    rng = random.Random(config.random_seed)
    values: list[float] = []
    for start in range(0, count, 4096):
        sample_count = min(4096, count - start)
        left, right = _sample_pairs(len(molecules), sample_count, rng)
        values.extend(_pair_distances(fingerprints[left], fingerprints[right]).tolist())
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return {
        "internal_diversity": mean,
        "internal_diversity_mc_se": se,
        "pair_samples_used": len(values),
    }


def _packed_fingerprints(molecules: Sequence[Chem.Mol]) -> np.ndarray[Any, Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    bits = np.empty((len(molecules), 1024), dtype=np.uint8)
    for index, molecule in enumerate(molecules):
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), bits[index])
    return np.packbits(bits, axis=1, bitorder="little").view(np.uint64)


def _sample_pairs(
    size: int, count: int, rng: random.Random
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    left = np.empty(count, dtype=np.intp)
    right = np.empty(count, dtype=np.intp)
    for index in range(count):
        first = rng.randrange(size)
        second = rng.randrange(size - 1)
        left[index] = first
        right[index] = second if second < first else second + 1
    return left, right


def _pair_distances(
    left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    xor = np.bitwise_xor(left, right)
    intersection = np.bitwise_and(left, right)
    xor_bits = _POPCOUNT_LOOKUP[xor.view(np.uint8)].sum(axis=1)
    intersection_bits = _POPCOUNT_LOOKUP[intersection.view(np.uint8)].sum(axis=1)
    union = xor_bits + intersection_bits
    return np.divide(xor_bits, union, out=np.zeros_like(xor_bits, dtype=float), where=union != 0)
