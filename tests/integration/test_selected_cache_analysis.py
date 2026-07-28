from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "analyze_selected_cache.py"


def test_selected_cache_analysis_end_to_end(tmp_path: Path) -> None:
    identifiers = np.asarray([11, 12, 13, 14], dtype=np.int64)
    manifest = pd.DataFrame(
        {
            "spacehastenid": [11, 12, 13, 14, 11],
            "reghash": ["h11", "h12", "h13", "h14", "h11"],
            "smiles": ["CC", "CCC", "CCN", "CCO", "CC"],
            "round": [1, 1, 2, 2, 3],
            "rank": [1, 2, 1, 2, 1],
            "dock_score": [-10.0, -8.0, -10.5, np.nan, np.nan],
            "is_scored": [True, True, True, False, False],
            "is_hit": [True, False, True, False, False],
            "clusterid": [101, 101, 102, 103, 101],
        }
    )
    structures = pd.DataFrame(
        {
            "spacehastenid": identifiers,
            "reghash": ["h11", "h12", "h13", "h14"],
            "typed_scaffold": ["t1", "t1", "t2", "t3"],
            "generic_framework": ["g1", "g1", "g2", "g3"],
            "MW": [100.0, 110.0, 120.0, 130.0],
            "cLogP": [1.0, 1.1, 1.2, 1.3],
            "TPSA": [20.0, 21.0, 22.0, 23.0],
            "HBD": [1, 1, 0, 0],
            "HBA": [1, 2, 2, 3],
            "rotatable": [0, 1, 1, 2],
            "rings": [0, 1, 1, 2],
            "Fsp3": [1.0, 0.5, 0.4, 0.3],
        }
    )
    manifest_path = tmp_path / "selected_manifest.csv.gz"
    structure_path = tmp_path / "structure_cache.csv.gz"
    fingerprint_path = tmp_path / "fingerprints.npz"
    nearest_path = tmp_path / "nearest.npz"
    families_path = tmp_path / "seed_families.csv.gz"
    enrichment_path = tmp_path / "portfolio_enrichment.csv"
    manifest.to_csv(manifest_path, index=False)
    structures.to_csv(structure_path, index=False)
    words = np.zeros((4, 16), dtype=np.uint64)
    words[:, 0] = [3, 5, 15, 9]
    np.savez_compressed(
        fingerprint_path,
        spacehastenid=identifiers,
        words=words,
        popcounts=np.asarray([2, 2, 4, 2], dtype=np.uint16),
    )
    np.savez_compressed(
        nearest_path,
        spacehastenid=identifiers[::-1],
        nearest_seed_id=np.asarray([4, 3, 2, 1]),
        tanimoto=np.asarray([0.2, 0.4, 0.6, 0.8]),
    )
    pd.DataFrame(
        {
            "family_type": ["typed_murcko", "generic_murcko"],
            "scaffold": ["t1", "g1"],
        }
    ).to_csv(families_path, index=False)
    pd.DataFrame(
        {
            "clusterid": [101, 102, 103],
            "centroid_source": ["seed", "virtual", "virtual"],
        }
    ).to_csv(enrichment_path, index=False)
    output = tmp_path / "analysis"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest_path),
            "--structure-cache",
            str(structure_path),
            "--fingerprints",
            str(fingerprint_path),
            "--nearest-seed",
            str(nearest_path),
            "--seed-families",
            str(families_path),
            "--portfolio-enrichment",
            str(enrichment_path),
            "--output-root",
            str(output),
            "--pair-samples",
            "20",
            "--dpi",
            "50",
        ],
        check=True,
    )

    receipt = json.loads((output / "_SUCCESS.json").read_text())
    assert receipt["selected_attempts"] == 5
    assert receipt["scored"] == 3
    assert receipt["hits"] == 2
    metrics = pd.read_csv(output / "diversity_metrics.csv")
    assert len(metrics) == 12
    final_hits = metrics[
        (metrics["round"] == 2) & (metrics["cohort"] == "cumulative_hit_only")
    ].iloc[0]
    assert final_hits["unique_compounds"] == 2
    assert final_hits["typed_q0"] == 2
    coverage = pd.read_csv(output / "seed_coverage_metrics.csv")
    assert len(coverage) == 6
    assert "seed_centred_atlas_fraction" in coverage
    assert (output / "figures" / "descriptor_shift.png").stat().st_size > 0
