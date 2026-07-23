#!/usr/bin/env python3
"""Validate the normalized-EI 50k docking study before submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    return parser.parse_args()


def validate(study_dir: Path) -> dict[str, object]:
    study_dir = study_dir.resolve()
    manifest = pd.read_csv(study_dir / "docking_manifest_50k.csv")
    population = pd.read_csv(study_dir / "policy_membership_population.csv.gz")
    selections = pd.read_csv(study_dir / "sequential_policy_selected_ids.csv.gz")
    strata = pd.read_csv(study_dir / "sampling_strata.csv")
    provenance = json.loads((study_dir / "study_provenance.json").read_text())

    if len(manifest) != 50_000:
        raise ValueError(f"manifest has {len(manifest)} rows")
    if manifest["spacehastenid"].nunique() != 50_000:
        raise ValueError("manifest IDs are not unique")
    if manifest["reghash"].nunique() != 50_000:
        raise ValueError("manifest reghashes are not unique")
    if manifest["has_existing_outcome"].any() or manifest["dock_score"].notna().any():
        raise ValueError("manifest contains existing outcomes")
    if not np.allclose(
        manifest["sampling_weight"], 1.0 / manifest["inclusion_probability"]
    ):
        raise ValueError("sampling weights do not invert inclusion probabilities")
    if manifest["study_index"].tolist() != list(range(1, 50_001)):
        raise ValueError("study indices are not contiguous")
    task_sizes = manifest.groupby("task_id").size()
    if task_sizes.to_dict() != {task: 125 for task in range(1, 401)}:
        raise ValueError("tasks are not exactly 1-400 with 125 compounds each")

    disagreement_population = set(
        population.loc[
            (~population["has_existing_outcome"])
            & population["is_policy_disagreement"],
            "spacehastenid",
        ].astype(int)
    )
    disagreement_manifest = set(
        manifest.loc[manifest["is_policy_disagreement"], "spacehastenid"].astype(int)
    )
    if disagreement_manifest != disagreement_population:
        raise ValueError("policy-disagreement census is incomplete")
    if len(disagreement_manifest) != 32_753:
        raise ValueError("unexpected policy-disagreement census size")

    sampled_counts = manifest.groupby("sampling_stratum").size()
    for row in strata.itertuples(index=False):
        if sampled_counts.get(row.sampling_stratum, 0) != row.sample_n:
            raise ValueError(f"stratum sample count mismatch: {row.sampling_stratum}")
        if not np.isclose(row.inclusion_probability, row.sample_n / row.population):
            raise ValueError(f"stratum probability mismatch: {row.sampling_stratum}")

    if (selections.groupby(["round", "policy"]).size() != 100_000).any():
        raise ValueError("sequential policy cells are not all 100,000")
    for policy, group in selections.groupby("policy"):
        round1 = set(group.loc[group["round"] == 1, "spacehastenid"].astype(int))
        round2 = set(group.loc[group["round"] == 2, "spacehastenid"].astype(int))
        if round1 & round2:
            raise ValueError(f"{policy} repeats compounds across rounds")

    for task_id in range(1, 401):
        expected = manifest.loc[
            manifest["task_id"] == task_id, "spacehastenid"
        ].astype(int).tolist()
        smi_path = study_dir / "inputs" / f"chunk_{task_id}.smi"
        inp_path = study_dir / "inputs" / f"chunk_{task_id}.inp"
        glide_path = study_dir / "inputs" / f"glide_chunk_{task_id}.in"
        task_paths = (smi_path, inp_path, glide_path)
        if not all(path.is_file() and path.stat().st_size for path in task_paths):
            raise ValueError(f"task {task_id} input files are missing or empty")
        observed = [
            int(line.rsplit(maxsplit=1)[1])
            for line in smi_path.read_text().splitlines()
        ]
        if observed != expected:
            raise ValueError(f"task {task_id} SMI titles differ from manifest")
        if f"chunk_{task_id}.smi" not in inp_path.read_text():
            raise ValueError(f"task {task_id} phase input references the wrong SMI")
        if f"chunk_{task_id}_1.maegz" not in glide_path.read_text():
            raise ValueError(f"task {task_id} Glide input references the wrong ligand file")

    grid = study_dir / "glide_grid.zip"
    if sha256(grid) != provenance["inputs"]["grid_sha256"]:
        raise ValueError("Glide grid checksum mismatch")
    submit = (study_dir / "submit_docking_50k.sh").read_text()
    if not re.search(r"^#SBATCH --array=1-400$", submit, re.MULTILINE):
        raise ValueError("submission script does not request a 400-task array")
    if f"#SBATCH --chdir={study_dir}" not in submit:
        raise ValueError("submission script has the wrong working directory")

    summary: dict[str, object] = {
        "status": "PASS",
        "manifest_rows": len(manifest),
        "policy_disagreement_census": len(disagreement_manifest),
        "probability_sample_rows": int(
            (manifest["sampling_reason"] == "common_policy_probability_sample").sum()
        ),
        "tasks": len(task_sizes),
        "molecules_per_task": int(task_sizes.iloc[0]),
        "minimum_inclusion_probability": float(manifest["inclusion_probability"].min()),
        "maximum_sampling_weight": float(manifest["sampling_weight"].max()),
        "grid_sha256": sha256(grid),
    }
    (study_dir / "preflight_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    try:
        print(json.dumps(validate(parse_args().study_dir), indent=2))
    except Exception as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
