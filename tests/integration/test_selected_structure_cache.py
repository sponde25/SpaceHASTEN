from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from spacehasten.core.db import (
    AcquisitionBatchRow,
    AcquisitionSelectionRow,
    Database,
    acquisition_selection_digest,
    canonical_json,
    sha256_hex,
)

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "selected_structure_cache.py"
)


def _database(path: Path) -> Path:
    with Database(path) as database:
        database.create_schema()
        identifiers = [
            database.insert_seed_undocked(f"hash-{index}", smiles, f"mol-{index}")
            for index, smiles in enumerate(("CCO", "c1ccccc1", "CCN", "CCOC"), 1)
        ]
        selections = [
            AcquisitionSelectionRow(
                batch_id="batch",
                selection_rank=rank,
                spacehastenid=identifier,
                clusterid=rank % 2,
                model_version=0,
                raw_mean=-9.0,
                raw_epistemic_std=0.2,
                calibrated_mean=-9.0,
                calibrated_std=0.3,
                p_hit=0.8,
                expected_improvement=0.4,
                quality=1.2,
                support_before=0.0,
                support_after=0.8,
                marginal_reward=0.1,
                crowding_penalty=0.0,
                final_utility=1.3,
                cluster_count_before=0,
                cap_reached_after=False,
                contributions_json=canonical_json(
                    {
                        "crowding_penalty": 0.0,
                        "final_utility": 1.3,
                        "marginal_reward": 0.1,
                        "quality": 1.2,
                    }
                ),
            )
            for rank, identifier in enumerate(identifiers, 1)
        ]
        policy = canonical_json({"schema_version": 1})
        database.plan_acquisition_batch(
            AcquisitionBatchRow(
                batch_id="batch",
                dock_iteration=1,
                strategy="portfolio",
                status="planned",
                policy_schema_version=1,
                policy_json=policy,
                policy_sha256=sha256_hex(policy),
                history_attempt_policy="once_per_campaign",
                model_version=0,
                atlas_id="atlas",
                atlas_version=0,
                candidate_count=4,
                candidate_watermark=max(identifiers),
                candidate_digest="candidate-digest",
                requested_count=4,
                selected_count=4,
                selection_digest=acquisition_selection_digest(selections),
                cap_scope=None,
                cap_limit=None,
            ),
            selections,
        )
        scores = {
            identifier: (-9.7 - rank / 10, "dock") for rank, identifier in enumerate(identifiers)
        }
        database.finalize_acquisition_outcomes("batch", scores, hit_threshold=-9.7)
        database.apply_dock_scores(
            [(score, 1, identifier) for identifier, (score, _source) in scores.items()]
        )
    return path


def _run(*arguments: str) -> None:
    subprocess.run([sys.executable, str(SCRIPT), *arguments], check=True)


def test_selected_structure_cache_end_to_end(tmp_path: Path) -> None:
    database = _database(tmp_path / "run.dbsh")
    output = tmp_path / "cache"
    _run(
        "prepare",
        str(database),
        "--output-root",
        str(output),
        "--hit-threshold",
        "-9.7",
        "--task-count",
        "2",
    )
    _run("worker", "--output-root", str(output), "--task-index", "1")
    _run("worker", "--output-root", str(output), "--task-index", "2")
    _run("combine", "--output-root", str(output))

    assert (output / "logs").is_dir()
    receipt = json.loads((output / "_SUCCESS.json").read_text())
    assert receipt["status"] == "complete"
    assert receipt["rows"] == 4
    assert receipt["task_count"] == 2
    with np.load(output / "fingerprints.npz") as data:
        assert data["spacehastenid"].shape == (4,)
        assert data["words"].shape == (4, 16)
        assert data["words"].dtype == np.uint64
        assert data["popcounts"].shape == (4,)
