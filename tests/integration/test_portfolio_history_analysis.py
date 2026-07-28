from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from spacehasten.core.db import (
    AcquisitionBatchRow,
    AcquisitionSelectionRow,
    Database,
    acquisition_selection_digest,
    canonical_json,
    sha256_hex,
)

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "analyze_portfolio_history.py"
)


def _selection(
    batch_id: str,
    rank: int,
    identifier: int,
    clusterid: int,
    support_before: float,
) -> AcquisitionSelectionRow:
    return AcquisitionSelectionRow(
        batch_id=batch_id,
        selection_rank=rank,
        spacehastenid=identifier,
        clusterid=clusterid,
        model_version=0,
        raw_mean=-10.0,
        raw_epistemic_std=0.2,
        calibrated_mean=-10.0,
        calibrated_std=0.3,
        p_hit=0.8,
        expected_improvement=0.4,
        quality=1.2,
        support_before=support_before,
        support_after=support_before + 0.8,
        marginal_reward=0.1,
        crowding_penalty=0.0,
        final_utility=1.3,
        cluster_count_before=rank - 1,
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


def _plan(
    database: Database,
    iteration: int,
    selections: list[AcquisitionSelectionRow],
) -> None:
    policy = canonical_json({"schema_version": 1})
    database.plan_acquisition_batch(
        AcquisitionBatchRow(
            batch_id=f"batch-{iteration}",
            dock_iteration=iteration,
            strategy="portfolio",
            status="planned",
            policy_schema_version=1,
            policy_json=policy,
            policy_sha256=sha256_hex(policy),
            history_attempt_policy="once_per_campaign",
            model_version=0,
            atlas_id="atlas",
            atlas_version=iteration,
            candidate_count=10,
            candidate_watermark=100,
            candidate_digest=f"candidate-{iteration}",
            requested_count=len(selections),
            selected_count=len(selections),
            selection_digest=acquisition_selection_digest(selections),
            cap_scope="batch",
            cap_limit=100,
        ),
        selections,
    )
    database.finalize_acquisition_outcomes(
        f"batch-{iteration}",
        {selection.spacehastenid: (-10.0, "dock") for selection in selections},
        hit_threshold=-9.7,
    )


def test_portfolio_history_analysis_outputs_productive_coverage(tmp_path: Path) -> None:
    database_path = tmp_path / "portfolio.dbsh"
    with Database(database_path) as database:
        database.create_schema()
        ids = [database.insert_seed_undocked(f"h{i}", "CC", f"m{i}") for i in range(4)]
        _plan(
            database,
            1,
            [
                _selection("batch-1", 1, ids[0], 1, 0.0),
                _selection("batch-1", 2, ids[1], 1, 0.8),
            ],
        )
        _plan(
            database,
            2,
            [
                _selection("batch-2", 1, ids[2], 1, 2.0),
                _selection("batch-2", 2, ids[3], 2, 0.0),
            ],
        )
    output = tmp_path / "analysis"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(database_path),
            "--output-root",
            str(output),
            "--hit-threshold",
            "-9.7",
            "--dpi",
            "50",
        ],
        check=True,
    )

    receipt = json.loads((output / "_SUCCESS.json").read_text())
    assert receipt["selected"] == 4
    assert receipt["hits"] == 4
    assert receipt["rounds"] == [1, 2]
    assert receipt["final_coverage"]["occupied_regions"] == 2
    assert receipt["final_coverage"]["u20"] == 4
    with (output / "production_atlas_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["broad_q2"] == "1.6"
    assert (output / "figures" / "coverage_depth.png").stat().st_size > 0
    assert (output / "figures" / "portfolio_contributions.png").stat().st_size > 0
