from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from spacehasten.core.db import (
    AcquisitionBatchRow,
    AcquisitionSelectionRow,
    ClusterAtlasAssignmentRow,
    ClusterAtlasCentroidRow,
    Database,
    acquisition_selection_digest,
    canonical_json,
    sha256_hex,
)
from spacehasten.core.portfolio_acquisition import candidate_pool_digest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "analyze_portfolio_enrichment.py"
)


def selection(batch_id: str, rank: int, identifier: int, cluster: int) -> AcquisitionSelectionRow:
    return AcquisitionSelectionRow(
        batch_id=batch_id,
        selection_rank=rank,
        spacehastenid=identifier,
        clusterid=cluster,
        model_version=0,
        raw_mean=-10.0,
        raw_epistemic_std=0.2,
        calibrated_mean=-10.0,
        calibrated_std=0.3,
        p_hit=0.8,
        expected_improvement=0.4,
        quality=1.2,
        support_before=0.0,
        support_after=0.8,
        marginal_reward=0.1,
        crowding_penalty=0.0,
        final_utility=1.3,
        cluster_count_before=rank - 1,
        cap_reached_after=False,
        contributions_json=canonical_json({"quality": 1.2}),
    )


def database(path: Path) -> Path:
    with Database(path) as db:
        db.create_schema()
        identifiers = [
            db.insert_seed_undocked(f"h{index}", "C" * index, f"m{index}")
            for index in range(1, 5)
        ]
        db.apply_predictions(
            [(identifier, 0, -10.0 + index / 10, 0.2, 0.1, 0.3)
             for index, identifier in enumerate(identifiers)]
        )
        db.append_cluster_atlas_assignments(
            [
                ClusterAtlasAssignmentRow("atlas", identifier, index % 2, 1.0, 1)
                for index, identifier in enumerate(identifiers)
            ]
        )
        db.append_cluster_atlas_centroids(
            [
                ClusterAtlasCentroidRow("atlas", 0, identifiers[0], 1),
                ClusterAtlasCentroidRow("atlas", 1, identifiers[1], 1),
            ]
        )
        pool = db.select_portfolio_candidate_pool("atlas", exclude_selected_attempts=True)
        selected = [
            selection("batch", 1, identifiers[0], 0),
            selection("batch", 2, identifiers[1], 1),
        ]
        policy = canonical_json({"schema_version": 1})
        db.plan_acquisition_batch(
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
                atlas_version=1,
                candidate_count=len(pool.ids),
                candidate_watermark=max(identifiers),
                candidate_digest=candidate_pool_digest(pool),
                requested_count=2,
                selected_count=2,
                selection_digest=acquisition_selection_digest(selected),
                cap_scope=None,
                cap_limit=None,
            ),
            selected,
        )
        scores = {identifiers[0]: (-10.0, "dock"), identifiers[1]: (-8.0, "dock")}
        db.finalize_acquisition_outcomes("batch", scores, hit_threshold=-9.7)
        db.apply_dock_scores([(score, 1, identifier) for identifier, (score, _) in scores.items()])
    return path


def test_portfolio_enrichment_reconstructs_exact_candidates(tmp_path: Path) -> None:
    source = database(tmp_path / "run.dbsh")
    output = tmp_path / "analysis"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
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
    assert receipt["candidate_reconstruction_exact"] is True
    assert receipt["selected"] == 2
    assert receipt["hits"] == 1
    clusters = pd.read_csv(output / "cluster_round_enrichment.csv")
    assert clusters["candidate_count"].sum() == 4
    assert clusters["selected_count"].sum() == 2
    assert (output / "figures" / "portfolio_enrichment.png").stat().st_size > 0
