"""Compact portfolio candidate loading and history eligibility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spacehasten.core.db import (
    AcquisitionBatchRow,
    AcquisitionSelectionRow,
    ClusterAtlasAssignmentRow,
    Database,
    acquisition_selection_digest,
    canonical_json,
    sha256_hex,
)


def _seed_portfolio_candidates(db: Database, count: int = 3) -> list[int]:
    db.create_schema()
    identifiers = []
    predictions = []
    assignments = []
    for index in range(count):
        identifier = db.insert_seed_undocked(
            f"hash-{index}",
            "C" * (index + 1),
            f"candidate-{index}",
        )
        identifiers.append(identifier)
        predictions.append((identifier, 0, -10.0 + index / 10, 0.2, 0.1, 0.3))
        assignments.append(ClusterAtlasAssignmentRow("atlas", identifier, index % 2, 1.0, 0))
    db.apply_predictions(predictions)
    db.append_cluster_atlas_assignments(assignments)
    db.commit()
    return identifiers


def _selection(batch_id: str, identifier: int) -> AcquisitionSelectionRow:
    return AcquisitionSelectionRow(
        batch_id=batch_id,
        selection_rank=1,
        spacehastenid=identifier,
        clusterid=0,
        model_version=0,
        raw_mean=-10.0,
        raw_epistemic_std=0.2,
        calibrated_mean=-10.0,
        calibrated_std=0.2,
        p_hit=0.9,
        expected_improvement=0.3,
        quality=1.2,
        support_before=0.0,
        support_after=0.9,
        marginal_reward=0.02,
        crowding_penalty=0.0,
        final_utility=1.22,
        cluster_count_before=0,
        cap_reached_after=False,
        contributions_json=canonical_json({"quality": 1.2}),
    )


def _plan(db: Database, identifier: int) -> None:
    selection = _selection("batch", identifier)
    policy_json = canonical_json({"schema_version": 1})
    db.plan_acquisition_batch(
        AcquisitionBatchRow(
            batch_id="batch",
            dock_iteration=1,
            strategy="portfolio",
            status="planned",
            policy_schema_version=1,
            policy_json=policy_json,
            policy_sha256=sha256_hex(policy_json),
            history_attempt_policy="once_per_campaign",
            model_version=0,
            atlas_id="atlas",
            atlas_version=0,
            candidate_count=3,
            candidate_watermark=3,
            candidate_digest="candidate-digest",
            requested_count=1,
            selected_count=1,
            selection_digest=acquisition_selection_digest([selection]),
            cap_scope=None,
            cap_limit=None,
        ),
        [selection],
    )


def test_portfolio_pool_excludes_attempts_only_when_requested(tmp_path: Path) -> None:
    with Database(tmp_path / "pool.dbsh") as db:
        identifiers = _seed_portfolio_candidates(db)
        all_pool = db.select_portfolio_candidate_pool("atlas", exclude_selected_attempts=False)
        assert all_pool.ids.tolist() == identifiers
        _plan(db, identifiers[0])
        excluded = db.select_portfolio_candidate_pool("atlas", exclude_selected_attempts=True)
        legacy = db.select_portfolio_candidate_pool("atlas", exclude_selected_attempts=False)
        assert excluded.ids.tolist() == identifiers[1:]
        assert legacy.ids.tolist() == identifiers


@pytest.mark.parametrize("missing", ["smiles", "uncertainty", "atlas"])
def test_portfolio_pool_fails_closed_on_incomplete_candidates(
    tmp_path: Path,
    missing: str,
) -> None:
    with Database(tmp_path / f"{missing}.dbsh") as db:
        identifiers = _seed_portfolio_candidates(db)
        identifier = identifiers[0]
        if missing == "smiles":
            db.connection.execute(
                "UPDATE data SET smiles = NULL WHERE spacehastenid = ?", (identifier,)
            )
        elif missing == "uncertainty":
            db.connection.execute(
                "UPDATE predictions SET epistemic_std = NULL WHERE spacehastenid = ?",
                (identifier,),
            )
        else:
            db.connection.execute(
                "DELETE FROM cluster_atlas_assignments WHERE atlas_id = ? AND spacehastenid = ?",
                ("atlas", identifier),
            )
        with pytest.raises(ValueError, match="missing SMILES"):
            db.select_portfolio_candidate_pool("atlas", exclude_selected_attempts=False)


def test_select_smiles_chunks_large_ordered_request(tmp_path: Path) -> None:
    with Database(tmp_path / "smiles.dbsh") as db:
        identifiers = _seed_portfolio_candidates(db, count=1_005)
        requested = list(reversed(identifiers))
        rows = db.select_smiles_by_ids(requested)
        assert [identifier for _smiles, identifier in rows] == requested
        assert rows[0][0] == "C" * 1_005
        with pytest.raises(ValueError, match="unique"):
            db.select_smiles_by_ids([identifiers[0], identifiers[0]])


def test_candidate_pool_arrays_are_compact_float64_and_int64(tmp_path: Path) -> None:
    with Database(tmp_path / "dtypes.dbsh") as db:
        _seed_portfolio_candidates(db)
        pool = db.select_portfolio_candidate_pool("atlas", exclude_selected_attempts=False)
        assert pool.ids.dtype == np.int64
        assert pool.cluster_ids.dtype == np.int64
        assert pool.model_versions.dtype == np.int64
        assert pool.raw_means.dtype == np.float64
        assert pool.raw_epistemic_stds.dtype == np.float64
