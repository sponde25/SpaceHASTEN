"""Immutable acquisition and calibration history in the SQLite wrapper."""

from __future__ import annotations

import math
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from spacehasten.core.db import (
    EXTENSION_SCHEMA_STATEMENTS,
    SCHEMA_STATEMENTS,
    AcquisitionBatchRow,
    AcquisitionSelectionRow,
    Database,
    ModelCalibrationRow,
    acquisition_selection_digest,
    canonical_json,
    sha256_hex,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
LEGACY_BASELINE = FIXTURES / "legacy_baseline.dbsh"


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "history.dbsh")
    database.create_schema()
    return database


def _selection(
    batch_id: str, rank: int, identifier: int, clusterid: int = 7
) -> AcquisitionSelectionRow:
    return AcquisitionSelectionRow(
        batch_id=batch_id,
        selection_rank=rank,
        spacehastenid=identifier,
        clusterid=clusterid,
        model_version=4,
        raw_mean=-7.123456789012345,
        raw_epistemic_std=0.234567890123456,
        calibrated_mean=-7.023456789012345,
        calibrated_std=0.334567890123456,
        p_hit=0.456789012345678,
        expected_improvement=0.123456789012345,
        quality=0.580245801358023,
        support_before=float(rank - 1),
        support_after=float(rank - 1) + 0.456789012345678,
        marginal_reward=0.1,
        crowding_penalty=0.02,
        final_utility=0.660245801358023,
        cluster_count_before=rank - 1,
        cap_reached_after=rank == 2,
        contributions_json=canonical_json({"quality": 0.580245801358023}),
    )


def _selections(batch_id: str) -> list[AcquisitionSelectionRow]:
    return [_selection(batch_id, 1, 101), _selection(batch_id, 2, 102, clusterid=8)]


def _batch(
    selections: list[AcquisitionSelectionRow],
    iteration: int = 1,
    *,
    batch_id: str = "batch-1",
    policy: dict[str, object] | None = None,
) -> AcquisitionBatchRow:
    policy_json = canonical_json(policy or {"schema_version": 1, "strategy": "portfolio"})
    return AcquisitionBatchRow(
        batch_id=batch_id,
        dock_iteration=iteration,
        strategy="portfolio",
        status="planned",
        policy_schema_version=1,
        policy_json=policy_json,
        policy_sha256=sha256_hex(policy_json),
        history_attempt_policy="once_per_campaign",
        model_version=4,
        atlas_id="atlas-a",
        atlas_version=2,
        candidate_count=10,
        candidate_watermark=100,
        candidate_digest="candidate-digest",
        requested_count=len(selections),
        selected_count=len(selections),
        selection_digest=acquisition_selection_digest(selections),
        cap_scope="batch",
        cap_limit=2,
    )


def _plan(
    db: Database,
    *,
    batch_id: str = "batch-1",
    iteration: int = 1,
    selections: list[AcquisitionSelectionRow] | None = None,
) -> tuple[AcquisitionBatchRow, list[AcquisitionSelectionRow]]:
    rows = _selections(batch_id) if selections is None else selections
    batch = _batch(rows, iteration, batch_id=batch_id)
    return db.plan_acquisition_batch(batch, rows), rows


def _calibration() -> ModelCalibrationRow:
    return ModelCalibrationRow(
        4,
        "affine_std_floor",
        "epistemic",
        0.1,
        1.2,
        0.03,
        "cross_validation",
        "validation",
        123,
        "split-hash",
        "artifact.json",
        "artifact-hash",
        canonical_json({"folds": 5}),
    )


def test_extension_schema_migrates_legacy_database_without_changing_legacy_data(
    tmp_path: Path,
) -> None:
    target = tmp_path / "legacy.dbsh"
    shutil.copy(LEGACY_BASELINE, target)
    with Database(target) as database:
        before = database.connection.execute("SELECT COUNT(*) FROM data").fetchone()
        database.ensure_extension_schema()
        tables = {
            row[0]
            for row in database.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert before == (5,)
        assert database.connection.execute("SELECT COUNT(*) FROM data").fetchone() == before
        assert database.load_model_blob(1) == b"dummytar"
        assert {"model_calibrations", "acquisition_batches", "acquisition_selections"} <= tables


def test_acquisition_schema_is_extension_only() -> None:
    legacy_schema = "\n".join(SCHEMA_STATEMENTS)
    extension_schema = "\n".join(EXTENSION_SCHEMA_STATEMENTS)
    assert "acquisition_batches" not in legacy_schema
    assert "model_calibrations" not in legacy_schema
    assert "raw_epistemic_std REAL NOT NULL" in extension_schema
    assert "CREATE TABLE IF NOT EXISTS acquisition_batches" in extension_schema


def test_calibration_registration_is_validated_immutable_and_idempotent(db: Database) -> None:
    calibration = _calibration()
    registered = db.register_model_calibration(calibration)
    assert registered.created_at is not None
    assert db.register_model_calibration(calibration) == registered
    assert db.load_model_calibration(4) == registered
    for invalid in (
        replace(calibration, mean_shift=math.nan),
        replace(calibration, std_scale=0.0),
        replace(calibration, std_floor=-0.1),
        replace(calibration, fit_row_count=-1),
        replace(calibration, metadata_json='{"folds":5 }'),
    ):
        with pytest.raises(ValueError):
            db.register_model_calibration(invalid)
    with pytest.raises(ValueError, match="different calibration"):
        db.register_model_calibration(replace(calibration, std_floor=0.04))
    initial = replace(calibration, model_version=0)
    assert db.register_model_calibration(initial).model_version == 0


def test_selection_digest_is_deterministic_and_covers_diagnostics(db: Database) -> None:
    selections = _selections("batch-1")
    digest = acquisition_selection_digest(selections)
    assert digest == acquisition_selection_digest(list(reversed(selections)))
    assert digest != acquisition_selection_digest(
        [replace(selections[0], quality=0.7), selections[1]]
    )
    fake = replace(_batch(selections), selection_digest="not-the-real-digest")
    with pytest.raises(ValueError, match="selection_digest"):
        db.plan_acquisition_batch(fake, selections)
    assert db.get_acquisition_batch(fake.batch_id) is None
    assert db.connection.execute("SELECT COUNT(*) FROM acquisition_selections").fetchone() == (0,)


def test_plan_is_atomic_one_based_and_roundtrips_diagnostics(db: Database) -> None:
    batch, selections = _plan(db)
    loaded = db.get_acquisition_batch_by_dock_iteration(1)
    assert loaded is not None and loaded.batch_id == batch.batch_id
    assert db.load_acquisition_selections(batch.batch_id) == selections
    assert [row.selection_rank for row in selections] == [1, 2]
    assert [outcome.status for outcome in db.load_acquisition_outcomes(batch.batch_id)] == [
        "pending",
        "pending",
    ]


def test_invalid_selection_and_batch_validation_leave_no_partial_rows(db: Database) -> None:
    selections = _selections("batch-1")
    bad_rank = [replace(selections[0], selection_rank=0), selections[1]]
    bad_batch = _batch(bad_rank)
    with pytest.raises(ValueError, match="start at one"):
        db.plan_acquisition_batch(bad_batch, bad_rank)
    bad_numeric = [replace(selections[0], p_hit=1.1), selections[1]]
    with pytest.raises(ValueError, match="p_hit"):
        db.plan_acquisition_batch(_batch(bad_numeric), bad_numeric)
    bad_json = [replace(selections[0], contributions_json='{"quality": 0.5}'), selections[1]]
    with pytest.raises(ValueError, match="canonical"):
        db.plan_acquisition_batch(_batch(bad_json), bad_json)
    assert db.connection.execute("SELECT COUNT(*) FROM acquisition_batches").fetchone() == (0,)
    assert db.connection.execute("SELECT COUNT(*) FROM acquisition_selections").fetchone() == (0,)


def test_initial_model_and_atlas_versions_are_valid(db: Database) -> None:
    selections = [replace(_selection("batch-v0", 1, 101), model_version=0)]
    batch = replace(
        _batch(selections, batch_id="batch-v0"),
        model_version=0,
        atlas_version=0,
    )
    assert db.plan_acquisition_batch(batch, selections).model_version == 0


def test_plan_reuse_requires_the_complete_immutable_plan(db: Database) -> None:
    batch, selections = _plan(db)
    assert db.plan_acquisition_batch(batch, selections) == batch
    changed_policy = replace(
        batch,
        policy_json=canonical_json({"schema_version": 1, "strategy": "different"}),
        policy_sha256=sha256_hex(canonical_json({"schema_version": 1, "strategy": "different"})),
    )
    for changed in (changed_policy, replace(batch, candidate_digest="other")):
        with pytest.raises(ValueError, match="different acquisition plan"):
            db.plan_acquisition_batch(changed, selections)


def test_outcome_finalization_retries_unresolved_and_preserves_scores(db: Database) -> None:
    batch, _ = _plan(db)
    db.finalize_acquisition_outcomes(batch.batch_id, {101: (-8.0, "dock")}, hit_threshold=-8.0)
    assert db.get_acquisition_batch(batch.batch_id).status == "partial"  # type: ignore[union-attr]
    summaries = db.finalize_acquisition_outcomes(
        batch.batch_id,
        {101: (-8.0, "dock"), 102: (-8.000000000000001, "dock")},
        hit_threshold=-8.0,
    )
    assert all(summary.unresolved_count == 0 for summary in summaries)
    assert db.get_acquisition_batch(batch.batch_id).status == "completed"  # type: ignore[union-attr]
    assert (
        db.finalize_acquisition_outcomes(
            batch.batch_id,
            {101: (-8.0, "dock"), 102: (-8.000000000000001, "dock")},
            hit_threshold=-8.0,
        )
        == summaries
    )
    with pytest.raises(ValueError, match="not planned"):
        db.finalize_acquisition_outcomes(batch.batch_id, {999: (-9.0, "dock")}, hit_threshold=-8.0)
    with pytest.raises(ValueError, match="finite"):
        db.finalize_acquisition_outcomes(
            batch.batch_id, {101: (math.nan, "dock")}, hit_threshold=-8.0
        )
    with pytest.raises(ValueError, match="conflicting immutable outcome"):
        db.finalize_acquisition_outcomes(batch.batch_id, {101: (-7.0, "dock")}, hit_threshold=-8.0)


def test_submission_and_failure_retries_are_idempotent(db: Database) -> None:
    batch, _ = _plan(db)
    db.update_acquisition_submitted(batch.batch_id, "job-1")
    db.update_acquisition_submitted(batch.batch_id, "job-1")
    with pytest.raises(ValueError, match="different job ID"):
        db.update_acquisition_submitted(batch.batch_id, "job-2")
    db.mark_acquisition_batch_failed(batch.batch_id)
    db.mark_acquisition_batch_failed(batch.batch_id)
    db.update_acquisition_submitted(batch.batch_id, "job-2")
    retried = db.get_acquisition_batch(batch.batch_id)
    assert retried is not None
    assert retried.scheduler_job_id == "job-2"
    assert retried.completed_at is None


def test_region_summaries_prior_counts_and_selected_history_exclude_seeds(db: Database) -> None:
    seed = db.insert_seed_docked("seed", "CC", "seed", -99.0)
    first, _ = _plan(db)
    db.finalize_acquisition_outcomes(
        first.batch_id, {101: (-8.0, "dock"), 102: (-6.0, "dock")}, hit_threshold=-8.0
    )
    second_rows = [_selection("batch-2", 1, 201)]
    second, _ = _plan(db, batch_id="batch-2", iteration=2, selections=second_rows)
    summaries = db.finalize_acquisition_outcomes(
        second.batch_id, {201: (-8.0, "dock")}, hit_threshold=-8.0
    )
    assert summaries[0].prior_observed_hits == 1
    assert summaries[0].observed_hits == 1
    assert db.load_acquisition_region_summaries(second.batch_id) == summaries
    assert db.prior_observed_hit_counts("atlas-a", before_dock_iteration=3, hit_threshold=-8.0) == {
        7: 2
    }
    assert seed not in db.selected_attempt_ids()
    assert db.selected_attempt_ids(before_dock_iteration=2) == {101, 102}
    assert db.selected_attempt_ids() == {101, 102, 201}
