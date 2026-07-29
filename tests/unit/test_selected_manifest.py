from __future__ import annotations

import csv
import gzip
import sqlite3
from pathlib import Path

from spacehasten.analysis.discovery import discover_run
from spacehasten.analysis.selected import selected_manifest, write_selected_manifest
from spacehasten.core.db import (
    AcquisitionBatchRow,
    AcquisitionSelectionRow,
    Database,
    acquisition_selection_digest,
    canonical_json,
    sha256_hex,
)


def _selection(batch_id: str, rank: int, identifier: int) -> AcquisitionSelectionRow:
    return AcquisitionSelectionRow(
        batch_id=batch_id,
        selection_rank=rank,
        spacehastenid=identifier,
        clusterid=rank,
        model_version=0,
        raw_mean=-9.0,
        raw_epistemic_std=0.2,
        calibrated_mean=-9.1,
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


def test_modern_selected_manifest_uses_immutable_history(tmp_path: Path) -> None:
    database_path = tmp_path / "modern.dbsh"
    with Database(database_path) as database:
        database.create_schema()
        first = database.insert_seed_undocked("hash-1", "CC", "one")
        second = database.insert_seed_undocked("hash-2", "CCC", "two")
        selections = [_selection("batch-1", 1, first), _selection("batch-1", 2, second)]
        policy = canonical_json({"schema_version": 1})
        database.plan_acquisition_batch(
            AcquisitionBatchRow(
                batch_id="batch-1",
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
                candidate_count=2,
                candidate_watermark=second,
                candidate_digest="candidate-digest",
                requested_count=2,
                selected_count=2,
                selection_digest=acquisition_selection_digest(selections),
                cap_scope=None,
                cap_limit=None,
            ),
            selections,
        )
        database.finalize_acquisition_outcomes(
            "batch-1", {first: (-9.7, "dock")}, hit_threshold=-9.7
        )
        database.apply_dock_scores([(-9.7, 1, first)])

    context = discover_run(database_path)
    rows = selected_manifest(context, hit_threshold=-9.7, strict_threshold=-11.0)

    assert [row["spacehastenid"] for row in rows] == [first, second]
    assert rows[0]["selection_id"] == "batch-1:1"
    assert rows[0]["is_scored"] is True
    assert rows[0]["is_hit"] is True
    assert rows[0]["is_strict_hit"] is False
    assert rows[1]["outcome_status"] == "unresolved"
    assert rows[1]["dock_score"] is None
    assert rows[1]["is_scored"] is False
    assert rows[0]["quality"] == 1.2

    output = tmp_path / "selected.csv.gz"
    write_selected_manifest(output, rows)
    with gzip.open(output, "rt", encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 2
    assert written[0]["selection_id"] == "batch-1:1"


def test_legacy_selected_manifest_uses_round_specific_outcomes(tmp_path: Path) -> None:
    run = tmp_path / "run"
    shared = run / "run_shared" / "docking"
    shared.mkdir(parents=True)
    database_path = run / "run.dbsh"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE data(spacehastenid INTEGER PRIMARY KEY,reghash TEXT,smiles TEXT,"
            "dock_score REAL,dock_iteration INTEGER)"
        )
        connection.executemany(
            "INSERT INTO data VALUES (?,?,?,?,?)",
            [
                (1, "h1", "CC", -8.0, 1),
                (2, "h2", "CCC", -9.0, 2),
            ],
        )
    for round_id, identifier in ((1, 1), (2, 2)):
        path = shared / f"iter{round_id}" / "acquisition.csv"
        path.parent.mkdir()
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["rank", "spacehastenid", "method", "base_score"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "rank": 1,
                    "spacehastenid": identifier,
                    "method": "ei",
                    "base_score": -0.5,
                }
            )

    rows = selected_manifest(discover_run(run), hit_threshold=-8.0, strict_threshold=-10.0)

    assert [row["selection_id"] for row in rows] == ["round-1:1", "round-2:1"]
    assert [row["dock_score"] for row in rows] == [-8.0, -9.0]
    assert all(row["is_hit"] for row in rows)
    assert rows[0]["acquisition_method"] == "ei"


def test_empty_modern_history_recovers_greedy_docking_inputs(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    database_path = run / "run.dbsh"
    with Database(database_path) as database:
        database.create_schema()
        database.connection.executemany(
            "INSERT INTO data(spacehastenid,reghash,smiles,dock_score,pred_score,"
            "dock_iteration,pred_version) VALUES (?,?,?,?,?,?,?)",
            [
                (1, "seed", "C", -5.0, None, 0, None),
                (2, "h2", "CC", -9.0, -8.0, 1, None),
                (3, "h3", "CCC", None, -9.0, None, None),
                (4, "h4", "CCCC", -10.0, -7.0, 2, 1),
            ],
        )
        database.connection.executemany(
            "INSERT INTO predictions(spacehastenid,model_version,pred_score,epistemic_std) "
            "VALUES (?,?,?,?)",
            [(2, 0, -8.0, 0.2), (4, 1, -7.0, 0.4)],
        )
        database.commit()

    first = run / "run_shared/docking/iter1/inputs/chunk_1.smi"
    second = run / "run_shared/docking/iter1/inputs/chunk_2.smi"
    round_two = run / "run_shared/docking/iter2/inputs/chunk_1.smi"
    for path, text in (
        (first, "CC 2\n"),
        (second, "CCC 3\n"),
        (round_two, "CCCC 4\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    context = discover_run(run)
    assert context.acquisition_paths == ()
    assert [round_id for round_id, _ in context.docking_input_paths] == [1, 2]
    rows = selected_manifest(context, hit_threshold=-8.0, strict_threshold=-9.5)

    assert [row["spacehastenid"] for row in rows] == [3, 2, 4]
    assert rows[0]["outcome_status"] == "unresolved"
    assert rows[0]["rank_source"] == "data_prediction_score_then_id"
    assert rows[0]["model_version"] == "0"
    assert rows[1]["is_hit"] is True
    assert rows[2]["is_strict_hit"] is True
    assert {row["selection_source"] for row in rows} == {"docking_input_chunks"}
