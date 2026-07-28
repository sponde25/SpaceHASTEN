#!/usr/bin/env python3
"""Analyze training metadata, leakage, prediction drift, and run timing."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from spacehasten.analysis.artifacts import write_json
from spacehasten.analysis.discovery import discover_run

TIMESTAMP = re.compile(r"\[(\d\d/\d\d/\d\d \d\d:\d\d:\d\d)\]")
ISO_TIMESTAMP = re.compile(r"^(\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d)")
ROUND_START = re.compile(r"Screening round (\d+)/(\d+)", re.I)
ROUND_END = re.compile(r"Updated dock_score for \d+ rows \(iter=(\d+)\)", re.I)
SUBMITTED = re.compile(r"Submitted .*? job (\d+)", re.I)
COMPLETED = re.compile(r"Job (\d+) \(([^)]+)\) completed successfully", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"spacehastenid", "round", "smiles", "model_version", "is_scored", "is_hit"}
    if missing := required - set(frame.columns):
        raise ValueError(f"selected manifest lacks columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("selected manifest is empty")
    if (frame.groupby("spacehastenid")["smiles"].nunique() > 1).any():
        raise ValueError("selected IDs map to multiple structures")
    return frame


def flatten_metadata(path: Path, version: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    row: dict[str, Any] = {"model_version": version, "metadata_path": str(path)}
    for key, value in payload.items():
        row[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
    return row


def training_tables(run: Path, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_root = run / "run_shared" / "models"
    metadata_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    directory_versions = {
        int(path.name[1:]) for path in model_root.glob("v*") if path.name[1:].isdigit()
    }
    used_versions = {
        int(value)
        for value in pd.to_numeric(manifest["model_version"], errors="coerce").dropna()
    }
    for version in sorted(directory_versions | used_versions):
        directory = model_root / f"v{version}"
        metadata_path = directory / "training_metadata.json"
        if metadata_path.is_file():
            metadata_rows.append(flatten_metadata(metadata_path, version))
        selected = manifest[pd.to_numeric(manifest["model_version"], errors="coerce") == version]
        train_path = directory / "train.csv"
        if selected.empty:
            status, overlap, reason = "not_used", 0, "no acquisition round used this model"
        elif not train_path.is_file():
            status, overlap, reason = "unavailable", 0, "training CSV is absent"
        else:
            training = pd.read_csv(train_path)
            if "smiles" not in training.columns:
                status, overlap, reason = "unavailable", 0, "training CSV lacks smiles"
            else:
                overlap = len(set(selected["smiles"].dropna()) & set(training["smiles"].dropna()))
                status = "passed" if overlap == 0 else "failed"
                reason = None
        leakage_rows.append(
            {
                "model_version": version,
                "selected_compounds": len(selected),
                "overlapping_smiles": overlap,
                "status": status,
                "reason": reason,
            }
        )
    return pd.DataFrame(metadata_rows), pd.DataFrame(leakage_rows)


def prediction_drift(database: Path) -> pd.DataFrame:
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(predictions)")}
        if not {"model_version", "pred_score"} <= columns:
            return pd.DataFrame()
        expressions = ["COUNT(*) AS prediction_count", "AVG(pred_score) AS pred_score_mean"]
        expressions.extend(
            f"AVG({column}) AS {column}_mean"
            for column in ("epistemic_std", "aleatoric_std", "total_std")
            if column in columns
        )
        query = (
            "SELECT model_version," + ",".join(expressions) + " FROM predictions "
            "GROUP BY model_version ORDER BY model_version"
        )
        return pd.read_sql_query(query, connection)


def log_events(paths: list[Path]) -> list[tuple[datetime, str]]:
    events: list[tuple[datetime, str]] = []
    for path in paths:
        current: datetime | None = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if match := TIMESTAMP.search(line):
                current = datetime.strptime(match.group(1), "%m/%d/%y %H:%M:%S")
            elif match := ISO_TIMESTAMP.search(line):
                current = datetime.fromisoformat(match.group(1).replace(" ", "T"))
            if current is not None:
                events.append((current, line.strip()))
    return sorted(set(events))


def stage_name(label: str) -> str:
    lower = label.lower()
    for prefix, stage in (
        ("dock", "docking"),
        ("train", "training"),
        ("predict", "prediction"),
        ("atlas", "atlas"),
        ("search", "simsearch"),
        ("control", "simsearch_control"),
    ):
        if lower.startswith(prefix):
            return stage
    return "other"


def timing_tables(
    paths: list[Path],
    manifest: pd.DataFrame,
    hit_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = log_events(paths)
    rounds = sorted(manifest["round"].unique().astype(int).tolist())
    starts: dict[int, datetime] = {}
    ends: dict[int, datetime] = {}
    for timestamp, text in events:
        if match := ROUND_START.search(text):
            starts.setdefault(int(match.group(1)), timestamp)
        if match := ROUND_END.search(text):
            ends[int(match.group(1))] = timestamp
    if set(starts) < set(rounds) or set(ends) < set(rounds):
        raise ValueError(
            "run logs lack explicit start or docking-completion boundaries for a round"
        )
    completed = {
        match.group(1): (match.group(2), timestamp)
        for timestamp, text in events
        if (match := COMPLETED.search(text))
    }
    submitted: dict[str, datetime] = {}
    for timestamp, text in events:
        if match := SUBMITTED.search(text):
            submitted.setdefault(match.group(1), timestamp)
    stage_rows: list[dict[str, Any]] = []
    for job_id, submitted_at in submitted.items():
        if job_id not in completed:
            continue
        label, completed_at = completed[job_id]
        round_id = next(
            (
                value
                for value in rounds
                if starts[value] <= submitted_at <= ends[value]
            ),
            None,
        )
        if round_id is None:
            continue
        stage_rows.append(
            {
                "round": round_id,
                "stage": stage_name(label),
                "job_id": job_id,
                "job_name": label,
                "submitted_at": submitted_at.isoformat(sep=" "),
                "completed_at": completed_at.isoformat(sep=" "),
                "log_wall_seconds": (completed_at - submitted_at).total_seconds(),
            }
        )
    round_rows = []
    cumulative = 0.0
    for round_id in rounds:
        seconds = (ends[round_id] - starts[round_id]).total_seconds()
        if seconds <= 0:
            raise ValueError(f"round {round_id} has non-positive wall time")
        cumulative += seconds
        current = manifest[manifest["round"] == round_id]
        scored = current[current["is_scored"]]
        hits = scored[scored["is_hit"]]
        round_rows.append(
            {
                "round": round_id,
                "round_start": starts[round_id].isoformat(sep=" "),
                "round_end": ends[round_id].isoformat(sep=" "),
                "end_to_end_wall_seconds": seconds,
                "cumulative_wall_seconds": cumulative,
                "selected": len(current),
                "scored": len(scored),
                "hits": len(hits),
                "selected_per_wall_hour": len(current) / (seconds / 3600),
                "scored_per_wall_hour": len(scored) / (seconds / 3600),
                "hits_per_wall_hour": len(hits) / (seconds / 3600),
                "hit_threshold": hit_threshold,
            }
        )
    return pd.DataFrame(stage_rows), pd.DataFrame(round_rows)


def scheduler_accounting(stages: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    identifiers = sorted(set(stages["job_id"].astype(str)))
    if not identifiers:
        return pd.DataFrame(), pd.DataFrame()
    columns = "JobIDRaw,JobName,State,ElapsedRaw,CPUTimeRAW,AllocCPUS,Submit,Start,End"
    completed = subprocess.run(
        ["sacct", "-j", ",".join(identifiers), "--parsable2", "--noheader", f"--format={columns}"],
        check=True,
        capture_output=True,
        text=True,
    )
    tasks = pd.read_csv(io.StringIO(completed.stdout), sep="|", names=columns.split(","))
    tasks = tasks.dropna(subset=["JobIDRaw", "State"])
    for column in ("ElapsedRaw", "CPUTimeRAW", "AllocCPUS"):
        tasks[column] = pd.to_numeric(tasks[column], errors="coerce")
    submitted = pd.to_datetime(tasks["Submit"], errors="coerce")
    started = pd.to_datetime(tasks["Start"], errors="coerce")
    tasks["queue_wait_seconds"] = (started - submitted).dt.total_seconds()
    tasks["parent_job_id"] = (
        tasks["JobIDRaw"].astype(str).str.split("_").str[0].str.split(".").str[0]
    )
    tasks = tasks.merge(
        stages[["job_id", "round", "stage"]].drop_duplicates("job_id"),
        left_on="parent_job_id",
        right_on="job_id",
        how="left",
    )
    summary = (
        tasks.groupby(["round", "stage"], dropna=False)
        .agg(
            tasks=("JobIDRaw", "size"),
            elapsed_median_seconds=("ElapsedRaw", "median"),
            elapsed_p95_seconds=("ElapsedRaw", lambda values: values.quantile(0.95)),
            elapsed_max_seconds=("ElapsedRaw", "max"),
            queue_wait_median_seconds=("queue_wait_seconds", "median"),
            allocated_cpus=("AllocCPUS", "sum"),
            cpu_time_seconds=("CPUTimeRAW", "sum"),
        )
        .reset_index()
    )
    return tasks, summary


def save_figures(root: Path, drift: pd.DataFrame, rounds: pd.DataFrame, dpi: int) -> None:
    plt.style.use("tableau-colorblind10")
    if not drift.empty:
        figure, axis = plt.subplots(figsize=(5.5, 3.7))
        axis.plot(drift["model_version"], drift["pred_score_mean"], marker="o")
        axis.set(xlabel="Model version", ylabel="Mean predicted score")
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure.savefig(root / "prediction_drift.png", dpi=dpi, bbox_inches="tight")
        figure.savefig(root / "prediction_drift.pdf", bbox_inches="tight")
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(5.5, 3.7))
    axis.bar(rounds["round"], rounds["end_to_end_wall_seconds"] / 3600)
    axis.set(xlabel="Round", ylabel="End-to-end wall time (hours)", xticks=rounds["round"])
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(root / "round_timing.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(root / "round_timing.pdf", bbox_inches="tight")
    plt.close(figure)


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def analyze(args: argparse.Namespace) -> None:
    context = discover_run(args.run_or_db, database_path=args.database)
    run = context.input_path if context.input_path.is_dir() else context.input_path.parent
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output root is non-empty; pass --overwrite: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest)
    training, leakage = training_tables(run, manifest)
    drift = prediction_drift(context.database_path)
    logs = args.log or [run / "run.log", run / "run_local" / "logs" / "spacehasten.log"]
    logs = [path for path in logs if path.is_file()]
    if not logs:
        raise ValueError("no run logs were found; pass one or more --log paths")
    stages, rounds = timing_tables(logs, manifest, args.hit_threshold)
    outputs = {
        "training_metadata.csv": training,
        "training_leakage_validation.csv": leakage,
        "candidate_prediction_drift.csv": drift,
        "stage_timing.csv": stages,
        "round_timing.csv": rounds,
    }
    if args.sacct:
        tasks, scheduler = scheduler_accounting(stages)
        outputs["sacct_tasks.csv"] = tasks
        outputs["sacct_summary.csv"] = scheduler
    for name, frame in outputs.items():
        write_frame(frame, root / name)
    save_figures(root, drift, rounds, args.dpi)
    failed = leakage[leakage["status"] == "failed"]
    artifacts = sorted(path for path in root.iterdir() if path.is_file())
    receipt = {
        "status": "complete" if failed.empty else "failed",
        "database": str(context.database_path),
        "manifest": str(args.manifest.resolve()),
        "logs": [str(path.resolve()) for path in logs],
        "rounds": rounds["round"].astype(int).tolist(),
        "models_with_metadata": len(training),
        "leakage_failures": len(failed),
        "stage_jobs": len(stages),
        "scheduler_accounting": bool(args.sacct),
        "outputs": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        ],
    }
    write_json(root / "_SUCCESS.json", receipt)
    if not failed.empty:
        raise ValueError("training leakage validation failed")
    print(json.dumps(receipt, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_or_db")
    parser.add_argument("--database", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log", type=Path, action="append")
    parser.add_argument("--hit-threshold", type=float, required=True)
    parser.add_argument("--sacct", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.dpi < 1 or not args.manifest.is_file():
        parser.error("dpi must be positive and manifest must exist")
    analyze(args)


if __name__ == "__main__":
    main()
