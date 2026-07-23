#!/usr/bin/env python3
"""Prepare a sequentialized 50k normalized-EI validation docking study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from analyze_ei_acquisition_attribution import (
    RoundDefinition,
    expected_improvement_scores,
    load_candidate_pool,
)
from analyze_lcb_acquisition_attribution import penalized_top_k
from replay_normalized_ei_diversity import frontier_scale

POLICIES = {"alpha_0p05": 0.05, "alpha_0p1": 0.10, "alpha_0p2": 0.20}
ATLAS_ID = "morgan-r2-1024-t040"
TASKS = 400
BUDGET = 50_000
MOLECULES_PER_TASK = BUDGET // TASKS
RANDOM_SEED = 42
RANK_BINS = (0, 10_000, 25_000, 50_000, 75_000, 100_000)
RANK_LABELS = ("1-10k", "10-25k", "25-50k", "50-75k", "75-100k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ei-db", type=Path, required=True)
    parser.add_argument("--lcb-docked-db", type=Path, required=True)
    parser.add_argument("--greedy-docked-db", type=Path, required=True)
    parser.add_argument("--source-replay", type=Path, required=True)
    parser.add_argument("--template-inputs", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=-9.7)
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--tasks", type=int, default=TASKS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    if args.budget != BUDGET or args.tasks != TASKS:
        parser.error("this approved study requires exactly 50,000 compounds and 400 tasks")
    if args.budget % args.tasks:
        parser.error("budget must divide evenly across tasks")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readonly_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def round_one_policies(source_replay: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        source_replay,
        usecols=[
            "round",
            "policy",
            "rank",
            "spacehastenid",
            "clusterid",
            "base_score",
            "expected_improvement",
            "cluster_count_before",
            "cluster_penalty",
            "penalized_score",
        ],
    )
    frame = frame[(frame["round"] == 1) & frame["policy"].isin(POLICIES)].copy()
    if len(frame) != 3 * 100_000:
        raise ValueError("source replay does not contain three complete round-1 policies")
    if (frame.groupby("policy").size() != 100_000).any():
        raise ValueError("source replay round-1 policy sizes are invalid")
    return frame


def sequential_round_two(
    database: Path,
    round_one: pd.DataFrame,
    threshold: float,
    xi: float,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    definition = RoundDefinition(
        2,
        1,
        2,
        7_425_038,
        "d.dock_iteration IS NULL OR d.dock_iteration=2",
    )
    pool = load_candidate_pool(database, definition)
    all_base = expected_improvement_scores(pool.means, pool.epistemic, threshold, xi)
    policy_frames: list[pd.DataFrame] = []
    scale_rows: list[dict[str, Any]] = []
    for policy, alpha in POLICIES.items():
        exclude = set(
            round_one.loc[round_one["policy"] == policy, "spacehastenid"].astype(int)
        )
        mask = np.fromiter(
            (int(identifier) not in exclude for identifier in pool.identifiers),
            dtype=bool,
            count=len(pool.identifiers),
        )
        identifiers = pool.identifiers[mask]
        clusters = pool.atlas_clusters[mask]
        base = all_base[mask]
        scale = frontier_scale(base, identifiers, batch_size)
        cluster_lambda = alpha * float(scale["primary_scale"]) / math.log(2.0)
        indices, counts, penalties = penalized_top_k(
            base, identifiers, clusters, batch_size, cluster_lambda
        )
        selected_ids = identifiers[indices]
        overlap = len(exclude & set(map(int, selected_ids)))
        if overlap:
            raise ValueError(f"{policy}: {overlap} round-1 IDs remain in round 2")
        policy_frames.append(
            pd.DataFrame(
                {
                    "round": 2,
                    "policy": policy,
                    "rank": np.arange(1, batch_size + 1),
                    "spacehastenid": selected_ids,
                    "clusterid": clusters[indices],
                    "base_score": base[indices],
                    "expected_improvement": -base[indices],
                    "cluster_count_before": counts,
                    "cluster_penalty": penalties,
                    "penalized_score": base[indices] + penalties,
                }
            )
        )
        scale_rows.append(
            {
                "policy": policy,
                "alpha": alpha,
                "candidate_count_before_exclusion": len(pool.identifiers),
                "candidate_count_after_exclusion": len(identifiers),
                "round1_exclusion_count": len(exclude),
                "cluster_lambda": cluster_lambda,
                **scale,
            }
        )
    return pd.concat(policy_frames, ignore_index=True), pd.DataFrame(scale_rows)


def load_compound_info(database: Path, identifiers: set[int]) -> pd.DataFrame:
    workspace = sqlite3.connect(":memory:")
    try:
        workspace.execute("ATTACH DATABASE ? AS source", (str(database.resolve()),))
        workspace.execute("CREATE TABLE selected(spacehastenid INTEGER PRIMARY KEY)")
        ordered = sorted(identifiers)
        for start in range(0, len(ordered), 10_000):
            workspace.executemany(
                "INSERT INTO selected VALUES (?)",
                ((value,) for value in ordered[start : start + 10_000]),
            )
        frame = pd.read_sql_query(
            "SELECT d.spacehastenid,d.reghash,d.smiles,d.dock_score "
            "FROM source.data d JOIN selected s USING(spacehastenid)",
            workspace,
        )
    finally:
        workspace.close()
    if len(frame) != len(identifiers):
        raise ValueError("failed to load every selected compound")
    if frame[["spacehastenid", "reghash", "smiles"]].isna().any().any():
        raise ValueError("selected compounds contain missing identifiers or structures")
    if frame["spacehastenid"].duplicated().any() or frame["reghash"].duplicated().any():
        raise ValueError("selected compounds are not unique by ID and reghash")
    return frame


def observed_hashes(path: Path) -> set[str]:
    with readonly_connection(path) as connection:
        return {
            str(value)
            for (value,) in connection.execute(
                "SELECT reghash FROM data WHERE dock_iteration > 0 "
                "AND dock_score IS NOT NULL"
            )
        }


def membership_table(selections: pd.DataFrame, compounds: pd.DataFrame) -> pd.DataFrame:
    keys = [(round_id, policy) for round_id in (1, 2) for policy in POLICIES]
    union = compounds.set_index("spacehastenid").copy()
    for round_id, policy in keys:
        subset = selections[
            (selections["round"] == round_id) & (selections["policy"] == policy)
        ].set_index("spacehastenid")
        prefix = f"r{round_id}_{policy}"
        union[f"member_{prefix}"] = union.index.isin(subset.index)
        union[f"rank_{prefix}"] = subset["rank"].reindex(union.index)
        union[f"cluster_{prefix}"] = subset["clusterid"].reindex(union.index)
        union[f"ei_{prefix}"] = subset["expected_improvement"].reindex(union.index)
    member_columns = [f"member_r{round_id}_{policy}" for round_id, policy in keys]
    union["membership_pattern"] = union[member_columns].astype(int).astype(str).agg("".join, axis=1)
    rank_columns = [f"rank_r{round_id}_{policy}" for round_id, policy in keys]
    union["minimum_policy_rank"] = union[rank_columns].min(axis=1)
    union["rank_band"] = pd.cut(
        union["minimum_policy_rank"],
        bins=RANK_BINS,
        labels=RANK_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)
    round1_count = union[[f"member_r1_{policy}" for policy in POLICIES]].sum(axis=1)
    round2_count = union[[f"member_r2_{policy}" for policy in POLICIES]].sum(axis=1)
    union["round1_membership_count"] = round1_count
    union["round2_membership_count"] = round2_count
    union["is_policy_disagreement"] = round1_count.isin([1, 2]) | round2_count.isin([1, 2])
    round1_best = union[[f"rank_r1_{policy}" for policy in POLICIES]].min(axis=1)
    round2_best = union[[f"rank_r2_{policy}" for policy in POLICIES]].min(axis=1)
    union["primary_round"] = np.where(
        round1_count.eq(0),
        2,
        np.where(round2_count.eq(0), 1, np.where(round1_best <= round2_best, 1, 2)),
    )
    return union.reset_index()


def allocate_proportional(
    populations: pd.Series, sample_size: int
) -> pd.Series:
    if sample_size > int(populations.sum()):
        raise ValueError("sample allocation exceeds available population")
    raw = populations / populations.sum() * sample_size
    allocation = np.floor(raw).astype(int)
    remainder = sample_size - int(allocation.sum())
    order = (raw - allocation).sort_values(ascending=False).index
    allocation.loc[order[:remainder]] += 1
    return allocation


def sample_manifest(
    frame: pd.DataFrame, budget: int, random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    disagreement = frame[frame["is_policy_disagreement"]].copy()
    if len(disagreement) >= budget:
        raise ValueError(
            f"policy-disagreement census ({len(disagreement)}) exhausts the 50k budget"
        )
    disagreement["sampling_reason"] = "policy_disagreement_census"
    disagreement["sampling_stratum"] = "disagreement:" + disagreement["membership_pattern"]
    disagreement["stratum_population"] = disagreement.groupby("sampling_stratum")[
        "spacehastenid"
    ].transform("size")
    disagreement["stratum_sample_n"] = disagreement["stratum_population"]
    disagreement["inclusion_probability"] = 1.0

    remaining = budget - len(disagreement)
    common = frame[~frame["is_policy_disagreement"]].copy()
    common["sampling_reason"] = "common_policy_probability_sample"
    common["sampling_stratum"] = (
        "common:r"
        + common["primary_round"].astype(str)
        + ":"
        + common["membership_pattern"]
        + ":"
        + common["rank_band"]
    )
    round_targets = {1: int(round(remaining * 0.60)), 2: remaining - int(round(remaining * 0.60))}
    sampled_parts: list[pd.DataFrame] = []
    allocation_rows: list[dict[str, Any]] = []
    for round_id in (1, 2):
        round_common = common[common["primary_round"] == round_id]
        populations = round_common.groupby("sampling_stratum").size()
        allocation = allocate_proportional(populations, round_targets[round_id])
        for stratum, population in populations.items():
            n = int(allocation.loc[stratum])
            stratum_frame = round_common[round_common["sampling_stratum"] == stratum]
            rng = np.random.default_rng(
                np.random.SeedSequence([random_seed, round_id, sum(map(ord, stratum))])
            )
            chosen = rng.choice(len(stratum_frame), n, replace=False)
            sample = stratum_frame.iloc[np.sort(chosen)].copy()
            sample["stratum_population"] = int(population)
            sample["stratum_sample_n"] = n
            sample["inclusion_probability"] = n / population
            sampled_parts.append(sample)
            allocation_rows.append(
                {
                    "sampling_stratum": stratum,
                    "primary_round": round_id,
                    "population": int(population),
                    "sample_n": n,
                    "inclusion_probability": n / population,
                }
            )
    sampled_common = pd.concat(sampled_parts, ignore_index=True)
    selected = pd.concat([disagreement, sampled_common], ignore_index=True)
    if len(selected) != budget or selected["spacehastenid"].duplicated().any():
        raise ValueError("sampling did not produce exactly 50,000 unique compounds")
    selected["sampling_weight"] = 1.0 / selected["inclusion_probability"]
    selected = selected.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    selected["study_index"] = np.arange(1, budget + 1)
    selected["task_id"] = (selected.index // MOLECULES_PER_TASK) + 1
    selected["task_row"] = (selected.index % MOLECULES_PER_TASK) + 1

    disagreement_allocations = (
        disagreement.groupby("sampling_stratum")
        .size()
        .rename("population")
        .reset_index()
    )
    disagreement_allocations["primary_round"] = 0
    disagreement_allocations["sample_n"] = disagreement_allocations["population"]
    disagreement_allocations["inclusion_probability"] = 1.0
    allocations = pd.concat(
        [disagreement_allocations, pd.DataFrame(allocation_rows)], ignore_index=True
    )
    return selected, allocations


def write_inputs(
    manifest: pd.DataFrame,
    output_dir: Path,
    template_inputs: Path,
    grid: Path,
) -> None:
    inputs = output_dir / "inputs"
    results = output_dir / "results"
    logs = output_dir / "logs" / "slurm"
    inputs.mkdir(parents=True, exist_ok=False)
    results.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    shutil.copy2(grid, output_dir / "glide_grid.zip")
    phase_template = (template_inputs / "chunk_1.inp").read_text(encoding="utf-8")
    glide_template = (template_inputs / "glide_chunk_1.in").read_text(encoding="utf-8")
    for task_id, chunk in manifest.groupby("task_id", sort=True):
        if len(chunk) != MOLECULES_PER_TASK:
            raise ValueError(f"task {task_id} has {len(chunk)} compounds, expected 125")
        with (inputs / f"chunk_{task_id}.smi").open("wt", encoding="utf-8") as handle:
            for row in chunk.sort_values("task_row").itertuples(index=False):
                handle.write(f"{row.smiles.strip()} {int(row.spacehastenid)}\n")
        (inputs / f"chunk_{task_id}.inp").write_text(
            phase_template.replace("chunk_1", f"chunk_{task_id}"), encoding="utf-8"
        )
        (inputs / f"glide_chunk_{task_id}.in").write_text(
            glide_template.replace("chunk_1", f"chunk_{task_id}"), encoding="utf-8"
        )


def write_submit_script(output_dir: Path, tasks: int) -> Path:
    script = output_dir / "submit_docking_50k.sh"
    log_root = output_dir / "logs" / "slurm"
    body = f"""#!/bin/bash
#SBATCH --job-name=ei_norm_50k
#SBATCH --partition=jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-{tasks}
#SBATCH --output={log_root}/task-%A_%a.out
#SBATCH --error={log_root}/task-%A_%a.err
#SBATCH --chdir={output_dir}

set -euo pipefail

TASK_ID="${{SLURM_ARRAY_TASK_ID}}"
RESULT="results/results-chunk_${{TASK_ID}}.tar.gz"
if [[ -s "$RESULT" ]] && tar -tzf "$RESULT" >/dev/null 2>&1; then
    echo "[task ${{TASK_ID}}] Valid result already exists"
    exit 0
fi
rm -f "$RESULT"

check_and_start_jobserver() {{
    status=$($SCHRODINGER/jsc local-server-status 2>&1 || true)
    if echo "$status" | grep -q "STOPPED"; then
        $SCHRODINGER/jsc local-server-start
    fi
}}
check_and_start_jobserver

CURDIR=$(pwd)
SCRATCH_DIR="/wrk/${{USER}}/ei_norm_50k_${{SLURM_ARRAY_JOB_ID}}_${{TASK_ID}}"
cleanup() {{ rm -rf "$SCRATCH_DIR"; }}
trap cleanup EXIT
rm -rf "$SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"
cp "inputs/chunk_${{TASK_ID}}.smi" "$SCRATCH_DIR/"
cp "inputs/chunk_${{TASK_ID}}.inp" "$SCRATCH_DIR/"
cp "inputs/glide_chunk_${{TASK_ID}}.in" "$SCRATCH_DIR/"
cp glide_grid.zip "$SCRATCH_DIR/"
cd "$SCRATCH_DIR"

echo "[task ${{TASK_ID}}] Building phase database"
$SCHRODINGER/pipeline -prog phase_db "chunk_${{TASK_ID}}.inp" -OVERWRITE -WAIT -NOJOBID -NJOBS 1
echo "[task ${{TASK_ID}}] Exporting structures"
$SCHRODINGER/phase_database "$(pwd)/chunk_${{TASK_ID}}.phdb" export \
    -omae "$(pwd)/chunk_${{TASK_ID}}" -get 1 -limit 99999999 -WAIT
rm -rf "$(pwd)/chunk_${{TASK_ID}}.phdb"
echo "[task ${{TASK_ID}}] Running Glide docking"
$SCHRODINGER/glide -new -OVERWRITE -WAIT -NJOBS 1 -HOST localhost:1 "glide_chunk_${{TASK_ID}}.in"
echo "[task ${{TASK_ID}}] Packaging results"
rm -f glide_grid.zip
TMP_RESULT="$CURDIR/results/.results-chunk_${{TASK_ID}}.tar.gz.tmp"
tar -czf "$TMP_RESULT" .
tar -tzf "$TMP_RESULT" >/dev/null
mv "$TMP_RESULT" "$CURDIR/$RESULT"
echo "[task ${{TASK_ID}}] Done"
"""
    script.write_text(body, encoding="utf-8")
    script.chmod(0o755)
    return script


def calculate(args: argparse.Namespace) -> None:
    started = datetime.now(UTC)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    round1 = round_one_policies(args.source_replay)
    round2, scales = sequential_round_two(
        args.ei_db, round1, args.threshold, args.xi, args.batch_size
    )
    selections = pd.concat([round1, round2], ignore_index=True)
    for policy in POLICIES:
        first = set(
            selections.loc[
                (selections["round"] == 1) & (selections["policy"] == policy),
                "spacehastenid",
            ].astype(int)
        )
        second = set(
            selections.loc[
                (selections["round"] == 2) & (selections["policy"] == policy),
                "spacehastenid",
            ].astype(int)
        )
        if first & second:
            raise ValueError(f"{policy} contains cross-round duplicate selections")
    identifiers = set(selections["spacehastenid"].astype(int))
    compounds = load_compound_info(args.ei_db, identifiers)
    known_hashes = observed_hashes(args.lcb_docked_db) | observed_hashes(
        args.greedy_docked_db
    )
    compounds["has_existing_outcome"] = compounds["dock_score"].notna() | compounds[
        "reghash"
    ].isin(known_hashes)
    membership = membership_table(selections, compounds)
    unobserved = membership[~membership["has_existing_outcome"]].copy()
    manifest, allocations = sample_manifest(unobserved, args.budget, args.random_seed)
    if manifest["reghash"].isin(known_hashes).any() or manifest["dock_score"].notna().any():
        raise ValueError("docking manifest contains compounds with existing outcomes")

    selections.to_csv(
        output_dir / "sequential_policy_selected_ids.csv.gz",
        index=False,
        compression="gzip",
    )
    scales.to_csv(output_dir / "sequential_round2_scales.csv", index=False)
    membership.to_csv(
        output_dir / "policy_membership_population.csv.gz",
        index=False,
        compression="gzip",
    )
    manifest.to_csv(output_dir / "docking_manifest_50k.csv", index=False)
    allocations.to_csv(output_dir / "sampling_strata.csv", index=False)
    write_inputs(manifest, output_dir, args.template_inputs, args.grid)
    submit_script = write_submit_script(output_dir, args.tasks)
    provenance = {
        "generated_at": started.isoformat(),
        "definition": {
            "policies": POLICIES,
            "beta": 1.0,
            "threshold": args.threshold,
            "xi": args.xi,
            "budget": args.budget,
            "tasks": args.tasks,
            "molecules_per_task": MOLECULES_PER_TASK,
            "sampling": (
                "census of all unobserved policy-disagreement compounds; remaining "
                "budget allocated 60/40 by primary round and proportionally across "
                "exact membership-pattern/rank-band strata"
            ),
            "round2_sequentialization": (
                "each policy excludes its own alternative round-1 selections from the "
                "historical model-v1 candidate pool; historical round-1 compounds lacking "
                "model-v1 predictions cannot be reintroduced"
            ),
        },
        "validation": {
            "manifest_rows": len(manifest),
            "unique_spacehastenids": manifest["spacehastenid"].nunique(),
            "unique_reghashes": manifest["reghash"].nunique(),
            "tasks": manifest["task_id"].nunique(),
            "task_size_min": int(manifest.groupby("task_id").size().min()),
            "task_size_max": int(manifest.groupby("task_id").size().max()),
            "known_outcomes_in_manifest": int(manifest["has_existing_outcome"].sum()),
            "policy_disagreement_census": int(manifest["is_policy_disagreement"].sum()),
        },
        "inputs": {
            "ei_db": str(args.ei_db.resolve()),
            "source_replay": str(args.source_replay.resolve()),
            "grid": str(args.grid.resolve()),
            "grid_sha256": sha256(args.grid),
            "phase_template": str((args.template_inputs / "chunk_1.inp").resolve()),
            "glide_template": str(
                (args.template_inputs / "glide_chunk_1.in").resolve()
            ),
        },
        "submit_script": str(submit_script),
    }
    (output_dir / "study_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2, default=str))


def main() -> int:
    try:
        calculate(parse_args())
    except Exception as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
