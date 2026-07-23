#!/usr/bin/env python3
# ruff: noqa: E501
"""Analyze the completed normalized-EI 50k docking validation study."""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import multiprocessing as mp
import sqlite3
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from compare_core_diversity import BYTE_POPCOUNT, structure_worker
from tqdm import tqdm

plt.switch_backend("Agg")
LOGGER = logging.getLogger("analyze_normalized_ei_docking_study")
POLICIES = ("alpha_0p05", "alpha_0p1", "alpha_0p2")
POLICY_LABELS = {
    "alpha_0p05": "alpha=0.05",
    "alpha_0p1": "alpha=0.10",
    "alpha_0p2": "alpha=0.20",
}
COLORS = {
    "alpha_0p05": "#0072B2",
    "alpha_0p1": "#009E73",
    "alpha_0p2": "#CC79A7",
}
CUTOFF = -9.7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--ei-db", type=Path, required=True)
    parser.add_argument("--lcb-docked-db", type=Path, required=True)
    parser.add_argument("--greedy-docked-db", type=Path, required=True)
    parser.add_argument("--reference-diversity", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=CUTOFF)
    parser.add_argument("--simulations", type=int, default=20_000)
    parser.add_argument("--pair-samples", type=int, default=500_000)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if min(args.simulations, args.pair_samples, args.processes, args.dpi) < 1:
        parser.error("simulation, pair, process, and DPI values must be positive")
    return args


def parse_result_archives(study_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_by_id: dict[int, float] = {}
    archive_rows: list[dict[str, Any]] = []
    for task_id in tqdm(range(1, 401), desc="Parsing Glide archives", unit="task"):
        path = study_dir / "results" / f"results-chunk_{task_id}.tar.gz"
        with tarfile.open(path, "r:gz") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.name.endswith(f"glide_chunk_{task_id}.csv")
            ]
            if len(members) != 1:
                raise ValueError(f"task {task_id}: expected one non-skip Glide CSV")
            handle = archive.extractfile(members[0])
            if handle is None:
                raise ValueError(f"task {task_id}: cannot read Glide CSV")
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8"))
            row_count = 0
            task_ids: set[int] = set()
            for row in reader:
                row_count += 1
                identifier = int(row["title"])
                score = float(row["r_i_docking_score"])
                task_ids.add(identifier)
                previous = score_by_id.get(identifier)
                if previous is None or score < previous:
                    score_by_id[identifier] = score
            archive_rows.append(
                {
                    "task_id": task_id,
                    "pose_rows": row_count,
                    "scored_titles": len(task_ids),
                    "archive_size_bytes": path.stat().st_size,
                }
            )
    scores = pd.DataFrame(
        sorted(score_by_id.items()), columns=["spacehastenid", "new_dock_score"]
    )
    return scores, pd.DataFrame(archive_rows)


def score_map(path: Path) -> dict[str, float]:
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
        return {
            str(reghash): float(score)
            for reghash, score in connection.execute(
                "SELECT reghash,dock_score FROM data WHERE dock_iteration > 0 "
                "AND dock_score IS NOT NULL"
            )
        }


def attach_existing_scores(
    population: pd.DataFrame,
    ei_db: Path,
    lcb_db: Path,
    greedy_db: Path,
) -> pd.DataFrame:
    result = population.copy()
    ei_scores = score_map(ei_db)
    lcb_scores = score_map(lcb_db)
    greedy_scores = score_map(greedy_db)
    ei = result["reghash"].map(ei_scores)
    lcb = result["reghash"].map(lcb_scores)
    greedy = result["reghash"].map(greedy_scores)
    result["existing_score"] = ei.fillna(lcb).fillna(greedy)
    result["existing_source"] = np.select(
        [ei.notna(), lcb.notna(), greedy.notna()],
        ["ei", "lcb", "greedy"],
        default="unobserved",
    )
    return result


def sampling_stratum(frame: pd.DataFrame) -> pd.Series:
    return np.where(
        frame["is_policy_disagreement"],
        "disagreement:" + frame["membership_pattern"],
        "common:r"
        + frame["primary_round"].astype(str)
        + ":"
        + frame["membership_pattern"]
        + ":"
        + frame["rank_band"],
    )


def prepare_analysis_population(
    study_dir: Path,
    scores: pd.DataFrame,
    ei_db: Path,
    lcb_db: Path,
    greedy_db: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dtypes = {"membership_pattern": "string"}
    population = pd.read_csv(
        study_dir / "policy_membership_population.csv.gz", dtype=dtypes
    )
    manifest = pd.read_csv(study_dir / "docking_manifest_50k.csv", dtype=dtypes)
    population = attach_existing_scores(population, ei_db, lcb_db, greedy_db)
    population["sampling_stratum"] = sampling_stratum(population)
    manifest_scores = manifest[["spacehastenid", "sampling_stratum"]].merge(
        scores, on="spacehastenid", how="left", validate="one_to_one"
    )
    manifest_scores["study_sampled"] = True
    population = population.merge(
        manifest_scores[["spacehastenid", "study_sampled", "new_dock_score"]],
        on="spacehastenid",
        how="left",
        validate="one_to_one",
    )
    population["study_sampled"] = population["study_sampled"].fillna(False)
    population["analysis_score"] = population["existing_score"].fillna(
        population["new_dock_score"]
    )
    population["score_source"] = np.where(
        population["existing_score"].notna(),
        population["existing_source"],
        np.where(population["new_dock_score"].notna(), "new_50k", "unobserved"),
    )
    return population, manifest_scores


def membership_column(round_id: int, policy: str) -> str:
    return f"member_r{round_id}_{policy}"


def stratum_statistics(population: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    unknown = population[population["existing_score"].isna()].copy()
    rows: list[dict[str, Any]] = []
    for stratum, group in unknown.groupby("sampling_stratum", observed=True):
        sampled = group[group["study_sampled"]]
        scored = sampled[sampled["new_dock_score"].notna()]
        n_population = len(group)
        n_sampled = len(sampled)
        n_scored = len(scored)
        if n_scored == 0:
            raise ValueError(f"stratum {stratum!r} has no successful docking outcomes")
        values = (scored["new_dock_score"] <= cutoff).astype(float).to_numpy()
        rows.append(
            {
                "sampling_stratum": stratum,
                "population": n_population,
                "sampled": n_sampled,
                "scored_sample": n_scored,
                "docking_failures": n_sampled - n_scored,
                "hit_count": int(values.sum()),
                "hit_rate": float(values.mean()),
                "sample_variance": float(values.var(ddof=1)) if n_scored > 1 else 0.0,
                "mean_dock_score": float(scored["new_dock_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def estimate_policy_rates(
    population: pd.DataFrame,
    cutoff: float,
    simulations: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strata = stratum_statistics(population, cutoff).set_index("sampling_stratum")
    rng = np.random.default_rng(random_seed)
    stratum_simulations: dict[str, np.ndarray] = {}
    for stratum, row in strata.iterrows():
        population_n = int(row["population"])
        n = int(row["scored_sample"])
        rate = float(row["hit_rate"])
        variance = float(row["sample_variance"])
        fraction = min(n / population_n, 1.0)
        standard_error = math.sqrt(max((1.0 - fraction) * variance / n, 0.0))
        if standard_error:
            stratum_simulations[stratum] = np.clip(
                rng.normal(rate, standard_error, size=simulations), 0.0, 1.0
            )
        else:
            stratum_simulations[stratum] = np.full(simulations, rate)
    policy_rows: list[dict[str, Any]] = []
    simulation_values: dict[tuple[int, str], np.ndarray] = {}
    for round_id in (1, 2):
        for policy in POLICIES:
            members = population[population[membership_column(round_id, policy)]]
            known = members[members["existing_score"].notna()]
            unknown = members[members["existing_score"].isna()]
            known_hits = int((known["existing_score"] <= cutoff).sum())
            estimates = np.full(simulations, known_hits, dtype=float)
            unknown_estimate = 0.0
            variance = 0.0
            for stratum, group in unknown.groupby("sampling_stratum", observed=True):
                stats_row = strata.loc[stratum]
                population_n = len(group)
                if population_n != int(stats_row["population"]):
                    # Exact membership-pattern strata mean a policy either owns all or none.
                    raise ValueError(f"partial policy membership inside stratum {stratum}")
                n = int(stats_row["scored_sample"])
                rate = float(stats_row["hit_rate"])
                sample_variance = float(stats_row["sample_variance"])
                fraction = min(n / population_n, 1.0)
                standard_error = math.sqrt(
                    max((1.0 - fraction) * sample_variance / n, 0.0)
                )
                unknown_estimate += population_n * rate
                variance += population_n**2 * standard_error**2
                estimates += population_n * stratum_simulations[stratum]
            estimate = (known_hits + unknown_estimate) / 100_000
            standard_error = math.sqrt(variance) / 100_000
            rates = estimates / 100_000
            simulation_values[(round_id, policy)] = rates
            policy_rows.append(
                {
                    "round": round_id,
                    "policy": policy,
                    "selected": len(members),
                    "known_outcomes": len(known),
                    "unknown_population": len(unknown),
                    "new_scored_in_policy": int(
                        unknown["new_dock_score"].notna().sum()
                    ),
                    "estimated_hit_rate": estimate,
                    "standard_error": standard_error,
                    "ci95_low": float(np.quantile(rates, 0.025)),
                    "ci95_high": float(np.quantile(rates, 0.975)),
                    "estimated_hits": estimate * 100_000,
                }
            )
    policy_estimates = pd.DataFrame(policy_rows)
    overall_rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        rates = 0.5 * (
            simulation_values[(1, policy)] + simulation_values[(2, policy)]
        )
        point = policy_estimates.loc[
            policy_estimates["policy"] == policy, "estimated_hit_rate"
        ].mean()
        overall_rows.append(
            {
                "policy": policy,
                "estimated_hit_rate": point,
                "ci95_low": float(np.quantile(rates, 0.025)),
                "ci95_high": float(np.quantile(rates, 0.975)),
                "estimated_hits_per_200k": point * 200_000,
            }
        )
        simulation_values[(0, policy)] = rates
    overall_estimates = pd.DataFrame(overall_rows)

    reference_rates = {
        "Greedy": {0: 0.58876, 1: 0.60683, 2: 0.57069},
        "SVDKL-LCB": {0: 0.47878005970686627, 1: 0.36388911112889033, 2: 0.5936790518577787},
    }
    comparison_rows: list[dict[str, Any]] = []
    for round_id in (0, 1, 2):
        for policy in POLICIES:
            rates = simulation_values[(round_id, policy)]
            point = float(rates.mean())
            for reference, by_round in reference_rates.items():
                differences = rates - by_round[round_id]
                comparison_rows.append(
                    {
                        "round": "overall" if round_id == 0 else round_id,
                        "policy": policy,
                        "reference": reference,
                        "reference_hit_rate": by_round[round_id],
                        "policy_hit_rate": point,
                        "difference_pp": 100 * (point - by_round[round_id]),
                        "difference_ci95_low_pp": 100
                        * float(np.quantile(differences, 0.025)),
                        "difference_ci95_high_pp": 100
                        * float(np.quantile(differences, 0.975)),
                        "probability_noninferior_margin_2pp": float(
                            np.mean(differences >= -0.02)
                        ),
                        "probability_superior": float(np.mean(differences > 0)),
                    }
                )
    pairwise_rows: list[dict[str, Any]] = []
    for round_id in (0, 1, 2):
        for left_index, left in enumerate(POLICIES):
            for right in POLICIES[left_index + 1 :]:
                differences = (
                    simulation_values[(round_id, right)]
                    - simulation_values[(round_id, left)]
                )
                pairwise_rows.append(
                    {
                        "round": "overall" if round_id == 0 else round_id,
                        "policy_left": left,
                        "policy_right": right,
                        "right_minus_left_pp": 100 * float(differences.mean()),
                        "ci95_low_pp": 100 * float(np.quantile(differences, 0.025)),
                        "ci95_high_pp": 100 * float(np.quantile(differences, 0.975)),
                        "probability_right_superior": float(np.mean(differences > 0)),
                    }
                )
    return (
        policy_estimates,
        overall_estimates,
        pd.DataFrame(comparison_rows),
        pd.DataFrame(pairwise_rows),
    )


def build_hit_structures(
    population: pd.DataFrame,
    cutoff: float,
    processes: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    hits = population[
        population["analysis_score"].notna() & (population["analysis_score"] <= cutoff)
    ].copy()
    chunksize = max(1, len(hits) // (processes * 16))
    typed: list[str] = []
    generic: list[str] = []
    binaries: list[bytes] = []
    popcounts: list[int] = []
    with mp.get_context("fork").Pool(processes) as pool:
        results = pool.imap(structure_worker, hits["smiles"], chunksize=chunksize)
        for typed_value, generic_value, binary, popcount in tqdm(
            results,
            total=len(hits),
            desc="Hit fingerprints and scaffolds",
            unit="mol",
        ):
            typed.append(typed_value)
            generic.append(generic_value)
            binaries.append(binary)
            popcounts.append(popcount)
    hits["typed_scaffold"] = typed
    hits["generic_framework"] = generic
    hits["hit_structure_index"] = np.arange(len(hits), dtype=np.int64)
    words = np.frombuffer(b"".join(binaries), dtype=np.uint64).reshape(len(hits), 16)
    return hits, words.copy(), np.asarray(popcounts, dtype=np.int16)


def weighted_family_metrics(
    values: pd.Series, weights: np.ndarray, prefix: str
) -> dict[str, float | int]:
    frame = pd.DataFrame({"family": values.to_numpy(), "weight": weights})
    counts = frame.groupby("family", sort=False)["weight"].sum().to_numpy()
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return {
        f"observed_{prefix}_richness": int(len(counts)),
        f"weighted_{prefix}_effective_number": float(math.exp(entropy)),
        f"weighted_{prefix}_largest_fraction": float(probabilities.max()),
        f"weighted_{prefix}_entropy": entropy,
    }


def weighted_internal_diversity(
    indices: np.ndarray,
    weights: np.ndarray,
    words: np.ndarray,
    popcounts: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> float:
    probabilities = weights / weights.sum()
    total = 0.0
    completed = 0
    while completed < samples:
        size = min(100_000, samples - completed)
        left = rng.choice(indices, size=size, replace=True, p=probabilities)
        right = rng.choice(indices, size=size, replace=True, p=probabilities)
        equal = left == right
        while equal.any():
            right[equal] = rng.choice(
                indices, size=int(equal.sum()), replace=True, p=probabilities
            )
            equal = left == right
        intersections = BYTE_POPCOUNT[
            np.bitwise_and(words[left], words[right]).view(np.uint8)
        ].sum(axis=1)
        similarities = intersections / (
            popcounts[left] + popcounts[right] - intersections
        )
        total += float(similarities.sum())
        completed += size
    return 1.0 - total / samples


def estimate_hit_diversity(
    population: pd.DataFrame,
    cutoff: float,
    pair_samples: int,
    processes: int,
    random_seed: int,
) -> pd.DataFrame:
    strata = stratum_statistics(population, cutoff).set_index("sampling_stratum")
    population = population.copy()
    population["analysis_weight"] = np.where(
        population["existing_score"].notna(), 1.0, np.nan
    )
    for stratum, row in strata.iterrows():
        mask = (
            population["existing_score"].isna()
            & population["new_dock_score"].notna()
            & (population["sampling_stratum"] == stratum)
        )
        population.loc[mask, "analysis_weight"] = row["population"] / row[
            "scored_sample"
        ]
    hits, words, popcounts = build_hit_structures(population, cutoff, processes)
    rows: list[dict[str, Any]] = []
    for round_id in (1, 2):
        for policy_index, policy in enumerate(POLICIES):
            frame = hits[hits[membership_column(round_id, policy)]].copy()
            weights = frame["analysis_weight"].to_numpy(dtype=float)
            if not np.isfinite(weights).all():
                raise ValueError(f"missing hit-analysis weights for round {round_id} {policy}")
            indices = frame["hit_structure_index"].to_numpy(dtype=np.int64)
            rows.append(
                {
                    "round": round_id,
                    "policy": policy,
                    "observed_hit_structures": len(frame),
                    "weighted_estimated_hits": float(weights.sum()),
                    "weighted_internal_diversity": weighted_internal_diversity(
                        indices,
                        weights,
                        words,
                        popcounts,
                        pair_samples,
                        np.random.default_rng(
                            random_seed + round_id * 100 + policy_index
                        ),
                    ),
                    **weighted_family_metrics(
                        frame["typed_scaffold"], weights, "typed_scaffold"
                    ),
                    **weighted_family_metrics(
                        frame["generic_framework"], weights, "generic_framework"
                    ),
                }
            )
    for policy_index, policy in enumerate(POLICIES):
        member = hits[membership_column(1, policy)] | hits[
            membership_column(2, policy)
        ]
        frame = hits[member].copy()
        weights = frame["analysis_weight"].to_numpy(dtype=float)
        indices = frame["hit_structure_index"].to_numpy(dtype=np.int64)
        rows.append(
            {
                "round": "overall",
                "policy": policy,
                "observed_hit_structures": len(frame),
                "weighted_estimated_hits": float(weights.sum()),
                "weighted_internal_diversity": weighted_internal_diversity(
                    indices,
                    weights,
                    words,
                    popcounts,
                    pair_samples,
                    np.random.default_rng(random_seed + 1000 + policy_index),
                ),
                **weighted_family_metrics(
                    frame["typed_scaffold"], weights, "typed_scaffold"
                ),
                **weighted_family_metrics(
                    frame["generic_framework"], weights, "generic_framework"
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_hit_diversity(
    policy_diversity: pd.DataFrame, reference_path: Path
) -> pd.DataFrame:
    reference = pd.read_csv(reference_path)
    reference = reference[
        (reference["cohort_type"] == "all_hits")
        | reference["cohort_type"].str.match(r"round_[12]_hits")
    ].copy()
    reference["round"] = np.where(
        reference["cohort_type"] == "all_hits",
        "overall",
        reference["round"].astype(str),
    )
    reference["weighted_internal_diversity"] = reference[
        "sampled_pair_internal_diversity"
    ]
    reference["weighted_typed_scaffold_effective_number"] = np.exp(
        reference["typed_scaffold_entropy"]
    )
    reference["weighted_generic_framework_effective_number"] = np.exp(
        reference["generic_framework_entropy"]
    )
    rows: list[dict[str, Any]] = []
    for policy_row in policy_diversity.itertuples(index=False):
        round_key = str(policy_row.round)
        for reference_label in ("Greedy", "SVDKL-LCB"):
            ref = reference[
                (reference["run"] == reference_label)
                & (reference["round"].astype(str) == round_key)
            ].iloc[0]
            rows.append(
                {
                    "round": policy_row.round,
                    "policy": policy_row.policy,
                    "reference": reference_label,
                    "policy_internal_diversity": policy_row.weighted_internal_diversity,
                    "reference_internal_diversity": ref.weighted_internal_diversity,
                    "internal_diversity_difference": (
                        policy_row.weighted_internal_diversity
                        - ref.weighted_internal_diversity
                    ),
                    "policy_typed_effective_number": policy_row.weighted_typed_scaffold_effective_number,
                    "reference_typed_effective_number": ref.weighted_typed_scaffold_effective_number,
                    "typed_effective_ratio": (
                        policy_row.weighted_typed_scaffold_effective_number
                        / ref.weighted_typed_scaffold_effective_number
                    ),
                    "policy_generic_effective_number": policy_row.weighted_generic_framework_effective_number,
                    "reference_generic_effective_number": ref.weighted_generic_framework_effective_number,
                    "generic_effective_ratio": (
                        policy_row.weighted_generic_framework_effective_number
                        / ref.weighted_generic_framework_effective_number
                    ),
                }
            )
    return pd.DataFrame(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    figure.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def make_figures(
    manifest_scores: pd.DataFrame,
    manifest: pd.DataFrame,
    policy_estimates: pd.DataFrame,
    overall_estimates: pd.DataFrame,
    pairwise: pd.DataFrame,
    diversity: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    configure_style()
    scored = manifest.merge(
        manifest_scores[["spacehastenid", "new_dock_score"]],
        on="spacehastenid",
        how="left",
        validate="one_to_one",
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    recovery = scored.groupby(["primary_round", "sampling_reason"]).agg(
        selected=("spacehastenid", "size"),
        scored=("new_dock_score", "count"),
        hits=("new_dock_score", lambda values: int(np.sum(values <= CUTOFF))),
    ).reset_index()
    recovery["score_recovery"] = recovery["scored"] / recovery["selected"]
    recovery["hit_rate_scored"] = recovery["hits"] / recovery["scored"]
    labels = [f"R{row.primary_round}\n{row.sampling_reason.replace('_', ' ')}" for row in recovery.itertuples()]
    x = np.arange(len(recovery))
    axes[0].bar(x, 100 * recovery["score_recovery"], color="#56B4E9")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Docking score recovery (%)")
    axes[0].set_title("50k study completion")
    axes[1].bar(x, 100 * recovery["hit_rate_scored"], color="#009E73")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("New-panel hit rate (%)")
    axes[1].set_title("Observed study outcomes")
    figure.tight_layout()
    save_figure(figure, output / "01_study_completion_yield", dpi)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True)
    references = {
        1: {"Greedy": 0.60683, "LCB": 0.36388911112889033},
        2: {"Greedy": 0.57069, "LCB": 0.5936790518577787},
        "overall": {"Greedy": 0.58876, "LCB": 0.47878005970686627},
    }
    for axis, round_id in zip(axes, (1, 2, "overall"), strict=True):
        data = (
            overall_estimates
            if round_id == "overall"
            else policy_estimates[policy_estimates["round"] == round_id]
        ).set_index("policy").loc[list(POLICIES)]
        x = np.arange(3)
        rate = data["estimated_hit_rate"].to_numpy()
        low = data["ci95_low"].to_numpy()
        high = data["ci95_high"].to_numpy()
        axis.errorbar(
            x,
            100 * rate,
            yerr=[100 * (rate - low), 100 * (high - rate)],
            fmt="none",
            color="black",
            capsize=3,
        )
        axis.bar(x, 100 * rate, color=[COLORS[policy] for policy in POLICIES])
        axis.axhline(
            100 * references[round_id]["Greedy"],
            color="#0072B2",
            linestyle="--",
            label="Greedy",
        )
        axis.axhline(
            100 * references[round_id]["LCB"],
            color="#D55E00",
            linestyle=":",
            label="LCB",
        )
        axis.set_xticks(x, [POLICY_LABELS[p] for p in POLICIES], rotation=20)
        axis.set(title=f"Round {round_id}", ylabel="Estimated hit rate (%)")
    axes[0].legend(frameon=False)
    figure.suptitle("Design-based normalized-EI hit-rate estimates", y=1.02)
    figure.tight_layout()
    save_figure(figure, output / "02_policy_hit_rates", dpi)

    overall_pairs = pairwise[pairwise["round"].astype(str) == "overall"].copy()
    figure, ax = plt.subplots(figsize=(7.5, 3.8))
    labels = [
        f"{POLICY_LABELS[row.policy_right]} - {POLICY_LABELS[row.policy_left]}"
        for row in overall_pairs.itertuples()
    ]
    x = np.arange(len(overall_pairs))
    point = overall_pairs["right_minus_left_pp"].to_numpy()
    low = overall_pairs["ci95_low_pp"].to_numpy()
    high = overall_pairs["ci95_high_pp"].to_numpy()
    ax.errorbar(
        x,
        point,
        yerr=[point - low, high - point],
        fmt="o",
        color="#333333",
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Overall hit-rate difference (percentage points)")
    ax.set_title("Pairwise normalized-EI policy differences")
    figure.tight_layout()
    save_figure(figure, output / "03_pairwise_hit_rate_differences", dpi)

    overall_diversity = diversity[diversity["round"].astype(str) == "overall"].set_index(
        "policy"
    ).loc[list(POLICIES)]
    reference = {
        "internal": {"Greedy": 0.7587461059575115, "LCB": 0.7696515954038189},
        "typed": {"Greedy": math.exp(8.14743703654415), "LCB": math.exp(8.64272839567231)},
        "generic": {"Greedy": math.exp(5.686037618574511), "LCB": math.exp(6.417833717312588)},
    }
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    metric_specs = (
        ("weighted_internal_diversity", "Hit internal diversity", "internal"),
        (
            "weighted_typed_scaffold_effective_number",
            "Effective typed scaffolds",
            "typed",
        ),
        (
            "weighted_generic_framework_effective_number",
            "Effective generic frameworks",
            "generic",
        ),
    )
    for axis, (column, label, ref_key) in zip(axes, metric_specs, strict=True):
        x = np.arange(3)
        axis.bar(
            x,
            overall_diversity[column],
            color=[COLORS[policy] for policy in POLICIES],
        )
        axis.axhline(
            reference[ref_key]["Greedy"], color="#0072B2", linestyle="--", label="Greedy"
        )
        axis.axhline(
            reference[ref_key]["LCB"], color="#D55E00", linestyle=":", label="LCB"
        )
        axis.set_xticks(x, [POLICY_LABELS[p] for p in POLICIES], rotation=20)
        axis.set_ylabel(label)
    axes[0].legend(frameon=False)
    figure.suptitle("Estimated hit diversity versus completed references", y=1.02)
    figure.tight_layout()
    save_figure(figure, output / "04_hit_diversity_comparison", dpi)


def write_report(
    score_summary: dict[str, Any],
    policy_estimates: pd.DataFrame,
    overall_estimates: pd.DataFrame,
    reference_comparison: pd.DataFrame,
    pairwise: pd.DataFrame,
    diversity: pd.DataFrame,
    diversity_comparison: pd.DataFrame,
    output: Path,
) -> None:
    overall = overall_estimates.set_index("policy")
    greedy = reference_comparison[
        (reference_comparison["round"].astype(str) == "overall")
        & (reference_comparison["reference"] == "Greedy")
    ].set_index("policy")
    lcb = reference_comparison[
        (reference_comparison["round"].astype(str) == "overall")
        & (reference_comparison["reference"] == "SVDKL-LCB")
    ].set_index("policy")
    overall_diversity = diversity[
        diversity["round"].astype(str) == "overall"
    ].set_index("policy")
    lines = [
        "# Normalized-EI 50k Docking Validation",
        "",
        "## Study Completion",
        "",
        f"All 400 SLURM tasks completed and produced valid result archives. Glide returned scores for **{score_summary['scored']:,} of 50,000 compounds ({100 * score_summary['score_recovery']:.2f}%)**. The analysis distinguishes missing docking outcomes from misses.",
        "",
        "The study docked every unobserved compound that distinguished `alpha=0.05`, `0.10`, and `0.20`, then probability-sampled common-policy compounds with recorded inclusion probabilities. Policy estimates combine exact historical outcomes with finite-population, response-adjusted estimates from the new panel.",
        "",
        "## Overall Hit Rate",
        "",
        "| Policy | Estimated hit rate | 95% CI | Versus Greedy | P(noninferior within 2 pp) | Versus LCB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        row = overall.loc[policy]
        g = greedy.loc[policy]
        lcb_row = lcb.loc[policy]
        lines.append(
            f"| {POLICY_LABELS[policy]} | {100 * row.estimated_hit_rate:.2f}% | "
            f"[{100 * row.ci95_low:.2f}, {100 * row.ci95_high:.2f}]% | "
            f"{g.difference_pp:+.2f} pp | {100 * g.probability_noninferior_margin_2pp:.1f}% | "
            f"{lcb_row.difference_pp:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## Round-Specific Hit Rate",
            "",
            "| Round | Policy | Estimated hit rate | 95% CI | Greedy | LCB |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in policy_estimates.itertuples(index=False):
        g = reference_comparison[
            (reference_comparison["round"].astype(str) == str(row.round))
            & (reference_comparison["policy"] == row.policy)
            & (reference_comparison["reference"] == "Greedy")
        ].iloc[0]
        lcb_row = reference_comparison[
            (reference_comparison["round"].astype(str) == str(row.round))
            & (reference_comparison["policy"] == row.policy)
            & (reference_comparison["reference"] == "SVDKL-LCB")
        ].iloc[0]
        lines.append(
            f"| {row.round} | {POLICY_LABELS[row.policy]} | "
            f"{100 * row.estimated_hit_rate:.2f}% | "
            f"[{100 * row.ci95_low:.2f}, {100 * row.ci95_high:.2f}]% | "
            f"{g.difference_pp:+.2f} pp | {lcb_row.difference_pp:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Policy Differences",
            "",
            "| Scope | Contrast | Difference | 95% CI | P(right better) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in pairwise.itertuples(index=False):
        lines.append(
            f"| {row.round} | {POLICY_LABELS[row.policy_right]} - "
            f"{POLICY_LABELS[row.policy_left]} | {row.right_minus_left_pp:+.2f} pp | "
            f"[{row.ci95_low_pp:+.2f}, {row.ci95_high_pp:+.2f}] | "
            f"{100 * row.probability_right_superior:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Estimated Hit Diversity",
            "",
            "Weighted structural estimates use inverse response-adjusted stratum weights. Effective family counts are `exp(Shannon entropy)` and are preferable to uncorrected raw richness under unequal sampling coverage.",
            "",
            "| Policy | Internal diversity | Effective typed scaffolds | Effective generic frameworks | Observed hit structures |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for policy in POLICIES:
        row = overall_diversity.loc[policy]
        lines.append(
            f"| {POLICY_LABELS[policy]} | {row.weighted_internal_diversity:.4f} | "
            f"{row.weighted_typed_scaffold_effective_number:,.0f} | "
            f"{row.weighted_generic_framework_effective_number:,.0f} | "
            f"{int(row.observed_hit_structures):,} |"
        )
    lines.extend(
        [
            "",
            "## Comparison With Greedy And LCB Hit Diversity",
            "",
            "| Policy | Reference | Internal diversity difference | Typed effective ratio | Generic effective ratio |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in diversity_comparison[
        diversity_comparison["round"].astype(str) == "overall"
    ].itertuples(index=False):
        lines.append(
            f"| {POLICY_LABELS[row.policy]} | {row.reference} | "
            f"{row.internal_diversity_difference:+.4f} | "
            f"{row.typed_effective_ratio:.2f}x | {row.generic_effective_ratio:.2f}x |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This study evaluates frozen historical round models after sequential exclusion. It identifies the acquisition rule with the best realized potency-diversity tradeoff conditional on those models. It does not recreate how alternative round-1 labels would change model-v1 training or the later search space.",
            "",
        ]
    )
    (output / "NORMALIZED_EI_50K_RESULTS.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def calculate(args: argparse.Namespace) -> None:
    started = time.monotonic()
    study_dir = args.study_dir.resolve()
    output = study_dir / "analysis"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(exist_ok=True)
    manifest = pd.read_csv(
        study_dir / "docking_manifest_50k.csv",
        dtype={"membership_pattern": "string"},
    )
    scores, archives = parse_result_archives(study_dir)
    unknown_titles = set(scores["spacehastenid"]) - set(manifest["spacehastenid"])
    if unknown_titles:
        raise ValueError(f"Glide results contain {len(unknown_titles)} unknown titles")
    score_summary = {
        "selected": len(manifest),
        "scored": len(scores),
        "missing": len(manifest) - len(scores),
        "score_recovery": len(scores) / len(manifest),
        "new_hits": int((scores["new_dock_score"] <= args.cutoff).sum()),
        "new_hit_rate_scored": float(
            np.mean(scores["new_dock_score"] <= args.cutoff)
        ),
        "minimum_score": float(scores["new_dock_score"].min()),
        "median_score": float(scores["new_dock_score"].median()),
        "maximum_score": float(scores["new_dock_score"].max()),
    }
    scores.to_csv(output / "new_docking_scores.csv", index=False)
    archives.to_csv(output / "archive_summary.csv", index=False)
    population, manifest_scores = prepare_analysis_population(
        study_dir,
        scores,
        args.ei_db,
        args.lcb_docked_db,
        args.greedy_docked_db,
    )
    strata = stratum_statistics(population, args.cutoff)
    strata.to_csv(output / "stratum_outcomes.csv", index=False)
    policy_estimates, overall_estimates, reference_comparison, pairwise = (
        estimate_policy_rates(
            population, args.cutoff, args.simulations, args.random_seed
        )
    )
    policy_estimates.to_csv(output / "policy_hit_rate_by_round.csv", index=False)
    overall_estimates.to_csv(output / "policy_hit_rate_overall.csv", index=False)
    reference_comparison.to_csv(
        output / "hit_rate_reference_comparison.csv", index=False
    )
    pairwise.to_csv(output / "pairwise_policy_hit_rate.csv", index=False)
    diversity = estimate_hit_diversity(
        population,
        args.cutoff,
        args.pair_samples,
        args.processes,
        args.random_seed,
    )
    diversity.to_csv(output / "policy_hit_diversity.csv", index=False)
    diversity_comparison = compare_hit_diversity(diversity, args.reference_diversity)
    diversity_comparison.to_csv(
        output / "hit_diversity_reference_comparison.csv", index=False
    )
    make_figures(
        manifest_scores,
        manifest,
        policy_estimates,
        overall_estimates,
        pairwise,
        diversity,
        output,
        args.dpi,
    )
    write_report(
        score_summary,
        policy_estimates,
        overall_estimates,
        reference_comparison,
        pairwise,
        diversity,
        diversity_comparison,
        output,
    )
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "definition": {
            "cutoff": args.cutoff,
            "simulations": args.simulations,
            "pair_samples": args.pair_samples,
            "estimator": (
                "exact historical outcomes plus stratum response-adjusted finite-population "
                "estimates; shared stratum simulations preserve policy covariance"
            ),
        },
        "score_summary": score_summary,
        "validation": {
            "archives": len(archives),
            "archives_with_pose_rows": int((archives["pose_rows"] > 0).sum()),
            "manifest_rows": len(manifest),
            "scored_titles": len(scores),
            "unknown_titles": len(unknown_titles),
            "policy_cells": len(policy_estimates),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    required = {
        "new_docking_scores.csv",
        "archive_summary.csv",
        "stratum_outcomes.csv",
        "policy_hit_rate_by_round.csv",
        "policy_hit_rate_overall.csv",
        "hit_rate_reference_comparison.csv",
        "pairwise_policy_hit_rate.csv",
        "policy_hit_diversity.csv",
        "hit_diversity_reference_comparison.csv",
        "NORMALIZED_EI_50K_RESULTS.md",
        "analysis_summary.json",
    }
    stems = (
        "01_study_completion_yield",
        "02_policy_hit_rates",
        "03_pairwise_hit_rate_differences",
        "04_hit_diversity_comparison",
    )
    required.update(
        f"{stem}.{suffix}" for stem in stems for suffix in ("png", "pdf")
    )
    missing = [
        name
        for name in sorted(required)
        if not (output / name).is_file() or (output / name).stat().st_size == 0
    ]
    if missing:
        raise ValueError(f"missing analysis outputs: {missing}")
    LOGGER.info("50k docking analysis complete in %.1f s", time.monotonic() - started)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    try:
        calculate(parse_args())
    except Exception:
        LOGGER.exception("50k normalized-EI study analysis failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
