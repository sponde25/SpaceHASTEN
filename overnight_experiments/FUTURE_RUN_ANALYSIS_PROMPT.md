# SpaceHASTEN Future-Run Analysis Prompt And Playbook

## Instructions To The Next Agent

Use this document as the initial prompt for analyzing a completed SpaceHASTEN overnight run.

Your first objective is **not** to write or rediscover analysis code. Use the implemented-command
table and exact sequence in `ANALYSIS_WORKFLOW.md`; inventory only the new run's artifacts and prior
compatible reference assets. Recompute only data specific to the selected-compound cohort or data
that are missing, corrupt, or incompatible.

If a compatible completed run has an `ANALYSIS_REPRODUCIBILITY.md`, use it as a worked reference.
It should record canonical artifacts, checksums, script ownership, completed SLURM jobs, failed
attempts, and exact regeneration commands. Never assume that a worked run's virtual-compound
artifacts are reusable for the new cohort.

Also read `ANALYSIS_WORKFLOW.md`. It is the implementation-status source of truth: use commands
marked implemented, treat conditional/reference sections accordingly, and do not create a new
run-local implementation for a capability that already exists centrally.

## Central-First Decision Tree

Before writing any analysis code:

```text
Does src/spacehasten/analysis or scripts/analysis already provide the capability?
  Yes -> use it unchanged.
  No -> is the missing logic target/run independent?
    Yes -> implement and test it centrally, then update ANALYSIS_WORKFLOW.md.
    No -> create the smallest possible run-local orchestration or report-narrative file.
```

Stop and ask before creating a run-local RDKit worker, fingerprint packer, nearest-seed combiner,
UMAP transform, diversity formula, coverage formula, or generic validator. Those belong centrally.

## Implemented Individual-Run Commands

| Purpose | Command |
|---|---|
| Transaction-consistent snapshot | `scripts/analysis/snapshot_sqlite_database.py` |
| Standard read-only analysis | `scripts/analysis/analyze_run.py` |
| Training metadata, leakage, prediction drift, timing | `scripts/analysis/analyze_run_metadata.py` |
| Normalize selected attempts | `scripts/analysis/export_selected_manifest.py` |
| Prepare/build/combine selected structure cache | `scripts/analysis/selected_structure_cache.py` |
| Portfolio support/reward/crowding/coverage analysis | `scripts/analysis/analyze_portfolio_history.py` |
| Exact portfolio candidate/region enrichment | `scripts/analysis/analyze_portfolio_enrichment.py` |
| Selected diversity/descriptors/seed coverage | `scripts/analysis/analyze_selected_cache.py` |
| Selected and seed count-matched resampling | `scripts/analysis/selected_resampling.py` |
| Prepare selected exact nearest-seed jobs | `scripts/analysis/prepare_selected_nearest_seed.py` |
| Exact nearest-seed worker/combine | `nearest_seed_similarity_chunk.py`, `combine_nearest_seed_chunks.py` |
| Prepare selected or centroid cached UMAP jobs | `scripts/analysis/prepare_cached_umap.py` |
| Fixed UMAP transform/combine | `transform_landmark_umap_chunk.py`, `combine_landmark_umap_chunks.py` |
| Prepare occupied atlas-centroid cache | `scripts/analysis/prepare_atlas_centroid_cache.py` |
| Selected fixed-reference chemical-space figures | `scripts/analysis/plot_selected_chemical_space.py` |
| Final artifact manifest and validation | `scripts/analysis/validate_run_analysis.py` |
| Standalone HTML export | `scripts/analysis/export_standalone_report.py` |

If this table and `ANALYSIS_WORKFLOW.md` disagree, inspect the repository and update both documents
before proceeding.

There is no implemented `analysis.toml` contract yet. Use explicit central CLI arguments and record
them in receipts. Do not invent a run-local TOML parser or lock/orchestrator framework. A shared
configuration file should be added only when multiple stable central commands require it.

## Non-Negotiable Rules

1. Do not launch a new SpaceHASTEN screening run unless the user explicitly requests it.
2. Do not repeat completed docking or cap-census jobs.
3. Do not rebuild a seed fingerprint index when the starting seed digest and fingerprint
   definition match an existing index.
4. Do not refit UMAP for each run. Reuse the fixed landmark model for descriptive projections.
5. Do not regenerate fingerprints, scaffolds, or descriptors separately for every cohort.
6. Build one selected-compound structure cache, then derive all cohorts from it.
7. Use exact selected-attempt denominators from acquisition CSVs and scored outcomes from the
   database. Never silently drop failed docking attempts.
8. Open canonical databases read-only. For an active writer or final archival analysis, create a
   SQLite online-backup snapshot first.
9. Put reusable logic in `src/spacehasten/analysis/` and generic thin CLIs/workers in
   `scripts/analysis/` only when truly reusable across runs.
10. Put target-, run-, policy-, or study-specific orchestration under
    `<RUN>/analysis/scripts/`.
11. Use SLURM arrays for independent molecule chunks, nearest-neighbor searches, UMAP transforms,
    resampling replicates, docking census tasks, and other embarrassingly parallel work.
12. While SLURM jobs run, continue independent local analysis and report integration. Do not wait
    idle.
13. Validate every expensive stage before downstream use and write a completion receipt.
14. Do not make cross-run or causal claims in a single-run report.
15. Do not copy a worked run's worker or combiner into a new run. Use the central command or improve
    it centrally with tests.
16. Do not build report or validation scripts before their canonical input tables and receipts
    exist.
17. Prefer modern acquisition-history tables over final-row reconstruction when available.
18. Treat a generated `_SUCCESS.json` as reusable only after validating its input/config hashes.

## Agent Delegation Rule

- Perform integration, scientific interpretation, architecture decisions, and multi-artifact
  reasoning in the main session or with a general reasoning agent.
- Use Terra only for a narrow execution task, compilation, bounded data extraction, or minimal
  script changes with explicit files, commands, and expected outputs.
- Do not give Terra an open-ended full analysis implementation.
- Independently review any delegated output before integrating it.

Every delegated prompt must state:

- workspace `/data/$USER/PROJECTS/SpaceHASTEN`;
- local setup `source /wrk/setup_conda.sh && conda activate spacehasten-quick`;
- expected local Python `/fastwrk/$USER/miniconda3/envs/spacehasten-quick/bin/python3`;
- no other user's filesystem namespace or conda environment;
- SLURM environment and `/fastwrk/$USER/.../${SLURM_JOB_ID}` staging rules;
- exact owned files, inputs, outputs, invariants, and verification commands.

## Required Inputs

Set these conceptual variables for the new run:

```text
REPO=<repository root>
RUN=<completed overnight experiment directory>
DB_SOURCE=<canonical or active run .dbsh database>
SNAPSHOT_ROOT=/data/$USER/SPACEHASTEN/<run_name>_analysis
DB=<SNAPSHOT_ROOT>/final.dbsh
SHARED=<RUN>/run_shared
ANALYSIS=<SNAPSHOT_ROOT>/analysis
HIT_CUTOFF=<target-specific docking cutoff>
SEED=42
```

Required evidence:

| Input | Purpose |
|---|---|
| Database | Authoritative structures, scores, iterations, predictions, atlas assignments |
| `run_shared/docking/iter*/acquisition.csv` | Selected attempts, rank, model, uncertainty, alpha/lambda/cap |
| `run_shared/models/v*/training_metadata.json` | Training behavior and model provenance |
| Run log and `logs/spacehasten.log` | Stage boundaries and scheduler job IDs |
| Docking templates/grid | Required only for a new validation census |
| Seed source metadata | Establish common starting-point identity |
| Verification summary | Completed-run integrity evidence |

Modern databases may additionally contain authoritative:

```text
acquisition_batches
acquisition_selections
acquisition_outcomes
acquisition_region_summaries
model_calibrations
```

Prefer these tables for selected attempts, unresolved outcomes, acquisition diagnostics, and prior
support. Use acquisition CSVs as portable mirrors and as the fallback for older runs.

## Authoritative Run-Time Data Model

Use this section before designing a new analysis or visualization. Always confirm the deployed
database with `PRAGMA table_info`; older runs may lack extension tables.

| Source | Grain and key | Important stored fields | Semantics |
|---|---|---|---|
| `data` | One current row per `spacehastenid` | `reghash`, `smiles`, `smilesid`, `dock_score`, `dock_iteration`, `pred_score`, `pred_version`, `spacelight`, `ftrees`, `query`, `simsearch_cycle` | Mutable latest compound state. Use for current structures and final scores, not for reconstructing historical selection state. |
| `docking_param`, `docking_grid` | One stored configuration blob | Docking parameter/input and grid archives | Exact docking setup retained for reproduction; treat as opaque binary unless using the established extraction path. |
| `properties` | One row per configured property filter | property name/type and min/max limits | Screening property constraints active for the run. |
| `predictions` | One row per (`spacehastenid`, `model_version`) | `pred_score`, `epistemic_std`, `aleatoric_std`, `total_std`, `created_at` | Versioned prediction history. Join a selected attempt to its exact stored model version. |
| `models` | One row per `model_version` | `model_tar` | Serialized model artifact. Prefer `run_shared/models/v*/training_metadata.json` for human-readable training provenance. |
| `model_calibrations` | One row per `model_version` | calibration kind/source, `mean_shift`, `std_scale`, `std_floor`, fit source/split/count, split and artifact hashes, metadata JSON | Exact calibration applied to portfolio predictions. |
| `clusters` | One row per compound | `clusterid` | Legacy/materialized clustering. Do not assume it is the persistent production atlas. |
| `cluster_atlases` | One row per `atlas_id` | similarity threshold, fingerprint type/parameters, partition count | Persistent atlas definition. |
| `cluster_atlas_versions` | One row per (`atlas_id`, `version`) | last compound watermark, compound count, centroid count, metadata path | Atlas growth and historical reconstruction boundary. |
| `cluster_atlas_centroids` | One row per (`atlas_id`, `clusterid`) | centroid compound ID, created version | Stable regional centroid and seed/virtual origin after joining to `data.dock_iteration`. |
| `cluster_atlas_assignments` | One row per (`atlas_id`, `spacehastenid`) | `clusterid`, centroid similarity, assigned version | Persistent compound-to-region assignment. Interpret `clusterid` only with its `atlas_id`. |
| `acquisition_batches` | One immutable row per batch/round | strategy/status, full policy JSON and hash, attempt policy, model and atlas versions, candidate count/watermark/digest, requested/selected counts, selection digest, cap, scheduler and timestamps | Authoritative batch provenance and the boundary for exact candidate reconstruction. |
| `acquisition_selections` | One immutable row per (`batch_id`, `selection_rank`) | compound/cluster/model, raw and calibrated mean/std, `p_hit`, EI, quality, support before/after, marginal reward, crowding penalty, final utility, cluster count, cap state, contribution JSON | Exact diagnostics at selection time. This is the primary mechanism-analysis table. |
| `acquisition_outcomes` | One row per selected attempt | status (`pending`, `scored`, `unresolved`), score, source, update time | Attempt-specific outcome authority. Keep unresolved attempts in selected denominators. |
| `acquisition_region_summaries` | One row per (`batch_id`, `clusterid`) | prior hits, selected count, expected hit mass, scored/hit/unresolved counts, cap reached | Persisted selected-region support and outcomes; it does not store candidate count per region. |

### Identity, Grain, And Mutability Rules

- `spacehastenid` is efficient for joins within one database but is run-local. Use `reghash` for
  cross-run compound identity.
- `data.pred_score`, `data.pred_version`, `dock_score`, and `dock_iteration` describe current/final
  row state. Use `predictions` and acquisition-history tables for historical analyses.
- A selection is an attempt, identified in normalized artifacts as `selection_id = batch_id:rank`.
  One compound may have multiple attempts under `unscored_eligible`; structure, nearest-seed, and
  UMAP caches remain one row per unique compound.
- `clusterid` is meaningful only with the matching persistent `atlas_id` and historical
  `atlas_version`. Do not compare numeric cluster IDs across incompatible atlases.
- `acquisition_batches.candidate_digest` is the acceptance test for reconstructed candidate pools.
  If count or digest differs, candidate-share and enrichment analyses are unavailable.
- `acquisition_outcomes` preserves failed or unresolved attempts even when `data` has no positive
  docking iteration for them.
- `policy_json`, contribution JSON, calibration metadata JSON, and receipts are structured data;
  parse them rather than matching human-readable strings.

### Run-Directory Evidence Outside SQLite

| Path | Stored evidence | Use |
|---|---|---|
| `run_shared/docking/iter*/acquisition.csv` | Portable selected-attempt mirror, rank, model, prediction/uncertainty, policy and cluster diagnostics available for that strategy | Legacy fallback and independent reconciliation with database history |
| `run_shared/models/v*/training_metadata.json` | Train/validation counts, epochs, best epoch/loss, stopping, target scaling, seed, batch/worker settings and implementation-specific metadata | Training behavior and provenance |
| `run_shared/models/v*/train.csv`, `val.csv` | Exact model split rows when retained | Leakage, split identity and distribution checks |
| `run_shared/models/v*/` model/calibration artifacts | Serialized fitted model and calibration outputs with hashes when recorded | Reproduction and calibration verification, not ad hoc refitting during analysis |
| `run.log`, `run_local/logs/spacehasten.log` | Round boundaries, submissions, completions, failures and stage messages | End-to-end and stage timing; correlate job IDs with `sacct` |
| `run_status.txt`, `run_shared/verification_summary.json` | Exit status and completed-run verification | Completion gate before any scientific analysis |
| Docking templates, grid and seed inputs | Exact physical-screening inputs | Reproduction or a separately approved new docking study only |

## Generated Analysis Artifact Model

The exact commands are in `ANALYSIS_WORKFLOW.md`. The table below describes what those commands
produce and the grain at which a future hypothesis can consume each artifact.

| Stage/artifact | Grain and key fields | Information available |
|---|---|---|
| Snapshot `final.dbsh.json` | One receipt | Source/output paths, online-backup method, source/snapshot quick checks, page count/size, core table counts, score counts by iteration, elapsed time |
| `standard/round_metrics.csv` | One row per round | Selected/scored/missing/hit counts, selected and scored rates with intervals, score summaries, cumulative counts |
| `standard/budget_curve.csv` | Ordered acquisition-budget checkpoints | Cumulative selected/scored/hits and yield versus budget |
| `standard/cutoff_curve.csv` | Round and score cutoff | Hit-count/rate sensitivity to the docking threshold |
| `standard/score_distribution.csv` | Round and score ECDF point | Full observed score-distribution source for custom plots |
| `standard/family_metrics.csv` | One selected cohort per round | Internal diversity, Murcko family concentration and persistent-atlas concentration when available |
| `standard/acquisition_metrics.csv` | One row per round | Portable acquisition CSV invariants such as candidate count, model, penalty/cap/frontier settings when present |
| `standard/calibration_metrics.csv` | One row per round/model | Prediction coverage, bias, MAE/RMSE, Pearson/Spearman, hit probability, Brier/log loss/ECE, uncertainty-error correlation and interval coverage |
| `standard/calibration_curve.csv` | Round/model/probability bin | Mean predicted hit probability and observed hit fraction |
| `standard/coverage.csv` | One row per round | Explicit selected, scored and missing-outcome audit/status |
| `run_metadata/training_metadata.csv` | One row per model | Flattened retained training metadata |
| `run_metadata/training_leakage_validation.csv` | One row per used model | Selected count, overlapping training SMILES, pass/unavailable/fail status |
| `run_metadata/candidate_prediction_drift.csv` | One row per model version | Prediction count and mean score/uncertainty components |
| `run_metadata/stage_timing.csv`, `round_timing.csv` | One job stage or round | Log wall time, throughput and exact boundaries |
| `run_metadata/sacct_tasks.csv`, `sacct_summary.csv` | Scheduler task or round/stage summary | State, elapsed/CPU time, CPUs, queue wait, median/p95/max task timing |
| `structure_cache/selected_manifest.csv.gz` | One row per selected attempt | Identity/outcome plus complete modern batch, policy, model, atlas, utility, support, crowding, cap and calibration diagnostics; legacy fields are namespaced `acquisition_*` |
| `structure_cache/structure_cache.csv.gz` | One row per unique selected compound | `spacehastenid`, `reghash`, typed Murcko scaffold, generic framework, MW, cLogP, TPSA, HBD/HBA, rotatable bonds, rings and Fsp3 |
| `structure_cache/fingerprints.npz` | One row per unique selected compound | `spacehastenid`, packed Morgan radius-2/1024 `words` (`N x 16 uint64`) and `popcounts` |
| `portfolio_history/*.csv` | Selection, region, round or threshold depending on table | Utility contributions, support-stratified outcomes/calibration, expected-versus-observed regions, cap binding, productive coverage, depth, transitions and first crossings |
| `portfolio_enrichment/cluster_round_enrichment.csv` | Round and persistent region | Candidate/selected/scored/hit counts and shares, selection/hit enrichment, share growth, expected mass, centroid ID/version/source and observed hit rate |
| `portfolio_enrichment/within_cluster_selection_order.csv` | One selected attempt | Selection order, cluster order, utility diagnostics and outcome flags |
| `portfolio_enrichment/cluster_concentration.csv` | One row per round | Candidate, selected and hit region richness, HHI, effective regions and top-10 share |
| `nearest_seed/nearest_seed_similarity.npz` | One row per unique selected compound | Selected ID, exact nearest seed ID, Tanimoto and fallback threshold tier |
| `selected_analysis/diversity_metrics.csv` | Round by selected/hit/cumulative cohort | Event and unique counts, internal diversity/MC error, typed/generic/atlas q0/q1/q2, HHI, entropy, largest/top-10 fractions and richness per 10k |
| `selected_analysis/new_productive_families.csv` | One row per round | New hit typed scaffolds, generic frameworks and atlas regions |
| `selected_analysis/descriptor_values.csv.gz`, `descriptor_summary.csv` | Attempt-level descriptor values and round/cohort summaries | Distribution and drift of all cached descriptors |
| `selected_analysis/selected_nearest_seed.csv.gz` | One row per selected attempt | Full manifest and descriptor context joined to nearest seed ID/Tanimoto, suitable for rank/uncertainty/score relationships |
| `selected_analysis/seed_coverage_metrics.csv` | Round and selected/hit cohort | Nearest-seed summaries/threshold fractions, seed-novel scaffold/framework fractions and seed-centred atlas fraction |
| `resampling/resampling_replicates.csv` | Design, replicate and round | Count-matched internal/family/atlas diversity and pair-sampling MC error |
| `resampling/resampling_intervals.csv` | Design, round and metric | Empirical mean/median/95% interval across replicates, separate from MC error |
| `selected_umap/landmark_umap_coordinates.npz` | One row per unique selected compound | Fixed-reference `spacehastenid` and finite 2D coordinates; visualization only |
| `centroid_umap/landmark_umap_coordinates.npz` | One row per occupied unique centroid | Coordinates needed for persistent-region maps when seed/selected coordinates do not already cover a centroid |
| `chemical_space/cluster_statistics.csv` | One row per occupied persistent region | Selected/scored/hit counts, centroid coordinates/source, posterior local hit rate and difference from global |
| `chemical_space/*.png`, `*.pdf`, `figure_metadata.json` | Four canonical figures and receipt | Seed/hit density, round shift, local hit enrichment and nearest-seed ECDF |
| `artifact_manifest.json`, `FINAL_VALIDATION.json` | Whole analysis | Canonical artifact hashes/sizes and cross-stage count, identity, range, finiteness, figure and report checks |

Every generated stage also writes `_SUCCESS.json`, `preparation.json`, metadata JSON, or an
equivalent receipt. Validate those receipts and their input/config hashes before reusing an artifact.

### Normalized Selected-Manifest Field Groups

For modern history, `selected_manifest.csv.gz` retains these groups in one attempt-level table:

- Identity/order: `selection_id`, `round`, `rank`, `spacehastenid`, `reghash`, `smiles`.
- Outcome: `outcome_status`, `dock_score`, `outcome_source`, `data_dock_iteration`,
  `data_dock_score`, `is_scored`, `is_hit`, `is_strict_hit`.
- Batch provenance: `batch_id`, strategy/status, policy schema/hash, history attempt policy,
  batch model, atlas ID/version, candidate count/watermark/digest, requested/selected counts,
  selection digest and cap scope/limit.
- Selection-time model values: cluster/model, raw mean/epistemic std, calibrated mean/std,
  `p_hit`, expected improvement and quality.
- Portfolio mechanism: support before/after, marginal reward, crowding penalty, final utility,
  prior cluster count, cap reached and complete contribution JSON.
- Calibration provenance: calibration kind/uncertainty source, shift/scale/floor, fit source/split,
  fit row count and split/artifact hashes.

Legacy acquisition CSV columns are preserved with an `acquisition_` prefix when there is no modern
history. Therefore, consumers must inspect headers and capability status rather than assuming every
policy-specific field exists.

### Canonical Join Patterns

- Attempt and outcome: join `acquisition_batches b` to `acquisition_selections s` on `batch_id`,
  then left-join `acquisition_outcomes o` on (`batch_id`, `spacehastenid`). Keep the left side so
  unresolved attempts remain present.
- Exact prospective prediction: join `s.spacehastenid` and `s.model_version` to both columns of
  `predictions`; do not join only by compound ID or use the latest `data.pred_score`.
- Structure/final state: left-join `data d` by `spacehastenid`, but derive attempt outcome from `o`,
  not from final `d.dock_iteration`.
- Calibration: join `model_calibrations` by the selection's model version.
- Persistent region: use the `clusterid` stored on the selection for selection-time mechanism
  analysis; use (`atlas_id`, `spacehastenid`) assignments for full candidate or compound cohorts.
- Centroid origin: join (`atlas_id`, `clusterid`) to `cluster_atlas_centroids`, then join
  `centroid_spacehastenid` to `data`; `dock_iteration = 0` denotes a seed centroid.
- Candidate-region shares are not directly persisted in `acquisition_region_summaries`. Consume
  the digest-validated `portfolio_enrichment/cluster_round_enrichment.csv` rather than inventing a
  candidate denominator.

### Exact Schema Introspection

Do not guess a column or array name. Use these read-only checks before writing a new consumer:

```bash
python3 - "$DB" <<'PY'
import sqlite3, sys
with sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True) as connection:
    for table in ("data", "predictions", "model_calibrations", "acquisition_batches",
                  "acquisition_selections", "acquisition_outcomes",
                  "acquisition_region_summaries", "cluster_atlas_assignments"):
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
        print(table, columns)
PY
```

```bash
python3 - "$ANALYSIS/structure_cache/selected_manifest.csv.gz" <<'PY'
import csv, gzip, sys
with gzip.open(sys.argv[1], "rt", newline="") as handle:
    print(csv.DictReader(handle).fieldnames)
PY
```

```bash
python3 - "$ANALYSIS/structure_cache/fingerprints.npz" <<'PY'
import numpy as np, sys
with np.load(sys.argv[1], allow_pickle=False) as data:
    print({name: (data[name].shape, str(data[name].dtype)) for name in data.files})
PY
```

For CSV artifacts, inspect `csv.DictReader(...).fieldnames` or `pandas.read_csv(..., nrows=5)` before
referencing columns. For JSON, inspect top-level keys and status first. For NPZ, require
`allow_pickle=False` and validate shape, dtype, finite values and IDs.

## Hypothesis-To-Data Map

Use this map to start a new scientific question from existing data instead of rerunning chemistry,
prediction, docking, nearest-neighbor search, or UMAP.

| Hypothesis or visualization | Primary data | Required controls/limits |
|---|---|---|
| Yield or potency changed by round | `round_metrics`, `score_distribution`, `cutoff_curve`, attempt outcomes | Show selected and scored denominators separately; round differences are descriptive within one adaptive trajectory |
| Model ranking or calibration improved | `calibration_metrics`, `calibration_curve`, versioned `predictions`, `model_calibrations` | Join each attempt to its acquisition model; report coverage and unavailable rows |
| Uncertainty selected useful compounds | Selected manifest raw/calibrated std, `p_hit`, EI and observed outcomes | Distinguish association from a counterfactual acquisition effect; stratify by round/support |
| Portfolio reward or crowding changed selection | Selection quality/support/reward/crowding/final utility and `portfolio_history` tables | Use diagnostics recorded at selection time; do not reconstruct them from final `data` state |
| Candidate supply explains regional selection | `cluster_round_enrichment` candidate and selected shares/growth | Require exact candidate count and digest; compare within matching atlas/version |
| Productive regions deepened or broadened | `production_atlas_metrics`, coverage depth, transitions and threshold crossings | State hit cutoff and atlas; report q0/q1/q2 together with U20/O20 and depth |
| Chemical diversity narrowed or expanded | `diversity_metrics`, productive-family growth, resampling intervals | Separate natural cohort size from count-matched estimands; MC error is not campaign uncertainty |
| Discoveries moved away from starting seeds | `selected_nearest_seed`, `seed_coverage_metrics`, seed-novel scaffolds/frameworks | Tanimoto is quantitative; UMAP is descriptive only |
| Physicochemical profile drifted | Descriptor values/summary joined to round, hit, rank, uncertainty or region | Use complete descriptor distributions, not only means; do not add drug-likeness claims without a defined rule |
| Particular regions are locally enriched | `cluster_statistics`, enrichment tables and centroid coordinates | Use scored count as hit-rate denominator, beta-binomial shrinkage, and centroid source outlines |
| Later within-region selections have diminishing returns | `within_cluster_selection_order` and order hit-rate table | Condition on round and scored status; this is observational marginal productivity |
| Operational throughput changed | Round/stage timing and `sacct` task/summary tables | Separate queue wait, wall time, CPU time and docking-only versus end-to-end throughput |
| Two runs found the same compounds | Attempt manifests joined by `reghash` | Verify seed identity, docking score identity/cutoff and comparable outcome completeness first |
| A new custom embedding or plot is useful | Structure/descriptor/fingerprint caches plus fixed UMAP and nearest outputs | Reuse cached features; a new embedding is secondary unless a new shared reference is explicitly approved |

## Rules For New Tools And Visualizations

1. State the observation grain first: selected attempt, unique compound, scored outcome, round,
   persistent region, model version, resampling replicate, or scheduler task.
2. State the estimand and denominator before calculating: selected yield, scored yield, natural
   richness, count-matched richness, expected hit mass, or observed hit rate are not interchangeable.
3. Use `selection_id` for attempts, `spacehastenid` for within-run compound joins, `reghash` for
   cross-run identity, (`atlas_id`, `clusterid`) for persistent regions, and `model_version` for
   prediction history.
4. Join from immutable history outward. Start with batches/selections/outcomes for mechanism
   questions; use `data` only for current structure/final-state fields.
5. Preserve unresolved attempts and explicit unavailable states. Never convert missing scores to
   non-hits without labeling that estimand.
6. Consume `structure_cache.csv.gz` and `fingerprints.npz`; do not repeat RDKit parsing,
   fingerprinting, scaffold generation, nearest-seed search, or UMAP transform.
7. Keep fixed-reference UMAP coordinates for comparable visuals. Quantitative chemical distance
   comes from fingerprints/Tanimoto, not 2D spacing.
8. For a new target-independent metric, add the smallest reusable function/CLI under the central
   analysis directories with synthetic tests and a receipt. A run-local file may contain only
   paths, labels, orchestration, or narrative.
9. Write derived tables with explicit keys, units, cutoff, cohort definition, random seed and input
   hashes. Put custom outputs in a new analysis subdirectory; never mutate the snapshot or canonical
   stage outputs.
10. Validate row counts, unique keys, join coverage, finite/range constraints and source hashes
    before interpreting a new result.

## Phase 0: Inventory Before Computation

Inspect these locations before writing or running anything:

```text
<RUN>/analysis/
<RUN>/analysis/scripts/
<RUN>/run_shared/
overnight_experiments/<related prior runs>/analysis/
```

Create a reuse table with:

```text
artifact path
owner run
row count
schema
fingerprint definition
seed digest
checksum
compatible yes/no
reason
```

Safe reusable references usually include:

- Identical starting-seed source and seed fingerprint index.
- Seed scaffold/framework/atlas code cache.
- Persistent starting-seed atlas.
- Fixed landmark UMAP model.
- Generic analysis workers and combiners.

Never reuse another run's virtual-compound outcomes, nearest-seed result rows, all-docked index,
UMAP coordinates, or cohort metrics as though they belong to the new run.

## Phase 1: Freeze And Validate The Run

### Script

`scripts/analysis/snapshot_sqlite_database.py`

### Why

It uses SQLite's online backup mechanism to create a transaction-consistent analysis database and
validates `PRAGMA quick_check` and core row counts. This prevents inconsistent reads and protects
the canonical workspace database.

### How

```bash
source /wrk/setup_conda.sh
conda activate spacehasten-quick
python scripts/analysis/snapshot_sqlite_database.py \
  --source "$DB_SOURCE" \
  --output "$DB"
```

### Validate

- Snapshot quick check is `ok`.
- Expected seed count.
- Expected acquisition rounds and models.
- Selected/scored counts reconcile with acquisition CSVs.
- Atlas latest watermark covers the final database.
- No unexpected later or partial round exists.

## Phase 2: Standard Per-Run Analysis

### Script

`scripts/analysis/analyze_run.py`

### Why

This is the generic read-only analyzer. It supports arbitrary rounds and evolving acquisition CSV
schemas and produces denominator-corrected yield, score, chemistry, atlas, acquisition, and
calibration tables plus standard figures.

### How

```bash
python scripts/analysis/analyze_run.py "$RUN" \
  --database "$DB" \
  --analysis-root "$ANALYSIS/standard" \
  --hit-threshold "$HIT_CUTOFF" \
  --cutoff-range -12 -8 0.25 \
  --pair-samples 1000000 \
  --random-seed 42 \
  --dpi 600
```

### Outputs And Why

| Artifact | Meaning |
|---|---|
| `round_metrics.csv` | Selected, scored, missing, hits, rates, CIs, score summaries |
| `budget_curve.csv` | Cumulative hit discovery within the docking budget |
| `cutoff_curve.csv` | Robustness to hit threshold |
| `score_distribution.csv` | Per-round score ECDF source |
| `family_metrics.csv` | Murcko and atlas concentration/effective diversity |
| `acquisition_metrics.csv` | Candidate pool, alpha/lambda/cap/frontier and penalty summaries |
| `calibration_metrics.csv` | Prospective model errors and probability calibration |
| `calibration_curve.csv` | Reliability plot source |
| `coverage.csv` | Explicit denominator and missing-outcome audit |

Do not duplicate these calculations in a run-local script.

## Phase 3: Model Behavior And Timing

Do not copy a prior run's model-analysis or timing script. Use
`standard/calibration_metrics.csv` and `standard/calibration_curve.csv` for prospective prediction
quality, then execute `scripts/analysis/analyze_run_metadata.py` exactly as shown in Section 0B of
`ANALYSIS_WORKFLOW.md` for training metadata, leakage, prediction drift, log timing, and `sacct`.

### Why

The inputs are run-specific, but their parsing and validation are centralized.

### Required Analyses

- Training set size, best epoch, validation behavior, and training duration.
- Prospective predicted-versus-observed error by exact acquisition model.
- MAE, RMSE, bias, rank correlation, Brier score, ECE, and interval coverage.
- Uncertainty versus absolute error.
- Candidate prediction and uncertainty drift.
- Training leakage check: current-round selections absent from prior training data.
- Round and stage wall time from logs.
- Raw `sacct` task median, p95, maximum, queue wait, allocated CPUs, and CPU time.

## Phase 4: Acquisition Attribution And Cluster Enrichment

### Reusable Interfaces

- `spacehasten.analysis.acquisition`
- `spacehasten.analysis.policies`
- Production acquisition formulas in `spacehasten.core.acquisition`
- Portfolio history: `scripts/analysis/analyze_portfolio_history.py`
- Exact candidate/region enrichment: `scripts/analysis/analyze_portfolio_enrichment.py`

### Why

For portfolio runs, execute both central analyzers against the immutable database override. The
enrichment command reconstructs each candidate pool from the exact model version, atlas
version/watermark, eligibility state, and prior selected IDs, and refuses output unless its count
and SHA-256 digest equal persisted history. Do not infer or fabricate candidate metrics.

### Required Outputs

- Policy-appropriate replay validation when reconstructable.
- Acquisition contribution and support-state diagnostics when recorded.
- Rank promotion/replacement metrics only for policies whose counterfactual is defined.
- Candidate, selected, and hit shares by persistent atlas cluster.
- Selection enrichment and hit enrichment.
- Candidate-supply growth separated from selected-share growth.
- Centroid source: seed-derived versus virtual-derived.
- Within-cluster order and marginal hit productivity.
- Effective clusters, HHI, and top-cluster contribution.

## Phase 5: One-Time Structure Cache

### Why

Every selected compound should be parsed, fingerprinted, assigned a scaffold, and described only
once. All diversity, descriptor, rarefaction, near-duplicate, and chemical-space analyses must use
this cache.

### Implemented Generic Commands

Prepare the selected-cohort cache:

```bash
python scripts/analysis/selected_structure_cache.py prepare "$RUN" \
  --database "$DB" \
  --output-root "$ANALYSIS/structure_cache" \
  --hit-threshold "$HIT_CUTOFF" \
  --task-count 80
```

Submit the generated `structure_cache/submit.sh`, then combine:

```bash
python scripts/analysis/selected_structure_cache.py combine \
  --output-root "$ANALYSIS/structure_cache"
```

### Execution

The generic command prefers modern acquisition-history tables and falls back to acquisition CSVs
for older runs. It writes the canonical normalized manifest, deterministic chunks, a SLURM script,
atomic outputs, task receipts, and a validated combined cache. Do not separately export a second
manifest into the same output root.

### Required Completion Receipt

`structure_cache/_SUCCESS.json`

Do not proceed to diversity analysis without it.

## Phase 6: Diversity And Resampling

Selected-cache diversity and resampling are fully implemented centrally. Execute Sections 0F and
0G of `ANALYSIS_WORKFLOW.md` exactly with `analyze_selected_cache.py` and
`selected_resampling.py`. Do not copy or adapt the worked-run cache, rarefaction, or combine scripts.

### Why

These scripts consume cached labels and packed fingerprints. They do not repeat RDKit parsing.

### Required Cohorts

- Per-round selected.
- Per-round hit-only.
- Cumulative selected.
- Cumulative hit-only.
- Per-round selected samples count-matched to hits.
- Starting-seed samples count-matched to the final hit count.

### Metrics

- Morgan internal diversity and Monte Carlo SE.
- Typed and generic Murcko q0/q1/q2, HHI, entropy, largest and top-10 fractions.
- Persistent-atlas equivalents.
- New productive families by round.
- Richness per 10,000 compounds.
- Deterministic rarefaction intervals.

Use SLURM arrays for resampling. Workers should consume NumPy-only caches when the compute-node
environment lacks pandas.

## Phase 7: Exact Starting-Seed Coverage

### Generic Workers

- `scripts/analysis/prepare_selected_nearest_seed.py`
- `scripts/analysis/nearest_seed_similarity_chunk.py`
- `scripts/analysis/combine_nearest_seed_chunks.py`

### Why

Nearest-seed Tanimoto is the quantitative measure of movement from the starting set. UMAP is only
a visualization.

### Reuse Rule

Reuse a seed FPSim2 index only after validating:

- Same ordered seed-content digest.
- Same seed count.
- Binary Morgan radius 2, 1024 bits.
- Compatible FPSim2 metadata.

The query compounds and result rows are run-specific and must be computed for each new run. Use the
preparation command and exact combine sequence from Section 0E of `ANALYSIS_WORKFLOW.md`; do not
manually split the manifest.

### Required Outputs

- Exact one-row-per-selected-compound nearest seed ID and Tanimoto.
- Mean, median, q05, q95.
- Fractions below 0.3, 0.4, 0.5, and 0.7.
- Relationships with round, hit status, score, uncertainty, and acquisition rank.
- Seed-scaffold/framework novelty.
- Seed-centred versus virtual-centred atlas fraction.

## Phase 8: Fixed-Reference Chemical Space

### Generic Transform Scripts

- `scripts/analysis/prepare_cached_umap.py`
- `scripts/analysis/transform_landmark_umap_chunk.py`
- `scripts/analysis/combine_landmark_umap_chunks.py`
- `scripts/analysis/prepare_atlas_centroid_cache.py`
- `scripts/analysis/plot_selected_chemical_space.py`

Use `prepare_cached_umap.py` with the selected `fingerprints.npz`; its generated array invokes the
generic `transform_landmark_umap_chunk.py --fingerprints` mode. Combine with
`combine_landmark_umap_chunks.py --skip-landmark-overwrite` exactly as shown in Sections 0H and 0I
of `ANALYSIS_WORKFLOW.md`. Do not copy old run-local transform workers and never refit the primary
UMAP for an individual run.

### Required Figures

1. Seed and virtual-hit density.
2. Acquisition shift by round.
3. Local virtual-hit enrichment.
4. Exact nearest-seed ECDF.

### Canonical Figure 3 Definition

- One point per occupied persistent-atlas cluster.
- Point location is the cluster centroid's fixed-reference UMAP coordinate.
- Marker area is proportional to virtual docked count.
- Color is beta-binomial posterior hit-rate difference from the run-global virtual hit rate,
  expressed in percentage points.
- Prior mean is the global hit rate and prior strength is 20 unless explicitly changed.
- Black outline denotes virtual-derived centroid.
- Diverging blue-white-orange color map centered at zero.
- Use a robust symmetric color limit such as the 98th percentile of absolute differences.

Do not replace Figure 3 with selected count versus hit rate.

## Phase 9: Hard-Cap Replay — Optional Study

Do not run a hard-cap study by default. It is a separate scientific experiment, not a required
individual-report stage. Skip it when the campaign already used the cap being evaluated or when a
validated prior census answers the same question.

### Rules

- Validate that no-cap replay reproduces historical IDs exactly.
- Hold the historical model, alpha/lambda, atlas watermark, and actual prior trajectory fixed.
- Exclude every prior selected ID, including failed docking attempts.
- Deduplicate all replay-selected compounds lacking outcomes into one census.
- Reuse the completed run's docking grid and templates.
- Count missing outcomes against selected-denominator yield.
- Label the result as frozen one-round replay, not sequential retraining evidence.
- Do not submit a census if one already exists and validates.

## Phase 10: Report And Validation

Report synthesis is allowed to be run-specific narrative over canonical central tables. Keep it
thin: no chemistry workers, metric formulas, transforms, or generic artifact validation. Use
`scripts/analysis/export_standalone_report.py` for HTML, then run
`scripts/analysis/validate_run_analysis.py` with every executed stage as shown in Section 0K of
`ANALYSIS_WORKFLOW.md`. Do not build a parallel report or validator framework inside one run.

### Required Report Sections

1. Integrity and provenance.
2. Hit quality and cutoff sensitivity.
3. Model training and prospective calibration.
4. Timing and efficiency.
5. Acquisition attribution.
6. Candidate and cluster enrichment.
7. Selected and hit diversity.
8. Starting-seed coverage.
9. Descriptor drift.
10. Fixed-reference chemical space.
11. Optional policy sensitivity, only when separately approved.
12. Interpretation limits.

### Final Validation

Validate:

- Exact selected and hit counts.
- No missing required artifacts.
- Structure, nearest-seed, and UMAP row counts.
- Finite coordinates and metrics.
- Expected diversity and cap cell counts.
- Expected resampling replicate counts.
- Nonempty PNG and PDF figures.
- All Markdown image links resolve.
- Standalone HTML embeds every report image.
- Snapshot receipt records source/snapshot quick checks as `ok`, database size matches, and an
  optional final `--quick-check` succeeds when requested.

## Expected Directory Layout

```text
<ANALYSIS>/
├── standard/
├── run_metadata/
├── portfolio_history/
├── portfolio_enrichment/
├── structure_cache/
├── nearest_seed/
├── selected_analysis/
├── resampling/
├── selected_umap/
├── centroid_cache/
├── centroid_umap/
├── chemical_space/
├── sensitivity/                   # optional, separately approved studies only
├── FULL_RUN_ANALYSIS.md
├── FULL_RUN_ANALYSIS.html
├── artifact_manifest.json
└── FINAL_VALIDATION.json
```

## Startup Checklist For A New Run

Copy this checklist into the working task list:

```text
[ ] Read this playbook and any compatible worked-run reproducibility guide.
[ ] Read ANALYSIS_WORKFLOW.md and its current implementation-status table.
[ ] Inventory the new run and existing compatible assets.
[ ] Validate run completion and modern acquisition history or acquisition CSV fallback.
[ ] Create an immutable SQLite analysis snapshot.
[ ] Run analyze_run.py with the run path plus immutable --database override.
[ ] Build one selected-compound cache with selected_structure_cache.py.
[ ] Run analyze_run_metadata.py for training, leakage, prediction drift, timing, and sacct.
[ ] Run analyze_portfolio_history.py and analyze_portfolio_enrichment.py for a portfolio run.
[ ] Reuse the validated seed index and seed reference cache.
[ ] Prepare, submit, and combine exact selected nearest-seed similarity.
[ ] Run analyze_selected_cache.py for diversity, descriptors, and seed coverage.
[ ] Prepare, submit, and combine selected_resampling.py.
[ ] Prepare, submit, and combine selected and occupied-centroid fixed-model UMAP transforms.
[ ] Generate the canonical four figures with plot_selected_chemical_space.py.
[ ] Run policy sensitivity only if separately approved and not already answered.
[ ] Write Markdown and export standalone HTML with export_standalone_report.py.
[ ] Run validate_run_analysis.py and inspect FINAL_VALIDATION.json.
[ ] Write a run-specific ANALYSIS_REPRODUCIBILITY.md.
[ ] Stop before cross-run comparisons unless separately requested.
```

## What Not To Claim

- Do not claim one round caused another round's improvement from a single adaptive trajectory.
- Do not treat molecules or close analogs as independent experimental replicates.
- Do not report Monte Carlo pair-sampling error as campaign uncertainty.
- Do not interpret UMAP distance quantitatively.
- Do not treat frozen cap replay as a retrained capped trajectory.
- Do not reuse another run's virtual-compound metrics as evidence for the current run.
- Do not combine starting-seed hits with virtual-selection hits without stating both counts.
