# SpaceHASTEN Future-Run Analysis Prompt And Playbook

## Instructions To The Next Agent

Use this document as the initial prompt for analyzing a completed SpaceHASTEN overnight run.

Your first objective is **not** to write new analysis code. Your first objective is to discover
and validate existing run artifacts, reusable caches, generic scripts, and prior compatible
reference assets. Recompute only data that are specific to the new selected-compound cohort or are
missing, corrupt, or incompatible.

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
| Normalize selected attempts | `scripts/analysis/export_selected_manifest.py` |
| Prepare/build/combine selected structure cache | `scripts/analysis/selected_structure_cache.py` |
| Portfolio support/reward/crowding/coverage analysis | `scripts/analysis/analyze_portfolio_history.py` |
| Exact nearest-seed worker/combine | `nearest_seed_similarity_chunk.py`, `combine_nearest_seed_chunks.py` |
| Fixed UMAP transform/combine | `transform_landmark_umap_chunk.py`, `combine_landmark_umap_chunks.py` |
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
WORKSPACE=<local SpaceHASTEN workspace or run_local target>
DB=<canonical .dbsh database>
SHARED=<RUN>/run_shared
ANALYSIS=<RUN>/analysis/full_run
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

## Phase 0: Inventory Before Computation

Inspect these locations before writing or running anything:

```text
<RUN>/analysis/
<RUN>/analysis/scripts/
<RUN>/run_shared/
overnight_experiments/<related prior runs>/analysis/
scripts/analysis/
src/spacehasten/analysis/
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
  --source "$DB" \
  --output <analysis snapshot>.dbsh
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

Do not copy a prior run's model-analysis script. Start with `standard/calibration_metrics.csv`,
`standard/calibration_curve.csv`, and model `training_metadata.json` files. If a reusable metric is
missing, add it to `src/spacehasten/analysis/` or a central CLI with synthetic tests.

### Why

Training metadata, exact model schedules, log boundaries, and stage naming are run-specific.

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

### Why

For portfolio runs, execute the central history analyzer against the immutable database override.
For historical candidate-pool reconstruction, use exact model version, atlas version/watermark,
eligibility state, and prior selected IDs. If the final database cannot reconstruct these inputs,
report the capability as unavailable; do not infer or fabricate candidate metrics.

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

Export a normalized selected manifest directly when needed:

```bash
python scripts/analysis/export_selected_manifest.py "$RUN" \
  --database "$DB" \
  --output "$ANALYSIS/structure_cache/selected_manifest.csv.gz" \
  --hit-threshold "$HIT_CUTOFF"
```

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
for older runs. It writes one manifest, deterministic chunks, a SLURM script, atomic outputs, task
receipts, and a validated combined cache.

### Required Completion Receipt

`structure_cache/_SUCCESS.json`

Do not proceed to diversity analysis without it.

## Phase 6: Diversity And Resampling

Selected-cache diversity and resampling are not yet fully centralized. Use existing generic
hit-only scripts only when their input contract matches. If selected/cumulative or portfolio-atlas
metrics are required, implement the missing target-independent operation centrally; do not copy the
normalized-EI run-local scripts.

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

The query compounds and result rows are run-specific and must be computed for each new run.

### Required Outputs

- Exact one-row-per-selected-compound nearest seed ID and Tanimoto.
- Mean, median, q05, q95.
- Fractions below 0.3, 0.4, 0.5, and 0.7.
- Relationships with round, hit status, score, uncertainty, and acquisition rank.
- Seed-scaffold/framework novelty.
- Seed-centred versus virtual-centred atlas fraction.

## Phase 8: Fixed-Reference Chemical Space

### Generic Transform Scripts

- `scripts/analysis/transform_landmark_umap_chunk.py`
- `scripts/analysis/combine_landmark_umap_chunks.py`

Use the generic index-based transform/combine commands when their input contract fits. A direct
cached-fingerprint transform is a known central gap; implement it centrally before using it on
multiple new targets. Do not copy old run-local transform workers.

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
`scripts/analysis/export_standalone_report.py` for HTML. A generic schema-driven report/validator is
a known future gap, not permission to build a parallel framework inside one run.

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
- Snapshot quick check is `ok` and source checksums are unchanged.

## Expected Directory Layout

```text
<RUN>/analysis/
├── scripts/                       # run-specific orchestration only
├── full_run/
│   ├── standard/
│   ├── model_behavior/
│   ├── timing/
│   ├── acquisition_attribution/
│   ├── cluster_enrichment/
│   ├── structure_cache/
│   ├── diversity/
│   ├── seed_coverage/
│   ├── descriptors/
│   ├── chemical_space/
│   ├── sensitivity/                # optional, separately approved studies only
│   ├── FULL_RUN_ANALYSIS.md
│   ├── FULL_RUN_ANALYSIS.html
│   ├── artifact_manifest.json
│   └── FINAL_VALIDATION.json
└── interim_*                      # optional immutable interim analyses
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
[ ] Export a normalized selected manifest with export_selected_manifest.py.
[ ] Build one selected-compound cache with selected_structure_cache.py.
[ ] Run analyze_portfolio_history.py when portfolio history is available.
[ ] Extend missing target-independent model/timing metrics centrally; do not copy worked scripts.
[ ] Reuse the validated seed index and seed reference cache.
[ ] Run exact nearest-seed similarity via SLURM.
[ ] Derive diversity and descriptor metrics from caches.
[ ] Run count-matched rarefaction/random-seed context via SLURM.
[ ] Transform selected compounds with the fixed UMAP model.
[ ] Generate the canonical four chemical-space figures.
[ ] Run policy sensitivity only if separately approved and not already answered.
[ ] Build Markdown and standalone HTML reports.
[ ] Run cross-artifact validation.
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
