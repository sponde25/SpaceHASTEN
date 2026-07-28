# SpaceHASTEN Overnight Experiment Analysis Workflow

This workflow analyzes a completed SpaceHASTEN experiment. Reusable metrics live under
`src/spacehasten/analysis/`; `scripts/analysis/` contains thin generic entry points and distributed
workers. Campaign-specific code under the owning run's `analysis/scripts/` must remain limited to
paths, labels, orchestration, and report narrative.

This is a living implementation guide. Commands marked **implemented** exist in the repository and
should be used directly. Sections marked **conditional/reference** describe older or one-time
workflows and must not be copied into every run.

## No-Duplication Rule

Before creating a run-local script:

1. Search `src/spacehasten/analysis/` and `scripts/analysis/` for the capability.
2. If a generic command exists, use it unchanged.
3. If target-independent logic is missing, add and test it centrally.
4. Create run-local code only when the logic is genuinely specific to that run, policy, target, or
   report narrative.

Run-local scripts must not reimplement generic RDKit workers, fingerprint packing, nearest-seed
combining, UMAP transforms, diversity formulas, coverage formulas, or artifact validation.

## Current Implementation Status

| Capability | Status | Current interface |
|---|---|---|
| Transaction-consistent SQLite snapshot | Implemented | `scripts/analysis/snapshot_sqlite_database.py` |
| Standard read-only run analysis | Implemented | `scripts/analysis/analyze_run.py` |
| Training metadata, leakage, prediction drift, timing | Implemented | `scripts/analysis/analyze_run_metadata.py` |
| Fingerprint-index construction | Implemented, conditional | `scripts/analysis/build_fingerprint_indexes.py` |
| FPSim2 nearest-seed worker/combine | Implemented | `nearest_seed_similarity_chunk.py`, `combine_nearest_seed_chunks.py` |
| Landmark UMAP transform/combine | Implemented | `transform_landmark_umap_chunk.py`, `combine_landmark_umap_chunks.py` |
| Fixed UMAP model fitting | Implemented, one-time reference creation only | `fit_landmark_umap.py` |
| Generic hit diversity | Implemented, older hit-only interface | `calculate_hit_diversity_metrics.py` |
| Random-seed comparison | Implemented, older interface | `compare_hits_to_random_seeds.py` |
| Cross-run hit quality | Implemented | `compare_hit_quality.py` |
| Acquisition-history selected-manifest export | Implemented | `scripts/analysis/export_selected_manifest.py` |
| Selected-cohort structure cache | Implemented | `scripts/analysis/selected_structure_cache.py` |
| Portfolio support/reward/crowding analysis | Implemented | `scripts/analysis/analyze_portfolio_history.py` |
| Exact portfolio candidate/region enrichment | Implemented | `scripts/analysis/analyze_portfolio_enrichment.py` |
| Selected-cache diversity/descriptors/seed coverage | Implemented | `scripts/analysis/analyze_selected_cache.py` |
| Selected-cache count matching and seed resampling | Implemented | `scripts/analysis/selected_resampling.py` |
| Selected nearest-seed preparation | Implemented | `scripts/analysis/prepare_selected_nearest_seed.py` |
| Selected/centroid fixed-model UMAP preparation | Implemented | `scripts/analysis/prepare_cached_umap.py` |
| Occupied atlas-centroid fingerprint cache | Implemented | `scripts/analysis/prepare_atlas_centroid_cache.py` |
| Selected fixed-reference chemical-space figures | Implemented | `scripts/analysis/plot_selected_chemical_space.py` |
| Cross-artifact validation and manifest | Implemented | `scripts/analysis/validate_run_analysis.py` |
| Standalone HTML export | Implemented | `scripts/analysis/export_standalone_report.py` |
| Single-command orchestration | Not required | Execute the exact stage sequence below |

Update this table whenever a missing capability is implemented. Remove obsolete run-local guidance
at the same time.

`analysis.toml` is not currently implemented. Use the documented CLI arguments and preserve the
exact commands in stage receipts. Add shared TOML configuration only after multiple implemented
central commands consume the same settings; do not introduce a second configuration system solely
for one report.

## Inputs

A completed experiment requires:

- `<run>.dbsh`: canonical SpaceHASTEN SQLite database.
- `run_shared/docking/iter*/acquisition.csv` or database acquisition-history tables.
- model training metadata and, when applicable, calibration artifacts.
- workflow logs or scheduler accounting for timing context.
- Docking input, grid, and pre-docked seed CSV for reproducing future runs.

Modern databases may provide authoritative `acquisition_batches`, `acquisition_selections`,
`acquisition_outcomes`, `acquisition_region_summaries`, and `model_calibrations`. Prefer those
tables over reconstructing attempts from final mutable row state. Use acquisition CSVs as portable
mirrors and as the fallback for older runs.

Define these shell variables before running the examples:

```bash
REPO=/data/$USER/PROJECTS/SpaceHASTEN
RUN=$REPO/overnight_experiments/<run_name>
DB_SOURCE=<canonical_or_live_run_database.dbsh>
SNAPSHOT_ROOT=/data/$USER/SPACEHASTEN/<run_name>_analysis
DB=$SNAPSHOT_ROOT/final.dbsh
ANALYSIS=$SNAPSHOT_ROOT/analysis
CUTOFF=-9.7
STRICT_CUTOFF=-11.0
SEED_INDEX=<validated_seed_morgan_r2_1024.h5>
SEED_REFERENCE=<validated_seed_reference_cache.npz>
SEED_FAMILIES=<validated_seed_scaffold_categories.csv.gz>
UMAP_MODEL=<validated_fixed_landmark_umap_model.joblib>
SEED_COORDINATES=<validated_fixed_seed_coordinates.npz>
```

Set every placeholder before execution. `SEED_INDEX`, `SEED_REFERENCE`, `SEED_FAMILIES`,
`UMAP_MODEL`, and `SEED_COORDINATES` must describe the same ordered starting-seed population and
Morgan radius-2/1024-bit definition. Do not silently substitute assets from another target or seed
set.

## Environments

Local preparation, analysis, and plotting use the project environment:

```bash
source /wrk/setup_conda.sh
conda activate spacehasten-quick
python3 -c 'import sys; print(sys.executable)'
```

For the current workspace, the expected interpreter is
`/fastwrk/$USER/miniconda3/envs/spacehasten-quick/bin/python3`. Never use another user's conda
environment or paths.

SLURM fingerprint and FPSim2 work uses:

```bash
source /data/programs/oce/actoce
conda activate fpsim2-0.7.3
```

UMAP requires a shared overlay visible to compute nodes. One-time setup:

```bash
source /data/programs/oce/actoce
conda activate fpsim2-0.7.3
python -m venv --system-site-packages /data/$USER/venvs/spacehasten-umap
source /data/$USER/venvs/spacehasten-umap/bin/activate
python -m pip install 'umap-learn==0.5.7'
```

## Snapshot Before Analysis — Implemented Default

If the canonical database is on node-local `/wrk`, or may have an active writer, create a durable
transaction-consistent SQLite backup before analysis:

```bash
python $REPO/scripts/analysis/snapshot_sqlite_database.py \
  --source "$DB_SOURCE" \
  --output "$DB"
```

Wait for the final database and JSON receipt to be atomically published. Never analyze a
`.tmp`, `-journal`, `-wal`, or `-shm` file. Treat the published `/data` snapshot as immutable.
I/O-heavy SLURM jobs must stage their own copy from `/data` to
`/fastwrk/$USER/.../${SLURM_JOB_ID}` inside the selected job.

## 0. Standard Per-Run Analysis — Implemented Default

Run the read-only standard profile first. It discovers arbitrary rounds and evolving acquisition CSV schemas, uses selected-attempt and scored denominators separately, and writes canonical coverage, yield, cutoff, chemistry, atlas, acquisition, calibration, provenance, and plot artifacts:

Preserve run/acquisition discovery while reading only the immutable snapshot:

```bash
python $REPO/scripts/analysis/analyze_run.py "$RUN" \
  --database "$DB" \
  --analysis-root "$ANALYSIS/standard" \
  --hit-threshold "$CUTOFF" \
  --cutoff-range -12 -8 0.25 \
  --pair-samples 1000000 \
  --random-seed 42 \
  --dpi 600
```

The analyzer opens the snapshot in SQLite read-only/query-only mode. Unsupported prediction or
atlas metrics are emitted with explicit availability status rather than inferred or imputed.

## 0A. Selected Structure Cache — Implemented

Prepare deterministic selected-cohort inputs and a SLURM script:

```bash
python $REPO/scripts/analysis/selected_structure_cache.py prepare "$RUN" \
  --database "$DB" \
  --output-root "$ANALYSIS/structure_cache" \
  --hit-threshold "$CUTOFF" \
  --strict-threshold -11.0 \
  --task-count 80
```

Submit only the generated `structure_cache/submit.sh`. It invokes the same central CLI's `worker`
subcommand in `fpsim2-0.7.3`:

```bash
sbatch "$ANALYSIS/structure_cache/submit.sh"
```

After `squeue -j <job_id>` no longer lists the array and every task exited successfully:

```bash
python $REPO/scripts/analysis/selected_structure_cache.py combine \
  --output-root "$ANALYSIS/structure_cache"
```

The combined receipt validates exact manifest order, unique IDs, scaffold/descriptor coverage, and
packed Morgan radius-2/1024 fingerprints. Do not create a run-local structure worker or combiner.
Its `selected_manifest.csv.gz` is the canonical normalized manifest; do not export a duplicate.

## 0B. Training Metadata And Timing — Implemented

Use the normalized manifest to collect model metadata, verify that each model's selected compounds
are absent from its training CSV, summarize candidate prediction drift, parse exact round
boundaries, and query SLURM accounting:

```bash
python $REPO/scripts/analysis/analyze_run_metadata.py "$RUN" \
  --database "$DB" \
  --manifest "$ANALYSIS/structure_cache/selected_manifest.csv.gz" \
  --output-root "$ANALYSIS/run_metadata" \
  --hit-threshold "$CUTOFF" \
  --sacct \
  --dpi 600
```

If scheduler accounting has aged out, rerun without `--sacct`; the command still emits exact
log-derived stage and round timing and records scheduler accounting as unavailable. Do not copy a
run-specific log parser.

## 0C. Portfolio History — Implemented When Available

For runs with portfolio acquisition-history tables:

```bash
python $REPO/scripts/analysis/analyze_portfolio_history.py "$RUN" \
  --database "$DB" \
  --output-root "$ANALYSIS/portfolio_history" \
  --hit-threshold "$CUTOFF" \
  --strict-threshold -11.0 \
  --dpi 600
```

This writes contribution, support-state calibration, cap-binding, expected-versus-observed region,
coverage-depth, U20/overfill, hit-depth, transition, and threshold-crossing tables plus standard
figures. Non-portfolio runs should skip this stage rather than emulate missing diagnostics.

## 0D. Exact Portfolio Enrichment — Implemented When Available

For the same portfolio run, reconstruct each historical candidate pool from its persisted model,
atlas version, watermark, and attempt policy. The command refuses to report candidate enrichment
unless both the persisted candidate count and digest match exactly:

```bash
python $REPO/scripts/analysis/analyze_portfolio_enrichment.py "$RUN" \
  --database "$DB" \
  --output-root "$ANALYSIS/portfolio_enrichment" \
  --hit-threshold "$CUTOFF" \
  --strict-threshold "$STRICT_CUTOFF" \
  --dpi 600
```

This writes candidate/selected/scored/hit shares, selection and hit enrichment, share growth,
centroid origin, within-region selection order, marginal hit productivity, concentration, and exact
candidate-reconstruction receipts. Run this I/O-heavy command against a node-local copy of `DB`
when the shared snapshot is large; keep `--database` pointing to that validated read-only copy.

## 0E. Exact Selected Nearest-Seed Similarity — Implemented

Prepare deterministic query chunks from the selected manifest and the validated seed index:

```bash
python $REPO/scripts/analysis/prepare_selected_nearest_seed.py \
  --manifest "$ANALYSIS/structure_cache/selected_manifest.csv.gz" \
  --seed-index "$SEED_INDEX" \
  --output-root "$ANALYSIS/nearest_seed" \
  --task-count 96
sbatch "$ANALYSIS/nearest_seed/submit.sh"
```

After the array succeeds, read exact counts from the preparation receipt and combine:

```bash
NEAREST_TASKS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_count"])' \
  "$ANALYSIS/nearest_seed/preparation.json")
SELECTED_COUNT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_compounds"])' \
  "$ANALYSIS/nearest_seed/preparation.json")
python $REPO/scripts/analysis/combine_nearest_seed_chunks.py \
  --chunks-dir "$ANALYSIS/nearest_seed/chunks" \
  --output "$ANALYSIS/nearest_seed/nearest_seed_similarity.npz" \
  --task-count "$NEAREST_TASKS" \
  --expected-count "$SELECTED_COUNT"
```

The existing worker performs exact top-1 FPSim2 Tanimoto search with validated Morgan
radius-2/1024 seed fingerprints. Do not replace it with approximate nearest neighbors.

## 0F. Selected Diversity, Descriptors, And Seed Coverage — Implemented

Consume the structure cache and exact nearest-seed result without reparsing SMILES or regenerating
selected fingerprints:

```bash
python $REPO/scripts/analysis/analyze_selected_cache.py \
  --manifest "$ANALYSIS/structure_cache/selected_manifest.csv.gz" \
  --structure-cache "$ANALYSIS/structure_cache/structure_cache.csv.gz" \
  --fingerprints "$ANALYSIS/structure_cache/fingerprints.npz" \
  --nearest-seed "$ANALYSIS/nearest_seed/nearest_seed_similarity.npz" \
  --seed-families "$SEED_FAMILIES" \
  --portfolio-enrichment "$ANALYSIS/portfolio_enrichment/cluster_round_enrichment.csv" \
  --output-root "$ANALYSIS/selected_analysis" \
  --pair-samples 1000000 \
  --random-seed 42 \
  --dpi 600
```

Outputs cover per-round selected and hit-only cohorts, cumulative selected and hit-only cohorts,
typed/generic/persistent-atlas q0/q1/q2 and concentration, productive-family growth, descriptor
drift, exact nearest-seed summaries, and seed scaffold/framework novelty. The selected-attempt
denominator includes unresolved outcomes.

## 0G. Count-Matched Resampling — Implemented

Prepare a NumPy-only worker cache. The optional seed inputs shown here enable both per-round
selected-to-hit count matching and starting-seed-to-final-hit matching:

```bash
python $REPO/scripts/analysis/selected_resampling.py prepare \
  --manifest "$ANALYSIS/structure_cache/selected_manifest.csv.gz" \
  --structure-cache "$ANALYSIS/structure_cache/structure_cache.csv.gz" \
  --fingerprints "$ANALYSIS/structure_cache/fingerprints.npz" \
  --seed-reference-cache "$SEED_REFERENCE" \
  --seed-index "$SEED_INDEX" \
  --output-root "$ANALYSIS/resampling" \
  --task-count 40 \
  --replicates 200 \
  --pair-samples 100000 \
  --random-seed 42
sbatch "$ANALYSIS/resampling/submit.sh"
```

After every task succeeds:

```bash
python $REPO/scripts/analysis/selected_resampling.py combine \
  --output-root "$ANALYSIS/resampling" \
  --dpi 600
```

The combined tables keep empirical between-replicate intervals separate from each replicate's
pair-sampling Monte Carlo error. Do not run the old hard-coded worked-run resampling workers.

## 0H. Fixed-Reference Selected UMAP — Implemented

Transform the existing packed selected fingerprints with the validated fixed model; never refit a
run-specific primary UMAP:

```bash
python $REPO/scripts/analysis/prepare_cached_umap.py \
  --fingerprints "$ANALYSIS/structure_cache/fingerprints.npz" \
  --model "$UMAP_MODEL" \
  --output-root "$ANALYSIS/selected_umap" \
  --task-count 80 \
  --batch-size 500
sbatch "$ANALYSIS/selected_umap/submit.sh"
```

After the array succeeds:

```bash
UMAP_TASKS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_count"])' \
  "$ANALYSIS/selected_umap/preparation.json")
UMAP_COUNT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["compound_count"])' \
  "$ANALYSIS/selected_umap/preparation.json")
python $REPO/scripts/analysis/combine_landmark_umap_chunks.py \
  --chunks-dir "$ANALYSIS/selected_umap/chunks" \
  --model "$UMAP_MODEL" \
  --output-dir "$ANALYSIS/selected_umap" \
  --task-count "$UMAP_TASKS" \
  --expected-count "$UMAP_COUNT" \
  --skip-landmark-overwrite
```

`--skip-landmark-overwrite` is required because selected IDs are not the complete reference-index
namespace. The combiner validates exact row count, unique IDs, two coordinate columns, and finite
values.

## 0I. Occupied Atlas-Centroid Coordinates — Implemented

Build the small occupied-centroid cache and transform it with the same fixed model. Seed and
selected coordinates take precedence later, so these transformed coordinates fill only missing
virtual-centroid positions:

```bash
python $REPO/scripts/analysis/prepare_atlas_centroid_cache.py "$RUN" \
  --database "$DB" \
  --manifest "$ANALYSIS/structure_cache/selected_manifest.csv.gz" \
  --output "$ANALYSIS/centroid_cache/fingerprints.npz"
python $REPO/scripts/analysis/prepare_cached_umap.py \
  --fingerprints "$ANALYSIS/centroid_cache/fingerprints.npz" \
  --model "$UMAP_MODEL" \
  --output-root "$ANALYSIS/centroid_umap" \
  --task-count 16 \
  --batch-size 500
sbatch "$ANALYSIS/centroid_umap/submit.sh"
```

After the centroid array succeeds:

```bash
CENTROID_TASKS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_count"])' \
  "$ANALYSIS/centroid_umap/preparation.json")
CENTROID_COUNT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["compound_count"])' \
  "$ANALYSIS/centroid_umap/preparation.json")
python $REPO/scripts/analysis/combine_landmark_umap_chunks.py \
  --chunks-dir "$ANALYSIS/centroid_umap/chunks" \
  --model "$UMAP_MODEL" \
  --output-dir "$ANALYSIS/centroid_umap" \
  --task-count "$CENTROID_TASKS" \
  --expected-count "$CENTROID_COUNT" \
  --skip-landmark-overwrite
```

## 0J. Canonical Chemical-Space Figures — Implemented

Generate the four fixed-reference figures from the validated selected, seed, centroid, enrichment,
and nearest-seed artifacts:

```bash
python $REPO/scripts/analysis/plot_selected_chemical_space.py \
  --manifest "$ANALYSIS/structure_cache/selected_manifest.csv.gz" \
  --selected-coordinates "$ANALYSIS/selected_umap/landmark_umap_coordinates.npz" \
  --seed-coordinates "$SEED_COORDINATES" \
  --seed-reference-cache "$SEED_REFERENCE" \
  --centroid-coordinates "$ANALYSIS/centroid_umap/landmark_umap_coordinates.npz" \
  --portfolio-enrichment "$ANALYSIS/portfolio_enrichment/cluster_round_enrichment.csv" \
  --selected-nearest "$ANALYSIS/selected_analysis/selected_nearest_seed.csv.gz" \
  --output-root "$ANALYSIS/chemical_space" \
  --label "<target and acquisition label>" \
  --prior-strength 20 \
  --dpi 600
```

The outputs are seed/hit density, round-wise acquisition shift, posterior local cluster hit
enrichment, and exact nearest-seed ECDF, each as 600-dpi PNG and vector PDF. UMAP remains a fixed
reference visualization; do not interpret two-dimensional distances quantitatively.

## 0K. Report Export And Final Validation — Implemented

Write the run-specific scientific narrative at `$ANALYSIS/FULL_RUN_ANALYSIS.md` using only the
canonical tables and figures above. Narrative synthesis is intentionally run-specific; metric
formulas, chemistry workers, and validation are not. Export it unchanged:

```bash
REPORT_MD="$ANALYSIS/FULL_RUN_ANALYSIS.md"
REPORT_HTML="$ANALYSIS/FULL_RUN_ANALYSIS.html"
python $REPO/scripts/analysis/export_standalone_report.py "$REPORT_MD" "$REPORT_HTML"
```

Then revalidate receipts, hashes, exact selected identities, nearest-seed ranges, finite UMAP
coordinates, figures, and report links, and generate the final artifact manifest:

```bash
python $REPO/scripts/analysis/validate_run_analysis.py \
  --analysis-root "$ANALYSIS" \
  --snapshot-receipt "$DB.json" \
  --database "$DB" \
  --standard-root "$ANALYSIS/standard" \
  --structure-root "$ANALYSIS/structure_cache" \
  --run-metadata-root "$ANALYSIS/run_metadata" \
  --selected-root "$ANALYSIS/selected_analysis" \
  --resampling-root "$ANALYSIS/resampling" \
  --portfolio-history-root "$ANALYSIS/portfolio_history" \
  --portfolio-enrichment-root "$ANALYSIS/portfolio_enrichment" \
  --nearest-seed "$ANALYSIS/nearest_seed/nearest_seed_similarity.npz" \
  --umap "$ANALYSIS/selected_umap/landmark_umap_coordinates.npz" \
  --figures-root "$ANALYSIS" \
  --markdown "$REPORT_MD" \
  --html "$REPORT_HTML"
```

For a non-portfolio legacy run, omit the two portfolio-root arguments and use the compatible
all-docked chemical-space command in legacy Section 6 instead of Section 0J. Do not omit any stage
that was executed. The final outputs are `artifact_manifest.json` and `FINAL_VALIDATION.json`.

## Legacy And Reference Appendix

The canonical future-run workflow ends at Section 0K. The remaining sections document the older
all-docked pipeline, one-time reference creation, and cross-run studies. Do not execute them in
addition to Sections 0A-0K unless the selected-cache workflow is incompatible with a legacy run or
the user explicitly requests a new shared reference or cross-run comparison.

## 1. Fingerprint Indexes — Conditional/Legacy

Create deterministic seed-only and seed-first all-docked Morgan indexes only when a compatible
validated index does not already exist. For modern selected-cohort analysis, prefer one selected
structure/fingerprint cache and reuse an existing seed index.

```bash
python $REPO/scripts/analysis/build_fingerprint_indexes.py \
  --database "$DB" \
  --output-dir "$ANALYSIS/fingerprints" \
  --radius 2 \
  --fp-size 1024
```

Expected outputs:

```text
analysis/fingerprints/
├── seeds.smi.gz
├── all_docked.smi.gz
├── seeds_morgan_r2_1024.h5
├── all_docked_morgan_r2_1024.h5
└── fingerprint_metadata.json
```

## 2. Seed-First Sphere-Exclusion Clustering — Conditional

Reuse the all-docked FPSim2 index without modifying the production clustering module:

```bash
PYTHONPATH=$REPO/src \
python $REPO/scripts/analysis/run_clustering_with_index.py \
  "$ANALYSIS/fingerprints/all_docked.smi.gz" \
  --fp-index "$ANALYSIS/fingerprints/all_docked_morgan_r2_1024.h5" \
  --output "$ANALYSIS/clustering/clustering.csv" \
  --processes "$(nproc)"
```

The input must remain seed-first. Seeds establish baseline centroids; virtual compounds outside seed-centroid coverage become virtual-derived centroids.

## 3. Landmark Jaccard UMAP — Reference Creation Only

Fit the centroid model only when establishing a new shared reference. Normal future runs must reuse
the validated fixed model and execute only transform/combine stages.

```bash
source /data/$USER/venvs/spacehasten-umap/bin/activate
python $REPO/scripts/analysis/fit_landmark_umap.py \
  --fp-index "$ANALYSIS/fingerprints/all_docked_morgan_r2_1024.h5" \
  --clustering "$ANALYSIS/clustering/clustering.csv" \
  --output-dir "$ANALYSIS/umap" \
  --n-neighbors 30 \
  --min-dist 0.1 \
  --processes "$(nproc)"
```

Transform all compounds using independent SLURM array tasks. Each task runs:

```bash
python $REPO/scripts/analysis/transform_landmark_umap_chunk.py \
  --fp-index "$ANALYSIS/fingerprints/all_docked_morgan_r2_1024.h5" \
  --model "$ANALYSIS/umap/landmark_umap_model.joblib" \
  --chunks-dir "$ANALYSIS/umap/chunks" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --task-count 96 \
  --batch-size 5000
```

Submit as `--array=1-96%96` without a node restriction so SLURM can distribute tasks. After successful completion, combine and validate:

```bash
python $REPO/scripts/analysis/combine_landmark_umap_chunks.py \
  --chunks-dir "$ANALYSIS/umap/chunks" \
  --model "$ANALYSIS/umap/landmark_umap_model.joblib" \
  --output-dir "$ANALYSIS/umap" \
  --task-count 96 \
  --expected-count <all_docked_count>
```

## 4. Exact Nearest-Seed Similarity — Implemented Reusable Workers

Split only the virtual-docked portion of the seed-first input:

```bash
python $REPO/scripts/analysis/split_virtual_smiles.py \
  --input "$ANALYSIS/fingerprints/all_docked.smi.gz" \
  --output-dir "$ANALYSIS/nearest_seed/inputs" \
  --seed-count <seed_count> \
  --virtual-count <virtual_docked_count> \
  --task-count 96
```

Each unpinned SLURM array task runs:

```bash
python $REPO/scripts/analysis/nearest_seed_similarity_chunk.py \
  --input "$ANALYSIS/nearest_seed/inputs/virtual_${CHUNK}_of_0096.smi.gz" \
  --seed-index "$ANALYSIS/fingerprints/seeds_morgan_r2_1024.h5" \
  --output "$ANALYSIS/nearest_seed/chunks/nearest_${CHUNK}_of_0096.npz"
```

Combine and validate:

```bash
python $REPO/scripts/analysis/combine_nearest_seed_chunks.py \
  --chunks-dir "$ANALYSIS/nearest_seed/chunks" \
  --output "$ANALYSIS/nearest_seed/nearest_seed_similarity.npz" \
  --task-count 96 \
  --expected-count <virtual_docked_count>
```

## 5. Virtual-Hit Quality (Run Before Diversity)

Establish score quality and denominator-corrected hit yield before interpreting any diversity result: a strategy can appear more novel simply because it retains weaker compounds. `compare_hit_quality.py` defines a virtual hit as a scored row with `dock_iteration > 0`, `dock_score <= $CUTOFF`, and a unique `reghash`; hit rates use both all selected virtual compounds and scored virtual compounds as explicit denominators.

```bash
python $REPO/scripts/analysis/compare_hit_quality.py \
  --greedy-db "$REFERENCE_DB" --lcb-db "$RUN_DB" \
  --greedy-timing "$REFERENCE_TIMING" --lcb-timing "$RUN_TIMING" \
  --greedy-selected "$REFERENCE_SELECTED" --lcb-selected "$RUN_SELECTED" \
  --cutoff "$CUTOFF" --output-dir "$ANALYSIS/hit_quality_vs_reference" --dpi 600
```

Expected outputs are `run_summary.csv`, `iteration_summary.csv`, `cutoff_sensitivity.csv`, `top_k_summary.csv`, `overlap_summary.csv`, `statistical_comparison.csv`, `operational_efficiency.csv`, `analysis_summary.json`, and the paired PNG/PDF score figures. Confirm shared seed `reghash` and score identity before comparison. Quality results set the context for diversity: report count-matched and potency-matched results whenever yield or score distributions differ.

## 6. Chemical-Space Figures

Generate four independent 600 dpi PNG and vector PDF figures:

```bash
source /wrk/setup_conda.sh
conda activate spacehasten-quick
python $REPO/scripts/analysis/plot_chemical_space.py \
  --database "$DB" \
  --coordinates "$ANALYSIS/umap/landmark_umap_coordinates.npz" \
  --clustering "$ANALYSIS/clustering/clustering.csv" \
  --nearest-seed "$ANALYSIS/nearest_seed/nearest_seed_similarity.npz" \
  --output-dir "$ANALYSIS/figures" \
  --summary "$RUN/CHEMICAL_SPACE_SUMMARY.md" \
  --label "<target and acquisition label>" \
  --cutoff "$CUTOFF" \
  --dpi 600
```

Outputs:

1. Seed and virtual-hit density.
2. Acquisition shift relative to seeds.
3. Cluster-level virtual-hit enrichment.
4. Exact nearest-seed Tanimoto ECDF.

## 7. Diversity Metrics

For large databases, stage the database inside the selected SLURM job because `/wrk` and `/fastwrk` are node-local. Keep the canonical database on shared `/data` immutable:

```bash
WORK_DIR=/fastwrk/$USER/SpaceHASTEN/<run_name>_${SLURM_JOB_ID}
mkdir -p "$WORK_DIR"
cp "$DB" "$WORK_DIR/run.dbsh"
```

Validate the staged copy with `PRAGMA quick_check` and expected hit count, then run:

```bash
PYTHONPATH=$REPO/src \
python $REPO/scripts/analysis/calculate_hit_diversity_metrics.py \
  --database "$WORK_DIR/run.dbsh" \
  --source-label "$DB" \
  --output-dir "$WORK_DIR/results" \
  --cutoff "$CUTOFF" \
  --cluster-similarity 0.4 \
  --pair-samples 10000000 \
  --pair-batch-size 250000 \
  --random-seed 42 \
  --processes "$SLURM_CPUS_PER_TASK"
```

Copy validated result files back to `$ANALYSIS/metrics/` and remove the node-local run directory before the job exits.

The metrics include:

- Unique virtual hits using SpaceHASTEN `reghash`.
- Sampled-pair internal diversity and Monte Carlo confidence interval.
- Typed Bemis-Murcko scaffold richness and concentration.
- Generic Murcko framework richness and concentration.
- Sphere-exclusion cluster richness, concentration, entropy, and normalized entropy.

## 8. Validation Checklist

- SQLite `PRAGMA quick_check` returns `ok`.
- Fingerprint index row counts match the intended cohorts.
- Clustering assigns every indexed molecule exactly once.
- UMAP IDs are unique, sorted, finite, and match the docked database cohort.
- Nearest-seed IDs are unique and all similarities lie in `[0, 1]`.
- Scaffold family counts sum to total unique virtual hits for both definitions.
- Cluster counts sum to total unique virtual hits.
- Every PNG is visually inspected for clipping, contrast, and correct cohort labels.
- Temporary SLURM `.out` logs are removed only after final artifacts pass validation.

## 9. Matched Random-Seed Comparison

Absolute hit diversity does not establish whether an acquisition strategy narrows chemical space. Build a common seed-first atlas at Tanimoto `>= 0.4`, then compare the complete hit set with matched random samples from the unique seed population.

Build the common atlas by rerunning the analysis wrapper with:

```bash
PYTHONPATH=$REPO/src \
python $REPO/scripts/analysis/run_clustering_with_index.py \
  "$ANALYSIS/fingerprints/all_docked.smi.gz" \
  --fp-index "$ANALYSIS/fingerprints/all_docked_morgan_r2_1024.h5" \
  --output "$ANALYSIS/random_seed_comparison/atlas_t040/clustering.csv" \
  --processes "$SLURM_CPUS_PER_TASK" \
  --similarity-threshold 0.4
```

For large jobs, stage the SMILES and HDF5 index into the selected node's `/fastwrk/$USER/...` directory inside the job before running this command.

Run matched resampling against a node-local database and seed index:

```bash
PYTHONPATH=$REPO/src \
python $REPO/scripts/analysis/compare_hits_to_random_seeds.py \
  --database "$WORK_DIR/run.dbsh" \
  --source-label "$DB" \
  --seed-index "$WORK_DIR/seeds.h5" \
  --atlas-clustering "$WORK_DIR/atlas_t040.csv" \
  --hit-metrics "$WORK_DIR/hit_metrics.json" \
  --output-dir "$WORK_DIR/results" \
  --cutoff "$CUTOFF" \
  --replicates 200 \
  --pair-samples 1000000 \
  --pair-batch-size 250000 \
  --random-seed 42 \
  --processes "$SLURM_CPUS_PER_TASK"
```

Each replicate samples the same number of unique seeds as observed virtual hits, without replacement. It reports random-seed means and empirical 95% intervals for internal diversity, typed and generic scaffold richness/concentration, and common-atlas cluster richness/concentration/evenness.

Generate the comparison figure:

```bash
source /wrk/setup_conda.sh
conda activate spacehasten-quick
python $REPO/scripts/analysis/plot_random_seed_comparison.py \
  --comparison "$ANALYSIS/random_seed_comparison/random_seed_comparison.json" \
  --output-dir "$ANALYSIS/figures" \
  --dpi 600
```

## Comparison Across Acquisition Functions

Cross-run comparison is a separate stage after each individual report validates. Require identical
starting-seed `reghash` values and scores, the same docking cutoff, the same fingerprint definition,
and a common fixed atlas and UMAP reference. Keep all source databases immutable and validate staged
copies before use.

Report full natural hit sets, count-matched cohorts, and potency-matched cohorts as separate
estimands. Raw richness is sample-size confounded; use deterministic without-replacement resampling
for count matching and keep pair-sampling Monte Carlo error separate from between-subsample
variation. Treat fixed-grid UMAP density and Jensen-Shannon distance as visualization descriptors,
not chemical distances or causal evidence.
