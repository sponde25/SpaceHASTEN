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
| Configuration-driven full-report orchestration | Not yet implemented | Do not build a parallel run-local framework |

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
DB=$RUN/<run_name>.dbsh
ANALYSIS=$RUN/analysis
CUTOFF=-9.7
```

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
  --source "$DB" \
  --output /data/$USER/SPACEHASTEN/<run_name>_analysis/final.dbsh
```

Wait for the final database and JSON receipt to be atomically published. Never analyze a
`.tmp`, `-journal`, `-wal`, or `-shm` file. Treat the published `/data` snapshot as immutable.
I/O-heavy SLURM jobs must stage their own copy from `/data` to
`/fastwrk/$USER/.../${SLURM_JOB_ID}` inside the selected job.

## 0. Standard Per-Run Analysis — Implemented Default

Run the read-only standard profile first. It discovers arbitrary rounds and evolving acquisition CSV schemas, uses selected-attempt and scored denominators separately, and writes canonical coverage, yield, cutoff, chemistry, atlas, acquisition, calibration, provenance, and plot artifacts:

```bash
source /wrk/setup_conda.sh
conda activate spacehasten-quick
python $REPO/scripts/analysis/analyze_run.py "$RUN" \
  --analysis-root "$ANALYSIS/standard" \
  --hit-threshold "$CUTOFF" \
  --cutoff-range -12 -8 0.25 \
  --pair-samples 10000000 \
  --random-seed 42 \
  --dpi 600
```

The analyzer opens the source database in SQLite read-only/query-only mode. Unsupported prediction or atlas metrics are emitted with explicit availability status rather than inferred or imputed.

When the immutable snapshot is outside the run directory, preserve run/acquisition discovery while
overriding only the database:

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

## 0A. Normalized Selected Manifest — Implemented

Export one row per selected attempt. Modern runs use immutable acquisition-history tables; older
runs fall back to acquisition CSVs and round-specific outcomes:

```bash
python $REPO/scripts/analysis/export_selected_manifest.py "$RUN" \
  --database "$DB" \
  --output "$ANALYSIS/structure_cache/selected_manifest.csv.gz" \
  --hit-threshold "$CUTOFF" \
  --strict-threshold -11.0
```

The manifest retains common identity/outcome columns and available policy diagnostics. Cross-run
identity uses `reghash`, never run-local numeric IDs.

## 0B. Selected Structure Cache — Implemented

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
subcommand in `fpsim2-0.7.3`. After all tasks complete:

```bash
python $REPO/scripts/analysis/selected_structure_cache.py combine \
  --output-root "$ANALYSIS/structure_cache"
```

The combined receipt validates exact manifest order, unique IDs, scaffold/descriptor coverage, and
packed Morgan radius-2/1024 fingerprints. Do not create a run-local structure worker or combiner.

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
  --task-count 96
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
