# SpaceHASTEN — Practical Usage Guide

A hands-on walkthrough of a typical screening campaign with the rewritten
SpaceHASTEN CLI. It is written for people who already know the legacy
SpaceHASTEN workflow and just want to see how the new commands map to it.

> **Working directory convention.** Every command needs to know which campaign
> (workspace) it operates on. If you run a command **from inside the campaign's
> work directory** you don't have to specify anything. If you run it from
> anywhere else, point at the campaign with the global `-w`/`--workspace` flag,
> e.g. `spacehasten status -w /wrk/user/spacehasten/run1`. All examples below
> assume you are standing in the work directory, so the workspace argument is
> omitted.

---

## 0. Activate the environment

```bash
source /data/programs/oce/actoce
conda activate spacehasten
```

## 1. Create a workspace on fast storage

Put the campaign on fast storage (`/wrk` or `/fastwrk`) — this is where the
database, stage scratch, and logs live.

```bash
mkdir -p /wrk/user/spacehasten/run1
cd /wrk/user/spacehasten/run1
```

## 2. Initialize the campaign

Bootstrap the workspace and register the docking parameters and grid:

```bash
spacehasten init . --dock-params /path/to/glide.in --dock-grid /path/to/grid.zip
```

`init` also writes a **`props.toml`** property-filter template into the work
directory. Edit it now if you want custom filters — see
[Property & substructure filters](#property--substructure-filters).

## 3. Pick seed compounds (optional)

If you need a fresh set of seed compounds, sample them from the configured seed
collection:

```bash
spacehasten pick-seeds --n-seeds 1000000 --output seeds_1M.smi
```

## 4. Import + dock seeds and train the first model

This imports the seeds, docks them, and trains the initial ML model in one
step. `--dock-cpus` sets how many cores are used concurrently for docking:

```bash
spacehasten seed-training --smi seeds_1M.smi --dock-cpus 250
```

## 5. Run screening cycles

A screening cycle does similarity search → dock → (re)train, repeated for the
requested number of rounds. A default 2-round campaign with 1,000 simsearch
queries and up to 1,000,000 docked compounds per round, using at most 250 cores
at a time:

```bash
spacehasten screening-cycle \
    --simsearch-top-n 1000 \
    --simsearch-jobs 125 \
    --dock-top-n 1000000 \
    --dock-cpus 250 \
    --rounds 2
```

## 6. Check progress at any time

```bash
spacehasten status --actives -8.5
```

Example output:

```
Workspace:                run1
Similarity search cycles: 6
Docking iterations:       2
Model versions:           2
Total compounds:          12528819
Docked compounds:         2821361
Actives (dock_score < -8.5): 1278785
```

`--actives` takes the virtual-hit threshold, and the report tells you how many
compounds fall below it.

Generate a docking-score distribution plot (written to `<workspace>/plots/`):

```bash
spacehasten plot --kind dock-scores
```

## 7. Export results

Once you are happy with the number of virtual hits, export them. Choose a
docking-score cutoff and an output file:

```bash
# Tabular results
spacehasten export csv --cutoff -8.5 --output results.csv

# Docking poses
spacehasten export poses --cutoff -8.5 --output poses.mae
```

---

## ML-screening of an enumerated library

Once a campaign has at least the initial ML model, you can ML-score a large
enumerated library and insert the best-predicted compounds into the database
for docking. Those docked compounds can then seed further similarity searches.

A **136M diversity subset of Enamine REAL** is currently available for
enumerated ML-screening.

```bash
# 1. Score the library and insert the survivors
spacehasten library-screen \
    --library /data/work/Enamine_REAL_2026-01_136M_subset \
    --jobs 250
# -> reports e.g. {'n_inserted': 138046}

# 2. Dock the freshly inserted compounds
spacehasten dock --top-n 138046 --cpus 250
```

The inserted compounds are now in the database and can be used both as docked
data for training and as query seeds for later similarity searches.

---

## Property & substructure filters

`spacehasten init` writes a **`props.toml`** into the work directory with the
default physicochemical ranges and two (empty) SMARTS lists:

```toml
[properties]
smarts_include = []
smarts_exclude = []

[properties.mw]
min = 0.0
max = 500.0

[properties.slogp]
min = -10.0
max = 5.0

[properties.hba]
min = 0
max = 10

[properties.hbd]
min = 0
max = 5

[properties.rotbonds]
min = 0
max = 10

[properties.tpsa]
min = 0.0
max = 140.0
```

### How the filter is picked up

You do **not** need to pass a flag on every command. When resolving the active
filter, SpaceHASTEN uses this precedence:

1. An explicit `--props-toml /path/to/file.toml` (always wins).
2. The workspace's own `props.toml` (the one written by `init`), if present.
3. The property ranges already stored in the database.
4. Built-in defaults.

Because the workspace `props.toml` is step 2, **editing that file in the work
directory is enough** — the next `seed-training`, `import-seeds`,
`screening-cycle`, or `library-screen` run will use it automatically. Running
`screening-cycle` (or `seed-training`) with an effective `props.toml` also
writes the ranges into the database, so the stored filter stays in sync.

To use a filter file kept elsewhere, pass it explicitly:

```bash
spacehasten seed-training --smi seeds_1M.smi --dock-cpus 250 \
    --props-toml /path/to/custom_props.toml
```

### Excluding (or requiring) substructures

The two SMARTS lists let you filter by substructure:

- **`smarts_exclude`** — a molecule is dropped if it matches **any** listed
  pattern. Use this to remove unwanted chemotypes, e.g. carboxylic acids or
  PAINS/structural alerts.
- **`smarts_include`** — a molecule is kept only if it matches **at least one**
  listed pattern (a scaffold requirement). An empty list means no include
  constraint.

Example — filter out carboxylic acids:

```toml
[properties]
smarts_include = []
smarts_exclude = ["C(=O)[OH]"]
```

Invalid SMARTS are detected when the filter runs (on the compute node) and will
abort that filtering task with an error, so validate your patterns.

### Filters during `library-screen`

`library-screen` applies **both** filter types, using the resolution precedence
`--props-toml` > workspace `props.toml` > DB (`properties` + `smarts_filters`
tables) > built-in defaults:

- **Physicochemical range filters (MW, SLogP, HBA, HBD, rotbonds, TPSA)** are
  applied first, vectorized over the library's pre-computed descriptor columns
  (no RDKit) — this is the fast path.
- **SMARTS substructure filters (`smarts_include` / `smarts_exclude`)** are then
  applied to the property-filter survivors. Because the library's pre-computed
  columns don't encode substructure matches, each surviving compound's SMILES is
  parsed with RDKit at this point; the cost stays proportional to the (small)
  surviving fraction and parallelizes across the chunk array-jobs. When no SMARTS
  patterns are configured this RDKit step is skipped entirely.

So editing `props.toml` (or passing `--props-toml`) tightens both the property
ranges *and* the substructure filtering of the library screen exactly as it does
for seeding:

```bash
spacehasten library-screen \
    --library /data/work/Enamine_REAL_2026-01_136M_subset \
    --jobs 250 \
    --props-toml /path/to/custom_props.toml
```

For example, with `smarts_exclude = ["C(=O)[OH]"]` in `props.toml`, carboxylic
acids are removed before the compounds are scored and inserted.

---

## Logs

The master log for a campaign is written to `<workspace>/logs/`.
