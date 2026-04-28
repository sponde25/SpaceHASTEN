# SpaceHASTEN Codebase Analysis & Rewrite Plan

> Source: in-tree analysis of SpaceHASTEN v0.10 (`/data/ajain/PROJECTS/SpaceHASTEN`). Total ~4.3k LOC across 21 Python files (flat layout).

---

## 1. Project Overview

SpaceHASTEN is an **active-learning virtual screening pipeline** for ultra-large BioSolveIT chemical spaces (REAL, FreedomSpace, etc.; tens of billions of compounds). Instead of docking the whole space, it iterates:

1. Dock seed molecules (Schrödinger Glide).
2. Train a Chemprop D-MPNN regressor on the docking scores.
3. Use top-scoring docked compounds as queries for similarity search (BioSolveIT **SpaceLight** + **FTrees**) against the .space file.
4. Filter retrieved compounds by RDKit physchem properties; predict their docking score with the model.
5. Pick top predicted compounds (greedy or clustering acquisition); dock them; loop back to 2.
6. Optionally cluster the database (RDKit + FPSim2 sphere-exclusion).
7. Export top hits as CSV or 3D poses (.maegz).

External commercial tools: **Schrödinger Suite** (LigPrep, Phase, Glide), **SpaceLight 2.0**, **FTrees 7.0**. Free: **chemprop 2.1.2** (+ Lightning), **FPSim2 0.7.3**, **RDKit**, **SLURM** (or SGE alpha). Reference: J. Chem. Inf. Model. 2025, 65, 1, 125-132.

---

## 2. Current Architecture

### Module map

| File | Role | LOC |
|---|---|---|
| [spacehasten.py](spacehasten.py) | Entrypoint; instantiates GUI; routes to CLI vs Tk mainloop | 39 |
| [spacehasten](spacehasten) | Bash wrapper that activates conda + `python3 spacehasten.py` | ~45 |
| [cmdline.py](cmdline.py) | `argparse` definitions for the CLI | 67 |
| [gui.py](gui.py) | Tkinter GUI **+ orchestration logic for both GUI and CLI** | 1059 |
| [cfg.py](cfg.py) | `SpaceHASTENConfiguration`: ini parser, defaults, scheduler keyword tables | 339 |
| [functions.py](functions.py) | Shared helpers (sqlite getters, RDKit hash, NFS check) | 220 |
| [scheduler_functions.py](scheduler_functions.py) | Bash job-script generators + completion polling | 263 |
| [importseeds_functions.py](importseeds_functions.py) | Build .dbsh, dock seeds, train first model, cluster | 142 |
| [simsearch_functions.py](simsearch_functions.py) | SpaceLight+FTrees jobs, property filter, predict scores, insert into db | 304 |
| [docking_functions.py](docking_functions.py) | Pick top compounds, write LigPrep+Glide input, submit, ingest scores | 270 |
| [training_functions.py](training_functions.py) | Train Chemprop, store model tar as BLOB in sqlite | 82 |
| [prediction_functions.py](prediction_functions.py) | Predict pred_score for all undocked rows | 199 |
| [cluster_functions.py](cluster_functions.py) | Submit clustering job, store clusters in sqlite | 83 |
| [archive_functions.py](archive_functions.py) | tar+pigz archive / restore / clean | 77 |
| [export_functions.py](export_functions.py) | CSV export + pose export driver | 87 |
| [export_poses.py](export_poses.py) | Schrödinger-API script invoked via `$SCHRODINGER/run` | 64 |
| [control.py](control.py) | RDKit property filter run on remote compute node | 78 |
| [chunkpredict.py](chunkpredict.py) | Older chemprop-CLI prediction helper (legacy) | 72 |
| [model_runner_train.py](model_runner_train.py) | Chemprop 2.x training via Python API | 211 |
| [model_runner_predict.py](model_runner_predict.py) | Chemprop 2.x prediction via Python API | 154 |
| [install_spacehasten.py](install_spacehasten.py) | Interactive installer; writes `spacehasten.ini`, copies files | 188 |
| [verify_spacehasten.py](verify_spacehasten.py) | Smoke test of full pipeline | 333 |
| [sec_clustering.sh](sec_clustering.sh) | Self-extracting bash that emits FPSim2/RDKit clustering Python | 80+ |

### Dependency graph (current)

```
spacehasten.py
   └─> gui.SpaceHASTENGUI  (always constructed, even for CLI)
        ├─> cfg.SpaceHASTENConfiguration  (loads ini, validates installs, builds scheduler keyword tables)
        ├─> cmdline.parse_cmdline()
        ├─> importseeds_functions.import_seeds
        │      ├─> functions (mol2hash, properties)
        │      ├─> docking_functions.dock(importing_seeds=True)
        │      │      └─> scheduler_functions.write_dock_scheduler / wait_until_jobs_done
        │      ├─> training_functions.train_new_model
        │      │      └─> scheduler_functions.write_train_scheduler
        │      └─> cluster_functions.cluster_dbsh
        │             └─> scheduler_functions.write_cluster_scheduler
        ├─> simsearch_functions.simsearch
        │      ├─> scheduler_functions.write_search_scheduler  (SpaceLight + FTrees)
        │      ├─> process_sim_results
        │      │      └─> scheduler_functions.write_control_scheduler
        │      │             └─> control.py (RDKit prop filter on node) + model_runner_predict.py
        │      └─> cluster_functions.cluster_dbsh
        ├─> docking_functions.dock  (greedy or clustering acquisition)
        ├─> prediction_functions.update_predicted_scores
        │      └─> scheduler_functions.write_predict_scheduler  → model_runner_predict.py
        ├─> cluster_functions.cluster_dbsh
        ├─> export_functions.export_results / export_poses
        │      └─> export_poses.py  (run under $SCHRODINGER/run)
        └─> archive_functions.archive / restore / clean
```

### Entry points

There is effectively **one** entrypoint, [spacehasten.py](spacehasten.py), wrapped by the bash launcher [spacehasten](spacehasten). It always builds a `Tk` root window — even in CLI mode `gui.SpaceHASTENGUI()` is instantiated, then `app.run_cmdline()` is called and the mainloop is skipped. There is no headless path.

---

## 3. Workflow Stages (end-to-end)

### Stage 0 — Install / verify
- [install_spacehasten.py](install_spacehasten.py) interactively asks for paths (BioSolveIT execs, conda envs, scratch, partition) and writes a `spacehasten.ini` plus copies all .py files into the install dir.
- [verify_spacehasten.py](verify_spacehasten.py) runs an end-to-end smoke test against bundled `examples.smi` / `test_dock.in` / `grid-test_dock.zip`.

### Stage 1 — Pick seeds (optional, GUI-only)
- `gui_pickseeds()` in [gui.py](gui.py): Pandas-loads a bzipped Enamine REAL or FreedomSpace cxsmiles file, samples N rows, parallelises `functions.cxsmi2smi` over `mp.Pool`, writes a `.smi`. Pure local, no scheduler.

### Stage 2 — Import seeds & build database
Driver: [`importseeds_functions.import_seeds`](importseeds_functions.py).

- Inputs: `.smi` (undocked) or `.csv` (already-docked Glide output), Glide `.in`, Glide grid `.zip`, property ranges.
- Creates `<name>.dbsh` (SQLite). Tables:
  - `data(spacehastenid PK, reghash, smiles, smilesid, dock_score, pred_score, spacelight, ftrees, query, dock_iteration, pred_version, simsearch_cycle)`
  - `docking_param(dock_param BLOB)` — original Glide .in
  - `docking_grid(dock_grid BLOB)` — original Glide grid .zip
  - `models(model_version UNIQUE, model_tar BLOB)` — gzipped chemprop models stored **inside** the sqlite database
  - `properties(...)` — physchem ranges
  - `clusters(spacehastenid PK, clusterid)` — added by clustering
- Hashes each molecule with RDKit `RegistrationHash` (TAUTOMER_HASH layer) for deduplication.
- If `.smi`, immediately fires Stage 4 (dock seeds, iter=0).
- Then fires Stage 3 (train model v1) and Stage 7 (cluster).

### Stage 3 — Train model (Chemprop)
Driver: [`training_functions.train_new_model`](training_functions.py).

- Pulls `(smiles, dock_score)` rows where `dock_score < TRAIN_DOCKING_CUTOFF` (10.0).
- Writes CSV; runs [model_runner_train.py](model_runner_train.py) under SLURM (GPU-exclusive node).
- Output: `model_<name>_ver<N>/model_0/pytorch_model.bin` is tarred + gzipped + inserted as BLOB into `models` table; on-disk model dir is deleted.
- 90/10 train/val split (deterministic last-10%); standard scaler on targets.

### Stage 4 — Docking (Schrödinger LigPrep + Phase + Glide)
Driver: [`docking_functions.dock`](docking_functions.py).

- Pick top compounds by `pred_score` (greedy) **or** lowest pred-score per cluster (clustering acquisition).
- Shuffle, chunk into `DOCKING_CHUNK` (1000) per task; chunks scale down if `chunks < cpus`.
- Per chunk write: `<base>.smi`, `<base>.inp` (Phase/LigPrep recipe), `glide_<base>.in` (template stripped of LIGANDFILE/GRIDFILE), shared `glide_grid.zip`.
- Submitted as a SLURM array (one task per chunk). Each task: copies inputs to `$SCRATCH/$USER/<jobname>`, runs `$SCHRODINGER/pipeline -prog phase_db`, exports to .mae, runs `$SCHRODINGER/glide -new`, tars results, copies back.
- `process_docking_results` untars all `.tar.gz` in parallel, reads `glide_*.csv`, takes min docking_score per title, UPDATEs `data.dock_score, data.dock_iteration`.

### Stage 5 — Similarity search (SpaceLight + FTrees)
Driver: [`simsearch_functions.simsearch`](simsearch_functions.py).

- Pick `args.top` query SMILES (greedy by dock_score or pred_score; or per-cluster).
- Mark them with `query = cycle_number`.
- SLURM array job: each task runs both SpaceLight and FTrees against the .space file with one query. Results written as `spacelightresult_*.csv`, `ftreesresult_*.csv`.
- `process_sim_results`:
  - Aggregates max similarity per SMILES per method.
  - Splits raw mols into chunks; **dispatches a second SLURM array** (`write_control_scheduler`) that on each node:
    1. Decompresses chunk
    2. Runs [control.py](control.py) for RDKit physchem filtering (writes `propoutput_*.csv`)
    3. Runs [model_runner_predict.py](model_runner_predict.py) to score survivors with the latest chemprop model
  - Reads back `predicted_propoutput_*.csv`, deduplicates by reghash, INSERTs new rows into `data` with `pred_score`, `simsearch_cycle`, `spacelight`, `ftrees`.
- Finally re-clusters the database.

### Stage 6 — Predict scores for entire DB
Driver: [`prediction_functions.update_predicted_scores`](prediction_functions.py).

- Pulls all rows where `dock_score IS NULL`. Splits into per-CPU CSVs. Submits a SLURM array running `model_runner_predict.py`. Reads back, UPDATEs `pred_score`, `pred_version`.
- Note: there is **also** an in-process variant `predict_dock` that uses `chemprop_predict` CLI via `mp.Pool` — appears legacy/unused for the main loop.

### Stage 7 — Clustering
Driver: [`cluster_functions.cluster_dbsh`](cluster_functions.py).

- Dumps SMILES from `data` to a gzipped `.smi`.
- Submits 1 SLURM job (large CPU + exclusive) running `sec_clustering.sh` (or override). The bash script itself writes a Python file that uses RDKit `LeaderPicker` + FPSim2 for sphere-exclusion clustering at distance 0.7 (Tanimoto similarity 0.3).
- Result CSV written, ingested by `process_cluster_results` into `clusters` table.

### Stage 8 — "Virtual screening" macro (greedy)
`gui_thread_virtual_screening` in [gui.py](gui.py) packages: simsearch (docked) → simsearch (predicted) → simsearch (predicted) → dock. If continuing, it first re-trains a model. This is the canonical iteration loop.

### Stage 9 — Export
- CSV: `export_results` joins `data ⋈ clusters` filtered by `dock_score <= cutoff`.
- Poses: `export_poses` decompresses every iteration's `results-*.tar.gz`, and runs [export_poses.py](export_poses.py) under `$SCHRODINGER/run` per `*_pv.maegz`. Concatenates `.mae` shards, gzips with `pigz`.

### Stage 10 — Archive / restore / clean
- Archive: `pigz` the `.dbsh`, symlink all `SIMSEARCH_*` and `DOCKING_*` dirs, `tar -ch` to follow links into a single `.archived-spacehasten`.
- Restore: tar -xf, gunzip dbsh, mv subdirs back into `~/SPACEHASTEN/`.
- Clean: `rm -fr` matched dirs and the `.dbsh`.

---

## 4. Configuration System

[`SpaceHASTENConfiguration`](cfg.py) is a god-object holding ~80 class-level constants and instance fields. It:

- Locates itself via `os.path.abspath(sys.argv[0])` (fragile if symlinked).
- Reads a single `spacehasten.ini` in the install dir with sections `[General] [Paths] [Slurm] [SGE] [Properties]`.
- Validates installs by `shutil.which("chemprop")`, `os.path.exists($SCHRODINGER/run)`, etc., and `sys.exit`s on failure (so importing `cfg` requires a fully-installed system — bad for unit tests).
- Builds **scheduler keyword strings** (e.g. `SCHEDULER_JOBNAME = "#SBATCH -J"`) that get string-concatenated into bash scripts.

Per-job property ranges are persisted in the SQLite `properties` table; everything else lives in the ini.

---

## 5. Scheduler / SLURM Layer

[scheduler_functions.py](scheduler_functions.py) contains six near-duplicate functions: `write_search_scheduler`, `write_dock_scheduler`, `write_predict_scheduler`, `write_train_scheduler`, `write_control_scheduler`, `write_cluster_scheduler`. Each:

1. Opens `submit_<job>.sh`.
2. Writes a hard-coded bash header using `args.c.SCHEDULER_*` strings.
3. Hard-codes the body (cd to scratch, copy inputs, run command, tar results).
4. `close_job` writes `touch jobdone-<name>-CPU<task_id>` as the completion sentinel.

Submission is `os.system("sbatch submit_*.sh")` from inside the working directory (no return code check). **Note**: `docking_functions.dock` calls `sbatch` directly even though `args.c.SCHEDULER_SUBMIT` exists — so if SCHEDULER=SGE, docking is currently broken.

Polling uses `wait_until_jobs_done`: every 5 s, `glob` the work dir for `jobdone-*-CPU*` files; tqdm bar updates as count grows. There is **no** SLURM polling, dependency chains, or failure detection — if a node dies the loop hangs forever.

The conda environment is re-activated inside every job script via `args.c.PREPARE_ANACONDA + ACTIVATE_*`.

---

## 6. File-system Layout During a Run

Working artefacts live under **two separate roots**, which is awkward:

- `<cwd>/<name>.dbsh` — sqlite DB (must be on local fast disk; NFS warning at startup).
- `$HOME/SPACEHASTEN/SIMSEARCH_<name>_cycle<N>/` — search work
  - `queries_<name>.smi`, `submit_queries_<name>.sh`, `spacelightresult_*.csv`, `ftreesresult_*.csv`
  - `CONTROL/control_<name>_cpu<i>.smi.gz`, `control.param`, `predicted_propoutput_*.csv`, model dir
  - `PREDICT/predict_<name>_cpu<i>.csv`, `predicted_predict_*.csv`, model dir
- `$HOME/SPACEHASTEN/DOCKING_<name>_iter<N>/` — docking inputs + `results-*.tar.gz`
- `$HOME/SPACEHASTEN/TRAIN_<name>_ver<N>/` — training scratch (deleted after model is BLOB'd)
- `$HOME/SPACEHASTEN/CLUSTERING_<name>_tmp/` — clustering scratch
- `$SCRATCH/$USER/<job>_cpu<i>/` — per-task per-node fast scratch (created and deleted by job).

Job-script files, scheduler `jobdone-*` sentinels, and tarballs of results are co-located in the work dir, mixing intermediate plumbing with deliverables.

Naming conventions are stringly-typed (`"DOCKING_"+name+"_iter"+str(n)`) and re-derived in many places — both producers and consumers compute their own path. There is no path/manifest module.

---

## 7. GUI Layer

[gui.py](gui.py) is **1059 LOC** and is the largest single file. It violates separation of concerns badly:

- It owns workflow orchestration: `gui_thread_virtual_screening`, `gui_thread_pickseeds`, `gui_thread_train`, `gui_thread_docking` are *the* drivers; `run_cmdline` (line ~140) duplicates argument-marshalling for headless runs.
- `cfg.SpaceHASTENConfiguration` is held on `self.c` and threaded into every job via `SimpleNamespace` "args" objects (~25 fields — see `gui_virtual_screening`).
- IPC is a `queue.Queue` of stringly-typed messages (`"Percent:50"`, `"UpdateModel:3"`, `"DoneTaskmenu"`) consumed by a 200 ms `after()` poll loop in `check_queue`.
- Threads are vanilla `threading.Thread` — no cancellation, no exception propagation.
- GUI uses raw Tk widgets with hardcoded grid coordinates, hex colors, and Pillow PNG logo. No theme. Window is fixed `541x400`, non-resizable.
- Five frames (`frame_main`, `frame_working`, `frame_task_menu`, `frame_export_menu`, `frame_props_menu`) are toggled with `grid()`/`grid_forget()`.
- Many file dialogs each duplicating filetype filters, default-dir logic, and cancel handling.

Since CLI runs through the same class (incl. NFS check, image load, Tk root creation), modernizing the GUI is blocked until the orchestration logic is extracted.

---

## 8. CLI Layer

[cmdline.py](cmdline.py) defines a **single flat argparse** with `--action {importsmiles, screen, exportcsv, cluster}` and ~14 optional flags. There are no subcommands, no validation per action, no `--help` per stage, no resume/status/check action, no JSON-config alternative.

`gui.run_cmdline` builds a `SimpleNamespace` of args and hand-dispatches to the relevant `*_functions` module, with significant copy-paste from the GUI handlers.

---

## 9. Chemprop / ML Integration

Two layers exist:

- [model_runner_train.py](model_runner_train.py) and [model_runner_predict.py](model_runner_predict.py) — modern, standalone, use the chemprop 2.x Python API + Lightning, run on SLURM nodes inside the `chemprop-2.1.2` conda env. These are clean.
- [chunkpredict.py](chunkpredict.py) and `prediction_functions.predict_dock` — older path that shells out to the `chemprop predict` and `chemprop_predict` CLIs. Looks legacy/dead in the main loop.

Models are persisted as gzipped tarballs **inside** the SQLite database (`models.model_tar` BLOB) and re-extracted to the work dir on every cycle. This makes the dbsh self-contained but expensive to load (multi-GB).

The Python API path saves only `pytorch_model.bin`; the `model_0/` directory layout is required by the predict script.

---

## 10. Pain Points & Code Smells

1. **GUI owns orchestration.** [gui.py](gui.py) is the de-facto controller; CLI runs through Tk root. Fixing the GUI requires extracting workflow logic first.
2. **No abstraction over the scheduler.** [scheduler_functions.py](scheduler_functions.py) writes raw bash strings; submission is `os.system("sbatch ...")` — silently ignores exit codes; SGE branch in cfg.py has typos (`elif self.SCHEDULER != "SGE"` should likely be `==`); `docking_functions.dock` hardcodes `sbatch`.
3. **Job completion is detected by sentinel files.** No polling of SLURM, no failure detection, no resume. A failed task = infinite hang.
4. **Stringly-typed everything.** Args are `SimpleNamespace` blobs constructed in 6+ places (each path different); messages between threads are `"Percent:NN"` strings; SQL is built by string concat (e.g. `"... LIMIT "+str(args.top)` in [docking_functions.py](docking_functions.py:113) and elsewhere — works because inputs are ints, but unsafe pattern).
5. **God-object cfg.** [cfg.SpaceHASTENConfiguration](cfg.py:35) loads ini, validates installs, builds scheduler scripts, all on import. Cannot be unit-tested without a full installation.
6. **Two work-dir roots.** `cwd` for the dbsh, `$HOME/SPACEHASTEN/` for everything else. Confusing for users; archive/clean depend on glob patterns matching the job name.
7. **Models stored as BLOBs.** Multi-GB `models.model_tar` makes the SQLite file enormous and writes slow. A model registry on disk would be cheaper.
8. **Massive copy-paste in scheduler-script writers** — six functions that share 80% of their bodies.
9. **Hardcoded `sbatch` calls** scattered across [docking_functions.py](docking_functions.py:151), [training_functions.py](training_functions.py:62), [simsearch_functions.py](simsearch_functions.py:148) — bypasses the `args.c.SCHEDULER_SUBMIT` abstraction.
10. **`os.system` everywhere** for tar/gunzip/pigz/cp/mv/rm/sbatch — no error handling, shell-injection risk if names contain whitespace (low risk in practice but a smell), and no progress hook.
11. **No typing, sparse docstrings, no tests.** Only `verify_spacehasten.py` smoke test.
12. **Naming via string ops.** `dbname.split("/")[-1].split(".")[0]` repeated dozens of times instead of `Path.stem`.
13. **Threads + Tk + multiprocessing mixed.** `mp.Pool` started inside a worker thread inside Tk (e.g. `gui_thread_pickseeds`).
14. **NFS check via subprocess to `stat -f`** ([functions.py](functions.py:115)) instead of `os.statvfs`.
15. **`functions.predict_dock` is dead code** — written for chemprop 1.x CLI workflow.
16. **`control.py` and `chunkpredict.py` are de-facto remote scripts** (called as `python3 control.py args` on compute nodes), but live in the same flat namespace as importable modules — easy to confuse.
17. **`get_latest_*` functions open a fresh sqlite connection on every call** ([functions.py](functions.py:131)); inside hot loops they re-execute on every cycle of a multi-thousand-task array.
18. **Schedulers and version bumps fight each other.** Glide-2026-1 changes triggered direct edits across `scheduler_functions.py`, `cfg.py`, `install_spacehasten.py`, `verify_spacehasten.py`. No abstraction over Schrödinger versions.
19. **Hardcoded paths in defaults**: `/data/programs/BiosolveIT/...`, `/wrk` — Orion-specific. The installer overrides them but the cfg defaults are surprising.
20. **`spacehasten` bash launcher activates conda by `grep`-ing the .ini** — fragile.

---

## 11. Rewrite Plan

Goal: **CLI-first**, modern, decoupled GUI, streamlined SLURM/file ops. Preserve the science (acquisition strategies, sqlite schema, BioSolveIT integrations) while replacing the plumbing.

### 11.1 Proposed package layout

```
spacehasten/
├── pyproject.toml                 # PEP 621; entrypoint: spacehasten = spacehasten.cli.main:app
├── src/spacehasten/
│   ├── __init__.py
│   ├── cli/
│   │   ├── main.py                # argparse app with subparsers (one per stage)
│   │   ├── _common.py             # shared options
│   │   └── progress.py            # Rich progress bars
│   ├── config/
│   │   ├── settings.py            # Pydantic Settings (.ini + env + flags merge)
│   │   └── schema.py              # typed dataclasses
│   ├── core/                      # pure logic, no I/O side-effects above sqlite
│   │   ├── db.py                  # sqlite ORM (sqlmodel/sqlalchemy) + migrations
│   │   ├── molecules.py           # RDKit hashing, dedup
│   │   ├── acquisition.py         # greedy / clustering / future BO
│   │   └── splitting.py           # chunking helpers
│   ├── stages/                    # one file per pipeline stage
│   │   ├── seeds.py
│   │   ├── docking.py
│   │   ├── simsearch.py
│   │   ├── training.py
│   │   ├── prediction.py
│   │   ├── clustering.py
│   │   ├── export.py
│   │   └── archive.py
│   ├── tools/                     # adapters around external CLIs
│   │   ├── glide.py               # builds .in/.inp; parses Glide .csv
│   │   ├── spacelight.py
│   │   ├── ftrees.py
│   │   ├── chemprop_runner.py     # wraps model_runner_{train,predict}
│   │   └── fpsim2.py
│   ├── scheduler/
│   │   ├── base.py                # Scheduler ABC (submit_array, wait, cancel, status)
│   │   ├── local.py               # subprocess pool
│   │   ├── slurm.py               # sbatch + sacct/squeue polling
│   │   └── sge.py
│   ├── workspace/
│   │   ├── layout.py              # WorkDir(path), JobDir, paths via methods not strings
│   │   └── manifest.py            # JSON manifest tracking artefacts + checksums
│   ├── workflows/
│   │   ├── screen.py              # the iterate loop
│   │   └── resume.py              # checkpoint/resume
│   ├── remote/                    # scripts shipped to compute nodes
│   │   ├── prop_filter.py         # replaces control.py
│   │   ├── train.py               # current model_runner_train.py
│   │   └── predict.py             # current model_runner_predict.py
│   └── ui/
│       ├── tui/                   # Textual TUI (interactive)
│       └── web/                   # FastAPI + small JS SPA (optional, deferred)
└── tests/
    ├── unit/                      # isolated, no SLURM
    ├── integration/               # local-scheduler end-to-end with tiny inputs
    └── fixtures/                  # minimal .space stub, fake glide
```

### 11.2 CLI design (argparse, subcommands)

Keep **stdlib `argparse`** (no Typer/Click dependency). Use `add_subparsers(dest="command")` so each stage gets its own parser, `--help`, and validation. The top-level `spacehasten` dispatches on `args.command`.

```
spacehasten init <db>                         # create empty .dbsh, write workspace/manifest.json
spacehasten import-seeds <smi-or-csv> [--top N] [--props props.toml]
spacehasten train [--config train.yaml]
spacehasten predict [--all-undocked]
spacehasten search <space> [--queries N] [--strategy greedy|clustering] [--source docked|predicted] [--props props.toml]
spacehasten dock [--top N] [--strategy greedy|clustering]
spacehasten cluster
spacehasten screen <space> --queries N --dock N [--rounds R] [--props props.toml]
spacehasten export csv   --cutoff -10 -o hits.csv
spacehasten export poses --cutoff -10 --iteration N -o hits.maegz
spacehasten archive create   [--bundle]
spacehasten archive extract  <archive>     # reverse of --bundle: rebuild dbsh + workdir from a self-contained archive
spacehasten archive restore  <archive>     # restore an unbundled archive (links/dirs)
spacehasten archive clean
spacehasten status         # job/cycle/iter/model summary, scheduler queue snapshot
spacehasten resume         # continues from last checkpoint
spacehasten verify         # smoke test (ports current verify_spacehasten.py)
```

Global flags: `--db`, `--config`, `--scheduler {auto,slurm,sge,local}`, `--partition`, `--scratch`, `--log-level`, `--json` (machine-readable output for scripting/UI).

**Property-filter file** (`--props props.toml`): the physchem windows are too many to specify ergonomically on a CLI. Use a small TOML (preferred) or JSON file:

```toml
# props.toml
[properties]
mw       = { min = 0.0,  max = 500.0 }
slogp    = { min = -10.0, max = 5.0 }
hba      = { min = 0,    max = 10 }
hbd      = { min = 0,    max = 5 }
rotbonds = { min = 0,    max = 10 }
tpsa     = { min = 0.0,  max = 140.0 }
```

Ranges still get persisted into the dbsh `properties` table on first use; subsequent stages read from there unless `--props` is passed again to override.

Every command should be implementable as a thin wrapper around a `stages.*` function so the same code path is used by CLI, TUI, and tests.

### 11.3 Scheduler abstraction

```python
class Scheduler(ABC):
    def submit_array(self, job: ArrayJob) -> ArrayHandle: ...
    def wait(self, handle, on_progress: Callable[[Progress], None]) -> ArrayResult: ...
    def cancel(self, handle) -> None: ...
    def status(self, handle) -> ArrayStatus: ...

@dataclass
class ArrayJob:
    name: str
    workdir: Path
    array_size: int
    max_concurrent: int
    cpus_per_task: int
    gpus: int = 0
    exclusive: bool = False
    env_setup: list[str]          # ["source ...", "conda activate ..."]
    command: list[str] | str      # task body; templating done by scheduler
    depends_on: list[ArrayHandle] = ()
```

Concrete `SlurmScheduler`:
- Renders `sbatch` script via Jinja2 template (one shared template, no per-stage duplication). *If Jinja2 turns out to be overkill for our handful of templates, fall back to a stdlib solution (`string.Template` or f-strings on a `textwrap.dedent`'d block) — the rendering is encapsulated in one module so swapping it is cheap.*
- Submits with `subprocess.run(["sbatch", "--parsable", script])`, captures job id.
- Polls with `sacct -j <id> -X --format=JobID,State -P` (parse `COMPLETED|FAILED|TIMEOUT|...`).
- Treats non-COMPLETED tasks as failures; surfaces failed-task indices for retry.
- Supports `--dependency=afterok:<id>` for chaining.

Replace sentinel-file polling everywhere. Keep an opt-in `LocalScheduler` for unit tests.

### 11.4 Workspace / file-ops streamlining

**Single working directory, on `/data/`, in plain sight.**

- One root: `WorkDir(path)`. The dbsh and *every* stage's artefacts (formerly `SIMSEARCH_*`, `DOCKING_*`, `TRAIN_*`, `CLUSTERING_*` under `$HOME/SPACEHASTEN/`) all live as **subdirectories of the same workdir**, not in `$HOME` and not behind a hidden `.spacehasten/` folder. Stage subdirs use plain visible names: `simsearch/cycle1/`, `docking/iter1/`, `models/v1/`, `clustering/`, `logs/`, `manifest.json`.
- **Disk policy**: SpaceHASTEN assumes the workdir is on `/data/` (slow NFS, but the canonical home). At startup:
  - If `cwd` is on `/data/` → proceed.
  - If `cwd` is on `/wrk/` (fast local scratch) → warn and offer to set up the canonical workdir at `/data/$USER/SPACEHASTEN/<run_name>/`. All stages live under that one folder; the old per-stage split across `$HOME/SPACEHASTEN/SIMSEARCH_*`, `DOCKING_*`, etc. is gone.
  - If somewhere else → warn but allow it (some users may have a different layout).
  - Scheduler jobs continue the existing `data → wrk → data` pattern *internally* (per-task `$SCRATCH/$USER/<jobname>` on the compute node), but the orchestrator only ever reads/writes one root.
- `WorkDir` exposes typed methods: `dock_dir(iteration)`, `simsearch_dir(cycle)`, `model_dir(version)`, `manifest_path()`, `slurm_log_dir()`. No more `os.getenv("HOME")+"/SPACEHASTEN/..."` strings.
- `Manifest` (JSON in workdir) records: stage state, scheduler-job IDs, input file checksums, model version, completion timestamps. Enables `spacehasten status` and `spacehasten resume`.
- Stop storing models as BLOBs by default. Disk-based `models/v<N>/` registry inside the workdir; keep a `--bundle` option that re-tars (or re-inlines) on archive, with `archive extract` as the inverse to recreate the workdir from a bundle.
- Replace `os.system` with `subprocess.run` (checked) or pure Python (`tarfile`, `gzip`, `shutil`).
- Use `pathlib.Path` everywhere; ban string-split path manipulation.
- Stream large CSVs (current `pd.read_csv` of 50M-row Enamine dump uses a lot of RAM); use `polars` or chunked pandas where it matters.

### 11.5 GUI modernization options

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Textual TUI** | Runs over SSH, in tmux, no X needed (matches HPC reality); same Python; rich progress; trivial to reuse `Scheduler.status()` | Less discoverable than GUI | **Recommended primary** |
| **PyQt6 / Qt for Python** | Modern look; easy file dialogs; native | Heavy dep; same single-user-on-login-node UX problem as Tk | Optional secondary |
| **FastAPI + small SPA** (htmx or React) | Multi-user, run on a head node, browser anywhere; serves status, plots, hit browser | More moving parts; auth needed; deployment overhead | Recommended for v2 once core stable |
| **Web extension to Jupyter / Streamlit** | Quick to prototype; great for plots | Awkward for long-running jobs | Nice-to-have, not core |

Plan: ship a **Textual TUI** as the day-1 replacement for tkinter; keep an optional **FastAPI** dashboard for status/progress/results browsing. Tk goes away.

The TUI will be a thin shell over the same `stages.*` and `Scheduler` APIs the CLI uses — no orchestration logic in the UI.

### 11.6 Configuration

- Replace ad-hoc ini with **Pydantic Settings** that merges (in priority): CLI flags > `~/.config/spacehasten/config.toml` (per-user) > `spacehasten.toml` in workdir > install-time `etc/spacehasten.toml` > defaults.
- **Per-project property filter** lives in its own file (`props.toml` or `props.json`) referenced via `--props`. First time it's used, the values are written into the dbsh `properties` table; later stages read from there unless `--props` is passed again to override. This replaces the current pile of `--mw_min/--mw_max/--logp_min/...` flags.
- Other per-job mutable state (acquisition method, cutoff, etc.) stays in the dbsh (already does).
- All site-specific paths (BioSolveIT execs, conda envs, partition) live in the install config and may be overridden per-run.

### 11.6a Logging

Three log streams, all in the **same workdir**:

1. **Master log** (`logs/spacehasten.log`): single file, append-only, timestamped (`YYYY-MM-DD HH:MM:SS`). Records every CLI invocation with the full command line and resolved arguments, every stage start/finish, scheduler job IDs, model versions, cycle/iteration bumps, and errors. This is the audit trail — a user should be able to reconstruct what was run, when, with which parameters, by reading just this file.
2. **Per-stage logs** (`logs/<stage>-<cycle-or-iter>.log`): orchestrator-side log of one stage's activity (queries built, chunks created, post-processing). Mirrors what currently goes to stdout.
3. **Scheduler logs** (`logs/slurm/<jobname>/`): `#SBATCH -o`/`-e` redirected into a dedicated subfolder, one folder per submitted job, with files like `task-0001.out`/`task-0001.err`. Off by default for array jobs (because a 250-task array creates 500 small files on NFS), but enabled with `--keep-slurm-logs` (or `keep_slurm_logs = true` in config). When disabled, stdout/stderr go to `/dev/null` per task; failures are detected via `sacct` state, not log scraping.

Use stdlib `logging` with a `RotatingFileHandler` for the master log and a `StreamHandler` for console. Per-stage logs are written by adding a temporary `FileHandler` for the duration of the stage.

### 11.7 Migration phases

**Phase 0 — Scaffolding (no behaviour change)**
- Create `pyproject.toml`, `src/` layout, `pre-commit` (ruff, black, mypy), CI.
- Move existing files in place under a compat shim that re-exports them so the current `spacehasten.py` keeps working.
- Add a `tests/` skeleton with at least: cfg loads, dbsh creation roundtrip.

**Phase 1 — Extract pure logic**
- Move sqlite schema + queries into `core/db.py` with typed accessors. Keep schema identical (forward-compat with existing .dbsh).
- Replace `SimpleNamespace`-args with dataclasses (`SimsearchParams`, `DockParams`, ...).
- Cache `get_latest_*` per-process; pass an open `Connection` rather than reopening per call.
- Make `cfg` lazy and not validate-on-import; introduce `Settings.from_files(...)`.

**Phase 2 — Scheduler abstraction**
- Implement `Scheduler` ABC, `SlurmScheduler`, `LocalScheduler`.
- Replace one stage at a time (start with **training**: smallest blast radius), then prediction, then docking, then simsearch, then clustering.
- Replace sentinel-file waits with real status polling. Add structured retries for failed array tasks.
- Delete `scheduler_functions.py` once all six writers are gone.

**Phase 3 — CLI-first entrypoint**
- Build the `argparse` app exposed as the new `spacehasten` console-script.
- Each subcommand calls `stages.*` directly; no Tk import.
- Ensure `verify` runs through the new CLI.
- Keep the old `gui.py`/`spacehasten.py` available as a deprecated `legacy-gui` console-script for one release.

**Phase 4 — Workspace cleanup**
- Introduce `WorkDir` + `Manifest`. Migrate `$HOME/SPACEHASTEN/*` into the visible single-root workdir on `/data/` (default: `<dbsh-dir>/` itself, or `/data/$USER/SPACEHASTEN/<run_name>/` if launched from `/wrk/`). One-shot `spacehasten migrate` helper for legacy runs.
- Add the structured `logs/` layout (master log, per-stage logs, optional `logs/slurm/`).
- Switch model storage to filesystem registry (`models/v<N>/`); add `archive create --bundle` for a self-contained tarball and `archive extract` as the inverse.

**Phase 5 — TUI**
- Build Textual TUI over the same stage APIs. Feature parity with current Tk: pick seeds, new/load job, run screen, export.
- Live job progress driven by `Scheduler.status()`.

**Phase 6 — Optional web dashboard**
- FastAPI service exposing read-only status + result browsing + simple "submit screen" form. Deferred until TUI is stable.

**Phase 7 — Polish**
- Type-check (mypy/strict on `core/`, `stages/`, `scheduler/`).
- Documentation site (mkdocs-material).
- Tagged release; keep the legacy code in a `legacy/` subpackage or a frozen branch.

### 11.8 Risks and what to preserve

- **Reproducibility of past runs.** Keep the SQLite schema (`data`, `models`, `clusters`, `properties`, `docking_param`, `docking_grid`) byte-compatible; old `.dbsh` files must continue to open. Add a schema-version row only as additive metadata.
- **BioSolveIT and Schrödinger contracts.** Don't change Glide `.in`/`.inp` templating semantics — they were tuned (LigPrep settings, Phase database). Wrap them but keep current outputs identical until tested.
- **Acquisition logic correctness.** The greedy/clustering SQL queries are subtle (clustering: `GROUP BY clusterid ORDER BY MIN(pred_score)`); preserve them under unit tests with fixed fixtures before refactoring.
- **HPC environment activation.** Conda env activation per job (`source PREPARE_ANACONDA && ACTIVATE_*`) is a real constraint — don't assume the new code can `import chemprop` from the orchestrator process; keep the remote-script pattern.
- **Verification.** Port `verify_spacehasten.py` to the new CLI early; treat it as the regression suite.
- **Rollout.** Run new and old in parallel against the same `.dbsh` for at least one production screen before retiring the legacy GUI path.

---

## Appendix A — Quick wins independent of the rewrite

These can be done in the existing tree to reduce risk before/while rewriting:

1. Fix `cfg.py` SGE branch typo: `elif self.SCHEDULER != "SGE"` → `== "SGE"`.
2. Replace direct `os.system("sbatch ...")` with `args.c.SCHEDULER_SUBMIT` in [docking_functions.py](docking_functions.py:151), [training_functions.py](training_functions.py:62), [simsearch_functions.py](simsearch_functions.py:148), [prediction_functions.py](prediction_functions.py).
3. Cache the sqlite connection in a single workflow run instead of opening it 100s of times.
4. Add `subprocess.run(check=True)` wrappers and stop ignoring tar/sbatch failures.
5. Centralize the "compute name from dbname" idiom into `Path(dbname).stem`.
6. Delete dead code: `prediction_functions.predict_dock`, `chunkpredict.py`.
