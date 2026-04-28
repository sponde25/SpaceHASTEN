# SpaceHASTEN Legacy Codebase Reference

> **Purpose.** Frozen reference for the rewrite. Every fact below is derived
> from the legacy tree at the workspace root (21 `.py` files + 1 bash wrapper
> + 1 sec_clustering.sh, ~4330 LOC) on 2026-04-28. Used together with
> [REWRITE_PLAN.md](../REWRITE_PLAN.md) and [SESSIONS.md](SESSIONS.md).
>
> **Scope.** Per-file public API, side effects, SQL, file paths, scheduler
> jobs, CLI flags, external-tool command lines. Do not modify this document
> while the rewrite is in progress unless the legacy code itself changes.

---

## 1. File-by-file analysis

### [cfg.py](../cfg.py) — configuration god-object

**Public API**
- `SpaceHASTENConfiguration()` (class, 35–368): instantiated once per run. Reads `${SPACEHASTEN_DIRECTORY}/spacehasten.ini`, sets ~100 attributes, validates external tools via `shutil.which` / `os.path.exists`, builds scheduler keyword strings.

**Behaviour on import**
- Resolves `SPACEHASTEN_DIRECTORY` from `os.path.abspath(sys.argv[0])` — fragile under symlinks.
- Hard-fails (`sys.exit`) on missing tools — cannot be imported in tests without a full install.

**Bug**: line 324 `elif self.SCHEDULER != "SGE":` should be `==`. With the typo, the SGE branch is unreachable when `scheduler=SGE`, and the "Unknown scheduler" branch becomes unreachable too.

**Scheduler keyword tables (lines 312–368)**
- SLURM: `SCHEDULER_SUBMIT="sbatch"`, `SCHEDULER_JOBNAME="#SBATCH -J"`, `SCHEDULER_OUTPUT_LOG="#SBATCH -o /dev/null"`, `SCHEDULER_OUTPUT_ERR="#SBATCH -e /dev/null"`, `SCHEDULER_PARTITION="#SBATCH -p {SLURM_PARTITION}"`, `SCHEDULER_CPU_PER_TASK="#SBATCH --cpus-per-task="`, `SCHEDULER_ARRAY_JOB="#SBATCH --array=1-"`, `SCHEDULER_ARRAY_ID="SLURM_ARRAY_TASK_ID"`, `SCHEDULER_GPU="#SBATCH {SLURM_GPU_PARAMETER}"`, `SCHEDULER_GPU_EXCLUSIVE="#SBATCH --exclusive"` (or `#nop`).
- SGE: same shape with `qsub` / `#$ -N` / `#$ -t 1-` / `SGE_TASK_ID`.

---

### [functions.py](../functions.py) — shared helpers

**Public API**
- `get_dbsh_properties(dbname) -> SimpleNamespace` (47–89): reads property ranges from the `properties` table; falls back to `cfg.SpaceHASTENConfiguration` defaults on any error (so importing `cfg` here also requires a full install — the bare-except is the only thing keeping legacy `.dbsh` files openable on dev boxes).
- `update_dbsh_properties(dbname, args)` (91–108): `DROP TABLE IF EXISTS properties` + `CREATE TABLE properties(property TEXT, is_double INTEGER, min_limit TEXT, max_limit TEXT)` + 6 `INSERT`s. Values are interpolated via string concat (callers pass strings).
- `check_glide_gridgen_input(glideinfile) -> bool` (110–121): looks for `GRID_CENTER|INNERBOX|OUTERBOX|RECEP_FILE`.
- `check_nfs(filename) -> bool` (123–133): subprocess `stat -f -c %T <filename>`, checks for `nfs` substring.
- `get_rdkit_properties(csv_filename) -> (list, list)` (135–142): pandas read; returns `(smilesid, docking_score)`.
- `get_latest_model(name) -> int` (144–154): `SELECT COUNT(*) FROM models`. Opens new connection.
- `get_latest_cycle(name) -> int` (156–167): `SELECT MAX(simsearch_cycle) FROM data`.
- `get_latest_iteration(name) -> int` (169–180): `SELECT MAX(dock_iteration) FROM data`.
- `mol2hash(line) -> str|None` (182–195): splits on `§`, `Chem.MolFromSmiles`, returns `RegistrationHash.GetMolLayers(mol)[HashLayer.TAUTOMER_HASH] + "§" + line`. Used in `mp.Pool`.
- `cxsmi2smi(cxsmiles_with_id) -> str|None` (197–207): canonicalises cxsmiles; returns `"<smiles> <id>\n"`.

**Cross-module**: imported by every stage module.

---

### [scheduler_functions.py](../scheduler_functions.py) — bash-script writers

Six near-duplicate writers, one polling helper. All write `submit_*.sh` and rely on `args.c.SCHEDULER_*` keyword strings. None of them call `sbatch` themselves — that is hard-coded in callers (see §A.4 below).

**Public API**
- `write_header(w, args, jobname)` (30–35).
- `close_job(w, args, personal_scratch)` (37–42): writes the `cd $curdir; rm -rf $personal_scratch; touch jobdone-{name}-CPU${TASK_ID}` block. **The sentinel pattern `jobdone-{name}-CPU{task_id}` is the universal completion signal.**
- `write_search_scheduler(cycle_dir, args)` (44–56): array job 1..top; each task runs SpaceLight + FTrees on one query line.
- `write_dock_scheduler(dock_dir, args, chunk_counter)` (58–80): array job 1..chunk_counter; each task runs Phase + LigPrep + Glide.
- `write_predict_scheduler(control_dir, args)` (82–103): array 1..cpu; each task runs `model_runner_predict.py`.
- `write_train_scheduler(control_dir, args)` (105–131): single-task GPU job; runs `model_runner_train.py`.
- `write_control_scheduler(control_dir, args)` (133–158): array 1..cpu; each task runs `control.py` then `model_runner_predict.py`.
- `write_cluster_scheduler(cluster_dir, args)` (160–176): single CPU-heavy job; runs `EXE_CLUSTERING_DEFAULT` (sec_clustering.sh).
- `wait_until_jobs_done(directory, name, num_jobs)` (178–189): `glob('jobdone-{name}-CPU*')`, sleep 5s, tqdm bar. **No SLURM polling, no failure detection.**

**Conda activation pattern**: every job script begins with `source <PREPARE_ANACONDA>` then `conda activate <ACTIVATE_*>` (chemprop or fpsim2).

---

### [importseeds_functions.py](../importseeds_functions.py)

**Public API**
- `import_seeds(args)` (42–151).

**SQL**
```sql
CREATE TABLE data (
  spacehastenid INTEGER PRIMARY KEY,
  reghash TEXT, smiles TEXT, smilesid TEXT,
  dock_score REAL, pred_score REAL,
  spacelight REAL, ftrees REAL,
  query INTEGER,
  dock_iteration INTEGER, pred_version INTEGER, simsearch_cycle INTEGER
);
CREATE TABLE docking_param (dock_param BLOB);
CREATE TABLE docking_grid  (dock_grid  BLOB);
CREATE TABLE models (model_version INTEGER UNIQUE, model_tar BLOB);
CREATE INDEX idx_reghash ON data(reghash);
-- Then per-row:
INSERT INTO data(reghash, smiles, smilesid, dock_score, dock_iteration) VALUES (?,?,?,?,0)  -- when CSV
INSERT INTO data(reghash, smiles, smilesid)                              VALUES (?,?,?)    -- when SMILES
```

**Inputs**: `args.smiles` (SMILES file with id) **or** `args.csv` (already-docked Glide CSV); `args.dock_params` (Glide .in); `args.dock_grid` (grid .zip); property bounds.

**Multiprocessing**: `mp.Pool(cores)` over `functions.mol2hash`.

**Calls**: `functions.update_dbsh_properties`, `docking_functions.dock(importing_seeds=True)` *(only for SMILES path)*, `training_functions.train_new_model`, `cluster_functions.cluster_dbsh`.

---

### [simsearch_functions.py](../simsearch_functions.py)

**Public API**
- `simsearch(args, do_not_update_gui=False)` (118–299): the canonical iteration step.
- `process_sim_results(args, cycle_dir)` (lines around 200–280, mp.Pool-based aggregator).

**SQL**
```sql
-- Pick queries:
-- greedy, docked source:
SELECT smiles, spacehastenid FROM data
 WHERE query IS NULL AND dock_score IS NOT NULL
 ORDER BY dock_score LIMIT ?;
-- clustering, docked source:
SELECT smiles, data.spacehastenid FROM data, clusters
 WHERE data.spacehastenid = clusters.spacehastenid
   AND query IS NULL AND dock_score IS NOT NULL
 GROUP BY clusterid ORDER BY MIN(dock_score) LIMIT ?;
-- greedy, predicted source:
SELECT smiles, spacehastenid FROM data
 WHERE query IS NULL AND pred_score IS NOT NULL AND dock_score IS NULL
 ORDER BY pred_score LIMIT ?;
-- clustering, predicted source: same shape with MIN(pred_score).

UPDATE data SET query = <cycle> WHERE spacehastenid = <id>;

SELECT model_tar FROM models WHERE model_version = <ver>;

INSERT INTO data(reghash, smiles, smilesid, spacelight, ftrees, pred_score, simsearch_cycle)
       VALUES (?,?,?,?,?,?,?);
```

**Files written** under `${HOME}/SPACEHASTEN/SIMSEARCH_{name}_cycle{cycle}/`:
- `queries_{name}.smi`
- `spacelightresult_{name}_{cpu}_1.csv`, `ftreesresult_{name}_{cpu}_1.csv`
- `CONTROL/control_{name}_cpu{cpu}.smi.gz`, `CONTROL/control.param`, `CONTROL/predicted_propoutput_control_{name}_cpu{cpu}.csv`, extracted model dir.

**Multiprocessing**: `mp.Pool` to read result CSVs and prediction CSVs.

**Calls**: `functions.get_latest_cycle/model`, `scheduler_functions.write_search_scheduler/write_control_scheduler`, `cluster_functions.cluster_dbsh`.

**Direct sbatch**: line 156 (`os.system("sbatch submit_ctrl_..."`) and line 253 (`sbatch submit_queries_...`). Bypasses `args.c.SCHEDULER_SUBMIT`.

---

### [docking_functions.py](../docking_functions.py)

**Public API**
- `dock(args, importing_seeds=False, do_not_update_gui=False)` (62–155).
- `process_docking_results(args, dock_iteration)` (13–42).
- `write_confgen_file(filename)` (157–200): writes Phase/LigPrep `.inp` template.
- `write_docking_file(filename, dbname, dock_dir)` (202–227): extracts `docking_param` and `docking_grid` blobs to disk; strips `LIGANDFILE`/`GRIDFILE` lines and rewrites them.

**SQL**
```sql
SELECT smiles, spacehastenid FROM data WHERE dock_score IS NULL;             -- greedy
SELECT smiles, data.spacehastenid FROM data, clusters
  WHERE data.spacehastenid = clusters.spacehastenid AND dock_score IS NULL
  GROUP BY clusterid ORDER BY MIN(pred_score) LIMIT ?;                       -- clustering
SELECT dock_param FROM docking_param;
SELECT dock_grid  FROM docking_grid;
UPDATE data SET dock_score = ?, dock_iteration = ? WHERE spacehastenid = ?;
```

**Files** under `${HOME}/SPACEHASTEN/DOCKING_{name}_iter{iter}/`:
- `dockinput_{name}_iter{iter}_cpu{cpu}.smi`
- `dockinput_{name}_iter{iter}_cpu{cpu}.inp`
- `glide_dockinput_{name}_iter{iter}_cpu{cpu}.in`
- `glide_grid.zip` (shared)
- `results-dockinput_{name}_iter{iter}_cpu{cpu}.tar.gz` (output)

**Per-task scratch**: `${SCRATCH}/${USER}/<jobname>_cpu${TASK_ID}/`.

**Direct sbatch**: line 160. Calls `process_docking_results` after `wait_until_jobs_done`.

**Multiprocessing**: `mp.Pool` over `os.system("tar xzf ...")` for result extraction.

---

### [training_functions.py](../training_functions.py)

**Public API**
- `train_new_model(args)` (40–91).

**SQL**
```sql
SELECT smiles, dock_score FROM data
 WHERE dock_score IS NOT NULL AND dock_score < <TRAIN_DOCKING_CUTOFF=10.0>;
INSERT INTO models VALUES(<version>, ?);   -- ? = gzipped tarball BLOB
```

**Behaviour**: writes `train_{name}_ver{N}.csv`; submits 1 GPU job; waits on `jobdone-train_{name}-CPU1`; tars `model_{name}_ver{N}/` → gzipped → INSERT BLOB; deletes on-disk model dir.

**Direct sbatch**: line 67.

---

### [prediction_functions.py](../prediction_functions.py)

**Public API**
- `update_predicted_scores(args)` (118–196): main orchestration.
- `predict_dock(mols, args)` (56–116): **legacy/dead** local path that shells out to `chemprop_predict` CLI via `mp.Pool`.
- `get_chemprop_predictions(param)` (24–35): helper for `predict_dock`.

**SQL**
```sql
SELECT smiles, spacehastenid FROM data WHERE dock_score IS NULL;
SELECT model_tar FROM models WHERE model_version = <ver>;
UPDATE data SET pred_score = ?, pred_version = ? WHERE spacehastenid = ?;
```

**Files** under `${HOME}/SPACEHASTEN/SIMSEARCH_{name}_cycle{cycle}/PREDICT/`:
- `predict_{name}_cpu{cpu}.csv` (input)
- `predicted_predict_{name}_cpu{cpu}.csv` (output)
- extracted `model_{name}_ver{ver}/`.

**Direct sbatch**: line 170.

---

### [cluster_functions.py](../cluster_functions.py)

**Public API**
- `cluster_dbsh(args)` (58–103).
- `process_cluster_results(args, cluster_dir)` (42–56).

**SQL**
```sql
SELECT COUNT(*) FROM data;
SELECT smiles, spacehastenid FROM data;
DROP TABLE IF EXISTS clusters;
CREATE TABLE clusters(spacehastenid INTEGER PRIMARY KEY, clusterid INTEGER);
-- then pandas .to_sql("clusters", ..., if_exists="append")
```

**Files** under `${HOME}/SPACEHASTEN/CLUSTERING_{name}_tmp/`:
- `clustering_input.smi.gz` (all SMILES, gzipped)
- `clustering.csv` (output: spacehastenid, clusterid)

**External script**: `EXE_CLUSTERING_DEFAULT` (default = `sec_clustering.sh`, see §1.20).

---

### [archive_functions.py](../archive_functions.py)

**Public API**
- `archive(args)` (32–56): `pigz` the `.dbsh`, symlink every `SIMSEARCH_*` and `DOCKING_*` cycle/iteration dir, `tar -ch` (follow links) into `{name}.archived-spacehasten`.
- `restore(args)` (58–75): inverse.
- `clean(args)` (77–92): `rm -fr` matching dirs and `.dbsh`.

**Staging**: `${SCRATCH}/${USER}/ARCHIVE_{name}/`.

---

### [export_functions.py](../export_functions.py)

**Public API**
- `export_results(args)` (32–57): CSV export.
- `export_poses(args)` (59–87): for each `DOCKING_{name}_iter{i}/` tarball, decompress, run [export_poses.py](../export_poses.py) under `$SCHRODINGER/run` per `*_pv.maegz`, concatenate `.mae` shards, `pigz`.

**SQL**
```sql
SELECT smiles, data.spacehastenid, smilesid, dock_score, pred_score,
       spacelight, ftrees, dock_iteration, clusterid
  FROM data, clusters
 WHERE data.spacehastenid = clusters.spacehastenid
   AND dock_score <= <cutoff>
 ORDER BY dock_score;
```

---

### [export_poses.py](../export_poses.py)

Schrödinger-side script invoked via `$SCHRODINGER/run`. Reads `_pv.maegz`, filters by docking score, looks up `smilesid` per pose:

```sql
SELECT smilesid FROM data WHERE spacehastenid = ?;
```

Writes `.mae` shards.

---

### [control.py](../control.py)

Remote script (runs on compute node, in `chemprop` conda env). Reads gzip'd SMILES + `control.param`; computes RDKit `Descriptors.MolWt`, `Crippen.MolLogP`, `CalcNumHBA`, `CalcNumHBD`, `CalcNumRotatableBonds`, `CalcTPSA`; filters by ranges; emits `propoutput_{name}.csv` with columns `smiles, rawmol` (rawmol holds reghash + original smiles + id, joined by `§`).

**Param-file format**: 12 lines, in this order: mw_min, mw_max, slogp_min, slogp_max, hba_min, hba_max, hbd_min, hbd_max, rotbonds_min, rotbonds_max, tpsa_min, tpsa_max.

---

### [chunkpredict.py](../chunkpredict.py)

Legacy/local chemprop-CLI predictor; chunks a CSV, calls `chemprop predict` per chunk, concatenates. **Dead in main loop**, only referenced from cfg.py path validation.

---

### [model_runner_train.py](../model_runner_train.py)

Standalone chemprop-2.x trainer (Python API + Lightning). Run on compute node in `chemprop-2.1.2` env.

CLI:
```
model_runner_train.py <data_csv> <save_dir>
  --batch-size <int>   --epochs <int>   --num-workers <int>
  --devices <str>      --mp-hidden-size <int>   --mp-depth <int>
  --ffn-hidden-size <int>   --ffn-layers <int>   --dropout <float>
  --activation <str>   --batch-norm <0|1>
  --warmup-epochs <int>   --init-lr <float>   --max-lr <float>   --final-lr <float>
```

Internally:
- `featurizers.SimpleMoleculeMolGraphFeaturizer()`
- `data.MoleculeDatapoint.from_smi(smi, y)` then `data.MoleculeDataset(..., featurizer)`
- 90/10 split (deterministic last-10% as val)
- Standard scaler on targets
- `chemprop_nn.BondMessagePassing → MeanAggregation → RegressionFFN`
- `chemprop_models.MPNN(message_passing, agg, predictor, batch_norm, warmup_epochs, init_lr, max_lr, final_lr)`
- `pl.Trainer.fit`

**Output**: `<save_dir>/model_0/pytorch_model.bin`.

---

### [model_runner_predict.py](../model_runner_predict.py)

Standalone chemprop-2.x predictor.

CLI:
```
model_runner_predict.py <data_csv> <model_dir> <output_csv>
  --batch-size <int>   --num-workers <int>
  --accelerator {cpu,gpu}   --devices <str>
```

Reads `<model_dir>/model_0/pytorch_model.bin` via `MPNN.load_from_checkpoint(weights_only=False)`.

**Input** CSV cols: `smiles, smilesid`. **Output** CSV cols: `smilesid, docking_score`.

---

### [cmdline.py](../cmdline.py)

Single flat argparse, no subcommands. Flags:

| Flag | Type | Choices | Used by |
|---|---|---|---|
| `--database` | str | — | all |
| `--action` | str | `importsmiles, screen, exportcsv, cluster` | dispatcher |
| `--smiles` | str | — | importsmiles |
| `--dock_params` | str | — | importsmiles |
| `--dock_grid` | str | — | importsmiles |
| `--mode` | str | `greedy, clustering` | screen |
| `--space` | str | — | screen |
| `--simsearch_queries` | int | — | screen |
| `--simsearch_cpus` | int | — | screen |
| `--dock_mols` | int | — | screen |
| `--dock_cpus` | int | — | screen |
| `--cutoff` | float | — | exportcsv |
| `--export_file` | str | — | exportcsv |

No `--help` per action, no per-stage subcommand, no resume/status, no JSON config.

---

### [spacehasten.py](../spacehasten.py)

```python
app = gui.SpaceHASTENGUI()                # always builds Tk root
app.command_line_args = cmdline.parse_cmdline()
if app.command_line_args.action is None:
    app.mainloop()
else:
    app.run_cmdline()
```
There is no headless path — Tk root is created even for CLI runs.

---

### [gui.py](../gui.py) — 1059 LOC, biggest single file

**Class** `SpaceHASTENGUI(tk.Tk)`. Owns workflow orchestration in addition to UI.

Major methods:
- `__init__` (101–149): instantiates `cfg.SpaceHASTENConfiguration`, `queue.Queue`, GUI state, builds 5 frames.
- `run_cmdline` (151–262): CLI dispatcher; builds `SimpleNamespace` per action; spawns worker threads; calls `*_functions` directly.
- `gui_thread_virtual_screening` (1008–1025): canonical loop = train (if continuing) → simsearch(docked) → simsearch(predicted) → simsearch(predicted) → dock.
- `gui_thread_train` (990–1006): `train_new_model` → `update_predicted_scores`.
- `gui_new` (857–938), `gui_load` (940–975), `gui_virtual_screening` (871–926), `gui_export` (1116–1145).

Frames:
- `frame_main`, `frame_working`, `frame_task_menu`, `frame_export_menu`, `frame_props_menu`. Toggled with `grid`/`grid_forget`.

**IPC**: `queue.Queue` of stringly-typed messages: `"Percent:NN"`, `"UpdateModel:N"`, `"DoneTaskmenu"`, `"Done"`. Polled at 200 ms by `perioidic_call` / `check_queue`.

**Threads**: vanilla `threading.Thread`; no cancellation, no exception propagation.

---

### [install_spacehasten.py](../install_spacehasten.py)

Interactive wizard. Prompts for paths/conda envs/partitions, writes `spacehasten.ini`, copies `.py` files + logo + test fixtures into target dir.

---

### [verify_spacehasten.py](../verify_spacehasten.py)

End-to-end smoke test. Uses bundled `examples.smi`, `test_dock.in`, `grid-test_dock.zip`. Tests:
1. `which sbatch`
2. `sbatch --partition=X --wrap='echo Hello World!'`
3. SpaceLight + FTrees locally
4. Submit a docking job
5. Submit a training job
6. Submit a clustering job

Polls with the same 5 s loop.

---

### [sec_clustering.sh](../sec_clustering.sh) — sphere-exclusion clustering script

Self-extracting bash. Embeds a Python script that uses RDKit `LeaderPicker` + FPSim2.

Steps:
1. `split --lines=10000 <smi>` → `x*` chunks.
2. For each chunk: `python3 sec_clustering.py make_fp <chunk>` → Morgan FPs.
3. `fpsim2-create-db <smi> fp.h5 --fp_type Morgan --fp_params '{"radius":2,"fpSize":1024}'`.
4. `python3 sec_clustering.py` → `LeaderPicker` at distance 0.7 (Tanimoto similarity 0.3); writes `cluster_centroids.smi`.
5. `split --lines=10 cluster_centroids.smi` → `x*` chunks.
6. For each centroid chunk: `python3 sec_clustering.py search_fp <chunk>` → similarity search.
7. `python3 sec_clustering.py compile <smi>` → `clustering.csv`.

Output: `clustering.csv` with columns `spacehastenid, clusterid`.

---

### [spacehasten](../spacehasten) (bash launcher)

```bash
PREPARE_ANACONDA=$(grep PREPARE_ANACONDA "$DIR/spacehasten.ini" | cut -d= -f2 | sed -e 's:^ *::g')
CHEMPROP_ACTIVATE=$(grep ACTIVATE_CHEMPROP "$DIR/spacehasten.ini" | cut -d= -f2 | sed -e 's:^ *::g')
eval "$PREPARE_ANACONDA"
eval "$CHEMPROP_ACTIVATE"
python3 "$DIR/spacehasten.py" $@
```

Fragile (`grep` of an INI), and double-`eval`s untrusted strings.

---

## A. Consolidated reference

### A.1 Complete SQLite schema

```sql
-- Created in importseeds_functions.import_seeds:
CREATE TABLE data (
  spacehastenid    INTEGER PRIMARY KEY,
  reghash          TEXT,
  smiles           TEXT,
  smilesid         TEXT,
  dock_score       REAL,
  pred_score       REAL,
  spacelight       REAL,
  ftrees           REAL,
  query            INTEGER,
  dock_iteration   INTEGER,
  pred_version     INTEGER,
  simsearch_cycle  INTEGER
);
CREATE INDEX idx_reghash ON data(reghash);

CREATE TABLE docking_param (dock_param BLOB);  -- one row, the original Glide .in
CREATE TABLE docking_grid  (dock_grid  BLOB);  -- one row, the original Glide grid .zip

CREATE TABLE models (
  model_version INTEGER UNIQUE,
  model_tar     BLOB                            -- gzipped tar of model_<name>_ver<N>/
);

-- Created lazily by functions.update_dbsh_properties:
CREATE TABLE properties (
  property   TEXT,
  is_double  INTEGER,        -- 1 if float, 0 if int
  min_limit  TEXT,           -- stored as text; cast at read time
  max_limit  TEXT
);
-- 6 rows: property in {'mw','slogp','hba','hbd','rotbonds','tpsa'}.

-- Created/recreated by cluster_functions.process_cluster_results:
CREATE TABLE clusters (
  spacehastenid INTEGER PRIMARY KEY,
  clusterid     INTEGER
);
```

**Lifecycle**: `data`, `docking_param`, `docking_grid`, `models` → created once. `properties` → DROP+CREATE per `update_dbsh_properties` call. `clusters` → DROP+CREATE per clustering run.

### A.2 spacehasten.ini key inventory

See cfg.py reading order. **All values stored as strings in ini; cast at read.**

#### `[General]`
`scheduler`, `prepare_anaconda`, `activate_chemprop`, `activate_clustering`, `gpu_exclusive`,
`cpu_count_search`, `cpu_count_dock`, `cpu_count_predict`, `cpu_count_control`, `cpu_count_clustering`,
`schrodinger_feature_flags`, `model_spec_path`, `model_hparams_path`,
`train_batch_size`, `train_epochs`, `train_num_workers`, `train_devices`,
`train_mp_hidden_size`, `train_mp_depth`, `train_ffn_hidden_size`, `train_ffn_layers`,
`train_dropout`, `train_activation`, `train_batch_norm`,
`train_warmup_epochs`, `train_init_lr`, `train_max_lr`, `train_final_lr`,
`pred_batch_size`, `pred_num_workers`, `pred_accelerator`, `pred_devices`.

#### `[Paths]`
`exe_spacelight_default`, `exe_ftrees_default`, `scratch_default`,
`spaces_dir_default`, `spaces_file_default`,
`seeds_dir_default`, `seeds_file_default`, `exe_clustering_default`.

#### `[Slurm]`
`slurm_partition`, `slurm_gpu_parameter`.

#### `[SGE]`
`sge_queue`, `sge_pe`, `sge_gpu_parameter`.

#### `[Properties]`
`mw_min/max`, `slogp_min/max`, `hba_min/max`, `hbd_min/max`, `rotbonds_min/max`, `tpsa_min/max`.

### A.3 File-system path inventory

Two roots in the legacy code:

1. **`<cwd>/`** — `<name>.dbsh`. Must be on local fast disk (NFS warning at startup).
2. **`${HOME}/SPACEHASTEN/`** — every other artefact:
   - `SIMSEARCH_{name}_cycle{cycle}/` — query SMILES, SpaceLight + FTrees results, `CONTROL/`, `PREDICT/`.
   - `DOCKING_{name}_iter{iter}/` — Glide inputs, `glide_grid.zip`, `results-*.tar.gz`.
   - `TRAIN_{name}_ver{ver}/` — training scratch (deleted after model is BLOB'd into `models`).
   - `CLUSTERING_{name}_tmp/` — clustering scratch.
3. **`${SCRATCH}/${USER}/<jobname>_cpu{TASK_ID}/`** — per-task per-node fast scratch.

Path templates are reconstructed by string concatenation in many call sites; there is no central path module.

### A.4 Scheduler-job inventory (six job types)

For all six: header rendered from `args.c.SCHEDULER_*` strings, conda activated inside the script, completion sentinel `jobdone-{name}-CPU{TASK_ID}`.

| # | Job | File pattern | Array | Body (compute node) |
|---|---|---|---|---|
| 1 | **search (SpaceLight + FTrees)** | `submit_queries_{name}.sh` | `1..top % cpu` | `spacelight -i "$smiles" -s {space} -o spacelightresult_{name}_${TASK}.csv ...`; `ftrees -i "$smiles" -s {space} -o ftreesresult_{name}_${TASK}.csv ...` |
| 2 | **dock** | `submit_dockinput_{name}_iter{iter}.sh` | `1..chunks % cpu` | `$SCHRODINGER/jsc local-server-start`; `$SCHRODINGER/pipeline -prog phase_db <cpu>.inp -OVERWRITE -WAIT -NOJOBID -NJOBS 1`; `$SCHRODINGER/phase_database <cpu>.phdb export -omae <cpu> -get 1 -limit 99999999 -WAIT`; `$SCHRODINGER/glide -new -OVERWRITE -WAIT -NJOBS 1 -HOST localhost:1 glide_<cpu>.in`; tar results back |
| 3 | **predict** | `submit_predict_{name}_cycle{cycle}.sh` | `1..cpu` | `model_runner_predict.py predict_<cpu>.csv model_<name>_ver<v> predicted_predict_<cpu>.csv ...` |
| 4 | **control (prop filter + predict)** | `submit_ctrl_{name}_cycle{cycle}.sh` | `1..cpu` | `control.py control_<cpu>.smi.gz control.param`; `gunzip -c propoutput_control_<cpu>.csv.gz > propoutput_<cpu>.csv`; `model_runner_predict.py propoutput_<cpu>.csv model_<name>_ver<v> predicted_propoutput_<cpu>.csv ...` |
| 5 | **train** | `submit_train_{name}_ver{ver}.sh` | single (GPU, exclusive) | `model_runner_train.py train_<name>_ver<v>.csv model_<name>_ver<v> --batch-size ... --final-lr ...`; `touch jobdone-train_<name>-CPU1` |
| 6 | **cluster** | `submit_cluster_{name}.sh` | single (CPU-heavy) | `{EXE_CLUSTERING_DEFAULT} clustering_input.smi`; `mv clustering.csv $curdir/` |

**Direct `os.system("sbatch ...")` call sites** (bypassing `args.c.SCHEDULER_SUBMIT`):
- [docking_functions.py:160](../docking_functions.py)
- [training_functions.py:67](../training_functions.py)
- [simsearch_functions.py:156](../simsearch_functions.py), [simsearch_functions.py:253](../simsearch_functions.py)
- [prediction_functions.py:170](../prediction_functions.py)

### A.5 External tool invocation matrix

| Tool | Stage | Exact command (template) |
|---|---|---|
| **SpaceLight** | search | `{exe_spacelight} -i "{smiles}" -s {space} -o spacelightresult_{name}_{cpu}.csv --max-nof-results {NNN_DEFAULT=10000} --min-similarity-threshold {SIM_SPACELIGHT_DEFAULT=0.5} --thread-count 1` |
| **FTrees** | search | `{exe_ftrees} -i "{smiles}" -s {space} -o ftreesresult_{name}_{cpu}.csv --max-nof-results {NNN_DEFAULT=10000} --min-similarity-threshold {SIM_FTREES_DEFAULT=0.9} --thread-count 1` |
| **$SCHRODINGER/jsc** | dock | `$SCHRODINGER/jsc local-server-start` |
| **$SCHRODINGER/pipeline** | dock | `$SCHRODINGER/pipeline -prog phase_db {cpu}.inp -OVERWRITE -WAIT -NOJOBID -NJOBS 1` |
| **$SCHRODINGER/phase_database** | dock | `$SCHRODINGER/phase_database {cpu}.phdb export -omae {cpu} -get 1 -limit 99999999 -WAIT` |
| **$SCHRODINGER/glide** | dock | `$SCHRODINGER/glide -new -OVERWRITE -WAIT -NJOBS 1 -HOST localhost:1 glide_{cpu}.in` |
| **$SCHRODINGER/run** | export poses | `$SCHRODINGER/run export_poses.py {pv_file} {cutoff} {dbsh}` |
| **chemprop predict** (v2) | predict / control | `chemprop predict --num-workers 0 --accelerator {cpu|gpu} --devices {N} --test-path {csv} --model-path {model} --preds-path {output}` (only via `chunkpredict.py`; legacy in main loop) |
| **model_runner_train.py** | train | `python3 model_runner_train.py {csv} {save_dir} --batch-size ... --final-lr ...` |
| **model_runner_predict.py** | predict / control | `python3 model_runner_predict.py {csv} {model_dir} {out_csv} --batch-size ... --devices ...` |
| **control.py** | control | `python3 control.py {smi.gz} control.param` |
| **fpsim2-create-db** | cluster | `fpsim2-create-db {smi} fp.h5 --fp_type Morgan --fp_params '{"radius":2,"fpSize":1024}' --processes $(nproc)` |
| **sec_clustering.sh** | cluster | `{EXE_CLUSTERING_DEFAULT} {smi}` → `clustering.csv` |
| **pigz** | archive / export | `pigz -c {file} > {file}.gz` / `pigz -d {file}.gz` |

### A.6 Acquisition-strategy SQL (preserve verbatim)

The exact queries that select the next batch of compounds. Any rewrite must reproduce these with fixed fixtures before refactoring.

```sql
-- DOCKING (docking_functions.dock):
-- greedy:
SELECT smiles, spacehastenid FROM data
 WHERE dock_score IS NULL
 ORDER BY pred_score LIMIT ?;
-- clustering:
SELECT smiles, data.spacehastenid FROM data, clusters
 WHERE data.spacehastenid = clusters.spacehastenid
   AND dock_score IS NULL
 GROUP BY clusterid ORDER BY MIN(pred_score) LIMIT ?;

-- SIMSEARCH queries (simsearch_functions.simsearch):
-- docked source, greedy:
SELECT smiles, spacehastenid FROM data
 WHERE query IS NULL AND dock_score IS NOT NULL
 ORDER BY dock_score LIMIT ?;
-- docked source, clustering:
SELECT smiles, data.spacehastenid FROM data, clusters
 WHERE data.spacehastenid = clusters.spacehastenid
   AND query IS NULL AND dock_score IS NOT NULL
 GROUP BY clusterid ORDER BY MIN(dock_score) LIMIT ?;
-- predicted source, greedy:
SELECT smiles, spacehastenid FROM data
 WHERE query IS NULL AND pred_score IS NOT NULL AND dock_score IS NULL
 ORDER BY pred_score LIMIT ?;
-- predicted source, clustering:
SELECT smiles, data.spacehastenid FROM data, clusters
 WHERE data.spacehastenid = clusters.spacehastenid
   AND query IS NULL AND pred_score IS NOT NULL AND dock_score IS NULL
 GROUP BY clusterid ORDER BY MIN(pred_score) LIMIT ?;
```

### A.7 Constants worth preserving

From cfg.py class-level defaults:

| Name | Value | Used by |
|---|---|---|
| `DOCKING_CHUNK` | 1000 | docking chunk size |
| `RDKIT_CHUNK_DEFAULT` | 12345 | RDKit chunk size for prop filter |
| `CHEMPROP_CHUNK_DEFAULT` | 12345 | Chemprop predict chunk size |
| `CHEMPROP_CPU_DEFAULT` | 250 | default predict CPUs |
| `MAX_CORES` | 250 | docking-array cap |
| `MAX_JOBNAME_LEN` | 15 | scheduler limit |
| `SIM_SPACELIGHT_DEFAULT` | 0.5 | similarity threshold |
| `SIM_FTREES_DEFAULT` | 0.9 | similarity threshold |
| `NNN_DEFAULT` | 10000 | max-nof-results per query |
| `TRAIN_DOCKING_CUTOFF` | 10.0 | training-set inclusion threshold |
| `FIELD_SCORE_DEFAULT` | `r_i_docking_score` | Glide CSV column |
| `FIELD_TITLE_DEFAULT` | `title` | Glide CSV column |
| `FIELD_SIMILARITY_SPACELIGHT` | `fingerprint-similarity` | SpaceLight CSV column |
| `FIELD_SIMILARITY_FTREES` | `pharmacophore-similarity` | FTrees CSV column |
