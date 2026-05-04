# Plan: CLI Restructuring — Workflow vs Manual Modes

Restructure the SpaceHASTEN CLI into two clearly separated usage patterns: **workflow mode** (opinionated end-to-end pipelines) and **manual mode** (individual stage commands for expert use). Rename `import-seeds --auto-train` → `seed-training`, rename `screen` → `screening-cycle` with updated flow, make `import-seeds` import-only, move dock-params/dock-grid to `init`, and reorganize help text.

**Steps**

### Phase 1: Extend `init` with dock-params/dock-grid
1. Add optional `--dock-params PATH` and `--dock-grid PATH` to `init`
2. Store them in the DB at init time (reuse existing `db.store_dock_param()`/`db.store_dock_grid()`)

### Phase 2: Create `seed-training` workflow command
3. New parser `_add_seed_training()` — args: `--smi` (required), `--props-toml`, `--processes`, `--dock-top-n`, `--dock-strategy`, `--dock-cpus`, `--train-cutoff`, `--dock-params` (optional override), `--dock-grid` (optional override)
4. Implement `_cmd_seed_training()` — import → dock → train → cluster; uses workspace-stored dock-params/grid by default

### Phase 3: Simplify `import-seeds`
5. Remove `--auto-train` and cascade params from `import-seeds`
6. Make `--dock-params`/`--dock-grid` optional (override workspace defaults, or set if not set at init)
7. `import-seeds`: pure DB import only

### Phase 4: Rename `screen` → `screening-cycle`  
8. Rename command, update flow to: Train (if not first screening-cycle) search(docked) → predict → search(predicted) → predict → search(predicted) → predict → dock
9. Remove `--train-first` option. The program should determine whether to train or not based on which screening cycle is taking place. we do not train directly after seed-training, but we do train after a previous screening-cycle (as there are new compounds that are docked that can be used for training a better model) 

### Phase 5: Reorganize help text
10. Group commands in argparse help:
    - **Setup**: `init`, `pick-seeds`
    - **Workflows**: `seed-training`, `screening-cycle`, `export`
    - **Manual stages**: `import-seeds`, `dock`, `train`, `search`, `predict`, `cluster`, `export`
    - **Utilities**: `status`, `resume`, `archive`, `verify`

### Phase 6: Wire dock-params from workspace
11. `dock` and `screening-cycle` accept optional `--dock-params`/`--dock-grid` overrides; default to DB-stored values
12. Add `db.load_dock_param()`/`db.load_dock_grid()` if not already present

### Phase 7: Refactor `seeds.import_seeds()` stage
13. Remove auto_train logic from `seeds.import_seeds()`; keep it as pure import
14. Cascade logic lives in the `_cmd_seed_training` CLI handler (or a thin workflow function)

**Relevant files**
- `src/spacehasten/cli/main.py` — parser definitions, command handlers
- `src/spacehasten/cli/_common.py` — global options, dock-params resolution helper
- `src/spacehasten/stages/seeds.py` — `import_seeds()` → remove cascade
- `src/spacehasten/stages/docking.py` — `dock()` → load dock-params from DB fallback
- `src/spacehasten/stages/prediction.py` — `predict_undocked()` called by screening-cycle
- `src/spacehasten/core/db.py` — add `load_dock_param()`/`load_dock_grid()` if missing
- `src/spacehasten/workspace/layout.py` — `WorkDir.bootstrap()`

**Verification**
1. `spacehasten --help` displays grouped command layout
2. `spacehasten seed-training --help` / `screening-cycle --help` show correct args
3. `spacehasten import-seeds --help` has NO `--auto-train`
4. `pytest tests/unit/test_cli.py` passes with updated expectations
5. Integration: `seed-training` runs full cascade; `screening-cycle` runs (train)→search→predict→…→dock

**Decisions**
- `seed-training` and `screening-cycle` as command names
- Flat commands with grouped help (no nested subgroups)
- `--auto-train` removed immediately
- Dock-params/dock-grid stored at `init`, overridable per-command
- `screening-cycle` flow: (train if not first screening-cycle)→search(docked)→predict→(search(predicted)→predict)×2→dock
