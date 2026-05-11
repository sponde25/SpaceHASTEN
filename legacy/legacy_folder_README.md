# Legacy SpaceHASTEN tree

This directory holds the **pre-rewrite** SpaceHASTEN codebase. It is kept
intact for **one release** as a regression baseline and as a fallback
while users migrate to the new `spacehasten` package (Session 15
cutover, MIGRATION_STATUS.md).

## Contents

| File | Replacement in the new package |
|---|---|
| `spacehasten.py`, `cmdline.py`, `spacehasten` (bash) | `spacehasten` console script (`spacehasten.cli.main`) |
| `gui.py` | `spacehasten-legacy-gui` console script (kept for one release) |
| `cfg.py` | `spacehasten.config.settings` |
| `functions.py` | `spacehasten.core.{db,molecules}` |
| `archive_functions.py` | `spacehasten.stages.archive` |
| `cluster_functions.py` | `spacehasten.stages.clustering` |
| `docking_functions.py` | `spacehasten.stages.docking` + `spacehasten.tools.glide` |
| `export_functions.py`, `export_poses.py` | `spacehasten.stages.export` (the latter still drives `export_poses.py` via `Settings.paths.export_poses_script`) |
| `importseeds_functions.py` | `spacehasten.stages.seeds` |
| `prediction_functions.py`, `chunkpredict.py` | `spacehasten.stages.prediction` (chunkpredict was dead code, not ported) |
| `simsearch_functions.py`, `control.py` | `spacehasten.stages.simsearch` + `spacehasten.remote.prop_filter` |
| `training_functions.py` | `spacehasten.stages.training` |
| `scheduler_functions.py` | `spacehasten.scheduler.{base,local,slurm,factory}` |
| `model_runner_train.py`, `model_runner_predict.py` | `spacehasten.remote.{train,predict}` |
| `sec_clustering.sh` | `spacehasten.remote.cluster` |
| `verify_spacehasten.py`, `verify` (bash) | `spacehasten verify` |

## Status

These files **must not be edited**. Bug fixes and improvements go into
the new package. This directory will be removed in the release after
the cutover.

## Running the legacy GUI

A `spacehasten-legacy-gui` console script is provided for one release:

```bash
spacehasten-legacy-gui
```

It runs `gui.py` from this directory. The legacy GUI requires the
legacy tree to remain in place; if you have only the new package
installed, the script will exit with a clear error.
