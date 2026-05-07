# CLI Help Text Plan

Goal: Every argument should clearly state whether it's **required** or **optional**, and if optional, what the **default** is.

Convention:
- Required args: no prefix needed (argparse already marks them)
- Optional args: help string starts with `"Optional."` and ends with `"Default: <value>."`

---

## Global Options (all subcommands)

| Argument | Required? | Current help | Proposed help |
|----------|-----------|--------------|---------------|
| `-w / --workspace` | Optional | Path to the SpaceHASTEN workspace directory. If omitted, the current working directory is used. | Optional. Workspace directory. Default: current working directory. |
| `--config` | Optional | Path to a TOML or INI config file (auto-detected by suffix). | Optional. TOML or INI config file (auto-detected by suffix). Default: none (built-in settings). |
| `--scheduler` | Optional | Scheduler backend (default: auto). | Optional. Scheduler backend. Default: auto. | # here the default should be slurm!
| `--partition` | Optional | SLURM partition (overrides the config file). | Optional. SLURM partition (overrides config file). Default: "jobs" | 
| `--scratch` | Optional | Override the scratch directory (paths.scratch_default). | Optional. Scratch directory override. Default: "/wrk" |
| `--log-level` | Optional | Logging verbosity (default: INFO). | Optional. Logging verbosity. Default: INFO. |
| `--quiet` | Optional | Suppress the startup banner. | Optional. Suppress the startup banner. |

---

## `init`

| Argument | Required? | Current help | Proposed help |
|----------|-----------|--------------|---------------|
| `path` (positional) | Required | Local root directory (should be on fast storage: /wrk or /fastwrk). | Local root directory (should be on fast storage: /wrk or /fastwrk). |
| `--name` | Optional | Project name (default: directory name). | Optional. Project name. Default: directory name. |
| `--shared-root` | Optional | NFS directory for stage artefacts visible to compute nodes. Default: /data/$USER/SPACEHASTEN/<name>/ | Optional. NFS directory for stage artefacts visible to compute nodes. Default: /data/$USER/SPACEHASTEN/<name>/. |
| `--dock-params` | Required | Glide .in template. | Glide docking parameter .in file. |
| `--dock-grid` | Required | Glide grid .zip. | Glide grid .zip file. |

---

## `pick-seeds`

| Argument | Required? | Current help | Proposed help |
|----------|-----------|--------------|---------------|
| `--seeds-file` | Optional | Path to seed collection (bz2/tsv). Default from settings. | Optional. Path to seed collection (bz2/tsv). Default: from config. |
| `--output / -o` | Required | Output .smi file path. | Output .smi file path. |
| `--n-seeds` | Optional | Number of seeds to sample (default: from settings, typically 1000000). | Optional. Number of seeds to sample. Default: from config (typically 1000000). | # this should not be optional, the user should always set the number of seeds
| `--cores` | Optional | Number of local cores for RDKit canonicalization (default: from settings). | Optional. Local cores for RDKit canonicalization. Default: from config. | # I think default should be all available CPUs

---

## `seed-training`

| Argument | Required? | Current help | Proposed help |
|----------|-----------|--------------|---------------|
| `--smi` | Required | SMI input (undocked seeds). | SMI file with undocked seed compounds. |
| `--dock-cpus` | Required | Concurrent docking tasks. | Number of concurrent docking tasks. |
| `--props-toml` | Optional | Optional. PropertyRanges TOML override. Default: built-in ranges. | Optional. PropertyRanges TOML override. Default: built-in ranges. |
| `--processes` | Optional | Optional. Worker pool size for hashing. Default: all available CPUs. | Optional. Worker pool size for hashing. Default: all available CPUs. |

---

## `screening-cycle`

| Argument | Required? | Current help | Proposed help |
|----------|-----------|--------------|---------------|
| `--rounds` | Optional | Number of screening rounds. | Optional. Number of screening cycle rounds. Default: 1. |
| `--strategy` | Optional | (no help text) | Optional. Acquisition strategy for choosing which compounds to dock. Default: greedy. |
| `--simsearch-top-n` | Optional | Queries per simsearch. | Optional. Number of query compounds per simsearch. Default: 100. | #should not be optional. should say "Number of simsearch queries. Recommendation: 1000"
| `--simsearch-cpu` | Optional | CPUs for simsearch. | Optional. CPUs for simsearch tasks. Default: 1. | # should not be optional. "Number of CPUs for simsearch tasks. Recommendation: max 250"
| `--space` | Optional | .space file override. | Optional. BioSolveIT .space file override. Default: from config. |
| `--dock-top-n` | Optional | Compounds to dock. | Optional. Number of compounds to dock per round. Default: 1000. | # should not be optional. should say "Number of compounds to dock per round. Recommendation: 1000000 (1M)"
| `--dock-cpus` | Optional | Concurrent docking tasks. | Optional. Concurrent docking tasks. Default: 1. | # Should not be optional. should say "Number of CPUs for docking tasks. Recommendation: max 250"

---

## `import-seeds`

| Argument | Required? | Current help | Proposed help |
|----------|-----------|--------------|---------------|
| `--smi` | Required (mutually exclusive with --csv) | SMI input (undocked seeds). | SMI file with undocked seed compounds. |
| `--csv` | Required (mutually exclusive with --smi) | CSV input (docked seeds). | CSV file with pre-docked seed compounds. |
| `--props-toml` | Optional | PropertyRanges TOML override. | Optional. PropertyRanges TOML override. Default: built-in ranges. |
| `--processes` | Optional | Worker pool size. | Optional. Worker pool size for hashing. Default: all available CPUs. |

---

## `dock`

| Argument | Required? | Current help | Proposed help |
|----------|-----------|--------------|---------------|
| `--top-n` | Required | (no help text) | Number of compounds to dock. |
| `--strategy` | Optional | (no help text) | Optional. Acquisition strategy for choosing which compounds to dock. Default: greedy. |
| `--cpus` | Optional | (no help text) | Optional. Concurrent docking tasks. Default: 1. | # Should not be optional. should say "Number of CPUs for docking tasks. Recommendation: max 250"

---

## `train`

| Argument | Required? | Current help | Proposed help |
|----------|-----------|--------------|---------------|
| `--cutoff` | Optional | (no help text) | Optional. Docking score cutoff for including compounds in the training set. Default: 10.0. |

---

## `predict`

| Argument | Required? | Current help | Proposed help |
|----------|-----------|--------------|---------------|
| `--model-version` | Optional | Model version (defaults to latest). | Optional. Model version to use. Default: latest. |

---

## `search`

| Argument | Required? | Current help | Proposed help |
|----------|-----------|--------------|---------------|
| `--source` | Required | (no help text) | Source compound pool: docked or predicted. |
| `--strategy` | Optional | (no help text) | Optional. Query acquisition strategy. Default: greedy. |
| `--top-n` | Required | (no help text) | Number of query compounds. |
| `--space` | Optional | (no help text) | Optional. BioSolveIT .space file override. Default: from config. |
| `--nnn` | Optional | (no help text) | Optional. Nearest-neighbor count for queries. Default: from config. |
| `--sim-spacelight` | Optional | (no help text) | Optional. SpaceLight similarity threshold. Default: from config. |
| `--sim-ftrees` | Optional | (no help text) | Optional. FTrees similarity threshold. Default: from config. |
| `--cpus` | Optional | (no help text) | Optional. CPUs for simsearch tasks. Default: 1. |  # should not be optional. "Number of CPUs for simsearch tasks. Recommendation: max 250"
| `--threads-per-task` | Optional | (no help text) | Optional. Threads per simsearch task. Default: 1. | #default should be 2
| `--cluster-after` | Optional | (no help text) | Optional. Run clustering after search completes. |

---

## `cluster`

No additional arguments (uses workspace context).

---

## `export csv`

| Argument | Required? | Current help | Proposed help |
|----------|-----------|--------------|---------------|
| `--cutoff` | Required | (no help text) | Docking score cutoff for export. |
| `--output` | Required | (no help text) | Output CSV file path. |

---

## `export poses`

| Argument | Required? | Current help | Proposed help |
|----------|-----------|--------------|---------------|
| `--cutoff` | Required | (no help text) | Docking score cutoff for export. |
| `--output` | Required | (no help text) | Output Maestro .mae file path. |
| `--iteration` | Optional | (no help text) | Optional. Limit to a specific docking iteration. Default: all iterations. |

---

## `archive create`

| Argument | Required? | Current help | Proposed help |
|----------|-----------|--------------|---------------|
| `--bundle` | Optional | Produce a .tgz bundle. | Optional. Produce a single .tgz bundle. |

---

## `archive extract`

| Argument | Required? | Current help | Proposed help |
|----------|-----------|--------------|---------------|
| `--archive` | Required | (no help text) | Path to the .tgz bundle to extract. |
| `--target` | Required | (no help text) | Target directory for extraction. |

---

## `archive restore`

| Argument | Required? | Current help | Proposed help |
|----------|-----------|--------------|---------------|
| `--archive` | Required | (no help text) | Path to .archived-spacehasten directory. |
| `--target` | Required | (no help text) | Target workspace directory. |

---

## `archive clean`

No additional arguments.

---

## `verify`

| Argument | Required? | Current help | Proposed help |
|----------|-----------|--------------|---------------|
| `--workdir` | Optional | Workspace root for the verify run (default: $HOME/SPACEHASTEN/VERIFY-<version>). | Optional. Workspace root for verify run. Default: $HOME/SPACEHASTEN/VERIFY-<version>. |
| `--fixtures-dir` | Optional | Directory containing examples.smi, example.smi, example.csv, test_dock.in and grid-test_dock.zip (default: this package's install root). | Optional. Input files directory for verification tests. Default: package install root. |
| `--only` | Optional | Run only these checks. | Optional. Run only these checks. (options: #list here the options) |
| `--skip` | Optional | Skip these checks. | Optional. Skip these checks. (options: #list here the options) Default: none. |
| `--keep-workdir` | Optional | Do not delete the verify workdir on success. | Optional. Keep the verify workdir after success. |

---

## `status`

No additional arguments.

---

## `resume`

No additional arguments.
