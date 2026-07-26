# SpaceHASTEN Copilot Instructions

## Environment

Conda is at `/wrk/lurvas/miniconda3`. Always activate it before running any Python or shell commands:

```bash
source /wrk/lurvas/miniconda3/etc/profile.d/conda.sh
conda activate spacehasten-quick
```

The primary environment is `spacehasten-quick` (Python 3.11, chemprop 2.1.2, lightning, FPSim2).

## Terminal

Always run commands inside a `tmux` session. If not already in one:

```bash
tmux new -s work   # new session
tmux attach -t work  # reattach
```

## GitHub

Create branches, pushes, and PRs against Anirudh Jain's fork:
`https://github.com/sponde25/SpaceHASTEN`.

Do not create PRs against the upstream/original SpaceHASTEN repository unless explicitly requested.

## Running code

Prefix all Python execution with the conda activation:

```bash
source /wrk/lurvas/miniconda3/etc/profile.d/conda.sh && conda activate spacehasten-quick && python3 ...
```

## Progress Reporting

Long-running scripts and analysis loops should expose progress whenever practical. Prefer a progress bar such as `tqdm`, or periodic structured logging when a progress bar is unsuitable. Include completed and total work, elapsed time, processing rate, and ETA when the total is known.

## System Conda Environments (for SLURM jobs)

For jobs running on compute nodes via SLURM, use the system-wide conda at `/data/programs/oce/`:

```bash
source /data/programs/oce/actoce
```

Available environments on compute nodes:
- `chemprop-2.1.2` - For training and prediction (has rdkit, chemprop, lightning)
- `fpsim2-0.7.3` - For clustering (has rdkit, FPSim2, tqdm)

**Important:** The local `spacehasten-quick` environment is NOT available on compute nodes. Always use the system conda environments for SLURM-submitted jobs.

## Large Database Analysis

For I/O-intensive analysis of large SQLite databases, keep the canonical database under `/data/$USER` but stage a read-only working copy under `/fastwrk/$USER/<project>/<run>` on the compute node selected for the job. `/wrk` and `/fastwrk` paths exist on every node but are node-local; copies created on the login host or another compute node are not shared. Stage from shared `/data` inside the job, run scans and temporary analysis on that node's `/fastwrk`, then copy only validated final results back to `/data/$USER`.

- Treat the canonical database on `/data` as immutable during analysis.
- If the source database may have an active writer, use SQLite's backup mechanism rather than copying the file directly.
- Verify the staged copy before analysis, for example with `PRAGMA quick_check` and expected row counts.
- Do not copy a modified analysis database back over the canonical database unless the user explicitly requests it.
- Use a run-specific directory, preferably including `$SLURM_JOB_ID`, to prevent concurrent jobs from sharing or overwriting temporary files.

## SpaceHASTEN Installation

### Installation Path
**Primary installation**: `/data/lurvas/projects/coding/SpaceHASTEN`

### System Conda Environments (for SLURM jobs)
For jobs running on compute nodes via SLURM, use the system-wide conda at `/data/programs/oce/`:

```bash
source /data/programs/oce/actoce
```

Available environments on compute nodes:
- `chemprop-2.1.2` - For training and prediction (has rdkit, chemprop, lightning)
- `fpsim2-0.7.3` - For clustering (has rdkit, FPSim2, tqdm)

### Installation Configuration (spacehasten.ini)
```ini
[General]
PREPARE_ANACONDA = source /data/programs/oce/actoce
ACTIVATE_CHEMPROP = conda activate chemprop-2.1.2
ACTIVATE_CLUSTERING = conda activate fpsim2-0.7.3
SCHEDULER = slurm
SLURM_PARTITION = jobs
```

### Installation Steps
1. Run the installer: `python3 install_spacehasten.py`
2. Specify installation path: `/data/lurvas/projects/coding/spacehasten_test/`
3. Use system conda paths for BioSolveIT tools
4. Set conda environments as shown above
5. Install `pigz` system-wide: `sudo apt install pigz`

### Verification
Run end-to-end verification:
```bash
/data/lurvas/projects/coding/spacehasten_test/verify
```

This tests: clustering, docking, chemprop training, SpaceLight, and FTrees.

## File System Restrictions

**Do not modify any files outside `/data/lurvas`, `/wrk/lurvas`, or `/fastwrk/lurvas`.** All other directories are read-only.
