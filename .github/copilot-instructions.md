# SpaceHASTEN Copilot Instructions

## Environment

Conda is at `/fastwrk/ajain/miniconda3`. Always activate it before running any Python or shell commands:

```bash
source /fastwrk/ajain/miniconda3/etc/profile.d/conda.sh
conda activate spacehasten-quick
```

The primary environment is `spacehasten-quick` (Python 3.11, chemprop 2.1.2, lightning, FPSim2).

## Terminal

Always run commands inside a `tmux` session. If not already in one:

```bash
tmux new -s work   # new session
tmux attach -t work  # reattach
```

## Running code

Prefix all Python execution with the conda activation:

```bash
source /fastwrk/ajain/miniconda3/etc/profile.d/conda.sh && conda activate spacehasten-quick && python3 ...
```

## System Conda Environments (for SLURM jobs)

For jobs running on compute nodes via SLURM, use the system-wide conda at `/data/programs/oce/`:

```bash
source /data/programs/oce/actoce
```

Available environments on compute nodes:
- `chemprop-2.1.2` - For training and prediction (has rdkit, chemprop, lightning)
- `fpsim2-0.7.3` - For clustering (has rdkit, FPSim2, tqdm)

**Important:** The local `spacehasten-quick` environment is NOT available on compute nodes. Always use the system conda environments for SLURM-submitted jobs.

## SpaceHASTEN Installation

### Installation Path
**Primary installation**: `/data/ajain/spacehasten-e2e-install/`

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
2. Specify installation path: `/data/ajain/spacehasten-e2e-install/`
3. Use system conda paths for BioSolveIT tools
4. Set conda environments as shown above
5. Install `pigz` system-wide: `sudo apt install pigz`

### Verification
Run end-to-end verification:
```bash
/data/ajain/spacehasten-e2e-install/verify
```

This tests: clustering, docking, chemprop training, SpaceLight, and FTrees.

## File System Restrictions

**Do not modify any files outside `/data/ajain`, `/wrk`, or `/fastwrk`.** All other directories are read-only.
