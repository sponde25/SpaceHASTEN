# SpaceHASTEN Copilot Instructions

## Environment

Local conda initialization is provided by `/wrk/setup_conda.sh`. Always activate it before
running Python or project shell commands:

```bash
source /wrk/setup_conda.sh
conda activate spacehasten-quick
```

The primary environment is `spacehasten-quick` at
`/fastwrk/$USER/miniconda3/envs/spacehasten-quick` (Python 3.11). Verify commands resolve to this
environment when behavior is ambiguous.

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
source /wrk/setup_conda.sh && conda activate spacehasten-quick && python3 ...
```

## Editing

For substantial changes, use small focused patches that modify one file or one narrow logical unit
at a time. Do not submit a single large multi-file patch; review and validate each capability group
before moving to the next.

## Progress Reporting

Long-running scripts and analysis loops should expose progress whenever practical. Prefer a progress bar such as `tqdm`, or periodic structured logging when a progress bar is unsuitable. Include completed and total work, elapsed time, processing rate, and ETA when the total is known.

## Agent Delegation

Use `general`/Sol agents for complex analysis, architecture, scientific reasoning, integration,
and multi-file implementation. Their prompts must include the exact workspace, owned files,
input/output paths, environment commands, scientific invariants, and verification commands.

Use Terra only for narrow, mechanical work after the main agent has already produced a complete
implementation plan. Suitable Terra tasks include a bounded file edit, compilation, formatting,
or deterministic data extraction from explicitly named inputs. Do not delegate open-ended
codebase exploration, scientific analysis design, architecture, report synthesis, or multi-stage
workflow implementation to Terra.

Every delegated prompt must state these environment rules explicitly:

- Workspace: `/data/$USER/PROJECTS/SpaceHASTEN`.
- Local setup: `source /wrk/setup_conda.sh && conda activate spacehasten-quick`.
- Expected local Python: `/fastwrk/$USER/miniconda3/envs/spacehasten-quick/bin/python3`.
- Never use another user's filesystem namespace or conda environment.
- SLURM jobs use `source /data/programs/oce/actoce` and the approved system environment.
- Node-local temporary paths use `/fastwrk/$USER/.../${SLURM_JOB_ID}`.

Independently review and verify all delegated output before integrating it. A subagent's successful
return is not evidence that scientific or runtime requirements were met.

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

Before copying a database to `/data`, NFS, or any other shared filesystem, report the source size,
destination, reason for the transfer, and expected reuse, then ask the user for explicit approval.
Do not treat a shared database snapshot as an automatic prerequisite when analysis can remain on the
user's personal login host.

Use separate roots by default: keep the immutable analysis database under the user's local
`/fastwrk/$USER/<project>/<run>` namespace and write compact derived inputs and final artifacts under
`/data/$USER`. A shared artifact root does not imply that the database must be shared.

For approved shared-database workflows, keep the immutable snapshot under `/data/$USER` and have
compute jobs open that shared file directly in read-only/query-only mode. Do not stage a node-local
database copy by default, and never copy the full database once per array task. Prefer preparing
compact shared chunks before submission so most array workers never open the database. Consider
node-local staging only after measured NFS performance demonstrates a material bottleneck and the
user explicitly approves the additional copy.

For user-approved local-only workflows, validate a local backup and place only derived inputs and
final artifacts on shared storage; ask again before a later database archive or SLURM stage that
requires the full database.

- Treat any validated local or approved shared canonical database as immutable during analysis.
- If the source database may have an active writer, use SQLite's backup mechanism rather than copying the file directly.
- Verify the immutable snapshot once before analysis, for example with `PRAGMA quick_check` and expected row counts.
- Do not copy a modified analysis database back over the canonical database unless the user explicitly requests it.
- Use a run-specific directory, preferably including `$SLURM_JOB_ID`, to prevent concurrent jobs from sharing or overwriting temporary files.

## SpaceHASTEN Installation

### Installation Path
**Repository workspace**: the current SpaceHASTEN checkout

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
2. Specify an installation path owned by the current user under `/data/$USER/`.
3. Use system conda paths for BioSolveIT tools
4. Set conda environments as shown above
5. Install `pigz` system-wide: `sudo apt install pigz`

### Verification
Run end-to-end verification:
```bash
/data/$USER/<installation>/verify
```

This tests: clustering, docking, chemprop training, SpaceLight, and FTrees.

## File System Restrictions

Modify only user-owned project and analysis paths under `/data/$USER`, `/wrk/$USER`, and
`/fastwrk/$USER`, unless the user explicitly approves another location. Treat shared system
software under `/data/programs` as read-only.
