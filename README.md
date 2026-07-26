<p align="center">
  <img src="spacehasten_logo.png" alt="SpaceHASTEN logo" width="400"/>
</p>

<h3 align="center">
  Iterative docking, similarity search, and ML-based exploration<br/>
  of nonenumerated chemical libraries
</h3>

<p align="center">
  <a href="https://doi.org/10.1021/acs.jcim.4c01790"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.4c01790-blue"/></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue"/>
  <img alt="License: BSD-3" src="https://img.shields.io/badge/license-BSD--3--Clause-green"/>
</p>

---

SpaceHASTEN is an **active-learning virtual screening pipeline** for
ultra-large BioSolveIT chemical spaces (Enamine REAL, FreedomSpace, etc. —
tens of billions of compounds). Instead of exhaustively enumerating and
docking the entire space, it iterates through cycles of molecular docking,
ML-based scoring, and similarity searching to funnel billions of
compounds down to a tractable set of promising hits — typically by
docking only a few million structures.

```
    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
     ░░░░░░░░░░   10⁹+ molecules   ░░░░░░░░░░░░
      ▒▒▒▒▒▒▒▒▒  similarity search ▒▒▒▒▒▒▒▒▒▒
        ▓▓▓▓▓▓▓  property filter   ▓▓▓▓▓▓▓▓
          ▓▓▓▓▓  chemprop predict  ▓▓▓▓▓▓
            ███   Glide docking    ████
               █      hits         ██
```

> Originally written by Tuomo Kalliokoski (Orion Pharma).
> v0.11 is a ground-up rewrite into a modern, typed Python package.

## Table of Contents

- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Migration from Legacy](#migration-from-legacy)
- [Development](#development)
- [Citation](#citation)
- [License](#license)
- [Version History](#version-history)

## How It Works

SpaceHASTEN implements an active-learning loop that progressively
explores chemical space:

1. **Dock seed molecules** — a small set of seed compounds is docked with
   Schrödinger Glide to produce initial docking scores.
2. **Train a surrogate model** — a chemprop D-MPNN regressor is trained 
   to predict docking scores directly from smiles.
3. **Similarity search** — top-scoring compounds serve as queries for
   SpaceLight and FTrees searches against the BioSolveIT `.space` file,
   retrieving structurally related molecules from billions of candidates.
4. **Filter and predict** — retrieved compounds are filtered by
   physicochemical properties (MW, SlogP, HBA, HBD, rotatable bonds,
   TPSA), then scored by the chemprop model.
5. **Dock top predictions** — the best-predicted compounds are docked with
   Glide, adding real docking scores to the database.
6. **Repeat** — the model is retrained on the expanded dataset, and
   the cycle continues, progressively enriching for active compounds.
7. **Export** — top hits are exported as CSV or 3D Maestro pose files.

Optionally, sphere-exclusion clustering (RDKit + FPSim2) can diversify
compound selection at each stage.

## Requirements

### Hardware

- A few hundred CPU cores (SLURM cluster)
- One GPU (for chemprop training)
- A few hundred GB of disk space
- Linux only (tested on Ubuntu 22.04 and Rocky Linux 8.8)

### Commercial Software

| Software | Tested Version | Purpose |
|---|---|---|
| [Schrödinger Suite](https://www.schrodinger.com/) | 2026-1 | LigPrep, Phase, Glide (docking) |
| [SpaceLight](https://www.biosolveit.de/download/?product=spacelight) | 2.0.0 | Similarity search in chemical spaces |
| [FTrees](https://www.biosolveit.de/download/?product=ftrees) | 7.0.0 | Pharmacophore-based similarity search |

### Free Software

| Software | Tested Version | Purpose |
|---|---|---|
| [chemprop](https://github.com/chemprop/chemprop) | 2.1.2 | D-MPNN training and prediction |
| [RDKit](https://www.rdkit.org/) | ≥ 2023.9 | Molecular hashing, property filtering |
| [FPSim2](https://github.com/chembl/FPSim2) | 0.7.3 | Sphere-exclusion clustering |
| [SLURM](https://slurm.schedmd.com/) | 21.08+ | Job scheduling |
| [pigz](https://zlib.net/pigz/) | any | Parallel gzip compression |
| Python | 3.11+ | Runtime |

### Chemical Spaces

Download `.space` files from
[BioSolveIT](https://www.biosolveit.de/chemical-spaces/).

Diverse seed compound sets can be obtained from:

- [Enamine REAL](https://enamine.net/compound-collections/real-compounds/real-database-subsets)
- [SYNPLE](https://www.synplechem.com/solutions/public-library-download)
- FreedomSpace (contact [ChemSpace](https://chem-space.com/) customer service)

## Installation

### 1. Set up conda environments

SpaceHASTEN dispatches compute-node work to separate conda environments.
Create them before installing:

**chemprop environment** (for training and prediction):

```bash
conda create -n chemprop-2.1.2 python=3.11 -y
conda activate chemprop-2.1.2
pip install chemprop==2.1.2
```

**FPSim2 environment** (for clustering):

```bash
conda create -n fpsim2-0.7.3 python=3.10 -y
conda activate fpsim2-0.7.3
export FPSIM2_MARCH_NATIVE=1
pip install FPSim2==0.7.3
```

### 2. Install SpaceHASTEN

```bash
git clone https://github.com/TuomoKalliokoski/SpaceHASTEN
cd SpaceHASTEN
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

### 3. Run the installer

The interactive installer writes a `spacehasten.ini` configuration file
tailored to your cluster:

```bash
python3 install_spacehasten.py
```

It will ask for:

- Installation path
- Paths to SpaceLight, FTrees, and Schrödinger executables
- Conda activation commands for the chemprop and clustering environments
- SLURM partition name
- Scratch directory location

### 4. Verify the installation

```bash
spacehasten verify --config /path/to/spacehasten.ini
```

This runs an end-to-end smoke test covering: pigz availability,
scheduler connectivity, clustering, docking, chemprop training, and
BioSolveIT tool integration. Takes approximately 15–30 minutes.

## Quick Start

A typical SpaceHASTEN campaign has three phases:

### Phase 1 — Initialize and train on seeds

```bash
# Create a workspace (use fast local storage for the DB)
spacehasten init /wrk/$USER/myscreen \
    --dock-params glide.in \
    --dock-grid grid.zip \
    --config spacehasten.ini

# Import seeds, dock them, and train the first model
spacehasten seed-training \
    --smi seeds.smi \
    --dock-cpus 250 \
    -w /wrk/$USER/myscreen
```

### Phase 2 — Run screening cycles

```bash
# Each round: train → search(docked) → search(predicted)×2 → dock
spacehasten screening-cycle \
    --simsearch-top-n 1000 \
    --simsearch-jobs 250 \
    --dock-top-n 1000000 \
    --dock-cpus 250 \
    --rounds 3 \
    -w /wrk/$USER/myscreen
```

### Phase 3 — Export results

```bash
# Export hits as CSV
spacehasten export csv \
    --cutoff -10.0 \
    --output results.csv \
    -w /wrk/$USER/myscreen

# Export 3D poses (requires Schrödinger)
spacehasten export poses \
    --cutoff -10.0 \
    --output results.mae \
    -w /wrk/$USER/myscreen
```

### Check status at any time

```bash
spacehasten status -w /wrk/$USER/myscreen --actives -8.0

# Machine-readable output
spacehasten status -w /wrk/$USER/myscreen --json
```

## CLI Reference

All commands share these **global options**:

| Flag | Description | Default |
|---|---|---|
| `-w`, `--workspace` | Workspace directory | Current directory |
| `--config` | TOML or INI config file | Auto-discovered |
| `--scheduler` | Backend: `auto`, `slurm`, `local` | `slurm` |
| `--partition` | SLURM partition | From config |
| `--scratch` | Scratch directory | `/wrk` |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `--json` | Machine-readable JSON output | Off |
| `--quiet` | Suppress the startup banner | Off |

### Setup Commands

#### `spacehasten init <path>`

Bootstrap a workspace: create the directory structure, database, and store
docking settings.

```
spacehasten init /wrk/$USER/myscreen \
    --dock-params glide.in \
    --dock-grid grid.zip \
    [--name myproject] \
    [--shared-root /data/$USER/SPACEHASTEN/myscreen]
```

#### `spacehasten pick-seeds`

Sample and canonicalize seeds from a large compound collection file.

```
spacehasten pick-seeds \
    --output seeds.smi \
    --n-seeds 1000000 \
    [--seeds-file collection.cxsmiles.bz2] \
    [--cores 8]
```

### Workflow Commands (Recommended)

#### `spacehasten seed-training`

Full bootstrap workflow: import seeds → dock → train.

```
spacehasten seed-training \
    --smi seeds.smi \
    --dock-cpus 250 \
    [--props-toml custom_props.toml] \
    [--processes 8]
```

#### `spacehasten screening-cycle`

The main active-learning loop: [train] → (search → predict)×3 → dock,
repeated for N rounds.

```
spacehasten screening-cycle \
    --simsearch-top-n 1000 \
    --simsearch-jobs 250 \
    --dock-top-n 1000000 \
    --dock-cpus 250 \
    [--rounds 3] \
    [--strategy greedy|clustering] \
    [--dock-acquisition greedy|clustering|lcb|ei|portfolio] \
    [--portfolio-policy policy.toml] \
    [--space /path/to/space.space] \
    [--nnn 10000]
```

Portfolio acquisition combines calibrated molecular quality with configurable
regional rewards, crowding, and constraints. See
[`docs/portfolio-acquisition.md`](docs/portfolio-acquisition.md) for policy
configuration, calibration, persistence, and retry semantics.

#### `spacehasten export csv`

Export docking results as a CSV file.

```
spacehasten export csv --cutoff -10.0 --output results.csv
```

#### `spacehasten export poses`

Export 3D docking poses as a Maestro file.

```
spacehasten export poses --cutoff -10.0 --output results.mae [--iteration 2]
```

### Manual Stage Commands (Expert)

These give fine-grained control over individual pipeline stages:

| Command | Description |
|---|---|
| `import-seeds --smi FILE` or `--csv FILE` | Import seed compounds (no training) |
| `train [--cutoff 10.0]` | Train one chemprop model |
| `predict [--model-version N]` | Predict scores for undocked rows |
| `search --source docked\|predicted --top-n N --cpus N [--strategy greedy\|clustering]` | Run one similarity search cycle |
| `dock --top-n N --cpus N [--strategy greedy\|clustering\|lcb\|ei\|portfolio]` | Dock the next batch of compounds |
| `cluster` | Run sphere-exclusion clustering |

`--strategy clustering` on `search`/`dock` requires cluster assignments to
already exist — run `cluster` first (these manual stages never cluster
automatically; that only happens inside `screening-cycle --strategy
clustering`, which re-clusters before each search/dock step).

### Utility Commands

| Command | Description |
|---|---|
| `status [--actives THRESHOLD]` | Print workspace summary |
| `resume` | Resume the last interrupted run |
| `archive create [--bundle]` | Archive the workspace |
| `archive extract --archive FILE --target DIR` | Extract a bundled archive |
| `archive restore --archive DIR --target DIR` | Restore an archived workspace |
| `archive clean` | Remove regenerable scratch directories |
| `verify` | End-to-end installation smoke test |

## Configuration

SpaceHASTEN reads configuration from multiple sources, with later sources
overriding earlier ones:

1. **Built-in defaults** (Pydantic models)
2. **INI file** (legacy `spacehasten.ini` format)
3. **TOML file** (new `spacehasten.toml` format)
4. **CLI flags** (`--partition`, `--scratch`, etc.)

Config files are auto-discovered in this order:

1. Explicit `--config` path
2. `spacehasten.toml` or `spacehasten.ini` in the workspace directory
3. Site-wide config installed alongside the package

### INI format (legacy)

```ini
[General]
SCHEDULER = slurm
PREPARE_ANACONDA = source /data/programs/oce/actoce
ACTIVATE_CHEMPROP = conda activate chemprop-2.1.2
ACTIVATE_CLUSTERING = conda activate fpsim2-0.7.3

[Paths]
EXE_SPACELIGHT_DEFAULT = /path/to/spacelight
EXE_FTREES_DEFAULT = /path/to/ftrees
SPACES_FILE_DEFAULT = /path/to/REALSpace.space
SCRATCH_DEFAULT = /wrk

[Slurm]
SLURM_PARTITION = jobs
SLURM_GPU_PARAMETER = --gpus=1

[Properties]
MW_MIN = 0.0
MW_MAX = 500.0
SLOGP_MIN = -10.0
SLOGP_MAX = 5.0
HBA_MIN = 0
HBA_MAX = 10
HBD_MIN = 0
HBD_MAX = 5
ROTBONDS_MIN = 0
ROTBONDS_MAX = 10
TPSA_MIN = 0.0
TPSA_MAX = 140.0
```

### TOML format (new)

```toml
[general]
scheduler = "slurm"
prepare_anaconda = "source /data/programs/oce/actoce"
activate_chemprop = "conda activate chemprop-2.1.2"
activate_clustering = "conda activate fpsim2-0.7.3"

[paths]
exe_spacelight_default = "/path/to/spacelight"
exe_ftrees_default = "/path/to/ftrees"
spaces_file_default = "/path/to/REALSpace.space"
scratch_default = "/wrk"

[slurm]
slurm_partition = "jobs"
slurm_gpu_parameter = "--gpus=1"
```

### Chemprop training hyperparameters

Training parameters are configurable via the `[General]` section:

| Parameter | Default | Description |
|---|---|---|
| `train_batch_size` | 250 | Training batch size |
| `train_epochs` | 30 | Number of training epochs |
| `train_mp_hidden_size` | 300 | Message-passing hidden layer size |
| `train_mp_depth` | 3 | Message-passing depth |
| `train_ffn_hidden_size` | 300 | Feed-forward hidden layer size |
| `train_ffn_layers` | 1 | Number of feed-forward layers |
| `train_dropout` | 0.1 | Dropout rate |
| `train_max_lr` | 1e-3 | Max learning rate |
| `train_warmup_epochs` | 2 | Learning rate warmup epochs |

## Architecture

SpaceHASTEN v0.11 is a PEP 621 package (`src/spacehasten/`) with a clean
separation of concerns:

```
src/spacehasten/
├── cli/                  # Command-line interface (argparse)
│   ├── main.py           #   Entry point and subcommand wiring
│   ├── _common.py        #   Global options, resource builders
│   ├── _banner.py        #   ASCII art banner
│   └── verify.py         #   Installation smoke tests
├── config/               # Configuration management
│   ├── settings.py       #   Pydantic Settings (INI/TOML/CLI merge)
│   └── properties.py     #   PropertyRanges (MW, logP, HBA, etc.)
├── core/                 # Domain logic
│   ├── db.py             #   SQLite wrapper, schema, acquisition SQL
│   └── molecules.py      #   RDKit tautomer hashing
├── stages/               # Pipeline stage orchestration
│   ├── seeds.py          #   Seed import
│   ├── docking.py        #   Glide docking orchestration
│   ├── training.py       #   Chemprop training orchestration
│   ├── prediction.py     #   Chemprop prediction orchestration
│   ├── simsearch.py      #   3-phase similarity search
│   ├── clustering.py     #   Sphere-exclusion clustering
│   ├── export.py         #   CSV and pose export
│   └── archive.py        #   Workspace archival
├── scheduler/            # Job scheduler abstraction
│   ├── base.py           #   Scheduler ABC, ArrayJob model
│   ├── slurm.py          #   SLURM backend (sbatch + sacct)
│   ├── local.py          #   Local backend (worker pool)
│   └── factory.py        #   make_scheduler() factory
├── tools/                # External tool adapters
│   ├── glide.py          #   Schrödinger Glide I/O
│   ├── spacelight.py     #   SpaceLight command builder
│   └── ftrees.py         #   FTrees command builder
├── remote/               # Scripts for compute nodes
│   ├── train.py          #   Chemprop training entry point
│   ├── predict.py        #   Chemprop prediction entry point
│   ├── cluster.py        #   FPSim2 clustering entry point
│   └── prop_filter.py    #   RDKit property filter
└── workspace/            # Workspace management
    ├── layout.py         #   WorkDir (dual-root directory layout)
    ├── manifest.py       #   Manifest (JSON state tracking)
    └── logging_setup.py  #   Three-tier logging
```

### Key design decisions

- **Scheduler abstraction** — stages submit `ArrayJob` objects to a
  `Scheduler` interface; SLURM and local backends are interchangeable.
- **Dual-root workspace** — the database and logs live on fast local
  storage (`/wrk`), while stage artefacts go to NFS shared storage
  (`/data`) visible to compute nodes.
- **On-disk model registry** — trained models are stored as files under
  `models/v<N>/`; the legacy SQLite BLOB column is retained for backward
  compatibility with old `.dbsh` databases.
- **Acquisition SQL regression lock** — all SQL queries that drive
  compound selection are stored as constants and frozen by unit tests,
  ensuring reproducibility across versions.
- **Remote script invocation** — compute nodes run `remote/*.py` scripts
  via absolute path, avoiding import-time dependency on the orchestrator
  package inside the chemprop/FPSim2 conda environments.

## Migration from Legacy

Version 0.11 is a complete rewrite. The legacy code (v0.1–v0.10) is
preserved in the `legacy/` directory for reference. A `spacehasten-legacy-gui`
console script is available for one release as a compatibility shim for
the Tkinter GUI.

Key changes from legacy:

| Aspect | Legacy (≤ 0.10) | New (0.11) |
|---|---|---|
| Entry point | Bash wrapper → `spacehasten.py` → Tkinter GUI | `pip install` console script → argparse CLI |
| Configuration | `cfg.py` reads flat INI | Pydantic Settings with INI/TOML/CLI layering |
| Scheduler | Hardcoded SLURM bash scripts | Scheduler ABC with SLURM and local backends |
| Database | Raw `sqlite3` calls scattered across modules | Typed `Database` class with locked SQL |
| Workspace | Split between `$HOME/SPACEHASTEN/` and CWD | Single-root or dual-root `WorkDir` |
| Clustering | Self-extracting bash script | Pure Python module (`remote/cluster.py`) |
| Type safety | None | mypy strict on core modules |
| Testing | None | 178+ pytest tests (unit + integration) |

Existing `.dbsh` database files from legacy SpaceHASTEN are fully
compatible — the schema is unchanged and acquisition SQL is
byte-identical.

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/

# Type check
mypy
```

### Code style

- Line length: 100
- Target: Python 3.11
- Linter rules: PEP 8, PyFlakes, isort, pyupgrade, bugbear, simplify

## Citation

If you use SpaceHASTEN in your research, please cite:

> Kalliokoski, T. SpaceHASTEN — Accelerating Hit Identification from
> Nonenumerated Chemical Libraries by Combining Docking, Similarity
> Search, and Machine Learning. *J. Chem. Inf. Model.* **2025**, 65 (1),
> 125–132. [DOI: 10.1021/acs.jcim.4c01790](https://doi.org/10.1021/acs.jcim.4c01790)

## License

BSD 3-Clause License. Copyright © 2024 Orion Corporation. See [LICENSE](LICENSE) for details.

## Version History

- **0.11** (in development): Complete rewrite as a typed PEP 621 Python package.
  CLI-first interface, Pydantic configuration, scheduler abstraction,
  dual-root workspaces, on-disk model registry, 178+ tests.
- **0.10**: Faster Glide docking, Schrödinger 2026-1 support.
- **0.9**: Clustering acquisition strategy, command-line interface, progress bars.
- **0.8**: Compressed similarity search results, workspace archiving.
- **0.7**: Dynamic docking chunk sizing.
- **0.6**: Per-screen adjustable properties, alpha SGE support.
- **0.5**: Various GUI and usability fixes.
- **0.4**: Migrated from chemprop 1.x to 2.x.
- **0.3**: Compound ID handling improvements, NFS detection.
- **0.2**: Updated for new BioSolveIT tool versions.
- **0.1**: Initial release.

See [legacy/OLD_README.md](legacy/OLD_README.md) for the full legacy changelog.
