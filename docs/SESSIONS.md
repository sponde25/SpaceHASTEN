# SpaceHASTEN Rewrite — Per-Session Instructions

> **How to use this file.**
>
> Each session below is a self-contained prompt that you (the human) can paste
> verbatim into a fresh Copilot chat. The agent in that session should treat
> the prompt as the spec, read the references it points to, implement only
> what the session scope describes, and update [MIGRATION_STATUS.md](../MIGRATION_STATUS.md)
> at the end.
>
> **Hard rules across all sessions:**
> 1. Do not modify legacy code at the workspace root (`*.py`, `spacehasten`,
>    `sec_clustering.sh`) unless a session explicitly says so. The legacy tree
>    is the regression baseline.
> 2. The new package lives under `src/spacehasten/`. The new console-script
>    entry point is `spacehasten` (added by `pyproject.toml`); to avoid
>    colliding with the legacy bash launcher of the same name, the legacy
>    launcher and its `spacehasten.py` stay untouched until the cutover
>    session.
> 3. Preserve the legacy SQLite schema byte-for-byte (see
>    [CODEBASE_REFERENCE.md §A.1](CODEBASE_REFERENCE.md#a1-complete-sqlite-schema)).
>    Any new column must be additive and nullable.
> 4. Preserve the acquisition SQL verbatim
>    ([§A.6](CODEBASE_REFERENCE.md#a6-acquisition-strategy-sql-preserve-verbatim))
>    until tests are in place; refactor only after.
> 5. Use stdlib `argparse` for the CLI. Do not introduce Typer/Click.
> 6. End every session with: (a) `pytest -q` green, (b) `MIGRATION_STATUS.md`
>    updated, (c) `git add` + commit message naming the session.
>
> **Environment** (from `.github/copilot-instructions.md`): activate
> `spacehasten-quick` for orchestrator code; SLURM jobs use system conda
> `chemprop-2.1.2` / `fpsim2-0.7.3`. All terminal commands inside `tmux`.

---

## Session ordering

Phases 0–2 are foundation; phases 3–6 are stages from smallest to largest
blast radius; phase 7 is integration; phase 8+ is optional UX. Do not skip
ahead — later sessions assume artefacts from earlier ones.

| # | Session | Phase | Depends on |
|---|---|---|---|
| 1 | Project scaffolding | 0 | — |
| 2 | Schema fixture & legacy `.dbsh` baseline | 0 | 1 |
| 3 | `core/db.py` — typed DB layer | 1 | 2 |
| 4 | `core/molecules.py` & `config/` | 1 | 3 |
| 5 | `scheduler/base.py` + `scheduler/local.py` | 2 | 1 |
| 6 | `scheduler/slurm.py` | 2 | 5 |
| 7 | `workspace/` — layout, manifest, logging | 4 | 1 |
| 8 | `stages/training.py` + `remote/train.py` | 3 | 4, 6, 7 |
| 9 | `stages/prediction.py` + `remote/predict.py` | 3 | 8 |
| 10 | `stages/clustering.py` (+ port `sec_clustering.sh`) | 3 | 6, 7 |
| 11 | `tools/glide.py` + `stages/docking.py` | 3 | 6, 7 |
| 12 | `tools/spacelight.py` + `tools/ftrees.py` + `remote/prop_filter.py` + `stages/simsearch.py` | 3 | 9, 10, 11 |
| 13 | `stages/seeds.py`, `stages/export.py`, `stages/archive.py` | 3 | 11, 12 |
| 14 | `cli/main.py` — argparse subcommands | 5 | 13 |
| 15 | Port `verify_spacehasten.py` to new CLI; cutover | 7 | 14 |
| 16 | Quick wins on legacy tree (parallel-safe) | A | — |
| 16b | Mypy clean-up of `remote/` modules (parallel-safe) | A | 12 |
| 17 | TUI (Textual) | 6 | 14 |
| 18 | FastAPI dashboard (optional) | 6 | 14 |

---

## Session 1 — Project scaffolding

**Scope.** Create a working `pip install -e .` package skeleton with no
behaviour. No domain logic. No SLURM. No SQL.

**Read first**
- [MIGRATION_STATUS.md](../MIGRATION_STATUS.md) (top section).
- [REWRITE_PLAN.md §11.1](../REWRITE_PLAN.md) (proposed package layout).
- [.github/copilot-instructions.md](../.github/copilot-instructions.md).

**Do**
1. Create `pyproject.toml` (PEP 621). Project name `spacehasten`. Python
   `>=3.11`. Build backend `setuptools>=68`. Console script
   `spacehasten = spacehasten.cli.main:main` (the function will not exist
   yet — that is fine; pip install will still succeed, it only resolves at
   call time).
2. Runtime deps (pinned to ranges that work in `spacehasten-quick`):
   `pandas`, `rdkit`, `pydantic>=2`, `pydantic-settings`, `tomli;python_version<"3.11"`,
   `tomli-w`, `tqdm`, `rich`, `jinja2`. Do **not** add chemprop or
   lightning here — those are remote-node only.
3. Optional deps groups: `[dev] = pytest, pytest-cov, ruff, mypy, types-pyyaml`,
   `[tui] = textual`, `[web] = fastapi, uvicorn`.
4. Create `src/spacehasten/__init__.py` exporting `__version__ = "0.11.0.dev0"`.
5. Create empty package directories (each with `__init__.py`):
   `cli/`, `config/`, `core/`, `scheduler/`, `stages/`, `tools/`, `workspace/`,
   `remote/`, `workflows/`.
6. Create `tests/unit/`, `tests/integration/`, `tests/fixtures/` with
   `conftest.py` files (empty).
7. Add `pytest.ini` (or `[tool.pytest.ini_options]` in `pyproject.toml`):
   `testpaths = ["tests"]`, `addopts = "-q --strict-markers"`.
8. Add `ruff.toml` with line-length 100, target-version py311, lint rules
   `E,F,I,UP,B,SIM`. Add `[tool.mypy]` section with `strict = true` scoped to
   `src/spacehasten/{core,stages,scheduler,workspace}`.
9. Add `.gitignore` entries for `dist/`, `build/`, `*.egg-info/`, `.venv/`,
   `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `__pycache__/`.
10. Add `tests/unit/test_smoke.py` with `def test_imports(): import spacehasten; assert spacehasten.__version__`.
11. Run: `pip install -e ".[dev]"` then `pytest -q` then `ruff check src tests` then
    `python -c "import spacehasten; print(spacehasten.__version__)"`. All must succeed.

**Do NOT**
- Implement any stage, scheduler, or DB code.
- Touch any legacy `.py` at the workspace root.
- Add CI yaml (deferred until Phase 7).

**Done when**
- `pytest -q` is green (1 test).
- `ruff check src tests` is clean.
- `pip show spacehasten` works.

**Update MIGRATION_STATUS.md** marking Session 1 complete.

---

## Session 2 — Schema fixture & legacy `.dbsh` baseline

**Scope.** Produce the *frozen* schema reference (a SQL dump and a tiny
pre-populated `.dbsh` fixture) that all later DB tests will use. Nothing else.

**Read first**
- [CODEBASE_REFERENCE.md §A.1](CODEBASE_REFERENCE.md#a1-complete-sqlite-schema)
  and [§1 importseeds_functions.py](CODEBASE_REFERENCE.md).
- Legacy [importseeds_functions.py](../importseeds_functions.py) (full file).
- Legacy [functions.py:91-108](../functions.py) (`update_dbsh_properties`).
- Legacy [cluster_functions.py:42-56](../cluster_functions.py) (`process_cluster_results`).

**Do**
1. Write a small standalone script `scripts/build_fixture_dbsh.py` that uses
   only `sqlite3` (no SpaceHASTEN imports) and creates
   `tests/fixtures/legacy_baseline.dbsh` containing:
   - `data` table with **5 hand-chosen rows**: 2 docked seeds (dock_score
     set, dock_iteration=0), 1 simsearch hit with pred_score, 1 simsearch
     hit with both spacelight+ftrees+pred_score, 1 query (`query=1`).
   - `docking_param` with one row (BLOB = bytes from
     `examples.smi` — any non-empty bytes will do; this is just to verify
     blob roundtrip).
   - `docking_grid` with one row (BLOB = b"PK\x05\x06" + b"\x00"*18 — minimum
     legal empty zip).
   - `models` with one row (version=1, model_tar = b"dummytar").
   - `properties` table with the six default rows.
   - `clusters` table with 5 rows (one per data row).
   - The `idx_reghash` index.
   - SMILES: ethanol (`CCO`), benzene (`c1ccccc1`), toluene (`Cc1ccccc1`),
     phenol (`Oc1ccccc1`), aniline (`Nc1ccccc1`).
2. Run the script; commit `tests/fixtures/legacy_baseline.dbsh`.
3. Add `tests/fixtures/legacy_schema.sql` containing the exact `CREATE TABLE`
   and `CREATE INDEX` statements as a frozen reference (copy from
   [CODEBASE_REFERENCE.md §A.1](CODEBASE_REFERENCE.md#a1-complete-sqlite-schema)).
4. Add `tests/unit/test_fixture.py` that:
   - Opens `legacy_baseline.dbsh`.
   - Asserts `sqlite_master` contains the expected 6 tables and 1 index.
   - Asserts the column lists match `legacy_schema.sql` exactly (use
     `PRAGMA table_info`).
   - Asserts the 5-row data invariants.

**Do NOT**
- Add any SpaceHASTEN package code.
- Modify the legacy tree.

**Done when**
- `tests/fixtures/legacy_baseline.dbsh` exists.
- `pytest -q tests/unit/test_fixture.py` passes.

---

## Session 3 — `core/db.py` typed DB layer

**Scope.** A thin, typed wrapper around `sqlite3` that supports every SQL
operation the legacy code performs. No business logic. Use only stdlib
`sqlite3` (no SQLAlchemy, no SQLModel — adds a dep with no payoff for this
schema size).

**Read first**
- [CODEBASE_REFERENCE.md §A.1, §A.6](CODEBASE_REFERENCE.md).
- Legacy [functions.py](../functions.py) (every `c.execute` line — there are ~10).
- Legacy [importseeds_functions.py](../importseeds_functions.py) (CREATE statements + INSERTs).
- Legacy [docking_functions.py:13-42, :76-100](../docking_functions.py).
- Legacy [simsearch_functions.py:140-280](../simsearch_functions.py).
- Legacy [training_functions.py:55-90](../training_functions.py).
- Legacy [prediction_functions.py:118-196](../prediction_functions.py).
- Legacy [cluster_functions.py](../cluster_functions.py).
- Legacy [export_functions.py:32-57](../export_functions.py).

**Do**
1. Create `src/spacehasten/core/db.py` with:
   - `@dataclass(frozen=True)` row types: `DataRow`, `ClusterRow`, `PropertyRow`, `ModelRow`.
   - `class Database` opening a single long-lived connection (`Connection` is
     stored on the instance — *no per-call open/close* like the legacy code).
     Context-manager protocol (`__enter__`, `__exit__`).
   - `Database.create_schema()` running the exact CREATE statements from
     [legacy_schema.sql](../tests/fixtures/legacy_schema.sql).
   - Methods, each named after the operation, parameterised, **using `?`
     placeholders, never f-strings**:
     - `insert_seed_undocked(reghash, smiles, smilesid)`
     - `insert_seed_docked(reghash, smiles, smilesid, dock_score)` (sets `dock_iteration=0`)
     - `insert_simsearch_hit(reghash, smiles, smilesid, spacelight, ftrees, pred_score, simsearch_cycle)`
     - `update_dock_score(spacehastenid, dock_score, dock_iteration)`
     - `update_pred_score(spacehastenid, pred_score, pred_version)`
     - `mark_as_query(spacehastenid, cycle)`
     - `latest_model_version() -> int` (0 if none)
     - `latest_simsearch_cycle() -> int` (0 if none)
     - `latest_dock_iteration() -> int` (0 if none)
     - `store_model_blob(version, blob)`, `load_model_blob(version) -> bytes`
     - `store_dock_param(blob)`, `store_dock_grid(blob)`,
       `load_dock_param() -> bytes`, `load_dock_grid() -> bytes`
     - `replace_clusters(rows: Iterable[ClusterRow])` (DROP+CREATE+INSERT-many)
     - `replace_properties(props: PropertyRanges)` (DROP+CREATE+INSERT 6 rows;
       *match legacy column types exactly: `is_double` int, `min_limit` and
       `max_limit` stored as TEXT*)
     - `load_properties() -> PropertyRanges | None`
     - `select_queries_for_simsearch(source: Literal["docked","predicted"], strategy: Literal["greedy","clustering"], limit: int) -> list[tuple[str,int]]`
     - `select_compounds_to_dock(strategy: Literal["greedy","clustering"], limit: int) -> list[tuple[str,int]]`
     - `select_undocked_for_prediction() -> Iterator[tuple[str,int]]` (use
       fetchmany; legacy reads them all into memory)
     - `select_training_data(cutoff: float = 10.0) -> list[tuple[str,float]]`
     - `select_export_rows(cutoff: float) -> list[ExportRow]`
   - Each `select_*` and `update_*` method MUST emit the exact SQL string
     in [§A.6](CODEBASE_REFERENCE.md#a6-acquisition-strategy-sql-preserve-verbatim).
     Add a unit test that asserts the SQL text (via a `_SQL` class constant
     or `inspect`) — this is the regression lock.
2. Add `tests/unit/test_db_schema.py`:
   - Roundtrip: `Database(path).create_schema()` followed by opening with
     raw `sqlite3` produces the same `sqlite_master` rows as
     `tests/fixtures/legacy_schema.sql`.
3. Add `tests/unit/test_db_acquisition.py`:
   - Open `tests/fixtures/legacy_baseline.dbsh` with `Database`.
   - Run each `select_*` method, assert exact result tuples (hand-computed
     from the 5 fixture rows).
4. Add `tests/unit/test_db_sql_locked.py`:
   - For each acquisition method, assert the SQL string is byte-identical to
     [§A.6](CODEBASE_REFERENCE.md#a6-acquisition-strategy-sql-preserve-verbatim).

**Do NOT**
- Add any I/O outside `sqlite3`.
- Import `cfg` or other legacy modules.
- Touch the legacy tree.

**Done when**
- All three test files green.
- `mypy src/spacehasten/core` clean.

---

## Session 4 — `core/molecules.py` and `config/`

**Scope.** Pure functions for RDKit hashing/dedup, and a Pydantic-Settings
based config system that supersedes `cfg.py` and `spacehasten.ini`.

**Read first**
- Legacy [functions.py:182-207](../functions.py) (`mol2hash`, `cxsmi2smi`).
- Legacy [cfg.py](../cfg.py) (full file).
- [CODEBASE_REFERENCE.md §A.2](CODEBASE_REFERENCE.md#a2-spacehastenini-key-inventory).
- [REWRITE_PLAN.md §11.6](../REWRITE_PLAN.md).

**Do**
1. `src/spacehasten/core/molecules.py`:
   - `tautomer_hash(smiles: str) -> str | None` — wraps `RegistrationHash.GetMolLayers(mol)[HashLayer.TAUTOMER_HASH]`.
   - `canonical_smiles(smiles: str) -> str | None` — `Chem.MolToSmiles(Chem.MolFromSmiles(...))`.
   - `parse_cxsmiles(line: str) -> tuple[str, str] | None` — `(canonical_smiles, id)`.
   - All return `None` on parse failure (no exceptions). Pure, no I/O.
2. `src/spacehasten/config/properties.py`:
   - Pydantic `PropertyRanges` model with `mw, slogp, hba, hbd, rotbonds, tpsa`,
     each a `Range(min: float, max: float)`. `hba/hbd/rotbonds` use int
     ranges (validators).
   - `PropertyRanges.from_toml(path)` and `.to_toml(path)`.
   - Default values match cfg.py defaults
     ([§A.2 Properties](CODEBASE_REFERENCE.md#a2-spacehastenini-key-inventory)).
3. `src/spacehasten/config/settings.py`:
   - Pydantic `BaseSettings` subclasses for `[General]`, `[Paths]`,
     `[Slurm]`, `[SGE]`, mirroring cfg.py keys.
   - `class Settings` aggregating them, with class methods:
     - `Settings.load(*, ini_path: Path | None = None, toml_path: Path | None = None, cli_overrides: Mapping[str, Any] = {}) -> Settings` — merges in priority cli > toml > ini > defaults.
     - `Settings.dump_toml(path)`.
   - **Critical**: do NOT call `shutil.which` / path-existence checks in
     `__init__` (legacy bug). Validation is opt-in: a `validate_install()`
     method users call explicitly.
4. Tests:
   - `tests/unit/test_molecules.py` — table-driven tautomer hash equality
     (e.g. `O=C(O)C` and `OC(=O)C` produce same hash).
   - `tests/unit/test_settings.py` — load a fixture `tests/fixtures/spacehasten.ini`
     copied from a real install; assert all attributes parsed correctly;
     assert TOML override wins over INI; assert CLI override wins over TOML.

**Do NOT**
- Touch sqlite (already in core/db).
- Implement scheduler-string assembly (that goes to `scheduler/`).
- Validate installs at construction time.

**Done when**
- All tests green; mypy clean on `core/`, `config/`.

---

## Session 5 — `scheduler/base.py` and `scheduler/local.py`

**Scope.** Define the scheduler abstraction and a local-subprocess
implementation. No SLURM yet. Local scheduler enables fast integration
tests.

**Read first**
- [REWRITE_PLAN.md §11.3](../REWRITE_PLAN.md).
- [CODEBASE_REFERENCE.md §A.4](CODEBASE_REFERENCE.md#a4-scheduler-job-inventory-six-job-types).
- Legacy [scheduler_functions.py](../scheduler_functions.py) (full file).

**Do**
1. `src/spacehasten/scheduler/base.py`:
   - `@dataclass(frozen=True) class ArrayJob`: `name: str`, `workdir: Path`,
     `array_size: int`, `max_concurrent: int`, `cpus_per_task: int`,
     `gpus: int = 0`, `exclusive: bool = False`,
     `env_setup: list[str] = []`, `command_template: str` (rendered with
     `${TASK_ID}`), `depends_on: list["ArrayHandle"] = []`.
   - `@dataclass class ArrayHandle`: `job_id: str`, `name: str`,
     `array_size: int`, `workdir: Path`.
   - `class ArrayStatus`: enum-like dataclass with per-task states.
   - `class Scheduler(ABC)`: `submit_array(job) -> ArrayHandle`,
     `wait(handle, on_progress) -> ArrayResult`, `cancel(handle) -> None`,
     `status(handle) -> ArrayStatus`. Default `wait()` calls `status()` in a
     loop with exponential backoff (start 1s, cap 30s).
2. `src/spacehasten/scheduler/local.py`:
   - `class LocalScheduler(Scheduler)`: runs each task as
     `subprocess.Popen(["bash","-c", body], cwd=workdir, env={...,"TASK_ID":str(i)})`.
   - Honours `max_concurrent` via a worker pool.
   - Captures stdout/stderr to `logs/local/<jobname>/task-NNNN.{out,err}`.
   - `status()` returns based on `Popen.poll()`.
3. Tests:
   - `tests/unit/test_local_scheduler.py`: submit a 4-task array running
     `echo $TASK_ID > out_${TASK_ID}.txt`, wait, assert files exist.
   - Failure case: one task `exit 1` → status reports it as failed; wait
     returns failed indices.

**Do NOT**
- Implement SLURM (next session).
- Render bash via Jinja yet (also next session).

**Done when**
- `pytest tests/unit/test_local_scheduler.py` green.

---

## Session 6 — `scheduler/slurm.py`

**Scope.** SLURM submission + `sacct` polling + dependency chains. Single
shared Jinja template for the script body. **Replaces** `scheduler_functions.py`'s
six near-duplicate writers.

**Read first**
- Legacy [scheduler_functions.py](../scheduler_functions.py) (every line).
- [CODEBASE_REFERENCE.md §A.4](CODEBASE_REFERENCE.md#a4-scheduler-job-inventory-six-job-types).
- [REWRITE_PLAN.md §11.3](../REWRITE_PLAN.md).

**Do**
1. `src/spacehasten/scheduler/_template.sh.j2` — Jinja2 template producing
   the same bash structure as legacy: header (`#SBATCH ...`), conda
   activation lines, per-task scratch setup, command, copy results back,
   `touch jobdone-<name>-CPU${SLURM_ARRAY_TASK_ID}` for backward
   compatibility (still emit it during the transition; SLURM polling is the
   real signal).
2. `src/spacehasten/scheduler/slurm.py`:
   - `class SlurmScheduler(Scheduler)`.
   - `submit_array`:
     - Renders template into `<workdir>/submit_<name>.sh`.
     - Runs `subprocess.run(["sbatch","--parsable", str(script)], capture_output=True, check=True)`; parses `<job_id>`.
   - `status`:
     - Runs `sacct -j <id> -X --format=JobID,State,ExitCode -P --noheader -n`.
     - Parses each task's state into `RUNNING|COMPLETED|FAILED|TIMEOUT|PENDING|CANCELLED`.
   - `cancel`: `scancel <job_id>`.
   - `wait`: poll `status` (delegating to base class default), report progress
     via callback after each poll.
   - Supports `depends_on=[ArrayHandle, ...]` → `--dependency=afterok:<id1>:<id2>`.
   - Treats any non-COMPLETED final state as failure, returns failed task
     indices in `ArrayResult`.
3. `src/spacehasten/scheduler/factory.py`:
   - `def make_scheduler(kind: Literal["auto","slurm","local"], settings: Settings) -> Scheduler` — `auto` returns `Slurm` if `which("sbatch")` else `Local`.
4. Tests (`tests/unit/test_slurm_scheduler.py`):
   - Use `monkeypatch` to replace `subprocess.run` with a fake that records
     calls. Assert the rendered script matches a snapshot
     (`tests/fixtures/expected_submit_*.sh`); assert `sbatch --parsable` is
     called with the right path; assert `sacct` polling parses
     `1234567_1|COMPLETED|0:0` correctly.

**Do NOT**
- Submit a real job.
- Touch any stage code yet.

**Done when**
- Snapshot tests pass; mypy clean on `scheduler/`.

---

## Session 7 — `workspace/` (layout, manifest, logging)

**Scope.** Single-root workspace layout; JSON manifest; three-tier log
layout (master, per-stage, optional per-task).

**Read first**
- [REWRITE_PLAN.md §11.4, §11.6a](../REWRITE_PLAN.md).
- [CODEBASE_REFERENCE.md §A.3](CODEBASE_REFERENCE.md#a3-file-system-path-inventory).

**Do**
1. `src/spacehasten/workspace/layout.py`:
   - `class WorkDir`. Constructor `WorkDir(root: Path)`.
   - Methods (return `Path`): `dbsh()`, `simsearch_dir(cycle: int)`,
     `docking_dir(iteration: int)`, `model_dir(version: int)`,
     `clustering_dir()`, `archive_dir()`, `logs_dir()`,
     `slurm_logs_dir(job_name: str)`, `manifest_path()`, `props_path()`.
   - `WorkDir.bootstrap(root, name)` creates root + `logs/` + `models/`
     and writes an empty manifest.
   - Disk-policy check: `WorkDir.warn_if_wrong_disk()` warns when root is on
     `/wrk` and suggests `/data/$USER/SPACEHASTEN/<name>/`.
2. `src/spacehasten/workspace/manifest.py`:
   - Pydantic `Manifest` model with: `schema_version: int = 1`,
     `name: str`, `created_at: datetime`,
     `stages: dict[str, StageRecord]`,
     `runs: list[RunRecord]` (each run: command line, args, started_at, ended_at, status).
   - `Manifest.load(path)`, `Manifest.save(path)` (atomic via tempfile +
     rename).
   - `Manifest.record_stage_start(stage, params)` /
     `.record_stage_finish(stage, status, scheduler_job_id)`.
3. `src/spacehasten/workspace/logging_setup.py`:
   - `configure_logging(workdir: WorkDir, level=INFO, console=True)`:
     - Master log: `RotatingFileHandler(workdir.logs_dir()/"spacehasten.log", maxBytes=20MB, backupCount=5)`.
     - Console: `RichHandler` if `rich` available, else `StreamHandler`.
   - `stage_log_context(workdir, stage_name)` — context manager that adds a
     `FileHandler` to `logs/<stage>-<n>.log` for the duration.
4. Tests:
   - `tests/unit/test_workdir.py` — bootstrap, paths, idempotency.
   - `tests/unit/test_manifest.py` — roundtrip, stage record append.
   - `tests/unit/test_logging.py` — assert master + stage handlers attach
     and write the expected files.

**Done when**
- All tests green; mypy clean on `workspace/`.

---

## Session 8 — Training stage

**Scope.** Port `model_runner_train.py` to `remote/train.py` (no changes
to the chemprop API call); write `stages/training.py` that uses
`Database`, `Scheduler`, and `WorkDir`. **Switch model storage from BLOB
to on-disk registry**, but keep a compatibility shim that can also load
legacy BLOBs.

**Read first**
- Legacy [training_functions.py](../training_functions.py) (full).
- Legacy [model_runner_train.py](../model_runner_train.py) (full).
- [CODEBASE_REFERENCE.md §A.4 row 5, §1 model_runner_train.py](CODEBASE_REFERENCE.md).

**Do**
1. `src/spacehasten/remote/train.py` — copy `model_runner_train.py`
   verbatim except: change argparse to use `argparse.ArgumentParser` (no
   semantic change), fix any obvious python-2-isms. Keep the chemprop API
   calls byte-identical.
2. `src/spacehasten/stages/training.py`:
   - `def train(db: Database, workdir: WorkDir, scheduler: Scheduler, settings: Settings, *, cutoff: float = 10.0) -> int`
     - Pull rows via `db.select_training_data(cutoff)`.
     - Write CSV under `workdir.model_dir(next_version)/train.csv`.
     - Build an `ArrayJob` of size 1 with command:
       `python3 -m spacehasten.remote.train <csv> <model_dir> --batch-size ... --final-lr ...`
       (passing every hyper-param from `settings.train_*`).
     - Submit; wait; on success, return `next_version`.
     - **Do NOT BLOB the model.** Just leave `model_dir/model_0/pytorch_model.bin`
       in place. Update the `models` table with `(version, b"")` for legacy
       schema compatibility. Add a `Manifest.record_model(version, model_dir)`
       entry which is the source of truth.
     - Add a `Database.load_model_path(version, workdir)` helper: returns
       `workdir.model_dir(v)/model_0/pytorch_model.bin` if it exists; else
       extracts `models.model_tar` BLOB to that path (back-compat).
3. Tests (`tests/integration/test_training_local.py`):
   - Use `LocalScheduler` and a stub `remote/train.py` that just touches
     the expected output file (skipping real chemprop). Verify the stage
     wires inputs/outputs correctly. Real chemprop training is covered by
     Session 15 verify smoke test.

**Do NOT**
- Re-implement chemprop logic.
- Submit a real GPU job in tests.

**Done when**
- Local-scheduler integration test green.

---

## Session 9 — Prediction stage

**Scope.** Port `model_runner_predict.py` to `remote/predict.py`; write
`stages/prediction.py` orchestrating chunked prediction over undocked rows.

**Read first**
- Legacy [prediction_functions.py](../prediction_functions.py) (full).
- Legacy [model_runner_predict.py](../model_runner_predict.py) (full).
- [CODEBASE_REFERENCE.md §A.4 row 3](CODEBASE_REFERENCE.md#a4-scheduler-job-inventory-six-job-types).

**Do**
1. `src/spacehasten/remote/predict.py` — port verbatim.
2. `src/spacehasten/stages/prediction.py`:
   - `predict_undocked(db, workdir, scheduler, settings, *, model_version: int, chunk_size: int = 12345) -> int`:
     - Stream undocked rows, write `predict_<i>.csv` chunks under
       `workdir.simsearch_dir(latest_cycle).joinpath("PREDICT")`.
     - Build an array job; one task per chunk; command runs
       `python3 -m spacehasten.remote.predict ...`.
     - On success, ingest `predicted_predict_*.csv` and call
       `db.update_pred_score` for each row (use a single
       transaction).
3. Tests:
   - Local-scheduler integration with a stub predict that emits a fixed
     `pred_score=-7.5` for every row in input. Assert DB rows updated.

**Done when**
- Integration test green; legacy `predict_dock` and `chunkpredict.py` are
  *not* ported (dead code per [CODEBASE_REFERENCE.md §1 chunkpredict.py](CODEBASE_REFERENCE.md)).

---

## Session 10 — Clustering stage

**Scope.** Reproduce sphere-exclusion clustering. Decision: **port the
embedded Python in `sec_clustering.sh` to a real Python module**
(`spacehasten.remote.cluster`), but keep the shell script around as a
drop-in fallback.

**Read first**
- Legacy [cluster_functions.py](../cluster_functions.py).
- Legacy [sec_clustering.sh](../sec_clustering.sh) — extract the embedded
  Python (the file is self-extracting; running it once produces
  `sec_clustering.py`).
- [CODEBASE_REFERENCE.md §1 sec_clustering.sh](CODEBASE_REFERENCE.md).

**Do**
1. `src/spacehasten/remote/cluster.py` — port the 7-step pipeline using
   RDKit `LeaderPicker` + FPSim2. Keep the same CLI:
   `python3 -m spacehasten.remote.cluster <smi>` produces `clustering.csv`.
2. `src/spacehasten/stages/clustering.py`:
   - `cluster(db, workdir, scheduler, settings) -> None`:
     - Stream `(smiles, spacehastenid)` from `db` to
       `workdir.clustering_dir()/clustering_input.smi.gz`.
     - Submit single-job array (size 1, exclusive, CPU-heavy).
     - On success, ingest `clustering.csv` via `db.replace_clusters`.
3. Tests with a 100-row synthetic SMILES set; local scheduler.

**Done when**
- Integration test green; results compared row-count-wise to a recorded
  baseline.

---

## Session 11 — Docking stage

**Scope.** Largest external integration. `tools/glide.py` builds Phase
`.inp` and Glide `.in` from templates + DB blobs. `stages/docking.py`
chunks, submits, and ingests results.

**Read first**
- Legacy [docking_functions.py](../docking_functions.py) (full).
- [CODEBASE_REFERENCE.md §A.4 row 2, §A.5](CODEBASE_REFERENCE.md).

**Do**
1. `src/spacehasten/tools/glide.py`:
   - `write_phase_inp(path, ...)` — port `write_confgen_file`.
   - `write_glide_in(path, dock_param_blob, dock_dir)` — port
     `write_docking_file`; strip `LIGANDFILE`/`GRIDFILE`, rewrite them.
   - `parse_glide_csv(csv_path) -> dict[str, float]` — title → min(score)
     reducer (legacy uses pandas groupby).
2. `src/spacehasten/stages/docking.py`:
   - `dock(db, workdir, scheduler, settings, *, top_n: int, strategy: Literal["greedy","clustering"]) -> int`:
     - Pull compounds via `db.select_compounds_to_dock`.
     - Shuffle, chunk into `DOCKING_CHUNK=1000` per task; cap chunks at
       configured CPU count.
     - Per chunk: write `.smi`, `.inp`, `glide_<chunk>.in`. Extract grid
       blob once to `glide_grid.zip`.
     - Build array job; command body matches [CODEBASE_REFERENCE.md §A.4 row 2](CODEBASE_REFERENCE.md#a4-scheduler-job-inventory-six-job-types).
     - On success, parallel-extract `results-*.tar.gz`, parse `glide_*.csv`,
       call `db.update_dock_score(...)` in one transaction.
     - Return the new `dock_iteration`.
3. Tests:
   - `test_glide_io.py`: write/parse roundtrip; CSV → score map matches a
     hand-computed expected dict.
   - Local-scheduler integration with a stub Glide that emits a fake
     `glide_<chunk>.csv`.

**Done when**
- Tests green; SQL writes for `dock_score`/`dock_iteration` match legacy
  byte-for-byte.

---

## Session 12 — Simsearch stage

**Scope.** SpaceLight + FTrees adapters; property filter as remote script;
two-phase array (search → control); result aggregation and dedup.

**Read first**
- Legacy [simsearch_functions.py](../simsearch_functions.py) (full).
- Legacy [control.py](../control.py).
- [CODEBASE_REFERENCE.md §A.4 rows 1 & 4, §A.5](CODEBASE_REFERENCE.md).

**Do**
1. `src/spacehasten/tools/spacelight.py`, `tools/ftrees.py`:
   - `class SpacelightAdapter`, `FTreesAdapter`. Each: `command_for(query, space, output, *, max_results, similarity, threads) -> list[str]`.
2. `src/spacehasten/remote/prop_filter.py` — port `control.py`. Same CLI.
3. `src/spacehasten/stages/simsearch.py`:
   - `simsearch(db, workdir, scheduler, settings, *, source, strategy, top_n) -> int`:
     - Phase A: pick queries (`db.select_queries_for_simsearch`), write
       `queries_<name>.smi`, mark them with cycle, submit search array,
       wait.
     - Phase B: aggregate result CSVs (max similarity per smiles per method),
       chunk into `control/control_<i>.smi.gz`, write `control.param`,
       extract latest model BLOB to `model_dir`, submit control array.
     - Phase C: ingest `predicted_propoutput_*.csv`, dedup by reghash
       (against `data.reghash` index), insert via
       `db.insert_simsearch_hit`.
     - Optionally trigger `clustering.cluster` (mirrors legacy).
   - Return new cycle number.
4. Tests:
   - `test_search_adapters.py`: command-line shape.
   - Local-scheduler integration with stub spacelight/ftrees binaries that
     emit fixed-shape CSVs; assert correct number of inserted rows after
     dedup.

**Done when**
- Integration green; dedup invariant holds (no duplicate `reghash`).

---

## Session 13 — Seeds, export, archive stages

**Scope.** Smaller stages.

**Read first**
- Legacy [importseeds_functions.py](../importseeds_functions.py),
  [export_functions.py](../export_functions.py),
  [export_poses.py](../export_poses.py),
  [archive_functions.py](../archive_functions.py).

**Do**
1. `src/spacehasten/stages/seeds.py`:
   - `import_seeds(db, workdir, *, smi_path: Path | None = None, csv_path: Path | None = None, dock_params_path: Path, dock_grid_path: Path, props: PropertyRanges)`.
   - Use `core.molecules.tautomer_hash` over `mp.Pool` (mirror legacy
     parallelism).
   - Insert rows via `db.insert_seed_*`.
   - Store dock_param/grid blobs.
   - Calls `training.train` and `clustering.cluster` only when invoked
     from a SMILES seed path AND `auto_train=True` (default True; expose
     a CLI flag to disable). The auto-dock-then-train flow stays.
2. `src/spacehasten/stages/export.py`:
   - `export_csv(db, output: Path, *, cutoff: float)`.
   - `export_poses(db, workdir, output: Path, *, cutoff: float, iteration: int | None)`:
     uses `subprocess.run(["$SCHRODINGER/run", str(EXPORT_POSES_PY), ...])`;
     `EXPORT_POSES_PY` is the legacy file at the workspace root for now
     (path resolved via `Settings`).
3. `src/spacehasten/stages/archive.py`:
   - `archive_create(workdir, *, bundle: bool = False)` — produces
     `<name>.archived-spacehasten` (link mode) or
     `<name>.archived-spacehasten.tgz` (bundle mode, self-contained:
     embed `.dbsh`, models, simsearch & docking dirs).
   - `archive_restore(archive_path, target_workdir)`.
   - `archive_extract(bundle_path, target_workdir)` — inverse of `bundle`.
   - `archive_clean(workdir)`.

**Done when**
- Tests green for each (`test_seeds.py`, `test_export.py`, `test_archive.py`).

---

## Session 14 — `cli/main.py`

**Scope.** Stitch all stages behind argparse subcommands.

**Read first**
- Legacy [cmdline.py](../cmdline.py) (legacy flags, for migration mapping).
- [REWRITE_PLAN.md §11.2](../REWRITE_PLAN.md).

**Do**
1. `src/spacehasten/cli/main.py`:
   - `def main(argv: list[str] | None = None) -> int`.
   - Top-level parser; `add_subparsers(dest="command", required=True)`.
   - Subcommands (each maps 1:1 to a `stages.*` function):
     `init`, `import-seeds`, `train`, `predict`, `search`, `dock`, `cluster`,
     `screen`, `export csv`, `export poses`, `archive create|extract|restore|clean`,
     `status`, `resume`, `verify`.
   - Global options: `--db`, `--config`, `--scheduler {auto,slurm,local}`,
     `--partition`, `--scratch`, `--log-level`, `--json`.
   - The `screen` command is the legacy macro: train? → simsearch(docked) →
     simsearch(predicted) ×2 → dock; one round per `--rounds`.
2. `src/spacehasten/cli/_common.py`: helpers to build `Database`,
   `WorkDir`, `Scheduler`, `Settings` from global args; logging setup.
3. Tests:
   - `tests/unit/test_cli.py`: each subcommand parses; required-arg
     validation; `--help` exits 0.
   - `tests/integration/test_cli_screen_local.py`: full screen with stubs.

**Done when**
- `spacehasten --help` works after `pip install -e .`.

---

## Session 15 — Verify port + cutover

**Scope.** Port `verify_spacehasten.py` to the new CLI; run end-to-end on
real SLURM. After it passes, mark legacy cutover.

**Read first**
- Legacy [verify_spacehasten.py](../verify_spacehasten.py).

**Do**
1. `src/spacehasten/cli/verify.py` — runs the same 6 checks via the new
   stage APIs against a temp workdir. Re-uses `examples.smi`,
   `test_dock.in`, `grid-test_dock.zip` from legacy tree.
2. Mark `gui.py`, `spacehasten.py` (legacy entrypoint), `cmdline.py`,
   `*_functions.py`, `chunkpredict.py`, `control.py` (legacy versions),
   `model_runner_*.py` (legacy versions), `cfg.py` and the `spacehasten`
   bash launcher as deprecated by:
   - Moving them to a top-level `legacy/` directory.
   - Adding a `legacy/README.md` explaining they are kept for one release.
3. Update install_spacehasten.py to install from the new package; keep one
   `spacehasten-legacy-gui` console script for the Tk GUI for one release.

**Done when**
- `spacehasten verify` passes on the cluster.
- One full screening round produces equivalent dbsh content (compare
  `data` row counts and score histograms) to a parallel legacy run.

---

## Session 16 — Quick-win patches on legacy tree (parallel-safe)

**Scope.** Tiny fixes to the legacy code that reduce drift while the
rewrite proceeds. Safe to do at any time after Session 1; no dependencies.

**Do**
- [cfg.py:324](../cfg.py): `elif self.SCHEDULER != "SGE":` → `== "SGE"`.
- Replace `os.system("sbatch ...")` with `subprocess.run([args.c.SCHEDULER_SUBMIT, ...], check=True)` in:
  - [docking_functions.py:160](../docking_functions.py)
  - [training_functions.py:67](../training_functions.py)
  - [simsearch_functions.py:156, :253](../simsearch_functions.py)
  - [prediction_functions.py:170](../prediction_functions.py)
- Add a `Path(dbname).stem` helper to [functions.py](../functions.py); replace
  the `dbname.split("/")[-1].split(".")[0]` idiom call sites.

**Done when**
- `verify_spacehasten.py` still passes on the cluster.

---

## Session 16b — Mypy clean-up of `remote/` modules (parallel-safe)

**Scope.** Eliminate the six pre-existing mypy errors in
`src/spacehasten/remote/{predict,train,cluster}.py` that surface when the
caller widens the strict scope from `core,stages,scheduler,workspace` to
include `remote/`. None of the errors block runtime; they are
import-stub gaps and one untyped union access. Safe to run at any time
after Session 12; no dependencies on later sessions.

**Read first**
- The six mypy errors as currently reported by:
  ```bash
  mypy src/spacehasten/core src/spacehasten/stages src/spacehasten/scheduler \
       src/spacehasten/workspace src/spacehasten/tools src/spacehasten/remote
  ```
  Expected output (Session 12 baseline):
  ```
  src/spacehasten/remote/cluster.py:162: error: Skipping analyzing "FPSim2": module is installed, but missing library stubs or py.typed marker  [import-untyped]
  src/spacehasten/remote/cluster.py:205: error: Skipping analyzing "FPSim2.io": module is installed, but missing library stubs or py.typed marker  [import-untyped]
  src/spacehasten/remote/train.py:24: error: Skipping analyzing "chemprop": module is installed, but missing library stubs or py.typed marker  [import-untyped]
  src/spacehasten/remote/predict.py:24: error: Skipping analyzing "chemprop": module is installed, but missing library stubs or py.typed marker  [import-untyped]
  src/spacehasten/remote/predict.py:98: error: Item "list[Any]" of "Any | list[Any]" has no attribute "cpu"  [union-attr]
  src/spacehasten/remote/predict.py:98: error: Item "None" of "list[Any] | list[list[Any]] | None" has no attribute "__iter__" (not iterable)  [union-attr]
  ```
- [src/spacehasten/remote/predict.py:80-110](../src/spacehasten/remote/predict.py)
  for the union-attr context (`trainer.predict(...)` returns
  `list[Tensor] | list[list[Tensor]] | None`).
- The Session 1 strict-mypy scope contract at [pyproject.toml](../pyproject.toml).

**Do**
1. Suppress the third-party import-stub errors at the *import site* —
   not via a global `[[tool.mypy.overrides]]` block, so the suppression
   is grep-able and we keep an audit trail when stubs eventually appear.
   - `src/spacehasten/remote/cluster.py`:
     - Line 162 (`import FPSim2 ...`): append `# type: ignore[import-untyped]`.
     - Line 205 (`from FPSim2.io import ...`): append `# type: ignore[import-untyped]`.
   - `src/spacehasten/remote/train.py:24` and
     `src/spacehasten/remote/predict.py:24` (`from chemprop import ...`):
     append `# type: ignore[import-untyped]`.
2. Fix the `predict.py:98` union-attr error properly (do NOT just
   `# type: ignore` it). The chemprop/Lightning return type for
   `trainer.predict(...)` widens to
   `list[Tensor] | list[list[Tensor]] | None`. Tighten the runtime
   handling so mypy can narrow:
   - Add an explicit `if batch_preds is None: raise RuntimeError("trainer.predict returned None")` immediately after the call.
   - Flatten the (potentially nested) list into a flat `list[Tensor]`
     before the `np.concatenate` step:
     ```python
     def _flatten(preds: list[Tensor] | list[list[Tensor]]) -> list[Tensor]:
         if preds and isinstance(preds[0], list):
             return [t for sub in preds for t in sub]  # type: ignore[misc]
         return cast(list[Tensor], preds)
     ```
     Use `from typing import cast` and import `Tensor` from `torch`.
     Keep the comprehension `[p.cpu().numpy() for p in flat_preds]`.
   - Verify the runtime behaviour is unchanged on a small CPU-only
     prediction (`pytest tests/integration/test_prediction_local.py -q`
     should still pass — the stub remote-predict isn't exercised here,
     so just smoke-test by importing the module:
     `python3 -c "import spacehasten.remote.predict"`).
3. Re-run mypy across the full `remote/` scope and confirm zero errors:
   ```bash
   rm -rf .mypy_cache
   mypy src/spacehasten/core src/spacehasten/stages src/spacehasten/scheduler \
        src/spacehasten/workspace src/spacehasten/tools src/spacehasten/remote
   # → Success: no issues found in 27 source files
   ```
4. Widen the strict scope: in `pyproject.toml`, the existing
   `[tool.mypy]` block (per Session 1) restricts strict mode to
   `src/spacehasten/{core,stages,scheduler,workspace}`. Append
   `tools` and `remote` to that list so future drift is caught
   automatically. Do **not** add `cli/` until Session 14 lands.

**Do NOT**
- Add a workspace-wide `ignore_missing_imports = true`. The whole point
  is per-import auditability.
- "Fix" the union-attr by silencing it with a blanket `# type: ignore`.
- Touch any of the legacy `*.py` files at the workspace root.
- Bump dependency versions or add new ones.

**Done when**
- The full mypy command above prints `Success: no issues found in 27 source files`.
- `pytest -q` is still green (130+ tests).
- `ruff check src tests` is still clean.
- `MIGRATION_STATUS.md` row 16b is marked `done` and the decision is
  logged in the decisions table.

---

## Session 17 — Textual TUI (optional)

Defer until 14 is stable. Build a TUI over the stage APIs only. No
orchestration in the UI. Out of scope for the migration's critical path.

---

## Session 18 — FastAPI dashboard (optional)

Read-only status + simple submit form. Out of scope until 17 is stable.
