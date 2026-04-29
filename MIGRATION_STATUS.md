# SpaceHASTEN Rewrite — Migration Status

**Started:** 2026-04-28
**Target package:** `spacehasten` (PEP 621, src layout)
**Legacy tree:** workspace root (do not modify until Session 15 cutover)

## Documents

- [REWRITE_PLAN.md](REWRITE_PLAN.md) — strategy and architecture decisions
- [docs/CODEBASE_REFERENCE.md](docs/CODEBASE_REFERENCE.md) — frozen
  reference of the legacy code (per-file, schema, paths, scheduler jobs,
  external tool commands)
- [docs/SESSIONS.md](docs/SESSIONS.md) — per-session prompts

## How sessions work

1. Pick the next not-started session from the table below.
2. Open a fresh Copilot chat **in this workspace**.
3. Paste the matching session block from
   [docs/SESSIONS.md](docs/SESSIONS.md) as your message.
4. The agent in that session implements only that session's scope, runs
   tests, and edits this file (the row below) to mark it complete.
5. Commit. The next session starts the same way.

**Hard rules** for every session — see top of [docs/SESSIONS.md](docs/SESSIONS.md).
The non-negotiables are:

- Do not touch the legacy tree until Session 15 (Session 16 is a separate
  parallel-safe quick-wins pass).
- Preserve the SQLite schema and acquisition SQL byte-for-byte until
  fixture-locked tests exist.
- Stdlib argparse only.
- `pytest -q` must be green at the end of every session.

## Session checklist

| # | Session | Status | Owner | Notes |
|---|---|---|---|---|
| 1 | Project scaffolding | done | | `pyproject.toml`, src layout, ruff/mypy/pytest |
| 2 | Schema fixture & legacy `.dbsh` baseline | done | | `tests/fixtures/legacy_baseline.dbsh` |
| 3 | `core/db.py` — typed DB layer | done | | locks acquisition SQL; 24 tests; mypy clean |
| 4 | `core/molecules.py` & `config/` | done | | RDKit hashing + Pydantic-Settings (INI/TOML/CLI merge); opt-in `validate_install`; 22 tests; mypy clean |
| 5 | `scheduler/base.py` + `scheduler/local.py` | done | | `ArrayJob`/`ArrayHandle`/`ArrayStatus`/`ArrayResult` + `Scheduler` ABC with backoff `wait()`; `LocalScheduler` with worker pool, per-task logs, cancel; 7 tests; ruff/mypy clean |
| 6 | `scheduler/slurm.py` | done | | `SlurmScheduler` (sbatch `--parsable` + sacct polling + afterok deps), shared Jinja template, `make_scheduler` factory; snapshot fixtures locked; 14 tests; ruff/mypy clean |
| 7 | `workspace/` — layout, manifest, logging | done | | `WorkDir` paths + bootstrap + /wrk warning, Pydantic `Manifest` (atomic save, stage/run records), three-tier logging (rotating master + Rich/Stream console + per-stage `FileHandler`); 22 tests; ruff/mypy clean |
| 8 | `stages/training.py` + `remote/train.py` | done | | on-disk model registry (manifest source of truth, legacy `models` BLOB shim retained); `Database.load_model_path` with BLOB fallback; `Manifest.record_model`; local-scheduler integration test with stub remote/train; 103 tests; ruff/mypy clean |
| 9 | `stages/prediction.py` + `remote/predict.py` | done | | chunked array prediction; `${TASK_ID}`-driven `predict_<i>.csv`/`predicted_predict_<i>.csv` under `simsearch/cycle<N>/PREDICT/`; reuses `Database.load_model_path` (BLOB fallback) and `apply_pred_scores` single-transaction bulk update; local-scheduler integration test with stub remote/predict; 105 tests; ruff/mypy clean. Legacy `predict_dock` and `chunkpredict.py` intentionally not ported (dead code). |
| 10 | `stages/clustering.py` (port `sec_clustering.sh`) | done | | `remote/cluster.py` ports the 7-step sphere-exclusion pipeline (Morgan-2/1024 + RDKit `LeaderPicker` at distance 0.7 + FPSim2 similarity search at >= 0.30) into a single in-process module with a multiprocessing pool for the FP pass; CLI `python3 -m spacehasten.remote.cluster <smi[.gz]>` produces `clustering.csv`. `stages/clustering.cluster` streams `(smiles, spacehastenid)` to gzipped input, submits a single exclusive CPU-heavy task, ingests via `db.replace_clusters`. Integration test exercises the real RDKit+FPSim2 pipeline on 100 synthetic SMILES via the local scheduler; 107 tests; ruff/mypy clean. |
| 11 | `tools/glide.py` + `stages/docking.py` | done | | `tools/glide.py` ports `write_confgen_file`/`write_docking_file`/`process_docking_results` reducer (`write_phase_inp`, `write_glide_in` (strips & rewrites `LIGANDFILE`/`GRIDFILE`), `parse_glide_csv` keeping `MIN(r_i_docking_score)` per `title` without pandas). `stages/docking.dock` shuffles, chunks at `min(N/cpus, DOCKING_CHUNK=1000)`, writes per-chunk `chunk_<i>.smi`/`.inp`/`glide_chunk_<i>.in`, extracts `glide_grid.zip` once, and submits an array job whose canonical body matches §A.4 row 2 (overridable for tests). Result tarballs are extracted under `extracted/`, parsed, and applied via `apply_dock_scores` in one transaction. Integration test uses a stub bash body emitting two-row `glide_chunk_<i>.csv` per compound to verify the min-reducer round-trips through scheduler+ingestion. 116 tests; ruff/mypy clean. |
| 12 | `tools/spacelight.py` + `tools/ftrees.py` + `remote/prop_filter.py` + `stages/simsearch.py` | done | | `SpacelightAdapter`/`FTreesAdapter` argv builders match legacy `--max-nof-results`/`--min-similarity-threshold`/`--thread-count` flag shape. `remote/prop_filter` ports `control.py` with the latent gunzip-on-plain-text bug fixed: input auto-detects `.gz`, output is plain CSV with columns `smiles,smilesid` (smilesid = legacy `<reghash>§<smiles>§<title>` packed string) and uses `\n` line endings so downstream bash `read` parses cleanly. `stages/simsearch.simsearch` runs the canonical three-phase flow: (A) `select_queries_for_simsearch` + `mark_as_query` + per-query SpaceLight/FTrees array, (B) per-method best-similarity-by-SMILES aggregation, gzipped chunking under `simsearch/cycle<N>/CONTROL/control_<i>.smi.gz`, `control.param` written from `db.load_properties()`, latest model materialised on disk via `Database.load_model_path` (BLOB fallback intact) and copied into `CONTROL/v<N>/`, prop-filter+chemprop predict array, (C) per-reghash min-score reduction, dedup against `data.reghash` index (chunked `IN (...)`), and `insert_simsearch_hit` per survivor. Search/control bodies are template-overridable for tests. New `Settings.general.{nnn_default,sim_spacelight_default,sim_ftrees_default,field_similarity_spacelight,field_similarity_ftrees}` fields complete the legacy cfg.py mapping. Tests: `test_search_adapters.py` (3 tests, command shape), `test_prop_filter.py` (7 tests inc. RDKit roundtrip + gzip + invalid-SMILES skip), `test_simsearch_local.py` (4 integration tests inc. dedup invariant via real RDKit reghash + per-method max-similarity bookkeeping + on-disk artefact layout + missing-model error). 130 tests; ruff/mypy clean. |
| 13 | `stages/seeds.py`, `stages/export.py`, `stages/archive.py` | not-started | | |
| 14 | `cli/main.py` — argparse subcommands | not-started | | `spacehasten --help` works |
| 15 | Port `verify_spacehasten.py`; cutover | not-started | | move legacy → `legacy/` |
| 16 | Quick-win patches on legacy tree | not-started | | parallel-safe; SGE typo + `sbatch` calls |
| 17 | Textual TUI | optional | | post-cutover |
| 18 | FastAPI dashboard | optional | | post-cutover |

Status values: `not-started` · `in-progress` · `blocked` · `done`.

## Decisions log

Append-only record of architectural decisions made during the rewrite.
Each entry: date, session, decision, rationale.

| Date | Session | Decision | Rationale |
|---|---|---|---|
| 2026-04-28 | 0 | Earlier `src/` scaffold deleted; restart fresh | Earlier attempt was incomplete and inconsistent with the plan. Easier to rebuild than reconcile. |
| 2026-04-28 | 0 | Stdlib `argparse` (not Typer/Click) | Avoids dep; legacy already uses argparse; sufficient for our needs. |
| 2026-04-28 | 0 | `sqlite3` stdlib (not SQLAlchemy/SQLModel) | Schema is small (6 tables) and stable; ORM is overkill. |
| 2026-04-28 | 0 | On-disk model registry, BLOB legacy fallback | Legacy BLOB store makes `.dbsh` huge. Keep loader compatible with old `.dbsh` files via fallback path. |
| 2026-04-28 | 0 | Single-root workspace under `/data/` | Plan §11.4. Eliminates `$HOME/SPACEHASTEN/` split. |
| 2026-04-28 | 1 | Editable install shadowed by legacy `spacehasten.py` at repo root when CWD=root | Expected per SESSIONS rule 2; resolved at Session 15 cutover. Tests run via `pytest` rootdir, package imports cleanly from any other CWD. |
| 2026-04-28 | 3 | Hardcode `SCHEMA_STATEMENTS` in `core/db.py` (with test asserting parity to `tests/fixtures/legacy_schema.sql`) | Avoids runtime dependency on the test-fixture file while preserving the fixture as the immutable source of truth. |
| 2026-04-28 | 3 | `PropertyRanges` defined in `core/db.py` as a frozen dataclass with `tuple[str,str]` limits | Matches legacy TEXT storage byte-for-byte; Session 4 will introduce a typed pydantic equivalent that converts to/from this representation. |
| 2026-04-28 | 3 | Acquisition SQL stored as `_SQL_*` class constants and locked by `tests/unit/test_db_sql_locked.py` | Spec requires byte-identical preservation of §A.6; class-constant + assert is the simplest regression lock. |
| 2026-04-29 | 12 | `remote/prop_filter` emits columns `smiles,smilesid` (not legacy `smiles,rawmol`) and writes plain CSV (legacy named output `.csv.gz` but wrote plain text, which the legacy scheduler then `gunzip`-ed — a latent bug). | The new format flows directly into `remote/predict` (which already expects `smiles,smilesid`), eliminating the gunzip step from the control task body. The `smilesid` field still packs `<reghash>§<smiles>§<title>` for downstream ingestion. |
| 2026-04-29 | 12 | `prop_filter` CSV writer pinned to `lineterminator="\n"`. | Default csv.writer emits `\r\n` which leaks `\r` into `read -r` in the control task's bash, corrupting the predicted CSV column order. |

## Open questions

Park here anything that would block forward progress and needs the user
to decide.

- (none currently — populated as sessions discover questions)

## Conventions

- **Branch per session**: `rewrite/sNN-short-name`. Squash-merge to `main`
  after tests pass.
- **Commit message**: `[sNN] <verb> <what>`, e.g. `[s03] add core/db.py with locked acquisition SQL`.
- **Test layout**: unit tests next to a peer module path
  (`tests/unit/test_<module>.py`); integration tests in
  `tests/integration/`.
- **Fixtures live under `tests/fixtures/`**; `legacy_baseline.dbsh` and
  `legacy_schema.sql` are immutable references.
- **Legacy code is read-only** until Session 15 except for the patches in
  Session 16.

## Environment quick reference

(See `.github/copilot-instructions.md` for full details.)

```bash
# Orchestrator dev environment (this workspace)
source /wrk/lurvas/miniconda3/etc/profile.d/conda.sh
conda activate spacehasten-quick

# Compute-node environments (used inside SLURM scripts)
source /data/programs/oce/actoce
conda activate chemprop-2.1.2     # for train/predict/control
conda activate fpsim2-0.7.3       # for clustering

# tmux required for any non-trivial command
tmux new -s rewrite
```

Workspace root: `/data/lurvas/projects/coding/SpaceHASTEN`. Writable
roots only: `/data/lurvas`, `/wrk/lurvas`, `/fastwrk/lurvas`.
