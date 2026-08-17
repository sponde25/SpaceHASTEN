# Plan: Library Screening (`spacehasten library-build` + `library-screen`)

> **Handoff spec.** This document is the complete, unambiguous
> implementation brief for the *library screening* feature. A fresh
> Copilot session should treat it as the spec: read the referenced source
> files, implement exactly what is described here, add the listed tests,
> keep `pytest -q` green, and update `MIGRATION_STATUS.md` at the end.
>
> **Hard invariants (from `docs/SESSIONS.md`):**
> 1. Preserve the legacy SQLite schema byte-for-byte. **No schema change
>    is required or permitted for this feature** (see §2, Decision D1).
> 2. Preserve the frozen acquisition SQL (`_SQL_*` in `core/db.py`) and its
>    regression lock (`tests/unit/test_db_sql_locked.py`). New SQL added by
>    this feature must be *new* constants/methods, never edits to existing
>    ones.
> 3. Use stdlib `argparse`. No Typer/Click.
> 4. Line length 100, Python 3.11, mypy-clean on new modules, ruff-clean.

---

## 1. Goal & scope

Add the ability to **ML-screen a large, pre-processed, diverse enumerated
library** (50M–1B compounds, e.g. Enamine REAL diverse subsets) with the
current chemprop surrogate model, filter by physicochemical (PC)
properties, and **insert only the high-scoring survivors into the `.dbsh`
database** so they become fresh, structurally diverse seeds for later
non-enumerated SpaceLight/FTrees searches.

Two CLI tools are delivered:

- **`spacehasten library-build`** — one-time (per library) conversion of an
  Enamine `.cxsmiles(.bz2)` diverse subset into a **chunked, canonicalized
  Parquet store** with precomputed PC properties + tautomer reghash, plus a
  self-describing `manifest.json`. The store is **target-agnostic** (SMILES
  + properties only), so it is built once and reused across every campaign.
- **`spacehasten library-screen`** — per-campaign: property-filter →
  chemprop-predict every compound in a library store (in parallel array
  tasks), select survivors by predicted docking score, dedup against the
  DB, and insert the winners (undocked, with `pred_score`).

**Out of scope (explicitly deferred):**
- Docking the inserted winners. The existing `spacehasten dock` command
  already selects `WHERE dock_score IS NULL ORDER BY pred_score` and will
  pick them up automatically. Diversity/cluster-based dock selection is a
  *separate* future feature of the `dock` command.
- Clustering before insert. Not done here.
- Workflow integration (`screening-cycle`). Ship as a standalone tool
  first; wire into workflows later.

---

## 2. Locked design decisions

**D1 — No schema change; provenance is implicit.**
Library-screened compounds are inserted with `reghash`, `smiles`,
`smilesid`, `pred_score`, `pred_version` set and **`simsearch_cycle` left
`NULL`** (also `dock_score`, `query`, `dock_iteration` `NULL`). The three
compound origins are then distinguishable without any new column:

| Origin            | `pred_score` | `simsearch_cycle` | `dock_score` (pre-dock) |
|-------------------|--------------|-------------------|-------------------------|
| Seed              | NULL         | NULL              | (docked → set)          |
| Simsearch hit     | maybe set    | **NOT NULL**      | NULL                    |
| **Library screen**| **NOT NULL** | **NULL**          | NULL                    |

So: *library-screened undocked* = `pred_score IS NOT NULL AND
simsearch_cycle IS NULL AND dock_score IS NULL`. Do **not** add a
`library_screen` column.

**D2 — Selection is score-based, modifiable, no clustering.**
Selection of which predicted compounds to insert supports three modes,
resolved in this precedence order:
1. `--top-n N` — insert the N compounds with the best (lowest) `pred_score`.
2. `--score-cutoff X` — insert all compounds with `pred_score <= X`.
3. **Default** (neither flag given) — cutoff derived from the seed docking
   scores: the value at the **top `--top-pct` percent** of seed
   `dock_score` (default `top_pct = 1.0` → 1st percentile). Insert all
   compounds with `pred_score <= cutoff`.

Lower docking score = better (Glide convention). All three are
user-facing; `--top-n` and `--score-cutoff` are mutually exclusive.

**D3 — Storage format: Parquet, zstd, chunked, properties + reghash precomputed.**
See §3. Rationale: chemprop 2.x featurizes molecular graphs *from SMILES at
inference time* (`SimpleMoleculeMolGraphFeaturizer` + `MoleculeDatapoint.
from_smi` in `remote/predict.py`) and does not consume precomputed feature
tensors — so SMILES is the required, most compact input. Parquet gives
columnar reads (SMILES-only for prediction, property-columns-only for
filtering), splittability (one file per array task), predicate pushdown,
and zstd compression (~half of gzipped CSV, faster decode). Carrying the 6
PC properties (**parsed from the Enamine columns**, see D5) + reghash turns
the per-campaign property "control" step into a **vectorized columnar
filter** (no per-campaign RDKit pass over 1B molecules) and makes reghash
dedup cheap.

**D4 — Property-filter first, then predict** (cheap → expensive funnel),
mirroring the existing `simsearch` control step order.

**D5 — Reuse Enamine's precomputed descriptors; do NOT recompute with RDKit.**
The Enamine REAL `.cxsmiles` files already ship the six PC descriptors we
filter on (verified header of `2026.01_Enamine_REAL_DB_13.6M.cxsmiles`):
```
smiles  id  MW  HAC  sLogP  HBA  HBD  RotBonds  FSP3  TPSA  QED
lead-like  350/3_lead-like  fragments  strict_fragments
PPI_modulators  natural_product-like  Type  InChiKey
```
So `library-build` **parses `MW/sLogP/HBA/HBD/RotBonds/TPSA` straight from
the source columns** (no `Descriptors.*`/`rdMolDescriptors.*` calls),
which removes the dominant per-molecule cost. RDKit is still required at
build time for **canonical SMILES + the tautomer `reghash`** (Enamine
provides an InChIKey, but the DB dedups on
`RegistrationHash.TAUTOMER_HASH`, so we must compute that ourselves for
cross-consistency with seed/simsearch rows).

*Caveat (document in README):* Enamine's descriptor values are not
guaranteed bit-identical to the RDKit values used by
`remote/prop_filter.py` on the seed/simsearch path (different toolkit /
versions). For coarse PC gates this is acceptable and keeps build cheap.
Provide a `--recompute-props` flag on `library-build` that forces RDKit
computation (exact parity with the rest of the pipeline) for users who
need it. Also parse and store `FSP3`, `QED`, and `InChIKey` as **optional
extra columns** (nullable) for possible future filters — they are free to
carry.

---

## 3. Library store format (produced by `library-build`)

### 3.1 On-disk layout
```
<store_dir>/
    manifest.json
    chunk_00000.parquet
    chunk_00001.parquet
    ...
```

### 3.2 Parquet chunk schema (one row per compound)
| Column        | Arrow type | Source | Notes                                       |
|---------------|-----------|--------|----------------------------------------------|
| `compound_id` | string    | col `id`   | Enamine catalog ID. Becomes `smilesid`.  |
| `smiles`      | string    | col `smiles` → RDKit | RDKit-canonical SMILES.        |
| `reghash`     | string    | RDKit  | `RegistrationHash.GetMolLayers(mol)[TAUTOMER_HASH]`. |
| `mw`          | float32   | col `MW`       | Parsed from cxsmiles (D5).           |
| `slogp`       | float32   | col `sLogP`    | Parsed from cxsmiles.                |
| `hba`         | int16     | col `HBA`      | Parsed from cxsmiles.                |
| `hbd`         | int16     | col `HBD`      | Parsed from cxsmiles.                |
| `rotbonds`    | int16     | col `RotBonds` | Parsed from cxsmiles.                |
| `tpsa`        | float32   | col `TPSA`     | Parsed from cxsmiles.                |
| `fsp3`        | float32   | col `FSP3`     | *Optional*, nullable — carried if present. |
| `qed`         | float32   | col `QED`      | *Optional*, nullable — carried if present. |
| `inchikey`    | string    | col `InChiKey` | *Optional*, nullable — carried if present. |

- Compression: `zstd`. Target ~1–5M rows/chunk (default **2_000_000**).
- Rows that fail `Chem.MolFromSmiles` are dropped at build time.
- The six filter properties are **taken from the Enamine columns by
  default** (D5); with `--recompute-props` they are computed with the exact
  RDKit calls from `remote/prop_filter.py` instead. `reghash` and canonical
  `smiles` are always RDKit-derived. The optional `fsp3/qed/inchikey`
  columns are best-effort: present only when the source provides them.
- Column mapping is resolved from the **cxsmiles header row by name**
  (case-insensitive), not by fixed position, so minor header reordering
  across Enamine releases is tolerated. Fall back to `--smiles-col/--id-col`
  and `--prop-cols` overrides when there is no header.

### 3.3 `manifest.json`
```json
{
  "format_version": 1,
  "source_files": ["Enamine_Diverse_REAL_..._48.2M.cxsmiles.bz2"],
  "n_compounds": 48200000,
  "n_chunks": 25,
  "chunk_glob": "chunk_*.parquet",
  "chunk_rows": [2000000, 2000000, "..."],
  "columns": ["compound_id","smiles","reghash","mw","slogp","hba","hbd","rotbonds","tpsa"],
  "optional_columns": ["fsp3","qed","inchikey"],
  "chunk_size": 2000000,
  "compression": "zstd",
  "props_source": "enamine",
  "rdkit_version": "2024.03.5",
  "reghash_algo": "RegistrationHash.TAUTOMER_HASH",
  "canonicalization": "rdkit-canonical",
  "built_at": "2026-08-03T15:50:00Z"
}
```
`props_source` is `"enamine"` (descriptors parsed from source columns) or
`"rdkit"` (built with `--recompute-props`); `library-screen` logs it so the
user knows which descriptor convention gated the filter.

A small `LibraryManifest` dataclass (load/save/validate) lives in the new
stage module (§5.1). `library-screen` loads it, validates `format_version
== 1` and that all 9 required `columns` are present, and uses `chunk_glob`
to enumerate tasks.

---

## 4. New / changed source files (index)

| File | Change |
|---|---|
| `src/spacehasten/remote/library_build.py` | **new** — cxsmiles → parquet chunk worker (single chunk) |
| `src/spacehasten/remote/library_infer.py` | **new** — parquet chunk → property filter (vectorized) → chemprop predict → output parquet |
| `src/spacehasten/stages/library_build.py` | **new** — orchestrate build: split source, submit build array, write manifest |
| `src/spacehasten/stages/library_screen.py` | **new** — orchestrate screen: submit infer array (resumable), ingest, select, insert |
| `src/spacehasten/core/db.py` | **add** `insert_library_hit()`, `seed_dock_score_percentile()`, `filter_existing_reghashes()` (new methods; no edits to `_SQL_*`) |
| `src/spacehasten/config/settings.py` | **add** `library_*` fields to `GeneralSettings`, `library_store_default` to `PathsSettings` |
| `src/spacehasten/cli/main.py` | **add** `_add_library_build` / `_add_library_screen` parsers + `_cmd_*` handlers, register in `Setup`/`Manual` help groups |
| `pyproject.toml` | ensure `pyarrow` (and `pandas`) are runtime deps of the orchestrator; the chemprop conda env must also have `pyarrow` |
| `README.md` | document both commands under CLI Reference |
| tests | see §8 |

---

## 5. Detailed implementation

### 5.1 `stages/library_build.py`

```python
def library_build(
    scheduler: Scheduler,
    settings: Settings,
    *,
    source_files: Sequence[Path],
    store_dir: Path,
    chunk_size: int = 2_000_000,
    recompute_props: bool = False,   # force RDKit descriptors (D5 parity)
    column_map: dict[str, str] | None = None,  # override header→field names
    build_command_prefix: Sequence[str] | None = None,
) -> LibraryManifest: ...
```
Algorithm:
1. `store_dir.mkdir(parents=True, exist_ok=True)`.
2. **Read the cxsmiles header** of the first source to resolve the
   column indices for `smiles, id, MW, sLogP, HBA, HBD, RotBonds, TPSA`
   (and optional `FSP3, QED, InChiKey`) **by name, case-insensitively**;
   `column_map` overrides. Record the resolved mapping to pass to the
   worker (as a small JSON sidecar in `_raw/`, or as CLI flags).
3. **Pre-split** the (bz2/gz/plain) source(s) into raw text shards of
   `chunk_size` lines each: `store_dir/_raw/shard_<i>.smi` (strip the
   header from every shard; the worker receives headerless data + the
   resolved column mapping). Stream-read (handle `.bz2`/`.gz` by suffix,
   like `prop_filter._open_input`). One array task per shard.
4. Submit an **array job** (`array_size = n_shards`) whose per-task body
   invokes `remote/library_build.py` on `shard_${TASK_ID}.smi` →
   `chunk_<i-1>.parquet` (5-digit zero-padded), passing the column mapping
   and `--recompute-props` when set. Env: `prepare_anaconda` +
   `activate_clustering` (RDKit lives in the fpsim2 env) — confirm RDKit is
   importable there; if not, use `activate_chemprop`.
5. On success: count rows per chunk (read parquet metadata `num_rows`,
   no full load), delete `_raw/`, write `manifest.json` (with
   `props_source = "rdkit" if recompute_props else "enamine"`).
6. Resumable: skip shards whose `chunk_<i>.parquet` already exists.

`LibraryManifest` dataclass: fields per §3.3; `.save(path)`, `.load(path)`,
`.validate()` (raises on missing columns / wrong `format_version`).

### 5.2 `remote/library_build.py` (compute node, RDKit env)

CLI:
```
library_build.py <shard.smi> <out.parquet> \
    --smiles-col N --id-col N \
    [--mw-col N --slogp-col N --hba-col N --hbd-col N --rotbonds-col N --tpsa-col N] \
    [--fsp3-col N --qed-col N --inchikey-col N] \
    [--recompute-props]
```
For each (headerless) line: split on tab/whitespace; take smiles + id;
`Chem.MolFromSmiles` (skip on None); canonicalize SMILES; compute the
tautomer `reghash` (RDKit, always). **Descriptors:**
- default: read `mw/slogp/hba/hbd/rotbonds/tpsa` from the parsed source
  columns (cast to the schema types; skip row on unparseable numeric);
  carry optional `fsp3/qed/inchikey` when their columns are provided.
- `--recompute-props`: ignore source descriptor columns and compute the six
  with the **exact RDKit calls from `prop_filter.py`** instead.

Accumulate into column lists; write a single Parquet file with pyarrow
(`compression="zstd"`), schema per §3.2. Batch-flush every ~200k rows to
bound memory. Use `pyarrow.Table.from_pydict` (pandas optional).

### 5.3 `remote/library_infer.py` (compute node, chemprop env)

CLI:
```
library_infer.py <chunk.parquet> <model_dir> <out.parquet> \
    --params <control.param> \
    [--score-cutoff FLOAT] [--top-n-per-chunk INT] \
    [--batch-size N] [--num-workers N] [--accelerator cpu] [--devices 1]
```
Algorithm:
1. `df = pd.read_parquet(chunk)` (needs `smiles, compound_id, reghash` +
   6 property cols).
2. **Vectorized property filter** using the precomputed columns against the
   12 bounds parsed from `--params` (same 12-line format as
   `prop_filter._Bounds.read`; reuse that parser — import it or duplicate
   the tiny reader). Build a boolean mask over `mw/slogp/hba/hbd/rotbonds/
   tpsa`; keep survivors. **No RDKit call here** — this is the columnar
   speed win.
3. If 0 survivors → write empty output parquet (schema:
   `reghash, smiles, compound_id, pred_score`) and exit 0.
4. Run chemprop prediction on `survivors["smiles"]` reusing the **exact
   chemprop inference block from `remote/predict.py`** (featurizer,
   `MoleculeDatapoint.from_smi`, `MPNN.load_from_checkpoint`,
   `pl.Trainer(...).predict`, unscale-transform warning). Factor that block
   into a shared helper if convenient, but do **not** change `predict.py`'s
   behavior.
5. Attach `pred_score` to survivors.
6. **Optional local pre-selection** to shrink output I/O at scale:
   - if `--score-cutoff X`: keep rows with `pred_score <= X`.
   - elif `--top-n-per-chunk K`: keep the K best by `pred_score`.
   - else: keep all.
7. Write output parquet `out.parquet` with columns
   `reghash, smiles, compound_id, pred_score` (`compound_id` maps to
   `smilesid` at ingest).

> **SMARTS note:** precomputed columns do not encode SMARTS matches. For
> v1, SMARTS filtering is **not** applied in `library-infer`. If needed
> later, apply it with RDKit on survivors only (after the property mask,
> before predict). Document this limitation.

### 5.4 `stages/library_screen.py`

```python
def library_screen(
    db: Database,
    workdir: WorkDir,
    scheduler: Scheduler,
    settings: Settings,
    *,
    library_dir: Path,
    model_version: int,
    props: PropertyRanges,
    top_n: int | None = None,
    score_cutoff: float | None = None,
    top_pct: float = 1.0,
    max_concurrent: int | None = None,
    dry_run: bool = False,
    report_path: Path | None = None,
    infer_command_prefix: Sequence[str] | None = None,
) -> int:   # returns number of compounds inserted (or would-insert if dry_run)
```

Orchestration:

1. **Load & validate** the library manifest from `library_dir`.
2. **Resolve the selection cutoff up-front** (so it can be pushed into the
   remote pre-filter):
   - `top_n` given → no cutoff pushdown; use `--top-n-per-chunk = top_n`
     in the remote command (bounded per-chunk top-N), then global top-N at
     ingest.
   - `score_cutoff` given → push `--score-cutoff score_cutoff`.
   - else → `cutoff = db.seed_dock_score_percentile(top_pct)`; if `None`
     (no seed docking yet) → **abort** with a clear error ("no seed
     docking scores; run seed-training first, or pass --score-cutoff/
     --top-n"). Push `--score-cutoff cutoff`.
3. **Materialize the model on disk** exactly as `prediction.predict_undocked`
   does: `bin_path = db.load_model_path(model_version, workdir)`;
   `model_dir = bin_path.parent.parent`; assert
   `model_dir/model_0/pytorch_model.bin` exists.
4. **Write `control.param`** (12-line PC bounds) from `props` into
   `workdir.library_dir_scratch()/inputs/control.param`. Reuse the writer
   pattern from `simsearch._write_control_param` (it reads from the DB
   `properties` table; here we already have a `PropertyRanges`, so write
   its 12 values directly in canonical order: mw_min,mw_max,slogp_min,...).
   Precedence for `props`: `--props-toml` > DB `properties` table >
   `PropertyRanges()` defaults.
5. **Enumerate chunks** from `manifest.chunk_glob`, sorted. Output dir:
   `<scratch>/results/predicted_<chunk_stem>.csv` (survivors are written as
   CSV, not Parquet, so they are easy to inspect; the library store itself
   stays Parquet).
6. **Resumable submission**: build the list of chunks whose output CSV
   does *not* yet exist. If all exist, skip to ingest. Submit an
   `ArrayJob` sized to the missing chunks. The library chunks are **read
   directly from wherever the store lives** — their absolute paths are
   written (one per line) to `inputs/chunks.txt`, and the command body
   selects its chunk via `sed -n "${TASK_ID}p" inputs/chunks.txt`. No
   library data is copied or symlinked into the run directory (this avoids
   creating a persistent duplicate of the library on every screen).
7. Command body (per task), built like `prediction._build_predict_command`:
   ```
   CHUNK_PATH=$(sed -n "${TASK_ID}p" inputs/chunks.txt)
   CHUNK_STEM=$(basename "$CHUNK_PATH"); CHUNK_STEM="${CHUNK_STEM%.parquet}"
   OUT_PATH="results/predicted_${CHUNK_STEM}.csv"
   python3 <remote/library_infer.py> \
       "$CHUNK_PATH" <model_dir> "$OUT_PATH" \
       --params <inputs/control.param> \
       [--score-cutoff X | --top-n-per-chunk K] \
       --batch-size <library_infer_batch_size> \
       --num-workers <library_infer_num_workers> \
       --accelerator <library_infer_accelerator> \
       --devices <library_infer_devices> \
       --block-size <library_infer_block_size>
   ```
   `env_setup = [prepare_anaconda, activate_chemprop]`. `cpus_per_task =
   int(settings.general.cpu_count_library or 1)`. `gpus` per accelerator
   setting. The worker streams each chunk in `--block-size` row slices
   (via pyarrow `iter_batches`) so peak memory is bounded by the slice, not
   the whole chunk — large (1-2M row) chunks can be screened, and many
   tasks run in parallel per node, without OOM. In the `top_n` path the
   worker trims each block to its own best `top_n` rows before accumulating
   (exact: a chunk-wide top-N row has at most N-1 better rows in the whole
   chunk, so it is always within its block's top-N), capping accumulation at
   `n_blocks * top_n` rather than every predicted survivor.
8. `scheduler.wait(...)`; on failure raise `RuntimeError` with
   `diagnostics.tail_logs(handle)` (same pattern as other stages).
9. **Ingest**: read every `results/predicted_*.csv`; concatenate
   `(reghash, smiles, compound_id, pred_score)`. Dedup by `reghash`
   keeping **min** `pred_score` (best), first-seen smiles/id.
10. **Global selection**:
    - `top_n` → sort by `pred_score` asc, take first `top_n`.
    - else (`score_cutoff` resolved in step 2) → keep `pred_score <= cutoff`.
11. **Dedup against DB**: `existing = db.filter_existing_reghashes(reghashes)`;
    drop those.
12. **Insert** survivors (unless `dry_run`) via
    `db.insert_library_hit(reghash, smiles, smilesid=compound_id,
    pred_score=score, pred_version=model_version)`; single transaction;
    `db.commit()`.
13. **Report**: log and optionally write `report_path` (JSON) with:
    `n_chunks, n_predicted (post-property survivors, summed), n_selected,
    n_deduped_existing, n_inserted, cutoff_used, model_version,
    pred_score_min/median/max of inserted`. In `dry_run`, additionally
    write the ranked survivor list to a CSV next to the report.
14. Return `n_inserted`.

**Scratch layout** (add to `WorkDir`, §5.5):
```
<shared_root>/library_screen/run<K>/
    inputs/  control.param  smarts.txt  chunks.txt (abs paths of pending chunks)
    results/ predicted_<chunk_stem>.csv (survivors only)
    report.json
```
`run<K>` = incrementing integer (mkdir the first free `runN`).

### 5.5 `WorkDir` addition (`workspace/layout.py`)
```python
def library_screen_dir(self, run: int) -> Path:
    return self.shared_root / "library_screen" / f"run{run}"
```

### 5.6 `core/db.py` additions (new methods only — do not touch `_SQL_*`)

```python
def insert_library_hit(
    self, reghash: str, smiles: str, smilesid: str,
    pred_score: float, pred_version: int,
) -> int:
    c = self._conn.execute(
        "INSERT INTO data(reghash, smiles, smilesid, pred_score, pred_version) "
        "VALUES (?, ?, ?, ?, ?)",
        (reghash, smiles, smilesid, pred_score, pred_version),
    )
    assert c.lastrowid is not None
    return c.lastrowid

def seed_dock_score_percentile(self, pct: float) -> float | None:
    """Value V such that `pct` percent of seed (dock_iteration=0) dock
    scores are <= V (lower = better). Returns None if no seed docking."""
    scores = self.dock_scores_by_iteration().get(0)
    if not scores:
        return None
    scores = sorted(scores)                      # ascending (best first)
    import math
    k = max(1, math.ceil(len(scores) * pct / 100.0))
    return float(scores[k - 1])

def filter_existing_reghashes(self, candidates: Iterable[str]) -> set[str]:
    """reghashes already present in `data` (chunked IN query, 500/batch).
    Extract the existing private `simsearch._existing_reghashes` logic
    here and have simsearch call this method to avoid duplication."""
```
Refactor `stages/simsearch.py::_existing_reghashes` to delegate to
`db.filter_existing_reghashes` (behavior-preserving).

### 5.7 `config/settings.py` additions
`GeneralSettings` (defaults mirror `pred_*`):
```python
cpu_count_library: str = "1"
library_infer_batch_size: int = 1000
library_infer_num_workers: int = 0
library_infer_accelerator: str = "cpu"
library_infer_devices: str = "1"
library_default_top_pct: float = 1.0
library_build_chunk_size: int = 2_000_000
```
`PathsSettings`:
```python
library_store_default: str | None = None
```

### 5.8 CLI (`cli/main.py`)

`_add_library_build`:
```
spacehasten library-build
    --source FILE [--source FILE ...]      (required; cxsmiles/.smi/.bz2/.gz)
    --output DIR                            (required; store dir)
    [--chunk-size 2000000]
    [--recompute-props]                     (force RDKit descriptors; D5)
    [--smiles-col NAME_OR_IDX]              (default: header 'smiles')
    [--id-col NAME_OR_IDX]                  (default: header 'id')
    [--jobs 250]                            (max array tasks run at once; 1 core each)
```
Header columns are resolved by name by default; the `--*-col` flags are
overrides for headerless inputs. Descriptors come from the Enamine columns
unless `--recompute-props` is given (§D5).
`_add_library_screen`:
```
spacehasten library-screen
    [--library DIR]                         (default: paths.library_store_default)
    [--model-version N]                     (default: latest)
    [--top-n N | --score-cutoff FLOAT]      (mutually exclusive)
    [--top-pct 1.0]                         (used only when neither above given)
    [--props-toml FILE]                     (override DB properties)
    [--jobs 250]                            (max screening array tasks run at once)
    [--dry-run]
    [--report FILE]
```
Handlers follow the `_cmd_predict` pattern: build `workdir`,
`settings`, `scheduler`; resolve `model_version` (error if none);
resolve `props` (toml > `db.load_properties()` > defaults); enforce
`--top-n`/`--score-cutoff` mutual exclusion (argparse mutually exclusive
group); call the stage; `print` a one-line summary. Register both in the
help groups: `library-build` under **Setup**, `library-screen` under
**Manual stages** (see `_add_*` grouping in `main.py`).

---

## 6. Scale & performance guidance (document in README)
- Property-filter-first + precomputed columns is what makes 50M–1B
  tractable; the D-MPNN only ever scores property-passing survivors.
- CPU D-MPNN inference on 1B compounds is hundreds of thousands of
  core-hours — recommend starting at the 50M subset and/or GPU
  (`library_infer_accelerator = gpu`).
- One array task per parquet chunk; tune `--jobs` to cluster
  size. `library_infer_batch_size` default 1000 (vs predict's 32) because
  chunks are large and throughput-bound.

---

## 7. Dependencies
- Orchestrator env (`spacehasten-quick`): add `pyarrow` to
  `pyproject.toml` runtime deps (pandas already used).
- Compute-node **chemprop env** must have `pyarrow` (for `library_infer`).
- Compute-node **RDKit env** (fpsim2 or chemprop) used for
  `library_build`. Confirm which env has RDKit; the codebase uses
  `activate_clustering` for RDKit-only remote work (`remote/cluster.py`)
  and `activate_chemprop` for RDKit+chemprop.

---

## 8. Tests (mirror existing integration-test style with bash stubs)

Unit:
- `tests/unit/test_library_manifest.py` — `LibraryManifest` save/load/
  validate (missing column, bad `format_version`).
- `tests/unit/test_db_library.py` — `insert_library_hit` sets
  `simsearch_cycle IS NULL`, `pred_score`/`pred_version` set;
  `seed_dock_score_percentile` correctness (1st/50th pct; None when no
  seeds); `filter_existing_reghashes` chunking.
- `tests/unit/test_library_build_worker.py` — `remote/library_build.py`
  descriptor parsing: given a small cxsmiles snippet with the real header
  (`smiles id MW HAC sLogP HBA HBD RotBonds FSP3 TPSA QED ... InChiKey`),
  assert the six filter columns are taken from the source values by
  default, and that `--recompute-props` instead produces the RDKit values;
  header-by-name resolution and `--*-col` overrides.
- Extend `tests/unit/test_db_sql_locked.py` expectations only if you added
  locked constants (you should **not** need to — `insert_library_hit` uses
  an inline string, consistent with `insert_simsearch_hit`).

Integration (LocalScheduler + stub remotes, like
`tests/integration/test_prediction_local.py` and `test_simsearch_local.py`):
- `test_library_build_local.py` — stub `library_build` writing a tiny
  parquet; assert manifest + chunk row counts.
- `test_library_screen_local.py` — pre-made 2-chunk parquet store + stub
  `library_infer` emitting fixed `pred_score`; assert: property filter
  applied, resumability (pre-existing output chunk skipped), dedup vs DB,
  `--score-cutoff` and `--top-n` selection, `--dry-run` inserts nothing,
  inserted rows have `simsearch_cycle IS NULL`.
- `test_cli.py` — arg parsing: mutual exclusion of `--top-n`/
  `--score-cutoff`; defaults.

---

## 9. Implementation phases (checklist)
1. **DB**: `insert_library_hit`, `seed_dock_score_percentile`,
   `filter_existing_reghashes`; refactor simsearch to use the last.
   Unit tests green.
2. **Config**: add `GeneralSettings`/`PathsSettings` fields.
3. **Format**: `LibraryManifest` + `remote/library_build.py` +
   `stages/library_build.py`; unit + integration tests.
4. **Infer**: `remote/library_infer.py` (share chemprop block with
   `predict.py` without changing it).
5. **Screen stage**: `stages/library_screen.py` (resumable, ingest,
   select, insert, report) + `WorkDir.library_screen_dir`.
6. **CLI**: `library-build` + `library-screen` parsers/handlers + help
   groups.
7. **Docs**: README CLI Reference entries; note SMARTS limitation & scale
   guidance. Update `MIGRATION_STATUS.md`.
8. `pytest -q` green, `ruff check`, `mypy` clean on new modules.

---

## 10. Open items to confirm with the user (if they arise during coding)
- Which conda env has RDKit for `library-build` (fpsim2 vs chemprop) on
  *this* cluster — pick based on `spacehasten.ini` and the copilot
  instructions (fpsim2-0.7.3 has rdkit; chemprop-2.1.2 also has rdkit).
- **Enamine `.cxsmiles` header is known** (confirmed by the user):
  `smiles  id  MW  HAC  sLogP  HBA  HBD  RotBonds  FSP3  TPSA  QED
  lead-like  350/3_lead-like  fragments  strict_fragments  PPI_modulators
  natural_product-like  Type  InChiKey`. Resolve columns by name; the
  six filter descriptors + optional `FSP3/QED/InChiKey` all come from here
  (D5). Verify the header is byte-identical across the specific subset(s)
  the user builds; fall back to `--*-col` overrides otherwise.
- Whether `pred_score` from an *old* model version should block re-screen
  with a newer model (v1: allow; each run overwrites nothing — inserts new
  rows, and duplicate reghashes are dropped against the DB).
