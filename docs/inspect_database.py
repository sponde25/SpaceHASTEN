# %% [markdown]
# # Inspecting a SpaceHASTEN database (`.dbsh`) with pandas
#
# A SpaceHASTEN workspace stores everything in a single SQLite file with a
# `.dbsh` extension (it is a plain SQLite database — the extension is just a
# convention). This file is a collection of copy-paste-ready recipes for
# loading and checking that database from Python / Jupyter.
#
# It is written with `# %%` cell markers, so you can:
#   - open it directly in **VS Code** or **JupyterLab** (via *jupytext*) and run
#     it cell-by-cell like a notebook, or
#   - just copy the snippets you need into your own notebook.
#
# **Schema of the main `data` table** (one row per compound):
#
# | column           | type    | meaning                                              |
# |------------------|---------|------------------------------------------------------|
# | `spacehastenid`  | INTEGER | primary key (internal row id)                        |
# | `reghash`        | TEXT    | structure hash (dedup key)                           |
# | `smiles`         | TEXT    | compound SMILES                                      |
# | `smilesid`       | TEXT    | external/source id                                   |
# | `dock_score`     | REAL    | actual docking score (lower is better; NULL if none) |
# | `pred_score`     | REAL    | chemprop predicted score (NULL if not predicted)     |
# | `spacelight`     | REAL    | SpaceLight similarity score                          |
# | `ftrees`         | REAL    | FTrees similarity score                              |
# | `query`          | INTEGER | simsearch cycle in which this row was used as a query|
# | `dock_iteration` | INTEGER | 0 = seed batch, 1..N = later screening rounds        |
# | `pred_version`   | INTEGER | chemprop model version that produced `pred_score`    |
# | `simsearch_cycle`| INTEGER | cycle in which this row was first acquired           |
#
# Other tables: `clusters` (spacehastenid -> clusterid), `properties`
# (property filter definitions), `smarts_filters`, plus BLOB-heavy tables
# `docking_param`, `docking_grid`, `models` (skip these when loading to pandas).

# %%
# ---------------------------------------------------------------------------
# 0. Imports and the path to your database
# ---------------------------------------------------------------------------
import sqlite3
from pathlib import Path

import pandas as pd

# Point this at your workspace's .dbsh file.
DBSH_PATH = Path("/path/to/workspace/your_workspace.dbsh")

assert DBSH_PATH.exists(), f"database not found: {DBSH_PATH}"

# %%
# ---------------------------------------------------------------------------
# 1. What tables exist, and how big is each one?
# ---------------------------------------------------------------------------
with sqlite3.connect(DBSH_PATH) as conn:
    table_names = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
    )["name"].tolist()

    counts = {
        name: pd.read_sql_query(f"SELECT COUNT(*) AS n FROM '{name}'", conn)["n"][0]
        for name in table_names
    }

pd.Series(counts, name="rows").to_frame()

# %%
# ---------------------------------------------------------------------------
# 2. Load the whole `data` table (the main compound table)
# ---------------------------------------------------------------------------
# This is what you almost always want. For very large workspaces this can be
# millions of rows; see section 4 for filtered/lighter alternatives.
with sqlite3.connect(DBSH_PATH) as conn:
    data = pd.read_sql_query("SELECT * FROM data", conn)

print(data.shape)
data.head()

# %%
# Quick overview of the main table.
data.info()
data.describe(include="all")

# %%
# ---------------------------------------------------------------------------
# 3. Load *all* tables at once (skipping the large BLOB tables)
# ---------------------------------------------------------------------------
SKIP_BLOB_TABLES = {"docking_param", "docking_grid", "models"}

with sqlite3.connect(DBSH_PATH) as conn:
    names = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", conn
    )["name"]
    tables = {
        name: pd.read_sql_query(f"SELECT * FROM '{name}'", conn)
        for name in names
        if name not in SKIP_BLOB_TABLES
    }

for name, df in tables.items():
    print(f"{name:16s} {df.shape}")

# Access any one with, e.g.:
# tables["data"], tables["clusters"], tables["properties"]

# %%
# ---------------------------------------------------------------------------
# 4. Loading only what you need (recommended for big databases)
# ---------------------------------------------------------------------------
# Filter in SQL so pandas never has to hold the full library in memory.

# 4a. Just the docked hits (dock_score below a cutoff; lower is better).
CUTOFF = -10.0
with sqlite3.connect(DBSH_PATH) as conn:
    hits = pd.read_sql_query(
        "SELECT * FROM data WHERE dock_score <= ? ORDER BY dock_score",
        conn,
        params=(CUTOFF,),
    )
print(f"{len(hits)} hits with dock_score <= {CUTOFF}")
hits.head()

# %%
# 4b. Only the seed batch (dock_iteration == 0).
with sqlite3.connect(DBSH_PATH) as conn:
    seeds = pd.read_sql_query(
        "SELECT * FROM data WHERE dock_iteration = 0", conn
    )
print(f"{len(seeds)} seed compounds")
seeds.head()

# %%
# 4c. Top-N best-scoring compounds overall.
N = 50
with sqlite3.connect(DBSH_PATH) as conn:
    top_n = pd.read_sql_query(
        "SELECT spacehastenid, smiles, smilesid, dock_score, pred_score,"
        " dock_iteration, simsearch_cycle"
        " FROM data WHERE dock_score IS NOT NULL"
        " ORDER BY dock_score ASC LIMIT ?",
        conn,
        params=(N,),
    )
top_n

# %%
# 4d. Only selected columns (much lighter than SELECT *).
with sqlite3.connect(DBSH_PATH) as conn:
    scores = pd.read_sql_query(
        "SELECT spacehastenid, dock_score, pred_score, dock_iteration,"
        " simsearch_cycle FROM data",
        conn,
    )
scores.head()

# %%
# ---------------------------------------------------------------------------
# 5. Streaming a huge table in chunks (avoids loading it all into RAM)
# ---------------------------------------------------------------------------
# Example: count rows per dock_iteration without materialising the full table.
counts_by_iter: dict = {}
with sqlite3.connect(DBSH_PATH) as conn:
    for chunk in pd.read_sql_query(
        "SELECT dock_iteration FROM data", conn, chunksize=100_000
    ):
        for it, n in chunk["dock_iteration"].value_counts().items():
            counts_by_iter[it] = counts_by_iter.get(it, 0) + int(n)
pd.Series(counts_by_iter, name="rows").sort_index().to_frame()

# %%
# ---------------------------------------------------------------------------
# 6. Handy aggregate checks (done in SQL, returned as small DataFrames)
# ---------------------------------------------------------------------------

# 6a. dock_score summary per iteration.
with sqlite3.connect(DBSH_PATH) as conn:
    per_iter = pd.read_sql_query(
        "SELECT dock_iteration,"
        "       COUNT(*)            AS n,"
        "       COUNT(dock_score)   AS n_docked,"
        "       MIN(dock_score)     AS best,"
        "       AVG(dock_score)     AS mean,"
        "       MAX(dock_score)     AS worst"
        " FROM data GROUP BY dock_iteration ORDER BY dock_iteration",
        conn,
    )
per_iter

# %%
# 6b. Sanity-check the score ranges — useful when a plot axis looks wrong.
# A pred_score far outside the dock_score range is usually an out-of-domain
# chemprop extrapolation, not a real hit.
with sqlite3.connect(DBSH_PATH) as conn:
    ranges = pd.read_sql_query(
        "SELECT MIN(dock_score) AS dock_min, MAX(dock_score) AS dock_max,"
        "       MIN(pred_score) AS pred_min, MAX(pred_score) AS pred_max"
        " FROM data",
        conn,
    )
ranges

# %%
# 6c. How many compounds were acquired per simsearch cycle.
with sqlite3.connect(DBSH_PATH) as conn:
    per_cycle = pd.read_sql_query(
        "SELECT simsearch_cycle, COUNT(*) AS n"
        " FROM data WHERE simsearch_cycle IS NOT NULL"
        " GROUP BY simsearch_cycle ORDER BY simsearch_cycle",
        conn,
    )
per_cycle

# %%
# ---------------------------------------------------------------------------
# 7. Joining with cluster assignments
# ---------------------------------------------------------------------------
# `clusters` is a separate table; use a LEFT JOIN so compounds without a
# cluster assignment are kept (clusterid is NaN for them).
with sqlite3.connect(DBSH_PATH) as conn:
    with_clusters = pd.read_sql_query(
        "SELECT d.spacehastenid, d.smiles, d.dock_score, c.clusterid"
        " FROM data d LEFT JOIN clusters c"
        "   ON d.spacehastenid = c.spacehastenid"
        " WHERE d.dock_score IS NOT NULL"
        " ORDER BY d.dock_score",
        conn,
    )
with_clusters.head()

# %%
# ---------------------------------------------------------------------------
# 8. Optional: run an arbitrary SQL string interactively
# ---------------------------------------------------------------------------
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a read-only SQL query against the database and return a DataFrame."""
    with sqlite3.connect(DBSH_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=params)


# Examples:
# query("SELECT * FROM data WHERE smilesid LIKE 'Z%' LIMIT 20")
# query("SELECT COUNT(*) FROM data WHERE pred_score IS NOT NULL")
