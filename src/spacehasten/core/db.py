"""Typed sqlite3 wrapper for the SpaceHASTEN ``.dbsh`` schema.

This module is a thin layer over :mod:`sqlite3`. It performs *no* business
logic; every method maps to a single SQL statement (or a tightly-scoped
group such as DROP+CREATE+INSERT). The legacy schema (see
``tests/fixtures/legacy_schema.sql`` and §A.1 of the codebase reference) is
preserved byte-for-byte.

Acquisition SQL strings (see §A.6 of the codebase reference) are stored as
class-level ``_SQL_*`` constants. They are the regression lock — tests in
``tests/unit/test_db_sql_locked.py`` assert byte-equality with §A.6.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final, Literal

# ---------------------------------------------------------------------------
# Schema (frozen — must stay byte-identical to tests/fixtures/legacy_schema.sql)
# ---------------------------------------------------------------------------

SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    (
        "CREATE TABLE IF NOT EXISTS data ("
        "spacehastenid INTEGER PRIMARY KEY,"
        "reghash TEXT,"
        "smiles TEXT,"
        "smilesid TEXT,"
        "dock_score REAL,"
        "pred_score REAL,"
        "spacelight REAL,"
        "ftrees REAL,"
        "query INTEGER,"
        "dock_iteration INTEGER,"
        "pred_version INTEGER,"
        "simsearch_cycle INTEGER"
        ")"
    ),
    "CREATE TABLE IF NOT EXISTS docking_param (dock_param BLOB)",
    "CREATE TABLE IF NOT EXISTS docking_grid (dock_grid BLOB)",
    "CREATE TABLE IF NOT EXISTS models (model_version INTEGER UNIQUE,model_tar BLOB)",
    (
        "CREATE TABLE IF NOT EXISTS properties ("
        "property TEXT,is_double INTEGER,min_limit TEXT,max_limit TEXT)"
    ),
    "CREATE TABLE IF NOT EXISTS clusters(spacehastenid INTEGER PRIMARY KEY,clusterid INTEGER)",
    "CREATE INDEX IF NOT EXISTS idx_reghash ON data(reghash)",
)

EXTENSION_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    (
        "CREATE TABLE IF NOT EXISTS predictions ("
        "spacehastenid INTEGER NOT NULL,"
        "model_version INTEGER NOT NULL,"
        "pred_score REAL NOT NULL,"
        "epistemic_std REAL,"
        "aleatoric_std REAL,"
        "total_std REAL,"
        "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
        "PRIMARY KEY(spacehastenid, model_version)"
        ")"
    ),
    "CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version)",
)


# ---------------------------------------------------------------------------
# Row dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataRow:
    spacehastenid: int
    reghash: str | None
    smiles: str | None
    smilesid: str | None
    dock_score: float | None
    pred_score: float | None
    spacelight: float | None
    ftrees: float | None
    query: int | None
    dock_iteration: int | None
    pred_version: int | None
    simsearch_cycle: int | None


@dataclass(frozen=True)
class ClusterRow:
    spacehastenid: int
    clusterid: int


@dataclass(frozen=True)
class PropertyRow:
    """One row of the ``properties`` table.

    ``min_limit`` / ``max_limit`` are stored as TEXT in the legacy schema and
    cast at read time; we keep them as ``str`` here to round-trip exactly.
    """

    property: str
    is_double: int  # 1 == float, 0 == int
    min_limit: str
    max_limit: str


@dataclass(frozen=True)
class ModelRow:
    model_version: int
    model_tar: bytes


@dataclass(frozen=True)
class PredictionRow:
    spacehastenid: int
    model_version: int
    pred_score: float
    epistemic_std: float | None
    aleatoric_std: float | None
    total_std: float | None
    created_at: str


@dataclass(frozen=True)
class SimsearchCycleStats:
    """Impact summary for :meth:`Database.undo_simsearch_cycle`.

    ``n_hits_docked`` and ``n_hits_used_as_query`` are the guardrail
    counts: if either is non-zero, ``undo_simsearch_cycle`` refuses to
    proceed (see its docstring).
    """

    cycle: int
    n_hits: int
    n_queries: int
    n_hits_docked: int
    n_hits_used_as_query: int


@dataclass(frozen=True)
class ExportRow:
    smiles: str
    spacehastenid: int
    smilesid: str
    dock_score: float
    pred_score: float | None
    spacelight: float | None
    ftrees: float | None
    dock_iteration: int | None
    clusterid: int | None


@dataclass(frozen=True)
class PropertyRanges:
    """Six property ranges, mirroring legacy cfg.py keys.

    Limits are stored as ``str`` to preserve the legacy TEXT storage format
    byte-for-byte. Session 4 introduces a typed pydantic equivalent that
    converts to/from this representation.
    """

    mw: tuple[str, str]
    slogp: tuple[str, str]
    hba: tuple[str, str]
    hbd: tuple[str, str]
    rotbonds: tuple[str, str]
    tpsa: tuple[str, str]

    def to_rows(self) -> list[PropertyRow]:
        return [
            PropertyRow("mw", 1, self.mw[0], self.mw[1]),
            PropertyRow("slogp", 1, self.slogp[0], self.slogp[1]),
            PropertyRow("hba", 0, self.hba[0], self.hba[1]),
            PropertyRow("hbd", 0, self.hbd[0], self.hbd[1]),
            PropertyRow("rotbonds", 0, self.rotbonds[0], self.rotbonds[1]),
            PropertyRow("tpsa", 1, self.tpsa[0], self.tpsa[1]),
        ]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class Database:
    """Long-lived sqlite3 connection wrapper.

    Unlike the legacy code (which opens/closes a connection per call), this
    class keeps a single connection for the lifetime of the instance. Use as
    a context manager or call :meth:`close` explicitly.
    """

    # ----- §A.6 acquisition SQL (preserve verbatim) -----
    _SQL_DOCK_GREEDY: Final[str] = (
        "SELECT smiles, spacehastenid FROM data\n"
        " WHERE dock_score IS NULL\n"
        " ORDER BY pred_score LIMIT ?"
    )
    _SQL_DOCK_CLUSTERING: Final[str] = (
        "SELECT smiles, data.spacehastenid FROM data, clusters\n"
        " WHERE data.spacehastenid = clusters.spacehastenid\n"
        "   AND dock_score IS NULL\n"
        " GROUP BY clusterid ORDER BY MIN(pred_score) LIMIT ?"
    )
    _SQL_SIMSEARCH_DOCKED_GREEDY: Final[str] = (
        "SELECT smiles, spacehastenid FROM data\n"
        " WHERE query IS NULL AND dock_score IS NOT NULL\n"
        " ORDER BY dock_score LIMIT ?"
    )
    _SQL_SIMSEARCH_DOCKED_CLUSTERING: Final[str] = (
        "SELECT smiles, data.spacehastenid FROM data, clusters\n"
        " WHERE data.spacehastenid = clusters.spacehastenid\n"
        "   AND query IS NULL AND dock_score IS NOT NULL\n"
        " GROUP BY clusterid ORDER BY MIN(dock_score) LIMIT ?"
    )
    _SQL_SIMSEARCH_PREDICTED_GREEDY: Final[str] = (
        "SELECT smiles, spacehastenid FROM data\n"
        " WHERE query IS NULL AND pred_score IS NOT NULL AND dock_score IS NULL\n"
        " ORDER BY pred_score LIMIT ?"
    )
    _SQL_SIMSEARCH_PREDICTED_CLUSTERING: Final[str] = (
        "SELECT smiles, data.spacehastenid FROM data, clusters\n"
        " WHERE data.spacehastenid = clusters.spacehastenid\n"
        "   AND query IS NULL AND pred_score IS NOT NULL AND dock_score IS NULL\n"
        " GROUP BY clusterid ORDER BY MIN(pred_score) LIMIT ?"
    )

    # ----- update SQL (parameterised) -----
    _SQL_UPDATE_DOCK_SCORE: Final[str] = (
        "UPDATE data SET dock_score = ?, dock_iteration = ? WHERE spacehastenid = ?"
    )
    _SQL_UPDATE_PRED_SCORE: Final[str] = (
        "UPDATE data SET pred_score = ?, pred_version = ? WHERE spacehastenid = ?"
    )
    _SQL_UPSERT_PREDICTION: Final[str] = (
        "INSERT INTO predictions("
        "spacehastenid, model_version, pred_score, epistemic_std, aleatoric_std, total_std"
        ") VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(spacehastenid, model_version) DO UPDATE SET "
        "pred_score = excluded.pred_score, "
        "epistemic_std = excluded.epistemic_std, "
        "aleatoric_std = excluded.aleatoric_std, "
        "total_std = excluded.total_std, "
        "created_at = CURRENT_TIMESTAMP"
    )
    _SQL_MARK_AS_QUERY: Final[str] = (
        "UPDATE data SET query = ? WHERE spacehastenid = ?"
    )

    # ----- supporting select SQL -----
    _SQL_SELECT_UNDOCKED: Final[str] = (
        "SELECT smiles, spacehastenid FROM data WHERE dock_score IS NULL"
    )
    _SQL_SELECT_TRAINING: Final[str] = (
        "SELECT smiles, dock_score FROM data\n"
        " WHERE dock_score IS NOT NULL AND dock_score < ?"
    )
    _SQL_SELECT_EXPORT: Final[str] = (
        "SELECT smiles, data.spacehastenid, smilesid, dock_score, pred_score,"
        " spacelight, ftrees, dock_iteration, clusterid\n"
        " FROM data LEFT JOIN clusters ON data.spacehastenid = clusters.spacehastenid\n"
        " WHERE dock_score <= ?\n"
        " ORDER BY dock_score"
    )
    _SQL_SELECT_SEEDS: Final[str] = (
        "SELECT smiles, smilesid, dock_score FROM data\n"
        " WHERE dock_iteration = 0\n"
        " ORDER BY spacehastenid"
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection = sqlite3.connect(self.path)

    # ----- context manager -----

    def __enter__(self) -> Database:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def commit(self) -> None:
        self._conn.commit()

    # ----- schema -----

    def create_schema(self) -> None:
        c = self._conn.cursor()
        for stmt in (*SCHEMA_STATEMENTS, *EXTENSION_SCHEMA_STATEMENTS):
            c.execute(stmt)
        self._conn.commit()

    def ensure_extension_schema(self) -> None:
        """Create additive SpaceHASTEN extension tables for an existing database."""
        for stmt in EXTENSION_SCHEMA_STATEMENTS:
            self._conn.execute(stmt)

    # ----- lookups -----

    def reghash_exists(self, reghash: str) -> bool:
        """Return True if a row with this reghash is already in the data table."""
        row = self._conn.execute(
            "SELECT 1 FROM data WHERE reghash = ? LIMIT 1", (reghash,)
        ).fetchone()
        return row is not None

    # ----- inserts -----

    def insert_seed_undocked(self, reghash: str, smiles: str, smilesid: str) -> int:
        c = self._conn.execute(
            "INSERT INTO data(reghash, smiles, smilesid) VALUES (?, ?, ?)",
            (reghash, smiles, smilesid),
        )
        assert c.lastrowid is not None
        return c.lastrowid

    def insert_seed_docked(
        self, reghash: str, smiles: str, smilesid: str, dock_score: float
    ) -> int:
        c = self._conn.execute(
            "INSERT INTO data(reghash, smiles, smilesid, dock_score, dock_iteration)"
            " VALUES (?, ?, ?, ?, 0)",
            (reghash, smiles, smilesid, dock_score),
        )
        assert c.lastrowid is not None
        return c.lastrowid

    def insert_simsearch_hit(
        self,
        reghash: str,
        smiles: str,
        smilesid: str,
        spacelight: float | None,
        ftrees: float | None,
        pred_score: float | None,
        simsearch_cycle: int,
        pred_version: int | None = None,
    ) -> int:
        c = self._conn.execute(
            "INSERT INTO data("
            "reghash, smiles, smilesid, spacelight, ftrees, pred_score, "
            "simsearch_cycle, pred_version"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                reghash,
                smiles,
                smilesid,
                spacelight,
                ftrees,
                pred_score,
                simsearch_cycle,
                pred_version,
            ),
        )
        assert c.lastrowid is not None
        return c.lastrowid

    # ----- updates -----

    def update_dock_score(
        self, spacehastenid: int, dock_score: float, dock_iteration: int
    ) -> None:
        self._conn.execute(
            self._SQL_UPDATE_DOCK_SCORE, (dock_score, dock_iteration, spacehastenid)
        )

    def update_pred_score(
        self, spacehastenid: int, pred_score: float, pred_version: int
    ) -> None:
        self._conn.execute(
            self._SQL_UPDATE_PRED_SCORE, (pred_score, pred_version, spacehastenid)
        )

    def store_prediction(
        self,
        spacehastenid: int,
        model_version: int,
        pred_score: float,
        epistemic_std: float | None,
        aleatoric_std: float | None,
        total_std: float | None,
    ) -> None:
        """Store one versioned prediction and its uncertainty decomposition."""
        self.ensure_extension_schema()
        self._conn.execute(
            self._SQL_UPSERT_PREDICTION,
            (
                spacehastenid,
                model_version,
                pred_score,
                epistemic_std,
                aleatoric_std,
                total_std,
            ),
        )

    def mark_as_query(self, spacehastenid: int, cycle: int) -> None:
        self._conn.execute(self._SQL_MARK_AS_QUERY, (cycle, spacehastenid))

    # ----- maxima -----

    def latest_model_version(self) -> int | None:
        """Return the highest model version, or None if no models exist."""
        row = self._conn.execute("SELECT MAX(model_version) FROM models").fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def latest_simsearch_cycle(self) -> int:
        row = self._conn.execute("SELECT MAX(simsearch_cycle) FROM data").fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def latest_dock_iteration(self) -> int | None:
        """Return the highest dock_iteration, or None if no rows have been docked."""
        row = self._conn.execute("SELECT MAX(dock_iteration) FROM data").fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def latest_search_attempt_cycle(self) -> int | None:
        """Return the highest simsearch cycle number *attempted* so far.

        Unlike :meth:`latest_simsearch_cycle` (which only looks at
        ``simsearch_cycle``, i.e. cycles that actually produced hits),
        this also considers the ``query`` column. ``simsearch()`` marks
        and commits queries *before* running the search job, so a cycle
        whose search job then fails leaves ``query = cycle`` rows but no
        ``simsearch_cycle = cycle`` rows. Taking the max of both columns
        recovers that failed attempt as "the latest cycle" so ``undo
        search`` can target it. Returns ``None`` if no simsearch cycle has
        ever been attempted.
        """
        row = self._conn.execute(
            "SELECT MAX(x) FROM ("
            "SELECT MAX(query) AS x FROM data "
            "UNION ALL SELECT MAX(simsearch_cycle) AS x FROM data"
            ")"
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def simsearch_cycle_stats(self, cycle: int) -> SimsearchCycleStats:
        """Return impact counts for reverting ``cycle`` (see :meth:`undo_simsearch_cycle`)."""
        n_hits = self._conn.execute(
            "SELECT COUNT(*) FROM data WHERE simsearch_cycle = ?", (cycle,)
        ).fetchone()[0]
        n_queries = self._conn.execute(
            "SELECT COUNT(*) FROM data WHERE query = ?", (cycle,)
        ).fetchone()[0]
        n_hits_docked = self._conn.execute(
            "SELECT COUNT(*) FROM data WHERE simsearch_cycle = ? AND dock_score IS NOT NULL",
            (cycle,),
        ).fetchone()[0]
        n_hits_used_as_query = self._conn.execute(
            "SELECT COUNT(*) FROM data WHERE simsearch_cycle = ? AND query IS NOT NULL",
            (cycle,),
        ).fetchone()[0]
        return SimsearchCycleStats(
            cycle=cycle,
            n_hits=int(n_hits),
            n_queries=int(n_queries),
            n_hits_docked=int(n_hits_docked),
            n_hits_used_as_query=int(n_hits_used_as_query),
        )

    def undo_simsearch_cycle(self, cycle: int) -> tuple[int, int]:
        """Revert simsearch ``cycle``: delete its hit compounds, release its query marks.

        Intended for the case where ``simsearch()`` marked queries (and
        committed) but then failed before inserting any hits, permanently
        stranding those compounds behind ``query IS NOT NULL``. Deletes
        every row with ``simsearch_cycle = cycle`` (plus any stale
        ``clusters`` rows for those ids) and resets ``query = NULL`` for
        every row with ``query = cycle``, in a single transaction.

        :raises ValueError: if any hit compound from ``cycle`` has
            already been docked (real docking data is never silently
            discarded — inspect manually instead), or has already been
            used as a query for a later cycle (which is structurally
            impossible if ``cycle`` is genuinely the latest search
            attempt; if this fires, undo the later cycle first).
        :returns: ``(hits_removed, queries_released)``.
        """
        stats = self.simsearch_cycle_stats(cycle)
        if stats.n_hits_docked:
            raise ValueError(
                f"cannot undo simsearch cycle {cycle}: {stats.n_hits_docked} of its "
                "hit compound(s) have already been docked; undoing would discard "
                "real docking results. Inspect the database manually before proceeding."
            )
        if stats.n_hits_used_as_query:
            raise ValueError(
                f"cannot undo simsearch cycle {cycle}: {stats.n_hits_used_as_query} of "
                "its hit compound(s) have already been used as queries for a later "
                f"cycle, so cycle {cycle} is not actually the latest search attempt. "
                "Undo the later cycle first."
            )
        hit_ids = [
            row[0]
            for row in self._conn.execute(
                "SELECT spacehastenid FROM data WHERE simsearch_cycle = ?", (cycle,)
            ).fetchall()
        ]
        if hit_ids:
            placeholders = ",".join("?" * len(hit_ids))
            self.ensure_extension_schema()
            self._conn.execute(
                f"DELETE FROM predictions WHERE spacehastenid IN ({placeholders})",
                hit_ids,
            )
            self._conn.execute(
                f"DELETE FROM clusters WHERE spacehastenid IN ({placeholders})", hit_ids
            )
            self._conn.execute("DELETE FROM data WHERE simsearch_cycle = ?", (cycle,))
        self._conn.execute("UPDATE data SET query = NULL WHERE query = ?", (cycle,))
        self.commit()
        return stats.n_hits, stats.n_queries

    # ----- counts -----

    def count_total(self) -> int:
        """Total number of compounds in the database."""
        row = self._conn.execute("SELECT COUNT(*) FROM data").fetchone()
        return int(row[0])

    def count_docked(self) -> int:
        """Number of compounds with a dock_score."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM data WHERE dock_score IS NOT NULL"
        ).fetchone()
        return int(row[0])

    def count_actives(self, threshold: float) -> int:
        """Number of docked compounds with dock_score < threshold."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM data WHERE dock_score IS NOT NULL AND dock_score < ?",
            (threshold,),
        ).fetchone()
        return int(row[0])

    # ----- model blob storage -----

    def store_model_blob(self, version: int, blob: bytes) -> None:
        self._conn.execute(
            "INSERT INTO models(model_version, model_tar) VALUES (?, ?)",
            (version, memoryview(blob)),
        )

    def load_model_blob(self, version: int) -> bytes:
        row = self._conn.execute(
            "SELECT model_tar FROM models WHERE model_version = ?", (version,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no model with version {version}")
        return bytes(row[0])

    def load_model_path(self, version: int, workdir: object) -> Path:
        """Resolve the on-disk path to a trained model checkpoint.

        Prefers the new on-disk registry layout
        (``workdir.model_dir(version)/model_0/pytorch_model.bin``). Falls
        back to extracting the legacy ``models.model_tar`` BLOB to that
        same location for back-compat with pre-Session-8 databases.

        ``workdir`` is typed as :class:`object` to avoid an import cycle
        with :mod:`spacehasten.workspace`; only its ``model_dir(version)``
        method is used.
        """
        model_dir_method = getattr(workdir, "model_dir", None)
        if not callable(model_dir_method):
            raise TypeError("workdir must expose a model_dir(version) method")
        model_dir = Path(model_dir_method(version))
        bin_path = model_dir / "model_0" / "pytorch_model.bin"
        if bin_path.exists():
            return bin_path

        # Back-compat: extract legacy BLOB.
        try:
            blob = self.load_model_blob(version)
        except KeyError as exc:
            raise FileNotFoundError(
                f"model version {version} not found on disk and no legacy BLOB available"
            ) from exc
        if not blob:
            raise FileNotFoundError(
                f"model version {version} has no on-disk checkpoint and the legacy"
                " BLOB is empty (likely written by Session-8+ training)"
            )
        import io
        import tarfile

        bin_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            tar.extractall(path=model_dir.parent)
        if not bin_path.exists():
            # Some legacy tars contain a top-level model_<name>_ver<N>/model_0/...
            # directory rather than model_0/ directly. Find the checkpoint.
            for candidate in model_dir.parent.rglob("pytorch_model.bin"):
                bin_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil as _shutil

                _shutil.copy(candidate, bin_path)
                break
        if not bin_path.exists():
            raise FileNotFoundError(
                "could not locate pytorch_model.bin after extracting legacy BLOB "
                f"for version {version}"
            )
        return bin_path

    # ----- docking blobs -----

    def store_dock_param(self, blob: bytes) -> None:
        self._conn.execute(
            "INSERT INTO docking_param VALUES (?)", (memoryview(blob),)
        )

    def store_dock_grid(self, blob: bytes) -> None:
        self._conn.execute(
            "INSERT INTO docking_grid VALUES (?)", (memoryview(blob),)
        )

    def load_dock_param(self) -> bytes:
        row = self._conn.execute("SELECT dock_param FROM docking_param").fetchone()
        if row is None:
            raise LookupError("docking_param is empty")
        return bytes(row[0])

    def load_dock_grid(self) -> bytes:
        row = self._conn.execute("SELECT dock_grid FROM docking_grid").fetchone()
        if row is None:
            raise LookupError("docking_grid is empty")
        return bytes(row[0])

    # ----- clusters -----

    def replace_clusters(self, rows: Iterable[ClusterRow]) -> None:
        c = self._conn.cursor()
        c.execute("DROP TABLE IF EXISTS clusters")
        c.execute(
            "CREATE TABLE clusters(spacehastenid INTEGER PRIMARY KEY,clusterid INTEGER)"
        )
        c.executemany(
            "INSERT INTO clusters(spacehastenid, clusterid) VALUES (?, ?)",
            ((r.spacehastenid, r.clusterid) for r in rows),
        )

    # ----- properties -----

    def replace_properties(self, props: PropertyRanges) -> None:
        c = self._conn.cursor()
        c.execute("DROP TABLE IF EXISTS properties")
        c.execute(
            "CREATE TABLE properties (property TEXT,is_double INTEGER,"
            "min_limit TEXT,max_limit TEXT)"
        )
        c.executemany(
            "INSERT INTO properties (property, is_double, min_limit, max_limit)"
            " VALUES (?, ?, ?, ?)",
            ((r.property, r.is_double, r.min_limit, r.max_limit) for r in props.to_rows()),
        )

    def load_properties(self) -> PropertyRanges | None:
        try:
            rows = self._conn.execute(
                "SELECT property, is_double, min_limit, max_limit FROM properties"
            ).fetchall()
        except sqlite3.OperationalError:
            return None
        if not rows:
            return None
        index: dict[str, tuple[str, str]] = {
            str(r[0]): (str(r[2]), str(r[3])) for r in rows
        }
        required = ("mw", "slogp", "hba", "hbd", "rotbonds", "tpsa")
        if not all(k in index for k in required):
            return None
        return PropertyRanges(
            mw=index["mw"],
            slogp=index["slogp"],
            hba=index["hba"],
            hbd=index["hbd"],
            rotbonds=index["rotbonds"],
            tpsa=index["tpsa"],
        )

    # ----- smarts filters -----

    def replace_smarts_filters(self, patterns: list[tuple[str, str]]) -> None:
        """Persist SMARTS include/exclude patterns.

        ``patterns`` is a list of ``(mode, smarts)`` pairs where *mode* is
        either ``'include'`` or ``'exclude'``.  Calling with an empty list
        clears any previously stored patterns.
        """
        c = self._conn.cursor()
        c.execute("DROP TABLE IF EXISTS smarts_filters")
        c.execute("CREATE TABLE smarts_filters (mode TEXT, pattern TEXT)")
        if patterns:
            c.executemany(
                "INSERT INTO smarts_filters (mode, pattern) VALUES (?, ?)",
                patterns,
            )

    def load_smarts_filters(self) -> list[tuple[str, str]]:
        """Return stored ``(mode, smarts)`` pairs, or ``[]`` if none stored."""
        try:
            rows = self._conn.execute(
                "SELECT mode, pattern FROM smarts_filters"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(str(r[0]), str(r[1])) for r in rows]


    def select_queries_for_simsearch(
        self,
        source: Literal["docked", "predicted"],
        strategy: Literal["greedy", "clustering"],
        limit: int,
    ) -> list[tuple[str, int]]:
        sql = self._simsearch_sql(source, strategy)
        return [(s, i) for s, i in self._conn.execute(sql, (limit,)).fetchall()]

    @classmethod
    def _simsearch_sql(
        cls,
        source: Literal["docked", "predicted"],
        strategy: Literal["greedy", "clustering"],
    ) -> str:
        if source == "docked" and strategy == "greedy":
            return cls._SQL_SIMSEARCH_DOCKED_GREEDY
        if source == "docked" and strategy == "clustering":
            return cls._SQL_SIMSEARCH_DOCKED_CLUSTERING
        if source == "predicted" and strategy == "greedy":
            return cls._SQL_SIMSEARCH_PREDICTED_GREEDY
        if source == "predicted" and strategy == "clustering":
            return cls._SQL_SIMSEARCH_PREDICTED_CLUSTERING
        raise ValueError(f"unknown simsearch SQL: source={source!r} strategy={strategy!r}")

    def select_compounds_to_dock(
        self, strategy: Literal["greedy", "clustering"], limit: int
    ) -> list[tuple[str, int]]:
        sql = (
            self._SQL_DOCK_GREEDY
            if strategy == "greedy"
            else self._SQL_DOCK_CLUSTERING
        )
        return [(s, i) for s, i in self._conn.execute(sql, (limit,)).fetchall()]

    def has_clusters(self) -> bool:
        """Whether ``clusters`` has any rows (i.e. ``cluster`` has run)."""
        return self._conn.execute("SELECT 1 FROM clusters LIMIT 1").fetchone() is not None

    def select_undocked_for_prediction(
        self, batch_size: int = 10000
    ) -> Iterator[tuple[str, int]]:
        cur = self._conn.execute(self._SQL_SELECT_UNDOCKED)
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            yield from rows

    def count_undocked_for_prediction(self) -> int:
        """Number of rows :meth:`select_undocked_for_prediction` would yield."""
        sql = self._SQL_SELECT_UNDOCKED.replace(
            "SELECT smiles, spacehastenid", "SELECT COUNT(*)", 1
        )
        row = self._conn.execute(sql).fetchone()
        return int(row[0]) if row else 0

    def select_training_data(self, cutoff: float = 10.0) -> list[tuple[str, float]]:
        return [
            (s, d)
            for s, d in self._conn.execute(self._SQL_SELECT_TRAINING, (cutoff,)).fetchall()
        ]

    def select_predictions(self, model_version: int | None = None) -> list[PredictionRow]:
        """Return persisted prediction history, optionally for one model version."""
        self.ensure_extension_schema()
        sql = (
            "SELECT spacehastenid, model_version, pred_score, epistemic_std, "
            "aleatoric_std, total_std, created_at FROM predictions"
        )
        params: tuple[int, ...] = ()
        if model_version is not None:
            sql += " WHERE model_version = ?"
            params = (model_version,)
        sql += " ORDER BY model_version, spacehastenid"
        return [PredictionRow(*row) for row in self._conn.execute(sql, params).fetchall()]

    def select_export_rows(self, cutoff: float) -> list[ExportRow]:
        return [
            ExportRow(
                smiles=row[0],
                spacehastenid=row[1],
                smilesid=row[2],
                dock_score=row[3],
                pred_score=row[4],
                spacelight=row[5],
                ftrees=row[6],
                dock_iteration=row[7],
                clusterid=row[8],
            )
            for row in self._conn.execute(self._SQL_SELECT_EXPORT, (cutoff,)).fetchall()
        ]

    def select_seed_rows(self) -> list[tuple[str, str, float]]:
        """Docked rows from the original seed batch (``dock_iteration == 0``).

        ``dock_iteration == 0`` uniquely identifies the seed round: it is
        set on pre-docked CSV seed imports (:meth:`insert_seed_docked`) and
        is also the iteration number assigned to the first ever ``dock``
        call (which docks previously-undocked ``.smi`` seeds). Every later
        ``dock`` call gets ``iteration = latest + 1 >= 1``, so compounds
        discovered in subsequent screening cycles never carry
        ``dock_iteration == 0``.
        """
        return [
            (row[0], row[1], row[2])
            for row in self._conn.execute(self._SQL_SELECT_SEEDS).fetchall()
        ]

    # ----- bulk applies (used by stages, not specified above but useful) -----

    def apply_dock_scores(self, rows: Sequence[tuple[float, int, int]]) -> None:
        """Apply many ``(dock_score, dock_iteration, spacehastenid)`` updates."""
        self._conn.executemany(self._SQL_UPDATE_DOCK_SCORE, rows)

    def apply_pred_scores(self, rows: Sequence[tuple[float, int, int]]) -> None:
        """Apply many ``(pred_score, pred_version, spacehastenid)`` updates."""
        self._conn.executemany(self._SQL_UPDATE_PRED_SCORE, rows)

    def apply_predictions(
        self,
        rows: Sequence[
            tuple[int, int, float, float | None, float | None, float | None]
        ],
    ) -> None:
        """Persist versioned predictions and update the legacy latest-score cache."""
        if not rows:
            return
        self.ensure_extension_schema()
        self._conn.executemany(
            self._SQL_UPDATE_PRED_SCORE,
            [(score, version, sid) for sid, version, score, _, _, _ in rows],
        )
        self._conn.executemany(self._SQL_UPSERT_PREDICTION, rows)


__all__ = [
    "ClusterRow",
    "Database",
    "DataRow",
    "EXTENSION_SCHEMA_STATEMENTS",
    "ExportRow",
    "ModelRow",
    "PredictionRow",
    "PropertyRanges",
    "PropertyRow",
    "SCHEMA_STATEMENTS",
    "SimsearchCycleStats",
]
