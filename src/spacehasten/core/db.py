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
    "CREATE TABLE IF NOT EXISTS properties (property TEXT,is_double INTEGER,min_limit TEXT,max_limit TEXT)",
    "CREATE TABLE IF NOT EXISTS clusters(spacehastenid INTEGER PRIMARY KEY,clusterid INTEGER)",
    "CREATE INDEX IF NOT EXISTS idx_reghash ON data(reghash)",
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
class ExportRow:
    smiles: str
    spacehastenid: int
    smilesid: str
    dock_score: float
    pred_score: float | None
    spacelight: float | None
    ftrees: float | None
    dock_iteration: int | None
    clusterid: int


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
        " FROM data, clusters\n"
        " WHERE data.spacehastenid = clusters.spacehastenid AND dock_score <= ?\n"
        " ORDER BY dock_score"
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
        for stmt in SCHEMA_STATEMENTS:
            c.execute(stmt)
        self._conn.commit()

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
    ) -> int:
        c = self._conn.execute(
            "INSERT INTO data("
            "reghash, smiles, smilesid, spacelight, ftrees, pred_score, simsearch_cycle"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (reghash, smiles, smilesid, spacelight, ftrees, pred_score, simsearch_cycle),
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

    # ----- acquisition selects (§A.6 verbatim) -----

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

    def select_undocked_for_prediction(
        self, batch_size: int = 10000
    ) -> Iterator[tuple[str, int]]:
        cur = self._conn.execute(self._SQL_SELECT_UNDOCKED)
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            yield from rows

    def select_training_data(self, cutoff: float = 10.0) -> list[tuple[str, float]]:
        return [
            (s, d)
            for s, d in self._conn.execute(self._SQL_SELECT_TRAINING, (cutoff,)).fetchall()
        ]

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

    # ----- bulk applies (used by stages, not specified above but useful) -----

    def apply_dock_scores(self, rows: Sequence[tuple[float, int, int]]) -> None:
        """Apply many ``(dock_score, dock_iteration, spacehastenid)`` updates."""
        self._conn.executemany(self._SQL_UPDATE_DOCK_SCORE, rows)

    def apply_pred_scores(self, rows: Sequence[tuple[float, int, int]]) -> None:
        """Apply many ``(pred_score, pred_version, spacehastenid)`` updates."""
        self._conn.executemany(self._SQL_UPDATE_PRED_SCORE, rows)


__all__ = [
    "ClusterRow",
    "Database",
    "DataRow",
    "ExportRow",
    "ModelRow",
    "PropertyRanges",
    "PropertyRow",
    "SCHEMA_STATEMENTS",
]
