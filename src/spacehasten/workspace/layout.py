"""Single-root workspace layout.

A :class:`WorkDir` represents one SpaceHASTEN run on disk. All artefacts —
the SQLite ``.dbsh``, simsearch/docking/clustering scratch, model registry,
logs, manifest — live as visible subdirectories of the same root.

This replaces the legacy split between ``<cwd>/<name>.dbsh`` and the
``$HOME/SPACEHASTEN/{SIMSEARCH,DOCKING,TRAIN,CLUSTERING}_*`` family of
directories (see CODEBASE_REFERENCE.md §A.3).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkDir:
    """Path provider for a single SpaceHASTEN workspace.

    The constructor performs no I/O. Use :meth:`bootstrap` to create the
    directory skeleton on disk.
    """

    root: Path

    # --------------------------------------------------------------------- #
    # Path accessors                                                        #
    # --------------------------------------------------------------------- #

    @property
    def name(self) -> str:
        """Workspace name = root directory basename."""
        return self.root.name

    def dbsh(self) -> Path:
        """Path to the canonical ``<name>.dbsh`` SQLite file."""
        return self.root / f"{self.name}.dbsh"

    def simsearch_dir(self, cycle: int) -> Path:
        return self.root / "simsearch" / f"cycle{cycle}"

    def docking_dir(self, iteration: int) -> Path:
        return self.root / "docking" / f"iter{iteration}"

    def model_dir(self, version: int) -> Path:
        return self.root / "models" / f"v{version}"

    def clustering_dir(self) -> Path:
        return self.root / "clustering"

    def archive_dir(self) -> Path:
        return self.root / "archive"

    def logs_dir(self) -> Path:
        return self.root / "logs"

    def slurm_logs_dir(self, job_name: str) -> Path:
        return self.logs_dir() / "slurm" / job_name

    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def props_path(self) -> Path:
        return self.root / "props.toml"

    # --------------------------------------------------------------------- #
    # Bootstrap                                                             #
    # --------------------------------------------------------------------- #

    @classmethod
    def bootstrap(cls, root: Path, name: str | None = None) -> WorkDir:
        """Create the workspace skeleton on disk.

        Creates ``root/``, ``root/logs/``, and ``root/models/`` (idempotent),
        and writes an empty manifest if none exists. ``name`` is recorded in
        the manifest; it defaults to ``root.name``.
        """
        # Local import to avoid a circular dependency at module import time.
        from .manifest import Manifest

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        wd = cls(root=root)
        wd.logs_dir().mkdir(exist_ok=True)
        (root / "models").mkdir(exist_ok=True)

        manifest_path = wd.manifest_path()
        if not manifest_path.exists():
            Manifest(name=name or wd.name).save(manifest_path)
        return wd

    # --------------------------------------------------------------------- #
    # Disk policy                                                           #
    # --------------------------------------------------------------------- #

    def warn_if_wrong_disk(self) -> str | None:
        """Warn (via :mod:`logging`) when ``root`` is on ``/wrk``.

        Returns the suggested canonical path when a warning was emitted,
        otherwise ``None``. Caller may surface the suggestion to the user.
        """
        try:
            resolved = self.root.resolve()
        except OSError:
            resolved = self.root
        if str(resolved).startswith("/wrk"):
            user = os.environ.get("USER", "user")
            suggested = f"/data/{user}/SPACEHASTEN/{self.name}/"
            logger.warning(
                "Workspace %s is on /wrk (fast scratch). The canonical "
                "location is on /data; consider %s",
                resolved,
                suggested,
            )
            return suggested
        return None
