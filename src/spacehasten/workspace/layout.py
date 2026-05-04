"""Dual-root workspace layout — fast local + shared NFS.

A :class:`WorkDir` represents one SpaceHASTEN run that spans two directory
trees:

* **root** — on fast local storage (``/wrk`` or ``/fastwrk``).  Houses the
  ``.dbsh`` SQLite database, manifest, property-ranges TOML, and
  application-level logs.  Only the orchestrating head-node process touches
  these files, so they never need to be visible to compute nodes.
* **shared_root** — on NFS (``/data/$USER/SPACEHASTEN/$name``).  Houses
  stage artefacts (simsearch, docking, clustering, models, export,
  archive) and SLURM stdout/stderr logs.  Compute nodes read inputs and
  write results here.

For backward compatibility, ``shared_root`` defaults to ``root`` when not
provided.  Existing workspaces that live entirely on ``/data`` therefore
keep working without migration.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkDir:
    """Path provider for a dual-root SpaceHASTEN workspace.

    The constructor performs no I/O.  Use :meth:`bootstrap` to create the
    directory skeleton on disk.

    Parameters
    ----------
    root:
        Local fast-storage directory.  The ``.dbsh``, manifest, props
        file, and application logs live here.
    shared_root:
        NFS directory visible to all compute nodes.  Stage artefacts,
        models, and SLURM logs go here.  Defaults to *root* for backward
        compatibility with single-root workspaces.
    """

    root: Path
    shared_root: Path = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # frozen=True means we must use object.__setattr__ for fixups.
        if self.shared_root is None:
            object.__setattr__(self, "shared_root", self.root)

    # --------------------------------------------------------------------- #
    # Path accessors — local (fast storage)                                 #
    # --------------------------------------------------------------------- #

    @property
    def name(self) -> str:
        """Workspace name = root directory basename."""
        return self.root.name

    def dbsh(self) -> Path:
        """Path to the canonical ``<name>.dbsh`` SQLite file."""
        return self.root / f"{self.name}.dbsh"

    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def props_path(self) -> Path:
        return self.root / "props.toml"

    def logs_dir(self) -> Path:
        """Application-level logs (master + per-stage), on fast storage."""
        return self.root / "logs"

    # --------------------------------------------------------------------- #
    # Path accessors — shared (NFS, visible to compute nodes)               #
    # --------------------------------------------------------------------- #

    def simsearch_dir(self, cycle: int) -> Path:
        return self.shared_root / "simsearch" / f"cycle{cycle}"

    def docking_dir(self, iteration: int) -> Path:
        return self.shared_root / "docking" / f"iter{iteration}"

    def model_dir(self, version: int) -> Path:
        return self.shared_root / "models" / f"v{version}"

    def clustering_dir(self) -> Path:
        return self.shared_root / "clustering"

    def archive_dir(self) -> Path:
        return self.shared_root / "archive"

    def slurm_logs_dir(self, job_name: str) -> Path:
        """SLURM stdout/stderr logs, on shared storage."""
        return self.shared_root / "logs" / "slurm" / job_name

    def export_dir(self) -> Path:
        return self.shared_root / "export"

    # --------------------------------------------------------------------- #
    # Bootstrap                                                             #
    # --------------------------------------------------------------------- #

    @classmethod
    def bootstrap(
        cls,
        root: Path,
        name: str | None = None,
        *,
        shared_root: Path | None = None,
    ) -> WorkDir:
        """Create the workspace skeleton on disk.

        Creates both the local root (``root/``, ``root/logs/``) and the
        shared root (``shared_root/``, ``shared_root/models/``).
        Idempotent — safe to call on an already-bootstrapped workspace.

        ``name`` is recorded in the manifest; it defaults to
        ``root.name``.  ``shared_root`` defaults to ``root`` for
        single-root backward compatibility.
        """
        from .manifest import Manifest

        root = Path(root)
        shared = Path(shared_root) if shared_root is not None else root

        # Local root skeleton.
        root.mkdir(parents=True, exist_ok=True)
        wd = cls(root=root, shared_root=shared)
        wd.logs_dir().mkdir(exist_ok=True)

        # Shared root skeleton.
        shared.mkdir(parents=True, exist_ok=True)
        (shared / "models").mkdir(exist_ok=True)

        manifest_path = wd.manifest_path()
        if not manifest_path.exists():
            Manifest(
                name=name or wd.name,
                shared_root=str(shared) if shared != root else None,
            ).save(manifest_path)
        return wd

    # --------------------------------------------------------------------- #
    # Disk policy                                                           #
    # --------------------------------------------------------------------- #

    def warn_if_wrong_disk(self) -> str | None:
        """Warn (via :mod:`logging`) when ``root`` is on ``/data``.

        The database (``root``) should live on fast local storage
        (``/wrk`` or ``/fastwrk``), not on NFS.

        Returns the suggested fast-storage path when a warning was
        emitted, otherwise ``None``.
        """
        try:
            resolved = self.root.resolve()
        except OSError:
            resolved = self.root
        if str(resolved).startswith("/data"):
            user = os.environ.get("USER", "user")
            fast = "/wrk"
            suggested = f"{fast}/{user}/SPACEHASTEN/{self.name}/"
            logger.warning(
                "Workspace root %s is on /data (NFS). The database will "
                "be slow. Consider placing the root on fast local storage: %s",
                resolved,
                suggested,
            )
            return suggested
        return None
