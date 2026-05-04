"""Archive stage — bundle, restore, extract, clean.

Replaces legacy ``archive_functions.archive``/``restore``/``clean``.
The new single-root workspace makes this almost trivial: an archive is
just a tar of ``workdir.root``. Two flavours::

    archive_create(workdir, bundle=False) → <root>.archived-spacehasten
    archive_create(workdir, bundle=True)  → <root>.archived-spacehasten.tgz

``archive_restore`` is the inverse of the non-bundle (``.tar``) form;
``archive_extract`` is the inverse of the bundle (``.tgz``) form.
``archive_clean`` removes the regenerable per-cycle / per-iteration
scratch directories from ``workdir`` (analogous to legacy clean), but
preserves the ``.dbsh``, manifest, models registry, and logs so the
workspace remains usable without restoring from the archive.
"""

from __future__ import annotations

import logging
import shutil
import tarfile
from pathlib import Path

from spacehasten.workspace.layout import WorkDir

logger = logging.getLogger(__name__)


_ARCHIVE_SUFFIX = ".archived-spacehasten"
_BUNDLE_SUFFIX = ".archived-spacehasten.tgz"

#: Top-level subdirectories considered "regenerable scratch" — removed by
#: :func:`archive_clean`. The ``.dbsh``, ``manifest.json``, ``props.toml``,
#: ``models/``, and ``logs/`` are preserved.
_REGENERABLE_DIRS: tuple[str, ...] = ("simsearch", "docking", "clustering")


def _archive_path(workdir: WorkDir, *, bundle: bool) -> Path:
    suffix = _BUNDLE_SUFFIX if bundle else _ARCHIVE_SUFFIX
    return workdir.root.parent / f"{workdir.name}{suffix}"


def archive_create(workdir: WorkDir, *, bundle: bool = False) -> Path:
    """Tar the workspace root.

    :param bundle: when ``True``, produce a gzipped, self-contained
        ``.tgz``; otherwise an uncompressed ``.tar`` (still self-contained
        in the new single-root layout — there are no cross-FS symlinks
        to follow).
    :returns: path to the created archive.
    """
    if not workdir.root.is_dir():
        raise FileNotFoundError(f"workspace root missing: {workdir.root}")

    out = _archive_path(workdir, bundle=bundle)
    if out.exists():
        out.unlink()
    if bundle:
        with tarfile.open(out, mode="w:gz", dereference=True) as tf:
            tf.add(workdir.root, arcname=workdir.name)
    else:
        with tarfile.open(out, mode="w", dereference=True) as tf:
            tf.add(workdir.root, arcname=workdir.name)
    logger.info("Created archive %s", out)
    return out


def _extract_archive(archive_path: Path, target_workdir: WorkDir) -> WorkDir:
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    target_root = target_workdir.root
    target_root.parent.mkdir(parents=True, exist_ok=True)
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty target {target_root}"
        )
    with tarfile.open(archive_path, mode="r:*") as tf:
        # Strip the leading ``<name>/`` prefix so contents land directly
        # under ``target_root``.
        members = []
        for m in tf.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) == 2:
                m.name = parts[1]
            else:
                m.name = ""
            if m.name:
                members.append(m)
        target_root.mkdir(parents=True, exist_ok=True)
        try:
            tf.extractall(target_root, members=members, filter="data")
        except TypeError:  # pragma: no cover - older Python fallback
            tf.extractall(target_root, members=members)
    logger.info("Extracted %s into %s", archive_path, target_root)
    return target_workdir


def archive_restore(archive_path: Path, target_workdir: WorkDir) -> WorkDir:
    """Restore a non-bundle (``.archived-spacehasten``) archive."""
    return _extract_archive(archive_path, target_workdir)


def archive_extract(bundle_path: Path, target_workdir: WorkDir) -> WorkDir:
    """Inverse of :func:`archive_create` with ``bundle=True``."""
    return _extract_archive(bundle_path, target_workdir)


def archive_clean(workdir: WorkDir) -> int:
    """Remove regenerable scratch (simsearch/, docking/, clustering/).

    Preserves ``.dbsh``, ``manifest.json``, ``props.toml``, ``models/``,
    and ``logs/`` so the workspace remains usable for new cycles.

    :returns: number of top-level directories removed.
    """
    n_removed = 0
    for name in _REGENERABLE_DIRS:
        target = workdir.shared_root / name
        if target.exists():
            shutil.rmtree(target)
            n_removed += 1
            logger.info("Removed %s", target)
    return n_removed


__all__ = [
    "archive_clean",
    "archive_create",
    "archive_extract",
    "archive_restore",
]
