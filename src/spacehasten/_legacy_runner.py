"""Console-script entry points that bridge to the legacy tree.

These exist to give users one release of overlap during the Session 15
cutover. They locate the ``legacy/`` directory adjacent to this package
in either an editable install (where ``legacy/`` sits at the repo root)
or in a sibling directory when shipped as data, then run the requested
legacy script via :func:`runpy.run_path` after putting that directory
on ``sys.path`` so the legacy ``import functions`` style imports
resolve.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _find_legacy_dir() -> Path:
    """Locate the ``legacy/`` directory.

    Search order:

    1. ``$SPACEHASTEN_LEGACY_DIR`` environment variable.
    2. Adjacent to the repo root that contains this installed package.
    3. ``legacy/`` two levels up from this file (``src/spacehasten/``).
    """
    env = os.environ.get("SPACEHASTEN_LEGACY_DIR")
    if env:
        p = Path(env)
        if p.is_dir():
            return p

    here = Path(__file__).resolve().parent
    for parent in here.parents:
        cand = parent / "legacy"
        if cand.is_dir() and (cand / "gui.py").exists():
            return cand

    raise SystemExit(
        "spacehasten-legacy-gui: could not locate the legacy/ directory. "
        "Set SPACEHASTEN_LEGACY_DIR to the absolute path that contains "
        "gui.py (typically the SpaceHASTEN install root)."
    )


def _run_legacy_script(name: str) -> None:
    legacy = _find_legacy_dir()
    script = legacy / name
    if not script.exists():
        raise SystemExit(f"spacehasten-legacy: {script} not found")
    sys.path.insert(0, str(legacy))
    os.chdir(legacy)
    runpy.run_path(str(script), run_name="__main__")


def gui_main() -> None:
    """Run the legacy Tk GUI (``legacy/gui.py``)."""
    _run_legacy_script("gui.py")
