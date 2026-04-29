"""Schrödinger Phase/LigPrep + Glide file builders and result parser.

Ports the three legacy helpers from ``docking_functions.py``:

- :func:`write_phase_inp` ← ``write_confgen_file``
- :func:`write_glide_in`  ← ``write_docking_file``
- :func:`parse_glide_csv` ← inline pandas reducer in
  ``process_docking_results``

The Phase ``.inp`` and Glide ``.in`` are pure-text templates with a
handful of substituted basenames. The legacy ``write_docking_file`` also
read the Glide template + grid from the SQLite ``docking_param`` /
``docking_grid`` BLOBs; here, blob loading is the caller's job (it lives
in :mod:`spacehasten.core.db`) and we accept the param blob as an
argument so this module is pure I/O over text.
"""

from __future__ import annotations

import csv
from pathlib import Path

# Glide CSV column names (cf. CODEBASE_REFERENCE.md §A.7).
_TITLE_COLUMN = "title"
_SCORE_COLUMN = "r_i_docking_score"

# Lines in the Glide .in template that we strip (the orchestrator
# rewrites them per chunk). Compared in upper-case after stripping the
# leading whitespace, matching legacy behaviour.
_GLIDE_STRIP_KEYS = ("LIGANDFILE", "GRIDFILE")


def write_phase_inp(path: Path) -> None:
    """Write a Phase/LigPrep ``.inp`` referencing the SMI/PHDB by basename.

    The file references ``<stem>.smi`` for ``[SET:ORIGINAL_LIGANDS] FILES``
    and ``<stem>.phdb`` for ``[STAGE:MANAGE] DATABASE`` where ``<stem>``
    is ``path.stem``. Mirrors legacy ``write_confgen_file`` byte-for-byte
    apart from the whitespace inside the rewritten template (which was
    already not load-bearing in Phase).
    """
    path = Path(path)
    stem = path.stem
    smi_basename = f"{stem}.smi"
    phdb_basename = f"{stem}.phdb"
    text = (
        "[SET:ORIGINAL_LIGANDS]\n"
        "    VARCLASS   Structures\n"
        f"    FILES   {smi_basename},\n"
        "\n"
        "[STAGE:LIGPREP]\n"
        "    STAGECLASS   ligprep.LigPrepStage\n"
        "    INPUTS   ORIGINAL_LIGANDS,\n"
        "    OUTPUTS   LIGPREP_OUT,\n"
        "    RECOMBINE   YES\n"
        "    RETITLE   YES\n"
        "    MIXLIGS   YES\n"
        "    SKIP_BAD_LIGANDS   YES\n"
        "    UNIQUEFIELD   s_m_title\n"
        "    OUTCOMPOUNDFIELD   s_m_title\n"
        "    USE_EPIK   YES\n"
        "    METAL_BINDING   NO\n"
        "    PH   7.0\n"
        "    PHT   2.0\n"
        "    NRINGCONFS   1\n"
        "    COMBINEOUTS   NO\n"
        "    STEREO_SOURCE   parities\n"
        "    NUM_STEREOISOMERS   32\n"
        "    REGULARIZE   NO\n"
        "\n"
        "[STAGE:POSTLIGPREP]\n"
        "    STAGECLASS   ligprep.PostLigPrepStage\n"
        "    INPUTS   LIGPREP_OUT,\n"
        "    OUTPUTS   POSTLIGPREP_OUT,\n"
        "    UNIQUEFIELD   s_m_title\n"
        "    OUTVARIANTFIELD   s_phase_variant\n"
        "    PRESERVE_NJOBS   YES\n"
        "    LIMIT_STEREOISOMERS   YES\n"
        "    MAXSTEREO   4\n"
        "    REMOVE_PENALIZED_STATES   YES\n"
        "\n"
        "[STAGE:MANAGE]\n"
        "    STAGECLASS   phase.DBManageStage\n"
        "    INPUTS   POSTLIGPREP_OUT,\n"
        "    OUTPUTS   DATABASE,\n"
        f"    DATABASE  {phdb_basename}\n"
        "    NEW   YES\n"
        "    MULTIPLE_CONFS   NO\n"
        "    CONSIDER_STEREO   NO\n"
        "    GENERATE_PROPS   NO\n"
        "    CREATE_SUBSET   NO\n"
        "    SKIP_DUPLICATES   NO\n"
        "\n"
        "[STAGE:CONFSITES]\n"
        "    STAGECLASS   phase.DBConfSitesStage\n"
        "    INPUTS   DATABASE,\n"
        "    CONFS   auto\n"
        "    MAX_CONFS   1\n"
        "    GENERATE_PROPS   YES\n"
        "\n"
        "[USEROUTS]\n"
        "    USEROUTS   DATABASE,\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_glide_in(
    path: Path,
    dock_param_blob: bytes,
    *,
    ligand_stem: str,
    grid_basename: str = "glide_grid.zip",
) -> None:
    """Write a Glide ``.in`` derived from the stored ``dock_param`` blob.

    The legacy template stored in ``docking_param`` is a Glide ``.in``
    file that may contain its own ``LIGANDFILE`` / ``GRIDFILE`` lines
    (those refer to the user's training run). We strip both and rewrite
    them to point at the per-chunk LigPrep output and to the local
    extracted grid zip.

    :param path: output ``.in`` path; its parent is created if missing.
    :param dock_param_blob: raw bytes of the stored Glide template.
    :param ligand_stem: per-chunk basename without extension; used to
        compute ``LIGANDFILE   <ligand_stem>_1.maegz`` (matches the
        ``$SCHRODINGER/phase_database ... -omae <stem> -get 1`` output).
    :param grid_basename: name of the grid zip in the per-task working
        directory; defaults to ``glide_grid.zip``.
    """
    template_text = dock_param_blob.decode("utf-8", errors="replace")
    kept_lines: list[str] = []
    for line in template_text.splitlines(keepends=True):
        stripped = line.strip().split()
        if stripped and stripped[0].upper() in _GLIDE_STRIP_KEYS:
            continue
        kept_lines.append(line)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8") as w:
        w.write(f"LIGANDFILE   {ligand_stem}_1.maegz\n")
        w.write(f"GRIDFILE     {grid_basename}\n")
        for line in kept_lines:
            w.write(line)


def parse_glide_csv(csv_path: Path) -> dict[str, float]:
    """Reduce a Glide pose CSV to ``{title: min_docking_score}``.

    Mirrors the legacy ``pandas.groupby('title').min()`` reducer without
    the pandas dependency: each row contributes its ``r_i_docking_score``
    if (a) the score is non-empty and parseable and (b) it is lower than
    any previous score for the same title.
    """
    csv_path = Path(csv_path)
    out: dict[str, float] = {}
    with csv_path.open("rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or _TITLE_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"{csv_path}: expected a column named {_TITLE_COLUMN!r}, "
                f"got {reader.fieldnames!r}"
            )
        if _SCORE_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"{csv_path}: expected a column named {_SCORE_COLUMN!r}, "
                f"got {reader.fieldnames!r}"
            )
        for row in reader:
            title = row.get(_TITLE_COLUMN)
            raw_score = row.get(_SCORE_COLUMN)
            if title is None or raw_score is None or raw_score == "":
                continue
            try:
                score = float(raw_score)
            except ValueError:
                continue
            prev = out.get(title)
            if prev is None or score < prev:
                out[title] = score
    return out


__all__ = [
    "parse_glide_csv",
    "write_glide_in",
    "write_phase_inp",
]
