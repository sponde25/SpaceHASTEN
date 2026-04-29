"""SpaceHASTEN installer (new package, Session 15 cutover).

The legacy installer copied loose ``.py`` files into a target directory.
The new installer instead installs the ``spacehasten`` Python package
(via ``pip install``) and writes a site-specific ``spacehasten.ini``
into the target directory along with the verify fixtures.

Usage (interactive)::

    python3 install_spacehasten.py

The installer never edits anything outside the chosen install
directory and the active Python environment.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Default values mirror the legacy installer; we no longer import
# legacy ``cfg.py`` so we duplicate the literals here.
SPACEHASTEN_VERSION = "0.11.0.dev0"

REPO_ROOT = Path(__file__).resolve().parent
VERIFY_FIXTURES = (
    "examples.smi",
    "example.smi",
    "example.csv",
    "test_dock.in",
    "grid-test_dock.zip",
)

DEFAULTS = {
    "spacelight": "/data/programs/BiosolveIT/spacelight-2.0.0-Linux-x64/spacelight",
    "ftrees": "/data/programs/BiosolveIT/ftrees-7.0.0-Linux-x64/ftrees",
    "spaces_dir": "/data/programs/BiosolveIT/spaces_new",
    "default_space": "/data/programs/BiosolveIT/spaces_new/REALSpace_83bn_2025-09.space",
    "default_seeds": (
        "/data/programs/BiosolveIT/spaces_seeds/"
        "Enamine_Diverse_REAL_drug-like_48.2M_cxsmiles.cxsmiles.bz2"
    ),
    "seeds_dir": "/data/programs/BiosolveIT/spaces_seeds",
    "scratch": "/wrk",
    "prepare_anaconda": "source /data/programs/oce/actoce",
    "activate_chemprop": "conda activate chemprop-2.1.2",
    "activate_clustering": "conda activate fpsim2-0.7.3",
    "slurm_queue": "jobs",
    "slurm_gpu_parameter": "--gpus=1",
    "slurm_cpu_clustering": "64",
    "gpu_exclusive": "1",
    "schrodinger_feature_flags": "",
}


def _ask_for_file(default: str, desc: str | None = None) -> str:
    if desc is None:
        desc = default.split("/")[-1]
    answer = input(
        f"Please enter the path to {desc} executable [default:{default}]: "
    ).strip()
    if not answer:
        answer = default
    if not Path(answer).exists():
        print(f"The specified path does not exist: {answer}")
        sys.exit(1)
    return answer


def _ask_for_dir(default: str, desc: str, exist: bool = True) -> str:
    answer = input(
        f"Please enter the path to {desc} directory [default:{default}]: "
    ).strip()
    if not answer:
        answer = default
    if exist and not Path(answer).is_dir():
        print(f"The specified path is not a directory: {answer}")
        sys.exit(1)
    return answer


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [default:{default}]" if default else ""
    answer = input(f"{prompt}{suffix}: ").strip()
    return answer or default


def _print_banner() -> None:
    print()
    print(r" ___                  _  _   _   ___ _____ ___ _  _")
    print(r"/ __|_ __  __ _ __ ___| || | /_\ / __|_   _| __| \| |")
    print(r"\__ \ '_ \/ _` / _/ -_) __ |/ _ \\__ \ | |  | _|| .` |")
    print(r"|___/ .__/\__,_\__\___|_||_/_/ \_\___/ |_| |___|_|\_|")
    print(r"    |_|")
    print()
    print(f"SpaceHASTEN installer {SPACEHASTEN_VERSION}\n")


def _write_ini(path: Path, *, answers: dict[str, str]) -> None:
    a = answers
    lines: list[str] = []
    lines.append("[General]")
    lines.append("SCHEDULER = slurm")
    lines.append(f"PREPARE_ANACONDA = {a['prepare_anaconda']}")
    lines.append(f"ACTIVATE_CHEMPROP = {a['activate_chemprop']}")
    lines.append(f"ACTIVATE_CLUSTERING = {a['activate_clustering']}")
    lines.append(f"GPU_EXCLUSIVE = {a['gpu_exclusive']}")
    lines.append("CPU_COUNT_SEARCH = 2")
    lines.append("CPU_COUNT_DOCK = 1")
    lines.append("CPU_COUNT_PREDICT = 1")
    lines.append("CPU_COUNT_CONTROL = 1")
    lines.append(f"CPU_COUNT_CLUSTERING = {a['slurm_cpu_clustering']}")
    if a["schrodinger_feature_flags"]:
        lines.append(
            f"SCHRODINGER_FEATURE_FLAGS = {a['schrodinger_feature_flags']}"
        )
    lines.append("")
    lines.append("[Paths]")
    lines.append(f"EXE_SPACELIGHT_DEFAULT = {a['spacelight']}")
    lines.append(f"EXE_FTREES_DEFAULT = {a['ftrees']}")
    lines.append(f"SPACES_DIR_DEFAULT = {a['spaces_dir']}")
    lines.append(f"SPACES_FILE_DEFAULT = {a['default_space']}")
    lines.append(f"SCRATCH_DEFAULT = {a['scratch']}")
    lines.append(f"SEEDS_DIR_DEFAULT = {a['seeds_dir']}")
    lines.append(f"SEEDS_FILE_DEFAULT = {a['default_seeds']}")
    # Point export_poses_script at the legacy file shipped with the repo.
    legacy_export = REPO_ROOT / "legacy" / "export_poses.py"
    if legacy_export.exists():
        lines.append(f"EXPORT_POSES_SCRIPT = {legacy_export}")
    lines.append("")
    lines.append("[Slurm]")
    lines.append(f"SLURM_PARTITION = {a['slurm_queue']}")
    lines.append(f"SLURM_GPU_PARAMETER = {a['slurm_gpu_parameter']}")
    lines.append("")
    lines.append("[SGE]")
    lines.append("SGE_QUEUE = jobs")
    lines.append("SGE_PE = smp")
    lines.append("SGE_GPU_PARAMETER = -l gpu=1")
    lines.append("")
    lines.append("[Properties]")
    lines.append("MW_MIN = 0.0")
    lines.append("MW_MAX = 500.0")
    lines.append("SLOGP_MIN = -10.0")
    lines.append("SLOGP_MAX = 5.0")
    lines.append("HBA_MIN = 0")
    lines.append("HBA_MAX = 10")
    lines.append("HBD_MIN = 0")
    lines.append("HBD_MAX = 5")
    lines.append("ROTBONDS_MIN = 0")
    lines.append("ROTBONDS_MAX = 10")
    lines.append("TPSA_MIN = 0.0")
    lines.append("TPSA_MAX = 140.0")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _pip_install_package() -> None:
    print("\nInstalling the spacehasten package into the active Python "
          "environment...")
    cmd = [sys.executable, "-m", "pip", "install", str(REPO_ROOT)]
    print("  $", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"pip install failed (rc={rc}). Aborting installation.")
        sys.exit(1)


def main() -> None:
    _print_banner()
    print("This installer writes a site config (`spacehasten.ini`) to a "
          "directory of your choice, copies the verify fixtures, and "
          "installs the `spacehasten` Python package into the active "
          "Python environment via `pip install`.\n")
    print("NOTE: the install directory must be visible to all compute "
          "nodes (NFS).")
    print("NOTE: SLURM is the default scheduler.\n")

    install_dir = _ask_for_dir(
        f"/data/programs/spacehasten-{SPACEHASTEN_VERSION}",
        "Installation",
        exist=False,
    )
    install_path = Path(install_dir)
    if install_path.exists():
        print(f"The specified path already exists: {install_path}")
        print("Installation aborted.")
        sys.exit(1)
    install_path.mkdir(parents=True)

    answers: dict[str, str] = {}
    answers["spacelight"] = _ask_for_file(DEFAULTS["spacelight"])
    answers["ftrees"] = _ask_for_file(DEFAULTS["ftrees"])
    answers["spaces_dir"] = _ask_for_dir(DEFAULTS["spaces_dir"], "BiosolveIT spaces")
    answers["default_space"] = _ask_for_file(DEFAULTS["default_space"], "default space")
    answers["default_seeds"] = _ask_for_file(
        DEFAULTS["default_seeds"], "default enumerated seeds"
    )
    answers["seeds_dir"] = _ask_for_dir(
        DEFAULTS["seeds_dir"], "Directory for enumerated seeds"
    )
    answers["scratch"] = _ask_for_dir(
        DEFAULTS["scratch"], "scratch (local fast disk)"
    )
    answers["prepare_anaconda"] = _ask(
        "Anaconda3 activation command", DEFAULTS["prepare_anaconda"]
    )
    answers["activate_chemprop"] = _ask(
        "Anaconda3 chemprop activation command", DEFAULTS["activate_chemprop"]
    )
    answers["activate_clustering"] = _ask(
        "Anaconda3 clustering activation command",
        DEFAULTS["activate_clustering"],
    )
    answers["gpu_exclusive"] = _ask(
        "Type 1 if you want node exclusivity for training/clustering, "
        "0 otherwise",
        DEFAULTS["gpu_exclusive"],
    )
    answers["slurm_queue"] = _ask("SLURM partition name", DEFAULTS["slurm_queue"])
    answers["slurm_gpu_parameter"] = _ask(
        "SLURM GPU parameter", DEFAULTS["slurm_gpu_parameter"]
    )
    answers["slurm_cpu_clustering"] = _ask(
        "Number of cores for clustering", DEFAULTS["slurm_cpu_clustering"]
    )
    answers["schrodinger_feature_flags"] = _ask(
        "SCHRODINGER_FEATURE_FLAGS such as -JOB_SERVER (ENTER to skip)",
        DEFAULTS["schrodinger_feature_flags"],
    )

    print("\nWriting site configuration...")
    ini_path = install_path / "spacehasten.ini"
    _write_ini(ini_path, answers=answers)

    print(f"  -> {ini_path}")
    for fixture in VERIFY_FIXTURES:
        src = REPO_ROOT / fixture
        if not src.exists():
            print(f"  WARNING: missing verify fixture {src}; verify --fixtures-dir "
                  "will need an explicit path.")
            continue
        dst = install_path / fixture
        shutil.copy(src, dst)
        print(f"  -> {dst}")

    _pip_install_package()

    print()
    print("=" * 60)
    print("SpaceHASTEN installed successfully.")
    print("=" * 60)
    print(f"Site config:    {ini_path}")
    print(f"Install dir:    {install_path}")
    print()
    print("Next steps:")
    print(f"  1. Verify the install on the cluster:")
    print(f"       spacehasten verify \\")
    print(f"           --config {ini_path} \\")
    print(f"           --fixtures-dir {install_path}")
    print(f"  2. Start a screening run, e.g.:")
    print(f"       spacehasten --config {ini_path} \\")
    print(f"           --db /data/$USER/SPACEHASTEN/myrun/myrun.dbsh \\")
    print(f"           import-seeds --smi seeds.smi \\")
    print(f"               --dock-params {install_path}/test_dock.in \\")
    print(f"               --dock-grid   {install_path}/grid-test_dock.zip")
    print()
    print(f"The legacy Tk GUI is available for one release as the "
          "`spacehasten-legacy-gui` console script.")
    print("This test should take around 15-30 minutes to run.")


if __name__ == "__main__":
    main()
