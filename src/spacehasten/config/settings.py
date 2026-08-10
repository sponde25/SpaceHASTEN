"""Pydantic-Settings replacement for legacy ``cfg.SpaceHASTENConfiguration``.

Loads configuration from (in priority order, lowest to highest):

    1. Class defaults
    2. INI file (legacy ``spacehasten.ini``)
    3. TOML file (new format)
    4. Explicit ``cli_overrides`` mapping

Validation of external tools and paths is **opt-in** via
:meth:`Settings.validate_install`. Construction never touches the filesystem
beyond reading configured INI/TOML files.
"""

from __future__ import annotations

import configparser
import os
import shutil
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Section models                                                              #
# --------------------------------------------------------------------------- #


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class GeneralSettings(_Section):
    scheduler: str = "slurm"
    prepare_anaconda: str | None = None
    activate_chemprop: str | None = None
    activate_clustering: str | None = None
    gpu_exclusive: str = "1"
    cpu_count_train: str = "8"
    cpu_count_search: str = "2"
    cpu_count_dock: str = "1"
    cpu_count_predict: str = "1"
    cpu_count_control: str = "1"
    cpu_count_clustering: str = "64"
    schrodinger_feature_flags: str | None = None
    model_spec_path: str | None = None
    model_hparams_path: str | None = None
    train_batch_size: int = 1024
    train_epochs: int = 50
    train_num_workers: int = 8
    train_devices: str = "1"
    train_mp_hidden_size: int = 300
    train_mp_depth: int = 3
    train_ffn_hidden_size: int = 300
    train_ffn_layers: int = 1
    train_dropout: float = 0.1
    train_activation: str = "relu"
    train_batch_norm: int = 0
    train_warmup_epochs: int = 2
    train_init_lr: float = 1e-4
    train_max_lr: float = 1e-3
    train_final_lr: float = 1e-4
    train_early_stopping_patience: int = 5
    train_early_stopping_min_delta: float = 0.0
    pred_batch_size: int = 32
    pred_num_workers: int = 0
    pred_accelerator: str = "cpu"
    pred_devices: str = "1"
    pred_chunk_size: int = 12345
    sim_spacelight_default: float = 0.5
    sim_ftrees_default: float = 0.9
    nnn_default: int = 10000
    field_similarity_spacelight: str = "fingerprint-similarity"
    field_similarity_ftrees: str = "pharmacophore-similarity"
    seeds_count: int = 1000000
    seeds_cpu: int = 4
    cpu_count_library: str = "1"
    library_infer_batch_size: int = 32
    library_infer_num_workers: int = 0
    library_infer_accelerator: str = "cpu"
    library_infer_devices: str = "1"
    library_default_top_pct: float = 1.0
    library_build_chunk_size: int = 2_000_000


class PathsSettings(_Section):
    exe_spacelight_default: str = "/data/programs/BiosolveIT/spacelight-2.0.0-Linux-x64/spacelight"
    exe_ftrees_default: str = "/data/programs/BiosolveIT/ftrees-7.0.0-Linux-x64/ftrees"
    scratch_default: str = "/wrk"
    shared_data_root: str = "/data"
    spaces_dir_default: str = "/data/programs/BiosolveIT/spaces_new"
    spaces_file_default: str = (
        "/data/programs/BiosolveIT/spaces_new/REALSpace_83bn_2025-09.space"
    )
    seeds_dir_default: str = "/data/programs/BiosolveIT/spaces_seeds"
    seeds_file_default: str = (
        "/data/programs/BiosolveIT/spaces_seeds/"
        "Enamine_Diverse_REAL_drug-like_48.2M_cxsmiles.cxsmiles.bz2"
    )
    exe_clustering_default: str | None = None  # resolved at install/use
    schrodinger_run: str = "$SCHRODINGER/run"
    export_poses_script: str | None = None  # path to legacy export_poses.py
    # Absolute path to the directory containing ``remote/{train,predict,...}``
    # on a filesystem visible to compute nodes. Set by ``install_spacehasten``
    # and used by ``Settings.remote_script_path``.
    spacehasten_src_dir: str | None = None
    # Default library store directory for `spacehasten library-screen`
    # (produced by `spacehasten library-build`); overridable with --library.
    library_store_default: str | None = None


class SlurmSettings(_Section):
    slurm_partition: str | None = None
    slurm_gpu_parameter: str | None = None


class SGESettings(_Section):
    sge_queue: str | None = None
    sge_pe: str | None = None
    sge_gpu_parameter: str | None = None


# --------------------------------------------------------------------------- #
# Section registry                                                            #
# --------------------------------------------------------------------------- #


_SECTIONS: dict[str, tuple[str, type[_Section]]] = {
    # canonical-name : (ini-section-name, model)
    "general": ("General", GeneralSettings),
    "paths": ("Paths", PathsSettings),
    "slurm": ("Slurm", SlurmSettings),
    "sge": ("SGE", SGESettings),
}


def _coerce(model_cls: type[_Section], raw: Mapping[str, Any]) -> dict[str, Any]:
    """Filter to known fields and let pydantic do the type coercion."""
    fields = model_cls.model_fields
    return {k: v for k, v in raw.items() if k in fields}


def _read_ini(path: Path) -> dict[str, dict[str, Any]]:
    parser = configparser.ConfigParser()
    parser.read(path)
    out: dict[str, dict[str, Any]] = {key: {} for key in _SECTIONS}
    for canonical, (ini_name, _model) in _SECTIONS.items():
        if parser.has_section(ini_name):
            out[canonical] = dict(parser.items(ini_name))
    return out


def _read_toml(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    out: dict[str, dict[str, Any]] = {key: {} for key in _SECTIONS}
    for canonical in _SECTIONS:
        section_data = data.get(canonical)
        if isinstance(section_data, Mapping):
            out[canonical] = dict(section_data)
    return out


def _deep_merge(
    base: dict[str, dict[str, Any]], overlay: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    for section, values in overlay.items():
        if section not in base:
            continue
        base[section].update(values)
    return base


# --------------------------------------------------------------------------- #
# Aggregate Settings                                                          #
# --------------------------------------------------------------------------- #


class Settings(BaseModel):
    """Aggregate configuration matching legacy ``[General]/[Paths]/[Slurm]/[SGE]``."""

    model_config = ConfigDict(extra="forbid")

    general: GeneralSettings = Field(default_factory=GeneralSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    slurm: SlurmSettings = Field(default_factory=SlurmSettings)
    sge: SGESettings = Field(default_factory=SGESettings)

    @classmethod
    def load(
        cls,
        *,
        ini_path: Path | None = None,
        toml_path: Path | None = None,
        cli_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Settings:
        """Build a :class:`Settings`, merging sources by priority cli > toml > ini > defaults."""
        merged: dict[str, dict[str, Any]] = {key: {} for key in _SECTIONS}
        if ini_path is not None:
            _deep_merge(merged, _read_ini(Path(ini_path)))
        if toml_path is not None:
            _deep_merge(merged, _read_toml(Path(toml_path)))
        if cli_overrides:
            _deep_merge(merged, cli_overrides)

        kwargs: dict[str, Any] = {}
        for canonical, (_ini_name, model_cls) in _SECTIONS.items():
            kwargs[canonical] = model_cls.model_validate(_coerce(model_cls, merged[canonical]))
        return cls(**kwargs)

    def remote_script_path(self, name: str) -> Path:
        """Return the absolute path to ``remote/<name>.py`` on shared storage.

        Compute-node tasks invoke remote scripts as
        ``python3 <abs path>`` rather than ``python3 -m`` so they do not
        depend on the orchestrator's package layout being importable
        inside the chemprop / fpsim2 conda environments.

        When ``paths.spacehasten_src_dir`` is not explicitly configured,
        the directory is auto-detected from the installed package location
        (i.e. the parent of ``spacehasten/__init__.py``).
        """
        src_dir = self.paths.spacehasten_src_dir
        if not src_dir:
            # Auto-detect from package location.
            import spacehasten as _pkg

            src_dir = str(Path(_pkg.__file__).resolve().parent)
        return Path(src_dir) / "remote" / f"{name}.py"

    def compute_shared_root(self, name: str) -> Path:
        """Derive the shared NFS root for a project.

        Returns ``<shared_data_root>/$USER/SPACEHASTEN/<name>/``.
        """
        user = os.environ.get("USER", "user")
        return Path(self.paths.shared_data_root) / user / "SPACEHASTEN" / name

    def dump_toml(self, path: Path) -> None:
        payload: dict[str, Any] = {}
        for section in _SECTIONS:
            section_data = getattr(self, section).model_dump()
            # tomli_w cannot serialize None; preserve unset fields by omission.
            payload[section] = {k: v for k, v in section_data.items() if v is not None}
        with Path(path).open("wb") as fh:
            tomli_w.dump(payload, fh)

    # --------------------------------------------------------------------- #
    # Opt-in install validation                                             #
    # --------------------------------------------------------------------- #

    def validate_install(self) -> list[str]:
        """Check external tools/paths required by the legacy code are present.

        Returns a list of human-readable error messages (empty if everything is OK).
        Never raises — callers decide what to do with the errors.
        """
        errors: list[str] = []
        if shutil.which("chemprop") is None:
            errors.append("chemprop 2.x not on PATH")
        if shutil.which("bzcat") is None:
            errors.append("bzcat not on PATH (install bzip2)")
        for label, p in (
            ("spacelight", self.paths.exe_spacelight_default),
            ("ftrees", self.paths.exe_ftrees_default),
        ):
            if p and not Path(p).exists():
                errors.append(f"{label} executable not found: {p}")
        if self.general.scheduler not in {"slurm", "SGE"}:
            errors.append(
                f"unknown scheduler '{self.general.scheduler}' "
                "(expected 'slurm' or 'SGE')"
            )
        return errors
