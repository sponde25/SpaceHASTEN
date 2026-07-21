"""Tests for ``spacehasten.config``."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomli_w

from spacehasten.config.properties import (
    FloatRange,
    IntRange,
    PropertyRanges,
)
from spacehasten.config.settings import Settings

FIXTURE_INI = Path(__file__).resolve().parents[1] / "fixtures" / "spacehasten.ini"


# --------------------------------------------------------------------------- #
# PropertyRanges                                                              #
# --------------------------------------------------------------------------- #


def test_property_ranges_defaults() -> None:
    p = PropertyRanges()
    assert p.mw == FloatRange(min=0.0, max=500.0)
    assert p.slogp == FloatRange(min=-10.0, max=5.0)
    assert p.hba == IntRange(min=0, max=10)
    assert p.hbd == IntRange(min=0, max=5)
    assert p.rotbonds == IntRange(min=0, max=10)
    assert p.tpsa == FloatRange(min=0.0, max=140.0)


def test_property_ranges_invalid_order() -> None:
    with pytest.raises(ValueError):
        FloatRange(min=10.0, max=0.0)
    with pytest.raises(ValueError):
        IntRange(min=5, max=2)


def test_property_ranges_toml_roundtrip(tmp_path: Path) -> None:
    p = PropertyRanges(
        mw=FloatRange(min=100.0, max=400.0),
        hba=IntRange(min=1, max=8),
    )
    target = tmp_path / "props.toml"
    p.to_toml(target)
    loaded = PropertyRanges.from_toml(target)
    assert loaded == p


# --------------------------------------------------------------------------- #
# Settings                                                                    #
# --------------------------------------------------------------------------- #


def test_settings_defaults_no_filesystem_access() -> None:
    """Constructing must not invoke any path/which validation."""
    s = Settings()
    assert s.general.scheduler == "slurm"
    assert s.general.cpu_count_train == "8"
    assert s.general.train_batch_size == 1024
    assert s.general.train_epochs == 30
    assert s.general.train_num_workers == 8
    assert s.general.train_early_stopping_patience == 8
    assert s.general.train_validation_fraction == pytest.approx(0.1)
    assert s.general.train_gradient_clip_val == pytest.approx(5.0)
    assert s.general.train_precision == "32-true"
    assert s.general.train_svdkl_gp_dim == 16
    assert s.general.train_svdkl_grid_size == 64
    assert s.general.train_svdkl_cholesky_jitter == pytest.approx(1e-3)
    assert s.general.train_svdkl_feature_transform == "tanh"
    assert s.general.train_svdkl_tanh_temperature == pytest.approx(3.0)
    assert s.general.train_seed == 42
    assert s.general.pred_accelerator == "cpu"
    assert s.general.pred_batch_size == 32
    assert s.general.pred_num_workers == 0
    assert s.general.pred_chunk_size == 12345
    assert s.general.cpu_count_predict == "1"
    assert s.general.cpu_count_control == "1"
    assert s.paths.scratch_default == "/wrk"


def test_settings_load_ini_fixture() -> None:
    s = Settings.load(ini_path=FIXTURE_INI)
    assert s.general.scheduler == "slurm"
    assert s.general.prepare_anaconda == "source /data/programs/oce/actoce"
    assert s.general.activate_chemprop == "conda activate chemprop-2.1.2"
    assert s.general.cpu_count_search == "4"
    assert s.general.train_batch_size == 128  # cast to int
    assert s.general.train_epochs == 25
    assert s.general.train_final_lr == pytest.approx(5e-5)
    assert s.general.pred_accelerator == "gpu"
    assert s.paths.exe_spacelight_default.endswith("/spacelight")
    assert s.slurm.slurm_partition == "jobs"
    assert s.slurm.slurm_gpu_parameter == "--gres=gpu:1"
    assert s.sge.sge_queue == "all.q"


def test_settings_toml_overrides_ini(tmp_path: Path) -> None:
    toml = tmp_path / "override.toml"
    payload = {"general": {"train_batch_size": 999, "scheduler": "SGE"}}
    with toml.open("wb") as fh:
        tomli_w.dump(payload, fh)

    s = Settings.load(ini_path=FIXTURE_INI, toml_path=toml)
    assert s.general.train_batch_size == 999
    assert s.general.scheduler == "SGE"
    # un-overridden ini values remain
    assert s.general.train_epochs == 25


def test_settings_cli_overrides_toml(tmp_path: Path) -> None:
    toml = tmp_path / "override.toml"
    payload = {"general": {"train_batch_size": 999}}
    with toml.open("wb") as fh:
        tomli_w.dump(payload, fh)

    s = Settings.load(
        ini_path=FIXTURE_INI,
        toml_path=toml,
        cli_overrides={"general": {"train_batch_size": 7}},
    )
    assert s.general.train_batch_size == 7


def test_settings_dump_toml_roundtrip(tmp_path: Path) -> None:
    s1 = Settings.load(ini_path=FIXTURE_INI)
    out = tmp_path / "settings.toml"
    s1.dump_toml(out)
    s2 = Settings.load(toml_path=out)
    assert s2 == s1


def test_validate_install_returns_errors_not_raises() -> None:
    """Running on dev box without BioSolveIT tools must not raise."""
    s = Settings()
    errors = s.validate_install()
    assert isinstance(errors, list)
    # On a dev box without chemprop / spacelight, errors will be present;
    # we only assert the call succeeds.


def test_remote_script_path_autodetects_when_unset() -> None:
    """``remote_script_path`` auto-detects from package location when src_dir is unset."""
    s = Settings()
    assert s.paths.spacehasten_src_dir is None
    result = s.remote_script_path("train")
    assert result.name == "train.py"
    assert result.parent.name == "remote"


def test_remote_script_path_returns_expected_path(tmp_path: Path) -> None:
    from spacehasten.config.settings import PathsSettings

    s = Settings(paths=PathsSettings(spacehasten_src_dir=str(tmp_path)))
    assert s.remote_script_path("train") == tmp_path / "remote" / "train.py"
    assert s.remote_script_path("predict") == tmp_path / "remote" / "predict.py"
