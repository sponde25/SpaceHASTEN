"""Tests for installer-generated configuration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from spacehasten.config.settings import Settings

INSTALLER = Path(__file__).resolve().parents[2] / "install_spacehasten.py"
spec = importlib.util.spec_from_file_location("install_spacehasten", INSTALLER)
assert spec is not None and spec.loader is not None
install_spacehasten = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install_spacehasten)


def test_installer_writes_training_and_prediction_defaults(tmp_path: Path) -> None:
    ini_path = tmp_path / "spacehasten.ini"
    answers = dict(install_spacehasten.DEFAULTS)

    install_spacehasten._write_ini(ini_path, answers=answers)
    settings = Settings.load(ini_path=ini_path)

    assert settings.general.cpu_count_train == "8"
    assert settings.general.train_batch_size == 1024
    assert settings.general.train_num_workers == 8
    assert settings.general.train_early_stopping_patience == 5
    assert settings.general.cpu_count_predict == "1"
    assert settings.general.cpu_count_control == "1"
    assert settings.general.pred_batch_size == 32
    assert settings.general.pred_num_workers == 0
    assert settings.general.pred_accelerator == "cpu"
    assert settings.general.pred_chunk_size == 12345
