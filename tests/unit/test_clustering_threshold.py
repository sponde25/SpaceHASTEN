"""Dependency-free checks for configurable sphere-exclusion thresholds."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from spacehasten.remote.cluster import run_clustering
from spacehasten.stages.clustering import _build_cluster_command, _wait_for_output


def test_cluster_command_propagates_similarity_threshold(tmp_path: Path) -> None:
    command = _build_cluster_command(
        tmp_path / "input.smi.gz",
        tmp_path / "clustering.csv",
        4,
        ("python3", "cluster.py"),
        0.4,
    )
    assert "--similarity-threshold 0.4" in command


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.1])
def test_remote_cluster_rejects_invalid_threshold_before_loading_rdkit(
    tmp_path: Path, threshold: float
) -> None:
    with pytest.raises(ValueError, match="similarity_threshold"):
        run_clustering(
            tmp_path / "input.smi",
            tmp_path / "clustering.csv",
            similarity_threshold=threshold,
        )


def test_clustering_waits_for_delayed_output_visibility(tmp_path: Path) -> None:
    output = tmp_path / "clustering.csv"
    timer = threading.Timer(0.05, output.touch)
    timer.start()
    try:
        assert _wait_for_output(output, timeout=1.0, poll_interval=0.01)
    finally:
        timer.join()
