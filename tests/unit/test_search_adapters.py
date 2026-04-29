"""Unit tests for the SpaceLight + FTrees command-line adapters."""

from __future__ import annotations

from spacehasten.tools.ftrees import FTreesAdapter
from spacehasten.tools.spacelight import SpacelightAdapter


def test_spacelight_adapter_command_shape() -> None:
    adapter = SpacelightAdapter(exe="/opt/biosolveit/spacelight")
    cmd = adapter.command_for(
        "CCO",
        "/spaces/REAL.space",
        "/out/spacelightresult_1.csv",
        max_results=5000,
        similarity=0.5,
        threads=1,
    )
    assert cmd[0] == "/opt/biosolveit/spacelight"
    assert cmd[1:5] == ["-i", "CCO", "-s", "/spaces/REAL.space"]
    assert cmd[5:7] == ["-o", "/out/spacelightresult_1.csv"]
    assert "--max-nof-results" in cmd
    assert cmd[cmd.index("--max-nof-results") + 1] == "5000"
    assert "--min-similarity-threshold" in cmd
    assert cmd[cmd.index("--min-similarity-threshold") + 1] == "0.5"
    assert "--thread-count" in cmd
    assert cmd[cmd.index("--thread-count") + 1] == "1"


def test_ftrees_adapter_command_shape() -> None:
    adapter = FTreesAdapter(exe="/opt/biosolveit/ftrees")
    cmd = adapter.command_for(
        "c1ccccc1",
        "/spaces/REAL.space",
        "/out/ftreesresult_2.csv",
        max_results=10000,
        similarity=0.9,
        threads=2,
    )
    assert cmd == [
        "/opt/biosolveit/ftrees",
        "-i", "c1ccccc1",
        "-s", "/spaces/REAL.space",
        "-o", "/out/ftreesresult_2.csv",
        "--max-nof-results", "10000",
        "--min-similarity-threshold", "0.9",
        "--thread-count", "2",
    ]


def test_adapters_handle_pathlib_inputs(tmp_path: object) -> None:
    from pathlib import Path

    sl = SpacelightAdapter(exe="sl")
    ft = FTreesAdapter(exe="ft")
    space = Path("/x/y.space")
    out = Path("/x/o.csv")
    a = sl.command_for("CCO", space, out, max_results=1, similarity=0.1, threads=1)
    b = ft.command_for("CCO", space, out, max_results=1, similarity=0.1, threads=1)
    assert a[4] == "/x/y.space"
    assert a[6] == "/x/o.csv"
    assert b[4] == "/x/y.space"
    assert b[6] == "/x/o.csv"
