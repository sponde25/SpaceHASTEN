"""Argparse-only tests for the SpaceHASTEN CLI."""

from __future__ import annotations

import pytest

from spacehasten.cli.main import _build_parser, main


def test_no_command_errors() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_top_level_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize(
    "subcommand",
    [
        ["init"],
        ["import-seeds"],
        ["seed-training"],
        ["train"],
        ["predict"],
        ["search"],
        ["dock"],
        ["cluster"],
        ["screening-cycle"],
        ["export"],
        ["export", "csv"],
        ["export", "poses"],
        ["archive"],
        ["archive", "create"],
        ["archive", "extract"],
        ["archive", "restore"],
        ["archive", "clean"],
        ["status"],
        ["resume"],
        ["verify"],
    ],
)
def test_subcommand_help_exits_zero(subcommand: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main([*subcommand, "--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize(
    "argv",
    [
        # Missing required args.
        ["init", "/tmp/ws"],  # missing --dock-params and --dock-grid
        ["import-seeds"],
        ["seed-training"],  # missing --smi
        ["search"],
        ["dock"],
        ["export", "csv"],
        ["export", "poses"],
        ["archive", "extract"],
        ["archive", "restore"],
        # Bad choices.
        ["search", "--source", "bogus", "--top-n", "1"],
        ["dock", "--top-n", "1", "--strategy", "bogus"],
    ],
)
def test_required_args_validated(argv: list[str]) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_each_subcommand_parses_minimally(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Smoke: every subcommand parses a minimal valid argv without raising."""
    parser = _build_parser()
    ws = tmp_path / "ws"
    ws.mkdir()
    smi = tmp_path / "seeds.smi"
    smi.write_text("CCO seed-1\n")
    dock_in = tmp_path / "dock.in"
    dock_in.write_bytes(b"")
    grid = tmp_path / "grid.zip"
    grid.write_bytes(b"")
    archive_path = tmp_path / "out.tgz"
    archive_path.write_bytes(b"")

    samples: list[list[str]] = [
        ["init", str(ws), "--dock-params", str(dock_in), "--dock-grid", str(grid)],
        ["-w", str(ws), "import-seeds", "--smi", str(smi)],
        ["-w", str(ws), "seed-training", "--smi", str(smi)],
        ["-w", str(ws), "train"],
        ["-w", str(ws), "predict"],
        ["-w", str(ws), "search", "--source", "docked", "--top-n", "10"],
        ["-w", str(ws), "dock", "--top-n", "10"],
        ["-w", str(ws), "cluster"],
        ["-w", str(ws), "screening-cycle"],
        ["-w", str(ws), "export", "csv", "--cutoff", "-7", "--output", "out.csv"],
        ["-w", str(ws), "export", "poses", "--cutoff", "-7", "--output", "out.mae"],
        ["-w", str(ws), "archive", "create"],
        ["archive", "extract", "--archive", str(archive_path), "--target", str(tmp_path / "t1")],
        ["archive", "restore", "--archive", str(archive_path), "--target", str(tmp_path / "t2")],
        ["-w", str(ws), "archive", "clean"],
        ["-w", str(ws), "status"],
        ["-w", str(ws), "resume"],
    ]
    for argv in samples:
        ns = parser.parse_args(argv)
        assert ns.command is not None


def test_init_creates_workspace(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ws = tmp_path / "ws"
    dock_in = tmp_path / "dock.in"
    dock_in.write_bytes(b"DOCK_PARAM_CONTENT")
    grid = tmp_path / "grid.zip"
    grid.write_bytes(b"GRID_CONTENT")
    rc = main(["init", str(ws), "--dock-params", str(dock_in), "--dock-grid", str(grid)])
    assert rc == 0
    assert (ws / "manifest.json").exists()
    assert (ws / "logs").is_dir()
    assert (ws / "models").is_dir()
    # .dbsh is created at init with schema and dock blobs
    assert (ws / "ws.dbsh").exists()
