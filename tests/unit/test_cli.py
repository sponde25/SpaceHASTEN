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
        ["train"],
        ["predict"],
        ["search"],
        ["dock"],
        ["cluster"],
        ["screen"],
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
        ["import-seeds"],
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
    db = tmp_path / "ws" / "ws.dbsh"
    smi = tmp_path / "seeds.smi"
    smi.write_text("CCO seed-1\n")
    dock_in = tmp_path / "dock.in"
    dock_in.write_bytes(b"")
    grid = tmp_path / "grid.zip"
    grid.write_bytes(b"")
    archive_path = tmp_path / "out.tgz"
    archive_path.write_bytes(b"")

    samples: list[list[str]] = [
        ["--db", str(db), "init"],
        ["--db", str(db), "import-seeds",
         "--smi", str(smi), "--dock-params", str(dock_in),
         "--dock-grid", str(grid)],
        ["--db", str(db), "train"],
        ["--db", str(db), "predict"],
        ["--db", str(db), "search", "--source", "docked", "--top-n", "10"],
        ["--db", str(db), "dock", "--top-n", "10"],
        ["--db", str(db), "cluster"],
        ["--db", str(db), "screen"],
        ["--db", str(db), "export", "csv", "--cutoff", "-7", "--output", "out.csv"],
        ["--db", str(db), "export", "poses", "--cutoff", "-7", "--output", "out.mae"],
        ["--db", str(db), "archive", "create"],
        ["archive", "extract", "--archive", str(archive_path), "--target", str(tmp_path / "t1")],
        ["archive", "restore", "--archive", str(archive_path), "--target", str(tmp_path / "t2")],
        ["--db", str(db), "archive", "clean"],
        ["--db", str(db), "status"],
        ["--db", str(db), "resume"],
    ]
    for argv in samples:
        ns = parser.parse_args(argv)
        assert ns.command is not None


def test_init_creates_workspace(tmp_path) -> None:  # type: ignore[no-untyped-def]
    db = tmp_path / "ws" / "ws.dbsh"
    rc = main(["--db", str(db), "init"])
    assert rc == 0
    assert (tmp_path / "ws" / "manifest.json").exists()
    assert (tmp_path / "ws" / "logs").is_dir()
    assert (tmp_path / "ws" / "models").is_dir()
