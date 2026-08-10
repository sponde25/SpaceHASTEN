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
        ["library-build"],
        ["library-screen"],
        ["screening-cycle"],
        ["export"],
        ["export", "csv"],
        ["export", "poses"],
        ["export", "seeds"],
        ["plot"],
        ["archive"],
        ["archive", "create"],
        ["archive", "extract"],
        ["archive", "restore"],
        ["archive", "clean"],
        ["status"],
        ["resume"],
        ["undo"],
        ["undo", "search"],
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
        ["export", "seeds"],
        ["archive", "extract"],
        ["archive", "restore"],
        ["undo"],  # missing undo_kind subcommand
        # Bad choices.
        ["search", "--source", "bogus", "--top-n", "1"],
        ["dock", "--top-n", "1", "--strategy", "bogus"],
        ["library-build"],  # missing --source and --output
        # top-n / score-cutoff are mutually exclusive.
        ["library-screen", "--top-n", "10", "--score-cutoff", "-8.0"],
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
        ["-w", str(ws), "seed-training", "--smi", str(smi), "--dock-cpus", "4"],
        ["-w", str(ws), "train"],
        ["-w", str(ws), "predict"],
        ["-w", str(ws), "search", "--source", "docked", "--top-n", "10", "--cpus", "4"],
        ["-w", str(ws), "dock", "--top-n", "10", "--cpus", "4"],
        ["-w", str(ws), "cluster"],
        ["library-build", "--source", str(smi), "--output", str(tmp_path / "libstore")],
        ["-w", str(ws), "library-screen"],
        ["-w", str(ws), "screening-cycle", "--simsearch-top-n", "100", "--simsearch-jobs", "4", "--dock-top-n", "1000", "--dock-cpus", "4"],
        ["-w", str(ws), "export", "csv", "--cutoff", "-7", "--output", "out.csv"],
        ["-w", str(ws), "export", "poses", "--cutoff", "-7", "--output", "out.mae"],
        ["-w", str(ws), "export", "seeds", "--output", "seeds.csv"],
        ["-w", str(ws), "plot"],
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
    shared = tmp_path / "shared"
    dock_in = tmp_path / "dock.in"
    dock_in.write_bytes(b"DOCK_PARAM_CONTENT")
    grid = tmp_path / "grid.zip"
    grid.write_bytes(b"GRID_CONTENT")
    rc = main(["init", str(ws), "--shared-root", str(shared), "--dock-params", str(dock_in), "--dock-grid", str(grid)])
    assert rc == 0
    assert (ws / "manifest.json").exists()
    assert (ws / "logs").is_dir()
    assert (shared / "models").is_dir()
    # .dbsh is created at init with schema and dock blobs
    assert (ws / "ws.dbsh").exists()


def _init_workspace(tmp_path) -> "Path":  # type: ignore[no-untyped-def]
    from pathlib import Path

    ws = tmp_path / "ws"
    shared = tmp_path / "shared"
    dock_in = tmp_path / "dock.in"
    dock_in.write_bytes(b"DOCK_PARAM_CONTENT")
    grid = tmp_path / "grid.zip"
    grid.write_bytes(b"GRID_CONTENT")
    rc = main([
        "init", str(ws), "--shared-root", str(shared),
        "--dock-params", str(dock_in), "--dock-grid", str(grid),
    ])
    assert rc == 0
    return Path(ws)


def test_screening_cycle_clustering_strategy_autoclusters(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``--strategy clustering`` must re-cluster before each of the 3
    search steps and before the dock step, every round (4x per round)."""
    from spacehasten.stages import clustering, docking, simsearch, training

    ws = _init_workspace(tmp_path)

    cluster_calls: list[int] = []
    monkeypatch.setattr(clustering, "cluster", lambda *a, **k: cluster_calls.append(1) or 0)
    monkeypatch.setattr(simsearch, "simsearch", lambda *a, **k: 1)
    monkeypatch.setattr(docking, "dock", lambda *a, **k: 1)
    monkeypatch.setattr(training, "train", lambda *a, **k: 1)

    rc = main([
        "-w", str(ws), "screening-cycle",
        "--simsearch-top-n", "10", "--simsearch-jobs", "1",
        "--dock-top-n", "10", "--dock-cpus", "1",
        "--rounds", "2", "--strategy", "clustering",
    ])
    assert rc == 0
    # 3 search steps + 1 dock step = 4 cluster calls per round.
    assert len(cluster_calls) == 4 * 2


def test_screening_cycle_greedy_strategy_never_clusters(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``--strategy greedy`` must never invoke clustering."""
    from spacehasten.stages import clustering, docking, simsearch, training

    ws = _init_workspace(tmp_path)

    cluster_calls: list[int] = []
    monkeypatch.setattr(clustering, "cluster", lambda *a, **k: cluster_calls.append(1) or 0)
    monkeypatch.setattr(simsearch, "simsearch", lambda *a, **k: 1)
    monkeypatch.setattr(docking, "dock", lambda *a, **k: 1)
    monkeypatch.setattr(training, "train", lambda *a, **k: 1)

    rc = main([
        "-w", str(ws), "screening-cycle",
        "--simsearch-top-n", "10", "--simsearch-jobs", "1",
        "--dock-top-n", "10", "--dock-cpus", "1",
        "--rounds", "2", "--strategy", "greedy",
    ])
    assert rc == 0
    assert cluster_calls == []


def test_library_build_dispatch_passes_args_to_stage(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from spacehasten.stages import library_build as library_build_mod

    ws = _init_workspace(tmp_path)
    src = tmp_path / "diverse.cxsmiles"
    src.write_text("smiles\tid\n")

    class _FakeManifest:
        n_compounds = 5
        n_chunks = 2

    calls: list[tuple[object, dict]] = []

    def fake_library_build(scheduler, settings, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((settings, kwargs))
        return _FakeManifest()

    monkeypatch.setattr(library_build_mod, "library_build", fake_library_build)

    rc = main([
        "-w", str(ws), "library-build",
        "--source", str(src), "--output", str(tmp_path / "libstore"),
        "--chunk-size", "100", "--cores", "3",
    ])
    assert rc == 0
    assert len(calls) == 1
    settings_used, kwargs_used = calls[0]
    assert settings_used.general.cpu_count_library == "3"
    assert kwargs_used["chunk_size"] == 100
    assert kwargs_used["store_dir"] == tmp_path / "libstore"
    assert kwargs_used["source_files"] == [src]


def test_library_build_runs_without_a_workspace(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """library-build must not require -w / cwd to be a spacehasten workspace.

    It builds a reusable library store independent of any run, so it should
    work from a bare directory with no manifest.json or .dbsh file.
    """
    from spacehasten.stages import library_build as library_build_mod

    bare_dir = tmp_path / "not_a_workspace"
    bare_dir.mkdir()
    assert not (bare_dir / "manifest.json").exists()
    assert not list(bare_dir.glob("*.dbsh"))

    src = tmp_path / "diverse.cxsmiles"
    src.write_text("smiles\tid\n")

    class _FakeManifest:
        n_compounds = 5
        n_chunks = 2

    calls: list[tuple[object, dict]] = []

    def fake_library_build(scheduler, settings, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((settings, kwargs))
        return _FakeManifest()

    monkeypatch.setattr(library_build_mod, "library_build", fake_library_build)
    monkeypatch.chdir(bare_dir)

    rc = main([
        "library-build",
        "--source", str(src), "--output", str(bare_dir / "libstore"),
    ])
    assert rc == 0
    assert len(calls) == 1
    # Logs should land under the output store, not require a workspace.
    assert (bare_dir / "libstore" / "logs" / "spacehasten.log").exists()


def test_library_screen_dispatch_resolves_model_and_forwards_args(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    from spacehasten.core.db import Database
    from spacehasten.stages import library_screen as library_screen_mod

    ws = _init_workspace(tmp_path)
    db_path = next(ws.glob("*.dbsh"))
    db = Database(db_path)
    db.store_model_blob(1, b"")
    db.commit()
    db.close()

    calls: list[dict] = []

    def fake_library_screen(db, workdir, scheduler, settings, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return 7

    monkeypatch.setattr(library_screen_mod, "library_screen", fake_library_screen)

    rc = main([
        "-w", str(ws), "library-screen",
        "--library", str(tmp_path / "libstore"), "--top-n", "50",
    ])
    assert rc == 0
    assert len(calls) == 1
    assert calls[0]["top_n"] == 50
    assert calls[0]["model_version"] == 1
    assert calls[0]["library_dir"] == tmp_path / "libstore"


def test_library_screen_dispatch_errors_without_trained_model(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    ws = _init_workspace(tmp_path)
    with pytest.raises(SystemExit, match="no trained model"):
        main(["-w", str(ws), "library-screen", "--library", str(tmp_path / "libstore")])

