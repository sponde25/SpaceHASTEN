"""Argparse-only tests for the SpaceHASTEN CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from spacehasten.cli.main import _build_parser, _cluster_alpha_schedule, main


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
        ["atlas"],
        ["atlas", "init"],
        ["atlas", "update"],
        ["atlas", "status"],
        ["screening-cycle"],
        ["export"],
        ["export", "csv"],
        ["export", "poses"],
        ["export", "seeds"],
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
        ["-w", str(ws), "atlas", "status"],
        [
            "-w",
            str(ws),
            "screening-cycle",
            "--simsearch-top-n",
            "100",
            "--simsearch-jobs",
            "4",
            "--dock-top-n",
            "1000",
            "--dock-cpus",
            "4",
        ],
        ["-w", str(ws), "export", "csv", "--cutoff", "-7", "--output", "out.csv"],
        ["-w", str(ws), "export", "poses", "--cutoff", "-7", "--output", "out.mae"],
        ["-w", str(ws), "export", "seeds", "--output", "seeds.csv"],
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


def test_dock_parses_uncertainty_acquisition_options() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "dock",
            "--top-n",
            "100",
            "--cpus",
            "4",
            "--strategy",
            "ei",
            "--ei-hit-threshold",
            "-9.7",
            "--ei-xi",
            "0.1",
            "--cluster-lambda",
            "0.5",
            "--atlas-id",
            "custom-atlas",
        ]
    )
    assert args.strategy == "ei"
    assert args.ei_hit_threshold == -9.7
    assert args.ei_xi == 0.1
    assert args.cluster_lambda == 0.5
    assert args.cluster_alpha is None
    assert args.atlas_id == "custom-atlas"


def test_screening_cycle_parses_cluster_alpha_schedule() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "screening-cycle",
            "--simsearch-top-n",
            "10",
            "--simsearch-jobs",
            "1",
            "--dock-top-n",
            "10",
            "--dock-cpus",
            "1",
            "--rounds",
            "2",
            "--dock-acquisition",
            "ei",
            "--ei-hit-threshold",
            "-9.7",
            "--cluster-alpha",
            "0.2",
            "0.1",
        ]
    )

    assert args.cluster_alpha == [0.2, 0.1]


def test_cluster_alpha_schedule_broadcasts_and_validates_length() -> None:
    assert _cluster_alpha_schedule([0.2], 2) == [0.2, 0.2]
    assert _cluster_alpha_schedule([0.2, 0.1], 2) == [0.2, 0.1]
    with pytest.raises(SystemExit, match="2 expected, got 3"):
        _cluster_alpha_schedule([0.2, 0.1, 0.05], 2)
    with pytest.raises(SystemExit, match="must be at least 1"):
        _cluster_alpha_schedule(None, 0)


def test_dock_rejects_cluster_alpha_with_fixed_lambda(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ws = _init_workspace(tmp_path)
    with pytest.raises(SystemExit, match="cannot be used together"):
        main(
            [
                "-w",
                str(ws),
                "dock",
                "--top-n",
                "10",
                "--cpus",
                "1",
                "--strategy",
                "ei",
                "--ei-hit-threshold",
                "-9.7",
                "--cluster-lambda",
                "0.5",
                "--cluster-alpha",
                "0.2",
            ]
        )


def test_search_rejects_uncertainty_acquisition() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "search",
                "--source",
                "predicted",
                "--top-n",
                "10",
                "--cpus",
                "2",
                "--strategy",
                "lcb",
            ]
        )


def test_init_creates_workspace(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ws = tmp_path / "ws"
    shared = tmp_path / "shared"
    dock_in = tmp_path / "dock.in"
    dock_in.write_bytes(b"DOCK_PARAM_CONTENT")
    grid = tmp_path / "grid.zip"
    grid.write_bytes(b"GRID_CONTENT")
    rc = main(
        [
            "init",
            str(ws),
            "--shared-root",
            str(shared),
            "--dock-params",
            str(dock_in),
            "--dock-grid",
            str(grid),
        ]
    )
    assert rc == 0
    assert (ws / "manifest.json").exists()
    assert (ws / "logs").is_dir()
    assert (shared / "models").is_dir()
    # .dbsh is created at init with schema and dock blobs
    assert (ws / "ws.dbsh").exists()


def _init_workspace(tmp_path) -> Path:  # type: ignore[no-untyped-def]
    ws = tmp_path / "ws"
    shared = tmp_path / "shared"
    dock_in = tmp_path / "dock.in"
    dock_in.write_bytes(b"DOCK_PARAM_CONTENT")
    grid = tmp_path / "grid.zip"
    grid.write_bytes(b"GRID_CONTENT")
    rc = main(
        [
            "init",
            str(ws),
            "--shared-root",
            str(shared),
            "--dock-params",
            str(dock_in),
            "--dock-grid",
            str(grid),
        ]
    )
    assert rc == 0
    return Path(ws)


def test_standalone_clustered_lcb_requires_current_atlas(tmp_path) -> None:  # type: ignore[no-untyped-def]
    ws = _init_workspace(tmp_path)
    with pytest.raises(SystemExit, match="atlas 'morgan-r2-1024-t040' is not initialized"):
        main(
            [
                "-w",
                str(ws),
                "dock",
                "--top-n",
                "1",
                "--cpus",
                "1",
                "--strategy",
                "lcb",
                "--cluster-lambda",
                "0.5",
            ]
        )


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

    rc = main(
        [
            "-w",
            str(ws),
            "screening-cycle",
            "--simsearch-top-n",
            "10",
            "--simsearch-jobs",
            "1",
            "--dock-top-n",
            "10",
            "--dock-cpus",
            "1",
            "--rounds",
            "2",
            "--strategy",
            "clustering",
        ]
    )
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

    rc = main(
        [
            "-w",
            str(ws),
            "screening-cycle",
            "--simsearch-top-n",
            "10",
            "--simsearch-jobs",
            "1",
            "--dock-top-n",
            "10",
            "--dock-cpus",
            "1",
            "--rounds",
            "2",
            "--strategy",
            "greedy",
        ]
    )
    assert rc == 0
    assert cluster_calls == []


def test_screening_cycle_lcb_only_changes_docking_acquisition(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from spacehasten.stages import atlas, clustering, docking, simsearch, training

    ws = _init_workspace(tmp_path)
    atlas_root = tmp_path / "seed-atlas"
    atlas_root.mkdir()
    cluster_calls: list[dict[str, object]] = []
    atlas_import_calls: list[dict[str, object]] = []
    atlas_update_calls: list[dict[str, object]] = []
    search_calls: list[dict[str, object]] = []
    dock_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        clustering,
        "cluster",
        lambda *a, **kwargs: cluster_calls.append(kwargs) or 0,
    )
    monkeypatch.setattr(
        simsearch,
        "simsearch",
        lambda *a, **kwargs: search_calls.append(kwargs) or 1,
    )
    monkeypatch.setattr(
        docking,
        "dock",
        lambda *a, **kwargs: dock_calls.append(kwargs) or 1,
    )
    monkeypatch.setattr(
        atlas,
        "import_initial_seed_atlas",
        lambda *a, **kwargs: atlas_import_calls.append(kwargs),
    )
    monkeypatch.setattr(
        atlas,
        "update_cluster_atlas",
        lambda *a, **kwargs: atlas_update_calls.append(kwargs),
    )
    monkeypatch.setattr(training, "train", lambda *a, **k: 1)

    rc = main(
        [
            "-w",
            str(ws),
            "screening-cycle",
            "--simsearch-top-n",
            "10",
            "--simsearch-jobs",
            "1",
            "--dock-top-n",
            "10",
            "--dock-cpus",
            "1",
            "--dock-acquisition",
            "lcb",
            "--lcb-beta",
            "2.0",
            "--cluster-lambda",
            "0.5",
            "--atlas-id",
            "test-atlas",
            "--atlas-root",
            str(atlas_root),
        ]
    )

    assert rc == 0
    assert [call["strategy"] for call in search_calls] == [
        "greedy",
        "greedy",
        "greedy",
    ]
    assert cluster_calls == []
    assert len(atlas_import_calls) == 1
    assert atlas_import_calls[0]["atlas_id"] == "test-atlas"
    assert atlas_import_calls[0]["atlas_root"] == atlas_root
    assert len(atlas_update_calls) == 1
    assert atlas_update_calls[0]["atlas_id"] == "test-atlas"
    assert dock_calls[0]["strategy"] == "lcb"
    assert dock_calls[0]["lcb_beta"] == 2.0
    assert dock_calls[0]["cluster_lambda"] == 0.5
    assert dock_calls[0]["cluster_alpha"] is None
    assert dock_calls[0]["atlas_id"] == "test-atlas"


def test_screening_cycle_applies_cluster_alpha_schedule(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from spacehasten.stages import atlas, docking, simsearch

    ws = _init_workspace(tmp_path)
    atlas_root = tmp_path / "seed-atlas"
    atlas_root.mkdir()
    atlas_import_calls: list[dict[str, object]] = []
    atlas_update_calls: list[dict[str, object]] = []
    dock_calls: list[dict[str, object]] = []
    monkeypatch.setattr(simsearch, "simsearch", lambda *a, **k: 1)
    monkeypatch.setattr(
        docking,
        "dock",
        lambda *a, **kwargs: dock_calls.append(kwargs) or 1,
    )
    monkeypatch.setattr(
        atlas,
        "import_initial_seed_atlas",
        lambda *a, **kwargs: atlas_import_calls.append(kwargs),
    )
    monkeypatch.setattr(
        atlas,
        "update_cluster_atlas",
        lambda *a, **kwargs: atlas_update_calls.append(kwargs),
    )

    rc = main(
        [
            "-w",
            str(ws),
            "screening-cycle",
            "--simsearch-top-n",
            "10",
            "--simsearch-jobs",
            "1",
            "--dock-top-n",
            "10",
            "--dock-cpus",
            "1",
            "--rounds",
            "2",
            "--dock-acquisition",
            "ei",
            "--ei-hit-threshold",
            "-9.7",
            "--cluster-alpha",
            "0.2",
            "0.1",
            "--atlas-id",
            "test-atlas",
            "--atlas-root",
            str(atlas_root),
        ]
    )

    assert rc == 0
    assert len(atlas_import_calls) == 1
    assert len(atlas_update_calls) == 2
    assert [call["cluster_alpha"] for call in dock_calls] == [0.2, 0.1]
    assert {call["cluster_lambda"] for call in dock_calls} == {0.0}


def test_screening_cycle_cluster_penalty_requires_seed_atlas_root(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from spacehasten.stages import docking, simsearch

    ws = _init_workspace(tmp_path)
    monkeypatch.setattr(
        simsearch,
        "simsearch",
        lambda *a, **k: pytest.fail("search should not run before atlas validation"),
    )
    monkeypatch.setattr(
        docking,
        "dock",
        lambda *a, **k: pytest.fail("dock should not run before atlas validation"),
    )

    with pytest.raises(SystemExit, match="pass --atlas-root PATH"):
        main(
            [
                "-w",
                str(ws),
                "screening-cycle",
                "--simsearch-top-n",
                "10",
                "--simsearch-jobs",
                "1",
                "--dock-top-n",
                "10",
                "--dock-cpus",
                "1",
                "--dock-acquisition",
                "lcb",
                "--cluster-lambda",
                "0.5",
            ]
        )
