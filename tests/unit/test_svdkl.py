"""Tests for the SVDKL GP head and actual Chemprop integration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

HAS_GPYTORCH = importlib.util.find_spec("gpytorch") is not None
HAS_CHEMPROP = importlib.util.find_spec("chemprop") is not None
HAS_PANDAS = importlib.util.find_spec("pandas") is not None

requires_gpytorch = pytest.mark.skipif(
    not HAS_GPYTORCH,
    reason="gpytorch is not installed in this environment",
)
requires_chemprop_svdkl = pytest.mark.skipif(
    not (HAS_GPYTORCH and HAS_CHEMPROP and HAS_PANDAS),
    reason="chemprop, gpytorch, and pandas are required for Chemprop SVDKL tests",
)

EXAMPLE_CSV = Path(__file__).resolve().parents[2] / "example.csv"


def test_import_without_instantiating_optional_dependencies() -> None:
    from spacehasten.remote import svdkl

    assert svdkl.SVDKLConfig(input_dim=3, gp_dim=2).grid_size == 128


@requires_gpytorch
def test_svdkl_head_trains_one_step_and_predicts() -> None:
    import gpytorch
    import torch

    from spacehasten.remote.svdkl import SVDKLConfig, SVDKLHead, predictive_mean_stds

    torch.manual_seed(0)
    embeddings = torch.randn(12, 5)
    target = torch.sin(embeddings[:, 0])
    head = SVDKLHead(SVDKLConfig(input_dim=5, gp_dim=3, grid_size=16))
    mll = gpytorch.mlls.PredictiveLogLikelihood(
        likelihood=head.likelihood,
        model=head.gp_layer,
        num_data=embeddings.size(0),
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)

    head.train()
    head.likelihood.train()
    optimizer.zero_grad()
    loss = -mll(head(embeddings), target)
    loss.backward()
    optimizer.step()

    mean, epistemic_std, aleatoric_std, total_std = predictive_mean_stds(
        head,
        embeddings[:4],
    )
    assert mean.shape == torch.Size([4])
    assert epistemic_std.shape == torch.Size([4])
    assert torch.isfinite(mean).all()
    assert torch.isfinite(epistemic_std).all()
    assert torch.isfinite(aleatoric_std).all()
    assert torch.isfinite(total_std).all()
    assert torch.all(epistemic_std >= 0)
    assert torch.allclose(
        total_std.square(),
        epistemic_std.square() + aleatoric_std.square(),
    )


@requires_gpytorch
def test_tanh_feature_transform_is_bounded_and_differentiable() -> None:
    import torch

    from spacehasten.remote.svdkl import SVDKLConfig, SVDKLHead

    head = SVDKLHead(
        SVDKLConfig(
            input_dim=3,
            gp_dim=2,
            grid_lower=-10.0,
            grid_upper=10.0,
            feature_transform="tanh",
            tanh_temperature=3.0,
            cholesky_jitter=1e-3,
        )
    )
    raw_features = torch.tensor([[-100.0, -3.0], [3.0, 100.0]], requires_grad=True)
    transformed = head.transform_features(raw_features)

    assert torch.all(transformed >= -9.5)
    assert torch.all(transformed <= 9.5)
    transformed.sum().backward()
    assert raw_features.grad is not None
    assert torch.isfinite(raw_features.grad).all()


@requires_chemprop_svdkl
def test_deterministic_random_split() -> None:
    import numpy as np

    from spacehasten.remote.train import deterministic_split_indices

    first_train, first_val = deterministic_split_indices(100, validation_fraction=0.1, seed=42)
    second_train, second_val = deterministic_split_indices(100, validation_fraction=0.1, seed=42)
    other_train, other_val = deterministic_split_indices(100, validation_fraction=0.1, seed=43)

    assert np.array_equal(first_train, second_train)
    assert np.array_equal(first_val, second_val)
    assert len(first_train) == 90
    assert len(first_val) == 10
    assert not np.array_equal(first_val, other_val)
    assert not np.array_equal(first_train, other_train)


@requires_chemprop_svdkl
def test_actual_chemprop_wrapper_uses_encoding() -> None:
    import torch

    model = _build_chemprop_svdkl_model(seed=0)
    loader, _ = _build_training_loader(_example_rows(4), batch_size=2)
    batch = next(iter(loader))

    with torch.no_grad():
        embeddings = model.encode_batch(batch)
        out = model(batch)

    assert embeddings.shape == torch.Size([2, 16])
    assert out.event_shape.numel() > 0


@requires_chemprop_svdkl
def test_actual_chemprop_wrapper_uses_ffn_embedding_width() -> None:
    import torch

    model = _build_chemprop_svdkl_model(seed=0, ffn_hidden_size=12)
    loader, _ = _build_training_loader(_example_rows(4), batch_size=2)
    batch = next(iter(loader))

    with torch.no_grad():
        embeddings = model.encode_batch(batch)
        out = model(batch)

    assert embeddings.shape == torch.Size([2, 12])
    assert model.head.projection.in_features == 12
    assert out.event_shape.numel() > 0


@requires_chemprop_svdkl
def test_actual_chemprop_svdkl_training_updates_all_relevant_parameter_groups() -> None:
    model = _build_chemprop_svdkl_model(seed=0)
    loader, _ = _build_training_loader(_example_rows(24), batch_size=6)

    before_message_passing = _clone_trainable_parameters(model.mpnn.message_passing)
    before_projection = _clone_trainable_parameters(model.head.projection)
    before_gp = _clone_trainable_parameters(model.gp_layer)
    before_mixing_weights = model.likelihood.mixing_weights.detach().clone()
    before_base_likelihood = _clone_trainable_parameters(model.likelihood.base_likelihood)

    _train_model(model, loader, num_data=24, epochs=2, seed=0)

    assert _any_parameter_changed(model.mpnn.message_passing, before_message_passing)
    assert _any_parameter_changed(model.head.projection, before_projection)
    assert _any_parameter_changed(model.gp_layer, before_gp)
    assert _any_parameter_changed(model.likelihood.base_likelihood, before_base_likelihood)
    assert not model.likelihood.mixing_weights.detach().allclose(before_mixing_weights)


@requires_gpytorch
def test_target_scaler_roundtrip() -> None:
    import torch

    from spacehasten.remote.svdkl import (
        fit_target_scaler,
        scale_targets,
        unscale_mean,
    )

    y = torch.tensor([-8.0, -7.0, -6.0])
    scaler = fit_target_scaler(y)
    restored = unscale_mean(scale_targets(y, scaler), scaler)
    assert torch.allclose(restored, y)


@requires_chemprop_svdkl
def test_actual_chemprop_svdkl_predictions_use_original_score_units() -> None:
    import torch

    from spacehasten.remote.svdkl import predictive_mean_std

    model, scaler = _train_actual_chemprop_svdkl(seed=0, rows=24, epochs=1)
    embeddings = _encode_dataframe(model, _example_rows(6, offset=24), batch_size=3)

    scaled_mean, scaled_std = predictive_mean_std(model.head, embeddings)
    unscaled_mean, unchanged_std = predictive_mean_std(
        model.head,
        embeddings,
        target_scaler=scaler,
    )
    expected_unscaled = torch.as_tensor(
        scaler.inverse_transform(scaled_mean.detach().numpy().reshape(-1, 1)).reshape(-1),
        dtype=unscaled_mean.dtype,
    )

    assert torch.allclose(unscaled_mean, expected_unscaled)
    assert torch.allclose(unchanged_std, scaled_std * float(scaler.scale_[0]))
    assert not torch.allclose(unscaled_mean, scaled_mean)


@requires_chemprop_svdkl
def test_actual_chemprop_svdkl_training_reproducible_with_seed() -> None:
    import torch

    first_model, first_scaler = _train_actual_chemprop_svdkl(seed=0, rows=24, epochs=1)
    second_model, second_scaler = _train_actual_chemprop_svdkl(seed=0, rows=24, epochs=1)
    pred_rows = _example_rows(8, offset=24)

    first_mean, first_std = _predict_dataframe(first_model, pred_rows, first_scaler, batch_size=4)
    second_mean, second_std = _predict_dataframe(
        second_model,
        pred_rows,
        second_scaler,
        batch_size=4,
    )

    assert torch.allclose(first_mean, second_mean, atol=1e-6)
    assert torch.allclose(first_std, second_std, atol=1e-6)


@requires_chemprop_svdkl
def test_actual_chemprop_svdkl_checkpoint_roundtrip_after_training(tmp_path: Path) -> None:
    import torch

    from spacehasten.remote.svdkl import (
        ChempropConfig,
        load_chemprop_svdkl_checkpoint,
        save_chemprop_svdkl_checkpoint,
    )

    model, scaler = _train_actual_chemprop_svdkl(seed=0, rows=24, epochs=1)
    pred_rows = _example_rows(8, offset=24)
    before_mean, before_std = _predict_dataframe(model, pred_rows, scaler, batch_size=4)

    path = tmp_path / "model_0" / "pytorch_model.bin"
    config = ChempropConfig(
        mp_hidden_size=16,
        mp_depth=2,
        ffn_hidden_size=16,
        ffn_layers=1,
        dropout=0.0,
        activation="relu",
        batch_norm=False,
        warmup_epochs=1,
        init_lr=1e-4,
        max_lr=1e-3,
        final_lr=1e-4,
    )
    save_chemprop_svdkl_checkpoint(
        path,
        model=model,
        target_scaler=scaler,
        chemprop_config=config,
        metadata={"source": "unit-test"},
    )
    loaded_model, loaded_scaler, metadata = load_chemprop_svdkl_checkpoint(path)
    after_mean, after_std = _predict_dataframe(loaded_model, pred_rows, loaded_scaler, batch_size=4)

    assert loaded_scaler.mean_.tolist() == scaler.mean_.tolist()
    assert loaded_scaler.scale_.tolist() == scaler.scale_.tolist()
    assert metadata == {"source": "unit-test"}
    assert torch.allclose(after_mean, before_mean)
    assert torch.allclose(after_std, before_std)


@requires_chemprop_svdkl
def test_actual_chemprop_svdkl_train_predict_roundtrip_output_schema() -> None:
    import numpy as np
    import pandas as pd

    model, scaler = _train_actual_chemprop_svdkl(seed=0, rows=24, epochs=1)
    pred_rows = _example_rows(10, offset=32)
    mean, epistemic_std, aleatoric_std, total_std = _predict_dataframe_stds(
        model,
        pred_rows,
        scaler,
        batch_size=5,
    )
    output_df = pd.DataFrame(
        {
            "smilesid": pred_rows["smilesid"].tolist(),
            "docking_score": mean.detach().numpy(),
            "docking_score_epistemic_std": epistemic_std.detach().numpy(),
            "docking_score_aleatoric_std": aleatoric_std.detach().numpy(),
            "docking_score_std": total_std.detach().numpy(),
        }
    )

    assert list(output_df.columns) == [
        "smilesid",
        "docking_score",
        "docking_score_epistemic_std",
        "docking_score_aleatoric_std",
        "docking_score_std",
    ]
    assert len(output_df) == len(pred_rows)
    assert np.isfinite(output_df["docking_score"]).all()
    assert np.isfinite(output_df["docking_score_std"]).all()
    assert (output_df["docking_score_std"] >= 0).all()


@requires_chemprop_svdkl
def test_remote_train_predict_roundtrip_writes_uncertainty(tmp_path: Path) -> None:
    import numpy as np
    import pandas as pd

    from spacehasten.remote.predict import predict
    from spacehasten.remote.svdkl import load_chemprop_svdkl_checkpoint
    from spacehasten.remote.train import train_model

    train_csv = tmp_path / "train.csv"
    pred_csv = tmp_path / "predict.csv"
    out_csv = tmp_path / "predictions.csv"
    model_dir = tmp_path / "model"
    _example_rows(24).to_csv(train_csv, index=False)
    pred_rows = _example_rows(9, offset=24)
    pred_rows[["smiles", "smilesid"]].to_csv(pred_csv, index=False)

    train_rc = train_model(
        str(train_csv),
        str(model_dir),
        batch_size=6,
        epochs=3,
        num_workers=0,
        devices="1",
        mp_hidden_size=16,
        mp_depth=2,
        ffn_hidden_size=12,
        ffn_layers=1,
        dropout=0.0,
        activation="relu",
        batch_norm=False,
        warmup_epochs=1,
        init_lr=1e-4,
        max_lr=1e-3,
        final_lr=1e-4,
        svdkl_gp_dim=2,
        svdkl_grid_size=16,
        svdkl_cholesky_jitter=1e-3,
        svdkl_feature_transform="tanh",
        svdkl_tanh_temperature=3.0,
        seed=42,
        early_stopping_patience=1,
        early_stopping_min_delta=1e6,
        validation_fraction=0.1,
        gradient_clip_val=5.0,
        precision="32-true",
    )
    assert train_rc == 0
    checkpoint_path = model_dir / "model_0" / "pytorch_model.bin"
    assert checkpoint_path.exists()
    loaded_model, target_scaler, metadata = load_chemprop_svdkl_checkpoint(checkpoint_path)
    from spacehasten.remote.train import deterministic_split_indices

    train_indices, _ = deterministic_split_indices(24, validation_fraction=0.1, seed=42)
    expected_train_mean = _example_rows(24).iloc[train_indices]["docking_score"].mean()
    assert float(target_scaler.mean_[0]) == pytest.approx(expected_train_mean)
    assert loaded_model.head.config.feature_transform == "tanh"
    assert loaded_model.head.config.tanh_temperature == pytest.approx(3.0)
    assert loaded_model.head.config.cholesky_jitter == pytest.approx(1e-3)
    assert metadata["stop_reason"] == "early_stopping"
    assert metadata["best_epoch"] == 1
    assert metadata["epochs_completed"] == 2
    assert (model_dir / "training_metadata.json").exists()

    pred_rc = predict(
        str(pred_csv),
        str(model_dir),
        str(out_csv),
        batch_size=4,
        num_workers=0,
        accelerator="cpu",
        devices="1",
    )
    assert pred_rc == 0

    output_df = pd.read_csv(out_csv)
    assert list(output_df.columns) == [
        "smilesid",
        "docking_score",
        "docking_score_epistemic_std",
        "docking_score_aleatoric_std",
        "docking_score_std",
    ]
    assert len(output_df) == 9
    assert np.isfinite(output_df["docking_score"]).all()
    std_columns = [
        "docking_score_epistemic_std",
        "docking_score_aleatoric_std",
        "docking_score_std",
    ]
    assert np.isfinite(output_df[std_columns]).all().all()
    assert (output_df[std_columns] >= 0).all().all()
    assert np.allclose(
        output_df["docking_score_std"].to_numpy() ** 2,
        output_df["docking_score_epistemic_std"].to_numpy() ** 2
        + output_df["docking_score_aleatoric_std"].to_numpy() ** 2,
    )


@requires_gpytorch
def test_remote_train_selects_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from spacehasten.remote import train as remote_train

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert remote_train._select_device(1) == torch.device("cuda:0")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert remote_train._select_device(1) == torch.device("cpu")
    assert remote_train._select_device(0) == torch.device("cpu")


@requires_gpytorch
def test_remote_predict_selects_cuda_for_auto_or_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from spacehasten.remote import predict as remote_predict

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert remote_predict._select_device("auto", 1) == torch.device("cuda:0")
    assert remote_predict._select_device("gpu", 1) == torch.device("cuda:0")
    assert remote_predict._select_device("cpu", 1) == torch.device("cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert remote_predict._select_device("auto", 1) == torch.device("cpu")


def _example_rows(rows: int, offset: int = 0) -> Any:
    import pandas as pd

    return pd.read_csv(EXAMPLE_CSV).iloc[offset : offset + rows].reset_index(drop=True)


def _build_training_loader(df: Any, batch_size: int) -> tuple[Any, Any]:
    import numpy as np
    from chemprop import data, featurizers

    from spacehasten.remote.svdkl import fit_target_scaler, scale_targets

    y = df["docking_score"].astype(float).to_numpy()
    scaler = fit_target_scaler(y)
    scaled_y = scale_targets(y, scaler)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    datapoints = [
        data.MoleculeDatapoint.from_smi(
            smi=smi,
            y=np.array([target], dtype=float),
        )
        for smi, target in zip(df["smiles"].astype(str), scaled_y, strict=False)
    ]
    dataset = data.MoleculeDataset(datapoints, featurizer)
    loader = data.build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return loader, scaler


def _build_prediction_loader(df: Any, batch_size: int) -> Any:
    from chemprop import data, featurizers

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    datapoints = [data.MoleculeDatapoint.from_smi(smi=smi) for smi in df["smiles"].astype(str)]
    dataset = data.MoleculeDataset(datapoints, featurizer)
    return data.build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def _build_chemprop_svdkl_model(seed: int, ffn_hidden_size: int = 16) -> Any:
    import torch

    from spacehasten.remote.svdkl import (
        ChempropConfig,
        ChempropSVDKLModel,
        SVDKLConfig,
        SVDKLHead,
        build_chemprop_mpnn,
    )

    torch.manual_seed(seed)
    d_h = 16
    config = ChempropConfig(
        mp_hidden_size=d_h,
        mp_depth=2,
        ffn_hidden_size=ffn_hidden_size,
        ffn_layers=1,
        dropout=0.0,
        activation="relu",
        batch_norm=False,
        warmup_epochs=1,
        init_lr=1e-4,
        max_lr=1e-3,
        final_lr=1e-4,
    )
    mpnn = build_chemprop_mpnn(config)
    head = SVDKLHead(SVDKLConfig(input_dim=ffn_hidden_size, gp_dim=2, grid_size=16))
    return ChempropSVDKLModel(mpnn, head, embedding_i=-1)


def _train_actual_chemprop_svdkl(seed: int, rows: int, epochs: int) -> tuple[Any, Any]:
    df = _example_rows(rows)
    model = _build_chemprop_svdkl_model(seed)
    loader, scaler = _build_training_loader(df, batch_size=6)
    _train_model(model, loader, num_data=len(df), epochs=epochs, seed=seed)
    return model, scaler


def _train_model(model: Any, loader: Any, num_data: int, epochs: int, seed: int) -> None:
    import gpytorch
    import torch

    torch.manual_seed(seed)
    mll = gpytorch.mlls.PredictiveLogLikelihood(
        likelihood=model.likelihood,
        model=model.gp_layer,
        num_data=num_data,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    model.likelihood.train()
    for _ in range(epochs):
        for batch in loader:
            target = _batch_targets(batch)
            optimizer.zero_grad()
            loss = -mll(model(batch), target)
            loss.backward()
            optimizer.step()


def _batch_targets(batch: Any) -> Any:
    import torch

    for attr in ("Y", "y", "targets"):
        if hasattr(batch, attr):
            target = getattr(batch, attr)
            if target is not None:
                break
    else:
        raise AttributeError("Could not find targets on Chemprop training batch")

    if not torch.is_tensor(target):
        target = torch.as_tensor(target)
    return target.float().reshape(target.shape[0], -1)[:, 0]


def _encode_dataframe(model: Any, df: Any, batch_size: int) -> Any:
    import torch

    loader = _build_prediction_loader(df, batch_size=batch_size)
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            embeddings.append(model.encode_batch(batch))
    return torch.cat(embeddings, dim=0)


def _predict_dataframe(model: Any, df: Any, scaler: Any, batch_size: int) -> tuple[Any, Any]:
    embeddings = _encode_dataframe(model, df, batch_size=batch_size)
    return _predict_embeddings(model.head, embeddings, scaler)


def _predict_dataframe_stds(
    model: Any,
    df: Any,
    scaler: Any,
    batch_size: int,
) -> tuple[Any, Any, Any, Any]:
    from spacehasten.remote.svdkl import predictive_mean_stds

    embeddings = _encode_dataframe(model, df, batch_size=batch_size)
    return tuple(
        value.detach()
        for value in predictive_mean_stds(
            model.head,
            embeddings,
            target_scaler=scaler,
        )
    )


def _predict_embeddings(head: Any, embeddings: Any, scaler: Any) -> tuple[Any, Any]:
    from spacehasten.remote.svdkl import predictive_mean_std

    mean, std = predictive_mean_std(head, embeddings, target_scaler=scaler)
    return mean.detach(), std.detach()


def _clone_trainable_parameters(module: Any) -> dict[str, Any]:
    return {
        name: param.detach().clone()
        for name, param in module.named_parameters()
        if param.requires_grad
    }


def _any_parameter_changed(module: Any, before: dict[str, Any]) -> bool:
    for name, param in module.named_parameters():
        if name in before and not param.detach().allclose(before[name]):
            return True
    return False
