"""SVDKL components used by the remote training and prediction scripts.

The model here is intentionally tensor-level: callers provide molecular
embeddings (Chemprop ``MPNN.encoding(..., i=-1)`` in the SpaceHASTEN training
path), and this module owns the GP head, likelihood, prediction, and checkpoint
round-trip. Keeping it independent of Chemprop lets us test the GP machinery
before wiring it into ``remote.train`` and ``remote.predict``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - exercised when optional deps are installed
    import gpytorch
    import torch
except ImportError:  # pragma: no cover - import-safe optional dependency guard
    gpytorch = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]


def _require_torch_gpytorch() -> None:
    if torch is None or gpytorch is None:
        raise ImportError(
            "spacehasten.remote.svdkl requires torch and gpytorch. "
            "Install them in the chemprop compute-node environment."
        )


@dataclass(frozen=True)
class SVDKLConfig:
    """Hyperparameters for the SVDKL GP head."""

    input_dim: int
    gp_dim: int
    grid_size: int = 128
    grid_lower: float = -10.0
    grid_upper: float = 10.0


@dataclass(frozen=True)
class ChempropConfig:
    """Chemprop MPNN architecture needed to rebuild a saved SVDKL model."""

    mp_hidden_size: int
    mp_depth: int
    ffn_hidden_size: int
    ffn_layers: int
    dropout: float
    activation: str
    batch_norm: bool
    warmup_epochs: int
    init_lr: float
    max_lr: float
    final_lr: float
    embedding_i: int = -1


def fit_target_scaler(y: Any) -> StandardScaler:
    """Fit a scikit-learn ``StandardScaler`` for 1D regression targets."""

    scaler = StandardScaler()
    scaler.fit(_as_numpy_column(y))
    return scaler


def scale_targets(y: Any, scaler: StandardScaler) -> Any:
    """Transform targets before SVDKL training."""

    scaled = scaler.transform(_as_numpy_column(y)).reshape(-1)
    return _like_input(scaled, y)


def unscale_mean(mean: Any, scaler: StandardScaler) -> Any:
    """Inverse-transform posterior means during prediction."""

    restored = scaler.inverse_transform(_as_numpy_column(mean)).reshape(-1)
    return _like_input(restored, mean)


def unscale_std(std: Any, scaler: StandardScaler) -> Any:
    """Convert posterior standard deviations to original target units."""

    scale = float(scaler.scale_[0])
    if torch is not None and isinstance(std, torch.Tensor):
        return std * scale
    return np.asarray(std) * scale


def _as_numpy_column(values: Any) -> np.ndarray:
    if torch is not None and isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    return array.astype(float).reshape(-1, 1)


def _like_input(values: np.ndarray, reference: Any) -> Any:
    if torch is not None and isinstance(reference, torch.Tensor):
        return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
    return values


class GaussianProcessLayer(gpytorch.models.ApproximateGP if gpytorch else object):
    """Independent 1D grid-interpolated GPs over each latent dimension."""

    def __init__(
        self,
        num_dim: int,
        grid_bounds: tuple[float, float] = (-10.0, 10.0),
        grid_size: int = 128,
    ) -> None:
        _require_torch_gpytorch()
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size,
            batch_shape=torch.Size([num_dim]),
        )
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self,
                grid_size=grid_size,
                grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ),
            num_tasks=num_dim,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

    def forward(self, x: Any) -> Any:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class MixingLikelihood(gpytorch.likelihoods.Likelihood if gpytorch else object):
    """Project latent GP samples into one scalar regression output."""

    def __init__(self, num_dim: int) -> None:
        _require_torch_gpytorch()
        super().__init__()
        self.num_dim = num_dim
        self.mixing_weights = torch.nn.Parameter(torch.randn(num_dim) / num_dim)
        self.base_likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, function_samples: Any, **kwargs: Any) -> Any:
        loc = function_samples @ self.mixing_weights
        return self.base_likelihood(loc, **kwargs)


class SVDKLHead(gpytorch.Module if gpytorch else object):
    """SVDKL head for precomputed neural embeddings."""

    def __init__(self, config: SVDKLConfig) -> None:
        _require_torch_gpytorch()
        super().__init__()
        self.config = config
        self.projection = torch.nn.Linear(config.input_dim, config.gp_dim)
        self.gp_layer = GaussianProcessLayer(
            config.gp_dim,
            grid_bounds=(config.grid_lower, config.grid_upper),
            grid_size=config.grid_size,
        )
        self.likelihood = MixingLikelihood(config.gp_dim)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(
            config.grid_lower,
            config.grid_upper,
        )

    def forward(self, embeddings: Any) -> Any:
        features = self.projection(embeddings.float())
        scaled = self.scale_to_bounds(features)
        scaled = scaled.transpose(-1, -2).unsqueeze(-1)
        return self.gp_layer(scaled)


class ChempropSVDKLModel(torch.nn.Module if torch else object):
    """End-to-end wrapper connecting Chemprop encodings to the SVDKL head."""

    def __init__(self, mpnn: Any, head: SVDKLHead, embedding_i: int = -1) -> None:
        _require_torch_gpytorch()
        super().__init__()
        self.mpnn = mpnn
        self.head = head
        self.embedding_i = embedding_i

    @property
    def gp_layer(self) -> GaussianProcessLayer:
        return self.head.gp_layer

    @property
    def likelihood(self) -> MixingLikelihood:
        return self.head.likelihood

    def encode_batch(self, batch: Any) -> Any:
        return self.mpnn.encoding(
            batch.bmg,
            getattr(batch, "V_d", None),
            getattr(batch, "X_d", None),
            i=self.embedding_i,
        )

    def forward(self, batch: Any) -> Any:
        return self.head(self.encode_batch(batch))


def move_batch_to_device(batch: Any, device: Any) -> Any:
    """Move Chemprop batch fields to a torch device."""

    _require_torch_gpytorch()
    if hasattr(batch, "_replace") and hasattr(batch, "_fields"):
        return batch._replace(
            **{
                field: _move_value_to_device(getattr(batch, field), device)
                for field in batch._fields
            }
        )
    return _move_value_to_device(batch, device)


def _move_value_to_device(value: Any, device: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to"):
        moved = value.to(device)
        return value if moved is None else moved
    if isinstance(value, dict):
        return {k: _move_value_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_value_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(v, device) for v in value)
    return value


def build_chemprop_mpnn(config: ChempropConfig) -> Any:
    """Rebuild the Chemprop encoder architecture used by a saved checkpoint."""

    _require_torch_gpytorch()
    try:
        from chemprop import models as chemprop_models
        from chemprop import nn as chemprop_nn
    except ImportError as e:  # pragma: no cover - remote environment guard
        raise ImportError(
            "spacehasten.remote.svdkl requires chemprop to rebuild the encoder"
        ) from e

    mp = chemprop_nn.BondMessagePassing(
        d_h=config.mp_hidden_size,
        depth=config.mp_depth,
        dropout=config.dropout,
        activation=config.activation,
    )
    agg = chemprop_nn.MeanAggregation()
    ffn = chemprop_nn.RegressionFFN(
        input_dim=config.mp_hidden_size,
        hidden_dim=config.ffn_hidden_size,
        n_layers=config.ffn_layers,
        dropout=config.dropout,
        activation=config.activation,
    )
    return chemprop_models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=config.batch_norm,
        warmup_epochs=config.warmup_epochs,
        init_lr=config.init_lr,
        max_lr=config.max_lr,
        final_lr=config.final_lr,
    )


def predictive_mean_std(
    head: SVDKLHead,
    embeddings: Any,
    *,
    target_scaler: StandardScaler | None = None,
) -> tuple[Any, Any]:
    """Return posterior mean and epistemic standard deviation for embeddings."""

    mean, epistemic_std, _, _ = predictive_mean_stds(
        head,
        embeddings,
        target_scaler=target_scaler,
    )
    return mean, epistemic_std


def predictive_mean_stds(
    head: SVDKLHead,
    embeddings: Any,
    *,
    target_scaler: StandardScaler | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Return mean plus epistemic, aleatoric, and total standard deviations.

    If ``target_scaler`` is provided, both the posterior mean and standard
    deviations are transformed back to docking-score units.
    """

    _require_torch_gpytorch()
    head.eval()
    head.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent_posterior = head(embeddings)
        mean, epistemic_std = _mix_latent_moments(
            latent_posterior.mean,
            latent_posterior.variance,
            head.likelihood.mixing_weights,
        )
        noise_variance = head.likelihood.base_likelihood.noise.clamp_min(0.0)
        aleatoric_std = noise_variance.sqrt().expand_as(epistemic_std)
        total_std = (epistemic_std.square() + noise_variance).sqrt()
    if target_scaler is not None:
        mean = unscale_mean(mean, target_scaler)
        epistemic_std = unscale_std(epistemic_std, target_scaler)
        aleatoric_std = unscale_std(aleatoric_std, target_scaler)
        total_std = unscale_std(total_std, target_scaler)
    return mean, epistemic_std, aleatoric_std, total_std


def _mix_latent_moments(mean: Any, variance: Any, mixing_weights: Any) -> tuple[Any, Any]:
    """Project independent latent GP moments through learned mixing weights."""

    mixed_mean = mean @ mixing_weights
    mixed_variance = variance.clamp_min(0.0) @ mixing_weights.square()
    return mixed_mean, mixed_variance.clamp_min(0.0).sqrt()


def save_svdkl_checkpoint(
    path: Path,
    *,
    head: SVDKLHead,
    target_scaler: StandardScaler,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save an SVDKL head checkpoint."""

    _require_torch_gpytorch()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "svdkl_head",
        "config": asdict(head.config),
        "head_state_dict": head.state_dict(),
        "target_scaler": target_scaler,
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def save_chemprop_svdkl_checkpoint(
    path: Path,
    *,
    model: ChempropSVDKLModel,
    target_scaler: StandardScaler,
    chemprop_config: ChempropConfig,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a full Chemprop encoder + SVDKL head checkpoint."""

    _require_torch_gpytorch()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "chemprop_svdkl",
        "chemprop_config": asdict(chemprop_config),
        "head_config": asdict(model.head.config),
        "model_state_dict": model.state_dict(),
        "target_scaler": target_scaler,
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_svdkl_checkpoint(
    path: Path,
    *,
    map_location: str = "cpu",
) -> tuple[SVDKLHead, StandardScaler, dict[str, Any]]:
    """Load an SVDKL head checkpoint."""

    _require_torch_gpytorch()
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("model_type") != "svdkl_head":
        raise ValueError(f"not an SVDKL head checkpoint: {path}")
    config = SVDKLConfig(**payload["config"])
    head = SVDKLHead(config)
    head.load_state_dict(payload["head_state_dict"])
    scaler = payload["target_scaler"]
    if not isinstance(scaler, StandardScaler):
        raise ValueError(f"checkpoint has invalid target scaler: {path}")
    metadata = dict(payload.get("metadata") or {})
    return head, scaler, metadata


def load_chemprop_svdkl_checkpoint(
    path: Path,
    *,
    map_location: str = "cpu",
) -> tuple[ChempropSVDKLModel, StandardScaler, dict[str, Any]]:
    """Load a full Chemprop encoder + SVDKL head checkpoint."""

    _require_torch_gpytorch()
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("model_type") != "chemprop_svdkl":
        raise ValueError(f"not a Chemprop SVDKL checkpoint: {path}")
    chemprop_config = ChempropConfig(**payload["chemprop_config"])
    mpnn = build_chemprop_mpnn(chemprop_config)
    head = SVDKLHead(SVDKLConfig(**payload["head_config"]))
    model = ChempropSVDKLModel(
        mpnn,
        head,
        embedding_i=chemprop_config.embedding_i,
    )
    model.load_state_dict(payload["model_state_dict"])
    scaler = payload["target_scaler"]
    if not isinstance(scaler, StandardScaler):
        raise ValueError(f"checkpoint has invalid target scaler: {path}")
    metadata = dict(payload.get("metadata") or {})
    return model, scaler, metadata


__all__ = [
    "GaussianProcessLayer",
    "ChempropConfig",
    "ChempropSVDKLModel",
    "MixingLikelihood",
    "SVDKLConfig",
    "SVDKLHead",
    "build_chemprop_mpnn",
    "fit_target_scaler",
    "load_chemprop_svdkl_checkpoint",
    "load_svdkl_checkpoint",
    "move_batch_to_device",
    "predictive_mean_std",
    "predictive_mean_stds",
    "save_chemprop_svdkl_checkpoint",
    "save_svdkl_checkpoint",
    "scale_targets",
    "unscale_mean",
    "unscale_std",
]
