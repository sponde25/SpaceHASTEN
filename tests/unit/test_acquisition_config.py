from __future__ import annotations

import pytest
from pydantic import ValidationError

from spacehasten.config.acquisition import PortfolioAcquisitionPolicy, load_acquisition_policy


def test_validated_policy_toml(tmp_path) -> None:
    path = tmp_path / "acquisition.toml"
    path.write_text(
        """schema_version = 1
name = "validated"
[quality]
kind = "gaussian_hit_ei"
hit_threshold = -9.7
probability_weight = 1
expected_improvement_weight = 1
xi = 0
uncertainty_source = "epistemic"
[support]
prior = "observed_hits"
current_batch_increment = "hit_probability"
[reward]
kind = "piecewise_linear"
breakpoints = [1, 5, 20]
slopes = [0.25, 1, 2]
weight = 0.1
[crowding]
kind = "logarithmic_post_target"
target = 20
weight = 0.04
scale = 20
[constraint]
kind = "per_cluster_cap"
limit = 100
scope = "batch"
[history]
attempt_policy = "once_per_campaign"
"""
    )
    policy = load_acquisition_policy(path)
    assert policy.name == "validated"
    assert policy.constraint.limit == 100
    assert policy.history.attempt_policy == "once_per_campaign"


def test_policy_rejects_unknown_and_invalid_components() -> None:
    with pytest.raises(ValidationError):
        PortfolioAcquisitionPolicy.model_validate(
            {
                "quality": {"kind": "gaussian_hit_ei", "hit_threshold": -9.7, "bad": 1},
                "reward": {"kind": "piecewise_linear", "breakpoints": [5, 1], "slopes": [1, 2]},
            }
        )
