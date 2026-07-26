# Portfolio Acquisition

Portfolio acquisition combines a molecular-quality model with regional rewards,
crowding terms, and hard constraints. It is enabled explicitly with a versioned
TOML policy; existing greedy, clustering, LCB, and EI behavior is unchanged.

## Policy Structure

```toml
schema_version = 1
name = "example_portfolio"

[quality]
kind = "gaussian_hit_ei"
hit_threshold = -8.0
probability_weight = 1.0
expected_improvement_weight = 1.0
xi = 0.0
uncertainty_source = "epistemic"

[support]
prior = "observed_hits"
current_batch_increment = "hit_probability"

[reward]
kind = "piecewise_linear"
breakpoints = [1.0, 5.0, 20.0]
slopes = [0.25, 1.0, 2.0]
weight = 0.1

[crowding]
kind = "logarithmic_post_target"
target = 20.0
weight = 0.04
scale = 20.0

[constraint]
kind = "per_cluster_cap"
limit = 100
scope = "batch"

[history]
attempt_policy = "once_per_campaign"
```

The hit threshold and weights are scientific inputs and must be chosen for the
campaign. They are not installation defaults.

The piecewise reward uses constant marginal slopes. For the example above, the
first support unit has slope 0.25, support from 1 to 5 has slope 1, support from
5 to 20 has slope 2, and support beyond 20 receives no additional reward.

`per_cluster_cap` limits selected compounds, not observed hits or expected hit
mass. `scope = "batch"` resets the count for every docking round. Prior observed
hits influence support but do not consume current-batch cap capacity.

## Calibration

Portfolio quality uses model-version-specific Gaussian calibration:

```text
calibrated_mean = raw_mean + mean_shift
calibrated_std = sqrt((std_scale * max(epistemic_std, 1e-8))^2 + std_floor^2)
```

Enable calibrator fitting during native training with:

```ini
[General]
TRAIN_FIT_GAUSSIAN_CALIBRATOR = true
```

The current fitter uses the deterministic validation split after restoring the
best checkpoint. Its metadata records that this split was also used for early
stopping. Existing model checkpoints remain valid for legacy acquisition modes;
portfolio acquisition fails closed if a required calibration is absent.

## Running

Standalone docking accepts one policy:

```bash
spacehasten -w WORKSPACE dock \
  --top-n 50000 \
  --cpus 250 \
  --strategy portfolio \
  --portfolio-policy portfolio.toml \
  --atlas-id morgan-r2-1024-t040
```

A screening cycle accepts one policy broadcast to every round or exactly one
policy per round:

```bash
spacehasten -w WORKSPACE screening-cycle \
  --rounds 4 \
  --simsearch-top-n 100 \
  --simsearch-jobs 125 \
  --dock-top-n 50000 \
  --dock-cpus 250 \
  --strategy greedy \
  --dock-acquisition portfolio \
  --portfolio-policy portfolio.toml \
  --atlas-id morgan-r2-1024-t040 \
  --atlas-root SEED_ATLAS
```

The persistent atlas must cover the current database before each acquisition.
Portfolio cap and crowding settings belong to the policy file; legacy
`--cluster-lambda`, `--cluster-alpha`, and `--cluster-cap` options are rejected
when portfolio acquisition is selected.

## Immutable History

Portfolio planning is committed before scheduler submission. The additive
database tables preserve:

- the canonical policy and model calibration;
- candidate watermark and digest;
- exact ordered selections and selection digest;
- raw and calibrated model outputs;
- probability, EI, quality, reward, crowding, support, and final utility;
- pending, scored, and unresolved docking outcomes;
- per-region selected, expected, scored, observed-hit, and unresolved counts.

With `attempt_policy = "once_per_campaign"`, every prior selected ID is excluded
from later newly planned batches, including unresolved attempts. Unresolved
attempts remain failures in the selected denominator and contribute no observed
hit support. Retrying an interrupted round reuses its persisted IDs rather than
recomputing acquisition.

`acquisition.csv` and `acquisition_policy.json` are atomic portable mirrors of
the database history. The database remains authoritative if shared artifacts are
cleaned or moved.
