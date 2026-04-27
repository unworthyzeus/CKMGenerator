# Try 80 prior formulas and references

## Scope

Try 80 does not invent a new hand-crafted prior.

It freezes and reuses:

- the `Try 78` path-loss prior
- the `Try 79` spread priors

This note documents the formulas that are actually consumed by the data loader.

## 1. Path loss prior from Try 78

### 1.1 LoS branch: coherent two-ray over FSPL

The LoS prior used by Try 80 is the `Try 78` coherent two-ray model:

`PL_LoS = FSPL + C_2ray(h, d) + bias(h)`

with

`C_2ray = -20 log10 |1 + rho(h) * (d_los / d_ref) * exp(-j (k (d_ref - d_los) + phi(h)))|`

where:

- `d_los` is the direct 3D path length
- `d_ref` is the reflected 3D path length
- `k = 2 pi / lambda`
- `rho(h)`, `phi(h)`, and `bias(h)` are interpolated from the calibrated `Try 78` height-bin fit

The implementation is taken from the calibrated LoS branch in:

- [prior_try78.py](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/prior_try78.py)
- [los_two_ray_calibration.json](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/final_calibrations/los_two_ray_calibration.json)

Try 80 also vendors this recalibration source locally:

- [try78_los_path_loss_prior.py](/c:/TFG/TFGpractice/TFGEightiethTry80/scripts/recalibrate_priors/try78_los_path_loss_prior.py)
- [try78_hybrid_path_loss_reference.py](/c:/TFG/TFGpractice/TFGEightiethTry80/scripts/recalibrate_priors/try78_hybrid_path_loss_reference.py)

Reference logic:

- free-space path loss
- coherent two-ray / ground-reflection correction

Relevant references already collected in the repo:

1. WOCC 2021 UAV A2G two-ray measurement paper, cited in [TRY78_LOS_DOCUMENTATION.md](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/TRY78_LOS_DOCUMENTATION.md)
2. UAV A2G survey cited there as support for two-ray-like LoS oscillations
3. Standard two-ray treatment as summarized in classical wireless texts such as Rappaport

### 1.2 NLoS branch: COST-231 + A2G envelope + regime calibration

The NLoS path-loss prior follows the embedded old `Try 78` formulation:

1. build a raw formula prior from:
   - COST-231 Hata style term
   - A2G LoS / NLoS elevation-angle envelope
   - LoS/NLoS masking
2. build multiscale morphology features
3. apply regime-wise linear calibration

The raw NLoS part is:

`PL_NLoS_raw = max(COST231, A2G_NLoS(theta))`

with the A2G NLoS envelope:

`A2G_NLoS(theta) = lambda0 + bias + amp * exp(-(90 - theta) / tau)`

and the raw LoS blend:

`PL_LoS_raw = 0.7 * LoS_path + 0.3 * min(LoS_path, A2G_LoS(theta))`

The calibrated map then becomes:

`PL_cal = X beta_regime`

with `X` built from:

- prior and prior squared
- `log(1 + d_2D)`
- local density and height at `15x15` and `41x41`
- local NLoS support
- shadow sigma proxy
- elevation-angle normalization

The exact implementation path is embedded in:

- [evaluate_hybrid_try78.py](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/evaluate_hybrid_try78.py)
- [nlos_regime_calibration.json](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/final_calibrations/nlos_regime_calibration.json)

Supporting references:

1. COST-231 / Hata family for urban macro path loss
2. elevation-aware UAV A2G path-loss formulations discussed in the Try 78 notes

## 2. Spread priors from Try 79

Try 80 uses the calibrated spread priors from `Try 79` for:

- `delay_spread`
- `angular_spread`

### 2.1 Raw spread prior

The raw spread prior is built in log-domain:

`prior_log = base(los/nlos) + topo_bias + a1 log(1 + d) + a2 theta_inv + a3 density_41 + a4 height_41 + a5 nlos_41 + a6 nlos_41 * theta_inv`

and then mapped back with:

`prior = expm1(prior_log)`

This directly follows the `Try 79` implementation in:

- [prior_try79.py](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/prior_try79.py)
- [try79_spread_priors.py](/c:/TFG/TFGpractice/TFGEightiethTry80/scripts/recalibrate_priors/try79_spread_priors.py)

### 2.2 Shared feature stack

The calibrated `Try 79` prior uses multiscale shared features:

- `density_15`, `density_41`
- `height_15`, `height_41`
- `nlos_15`, `nlos_41`
- `theta_norm`, `theta_inv`
- `tx_clearance_41`
- `tx_below_frac_41`
- `blocker_41`

These are built from the HDF5 geometry only, without DL.

### 2.3 Regime-wise calibration

The final calibrated spread prediction is a ridge-regression model in `log1p(target)`:

`y_log = X beta_regime`

with regimes keyed by:

- metric
- 6-class topology class
- LoS / NLoS
- antenna-height bin

and the same fallback hierarchy documented in `Try 79`.

This is loaded from:

- [calibration.json](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/calibration.json)
- [try79_calibration.json](/c:/TFG/TFGpractice/TFGEightiethTry80/calibrations/try79_calibration.json)

### 2.4 Why log-domain

The spread priors are calibrated in `log1p` space because:

- spreads are non-negative
- they are heavy-tailed
- 3GPP TR 38.901 models `DS`, `ASD`, `ASA`, `ZSD`, `ZSA` in log domain

This rationale is already documented in:

- [README.md](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/README.md)
- [LOS_ANGULAR_SPREAD_NOTE.md](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/LOS_ANGULAR_SPREAD_NOTE.md)

## 3. Final priors consumed by Try 80

Per sample, Try 80 constructs:

- `path_loss_prior_los`
- `path_loss_prior_nlos`
- `path_loss_prior = los_mask * prior_los + nlos_mask * prior_nlos`
- `delay_spread_prior`
- `angular_spread_prior`

These priors are then fed as frozen input channels.

## 4. References

1. 3GPP TR 38.901, “Study on channel model for frequencies from 0.5 to 100 GHz”.
2. WINNER II channel model documentation for log-domain large-scale parameters.
3. Try 78 references collected in [TRY78_LOS_DOCUMENTATION.md](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/TRY78_LOS_DOCUMENTATION.md).
4. Try 79 references collected in [README.md](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/README.md).
