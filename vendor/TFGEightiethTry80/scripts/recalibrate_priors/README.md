# Try 80 Prior Recalibration Sources

This folder vendors the original Try 78 / Try 79 calibration code so Try 80
contains both:

- the frozen calibration assets used at runtime, under `calibrations/`
- the source scripts needed to regenerate or audit those priors later

These scripts are copied for traceability and optional future recalibration.
They are not executed by normal Try 80 training or evaluation.

For Try 80 prior-only evaluation, use:

- `../evaluate_frozen_priors.py`

## Files

- `try78_los_path_loss_prior.py`
  - Source: `TFGSeventyEighthTry78/prior_try78.py`
  - Fits the Try 78 LoS path-loss prior calibration:
    radial residual and coherent two-ray parameters.

- `try78_hybrid_path_loss_reference.py`
  - Source: `TFGSeventyEighthTry78/evaluate_hybrid_try78.py`
  - Reference implementation for the final Try 78 hybrid path-loss prior:
    LoS from the calibrated two-ray branch plus NLoS from the vendored
    regime-calibrated coefficients. This is evaluation/reference code, not a
    fresh fit for the NLoS coefficients.

- `try79_spread_priors.py`
  - Source: `TFGSeventyNinthTry79/prior_try79.py`
  - Fits the Try 79 delay-spread and angular-spread prior calibrations with
    regime-wise ridge regression in `log1p` space.

## Current Runtime Link

Try 80 still uses frozen JSON files by default:

- `calibrations/try78_los_two_ray_calibration.json`
- `calibrations/try78_nlos_regime_calibration.json`
- `calibrations/try79_calibration.json`

If the priors are recalibrated in the future, write the new JSON files to
`calibrations/` and update the `prior:` paths in the Try 80 experiment YAML.
