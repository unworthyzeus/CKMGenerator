# Try 80 Vendored Calibrations

This directory vendors the frozen prior calibration JSON files that Try 80
needs at runtime so cluster uploads do not depend on sibling Try 78 / Try 79
folders being present.

Current files:

- `try78_los_two_ray_calibration.json`
- `try78_nlos_regime_calibration.json`
- `try79_calibration.json`

Source of truth:

- `TFGSeventyEighthTry78/final_calibrations/los_two_ray_calibration.json`
- `TFGSeventyEighthTry78/final_calibrations/nlos_regime_calibration.json`
- `TFGSeventyNinthTry79/calibration.json`

Recalibration/reference code is vendored in:

- `../scripts/recalibrate_priors/try78_los_path_loss_prior.py`
- `../scripts/recalibrate_priors/try78_hybrid_path_loss_reference.py`
- `../scripts/recalibrate_priors/try79_spread_priors.py`

Prior-only evaluation for the current Try 80 frozen JSONs is in:

- `../scripts/evaluate_frozen_priors.py`

Those scripts are kept for reproducibility and future recalibration. The
current Try 80 training/evaluation path does not run them automatically; it
loads the JSON files listed above.
