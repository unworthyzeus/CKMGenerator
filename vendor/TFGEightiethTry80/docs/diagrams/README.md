# Try 78 + Try 79 priors — supervisor-facing explanation

This folder documents the two analytic priors that `Try 80` freezes as input
channels:

- **Try 78** — path-loss prior (LoS coherent two-ray + NLoS COST-231 envelope
  with regime-wise residual calibration).
- **Try 79** — spread priors for `delay_spread` and `angular_spread`
  (log-domain ridge regression over multiscale morphology features, regime
  keyed by topology × LoS/NLoS × antenna-height bin).

Neither uses deep learning. Both are the deterministic, physics-grounded
reference that any learned model (`Try 80`) must meet or beat.

## How to use this folder

| File | What it covers |
|------|----------------|
| [`try78_path_loss_prior.md`](try78_path_loss_prior.md) | FSPL, 3D ray geometry, coherent two-ray interference, radial residual fallback, and NLoS COST-231 + A2G envelope — with the formulas the code actually evaluates and the bibliographic references behind each term. |
| [`try79_spread_prior.md`](try79_spread_prior.md) | Log-domain regression for delay/angular spread: raw prior, 23-feature design matrix, regime keying, LoS-specific clamp for angular spread, and the 3GPP / WINNER II rationale. |
| [`try78_vs_try79_pipeline.md`](try78_vs_try79_pipeline.md) | One-page comparison diagram showing both priors flowing into `Try 80` (how they fit together as input channels to the DL model). |

## Supervisor-level one-liner

> "We split the channel model into two well-established blocks — a
> deterministic **two-ray LoS** model plus a **COST-231-style NLoS
> envelope** for path loss (`Try 78`), and a **log-domain regime regressor**
> for the large-scale spreads (`Try 79`). Both are calibrated on the
> training cities with city-holdout integrity, and both follow the
> log-domain conventions of 3GPP TR 38.901 §7.5. The deep model in `Try 80`
> only has to learn a bounded residual on top."

## Where the code lives

- Try 78: [`TFGSeventyEighthTry78/prior_try78.py`](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/prior_try78.py),
  [`evaluate_hybrid_try78.py`](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/evaluate_hybrid_try78.py),
  calibrations under `final_calibrations/`.
- Try 79: [`TFGSeventyNinthTry79/prior_try79.py`](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/prior_try79.py),
  calibration under `test_eval_dml_hz_v2/calibration.json`.
- Prior-consumer in Try 80: [`PRIOR_FORMULAS_TRY80.md`](../PRIOR_FORMULAS_TRY80.md).
