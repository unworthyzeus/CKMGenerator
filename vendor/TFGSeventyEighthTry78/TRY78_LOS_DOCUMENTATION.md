# Try 78 LoS Documentation

## Goal

Try 78 is now a **LoS-only, no-DL baseline**.

NLoS has been removed completely from this branch.

The progression inside Try 78 was:

1. Start from `FSPL`.
2. Verify that the LoS residual has a strong **radial / ring-like** pattern.
3. Implement a **radial residual lookup** by height.
4. Upgrade that into a **coherent two-ray physics model** with fitted effective reflection parameters.

## Current implementation

Main file:

- [prior_try78.py](C:/TFG/TFGpractice/TFGSeventyEighthTry78/prior_try78.py)
- [evaluate_hybrid_try78.py](C:/TFG/TFGpractice/TFGSeventyEighthTry78/evaluate_hybrid_try78.py)

The script now compares three LoS-only models:

1. `FSPL`
2. `FSPL + radial residual`
3. `coherent two-ray`

The coherent two-ray model is:

`PL = FSPL - 20 log10 |1 + rho * (d_los / d_ref) * exp(-j * (k * (d_ref - d_los) + phi))| + bias`

with effective parameters fitted per height bin:

- `rho(height_bin)`
- `phi(height_bin)`
- `bias(height_bin)`

Default setting currently used:

- `height_bin_m = 5.0`

Current two-ray search defaults:

- `phi_grid_size = 48`
- `rho_max = 1.5`
- `rho_step = 0.05`
- `two_ray_per_sample_cap = 2500`
- `two_ray_per_bin_cap = 8000`

## Main outputs

Primary outputs:

- [hybrid_out_final_dml/hybrid_eval_summary.json](C:/TFG/TFGpractice/TFGSeventyEighthTry78/hybrid_out_final_dml/hybrid_eval_summary.json)
- [hybrid_out_final_dml/progress.json](C:/TFG/TFGpractice/TFGSeventyEighthTry78/hybrid_out_final_dml/progress.json)
- [hybrid_out_final_dml/progress.out](C:/TFG/TFGpractice/TFGSeventyEighthTry78/hybrid_out_final_dml/progress.out)
- [final_calibrations/los_two_ray_calibration.json](C:/TFG/TFGpractice/TFGSeventyEighthTry78/final_calibrations/los_two_ray_calibration.json)
- [final_calibrations/nlos_regime_calibration.json](C:/TFG/TFGpractice/TFGSeventyEighthTry78/final_calibrations/nlos_regime_calibration.json)

## Best smoke-test result so far

Command used:

```bash
python TFGpractice/TFGSeventyEighthTry78/prior_try78.py --max-samples 3000 --out-dir TFGpractice/TFGSeventyEighthTry78/prior_out
```

Eval result on LoS pixels only:

- `FSPL RMSE`: `3.6643 dB`
- `FSPL + radial residual RMSE`: `3.4071 dB`
- `coherent two-ray RMSE`: `1.7649 dB`

This means the current physics-only LoS branch is already **sub-2 dB** on the 3000-sample smoke test.

## Full-dataset result

Command used for the new LoS model on all samples:

```bash
python TFGpractice/TFGSeventyEighthTry78/prior_try78.py --skip-plots --out-dir TFGpractice/TFGSeventyEighthTry78/prior_out_all
```

Result on all samples, eval split only, LoS pixels:

- `FSPL RMSE`: `3.9540 dB`
- `FSPL + radial residual RMSE`: `3.6496 dB`
- `coherent two-ray RMSE`: `1.7574 dB`

## Final hybrid result

Hybrid definition:

- `LoS = coherent two-ray` from the new Try 78
- `NLoS = regime-calibrated NLoS` embedded in the final unified evaluator

Final output:

- [hybrid_out_final_dml/hybrid_eval_summary.json](C:/TFG/TFGpractice/TFGSeventyEighthTry78/hybrid_out_final_dml/hybrid_eval_summary.json)

Corrected full-dataset eval result:

- `hybrid LoS RMSE`: `1.7516 dB`
- `hybrid NLoS RMSE`: `3.3967 dB`
- `hybrid overall RMSE`: `1.9246 dB`

This is the strongest current Try 78 result.

## What Comes From Papers

Transferred from the literature / standards:

- The idea that LoS A2G behavior can be dominated by a **two-ray / ground-reflection** term.
- The use of a **coherent interference correction** on top of `FSPL`.
- The interpretation that altitude and elevation angle dominate large-scale LoS behavior.

These are the parts that are mainly backed by the UAV A2G literature and two-ray references.

## What Seems To Be Our Own Dataset Insight

What appears to be specific to our work on CKM:

- The decision to inspect the actual CKM LoS maps and notice that the residual is almost a **ring field**.
- The conclusion that in CKM, buildings often behave more like a **visibility mask** over a global radial structure than like the main source of LoS residual shape.
- The practical hybrid recipe:
  - `LoS = fitted two-ray`
  - `NLoS = old regime calibration`
- The choice to store the two final calibrations together in:
  - [final_calibrations](C:/TFG/TFGpractice/TFGSeventyEighthTry78/final_calibrations)

## Important note about the old overall metric

The old `old_try_78_with_nlos/prior_try78.py` has a bug in its aggregate:

- `prior_rmse_overall_pw`
- `calib_rmse_overall_pw`

were weighted using `n_los` instead of using all valid pixels.

So the old reported `3.9197 dB` overall is not the clean final overall number to use.

The hybrid evaluation script fixes that and computes the true overall RMSE over all valid pixels.

## Radial baseline result

Alternative height bins tested on the same 3000-sample subset for the radial branch:

- `5 m`: `3.4071 dB`
- `10 m`: `3.5418 dB`
- `20 m`: `3.6228 dB`

So `5 m` remains the current default.

## What the images show

Representative panel images:

- [Abidjan sample_00022](C:/TFG/TFGpractice/TFGSeventyEighthTry78/tmp_visuals/Abidjan_sample_00022_panels.png)
- [Abidjan sample_00112](C:/TFG/TFGpractice/TFGSeventyEighthTry78/tmp_visuals/Abidjan_sample_00112_panels.png)
- [Abidjan sample_00055](C:/TFG/TFGpractice/TFGSeventyEighthTry78/tmp_visuals/Abidjan_sample_00055_panels.png)
- [Abu Dhabi sample_00190](C:/TFG/TFGpractice/TFGSeventyEighthTry78/tmp_visuals/Abu Dhabi_sample_00190_panels.png)

Observed pattern:

- The LoS `path_loss` map shows **clear concentric rings** around the transmitter.
- The residual `path_loss - FSPL` also shows these rings strongly.
- The building map does **not** appear to define the residual shape directly.
- Buildings mostly clip the visible LoS region, acting like a geometric mask over a more global radial pattern.

Interpretation:

- LoS in CKM does not look like generic urban complexity.
- It looks much closer to **FSPL plus a coherent interference term**.
- This is consistent with a **two-ray / ground-reflection type mechanism** or a closely related deterministic simulator artifact.

## 50-sample analysis

File:

- [analysis_50_samples.json](C:/TFG/TFGpractice/TFGSeventyEighthTry78/prior_out/analysis_50_samples.json)

Setup:

- 50 eval samples
- height-stratified
- taken from the same 3000-sample subset as the smoke test

Key finding:

- Mean `radial_r2`: about `0.70`
- Median `radial_r2`: about `0.74`
- Mean oracle radial gain: about `1.70 dB`

Meaning:

- If each sample could use its own perfect radial residual profile, the gain over raw FSPL would be much larger than what the train-split radial lookup obtains.
- That analysis was the bridge from the radial baseline to the coherent two-ray implementation.

## Online research notes

The online literature points in the same direction:

- The WOCC 2021 UAV measurement paper reports that a **two-ray ground-reflection model** is more applicable than a single-ray model for short-distance UAV A2G links:
  - https://doi.org/10.1109/WOCC53213.2021.9603250
  - metadata page: https://ntut.elsevierpure.com/en/publications/channel-modeling-of-air-to-ground-signal-measurement-with-two-ray/
- The UAV path-loss survey explicitly notes that in LoS, the **rapidly varying path loss and peaks** are attributable to **two-ray model behavior**:
  - https://www.mdpi.com/1424-8220/23/10/4775/htm
- A recent elevation-aware UAV A2G model also reinforces that angle and altitude dominate large-scale LoS behavior:
  - https://www.mdpi.com/2227-7390/13/21/3377

Current inference from those sources plus the CKM images:

- The ring structure in CKM LoS is very likely not accidental.
- A coherent two-ray-like mechanism is now the best working explanation.

## Current fitted two-ray behavior

On the 3000-sample smoke test, the fitted parameters show:

- `phi` consistently lands at or very near `0`
- `rho` often falls around `0.5 - 0.65`
- some high bins jump to `rho = 1.5` with a positive bias around `+3.4 dB`

Interpretation:

- The dataset seems to prefer a largely in-phase reflection term.
- The main variation is currently being absorbed by `rho` and `bias`.
- The high-altitude bins probably need a smoother or more constrained parameterization.

## Why the radial branch was not enough

The radial lookup was too crude:

- It averages samples that share similar height but not necessarily the same effective physical state.
- It uses only `radius` and `height_bin`.
- It does not explicitly model:
  - coherent phase difference between direct and reflected rays
  - reflection coefficient magnitude
  - path-difference physics
  - any effective simulator-specific bias

So it captured the existence of rings, but not the mechanism that places them.

## Best next experiments

### 1. Smooth the two-ray parameter curves

Right now `rho`, `phi`, and `bias` are fit independently per bin.

Next step:

- regularize or smooth them across height
- avoid suspicious spikes at sparse high-altitude bins

### 2. Increase two-ray search fidelity

Try:

- finer `phi` grid
- finer `rho` step
- optional `rho_max > 1.5` only if justified

### 3. Add a small residual on top of two-ray

Try:

- `PL = two_ray + small_residual(height, radius)`

This keeps the physics as the backbone while allowing a small structured correction.

### 4. Fit in phase/path-difference domain directly

Use features like:

- `delta_d = d_ref - d_los`
- phase `2*pi*delta_d/lambda`

This may be cleaner than working only in raw radial distance.

## How to rerun

Main command:

```bash
python TFGpractice/TFGSeventyEighthTry78/prior_try78.py --out-dir TFGpractice/TFGSeventyEighthTry78/prior_out
```

Quick smoke test:

```bash
python TFGpractice/TFGSeventyEighthTry78/prior_try78.py --max-samples 3000 --skip-plots --out-dir TFGpractice/TFGSeventyEighthTry78/prior_out_smoke
```

Load an existing calibration without refitting:

```bash
python TFGpractice/TFGSeventyEighthTry78/prior_try78.py --calibration-json TFGpractice/TFGSeventyEighthTry78/prior_out/calibration.json --skip-fit
```

## Current status in one line

Try 78 has already established that:

- **LoS is strongly deterministic**
- **the dominant residual is radial/oscillatory**
- **a coherent two-ray model is enough to reach sub-2 dB on the smoke test**

The path now looks much more like **refining coherent LoS physics** than **training a better DL model**.
