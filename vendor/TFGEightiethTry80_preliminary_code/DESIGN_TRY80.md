# Try 80 - Joint prior-anchored mega-model

## Goal

Try 80 is a **single large multi-task model** that predicts:

- `path_loss`
- `delay_spread`
- `angular_spread`

with one shared backbone and three task heads.

The design intention is:

1. keep the strong deterministic priors from `Try 78` and `Try 79`
2. make DL learn only bounded residual structure
3. preserve the `Try 76 / 77` distribution-first idea
4. avoid regressions versus the frozen prior

## Inputs

Per sample, the model receives 9 channels:

1. normalized `topology_map`
2. `LoS` mask
3. `NLoS` mask
4. `ground_mask`
5. combined `path_loss` prior
6. `LoS` path-loss prior
7. `NLoS` path-loss prior
8. `delay_spread` prior
9. `angular_spread` prior

The model also receives UAV height as a scalar, embedded through the same sinusoidal conditioning family used in `Try 76 / 77`.

All supervision and metrics are restricted to:

- pixels with `topology == 0`
- valid target pixels for the corresponding output

## Split protocol

Try 80 reuses the `Try 76 / 77` split semantics:

- `city_holdout`
- `val_ratio = 0.15`
- `test_ratio = 0.15`
- `split_seed = 42`

So this try stays comparable to the recent distribution-first family.

## Architecture

### Shared trunk

- 4-scale encoder-decoder
- GroupNorm throughout
- base width `96`
- FiLM modulation from sinusoidal height embedding

This gives one large shared spatial representation for all three outputs.

### Distribution-first residual heads

Each task head predicts two region-conditioned residual models:

- one for `LoS`
- one for `NLoS`

For each region, the head has:

1. a **global residual GMM** from pooled encoder features
2. a **local decoder map** that predicts:
   - component weights
   - bounded local residual offset
   - anchor gate `alpha`
   - residual variance term

So the final prediction is not free-form. It is:

`prediction = prior + alpha * bounded_residual`

with `alpha in [0, 1]`.

This is the main safety mechanism that keeps the DL model close to the physical prior whenever evidence is weak.

## Why this is based on Try 76 / 77

The model keeps the two-stage logic:

- Stage A: infer sample-level distribution structure
- Stage B: place that structure spatially on the map

What changes versus `Try 76 / 77` is that the target distribution is now modeled in **residual space around the prior**, not in raw target space from scratch.

That is the adaptation recommended by:

- [TRY78_DL_RECOMMENDATIONS.md](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/TRY78_DL_RECOMMENDATIONS.md)
- [DL_OVER_PRIOR_RECOMMENDATION.md](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/DL_OVER_PRIOR_RECOMMENDATION.md)

## Loss bundle

Per task, Try 80 combines:

- `map_nll`
- `dist_kl`
- `moment_match`
- `anchor`
- `prior_guard`
- `outlier_budget`
- `rmse`
- `mae`

The key new term is `prior_guard`:

- it penalizes pixels where the model becomes worse than the prior
- so the network is explicitly discouraged from drifting away from a good prior unless it wins back error

## Metrics

Validation and test JSONs report pixel-weighted RMSE and MAE for:

- `path_loss`
- `delay_spread`
- `angular_spread`

at these levels:

1. overall
2. LoS
3. NLoS
4. 9 macro-experts = `3 topology groups x 3 antenna bins`

The same metrics are also reported for the frozen prior, so every epoch can be compared directly against the baseline it is supposed to improve.

## Optional precompute

Try 80 can train in two modes:

- on-the-fly prior computation
- reading priors from an auxiliary HDF5 cache

The cache is produced by:

- [precompute_priors_hdf5.py](/c:/TFG/TFGpractice/TFGEightiethTry80/scripts/precompute_priors_hdf5.py)

This is optional. The trainer still works without it.
