# Try 79 — delay-spread and angular-spread prior

Analytic (non-DL) prior for the two spread maps `delay_spread` (ns) and
`angular_spread` (degrees). Log-domain ridge regression over 23
morphology features, regime-keyed by topology × LoS/NLoS × antenna-height
bin. Implementation in
[`prior_try79.py`](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/prior_try79.py).

## 1. Why log-domain

3GPP TR 38.901 §7.5 generates delay spread (`lgDS`), azimuth and zenith
angular spreads (`lgASD`, `lgASA`, `lgZSD`, `lgZSA`) from **log-normal**
random variables. WINNER II does the same. That makes `log1p(target)`
the natural regression space for a prior:

```
y_log = log1p(target)          (target in ns or deg, ≥ 0)
```

The model predicts `ŷ_log` and maps back via `expm1(ŷ_log)`, then
clamps.

## 2. Pipeline overview

```
┌─────────────────────────────────────────────────────────────────┐
│ HDF5 sample  (topology, los_mask, uav_height)                  │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌───────────────────────────────┐    ┌──────────────────────────────┐
│ compute_shared_features       │    │ regime key                   │
│  - d_2D, theta (elevation)    │    │  metric × topology_class     │
│  - density/height/nlos 15×15  │    │          × LoS/NLoS          │
│  - density/height/nlos 41×41  │    │          × ant_bin(h_tx)     │
│  - tx_clearance_41            │    └──────────────────────────────┘
│  - tx_below_frac_41           │                   │
│  - blocker_41                 │                   │
└───────────────────────────────┘                   │
          │                                         │
          ▼                                         ▼
┌───────────────────────────────┐    ┌──────────────────────────────┐
│ compute_raw_prior             │    │ regime_coeffs[key] (ridge β) │
│  log-domain closed form       │    │   fit on training cities     │
│  → raw_prior (ns / deg)       │    └──────────────────────────────┘
└───────────────────────────────┘                   │
          │                                         │
          ▼                                         │
┌───────────────────────────────┐                   │
│ build_design_matrix           │                   │
│  → X ∈ R^{H×W×23}             │                   │
└───────────────────────────────┘                   │
          │                                         │
          └──────────────┬──────────────────────────┘
                         ▼
          ŷ_log = X · β_regime          (ridge prediction)
                         │
                         ▼
          ŷ = expm1(ŷ_log)  →  clamp   (LoS / NLoS specific)
                         │
                         ▼
          calibrated prior map  (513 × 513)
```

## 3. Raw prior (closed form, log-domain)

`compute_raw_prior` in
[`prior_try79.py:488`](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/prior_try79.py#L488):

```
prior_log(i, j) =
    base_los/nlos(metric)            ← log1p of a LoS/NLoS base constant
    + topo_bias(metric, topo_class)  ← log offset per topology regime
    + a1 · log(1 + d_2D(i, j))
    + a2 · θ⁻¹(i, j)                 ← rises when UAV is near the horizon
    + a3 · density_41(i, j)
    + a4 · height_41(i, j)
    + a5 · nlos_41(i, j)
    + a6 · nlos_41(i, j) · θ⁻¹(i, j)

raw_prior = clip( expm1( clip(prior_log, 0, 8) ),  0,  clip_hi )
```

The coefficients `a_k` are fixed per `(metric, topology_class, LoS/NLoS)`
from physical intuition — they are **not** fitted end-to-end. The ridge
stage in §4 does all the data fitting on top.

## 4. Design matrix (23 features)

`build_design_matrix` in
[`prior_try79.py:508`](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/prior_try79.py#L508):

| # | Feature | Source | Physics / intuition |
|---|---------|--------|---------------------|
| 1 | `prior_log²`       | raw prior | capture curvature around the physical anchor |
| 2 | `prior_log`        | raw prior | **the anchor** — without this the model has no physics starting point |
| 3 | `log(1 + d_2D)`    | geometry  | Friis-style log-distance term |
| 4 | `theta_norm`       | geometry  | normalised elevation ∈ [0,1] |
| 5 | `theta_inv`        | geometry  | `1 − theta_norm`, rises toward the horizon |
| 6 | `h_norm`           | Tx scalar | UAV height normalised over dataset range |
| 7 | `h_norm²`          | Tx scalar | second-order height term |
| 8 | `density_15`       | 15×15 box | local building density (fine) |
| 9 | `density_41`       | 41×41 box | local building density (coarse) |
| 10 | `height_15`       | 15×15 box | local mean building height (fine) |
| 11 | `height_41`       | 41×41 box | local mean building height (coarse) |
| 12 | `nlos_15`         | 15×15 box | local NLoS pixel fraction (fine) |
| 13 | `nlos_41`         | 41×41 box | local NLoS pixel fraction (coarse) |
| 14 | `nlos_41 · log(1+d_2D)` | cross | far-NLoS rays spread more than near-NLoS |
| 15 | `density_41 · theta_inv` | cross | low-θ dense urban = rich multipath |
| 16 | `blocker_41`      | 41×41 | fraction of pixels with roof above Tx–Rx line |
| 17 | `bias`            | const | intercept |
| 18 | `tx_clearance_41` | height-aware 41×41 | `(h_tx − roof) / 90 m`, clamped — how far above rooftops the UAV is |
| 19 | `tx_below_frac_41` | height-aware 41×41 | fraction of pixels where `h_tx < roof` (rare, but diagnostic of urban-canyon illumination) |
| 20 | `theta_norm · density_41` | cross | high-θ but dense = edge-scattering regime |
| 21 | `tx_clearance · theta_inv` | height-aware × geometry | clearance matters most near the horizon |
| 22 | `tx_below_frac · density_41` | height-aware × morphology | low-UAV rich-urban multipath pocket |
| 23 | `tx_below_frac · nlos_41` | height-aware × NLoS | isolates the specific "UAV below roofs and in NLoS" regime |

Features `18–23` are the **height-aware addition** from the HZ branch;
they replace the earlier `h_x_*` scalar interactions that were fragile on
small samples (see
[`LOS_ANGULAR_SPREAD_NOTE.md`](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/LOS_ANGULAR_SPREAD_NOTE.md)
for context on why LoS angular spread is measured separately).

## 5. Regime keying and ridge fit

Regime key (a string):

```
metric | topology_class | los|nlos | ant_bin
```

with:

- `metric ∈ {delay_spread, angular_spread}`
- `topology_class ∈ 6 classes`
  (`open_sparse_lowrise`, `open_sparse_vertical`,
   `mixed_compact_lowrise`, `mixed_compact_midrise`,
   `dense_block_midrise`, `dense_block_highrise`)
- `ant_bin ∈ 3 bins` at `q1 ≈ 58.12 m`, `q2 ≈ 103.85 m` (dataset quantiles)

Per regime, a ridge fit in `log1p` space:

```
β_regime = ( Xᵀ X + λ · I )⁻¹ Xᵀ y_log       with λ = 1e-3
```

Fallback chain if a regime is under-sampled (
[`fit_keys`](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/prior_try79.py#L547)):

```
1. exact regime                                      metric|topo|los_state|ant_bin
2. same topology + LoS state,      all heights       metric|topo|los_state|all_ant
3. same topology,                  all states        metric|topo|all_los|all_ant
4. global + LoS state,             all heights       metric|global|los_state|all_ant
5. global                                             metric|global|all_los|all_ant
```

At inference, `apply_calibration` walks the same chain top-down until it
finds a fitted β.

## 6. Output clamps (key LoS angular insight)

Defined at
[`prior_try79.py:166`](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/prior_try79.py#L166):

```python
LOS_CLIP_HI  = {"angular_spread": 15.0, "delay_spread": 400.0}
NLOS_CLIP_HI = {"angular_spread": 90.0, "delay_spread": 400.0}
```

Two clamps per metric — different for LoS vs NLoS. Reason for the tight
`15°` LoS angular clamp: in LoS the **direct ray** carries the link; the
angular spread target is a spike distribution near 0°, and a `90°`
headroom only amplifies rare outlier pixels. Full argument in
[`LOS_ANGULAR_SPREAD_NOTE.md`](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/LOS_ANGULAR_SPREAD_NOTE.md).

## 7. Numerical behaviour (validated on the official test split)

From
[`test_eval_dml_hz_v2/eval_summary_test.json`](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/test_eval_dml_hz_v2/eval_summary_test.json)
(2590 held-out samples, 114 regimes fitted, city-holdout):

| Metric | Overall RMSE | LoS RMSE | NLoS RMSE |
|--------|--------------|----------|-----------|
| `delay_spread`   | **28.10 ns**  | 27.49 ns | 29.62 ns |
| `angular_spread` | **13.74 °**   | 15.28 °  | 8.66 °   |

The **raw** (uncalibrated) priors for comparison: delay 65.37 ns / 28.04
LoS / 114.67 NLoS — the calibration layer is the part that closes the
NLoS gap.

## 8. Why this formulation

| Choice | Reason |
|--------|--------|
| `log1p` regression | Matches 3GPP TR 38.901 §7.5 convention (`lgDS`, `lgASD`, `lgASA`, `lgZSD`, `lgZSA`) and suits non-negative heavy-tailed spreads. |
| Ridge, not deep net | Enables per-coefficient interpretability and quick retraining; plenty of regime capacity already via the 6×2×3 split. |
| Regime fallback | Prevents sample-starved bins from producing unstable β; guarantees a prediction always exists. |
| Multiscale box filters (15×15 and 41×41) | Simple, fast, non-trainable morphology descriptor — captures block-level density + corridor-level NLoS support. |
| Height-aware features | UAV altitude changes whether the Tx is above or below nearby rooftops — this fundamentally changes LoS/NLoS propagation statistics. |
| LoS angular clamp at 15° | Physically motivated: direct-ray-dominated LoS has inherently small AS. |

## 9. References

1. **3GPP TR 38.901** §7.5, *Study on channel model for frequencies from 0.5 to 100 GHz* — defines `lgDS`, `lgASD`, `lgASA`, `lgZSD`, `lgZSA` as log-normal large-scale parameters.
2. **WINNER II** channel model (IST-4-027756 final report) — joint log-domain large-scale parameters with correlated evolution.
3. **X. Cai et al.**, *An Empirical Air-to-Ground Channel Model Based on Passive Measurements in LTE*, 2019, [arXiv:1901.07930](https://arxiv.org/abs/1901.07930) — UAV A2G delay spread vs height / horizontal distance.
4. **T. Izydorczyk et al.**, *Angular Distribution of Cellular Signals for UAVs in Urban and Rural Scenarios*, EuCAP 2019 — measured angular spread shrinks with height and LoS dominance.
5. **W. Khawaja et al.**, *UWB A2G Propagation Channel Characterization in an Open Area*, 2019, [arXiv:1906.04013](https://arxiv.org/abs/1906.04013) — Saleh-Valenzuela as a fit for UAV A2G wideband delay.
6. **W. Khawaja et al.**, *Survey of A2G Propagation Channel Modeling for UAVs*, IEEE Comm. Surveys & Tutorials, 2018, [arXiv:1801.01656](https://arxiv.org/abs/1801.01656).
7. **T. Hoerl, R. Kennard**, *Ridge Regression*, Technometrics, 1970 — regularized least squares.
8. In-repo: [`README.md`](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/README.md), [`LOS_ANGULAR_SPREAD_NOTE.md`](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/LOS_ANGULAR_SPREAD_NOTE.md).
