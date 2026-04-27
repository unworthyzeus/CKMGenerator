# Try 78 + Try 79 → Try 80 — one-page pipeline

```
                     HDF5 sample  (513×513)
     ┌────────────────────────────────────────────────────────────────┐
     │  topology_map · los_mask · path_loss · delay_spread · ang_sp.  │
     │  uav_height (scalar)                                           │
     └────────────────────────────────────────────────────────────────┘
                            │                 │
       ┌────────────────────┘                 └────────────────────┐
       ▼                                                           ▼
┌──────────────────────────────────┐       ┌──────────────────────────────────┐
│      Try 78 — path-loss prior    │       │      Try 79 — spread priors      │
│                                  │       │                                  │
│  LoS branch:                     │       │  compute_shared_features         │
│    FSPL(d_los, f)                │       │    (15×15 & 41×41 morphology,    │
│    + C_2ray(ρ(h),φ(h),d_los,d_ref)│      │     tx_clearance, tx_below,      │
│    + bias(h)                     │       │     θ, θ⁻¹, d_2D, ...)           │
│                                  │       │                                  │
│  NLoS branch:                    │       │  raw_prior (log1p, closed form)  │
│    max(COST-231, A2G_NLoS(θ))    │       │                                  │
│    → regime ridge calibration    │       │  design matrix X  (23 features)  │
│                                  │       │                                  │
│  PL_prior = mask · LoS + ~ · NLoS│       │  ŷ_log = X · β_{regime}          │
│  clamp [20 dB, 180 dB]           │       │  ŷ = expm1(clamp)                │
└──────────────────────────────────┘       └──────────────────────────────────┘
                 │                                           │
                 │  path_loss_prior (513×513)                │
                 │                                           │  delay_spread_prior
                 │                                           │  angular_spread_prior
                 ▼                                           ▼
     ┌────────────────────────────────────────────────────────────────┐
     │                       Try 80 — DL model                        │
     │                                                                │
     │  Input channels (frozen priors + raw geometry):                │
     │    • topology_map / 90                                         │
     │    • los_mask, nlos_mask, ground_mask                          │
     │    • path_loss_prior (Try 78)          ← anchor                │
     │    • delay_spread_prior  (Try 79)      ← anchor                │
     │    • angular_spread_prior (Try 79)     ← anchor                │
     │    • θ, d_2D, tx_clearance, ...                                │
     │                                                                │
     │  UNet trunk + sinusoidal FiLM on uav_height                    │
     │  Residual heads (bounded `tanh · δ_max` around the priors)     │
     │  GMM head (K=3) for spread targets                             │
     │                                                                │
     │  Output: calibrated dense maps (513×513) for path_loss,        │
     │          delay_spread, angular_spread                          │
     └────────────────────────────────────────────────────────────────┘
```

## Key property

Each DL output is `prior + bounded_residual`, so:

1. On unseen cities, if the DL signal is weak, the prediction **falls
   back to the physics prior** (safe baseline).
2. The DL model only has to learn the **residual structure** (corridor
   reflections, knife-edge fringes, tail behaviour) — not the large-scale
   geometry that Try 78 and Try 79 already solve in closed form.
3. Both priors are **log-domain-friendly**: Try 78 in dB, Try 79 in
   `log1p` — aligned with 3GPP TR 38.901 conventions.

## Split integrity

Priors are fit on **training cities only** (city-holdout). The
calibration files in `Try 78` and `Try 79` must be used as-is; they
**must not** be re-fit on the same cities Try 80 is evaluated on,
because that would leak test-city statistics into the prior channels.

## See also

- [`try78_path_loss_prior.md`](try78_path_loss_prior.md)
- [`try79_spread_prior.md`](try79_spread_prior.md)
- [`../PRIOR_FORMULAS_TRY80.md`](../PRIOR_FORMULAS_TRY80.md) (Try 80
  consumer-side formulas)
- [`/c:/TFG/TFGpractice/TFGSeventyNinthTry79/DL_OVER_PRIOR_RECOMMENDATION.md`](/c:/TFG/TFGpractice/TFGSeventyNinthTry79/DL_OVER_PRIOR_RECOMMENDATION.md)
  — recommended residual-on-prior architecture for Try 80.
