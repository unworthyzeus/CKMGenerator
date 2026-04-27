# DL model on top of the Try 79 prior — recommendation

This note describes how a deep-learning model should consume the `Try 79`
height-aware calibrated prior as an input, without throwing away the
physics-grounded behaviour the prior already delivers.

## TL;DR

- Freeze the `Try 79` prior. Feed its calibrated `delay_spread` and
  `angular_spread` maps as **input channels**, not as training targets.
- Let the DL model learn a **bounded residual** on top of the prior
  (`y = prior + delta`, with `delta` soft-clamped), conditioned on UAV
  height via FiLM, split LoS/NLoS at the output stage.
- Supervise in the same space the prior works in: log-domain
  (`log1p(target)`), with a pixel-wise Gaussian or GMM head so the tail
  structure of the spread distributions is preserved.
- Train per-topology-class experts (6 classes) like Try 76 / Try 77
  already do. Reuse those split definitions for integrity.

## Why the prior as input, not teacher

The calibrated prior is already:

- regime-aware (topology × LoS/NLoS × antenna bin),
- log-domain (matches 3GPP / WINNER II conventions for `lgDS`, `lgASD`,
  `lgASA`),
- LoS-clamped for angular spread (direct-ray physics, see
  `LOS_ANGULAR_SPREAD_NOTE.md`).

A vanilla CNN trained end-to-end on these targets collapses the NLoS
tail and smears the LoS spike — exactly the failure mode that motivated
Try 76 / Try 77's distribution-first GMM experts. Using the prior as an
**anchor input** gives the DL model a strong baseline to correct instead
of one to rediscover from pixels.

## Inputs (per sample, 513×513)

From the HDF5 plus Try 79 shared features (use `compute_shared_features`
and `apply_calibration` directly at dataloader time or preload):

| Channel group | Channels | Notes |
|---------------|----------|-------|
| Geometry | `topology_map/90`, `los_mask`, `nlos_mask`, `ground_mask` | same as Try 76 |
| Tx geometry | normalized radius, `sin/cos(azimuth)`, `theta_norm`, `theta_inv` | already computed in `shared_features` |
| Height-aware | `tx_clearance_41`, `tx_below_frac_41` | key channels from Try 79 HZ fix |
| Morphology | `density_15/41`, `height_15/41`, `nlos_15/41`, `blocker_41` | box-filter features |
| **Prior (frozen)** | `prior_delay_calibrated`, `prior_angular_calibrated` | Try 79 output, log1p space |
| **Prior (raw)** | `prior_delay_raw`, `prior_angular_raw` | optional, pre-calibration |

Scalar conditioning: `antenna_height_m` → sinusoidal FiLM (16 log-spaced
frequencies over [12, 478] m), injected at 3–4 decoder blocks — same
recipe as PMHHNet.

## Architecture (small, not PMNet-sized)

The prior does most of the geometry work. The DL model only needs to
model **local residual structure** (corridor reflections, knife-edge
fringes, building-edge spikes). A lightweight UNet is enough:

- Encoder: 4 scales, width 48→384, GroupNorm, SiLU.
- Decoder: symmetric, skip connections, FiLM at every decoder block.
- No HF Laplacian stem needed — the prior already carries large-scale
  structure; the encoder sees raw topology for edges.
- Dropout2d(0.1), weight_decay 0.01.

### Output head — residual, not raw target

```
mu_residual, log_sigma = head(features)           # 2 channels
mu = prior_log1p + tanh(mu_residual) * delta_max  # soft clamp in log1p
y_hat = expm1(mu)
```

`delta_max` per metric (log1p-units): ~0.7 for `delay_spread`, ~0.5 for
`angular_spread`. This prevents the DL model from drifting far from the
prior on out-of-distribution cities — the worst-case failure mode for
the CKM city-holdout protocol.

For the tail-heavy targets (both spreads are spike+exp per
`distribution_classes.md`), a GMM head with K=3 is a better fit than a
single Gaussian — same rationale as Try 77:

```
pi_k(i,j), delta_mu_k(i,j), log_sigma_k(i,j)   # per component
mu_k(i,j) = prior_log1p(i,j) + tanh(delta_mu_k) * delta_max
```

Reconstruction: `y_hat = Σ_k pi_k * expm1(mu_k)`.

## Loss bundle

- `map_nll`: pixel-wise GMM NLL on `log1p(target)`. Use `nan_mask` and
  `ground_mask` exactly like Try 77.
- `residual_anchor`: `lambda_anchor * mean(delta_mu**2)`, 0.05–0.1.
  Keeps the DL model close to the prior when evidence is weak (unseen
  city, sparse features).
- `los_angular_suppress`: down-weight LoS angular residual by 0.3–0.5.
  See `LOS_ANGULAR_SPREAD_NOTE.md` — chasing LoS angular RMSE overfits
  spike outliers.
- `moment_match` (Try 77) + `outlier_budget` optional.

Avoid: Huber on NLoS (underweights the tail); corridor weighting
centred on Tx (hurts edges for full 513² spread maps).

## Training protocol

- **Splits**: reuse Try 74 / Try 75 official city-holdout split for
  `train/val/test` integrity. Do **not** rebuild from Try 58/59.
- **Per-topology experts**: 6 experts, one per topology class, same
  partition used by Try 76 / Try 77. Each model is small enough that 6
  experts stay in budget.
- **Per-metric**: one model per metric (delay, angular) — 12 experts
  total, matching Try 77's 6×2 layout.
- Optimizer: AdamW lr=3e-4, wd=0.01, warmup 3 epochs,
  `ReduceLROnPlateau` patience=15.
- Label noise `sigma_db`-equivalent on log1p targets (σ≈0.02) — fixes
  the Try 76/77 late-epoch overfit described in `CLAUDE.md`.
- Batch 1–2 per GPU, grad accumulation 8, D4 TTA on validation.
- No SWA, no EMA initially — keep it lean like Try 76/77 and add later
  only if val instability shows up.

## Evaluation

- Report **four** aggregate numbers per metric: overall / LoS / NLoS /
  NLoS-only-weighted. Follow the guidance in
  `LOS_ANGULAR_SPREAD_NOTE.md`: the NLoS number is the one worth
  optimising for angular spread.
- Ablation baselines to beat:
  - Prior only: delay ~28.1 ns, angular ~13.7° overall (test).
  - Prior + residual DL: target is to cut NLoS delay RMSE
    below ~25 ns and match or beat prior on LoS (no regression).
- Sanity: the DL prediction **must not** underperform the prior on any
  (topology × LoS/NLoS × height_bin) regime. If it does, raise
  `lambda_anchor` and retrain — the anchor term is the safety net for
  city-holdout generalisation.

## Integration with Try 80 (path loss + spreads)

If this is a subcomponent of the Try 80 joint model described in
`prompt_de_la_ostia_para_manana.md`:

- Keep the Try 79 prior as a frozen channel fed into the shared trunk
  alongside the Try 78 two-ray path-loss prior.
- Use a single PMHHNet-style backbone with **three output heads**
  (path_loss, delay_spread, angular_spread), each anchored to its own
  prior via the `prior + bounded_residual` pattern above.
- FiLM conditioning on UAV height is shared across heads; GMM head is
  used only for the spread heads (path loss is unimodal enough for a
  Gaussian head).

## Files to add (when implementing)

```
TFGEightiethTry80/
├── src/
│   ├── model_try80.py           # UNet + FiLM + residual-on-prior head
│   ├── losses_try80.py          # GMM-NLL + residual anchor + LoS-weight
│   ├── data_utils.py            # loads Try 79 prior at __getitem__
│   └── config_try80.py
├── experiments/
│   └── try80_expert_<topology>_<metric>.yaml
└── train_try80.py
```

## References

- `prior_try79.py` — source of the calibrated prior and shared features.
- `LOS_ANGULAR_SPREAD_NOTE.md` — why LoS angular RMSE is a weak metric.
- `TFGSeventySixthTry76/DESIGN_TRY76.md` — GMM head reference.
- `TFGSeventySeventhTry77/` — spread-target GMM experts (closest prior
  art in this repo).
- 3GPP TR 38.901 §7.5 — log-domain LSP convention that the prior
  already matches.
