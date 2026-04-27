# Try 78 — path-loss prior

Analytic path-loss model used by `Try 80` as a frozen input channel.
Two branches: LoS (coherent two-ray over FSPL) and NLoS (COST-231 +
elevation envelope, then regime-wise residual calibration).

All formulas below are the exact expressions evaluated in
[`prior_try78.py`](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/prior_try78.py)
and [`evaluate_hybrid_try78.py`](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/evaluate_hybrid_try78.py).

## 1. Geometry and constants

The simulation grid is `513 × 513` pixels at `1 m / pixel`; the UAV
transmitter sits at the centre pixel `(256, 256)` at height `h_tx`, the
receiver is a **ground UE at `h_rx = 1.5 m`**, and the carrier is
`f = 7.125 GHz` (`λ = c / f ≈ 4.21 cm`, `k = 2π / λ`).

```
              (UAV, h_tx)
                  ●
                 /|\
                / | \
   d_los      /  |  \   d_ref
   (direct) /    |    \ (reflected)
           /     |     \
          /      |      \
  (Rx) ●---------·--------● (image Rx, −h_rx)
       h_rx     d_2D
       Ground (h=0)
```

- `d_2D(i, j)`: horizontal distance from Tx to pixel `(i, j)` in metres.
- `d_los = √(d_2D² + (h_tx − h_rx)²)`: direct 3D path length.
- `d_ref = √(d_2D² + (h_tx + h_rx)²)`: reflected 3D path length
  (image-Rx construction).

## 2. LoS branch: coherent two-ray over FSPL

### 2.1 Free-space anchor (Friis)

```
FSPL(d) = 32.45 + 20 · log10(d_los_km)   + 20 · log10(f_MHz)
          └── const ──┘   ↑                ↑
                         3D slant path    7125 MHz here
```

This is the standard Friis equation in dB, with the `d` in km and `f` in
MHz form that 3GPP documents reuse. See
[`prior_try78.py:63`](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/prior_try78.py#L63).

### 2.2 Two-ray coherent correction

The direct and ground-reflected rays add coherently at the receiver. In
dB the correction that gets *added* to FSPL is:

```
C_2ray(i, j; h_tx) =
    −20 · log10 | 1 + ρ(h_tx) · (d_los / d_ref) · exp( −j · [ k·(d_ref − d_los) + φ(h_tx) ] ) |
```

with three **fitted per-height-bin** scalars:

- `ρ(h_tx)` — effective ground reflection magnitude (≤ `ρ_max = 1.5`).
- `φ(h_tx)` — reflection phase offset (constant term, smoothed).
- `bias(h_tx)` — residual mean-zero offset absorbed into the total.

Grid search in code:
[`fit_two_ray_calibration`](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/prior_try78.py#L406) —
over `ρ ∈ [0, 1.5]` (step 0.05) × `φ ∈ [−π, π)` (48 equispaced bins), per
UAV-height bin, then Gaussian-smoothed in height (`σ = 10 m`).

### 2.3 Final LoS prediction

```
PL_LoS(i, j) = clip( FSPL(i, j) + C_2ray(i, j) + bias(h_tx),
                     PL_min = 20 dB, PL_max = 180 dB )
```

ρ, φ, bias are **linearly interpolated** in `h_tx` between adjacent
height bins.

### 2.4 Radial residual lookup (fallback)

A non-parametric alternative, kept as a safety net and inspection tool:

```
PL_radial(i, j) = FSPL(i, j) + r̄(h_tx_bin, radius_px[i, j])
```

where `r̄(h_bin, R)` is the **mean residual of FSPL** at radius `R`
pixels, computed from training LoS pixels (`fit_radial_calibration` in
[`prior_try78.py:234`](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/prior_try78.py#L234)).
This is the sanity-check that the two-ray fit is not worse than a
structure-free radial lookup.

## 3. NLoS branch: COST-231 + A2G envelope + regime calibration

Implemented in
[`evaluate_hybrid_try78.py`](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/evaluate_hybrid_try78.py).

### 3.1 Raw formula prior

Two components, combined by a max-envelope:

```
PL_COST231(d, h_tx) = 46.3 + 33.9·log10(f_MHz) − 13.82·log10(h_tx)
                     + (44.9 − 6.55·log10(h_tx)) · log10(d_km)
                     + C_urban       (= 3 dB for medium/large cities)
```

```
A2G_NLoS(θ) = λ₀ + bias + amp · exp( −(90° − θ) / τ )
```

where `θ` is the Tx elevation angle from the Rx pixel.

```
PL_NLoS_raw = max(PL_COST231, A2G_NLoS(θ))
```

(rationale: at high elevation the COST-231 urban term under-predicts the
loss — the A2G exponential envelope takes over as `θ → 90°`.)

### 3.2 LoS-ish blend for transitions

```
PL_LoS_raw = 0.7 · LoS_path   +   0.3 · min(LoS_path, A2G_LoS(θ))
```

Used when the pixel is nominally LoS but the geometry is borderline.

### 3.3 Regime-wise calibration

A ridge-regression correction on top of the raw formula:

```
PL_cal(i, j) = X(i, j) · β_regime
```

with design matrix `X` including:

- `prior` and `prior²`
- `log(1 + d_2D)`
- local density at `15×15` and `41×41`
- local mean building height at `15×15` and `41×41`
- local NLoS support at `41×41`
- shadow-σ proxy
- normalized elevation angle

Regime keying (same spirit as Try 79): topology class × LoS/NLoS ×
antenna-height bin, with the usual four-level fallback if the exact
regime has too few samples. Coefficients are stored in
[`final_calibrations/nlos_regime_calibration.json`](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/final_calibrations/nlos_regime_calibration.json).

## 4. Final prior consumed by Try 80

Per sample, `Try 80` builds (see
[`PRIOR_FORMULAS_TRY80.md`](../PRIOR_FORMULAS_TRY80.md)):

```
path_loss_prior = los_mask · PL_LoS  +  nlos_mask · PL_NLoS
```

Both components are clamped to `[20 dB, 180 dB]` and match the
`uint8 dB` convention of the CKM HDF5.

## 5. Why this formulation

| Choice | Reason |
|--------|--------|
| FSPL in dB with `20·log10(f_MHz)` | Standard Friis equation for free-space path loss. |
| Coherent two-ray with `ρ, φ` | The Earth-reflection pattern produces the characteristic oscillatory LoS behaviour seen in CKM LoS pixels (especially 80–478 m UAV heights). A simple PEC-reflector model is too idealized. |
| Height-bin interpolation | UAV A2G measurements show ρ and the effective bias vary smoothly with altitude. |
| COST-231 Hata for NLoS | Validated for urban macrocell path loss in the 1.5–2 GHz range; extrapolated via `20·log10(f/f_ref)` (standard 3GPP reference bracket for 7.125 GHz). |
| A2G exponential envelope | Matches the elevation dependence measured for UAV-to-ground NLoS in the literature. |
| Regime calibration | Removes the residual bias between formula and ray-traced ground truth on training cities, while keeping the physical structure. |

## 6. References

1. **H. T. Friis**, *A note on a simple transmission formula*, Proc. IRE, 1946 — FSPL.
2. **3GPP TR 38.901** §7.4, *Study on channel model for frequencies from 0.5 to 100 GHz* — Friis, COST-231 extrapolation, A2G elevation dependence.
3. **COST Action 231** Final Report (1999), *Digital mobile radio towards future generation systems* — COST-231–Hata urban macro model.
4. **Al-Hourani, Kandeepan, Lardner**, *Optimal LAP altitude for maximum coverage*, IEEE Wireless Commun. Letters, 2014 — A2G LoS-probability envelope and elevation dependence.
5. **A. A. Khuwaja, Y. Chen, N. Zhao, M.-S. Alouini, P. Dobbins**, *A Survey of Channel Modeling for UAV Communications*, IEEE Comm. Surveys & Tutorials, 2018 — two-ray / multi-ray structure for UAV A2G.
6. **T. S. Rappaport**, *Wireless Communications: Principles and Practice* — textbook treatment of two-ray ground-reflection model.
7. In-repo: [`TRY78_LOS_DOCUMENTATION.md`](/c:/TFG/TFGpractice/TFGSeventyEighthTry78/TRY78_LOS_DOCUMENTATION.md) — collects the two-ray source papers that motivated `ρ(h)` fitting.
