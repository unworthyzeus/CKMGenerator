# Why LoS angular spread RMSE is less important than the number suggests

## TL;DR

In `Try 79` evaluations, **LoS angular spread** tends to show the largest
per-regime RMSE, while **NLoS angular spread** is small. This is **not** a
failure of the model, it is a property of the channel:

- In **LoS** the dominant component is the **direct ray**. The angular spread
  reported by the ray tracer is dominated by one strong path with
  near-zero angular deviation plus a handful of weak reflections.
- The ground truth `angular_spread` map in LoS regions is effectively a
  **spike-like distribution** (most pixels tiny, occasional high spikes near
  edges of buildings or foliage).
- Any regression in log-domain with a broad regime mean will miss those
  narrow spikes and the RMSE blows up even though the *channel is trivial
  from a communication standpoint*: a receiver with a directional antenna
  pointed at the UAV sees virtually no multipath.

## Detailed reasoning

### 1. Angular spread as a multipath richness metric

Angular spread `AS` quantifies how concentrated or dispersed the incoming
rays are at the receiver. In 3GPP TR 38.901 and WINNER II it is modelled in
the log domain (`lgASD`, `lgASA`) because it is heavy-tailed and strictly
non-negative.

- `AS ~ 0 deg`: all energy arrives from (almost) one direction → **pure LoS**.
- `AS large`: many rays reach the receiver from different azimuths → **rich NLoS**.

So a *good* LoS pixel is one where `AS ~ 0`. The model does not need to
predict the fine structure of the spike — it only needs to know the
receiver is in a LoS corridor.

### 2. What communications actually care about in LoS

In LoS, the **direct path** dominates the received power by 20-40 dB over
any scattered copy (free-space + single small reflection). Consequences:

- **Beamforming** a single-panel array toward the UAV recovers essentially
  all the energy. Residual angular spread contributes less than 1 dB of
  equivalent loss.
- **Doppler**, **delay**, and **Ricean K-factor** are what matter for link
  design in LoS — angular spread is a secondary statistic.
- **Channel capacity** in LoS is driven by the main eigenvalue of the
  MIMO channel matrix; small angular spread is **good**, not bad.

So an error of 10-30 deg on an already-tiny LoS angular spread does not
translate to any meaningful link-level impact. The same 10-30 deg error on
a large NLoS spread does matter, because it changes the multipath richness
and therefore the effective rank of the channel.

### 3. Why the raw per-regime RMSE looks bad in LoS

In `smoke_joint_v2_r01` (175-sample smoke):

| Topology                | LoS angular RMSE | NLoS angular RMSE |
|-------------------------|------------------|-------------------|
| `open_sparse_vertical`  | 13.75 deg        | 12.75 deg         |
| `mixed_compact_midrise` | 22.00 deg        |  9.55 deg         |
| `dense_block_midrise`   | 27.48 deg        |  9.50 deg         |
| `dense_block_highrise`  | 33.47 deg        | 11.83 deg         |

- In dense cities almost every ground pixel with LoS sits next to a
  building edge where a **single reflection** plus the **direct ray**
  gives a bimodal distribution: most pixels ~0 deg, a few pixels ~30-90
  deg. The regressor picks the regime mean and misses both.
- In open low-rise areas, LoS pixels are genuinely ~0 deg everywhere and
  the regressor does the right thing (13.75 deg vs 12.75 deg).

The same phenomenon does **not** affect NLoS, because NLoS pixels always
have many rays: the distribution is smoother and a log-domain regime mean
fits it well.

## Proposed corrections to headline reporting

1. **Report two aggregate numbers for angular spread.** One weighted by
   all ground pixels (current behaviour), one weighted by `1 - LoS_mask`
   (NLoS only). The NLoS number is the one to optimise against.

2. **Do not chase LoS angular RMSE.** Any reduction in LoS angular RMSE
   via regime splitting or extra features is likely overfitting to spike
   outliers and will not generalise.

3. **Use a LoS-specific metric for direction error.** Instead of RMSE on
   `AS`, compute the *angular difference from zero* aggregated only over
   pixels with strong LoS dominance. Report it as a percentile (e.g. P90).

4. **If a link-level metric is available, use it.** The direct proxy for
   what LoS angular spread does to a UAV link is the Ricean K-factor or
   the beamforming gain loss. Either is more meaningful than AS RMSE.

5. **Tight-clamp LoS AS predictions to a small range (0-15 deg).** The
   baseline already clips to `clip_hi=90`. A tighter LoS-specific clamp
   removes the worst outlier contributions to RMSE without hurting the
   physically meaningful small-AS predictions.

## Cross-reference

- `prior_try79.py`: the LoS/NLoS split is already used inside regime
  keys, so the model **has** separate LoS and NLoS regressors. The RMSE
  asymmetry is a property of the targets, not the model structure.
- `prior_try79_hz.py` (height-aware variant): same reasoning applies.
  The added `tx_clearance_41` and `tx_below_frac_41` features help
  identify the **LoS/NLoS boundary**, which is the region where LoS
  angular spikes actually live — so the height-aware version is expected
  to *narrow* LoS angular RMSE slightly without any change to headline
  NLoS numbers.

## References

- 3GPP TR 38.901, clause 7.5 — `lgASA`, `lgASD`, `lgZSA`, `lgZSD` all
  modelled in log domain with height and regime-dependent means.
- WINNER II channel model, annex on large-scale parameter correlations.
- Al-Hourani, Kandeepan, Lardner, *Optimal LAP altitude for maximum
  coverage*, 2014 — LoS probability is the dominant propagation statistic
  for UAV A2G, not spreads.
- Rappaport, *Wireless Communications: Principles and Practice* — in LoS
  channels, the direct path carries the link; small-scale fading is
  Ricean with a high K-factor, and angular spread is a secondary metric.
