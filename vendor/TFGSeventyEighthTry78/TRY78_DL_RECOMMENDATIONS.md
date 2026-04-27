# Try 78 - DL Recommendations

## Goal

This note is about **how to improve Try 78 with deep learning**, starting from the current result:

- physics LoS model already works very well
- final hybrid result is about `1.9246 dB` overall
- LoS alone is already about `1.75 dB`

So the recommendation is **not** to replace the physics with a black box.

The recommendation is to use DL as a **small residual corrector** on top of the current physics.

## Current baseline to respect

Current strong prior:

- `LoS = coherent two-ray`
- `NLoS = regime-calibrated model`

For LoS specifically, the most important lesson from Try 78 is:

- the dataset is highly structured
- a large part of LoS is already explained by deterministic physics
- DL should focus on the **remaining residual**, not on rediscovering `FSPL` or two-ray from scratch

## Main recommendation

If we revisit Try 78 with DL, the best direction is:

1. keep the current two-ray model as the base prediction
2. train a model to predict only the residual
3. start with the `LoS` branch first

Target:

`residual_dl = GT_path_loss - path_loss_two_ray`

Final prediction:

`path_loss_pred = path_loss_two_ray + residual_dl_pred`

This is much safer than training a raw image-to-image network directly on path loss.

## Best DL architectures to try

### 1. Residual U-Net on top of the physics prior

This is the first thing I would try.

Inputs:

- topology / building-height map
- LoS mask
- two-ray prediction
- `FSPL`
- transmitter height as a scalar channel
- maybe direct geometry channels:
  - `d2d`
  - elevation angle
  - reflected-path phase difference

Output:

- residual correction in dB

Why:

- simple
- stable
- easy to compare against the current baseline
- most likely to beat the pure physics by a small but real margin

### 2. Two-head model: mean residual + uncertainty

Second option if the first works.

Outputs:

- residual mean
- residual scale / uncertainty

Why:

- some LoS bins are clearly easier than others
- uncertainty could help identify where the physics is already enough and where learned correction matters

This is more useful if the final paper wants confidence maps or calibrated reliability.

### 3. Height-conditioned residual expert model

If one model is not enough, split the residual correction by altitude regime.

Examples:

- low altitude
- mid altitude
- high altitude

This is better than a generic mixture of experts because Try 78 already showed that **height is one of the main drivers**.

## Inputs I would definitely include

For a DL residual model, the most useful input set is probably:

- `topology_map`
- `ground/building mask`
- `LoS mask`
- `FSPL`
- `two-ray prediction`
- `two-ray residual candidate = two_ray - FSPL`
- `d2d`
- elevation angle
- UAV height repeated as a full channel

If more physics channels are cheap to build, also include:

- reflected path length
- path difference `d_ref - d_los`
- fitted height-bin parameters:
  - `rho`
  - `phi`
  - `bias`

That gives the network a way to learn **when the fitted physics is locally wrong**.

## Training strategy I would use

### Stage 1. Residual training on the LoS branch

Train only on valid LoS pixels.

Loss:

- main: `L1` or Huber on residual dB
- metric: LoS RMSE in final path loss

Why:

- cleaner problem
- directly tied to the strongest Try 78 insight
- less chance of the network being distracted by NLoS complexity

### Stage 2. Small-capacity model first

Do not start with a huge model.

Start with:

- small U-Net
- moderate channels
- no fancy attention at first

Reason:

- the residual is probably low-amplitude and structured
- overcapacity will make it easier to overfit city patterns

### Stage 3. Freeze physics, learn only correction

The two-ray part should stay fixed.

Do not let the network modify the whole prediction implicitly.

Reason:

- we already know the deterministic part is good
- this makes the experiment interpretable

## Losses I would try

Preferred order:

1. Huber on residual dB
2. `L1` on residual dB
3. mixed loss:
   - Huber on residual
   - small smoothness penalty on residual map

I would avoid fancy perceptual losses here.

The output is physical and metric-driven, not visual.

## How to adapt the Try 76 distribution-first idea

A very good extension would be to adapt the **distribution-first** logic from Try 76 instead of predicting only a single residual value.

For Try 78, that adaptation should look like this:

- keep the current physics prior
- let DL predict a **distribution over the residual**
- derive the final point prediction from that distribution

That means replacing:

- single residual mean

with something like:

- mixture weights
- mixture means
- mixture scales

for the residual in dB.

So the final prediction becomes:

- prior from physics
- plus a learned residual distribution

This is especially attractive because:

- LoS residual is usually small, but not perfectly Gaussian
- there are rare bins and local zones where the physics misses more strongly
- a mixture model can represent that better than a single deterministic head

## Distribution-first for LoS

For `LoS`, I would adapt Try 76 conservatively.

The target would be:

`residual_los = GT_path_loss - path_loss_two_ray`

The network would output a small mixture model for that residual, for example:

- `K=2` or `K=3` components
- per-pixel mixture logits
- per-pixel residual means
- per-pixel residual scales

Then:

- training loss = negative log-likelihood on residual dB
- point prediction = mixture mean or MAP component mean

Why this is useful in `LoS`:

- most pixels are close to the physics prior
- but some pixels have structured errors that are not well captured by a single mean
- a small mixture is a clean way to handle asymmetric or heavy-tailed residuals

My recommendation here is:

- keep it small
- `K=2` is probably enough for the first test
- do not jump to a huge `GMM` head immediately

## Distribution-first for NLoS

For `NLoS`, the Try 76 adaptation may be even more valuable.

Reason:

- `NLoS` is much less deterministic
- multiple propagation mechanisms can coexist
- the target distribution is much more likely to be multimodal or heteroscedastic

So for `NLoS`, a distribution-first head is not just a refinement. It may be the right formulation.

The cleanest first version would be:

- prior = current calibrated NLoS estimate
- target = `GT - prior`
- network predicts a residual mixture distribution

That gives:

- stable baseline anchoring
- richer uncertainty structure than plain regression
- a way to represent multiple plausible attenuation behaviors

## Best practical distribution parameterization

If we adapt Try 76 ideas, I would use this order:

### Option 1. Small Gaussian mixture on residual dB

Outputs:

- mixture logits
- residual means
- residual log-variances

This is the closest to Try 76 in spirit.

### Option 2. Spike-plus-continuous residual model

This may be especially useful for `LoS`, and maybe also for sparse `NLoS` corrections.

Idea:

- one branch predicts probability that residual is near zero
- another branch predicts continuous residual if not near zero

Why:

- in `LoS`, a large fraction of pixels are already very well explained by physics
- so the residual often behaves like:
  - a near-zero spike
  - plus a continuous correction tail

This is actually a very natural match to the Try 78 setup.

### Option 3. Quantized distribution / ordinal bins

If full `GMM` training is unstable, a simpler backup is:

- discretize residual dB into bins
- predict per-bin probabilities
- recover mean / median residual from that distribution

This is less elegant than a mixture head, but often easier to stabilize.

## Recommended adaptation order

If I were porting Try 76 ideas into Try 78, I would do it in this order:

1. deterministic residual U-Net
2. small distribution head on the residual
3. only then expert splits or more complex mixtures

That way, we first verify that the residual itself is learnable, and only then ask the model to learn uncertainty or multimodality.

## What I would actually implement first

If the goal is one realistic next experiment, I would do:

### LoS distribution-first experiment

- prior: two-ray
- target: residual in dB
- model: small U-Net
- head: `K=2` Gaussian mixture on residual
- inference: use mixture mean

### NLoS distribution-first experiment

- prior: current NLoS calibrated prior
- target: residual in dB
- model: geometry-aware U-Net
- head: `K=3` Gaussian mixture on residual
- inference: use mixture mean

That is the cleanest “Try 76 style” adaptation without throwing away the strong parts of Try 78.

## What I would not do

I would avoid these first:

- training raw DL from image to full path loss
- mixing LoS and NLoS in the first DL version
- forcing a giant MoE architecture immediately
- using only topology image without physics channels
- predicting path loss in linear power units

Those are much more likely to throw away the main gain from Try 78.

## Most promising concrete experiments

### Experiment A

Residual U-Net for the LoS branch.

Inputs:

- topology
- LoS mask
- `FSPL`
- two-ray prediction
- height channel

Target:

- residual to two-ray

This is the cleanest next experiment.

### Experiment B

Same as A, but predict:

- residual mean
- residual uncertainty

Useful if A already improves RMSE.

### Experiment C

Residual expert by height regime.

This only makes sense if A shows that different altitude bands still have different error behavior.

## What comes from papers

Mostly literature-aligned:

- using physics-guided neural correction instead of pure black-box prediction
- using geometry-aware channels
- using uncertainty heads when the residual difficulty is non-uniform

These ideas are common in hybrid physics + ML modeling.

## What would be our likely innovation

The more original part would be:

- discovering that CKM LoS is already strongly explained by a two-ray-like field
- using the fitted two-ray map itself as the main prior
- training DL only on the **small LoS residual left after that**

That is a much more specific proposal than generic physics-informed DL.

## Final recommendation

If we do a DL version of Try 78, I would do this exact order:

1. residual U-Net over the current two-ray model for the LoS branch
2. verify whether LoS RMSE beats `1.75 dB`
3. only then decide whether to add uncertainty or height experts

In short:

- do **not** replace the current physics
- let DL correct only what the physics still misses
- keep the first experiment narrow, interpretable, and focused on the LoS branch

## NLoS recommendations

For `NLoS`, I would not follow exactly the same recipe as `LoS`.

Reason:

- in `LoS`, the deterministic physics already explains a lot
- in `NLoS`, the current strong part is much more heuristic and regime-based
- so for `NLoS`, DL has more room to learn genuinely useful structure

In other words:

- `LoS`: DL should be a small correction
- `NLoS`: DL can be a larger part of the model

## Main recommendation for NLoS

If we add DL for `NLoS`, I would do this:

1. keep the current calibrated `NLoS` prior as a baseline input
2. train a DL model only on `NLoS` pixels
3. predict either:
   - `NLoS residual over the calibrated prior`, or
   - directly `NLoS path loss`

My preferred order is:

1. residual over the current `NLoS` prior
2. only if that saturates, try direct `NLoS` prediction

Target for the first version:

`residual_nlos_dl = GT_path_loss - path_loss_nlos_prior`

Final:

`path_loss_nlos_pred = path_loss_nlos_prior + residual_nlos_dl_pred`

## Best DL architectures for NLoS

### 1. Geometry-aware U-Net residual model

This is the first NLoS model I would try.

Inputs:

- topology / height map
- ground/building mask
- LoS mask
- calibrated NLoS prior
- `FSPL`
- `d2d`
- UAV height channel

Output:

- residual over the current NLoS prior

Why:

- easy to compare against the current hybrid
- still anchored to the existing good heuristic
- lets DL focus on diffraction / blockage / shadow complexity

### 2. Expert model by topology class

If one global NLoS network is too blunt, split by topology.

Examples:

- open sparse
- mixed compact
- dense block

This is a very natural move because the old Try 78 NLoS calibration already showed strong topology dependence.

### 3. Two-stage NLoS model

This may be the most powerful version later.

Stage 1:

- classify NLoS regime or severity

Stage 2:

- regress path loss inside that regime

This is basically the DL analogue of the old regime-calibration idea.

## Inputs I would include for NLoS

The most important NLoS channels are not exactly the same as in LoS.

I would include:

- `topology_map`
- building mask
- `LoS mask`
- current NLoS prior
- `FSPL`
- `d2d`
- UAV height channel

And I would strongly consider extra geometric channels like:

- local building density
- local mean building height
- distance to nearest building / blocker
- directional blockage maps around the transmitter
- local accumulated obstruction depth along Tx-Rx direction

These are especially relevant for `NLoS`, because here the map structure is probably doing real causal work.

## Training strategy for NLoS

### Stage 1. Train on NLoS only

Mask the loss to valid `NLoS` pixels only.

Why:

- much cleaner learning signal
- avoids LoS dominating gradients
- NLoS is sparse and physically different enough to justify its own model

### Stage 2. Start as residual learning

Even for `NLoS`, I would still start with residual learning over the current calibrated prior.

Reason:

- safer optimization
- easier ablation
- easier to prove gain against the non-DL baseline

### Stage 3. Move to experts only if needed

If one residual U-Net is not enough, then split by:

- topology class
- altitude regime
- maybe dense vs sparse morphology

But I would not begin with a giant MoE unless the simple residual model already shows clear topology-specific failure modes.

## Losses I would try for NLoS

Preferred order:

1. Huber on NLoS residual dB
2. `L1` on NLoS residual dB
3. mixed loss:
   - Huber on residual
   - edge-aware or local-gradient penalty, very small

Why a tiny structural penalty may help here:

- NLoS maps often have sharp transitions caused by buildings and occlusion
- but you do not want to oversmooth those edges

So if used, that regularizer should be weak.

## How Try 76 distribution-first should influence NLoS

For `NLoS`, I would take the distribution-first idea more seriously than for `LoS`.

If I had to choose one advanced version for NLoS, it would be:

- geometry-aware residual network
- distribution head instead of a single regression head

Why:

- NLoS uncertainty is real, not just noise
- different blockage situations can produce different plausible path-loss corrections
- a single mean can blur together physically different cases

So the most Try-76-like adaptation for NLoS is:

- residual `GMM` over the current NLoS prior
- maybe later with topology experts

This is probably the most principled DL path for NLoS inside Try 78.

## What I would not do for NLoS

I would avoid these first:

- mixing LoS and NLoS pixels in one first model
- predicting only from the topology image without any prior channels
- making the first model huge and highly specialized
- switching immediately to a diffusion-style or generative formulation

That would make debugging much harder.

## Most promising NLoS experiments

### Experiment D

NLoS-only residual U-Net over the current NLoS calibrated prior.

Inputs:

- topology
- building mask
- LoS mask
- current NLoS prior
- `FSPL`
- height

This is the cleanest first NLoS DL experiment.

### Experiment E

Same as D, but with extra geometric channels:

- local density
- local mean height
- obstruction depth

This is probably the strongest practical next step if D already helps.

### Experiment F

Topology-specialized NLoS residual experts.

This only makes sense if the error analysis shows that different topology families fail in clearly different ways.

## Likely combined DL roadmap

If the objective is to improve the whole Try 78 with DL, I would do it in this order:

1. `LoS` residual U-Net over two-ray
2. `LoS` residual distribution head if deterministic residual works
3. measure LoS gain
4. `NLoS` residual U-Net over the calibrated NLoS prior
5. `NLoS` distribution-first head if deterministic residual works
6. measure overall gain
7. only then consider expert splits

That keeps the work incremental and interpretable.

## Final practical view

If I had to bet on where DL helps more:

- `LoS`: smaller but cleaner gain
- `NLoS`: potentially larger gain, but harder and noisier

So the best full strategy is probably:

- use physics-first DL for `LoS`
- use geometry-first residual DL for `NLoS`
- and, where useful, replace single-value residual heads with **Try 76 style distribution heads**
