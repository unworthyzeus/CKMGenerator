# Try 79 - Pure `numpy` non-DL spread baseline

`Try 79` is a transparent baseline for `delay_spread` and `angular_spread` that avoids deep learning entirely.

It is intentionally modeled after the workflow of old `Try 78`:

1. build a physically sensible prior from observable geometry
2. extract multiscale map features directly from the HDF5 tensors
3. fit a small regime-wise calibration model
4. optionally accelerate the multiscale box-filter feature extraction with `torch-directml`
4. evaluate under city holdout

The difference is that `Try 79` targets the spread maps from `Try 58`, `Try 59`, and `Try 77`, not path loss.

## Why this model looks like this

The design choices are tied to the literature rather than to neural-network convenience:

- `log1p(spread)` regression:
  3GPP TR 38.901 generates delay spread and angular spreads from log-domain random variables (`lgDS`, `lgASD`, `lgASA`, etc.), so a log-domain baseline is a natural first choice for non-negative heavy-tailed spreads.
- elevation angle and distance features:
  UAV air-to-ground measurements show that channel statistics vary with flight height and horizontal distance.
- explicit LoS / NLoS handling:
  both delay and angular dispersion become smaller when LoS dominates, but urban and low-height cases retain richer multipath.
- multiscale obstruction features:
  local building density, height, and NLoS support are simple map-derived proxies for the richness of the multipath environment.

## Model summary

For each metric (`delay_spread`, `angular_spread`), `prior_try79.py` does the following:

1. Compute a raw log-domain prior from:
   - `log(1 + d_2D)`
   - normalized radius from the transmitter
   - azimuth harmonics around the transmitter
   - normalized elevation angle
   - topology class
   - local building density
   - local mean building height
   - local NLoS support
2. Build a feature matrix with pure `numpy` box filters at `15 x 15` and `41 x 41`.
3. Add joint-LSP features inspired by 3GPP / WINNER:
   - the raw prior of the other spread metric
   - a simple LoS concentration proxy
   - a diffuse-multipath proxy
4. Fit ridge regressors in `log1p(target)` space for:
   - `metric x topology_class x LoS/NLoS x antenna_bin`
5. Add fallback regimes for sparse cases:
   - exact regime
   - same topology + LoS/NLoS + all heights
   - same topology + all LoS states + all heights
   - global + LoS/NLoS
   - fully global
6. Evaluate both:
   - the raw prior
   - the calibrated prediction

The split matches the `Try 77` family semantics:

- city holdout
- train / val / test
- default `15% / 15%` held out

## Dependencies

Only:

- `numpy`
- optional: `torch` + `torch-directml` for `--device dml`
- `h5py`

No deep learning. By default the script stays on CPU and pure `numpy`; if `torch-directml`
is available, `--device dml` can accelerate the box-filter feature extraction on AMD GPUs.

## Usage

Full run:

```powershell
python prior_try79.py `
  --hdf5 c:/TFG/TFGpractice/Datasets/CKM_Dataset_270326.h5 `
  --out-dir prior_out
```

Quick smoke test:

```powershell
python prior_try79.py --max-samples 24 --pixel-subsample 0.005
```

Evaluate only from a saved calibration:

```powershell
python prior_try79.py `
  --calibration-json prior_out/calibration.json `
  --skip-fit `
  --eval-split val
```

## Outputs

- `calibration.json`
- `eval_summary_val.json` or `eval_summary_test.json`
- `progress.json`
- `progress.out`

Map inspection panels are also saved when we analyze the dataset manually, for example in:

- [map_inspection](C:/TFG/TFGpractice/TFGSeventyNinthTry79/map_inspection)

Each summary contains:

- aggregate pixel-weighted RMSE / MAE
- LoS / NLoS breakdown
- per-topology breakdown
- per-sample metrics

## Interpretation

This try is not expected to beat a well-trained distribution-first model like `Try 77`.

The purpose is different:

- establish how much of the spread prediction problem can be explained with simple geometry and morphology
- provide a reproducible non-DL baseline
- keep a thesis-friendly model whose behavior can be inspected coefficient by coefficient

## What Comes From Papers

Main ideas imported from standards / papers:

- `3GPP TR 38.901` models `DS`, `ASD`, and `ASA` in the log domain and treats them as **joint large-scale parameters**.
- `WINNER II` also models delay and angle spreads as **correlated large-scale parameters** with joint evolution.
- UAV A2G measurement papers support dependence on **distance, height, LoS dominance, and angular concentration**.

Those sources justify:

- `log1p` regression for spreads
- explicit distance / elevation features
- explicit LoS / NLoS treatment
- cross-metric features between `delay_spread` and `angular_spread`
- a LoS concentration proxy and a diffuse-multipath proxy

## What Seems To Be Our Own Dataset Insight

What appears more specific to our work on CKM:

- Inspecting the actual spread maps and noticing two recurring families:
  - almost radial / quantized patterns in easy open cases
  - strongly directional sector / streak patterns in hard urban cases
- Adding explicit transmitter-centered geometry features:
  - normalized radius
  - azimuth harmonics
- Using very simple map-derived morphology plus regime-wise ridge calibration instead of a full stochastic channel simulator
- Keeping the whole model interpretable and HDF5-native rather than rebuilding the full generative model from the papers

## References

1. 3GPP TR 38.901, "Study on channel model for frequencies from 0.5 to 100 GHz".
   Link: https://www.3gpp.org/ftp/Meetings_3GPP_SYNC/SA3/Inbox/Drafts/tr_138901v140200p.pdf
   Why it matters here: DS and angular spreads are modeled in the log domain (`lgDS`, `lgASD`, `lgASA`, `lgZSA`, `lgZSD`), which motivates `log1p` calibration.

2. X. Cai, J. Rodriguez-Pineiro, X. Yin, B. Ai, G. F. Pedersen, A. Perez Yuste,
   "An Empirical Air-to-Ground Channel Model Based on Passive Measurements in LTE", 2019.
   Link: https://arxiv.org/abs/1901.07930
   Why it matters here: the extracted UAV A2G stochastic model explicitly studies delay spread versus height and horizontal distance.

3. T. Izydorczyk, F. M. L. Tavares, G. Berardinelli, M. Bucur, P. Mogensen,
   "Angular Distribution of Cellular Signals for UAVs in Urban and Rural Scenarios", EuCAP 2019.
   Link: https://vbn.aau.dk/ws/files/306998488/Angular_Distribution_of_Cellular_Signals_for_UAVs_in_Urban_and_Rural_Scenarios.pdf
   Why it matters here: measured angular spread shrinks as height increases and LoS becomes more dominant, but urban scenarios still retain relevant multipath.

4. W. Khawaja, O. Ozdemir, F. Erden, I. Guvenc, D. Matolak,
   "Ultra-Wideband Air-to-Ground Propagation Channel Characterization in an Open Area", 2019.
   Link: https://arxiv.org/abs/1906.04013
   Why it matters here: the Saleh-Valenzuela family is reported as a good fit for UAV A2G wideband delay structure, reinforcing the idea that spreads are sparse/heavy-tailed rather than simple Gaussian fields.

5. W. Khawaja, I. Guvenc, D. Matolak, U.-C. Fiebig, N. Schneckenberger,
   "A Survey of Air-to-Ground Propagation Channel Modeling for Unmanned Aerial Vehicles", 2018.
   Link: https://arxiv.org/abs/1801.01656
   Why it matters here: survey support for separating large-scale geometry effects from small-scale fading/channel-dispersion statistics in UAV A2G modeling.
