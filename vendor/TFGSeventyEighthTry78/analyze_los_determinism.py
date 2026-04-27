"""Try 78 — LoS determinism analysis (no deep learning).

Hypothesis: LoS path loss on CKM is (nearly) a deterministic function of
geometry — Tx height, 2D distance from map center, and whatever physics
piece we are currently missing (ground reflection / Fresnel clearance /
knife-edge from the nearest building). If this is true, Try 78 can replace
LoS prediction with a closed-form formula (+ small lookup) instead of a CNN.

This script runs three exploratory phases:

  PHASE 1 — Pairing test.
    Bucket samples by UAV height (rounded to `--height-bin-m`). For each
    bucket with >= 2 samples, sample random pairs; intersect their LoS
    masks (AND with ground on both maps); compare `path_loss` pixel-by-pixel.
    If LoS is purely geometric, |ΔPL| at the intersection should be ~ the
    quantisation floor (uint8 = 1 dB). A wide distribution means there is
    sample-specific physics (buildings near the ray, multipath).

  PHASE 2 — Analytic baselines on LoS pixels only.
    Compute FSPL(d_3D, f) and a coherent two-ray model at every LoS ground
    pixel. Residual = observed_path_loss - baseline. Report per-sample RMSE
    and aggregate residual statistics vs d_3D and vs h_tx.

  PHASE 3 — Diagnostic plots.
    - Histogram of |ΔPL| from pairing test, overall and per height bucket.
    - Scatter of observed PL vs d_3D, coloured by h_tx.
    - Residual (PL - FSPL) vs d_3D and vs h_tx.
    - Per-sample RMSE vs h_tx.

Assumptions (from repo CLAUDE.md and Try 67 data utils):
    - Tx is at the center of the 513x513 map (pixel (256, 256)).
    - Rx is a fixed 1.5 m ground UE at every ground pixel (topology == 0).
    - Frequency is 7.125 GHz (from Try 67 YAMLs).
    - Resolution is 1 m / pixel.
    - path_loss < 20 dB is a sentinel / invalid marker (see Try 76).

Usage:
    python analyze_los_determinism.py \
        --hdf5 c:/TFG/TFGpractice/Datasets/CKM_Dataset_270326.h5 \
        --out-dir c:/TFG/TFGpractice/TFGSeventyEighthTry78/analysis_out

Cheap dry-run (a few samples, no plots):
    python analyze_los_determinism.py --max-samples 20 --skip-plots
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Constants matching the CKM dataset / Try 67 YAMLs
# ---------------------------------------------------------------------------
IMG_SIZE: int = 513
TX_ROW: int = 256
TX_COL: int = 256
METERS_PER_PIXEL: float = 1.0
FREQ_GHZ: float = 7.125
RX_HEIGHT_M: float = 1.5
PATH_LOSS_MIN_DB: float = 20.0            # below this, target is sentinel noise
EPS_R_GROUND: float = 5.0                 # relative permittivity ~ dry soil/asphalt
DEFAULT_HDF5 = Path("c:/TFG/TFGpractice/Datasets/CKM_Dataset_270326.h5")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def distance_maps_m() -> Tuple[np.ndarray, np.ndarray]:
    """Return (d2d, d3d_for_rx=1.5m -- filled later) as float32 in meters.

    d2d[i,j] = METERS_PER_PIXEL * sqrt((i - TX_ROW)^2 + (j - TX_COL)^2)
    """
    ii, jj = np.indices((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    dy = (ii - TX_ROW) * METERS_PER_PIXEL
    dx = (jj - TX_COL) * METERS_PER_PIXEL
    d2d = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    # d3d is h_tx-dependent; compute per-sample in caller. Return d2d + sentinel.
    return d2d, np.zeros_like(d2d)


def d3d_for_height(d2d: np.ndarray, h_tx: float) -> np.ndarray:
    dz = float(h_tx) - RX_HEIGHT_M
    return np.sqrt(d2d * d2d + dz * dz).astype(np.float32)


def fspl_db(d3d_m: np.ndarray, freq_ghz: float = FREQ_GHZ) -> np.ndarray:
    """Free-space path loss in dB. Uses the standard 32.45 + 20log10(d_km) + 20log10(f_MHz)."""
    d_km = np.maximum(d3d_m, 1.0) / 1000.0
    return 32.45 + 20.0 * np.log10(d_km) + 20.0 * math.log10(freq_ghz * 1000.0)


def incoherent_two_ray_db(
    d2d: np.ndarray,
    h_tx: float,
    h_rx: float = RX_HEIGHT_M,
    freq_ghz: float = FREQ_GHZ,
) -> np.ndarray:
    """Plane-earth / incoherent two-ray with FSPL crossover.

    This is the baseline used by the older prior line (Try 41-47, see
    ``TFGSixtySeventhTry67/data_utils.py`` mode ``two_ray_ground`` and
    ``FORMULA_PRIOR_CALIBRATION_SYSTEM.md``):

        d_c      = 4 * pi * h_tx * h_rx / wavelength         (breakpoint)
        PL(d3d)  = FSPL(d3d)                 if d3d <= d_c
        PL(d3d)  = 40*log10(d3d) - 20*log10(h_tx) - 20*log10(h_rx)   else

    Smoother than the coherent variant (no destructive-interference nulls),
    and historically gave a better LoS prior RMSE on CKM once regime
    calibration was fit on top.
    """
    wavelength = 0.299_792_458 / max(freq_ghz, 0.1)
    h_tx_c = max(float(h_tx), 1.0)
    h_rx_c = max(float(h_rx), 0.5)
    d_c = max(4.0 * math.pi * h_tx_c * h_rx_c / wavelength, 1.0)
    d3d = np.sqrt(d2d * d2d + (h_tx_c - h_rx_c) ** 2).astype(np.float32)
    d3d = np.maximum(d3d, 1.0)
    fspl = fspl_db(d3d, freq_ghz)
    plane_earth = 40.0 * np.log10(d3d) - 20.0 * math.log10(h_tx_c) - 20.0 * math.log10(h_rx_c)
    return np.where(d3d <= d_c, fspl, plane_earth).astype(np.float32)


def two_ray_pathloss_db(
    d2d: np.ndarray,
    h_tx: float,
    h_rx: float = RX_HEIGHT_M,
    freq_ghz: float = FREQ_GHZ,
    eps_r: float = EPS_R_GROUND,
) -> np.ndarray:
    """Coherent two-ray ground-reflection path loss in dB.

    PL = -20 log10 | (lambda / 4 pi) * ( e^{-j k d_los}/d_los
                                          + Gamma * e^{-j k d_ref}/d_ref ) |
    This collapses to FSPL(d_los) when the ground reflection adds in phase and
    rapidly diverges otherwise; captures the classical distance-breakpoint
    behaviour without any tuning.
    """
    wavelength = 0.299_792_458 / max(freq_ghz, 0.1)
    k = 2.0 * math.pi / wavelength
    d_los = np.sqrt(d2d * d2d + (h_tx - h_rx) ** 2).astype(np.float64)
    d_ref = np.sqrt(d2d * d2d + (h_tx + h_rx) ** 2).astype(np.float64)
    d_los = np.maximum(d_los, 1.0)
    d_ref = np.maximum(d_ref, 1.0)

    # Fresnel reflection coefficient (average of TE/TM, smooth surface).
    cos_i = np.clip((h_tx + h_rx) / d_ref, 1e-4, 1.0)
    sin_i_sq = np.clip(1.0 - cos_i * cos_i, 0.0, 1.0)
    root = np.sqrt(np.clip(eps_r - sin_i_sq, 1e-4, None))
    gamma_h = (cos_i - root) / (cos_i + root + 1e-8)
    gamma_v = (eps_r * cos_i - root) / (eps_r * cos_i + root + 1e-8)
    gamma = 0.5 * (gamma_h + gamma_v)
    gamma = np.clip(gamma, -0.95, 0.95)

    phase_los = np.exp(-1j * k * d_los)
    phase_ref = np.exp(-1j * k * d_ref)
    field = (wavelength / (4.0 * math.pi)) * (phase_los / d_los + gamma * phase_ref / d_ref)
    amp_sq = np.maximum(np.abs(field) ** 2, 1e-30)
    pl_db = -10.0 * np.log10(amp_sq)
    return pl_db.astype(np.float32)


# ---------------------------------------------------------------------------
# HDF5 enumeration
# ---------------------------------------------------------------------------

@dataclass
class SampleRef:
    city: str
    sample: str
    uav_height_m: float


def enumerate_samples(hdf5_path: Path, max_samples: Optional[int] = None) -> List[SampleRef]:
    refs: List[SampleRef] = []
    with h5py.File(str(hdf5_path), "r") as handle:
        cities = sorted(handle.keys())
        for city in cities:
            for sample in sorted(handle[city].keys()):
                uav = float(np.asarray(handle[city][sample]["uav_height"][...]).reshape(-1)[0])
                refs.append(SampleRef(city=city, sample=sample, uav_height_m=uav))
                if max_samples is not None and len(refs) >= max_samples:
                    return refs
    return refs


def load_sample(handle: h5py.File, ref: SampleRef) -> Dict[str, np.ndarray]:
    grp = handle[ref.city][ref.sample]
    topology = np.asarray(grp["topology_map"][...], dtype=np.float32)
    los = np.asarray(grp["los_mask"][...], dtype=np.uint8)
    pl = np.asarray(grp["path_loss"][...], dtype=np.float32)
    return {
        "topology": topology,
        "los_mask": los,
        "path_loss": pl,
        "ground": (topology == 0.0),
    }


# ---------------------------------------------------------------------------
# Phase 1: pairing test
# ---------------------------------------------------------------------------

@dataclass
class PairingBucketStats:
    height_bin_m: float
    n_samples: int
    n_pairs: int
    n_pixels: int
    mean_abs_delta_db: float
    std_abs_delta_db: float
    p50_abs_delta_db: float
    p95_abs_delta_db: float
    p99_abs_delta_db: float


def run_pairing_test(
    hdf5_path: Path,
    refs: Sequence[SampleRef],
    height_bin_m: float,
    max_pairs_per_bucket: int,
    max_pixels_per_pair: int,
    rng: random.Random,
) -> Tuple[List[PairingBucketStats], np.ndarray]:
    """Returns per-bucket stats and a flat array of sampled |ΔPL| values."""
    buckets: Dict[float, List[SampleRef]] = {}
    for ref in refs:
        key = round(ref.uav_height_m / height_bin_m) * height_bin_m
        buckets.setdefault(float(key), []).append(ref)

    all_deltas: List[np.ndarray] = []
    stats: List[PairingBucketStats] = []

    with h5py.File(str(hdf5_path), "r") as handle:
        for h_bin in sorted(buckets.keys()):
            bucket = buckets[h_bin]
            if len(bucket) < 2:
                continue
            pairs = []
            # unique unordered pairs, capped
            candidates = list(range(len(bucket)))
            rng.shuffle(candidates)
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    pairs.append((candidates[i], candidates[j]))
                    if len(pairs) >= max_pairs_per_bucket:
                        break
                if len(pairs) >= max_pairs_per_bucket:
                    break

            bucket_deltas: List[np.ndarray] = []
            for ai, bi in pairs:
                a = load_sample(handle, bucket[ai])
                b = load_sample(handle, bucket[bi])
                valid_a = (a["path_loss"] >= PATH_LOSS_MIN_DB) & a["ground"] & (a["los_mask"] > 0)
                valid_b = (b["path_loss"] >= PATH_LOSS_MIN_DB) & b["ground"] & (b["los_mask"] > 0)
                inter = valid_a & valid_b
                n_inter = int(inter.sum())
                if n_inter == 0:
                    continue
                delta = np.abs(a["path_loss"][inter] - b["path_loss"][inter]).astype(np.float32)
                if delta.size > max_pixels_per_pair:
                    idx = rng.sample(range(delta.size), max_pixels_per_pair)
                    delta = delta[np.asarray(idx)]
                bucket_deltas.append(delta)

            if not bucket_deltas:
                continue
            flat = np.concatenate(bucket_deltas)
            all_deltas.append(flat)
            stats.append(
                PairingBucketStats(
                    height_bin_m=float(h_bin),
                    n_samples=len(bucket),
                    n_pairs=len(pairs),
                    n_pixels=int(flat.size),
                    mean_abs_delta_db=float(flat.mean()),
                    std_abs_delta_db=float(flat.std()),
                    p50_abs_delta_db=float(np.percentile(flat, 50)),
                    p95_abs_delta_db=float(np.percentile(flat, 95)),
                    p99_abs_delta_db=float(np.percentile(flat, 99)),
                )
            )

    flat_all = np.concatenate(all_deltas) if all_deltas else np.zeros(0, dtype=np.float32)
    return stats, flat_all


# ---------------------------------------------------------------------------
# Phase 2: analytic baselines
# ---------------------------------------------------------------------------

@dataclass
class SampleResidualStats:
    city: str
    sample: str
    split: str                       # "train" or "eval"
    uav_height_m: float
    n_los_pixels: int
    pl_mean_db: float
    pl_std_db: float
    # raw (no calibration)
    fspl_rmse_db: float
    fspl_bias_db: float
    two_ray_rmse_db: float
    two_ray_bias_db: float
    incoherent_two_ray_rmse_db: float
    incoherent_two_ray_bias_db: float
    # global-affine calibration (a, b fit ONCE on train pixels only, applied here)
    fspl_calib_rmse_db: float
    incoherent_two_ray_calib_rmse_db: float


def _ols_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y ≈ a*x + b by least squares. Returns (a, b). No residual leak."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    dx = x - x_mean
    var_x = float(np.sum(dx * dx))
    if var_x < 1e-9:
        return 0.0, float(y_mean)
    a = float(np.sum(dx * (y - y_mean)) / var_x)
    b = float(y_mean - a * x_mean)
    return a, b


def split_city_holdout_78(
    refs: Sequence[SampleRef],
    eval_ratio: float = 0.30,
    split_seed: int = 42,
) -> Tuple[List[SampleRef], List[SampleRef]]:
    """2-way city-holdout split. Whole cities go to one side — no leakage."""
    by_city: Dict[str, List[SampleRef]] = {}
    for r in refs:
        by_city.setdefault(r.city, []).append(r)
    cities = list(by_city.keys())
    rng = random.Random(split_seed)
    rng.shuffle(cities)
    total = len(refs)
    target_eval = int(round(total * eval_ratio))
    eval_refs: List[SampleRef] = []
    fit_refs: List[SampleRef] = []
    # Greedy fill eval, always leave at least 1 city for fit.
    for idx, c in enumerate(cities):
        city_samples = by_city[c]
        remaining_cities = len(cities) - idx
        if (
            len(eval_refs) < target_eval
            and remaining_cities > 1                # leave >=1 city for fit
            and len(eval_refs) + len(city_samples) <= max(target_eval * 2, len(city_samples))
        ):
            eval_refs.extend(city_samples)
        else:
            fit_refs.extend(city_samples)
    # Safety: if either side empty (happens with --max-samples capping to one city),
    # fall back to per-sample split so the script still runs.
    if not fit_refs or not eval_refs:
        shuffled = list(refs)
        random.Random(split_seed).shuffle(shuffled)
        cut = max(1, int(round(len(shuffled) * (1.0 - eval_ratio))))
        fit_refs = shuffled[:cut]
        eval_refs = shuffled[cut:]
    return fit_refs, eval_refs


FEATURE_NAMES: Tuple[str, ...] = (
    "h_tx",
    "log_h_tx",
    "h_tx_sq",
    "mean_building_height",
    "std_building_height",
    "building_density",
    "mean_bh_ring100",
    "mean_bh_ring250",
    "frac_bh_above_tx",
    "los_fraction",
    "mean_bh_over_los_pixels",
    "bias",
)


def compute_sample_features(topology: np.ndarray, los_mask: np.ndarray, h_tx: float) -> np.ndarray:
    """Topology-only features (no path_loss).

    Everything here comes from inputs available at inference time — topology
    heights, LoS mask, UAV height. Shape: (len(FEATURE_NAMES),).
    """
    ground = (topology == 0.0)
    bh = topology[~ground] if (~ground).any() else np.array([0.0], dtype=np.float32)
    density = float((~ground).mean())

    ii, jj = np.indices(topology.shape, dtype=np.float32)
    d2d_px = np.sqrt((ii - TX_ROW) ** 2 + (jj - TX_COL) ** 2)
    ring100 = (~ground) & (d2d_px <= 100.0)
    ring250 = (~ground) & (d2d_px <= 250.0)
    mean_bh_ring100 = float(topology[ring100].mean()) if ring100.any() else 0.0
    mean_bh_ring250 = float(topology[ring250].mean()) if ring250.any() else 0.0

    frac_above_tx = float(((topology > h_tx) & (~ground)).mean())

    los_pixels = (los_mask > 0) & ground
    los_fraction = float(los_pixels.sum() / max(ground.sum(), 1))
    # mean building height on the map restricted to regions around LoS rays:
    # proxy = mean building height in a dilated neighborhood of LoS pixels.
    # Cheap approximation: mean building height in any non-ground pixel
    # that lies within 10 m of a LoS pixel. For speed we sample directly
    # mean nearby-buildings via the shared ring250 statistic instead.
    mean_bh_over_los = mean_bh_ring250  # cheap proxy, same ring statistic

    return np.array([
        float(h_tx),
        float(math.log(max(h_tx, 1.0))),
        float(h_tx) ** 2,
        float(bh.mean()),
        float(bh.std()),
        density,
        mean_bh_ring100,
        mean_bh_ring250,
        frac_above_tx,
        los_fraction,
        mean_bh_over_los,
        1.0,  # bias term
    ], dtype=np.float64)


@dataclass
class Calibration:
    baseline_name: str
    global_a: float
    global_b: float
    feat_coef_a: np.ndarray          # shape (len(FEATURE_NAMES),)
    feat_coef_b: np.ndarray
    feature_names: Tuple[str, ...] = FEATURE_NAMES

    def apply_global(self, baseline: np.ndarray) -> np.ndarray:
        return self.global_a * baseline + self.global_b

    def apply_feature_based(self, baseline: np.ndarray, feats: np.ndarray) -> np.ndarray:
        a = float(feats @ self.feat_coef_a)
        b = float(feats @ self.feat_coef_b)
        return a * baseline + b


def _ridge_solve(X: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    XtX = X.T @ X
    reg = lam * np.eye(XtX.shape[0])
    # Do not regularise the bias column (last column).
    reg[-1, -1] = 0.0
    return np.linalg.solve(XtX + reg, X.T @ y)


def _gather_pixels_and_oracle(
    hdf5_path: Path,
    refs: Sequence[SampleRef],
    max_pixels_per_sample: int,
    rng: random.Random,
) -> Dict[str, np.ndarray]:
    """Walk samples, collect subsampled LoS pixels + per-sample oracle (a, b)
    and features. Only called on the FIT split — oracle labels are used only
    as supervision for the feature regressor, never on eval samples."""
    d2d_map, _ = distance_maps_m()
    pix_fspl: List[np.ndarray] = []
    pix_incoh: List[np.ndarray] = []
    pix_pl: List[np.ndarray] = []
    oracle_a_fspl: List[float] = []
    oracle_b_fspl: List[float] = []
    oracle_a_incoh: List[float] = []
    oracle_b_incoh: List[float] = []
    feats: List[np.ndarray] = []

    with h5py.File(str(hdf5_path), "r") as handle:
        for ref in refs:
            s = load_sample(handle, ref)
            valid = (s["path_loss"] >= PATH_LOSS_MIN_DB) & s["ground"] & (s["los_mask"] > 0)
            n = int(valid.sum())
            if n == 0:
                continue
            fspl = fspl_db(d3d_for_height(d2d_map, ref.uav_height_m))
            incoh = incoherent_two_ray_db(d2d_map, ref.uav_height_m)
            pl = s["path_loss"][valid].astype(np.float32)
            fspl_at = fspl[valid].astype(np.float32)
            incoh_at = incoh[valid].astype(np.float32)

            a_fs, b_fs = _ols_affine(fspl_at, pl)
            a_ic, b_ic = _ols_affine(incoh_at, pl)
            oracle_a_fspl.append(a_fs); oracle_b_fspl.append(b_fs)
            oracle_a_incoh.append(a_ic); oracle_b_incoh.append(b_ic)
            feats.append(compute_sample_features(s["topology"], s["los_mask"], ref.uav_height_m))

            if n > max_pixels_per_sample:
                idx = rng.sample(range(n), max_pixels_per_sample)
                idx_arr = np.asarray(idx)
            else:
                idx_arr = np.arange(n)
            pix_fspl.append(fspl_at[idx_arr])
            pix_incoh.append(incoh_at[idx_arr])
            pix_pl.append(pl[idx_arr])

    return {
        "pix_fspl": np.concatenate(pix_fspl) if pix_fspl else np.zeros(0, np.float32),
        "pix_incoh": np.concatenate(pix_incoh) if pix_incoh else np.zeros(0, np.float32),
        "pix_pl": np.concatenate(pix_pl) if pix_pl else np.zeros(0, np.float32),
        "oracle_a_fspl": np.asarray(oracle_a_fspl, dtype=np.float64),
        "oracle_b_fspl": np.asarray(oracle_b_fspl, dtype=np.float64),
        "oracle_a_incoh": np.asarray(oracle_a_incoh, dtype=np.float64),
        "oracle_b_incoh": np.asarray(oracle_b_incoh, dtype=np.float64),
        "features": np.asarray(feats, dtype=np.float64) if feats else np.zeros((0, len(FEATURE_NAMES)), np.float64),
    }


def fit_calibrations(fit_pack: Dict[str, np.ndarray], ridge_lambda: float = 1e-2) -> Dict[str, Calibration]:
    """Fit:
      - global (a, b) from ALL fit-split LoS pixels (OLS, 1 coefficient pair per baseline)
      - feature-based predictors coef_a, coef_b from per-sample oracle (a_s, b_s)
        supervised by topology features. Ridge with tiny λ, bias column unregularised.
    """
    cals: Dict[str, Calibration] = {}
    X = fit_pack["features"]
    for name, pix_key, ora_a_key, ora_b_key in (
        ("fspl", "pix_fspl", "oracle_a_fspl", "oracle_b_fspl"),
        ("incoherent_two_ray", "pix_incoh", "oracle_a_incoh", "oracle_b_incoh"),
    ):
        pix_base = fit_pack[pix_key]
        pix_y = fit_pack["pix_pl"]
        if pix_base.size:
            ga, gb = _ols_affine(pix_base, pix_y)
        else:
            ga, gb = 1.0, 0.0
        if X.size:
            coef_a = _ridge_solve(X, fit_pack[ora_a_key], lam=ridge_lambda)
            coef_b = _ridge_solve(X, fit_pack[ora_b_key], lam=ridge_lambda)
        else:
            coef_a = np.zeros(len(FEATURE_NAMES)); coef_a[-1] = 1.0
            coef_b = np.zeros(len(FEATURE_NAMES))
        cals[name] = Calibration(
            baseline_name=name,
            global_a=ga, global_b=gb,
            feat_coef_a=coef_a, feat_coef_b=coef_b,
        )
    return cals


def run_baseline_evaluation(
    hdf5_path: Path,
    refs: Sequence[SampleRef],
    split_tag: str,
    calibrations: Optional[Dict[str, Calibration]],
    max_samples: Optional[int],
    rng: random.Random,
    collect_scatter: int = 200_000,
) -> Tuple[List[SampleResidualStats], Dict[str, np.ndarray]]:
    """Evaluate raw + calibrated baselines on a given sample list.

    If `calibrations` is provided, also reports calibrated RMSE (both global
    and feature-based per-sample). Calibrations must be fit on a disjoint
    (fit) split before calling this on eval samples.
    """
    d2d_map, _ = distance_maps_m()

    chosen = list(refs)
    rng.shuffle(chosen)
    if max_samples is not None:
        chosen = chosen[:max_samples]

    per_sample: List[SampleResidualStats] = []
    scat_d3d: List[np.ndarray] = []
    scat_h: List[np.ndarray] = []
    scat_pl: List[np.ndarray] = []
    scat_fr: List[np.ndarray] = []
    scat_tr: List[np.ndarray] = []
    feat_pred_sse_fspl = 0.0
    feat_pred_sse_incoh = 0.0
    feat_pred_n = 0

    n_keep_per_sample = max(64, collect_scatter // max(len(chosen), 1))

    with h5py.File(str(hdf5_path), "r") as handle:
        for ref in chosen:
            s = load_sample(handle, ref)
            valid = (s["path_loss"] >= PATH_LOSS_MIN_DB) & s["ground"] & (s["los_mask"] > 0)
            n = int(valid.sum())
            if n == 0:
                continue
            d3d = d3d_for_height(d2d_map, ref.uav_height_m)
            fspl = fspl_db(d3d)
            two_ray = two_ray_pathloss_db(d2d_map, ref.uav_height_m)
            incoh = incoherent_two_ray_db(d2d_map, ref.uav_height_m)

            pl_obs = s["path_loss"][valid].astype(np.float32)
            fspl_at = fspl[valid]
            two_ray_at = two_ray[valid]
            incoh_at = incoh[valid]
            d3d_at = d3d[valid]

            fspl_resid = pl_obs - fspl_at
            two_ray_resid = pl_obs - two_ray_at
            incoh_resid = pl_obs - incoh_at

            if calibrations is not None:
                fspl_cal = calibrations["fspl"].apply_global(fspl_at)
                incoh_cal = calibrations["incoherent_two_ray"].apply_global(incoh_at)
                fspl_calib_rmse = float(np.sqrt(np.mean((pl_obs - fspl_cal) ** 2)))
                incoh_calib_rmse = float(np.sqrt(np.mean((pl_obs - incoh_cal) ** 2)))

                feats = compute_sample_features(s["topology"], s["los_mask"], ref.uav_height_m)
                fspl_feat = calibrations["fspl"].apply_feature_based(fspl_at, feats)
                incoh_feat = calibrations["incoherent_two_ray"].apply_feature_based(incoh_at, feats)
                feat_pred_sse_fspl += float(np.sum((pl_obs - fspl_feat) ** 2))
                feat_pred_sse_incoh += float(np.sum((pl_obs - incoh_feat) ** 2))
                feat_pred_n += int(pl_obs.size)
            else:
                fspl_calib_rmse = float("nan")
                incoh_calib_rmse = float("nan")

            per_sample.append(
                SampleResidualStats(
                    city=ref.city,
                    sample=ref.sample,
                    split=split_tag,
                    uav_height_m=ref.uav_height_m,
                    n_los_pixels=n,
                    pl_mean_db=float(pl_obs.mean()),
                    pl_std_db=float(pl_obs.std()),
                    fspl_rmse_db=float(np.sqrt(np.mean(fspl_resid ** 2))),
                    fspl_bias_db=float(fspl_resid.mean()),
                    two_ray_rmse_db=float(np.sqrt(np.mean(two_ray_resid ** 2))),
                    two_ray_bias_db=float(two_ray_resid.mean()),
                    incoherent_two_ray_rmse_db=float(np.sqrt(np.mean(incoh_resid ** 2))),
                    incoherent_two_ray_bias_db=float(incoh_resid.mean()),
                    fspl_calib_rmse_db=fspl_calib_rmse,
                    incoherent_two_ray_calib_rmse_db=incoh_calib_rmse,
                )
            )

            if n > n_keep_per_sample:
                idx = rng.sample(range(n), n_keep_per_sample)
                idx_arr = np.asarray(idx)
            else:
                idx_arr = np.arange(n)
            scat_d3d.append(d3d_at[idx_arr].astype(np.float32))
            scat_h.append(np.full(idx_arr.size, ref.uav_height_m, dtype=np.float32))
            scat_pl.append(pl_obs[idx_arr])
            scat_fr.append(fspl_resid[idx_arr].astype(np.float32))
            scat_tr.append(two_ray_resid[idx_arr].astype(np.float32))

    scatter = {
        "d3d_m": np.concatenate(scat_d3d) if scat_d3d else np.zeros(0, np.float32),
        "h_tx_m": np.concatenate(scat_h) if scat_h else np.zeros(0, np.float32),
        "pl_obs_db": np.concatenate(scat_pl) if scat_pl else np.zeros(0, np.float32),
        "fspl_residual_db": np.concatenate(scat_fr) if scat_fr else np.zeros(0, np.float32),
        "two_ray_residual_db": np.concatenate(scat_tr) if scat_tr else np.zeros(0, np.float32),
        "feat_pred_rmse_fspl_db": (math.sqrt(feat_pred_sse_fspl / feat_pred_n)
                                   if feat_pred_n else float("nan")),
        "feat_pred_rmse_incoh_db": (math.sqrt(feat_pred_sse_incoh / feat_pred_n)
                                    if feat_pred_n else float("nan")),
    }
    return per_sample, scatter


# ---------------------------------------------------------------------------
# Phase 3: plots (lazy-import matplotlib so the script runs headless too)
# ---------------------------------------------------------------------------

def make_plots(
    pairing_stats: Sequence[PairingBucketStats],
    pairing_deltas: np.ndarray,
    sample_stats: Sequence[SampleResidualStats],
    scatter: Dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) |ΔPL| histogram (pairing test, all pairs)
    if pairing_deltas.size > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(pairing_deltas, bins=80, range=(0.0, min(40.0, float(pairing_deltas.max()) + 1)))
        ax.set_xlabel("|ΔPL| at LoS-intersection pixels (dB)")
        ax.set_ylabel("count")
        ax.set_title(f"Pairing test: same-height LoS-intersection (n={pairing_deltas.size:,})")
        ax.axvline(1.0, color="r", ls="--", lw=1, label="1 dB (uint8 floor)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "pairing_abs_delta_hist.png", dpi=120)
        plt.close(fig)

    # 2) per-bucket mean/p95 |ΔPL| vs height
    if pairing_stats:
        hs = np.array([s.height_bin_m for s in pairing_stats])
        means = np.array([s.mean_abs_delta_db for s in pairing_stats])
        p95 = np.array([s.p95_abs_delta_db for s in pairing_stats])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(hs, means, "o-", label="mean |ΔPL|")
        ax.plot(hs, p95, "s--", label="p95 |ΔPL|")
        ax.set_xlabel("UAV height bin (m)")
        ax.set_ylabel("|ΔPL| (dB)")
        ax.set_title("Pairing determinism vs UAV height")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "pairing_vs_height.png", dpi=120)
        plt.close(fig)

    # 3) PL vs d3d coloured by h_tx
    if scatter["d3d_m"].size > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(
            scatter["d3d_m"],
            scatter["pl_obs_db"],
            c=scatter["h_tx_m"],
            s=2,
            alpha=0.35,
            cmap="viridis",
        )
        # overlay FSPL curves for a few representative heights
        d_lin = np.linspace(1.0, float(scatter["d3d_m"].max()) + 10.0, 400)
        for h in (30.0, 100.0, 300.0):
            ax.plot(np.sqrt(d_lin ** 2 + (h - RX_HEIGHT_M) ** 2), fspl_db(d_lin), lw=1,
                    label=f"FSPL, h_tx={h:.0f} m")
        ax.set_xlabel("d_3D (m)")
        ax.set_ylabel("path_loss (dB)")
        ax.set_title("LoS pixels: observed PL vs d_3D")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("UAV height (m)")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "pl_vs_d3d.png", dpi=120)
        plt.close(fig)

    # 4) residual vs d3d
    if scatter["fspl_residual_db"].size > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, key, title in (
            (axes[0], "fspl_residual_db", "PL - FSPL"),
            (axes[1], "two_ray_residual_db", "PL - two-ray"),
        ):
            ax.scatter(scatter["d3d_m"], scatter[key], c=scatter["h_tx_m"],
                       s=2, alpha=0.3, cmap="viridis")
            ax.axhline(0.0, color="k", lw=0.7)
            ax.set_xlabel("d_3D (m)")
            ax.set_title(title)
        axes[0].set_ylabel("residual (dB)")
        fig.tight_layout()
        fig.savefig(out_dir / "residuals_vs_d3d.png", dpi=120)
        plt.close(fig)

    # 5) per-sample RMSE vs h_tx
    if sample_stats:
        hs = np.array([s.uav_height_m for s in sample_stats])
        fs = np.array([s.fspl_rmse_db for s in sample_stats])
        tr = np.array([s.two_ray_rmse_db for s in sample_stats])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(hs, fs, s=12, alpha=0.6, label="FSPL RMSE")
        ax.scatter(hs, tr, s=12, alpha=0.6, label="two-ray RMSE")
        ax.set_xlabel("UAV height (m)")
        ax.set_ylabel("per-sample LoS RMSE (dB)")
        ax.set_title("Analytic-baseline LoS RMSE vs UAV height")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "baseline_rmse_vs_height.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def aggregate_baseline(sample_stats: Sequence[SampleResidualStats]) -> Dict[str, float]:
    if not sample_stats:
        return {}
    n_px = np.array([s.n_los_pixels for s in sample_stats], dtype=np.float64)
    w = n_px / n_px.sum()
    def _pw(field: str) -> float:
        arr = np.array([getattr(s, field) for s in sample_stats], dtype=np.float64)
        return float(np.sqrt(np.sum(w * arr ** 2)))
    fspl_rmse = np.array([s.fspl_rmse_db for s in sample_stats])
    fspl_bias = np.array([s.fspl_bias_db for s in sample_stats])
    tr_rmse = np.array([s.two_ray_rmse_db for s in sample_stats])
    ic_rmse = np.array([s.incoherent_two_ray_rmse_db for s in sample_stats])
    ic_bias = np.array([s.incoherent_two_ray_bias_db for s in sample_stats])
    fspl_cal = np.array([s.fspl_calib_rmse_db for s in sample_stats])
    ic_cal = np.array([s.incoherent_two_ray_calib_rmse_db for s in sample_stats])
    has_calib = np.isfinite(fspl_cal).all() and np.isfinite(ic_cal).all()
    out = {
        "n_samples": int(len(sample_stats)),
        "total_los_pixels": int(n_px.sum()),
        "fspl_rmse_mean_db": float(fspl_rmse.mean()),
        "fspl_rmse_pixelweighted_db": _pw("fspl_rmse_db"),
        "fspl_bias_mean_db": float(fspl_bias.mean()),
        "two_ray_rmse_mean_db": float(tr_rmse.mean()),
        "two_ray_rmse_pixelweighted_db": _pw("two_ray_rmse_db"),
        "incoherent_two_ray_rmse_mean_db": float(ic_rmse.mean()),
        "incoherent_two_ray_rmse_pixelweighted_db": _pw("incoherent_two_ray_rmse_db"),
        "incoherent_two_ray_bias_mean_db": float(ic_bias.mean()),
    }
    if has_calib:
        out.update({
            "fspl_calib_rmse_mean_db": float(fspl_cal.mean()),
            "fspl_calib_rmse_pixelweighted_db": _pw("fspl_calib_rmse_db"),
            "incoherent_two_ray_calib_rmse_mean_db": float(ic_cal.mean()),
            "incoherent_two_ray_calib_rmse_pixelweighted_db": _pw("incoherent_two_ray_calib_rmse_db"),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "analysis_out")
    parser.add_argument("--height-bin-m", type=float, default=5.0,
                        help="Bucket UAV heights to this resolution for pairing test.")
    parser.add_argument("--max-pairs-per-bucket", type=int, default=20)
    parser.add_argument("--max-pixels-per-pair", type=int, default=4000)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap on enumerated samples (for smoke tests).")
    parser.add_argument("--max-baseline-samples", type=int, default=400,
                        help="Cap on samples evaluated in phase 2 (fit+eval splits combined).")
    parser.add_argument("--eval-ratio", type=float, default=0.30,
                        help="Fraction of samples (by city holdout) used for calibration eval.")
    parser.add_argument("--fit-pixels-per-sample", type=int, default=2000,
                        help="Pixels subsampled per fit-split sample when fitting global affine.")
    parser.add_argument("--ridge-lambda", type=float, default=1e-2,
                        help="Ridge regularisation for the feature-based (a, b) predictor.")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="City-holdout seed (match Try 76 uses 42).")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-pairing", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip fitting global + feature-based calibration.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[try78] HDF5={args.hdf5}")
    refs = enumerate_samples(args.hdf5, max_samples=args.max_samples)
    print(f"[try78] enumerated {len(refs)} samples; "
          f"height range [{min(r.uav_height_m for r in refs):.1f}, "
          f"{max(r.uav_height_m for r in refs):.1f}] m")

    pairing_stats: List[PairingBucketStats] = []
    pairing_deltas = np.zeros(0, dtype=np.float32)
    if not args.skip_pairing:
        print("[try78] phase 1: pairing test")
        pairing_stats, pairing_deltas = run_pairing_test(
            args.hdf5, refs,
            height_bin_m=args.height_bin_m,
            max_pairs_per_bucket=args.max_pairs_per_bucket,
            max_pixels_per_pair=args.max_pixels_per_pair,
            rng=rng,
        )
        if pairing_deltas.size:
            print(f"  total intersection pixels sampled: {pairing_deltas.size:,}")
            print(f"  overall  mean |ΔPL|: {pairing_deltas.mean():.3f} dB")
            print(f"  overall  p50  |ΔPL|: {np.percentile(pairing_deltas, 50):.3f} dB")
            print(f"  overall  p95  |ΔPL|: {np.percentile(pairing_deltas, 95):.3f} dB")
            print(f"  overall  p99  |ΔPL|: {np.percentile(pairing_deltas, 99):.3f} dB")
        else:
            print("  (no pairs found — check --height-bin-m)")

    sample_stats: List[SampleResidualStats] = []
    scatter: Dict[str, np.ndarray] = {
        k: np.zeros(0, np.float32)
        for k in ("d3d_m", "h_tx_m", "pl_obs_db", "fspl_residual_db", "two_ray_residual_db")
    }
    calibrations: Optional[Dict[str, Calibration]] = None
    calib_summary: Dict[str, float] = {}
    fit_agg: Dict[str, float] = {}
    eval_agg: Dict[str, float] = {}
    fit_scatter: Dict[str, object] = {}
    eval_scatter: Dict[str, object] = {}
    if not args.skip_baseline:
        fit_refs, eval_refs = split_city_holdout_78(
            refs, eval_ratio=args.eval_ratio, split_seed=args.split_seed,
        )
        cap_fit = args.max_baseline_samples
        cap_eval = args.max_baseline_samples
        if cap_fit is not None:
            cap_fit = int(round(cap_fit * (1.0 - args.eval_ratio)))
            cap_eval = args.max_baseline_samples - cap_fit

        # downsample per-split sample lists if needed
        rng_split = random.Random(args.split_seed)
        rng_split.shuffle(fit_refs)
        rng_split.shuffle(eval_refs)
        if cap_fit is not None:
            fit_refs = fit_refs[:cap_fit]
            eval_refs = eval_refs[:cap_eval]
        print(f"[try78] split (seed={args.split_seed}): "
              f"fit cities={len({r.city for r in fit_refs})} samples={len(fit_refs)}; "
              f"eval cities={len({r.city for r in eval_refs})} samples={len(eval_refs)}")

        if not args.skip_calibration and fit_refs and eval_refs:
            print("[try78] phase 2a: collecting FIT-split pixels + per-sample oracle (a, b)")
            fit_pack = _gather_pixels_and_oracle(
                args.hdf5, fit_refs,
                max_pixels_per_sample=args.fit_pixels_per_sample,
                rng=rng,
            )
            print(f"  fit pixels: {fit_pack['pix_pl'].size:,}  "
                  f"fit samples: {fit_pack['features'].shape[0]}")
            calibrations = fit_calibrations(fit_pack, ridge_lambda=args.ridge_lambda)
            for name, cal in calibrations.items():
                print(f"  calibration[{name}] global a={cal.global_a:.4f}  b={cal.global_b:.4f}")
            calib_summary = {
                "fit_n_samples": int(fit_pack["features"].shape[0]),
                "fit_n_pixels": int(fit_pack["pix_pl"].size),
                **{f"{name}_global_a": cal.global_a for name, cal in calibrations.items()},
                **{f"{name}_global_b": cal.global_b for name, cal in calibrations.items()},
            }

        print("[try78] phase 2b: evaluate on FIT split (sanity, same data as calibration)")
        fit_stats, fit_scatter = run_baseline_evaluation(
            args.hdf5, fit_refs,
            split_tag="fit",
            calibrations=calibrations,
            max_samples=None,
            rng=rng,
        )
        fit_agg = aggregate_baseline(fit_stats)
        for k, v in fit_agg.items():
            print(f"  fit/{k}: {v}")
        if not math.isnan(fit_scatter.get("feat_pred_rmse_fspl_db", float("nan"))):
            print(f"  fit/fspl_feature_predicted_rmse_db (pixel-weighted): "
                  f"{fit_scatter['feat_pred_rmse_fspl_db']:.4f}")
            print(f"  fit/incoherent_two_ray_feature_predicted_rmse_db (pixel-weighted): "
                  f"{fit_scatter['feat_pred_rmse_incoh_db']:.4f}")

        print("[try78] phase 2c: evaluate on EVAL split (held-out cities)")
        eval_stats, eval_scatter = run_baseline_evaluation(
            args.hdf5, eval_refs,
            split_tag="eval",
            calibrations=calibrations,
            max_samples=None,
            rng=rng,
        )
        eval_agg = aggregate_baseline(eval_stats)
        for k, v in eval_agg.items():
            print(f"  eval/{k}: {v}")
        if not math.isnan(eval_scatter.get("feat_pred_rmse_fspl_db", float("nan"))):
            print(f"  eval/fspl_feature_predicted_rmse_db (pixel-weighted): "
                  f"{eval_scatter['feat_pred_rmse_fspl_db']:.4f}")
            print(f"  eval/incoherent_two_ray_feature_predicted_rmse_db (pixel-weighted): "
                  f"{eval_scatter['feat_pred_rmse_incoh_db']:.4f}")

        sample_stats = fit_stats + eval_stats
        # prefer eval scatter for the diagnostic plots (honest numbers)
        scatter = eval_scatter

    # Serialise feature-based predictor coefficients if available
    calib_serialised: Dict[str, Dict[str, object]] = {}
    if calibrations is not None:
        for name, cal in calibrations.items():
            calib_serialised[name] = {
                "global_a": cal.global_a,
                "global_b": cal.global_b,
                "feature_names": list(cal.feature_names),
                "feat_coef_a": cal.feat_coef_a.tolist(),
                "feat_coef_b": cal.feat_coef_b.tolist(),
            }

    # Feature-based predictor pixel-weighted RMSE from the eval-phase scatter
    feat_predicted = {
        "fit_fspl_rmse_db": fit_scatter.get("feat_pred_rmse_fspl_db", float("nan")),
        "fit_incoh_two_ray_rmse_db": fit_scatter.get("feat_pred_rmse_incoh_db", float("nan")),
        "eval_fspl_rmse_db": eval_scatter.get("feat_pred_rmse_fspl_db", float("nan")),
        "eval_incoh_two_ray_rmse_db": eval_scatter.get("feat_pred_rmse_incoh_db", float("nan")),
    }

    # Persist JSON / NPZ
    summary = {
        "config": {
            "hdf5": str(args.hdf5),
            "img_size": IMG_SIZE,
            "tx_pixel": [TX_ROW, TX_COL],
            "meters_per_pixel": METERS_PER_PIXEL,
            "freq_ghz": FREQ_GHZ,
            "rx_height_m": RX_HEIGHT_M,
            "path_loss_min_db": PATH_LOSS_MIN_DB,
            "height_bin_m": args.height_bin_m,
            "max_pairs_per_bucket": args.max_pairs_per_bucket,
            "max_pixels_per_pair": args.max_pixels_per_pair,
            "max_baseline_samples": args.max_baseline_samples,
            "eval_ratio": args.eval_ratio,
            "fit_pixels_per_sample": args.fit_pixels_per_sample,
            "ridge_lambda": args.ridge_lambda,
            "split_seed": args.split_seed,
            "seed": args.seed,
        },
        "pairing_buckets": [asdict(s) for s in pairing_stats],
        "baseline_aggregate_combined": aggregate_baseline(sample_stats),
        "baseline_aggregate_fit": fit_agg,
        "baseline_aggregate_eval": eval_agg,
        "calibration_summary": calib_summary,
        "calibrations": calib_serialised,
        "feature_predicted_rmse": feat_predicted,
        "baseline_per_sample": [asdict(s) for s in sample_stats],
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[try78] wrote {summary_path}")

    npz_payload = {k: v for k, v in scatter.items() if isinstance(v, np.ndarray)}
    if pairing_deltas.size or any(v.size for v in npz_payload.values()):
        np.savez_compressed(
            args.out_dir / "arrays.npz",
            pairing_deltas=pairing_deltas,
            **npz_payload,
        )
        print(f"[try78] wrote {args.out_dir / 'arrays.npz'}")

    if not args.skip_plots:
        print("[try78] phase 3: plots")
        make_plots(pairing_stats, pairing_deltas, sample_stats, scatter, args.out_dir)
        print(f"[try78] plots in {args.out_dir}")

    print("[try78] done")


if __name__ == "__main__":
    main()
