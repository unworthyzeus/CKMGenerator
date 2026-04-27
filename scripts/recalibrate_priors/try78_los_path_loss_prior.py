"""Try 78 - LoS-only physics baseline (no DL, no NLoS).

This branch now compares three LoS-only models:

1. ``FSPL`` baseline.
2. ``FSPL + radial residual`` lookup conditioned on UAV height.
3. ``Coherent two-ray`` with fitted effective reflection parameters
   ``rho(height)``, ``phi(height)``, and ``bias(height)``.

The current main hypothesis is that CKM LoS is dominated by a coherent
ground-reflection pattern. The radial lookup is kept as a non-parametric
reference, but the two-ray model is the main physics candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np


IMG_SIZE = 513
TX_ROW = 256
TX_COL = 256
METERS_PER_PIXEL = 1.0
FREQ_GHZ = 7.125
RX_HEIGHT_M = 1.5
PATH_LOSS_MIN_DB = 20.0
PATH_LOSS_MAX_DB = 180.0
DEFAULT_HDF5 = Path("c:/TFG/TFGpractice/Datasets/CKM_Dataset_270326.h5")
WAVELENGTH_M = 0.299792458 / FREQ_GHZ
WAVENUMBER = 2.0 * math.pi / WAVELENGTH_M


def _build_geometry() -> Tuple[np.ndarray, np.ndarray]:
    ii, jj = np.indices((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    dy = (ii - TX_ROW) * METERS_PER_PIXEL
    dx = (jj - TX_COL) * METERS_PER_PIXEL
    d2d = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    radius_px = np.rint(d2d / METERS_PER_PIXEL).astype(np.int16)
    return d2d, radius_px


_D2D_M, _RADIUS_PX = _build_geometry()
MAX_RADIUS_PX = int(_RADIUS_PX.max())


def d_los_map_m(h_tx_m: float) -> np.ndarray:
    return np.sqrt(_D2D_M * _D2D_M + (float(h_tx_m) - RX_HEIGHT_M) ** 2).astype(np.float32)


def d_ref_map_m(h_tx_m: float) -> np.ndarray:
    return np.sqrt(_D2D_M * _D2D_M + (float(h_tx_m) + RX_HEIGHT_M) ** 2).astype(np.float32)


def fspl_db(h_tx_m: float) -> np.ndarray:
    d3d = d_los_map_m(h_tx_m)
    d_km = np.maximum(d3d, 1.0) / 1000.0
    return (
        32.45
        + 20.0 * np.log10(d_km)
        + 20.0 * math.log10(FREQ_GHZ * 1000.0)
    ).astype(np.float32)


def coherent_two_ray_correction_db(
    h_tx_m: float,
    rho: float,
    phi_rad: float,
) -> np.ndarray:
    d_los = d_los_map_m(h_tx_m)
    d_ref = d_ref_map_m(h_tx_m)
    ratio = np.clip(d_los / np.maximum(d_ref, 1e-6), 0.0, 2.0).astype(np.float32)
    phase = (WAVENUMBER * (d_ref - d_los)).astype(np.float32)
    amp = np.abs(1.0 + float(rho) * ratio * np.exp(-1j * (phase + float(phi_rad))))
    return (-20.0 * np.log10(np.maximum(amp, 1e-6))).astype(np.float32)


@dataclass
class SampleRef:
    city: str
    sample: str
    uav_height_m: float


@dataclass
class SampleEvalResult:
    city: str
    sample: str
    uav_height_m: float
    n_los: int
    fspl_rmse_los: float
    radial_rmse_los: float
    two_ray_rmse_los: float
    fspl_bias_los: float
    radial_bias_los: float
    two_ray_bias_los: float
    radial_improvement_db: float
    two_ray_improvement_db: float


def enumerate_samples(hdf5_path: Path) -> List[SampleRef]:
    refs: List[SampleRef] = []
    with h5py.File(str(hdf5_path), "r") as handle:
        for city in sorted(handle.keys()):
            for sample in sorted(handle[city].keys()):
                h_tx = float(np.asarray(handle[city][sample]["uav_height"][...]).reshape(-1)[0])
                refs.append(SampleRef(city=city, sample=sample, uav_height_m=h_tx))
    return refs


def subsample_refs(
    refs: Sequence[SampleRef],
    max_samples: Optional[int],
    seed: int,
) -> List[SampleRef]:
    picked = list(refs)
    if max_samples is None or len(picked) <= max_samples:
        return picked
    rng = random.Random(seed)
    rng.shuffle(picked)
    return picked[:max_samples]


def split_city_holdout(
    refs: Sequence[SampleRef],
    eval_ratio: float = 0.30,
    split_seed: int = 42,
) -> Tuple[List[SampleRef], List[SampleRef]]:
    by_city: Dict[str, List[SampleRef]] = {}
    for ref in refs:
        by_city.setdefault(ref.city, []).append(ref)

    cities = list(by_city.keys())
    rng = random.Random(split_seed)
    rng.shuffle(cities)

    target_eval = int(round(len(refs) * eval_ratio))
    fit_refs: List[SampleRef] = []
    eval_refs: List[SampleRef] = []

    for idx, city in enumerate(cities):
        remaining_cities = len(cities) - idx
        city_refs = by_city[city]
        if len(eval_refs) < target_eval and remaining_cities > 1:
            eval_refs.extend(city_refs)
        else:
            fit_refs.extend(city_refs)

    if not fit_refs or not eval_refs:
        shuffled = list(refs)
        rng.shuffle(shuffled)
        cut = max(1, int(round(len(shuffled) * (1.0 - eval_ratio))))
        fit_refs = shuffled[:cut]
        eval_refs = shuffled[cut:]
    return fit_refs, eval_refs


def load_sample(handle: h5py.File, ref: SampleRef) -> Dict[str, np.ndarray]:
    grp = handle[ref.city][ref.sample]
    topology = np.asarray(grp["topology_map"][...], dtype=np.float32)
    los_mask = np.asarray(grp["los_mask"][...], dtype=np.uint8)
    path_loss = np.asarray(grp["path_loss"][...], dtype=np.float32)
    ground = topology == 0.0
    valid_los = ground & (los_mask > 0) & np.isfinite(path_loss) & (path_loss >= PATH_LOSS_MIN_DB)
    return {
        "topology": topology,
        "los_mask": los_mask,
        "path_loss": path_loss,
        "ground": ground,
        "valid_los": valid_los,
    }


def height_bin_key(h_tx_m: float, height_bin_m: float) -> float:
    return float(round(float(h_tx_m) / float(height_bin_m)) * float(height_bin_m))


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.float32)
    mask = den > 0
    out[mask] = (num[mask] / den[mask]).astype(np.float32)
    return out


def _gaussian_weighted_smooth(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    sigma: float,
) -> np.ndarray:
    if y.size == 0:
        return y.astype(np.float32)
    sigma = max(float(sigma), 1e-3)
    out = np.zeros_like(y, dtype=np.float32)
    for i, xi in enumerate(x):
        kernel = np.exp(-0.5 * ((x - xi) / sigma) ** 2).astype(np.float64)
        wk = kernel * np.maximum(weights.astype(np.float64), 1e-6)
        den = float(wk.sum())
        out[i] = float(np.sum(wk * y) / den) if den > 0.0 else float(y[i])
    return out


def _smooth_radial_profiles(
    radial_profile_db: np.ndarray,
    radial_count: np.ndarray,
    global_profile_db: np.ndarray,
    radius_sigma_px: float = 1.5,
) -> np.ndarray:
    if radial_profile_db.size == 0:
        return radial_profile_db.astype(np.float32)
    radius_axis = np.arange(radial_profile_db.shape[1], dtype=np.float32)
    out = np.zeros_like(radial_profile_db, dtype=np.float32)
    for i in range(radial_profile_db.shape[0]):
        profile = radial_profile_db[i].astype(np.float32).copy()
        missing = radial_count[i] == 0
        profile[missing] = global_profile_db[missing]
        out[i] = _gaussian_weighted_smooth(
            radius_axis,
            profile.astype(np.float64),
            np.maximum(radial_count[i], 1).astype(np.float64),
            sigma=radius_sigma_px,
        )
    return out


def fit_radial_calibration(
    hdf5_path: Path,
    fit_refs: Sequence[SampleRef],
    height_bin_m: float,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    height_bins = sorted({height_bin_key(ref.uav_height_m, height_bin_m) for ref in fit_refs})
    bin_to_idx = {hb: idx for idx, hb in enumerate(height_bins)}

    radial_sum = np.zeros((len(height_bins), MAX_RADIUS_PX + 1), dtype=np.float64)
    radial_count = np.zeros((len(height_bins), MAX_RADIUS_PX + 1), dtype=np.uint32)
    global_sum = np.zeros(MAX_RADIUS_PX + 1, dtype=np.float64)
    global_count = np.zeros(MAX_RADIUS_PX + 1, dtype=np.uint32)

    with h5py.File(str(hdf5_path), "r") as handle:
        for idx, ref in enumerate(fit_refs, start=1):
            sample = load_sample(handle, ref)
            valid_los = sample["valid_los"]
            if not valid_los.any():
                continue

            residual = sample["path_loss"] - fspl_db(ref.uav_height_m)
            rr = _RADIUS_PX[valid_los]
            values = residual[valid_los].astype(np.float32)

            hb = height_bin_key(ref.uav_height_m, height_bin_m)
            bi = bin_to_idx[hb]
            bincount_sum = np.bincount(rr, weights=values, minlength=MAX_RADIUS_PX + 1)
            bincount_cnt = np.bincount(rr, minlength=MAX_RADIUS_PX + 1).astype(np.uint32)

            radial_sum[bi] += bincount_sum
            radial_count[bi] += bincount_cnt
            global_sum += bincount_sum
            global_count += bincount_cnt

            if verbose and idx % 500 == 0:
                print(f"  radial-fit [{idx}/{len(fit_refs)}] processed")

    radial_profile = _safe_divide(radial_sum, radial_count)
    global_profile = _safe_divide(global_sum, global_count)
    return {
        "height_bin_m": np.asarray([height_bin_m], dtype=np.float32),
        "height_bins_m": np.asarray(height_bins, dtype=np.float32),
        "radial_profile_db": radial_profile,
        "radial_profile_smooth_db": _smooth_radial_profiles(radial_profile, radial_count, global_profile),
        "radial_count": radial_count,
        "global_profile_db": global_profile,
        "global_count": global_count,
    }


def _interpolate_profiles(
    h_tx_m: float,
    height_bins_m: np.ndarray,
    radial_profile_db: np.ndarray,
    radial_count: np.ndarray,
    global_profile_db: np.ndarray,
) -> np.ndarray:
    if height_bins_m.size == 1:
        profile = radial_profile_db[0].copy()
        missing = radial_count[0] == 0
        profile[missing] = global_profile_db[missing]
        return profile

    if h_tx_m <= float(height_bins_m[0]):
        profile = radial_profile_db[0].copy()
        missing = radial_count[0] == 0
        profile[missing] = global_profile_db[missing]
        return profile

    if h_tx_m >= float(height_bins_m[-1]):
        profile = radial_profile_db[-1].copy()
        missing = radial_count[-1] == 0
        profile[missing] = global_profile_db[missing]
        return profile

    hi = int(np.searchsorted(height_bins_m, h_tx_m))
    lo = hi - 1
    h_lo = float(height_bins_m[lo])
    h_hi = float(height_bins_m[hi])
    t = 0.0 if h_hi <= h_lo else float((h_tx_m - h_lo) / (h_hi - h_lo))

    prof_lo = radial_profile_db[lo]
    prof_hi = radial_profile_db[hi]
    cnt_lo = radial_count[lo]
    cnt_hi = radial_count[hi]

    out = np.empty_like(global_profile_db, dtype=np.float32)
    both = (cnt_lo > 0) & (cnt_hi > 0)
    only_lo = (cnt_lo > 0) & (cnt_hi == 0)
    only_hi = (cnt_lo == 0) & (cnt_hi > 0)
    neither = (cnt_lo == 0) & (cnt_hi == 0)

    out[both] = ((1.0 - t) * prof_lo[both] + t * prof_hi[both]).astype(np.float32)
    out[only_lo] = prof_lo[only_lo]
    out[only_hi] = prof_hi[only_hi]
    out[neither] = global_profile_db[neither]
    return out


def _stabilize_two_ray_params(
    height_bins_m: np.ndarray,
    rho: np.ndarray,
    phi_rad: np.ndarray,
    bias_db: np.ndarray,
    fit_count: np.ndarray,
    rho_max: float,
    height_sigma_m: float = 10.0,
    rho_outlier_thresh: float = 0.25,
    bias_outlier_thresh: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = np.sqrt(np.maximum(fit_count.astype(np.float64), 1.0))
    rho_smooth = _gaussian_weighted_smooth(height_bins_m, rho.astype(np.float64), weights, sigma=height_sigma_m)
    phi_smooth = _gaussian_weighted_smooth(height_bins_m, phi_rad.astype(np.float64), weights, sigma=height_sigma_m)
    bias_smooth = _gaussian_weighted_smooth(height_bins_m, bias_db.astype(np.float64), weights, sigma=height_sigma_m)

    suspicious = np.zeros_like(rho, dtype=bool)
    rho_cap_flag = rho >= float(rho_max) - 0.05
    suspicious |= rho_cap_flag & (np.abs(rho - rho_smooth) > rho_outlier_thresh)
    suspicious |= np.abs(bias_db - bias_smooth) > bias_outlier_thresh
    suspicious |= fit_count < 0.5 * float(np.median(fit_count[fit_count > 0])) if np.any(fit_count > 0) else False

    rho_final = rho.astype(np.float32).copy()
    phi_final = phi_rad.astype(np.float32).copy()
    bias_final = bias_db.astype(np.float32).copy()

    # Always use gently smoothed phi; it is effectively near-constant here.
    phi_final = phi_smooth.astype(np.float32)
    # For rho and bias, keep raw unless the bin is suspicious.
    rho_final[suspicious] = rho_smooth[suspicious]
    bias_final[suspicious] = bias_smooth[suspicious]

    return (
        np.clip(rho_smooth.astype(np.float32), 0.0, float(rho_max)),
        phi_smooth.astype(np.float32),
        bias_smooth.astype(np.float32),
        suspicious,
    )


def _interpolate_scalar(
    h_tx_m: float,
    height_bins_m: np.ndarray,
    values: np.ndarray,
) -> float:
    if values.size == 1:
        return float(values[0])
    if h_tx_m <= float(height_bins_m[0]):
        return float(values[0])
    if h_tx_m >= float(height_bins_m[-1]):
        return float(values[-1])
    hi = int(np.searchsorted(height_bins_m, h_tx_m))
    lo = hi - 1
    h_lo = float(height_bins_m[lo])
    h_hi = float(height_bins_m[hi])
    t = 0.0 if h_hi <= h_lo else float((h_tx_m - h_lo) / (h_hi - h_lo))
    return float((1.0 - t) * values[lo] + t * values[hi])


def predict_radial_map(h_tx_m: float, calibration: Dict[str, np.ndarray]) -> np.ndarray:
    base = fspl_db(h_tx_m)
    profile = _interpolate_profiles(
        h_tx_m=h_tx_m,
        height_bins_m=calibration["height_bins_m"],
        radial_profile_db=calibration.get("radial_profile_smooth_db", calibration["radial_profile_db"]),
        radial_count=calibration["radial_count"],
        global_profile_db=calibration["global_profile_db"],
    )
    pred = base + profile[_RADIUS_PX]
    return np.clip(pred, PATH_LOSS_MIN_DB, PATH_LOSS_MAX_DB).astype(np.float32)


def fit_two_ray_calibration(
    hdf5_path: Path,
    fit_refs: Sequence[SampleRef],
    height_bin_m: float,
    seed: int,
    phi_grid_size: int = 48,
    rho_max: float = 1.5,
    rho_step: float = 0.05,
    per_sample_cap: int = 2500,
    per_bin_cap: int = 8000,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    height_bins = sorted({height_bin_key(ref.uav_height_m, height_bin_m) for ref in fit_refs})
    bin_to_idx = {hb: idx for idx, hb in enumerate(height_bins)}
    rng = np.random.default_rng(seed)

    packed_phase: List[List[np.ndarray]] = [[] for _ in height_bins]
    packed_ratio: List[List[np.ndarray]] = [[] for _ in height_bins]
    packed_base: List[List[np.ndarray]] = [[] for _ in height_bins]
    packed_target: List[List[np.ndarray]] = [[] for _ in height_bins]

    with h5py.File(str(hdf5_path), "r") as handle:
        for idx, ref in enumerate(fit_refs, start=1):
            sample = load_sample(handle, ref)
            valid_los = sample["valid_los"]
            if not valid_los.any():
                continue

            hb = height_bin_key(ref.uav_height_m, height_bin_m)
            bi = bin_to_idx[hb]
            d_los = d_los_map_m(ref.uav_height_m)
            d_ref = d_ref_map_m(ref.uav_height_m)
            phase = (WAVENUMBER * (d_ref - d_los)).astype(np.float32)[valid_los]
            ratio = np.clip((d_los / np.maximum(d_ref, 1e-6)), 0.0, 2.0).astype(np.float32)[valid_los]
            base = fspl_db(ref.uav_height_m)[valid_los]
            target = sample["path_loss"][valid_los].astype(np.float32)

            if target.size > per_sample_cap:
                chosen = rng.choice(target.size, size=per_sample_cap, replace=False)
                phase = phase[chosen]
                ratio = ratio[chosen]
                base = base[chosen]
                target = target[chosen]

            packed_phase[bi].append(phase)
            packed_ratio[bi].append(ratio)
            packed_base[bi].append(base)
            packed_target[bi].append(target)

            if verbose and idx % 500 == 0:
                print(f"  two-ray-pack [{idx}/{len(fit_refs)}] processed")

    phi_grid = np.linspace(-math.pi, math.pi, phi_grid_size, endpoint=False, dtype=np.float32)
    rho_grid = np.arange(0.0, rho_max + 0.5 * rho_step, rho_step, dtype=np.float32)

    fitted_rho = np.zeros(len(height_bins), dtype=np.float32)
    fitted_phi = np.zeros(len(height_bins), dtype=np.float32)
    fitted_bias = np.zeros(len(height_bins), dtype=np.float32)
    fitted_rmse = np.full(len(height_bins), np.nan, dtype=np.float32)
    fitted_count = np.zeros(len(height_bins), dtype=np.int32)

    for bi, hb in enumerate(height_bins):
        if not packed_target[bi]:
            continue

        phase = np.concatenate(packed_phase[bi])
        ratio = np.concatenate(packed_ratio[bi])
        base = np.concatenate(packed_base[bi])
        target = np.concatenate(packed_target[bi])

        if target.size > per_bin_cap:
            chosen = rng.choice(target.size, size=per_bin_cap, replace=False)
            phase = phase[chosen]
            ratio = ratio[chosen]
            base = base[chosen]
            target = target[chosen]

        phase_matrix = phase[:, None] + phi_grid[None, :]
        exp_matrix = np.exp(-1j * phase_matrix)

        best_rmse = float("inf")
        best_rho = 0.0
        best_phi = 0.0
        best_bias = 0.0

        for rho in rho_grid:
            corr = -20.0 * np.log10(
                np.maximum(np.abs(1.0 + float(rho) * ratio[:, None] * exp_matrix), 1e-6)
            )
            pred_no_bias = base[:, None] + corr
            bias_candidates = (target[:, None] - pred_no_bias).mean(axis=0)
            rmse_candidates = np.sqrt(
                np.mean((pred_no_bias + bias_candidates[None, :] - target[:, None]) ** 2, axis=0)
            )
            best_phi_idx = int(rmse_candidates.argmin())
            rmse_here = float(rmse_candidates[best_phi_idx])
            if rmse_here < best_rmse:
                best_rmse = rmse_here
                best_rho = float(rho)
                best_phi = float(phi_grid[best_phi_idx])
                best_bias = float(bias_candidates[best_phi_idx])

        fitted_rho[bi] = best_rho
        fitted_phi[bi] = best_phi
        fitted_bias[bi] = best_bias
        fitted_rmse[bi] = best_rmse
        fitted_count[bi] = int(target.size)

        if verbose:
            print(
                f"  two-ray-fit h={hb:.1f} m  n={target.size:,}  "
                f"rho={best_rho:.3f}  phi={best_phi:.3f}  bias={best_bias:.3f}  rmse={best_rmse:.3f}"
            )

    rho_smooth, phi_smooth, bias_smooth, suspicious = _stabilize_two_ray_params(
        np.asarray(height_bins, dtype=np.float32),
        fitted_rho,
        fitted_phi,
        fitted_bias,
        fitted_count,
        rho_max=rho_max,
    )

    return {
        "height_bin_m": np.asarray([height_bin_m], dtype=np.float32),
        "height_bins_m": np.asarray(height_bins, dtype=np.float32),
        "rho_raw": fitted_rho,
        "phi_rad_raw": fitted_phi,
        "bias_db_raw": fitted_bias,
        "rho": rho_smooth,
        "phi_rad": phi_smooth,
        "bias_db": bias_smooth,
        "suspicious_bin_mask": suspicious.astype(np.int8),
        "fit_rmse_db": fitted_rmse,
        "fit_count": fitted_count,
        "phi_grid_size": np.asarray([phi_grid_size], dtype=np.int32),
        "rho_max": np.asarray([rho_max], dtype=np.float32),
        "rho_step": np.asarray([rho_step], dtype=np.float32),
        "per_sample_cap": np.asarray([per_sample_cap], dtype=np.int32),
        "per_bin_cap": np.asarray([per_bin_cap], dtype=np.int32),
    }


def fit_two_ray_residual_calibration(
    hdf5_path: Path,
    fit_refs: Sequence[SampleRef],
    two_ray_calibration: Dict[str, np.ndarray],
    height_bin_m: float,
    residual_clip_db: float = 2.5,
    radius_sigma_px: float = 1.5,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    height_bins = sorted({height_bin_key(ref.uav_height_m, height_bin_m) for ref in fit_refs})
    bin_to_idx = {hb: idx for idx, hb in enumerate(height_bins)}

    residual_sum = np.zeros((len(height_bins), MAX_RADIUS_PX + 1), dtype=np.float64)
    residual_count = np.zeros((len(height_bins), MAX_RADIUS_PX + 1), dtype=np.uint32)
    global_sum = np.zeros(MAX_RADIUS_PX + 1, dtype=np.float64)
    global_count = np.zeros(MAX_RADIUS_PX + 1, dtype=np.uint32)

    with h5py.File(str(hdf5_path), "r") as handle:
        for idx, ref in enumerate(fit_refs, start=1):
            sample = load_sample(handle, ref)
            valid_los = sample["valid_los"]
            if not valid_los.any():
                continue

            hb = height_bin_key(ref.uav_height_m, height_bin_m)
            bi = bin_to_idx[hb]
            two_ray_pred = predict_two_ray_map(ref.uav_height_m, two_ray_calibration)
            residual = np.clip(sample["path_loss"] - two_ray_pred, -residual_clip_db, residual_clip_db)
            rr = _RADIUS_PX[valid_los]
            values = residual[valid_los].astype(np.float32)

            bincount_sum = np.bincount(rr, weights=values, minlength=MAX_RADIUS_PX + 1)
            bincount_cnt = np.bincount(rr, minlength=MAX_RADIUS_PX + 1).astype(np.uint32)
            residual_sum[bi] += bincount_sum
            residual_count[bi] += bincount_cnt
            global_sum += bincount_sum
            global_count += bincount_cnt

            if verbose and idx % 500 == 0:
                print(f"  two-ray-residual-fit [{idx}/{len(fit_refs)}] processed")

    residual_profile = _safe_divide(residual_sum, residual_count)
    global_profile = _safe_divide(global_sum, global_count)
    smooth_profile = _smooth_radial_profiles(
        residual_profile,
        residual_count,
        global_profile,
        radius_sigma_px=radius_sigma_px,
    )
    return {
        "height_bin_m": np.asarray([height_bin_m], dtype=np.float32),
        "height_bins_m": np.asarray(height_bins, dtype=np.float32),
        "residual_profile_db": residual_profile,
        "residual_profile_smooth_db": smooth_profile,
        "residual_count": residual_count,
        "global_residual_db": global_profile,
        "global_count": global_count,
        "residual_clip_db": np.asarray([residual_clip_db], dtype=np.float32),
        "radius_sigma_px": np.asarray([radius_sigma_px], dtype=np.float32),
    }


def predict_two_ray_map(h_tx_m: float, calibration: Dict[str, np.ndarray]) -> np.ndarray:
    rho = _interpolate_scalar(h_tx_m, calibration["height_bins_m"], calibration["rho"])
    phi = _interpolate_scalar(h_tx_m, calibration["height_bins_m"], calibration["phi_rad"])
    bias = _interpolate_scalar(h_tx_m, calibration["height_bins_m"], calibration["bias_db"])
    pred = fspl_db(h_tx_m) + coherent_two_ray_correction_db(h_tx_m, rho=rho, phi_rad=phi) + bias
    if "residual_profile_db" in calibration:
        residual_profile = _interpolate_profiles(
            h_tx_m=h_tx_m,
            height_bins_m=calibration["height_bins_m"],
            radial_profile_db=calibration.get("residual_profile_smooth_db", calibration["residual_profile_db"]),
            radial_count=calibration["residual_count"],
            global_profile_db=calibration["global_residual_db"],
        )
        residual_clip_db = float(calibration.get("residual_clip_db", np.asarray([2.5], dtype=np.float32))[0])
        pred = pred + np.clip(residual_profile[_RADIUS_PX], -residual_clip_db, residual_clip_db)
    return np.clip(pred, PATH_LOSS_MIN_DB, PATH_LOSS_MAX_DB).astype(np.float32)


def evaluate(
    hdf5_path: Path,
    eval_refs: Sequence[SampleRef],
    radial_calibration: Dict[str, np.ndarray],
    two_ray_calibration: Dict[str, np.ndarray],
    verbose: bool = True,
    log_every: int = 100,
) -> List[SampleEvalResult]:
    results: List[SampleEvalResult] = []
    with h5py.File(str(hdf5_path), "r") as handle:
        for idx, ref in enumerate(eval_refs, start=1):
            sample = load_sample(handle, ref)
            valid_los = sample["valid_los"]
            if not valid_los.any():
                continue

            target = sample["path_loss"][valid_los]
            fspl_pred = fspl_db(ref.uav_height_m)[valid_los]
            radial_pred = predict_radial_map(ref.uav_height_m, radial_calibration)[valid_los]
            two_ray_pred = predict_two_ray_map(ref.uav_height_m, two_ray_calibration)[valid_los]

            fspl_err = fspl_pred - target
            radial_err = radial_pred - target
            two_ray_err = two_ray_pred - target

            fspl_rmse = float(np.sqrt(np.mean(fspl_err ** 2)))
            radial_rmse = float(np.sqrt(np.mean(radial_err ** 2)))
            two_ray_rmse = float(np.sqrt(np.mean(two_ray_err ** 2)))

            results.append(
                SampleEvalResult(
                    city=ref.city,
                    sample=ref.sample,
                    uav_height_m=ref.uav_height_m,
                    n_los=int(valid_los.sum()),
                    fspl_rmse_los=fspl_rmse,
                    radial_rmse_los=radial_rmse,
                    two_ray_rmse_los=two_ray_rmse,
                    fspl_bias_los=float(fspl_err.mean()),
                    radial_bias_los=float(radial_err.mean()),
                    two_ray_bias_los=float(two_ray_err.mean()),
                    radial_improvement_db=float(fspl_rmse - radial_rmse),
                    two_ray_improvement_db=float(fspl_rmse - two_ray_rmse),
                )
            )

            if verbose and idx % log_every == 0:
                cur = results[-1]
                print(
                    f"  eval [{idx}/{len(eval_refs)}] h={cur.uav_height_m:.1f} m  "
                    f"FSPL={cur.fspl_rmse_los:.3f} dB  "
                    f"radial={cur.radial_rmse_los:.3f} dB  "
                    f"2ray={cur.two_ray_rmse_los:.3f} dB"
                )
    return results


def aggregate(results: Sequence[SampleEvalResult]) -> Dict[str, float]:
    if not results:
        return {}
    total_los = sum(r.n_los for r in results)
    fspl_sse = sum(r.n_los * (r.fspl_rmse_los ** 2) for r in results)
    radial_sse = sum(r.n_los * (r.radial_rmse_los ** 2) for r in results)
    two_ray_sse = sum(r.n_los * (r.two_ray_rmse_los ** 2) for r in results)
    return {
        "n_samples": len(results),
        "total_los_pixels": total_los,
        "fspl_rmse_los_pw": math.sqrt(fspl_sse / total_los) if total_los else float("nan"),
        "radial_rmse_los_pw": math.sqrt(radial_sse / total_los) if total_los else float("nan"),
        "two_ray_rmse_los_pw": math.sqrt(two_ray_sse / total_los) if total_los else float("nan"),
        "mean_radial_gain_db": float(np.mean([r.radial_improvement_db for r in results])),
        "median_radial_gain_db": float(np.median([r.radial_improvement_db for r in results])),
        "mean_two_ray_gain_db": float(np.mean([r.two_ray_improvement_db for r in results])),
        "median_two_ray_gain_db": float(np.median([r.two_ray_improvement_db for r in results])),
    }


def save_calibration(
    radial_calibration: Dict[str, np.ndarray],
    two_ray_calibration: Dict[str, np.ndarray],
    path: Path,
    meta: Dict[str, object],
) -> None:
    payload = {
        "model_type": "los_physics_try78",
        "freq_ghz": FREQ_GHZ,
        "rx_height_m": RX_HEIGHT_M,
        "meters_per_pixel": METERS_PER_PIXEL,
        "radial": {
            "height_bin_m": float(radial_calibration["height_bin_m"][0]),
            "height_bins_m": radial_calibration["height_bins_m"].tolist(),
            "global_profile_db": radial_calibration["global_profile_db"].tolist(),
            "global_count": radial_calibration["global_count"].tolist(),
            "radial_profile_db": radial_calibration["radial_profile_db"].tolist(),
            "radial_profile_smooth_db": radial_calibration["radial_profile_smooth_db"].tolist(),
            "radial_count": radial_calibration["radial_count"].tolist(),
        },
        "two_ray": {
            "height_bin_m": float(two_ray_calibration["height_bin_m"][0]),
            "height_bins_m": two_ray_calibration["height_bins_m"].tolist(),
            "rho_raw": two_ray_calibration["rho_raw"].tolist(),
            "phi_rad_raw": two_ray_calibration["phi_rad_raw"].tolist(),
            "bias_db_raw": two_ray_calibration["bias_db_raw"].tolist(),
            "rho": two_ray_calibration["rho"].tolist(),
            "phi_rad": two_ray_calibration["phi_rad"].tolist(),
            "bias_db": two_ray_calibration["bias_db"].tolist(),
            "suspicious_bin_mask": two_ray_calibration["suspicious_bin_mask"].tolist(),
            "fit_rmse_db": two_ray_calibration["fit_rmse_db"].tolist(),
            "fit_count": two_ray_calibration["fit_count"].tolist(),
            "residual_profile_db": two_ray_calibration["residual_profile_db"].tolist(),
            "residual_profile_smooth_db": two_ray_calibration["residual_profile_smooth_db"].tolist(),
            "residual_count": two_ray_calibration["residual_count"].tolist(),
            "global_residual_db": two_ray_calibration["global_residual_db"].tolist(),
            "global_count": two_ray_calibration["global_count"].tolist(),
            "residual_clip_db": float(two_ray_calibration["residual_clip_db"][0]),
            "radius_sigma_px": float(two_ray_calibration["radius_sigma_px"][0]),
            "phi_grid_size": int(two_ray_calibration["phi_grid_size"][0]),
            "rho_max": float(two_ray_calibration["rho_max"][0]),
            "rho_step": float(two_ray_calibration["rho_step"][0]),
            "per_sample_cap": int(two_ray_calibration["per_sample_cap"][0]),
            "per_bin_cap": int(two_ray_calibration["per_bin_cap"][0]),
        },
        "meta": meta,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_calibration(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    payload = json.loads(path.read_text())
    radial = payload["radial"]
    two_ray = payload["two_ray"]
    radial_calibration = {
        "height_bin_m": np.asarray([radial["height_bin_m"]], dtype=np.float32),
        "height_bins_m": np.asarray(radial["height_bins_m"], dtype=np.float32),
        "global_profile_db": np.asarray(radial["global_profile_db"], dtype=np.float32),
        "global_count": np.asarray(radial["global_count"], dtype=np.uint32),
        "radial_profile_db": np.asarray(radial["radial_profile_db"], dtype=np.float32),
        "radial_profile_smooth_db": np.asarray(radial.get("radial_profile_smooth_db", radial["radial_profile_db"]), dtype=np.float32),
        "radial_count": np.asarray(radial["radial_count"], dtype=np.uint32),
    }
    two_ray_calibration = {
        "height_bin_m": np.asarray([two_ray["height_bin_m"]], dtype=np.float32),
        "height_bins_m": np.asarray(two_ray["height_bins_m"], dtype=np.float32),
        "rho_raw": np.asarray(two_ray.get("rho_raw", two_ray["rho"]), dtype=np.float32),
        "phi_rad_raw": np.asarray(two_ray.get("phi_rad_raw", two_ray["phi_rad"]), dtype=np.float32),
        "bias_db_raw": np.asarray(two_ray.get("bias_db_raw", two_ray["bias_db"]), dtype=np.float32),
        "rho": np.asarray(two_ray["rho"], dtype=np.float32),
        "phi_rad": np.asarray(two_ray["phi_rad"], dtype=np.float32),
        "bias_db": np.asarray(two_ray["bias_db"], dtype=np.float32),
        "suspicious_bin_mask": np.asarray(two_ray.get("suspicious_bin_mask", [0] * len(two_ray["rho"])), dtype=np.int8),
        "fit_rmse_db": np.asarray(two_ray["fit_rmse_db"], dtype=np.float32),
        "fit_count": np.asarray(two_ray["fit_count"], dtype=np.int32),
        "residual_profile_db": np.asarray(two_ray.get("residual_profile_db", np.zeros((len(two_ray["rho"]), MAX_RADIUS_PX + 1))), dtype=np.float32),
        "residual_profile_smooth_db": np.asarray(two_ray.get("residual_profile_smooth_db", two_ray.get("residual_profile_db", np.zeros((len(two_ray["rho"]), MAX_RADIUS_PX + 1)))), dtype=np.float32),
        "residual_count": np.asarray(two_ray.get("residual_count", np.zeros((len(two_ray["rho"]), MAX_RADIUS_PX + 1))), dtype=np.uint32),
        "global_residual_db": np.asarray(two_ray.get("global_residual_db", np.zeros(MAX_RADIUS_PX + 1)), dtype=np.float32),
        "global_count": np.asarray(two_ray.get("global_count", np.zeros(MAX_RADIUS_PX + 1)), dtype=np.uint32),
        "residual_clip_db": np.asarray([two_ray.get("residual_clip_db", 2.5)], dtype=np.float32),
        "radius_sigma_px": np.asarray([two_ray.get("radius_sigma_px", 1.5)], dtype=np.float32),
        "phi_grid_size": np.asarray([two_ray["phi_grid_size"]], dtype=np.int32),
        "rho_max": np.asarray([two_ray["rho_max"]], dtype=np.float32),
        "rho_step": np.asarray([two_ray["rho_step"]], dtype=np.float32),
        "per_sample_cap": np.asarray([two_ray["per_sample_cap"]], dtype=np.int32),
        "per_bin_cap": np.asarray([two_ray["per_bin_cap"]], dtype=np.int32),
    }
    return radial_calibration, two_ray_calibration


def make_plots(
    results: Sequence[SampleEvalResult],
    radial_calibration: Dict[str, np.ndarray],
    two_ray_calibration: Dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs = np.asarray([r.uav_height_m for r in results], dtype=np.float32)
    fspl_rmse = np.asarray([r.fspl_rmse_los for r in results], dtype=np.float32)
    radial_rmse = np.asarray([r.radial_rmse_los for r in results], dtype=np.float32)
    two_ray_rmse = np.asarray([r.two_ray_rmse_los for r in results], dtype=np.float32)
    radial_gains = np.asarray([r.radial_improvement_db for r in results], dtype=np.float32)
    two_ray_gains = np.asarray([r.two_ray_improvement_db for r in results], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(hs, fspl_rmse, s=9, alpha=0.45, label="FSPL")
    ax.scatter(hs, radial_rmse, s=9, alpha=0.45, label="FSPL + radial residual")
    ax.scatter(hs, two_ray_rmse, s=9, alpha=0.45, label="coherent two-ray")
    ax.set_xlabel("UAV height (m)")
    ax.set_ylabel("LoS RMSE (dB)")
    ax.set_title("Try 78 LoS-only RMSE vs height")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "los_rmse_vs_height.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(hs, radial_gains, s=9, alpha=0.55, label="radial gain")
    ax.scatter(hs, two_ray_gains, s=9, alpha=0.55, label="two-ray gain")
    ax.axhline(0.0, color="k", lw=1, ls="--")
    ax.set_xlabel("UAV height (m)")
    ax.set_ylabel("FSPL RMSE - model RMSE (dB)")
    ax.set_title("LoS gain over FSPL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "los_gain_vs_height.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    representative = radial_calibration["height_bins_m"]
    if representative.size > 8:
        idx = np.linspace(0, representative.size - 1, 8, dtype=int)
        representative = representative[idx]
    radius_axis = np.arange(MAX_RADIUS_PX + 1)
    for hb in representative:
        prof = _interpolate_profiles(
            h_tx_m=float(hb),
            height_bins_m=radial_calibration["height_bins_m"],
            radial_profile_db=radial_calibration["radial_profile_db"],
            radial_count=radial_calibration["radial_count"],
            global_profile_db=radial_calibration["global_profile_db"],
        )
        ax.plot(radius_axis, prof, label=f"{hb:.0f} m")
    ax.set_xlabel("Radius from Tx (pixels)")
    ax.set_ylabel("Residual over FSPL (dB)")
    ax.set_title("Learned radial residual profiles")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "radial_profiles.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(two_ray_calibration["height_bins_m"], two_ray_calibration["rho_raw"], alpha=0.35, label="rho raw")
    ax.plot(two_ray_calibration["height_bins_m"], two_ray_calibration["rho"], label="rho smooth")
    ax.plot(two_ray_calibration["height_bins_m"], two_ray_calibration["bias_db_raw"], alpha=0.35, label="bias raw")
    ax.plot(two_ray_calibration["height_bins_m"], two_ray_calibration["bias_db"], label="bias smooth")
    ax.set_xlabel("Height bin (m)")
    ax.set_ylabel("Value")
    ax.set_title("Fitted coherent two-ray parameters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "two_ray_params.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "prior_out",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-ratio", type=float, default=0.30)
    parser.add_argument("--height-bin-m", type=float, default=5.0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--calibration-json", type=Path, default=None)
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--phi-grid-size", type=int, default=48)
    parser.add_argument("--rho-max", type=float, default=1.5)
    parser.add_argument("--rho-step", type=float, default=0.05)
    parser.add_argument("--two-ray-per-sample-cap", type=int, default=2500)
    parser.add_argument("--two-ray-per-bin-cap", type=int, default=8000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[try78-los] HDF5={args.hdf5}")

    refs = enumerate_samples(args.hdf5)
    refs = subsample_refs(refs, args.max_samples, seed=args.seed)
    print(f"[try78-los] using {len(refs)} samples")

    fit_refs, eval_refs = split_city_holdout(refs, eval_ratio=args.eval_ratio, split_seed=args.split_seed)
    print(
        f"[try78-los] split seed={args.split_seed}: "
        f"fit {len({r.city for r in fit_refs})} cities / {len(fit_refs)} samples  |  "
        f"eval {len({r.city for r in eval_refs})} cities / {len(eval_refs)} samples"
    )

    radial_calibration: Dict[str, np.ndarray]
    two_ray_calibration: Dict[str, np.ndarray]
    if args.calibration_json:
        print(f"[try78-los] loading calibration from {args.calibration_json}")
        radial_calibration, two_ray_calibration = load_calibration(args.calibration_json)
    elif not args.skip_fit:
        print(f"[try78-los] fitting radial LoS profiles (height_bin={args.height_bin_m:.1f} m)")
        radial_calibration = fit_radial_calibration(
            args.hdf5,
            fit_refs,
            height_bin_m=args.height_bin_m,
            verbose=True,
        )
        print("[try78-los] fitting coherent two-ray parameters by height bin")
        two_ray_calibration = fit_two_ray_calibration(
            args.hdf5,
            fit_refs,
            height_bin_m=args.height_bin_m,
            seed=args.seed,
            phi_grid_size=args.phi_grid_size,
            rho_max=args.rho_max,
            rho_step=args.rho_step,
            per_sample_cap=args.two_ray_per_sample_cap,
            per_bin_cap=args.two_ray_per_bin_cap,
            verbose=True,
        )
        print("[try78-los] fitting small residual on top of smoothed two-ray")
        two_ray_residual = fit_two_ray_residual_calibration(
            args.hdf5,
            fit_refs,
            two_ray_calibration=two_ray_calibration,
            height_bin_m=args.height_bin_m,
            residual_clip_db=2.5,
            radius_sigma_px=1.5,
            verbose=True,
        )
        two_ray_calibration.update(two_ray_residual)
        cal_path = args.out_dir / "calibration.json"
        save_calibration(
            radial_calibration,
            two_ray_calibration,
            cal_path,
            meta={
                "n_fit_samples": len(fit_refs),
                "n_eval_samples": len(eval_refs),
                "seed": args.seed,
            },
        )
        print(f"[try78-los] calibration saved -> {cal_path}")
    else:
        raise ValueError("Need either --calibration-json or a fit pass (do not use --skip-fit alone).")

    eval_results: List[SampleEvalResult] = []
    if not args.skip_eval and eval_refs:
        print("[try78-los] evaluating on LoS pixels only ...")
        eval_results = evaluate(
            args.hdf5,
            eval_refs,
            radial_calibration,
            two_ray_calibration,
            verbose=True,
            log_every=args.log_every,
        )
        agg = aggregate(eval_results)
        print("\n[try78-los] === EVAL AGGREGATE (LoS pixel-weighted) ===")
        for key, value in agg.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    summary = {
        "model_type": "los_physics_try78",
        "split_seed": args.split_seed,
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "height_bin_m": float(args.height_bin_m),
        "n_fit_samples": len(fit_refs),
        "n_eval_samples": len(eval_refs),
        "aggregate": aggregate(eval_results),
        "two_ray_fit": {
            "height_bins_m": two_ray_calibration["height_bins_m"].tolist(),
            "rho_raw": two_ray_calibration["rho_raw"].tolist(),
            "rho": two_ray_calibration["rho"].tolist(),
            "phi_rad_raw": two_ray_calibration["phi_rad_raw"].tolist(),
            "phi_rad": two_ray_calibration["phi_rad"].tolist(),
            "bias_db_raw": two_ray_calibration["bias_db_raw"].tolist(),
            "bias_db": two_ray_calibration["bias_db"].tolist(),
            "suspicious_bin_mask": two_ray_calibration["suspicious_bin_mask"].tolist(),
            "fit_rmse_db": two_ray_calibration["fit_rmse_db"].tolist(),
            "fit_count": two_ray_calibration["fit_count"].tolist(),
            "residual_clip_db": float(two_ray_calibration["residual_clip_db"][0]),
        },
        "per_sample": [
            {
                "city": r.city,
                "sample": r.sample,
                "uav_height_m": r.uav_height_m,
                "n_los": r.n_los,
                "fspl_rmse_los": r.fspl_rmse_los,
                "radial_rmse_los": r.radial_rmse_los,
                "two_ray_rmse_los": r.two_ray_rmse_los,
                "radial_gain_db": r.radial_improvement_db,
                "two_ray_gain_db": r.two_ray_improvement_db,
            }
            for r in eval_results
        ],
    }
    summary_path = args.out_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[try78-los] summary -> {summary_path}")

    if not args.skip_plots and eval_results:
        print("[try78-los] generating plots ...")
        make_plots(eval_results, radial_calibration, two_ray_calibration, args.out_dir)
        print(f"[try78-los] plots -> {args.out_dir}")

    print("[try78-los] done")


if __name__ == "__main__":
    main()
