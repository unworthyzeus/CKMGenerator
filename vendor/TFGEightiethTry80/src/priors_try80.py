"""Try 80 - frozen prior bundle from Try 78 (path loss) and Try 79 (spreads)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


IMG_SIZE = 513
TX_ROW = 256
TX_COL = 256
METERS_PER_PIXEL = 1.0
FREQ_GHZ = 7.125
RX_HEIGHT_M = 1.5
PATH_LOSS_MIN_DB = 20.0
PATH_LOSS_MAX_DB = 180.0
HEIGHT_NORM_M = 90.0
KERNEL_SIZES = (15, 41)
ANT_Q1 = 58.12
ANT_Q2 = 103.85

TOPOLOGY_THRESHOLDS = {
    "density_q1": 0.12,
    "density_q2": 0.28,
    "height_q1": 12.0,
    "height_q2": 28.0,
}

TOPOLOGY_CLASSES = (
    "open_sparse_lowrise",
    "open_sparse_vertical",
    "mixed_compact_lowrise",
    "mixed_compact_midrise",
    "dense_block_midrise",
    "dense_block_highrise",
)

# Try 78 embedded NLoS constants.
SIGMA_LOS_RHO = 0.0272
SIGMA_LOS_MU = 0.7475
SIGMA_NLOS_RHO = 2.3197
SIGMA_NLOS_MU = 0.2361
A2G_NLOS_BIAS = -16.16
A2G_NLOS_AMP = 12.0436
A2G_NLOS_TAU = 7.52
A2G_LOS_LOG_COEFF = -20.0
A2G_LOS_BIAS = 0.0
DENSITY_Q1_78 = 0.1957
DENSITY_Q2_78 = 0.2549
HEIGHT_Q1_78 = 10.91
HEIGHT_Q2_78 = 15.95

PATH_FEATURE_NAMES = (
    "prior_sq",
    "prior",
    "log1p_d2d",
    "density_15",
    "density_41",
    "height_15",
    "height_41",
    "density_41_x_logd",
    "nlos_15",
    "nlos_41",
    "nlos_41_x_logd",
    "shadow_sigma",
    "theta_norm",
    "nlos_41_x_theta",
    "bias",
)
PATH_N_FEAT = len(PATH_FEATURE_NAMES)

METRIC_SPECS: Dict[str, Dict[str, float | str]] = {
    "delay_spread": {
        "clip_hi": 400.0,
        "base_los": 4.0,
        "base_nlos": 11.0,
        "coef_logd": 0.11,
        "coef_theta_inv": 0.62,
        "coef_density": 0.48,
        "coef_height": 0.32,
        "coef_nlos": 0.86,
        "coef_interaction": 0.60,
    },
    "angular_spread": {
        "clip_hi": 90.0,
        "base_los": 3.0,
        "base_nlos": 7.0,
        "coef_logd": 0.05,
        "coef_theta_inv": 0.52,
        "coef_density": 0.60,
        "coef_height": 0.38,
        "coef_nlos": 0.58,
        "coef_interaction": 0.35,
    },
}

TOPOLOGY_PRIOR_BIAS_LOG = {
    "delay_spread": {
        "open_sparse_lowrise": 0.00,
        "open_sparse_vertical": 0.18,
        "mixed_compact_lowrise": 0.06,
        "mixed_compact_midrise": 0.22,
        "dense_block_midrise": 0.24,
        "dense_block_highrise": 0.34,
    },
    "angular_spread": {
        "open_sparse_lowrise": 0.00,
        "open_sparse_vertical": 0.09,
        "mixed_compact_lowrise": 0.05,
        "mixed_compact_midrise": 0.16,
        "dense_block_midrise": 0.18,
        "dense_block_highrise": 0.24,
    },
}

LOS_CLIP_HI = {"delay_spread": 400.0, "angular_spread": 15.0}
NLOS_CLIP_HI = {"delay_spread": 400.0, "angular_spread": 90.0}


def _build_d2d() -> np.ndarray:
    ii, jj = np.indices((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    dy = (ii - TX_ROW) * METERS_PER_PIXEL
    dx = (jj - TX_COL) * METERS_PER_PIXEL
    return np.sqrt(dy * dy + dx * dx).astype(np.float32)


_D2D = _build_d2d()
_LOGD = np.log1p(_D2D).astype(np.float32)
_RADIUS_PX = np.rint(_D2D / METERS_PER_PIXEL).astype(np.int16)
_MAX_RADIUS_PX = int(_RADIUS_PX.max())
_WAVELENGTH_M = 0.299792458 / FREQ_GHZ
_WAVENUMBER = 2.0 * math.pi / _WAVELENGTH_M


@dataclass
class JointPriors:
    path_loss_prior: np.ndarray
    path_loss_los_prior: np.ndarray
    path_loss_nlos_prior: np.ndarray
    delay_spread_prior: np.ndarray
    angular_spread_prior: np.ndarray
    topology_class_6: str
    topology_class_3: str
    antenna_bin: str


class Try80PriorComputer:
    """Frozen prior loader and evaluator.

    The class computes:
    - LoS path-loss prior from Try 78 coherent two-ray calibration.
    - NLoS path-loss prior from Try 78 regime calibration.
    - Delay/angular spread priors from Try 79 log-domain calibration.
    """

    def __init__(
        self,
        try78_los_calibration_json: Path,
        try78_nlos_calibration_json: Path,
        try79_calibration_json: Path,
    ) -> None:
        self._path_los_cal = _load_try78_los_calibration(try78_los_calibration_json)
        self._path_nlos_cal = _load_old_calibration(try78_nlos_calibration_json)
        self._spread_cal = _load_spread_calibration(try79_calibration_json)

    def compute(self, topology: np.ndarray, los_mask: np.ndarray, h_tx: float) -> JointPriors:
        topology = np.asarray(topology, dtype=np.float32)
        los_mask = np.asarray(los_mask, dtype=np.float32)
        ct6 = classify_topology(topology)
        ct3 = macro_topology_class(ct6)
        ant = ant_bin(h_tx)

        los_prior = _predict_two_ray_map(h_tx, self._path_los_cal)
        nlos_prior = _compute_try78_nlos_map(topology, los_mask, h_tx, self._path_nlos_cal)
        path_prior = np.where(los_mask > 0.5, los_prior, nlos_prior).astype(np.float32)

        shared = compute_shared_features(topology, los_mask, h_tx)
        delay_prior = _compute_try79_metric_prior(
            metric="delay_spread",
            topology_class=ct6,
            ant_label=ant,
            los_mask=los_mask,
            shared=shared,
            coefs=self._spread_cal,
        )
        angular_prior = _compute_try79_metric_prior(
            metric="angular_spread",
            topology_class=ct6,
            ant_label=ant,
            los_mask=los_mask,
            shared=shared,
            coefs=self._spread_cal,
        )

        return JointPriors(
            path_loss_prior=path_prior,
            path_loss_los_prior=los_prior.astype(np.float32),
            path_loss_nlos_prior=nlos_prior.astype(np.float32),
            delay_spread_prior=delay_prior,
            angular_spread_prior=angular_prior,
            topology_class_6=ct6,
            topology_class_3=ct3,
            antenna_bin=ant,
        )


def classify_topology(topo_m: np.ndarray) -> str:
    non_ground = topo_m != 0.0
    density = float(np.mean(non_ground)) if non_ground.size else 0.0
    heights = topo_m[non_ground]
    mean_h = float(np.mean(heights)) if heights.size else 0.0
    d1 = TOPOLOGY_THRESHOLDS["density_q1"]
    d2 = TOPOLOGY_THRESHOLDS["density_q2"]
    h1 = TOPOLOGY_THRESHOLDS["height_q1"]
    h2 = TOPOLOGY_THRESHOLDS["height_q2"]
    if density <= d1:
        return "open_sparse_lowrise" if mean_h <= h1 else "open_sparse_vertical"
    if density >= d2:
        return "dense_block_midrise" if mean_h <= h2 else "dense_block_highrise"
    return "mixed_compact_lowrise" if mean_h <= h1 else "mixed_compact_midrise"


def macro_topology_class(topology_class_6: str) -> str:
    if topology_class_6.startswith("open_sparse"):
        return "open"
    if topology_class_6.startswith("mixed_compact"):
        return "mixed"
    return "dense"


def ant_bin(h_tx: float) -> str:
    if h_tx <= ANT_Q1:
        return "low_ant"
    if h_tx <= ANT_Q2:
        return "mid_ant"
    return "high_ant"


def box_mean(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return arr.astype(np.float32, copy=True)
    pad = k // 2
    padded = np.pad(arr.astype(np.float32), ((pad, pad), (pad, pad)), mode="reflect")
    integ = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    out = integ[k:, k:] - integ[:-k, k:] - integ[k:, :-k] + integ[:-k, :-k]
    return (out / float(k * k)).astype(np.float32)


def compute_shared_features(topology: np.ndarray, los_mask: np.ndarray, h_tx: float) -> Dict[str, np.ndarray]:
    ground = (topology == 0.0).astype(np.float32)
    buildings = 1.0 - ground
    nlos_ground = ((los_mask <= 0.5) & (ground > 0.5)).astype(np.float32)

    theta_deg = np.degrees(np.arctan2(max(float(h_tx) - RX_HEIGHT_M, 0.1), np.maximum(_D2D, 1.0)))
    theta_norm = np.clip(theta_deg / 90.0, 0.0, 1.0).astype(np.float32)
    h_norm_scalar = float(np.clip(np.log1p(max(float(h_tx) - RX_HEIGHT_M, 0.0)) / math.log1p(400.0), 0.0, 1.0))
    h_norm = np.full_like(_LOGD, h_norm_scalar, dtype=np.float32)

    density_15 = box_mean(buildings, KERNEL_SIZES[0])
    density_41 = box_mean(buildings, KERNEL_SIZES[1])
    height_15 = np.clip(box_mean(buildings * topology, KERNEL_SIZES[0]) / HEIGHT_NORM_M, 0.0, None)
    height_41 = np.clip(box_mean(buildings * topology, KERNEL_SIZES[1]) / HEIGHT_NORM_M, 0.0, None)
    nlos_15 = np.clip(box_mean(nlos_ground, KERNEL_SIZES[0]), 0.0, 1.0)
    nlos_41 = np.clip(box_mean(nlos_ground, KERNEL_SIZES[1]), 0.0, 1.0)
    blocker_depth = np.clip(box_mean(np.maximum(topology - RX_HEIGHT_M, 0.0) * buildings, KERNEL_SIZES[1]) / HEIGHT_NORM_M, 0.0, None)

    clearance = np.maximum(float(h_tx) - topology, 0.0).astype(np.float32) * buildings
    tx_clearance_41 = np.clip(box_mean(clearance, KERNEL_SIZES[1]) / HEIGHT_NORM_M, 0.0, None)
    taller = (topology > float(h_tx)).astype(np.float32) * buildings
    tx_below_frac_41 = np.clip(box_mean(taller, KERNEL_SIZES[1]), 0.0, 1.0)

    return {
        "theta_norm": theta_norm,
        "theta_inv": 1.0 - theta_norm,
        "h_norm": h_norm,
        "logd": _LOGD,
        "density_15": density_15,
        "density_41": density_41,
        "height_15": height_15,
        "height_41": height_41,
        "nlos_15": nlos_15,
        "nlos_41": nlos_41,
        "blocker_41": blocker_depth.astype(np.float32),
        "tx_clearance_41": tx_clearance_41.astype(np.float32),
        "tx_below_frac_41": tx_below_frac_41.astype(np.float32),
    }


def _load_try78_los_calibration(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    two_ray = payload["two_ray"]
    return {
        "height_bins_m": np.asarray(two_ray["height_bins_m"], dtype=np.float32),
        "rho": np.asarray(two_ray["rho"], dtype=np.float32),
        "phi_rad": np.asarray(two_ray["phi_rad"], dtype=np.float32),
        "bias_db": np.asarray(two_ray["bias_db"], dtype=np.float32),
        "residual_profile_db": np.asarray(two_ray.get("residual_profile_db", np.zeros((len(two_ray["rho"]), _MAX_RADIUS_PX + 1))), dtype=np.float32),
        "residual_profile_smooth_db": np.asarray(two_ray.get("residual_profile_smooth_db", two_ray.get("residual_profile_db", np.zeros((len(two_ray["rho"]), _MAX_RADIUS_PX + 1)))), dtype=np.float32),
        "residual_count": np.asarray(two_ray.get("residual_count", np.zeros((len(two_ray["rho"]), _MAX_RADIUS_PX + 1))), dtype=np.uint32),
        "global_residual_db": np.asarray(two_ray.get("global_residual_db", np.zeros(_MAX_RADIUS_PX + 1)), dtype=np.float32),
        "residual_clip_db": float(two_ray.get("residual_clip_db", 2.5)),
    }


def _load_old_calibration(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {k: np.asarray(v, dtype=np.float64) for k, v in payload["coefficients"].items()}


def _load_spread_calibration(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {k: np.asarray(v, dtype=np.float64) for k, v in payload["coefficients"].items()}


def _d_los_map_m(h_tx_m: float) -> np.ndarray:
    return np.sqrt(_D2D * _D2D + (float(h_tx_m) - RX_HEIGHT_M) ** 2).astype(np.float32)


def _d_ref_map_m(h_tx_m: float) -> np.ndarray:
    return np.sqrt(_D2D * _D2D + (float(h_tx_m) + RX_HEIGHT_M) ** 2).astype(np.float32)


def _fspl_db(h_tx_m: float) -> np.ndarray:
    d3d = _d_los_map_m(h_tx_m)
    d_km = np.maximum(d3d, 1.0) / 1000.0
    return (32.45 + 20.0 * np.log10(d_km) + 20.0 * math.log10(FREQ_GHZ * 1000.0)).astype(np.float32)


def _coherent_two_ray_correction_db(h_tx_m: float, rho: float, phi_rad: float) -> np.ndarray:
    d_los = _d_los_map_m(h_tx_m)
    d_ref = _d_ref_map_m(h_tx_m)
    ratio = np.clip(d_los / np.maximum(d_ref, 1e-6), 0.0, 2.0).astype(np.float32)
    phase = (_WAVENUMBER * (d_ref - d_los)).astype(np.float32)
    amp = np.abs(1.0 + float(rho) * ratio * np.exp(-1j * (phase + float(phi_rad))))
    return (-20.0 * np.log10(np.maximum(amp, 1e-6))).astype(np.float32)


def _interpolate_scalar(h_tx_m: float, height_bins_m: np.ndarray, values: np.ndarray) -> float:
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


def _predict_two_ray_map(h_tx_m: float, calibration: Dict[str, np.ndarray]) -> np.ndarray:
    rho = _interpolate_scalar(h_tx_m, calibration["height_bins_m"], calibration["rho"])
    phi = _interpolate_scalar(h_tx_m, calibration["height_bins_m"], calibration["phi_rad"])
    bias = _interpolate_scalar(h_tx_m, calibration["height_bins_m"], calibration["bias_db"])
    pred = _fspl_db(h_tx_m) + _coherent_two_ray_correction_db(h_tx_m, rho=rho, phi_rad=phi) + bias
    residual_profile = _interpolate_profiles(
        h_tx_m=h_tx_m,
        height_bins_m=calibration["height_bins_m"],
        radial_profile_db=calibration.get("residual_profile_smooth_db", calibration["residual_profile_db"]),
        radial_count=calibration["residual_count"],
        global_profile_db=calibration["global_residual_db"],
    )
    pred = pred + np.clip(residual_profile[_RADIUS_PX], -calibration["residual_clip_db"], calibration["residual_clip_db"])
    return np.clip(pred, PATH_LOSS_MIN_DB, PATH_LOSS_MAX_DB).astype(np.float32)


def _sample_city_type_78(topology: np.ndarray) -> str:
    ground = topology == 0.0
    non_ground = ~ground
    density = float(non_ground.mean())
    bh = topology[non_ground]
    mean_bh = float(bh.mean()) if bh.size else 0.0
    if density >= DENSITY_Q2_78 or mean_bh >= HEIGHT_Q2_78:
        return "dense_highrise"
    if density <= DENSITY_Q1_78 and mean_bh <= HEIGHT_Q1_78:
        return "open_lowrise"
    return "mixed_midrise"


def _regime_key_78(ct: str, los_label: str, ab: str) -> str:
    return f"{ct}|{los_label}|{ab}"


def _compute_formula_prior_78(los_mask: np.ndarray, h_tx: float) -> np.ndarray:
    h_tx_c = max(float(h_tx), 1.0)
    h_rx_c = max(float(RX_HEIGHT_M), 0.5)
    d2d = _D2D.astype(np.float64)
    d3d = np.sqrt(d2d ** 2 + (h_tx_c - h_rx_c) ** 2)
    d2d = np.maximum(d2d, 1.0)
    d3d = np.maximum(d3d, 1.0)

    freq_mhz = FREQ_GHZ * 1000.0
    log_f = math.log10(freq_mhz)
    fspl = 32.45 + 20.0 * np.log10(d3d / 1000.0) + 20.0 * math.log10(freq_mhz)

    a_hm = (1.1 * log_f - 0.7) * h_rx_c - (1.56 * log_f - 0.8)
    d_km = np.maximum(d2d / 1000.0, 0.001)
    hb_log = math.log10(max(h_tx_c, 1.0))
    cost231 = 46.3 + 33.9 * log_f - 13.82 * hb_log - a_hm + (44.9 - 6.55 * hb_log) * np.log10(d_km) + 3.0

    crossover = max(4.0 * math.pi * h_tx_c * h_rx_c / _WAVELENGTH_M, 1.0)
    two_ray = 40.0 * np.log10(d3d) - 20.0 * math.log10(h_tx_c) - 20.0 * math.log10(h_rx_c)
    los_path = np.where(d3d <= crossover, fspl, two_ray)

    theta_deg = np.degrees(np.arctan2(h_tx_c - h_rx_c, np.maximum(d2d, 1.0)))
    sin_theta = np.clip(np.sin(np.radians(theta_deg)), 1e-4, 1.0)
    lambda0_db = 20.0 * math.log10((4.0 * math.pi * h_tx_c * FREQ_GHZ * 1e9) / 299792458.0)
    a2g_los = lambda0_db + A2G_LOS_BIAS + A2G_LOS_LOG_COEFF * np.log10(sin_theta)
    a2g_nlos = lambda0_db + (A2G_NLOS_BIAS + A2G_NLOS_AMP * np.exp(-(90.0 - theta_deg) / A2G_NLOS_TAU))
    nlos_path = np.maximum(cost231, a2g_nlos)

    los_prob = los_mask.astype(np.float64)
    los_blend = 0.7 * los_path + 0.3 * np.minimum(los_path, a2g_los)
    prior = los_prob * los_blend + (1.0 - los_prob) * nlos_path
    return np.clip(prior, 0.0, 180.0).astype(np.float32)


def _avg_pool_78(arr: np.ndarray, k: int) -> np.ndarray:
    return box_mean(arr.astype(np.float32), k)


def _compute_pixel_features_78(topology: np.ndarray, los_mask: np.ndarray, prior_db: np.ndarray, h_tx: float) -> np.ndarray:
    ground = (topology == 0.0).astype(np.float32)
    building_mask = (1.0 - ground).astype(np.float32)
    nlos_mask_f = ((los_mask <= 0.5) & (ground > 0)).astype(np.float32)
    logd = _LOGD

    density_15 = _avg_pool_78(building_mask, KERNEL_SIZES[0])
    density_41 = _avg_pool_78(building_mask, KERNEL_SIZES[1])
    bh_sum_15 = _avg_pool_78(topology * building_mask, KERNEL_SIZES[0])
    bh_sum_41 = _avg_pool_78(topology * building_mask, KERNEL_SIZES[1])
    height_15 = np.clip(bh_sum_15 / HEIGHT_NORM_M, 0.0, None)
    height_41 = np.clip(bh_sum_41 / HEIGHT_NORM_M, 0.0, None)
    nlos_15 = np.clip(_avg_pool_78(nlos_mask_f, KERNEL_SIZES[0]), 0.0, 1.0)
    nlos_41 = np.clip(_avg_pool_78(nlos_mask_f, KERNEL_SIZES[1]), 0.0, 1.0)

    theta_deg = np.degrees(np.arctan2(max(float(h_tx), 1.0) - RX_HEIGHT_M, np.maximum(_D2D, 1.0))).astype(np.float32)
    los_prob = los_mask.astype(np.float32)
    sigma_los = SIGMA_LOS_RHO * np.power(np.clip(90.0 - theta_deg, 0.0, None), SIGMA_LOS_MU)
    sigma_nlos = SIGMA_NLOS_RHO * np.power(np.clip(90.0 - theta_deg, 0.0, None), SIGMA_NLOS_MU)
    shadow_sigma = los_prob * sigma_los + (1.0 - los_prob) * sigma_nlos
    theta_norm = np.clip(theta_deg / 90.0, 0.0, 1.0)
    bias = np.ones_like(prior_db, dtype=np.float32)
    return np.stack(
        [
            prior_db * prior_db,
            prior_db,
            logd,
            density_15,
            density_41,
            height_15,
            height_41,
            density_41 * logd,
            nlos_15,
            nlos_41,
            nlos_41 * logd,
            shadow_sigma,
            theta_norm,
            nlos_41 * theta_norm,
            bias,
        ],
        axis=-1,
    ).astype(np.float32)


def _apply_calibration_78(prior: np.ndarray, x_all: np.ndarray, los_mask: np.ndarray, ct: str, ab: str, coefs: Dict[str, np.ndarray]) -> np.ndarray:
    out = prior.copy()
    x_flat = x_all.reshape(-1, PATH_N_FEAT)
    for los_label, los_flag in (("LoS", True), ("NLoS", False)):
        region = (los_mask > 0.5) if los_flag else (los_mask <= 0.5)
        if not region.any():
            continue
        for ab_try in (ab, "mid_ant", "low_ant", "high_ant"):
            key = _regime_key_78(ct, los_label, ab_try)
            if key in coefs:
                pred_flat = (x_flat @ coefs[key]).astype(np.float32)
                pred = pred_flat.reshape(prior.shape)
                out[region] = np.clip(pred[region], PATH_LOSS_MIN_DB, PATH_LOSS_MAX_DB)
                break
    return out.astype(np.float32)


def _compute_try78_nlos_map(topology: np.ndarray, los_mask: np.ndarray, h_tx: float, coefs: Dict[str, np.ndarray]) -> np.ndarray:
    ct = _sample_city_type_78(topology)
    ab = ant_bin(h_tx)
    prior = _compute_formula_prior_78(los_mask, h_tx)
    x_all = _compute_pixel_features_78(topology, los_mask, prior, h_tx)
    return _apply_calibration_78(prior, x_all, los_mask, ct, ab, coefs)


def _build_design_matrix_79(shared: Dict[str, np.ndarray], raw_prior: np.ndarray) -> np.ndarray:
    prior_log = np.log1p(np.clip(raw_prior, 0.0, None))
    tx_clearance = shared["tx_clearance_41"]
    tx_below = shared["tx_below_frac_41"]
    bias = np.ones_like(raw_prior, dtype=np.float32)
    return np.stack(
        [
            prior_log * prior_log,
            prior_log,
            shared["logd"],
            shared["theta_norm"],
            shared["theta_inv"],
            shared["h_norm"],
            shared["h_norm"] * shared["h_norm"],
            shared["density_15"],
            shared["density_41"],
            shared["height_15"],
            shared["height_41"],
            shared["nlos_15"],
            shared["nlos_41"],
            shared["nlos_41"] * shared["logd"],
            shared["density_41"] * shared["theta_inv"],
            shared["blocker_41"],
            bias,
            tx_clearance,
            tx_below,
            shared["theta_norm"] * shared["density_41"],
            tx_clearance * shared["theta_inv"],
            tx_below * shared["density_41"],
            tx_below * shared["nlos_41"],
        ],
        axis=-1,
    ).astype(np.float32)


def _regime_key_79(metric: str, topology_class: str, los_label: str, ant_label: str) -> str:
    return f"{metric}|{topology_class}|{los_label}|{ant_label}"


def _inference_keys_79(metric: str, topology_class: str, los_label: str, ant_label: str) -> Tuple[str, ...]:
    return (
        _regime_key_79(metric, topology_class, los_label, ant_label),
        _regime_key_79(metric, topology_class, los_label, "all_ant"),
        _regime_key_79(metric, topology_class, "all_los", "all_ant"),
        _regime_key_79(metric, "global", los_label, "all_ant"),
        _regime_key_79(metric, "global", "all_los", "all_ant"),
    )


def _compute_raw_prior_79(metric: str, topology_class: str, shared: Dict[str, np.ndarray], los_mask: np.ndarray) -> np.ndarray:
    spec = METRIC_SPECS[metric]
    topo_bias = TOPOLOGY_PRIOR_BIAS_LOG[metric][topology_class]
    base_los = math.log1p(float(spec["base_los"]))
    base_nlos = math.log1p(float(spec["base_nlos"]))
    base = np.where(los_mask > 0.5, base_los, base_nlos).astype(np.float32)
    prior_log = (
        base
        + topo_bias
        + float(spec["coef_logd"]) * shared["logd"]
        + float(spec["coef_theta_inv"]) * shared["theta_inv"]
        + float(spec["coef_density"]) * shared["density_41"]
        + float(spec["coef_height"]) * shared["height_41"]
        + float(spec["coef_nlos"]) * shared["nlos_41"]
        + float(spec["coef_interaction"]) * shared["nlos_41"] * shared["theta_inv"]
    )
    prior = np.expm1(np.clip(prior_log, 0.0, 8.0))
    return np.clip(prior, 0.0, float(spec["clip_hi"])).astype(np.float32)


def _apply_calibration_79(metric: str, topology_class: str, ant_label: str, los_mask: np.ndarray, raw_prior: np.ndarray, x_all: np.ndarray, coefs: Dict[str, np.ndarray]) -> np.ndarray:
    pred_log = np.log1p(np.clip(raw_prior, 0.0, None)).astype(np.float32)
    x_flat = x_all.reshape(-1, x_all.shape[-1]).astype(np.float64)
    los_region = los_mask > 0.5
    nlos_region = los_mask <= 0.5
    for los_label, region in (("LoS", los_region), ("NLoS", nlos_region)):
        if not np.any(region):
            continue
        for key in _inference_keys_79(metric, topology_class, los_label, ant_label):
            if key in coefs:
                reg_pred = (x_flat @ coefs[key]).reshape(raw_prior.shape).astype(np.float32)
                pred_log[region] = reg_pred[region]
                break
    pred = np.expm1(pred_log).astype(np.float32)
    pred = np.where(los_region, np.clip(pred, 0.0, LOS_CLIP_HI[metric]), pred)
    pred = np.where(nlos_region, np.clip(pred, 0.0, NLOS_CLIP_HI[metric]), pred)
    return np.clip(pred, 0.0, max(LOS_CLIP_HI[metric], NLOS_CLIP_HI[metric])).astype(np.float32)


def _compute_try79_metric_prior(
    metric: str,
    topology_class: str,
    ant_label: str,
    los_mask: np.ndarray,
    shared: Dict[str, np.ndarray],
    coefs: Dict[str, np.ndarray],
) -> np.ndarray:
    raw_prior = _compute_raw_prior_79(metric, topology_class, shared, los_mask)
    x_all = _build_design_matrix_79(shared, raw_prior)
    return _apply_calibration_79(metric, topology_class, ant_label, los_mask, raw_prior, x_all, coefs)
