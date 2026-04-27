"""Evaluate the final unified Try 78 hybrid model.

- LoS from the refined coherent two-ray prior in ``prior_try78.py``
- NLoS from the old regime-calibrated model, now embedded locally here

This keeps the best-performing Try 78 hybrid while removing the runtime
dependency on ``old_try_78_with_nlos`` so that folder can be deleted after
validation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
from scipy.ndimage import uniform_filter

import prior_try78 as los_model

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

try:
    import torch_directml
except Exception:
    torch_directml = None


ROOT = Path(__file__).resolve().parent
CALIB_DIR = ROOT / "final_calibrations"


# ---------------------------------------------------------------------------
# Embedded old Try 78 NLoS calibration pieces
# ---------------------------------------------------------------------------

IMG_SIZE = 513
TX_ROW = 256
TX_COL = 256
METERS_PER_PIXEL = 1.0
FREQ_GHZ = 7.125
RX_HEIGHT_M = 1.5
PATH_LOSS_MIN_DB = 20.0
HEIGHT_SCALE = 90.0
KERNEL_SIZES = (15, 41)

SIGMA_LOS_RHO = 0.0272
SIGMA_LOS_MU = 0.7475
SIGMA_NLOS_RHO = 2.3197
SIGMA_NLOS_MU = 0.2361

A2G_Nlos_BIAS = -16.16
A2G_NLoS_AMP = 12.0436
A2G_NLoS_TAU = 7.52
A2G_LOS_LOG_COEFF = -20.0
A2G_LOS_BIAS = 0.0

DENSITY_Q1 = 0.1957
DENSITY_Q2 = 0.2549
HEIGHT_Q1 = 10.91
HEIGHT_Q2 = 15.95
ANT_Q1 = 58.12
ANT_Q2 = 103.85

FEATURE_NAMES = (
    "prior_sq", "prior", "log1p_d2d",
    "density_15", "density_41",
    "height_15", "height_41",
    "density_41_x_logd",
    "nlos_15", "nlos_41",
    "nlos_41_x_logd",
    "shadow_sigma",
    "theta_norm",
    "nlos_41_x_theta",
    "bias",
)
N_FEAT = len(FEATURE_NAMES)


def _build_d2d_map() -> np.ndarray:
    ii, jj = np.indices((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    dy = (ii - TX_ROW) * METERS_PER_PIXEL
    dx = (jj - TX_COL) * METERS_PER_PIXEL
    return np.sqrt(dy * dy + dx * dx).astype(np.float32)


_D2D = _build_d2d_map()
_TORCH_D2D_CACHE: Dict[str, "torch.Tensor"] = {}


def load_old_calibration(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(path.read_text())
    return {k: np.asarray(v, dtype=np.float64) for k, v in payload["coefficients"].items()}


def sample_city_type(topology: np.ndarray) -> str:
    ground = topology == 0.0
    non_ground = ~ground
    density = float(non_ground.mean())
    bh = topology[non_ground]
    mean_bh = float(bh.mean()) if bh.size else 0.0
    if density >= DENSITY_Q2 or mean_bh >= HEIGHT_Q2:
        return "dense_highrise"
    if density <= DENSITY_Q1 and mean_bh <= HEIGHT_Q1:
        return "open_lowrise"
    return "mixed_midrise"


def ant_bin(h_tx: float) -> str:
    if h_tx <= ANT_Q1:
        return "low_ant"
    if h_tx <= ANT_Q2:
        return "mid_ant"
    return "high_ant"


def regime_key(ct: str, los_label: str, ab: str) -> str:
    return f"{ct}|{los_label}|{ab}"


def compute_formula_prior(
    los_mask: np.ndarray,
    h_tx: float,
    freq_ghz: float = FREQ_GHZ,
    rx_h: float = RX_HEIGHT_M,
) -> np.ndarray:
    h_tx_c = max(float(h_tx), 1.0)
    h_rx_c = max(float(rx_h), 0.5)
    d2d = _D2D.astype(np.float64)
    d3d = np.sqrt(d2d ** 2 + (h_tx_c - h_rx_c) ** 2)
    d2d = np.maximum(d2d, 1.0)
    d3d = np.maximum(d3d, 1.0)

    freq_mhz = freq_ghz * 1000.0
    log_f = math.log10(freq_mhz)

    fspl = 32.45 + 20.0 * np.log10(d3d / 1000.0) + 20.0 * math.log10(freq_mhz)

    a_hm = (1.1 * log_f - 0.7) * h_rx_c - (1.56 * log_f - 0.8)
    d_km = np.maximum(d2d / 1000.0, 0.001)
    hb_log = math.log10(max(h_tx_c, 1.0))
    cost231 = (
        46.3 + 33.9 * log_f - 13.82 * hb_log - a_hm
        + (44.9 - 6.55 * hb_log) * np.log10(d_km)
        + 3.0
    )

    wavelength = 0.299792458 / freq_ghz
    crossover = max(4.0 * math.pi * h_tx_c * h_rx_c / wavelength, 1.0)
    two_ray = 40.0 * np.log10(d3d) - 20.0 * math.log10(h_tx_c) - 20.0 * math.log10(h_rx_c)
    los_path = np.where(d3d <= crossover, fspl, two_ray)

    theta_deg = np.degrees(np.arctan2(h_tx_c - h_rx_c, np.maximum(d2d, 1.0)))
    sin_theta = np.clip(np.sin(np.radians(theta_deg)), 1e-4, 1.0)
    lambda0_db = 20.0 * math.log10((4.0 * math.pi * h_tx_c * freq_ghz * 1e9) / 299792458.0)
    a2g_los = lambda0_db + A2G_LOS_BIAS + A2G_LOS_LOG_COEFF * np.log10(sin_theta)
    a2g_nlos = lambda0_db + (A2G_Nlos_BIAS + A2G_NLoS_AMP * np.exp(-(90.0 - theta_deg) / A2G_NLoS_TAU))
    nlos_path = np.maximum(cost231, a2g_nlos)

    los_prob = los_mask.astype(np.float64)
    los_blend = 0.7 * los_path + 0.3 * np.minimum(los_path, a2g_los)
    prior = los_prob * los_blend + (1.0 - los_prob) * nlos_path
    return np.clip(prior, 0.0, 180.0).astype(np.float32)


def _avg_pool(arr: np.ndarray, k: int) -> np.ndarray:
    size = k if k % 2 == 1 else k + 1
    return uniform_filter(arr.astype(np.float32), size=size, mode="reflect")


def compute_pixel_features(
    topology: np.ndarray,
    los_mask: np.ndarray,
    prior_db: np.ndarray,
    h_tx: float,
) -> np.ndarray:
    ground = (topology == 0.0).astype(np.float32)
    building_mask = (1.0 - ground).astype(np.float32)
    nlos_mask_f = ((los_mask <= 0.5) & (ground > 0)).astype(np.float32)

    d2d = _D2D.astype(np.float32)
    logd = np.log1p(d2d)

    density_15 = _avg_pool(building_mask, KERNEL_SIZES[0])
    density_41 = _avg_pool(building_mask, KERNEL_SIZES[1])
    bh_sum_15 = _avg_pool(topology * building_mask, KERNEL_SIZES[0])
    bh_sum_41 = _avg_pool(topology * building_mask, KERNEL_SIZES[1])
    height_15 = np.clip(bh_sum_15 / HEIGHT_SCALE, 0.0, None)
    height_41 = np.clip(bh_sum_41 / HEIGHT_SCALE, 0.0, None)

    nlos_15 = np.clip(_avg_pool(nlos_mask_f, KERNEL_SIZES[0]), 0.0, 1.0)
    nlos_41 = np.clip(_avg_pool(nlos_mask_f, KERNEL_SIZES[1]), 0.0, 1.0)

    h_tx_c = max(float(h_tx), 1.0)
    theta_deg = np.degrees(np.arctan2(h_tx_c - RX_HEIGHT_M, np.maximum(d2d, 1.0))).astype(np.float32)
    los_prob = los_mask.astype(np.float32)
    sigma_los = SIGMA_LOS_RHO * np.power(np.clip(90.0 - theta_deg, 0.0, None), SIGMA_LOS_MU)
    sigma_nlos = SIGMA_NLOS_RHO * np.power(np.clip(90.0 - theta_deg, 0.0, None), SIGMA_NLOS_MU)
    shadow_sigma = los_prob * sigma_los + (1.0 - los_prob) * sigma_nlos
    theta_norm = np.clip(theta_deg / 90.0, 0.0, 1.0)

    bias = np.ones_like(prior_db)
    return np.stack([
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
    ], axis=-1).astype(np.float32)


def apply_calibration(
    prior: np.ndarray,
    x_all: np.ndarray,
    los_label_map: np.ndarray,
    ct: str,
    ab: str,
    coefs: Dict[str, np.ndarray],
) -> np.ndarray:
    out = prior.copy()
    x_flat = x_all.reshape(-1, N_FEAT)
    for los_label, los_flag in (("LoS", True), ("NLoS", False)):
        region = (los_label_map > 0) if los_flag else (los_label_map == 0)
        if not region.any():
            continue
        for ab_try in (ab, "mid_ant", "low_ant", "high_ant"):
            key = regime_key(ct, los_label, ab_try)
            if key in coefs:
                pred_flat = (x_flat @ coefs[key]).astype(np.float32)
                pred = pred_flat.reshape(prior.shape)
                out[region] = np.clip(pred[region], PATH_LOSS_MIN_DB, 180.0)
                break
    return out


def load_hybrid_sample(handle: h5py.File, ref: los_model.SampleRef) -> Dict[str, np.ndarray]:
    grp = handle[ref.city][ref.sample]
    topology = np.asarray(grp["topology_map"][...], dtype=np.float32)
    los_mask = np.asarray(grp["los_mask"][...], dtype=np.uint8)
    path_loss = np.asarray(grp["path_loss"][...], dtype=np.float32)
    ground = topology == 0.0
    valid = ground & np.isfinite(path_loss) & (path_loss >= PATH_LOSS_MIN_DB)
    return {
        "topology": topology,
        "los_mask": los_mask,
        "path_loss": path_loss,
        "ground": ground,
        "valid": valid,
    }


def rmse_from_sse(sse: float, n: int) -> float:
    return math.sqrt(sse / n) if n else float("nan")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_progress_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def append_progress_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def build_progress_payload(
    *,
    status: str,
    started_at: str,
    device_name: str,
    out_dir: Path,
    n_fit_samples: int,
    n_eval_samples: int,
    processed_eval_samples: int,
    fspl_los_sse: float,
    radial_los_sse: float,
    two_los_sse: float,
    hybrid_all_sse: float,
    hybrid_los_sse: float,
    hybrid_nlos_sse: float,
    n_all: int,
    n_los: int,
    n_nlos: int,
    last_sample: Optional[Dict] = None,
    error: Optional[str] = None,
) -> Dict:
    aggregate = {
        "fspl_rmse_los_pw": rmse_from_sse(fspl_los_sse, n_los),
        "radial_rmse_los_pw": rmse_from_sse(radial_los_sse, n_los),
        "two_ray_rmse_los_pw": rmse_from_sse(two_los_sse, n_los),
        "hybrid_rmse_los_pw": rmse_from_sse(hybrid_los_sse, n_los),
        "hybrid_rmse_nlos_pw": rmse_from_sse(hybrid_nlos_sse, n_nlos),
        "hybrid_rmse_overall_pw": rmse_from_sse(hybrid_all_sse, n_all),
        "total_valid_pixels": n_all,
        "total_los_pixels": n_los,
        "total_nlos_pixels": n_nlos,
    }
    return {
        "status": status,
        "started_at_utc": started_at,
        "updated_at_utc": utc_now_iso(),
        "device": device_name,
        "out_dir": str(out_dir),
        "n_fit_samples": n_fit_samples,
        "n_eval_samples": n_eval_samples,
        "processed_eval_samples": processed_eval_samples,
        "progress_ratio": (processed_eval_samples / n_eval_samples) if n_eval_samples else 0.0,
        "aggregate_so_far": aggregate,
        "last_sample": last_sample,
        "error": error,
    }


def resolve_device(device_name: str):
    if device_name == "cpu":
        return None
    if torch is None:
        if device_name == "auto":
            return None
        raise RuntimeError("torch is not available, so --device dml cannot be used")
    if device_name == "dml":
        if torch_directml is None:
            raise RuntimeError("torch-directml is not installed in this Python environment")
        return torch_directml.device()
    if device_name == "auto":
        if torch_directml is not None:
            try:
                return torch_directml.device()
            except Exception:
                return None
        return None
    raise ValueError(f"unknown device: {device_name}")


def _torch_d2d(device) -> "torch.Tensor":
    key = str(device)
    cached = _TORCH_D2D_CACHE.get(key)
    if cached is None:
        _TORCH_D2D_CACHE[key] = torch.from_numpy(_D2D.astype(np.float32)).to(device)
        cached = _TORCH_D2D_CACHE[key]
    return cached


def _avg_pool_torch(arr: "torch.Tensor", k: int) -> "torch.Tensor":
    pad = k // 2
    x = arr.unsqueeze(0).unsqueeze(0)
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x, kernel_size=k, stride=1).squeeze(0).squeeze(0)


def predict_nlos_map_torch(
    sample: Dict[str, np.ndarray],
    h_tx: float,
    ct: str,
    ab: str,
    coefs: Dict[str, np.ndarray],
    device,
) -> np.ndarray:
    d2d = _torch_d2d(device)
    topology = torch.from_numpy(sample["topology"].astype(np.float32)).to(device)
    los_mask = torch.from_numpy(sample["los_mask"].astype(np.float32)).to(device)

    h_tx_c = max(float(h_tx), 1.0)
    h_rx_c = max(float(RX_HEIGHT_M), 0.5)
    freq_mhz = FREQ_GHZ * 1000.0
    log_f = math.log10(freq_mhz)

    d3d = torch.sqrt(d2d * d2d + (h_tx_c - h_rx_c) ** 2)
    d2d_safe = torch.clamp(d2d, min=1.0)
    d3d_safe = torch.clamp(d3d, min=1.0)
    fspl = 32.45 + 20.0 * torch.log10(d3d_safe / 1000.0) + 20.0 * math.log10(freq_mhz)

    a_hm = (1.1 * log_f - 0.7) * h_rx_c - (1.56 * log_f - 0.8)
    d_km = torch.clamp(d2d_safe / 1000.0, min=0.001)
    hb_log = math.log10(max(h_tx_c, 1.0))
    cost231 = (
        46.3 + 33.9 * log_f - 13.82 * hb_log - a_hm
        + (44.9 - 6.55 * hb_log) * torch.log10(d_km)
        + 3.0
    )

    wavelength = 0.299792458 / FREQ_GHZ
    crossover = max(4.0 * math.pi * h_tx_c * h_rx_c / wavelength, 1.0)
    two_ray = 40.0 * torch.log10(d3d_safe) - 20.0 * math.log10(h_tx_c) - 20.0 * math.log10(h_rx_c)
    los_path = torch.where(d3d_safe <= crossover, fspl, two_ray)

    theta_deg = torch.rad2deg(torch.atan2(torch.full_like(d2d_safe, h_tx_c - h_rx_c), d2d_safe))
    sin_theta = torch.clamp(torch.sin(torch.deg2rad(theta_deg)), min=1e-4, max=1.0)
    lambda0_db = 20.0 * math.log10((4.0 * math.pi * h_tx_c * FREQ_GHZ * 1e9) / 299792458.0)
    a2g_los = lambda0_db + A2G_LOS_BIAS + A2G_LOS_LOG_COEFF * torch.log10(sin_theta)
    a2g_nlos = lambda0_db + (A2G_Nlos_BIAS + A2G_NLoS_AMP * torch.exp(-(90.0 - theta_deg) / A2G_NLoS_TAU))
    nlos_path = torch.maximum(cost231, a2g_nlos)

    los_blend = 0.7 * los_path + 0.3 * torch.minimum(los_path, a2g_los)
    prior = torch.clamp(los_mask * los_blend + (1.0 - los_mask) * nlos_path, min=0.0, max=180.0)

    ground = (topology == 0.0).to(torch.float32)
    building_mask = 1.0 - ground
    nlos_mask_f = ((los_mask <= 0.5) & (ground > 0)).to(torch.float32)
    logd = torch.log1p(d2d)

    density_15 = _avg_pool_torch(building_mask, KERNEL_SIZES[0])
    density_41 = _avg_pool_torch(building_mask, KERNEL_SIZES[1])
    bh_sum_15 = _avg_pool_torch(topology * building_mask, KERNEL_SIZES[0])
    bh_sum_41 = _avg_pool_torch(topology * building_mask, KERNEL_SIZES[1])
    height_15 = torch.clamp(bh_sum_15 / HEIGHT_SCALE, min=0.0)
    height_41 = torch.clamp(bh_sum_41 / HEIGHT_SCALE, min=0.0)
    nlos_15 = torch.clamp(_avg_pool_torch(nlos_mask_f, KERNEL_SIZES[0]), min=0.0, max=1.0)
    nlos_41 = torch.clamp(_avg_pool_torch(nlos_mask_f, KERNEL_SIZES[1]), min=0.0, max=1.0)

    sigma_los = SIGMA_LOS_RHO * torch.pow(torch.clamp(90.0 - theta_deg, min=0.0), SIGMA_LOS_MU)
    sigma_nlos = SIGMA_NLOS_RHO * torch.pow(torch.clamp(90.0 - theta_deg, min=0.0), SIGMA_NLOS_MU)
    shadow_sigma = los_mask * sigma_los + (1.0 - los_mask) * sigma_nlos
    theta_norm = torch.clamp(theta_deg / 90.0, min=0.0, max=1.0)
    bias = torch.ones_like(prior)

    x_all = torch.stack([
        prior * prior,
        prior,
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
    ], dim=-1)
    x_flat = x_all.reshape(-1, N_FEAT)

    out = prior.clone()
    region = los_mask <= 0.5
    for ab_try in (ab, "mid_ant", "low_ant", "high_ant"):
        key = regime_key(ct, "NLoS", ab_try)
        if key in coefs:
            w = torch.from_numpy(coefs[key].astype(np.float32)).to(device)
            pred = torch.matmul(x_flat, w).reshape(prior.shape)
            pred = torch.clamp(pred, min=PATH_LOSS_MIN_DB, max=180.0)
            out = torch.where(region, pred, out)
            break

    return out.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, default=Path("c:/TFG/TFGpractice/Datasets/CKM_Dataset_270326.h5"))
    parser.add_argument("--new-calibration-json", type=Path, default=CALIB_DIR / "los_two_ray_calibration.json")
    parser.add_argument("--old-calibration-json", type=Path, default=CALIB_DIR / "nlos_regime_calibration.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "hybrid_out_final")
    parser.add_argument("--eval-ratio", type=float, default=0.30)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cpu", "dml"), default="auto")
    parser.add_argument("--progress-json", type=Path, default=None)
    parser.add_argument("--progress-log", type=Path, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    device_name = str(device) if device is not None else "cpu"
    progress_json = args.progress_json or (args.out_dir / "progress.json")
    progress_log = args.progress_log or (args.out_dir / "progress.out")
    started_at = utc_now_iso()

    refs = los_model.enumerate_samples(args.hdf5)
    refs = los_model.subsample_refs(refs, args.max_samples, seed=args.seed)
    fit_refs, eval_refs = los_model.split_city_holdout(refs, eval_ratio=args.eval_ratio, split_seed=args.split_seed)

    print(f"[try78-hybrid] loading new calibration from {args.new_calibration_json}")
    radial_calibration, two_ray_calibration = los_model.load_calibration(args.new_calibration_json)
    print(f"[try78-hybrid] loading embedded-old NLoS calibration from {args.old_calibration_json}")
    old_coefs = load_old_calibration(args.old_calibration_json)
    print(f"[try78-hybrid] execution device: {device_name}")
    append_progress_log(progress_log, f"{utc_now_iso()} [start] device={device_name} fit={len(fit_refs)} eval={len(eval_refs)}")
    write_progress_json(
        progress_json,
        build_progress_payload(
            status="running",
            started_at=started_at,
            device_name=device_name,
            out_dir=args.out_dir,
            n_fit_samples=len(fit_refs),
            n_eval_samples=len(eval_refs),
            processed_eval_samples=0,
            fspl_los_sse=0.0,
            radial_los_sse=0.0,
            two_los_sse=0.0,
            hybrid_all_sse=0.0,
            hybrid_los_sse=0.0,
            hybrid_nlos_sse=0.0,
            n_all=0,
            n_los=0,
            n_nlos=0,
        ),
    )

    fspl_los_sse = 0.0
    radial_los_sse = 0.0
    two_los_sse = 0.0
    hybrid_all_sse = 0.0
    hybrid_los_sse = 0.0
    hybrid_nlos_sse = 0.0
    n_los = 0
    n_nlos = 0
    n_all = 0
    per_sample = []

    try:
        with h5py.File(str(args.hdf5), "r") as handle:
            for idx, ref in enumerate(eval_refs, start=1):
                sample = load_hybrid_sample(handle, ref)
                if not sample["valid"].any():
                    continue

                ct = sample_city_type(sample["topology"])
                ab = ant_bin(ref.uav_height_m)

                fspl_map = los_model.fspl_db(ref.uav_height_m)
                radial_map = los_model.predict_radial_map(ref.uav_height_m, radial_calibration)
                two_ray_map = los_model.predict_two_ray_map(ref.uav_height_m, two_ray_calibration)

                hybrid_map = two_ray_map.copy()
                if device is None:
                    old_prior = compute_formula_prior(sample["los_mask"], ref.uav_height_m)
                    x_all = compute_pixel_features(sample["topology"], sample["los_mask"], old_prior, ref.uav_height_m)
                    nlos_map = apply_calibration(old_prior, x_all, sample["los_mask"], ct, ab, old_coefs)
                else:
                    nlos_map = predict_nlos_map_torch(sample, ref.uav_height_m, ct, ab, old_coefs, device)
                hybrid_map[sample["los_mask"] == 0] = nlos_map[sample["los_mask"] == 0]

                target = sample["path_loss"]
                valid = sample["valid"]
                los_mask = valid & (sample["los_mask"] > 0)
                nlos_mask = valid & (sample["los_mask"] == 0)

                def accum(pred: np.ndarray, mask: np.ndarray) -> Tuple[float, int]:
                    if not mask.any():
                        return 0.0, 0
                    err = pred[mask] - target[mask]
                    return float(np.sum(err * err)), int(mask.sum())

                sse, cnt = accum(fspl_map, los_mask)
                fspl_los_sse += sse
                n_los += cnt

                sse, _ = accum(radial_map, los_mask)
                radial_los_sse += sse

                sse, _ = accum(two_ray_map, los_mask)
                two_los_sse += sse

                sse, cnt = accum(hybrid_map, valid)
                hybrid_all_sse += sse
                n_all += cnt

                sse, _ = accum(hybrid_map, los_mask)
                hybrid_los_sse += sse

                sse, cnt = accum(hybrid_map, nlos_mask)
                hybrid_nlos_sse += sse
                n_nlos += cnt

                last_sample = None
                if los_mask.any() or nlos_mask.any():
                    last_sample = {
                        "city": ref.city,
                        "sample": ref.sample,
                        "uav_height_m": ref.uav_height_m,
                        "n_los": int(los_mask.sum()),
                        "n_nlos": int(nlos_mask.sum()),
                        "two_ray_los_rmse": rmse_from_sse(*accum(two_ray_map, los_mask)),
                        "hybrid_overall_rmse": rmse_from_sse(*accum(hybrid_map, valid)),
                        "hybrid_nlos_rmse": rmse_from_sse(*accum(hybrid_map, nlos_mask)),
                    }
                    per_sample.append(last_sample)

                if idx % args.log_every == 0:
                    message = (
                        f"{utc_now_iso()} [progress] eval={idx}/{len(eval_refs)} "
                        f"los_2ray={rmse_from_sse(two_los_sse, n_los):.3f}dB "
                        f"hybrid_overall={rmse_from_sse(hybrid_all_sse, n_all):.3f}dB"
                    )
                    print(
                        f"[try78-hybrid] eval [{idx}/{len(eval_refs)}] "
                        f"LoS-2ray={rmse_from_sse(two_los_sse, n_los):.3f} dB  "
                        f"hybrid-overall={rmse_from_sse(hybrid_all_sse, n_all):.3f} dB"
                    )
                    append_progress_log(progress_log, message)
                    write_progress_json(
                        progress_json,
                        build_progress_payload(
                            status="running",
                            started_at=started_at,
                            device_name=device_name,
                            out_dir=args.out_dir,
                            n_fit_samples=len(fit_refs),
                            n_eval_samples=len(eval_refs),
                            processed_eval_samples=idx,
                            fspl_los_sse=fspl_los_sse,
                            radial_los_sse=radial_los_sse,
                            two_los_sse=two_los_sse,
                            hybrid_all_sse=hybrid_all_sse,
                            hybrid_los_sse=hybrid_los_sse,
                            hybrid_nlos_sse=hybrid_nlos_sse,
                            n_all=n_all,
                            n_los=n_los,
                            n_nlos=n_nlos,
                            last_sample=last_sample,
                        ),
                    )
    except Exception as exc:
        append_progress_log(progress_log, f"{utc_now_iso()} [error] {exc}")
        write_progress_json(
            progress_json,
            build_progress_payload(
                status="error",
                started_at=started_at,
                device_name=device_name,
                out_dir=args.out_dir,
                n_fit_samples=len(fit_refs),
                n_eval_samples=len(eval_refs),
                processed_eval_samples=len(per_sample),
                fspl_los_sse=fspl_los_sse,
                radial_los_sse=radial_los_sse,
                two_los_sse=two_los_sse,
                hybrid_all_sse=hybrid_all_sse,
                hybrid_los_sse=hybrid_los_sse,
                hybrid_nlos_sse=hybrid_nlos_sse,
                n_all=n_all,
                n_los=n_los,
                n_nlos=n_nlos,
                last_sample=per_sample[-1] if per_sample else None,
                error=str(exc),
            ),
        )
        raise

    summary = {
        "n_fit_samples": len(fit_refs),
        "n_eval_samples": len(eval_refs),
        "aggregate": {
            "fspl_rmse_los_pw": rmse_from_sse(fspl_los_sse, n_los),
            "radial_rmse_los_pw": rmse_from_sse(radial_los_sse, n_los),
            "two_ray_rmse_los_pw": rmse_from_sse(two_los_sse, n_los),
            "hybrid_rmse_los_pw": rmse_from_sse(hybrid_los_sse, n_los),
            "hybrid_rmse_nlos_pw": rmse_from_sse(hybrid_nlos_sse, n_nlos),
            "hybrid_rmse_overall_pw": rmse_from_sse(hybrid_all_sse, n_all),
            "total_valid_pixels": n_all,
            "total_los_pixels": n_los,
            "total_nlos_pixels": n_nlos,
        },
        "per_sample": per_sample,
    }

    out_path = args.out_dir / "hybrid_eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    append_progress_log(progress_log, f"{utc_now_iso()} [done] summary={out_path}")
    write_progress_json(
        progress_json,
        build_progress_payload(
            status="done",
            started_at=started_at,
            device_name=device_name,
            out_dir=args.out_dir,
            n_fit_samples=len(fit_refs),
            n_eval_samples=len(eval_refs),
            processed_eval_samples=len(eval_refs),
            fspl_los_sse=fspl_los_sse,
            radial_los_sse=radial_los_sse,
            two_los_sse=two_los_sse,
            hybrid_all_sse=hybrid_all_sse,
            hybrid_los_sse=hybrid_los_sse,
            hybrid_nlos_sse=hybrid_nlos_sse,
            n_all=n_all,
            n_los=n_los,
            n_nlos=n_nlos,
            last_sample=per_sample[-1] if per_sample else None,
        ),
    )
    print(f"[try78-hybrid] summary -> {out_path}")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
