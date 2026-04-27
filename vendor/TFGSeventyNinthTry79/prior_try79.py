"""Try 79 - pure-numpy non-DL baseline for delay spread and angular spread.

This try mirrors the "prior + calibration" spirit of Try 78, but targets the
small-scale spread maps used in Try 58 / 59 / 77.

Pipeline:

1. Build a lightweight log-domain spread prior from observable geometry:
   distance, elevation angle, topology class, local building density/height,
   local NLoS support, and per-pixel UAV-to-skyline geometry.
2. Extract multiscale morphology features with pure numpy box filters
   (no torch, no scipy, no deep learning).
3. Fit regime-wise ridge regressors in log1p(spread) space:
      metric x topology_class x LoS/NLoS x antenna-height-bin
   with broad fallback regimes for sparse cases.
4. Apply LoS-specific output clamps: angular spread in LoS regions is
   clamped to [0, 15] deg because the direct ray dominates and the GT
   spike-at-zero distribution is not informative above that.
   See ``LOS_ANGULAR_SPREAD_NOTE.md`` for the rationale.
5. Evaluate on a city-holdout split matching the Try 77 semantics.

Height awareness: UAV height is used both as a regime key (``ant_bin``) and
as three per-pixel geometry features (``tx_clearance_41``,
``tx_below_frac_41``, and their cross-products with ``theta_inv``,
``density_41``, ``nlos_41``). The older scalar ``h_norm * feature``
interactions were removed - they were redundant with ``ant_bin`` keying.

The model is intentionally simple and transparent. It is meant as a strong
non-DL baseline, not as a replacement for the distribution-first Try 77 model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np

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


IMG_SIZE = 513
TX_ROW = 256
TX_COL = 256
RX_HEIGHT_M = 1.5
METERS_PER_PIXEL = 1.0
HEIGHT_NORM_M = 90.0
DEFAULT_HDF5 = Path("c:/TFG/TFGpractice/Datasets/CKM_Dataset_270326.h5")
RIDGE_LAMBDA = 1e-3
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

METRIC_SPECS: Dict[str, Dict[str, float | str]] = {
    "delay_spread": {
        "field": "delay_spread",
        "unit": "ns",
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
        "field": "angular_spread",
        "unit": "deg",
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

FEATURE_NAMES = (
    "prior_log_sq",
    "prior_log",
    "log1p_d2d",
    "theta_norm",
    "theta_inv",
    "h_norm",
    "h_norm_sq",
    "density_15",
    "density_41",
    "height_15",
    "height_41",
    "nlos_15",
    "nlos_41",
    "nlos_41_x_logd",
    "density_41_x_theta_inv",
    "blocker_41",
    "bias",
    "tx_clearance_41",
    "tx_below_frac_41",
    "theta_x_density_41",
    "tx_clearance_x_theta_inv",
    "tx_below_frac_x_density_41",
    "tx_below_frac_x_nlos_41",
)
N_FEAT = len(FEATURE_NAMES)

# LoS-specific output clamp for angular spread. In LoS the direct ray
# dominates and the GT angular spread is spike-like near zero; the
# 0..90 deg headroom (NLOS_CLIP_HI) would amplify rare spike outliers
# in LoS regions. See LOS_ANGULAR_SPREAD_NOTE.md.
LOS_CLIP_HI = {
    "angular_spread": 15.0,
    "delay_spread": float(METRIC_SPECS["delay_spread"]["clip_hi"]),
}
NLOS_CLIP_HI = {
    "angular_spread": float(METRIC_SPECS["angular_spread"]["clip_hi"]),
    "delay_spread": float(METRIC_SPECS["delay_spread"]["clip_hi"]),
}


def _d2d_map() -> np.ndarray:
    ii, jj = np.indices((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    dy = (ii - TX_ROW) * METERS_PER_PIXEL
    dx = (jj - TX_COL) * METERS_PER_PIXEL
    return np.sqrt(dy * dy + dx * dx).astype(np.float32)


_D2D = _d2d_map()
_LOGD = np.log1p(_D2D).astype(np.float32)
_TORCH_D2D_CACHE: Dict[str, "torch.Tensor"] = {}
_TORCH_LOGD_CACHE: Dict[str, "torch.Tensor"] = {}


def box_mean(arr: np.ndarray, k: int) -> np.ndarray:
    """Pure-numpy same-size box average with reflect padding."""
    if k <= 1:
        return arr.astype(np.float32, copy=True)
    pad = k // 2
    padded = np.pad(arr.astype(np.float32), ((pad, pad), (pad, pad)), mode="reflect")
    integ = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    out = integ[k:, k:] - integ[:-k, k:] - integ[k:, :-k] + integ[:-k, :-k]
    return (out / float(k * k)).astype(np.float32)


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
        cached = torch.from_numpy(_D2D.astype(np.float32)).to(device)
        _TORCH_D2D_CACHE[key] = cached
    return cached


def _torch_logd(device) -> "torch.Tensor":
    key = str(device)
    cached = _TORCH_LOGD_CACHE.get(key)
    if cached is None:
        cached = torch.from_numpy(_LOGD.astype(np.float32)).to(device)
        _TORCH_LOGD_CACHE[key] = cached
    return cached


def _avg_pool_torch(arr: "torch.Tensor", k: int) -> "torch.Tensor":
    if k <= 1:
        return arr
    pad = k // 2
    x = arr.unsqueeze(0).unsqueeze(0)
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x, kernel_size=k, stride=1).squeeze(0).squeeze(0)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_progress_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def write_progress_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_progress_payload(
    *,
    status: str,
    started_at: str,
    device_name: str,
    phase: str,
    out_dir: Path,
    metrics: Sequence[str],
    n_train_samples: int,
    n_eval_samples: int,
    processed_samples: int,
    n_regimes_fitted: int,
    eval_results: Optional[Dict[str, Dict[str, object]]] = None,
    error: Optional[str] = None,
) -> Dict[str, object]:
    return {
        "status": status,
        "started_at_utc": started_at,
        "updated_at_utc": utc_now_iso(),
        "device": device_name,
        "phase": phase,
        "out_dir": str(out_dir),
        "metrics": list(metrics),
        "n_train_samples": n_train_samples,
        "n_eval_samples": n_eval_samples,
        "processed_samples": processed_samples,
        "progress_ratio": (processed_samples / n_train_samples) if phase == "fit" and n_train_samples else ((processed_samples / n_eval_samples) if phase == "eval" and n_eval_samples else 0.0),
        "n_regimes_fitted": n_regimes_fitted,
        "aggregate_so_far": {
            metric: metric_payload.get("aggregate", {})
            for metric, metric_payload in (eval_results or {}).items()
        },
        "error": error,
    }


def read_field(grp: h5py.Group, name: str) -> np.ndarray:
    if name in grp:
        return np.asarray(grp[name][...], dtype=np.float32)
    alts = [name + "_map", name.replace("_", "") + "_map"]
    for alt in alts:
        if alt in grp:
            return np.asarray(grp[alt][...], dtype=np.float32)
    raise KeyError(f"Missing field {name!r} in sample with keys {list(grp.keys())}")


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


def ant_bin(h_tx: float) -> str:
    if h_tx <= ANT_Q1:
        return "low_ant"
    if h_tx <= ANT_Q2:
        return "mid_ant"
    return "high_ant"


@dataclass
class SampleRef:
    city: str
    sample: str
    uav_height_m: float


@dataclass
class SampleArrays:
    topology: np.ndarray
    los_mask: np.ndarray
    ground: np.ndarray
    topology_class: str
    targets: Dict[str, np.ndarray]
    valid_masks: Dict[str, np.ndarray]


def enumerate_samples(hdf5_path: Path, max_samples: Optional[int] = None) -> List[SampleRef]:
    refs: List[SampleRef] = []
    with h5py.File(str(hdf5_path), "r") as handle:
        for city in sorted(handle.keys()):
            for sample in sorted(handle[city].keys()):
                h = float(np.asarray(handle[city][sample]["uav_height"][...]).reshape(-1)[0])
                refs.append(SampleRef(city=city, sample=sample, uav_height_m=h))
                if max_samples and len(refs) >= max_samples:
                    return refs
    return refs


def split_city_holdout(
    refs: Sequence[SampleRef],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_seed: int = 42,
) -> Tuple[List[SampleRef], List[SampleRef], List[SampleRef]]:
    refs = list(refs)
    if len(refs) < 2:
        return refs, list(refs), []
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    by_city: Dict[str, List[SampleRef]] = {}
    for ref in refs:
        by_city.setdefault(ref.city, []).append(ref)
    cities = list(by_city.keys())

    rng = random.Random(split_seed)
    rng.shuffle(cities)

    total = len(refs)
    val_target = max(1, int(round(total * val_ratio))) if val_ratio > 0.0 else 0
    test_target = max(1, int(round(total * test_ratio))) if test_ratio > 0.0 else 0

    train_refs: List[SampleRef] = []
    val_refs: List[SampleRef] = []
    test_refs: List[SampleRef] = []

    n_test_cities = 0
    n_val_cities = 0
    for idx, city in enumerate(cities):
        remaining = len(cities) - idx
        city_refs = by_city[city]
        if len(test_refs) < test_target and remaining > 2:
            test_refs.extend(city_refs)
            n_test_cities += 1
            continue
        if len(val_refs) < val_target and remaining > 1:
            val_refs.extend(city_refs)
            n_val_cities += 1
            continue
        train_refs.extend(city_refs)

    if not train_refs:
        shuffled = refs[:]
        rng.shuffle(shuffled)
        n_test = test_target
        n_val = val_target
        test_refs = shuffled[:n_test]
        val_refs = shuffled[n_test:n_test + n_val]
        train_refs = shuffled[n_test + n_val:]

    return train_refs, val_refs, test_refs


def load_sample(handle: h5py.File, ref: SampleRef, metrics: Iterable[str]) -> SampleArrays:
    grp = handle[ref.city][ref.sample]
    topology = np.asarray(grp["topology_map"][...], dtype=np.float32)
    los_mask = np.asarray(grp["los_mask"][...], dtype=np.uint8)
    ground = topology == 0.0
    topology_class = classify_topology(topology)

    targets: Dict[str, np.ndarray] = {}
    valid_masks: Dict[str, np.ndarray] = {}
    for metric in metrics:
        field = str(METRIC_SPECS[metric]["field"])
        target = read_field(grp, field)
        valid = ground & np.isfinite(target) & (target >= 0.0)
        targets[metric] = np.where(valid, target, 0.0).astype(np.float32)
        valid_masks[metric] = valid

    return SampleArrays(
        topology=topology,
        los_mask=los_mask,
        ground=ground,
        topology_class=topology_class,
        targets=targets,
        valid_masks=valid_masks,
    )


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

    h = float(h_tx)
    clearance = np.maximum(h - topology, 0.0).astype(np.float32) * buildings
    tx_clearance_41 = np.clip(box_mean(clearance, KERNEL_SIZES[1]) / HEIGHT_NORM_M, 0.0, None).astype(np.float32)
    taller = (topology > h).astype(np.float32) * buildings
    tx_below_frac_41 = np.clip(box_mean(taller, KERNEL_SIZES[1]), 0.0, 1.0).astype(np.float32)

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
        "tx_clearance_41": tx_clearance_41,
        "tx_below_frac_41": tx_below_frac_41,
    }


def compute_shared_features_torch(topology: np.ndarray, los_mask: np.ndarray, h_tx: float, device) -> Dict[str, np.ndarray]:
    # Keep the DML path conservative: accelerate at the script level, but
    # compute shared features with the exact same numpy implementation.
    return compute_shared_features(topology, los_mask, h_tx)


def compute_raw_prior(metric: str, topology_class: str, shared: Dict[str, np.ndarray], los_mask: np.ndarray) -> np.ndarray:
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


def build_design_matrix(shared: Dict[str, np.ndarray], raw_prior: np.ndarray) -> np.ndarray:
    prior_log = np.log1p(np.clip(raw_prior, 0.0, None))
    bias = np.ones_like(raw_prior, dtype=np.float32)
    tx_clearance = shared["tx_clearance_41"]
    tx_below = shared["tx_below_frac_41"]
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


def regime_key(metric: str, topology_class: str, los_label: str, ant_label: str) -> str:
    return f"{metric}|{topology_class}|{los_label}|{ant_label}"


def fit_keys(metric: str, topology_class: str, los_label: str, ant_label: str) -> Tuple[str, ...]:
    return (
        regime_key(metric, topology_class, los_label, ant_label),
        regime_key(metric, topology_class, los_label, "all_ant"),
        regime_key(metric, topology_class, "all_los", "all_ant"),
        regime_key(metric, "global", los_label, "all_ant"),
        regime_key(metric, "global", "all_los", "all_ant"),
    )


def inference_keys(metric: str, topology_class: str, los_label: str, ant_label: str) -> Tuple[str, ...]:
    return fit_keys(metric, topology_class, los_label, ant_label)


@dataclass
class RegimeAccum:
    xtx: np.ndarray = field(default_factory=lambda: np.zeros((N_FEAT, N_FEAT), dtype=np.float64))
    xty: np.ndarray = field(default_factory=lambda: np.zeros((N_FEAT,), dtype=np.float64))
    count: int = 0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.shape[0] == 0:
            return
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.count += int(x.shape[0])

    def solve(self, ridge: float) -> Optional[np.ndarray]:
        if self.count < N_FEAT * 4:
            return None
        reg = np.eye(N_FEAT, dtype=np.float64) * ridge
        reg[-1, -1] = 0.0
        try:
            return np.linalg.solve(self.xtx + reg, self.xty)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(self.xtx + reg, self.xty, rcond=None)[0]


def fit_calibration(
    hdf5_path: Path,
    train_refs: Sequence[SampleRef],
    metrics: Sequence[str],
    pixel_subsample: float,
    ridge: float,
    seed: int,
    log_every: int,
    device=None,
    progress_json: Optional[Path] = None,
    progress_log: Optional[Path] = None,
    started_at: Optional[str] = None,
    device_name: str = "cpu",
    out_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    rng = random.Random(seed)
    accums: Dict[str, RegimeAccum] = {}

    with h5py.File(str(hdf5_path), "r") as handle:
        for i, ref in enumerate(train_refs):
            sample = load_sample(handle, ref, metrics)
            ab = ant_bin(ref.uav_height_m)
            if device is None:
                shared = compute_shared_features(sample.topology, sample.los_mask, ref.uav_height_m)
            else:
                shared = compute_shared_features_torch(sample.topology, sample.los_mask, ref.uav_height_m, device)

            for metric in metrics:
                raw_prior = compute_raw_prior(metric, sample.topology_class, shared, sample.los_mask)
                x_all = build_design_matrix(shared, raw_prior)
                y_all = np.log1p(np.clip(sample.targets[metric], 0.0, None)).astype(np.float32)
                valid = sample.valid_masks[metric]

                for los_label, region in (
                    ("LoS", valid & (sample.los_mask > 0)),
                    ("NLoS", valid & (sample.los_mask == 0)),
                ):
                    if not region.any():
                        continue
                    idx = np.flatnonzero(region.reshape(-1))
                    if pixel_subsample < 1.0:
                        n_keep = max(1, int(len(idx) * pixel_subsample))
                        if n_keep < len(idx):
                            chosen = rng.sample(range(len(idx)), n_keep)
                            idx = idx[np.asarray(chosen, dtype=np.int64)]

                    x = x_all.reshape(-1, N_FEAT)[idx].astype(np.float64)
                    y = y_all.reshape(-1)[idx].astype(np.float64)
                    for key in fit_keys(metric, sample.topology_class, los_label, ab):
                        accums.setdefault(key, RegimeAccum()).update(x, y)

            if (i + 1) % max(log_every, 1) == 0:
                print(f"[try79] fit {i + 1}/{len(train_refs)} samples  regimes={len(accums)}")
                if progress_log is not None:
                    append_progress_log(
                        progress_log,
                        f"{utc_now_iso()} [fit] {i + 1}/{len(train_refs)} regimes={len(accums)}",
                    )
                if progress_json is not None and started_at is not None and out_dir is not None:
                    write_progress_json(
                        progress_json,
                        build_progress_payload(
                            status="running",
                            started_at=started_at,
                            device_name=device_name,
                            phase="fit",
                            out_dir=out_dir,
                            metrics=metrics,
                            n_train_samples=len(train_refs),
                            n_eval_samples=0,
                            processed_samples=i + 1,
                            n_regimes_fitted=len(accums),
                        ),
                    )

    coefs: Dict[str, np.ndarray] = {}
    for key, accum in accums.items():
        coef = accum.solve(ridge=ridge)
        if coef is not None:
            coefs[key] = coef
    return coefs


def apply_calibration(
    metric: str,
    topology_class: str,
    ant_label: str,
    los_mask: np.ndarray,
    raw_prior: np.ndarray,
    x_all: np.ndarray,
    coefs: Dict[str, np.ndarray],
) -> np.ndarray:
    pred_log = np.log1p(np.clip(raw_prior, 0.0, None)).astype(np.float32)
    x_flat = x_all.reshape(-1, N_FEAT).astype(np.float64)
    los_region = los_mask > 0
    nlos_region = los_mask == 0
    for los_label, region in (("LoS", los_region), ("NLoS", nlos_region)):
        if not np.any(region):
            continue
        for key in inference_keys(metric, topology_class, los_label, ant_label):
            if key in coefs:
                reg_pred = (x_flat @ coefs[key]).reshape(raw_prior.shape).astype(np.float32)
                pred_log[region] = reg_pred[region]
                break
    pred = np.expm1(pred_log).astype(np.float32)
    los_hi = float(LOS_CLIP_HI.get(metric, METRIC_SPECS[metric]["clip_hi"]))
    nlos_hi = float(NLOS_CLIP_HI.get(metric, METRIC_SPECS[metric]["clip_hi"]))
    pred = np.where(los_region, np.clip(pred, 0.0, los_hi), pred).astype(np.float32)
    pred = np.where(nlos_region, np.clip(pred, 0.0, nlos_hi), pred).astype(np.float32)
    return np.clip(pred, 0.0, max(los_hi, nlos_hi)).astype(np.float32)


def rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - target[mask]) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def aggregate_rows(rows: Sequence[Dict[str, float | int | str]]) -> Dict[str, float | int]:
    def weighted(metric_key: str, count_key: str, squared: bool) -> float:
        num = 0.0
        den = 0
        for row in rows:
            value = float(row[metric_key])
            count = int(row[count_key])
            if math.isfinite(value) and count > 0:
                num += count * (value * value if squared else value)
                den += count
        if den == 0:
            return float("nan")
        return float(math.sqrt(num / den) if squared else num / den)

    return {
        "n_samples": len(rows),
        "raw_rmse_overall_pw": weighted("raw_rmse_overall", "n_valid", True),
        "calib_rmse_overall_pw": weighted("calib_rmse_overall", "n_valid", True),
        "raw_mae_overall_pw": weighted("raw_mae_overall", "n_valid", False),
        "calib_mae_overall_pw": weighted("calib_mae_overall", "n_valid", False),
        "raw_rmse_los_pw": weighted("raw_rmse_los", "n_los", True),
        "calib_rmse_los_pw": weighted("calib_rmse_los", "n_los", True),
        "raw_rmse_nlos_pw": weighted("raw_rmse_nlos", "n_nlos", True),
        "calib_rmse_nlos_pw": weighted("calib_rmse_nlos", "n_nlos", True),
    }


def evaluate(
    hdf5_path: Path,
    refs: Sequence[SampleRef],
    metrics: Sequence[str],
    coefs: Dict[str, np.ndarray],
    log_every: int,
    device=None,
    progress_json: Optional[Path] = None,
    progress_log: Optional[Path] = None,
    started_at: Optional[str] = None,
    device_name: str = "cpu",
    out_dir: Optional[Path] = None,
    n_train_samples: int = 0,
) -> Dict[str, Dict[str, object]]:
    per_metric_rows: Dict[str, List[Dict[str, object]]] = {metric: [] for metric in metrics}

    with h5py.File(str(hdf5_path), "r") as handle:
        for i, ref in enumerate(refs):
            sample = load_sample(handle, ref, metrics)
            ab = ant_bin(ref.uav_height_m)
            if device is None:
                shared = compute_shared_features(sample.topology, sample.los_mask, ref.uav_height_m)
            else:
                shared = compute_shared_features_torch(sample.topology, sample.los_mask, ref.uav_height_m, device)

            for metric in metrics:
                target = sample.targets[metric]
                valid = sample.valid_masks[metric]
                raw_prior = compute_raw_prior(metric, sample.topology_class, shared, sample.los_mask)
                x_all = build_design_matrix(shared, raw_prior)
                calib = apply_calibration(metric, sample.topology_class, ab, sample.los_mask, raw_prior, x_all, coefs)

                los_mask = valid & (sample.los_mask > 0)
                nlos_mask = valid & (sample.los_mask == 0)
                row = {
                    "city": ref.city,
                    "sample": ref.sample,
                    "uav_height_m": ref.uav_height_m,
                    "topology_class": sample.topology_class,
                    "n_valid": int(valid.sum()),
                    "n_los": int(los_mask.sum()),
                    "n_nlos": int(nlos_mask.sum()),
                    "raw_rmse_overall": rmse(raw_prior, target, valid),
                    "calib_rmse_overall": rmse(calib, target, valid),
                    "raw_mae_overall": mae(raw_prior, target, valid),
                    "calib_mae_overall": mae(calib, target, valid),
                    "raw_rmse_los": rmse(raw_prior, target, los_mask),
                    "calib_rmse_los": rmse(calib, target, los_mask),
                    "raw_rmse_nlos": rmse(raw_prior, target, nlos_mask),
                    "calib_rmse_nlos": rmse(calib, target, nlos_mask),
                }
                per_metric_rows[metric].append(row)

            if (i + 1) % max(log_every, 1) == 0:
                print(f"[try79] eval {i + 1}/{len(refs)} samples")
                partial = {
                    metric: {
                        "aggregate": aggregate_rows(rows)
                    }
                    for metric, rows in per_metric_rows.items()
                    if rows
                }
                if progress_log is not None:
                    agg_txt = " ".join(
                        f"{metric}={partial[metric]['aggregate'].get('calib_rmse_overall_pw', float('nan')):.3f}"
                        for metric in metrics
                        if metric in partial
                    )
                    append_progress_log(
                        progress_log,
                        f"{utc_now_iso()} [eval] {i + 1}/{len(refs)} {agg_txt}".strip(),
                    )
                if progress_json is not None and started_at is not None and out_dir is not None:
                    write_progress_json(
                        progress_json,
                        build_progress_payload(
                            status="running",
                            started_at=started_at,
                            device_name=device_name,
                            phase="eval",
                            out_dir=out_dir,
                            metrics=metrics,
                            n_train_samples=n_train_samples,
                            n_eval_samples=len(refs),
                            processed_samples=i + 1,
                            n_regimes_fitted=len(coefs),
                            eval_results=partial,
                        ),
                    )

    out: Dict[str, Dict[str, object]] = {}
    for metric, rows in per_metric_rows.items():
        per_topology = {}
        for topology_class in TOPOLOGY_CLASSES:
            topo_rows = [row for row in rows if row["topology_class"] == topology_class]
            if topo_rows:
                per_topology[topology_class] = aggregate_rows(topo_rows)
        out[metric] = {
            "aggregate": aggregate_rows(rows),
            "per_topology": per_topology,
            "per_sample": rows,
        }
    return out


def save_calibration(path: Path, coefs: Dict[str, np.ndarray], meta: Dict[str, object]) -> None:
    payload = {
        "model_type": "try79_numpy_spread_prior",
        "feature_names": list(FEATURE_NAMES),
        "metrics": list(METRIC_SPECS.keys()),
        "topology_thresholds": TOPOLOGY_THRESHOLDS,
        "antenna_height_thresholds": {"q1": ANT_Q1, "q2": ANT_Q2},
        "kernel_sizes": list(KERNEL_SIZES),
        "height_norm_m": HEIGHT_NORM_M,
        "meta": meta,
        "coefficients": {k: v.tolist() for k, v in coefs.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibration(path: Path) -> Dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {k: np.asarray(v, dtype=np.float64) for k, v in payload["coefficients"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "prior_out")
    parser.add_argument("--metrics", nargs="*", default=["delay_spread", "angular_spread"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--pixel-subsample", type=float, default=0.02)
    parser.add_argument("--ridge", type=float, default=RIDGE_LAMBDA)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--calibration-json", type=Path, default=None)
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", choices=["auto", "cpu", "dml"], default="auto")
    parser.add_argument("--progress-json", type=Path, default=None)
    parser.add_argument("--progress-log", type=Path, default=None)
    args = parser.parse_args()

    metrics = [metric for metric in args.metrics if metric in METRIC_SPECS]
    if not metrics:
        raise ValueError(f"No valid metrics requested. Choose from {list(METRIC_SPECS)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    device_name = str(device) if device is not None else "cpu"
    progress_json = args.progress_json or (args.out_dir / "progress.json")
    progress_log = args.progress_log or (args.out_dir / "progress.out")
    started_at = utc_now_iso()

    print(f"[try79] hdf5={args.hdf5}")
    refs = enumerate_samples(args.hdf5, max_samples=args.max_samples)
    train_refs, val_refs, test_refs = split_city_holdout(
        refs,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )
    eval_refs = val_refs if args.eval_split == "val" else test_refs

    print(
        f"[try79] split seed={args.split_seed}: "
        f"train={len(train_refs)} val={len(val_refs)} test={len(test_refs)}"
    )
    print(f"[try79] execution device={device_name}")
    append_progress_log(progress_log, f"{utc_now_iso()} [start] device={device_name} train={len(train_refs)} eval={len(eval_refs)} split={args.eval_split}")
    write_progress_json(
        progress_json,
        build_progress_payload(
            status="running",
            started_at=started_at,
            device_name=device_name,
            phase="setup",
            out_dir=args.out_dir,
            metrics=metrics,
            n_train_samples=len(train_refs),
            n_eval_samples=len(eval_refs),
            processed_samples=0,
            n_regimes_fitted=0,
        ),
    )

    try:
        if args.calibration_json:
            print(f"[try79] loading calibration from {args.calibration_json}")
            coefs = load_calibration(args.calibration_json)
        elif args.skip_fit:
            coefs = {}
        else:
            print(f"[try79] fitting metrics={metrics} pixel_subsample={args.pixel_subsample}")
            coefs = fit_calibration(
                args.hdf5,
                train_refs,
                metrics=metrics,
                pixel_subsample=args.pixel_subsample,
                ridge=args.ridge,
                seed=args.seed,
                log_every=args.log_every,
                device=device,
                progress_json=progress_json,
                progress_log=progress_log,
                started_at=started_at,
                device_name=device_name,
                out_dir=args.out_dir,
            )
            cal_path = args.out_dir / "calibration.json"
            save_calibration(
                cal_path,
                coefs,
                meta={
                    "n_train_samples": len(train_refs),
                    "metrics": metrics,
                    "split_seed": args.split_seed,
                    "val_ratio": args.val_ratio,
                    "test_ratio": args.test_ratio,
                },
            )
            print(f"[try79] calibration -> {cal_path}")
            append_progress_log(progress_log, f"{utc_now_iso()} [fit_done] regimes={len(coefs)} calibration={cal_path}")

        summary = {
            "metrics": metrics,
            "split_seed": args.split_seed,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "n_train_samples": len(train_refs),
            "n_val_samples": len(val_refs),
            "n_test_samples": len(test_refs),
            "n_regimes_fitted": len(coefs),
            "eval_split": args.eval_split,
            "results": {},
        }

        if not args.skip_eval and eval_refs:
            print(f"[try79] evaluating on {args.eval_split} split")
            summary["results"] = evaluate(
                args.hdf5,
                eval_refs,
                metrics=metrics,
                coefs=coefs,
                log_every=args.log_every,
                device=device,
                progress_json=progress_json,
                progress_log=progress_log,
                started_at=started_at,
                device_name=device_name,
                out_dir=args.out_dir,
                n_train_samples=len(train_refs),
            )

        out_path = args.out_dir / f"eval_summary_{args.eval_split}.json"
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        append_progress_log(progress_log, f"{utc_now_iso()} [done] summary={out_path}")
        write_progress_json(
            progress_json,
            build_progress_payload(
                status="done",
                started_at=started_at,
                device_name=device_name,
                phase="done",
                out_dir=args.out_dir,
                metrics=metrics,
                n_train_samples=len(train_refs),
                n_eval_samples=len(eval_refs),
                processed_samples=len(eval_refs) if not args.skip_eval else len(train_refs),
                n_regimes_fitted=len(coefs),
                eval_results=summary["results"],
            ),
        )
        print(f"[try79] summary -> {out_path}")
    except Exception as exc:
        append_progress_log(progress_log, f"{utc_now_iso()} [error] {exc}")
        write_progress_json(
            progress_json,
            build_progress_payload(
                status="error",
                started_at=started_at,
                device_name=device_name,
                phase="error",
                out_dir=args.out_dir,
                metrics=metrics,
                n_train_samples=len(train_refs),
                n_eval_samples=len(eval_refs),
                processed_samples=0,
                n_regimes_fitted=0,
                error=str(exc),
            ),
        )
        raise


if __name__ == "__main__":
    main()
