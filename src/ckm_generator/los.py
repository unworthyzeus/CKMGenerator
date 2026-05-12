"""LoS/NLoS mask generation from a topology height map."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.ndimage import maximum_filter
except Exception:  # pragma: no cover - optional speed/quality dependency
    maximum_filter = None


@dataclass
class MaskComparison:
    n_pixels: int
    mismatches: int
    mismatch_fraction: float
    false_los: int
    false_nlos: int
    los_iou: float
    nlos_iou: float


def compute_los_mask(
    topology: np.ndarray,
    antenna_height_m: float,
    *,
    rx_height_m: float = 1.5,
    tx_row: Optional[int] = None,
    tx_col: Optional[int] = None,
    ground_epsilon_m: float = 1.0e-6,
    sample_step_px: float = 1.0,
    clearance_m: float = 0.0,
    building_dilation_px: int = 0,
    chunk_size: int = 4096,
    backend: str = "numpy",
    torch_device: Optional[str] = None,
) -> np.ndarray:
    """Generate a binary LoS mask by ray-casting from the center transmitter.

    A ground receiver pixel is LoS when no building cell crossed by the
    center-to-pixel ray rises above the straight Tx-to-Rx segment. Building
    pixels are always returned as non-LoS, matching the Try 80 masking rule.
    """
    resolved_backend = backend.lower().strip()
    if resolved_backend in {"auto", "cuda"}:
        resolved_backend = "torch" if _torch_cuda_available() else "numpy"
    if resolved_backend == "torch":
        return _compute_los_mask_torch(
            topology,
            antenna_height_m,
            rx_height_m=rx_height_m,
            tx_row=tx_row,
            tx_col=tx_col,
            ground_epsilon_m=ground_epsilon_m,
            sample_step_px=sample_step_px,
            clearance_m=clearance_m,
            building_dilation_px=building_dilation_px,
            chunk_size=chunk_size,
            torch_device=torch_device,
        )
    if resolved_backend != "numpy":
        raise ValueError("backend must be 'numpy', 'torch', 'cuda', or 'auto'")
    return _compute_los_mask_numpy(
        topology,
        antenna_height_m,
        rx_height_m=rx_height_m,
        tx_row=tx_row,
        tx_col=tx_col,
        ground_epsilon_m=ground_epsilon_m,
        sample_step_px=sample_step_px,
        clearance_m=clearance_m,
        building_dilation_px=building_dilation_px,
        chunk_size=chunk_size,
    )


def _compute_los_mask_numpy(
    topology: np.ndarray,
    antenna_height_m: float,
    *,
    rx_height_m: float,
    tx_row: Optional[int],
    tx_col: Optional[int],
    ground_epsilon_m: float,
    sample_step_px: float,
    clearance_m: float,
    building_dilation_px: int,
    chunk_size: int,
) -> np.ndarray:
    topo = np.nan_to_num(np.asarray(topology, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if topo.ndim != 2:
        raise ValueError(f"Expected a 2D topology map, got shape {topo.shape}")
    h, w = topo.shape
    tx_r = h // 2 if tx_row is None else int(tx_row)
    tx_c = w // 2 if tx_col is None else int(tx_col)
    ground = topo <= float(ground_epsilon_m)

    blockers = topo
    if building_dilation_px > 0:
        if maximum_filter is None:
            raise RuntimeError("scipy is required for building_dilation_px > 0")
        size = int(building_dilation_px) * 2 + 1
        blockers = maximum_filter(topo, size=size, mode="nearest")

    rows, cols = np.nonzero(ground)
    los_values = np.zeros(rows.shape[0], dtype=bool)
    step_px = max(float(sample_step_px), 0.25)
    chunk = max(int(chunk_size), 256)
    step_axis_cache: dict[int, np.ndarray] = {}

    for start in range(0, rows.shape[0], chunk):
        end = min(start + chunk, rows.shape[0])
        rr = rows[start:end].astype(np.float32)
        cc = cols[start:end].astype(np.float32)
        dr = rr - float(tx_r)
        dc = cc - float(tx_c)
        steps = np.ceil(np.maximum(np.abs(dr), np.abs(dc)) / step_px).astype(np.int32)
        steps = np.maximum(steps, 1)
        max_steps = int(steps.max(initial=1))
        step_ids = step_axis_cache.get(max_steps)
        if step_ids is None:
            step_ids = np.arange(1, max_steps, dtype=np.float32)[:, None]
            step_axis_cache[max_steps] = step_ids
        if step_ids.size == 0:
            los_values[start:end] = True
            continue

        active = step_ids < steps[None, :]
        t = step_ids / steps[None, :]
        sample_r = np.rint(float(tx_r) + dr[None, :] * t).astype(np.int32)
        sample_c = np.rint(float(tx_c) + dc[None, :] * t).astype(np.int32)
        sample_r = np.clip(sample_r, 0, h - 1)
        sample_c = np.clip(sample_c, 0, w - 1)

        target_r = rr.astype(np.int32)[None, :]
        target_c = cc.astype(np.int32)[None, :]
        not_target = (sample_r != target_r) | (sample_c != target_c)
        not_tx = (sample_r != tx_r) | (sample_c != tx_c)
        z_line = float(antenna_height_m) + (float(rx_height_m) - float(antenna_height_m)) * t
        sampled_heights = blockers[sample_r, sample_c]
        blocked = active & not_target & not_tx & ((sampled_heights + float(clearance_m)) >= z_line)
        los_values[start:end] = ~np.any(blocked, axis=0)

    los = np.zeros_like(ground, dtype=np.float32)
    los[rows, cols] = los_values.astype(np.float32)
    return los * ground.astype(np.float32)


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _compute_los_mask_torch(
    topology: np.ndarray,
    antenna_height_m: float,
    *,
    rx_height_m: float,
    tx_row: Optional[int],
    tx_col: Optional[int],
    ground_epsilon_m: float,
    sample_step_px: float,
    clearance_m: float,
    building_dilation_px: int,
    chunk_size: int,
    torch_device: Optional[str],
) -> np.ndarray:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional backend guard
        raise RuntimeError("The torch LoS backend requires PyTorch") from exc

    device = torch.device(torch_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    topo = np.nan_to_num(np.asarray(topology, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if topo.ndim != 2:
        raise ValueError(f"Expected a 2D topology map, got shape {topo.shape}")
    h, w = topo.shape
    tx_r = h // 2 if tx_row is None else int(tx_row)
    tx_c = w // 2 if tx_col is None else int(tx_col)
    ground = topo <= float(ground_epsilon_m)

    blockers_np = topo
    if building_dilation_px > 0:
        if maximum_filter is None:
            raise RuntimeError("scipy is required for building_dilation_px > 0")
        size = int(building_dilation_px) * 2 + 1
        blockers_np = maximum_filter(topo, size=size, mode="nearest")

    rows, cols = np.nonzero(ground)
    los_values = np.zeros(rows.shape[0], dtype=bool)
    step_px = max(float(sample_step_px), 0.25)
    chunk = max(int(chunk_size), 256)
    blockers = torch.as_tensor(blockers_np, device=device, dtype=torch.float64)
    step_axis_cache: dict[int, object] = {}

    with torch.inference_mode():
        for start in range(0, rows.shape[0], chunk):
            end = min(start + chunk, rows.shape[0])
            rr_np = rows[start:end].astype(np.float32)
            cc_np = cols[start:end].astype(np.float32)
            dr_np = rr_np - float(tx_r)
            dc_np = cc_np - float(tx_c)
            steps_np = np.ceil(np.maximum(np.abs(dr_np), np.abs(dc_np)) / step_px).astype(np.int32)
            steps_np = np.maximum(steps_np, 1)
            max_steps = int(steps_np.max(initial=1))
            if max_steps <= 1:
                los_values[start:end] = True
                continue

            step_ids = step_axis_cache.get(max_steps)
            if step_ids is None:
                step_ids = torch.arange(1, max_steps, device=device, dtype=torch.float64)[:, None]
                step_axis_cache[max_steps] = step_ids

            steps = torch.as_tensor(steps_np, device=device, dtype=torch.float64)[None, :]
            rr = torch.as_tensor(rr_np, device=device, dtype=torch.float64)
            cc = torch.as_tensor(cc_np, device=device, dtype=torch.float64)
            dr = torch.as_tensor(dr_np, device=device, dtype=torch.float64)
            dc = torch.as_tensor(dc_np, device=device, dtype=torch.float64)

            active = step_ids < steps
            t = step_ids / steps
            sample_r = torch.round(float(tx_r) + dr[None, :] * t).to(torch.int64).clamp_(0, h - 1)
            sample_c = torch.round(float(tx_c) + dc[None, :] * t).to(torch.int64).clamp_(0, w - 1)

            target_r = rr.to(torch.int64)[None, :]
            target_c = cc.to(torch.int64)[None, :]
            not_target = (sample_r != target_r) | (sample_c != target_c)
            not_tx = (sample_r != tx_r) | (sample_c != tx_c)
            z_line = float(antenna_height_m) + (float(rx_height_m) - float(antenna_height_m)) * t
            sampled_heights = blockers[sample_r, sample_c]
            blocked = active & not_target & not_tx & ((sampled_heights + float(clearance_m)) >= z_line)
            los_values[start:end] = (~torch.any(blocked, dim=0)).detach().cpu().numpy()

    los = np.zeros_like(ground, dtype=np.float32)
    los[rows, cols] = los_values.astype(np.float32)
    return los * ground.astype(np.float32)


def compute_nlos_mask(topology: np.ndarray, los_mask: np.ndarray, *, ground_epsilon_m: float = 1.0e-6) -> np.ndarray:
    topo = np.asarray(topology, dtype=np.float32)
    ground = topo <= float(ground_epsilon_m)
    los = np.asarray(los_mask, dtype=np.float32) > 0.5
    return (ground & ~los).astype(np.float32)


def compare_los_masks(generated_los: np.ndarray, reference_los: np.ndarray, topology: np.ndarray) -> MaskComparison:
    ground = np.asarray(topology, dtype=np.float32) <= 1.0e-6
    gen = (np.asarray(generated_los, dtype=np.float32) > 0.5) & ground
    ref = (np.asarray(reference_los, dtype=np.float32) > 0.5) & ground
    mism = gen != ref
    false_los = gen & ~ref
    false_nlos = ~gen & ref & ground
    los_union = gen | ref
    nlos_gen = ~gen & ground
    nlos_ref = ~ref & ground
    nlos_union = nlos_gen | nlos_ref
    n_pixels = int(ground.sum())
    return MaskComparison(
        n_pixels=n_pixels,
        mismatches=int(mism.sum()),
        mismatch_fraction=float(mism.sum() / max(n_pixels, 1)),
        false_los=int(false_los.sum()),
        false_nlos=int(false_nlos.sum()),
        los_iou=float((gen & ref).sum() / max(los_union.sum(), 1)),
        nlos_iou=float((nlos_gen & nlos_ref).sum() / max(nlos_union.sum(), 1)),
    )
