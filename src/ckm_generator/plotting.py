"""PNG rendering helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


TASK_LABELS = {
    "path_loss": ("Path loss", "dB", "viridis"),
    "delay_spread": ("Delay spread", "ns", "magma"),
    "angular_spread": ("Angular spread", "deg", "plasma"),
}


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(path)


def save_map_png(
    values: np.ndarray,
    path: Path,
    *,
    mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    unit: str = "",
    robust: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype=np.float32)
    if mask is not None:
        arr = np.where(np.asarray(mask, dtype=np.float32) > 0.5, arr, np.nan)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        finite = np.array([0.0, 1.0], dtype=np.float32)
    if robust and finite.size > 16:
        vmin, vmax = np.percentile(finite, [1.0, 99.0])
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
    if abs(float(vmax) - float(vmin)) < 1.0e-6:
        vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(5.4, 5.1), dpi=120)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if unit:
        cbar.set_label(unit, fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def save_joint_prediction_png(predictions: Dict[str, np.ndarray], path: Path, *, ground_mask: np.ndarray, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=130)
    for ax, task in zip(axes, ("path_loss", "delay_spread", "angular_spread")):
        label, unit, cmap = TASK_LABELS[task]
        arr = np.where(ground_mask > 0.5, predictions[task], np.nan)
        finite = arr[np.isfinite(arr)]
        if finite.size > 16:
            vmin, vmax = np.percentile(finite, [1.0, 99.0])
        elif finite.size:
            vmin, vmax = float(finite.min()), float(finite.max())
        else:
            vmin, vmax = 0.0, 1.0
        if abs(float(vmax) - float(vmin)) < 1.0e-6:
            vmax = vmin + 1.0
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=11)
        ax.set_axis_off()
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label(unit, fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
