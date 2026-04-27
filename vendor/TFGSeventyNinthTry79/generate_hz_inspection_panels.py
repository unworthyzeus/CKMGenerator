"""Generate GT vs height-aware-prediction inspection panels.

Picks 10 diverse test-split samples (mix of topology classes and LoS
coverage, spanning UAV heights) and saves one 8-panel figure per sample
under `hz_inspection/`.

Each figure shows:
  row 0: topology | LoS mask | GT delay_spread | predicted delay_spread
  row 1: height-aware tx_clearance_41 | error (pred - gt delay) | GT angular_spread | predicted angular_spread

Comparable to the existing `map_inspection/*_panels.png` style but with
both delay and angular spreads plus the height-aware clearance map.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

import prior_try79 as base


HERE = Path(__file__).resolve().parent
HDF5 = base.DEFAULT_HDF5
EVAL_JSON = HERE / "test_eval_dml_hz_v2" / "eval_summary_test.json"
OUT_DIR = HERE / "hz_inspection"
CAL_PATH = HERE / "test_eval_dml_hz_v2" / "calibration.json"
N_PICK = 10


def pick_samples() -> list[dict]:
    results = json.loads(EVAL_JSON.read_text())["results"]
    rows = results["delay_spread"]["per_sample"]
    by_topo: dict[str, list[dict]] = {}
    for row in rows:
        by_topo.setdefault(row["topology_class"], []).append(row)
    for k in by_topo:
        by_topo[k].sort(key=lambda r: r["uav_height_m"])

    picks: list[dict] = []
    topo_order = [
        "open_sparse_lowrise",
        "mixed_compact_lowrise",
        "mixed_compact_midrise",
        "dense_block_midrise",
        "open_sparse_vertical",
        "dense_block_highrise",
    ]
    for topo in topo_order:
        bucket = by_topo.get(topo, [])
        if not bucket:
            continue
        low = bucket[len(bucket) // 6]
        high = bucket[-max(1, len(bucket) // 6)]
        picks.append(low)
        if len(picks) < N_PICK and high["sample"] != low["sample"]:
            picks.append(high)
        if len(picks) >= N_PICK:
            break

    for topo in topo_order:
        if len(picks) >= N_PICK:
            break
        bucket = by_topo.get(topo, [])
        if len(bucket) >= 3:
            picks.append(bucket[len(bucket) // 2])
    return picks[:N_PICK]


def render_panel(out_path: Path, row: dict, arrays: dict) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    topology = arrays["topology"]
    los = arrays["los_mask"]
    gt_delay = arrays["gt_delay"]
    pr_delay = arrays["pr_delay"]
    gt_ang = arrays["gt_ang"]
    pr_ang = arrays["pr_ang"]
    clearance = arrays["tx_clearance"]
    delay_err = pr_delay - gt_delay
    valid = arrays["valid"]
    delay_err = np.where(valid, delay_err, 0.0)

    ax = axes[0, 0]
    ax.imshow(topology, cmap="terrain")
    ax.set_title(f"topology  h_tx={row['uav_height_m']:.1f} m")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(los, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"LoS mask  ({100.0 * los.mean():.1f}% LoS)")
    ax.axis("off")

    vmax_d = float(np.nanpercentile(gt_delay[valid], 99.0)) if valid.any() else 1.0
    ax = axes[0, 2]
    im = ax.imshow(gt_delay, cmap="viridis", vmin=0, vmax=vmax_d)
    ax.set_title("GT delay_spread (ns)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 3]
    im = ax.imshow(pr_delay, cmap="viridis", vmin=0, vmax=vmax_d)
    ax.set_title("HZ pred delay_spread (ns)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    im = ax.imshow(clearance, cmap="magma")
    ax.set_title("tx_clearance_41 (UAV above rooftops / 90m)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    amax = max(abs(float(np.nanpercentile(delay_err[valid], 1.0))),
               abs(float(np.nanpercentile(delay_err[valid], 99.0)))) if valid.any() else 1.0
    ax = axes[1, 1]
    im = ax.imshow(delay_err, cmap="seismic", vmin=-amax, vmax=amax)
    ax.set_title(f"delay error (pred-GT)  rmse={row['calib_rmse_overall']:.1f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    vmax_a = float(np.nanpercentile(gt_ang[valid], 99.0)) if valid.any() else 1.0
    ax = axes[1, 2]
    im = ax.imshow(gt_ang, cmap="plasma", vmin=0, vmax=vmax_a)
    ax.set_title("GT angular_spread (deg)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 3]
    im = ax.imshow(pr_ang, cmap="plasma", vmin=0, vmax=vmax_a)
    ax.set_title("HZ pred angular_spread (deg)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"{row['city']} / {row['sample']}  topo={row['topology_class']}  "
        f"delay rmse={row['calib_rmse_overall']:.2f} ns",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CAL_PATH.exists():
        raise FileNotFoundError(CAL_PATH)
    coefs = base.load_calibration(CAL_PATH)
    print(f"[panels] using calibration from {CAL_PATH}")

    picks = pick_samples()
    print(f"[panels] picked {len(picks)} samples")

    with h5py.File(str(HDF5), "r") as handle:
        for i, row in enumerate(picks):
            ref = base.SampleRef(
                city=row["city"],
                sample=row["sample"],
                uav_height_m=row["uav_height_m"],
            )
            sample = base.load_sample(handle, ref, ("delay_spread", "angular_spread"))
            shared = base.compute_shared_features(sample.topology, sample.los_mask, ref.uav_height_m)
            ab = base.ant_bin(ref.uav_height_m)

            gt_delay = sample.targets["delay_spread"]
            gt_ang = sample.targets["angular_spread"]

            raw_d = base.compute_raw_prior("delay_spread", sample.topology_class, shared, sample.los_mask)
            raw_a = base.compute_raw_prior("angular_spread", sample.topology_class, shared, sample.los_mask)
            x_d = base.build_design_matrix(shared, raw_d)
            x_a = base.build_design_matrix(shared, raw_a)
            pred_delay = base.apply_calibration(
                "delay_spread", sample.topology_class, ab, sample.los_mask, raw_d, x_d, coefs
            )
            pred_ang = base.apply_calibration(
                "angular_spread", sample.topology_class, ab, sample.los_mask, raw_a, x_a, coefs
            )

            arrays = {
                "topology": sample.topology,
                "los_mask": sample.los_mask,
                "gt_delay": gt_delay,
                "pr_delay": pred_delay,
                "gt_ang": gt_ang,
                "pr_ang": pred_ang,
                "tx_clearance": shared["tx_clearance_41"],
                "valid": sample.valid_masks["delay_spread"],
            }

            out_path = OUT_DIR / f"{row['city']}_{row['sample']}_hz_panel.png"
            render_panel(out_path, row, arrays)
            print(f"[panels] {i + 1}/{len(picks)}  {out_path.name}")


if __name__ == "__main__":
    main()
