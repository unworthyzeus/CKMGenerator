"""Evaluate the frozen Try 78 / Try 79 priors used by Try 80.

This is a model-free evaluator: it loads the Try 80 config, builds the same
city-holdout dataset split, computes or reads the frozen prior maps, and
reports prior-only RMSE/MAE against the CKM targets.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_try80 import Try80Cfg
from src.data_utils import Try80DataConfig, build_joint_datasets
from src.metrics_try80 import TASKS, MultiTaskMetricAccumulator


def build_data_cfg(cfg: Try80Cfg) -> Try80DataConfig:
    return Try80DataConfig(
        hdf5_path=cfg.data.hdf5_path,
        try78_los_calibration_json=cfg.prior.try78_los_calibration_json,
        try78_nlos_calibration_json=cfg.prior.try78_nlos_calibration_json,
        try79_calibration_json=cfg.prior.try79_calibration_json,
        image_size=cfg.data.image_size,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        split_seed=cfg.data.split_seed,
        topology_norm_m=cfg.data.topology_norm_m,
        path_loss_no_data_mask_column=cfg.data.path_loss_no_data_mask_column,
        derive_no_data_from_non_ground=cfg.data.derive_no_data_from_non_ground,
        augment_d4=False,
        precomputed_priors_hdf5_path=cfg.data.precomputed_priors_hdf5_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = Try80Cfg.load(args.config)
    out_dir = args.out_dir or (cfg.runtime.output_dir / f"prior_eval_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = build_joint_datasets(build_data_cfg(cfg))
    ds_by_split = {"train": train_ds, "val": val_ds, "test": test_ds}
    ds = ds_by_split[args.split]

    num_workers = max(0, cfg.training.num_workers // 2) if args.num_workers is None else max(0, args.num_workers)
    loader = DataLoader(ds, batch_size=max(1, args.batch_size), num_workers=num_workers, shuffle=False)

    metrics = MultiTaskMetricAccumulator(store_per_sample=True)
    started = time.time()
    seen = 0

    with torch.no_grad():
        for batch in loader:
            priors_native = {task: batch[f"{task}_prior"] for task in TASKS}
            metrics.update_batch(batch, preds_native=priors_native, priors_native=priors_native)
            seen += int(priors_native["path_loss"].shape[0])
            if args.max_samples is not None and seen >= args.max_samples:
                break

    summary = {
        "split": args.split,
        "n_samples_available": len(ds),
        "n_samples_evaluated": metrics.n_samples,
        "elapsed_seconds": time.time() - started,
        "metrics": metrics.summary(),
    }

    (out_dir / "prior_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "prior_per_sample.json").write_text(json.dumps(summary["metrics"]["per_sample"], indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"]["prior"]["flat"], indent=2))


if __name__ == "__main__":
    main()
