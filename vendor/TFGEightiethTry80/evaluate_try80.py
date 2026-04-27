"""Try 80 - evaluation for the joint prior-anchored model."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config_try80 import Try80Cfg
from src.data_utils import HeightEmbedding, Try80DataConfig, build_joint_datasets
from src.losses_try80 import LossWeights, combined_loss
from src.metrics_try80 import TASKS, inverse_transform, transform_target
from src.model_try80 import Try80Model, Try80ModelConfig


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


def build_model_cfg(cfg: Try80Cfg) -> Try80ModelConfig:
    return Try80ModelConfig(
        in_channels=cfg.model.in_channels,
        cond_dim=cfg.model.cond_dim,
        height_embed_dim=cfg.model.height_embed_dim,
        base_width=cfg.model.base_width,
        num_components=cfg.model.num_components,
        decoder_dropout=cfg.model.decoder_dropout,
        alpha_bias=cfg.model.alpha_bias,
        sigma_min=cfg.model.sigma_min,
        sigma_max=cfg.model.sigma_max,
        path_residual_los_max=cfg.model.path_residual_los_max,
        path_residual_nlos_max=cfg.model.path_residual_nlos_max,
        delay_residual_los_max=cfg.model.delay_residual_los_max,
        delay_residual_nlos_max=cfg.model.delay_residual_nlos_max,
        angular_residual_los_max=cfg.model.angular_residual_los_max,
        angular_residual_nlos_max=cfg.model.angular_residual_nlos_max,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()

    cfg = Try80Cfg.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = cfg.runtime.output_dir / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = build_data_cfg(cfg)
    _, val_ds, test_ds = build_joint_datasets(data_cfg)
    ds = val_ds if args.split == "val" else test_ds
    loader = DataLoader(ds, batch_size=1, num_workers=max(0, cfg.training.num_workers // 2), shuffle=False)

    model = Try80Model(build_model_cfg(cfg)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    height_embed = HeightEmbedding()
    weights = LossWeights(**cfg.losses.__dict__)
    from src.metrics_try80 import MultiTaskMetricAccumulator

    totals = {"total": 0.0, "map_nll": 0.0, "dist_kl": 0.0, "moment_match": 0.0, "anchor": 0.0,
              "prior_guard": 0.0, "outlier_budget": 0.0, "rmse": 0.0, "mae": 0.0}
    metrics = MultiTaskMetricAccumulator(store_per_sample=True)
    started = time.time()

    with torch.no_grad():
        for raw_batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw_batch.items()}
            priors_native = {task: batch[f"{task}_prior"] for task in TASKS}
            priors_trans = {task: transform_target(task, priors_native[task]) for task in TASKS}
            outputs = model(batch["inputs"], height_embed(batch["antenna_height_m"]), priors_trans)
            preds_native = {task: inverse_transform(task, outputs[task]["pred_trans"]) for task in TASKS}
            loss_terms = combined_loss(batch, outputs, preds_native, priors_native, weights=weights)
            for key in totals:
                totals[key] += float(loss_terms[key].detach().item())
            metrics.update_batch(batch, preds_native, priors_native)

    denom = max(len(ds), 1)
    summary = {
        "split": args.split,
        "n_samples": len(ds),
        "elapsed_seconds": time.time() - started,
        "losses": {key: value / denom for key, value in totals.items()},
        "metrics": metrics.summary(),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "per_sample.json").write_text(json.dumps(summary["metrics"]["per_sample"], indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"]["model"]["flat"], indent=2))


if __name__ == "__main__":
    main()
