"""Try 80 - joint multi-task training entry point."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from src.config_try80 import Try80Cfg
from src.data_utils import HeightEmbedding, Try80DataConfig, Try80JointDataset, build_joint_datasets
from src.losses_try80 import LossWeights, combined_loss
from src.metrics_try80 import MultiTaskMetricAccumulator, TASKS, inverse_transform, transform_target
from src.model_try80 import Try80Model, Try80ModelConfig


CODE_MARKER = "try80_joint_prior_anchor_2026-04-22a"


def resolve_single_process_device(preferred: str | None = None) -> torch.device:
    preferred = (preferred or "").strip().lower()
    if preferred == "directml":
        try:
            import torch_directml
        except ImportError as exc:
            raise RuntimeError("runtime.device=directml requested, but torch_directml is not installed") from exc
        return torch_directml.device()
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preferred)
    if preferred == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_distributed(preferred_device: str | None = None) -> Tuple[bool, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = os.environ.get("DDP_BACKEND", "nccl")
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        return True, rank, world, device
    return False, 0, 1, resolve_single_process_device(preferred_device)


def maybe_barrier(ddp: bool) -> None:
    if ddp and dist.is_initialized():
        dist.barrier()


def is_main(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_lambda(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


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
        augment_d4=cfg.data.augment_d4,
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


def build_priors(batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
    return {task: batch[f"{task}_prior"] for task in TASKS}


def build_priors_trans(batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
    return {task: transform_target(task, batch[f"{task}_prior"]) for task in TASKS}


def build_preds_native(outputs: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {task: inverse_transform(task, outputs[task]["pred_trans"]) for task in TASKS}


LOG_STEP_INTERVAL = 25


def _has_nan_params(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if p.data.isnan().any() or p.data.isinf().any():
            return True
    return False


def _has_nan_grads(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and (p.grad.isnan().any() or p.grad.isinf().any()):
            return True
    return False


def run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    height_embed: HeightEmbedding,
    weights: LossWeights,
    grad_accum: int,
    epoch: int = 0,
    main_: bool = True,
) -> Dict[str, float]:
    model.train(True)
    totals = {"total": 0.0, "map_nll": 0.0, "dist_kl": 0.0, "moment_match": 0.0, "anchor": 0.0,
              "prior_guard": 0.0, "outlier_budget": 0.0, "rmse": 0.0, "mae": 0.0}
    steps = 0
    skipped = 0
    n_total = len(loader)
    optimizer.zero_grad(set_to_none=True)
    for step, raw_batch in enumerate(loader):
        # Guard: if model weights are already NaN, abort immediately.
        if _has_nan_params(model):
            raise RuntimeError(
                f"[epoch {epoch:03d} step {step:05d}] NaN/Inf detected in model parameters. "
                "Training aborted to prevent silent weight corruption."
            )

        steps = step + 1
        batch = to_device(raw_batch, device)
        priors_trans = build_priors_trans(batch)
        out = model(batch["inputs"], height_embed(batch["antenna_height_m"]), priors_trans)
        preds_native = build_preds_native(out)
        priors_native = build_priors(batch)
        loss_terms = combined_loss(batch, out, preds_native, priors_native, weights=weights)
        loss_val = loss_terms["total"]

        # Skip batch if loss is NaN or Inf — do NOT backward through it.
        if not torch.isfinite(loss_val):
            skipped += 1
            if main_:
                print(
                    f"[epoch {epoch:03d} step {steps:05d}/{n_total}] WARNING: non-finite loss "
                    f"({float(loss_val):.4g}), skipping batch ({skipped} skipped so far).",
                    flush=True,
                )
            # Zero out any accumulated gradients for this microstep to keep accum clean.
            if (step + 1) % max(grad_accum, 1) == 0:
                optimizer.zero_grad(set_to_none=True)
            continue

        loss = loss_val / max(grad_accum, 1)
        loss.backward()

        if (step + 1) % max(grad_accum, 1) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if _has_nan_grads(model):
                if main_:
                    print(
                        f"[epoch {epoch:03d} step {steps:05d}/{n_total}] WARNING: NaN/Inf in "
                        "gradients after clipping, skipping optimizer step.",
                        flush=True,
                    )
                skipped += 1
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for key in totals:
            v = float(loss_terms[key].detach().item())
            totals[key] += v if math.isfinite(v) else 0.0

        if main_ and steps % LOG_STEP_INTERVAL == 0:
            avg = totals["total"] / max(steps - skipped, 1)
            print(f"[epoch {epoch:03d} step {steps:05d}/{n_total}] loss={avg:.4f}", flush=True)

    if steps > 0 and steps % max(grad_accum, 1) != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not _has_nan_grads(model):
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if main_ and skipped > 0:
        print(f"[epoch {epoch:03d}] INFO: {skipped} batch(es) skipped due to non-finite loss/grads.", flush=True)

    denom = max(steps - skipped, 1)
    return {key: value / denom for key, value in totals.items()}


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    height_embed: HeightEmbedding,
    weights: LossWeights,
    *,
    store_per_sample: bool = False,
) -> Dict[str, object]:
    model.train(False)
    totals = {"total": 0.0, "map_nll": 0.0, "dist_kl": 0.0, "moment_match": 0.0, "anchor": 0.0,
              "prior_guard": 0.0, "outlier_budget": 0.0, "rmse": 0.0, "mae": 0.0}
    steps = 0
    metrics = MultiTaskMetricAccumulator(store_per_sample=store_per_sample)

    for raw_batch in loader:
        steps += 1
        batch = to_device(raw_batch, device)
        priors_trans = build_priors_trans(batch)
        out = model(batch["inputs"], height_embed(batch["antenna_height_m"]), priors_trans)
        preds_native = build_preds_native(out)
        priors_native = build_priors(batch)
        loss_terms = combined_loss(batch, out, preds_native, priors_native, weights=weights)
        for key in totals:
            totals[key] += float(loss_terms[key].detach().item())
        metrics.update_batch(batch, preds_native, priors_native)

    denom = max(steps, 1)
    return {
        "losses": {key: value / denom for key, value in totals.items()},
        "metrics": metrics.summary(),
    }


def composite_score(summary: Dict[str, object]) -> float:
    flat = summary["metrics"]["model"]["flat"]
    return (
        float(flat["path_loss_rmse_overall_pw"])
        + 0.05 * float(flat["delay_spread_rmse_overall_pw"])
        + 0.10 * float(flat["angular_spread_rmse_overall_pw"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = Try80Cfg.load(args.config)
    set_seed(cfg.seed)

    ddp, rank, world, device = init_distributed(cfg.runtime.device)
    main_ = is_main(rank)
    cfg.runtime.output_dir.mkdir(parents=True, exist_ok=True)

    if main_:
        with open(cfg.runtime.output_dir / "resolved_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "code_marker": CODE_MARKER,
                    "seed": cfg.seed,
                    "data": {**asdict(cfg.data), "hdf5_path": str(cfg.data.hdf5_path),
                              "precomputed_priors_hdf5_path": str(cfg.data.precomputed_priors_hdf5_path) if cfg.data.precomputed_priors_hdf5_path else None},
                    "prior": {**asdict(cfg.prior),
                               "try78_los_calibration_json": str(cfg.prior.try78_los_calibration_json),
                               "try78_nlos_calibration_json": str(cfg.prior.try78_nlos_calibration_json),
                               "try79_calibration_json": str(cfg.prior.try79_calibration_json)},
                    "model": asdict(cfg.model),
                    "losses": asdict(cfg.losses),
                    "training": asdict(cfg.training),
                    "runtime": {**asdict(cfg.runtime), "output_dir": str(cfg.runtime.output_dir),
                                 "resume_checkpoint": str(cfg.runtime.resume_checkpoint) if cfg.runtime.resume_checkpoint else None},
                },
                f,
                indent=2,
            )

    data_cfg = build_data_cfg(cfg)
    train_ds, val_ds, test_ds = build_joint_datasets(data_cfg)
    if main_:
        print(f"[rank0] code_marker={CODE_MARKER} device={device} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    def make_loader(ds: Try80JointDataset, train: bool) -> DataLoader:
        sampler = DistributedSampler(ds, shuffle=train, drop_last=False) if ddp else None
        return DataLoader(
            ds,
            batch_size=cfg.training.batch_size,
            shuffle=(sampler is None and train),
            sampler=sampler,
            num_workers=cfg.training.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            persistent_workers=cfg.training.num_workers > 0,
        )

    train_loader = make_loader(train_ds, train=True)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=max(0, cfg.training.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=max(0, cfg.training.num_workers // 2) > 0,
    )

    model = Try80Model(build_model_cfg(cfg)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: lr_lambda(epoch, cfg.training.warmup_epochs, cfg.training.epochs),
    )
    height_embed = HeightEmbedding()
    loss_weights = LossWeights(**asdict(cfg.losses))

    start_epoch = 0
    best_score = float("inf")
    history: List[Dict[str, object]] = []

    if not cfg.runtime.resume_checkpoint:
        best_ckpt = cfg.runtime.output_dir / "best_model.pt"
        if best_ckpt.exists():
            cfg.runtime.resume_checkpoint = best_ckpt

    if cfg.runtime.resume_checkpoint and cfg.runtime.resume_checkpoint.exists():
        state = torch.load(cfg.runtime.resume_checkpoint, map_location=device)
        model.load_state_dict(state["model"], strict=False)
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_score = float(state.get("best_score", best_score))
        history = list(state.get("history", []))
        if main_:
            print(f"[rank0] resumed from {cfg.runtime.resume_checkpoint} at epoch {start_epoch}")

    model_wrapped: torch.nn.Module = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=False) if ddp and device.type == "cuda" else model

    patience_counter = 0
    best_summary: Dict[str, object] | None = None
    for epoch in range(start_epoch, cfg.training.epochs):
        epoch_start = time.time()
        if ddp and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_time_start = time.time()
        train_losses = run_train_epoch(
            model=model_wrapped,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            height_embed=height_embed,
            weights=loss_weights,
            grad_accum=cfg.training.grad_accum_steps,
            epoch=epoch,
            main_=main_,
        )
        train_seconds = time.time() - train_time_start
        scheduler.step()

        maybe_barrier(ddp)
        if main_:
            val_time_start = time.time()
            val_summary = run_validation(
                model=model,
                loader=val_loader,
                device=device,
                height_embed=height_embed,
                weights=loss_weights,
                store_per_sample=False,
            )
            val_seconds = time.time() - val_time_start
            score = composite_score(val_summary)
            epoch_row = {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_seconds": train_seconds,
                "val_seconds": val_seconds,
                "epoch_seconds": time.time() - epoch_start,
                "train": train_losses,
                "val": val_summary,
                "score": score,
            }
            history.append(epoch_row)
            (cfg.runtime.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

            last_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_score": best_score,
                "history": history,
                "model_cfg": asdict(build_model_cfg(cfg)),
            }
            torch.save(last_state, cfg.runtime.output_dir / "last_model.pt")

            flat_model = val_summary["metrics"]["model"]["flat"]
            flat_prior = val_summary["metrics"]["prior"]["flat"]
            print(
                f"[epoch {epoch:03d}] "
                f"train_total={train_losses['total']:.4f} "
                f"val_path={flat_model['path_loss_rmse_overall_pw']:.4f} "
                f"val_delay={flat_model['delay_spread_rmse_overall_pw']:.4f} "
                f"val_angular={flat_model['angular_spread_rmse_overall_pw']:.4f} "
                f"prior_path={flat_prior['path_loss_rmse_overall_pw']:.4f} "
                f"score={score:.4f}"
            )

            improved = score < best_score
            if improved:
                best_score = score
                best_summary = val_summary
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_score": best_score,
                        "history": history,
                        "model_cfg": asdict(build_model_cfg(cfg)),
                    },
                    cfg.runtime.output_dir / "best_model.pt",
                )
                (cfg.runtime.output_dir / "best_summary_val.json").write_text(json.dumps(val_summary, indent=2), encoding="utf-8")
            else:
                patience_counter += 1
            if patience_counter >= cfg.training.patience:
                print(f"[rank0] early stopping at epoch {epoch} (patience={cfg.training.patience})")
                break
        maybe_barrier(ddp)

    if main_ and best_summary is not None:
        print("[rank0] training complete")
        print(json.dumps(best_summary["metrics"]["model"]["flat"], indent=2))

    maybe_barrier(ddp)
    if ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
