"""Try 80 - joint multi-task losses for prior-anchored residual GMM prediction."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from .metrics_try80 import TASKS, transform_target


EPS = 1.0e-6
LOG_2PI = math.log(2.0 * math.pi)

RESIDUAL_HIST_RANGE = {
    "path_loss": {"los": 2.0, "nlos": 4.0},
    "delay_spread": {"los": 30.0, "nlos": 40.0},
    "angular_spread": {"los": 9.0, "nlos": 13.0},
}


@dataclass
class LossWeights:
    map_nll: float = 1.0
    dist_kl: float = 0.25
    moment_match: float = 0.10
    anchor: float = 0.05
    prior_guard: float = 0.10
    rmse: float = 0.40
    mae: float = 0.10
    outlier_budget: float = 0.02
    outlier_budget_threshold: float = 0.60


def combined_loss(
    batch: Dict[str, torch.Tensor | object],
    outputs: Dict[str, Dict[str, torch.Tensor]],
    preds_native: Dict[str, torch.Tensor],
    priors_native: Dict[str, torch.Tensor],
    weights: LossWeights = LossWeights(),
) -> Dict[str, torch.Tensor | Dict[str, Dict[str, float]]]:
    total = None
    agg = {
        "map_nll": 0.0,
        "dist_kl": 0.0,
        "moment_match": 0.0,
        "anchor": 0.0,
        "prior_guard": 0.0,
        "outlier_budget": 0.0,
        "rmse": 0.0,
        "mae": 0.0,
    }
    per_task: Dict[str, Dict[str, float]] = {}

    los_mask = batch["los_mask"]
    nlos_mask = batch["nlos_mask"]

    for task in TASKS:
        target_native = batch[f"{task}_target"]
        valid_mask = batch[f"{task}_mask"]
        target_trans = transform_target(task, target_native)
        pred_trans = outputs[task]["pred_trans"]
        pred_native = preds_native[task]

        l_map = _map_nll_loss(task, target_trans, valid_mask, los_mask, nlos_mask, outputs[task])
        l_kl = _dist_kl_loss(
            task,
            target_native,
            priors_native[task],
            valid_mask,
            los_mask,
            nlos_mask,
            outputs[task]["pi"],
            outputs[task]["global_delta_native"],
        )
        l_mom = _moment_match_loss(target_native, pred_native, valid_mask)
        l_anchor = _anchor_loss(outputs[task]["alpha"], outputs[task]["global_delta"], outputs[task]["local_delta"])
        l_guard = _prior_guard_loss(target_native, pred_native, priors_native[task], valid_mask)
        l_out = _outlier_budget_loss(outputs[task]["p"], valid_mask, threshold=weights.outlier_budget_threshold)
        l_rmse = _rmse_loss(target_native, pred_native, valid_mask)
        l_mae = _mae_loss(target_native, pred_native, valid_mask)

        task_total = (
            weights.map_nll * l_map
            + weights.dist_kl * l_kl
            + weights.moment_match * l_mom
            + weights.anchor * l_anchor
            + weights.prior_guard * l_guard
            + weights.outlier_budget * l_out
            + weights.rmse * l_rmse
            + weights.mae * l_mae
        )
        total = task_total if total is None else total + task_total

        agg["map_nll"] += float(l_map.detach().item())
        agg["dist_kl"] += float(l_kl.detach().item())
        agg["moment_match"] += float(l_mom.detach().item())
        agg["anchor"] += float(l_anchor.detach().item())
        agg["prior_guard"] += float(l_guard.detach().item())
        agg["outlier_budget"] += float(l_out.detach().item())
        agg["rmse"] += float(l_rmse.detach().item())
        agg["mae"] += float(l_mae.detach().item())
        per_task[task] = {
            "map_nll": float(l_map.detach().item()),
            "dist_kl": float(l_kl.detach().item()),
            "moment_match": float(l_mom.detach().item()),
            "anchor": float(l_anchor.detach().item()),
            "prior_guard": float(l_guard.detach().item()),
            "outlier_budget": float(l_out.detach().item()),
            "rmse": float(l_rmse.detach().item()),
            "mae": float(l_mae.detach().item()),
        }

    if total is None:
        total = torch.zeros((), device=next(iter(outputs.values()))["pred_trans"].device)
    # Do NOT silently mask NaN/Inf here — let the training loop detect and skip bad batches.
    # nan_to_num was hiding numerical instability and allowing corrupted weights to go unnoticed.
    return {
        "total": total,
        "map_nll": torch.tensor(agg["map_nll"], device=total.device),
        "dist_kl": torch.tensor(agg["dist_kl"], device=total.device),
        "moment_match": torch.tensor(agg["moment_match"], device=total.device),
        "anchor": torch.tensor(agg["anchor"], device=total.device),
        "prior_guard": torch.tensor(agg["prior_guard"], device=total.device),
        "outlier_budget": torch.tensor(agg["outlier_budget"], device=total.device),
        "rmse": torch.tensor(agg["rmse"], device=total.device),
        "mae": torch.tensor(agg["mae"], device=total.device),
        "per_task": per_task,
    }


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask > 0.5
    if not torch.any(active):
        return x.new_zeros(())
    return x.masked_select(active).mean()


def _map_nll_loss(
    task: str,
    target_trans: torch.Tensor,
    valid_mask: torch.Tensor,
    los_mask: torch.Tensor,
    nlos_mask: torch.Tensor,
    out: Dict[str, torch.Tensor],
) -> torch.Tensor:
    mu = torch.nan_to_num(out["mu"], nan=0.0, posinf=1.0e3, neginf=-1.0e3)
    sigma = torch.nan_to_num(out["sigma"], nan=1.0, posinf=1.0e3, neginf=1.0).clamp_min(EPS)
    p = torch.nan_to_num(out["p"], nan=1.0 / max(out["p"].shape[2], 1), posinf=1.0, neginf=EPS).clamp_min(EPS)
    p = p / p.sum(dim=2, keepdim=True).clamp_min(EPS)
    masks = torch.stack([valid_mask * los_mask, valid_mask * nlos_mask], dim=1)  # (B, 2, 1, H, W)
    target = torch.nan_to_num(target_trans, nan=0.0, posinf=1.0e3, neginf=-1.0e3).unsqueeze(1).unsqueeze(2)
    diff = target - mu
    log_pdf = -0.5 * ((diff * diff) / (sigma * sigma) + 2.0 * sigma.log() + LOG_2PI)
    log_pdf = torch.nan_to_num(log_pdf, nan=-1.0e6, neginf=-1.0e6, posinf=1.0e6)
    log_mix = torch.logsumexp(log_pdf + p.log(), dim=2)
    log_mix = torch.nan_to_num(log_mix, nan=-1.0e6, neginf=-1.0e6, posinf=1.0e6)
    losses = []
    for region in range(2):
        region_mask = masks[:, region]
        if torch.any(region_mask > 0.5):
            losses.append(torch.nan_to_num(_masked_mean(-log_mix[:, region : region + 1], region_mask), nan=0.0, posinf=1.0e6, neginf=-1.0e6))
    if not losses:
        return target_trans.new_zeros(())
    return torch.nan_to_num(torch.stack(losses).mean(), nan=0.0, posinf=1.0e6, neginf=-1.0e6)


def _dist_kl_loss(
    task: str,
    target_native: torch.Tensor,
    prior_native: torch.Tensor,
    valid_mask: torch.Tensor,
    los_mask: torch.Tensor,
    nlos_mask: torch.Tensor,
    pi: torch.Tensor,
    global_delta: torch.Tensor,
    n_bins: int = 64,
) -> torch.Tensor:
    losses = []
    for region_idx, region_name in enumerate(("los", "nlos")):
        region_mask = (valid_mask > 0.5) & ((los_mask if region_name == "los" else nlos_mask) > 0.5)
        if not torch.any(region_mask):
            continue
        residual = target_native - prior_native
        clip = RESIDUAL_HIST_RANGE[task][region_name]
        emp = _soft_bin(residual, region_mask, -clip, clip, n_bins=n_bins)
        mix = _global_delta_pmf(pi[:, region_idx], global_delta[:, region_idx], -clip, clip, n_bins=n_bins)
        emp = emp.clamp_min(EPS)
        mix = mix.clamp_min(EPS)
        losses.append(torch.nan_to_num((emp * (emp.log() - mix.log())).sum(dim=1).mean(), nan=0.0, posinf=1.0e6, neginf=-1.0e6))
    if not losses:
        return target_native.new_zeros(())
    return torch.stack(losses).mean()


def _soft_bin(values: torch.Tensor, mask: torch.Tensor, lo: float, hi: float, n_bins: int, kernel_sigma_bins: float = 0.6) -> torch.Tensor:
    centers = torch.linspace(lo, hi, n_bins, device=values.device)
    binwidth = (hi - lo) / max(n_bins - 1, 1)
    diffs = (values.unsqueeze(-1) - centers) / (kernel_sigma_bins * binwidth + EPS)
    weights = torch.exp(-0.5 * diffs.pow(2))
    weighted = weights * mask.unsqueeze(-1)
    counts = weighted.flatten(1, 3).sum(dim=1)
    total = counts.sum(dim=1, keepdim=True).clamp_min(EPS)
    return counts / total


def _global_delta_pmf(pi: torch.Tensor, global_delta: torch.Tensor, lo: float, hi: float, n_bins: int) -> torch.Tensor:
    centers = torch.linspace(lo, hi, n_bins, device=pi.device)
    binwidth = (hi - lo) / max(n_bins - 1, 1)
    mu = global_delta.squeeze(-1).squeeze(-1)
    sigma = torch.full_like(mu, max(binwidth * 1.5, EPS))
    diffs = (centers.view(1, 1, -1) - mu.unsqueeze(-1)) / sigma.unsqueeze(-1).clamp_min(EPS)
    pdf = torch.exp(-0.5 * diffs.pow(2)) / (sigma.unsqueeze(-1) * math.sqrt(2.0 * math.pi))
    pmf = (pi.unsqueeze(-1) * pdf * binwidth).sum(dim=1)
    total = pmf.sum(dim=1, keepdim=True).clamp_min(EPS)
    return pmf / total


def _moment_match_loss(target: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    tgt_mean = (target * mask).sum(dim=(1, 2, 3)) / denom
    pred_mean = (pred * mask).sum(dim=(1, 2, 3)) / denom
    tgt_var = ((target - tgt_mean.view(-1, 1, 1, 1)) ** 2 * mask).sum(dim=(1, 2, 3)) / denom
    pred_var = ((pred - pred_mean.view(-1, 1, 1, 1)) ** 2 * mask).sum(dim=(1, 2, 3)) / denom
    return ((pred_mean - tgt_mean) ** 2).mean() + ((pred_var.sqrt() - tgt_var.sqrt()) ** 2).mean()


def _anchor_loss(alpha: torch.Tensor, global_delta: torch.Tensor, local_delta: torch.Tensor) -> torch.Tensor:
    return (alpha.pow(2).mean() + 0.25 * global_delta.pow(2).mean() + 0.50 * local_delta.pow(2).mean())


def _prior_guard_loss(target: torch.Tensor, pred: torch.Tensor, prior: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask > 0.5
    if not torch.any(active):
        return target.new_zeros(())
    pred_sq = (pred - target) ** 2
    prior_sq = (prior - target) ** 2
    return F.relu(pred_sq - prior_sq).masked_select(active).mean()


def _outlier_budget_loss(p: torch.Tensor, valid_mask: torch.Tensor, threshold: float) -> torch.Tensor:
    outlier_p = p[:, :, -1:]
    mask = valid_mask.unsqueeze(1)
    mean_used = _masked_mean(outlier_p, mask)
    return F.relu(mean_used - threshold)


def _rmse_loss(target: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return torch.sqrt((diff.sum() / denom).clamp_min(EPS))


def _mae_loss(target: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom
