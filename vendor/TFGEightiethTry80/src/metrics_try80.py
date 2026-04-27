"""Try 80 - transforms and pixel-weighted metric aggregation."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, MutableMapping, Optional

import torch


TASKS = ("path_loss", "delay_spread", "angular_spread")
SCOPES = ("overall", "los", "nlos")


def transform_target(task: str, x: torch.Tensor) -> torch.Tensor:
    if task == "path_loss":
        return x
    return torch.log1p(torch.clamp(x, min=0.0))


def inverse_transform(task: str, x: torch.Tensor) -> torch.Tensor:
    if task == "path_loss":
        return x
    return torch.expm1(x)


def rmse_from_sse(sse: float, n: int) -> float:
    return math.sqrt(sse / n) if n > 0 else float("nan")


def mae_from_sae(sae: float, n: int) -> float:
    return sae / n if n > 0 else float("nan")


@dataclass
class _Stat:
    sse: float = 0.0
    sae: float = 0.0
    n: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        if not torch.any(mask):
            return
        diff = pred[mask] - target[mask]
        self.sse += float((diff * diff).sum().item())
        self.sae += float(diff.abs().sum().item())
        self.n += int(mask.sum().item())

    def summary(self) -> Dict[str, float | int]:
        return {
            "rmse_pw": rmse_from_sse(self.sse, self.n),
            "mae_pw": mae_from_sae(self.sae, self.n),
            "n_pixels": self.n,
        }


@dataclass
class MultiTaskMetricAccumulator:
    store_per_sample: bool = False
    model_stats: Dict[str, Dict[str, _Stat]] = field(default_factory=dict)
    prior_stats: Dict[str, Dict[str, _Stat]] = field(default_factory=dict)
    model_expert_stats: Dict[str, Dict[str, _Stat]] = field(default_factory=dict)
    prior_expert_stats: Dict[str, Dict[str, _Stat]] = field(default_factory=dict)
    per_sample_rows: List[Dict[str, object]] = field(default_factory=list)
    n_samples: int = 0

    def __post_init__(self) -> None:
        if not self.model_stats:
            self.model_stats = {task: {scope: _Stat() for scope in SCOPES} for task in TASKS}
        if not self.prior_stats:
            self.prior_stats = {task: {scope: _Stat() for scope in SCOPES} for task in TASKS}

    def update_batch(
        self,
        batch: Dict[str, object],
        preds_native: Dict[str, torch.Tensor],
        priors_native: Dict[str, torch.Tensor],
    ) -> None:
        bsz = int(preds_native["path_loss"].shape[0])
        los_mask = batch["los_mask"]
        nlos_mask = batch["nlos_mask"]
        top3 = batch["topology_class_3"]
        ant = batch["antenna_bin"]
        cities = batch["city"]
        samples = batch["sample"]

        for bi in range(bsz):
            expert_key = f"{top3[bi]}|{ant[bi]}"
            self.n_samples += 1
            sample_row: Dict[str, object] = {
                "city": cities[bi],
                "sample": samples[bi],
                "topology_class_3": top3[bi],
                "antenna_bin": ant[bi],
            }
            for task in TASKS:
                target = batch[f"{task}_target"][bi, 0]
                valid = batch[f"{task}_mask"][bi, 0] > 0.5
                los = valid & (los_mask[bi, 0] > 0.5)
                nlos = valid & (nlos_mask[bi, 0] > 0.5)
                pred = preds_native[task][bi, 0]
                prior = priors_native[task][bi, 0]

                self.model_stats[task]["overall"].update(pred, target, valid)
                self.model_stats[task]["los"].update(pred, target, los)
                self.model_stats[task]["nlos"].update(pred, target, nlos)
                self.prior_stats[task]["overall"].update(prior, target, valid)
                self.prior_stats[task]["los"].update(prior, target, los)
                self.prior_stats[task]["nlos"].update(prior, target, nlos)

                self._expert_stat(self.model_expert_stats, expert_key, task).update(pred, target, valid)
                self._expert_stat(self.prior_expert_stats, expert_key, task).update(prior, target, valid)

                if self.store_per_sample:
                    sample_row[f"{task}_model_rmse_overall"] = _rmse_tensor(pred, target, valid)
                    sample_row[f"{task}_model_rmse_los"] = _rmse_tensor(pred, target, los)
                    sample_row[f"{task}_model_rmse_nlos"] = _rmse_tensor(pred, target, nlos)
                    sample_row[f"{task}_prior_rmse_overall"] = _rmse_tensor(prior, target, valid)
                    sample_row[f"{task}_prior_rmse_los"] = _rmse_tensor(prior, target, los)
                    sample_row[f"{task}_prior_rmse_nlos"] = _rmse_tensor(prior, target, nlos)

            if self.store_per_sample:
                self.per_sample_rows.append(sample_row)

    def _expert_stat(self, store: MutableMapping[str, Dict[str, _Stat]], expert_key: str, task: str) -> _Stat:
        if expert_key not in store:
            store[expert_key] = {name: _Stat() for name in TASKS}
        return store[expert_key][task]

    def summary(self) -> Dict[str, object]:
        model_agg = self._aggregate_summary(self.model_stats)
        prior_agg = self._aggregate_summary(self.prior_stats)
        model_experts = {key: {task: stats[task].summary() for task in TASKS} for key, stats in sorted(self.model_expert_stats.items())}
        prior_experts = {key: {task: stats[task].summary() for task in TASKS} for key, stats in sorted(self.prior_expert_stats.items())}

        return {
            "n_samples": self.n_samples,
            "model": {
                "aggregate": model_agg,
                "macro_experts": model_experts,
                "flat": self._flatten(model_agg, model_experts),
            },
            "prior": {
                "aggregate": prior_agg,
                "macro_experts": prior_experts,
                "flat": self._flatten(prior_agg, prior_experts),
            },
            "per_sample": self.per_sample_rows if self.store_per_sample else [],
        }

    @staticmethod
    def _aggregate_summary(store: Dict[str, Dict[str, _Stat]]) -> Dict[str, Dict[str, float | int]]:
        return {
            task: {
                "overall": store[task]["overall"].summary(),
                "los": store[task]["los"].summary(),
                "nlos": store[task]["nlos"].summary(),
            }
            for task in TASKS
        }

    @staticmethod
    def _flatten(
        agg: Dict[str, Dict[str, Dict[str, float | int]]],
        expert: Dict[str, Dict[str, Dict[str, float | int]]],
    ) -> Dict[str, float | int]:
        out: Dict[str, float | int] = {}
        for task in TASKS:
            out[f"{task}_rmse_overall_pw"] = agg[task]["overall"]["rmse_pw"]
            out[f"{task}_rmse_los_pw"] = agg[task]["los"]["rmse_pw"]
            out[f"{task}_rmse_nlos_pw"] = agg[task]["nlos"]["rmse_pw"]
            out[f"{task}_mae_overall_pw"] = agg[task]["overall"]["mae_pw"]
        for expert_key, metrics in expert.items():
            safe_key = expert_key.replace("|", "_")
            for task in TASKS:
                out[f"{safe_key}_{task}_rmse_pw"] = metrics[task]["rmse_pw"]
        return out


def _rmse_tensor(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not torch.any(mask):
        return float("nan")
    diff = pred[mask] - target[mask]
    return float(torch.sqrt((diff * diff).mean()).item())
