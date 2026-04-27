"""Try 80 - YAML config loader for the joint multi-task prior-anchored model."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataSection:
    hdf5_path: Path
    image_size: int = 513
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42
    topology_norm_m: float = 90.0
    path_loss_no_data_mask_column: Optional[str] = None
    derive_no_data_from_non_ground: bool = True
    augment_d4: bool = True
    precomputed_priors_hdf5_path: Optional[Path] = None


@dataclass
class PriorSection:
    try78_los_calibration_json: Path
    try78_nlos_calibration_json: Path
    try79_calibration_json: Path


@dataclass
class ModelSection:
    in_channels: int = 9
    cond_dim: int = 128
    height_embed_dim: int = 32
    base_width: int = 96
    num_components: int = 3
    decoder_dropout: float = 0.10
    alpha_bias: float = -2.0
    sigma_min: float = 0.05
    sigma_max: float = 3.00
    path_residual_los_max: float = 2.0
    path_residual_nlos_max: float = 4.0
    delay_residual_los_max: float = 30.0
    delay_residual_nlos_max: float = 40.0
    angular_residual_los_max: float = 9.0
    angular_residual_nlos_max: float = 13.0


@dataclass
class LossSection:
    map_nll: float = 1.0
    dist_kl: float = 0.25
    moment_match: float = 0.10
    anchor: float = 0.05
    prior_guard: float = 0.10
    rmse: float = 0.40
    mae: float = 0.10
    outlier_budget: float = 0.02
    outlier_budget_threshold: float = 0.60


@dataclass
class TrainingSection:
    optimizer: str = "adamw"
    lr: float = 2.0e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    epochs: int = 120
    batch_size: int = 1
    grad_accum_steps: int = 8
    patience: int = 20
    num_workers: int = 4


@dataclass
class RuntimeSection:
    device: str = "cuda"
    output_dir: Path = field(default_factory=lambda: Path("outputs/try80_joint_big"))
    resume_checkpoint: Optional[Path] = None


@dataclass
class Try80Cfg:
    seed: int
    data: DataSection
    prior: PriorSection
    model: ModelSection
    losses: LossSection
    training: TrainingSection
    runtime: RuntimeSection

    @classmethod
    def load(cls, path: Path) -> "Try80Cfg":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw, root=path.parent)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], root: Path) -> "Try80Cfg":
        d = raw.get("data", {}) or {}
        data = DataSection(
            hdf5_path=_resolve_path(root, d.get("hdf5_path")),
            image_size=int(d.get("image_size", 513)),
            val_ratio=float(d.get("val_ratio", 0.15)),
            test_ratio=float(d.get("test_ratio", 0.15)),
            split_seed=int(d.get("split_seed", 42)),
            topology_norm_m=float(d.get("topology_norm_m", 90.0)),
            path_loss_no_data_mask_column=(
                str(d.get("path_loss_no_data_mask_column")).strip()
                if d.get("path_loss_no_data_mask_column") is not None
                else None
            ),
            derive_no_data_from_non_ground=bool(d.get("derive_no_data_from_non_ground", True)),
            augment_d4=bool(d.get("augment_d4", True)),
            precomputed_priors_hdf5_path=(
                _resolve_path(root, d.get("precomputed_priors_hdf5_path"))
                if d.get("precomputed_priors_hdf5_path")
                else None
            ),
        )

        p = raw.get("prior", {}) or {}
        prior = PriorSection(
            try78_los_calibration_json=_resolve_path(root, p.get("try78_los_calibration_json")),
            try78_nlos_calibration_json=_resolve_path(root, p.get("try78_nlos_calibration_json")),
            try79_calibration_json=_resolve_path(root, p.get("try79_calibration_json")),
        )

        m = raw.get("model", {}) or {}
        model = ModelSection(
            in_channels=int(m.get("in_channels", 9)),
            cond_dim=int(m.get("cond_dim", 128)),
            height_embed_dim=int(m.get("height_embed_dim", 32)),
            base_width=int(m.get("base_width", 96)),
            num_components=int(m.get("num_components", 3)),
            decoder_dropout=float(m.get("decoder_dropout", 0.10)),
            alpha_bias=float(m.get("alpha_bias", -2.0)),
            sigma_min=float(m.get("sigma_min", 0.05)),
            sigma_max=float(m.get("sigma_max", 3.00)),
            path_residual_los_max=float(m.get("path_residual_los_max", 2.0)),
            path_residual_nlos_max=float(m.get("path_residual_nlos_max", 4.0)),
            delay_residual_los_max=float(m.get("delay_residual_los_max", 30.0)),
            delay_residual_nlos_max=float(m.get("delay_residual_nlos_max", 40.0)),
            angular_residual_los_max=float(m.get("angular_residual_los_max", 9.0)),
            angular_residual_nlos_max=float(m.get("angular_residual_nlos_max", 13.0)),
        )

        l = raw.get("losses", {}) or {}
        losses = LossSection(
            map_nll=float(l.get("map_nll", 1.0)),
            dist_kl=float(l.get("dist_kl", 0.25)),
            moment_match=float(l.get("moment_match", 0.10)),
            anchor=float(l.get("anchor", 0.05)),
            prior_guard=float(l.get("prior_guard", 0.10)),
            rmse=float(l.get("rmse", 0.40)),
            mae=float(l.get("mae", 0.10)),
            outlier_budget=float(l.get("outlier_budget", 0.02)),
            outlier_budget_threshold=float(l.get("outlier_budget_threshold", 0.60)),
        )

        t = raw.get("training", {}) or {}
        training = TrainingSection(
            optimizer=str(t.get("optimizer", "adamw")),
            lr=float(t.get("lr", 2.0e-4)),
            weight_decay=float(t.get("weight_decay", 0.01)),
            warmup_epochs=int(t.get("warmup_epochs", 2)),
            epochs=int(t.get("epochs", 120)),
            batch_size=int(t.get("batch_size", 1)),
            grad_accum_steps=int(t.get("grad_accum_steps", 8)),
            patience=int(t.get("patience", 20)),
            num_workers=int(t.get("num_workers", 4)),
        )

        r = raw.get("runtime", {}) or {}
        output_dir = r.get("output_dir", "outputs/try80_joint_big")
        resume = r.get("resume_checkpoint", "") or None
        runtime = RuntimeSection(
            device=str(r.get("device", "cuda")),
            output_dir=_resolve_path(root, output_dir),
            resume_checkpoint=_resolve_path(root, resume) if resume else None,
        )

        return cls(
            seed=int(raw.get("seed", 42)),
            data=data,
            prior=prior,
            model=model,
            losses=losses,
            training=training,
            runtime=runtime,
        )


def _resolve_path(root: Path, value: Any) -> Path:
    if value in (None, ""):
        raise ValueError("Required path value is missing in Try80 config")
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()
