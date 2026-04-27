"""Try 80 - joint dataset utilities for path loss + spread prediction."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .priors_try80 import Try80PriorComputer


SampleRef = Tuple[str, str]
LOG1P_DELAY_NORM = float(math.log1p(400.0))
LOG1P_ANGULAR_NORM = float(math.log1p(90.0))
PATH_LOSS_MIN_DB = 20.0


@dataclass
class HeightEmbedding:
    n_freq: int = 16
    min_height_m: float = 12.0
    max_height_m: float = 478.0

    def __call__(self, h_m: torch.Tensor) -> torch.Tensor:
        log_min = math.log(max(self.min_height_m, 1e-3))
        log_max = math.log(max(self.max_height_m, self.min_height_m + 1e-3))
        freqs = torch.exp(torch.linspace(log_min, log_max, self.n_freq, device=h_m.device))
        angles = h_m.unsqueeze(-1) / freqs.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


@dataclass
class Try80DataConfig:
    hdf5_path: Path
    try78_los_calibration_json: Path
    try78_nlos_calibration_json: Path
    try79_calibration_json: Path
    image_size: int = 513
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42
    topology_norm_m: float = 90.0
    path_loss_no_data_mask_column: Optional[str] = None
    derive_no_data_from_non_ground: bool = True
    augment_d4: bool = True
    precomputed_priors_hdf5_path: Optional[Path] = None


def read_field(grp: h5py.Group, name: str) -> np.ndarray:
    if name in grp:
        return np.asarray(grp[name][...], dtype=np.float32)
    alts = [name + "_map", name.replace("_", "") + "_map"]
    for alt in alts:
        if alt in grp:
            return np.asarray(grp[alt][...], dtype=np.float32)
    raise KeyError(f"Missing field {name!r} in sample with keys {list(grp.keys())}")


def list_hdf5_samples(hdf5_path: Path) -> List[SampleRef]:
    refs: List[SampleRef] = []
    with h5py.File(str(hdf5_path), "r") as handle:
        for city in sorted(handle.keys()):
            for sample in sorted(handle[city].keys()):
                refs.append((city, sample))
    return refs


def split_city_holdout(
    sample_refs: Sequence[SampleRef],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_seed: int = 42,
) -> Tuple[List[SampleRef], List[SampleRef], List[SampleRef]]:
    refs = list(sample_refs)
    if len(refs) < 2:
        return refs, list(refs), []
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    total = len(refs)
    test_size = max(1, int(round(total * test_ratio))) if test_ratio > 0.0 else 0
    val_size = max(1, int(round(total * val_ratio))) if val_ratio > 0.0 else 0

    rng = random.Random(split_seed)
    by_city: Dict[str, List[SampleRef]] = {}
    for city, sample in refs:
        by_city.setdefault(city, []).append((city, sample))
    city_names = list(by_city.keys())

    if len(city_names) < 3:
        rng.shuffle(refs)
        test_refs = refs[:test_size]
        val_refs = refs[test_size : test_size + val_size]
        train_refs = refs[test_size + val_size :]
        return train_refs, val_refs, test_refs

    rng.shuffle(city_names)
    train_refs: List[SampleRef] = []
    val_refs: List[SampleRef] = []
    test_refs: List[SampleRef] = []
    test_city_count = 0
    val_city_count = 0
    for city in city_names:
        remaining = len(city_names) - test_city_count - val_city_count
        city_refs = by_city[city]
        if len(test_refs) < test_size and remaining > 2:
            test_refs.extend(city_refs)
            test_city_count += 1
            continue
        if len(val_refs) < val_size and remaining > 1:
            val_refs.extend(city_refs)
            val_city_count += 1
            continue
        train_refs.extend(city_refs)
    if not train_refs:
        rng.shuffle(refs)
        test_refs = refs[:test_size]
        val_refs = refs[test_size : test_size + val_size]
        train_refs = refs[test_size + val_size :]
    return train_refs, val_refs, test_refs


def _valid_path_loss_mask(
    grp: h5py.Group,
    path_loss: np.ndarray,
    ground: np.ndarray,
    no_data_mask_column: Optional[str],
    derive_no_data_from_non_ground: bool,
) -> np.ndarray:
    valid = np.isfinite(path_loss) & (path_loss >= PATH_LOSS_MIN_DB)
    if no_data_mask_column:
        key = str(no_data_mask_column).strip()
        if key and key in grp:
            no_data = np.asarray(grp[key][...], dtype=np.float32) > 0.5
            valid &= ~no_data
    if derive_no_data_from_non_ground:
        valid &= ground
    return valid


def _valid_nonnegative_mask(target: np.ndarray, ground: np.ndarray) -> np.ndarray:
    return ground & np.isfinite(target) & (target >= 0.0)


class Try80JointDataset(Dataset):
    def __init__(
        self,
        cfg: Try80DataConfig,
        sample_refs: Sequence[SampleRef],
        augment: bool = False,
    ) -> None:
        self.cfg = cfg
        self._refs = list(sample_refs)
        self.augment = bool(augment)
        self._prior_computer: Optional[Try80PriorComputer] = None

    def __len__(self) -> int:
        return len(self._refs)

    def _get_prior_computer(self) -> Try80PriorComputer:
        if self._prior_computer is None:
            self._prior_computer = Try80PriorComputer(
                try78_los_calibration_json=self.cfg.try78_los_calibration_json,
                try78_nlos_calibration_json=self.cfg.try78_nlos_calibration_json,
                try79_calibration_json=self.cfg.try79_calibration_json,
            )
        return self._prior_computer

    @staticmethod
    def _apply_aug(arrays: List[np.ndarray]) -> List[np.ndarray]:
        k = random.randint(0, 3)
        flip_h = random.random() < 0.5
        flip_v = random.random() < 0.5
        out: List[np.ndarray] = []
        for a in arrays:
            b = np.rot90(a, k=k, axes=(-2, -1)) if k else a
            if flip_h:
                b = b[..., :, ::-1]
            if flip_v:
                b = b[..., ::-1, :]
            out.append(np.ascontiguousarray(b))
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        city, sample = self._refs[idx]
        with h5py.File(str(self.cfg.hdf5_path), "r") as handle:
            grp = handle[city][sample]
            topology = np.asarray(grp["topology_map"][...], dtype=np.float32)
            los_mask = np.asarray(grp["los_mask"][...], dtype=np.float32)
            path_loss = np.asarray(grp["path_loss"][...], dtype=np.float32)
            delay_spread = read_field(grp, "delay_spread")
            angular_spread = read_field(grp, "angular_spread")
            uav_height = float(np.asarray(grp["uav_height"][...]).reshape(-1)[0])

            ground = topology == 0.0
            valid_path = _valid_path_loss_mask(
                grp,
                path_loss,
                ground,
                self.cfg.path_loss_no_data_mask_column,
                self.cfg.derive_no_data_from_non_ground,
            )
            valid_delay = _valid_nonnegative_mask(delay_spread, ground)
            valid_angular = _valid_nonnegative_mask(angular_spread, ground)

        priors = self._load_or_compute_priors(city, sample, topology, los_mask, uav_height)

        ground_f = ground.astype(np.float32)
        los = los_mask * ground_f
        nlos = (1.0 - los_mask) * ground_f
        topology_input = topology * ground_f / max(self.cfg.topology_norm_m, 1e-3)
        path_prior = priors.path_loss_prior.astype(np.float32)
        path_prior_los = priors.path_loss_los_prior.astype(np.float32)
        path_prior_nlos = priors.path_loss_nlos_prior.astype(np.float32)
        delay_prior = priors.delay_spread_prior.astype(np.float32)
        angular_prior = priors.angular_spread_prior.astype(np.float32)

        channels = np.nan_to_num(
            np.stack(
                [
                    topology_input,
                    los,
                    nlos,
                    ground_f,
                    path_prior / 180.0,
                    path_prior_los / 180.0,
                    path_prior_nlos / 180.0,
                    np.log1p(np.clip(delay_prior, 0.0, None)) / LOG1P_DELAY_NORM,
                    np.log1p(np.clip(angular_prior, 0.0, None)) / LOG1P_ANGULAR_NORM,
                ],
                axis=0,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        path_loss = np.where(valid_path, path_loss, 0.0).astype(np.float32)
        delay_spread = np.where(valid_delay, delay_spread, 0.0).astype(np.float32)
        angular_spread = np.where(valid_angular, angular_spread, 0.0).astype(np.float32)
        valid_path_f = valid_path.astype(np.float32)
        valid_delay_f = valid_delay.astype(np.float32)
        valid_angular_f = valid_angular.astype(np.float32)

        arrays: List[np.ndarray] = [
            channels,
            path_loss,
            delay_spread,
            angular_spread,
            valid_path_f,
            valid_delay_f,
            valid_angular_f,
            path_prior,
            path_prior_los,
            path_prior_nlos,
            delay_prior,
            angular_prior,
            ground_f,
            los,
            nlos,
        ]
        if self.augment and self.cfg.augment_d4:
            (
                channels,
                path_loss,
                delay_spread,
                angular_spread,
                valid_path_f,
                valid_delay_f,
                valid_angular_f,
                path_prior,
                path_prior_los,
                path_prior_nlos,
                delay_prior,
                angular_prior,
                ground_f,
                los,
                nlos,
            ) = self._apply_aug(arrays)

        return {
            "city": city,
            "sample": sample,
            "inputs": torch.from_numpy(channels),
            "path_loss_target": torch.from_numpy(path_loss).unsqueeze(0),
            "delay_spread_target": torch.from_numpy(delay_spread).unsqueeze(0),
            "angular_spread_target": torch.from_numpy(angular_spread).unsqueeze(0),
            "path_loss_mask": torch.from_numpy(valid_path_f).unsqueeze(0),
            "delay_spread_mask": torch.from_numpy(valid_delay_f).unsqueeze(0),
            "angular_spread_mask": torch.from_numpy(valid_angular_f).unsqueeze(0),
            "path_loss_prior": torch.from_numpy(path_prior).unsqueeze(0),
            "path_loss_los_prior": torch.from_numpy(path_prior_los).unsqueeze(0),
            "path_loss_nlos_prior": torch.from_numpy(path_prior_nlos).unsqueeze(0),
            "delay_spread_prior": torch.from_numpy(delay_prior).unsqueeze(0),
            "angular_spread_prior": torch.from_numpy(angular_prior).unsqueeze(0),
            "ground_mask": torch.from_numpy(ground_f).unsqueeze(0),
            "los_mask": torch.from_numpy(los).unsqueeze(0),
            "nlos_mask": torch.from_numpy(nlos).unsqueeze(0),
            "antenna_height_m": torch.tensor(uav_height, dtype=torch.float32),
            "topology_class_6": priors.topology_class_6,
            "topology_class_3": priors.topology_class_3,
            "antenna_bin": priors.antenna_bin,
        }

    def _load_or_compute_priors(
        self,
        city: str,
        sample: str,
        topology: np.ndarray,
        los_mask: np.ndarray,
        uav_height: float,
    ):
        path = self.cfg.precomputed_priors_hdf5_path
        if path and path.exists():
            with h5py.File(str(path), "r") as handle:
                if city in handle and sample in handle[city]:
                    grp = handle[city][sample]
                    return _PriorRecord(
                        path_loss_prior=np.asarray(grp["path_loss_prior"][...], dtype=np.float32),
                        path_loss_los_prior=np.asarray(grp["path_loss_los_prior"][...], dtype=np.float32),
                        path_loss_nlos_prior=np.asarray(grp["path_loss_nlos_prior"][...], dtype=np.float32),
                        delay_spread_prior=np.asarray(grp["delay_spread_prior"][...], dtype=np.float32),
                        angular_spread_prior=np.asarray(grp["angular_spread_prior"][...], dtype=np.float32),
                        topology_class_6=str(grp.attrs["topology_class_6"]),
                        topology_class_3=str(grp.attrs["topology_class_3"]),
                        antenna_bin=str(grp.attrs["antenna_bin"]),
                    )
        return self._get_prior_computer().compute(topology, los_mask, uav_height)


@dataclass
class _PriorRecord:
    path_loss_prior: np.ndarray
    path_loss_los_prior: np.ndarray
    path_loss_nlos_prior: np.ndarray
    delay_spread_prior: np.ndarray
    angular_spread_prior: np.ndarray
    topology_class_6: str
    topology_class_3: str
    antenna_bin: str


def build_joint_datasets(cfg: Try80DataConfig) -> Tuple[Try80JointDataset, Try80JointDataset, Try80JointDataset]:
    refs = list_hdf5_samples(cfg.hdf5_path)
    train_refs, val_refs, test_refs = split_city_holdout(
        refs,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        split_seed=cfg.split_seed,
    )
    train_filtered = _filter_small_samples(cfg, train_refs)
    return (
        Try80JointDataset(cfg, train_filtered, augment=True),
        Try80JointDataset(cfg, val_refs, augment=False),
        Try80JointDataset(cfg, test_refs, augment=False),
    )


def _filter_small_samples(cfg: Try80DataConfig, refs: Sequence[SampleRef]) -> List[SampleRef]:
    filtered: List[SampleRef] = []
    with h5py.File(str(cfg.hdf5_path), "r") as handle:
        for city, sample in refs:
            grp = handle[city][sample]
            topology = np.asarray(grp["topology_map"][...], dtype=np.float32)
            ground = topology == 0.0
            path_loss = np.asarray(grp["path_loss"][...], dtype=np.float32)
            delay_spread = read_field(grp, "delay_spread")
            angular_spread = read_field(grp, "angular_spread")
            valid_path = _valid_path_loss_mask(
                grp,
                path_loss,
                ground,
                cfg.path_loss_no_data_mask_column,
                cfg.derive_no_data_from_non_ground,
            )
            valid_delay = _valid_nonnegative_mask(delay_spread, ground)
            valid_angular = _valid_nonnegative_mask(angular_spread, ground)
            valid_any = valid_path | valid_delay | valid_angular
            if int(valid_any.sum()) >= 64:
                filtered.append((city, sample))
    return filtered
