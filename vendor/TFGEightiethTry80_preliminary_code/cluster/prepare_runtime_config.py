"""Prepare a runtime config for Try 80 cluster runs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping.")
    return data


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _resolve(root: Path, value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    path = Path(str(value))
    return str(path if path.is_absolute() else (root / path).resolve())


def resolve_runtime_paths(cfg: Dict[str, Any], config_root: Path) -> Dict[str, Any]:
    runtime_cfg = dict(cfg.get("runtime", {}))
    data_cfg = dict(cfg.get("data", {}))
    prior_cfg = dict(cfg.get("prior", {}))

    for key in ("output_dir", "resume_checkpoint"):
        if runtime_cfg.get(key):
            runtime_cfg[key] = _resolve(config_root, runtime_cfg[key])
    for key in ("hdf5_path", "precomputed_priors_hdf5_path"):
        if data_cfg.get(key):
            data_cfg[key] = _resolve(config_root, data_cfg[key])
    for key in ("try78_los_calibration_json", "try78_nlos_calibration_json", "try79_calibration_json"):
        if prior_cfg.get(key):
            prior_cfg[key] = _resolve(config_root, prior_cfg[key])

    cfg["runtime"] = runtime_cfg
    cfg["data"] = data_cfg
    cfg["prior"] = prior_cfg
    return cfg


def apply_overrides(
    cfg: Dict[str, Any],
    hdf5_path: Optional[str],
    precomputed_priors_hdf5_path: Optional[str],
    resume_checkpoint: Optional[str],
) -> Dict[str, Any]:
    data_cfg = dict(cfg.get("data", {}))
    runtime_cfg = dict(cfg.get("runtime", {}))
    if hdf5_path:
        data_cfg["hdf5_path"] = hdf5_path
    if precomputed_priors_hdf5_path is not None:
        data_cfg["precomputed_priors_hdf5_path"] = precomputed_priors_hdf5_path
    if resume_checkpoint:
        runtime_cfg["resume_checkpoint"] = resume_checkpoint
    cfg["data"] = data_cfg
    cfg["runtime"] = runtime_cfg
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--hdf5-path", default="")
    parser.add_argument("--precomputed-priors-hdf5-path", default="")
    parser.add_argument("--resume-checkpoint", default="")
    args = parser.parse_args()

    in_path = Path(args.input_config)
    out_path = Path(args.output_config)
    cfg = load_yaml(in_path)
    cfg = resolve_runtime_paths(cfg, in_path.parent)
    cfg = apply_overrides(
        cfg,
        args.hdf5_path or None,
        args.precomputed_priors_hdf5_path if args.precomputed_priors_hdf5_path != "" else None,
        args.resume_checkpoint or None,
    )
    save_yaml(out_path, cfg)
    print(f"input_config={in_path}")
    print(f"output_config={out_path}")
    print(f"hdf5_path={cfg.get('data', {}).get('hdf5_path')}")
    print(f"precomputed_priors_hdf5_path={cfg.get('data', {}).get('precomputed_priors_hdf5_path')}")
    print(f"output_dir={cfg.get('runtime', {}).get('output_dir')}")
    print(f"resume_checkpoint={cfg.get('runtime', {}).get('resume_checkpoint')}")


if __name__ == "__main__":
    main()
