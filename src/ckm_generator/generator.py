"""High-level CKM Generator runtime."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import yaml

from .diagnostics import RuntimeDependencyError, build_torch_device, ensure_torch_runtime
from .io_utils import SampleInput, format_height, slugify
from .los import MaskComparison, compare_los_masks, compute_los_mask, compute_nlos_mask
from .paths import DEFAULT_CONFIG_PATH, DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_DIR, PROJECT_ROOT
from .plotting import TASK_LABELS, save_joint_prediction_png, save_map_png, save_mask_png
from .try80.data_utils import HeightEmbedding, LOG1P_ANGULAR_NORM, LOG1P_DELAY_NORM
from .try80.metrics_try80 import inverse_transform, transform_target
from .try80.model_try80 import Try80Model, Try80ModelConfig
from .try80.priors_try80 import Try80PriorComputer


TASKS = ("path_loss", "delay_spread", "angular_spread")


@dataclass
class PredictionResult:
    sample_id: str
    antenna_height_m: float
    source: str
    topology: np.ndarray
    ground_mask: np.ndarray
    los_mask: np.ndarray
    nlos_mask: np.ndarray
    generated_los_mask: np.ndarray
    generated_nlos_mask: np.ndarray
    reference_los_mask: Optional[np.ndarray]
    priors: Dict[str, np.ndarray]
    predictions: Dict[str, np.ndarray] = field(default_factory=dict)
    mask_source: str = "generated"
    requested_mask_source: str = "auto"
    topology_class_6: str = ""
    topology_class_3: str = ""
    antenna_bin: str = ""
    mask_comparison: Optional[MaskComparison] = None
    raycast_comparison: Optional[MaskComparison] = None
    metadata: Dict[str, object] = field(default_factory=dict)


class CKMGenerator:
    def __init__(
        self,
        *,
        config_path: Path = DEFAULT_CONFIG_PATH,
        checkpoint_path: Optional[Path] = None,
        device: str = "auto",
        load_model: bool = True,
    ) -> None:
        self.project_root = PROJECT_ROOT
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) if self.config_path.exists() else {}
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else self._resolve(self.config.get("model", {}).get("checkpoint", DEFAULT_MODEL_PATH))
        self.device_name = device
        self.device = None
        self.model = None
        self.runtime_report = None
        self.height_embed = HeightEmbedding()
        prior_cfg = self.config.get("priors", {})
        self.prior_computer = Try80PriorComputer(
            try78_los_calibration_json=self._resolve(prior_cfg.get("try78_los_calibration_json", "calibrations/try78_los_two_ray_calibration.json")),
            try78_nlos_calibration_json=self._resolve(prior_cfg.get("try78_nlos_calibration_json", "calibrations/try78_nlos_regime_calibration.json")),
            try79_calibration_json=self._resolve(prior_cfg.get("try79_calibration_json", "calibrations/try79_calibration.json")),
        )
        if load_model:
            self.load_model()

    def load_model(self) -> None:
        if not self.checkpoint_path.exists():
            raise RuntimeDependencyError(f"Model checkpoint not found: {self.checkpoint_path}")
        report = ensure_torch_runtime(self.device_name)
        self.runtime_report = report
        import torch

        self.device = build_torch_device(report.selected_device or "cpu")
        try:
            state = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeDependencyError(f"Could not load checkpoint {self.checkpoint_path}: {type(exc).__name__}: {exc}") from exc
        model_cfg_raw = state.get("model_cfg") if isinstance(state, dict) else None
        if not model_cfg_raw:
            model_cfg_raw = self.config.get("model", {}).get("model_cfg", {})
        try:
            model_cfg = Try80ModelConfig(**model_cfg_raw)
            model = Try80Model(model_cfg)
        except Exception as exc:
            raise RuntimeDependencyError(f"Could not build Try 80 model from config: {type(exc).__name__}: {exc}") from exc
        state_dict = state.get("model", state) if isinstance(state, dict) else state
        try:
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            if report.selected_device == "cuda":
                try:
                    import torch

                    torch.backends.cudnn.benchmark = True
                    model.to(memory_format=torch.channels_last)
                except Exception:
                    pass
        except Exception as exc:
            raise RuntimeDependencyError(f"Could not move/load Try 80 model on {report.selected_device}: {type(exc).__name__}: {exc}") from exc
        model.eval()
        self.model = model

    def predict_results(
        self,
        results: Sequence[PredictionResult],
        *,
        batch_size: int = 1,
        mixed_precision: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        if not results:
            return
        if self.model is None:
            self.load_model()
        total = len(results)
        batch_size = max(1, int(batch_size))
        done = 0
        while done < total:
            current_batch_size = min(batch_size, total - done)
            chunk = results[done : done + current_batch_size]
            try:
                predictions = self._predict_batch(chunk, mixed_precision=mixed_precision)
            except RuntimeDependencyError as exc:
                if current_batch_size <= 1 or "out of memory" not in str(exc).lower():
                    raise
                self._empty_cuda_cache()
                batch_size = max(1, current_batch_size // 2)
                continue
            for result, prediction in zip(chunk, predictions):
                result.predictions = prediction
            done += current_batch_size
            if progress_callback is not None:
                progress_callback(done, total)

    def generate(
        self,
        sample: SampleInput,
        *,
        height_m: Optional[float] = None,
        mask_source: str = "auto",
        run_model: bool = True,
        los_sample_step_px: Optional[float] = None,
        los_clearance_m: Optional[float] = None,
        los_building_dilation_px: Optional[int] = None,
    ) -> PredictionResult:
        h_m = float(height_m if height_m is not None else sample.height_m)
        self._validate_height(h_m)
        topology = self._prepare_topology(sample.topology)
        ground = (topology <= 1.0e-6).astype(np.float32)

        los_cfg = self.config.get("los", {})
        reference_los = self._prepare_mask(sample.reference_los_mask) if sample.reference_los_mask is not None else None

        raycast_los: Optional[np.ndarray] = None

        def get_raycast_los() -> np.ndarray:
            nonlocal raycast_los
            if raycast_los is None:
                raycast_los = compute_los_mask(
                    topology,
                    h_m,
                    rx_height_m=float(los_cfg.get("rx_height_m", 1.5)),
                    sample_step_px=float(los_sample_step_px if los_sample_step_px is not None else los_cfg.get("sample_step_px", 1.0)),
                    clearance_m=float(los_clearance_m if los_clearance_m is not None else los_cfg.get("clearance_m", 0.0)),
                    building_dilation_px=int(los_building_dilation_px if los_building_dilation_px is not None else los_cfg.get("building_dilation_px", 0)),
                    chunk_size=int(los_cfg.get("chunk_size", 4096)),
                )
            return raycast_los

        resolved_mask_source = mask_source
        if mask_source == "auto":
            if reference_los is not None:
                los = reference_los * ground
                generated_los = los.copy()
                resolved_mask_source = "provided"
            else:
                los = get_raycast_los()
                generated_los = los
                resolved_mask_source = "generated"
        elif mask_source == "provided":
            if reference_los is None:
                raise ValueError("mask_source='provided' was requested, but no LoS/NLoS mask was provided")
            los = reference_los * ground
            generated_los = los.copy()
            resolved_mask_source = "provided"
        elif mask_source == "generated":
            los = get_raycast_los()
            generated_los = los
            resolved_mask_source = "generated"
        else:
            raise ValueError("mask_source must be 'auto', 'generated', or 'provided'")
        nlos = compute_nlos_mask(topology, los)
        generated_nlos = compute_nlos_mask(topology, generated_los)

        priors_obj = self.prior_computer.compute(topology, los, h_m)
        priors = {
            "path_loss": priors_obj.path_loss_prior,
            "path_loss_los": priors_obj.path_loss_los_prior,
            "path_loss_nlos": priors_obj.path_loss_nlos_prior,
            "delay_spread": priors_obj.delay_spread_prior,
            "angular_spread": priors_obj.angular_spread_prior,
        }

        predictions: Dict[str, np.ndarray] = {}
        if run_model:
            if self.model is None:
                self.load_model()
            predictions = self._predict(topology, ground, los, nlos, priors, h_m)

        comparison = compare_los_masks(los, reference_los, topology) if reference_los is not None else None
        raycast_comparison = None
        if reference_los is not None and raycast_los is not None:
            raycast_comparison = compare_los_masks(raycast_los, reference_los, topology)
        if comparison is not None and mask_source in {"auto", "provided"} and comparison.mismatches != 0:
            raise RuntimeError(
                f"Provided LoS mismatch detected for {sample.sample_id}: "
                f"{comparison.mismatches} pixels ({comparison.mismatch_fraction:.8%})."
            )
        return PredictionResult(
            sample_id=sample.sample_id,
            antenna_height_m=h_m,
            source=sample.source,
            topology=topology,
            ground_mask=ground,
            los_mask=los,
            nlos_mask=nlos,
            generated_los_mask=generated_los,
            generated_nlos_mask=generated_nlos,
            reference_los_mask=reference_los,
            priors=priors,
            predictions=predictions,
            mask_source=resolved_mask_source,
            requested_mask_source=mask_source,
            topology_class_6=priors_obj.topology_class_6,
            topology_class_3=priors_obj.topology_class_3,
            antenna_bin=priors_obj.antenna_bin,
            mask_comparison=comparison,
            raycast_comparison=raycast_comparison,
            metadata=dict(sample.metadata),
        )

    def save_result(
        self,
        result: PredictionResult,
        out_dir: Path = DEFAULT_OUTPUT_DIR,
        *,
        save_arrays: bool = True,
        save_masks: bool = True,
        save_visual_maps: bool = True,
    ) -> Dict[str, str]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{slugify(result.sample_id)}_h{format_height(result.antenna_height_m)}m"
        written: Dict[str, str] = {}

        mask_dir = out_dir / "masks"
        prior_dir = out_dir / "priors"
        pred_dir = out_dir / "predictions"
        array_dir = out_dir / "arrays"
        meta_dir = out_dir / "metadata"

        if save_masks:
            paths = {
                "los_mask": mask_dir / f"{prefix}_los_mask.png",
                "nlos_mask": mask_dir / f"{prefix}_nlos_mask.png",
                "generated_los_mask": mask_dir / f"{prefix}_generated_los_mask.png",
                "generated_nlos_mask": mask_dir / f"{prefix}_generated_nlos_mask.png",
            }
            for key, path in paths.items():
                save_mask_png(getattr(result, key), path)
                written[key] = str(path)

        if save_visual_maps:
            for key, arr in result.priors.items():
                title = f"{result.sample_id} | h={result.antenna_height_m:.2f} m | prior {key}"
                unit = "dB" if "path_loss" in key else ("ns" if "delay" in key else "deg")
                path = prior_dir / f"{prefix}_prior_{key}.png"
                save_map_png(arr, path, mask=result.ground_mask, title=title, unit=unit)
                written[f"prior_{key}"] = str(path)

            for task, arr in result.predictions.items():
                label, unit, cmap = TASK_LABELS[task]
                path = pred_dir / f"{prefix}_pred_{task}.png"
                save_map_png(arr, path, mask=result.ground_mask, title=f"{result.sample_id} | h={result.antenna_height_m:.2f} m | {label}", cmap=cmap, unit=unit)
                written[f"pred_{task}"] = str(path)

            if result.predictions:
                joint_path = pred_dir / f"{prefix}_pred_joint.png"
                save_joint_prediction_png(
                    result.predictions,
                    joint_path,
                    ground_mask=result.ground_mask,
                    title=f"{result.sample_id} | antenna height {result.antenna_height_m:.2f} m",
                )
                written["pred_joint"] = str(joint_path)

        if save_arrays:
            array_dir.mkdir(parents=True, exist_ok=True)
            arrays = {
                "topology": result.topology,
                "ground_mask": result.ground_mask,
                "los_mask": result.los_mask,
                "nlos_mask": result.nlos_mask,
                "generated_los_mask": result.generated_los_mask,
                "generated_nlos_mask": result.generated_nlos_mask,
                **{f"prior_{k}": v for k, v in result.priors.items()},
                **{f"pred_{k}": v for k, v in result.predictions.items()},
            }
            if result.reference_los_mask is not None:
                arrays["reference_los_mask"] = result.reference_los_mask
            npz_path = array_dir / f"{prefix}.npz"
            np.savez_compressed(npz_path, **arrays)
            written["arrays"] = str(npz_path)

        metadata = self._metadata(result, written)
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / f"{prefix}.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        written["metadata"] = str(meta_path)
        return written

    def _predict(
        self,
        topology: np.ndarray,
        ground: np.ndarray,
        los: np.ndarray,
        nlos: np.ndarray,
        priors: Dict[str, np.ndarray],
        height_m: float,
    ) -> Dict[str, np.ndarray]:
        import torch

        assert self.model is not None
        device = self.device
        channels = self._build_channels(topology, ground, los, nlos, priors)
        inputs = torch.from_numpy(channels).unsqueeze(0).to(device)
        if self._uses_cuda():
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        height = torch.tensor([height_m], dtype=torch.float32, device=device)
        priors_native = {
            "path_loss": torch.from_numpy(priors["path_loss"]).unsqueeze(0).unsqueeze(0).to(device),
            "delay_spread": torch.from_numpy(priors["delay_spread"]).unsqueeze(0).unsqueeze(0).to(device),
            "angular_spread": torch.from_numpy(priors["angular_spread"]).unsqueeze(0).unsqueeze(0).to(device),
        }
        if self._uses_cuda():
            priors_native = {key: value.contiguous(memory_format=torch.channels_last) for key, value in priors_native.items()}
        priors_trans = {task: transform_target(task, value) for task, value in priors_native.items()}
        with torch.inference_mode():
            try:
                outputs = self.model(inputs, self.height_embed(height), priors_trans)
                preds_native = {task: inverse_transform(task, outputs[task]["pred_trans"]) for task in TASKS}
            except RuntimeError as exc:
                msg = str(exc)
                low = msg.lower()
                if "out of memory" in low or "e_outofmemory" in low or "memory" in low:
                    raise RuntimeDependencyError(
                        f"Model inference ran out of memory on {self.device}. "
                        "The generator uses batch 1; try --device cpu, close GPU-heavy apps, "
                        "or use a CUDA build/GPU with more available VRAM. "
                        f"Original error: {type(exc).__name__}: {exc}"
                    ) from exc
                raise RuntimeDependencyError(f"Model inference failed on {self.device}: {type(exc).__name__}: {exc}") from exc
        return {task: preds_native[task][0, 0].detach().cpu().numpy().astype(np.float32) for task in TASKS}

    def _predict_batch(self, results: Sequence[PredictionResult], *, mixed_precision: bool = False) -> list[Dict[str, np.ndarray]]:
        import torch

        assert self.model is not None
        device = self.device
        channels = np.stack(
            [
                self._build_channels(result.topology, result.ground_mask, result.los_mask, result.nlos_mask, result.priors)
                for result in results
            ],
            axis=0,
        )
        inputs = torch.from_numpy(channels).to(device)
        if self._uses_cuda():
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        height = torch.tensor([result.antenna_height_m for result in results], dtype=torch.float32, device=device)
        priors_native = {
            task: torch.from_numpy(np.stack([result.priors[task] for result in results], axis=0)[:, None].astype(np.float32)).to(device)
            for task in TASKS
        }
        if self._uses_cuda():
            priors_native = {key: value.contiguous(memory_format=torch.channels_last) for key, value in priors_native.items()}
        priors_trans = {task: transform_target(task, value) for task, value in priors_native.items()}

        try:
            with torch.inference_mode():
                if mixed_precision and self._uses_cuda():
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(inputs, self.height_embed(height), priors_trans)
                else:
                    outputs = self.model(inputs, self.height_embed(height), priors_trans)
                preds_native = {task: inverse_transform(task, outputs[task]["pred_trans"]) for task in TASKS}
        except RuntimeError as exc:
            msg = str(exc)
            low = msg.lower()
            if "out of memory" in low or "e_outofmemory" in low or "memory" in low:
                self._empty_cuda_cache()
                raise RuntimeDependencyError(
                    f"Model inference ran out of memory on {self.device} with batch size {len(results)}. "
                    "Lower the model batch size, close GPU-heavy apps, or use --device cpu if needed. "
                    f"Original error: {type(exc).__name__}: {exc}"
                ) from exc
            raise RuntimeDependencyError(f"Model inference failed on {self.device}: {type(exc).__name__}: {exc}") from exc

        batch_arrays = {
            task: preds_native[task][:, 0].detach().cpu().numpy().astype(np.float32)
            for task in TASKS
        }
        return [
            {task: batch_arrays[task][idx] for task in TASKS}
            for idx in range(len(results))
        ]

    def _build_channels(
        self,
        topology: np.ndarray,
        ground: np.ndarray,
        los: np.ndarray,
        nlos: np.ndarray,
        priors: Dict[str, np.ndarray],
    ) -> np.ndarray:
        topology_norm_m = float(self.config.get("io", {}).get("topology_norm_m", 90.0))
        topology_input = topology * ground / max(topology_norm_m, 1.0e-3)
        channels = np.stack(
            [
                topology_input,
                los * ground,
                nlos * ground,
                ground,
                priors["path_loss"] / 180.0,
                priors["path_loss_los"] / 180.0,
                priors["path_loss_nlos"] / 180.0,
                np.log1p(np.clip(priors["delay_spread"], 0.0, None)) / LOG1P_DELAY_NORM,
                np.log1p(np.clip(priors["angular_spread"], 0.0, None)) / LOG1P_ANGULAR_NORM,
            ],
            axis=0,
        )
        return np.nan_to_num(channels.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _uses_cuda(self) -> bool:
        return str(getattr(self.runtime_report, "selected_device", "")).lower() == "cuda"

    def _empty_cuda_cache(self) -> None:
        if not self._uses_cuda():
            return
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def _prepare_topology(self, topology: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(np.asarray(topology, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape != (513, 513):
            from .io_utils import fit_array_to_model_grid

            arr = fit_array_to_model_grid(arr, image_size=513, is_mask=False, source="sample topology")
        return np.clip(arr, 0.0, None).astype(np.float32)

    def _prepare_mask(self, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        from .io_utils import fit_array_to_model_grid, normalize_mask

        arr = fit_array_to_model_grid(mask, image_size=513, is_mask=True, source="sample mask")
        return normalize_mask(arr)

    def _metadata(self, result: PredictionResult, written: Dict[str, str]) -> Dict[str, object]:
        return {
            "sample_id": result.sample_id,
            "antenna_height_m": result.antenna_height_m,
            "source": result.source,
            "checkpoint": str(self.checkpoint_path),
            "mask_source": result.mask_source,
            "requested_mask_source": result.requested_mask_source,
            "topology_class_6": result.topology_class_6,
            "topology_class_3": result.topology_class_3,
            "antenna_bin": result.antenna_bin,
            "mask_comparison": asdict(result.mask_comparison) if result.mask_comparison else None,
            "raycast_comparison": asdict(result.raycast_comparison) if result.raycast_comparison else None,
            "metadata": result.metadata,
            "files": written,
        }

    def _resolve(self, value: object) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else (self.project_root / path).resolve()

    @staticmethod
    def _validate_height(height_m: float) -> None:
        if not (10.0 <= float(height_m) <= 478.0):
            raise ValueError(f"antenna height must be between 10 m and 478 m, got {height_m}")

    @staticmethod
    def _resolve_device(name: str):
        report = ensure_torch_runtime(name)
        return build_torch_device(report.selected_device or "cpu")

