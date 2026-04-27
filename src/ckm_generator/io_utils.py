"""Input loading helpers for CKM Generator."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence

import h5py
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ARRAY_SUFFIXES = {".npy", ".npz", ".csv", ".txt", ".json"}
HDF5_SUFFIXES = {".h5", ".hdf5", ".hdf"}

TOPOLOGY_KEYS = (
    "topology_map",
    "topology",
    "building_height",
    "building_heights",
    "height_map",
    "terrain",
)
LOS_KEYS = ("los_mask", "LoS_mask", "los", "LoS")
NLOS_KEYS = ("nlos_mask", "nLoS_mask", "NLoS_mask", "nlos", "NLoS")
HEIGHT_KEYS = ("uav_height", "antenna_height", "antenna_height_m", "height", "height_m", "h_tx")
PAD_BUILDING_HEIGHT_M = 1.0


@dataclass
class SampleInput:
    sample_id: str
    topology: np.ndarray
    height_m: Optional[float] = None
    reference_los_mask: Optional[np.ndarray] = None
    reference_nlos_mask: Optional[np.ndarray] = None
    source: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)


def slugify(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    return safe.strip("_") or "sample"


def format_height(height_m: float) -> str:
    text = f"{float(height_m):.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def load_single_input(
    input_path: Path,
    *,
    height_m: Optional[float] = None,
    los_mask_path: Optional[Path] = None,
    nlos_mask_path: Optional[Path] = None,
    topology_max_m: float = 90.0,
    image_values_are_metres: bool = False,
    meters_per_pixel: float = 1.0,
    image_size: int = 513,
) -> SampleInput:
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()
    if suffix in HDF5_SUFFIXES:
        samples = list(load_hdf5_samples(input_path, image_size=image_size, meters_per_pixel=meters_per_pixel))
        if not samples:
            raise ValueError(f"No topology sample found inside {input_path}")
        sample = samples[0]
        if height_m is not None:
            sample.height_m = float(height_m)
        return sample
    if suffix == ".npz":
        return _read_npz_sample(input_path, height_m=height_m, image_size=image_size, meters_per_pixel=meters_per_pixel)

    topology = load_2d_array(
        input_path,
        topology_max_m=topology_max_m,
        image_values_are_metres=image_values_are_metres,
    )
    topology = fit_array_to_model_grid(
        topology,
        image_size=image_size,
        is_mask=False,
        source=f"{input_path} topology",
        meters_per_pixel=meters_per_pixel,
    )

    auto_los_mask_path = None
    auto_nlos_mask_path = None
    if los_mask_path is None and nlos_mask_path is None:
        auto_los_mask_path, auto_nlos_mask_path = _find_sidecar_mask_paths(input_path)
        los_mask_path = auto_los_mask_path
        nlos_mask_path = auto_nlos_mask_path

    los = load_mask(los_mask_path, image_size=image_size, meters_per_pixel=meters_per_pixel) if los_mask_path else None
    nlos = load_mask(nlos_mask_path, image_size=image_size, meters_per_pixel=meters_per_pixel) if nlos_mask_path else None
    if los is None and nlos is not None:
        los = 1.0 - nlos
    if nlos is None and los is not None:
        nlos = 1.0 - los

    return SampleInput(
        sample_id=slugify(input_path.stem),
        topology=topology,
        height_m=height_m,
        reference_los_mask=los,
        reference_nlos_mask=nlos,
        source=str(input_path),
        metadata={
            "input_type": suffix.lstrip(".") or "file",
            "auto_los_mask_path": str(auto_los_mask_path) if auto_los_mask_path else "",
            "auto_nlos_mask_path": str(auto_nlos_mask_path) if auto_nlos_mask_path else "",
        },
    )


def load_2d_array(
    path: Path,
    *,
    topology_max_m: float = 90.0,
    image_values_are_metres: bool = False,
) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return load_topology_image(path, topology_max_m=topology_max_m, values_are_metres=image_values_are_metres)
    if suffix == ".npy":
        return _as_2d(np.load(path))
    if suffix == ".npz":
        payload = np.load(path)
        key = _first_present(payload.files, TOPOLOGY_KEYS) or payload.files[0]
        return _as_2d(payload[key])
    if suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        return _as_2d(np.loadtxt(path, delimiter=delimiter))
    if suffix == ".json":
        return _as_2d(np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float32))
    raise ValueError(f"Unsupported input type: {path.suffix}")


def load_topology_image(path: Path, *, topology_max_m: float = 90.0, values_are_metres: bool = False) -> np.ndarray:
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    if values_are_metres:
        return arr
    max_value = 65535.0 if arr.max(initial=0.0) > 255.0 else 255.0
    return (arr / max(max_value, 1.0) * float(topology_max_m)).astype(np.float32)


def load_mask(path: Path, *, image_size: int = 513, meters_per_pixel: float = 1.0) -> np.ndarray:
    arr = load_2d_array(Path(path), topology_max_m=1.0, image_values_are_metres=True)
    arr = fit_array_to_model_grid(
        arr,
        image_size=image_size,
        is_mask=True,
        source=f"{path} mask",
        meters_per_pixel=meters_per_pixel,
    )
    return normalize_mask(arr)


def _read_npz_sample(input_path: Path, *, height_m: Optional[float], image_size: int, meters_per_pixel: float) -> SampleInput:
    payload = np.load(input_path)
    topology_key = _first_present(payload.files, TOPOLOGY_KEYS) or payload.files[0]
    topology = fit_array_to_model_grid(
        _as_2d(payload[topology_key]),
        image_size=image_size,
        is_mask=False,
        source=f"{input_path}:{topology_key}",
        meters_per_pixel=meters_per_pixel,
    )

    los_key = _first_present(payload.files, LOS_KEYS)
    nlos_key = _first_present(payload.files, NLOS_KEYS)
    los = (
        normalize_mask(
            fit_array_to_model_grid(
                np.asarray(payload[los_key]),
                image_size=image_size,
                is_mask=True,
                source=f"{input_path}:{los_key}",
                meters_per_pixel=meters_per_pixel,
            )
        )
        if los_key
        else None
    )
    nlos = (
        normalize_mask(
            fit_array_to_model_grid(
                np.asarray(payload[nlos_key]),
                image_size=image_size,
                is_mask=True,
                source=f"{input_path}:{nlos_key}",
                meters_per_pixel=meters_per_pixel,
            )
        )
        if nlos_key
        else None
    )
    if los is None and nlos is not None:
        los = 1.0 - nlos
    if nlos is None and los is not None:
        nlos = 1.0 - los

    height_key = _first_present(payload.files, HEIGHT_KEYS)
    resolved_height = float(np.asarray(payload[height_key]).reshape(-1)[0]) if height_key else None
    if height_m is not None:
        resolved_height = float(height_m)

    return SampleInput(
        sample_id=slugify(input_path.stem),
        topology=topology,
        height_m=resolved_height,
        reference_los_mask=los,
        reference_nlos_mask=nlos,
        source=str(input_path),
        metadata={
            "input_type": "npz",
            "npz_keys": list(payload.files),
            "topology_key": topology_key,
            "height_key": height_key or "",
            "los_key": los_key or "",
            "nlos_key": nlos_key or "",
            "meters_per_pixel": float(meters_per_pixel),
        },
    )


def _find_sidecar_mask_paths(input_path: Path) -> tuple[Optional[Path], Optional[Path]]:
    stem = input_path.stem
    base = stem[: -len("_topology")] if stem.endswith("_topology") else stem
    candidate_dirs = [input_path.parent, input_path.parent.parent / "masks_png"]
    los_names = (f"{base}_los_mask.png", f"{base}_los.png", f"{base}_LoS_mask.png")
    nlos_names = (f"{base}_nlos_mask.png", f"{base}_nlos.png", f"{base}_NLoS_mask.png")

    los_path = _first_existing(candidate_dirs, los_names)
    nlos_path = _first_existing(candidate_dirs, nlos_names)
    return los_path, nlos_path


def _first_existing(dirs: Sequence[Path], names: Sequence[str]) -> Optional[Path]:
    for directory in dirs:
        for name in names:
            path = directory / name
            if path.exists():
                return path
    return None


def fit_array_to_model_grid(
    arr: np.ndarray,
    *,
    image_size: int,
    is_mask: bool,
    source: str,
    meters_per_pixel: float = 1.0,
) -> np.ndarray:
    arr = _as_2d(arr)
    arr = _resample_to_one_meter_pixels(arr, meters_per_pixel=meters_per_pixel, is_mask=is_mask, source=source)
    expected = (image_size, image_size)
    if arr.shape == expected:
        return arr.astype(np.float32, copy=False)
    fill_value = 0.0 if is_mask else PAD_BUILDING_HEIGHT_M
    out = np.full(expected, fill_value, dtype=np.float32)

    h, w = arr.shape
    crop_top = max((h - image_size) // 2, 0)
    crop_left = max((w - image_size) // 2, 0)
    crop = arr[crop_top : crop_top + min(h, image_size), crop_left : crop_left + min(w, image_size)]

    out_top = max((image_size - crop.shape[0]) // 2, 0)
    out_left = max((image_size - crop.shape[1]) // 2, 0)
    out[out_top : out_top + crop.shape[0], out_left : out_left + crop.shape[1]] = crop.astype(np.float32, copy=False)
    return out


def _resample_to_one_meter_pixels(
    arr: np.ndarray,
    *,
    meters_per_pixel: float,
    is_mask: bool,
    source: str,
) -> np.ndarray:
    mpp = float(meters_per_pixel)
    if not np.isfinite(mpp) or mpp <= 0.0:
        raise ValueError(f"{source} has invalid pixel spacing {meters_per_pixel!r}; it must be positive metres per pixel.")
    if abs(mpp - 1.0) < 1.0e-6:
        return arr.astype(np.float32, copy=False)

    h, w = arr.shape
    out_h = max(int(round(h * mpp)), 1)
    out_w = max(int(round(w * mpp)), 1)
    if (out_h, out_w) == arr.shape:
        return arr.astype(np.float32, copy=False)

    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    if is_mask:
        img = Image.fromarray((normalize_mask(arr) * 255).astype(np.uint8), mode="L")
        out = np.asarray(img.resize((out_w, out_h), resample), dtype=np.float32) / 255.0
        return normalize_mask(out)
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    return np.asarray(img.resize((out_w, out_h), resample), dtype=np.float32)


def normalize_mask(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    threshold = 127.0 if float(arr.max(initial=0.0)) > 1.5 else 0.5
    return (arr > threshold).astype(np.float32)


def resize_array(arr: np.ndarray, *, image_size: int = 513, is_mask: bool = False) -> np.ndarray:
    arr = _as_2d(arr)
    if arr.shape == (image_size, image_size):
        return arr.astype(np.float32, copy=False)
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    if is_mask:
        img = Image.fromarray((normalize_mask(arr) * 255).astype(np.uint8), mode="L")
        out = np.asarray(img.resize((image_size, image_size), resample), dtype=np.float32) / 255.0
        return normalize_mask(out)
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    return np.asarray(img.resize((image_size, image_size), resample), dtype=np.float32)


def load_hdf5_samples(
    hdf5_path: Path,
    *,
    city: Optional[str] = None,
    sample: Optional[str] = None,
    image_size: int = 513,
    meters_per_pixel: float = 1.0,
) -> Iterator[SampleInput]:
    hdf5_path = Path(hdf5_path)
    with h5py.File(str(hdf5_path), "r") as handle:
        if city and sample and city in handle and isinstance(handle[city], h5py.Group) and sample in handle[city]:
            grp = handle[city][sample]
            if isinstance(grp, h5py.Group) and _find_dataset(grp, TOPOLOGY_KEYS) is not None:
                yield _read_hdf5_group(hdf5_path, f"{city}/{sample}", grp, image_size=image_size, meters_per_pixel=meters_per_pixel)
                return

        if city and city in handle and isinstance(handle[city], h5py.Group):
            city_grp = handle[city]
            for sample_name in sorted(city_grp.keys()):
                if sample and sample_name != sample:
                    continue
                obj = city_grp[sample_name]
                if isinstance(obj, h5py.Group) and _find_dataset(obj, TOPOLOGY_KEYS) is not None:
                    yield _read_hdf5_group(hdf5_path, f"{city}/{sample_name}", obj, image_size=image_size, meters_per_pixel=meters_per_pixel)
            return

        if sample and not city:
            for city_name in sorted(handle.keys()):
                city_grp = handle[city_name]
                if not isinstance(city_grp, h5py.Group) or sample not in city_grp:
                    continue
                obj = city_grp[sample]
                if isinstance(obj, h5py.Group) and _find_dataset(obj, TOPOLOGY_KEYS) is not None:
                    yield _read_hdf5_group(hdf5_path, f"{city_name}/{sample}", obj, image_size=image_size, meters_per_pixel=meters_per_pixel)
            return

        if _looks_like_city_sample_hdf5(handle):
            for city_name in sorted(handle.keys()):
                city_grp = handle[city_name]
                if not isinstance(city_grp, h5py.Group):
                    continue
                for sample_name in sorted(city_grp.keys()):
                    obj = city_grp[sample_name]
                    if isinstance(obj, h5py.Group) and _find_dataset(obj, TOPOLOGY_KEYS) is not None:
                        yield _read_hdf5_group(hdf5_path, f"{city_name}/{sample_name}", obj, image_size=image_size, meters_per_pixel=meters_per_pixel)
            return

        for group_name, grp in _iter_topology_groups(handle):
            parts = [p for p in group_name.split("/") if p]
            group_city = parts[-2] if len(parts) >= 2 else None
            group_sample = parts[-1] if parts else hdf5_path.stem
            if city and group_city != city:
                continue
            if sample and group_sample != sample:
                continue
            yield _read_hdf5_group(hdf5_path, group_name, grp, image_size=image_size, meters_per_pixel=meters_per_pixel)


def _iter_topology_groups(handle: h5py.File) -> Iterator[tuple[str, h5py.Group]]:
    if _find_dataset(handle, TOPOLOGY_KEYS) is not None:
        yield "", handle
        return

    found: list[tuple[str, h5py.Group]] = []

    def visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Group) and _find_dataset(obj, TOPOLOGY_KEYS) is not None:
            found.append((name, obj))

    handle.visititems(visit)
    for item in found:
        yield item


def _looks_like_city_sample_hdf5(handle: h5py.File) -> bool:
    for city_name in handle.keys():
        city_grp = handle[city_name]
        if not isinstance(city_grp, h5py.Group):
            continue
        for sample_name in city_grp.keys():
            obj = city_grp[sample_name]
            if isinstance(obj, h5py.Group) and _find_dataset(obj, TOPOLOGY_KEYS) is not None:
                return True
    return False


def _read_hdf5_group(hdf5_path: Path, group_name: str, grp: h5py.Group, *, image_size: int, meters_per_pixel: float) -> SampleInput:
    topo_ds = _find_dataset(grp, TOPOLOGY_KEYS)
    if topo_ds is None:
        raise ValueError(f"HDF5 group {group_name or '/'} has no topology dataset")
    topology = fit_array_to_model_grid(
        np.asarray(topo_ds[...], dtype=np.float32),
        image_size=image_size,
        is_mask=False,
        source=f"{hdf5_path}:{group_name or '/'} topology",
        meters_per_pixel=meters_per_pixel,
    )

    los_ds = _find_dataset(grp, LOS_KEYS)
    nlos_ds = _find_dataset(grp, NLOS_KEYS)
    los = (
        normalize_mask(
            fit_array_to_model_grid(
                np.asarray(los_ds[...]),
                image_size=image_size,
                is_mask=True,
                source=f"{hdf5_path}:{group_name or '/'} los_mask",
                meters_per_pixel=meters_per_pixel,
            )
        )
        if los_ds is not None
        else None
    )
    nlos = (
        normalize_mask(
            fit_array_to_model_grid(
                np.asarray(nlos_ds[...]),
                image_size=image_size,
                is_mask=True,
                source=f"{hdf5_path}:{group_name or '/'} nlos_mask",
                meters_per_pixel=meters_per_pixel,
            )
        )
        if nlos_ds is not None
        else None
    )
    if los is None and nlos is not None:
        los = 1.0 - nlos
    if nlos is None and los is not None:
        nlos = 1.0 - los

    height_ds = _find_dataset(grp, HEIGHT_KEYS)
    height = float(np.asarray(height_ds[...]).reshape(-1)[0]) if height_ds is not None else None

    parts = [p for p in group_name.split("/") if p]
    sample_id = "_".join(parts) if parts else hdf5_path.stem
    return SampleInput(
        sample_id=slugify(sample_id),
        topology=topology,
        height_m=height,
        reference_los_mask=los,
        reference_nlos_mask=nlos,
        source=f"{hdf5_path}:{group_name or '/'}",
        metadata={"hdf5_group": group_name or "/", "meters_per_pixel": float(meters_per_pixel)},
    )


def _find_dataset(grp: h5py.Group, names: Sequence[str]) -> Optional[h5py.Dataset]:
    for name in names:
        if name in grp and isinstance(grp[name], h5py.Dataset):
            return grp[name]
    lower = {key.lower(): key for key in grp.keys()}
    for name in names:
        key = lower.get(name.lower())
        if key and isinstance(grp[key], h5py.Dataset):
            return grp[key]
    return None


def _first_present(values: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {v.lower(): v for v in values}
    for key in candidates:
        hit = lowered.get(key.lower())
        if hit:
            return hit
    return None


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")
    return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
