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
    image_size: int = 513,
) -> SampleInput:
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()
    if suffix in HDF5_SUFFIXES:
        samples = list(load_hdf5_samples(input_path, image_size=image_size))
        if not samples:
            raise ValueError(f"No topology sample found inside {input_path}")
        sample = samples[0]
        if height_m is not None:
            sample.height_m = float(height_m)
        return sample

    topology = load_2d_array(
        input_path,
        topology_max_m=topology_max_m,
        image_values_are_metres=image_values_are_metres,
    )
    topology = resize_array(topology, image_size=image_size, is_mask=False)

    los = load_mask(los_mask_path, image_size=image_size) if los_mask_path else None
    nlos = load_mask(nlos_mask_path, image_size=image_size) if nlos_mask_path else None
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
        metadata={"input_type": suffix.lstrip(".") or "file"},
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


def load_mask(path: Path, *, image_size: int = 513) -> np.ndarray:
    arr = load_2d_array(Path(path), topology_max_m=1.0, image_values_are_metres=True)
    arr = resize_array(arr, image_size=image_size, is_mask=True)
    return normalize_mask(arr)


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
) -> Iterator[SampleInput]:
    hdf5_path = Path(hdf5_path)
    with h5py.File(str(hdf5_path), "r") as handle:
        if city and sample and city in handle and isinstance(handle[city], h5py.Group) and sample in handle[city]:
            grp = handle[city][sample]
            if isinstance(grp, h5py.Group) and _find_dataset(grp, TOPOLOGY_KEYS) is not None:
                yield _read_hdf5_group(hdf5_path, f"{city}/{sample}", grp, image_size=image_size)
                return

        if city and city in handle and isinstance(handle[city], h5py.Group):
            city_grp = handle[city]
            for sample_name in sorted(city_grp.keys()):
                if sample and sample_name != sample:
                    continue
                obj = city_grp[sample_name]
                if isinstance(obj, h5py.Group) and _find_dataset(obj, TOPOLOGY_KEYS) is not None:
                    yield _read_hdf5_group(hdf5_path, f"{city}/{sample_name}", obj, image_size=image_size)
            return

        if sample and not city:
            for city_name in sorted(handle.keys()):
                city_grp = handle[city_name]
                if not isinstance(city_grp, h5py.Group) or sample not in city_grp:
                    continue
                obj = city_grp[sample]
                if isinstance(obj, h5py.Group) and _find_dataset(obj, TOPOLOGY_KEYS) is not None:
                    yield _read_hdf5_group(hdf5_path, f"{city_name}/{sample}", obj, image_size=image_size)
            return

        if _looks_like_city_sample_hdf5(handle):
            for city_name in sorted(handle.keys()):
                city_grp = handle[city_name]
                if not isinstance(city_grp, h5py.Group):
                    continue
                for sample_name in sorted(city_grp.keys()):
                    obj = city_grp[sample_name]
                    if isinstance(obj, h5py.Group) and _find_dataset(obj, TOPOLOGY_KEYS) is not None:
                        yield _read_hdf5_group(hdf5_path, f"{city_name}/{sample_name}", obj, image_size=image_size)
            return

        for group_name, grp in _iter_topology_groups(handle):
            parts = [p for p in group_name.split("/") if p]
            group_city = parts[-2] if len(parts) >= 2 else None
            group_sample = parts[-1] if parts else hdf5_path.stem
            if city and group_city != city:
                continue
            if sample and group_sample != sample:
                continue
            yield _read_hdf5_group(hdf5_path, group_name, grp, image_size=image_size)


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


def _read_hdf5_group(hdf5_path: Path, group_name: str, grp: h5py.Group, *, image_size: int) -> SampleInput:
    topo_ds = _find_dataset(grp, TOPOLOGY_KEYS)
    if topo_ds is None:
        raise ValueError(f"HDF5 group {group_name or '/'} has no topology dataset")
    topology = resize_array(np.asarray(topo_ds[...], dtype=np.float32), image_size=image_size, is_mask=False)

    los_ds = _find_dataset(grp, LOS_KEYS)
    nlos_ds = _find_dataset(grp, NLOS_KEYS)
    los = normalize_mask(resize_array(np.asarray(los_ds[...]), image_size=image_size, is_mask=True)) if los_ds is not None else None
    nlos = normalize_mask(resize_array(np.asarray(nlos_ds[...]), image_size=image_size, is_mask=True)) if nlos_ds is not None else None
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
        metadata={"hdf5_group": group_name or "/"},
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
