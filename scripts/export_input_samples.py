"""Export CKM HDF5 samples as standalone interface-test inputs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ckm_generator.io_utils import slugify  # noqa: E402
from ckm_generator.los import compute_nlos_mask  # noqa: E402
from ckm_generator.plotting import save_map_png, save_mask_png  # noqa: E402


DEFAULT_HDF5 = Path("C:/TFG/TFGpractice/Datasets/CKM_Dataset_270326.h5")
TARGET_KEYS = ("path_loss", "delay_spread", "angular_spread")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CKM samples for manual CKM Generator interface testing.")
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--out", type=Path, default=ROOT / "inputs_prueba")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--los-mask", action=argparse.BooleanOptionalAction, default=True, help="Include/export LoS mask.")
    parser.add_argument("--nlos-mask", action=argparse.BooleanOptionalAction, default=True, help="Include/export derived NLoS mask.")
    parser.add_argument("--targets", action=argparse.BooleanOptionalAction, default=True, help="Include GT output datasets: path_loss, delay_spread, angular_spread.")
    parser.add_argument("--matrices", action=argparse.BooleanOptionalAction, default=True, help="Save exact numeric matrices as compressed .npz files.")
    parser.add_argument("--png", action=argparse.BooleanOptionalAction, default=True, help="Save raw topology PNG inputs and mask PNGs.")
    parser.add_argument(
        "--topology-max-m",
        type=float,
        default=None,
        help="Metres encoded by white in raw topology PNG inputs. Defaults to each sample's own maximum, so PNG export does not clip.",
    )
    parser.add_argument("--preview-png", action=argparse.BooleanOptionalAction, default=True, help="Also save topology preview PNGs with title and colorbar.")
    parser.add_argument("--bundle-hdf5", action=argparse.BooleanOptionalAction, default=False, help="Also write a single HDF5 containing all exported samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise SystemExit("--n must be positive.")
    if not args.hdf5.exists():
        raise SystemExit(f"HDF5 not found: {args.hdf5}")

    args.out.mkdir(parents=True, exist_ok=True)
    sample_dir = args.out / "samples"
    png_dir = args.out / "topology_png"
    preview_png_dir = args.out / "topology_preview_png"
    mask_png_dir = args.out / "masks_png"
    matrix_dir = args.out / "matrices"
    sample_dir.mkdir(parents=True, exist_ok=True)
    if args.png:
        png_dir.mkdir(parents=True, exist_ok=True)
        if args.los_mask or args.nlos_mask:
            mask_png_dir.mkdir(parents=True, exist_ok=True)
    if args.preview_png:
        preview_png_dir.mkdir(parents=True, exist_ok=True)
    if args.matrices:
        matrix_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    with h5py.File(str(args.hdf5), "r") as src:
        selected = _select_round_robin(_collect_samples(src), args.n)
        bundle = None
        try:
            if args.bundle_hdf5:
                bundle = h5py.File(str(args.out / "inputs_prueba_bundle.h5"), "w")
            for idx, (city, sample_name) in enumerate(selected, start=1):
                row = _export_one(
                    src,
                    bundle,
                    city=city,
                    sample_name=sample_name,
                    sample_dir=sample_dir,
                    png_dir=png_dir,
                    preview_png_dir=preview_png_dir,
                    mask_png_dir=mask_png_dir,
                    matrix_dir=matrix_dir,
                    include_los=args.los_mask,
                    include_nlos=args.nlos_mask,
                    include_targets=args.targets,
                    save_matrices=args.matrices,
                    save_png=args.png,
                    save_preview_png=args.preview_png,
                    topology_max_m=args.topology_max_m,
                )
                rows.append({"index": idx, **row})
                print(f"{idx:03d}/{len(selected):03d} {row['sample_id']} h={float(row['antenna_height_m']):.2f}m")
        finally:
            if bundle is not None:
                bundle.close()

    _write_manifest(args.out, rows)
    print(json.dumps({"n_samples": len(rows), "out": str(args.out), "cities": len({r["city"] for r in rows})}, indent=2))


def _collect_samples(handle: h5py.File) -> dict[str, list[str]]:
    city_samples: dict[str, list[str]] = {}
    for city in sorted(handle.keys()):
        city_grp = handle[city]
        if not isinstance(city_grp, h5py.Group):
            continue
        samples = []
        for sample_name in sorted(city_grp.keys()):
            grp = city_grp[sample_name]
            if isinstance(grp, h5py.Group) and "topology_map" in grp and "uav_height" in grp:
                samples.append(sample_name)
        if samples:
            city_samples[city] = samples
    return city_samples


def _select_round_robin(city_samples: dict[str, list[str]], n: int) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    cities = sorted(city_samples)
    offset = 0
    while len(selected) < n:
        added = False
        for city in cities:
            samples = city_samples[city]
            if offset >= len(samples):
                continue
            selected.append((city, samples[offset]))
            added = True
            if len(selected) >= n:
                break
        if not added:
            break
        offset += 1
    return selected


def _export_one(
    src: h5py.File,
    bundle: h5py.File | None,
    *,
    city: str,
    sample_name: str,
    sample_dir: Path,
    png_dir: Path,
    preview_png_dir: Path,
    mask_png_dir: Path,
    matrix_dir: Path,
    include_los: bool,
    include_nlos: bool,
    include_targets: bool,
    save_matrices: bool,
    save_png: bool,
    save_preview_png: bool,
    topology_max_m: float | None,
) -> dict[str, object]:
    grp = src[city][sample_name]
    sample_id = slugify(f"{city}_{sample_name}")
    topology = np.asarray(grp["topology_map"][...], dtype=np.float32)
    height = np.asarray(grp["uav_height"][...], dtype=np.float32)
    height_m = float(height.reshape(-1)[0])

    matrices: dict[str, np.ndarray] = {
        "topology_map": topology,
        "uav_height": height.astype(np.float32, copy=False),
    }
    los = None
    if include_los and "los_mask" in grp:
        los = (np.asarray(grp["los_mask"][...]) > 0).astype(np.uint8)
        matrices["los_mask"] = los
    if include_nlos and los is not None:
        matrices["nlos_mask"] = compute_nlos_mask(topology, los).astype(np.uint8)
    if include_targets:
        for key in TARGET_KEYS:
            if key in grp:
                matrices[key] = np.asarray(grp[key][...])

    sample_h5 = sample_dir / f"{sample_id}.h5"
    with h5py.File(str(sample_h5), "w") as out:
        out.attrs["source_city"] = city
        out.attrs["source_sample"] = sample_name
        out.attrs["sample_id"] = sample_id
        for key, arr in matrices.items():
            _write_dataset(out, key, arr)

    bundle_group = ""
    if bundle is not None:
        bgrp = bundle.require_group(city).create_group(sample_name)
        bgrp.attrs["sample_id"] = sample_id
        for key, arr in matrices.items():
            _write_dataset(bgrp, key, arr)
        bundle_group = f"{city}/{sample_name}"

    topology_png = ""
    topology_png_max_m = None
    topology_preview_png = ""
    los_png = ""
    nlos_png = ""
    if save_png:
        topology_png_path = png_dir / f"{sample_id}_topology.png"
        topology_png_max_m = _save_raw_topology_png(topology, topology_png_path, topology_max_m=topology_max_m)
        topology_png = str(topology_png_path)
        if include_los and los is not None:
            los_png_path = mask_png_dir / f"{sample_id}_los_mask.png"
            save_mask_png(los, los_png_path)
            los_png = str(los_png_path)
        if include_nlos and "nlos_mask" in matrices:
            nlos_png_path = mask_png_dir / f"{sample_id}_nlos_mask.png"
            save_mask_png(matrices["nlos_mask"], nlos_png_path)
            nlos_png = str(nlos_png_path)
    if save_preview_png:
        topology_preview_png_path = preview_png_dir / f"{sample_id}_topology_preview.png"
        save_map_png(topology, topology_preview_png_path, title=f"{sample_id} topology", unit="m", robust=True)
        topology_preview_png = str(topology_preview_png_path)

    matrix_npz = ""
    if save_matrices:
        matrix_npz_path = matrix_dir / f"{sample_id}.npz"
        np.savez_compressed(matrix_npz_path, **matrices)
        matrix_npz = str(matrix_npz_path)

    return {
        "sample_id": sample_id,
        "city": city,
        "sample": sample_name,
        "antenna_height_m": height_m,
        "hdf5_path": str(sample_h5),
        "bundle_group": bundle_group,
        "matrix_npz": matrix_npz,
        "topology_png": topology_png,
        "topology_png_max_m": "" if topology_png_max_m is None else float(topology_png_max_m),
        "topology_preview_png": topology_preview_png,
        "los_mask_png": los_png,
        "nlos_mask_png": nlos_png,
        "datasets": ",".join(matrices.keys()),
    }


def _save_raw_topology_png(topology: np.ndarray, path: Path, *, topology_max_m: float | None) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.nan_to_num(np.asarray(topology, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if topology_max_m is None:
        max_m = float(np.max(arr, initial=0.0))
    else:
        max_m = float(topology_max_m)
    max_m = max(max_m, 1.0e-6)
    encoded = np.clip(arr / max_m, 0.0, 1.0)
    Image.fromarray(np.rint(encoded * 65535.0).astype(np.uint16), mode="I;16").save(path)
    return max_m


def _write_dataset(group: h5py.Group | h5py.File, key: str, arr: np.ndarray) -> None:
    data = np.asarray(arr)
    kwargs = {}
    if data.ndim >= 2:
        kwargs = {"compression": "gzip", "shuffle": True}
    group.create_dataset(key, data=data, **kwargs)


def _write_manifest(out_dir: Path, rows: list[dict[str, object]]) -> None:
    csv_path = out_dir / "manifest.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
