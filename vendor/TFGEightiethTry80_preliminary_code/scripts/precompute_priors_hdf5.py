"""Precompute frozen Try 80 priors into an auxiliary HDF5 file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.priors_try80 import Try80PriorComputer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--try78-los-calibration-json", type=Path, required=True)
    parser.add_argument("--try78-nlos-calibration-json", type=Path, required=True)
    parser.add_argument("--try79-calibration-json", type=Path, required=True)
    args = parser.parse_args()

    computer = Try80PriorComputer(
        try78_los_calibration_json=args.try78_los_calibration_json,
        try78_nlos_calibration_json=args.try78_nlos_calibration_json,
        try79_calibration_json=args.try79_calibration_json,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(args.hdf5), "r") as src, h5py.File(str(args.out), "w") as dst:
        total = sum(len(src[city].keys()) for city in src.keys())
        done = 0
        for city in sorted(src.keys()):
            g_city = dst.require_group(city)
            for sample in sorted(src[city].keys()):
                grp = src[city][sample]
                topology = np.asarray(grp["topology_map"][...], dtype=np.float32)
                los_mask = np.asarray(grp["los_mask"][...], dtype=np.float32)
                h_tx = float(np.asarray(grp["uav_height"][...]).reshape(-1)[0])
                priors = computer.compute(topology, los_mask, h_tx)

                g = g_city.require_group(sample)
                for name, arr in (
                    ("path_loss_prior", priors.path_loss_prior),
                    ("path_loss_los_prior", priors.path_loss_los_prior),
                    ("path_loss_nlos_prior", priors.path_loss_nlos_prior),
                    ("delay_spread_prior", priors.delay_spread_prior),
                    ("angular_spread_prior", priors.angular_spread_prior),
                ):
                    if name in g:
                        del g[name]
                    g.create_dataset(name, data=arr, compression="gzip")
                g.attrs["topology_class_6"] = priors.topology_class_6
                g.attrs["topology_class_3"] = priors.topology_class_3
                g.attrs["antenna_bin"] = priors.antenna_bin

                done += 1
                if done % 100 == 0:
                    print(f"[precompute-priors] {done}/{total} samples")

    print(f"[precompute-priors] wrote {args.out}")


if __name__ == "__main__":
    main()
