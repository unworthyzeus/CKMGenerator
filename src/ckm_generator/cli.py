"""Command-line interface for CKM Generator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .diagnostics import RuntimeDependencyError, inspect_runtime
from .generator import CKMGenerator
from .io_utils import HDF5_SUFFIXES, load_hdf5_samples, load_single_input
from .output_utils import resolve_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CKM LoS/NLoS masks, priors, and Try 80 predictions.")
    parser.add_argument("--input", type=Path, required=True, help="Topology image/array or HDF5 file.")
    parser.add_argument("--height", type=float, default=None, help="Antenna height in metres. Used when the input does not provide one.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults to outputs/runs/<timestamp>_cli.")
    parser.add_argument("--run-name", type=str, default="cli", help="Name suffix used for the default timestamped output folder.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "directml"), default="auto")
    parser.add_argument("--mask-source", choices=("auto", "exact", "generated", "provided"), default="auto")
    parser.add_argument("--skip-model", action="store_true", help="Only generate masks and priors.")
    parser.add_argument("--check-runtime", action="store_true", help="Print PyTorch/DirectML/CUDA diagnostics and exit.")
    parser.add_argument("--save-arrays", action=argparse.BooleanOptionalAction, default=True, help="Save numeric .npz matrices for masks, priors, and predictions.")
    parser.add_argument("--save-masks", action=argparse.BooleanOptionalAction, default=True, help="Save LoS/NLoS mask PNG files.")

    parser.add_argument("--los-mask", type=Path, default=None, help="Optional LoS reference mask for non-HDF5 inputs.")
    parser.add_argument("--nlos-mask", type=Path, default=None, help="Optional NLoS reference mask for non-HDF5 inputs.")
    parser.add_argument("--topology-max-m", type=float, default=90.0, help="Scale for 8/16-bit topology images when not encoded in metres.")
    parser.add_argument("--image-values-are-metres", action="store_true")

    parser.add_argument("--hdf5-city", type=str, default=None)
    parser.add_argument("--hdf5-sample", type=str, default=None)
    parser.add_argument("--all-hdf5", action="store_true")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--los-sample-step-px", type=float, default=None)
    parser.add_argument("--los-clearance-m", type=float, default=None)
    parser.add_argument("--los-building-dilation-px", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_runtime:
        print(json.dumps(inspect_runtime(args.device).as_dict(), indent=2))
        return

    input_suffix = args.input.suffix.lower()

    if input_suffix in HDF5_SUFFIXES:
        samples = list(load_hdf5_samples(args.input, city=args.hdf5_city, sample=args.hdf5_sample))
        if not args.all_hdf5 and not args.hdf5_city and not args.hdf5_sample:
            samples = samples[:1]
        if args.limit is not None:
            samples = samples[: max(args.limit, 0)]
        if not samples:
            raise SystemExit("No matching HDF5 samples found.")
        if args.height is not None:
            for sample in samples:
                sample.height_m = float(args.height)
    else:
        samples = [
            load_single_input(
                args.input,
                height_m=args.height,
                los_mask_path=args.los_mask,
                nlos_mask_path=args.nlos_mask,
                topology_max_m=args.topology_max_m,
                image_values_are_metres=args.image_values_are_metres,
            )
        ]

    try:
        generator = CKMGenerator(checkpoint_path=args.checkpoint, device=args.device, load_model=not args.skip_model)
    except RuntimeDependencyError as exc:
        raise SystemExit(f"Runtime error: {exc}") from exc
    out_dir = resolve_output_dir(args.out, label=args.run_name)
    summaries = []
    for sample in samples:
        if sample.height_m is None:
            raise SystemExit(f"Sample {sample.sample_id} has no antenna height. Pass --height.")
        try:
            result = generator.generate(
                sample,
                mask_source=args.mask_source,
                run_model=not args.skip_model,
                los_sample_step_px=args.los_sample_step_px,
                los_clearance_m=args.los_clearance_m,
                los_building_dilation_px=args.los_building_dilation_px,
            )
        except RuntimeDependencyError as exc:
            raise SystemExit(f"Generation error for {sample.sample_id}: {exc}") from exc
        files = generator.save_result(result, out_dir, save_arrays=args.save_arrays, save_masks=args.save_masks)
        summaries.append(
            {
                "sample_id": result.sample_id,
                "antenna_height_m": result.antenna_height_m,
                "mask_source": result.mask_source,
                "requested_mask_source": result.requested_mask_source,
                "topology_class_6": result.topology_class_6,
                "antenna_bin": result.antenna_bin,
                "mask_comparison": None if result.mask_comparison is None else result.mask_comparison.__dict__,
                "files": files,
            }
        )
    runtime = None
    if getattr(generator, "runtime_report", None) is not None:
        runtime = generator.runtime_report.as_dict()
    print(json.dumps({"n_samples": len(summaries), "output_dir": str(out_dir), "runtime": runtime, "samples": summaries}, indent=2))


if __name__ == "__main__":
    main()
