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
    parser.add_argument("--input", type=Path, default=None, help="Topology image/array or HDF5 file.")
    parser.add_argument("--height", type=float, default=None, help="Antenna height in metres. Used when the input does not provide one.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory. Defaults to outputs/runs/<timestamp>_cli.")
    parser.add_argument("--run-name", type=str, default="cli", help="Name suffix used for the default timestamped output folder.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "directml"), default="auto")
    parser.add_argument("--mask-source", choices=("auto", "generated", "provided"), default="auto")
    parser.add_argument("--skip-model", action="store_true", help="Only generate masks and priors.")
    parser.add_argument("--check-runtime", action="store_true", help="Print PyTorch/DirectML/CUDA diagnostics and exit.")
    parser.add_argument("--batch-size", type=int, default=None, help="Model inference batch size. Defaults to the runtime recommendation.")
    parser.add_argument("--mixed-precision", action="store_true", help="Use CUDA fp16 autocast for model inference. Faster on many NVIDIA GPUs, with small numeric differences.")
    parser.add_argument("--save-arrays", action=argparse.BooleanOptionalAction, default=True, help="Save numeric .npz matrices for masks, priors, and predictions.")
    parser.add_argument("--save-masks", action=argparse.BooleanOptionalAction, default=True, help="Save LoS/NLoS mask PNG files.")
    parser.add_argument("--save-visual-maps", action=argparse.BooleanOptionalAction, default=True, help="Save prior/prediction PNG figures.")

    parser.add_argument("--los-mask", type=Path, default=None, help="Optional LoS reference mask for non-HDF5 inputs.")
    parser.add_argument("--nlos-mask", type=Path, default=None, help="Optional NLoS reference mask for non-HDF5 inputs.")
    parser.add_argument("--topology-max-m", type=float, default=90.0, help="Scale for 8/16-bit topology images when not encoded in metres.")
    parser.add_argument("--image-values-are-metres", action="store_true")
    parser.add_argument("--meters-per-pixel", type=float, default=1.0, help="Input grid spacing. Inputs are resampled to the model's 1 m/pixel grid before 513x513 crop/pad.")

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
    if args.input is None:
        raise SystemExit("--input is required unless --check-runtime is used.")

    input_suffix = args.input.suffix.lower()

    if input_suffix in HDF5_SUFFIXES:
        samples = list(load_hdf5_samples(args.input, city=args.hdf5_city, sample=args.hdf5_sample, meters_per_pixel=args.meters_per_pixel))
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
                meters_per_pixel=args.meters_per_pixel,
            )
        ]

    try:
        generator = CKMGenerator(checkpoint_path=args.checkpoint, device=args.device, load_model=not args.skip_model)
    except RuntimeDependencyError as exc:
        raise SystemExit(f"Runtime error: {exc}") from exc
    out_dir = resolve_output_dir(args.out, label=args.run_name)
    recommended_batch = int(getattr(getattr(generator, "runtime_report", None), "recommended_batch_size", 1) or 1)
    batch_size = max(1, int(args.batch_size or recommended_batch))
    summaries = []
    pending = []

    def flush_pending() -> None:
        if not pending:
            return
        if not args.skip_model:
            generator.predict_results(pending, batch_size=batch_size, mixed_precision=args.mixed_precision)
        for result in pending:
            files = generator.save_result(
                result,
                out_dir,
                save_arrays=args.save_arrays,
                save_masks=args.save_masks,
                save_visual_maps=args.save_visual_maps,
            )
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
        pending.clear()

    for sample in samples:
        if sample.height_m is None:
            raise SystemExit(f"Sample {sample.sample_id} has no antenna height. Pass --height.")
        try:
            result = generator.generate(
                sample,
                mask_source=args.mask_source,
                run_model=False,
                los_sample_step_px=args.los_sample_step_px,
                los_clearance_m=args.los_clearance_m,
                los_building_dilation_px=args.los_building_dilation_px,
            )
        except (RuntimeDependencyError, ValueError, RuntimeError) as exc:
            raise SystemExit(f"Generation error for {sample.sample_id}: {exc}") from exc
        pending.append(result)
        if args.skip_model or len(pending) >= batch_size:
            try:
                flush_pending()
            except (RuntimeDependencyError, ValueError, RuntimeError) as exc:
                raise SystemExit(f"Generation error while saving/predicting: {exc}") from exc
    try:
        flush_pending()
    except (RuntimeDependencyError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Generation error while saving/predicting: {exc}") from exc
    runtime = None
    if getattr(generator, "runtime_report", None) is not None:
        runtime = generator.runtime_report.as_dict()
    print(json.dumps({"n_samples": len(summaries), "output_dir": str(out_dir), "runtime": runtime, "batch_size": batch_size, "samples": summaries}, indent=2))


if __name__ == "__main__":
    main()
