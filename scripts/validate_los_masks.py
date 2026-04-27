"""Validate CKM Generator LoS/NLoS masks against an HDF5 reference dataset."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ckm_generator.generator import CKMGenerator  # noqa: E402
from ckm_generator.io_utils import SampleInput, load_hdf5_samples  # noqa: E402
from ckm_generator.los import compare_los_masks, compute_los_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "los_validation")
    parser.add_argument("--city", type=str, default=None)
    parser.add_argument("--sample", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=("generator-auto", "generator-auto-no-input-mask", "exact-copy", "raycast"),
        default="generator-auto",
        help=(
            "generator-auto validates the real default generator path. "
            "generator-auto-no-input-mask removes HDF5 masks before generation, so the configured exact reference lookup is tested. "
            "exact-copy is a trivial GT self-check. raycast checks the approximate geometry fallback."
        ),
    )
    parser.add_argument("--sample-step-px", type=float, default=0.25)
    parser.add_argument("--clearance-m", type=float, default=0.0)
    parser.add_argument("--building-dilation-px", type=int, default=0)
    parser.add_argument("--keep-hdf5-group", action="store_true", help="With generator-auto-no-input-mask, keep the original group path instead of forcing fingerprint lookup.")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--allow-mismatch", action="store_true", help="Exit 0 even when mismatches are found.")
    return parser.parse_args()


def _without_input_masks(sample: SampleInput, *, keep_hdf5_group: bool) -> SampleInput:
    metadata = dict(sample.metadata)
    if not keep_hdf5_group:
        metadata.pop("hdf5_group", None)
    return SampleInput(
        sample_id=sample.sample_id,
        topology=sample.topology,
        height_m=sample.height_m,
        reference_los_mask=None,
        reference_nlos_mask=None,
        source=sample.source,
        metadata=metadata,
    )


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    generator = None
    if args.mode.startswith("generator-"):
        generator = CKMGenerator(load_model=False)

    samples = load_hdf5_samples(args.hdf5, city=args.city, sample=args.sample)
    for idx, sample in enumerate(samples):
        if args.limit is not None and idx >= args.limit:
            break
        if sample.height_m is None or sample.reference_los_mask is None:
            continue

        if args.mode == "exact-copy":
            generated = np.asarray(sample.reference_los_mask, dtype=np.float32)
            cmp = compare_los_masks(generated, sample.reference_los_mask, sample.topology)
            resolved_source = "exact-copy"
        elif args.mode == "raycast":
            generated = compute_los_mask(
                sample.topology,
                sample.height_m,
                sample_step_px=args.sample_step_px,
                clearance_m=args.clearance_m,
                building_dilation_px=args.building_dilation_px,
            )
            cmp = compare_los_masks(generated, sample.reference_los_mask, sample.topology)
            resolved_source = "raycast"
        else:
            assert generator is not None
            gen_sample = sample
            if args.mode == "generator-auto-no-input-mask":
                gen_sample = _without_input_masks(sample, keep_hdf5_group=args.keep_hdf5_group)
            result = generator.generate(gen_sample, mask_source="auto", run_model=False)
            cmp = compare_los_masks(result.los_mask, sample.reference_los_mask, sample.topology)
            resolved_source = result.mask_source

        row = {
            "sample_id": sample.sample_id,
            "height_m": sample.height_m,
            "resolved_mask_source": resolved_source,
            **cmp.__dict__,
        }
        rows.append(row)
        if cmp.mismatches != 0 or args.progress_every <= 1 or len(rows) % args.progress_every == 0:
            print(
                f"{len(rows)} {sample.sample_id}: mismatch={cmp.mismatch_fraction:.8f} "
                f"pixels={cmp.mismatches} source={resolved_source}"
            )

    csv_path = args.out / "los_validation.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "n_samples": len(rows),
        "mean_mismatch_fraction": sum(r["mismatch_fraction"] for r in rows) / max(len(rows), 1),
        "max_mismatch_fraction": max((r["mismatch_fraction"] for r in rows), default=0.0),
        "total_mismatches": sum(r["mismatches"] for r in rows),
        "total_pixels": sum(r["n_pixels"] for r in rows),
        "mask_sources": {source: sum(1 for r in rows if r["resolved_mask_source"] == source) for source in sorted({r["resolved_mask_source"] for r in rows})},
        "options": {
            "sample_step_px": args.sample_step_px,
            "clearance_m": args.clearance_m,
            "building_dilation_px": args.building_dilation_px,
            "mode": args.mode,
            "keep_hdf5_group": args.keep_hdf5_group,
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["total_mismatches"] != 0 and not args.allow_mismatch:
        raise SystemExit(f"LoS validation failed: {summary['total_mismatches']} mismatched pixels.")


if __name__ == "__main__":
    main()
