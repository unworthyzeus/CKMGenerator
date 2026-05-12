"""Streamlit interface for CKM Generator."""
from __future__ import annotations

import os
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import streamlit as st

from .diagnostics import inspect_runtime
from .generator import CKMGenerator
from .io_utils import HDF5_SUFFIXES, load_hdf5_samples, load_single_input
from .output_utils import list_existing_output_dirs, make_timestamped_output_dir, safe_output_name
from .paths import DEFAULT_OUTPUT_DIR
from .plotting import TASK_LABELS


def main() -> None:
    st.set_page_config(page_title="CKM Generator", layout="wide")
    st.title("CKM Generator")

    with st.sidebar:
        input_path = st.text_input("Local input path", value="")
        uploaded = st.file_uploader(
            "Or upload topology / data / HDF5",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "npy", "npz", "csv", "txt", "json", "h5", "hdf5", "hdf"],
            accept_multiple_files=True,
            help="You can upload several topology/data files at once. LoS/NLoS sidecar mask files are kept available but not generated as separate samples.",
        )
        height = st.number_input("Antenna height (m)", min_value=10.0, max_value=478.0, value=56.0, step=0.1, format="%.2f")
        device = st.selectbox("Device", ["auto", "directml", "cpu", "cuda"], index=0)
        runtime_report = inspect_runtime(device)
        st.caption(f"Runtime: torch={runtime_report.torch_version or 'missing'} | selected={runtime_report.selected_device}")
        st.caption("Rx height is fixed at 1.5 m for the CKM calibration and Try 78/79/80 priors.")
        mask_source = st.selectbox(
            "Mask used by priors/model",
            ["auto", "provided", "generated"],
            index=0,
            help=(
                "auto: use masks shipped with the input/upload when available, otherwise ray-cast. "
                "provided: require masks from the input/upload. generated: always ray-cast from topology + antenna height."
            ),
        )
        prior_backend = st.selectbox(
            "Prior backend",
            ["auto", "numpy", "torch", "cuda", "directml", "torch-cpu"],
            index=0,
            help="auto uses the selected torch runtime when available; numpy is the exact CPU fallback.",
        )
        run_model = st.checkbox("Run Try 80 model", value=True)
        model_batch_size = st.number_input(
            "Model batch size",
            min_value=1,
            max_value=8,
            value=max(1, int(runtime_report.recommended_batch_size or 1)),
            step=1,
            disabled=not run_model,
            help=f"{runtime_report.recommendation_reason} Increase if your GPU has spare VRAM; lower it if you hit out-of-memory.",
        )
        mixed_precision = st.checkbox(
            "CUDA mixed precision",
            value=False,
            disabled=(not run_model or runtime_report.selected_device != "cuda"),
            help="Faster on many NVIDIA GPUs, but predictions can differ slightly from full fp32.",
        )
        save_arrays = st.checkbox("Save numeric arrays (.npz)", value=True)
        save_masks = st.checkbox("Save LoS/NLoS mask PNGs", value=True)
        save_visual_maps = st.checkbox(
            "Save visual map PNGs",
            value=True,
            help="Saves prior/prediction PNG figures. Turn off for faster batch exports; arrays, masks, and metadata can still be saved.",
        )
        topology_max_m = st.number_input(
            "Image max height (m)",
            min_value=1.0,
            max_value=500.0,
            value=90.0,
            step=1.0,
            help="Only used for PNG/JPG/TIFF/BMP topology images. Ignored for HDF5, NPZ, NPY, CSV, TXT, and JSON inputs.",
        )
        image_values_are_metres = st.checkbox("Image pixels are already metres", value=True)
        meters_per_pixel = st.number_input(
            "Distance between pixels (m)",
            min_value=0.05,
            max_value=20.0,
            value=1.0,
            step=0.05,
            format="%.2f",
            help="Input grid spacing. The generator converts inputs to the calibrated 1 m/pixel, 513x513 model grid before inference.",
        )
        hdf5_city = st.text_input("HDF5 city filter", value="")
        hdf5_sample = st.text_input("HDF5 sample filter", value="")
        output_mode = st.selectbox(
            "Output folder",
            ["New run in outputs/runs", "Choose existing output folder", "Custom path"],
            index=0,
            help="Custom path shows a text field. Browsers cannot open a native folder picker here.",
        )
        run_name = "streamlit"
        selected_existing = None
        custom_out_dir = ""
        folder_to_open = None
        if output_mode == "New run in outputs/runs":
            run_name = st.text_input("Run name", value="streamlit")
        elif output_mode == "Choose existing output folder":
            existing_dirs = list_existing_output_dirs()
            labels = [str(path.relative_to(DEFAULT_OUTPUT_DIR)) if path.is_relative_to(DEFAULT_OUTPUT_DIR) else str(path) for path in existing_dirs]
            if labels:
                selected_label = st.selectbox("Existing folder", labels, index=0)
                selected_existing = existing_dirs[labels.index(selected_label)]
            else:
                st.caption("No output folders yet; a manual folder will be created.")
                selected_existing = DEFAULT_OUTPUT_DIR / "manual"
            folder_to_open = selected_existing
        else:
            custom_out_dir = st.text_input(
                "Custom output folder path",
                value=str(DEFAULT_OUTPUT_DIR / "manual"),
                help="Type or paste a local folder path. It will be created when you generate, or when you press Open/create.",
            )
            folder_to_open = Path(custom_out_dir).expanduser() if custom_out_dir.strip() else None

        if folder_to_open is not None and st.button("Open/create output folder"):
            try:
                _open_folder(folder_to_open)
            except Exception as exc:
                st.error(f"Could not open folder: {exc}")
        run = st.button("Generate", type="primary")

    if not run:
        return

    paths = _materialize_inputs(input_path, uploaded)
    if not paths:
        st.error("Choose a local path or upload one or more files.")
        return

    try:
        generator = CKMGenerator(device=device, load_model=run_model)
        generator.config.setdefault("priors", {})["backend"] = prior_backend
        out_dir = _selected_output_dir(
            output_mode,
            run_name=run_name,
            selected_existing=selected_existing,
            custom_out_dir=custom_out_dir,
        )
        rendered = []
        progress = st.progress(0.0, text="Generating...")
        phase_count = 3 if run_model else 2
        for idx, path in enumerate(paths, start=1):
            sample = _load_first_sample(
                path,
                height=float(height),
                topology_max_m=float(topology_max_m),
                image_values_are_metres=bool(image_values_are_metres),
                meters_per_pixel=float(meters_per_pixel),
                hdf5_city=hdf5_city or None,
                hdf5_sample=hdf5_sample or None,
            )
            result = generator.generate(sample, mask_source=mask_source, run_model=False)
            rendered.append((result, {}))
            progress.progress(idx / (len(paths) * phase_count), text=f"Prepared {idx}/{len(paths)}")
        if run_model:
            def _model_progress(done: int, total: int) -> None:
                progress.progress((len(paths) + done) / (len(paths) * phase_count), text=f"Ran model {done}/{total}")

            generator.predict_results(
                [result for result, _ in rendered],
                batch_size=int(model_batch_size),
                mixed_precision=bool(mixed_precision),
                progress_callback=_model_progress,
            )
        saved = []
        save_offset = len(paths) * (phase_count - 1)
        for idx, (result, _) in enumerate(rendered, start=1):
            files = generator.save_result(
                result,
                out_dir,
                save_arrays=save_arrays,
                save_masks=save_masks,
                save_visual_maps=save_visual_maps,
            )
            saved.append((result, files))
            progress.progress((save_offset + idx) / (len(paths) * phase_count), text=f"Saved {idx}/{len(paths)}")
        rendered = saved
        progress.empty()
    except Exception as exc:
        _render_error(exc, traceback.format_exc())
        return

    st.success(f"Generated {len(rendered)} sample{'s' if len(rendered) != 1 else ''}")
    st.caption(f"Output: {out_dir}")
    if len(rendered) == 1:
        _render_result(*rendered[0])
    else:
        tabs = st.tabs([result.sample_id for result, _ in rendered])
        for tab, (result, files) in zip(tabs, rendered):
            with tab:
                _render_result(result, files)

    with st.expander("Generated files", expanded=False):
        st.json({result.sample_id: files for result, files in rendered})


def _render_result(result, files: dict[str, str]) -> None:
    st.caption(
        f"{result.sample_id} | h={result.antenna_height_m:.2f} m | "
        f"mask={result.mask_source} (requested {result.requested_mask_source})"
    )
    if result.mask_source == "generated":
        st.warning("No provided LoS/NLoS mask was used for this sample; masks were ray-cast from topology + antenna height.")
    if result.mask_comparison:
        st.json(result.mask_comparison.__dict__)

    cols = st.columns(3)
    cols[0].image(_topology_display(result.topology), caption="Topology", use_container_width=True)
    cols[1].image(_mask_display(result.los_mask), caption="LoS mask", use_container_width=True)
    cols[2].image(_mask_display(result.nlos_mask), caption="NLoS mask", use_container_width=True)

    if result.predictions:
        if "pred_joint" in files:
            st.image(files["pred_joint"], caption="Joint prediction", use_container_width=True)
        pcols = st.columns(3)
        for col, task in zip(pcols, ("path_loss", "delay_spread", "angular_spread")):
            if f"pred_{task}" in files:
                col.image(files[f"pred_{task}"], use_container_width=True)
            else:
                label, unit, _ = TASK_LABELS[task]
                col.image(
                    _map_display(result.predictions[task], result.ground_mask),
                    caption=f"{label} ({unit})",
                    use_container_width=True,
                )


def _render_error(exc: Exception, trace: str) -> None:
    st.error(f"{type(exc).__name__}: {exc}")
    with st.expander("Traceback", expanded=False):
        st.code(trace, language="text")


def _materialize_inputs(input_path: str, uploaded: Iterable | None) -> list[Path]:
    if input_path.strip():
        return [Path(input_path.strip())]
    uploaded_files = list(uploaded or [])
    if not uploaded_files:
        return []
    tmp_dir = Path(tempfile.mkdtemp(prefix="ckm_generator_uploads_"))
    paths: list[Path] = []
    for idx, file in enumerate(uploaded_files, start=1):
        safe_name = _safe_upload_name(file.name, fallback=f"uploaded_{idx}")
        path = tmp_dir / safe_name
        path.write_bytes(file.getbuffer())
        paths.append(path)
    return [path for path in paths if not _looks_like_sidecar_mask(path)]


def _load_first_sample(
    path: Path,
    *,
    height: float,
    topology_max_m: float,
    image_values_are_metres: bool,
    meters_per_pixel: float,
    hdf5_city: str | None,
    hdf5_sample: str | None,
):
    if path.suffix.lower() in HDF5_SUFFIXES:
        sample = next(load_hdf5_samples(path, city=hdf5_city, sample=hdf5_sample, meters_per_pixel=meters_per_pixel), None)
        if sample is None:
            raise ValueError(f"No matching HDF5 sample found in {path}")
        if sample.height_m is None:
            sample.height_m = height
        return sample
    return load_single_input(
        path,
        height_m=height,
        topology_max_m=topology_max_m,
        image_values_are_metres=image_values_are_metres,
        meters_per_pixel=meters_per_pixel,
    )


def _safe_upload_name(name: str, *, fallback: str) -> str:
    path = Path(name)
    stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in path.stem).strip("_")
    suffix = "".join(ch for ch in path.suffix if ch.isalnum() or ch == ".")
    return f"{stem or fallback}{suffix}"


def _looks_like_sidecar_mask(path: Path) -> bool:
    stem = path.stem.lower()
    return stem.endswith("_los_mask") or stem.endswith("_nlos_mask") or stem.endswith("_los") or stem.endswith("_nlos")


def _selected_output_dir(
    output_mode: str,
    *,
    run_name: str,
    selected_existing: Path | None,
    custom_out_dir: str,
) -> Path:
    if output_mode == "New run in outputs/runs":
        return make_timestamped_output_dir(label=safe_output_name(run_name, fallback="streamlit"))
    if output_mode == "Choose existing output folder":
        return selected_existing or (DEFAULT_OUTPUT_DIR / "manual")
    return Path(custom_out_dir).expanduser().resolve()


def _open_folder(path: Path) -> None:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif os.name == "posix":
        opener = "open" if sys_platform_is_darwin() else "xdg-open"
        subprocess.Popen([opener, str(path)])


def sys_platform_is_darwin() -> bool:
    import sys

    return sys.platform == "darwin"


def _topology_display(topology: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(topology, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(arr.max(initial=0.0))
    if vmax <= 1.0e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip(arr / vmax * 255.0, 0.0, 255.0).astype(np.uint8)


def _mask_display(mask: np.ndarray) -> np.ndarray:
    return ((np.asarray(mask, dtype=np.float32) > 0.5).astype(np.uint8) * 255)


def _map_display(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = (np.asarray(mask, dtype=np.float32) > 0.5) & np.isfinite(arr)
    finite = arr[valid]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    if finite.size > 16:
        vmin, vmax = np.percentile(finite, [1.0, 99.0])
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
    if abs(float(vmax) - float(vmin)) < 1.0e-6:
        vmax = float(vmin) + 1.0
    out = np.zeros(arr.shape, dtype=np.uint8)
    out[valid] = np.clip((arr[valid] - float(vmin)) / (float(vmax) - float(vmin)) * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


if __name__ == "__main__":
    main()
