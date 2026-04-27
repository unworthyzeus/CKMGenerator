"""Streamlit interface for CKM Generator."""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from .diagnostics import inspect_runtime
from .generator import CKMGenerator
from .io_utils import HDF5_SUFFIXES, load_hdf5_samples, load_single_input
from .output_utils import list_existing_output_dirs, make_timestamped_output_dir, safe_output_name
from .paths import DEFAULT_OUTPUT_DIR


def main() -> None:
    st.set_page_config(page_title="CKM Generator", layout="wide")
    st.title("CKM Generator")

    with st.sidebar:
        input_path = st.text_input("Local input path", value="")
        uploaded = st.file_uploader("Or upload topology / HDF5", type=["png", "jpg", "jpeg", "tif", "tiff", "npy", "npz", "csv", "txt", "json", "h5", "hdf5", "hdf"])
        height = st.number_input("Antenna height (m)", min_value=10.0, max_value=478.0, value=56.0, step=0.1, format="%.2f")
        device = st.selectbox("Device", ["auto", "directml", "cpu", "cuda"], index=0)
        runtime_report = inspect_runtime(device)
        st.caption(f"Runtime: torch={runtime_report.torch_version or 'missing'} | selected={runtime_report.selected_device}")
        st.caption("Rx height is fixed at 1.5 m for the CKM calibration and Try 78/79/80 priors.")
        mask_source = st.selectbox("Mask used by priors/model", ["auto", "exact", "generated", "provided"], index=0)
        run_model = st.checkbox("Run Try 80 model", value=True)
        save_arrays = st.checkbox("Save numeric arrays (.npz)", value=True)
        save_masks = st.checkbox("Save LoS/NLoS mask PNGs", value=True)
        topology_max_m = st.number_input("Image max height (m)", min_value=1.0, max_value=500.0, value=90.0, step=1.0)
        image_values_are_metres = st.checkbox("Image pixels are already metres", value=False)
        hdf5_city = st.text_input("HDF5 city filter", value="")
        hdf5_sample = st.text_input("HDF5 sample filter", value="")
        output_mode = st.selectbox("Output folder", ["New run in outputs/runs", "Choose existing output folder", "Custom path"], index=0)
        run_name = "streamlit"
        selected_existing = None
        custom_out_dir = ""
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
        else:
            custom_out_dir = st.text_input("Custom output directory", value=str(DEFAULT_OUTPUT_DIR / "manual"))
        run = st.button("Generate", type="primary")

    if not run:
        return

    path = _materialize_input(input_path, uploaded)
    if path is None:
        st.error("Choose a local path or upload a file.")
        return

    try:
        if path.suffix.lower() in HDF5_SUFFIXES:
            samples = list(load_hdf5_samples(path, city=hdf5_city or None, sample=hdf5_sample or None))
            if not samples:
                st.error("No matching HDF5 sample found.")
                return
            sample = samples[0]
            if sample.height_m is None:
                sample.height_m = float(height)
        else:
            sample = load_single_input(
                path,
                height_m=float(height),
                topology_max_m=float(topology_max_m),
                image_values_are_metres=bool(image_values_are_metres),
            )

        generator = CKMGenerator(device=device, load_model=run_model)
        result = generator.generate(sample, mask_source=mask_source, run_model=run_model)
        out_dir = _selected_output_dir(
            output_mode,
            run_name=run_name,
            selected_existing=selected_existing,
            custom_out_dir=custom_out_dir,
        )
        files = generator.save_result(result, out_dir, save_arrays=save_arrays, save_masks=save_masks)
    except Exception as exc:
        st.exception(exc)
        return

    st.success(f"Generated {result.sample_id} at h={result.antenna_height_m:.2f} m")
    st.caption(f"Output: {out_dir}")
    if result.mask_comparison:
        st.json(result.mask_comparison.__dict__)

    cols = st.columns(2)
    cols[0].image(result.los_mask, caption="LoS mask", clamp=True)
    cols[1].image(result.nlos_mask, caption="NLoS mask", clamp=True)

    if result.predictions:
        st.image(files["pred_joint"], caption="Joint prediction")
        pcols = st.columns(3)
        for col, task in zip(pcols, ("path_loss", "delay_spread", "angular_spread")):
            col.image(files[f"pred_{task}"], caption=task)

    with st.expander("Generated files", expanded=False):
        st.json(files)


def _materialize_input(input_path: str, uploaded) -> Path | None:
    if input_path.strip():
        return Path(input_path.strip())
    if uploaded is None:
        return None
    suffix = Path(uploaded.name).suffix
    tmp = Path(tempfile.mkdtemp(prefix="ckm_generator_")) / f"uploaded{suffix}"
    tmp.write_bytes(uploaded.getbuffer())
    return tmp


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


if __name__ == "__main__":
    main()
