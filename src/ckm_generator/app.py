"""Streamlit interface for CKM Generator."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import traceback
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import streamlit as st

if __package__ in {None, ""}:
    # Streamlit runs app.py as a script, so relative imports need a package fallback.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ckm_generator.diagnostics import inspect_runtime
    from ckm_generator.generator import CKMGenerator
    from ckm_generator.io_utils import HDF5_SUFFIXES, load_hdf5_samples, load_single_input
    from ckm_generator.output_utils import list_existing_output_dirs, make_timestamped_output_dir, safe_output_name
    from ckm_generator.paths import DEFAULT_OUTPUT_DIR
    from ckm_generator.plotting import TASK_LABELS
else:
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
        st.caption("Rx height is fixed at 1.5 m for the CKM calibration and final calibrated priors.")
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
        run_model = st.checkbox("Run final residual model", value=True)
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
        save_arrays = st.checkbox(
            "Save numeric arrays (.npz)",
            value=True,
            help="Fast authoritative numeric export. Includes topology, masks, priors, and predictions.",
        )
        compress_arrays = st.checkbox(
            "Compress numeric arrays",
            value=False,
            disabled=not save_arrays,
            help="Smaller .npz files, but slower. Leave off for batch speed.",
        )
        save_masks = st.checkbox(
            "Save LoS/NLoS mask PNGs",
            value=False,
            help="Optional visual mask files. Masks are still stored in the .npz arrays.",
        )
        save_visual_maps = st.checkbox(
            "Save visual map PNGs",
            value=False,
            help="Optional rendered prior/prediction figures. Leave off for faster batch exports; previews still appear in the app.",
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
        cached = st.session_state.get("last_preview_manifest")
        if cached:
            _render_preview(cached, cached=True)
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
                compress_arrays=compress_arrays,
            )
            saved.append((result, files))
            progress.progress((save_offset + idx) / (len(paths) * phase_count), text=f"Saved {idx}/{len(paths)}")
        rendered = saved
        preview_manifest = _preview_manifest(rendered, out_dir)
        st.session_state["last_preview_manifest"] = preview_manifest
        progress.empty()
    except Exception as exc:
        _render_error(exc, traceback.format_exc())
        return

    _render_preview(preview_manifest, cached=False, memory_results=rendered)


def _render_preview(preview: dict[str, object], *, cached: bool = False, memory_results=None) -> None:
    items = list(preview.get("items", []))
    out_dir = preview.get("out_dir")
    if cached:
        st.info(f"Showing last generated batch: {len(items)} sample{'s' if len(items) != 1 else ''}")
    else:
        st.success(f"Generated {len(items)} sample{'s' if len(items) != 1 else ''}")
    if out_dir:
        st.caption(f"Output: {out_dir}")

    memory_by_label = {}
    if memory_results:
        memory_by_label = {_result_label(idx, result): (result, files) for idx, (result, files) in enumerate(memory_results, start=1)}

    if len(items) > 1:
        st.dataframe(_preview_summary(items), use_container_width=True, hide_index=True)
    if len(items) == 1:
        _render_preview_item(items[0], memory_by_label=memory_by_label)
    elif len(items) > 6:
        labels = [str(item["label"]) for item in items]
        current = st.session_state.get("ckm_preview_sample")
        index = labels.index(current) if current in labels else 0
        selected = st.selectbox(
            "Preview sample",
            labels,
            index=index,
            key="ckm_preview_sample",
            help="All samples were generated and saved; choose any row here to inspect its preview.",
        )
        selected_item = items[labels.index(selected)]
        _render_preview_item(selected_item, memory_by_label=memory_by_label)
    else:
        tabs = st.tabs([str(item["label"]) for item in items])
        for tab, item in zip(tabs, items):
            with tab:
                _render_preview_item(item, memory_by_label=memory_by_label)

    with st.expander("Generated files", expanded=False):
        st.json({str(item["label"]): item.get("files", {}) for item in items})


def _preview_manifest(rendered, out_dir: Path) -> dict[str, object]:
    return {
        "out_dir": str(out_dir),
        "items": [_preview_item(idx, result, files) for idx, (result, files) in enumerate(rendered, start=1)],
    }


def _preview_item(idx: int, result, files: dict[str, str]) -> dict[str, object]:
    return {
        "label": _result_label(idx, result),
        "sample_id": result.sample_id,
        "antenna_height_m": float(result.antenna_height_m),
        "mask_source": result.mask_source,
        "requested_mask_source": result.requested_mask_source,
        "files": {key: str(value) for key, value in files.items()},
        "mask_comparison": _comparison_dict(result.mask_comparison),
    }


def _comparison_dict(comparison) -> dict[str, object] | None:
    if comparison is None:
        return None
    if isinstance(comparison, dict):
        return comparison
    return dict(comparison.__dict__)


def _result_label(idx: int, result) -> str:
    return f"{idx:03d} | {result.sample_id} | h={result.antenna_height_m:.2f} m"


def _preview_summary(items) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, item in enumerate(items, start=1):
        files = item.get("files", {})
        if not isinstance(files, dict):
            files = {}
        rows.append(
            {
                "#": idx,
                "sample": item.get("sample_id", ""),
                "height_m": round(float(item.get("antenna_height_m", 0.0)), 2),
                "mask": item.get("mask_source", ""),
                "arrays": files.get("arrays", ""),
                "metadata": files.get("metadata", ""),
                "png_files": sum(1 for value in files.values() if str(value).lower().endswith(".png")),
            }
        )
    return rows


def _render_preview_item(item: dict[str, object], *, memory_by_label: dict[str, tuple[object, dict[str, str]]]) -> None:
    label = str(item["label"])
    if label in memory_by_label:
        _render_result(*memory_by_label[label])
        return
    _render_saved_result(item)


def _render_saved_result(item: dict[str, object]) -> None:
    files = item.get("files", {})
    if not isinstance(files, dict):
        files = {}
    metadata = _load_metadata(files.get("metadata"))
    sample_id = metadata.get("sample_id", item.get("sample_id", ""))
    height = float(metadata.get("antenna_height_m", item.get("antenna_height_m", 0.0)))
    mask_source = metadata.get("mask_source", item.get("mask_source", ""))
    requested_mask_source = metadata.get("requested_mask_source", item.get("requested_mask_source", ""))
    st.caption(f"{sample_id} | h={height:.2f} m | mask={mask_source} (requested {requested_mask_source})")
    if mask_source == "generated":
        st.warning("No provided LoS/NLoS mask was used for this sample; masks were ray-cast from topology + antenna height.")
    comparison = metadata.get("mask_comparison") or item.get("mask_comparison")
    if comparison:
        _render_mask_comparison(comparison)

    arrays_path = files.get("arrays")
    if not arrays_path:
        st.warning("No .npz arrays were saved for this sample, so the preview cannot be reconstructed after the app rerun.")
        return
    arrays_path = Path(str(arrays_path))
    if not arrays_path.exists():
        st.warning(f"Saved array archive not found: {arrays_path}")
        return
    with np.load(arrays_path) as data:
        arrays = {name: data[name] for name in data.files}

    topology = arrays.get("topology")
    los = arrays.get("los_mask")
    nlos = arrays.get("nlos_mask")
    ground = arrays.get("ground_mask")
    if topology is None or los is None or nlos is None:
        st.warning(f"Saved array archive is missing topology or mask arrays: {arrays_path}")
        return
    if ground is None:
        ground = (np.asarray(topology) <= 1.0e-6).astype(np.float32)

    cols = st.columns(3)
    cols[0].image(_topology_display(topology), caption="Topology", use_container_width=True)
    cols[1].image(_mask_display(los), caption="LoS mask", use_container_width=True)
    cols[2].image(_mask_display(nlos), caption="NLoS mask", use_container_width=True)

    predictions = {task: arrays[f"pred_{task}"] for task in ("path_loss", "delay_spread", "angular_spread") if f"pred_{task}" in arrays}
    if predictions:
        if "pred_joint" in files and Path(str(files["pred_joint"])).exists():
            st.image(files["pred_joint"], caption="Joint prediction", use_container_width=True)
        pcols = st.columns(3)
        for col, task in zip(pcols, ("path_loss", "delay_spread", "angular_spread")):
            label, unit, _ = TASK_LABELS[task]
            if f"pred_{task}" in files and Path(str(files[f"pred_{task}"])).exists():
                col.image(files[f"pred_{task}"], use_container_width=True)
            elif task in predictions:
                col.image(_map_display(predictions[task], ground), caption=f"{label} ({unit})", use_container_width=True)


def _load_metadata(path: object) -> dict[str, object]:
    if not path:
        return {}
    meta_path = Path(str(path))
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _render_result(result, files: dict[str, str]) -> None:
    st.caption(
        f"{result.sample_id} | h={result.antenna_height_m:.2f} m | "
        f"mask={result.mask_source} (requested {result.requested_mask_source})"
    )
    if result.mask_source == "generated":
        st.warning("No provided LoS/NLoS mask was used for this sample; masks were ray-cast from topology + antenna height.")
    if result.mask_comparison:
        _render_mask_comparison(result.mask_comparison.__dict__)

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


def _render_mask_comparison(comparison: dict[str, object]) -> None:
    mismatches = int(comparison.get("mismatches", 0) or 0)
    n_pixels = int(comparison.get("n_pixels", 0) or 0)
    los_iou = float(comparison.get("los_iou", 0.0) or 0.0)
    nlos_iou = float(comparison.get("nlos_iou", 0.0) or 0.0)
    if mismatches == 0:
        st.success(f"Mask validation: 0 mismatches over {n_pixels:,} pixels | LoS IoU={los_iou:.6f} | NLoS IoU={nlos_iou:.6f}")
        return
    st.warning(f"Mask validation: {mismatches:,} mismatches over {n_pixels:,} pixels")
    st.dataframe(
        [
            {
                "mismatch_fraction": comparison.get("mismatch_fraction", 0.0),
                "false_los": comparison.get("false_los", 0),
                "false_nlos": comparison.get("false_nlos", 0),
                "los_iou": los_iou,
                "nlos_iou": nlos_iou,
            }
        ],
        use_container_width=True,
        hide_index=True,
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
