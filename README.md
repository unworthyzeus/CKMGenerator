# CKM Generator

Standalone final generator for CKM topology inputs.

It is intentionally outside `TFGpractice` and includes:

- `models/best_model.pt`: local current Try 80 checkpoint copied from `try80_joint_huge_pathloss_finetune`.
- `calibrations/`: frozen Try 78/79 calibration JSONs used by Try 80.
- `src/ckm_generator/`: clean generator interface, CLI, Streamlit app, loaders, LoS/NLoS ray-caster, plotting.
- `vendor/`: full copied code for Try 80, Try 78, Try 79, and the preliminary final Try 80 bundle.

`models/best_model.pt` is intentionally ignored by Git for now because the
checkpoint is larger than GitHub's regular file limit. Keep it locally under
`models/`, or pass `--checkpoint C:\path\best_model.pt`.

## Supported Inputs

- Images: PNG, JPG, TIFF, BMP topology maps.
- Arrays: `.npy`, `.npz`, `.csv`, `.txt`, `.json`.
- HDF5: full CKM city/sample layout or a simple file/group with only `topology_map`.
- Optional masks: `los_mask` or `nlos_mask` in HDF5, or separate mask files for non-HDF5 inputs.
- Optional antenna height in HDF5: `uav_height`, `antenna_height`, `antenna_height_m`, `height`, `height_m`, or `h_tx`.

If an image is 8/16-bit normalized, pass `--topology-max-m`. If pixels already store metres, pass `--image-values-are-metres`.

## LoS vs NLoS Mask

The original CKM HDF5 normally stores only `los_mask`. `nlos_mask` is derived,
not an independent original label:

```python
ground = topology_map == 0
los = los_mask * ground
nlos = (1 - los_mask) * ground
```

The `ground` term matters: building pixels are excluded from both masks because
they are not valid receiver pixels. A raw `1 - los_mask` would incorrectly turn
buildings into NLoS.

## Is Generated LoS 100% Equal To GT?

Yes when GT/reference is available. The default `auto` mode uses the exact
`los_mask`/`nlos_mask` from the input HDF5 or configured reference HDF5 when it
can find one. If there is no GT/reference, it falls back to the topology +
height ray-caster without throwing an error.

The pure geometry ray-caster is still available, but it is not bit-perfect
against CKM. On the first two CKM samples tested with the strict ray-caster
(`sample_step_px=0.25`), mismatch was about `3.68%` and `2.07%`.

Use modes explicitly:

- `--mask-source auto`: exact GT/reference if available, otherwise ray-caster. This is the default.
- `--mask-source exact`: exact GT/reference if available, otherwise ray-caster fallback.
- `--mask-source generated`: generate LoS/NLoS from topology + antenna height with the ray-caster fallback.
- `--mask-source provided`: use the HDF5/file LoS or NLoS mask as GT input.

For topology-only inputs that are not in the reference HDF5, the generator still
produces a mask using the ray-caster. GT is a temporary accelerator for dataset
validation, not a permanent requirement.

Rx height is fixed to `1.5 m` by the CKM calibration and the Try 78/79/80 prior
formulas.

## CLI

From `C:\TFG\CKMGenerator`:

```powershell
C:\TFG\.venv\Scripts\python.exe C:\TFG\CKMGenerator\ckm_doctor.py
```

The generator checks PyTorch first, then uses device priority `CUDA -> DirectML -> CPU`.
The doctor output includes CUDA VRAM when CUDA is available, Windows adapter VRAM
for DirectML/AMD when Windows exposes it, available system RAM, and a conservative
batch-size recommendation. DirectML is treated cautiously because it tends to use
more VRAM on this model. If PyTorch, DirectML, CUDA, the checkpoint, memory, or
model execution fail, the CLI exits with a specific error.

If DirectML is missing on Windows:

```powershell
C:\TFG\.venv\Scripts\python.exe -m pip install -r C:\TFG\CKMGenerator\requirements-directml.txt
```

```powershell
C:\TFG\.venv\Scripts\python.exe C:\TFG\CKMGenerator\ckm_generate.py `
  --input C:\TFG\TFGpractice\Datasets\CKM_Dataset_270326.h5 `
  --hdf5-city Abidjan `
  --hdf5-sample sample_00001 `
  --run-name abidjan_check
```

The same generator engine is used by both terminal and Streamlit.

For an image:

```powershell
C:\TFG\.venv\Scripts\python.exe C:\TFG\CKMGenerator\ckm_generate.py `
  --input C:\path\topology_map.png `
  --height 56.65 `
  --topology-max-m 90 `
  --run-name topology_demo
```

By default, CLI outputs go to `outputs/runs/<timestamp>_<run-name>`. Pass
`--out C:\path\chosen_folder` to write somewhere specific.
Use `--no-save-arrays` to skip numeric `.npz` matrices, or `--no-save-masks`
to skip LoS/NLoS PNG files.

Outputs are named with sample and antenna height:

- `masks/*_los_mask.png`
- `masks/*_nlos_mask.png`
- `priors/*_prior_path_loss.png`
- `priors/*_prior_path_loss_los.png`
- `priors/*_prior_path_loss_nlos.png`
- `priors/*_prior_delay_spread.png`
- `priors/*_prior_angular_spread.png`
- `predictions/*_pred_path_loss.png`
- `predictions/*_pred_delay_spread.png`
- `predictions/*_pred_angular_spread.png`
- `predictions/*_pred_joint.png`
- `arrays/*.npz`
- `metadata/*.json`

## Streamlit Interface

Install Streamlit if needed, then:

```powershell
C:\TFG\.venv\Scripts\python.exe -m streamlit run C:\TFG\CKMGenerator\ckm_app.py
```

The sidebar has an output selector: create a new timestamped run under
`outputs/runs`, reuse an existing output folder, or enter a custom path.
It also has export toggles for numeric arrays (`.npz`) and LoS/NLoS mask PNGs.

## Export Manual Test Inputs

To export 100 interface-ready samples spread across cities:

```powershell
C:\TFG\.venv\Scripts\python.exe C:\TFG\CKMGenerator\ckm_export_inputs.py `
  --hdf5 C:\TFG\TFGpractice\Datasets\CKM_Dataset_270326.h5 `
  --out C:\TFG\CKMGenerator\inputs_prueba `
  --n 100 `
  --los-mask `
  --matrices
```

This creates individual HDF5 inputs under `inputs_prueba\samples`, visual PNGs,
exact `.npz` matrices under `inputs_prueba\matrices`, and a manifest CSV/JSON.
Use `--no-los-mask`, `--no-matrices`, or `--no-png` to reduce what is exported.

## Mask Validation

To validate the real default generator path against CKM HDF5 reference masks:

```powershell
C:\TFG\.venv\Scripts\python.exe C:\TFG\CKMGenerator\scripts\validate_los_masks.py `
  --hdf5 C:\TFG\TFGpractice\Datasets\CKM_Dataset_270326.h5 `
  --limit 10
```

This writes `outputs/los_validation/los_validation.csv` and `summary.json`.
The validator exits with an error if any pixel mismatches, unless
`--allow-mismatch` is passed.

To test the exact-reference lookup as if the input HDF5 had no LoS/NLoS masks:

```powershell
C:\TFG\.venv\Scripts\python.exe C:\TFG\CKMGenerator\scripts\validate_los_masks.py `
  --hdf5 C:\TFG\TFGpractice\Datasets\CKM_Dataset_270326.h5 `
  --mode generator-auto-no-input-mask
```

The ray-caster fallback can be measured separately with `--mode raycast`, but it
is an approximate no-GT fallback and is not expected to match CKM bit-for-bit.

## Updating the Model Later

Replace `models/best_model.pt` with the newer compatible Try 80 checkpoint. If the architecture changes, update `config/generator_config.yaml` under `model.model_cfg`.

## Future Git LFS

When the repository is ready to version model checkpoints, enable Git LFS and
track weights explicitly instead of committing raw `.pt` files:

```powershell
git lfs install
git lfs track "models/*.pt"
git add .gitattributes models/best_model.pt
git commit -m "Track model checkpoint with Git LFS"
```

Until then, checkpoint files stay in `.gitignore`.
