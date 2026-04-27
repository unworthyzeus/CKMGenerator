# Try 80 - codigo final preliminar

Este directorio contiene una copia autocontenida del codigo de `Try 80`
fuera de `TFGpractice`.

Try 80 es el modelo conjunto final para:

- `path_loss`
- `delay_spread`
- `angular_spread`

El modelo usa priors no-DL congelados:

- Try 78 para `path_loss`
- Try 79 para `delay_spread` y `angular_spread`

La version final del checkpoint aun esta entrenando. De momento, el checkpoint
preliminar disponible es:

```powershell
C:\TFG\TFGpractice\cluster_outputs\TFGEightiethTry80\try80_joint_big\best_model.pt
```

Cuando este el modelo final, basta con cambiar esa ruta por el nuevo
`best_model.pt`.

## Estructura

- `src/`
  - dataset, priors, metricas, perdidas y arquitectura.
- `train_try80.py`
  - entrenamiento del modelo conjunto.
- `evaluate_try80.py`
  - evaluacion de un checkpoint entrenado.
- `experiments/try80_joint_big.yaml`
  - configuracion principal.
- `calibrations/`
  - JSONs congelados de Try 78 / Try 79 usados por defecto.
- `scripts/precompute_priors_hdf5.py`
  - precalculo opcional de priors a HDF5 auxiliar.
- `scripts/evaluate_frozen_priors.py`
  - evaluacion de los priors sin modelo neuronal.
- `scripts/recalibrate_priors/`
  - codigo fuente vendorizado para auditar o recalibrar los priors.

## Requisitos esperados

El codigo se ha probado localmente con:

```powershell
C:\TFG\.venv\Scripts\python.exe
```

Ese entorno tiene `torch` disponible. Si se usa otro entorno, debe incluir al
menos:

- `torch`
- `numpy`
- `h5py`
- `pyyaml`
- `scipy`
- `matplotlib`

## Dataset esperado

La configuracion por defecto apunta a:

```powershell
C:\TFG\TFGpractice\Datasets\CKM_Dataset_270326.h5
```

Si el dataset se mueve, editar:

```powershell
experiments\try80_joint_big.yaml
```

campo:

```yaml
data:
  hdf5_path: ...
```

## Evaluar el checkpoint preliminar

Desde `C:\TFG`:

```powershell
C:\TFG\.venv\Scripts\python.exe C:\TFG\Final_Models_TFG\TFGEightiethTry80_preliminary_code\evaluate_try80.py `
  --config C:\TFG\Final_Models_TFG\TFGEightiethTry80_preliminary_code\experiments\try80_joint_big.yaml `
  --checkpoint C:\TFG\TFGpractice\cluster_outputs\TFGEightiethTry80\try80_joint_big\best_model.pt `
  --split test
```

La salida se guarda bajo el `runtime.output_dir` definido en el YAML.

## Evaluar solo los priors congelados

Esto no usa checkpoint. Sirve para comprobar la calidad del baseline fisico /
estadistico que Try 80 mejora con residuales.

```powershell
C:\TFG\.venv\Scripts\python.exe C:\TFG\Final_Models_TFG\TFGEightiethTry80_preliminary_code\scripts\evaluate_frozen_priors.py `
  --config C:\TFG\Final_Models_TFG\TFGEightiethTry80_preliminary_code\experiments\try80_joint_big.yaml `
  --split test `
  --num-workers 0
```

## Precalcular priors

El YAML puede leer priors desde:

```yaml
data:
  precomputed_priors_hdf5_path: ...
```

Para regenerar ese cache:

```powershell
C:\TFG\.venv\Scripts\python.exe C:\TFG\Final_Models_TFG\TFGEightiethTry80_preliminary_code\scripts\precompute_priors_hdf5.py `
  --hdf5 C:\TFG\TFGpractice\Datasets\CKM_Dataset_270326.h5 `
  --out C:\TFG\TFGpractice\Datasets\try80_precomputed_priors.h5 `
  --try78-los-calibration-json C:\TFG\Final_Models_TFG\TFGEightiethTry80_preliminary_code\calibrations\try78_los_two_ray_calibration.json `
  --try78-nlos-calibration-json C:\TFG\Final_Models_TFG\TFGEightiethTry80_preliminary_code\calibrations\try78_nlos_regime_calibration.json `
  --try79-calibration-json C:\TFG\Final_Models_TFG\TFGEightiethTry80_preliminary_code\calibrations\try79_calibration.json
```

## Recalibrar priors

Normalmente no hace falta: Try 80 usa los JSONs congelados en `calibrations/`.
El codigo de recalibracion queda incluido para reproducibilidad.

LoS path-loss Try 78:

```powershell
C:\TFG\.venv\Scripts\python.exe C:\TFG\Final_Models_TFG\TFGEightiethTry80_preliminary_code\scripts\recalibrate_priors\try78_los_path_loss_prior.py `
  --hdf5 C:\TFG\TFGpractice\Datasets\CKM_Dataset_270326.h5 `
  --out-dir C:\TFG\scratch_try78_los_recalibration
```

Spreads Try 79:

```powershell
C:\TFG\.venv\Scripts\python.exe C:\TFG\Final_Models_TFG\TFGEightiethTry80_preliminary_code\scripts\recalibrate_priors\try79_spread_priors.py `
  --hdf5 C:\TFG\TFGpractice\Datasets\CKM_Dataset_270326.h5 `
  --out-dir C:\TFG\scratch_try79_spread_recalibration
```

Referencia/evaluacion del prior hibrido Try 78:

```powershell
C:\TFG\.venv\Scripts\python.exe C:\TFG\Final_Models_TFG\TFGEightiethTry80_preliminary_code\scripts\recalibrate_priors\try78_hybrid_path_loss_reference.py `
  --hdf5 C:\TFG\TFGpractice\Datasets\CKM_Dataset_270326.h5 `
  --out-dir C:\TFG\scratch_try78_hybrid_eval
```

Si se aceptan nuevas calibraciones, copiar los JSON generados a
`calibrations/` y actualizar los paths del bloque `prior:` en el YAML.

## Smoke test realizado

El 2026-04-23 se comprobo:

- smoke de calibracion LoS Try 78 con 8 muestras: OK
- smoke de calibracion Try 79 con 8 muestras y subsampling minimo: OK
- smoke de evaluacion de priors congelados Try 80 con 2 muestras: OK

Las smoke runs escribieron solo en `%TEMP%` y el directorio temporal fue
eliminado al terminar.
