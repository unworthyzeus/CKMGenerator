# Try 80 Masking Equivalence

This note records the two masking implementations used in Try 80 and explains
why they are equivalent for the specific question "are building pixels
excluded?".

## Path Loss Mask

In [src/data_utils.py](/c:/TFG/TFGpractice/TFGEightiethTry80/src/data_utils.py#L129),
path loss uses:

```python
def _valid_path_loss_mask(
    grp: h5py.Group,
    path_loss: np.ndarray,
    ground: np.ndarray,
    no_data_mask_column: Optional[str],
    derive_no_data_from_non_ground: bool,
) -> np.ndarray:
    valid = np.isfinite(path_loss) & (path_loss >= PATH_LOSS_MIN_DB)
    if no_data_mask_column:
        key = str(no_data_mask_column).strip()
        if key and key in grp:
            no_data = np.asarray(grp[key][...], dtype=np.float32) > 0.5
            valid &= ~no_data
    if derive_no_data_from_non_ground:
        valid &= ground
    return valid
```

With the default Try 80 config in
[experiments/try80_joint_big.yaml](/c:/TFG/TFGpractice/TFGEightiethTry80/experiments/try80_joint_big.yaml),
we have:

```yaml
path_loss_no_data_mask_column: path_loss_no_data_mask
derive_no_data_from_non_ground: true
```

So the path-loss mask includes the clause:

```python
valid &= ground
```

which removes every non-ground pixel, i.e. every pixel where `topology != 0`.

## Delay / Angular Mask

In [src/data_utils.py](/c:/TFG/TFGpractice/TFGEightiethTry80/src/data_utils.py#L147),
delay spread and angular spread use:

```python
def _valid_nonnegative_mask(target: np.ndarray, ground: np.ndarray) -> np.ndarray:
    return ground & np.isfinite(target) & (target >= 0.0)
```

This also contains the same building-exclusion clause:

```python
ground
```

and `ground` is defined in
[src/data_utils.py](/c:/TFG/TFGpractice/TFGEightiethTry80/src/data_utils.py#L201)
as:

```python
ground = topology == 0.0
```

So delay spread and angular spread also remove every non-ground pixel, i.e.
every pixel where `topology != 0`.

## Why They Are The Same

The two implementations are not textually identical, but they are equivalent
with respect to building suppression:

- Path loss excludes buildings via `valid &= ground`.
- Delay spread excludes buildings via `ground & ...`.
- Angular spread excludes buildings via `ground & ...`.

Since `ground` means `topology == 0.0`, all three tasks reject building pixels
in the same way.

In logical form:

```text
building pixel  <=>  topology != 0
ground pixel    <=>  topology == 0
```

Therefore:

- if `topology != 0`, then `ground == False`
- path loss becomes invalid because `valid &= ground`
- delay spread becomes invalid because `ground & ... == False`
- angular spread becomes invalid because `ground & ... == False`

## Important Nuance

Path loss has one extra restriction that the other two tasks do not use:

- the optional dataset column `path_loss_no_data_mask`

That extra clause can remove additional pixels from the path-loss target, but it
does not change the building rule itself. The building rule is still the same
for all three tasks: only `ground` pixels are allowed.

## Consequence For Predictions

The dataset also builds LoS / NLoS masks only on ground:

```python
ground_f = ground.astype(np.float32)
los = los_mask * ground_f
nlos = (1.0 - los_mask) * ground_f
```

and the model prediction is blended only through those region masks in
[src/model_try80.py](/c:/TFG/TFGpractice/TFGEightiethTry80/src/model_try80.py#L263).
So building pixels are not only excluded from the losses and metrics, they are
also suppressed in the final decoded prediction maps.
