"""Generate Try 80 inspection panels for the thesis appendix.

Each panel is rendered twice, in two layouts that share the same content:

  - 3x6 (wide):  rows = tasks (path loss, delay spread, angular spread)
                 cols = ground truth, frozen prior, Try 80 output, context,
                        prior minus ground truth (prior error),
                        Try 80 minus prior (model correction)

  - 6x3 (tall):  rows = the six stages listed above
                 cols = tasks

The two right-most columns describe the prior-anchored residual story:
column 5 shows what the frozen prior got wrong, and column 6 shows the signed
correction the Try 80 residual head applied on top of the prior.

Sample selection: the script only keeps candidates for which the Try 80
prediction beats the frozen prior on all three outputs simultaneously (path
loss, delay spread, angular spread). Each macro-expert x antenna-bin bucket
then picks the candidate with the best mean model RMSE.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import torch_directml
except ImportError:  # pragma: no cover - depends on local Windows runtime
    torch_directml = None


HERE = Path(__file__).resolve()
TRY80_ROOT = HERE.parents[1]
PRACTICE_ROOT = TRY80_ROOT.parent
REPO_ROOT = PRACTICE_ROOT.parent
if str(TRY80_ROOT) not in sys.path:
    sys.path.insert(0, str(TRY80_ROOT))

from evaluate_try80 import build_data_cfg, build_model_cfg  # noqa: E402
from src.config_try80 import Try80Cfg  # noqa: E402
from src.data_utils import HeightEmbedding, build_joint_datasets, read_field  # noqa: E402
from src.metrics_try80 import TASKS, inverse_transform, transform_target  # noqa: E402
from src.model_try80 import Try80Model  # noqa: E402
from src.priors_try80 import ant_bin, classify_topology, macro_topology_class  # noqa: E402


TASK_LABELS = {
    "path_loss": ("Path loss", "dB", "viridis"),
    "delay_spread": ("Delay spread", "ns", "viridis"),
    "angular_spread": ("Angular spread", "deg", "plasma"),
}
EXPERT_ORDER = (
    "open|low_ant",
    "open|mid_ant",
    "open|high_ant",
    "mixed|low_ant",
    "mixed|mid_ant",
    "mixed|high_ant",
    "dense|low_ant",
    "dense|mid_ant",
    "dense|high_ant",
)


@dataclass
class Candidate:
    idx: int
    city: str
    sample: str
    h_tx: float
    topology_class_6: str
    expert: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=TRY80_ROOT / "experiments" / "try80_joint_big.yaml",
        help="Try 80 YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PRACTICE_ROOT / "cluster_outputs" / "TFGEightiethTry80" / "try80_joint_big" / "best_model.pt",
        help="Try 80 checkpoint.",
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "FINAL_THESIS" / "TFG" / "img" / "thesis_figures" / "try80_appendix_panels",
    )
    parser.add_argument("--max-experts", type=int, default=6)
    parser.add_argument("--candidates-per-expert", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument(
        "--require-improvement",
        action="store_true",
        default=True,
        help="Only keep candidates where Try 80 beats the prior on all three outputs.",
    )
    parser.add_argument(
        "--no-require-improvement",
        dest="require_improvement",
        action="store_false",
        help="Disable the all-tasks-improved filter.",
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("3x6", "6x3"),
        default=("3x6", "6x3"),
        help="Which layouts to render (default: both).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "directml", "cuda", "cpu"),
        default="auto",
        help="Inference device. Auto prefers CUDA, then DirectML, then CPU.",
    )
    return parser.parse_args()


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_")


def load_raw_sample(hdf5_path: Path, city: str, sample: str) -> Dict[str, np.ndarray | float]:
    with h5py.File(str(hdf5_path), "r") as handle:
        grp = handle[city][sample]
        topology = np.asarray(grp["topology_map"][...], dtype=np.float32)
        return {
            "topology": topology,
            "los_mask_raw": np.asarray(grp["los_mask"][...], dtype=np.float32),
            "path_loss": np.asarray(grp["path_loss"][...], dtype=np.float32),
            "delay_spread": read_field(grp, "delay_spread"),
            "angular_spread": read_field(grp, "angular_spread"),
            "h_tx": float(np.asarray(grp["uav_height"][...]).reshape(-1)[0]),
        }


def collect_candidates(ds, hdf5_path: Path, per_expert: int, max_experts: int) -> List[Candidate]:
    buckets: Dict[str, List[Candidate]] = {key: [] for key in EXPERT_ORDER}
    wanted = set(EXPERT_ORDER)
    for idx, (city, sample) in enumerate(ds._refs):  # noqa: SLF001 - script-level inspection
        if all(len(v) >= per_expert for v in buckets.values()):
            break
        raw = load_raw_sample(hdf5_path, city, sample)
        topology = raw["topology"]
        h_tx = float(raw["h_tx"])
        cls6 = classify_topology(topology)
        expert = f"{macro_topology_class(cls6)}|{ant_bin(h_tx)}"
        if expert in wanted and len(buckets[expert]) < per_expert:
            buckets[expert].append(Candidate(idx, city, sample, h_tx, cls6, expert))

    selected: List[Candidate] = []
    for expert in EXPERT_ORDER:
        selected.extend(buckets[expert])
        if len({cand.expert for cand in selected}) >= max_experts:
            break
    return selected


def tensor_batch(item: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            out[key] = value.unsqueeze(0).to(device)
        else:
            out[key] = [value]
    return out


def resolve_device(name: str):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if name == "directml":
        if torch_directml is None:
            raise RuntimeError("DirectML was requested, but torch_directml is not installed.")
        return torch_directml.device()
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch_directml is not None:
        return torch_directml.device()
    return torch.device("cpu")


def rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0.5
    if not np.any(valid):
        return float("nan")
    diff = pred[valid] - target[valid]
    return float(np.sqrt(np.mean(diff * diff)))


def infer_one(model, height_embed, ds, cand: Candidate, device: torch.device) -> Dict[str, object]:
    item = ds[cand.idx]
    batch = tensor_batch(item, device)
    priors_native = {task: batch[f"{task}_prior"] for task in TASKS}
    priors_trans = {task: transform_target(task, priors_native[task]) for task in TASKS}
    with torch.no_grad():
        outputs = model(batch["inputs"], height_embed(batch["antenna_height_m"]), priors_trans)
        preds_native = {task: inverse_transform(task, outputs[task]["pred_trans"]) for task in TASKS}

    arrays: Dict[str, Dict[str, np.ndarray]] = {"target": {}, "prior": {}, "pred": {}, "mask": {}}
    metrics: Dict[str, Dict[str, float]] = {}
    for task in TASKS:
        target = item[f"{task}_target"].squeeze(0).numpy()
        prior = item[f"{task}_prior"].squeeze(0).numpy()
        pred = preds_native[task][0, 0].detach().cpu().numpy()
        mask = item[f"{task}_mask"].squeeze(0).numpy()
        arrays["target"][task] = target
        arrays["prior"][task] = prior
        arrays["pred"][task] = pred
        arrays["mask"][task] = mask
        metrics[task] = {
            "model_rmse": rmse(pred, target, mask),
            "prior_rmse": rmse(prior, target, mask),
        }

    raw = load_raw_sample(ds.cfg.hdf5_path, cand.city, cand.sample)
    ground = (raw["topology"] == 0.0).astype(np.float32)
    los = np.asarray(item["los_mask"].squeeze(0).numpy(), dtype=np.float32)
    nlos = np.asarray(item["nlos_mask"].squeeze(0).numpy(), dtype=np.float32)

    return {
        "candidate": cand,
        "arrays": arrays,
        "raw": raw,
        "ground": ground,
        "los": los,
        "nlos": nlos,
        "metrics": metrics,
    }


def masked(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0.5, arr, np.nan)


def robust_range(values: Iterable[np.ndarray], masks: Iterable[np.ndarray], lo=1.0, hi=99.0) -> Tuple[float, float]:
    flat = []
    for arr, mask in zip(values, masks):
        valid = arr[mask > 0.5]
        valid = valid[np.isfinite(valid)]
        if valid.size:
            flat.append(valid)
    if not flat:
        return 0.0, 1.0
    vec = np.concatenate(flat)
    vmin = float(np.percentile(vec, lo))
    vmax = float(np.percentile(vec, hi))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def symmetric_range(values: Iterable[np.ndarray], masks: Iterable[np.ndarray]) -> float:
    flat = []
    for arr, mask in zip(values, masks):
        valid = arr[mask > 0.5]
        valid = valid[np.isfinite(valid)]
        if valid.size:
            flat.append(valid)
    if not flat:
        return 1.0
    vec = np.concatenate(flat)
    vmax = max(abs(float(np.percentile(vec, 1.0))), abs(float(np.percentile(vec, 99.0))))
    return max(vmax, 1.0e-6)


def add_map(ax, image, title: str, cmap: str, vmin=None, vmax=None, colorbar=True):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color="#f2f2f2")
    im = ax.imshow(image, cmap=cm, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    return im


STAGE_ORDER = ("gt", "prior", "pred", "context", "prior_err", "model_corr")


def _build_stage_plan(record: Dict[str, object]) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Return a stage -> task -> {image, title, cmap, vmin, vmax} plan.

    Centralising the plan makes the 3x6 and 6x3 renderers share identical
    content: the only difference between them is the axes orientation.
    """
    arrays = record["arrays"]
    raw = record["raw"]
    metrics = record["metrics"]

    task_ranges = {
        task: robust_range(
            [arrays["target"][task], arrays["prior"][task], arrays["pred"][task]],
            [arrays["mask"][task]] * 3,
        )
        for task in TASKS
    }
    prior_err_ranges = {
        task: symmetric_range(
            [arrays["prior"][task] - arrays["target"][task]], [arrays["mask"][task]]
        )
        for task in TASKS
    }
    model_corr_ranges = {
        task: symmetric_range(
            [arrays["pred"][task] - arrays["prior"][task]], [arrays["mask"][task]]
        )
        for task in TASKS
    }
    context_maps = {
        "path_loss": (raw["topology"], "Topology height (m)", "terrain", None, None),
        "delay_spread": (
            record["nlos"],
            f"NLoS support ({100.0 * np.mean(record['nlos']):.1f}%)",
            "gray",
            0,
            1,
        ),
        "angular_spread": (
            record["los"],
            f"LoS support ({100.0 * np.mean(record['los']):.1f}%)",
            "gray",
            0,
            1,
        ),
    }

    plan: Dict[str, Dict[str, Dict[str, object]]] = {s: {} for s in STAGE_ORDER}
    for task in TASKS:
        label, unit, cmap = TASK_LABELS[task]
        mask = arrays["mask"][task]
        vmin, vmax = task_ranges[task]

        plan["gt"][task] = {
            "image": masked(arrays["target"][task], mask),
            "title": f"Ground truth\n{label} ({unit})",
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
        }
        plan["prior"][task] = {
            "image": masked(arrays["prior"][task], mask),
            "title": (
                f"Frozen prior (Try 78/79)\n{label} - RMSE "
                f"{metrics[task]['prior_rmse']:.2f} {unit}"
            ),
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
        }
        plan["pred"][task] = {
            "image": masked(arrays["pred"][task], mask),
            "title": (
                f"Try 80 prediction\n{label} - RMSE "
                f"{metrics[task]['model_rmse']:.2f} {unit}"
            ),
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
        }
        ctx, ctx_title, ctx_cmap, ctx_vmin, ctx_vmax = context_maps[task]
        plan["context"][task] = {
            "image": ctx,
            "title": f"Context\n{ctx_title}",
            "cmap": ctx_cmap,
            "vmin": ctx_vmin,
            "vmax": ctx_vmax,
        }

        prior_err = arrays["prior"][task] - arrays["target"][task]
        plan["prior_err"][task] = {
            "image": masked(prior_err, mask),
            "title": f"Prior error\n(Prior - GT) {label} ({unit})",
            "cmap": "seismic",
            "vmin": -prior_err_ranges[task],
            "vmax": prior_err_ranges[task],
        }
        model_corr = arrays["pred"][task] - arrays["prior"][task]
        plan["model_corr"][task] = {
            "image": masked(model_corr, mask),
            "title": f"Model correction\n(Try 80 - Prior) {label} ({unit})",
            "cmap": "coolwarm",
            "vmin": -model_corr_ranges[task],
            "vmax": model_corr_ranges[task],
        }

    return plan


def _render_layout(plan, cand, layout: str, out_path: Path, dpi: int) -> None:
    if layout == "3x6":
        fig, axes = plt.subplots(3, 6, figsize=(24, 11), constrained_layout=True)
        for row, task in enumerate(TASKS):
            for col, stage in enumerate(STAGE_ORDER):
                spec = plan[stage][task]
                add_map(
                    axes[row, col],
                    spec["image"],
                    spec["title"],
                    spec["cmap"],
                    spec["vmin"],
                    spec["vmax"],
                )
    elif layout == "6x3":
        fig, axes = plt.subplots(6, 3, figsize=(12, 22), constrained_layout=True)
        for row, stage in enumerate(STAGE_ORDER):
            for col, task in enumerate(TASKS):
                spec = plan[stage][task]
                add_map(
                    axes[row, col],
                    spec["image"],
                    spec["title"],
                    spec["cmap"],
                    spec["vmin"],
                    spec["vmax"],
                )
    else:
        raise ValueError(f"Unknown layout {layout!r}")

    fig.suptitle(
        f"{cand.city} / {cand.sample}    {cand.topology_class_6}    {cand.expert}    "
        f"h_tx={cand.h_tx:.1f} m",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def render_panel(
    record: Dict[str, object],
    out_path_base: Path,
    dpi: int,
    layouts: Iterable[str] = ("3x6", "6x3"),
) -> Dict[str, object]:
    cand: Candidate = record["candidate"]
    metrics = record["metrics"]
    plan = _build_stage_plan(record)

    files: Dict[str, str] = {}
    for layout in layouts:
        if layout == "3x6":
            out_path = out_path_base
        else:
            out_path = out_path_base.with_name(
                out_path_base.stem + f"_{layout}" + out_path_base.suffix
            )
        _render_layout(plan, cand, layout, out_path, dpi)
        files[layout] = out_path.name

    score = float(np.nanmean([metrics[task]["model_rmse"] for task in TASKS]))
    return {
        "file": files.get("3x6", next(iter(files.values()))),
        "files": files,
        "city": cand.city,
        "sample": cand.sample,
        "h_tx_m": cand.h_tx,
        "topology_class_6": cand.topology_class_6,
        "expert": cand.expert,
        "score_mean_model_rmse": score,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Try80Cfg.load(args.config)
    data_cfg = build_data_cfg(cfg)
    _, val_ds, test_ds = build_joint_datasets(data_cfg)
    ds = val_ds if args.split == "val" else test_ds
    candidates = collect_candidates(ds, data_cfg.hdf5_path, args.candidates_per_expert, args.max_experts)
    if not candidates:
        raise RuntimeError("No panel candidates were found.")

    device = resolve_device(args.device)
    model = Try80Model(build_model_cfg(cfg)).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    model.to(device)
    model.eval()
    height_embed = HeightEmbedding()

    by_expert: Dict[str, List[Dict[str, object]]] = {}
    skipped_no_improvement = 0
    for cand in candidates:
        print(f"[try80-panels] scoring {cand.expert} {cand.city}/{cand.sample}")
        record = infer_one(model, height_embed, ds, cand, device)
        metrics = record["metrics"]
        improves_all = all(
            metrics[task]["model_rmse"] < metrics[task]["prior_rmse"] for task in TASKS
        )
        if args.require_improvement and not improves_all:
            skipped_no_improvement += 1
            continue
        score = float(np.nanmean([metrics[task]["model_rmse"] for task in TASKS]))
        by_expert.setdefault(cand.expert, []).append({"record": record, "score": score})

    if skipped_no_improvement:
        print(
            f"[try80-panels] skipped {skipped_no_improvement} candidates that did not "
            "beat the prior on all three outputs."
        )

    rendered: List[Dict[str, object]] = []
    for expert in EXPERT_ORDER:
        bucket = by_expert.get(expert)
        if not bucket:
            print(f"[try80-panels] no improving candidate for expert {expert}")
            continue
        chosen = sorted(bucket, key=lambda row: row["score"])[0]["record"]
        cand = chosen["candidate"]
        safe_expert = slug(cand.expert.replace("|", "_"))
        out_path = args.out_dir / f"try80_panel_{safe_expert}_{slug(cand.city)}_{slug(cand.sample)}.png"
        print(f"[try80-panels] rendering {out_path.name} ({', '.join(args.layouts)})")
        rendered.append(render_panel(chosen, out_path, args.dpi, layouts=args.layouts))

    main_row = sorted(rendered, key=lambda row: row["score_mean_model_rmse"])[0]
    main_src = args.out_dir / str(main_row["files"].get("6x3", main_row["file"]))
    main_dst = args.out_dir / "try80_main_panel.png"
    shutil.copyfile(main_src, main_dst)

    manifest = {
        "split": args.split,
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "main_panel_layout": "6x3" if "6x3" in main_row["files"] else "3x6",
        "main_panel": main_dst.name,
        "panels": rendered,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[try80-panels] wrote {len(rendered)} panels to {args.out_dir}")
    print(f"[try80-panels] main panel: {main_dst.name}")


if __name__ == "__main__":
    main()
