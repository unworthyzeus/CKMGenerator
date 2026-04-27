"""Plot Try 80 training history and model-vs-prior validation errors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


TASKS = ("path_loss", "delay_spread", "angular_spread")
SCOPES = ("overall", "los", "nlos")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    args = parser.parse_args()

    history = json.loads(args.history.read_text(encoding="utf-8"))
    if not history:
        raise SystemExit("history is empty")

    epochs = [int(row["epoch"]) for row in history]
    out_dir = args.history.parent

    _plot_losses(history, epochs, out_dir / "history_losses.png")
    _plot_metric_errors(history, epochs, out_dir / "history_metrics_vs_prior.png")
    print(f"[plot-history] plots saved under {out_dir}")


def _plot_losses(history, epochs, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    keys = ("total", "map_nll", "dist_kl", "prior_guard")
    for ax, key in zip(axes, keys):
        ax.plot(epochs, [row["train"][key] for row in history], label="train")
        ax.plot(epochs, [row["val"]["losses"][key] for row in history], label="val")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_metric_errors(history, epochs, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for row_idx, task in enumerate(TASKS):
        for col_idx, scope in enumerate(SCOPES):
            ax = axes[row_idx, col_idx]
            model_vals = [row["val"]["metrics"]["model"]["aggregate"][task][scope]["rmse_pw"] for row in history]
            prior_vals = [row["val"]["metrics"]["prior"]["aggregate"][task][scope]["rmse_pw"] for row in history]
            gain_vals = [p - m for p, m in zip(prior_vals, model_vals)]
            ax.plot(epochs, model_vals, label="model")
            ax.plot(epochs, prior_vals, label="prior", linestyle="--")
            ax.plot(epochs, gain_vals, label="prior-model", linestyle=":")
            ax.set_title(f"{task} / {scope}")
            ax.set_xlabel("epoch")
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
