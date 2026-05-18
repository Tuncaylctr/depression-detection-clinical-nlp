#!/usr/bin/env python3
"""
plot_learning_curves_llm
python3  / scripts/llm/plot_learning_curves_llm.py  .— Training curves for LLM LoRA experiments (Exp 7 & 8).

For each model (Gemma-3-12B, Qwen3.5-9B):
  • Finds the best run from llm_grid_search_results.csv (by val macro-F1)
  • Plots train loss + val macro-F1 per epoch for that run

Usage:
  python scripts/llm/plot_learning_curves_llm.py          # best run per model
  python scripts/llm/plot_learning_curves_llm.py --all    # every run
"""

import argparse
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parent.parent.parent
RESULTS_DIR = BASE_DIR / "results" / "llm"
FIGURES_DIR = RESULTS_DIR / "figures"
GRID_CSV    = RESULTS_DIR / "llm_grid_search_results.csv"

DPI     = 150
FIGSIZE = (5.0, 3.2)

MODELS = [
    ("google/gemma-3-12b-it", "Gemma 3 12B"),
    ("Qwen/Qwen3.5-9B",       "Qwen3.5 9B"),
]


def _style_ax(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors="black", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"   Saved → {path.resolve()}")


def _plot_run(run_name: str, label: str, df: pd.DataFrame, out_path: Path):
    epochs = df["epoch"].tolist()

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    fig.patch.set_facecolor("white")
    _style_ax(ax1)

    ax1.plot(epochs, df["train_loss"], "o-", color="#1f77b4",
             linewidth=1.5, markersize=4, label="Train loss")
    ax1.set_xlabel("Epoch", color="black", fontsize=7)
    ax1.set_ylabel("Train loss", color="#1f77b4", fontsize=7)
    ax1.tick_params(axis="y", labelcolor="#1f77b4", labelsize=6)

    ax2 = ax1.twinx()
    ax2.set_facecolor("white")
    ax2.plot(epochs, df["val_macro_f1"], "s--", color="#d62728",
             linewidth=1.5, markersize=4, label="Val macro-F1")
    ax2.set_ylabel("Val macro-F1", color="#d62728", fontsize=7)
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=6)
    ax2.set_ylim(0, 1)

    best_idx   = df["val_macro_f1"].idxmax()
    best_epoch = df.loc[best_idx, "epoch"]
    best_f1    = df.loc[best_idx, "val_macro_f1"]
    ax2.axvline(best_epoch, color="grey", linestyle=":", linewidth=1)
    ax2.annotate(
        f"best\nepoch {best_epoch}\nF1={best_f1:.3f}",
        xy=(best_epoch, best_f1),
        xytext=(best_epoch + 0.3, best_f1 - 0.10),
        fontsize=5, color="grey",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper right")

    ax1.set_title(f"Training Curve — {label}", color="black",
                  fontsize=8, fontweight="bold", pad=6)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    _save(fig, out_path)


def run(plot_all: bool = False):
    if not GRID_CSV.exists():
        print(f"Grid results not found: {GRID_CSV}")
        print("Run training first: python scripts/llm/train_llm.py --model_name ...")
        return

    grid = pd.read_csv(GRID_CSV)
    grid = grid.dropna(subset=["best_val_f1"])

    if grid.empty:
        print("No completed runs found in the grid results CSV.")
        return

    for model_id, model_label in MODELS:
        model_df = grid[grid["model"] == model_id]
        if model_df.empty:
            print(f"\n  SKIP {model_label} — no completed runs found")
            continue

        runs_to_plot = (
            model_df["run_name"].tolist()
            if plot_all
            else [model_df.loc[model_df["macro_f1"].idxmax(), "run_name"]]
        )

        for run_name in runs_to_plot:
            epoch_csv = RESULTS_DIR / f"epoch_metrics_{run_name}.csv"
            if not epoch_csv.exists():
                print(f"\n  SKIP {run_name} — epoch metrics not found: {epoch_csv}")
                continue

            df  = pd.read_csv(epoch_csv)
            row = model_df[model_df["run_name"] == run_name].iloc[0]
            label = f"{model_label}  lr={row['lr']}  r={int(row['lora_r'])}"
            print(f"\n  {label} …")
            _plot_run(run_name, label, df, FIGURES_DIR / f"tc_{run_name}.png")


def parse_args():
    p = argparse.ArgumentParser(description="Plot LLM training curves.")
    p.add_argument("--all", action="store_true",
                   help="Plot every run instead of just the best per model.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(plot_all=args.all)
    print("\n Done.")
