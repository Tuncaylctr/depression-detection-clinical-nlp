#!/usr/bin/env python3
"""
evaluate.py — Evaluation utilities for depression detection models.

Provides two public functions consumed by train_classical.py (and any future
LLM-based pipeline):

  • print_metrics(model_name, y_true, y_pred)
      Prints a formatted scikit-learn classification report plus a text
      confusion matrix to stdout.

  • save_confusion_matrix_plot(model_name, y_true, y_pred, output_dir)
      Saves a publication-quality confusion matrix as a PNG figure.

Reference
---------
Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
Journal of Machine Learning Research, 12, 2825–2830.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ["Non-depressed (0)", "Depressed (1)"]
OUTPUT_DIR_DEFAULT = "results/figures"


# ---------------------------------------------------------------------------
# 1. Console metrics printer
# ---------------------------------------------------------------------------

def print_metrics(
    model_name: str,
    y_true,
    y_pred,
    *,
    digits: int = 4,
) -> None:
    """
    Print a detailed classification report and confusion matrix to stdout.

    Parameters
    ----------
    model_name : Human-readable name of the model (used as heading).
    y_true     : Ground-truth binary labels (array-like, values 0/1).
    y_pred     : Predicted binary labels (array-like, values 0/1).
    digits     : Number of decimal places in the classification report.
    """
    separator = "─" * 60

    print(f"\n{separator}")
    print(f"    {model_name}")
    print(separator)

    # ── Classification report ──────────────────────────────────────────────
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Non-depressed", "Depressed"],
        digits=digits,
        zero_division=0,
    )
    print(report)

    # ── Confusion matrix (text) ────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion matrix:")
    print(f"                Predicted 0   Predicted 1")
    print(f"  Actual 0      {cm[0, 0]:>9d}   {cm[0, 1]:>9d}")
    print(f"  Actual 1      {cm[1, 0]:>9d}   {cm[1, 1]:>9d}")
    print()

    # Quick summary line
    tn, fp, fn, tp = cm.ravel()
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(separator)


# ---------------------------------------------------------------------------
# 2. Confusion matrix PNG saver
# ---------------------------------------------------------------------------

def save_confusion_matrix_plot(
    model_name: str,
    y_true,
    y_pred,
    *,
    output_dir: str = OUTPUT_DIR_DEFAULT,
    normalize: bool = True,
    colormap: str = "Blues",
    dpi: int = 150,
) -> Path:
    """
    Save a confusion matrix figure as a PNG file.

    Parameters
    ----------
    model_name  : Used as the figure title and to derive the output filename.
    y_true      : Ground-truth binary labels.
    y_pred      : Predicted binary labels.
    output_dir  : Directory where the PNG will be saved.
    normalize   : If True, show normalised (proportional) counts; raw counts
                  are shown alongside in parentheses.
    colormap    : Matplotlib colormap name.
    dpi         : Image resolution in dots per inch.

    Returns
    -------
    Path to the saved PNG file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_raw  = confusion_matrix(y_true, y_pred)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Heatmap
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=colormap, vmin=0.0, vmax=1.0)

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    cbar.ax.tick_params(colors="white", labelsize=8)
    cbar.outline.set_edgecolor("#333344")

    # Tick labels
    tick_labels = ["Non-dep (0)", "Depressed (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels, color="white", fontsize=9)
    ax.set_yticklabels(tick_labels, color="white", fontsize=9)

    # Annotate cells with normalised % and raw count
    thresh = cm_norm.max() / 2.0
    for i in range(2):
        for j in range(2):
            pct = cm_norm[i, j]
            raw = cm_raw[i, j]
            color = "white" if pct < thresh else "#0f1117"
            ax.text(
                j, i,
                f"{pct:.1%}\n(n={raw})",
                ha="center", va="center",
                color=color,
                fontsize=10, fontweight="bold",
            )

    # Labels & title
    ax.set_xlabel("Predicted label", color="white", labelpad=8)
    ax.set_ylabel("True label", color="white", labelpad=8)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    title = f"Confusion Matrix — {model_name}"
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)

    fig.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────
    safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
    out_file  = out_dir / f"cm_{safe_name}.png"
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"    Confusion matrix saved → {out_file.resolve()}")
    return out_file


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate a small imbalanced prediction scenario
    rng    = np.random.default_rng(42)
    y_true = np.array([0] * 20 + [1] * 9)
    y_pred = rng.choice([0, 1], size=len(y_true), p=[0.65, 0.35])

    print_metrics("Demo Model", y_true, y_pred)
    out = save_confusion_matrix_plot("Demo Model", y_true, y_pred, output_dir="/tmp/cm_test")
    print(f"Saved to: {out}")
