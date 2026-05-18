#!/usr/bin/env python3
"""
plot_learning_curves.py — Learning / training curves for all models.

Classical models  : sklearn learning_curve — macro-F1 vs training set size,
                    using the best hyperparameters from results_classical.csv.
Transformer models: per-epoch train loss + val macro-F1, loaded from
                    epoch_metrics_<model>.csv files saved by train_sentence_transformer.py.

Usage:
  python scripts/plot_learning_curves.py          # both classical and transformer
  python scripts/plot_learning_curves.py --classical
  python scripts/plot_learning_curves.py --transformers
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts" / "classical"))
sys.path.insert(0, str(BASE_DIR / "scripts" / "data"))
sys.path.insert(0, str(BASE_DIR))

LABEL_DIR = BASE_DIR / "data" / "labels"
TRANS_DIR = BASE_DIR / "data" / "transcripts"

#  Plot style 

DPI        = 150
FIGSIZE_LC = (4.5, 3.2)   # classical learning curve
FIGSIZE_TC = (5.0, 3.2)   # transformer training curve (two y-axes)

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


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_train_pool():
    from scripts.data.preprocess   import load_participant_transcripts
    from scripts.data.clean_labels import clean_labels

    labels_df = clean_labels(
        label_dir=str(LABEL_DIR),
        output_dir=str(BASE_DIR / "data" / "processed"),
        output_filename="labels.csv",
        verbose=False,
    )
    dataset = load_participant_transcripts(TRANS_DIR, labels_df, verbose=False)

    train_ids = set(pd.read_csv(LABEL_DIR / "train_split_Depression_AVEC2017.csv")
                    .iloc[:, 0].astype(int))
    dev_ids   = set(pd.read_csv(LABEL_DIR / "dev_split_Depression_AVEC2017.csv")
                    .iloc[:, 0].astype(int))
    pool_ids  = train_ids | dev_ids

    pool_df = dataset[dataset["participant_id"].isin(pool_ids)].reset_index(drop=True)
    X = pool_df["text"].tolist()
    y = pool_df["PHQ_Binary"].tolist()
    print(f"   Training pool: {len(pool_df)} participants  "
          f"(dep={sum(y)}, non-dep={len(y)-sum(y)})")
    return X, y


#  Classical learning curves 

def _build_classical_pipelines():
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model  import LogisticRegression
    from sklearn.svm           import LinearSVC
    from sklearn.ensemble      import RandomForestClassifier

    tfidf_base = dict(
        sublinear_tf=True, strip_accents="unicode",
        analyzer="word", stop_words="english", min_df=2,
        max_features=5000, ngram_range=(1, 2),
    )

    return [
        (
            "Logistic Regression",
            Pipeline([
                ("tfidf", TfidfVectorizer(**tfidf_base)),
                ("clf",   LogisticRegression(C=0.01, class_weight="balanced",
                                             max_iter=2000, solver="lbfgs",
                                             random_state=42)),
            ]),
        ),
        (
            "LinearSVC (SVM)",
            Pipeline([
                ("tfidf", TfidfVectorizer(**tfidf_base)),
                ("clf",   LinearSVC(C=0.01, class_weight="balanced",
                                    max_iter=5000, random_state=42)),
            ]),
        ),
        (
            "Random Forest",
            Pipeline([
                ("tfidf", TfidfVectorizer(**tfidf_base)),
                ("clf",   RandomForestClassifier(n_estimators=100, max_depth=None,
                                                 class_weight="balanced",
                                                 random_state=42, n_jobs=-1)),
            ]),
        ),
    ]


def _plot_classical_lc(name, train_sizes_abs, train_scores, val_scores, out_path):
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=FIGSIZE_LC)
    fig.patch.set_facecolor("white")
    _style_ax(ax)

    ax.plot(train_sizes_abs, train_mean, "o-", color="#1f77b4", linewidth=1.5,
            markersize=4, label="Training score")
    ax.fill_between(train_sizes_abs,
                    train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#1f77b4")

    ax.plot(train_sizes_abs, val_mean, "s--", color="#d62728", linewidth=1.5,
            markersize=4, label="Cross-val score")
    ax.fill_between(train_sizes_abs,
                    val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="#d62728")

    ax.set_xlabel("Training set size", color="black", fontsize=7)
    ax.set_ylabel("Macro-F1", color="black", fontsize=7)
    ax.set_title(f"Learning Curve — {name}", color="black",
                 fontsize=8, fontweight="bold", pad=6)
    ax.legend(fontsize=6, framealpha=0.8)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    _save(fig, out_path)


def run_classical():
    from sklearn.model_selection import learning_curve

    print("\n── Classical learning curves ──────────────────────────────")
    X, y = _load_train_pool()
    pipelines  = _build_classical_pipelines()
    figures_dir = BASE_DIR / "results" / "classical" / "figures"

    for name, pipe in pipelines:
        print(f"\n  {name} …")
        train_sizes, train_scores, val_scores = learning_curve(
            pipe, X, y,
            cv=5,
            scoring="f1_macro",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
            shuffle=True,
            random_state=42,
        )
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        _plot_classical_lc(
            name, train_sizes, train_scores, val_scores,
            figures_dir / f"lc_{safe}.png",
        )


#  Transformer training curves 

# (model_label, epoch_csv_path, figures_output_dir)
TRANSFORMER_TARGETS = [
    (
        "mBERT (bert_lr2e-05_wr0.0_dwAuto)",
        BASE_DIR / "results" / "sentence_transformers" / "epoch_metrics_bert-base-multilingual-cased.csv",
        BASE_DIR / "results" / "sentence_transformers" / "figures",
        "tc_bert_lr2e-05_wr0.0_dwauto",
    ),
    (
        "XLM-RoBERTa (xlm_lr5e-06_wr0.0_dw4.0)",
        BASE_DIR / "results" / "sentence_transformers" / "epoch_metrics_xlm-roberta-base.csv",
        BASE_DIR / "results" / "sentence_transformers" / "figures",
        "tc_xlm_lr5e-06_wr0.0_dw4.0",
    ),
    (
        "mmBERT-base (jhu_lr2e-05_wr0.0_dwAuto)",
        BASE_DIR / "results" / "sentence_transformers" / "epoch_metrics_jhu_lr2e-05_wr0.0_dwauto.csv",
        BASE_DIR / "results" / "sentence_transformers" / "figures",
        "tc_mmbert-base",
    ),
]


def _plot_transformer_tc(label, df, out_path):
    epochs = df["epoch"].tolist()

    fig, ax1 = plt.subplots(figsize=FIGSIZE_TC)
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

    # Mark best val epoch
    best_epoch = df.loc[df["val_macro_f1"].idxmax(), "epoch"]
    best_f1    = df["val_macro_f1"].max()
    ax2.axvline(best_epoch, color="grey", linestyle=":", linewidth=1)
    ax2.annotate(f"best\nepoch {best_epoch}\nF1={best_f1:.3f}",
                 xy=(best_epoch, best_f1),
                 xytext=(best_epoch + 0.4, best_f1 - 0.08),
                 fontsize=5, color="grey")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper right")

    ax1.set_title(f"Training Curve — {label}", color="black",
                  fontsize=8, fontweight="bold", pad=6)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    _save(fig, out_path)


def run_transformers():
    print("\n── Transformer training curves ────────────────────────────")
    for label, csv_path, figures_dir, stem in TRANSFORMER_TARGETS:
        if not csv_path.exists():
            print(f"\n  SKIP {label} — epoch metrics not found: {csv_path}")
            print(f"       Re-run training with the updated train_sentence_transformer.py")
            continue
        print(f"\n  {label} …")
        df = pd.read_csv(csv_path)
        _plot_transformer_tc(label, df, figures_dir / f"{stem}.png")


# CLI 

def parse_args():
    p = argparse.ArgumentParser(description="Generate learning / training curves.")
    p.add_argument("--classical",    action="store_true")
    p.add_argument("--transformers", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    both = not args.classical and not args.transformers
    if args.classical or both:
        run_classical()
    if args.transformers or both:
        run_transformers()
    print("\n Done.")


if __name__ == "__main__":
    main()
