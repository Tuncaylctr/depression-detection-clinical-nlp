#!/usr/bin/env python3
"""
train_sentence_transformer.py — Fine-tune a multilingual transformer encoder
for DAIC-WOZ binary depression detection (Experiments 4 & 5).

Supported models (change only --model_name):
  Experiment 4 : bert-base-multilingual-cased   (mBERT)
  Experiment 5 : xlm-roberta-base               (XLM-RoBERTa)

Pipeline overview
-----------------
1.  Clean labels via clean_labels.py
2.  Load participant transcripts via preprocess.py
3.  Partition using the official DAIC-WOZ split files:
      Training pool = train_split  +  dev_split  participants
      Test set      = full_test_split participants
4.  Hold out 15 % of the training pool as a validation set
    (stratified, random_state = --seed)
5.  Compute inverse-frequency class weights and apply to CrossEntropyLoss
6.  Fine-tune AutoModelForSequenceClassification with AdamW + linear warmup
7.  Early stopping on validation macro-F1 (patience = --patience)
8.  Save best checkpoint with model.save_pretrained + tokenizer.save_pretrained
9.  Evaluate best checkpoint on test set; save metrics + confusion matrix PNG

Usage
-----
  # Experiment 4 — mBERT
  python scripts/train_sentence_transformer.py \\
      --model_name bert-base-multilingual-cased \\
      --epochs 10 --batch_size 8 --lr 2e-5 --patience 3 \\
      --output_dir models/mbert

  # Experiment 5 — XLM-RoBERTa
  python scripts/train_sentence_transformer.py \\
      --model_name xlm-roberta-base \\
      --epochs 10 --batch_size 8 --lr 2e-5 --patience 3 \\
      --output_dir models/xlm_roberta
"""

import os
import sys
import random
import argparse
import warnings

import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local modules — resolved relative to this file so the script works from
# any working directory (project root, scripts/, etc.)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "classical"))
from preprocess   import load_participant_transcripts
from clean_labels import clean_labels
from evaluate     import print_metrics, save_confusion_matrix_plot
from dataset_st   import DepressionDataset

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths (resolved from project root = three levels above this file)
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent.parent.parent
LABEL_DIR   = BASE_DIR / "data" / "labels"
TRANS_DIR   = BASE_DIR / "data" / "transcripts"

TRAIN_SPLIT_FILE = LABEL_DIR / "train_split_Depression_AVEC2017.csv"
DEV_SPLIT_FILE   = LABEL_DIR / "dev_split_Depression_AVEC2017.csv"
TEST_SPLIT_FILE  = LABEL_DIR / "full_test_split.csv"


# ---------------------------------------------------------------------------
# Shared helpers (also imported by evaluate_sentence_transformer.py)
# ---------------------------------------------------------------------------

def load_split_ids(csv_path: Path, id_col: str = "Participant_ID") -> set:
    """Return a set of integer participant IDs from a DAIC-WOZ split CSV."""
    df = pd.read_csv(csv_path)
    # full_test_split.csv uses lowercase 'participant_ID'
    if id_col not in df.columns:
        id_col = df.columns[0]
    return set(df[id_col].astype(int).tolist())


def compute_metrics(y_true, y_pred) -> dict:
    """Return the six scalar evaluation metrics used throughout the thesis."""
    return {
        "accuracy":         round(accuracy_score(y_true, y_pred), 4),
        "macro_precision":  round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_recall":     round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "macro_f1":         round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_depressed":     round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "f1_non_depressed": round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
    }


def save_results(model_name: str, metrics: dict, results_dir: Path) -> None:
    """
    Append (or overwrite) one row in sentence_transformer_results.csv.

    If the model already has a row (e.g. from a previous run), that row is
    replaced so the CSV always holds the most recent evaluation.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "sentence_transformer_results.csv"

    row = {"model": model_name, **metrics}
    df_new = pd.DataFrame([row])

    if out_csv.exists():
        df_existing = pd.read_csv(out_csv)
        df_existing = df_existing[df_existing["model"] != model_name]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(out_csv, index=False)
    print(f"   Results saved → {out_csv.resolve()}")


def run_inference(model, loader, device) -> tuple:
    """
    Run forward passes on a DataLoader in evaluation mode.

    Returns
    -------
    (y_true, y_pred) : two Python lists of integer labels.
    """
    model.eval()
    all_true  = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = outputs.logits.argmax(dim=-1)

            all_true.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    return all_true, all_preds


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args) -> None:
    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice : {device}")
    print("=" * 65)
    print(f"  DAIC-WOZ Sentence Transformer Pipeline — {args.model_name}")
    print("=" * 65)

    # ── Resolve output directories from project root ──────────────────────────
    output_dir  = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir
    if not results_dir.is_absolute():
        results_dir = BASE_DIR / results_dir
    figures_dir = results_dir / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Labels ──────────────────────────────────────────────────────────────
    print("\n[1/5] Cleaning labels…")
    labels_df = clean_labels(
        label_dir=str(LABEL_DIR),
        output_dir=str(BASE_DIR / "data" / "processed"),
        output_filename="labels.csv",
        verbose=True,
    )

    # ── 2. Transcripts ─────────────────────────────────────────────────────────
    print("\n[2/5] Loading transcripts…")
    dataset = load_participant_transcripts(TRANS_DIR, labels_df, verbose=True)

    if dataset.empty:
        raise RuntimeError("Dataset is empty — check transcript and label paths.")

    # ── 3. Official train / test split ─────────────────────────────────────────
    print("\n[3/5] Partitioning data using official DAIC-WOZ splits…")
    train_ids = load_split_ids(TRAIN_SPLIT_FILE)
    dev_ids   = load_split_ids(DEV_SPLIT_FILE)
    test_ids  = load_split_ids(TEST_SPLIT_FILE)

    # Combine train + dev as the training pool (consistent with Classical ML
    # pipeline in train_classical.py and standard AVEC2017 practice)
    train_pool_ids = train_ids | dev_ids

    pool_df = dataset[dataset["participant_id"].isin(train_pool_ids)].reset_index(drop=True)
    test_df = dataset[dataset["participant_id"].isin(test_ids)].reset_index(drop=True)

    X_pool = pool_df["text"].tolist()
    y_pool = pool_df["PHQ_Binary"].tolist()
    X_test = test_df["text"].tolist()
    y_test = test_df["PHQ_Binary"].tolist()

    # ── 4. Stratified 15 % validation hold-out from the training pool ──────────
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool,
        test_size=0.15,
        stratify=y_pool,
        random_state=args.seed,
    )

    print(f"   Train : {len(X_train):3d}  "
          f"(dep={sum(y_train)}, non-dep={len(y_train) - sum(y_train)})")
    print(f"   Val   : {len(X_val):3d}  "
          f"(dep={sum(y_val)}, non-dep={len(y_val) - sum(y_val)})")
    print(f"   Test  : {len(X_test):3d}  "
          f"(dep={sum(y_test)}, non-dep={len(y_test) - sum(y_test)})")

    # ── 5. Model + tokeniser ───────────────────────────────────────────────────
    print(f"\n[4/5] Loading tokeniser and model: {args.model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Inverse-frequency class weights to address the ~1:2.3 class imbalance.
    # compute_class_weight('balanced') assigns weight[c] = n_samples /
    # (n_classes * n_samples_c), giving the minority (depressed) class
    # proportionally higher weight in the loss.
    # --depressed_weight overrides the computed weight for label=1 when provided.
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=np.array(y_train),
    )
    if args.depressed_weight is not None:
        cw[1] = args.depressed_weight
    class_weights = torch.FloatTensor(cw).to(device)
    print(f"   Class weights → non-depressed: {cw[0]:.4f}  depressed: {cw[1]:.4f}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
    )
    model.to(device)

    # ── DataLoaders ────────────────────────────────────────────────────────────
    train_ds = DepressionDataset(X_train, y_train, tokenizer, max_length=args.max_length)
    val_ds   = DepressionDataset(X_val,   y_val,   tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=16,              shuffle=False)

    # ── Optimiser & scheduler ──────────────────────────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\n[5/5] Fine-tuning {args.model_name} …")
    print(f"   Epochs={args.epochs}  BatchSize={args.batch_size}  "
          f"LR={args.lr}  WarmupRatio={args.warmup_ratio}  Patience={args.patience}\n")

    best_val_f1        = -1.0
    epochs_no_improve  = 0

    for epoch in range(1, args.epochs + 1):
        # ── Train pass ─────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch:>2d}/{args.epochs}", leave=True)

        for batch in bar:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, labels_batch)
            loss.backward()

            # Gradient clipping prevents exploding gradients during fine-tuning
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)

        # ── Validation pass ────────────────────────────────────────────────────
        y_val_true, y_val_pred = run_inference(model, val_loader, device)
        val_f1 = f1_score(y_val_true, y_val_pred, average="macro", zero_division=0)

        print(f"  Epoch {epoch:>2d}/{args.epochs}  "
              f"train_loss={avg_train_loss:.4f}  val_macro_f1={val_f1:.4f}")

        # ── Early stopping / checkpoint ────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"    Checkpoint saved → {output_dir}  (val macro-F1 = {best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"    No improvement for {epochs_no_improve}/{args.patience} epoch(s)")
            if epochs_no_improve >= args.patience:
                print(f"\n  Early stopping after epoch {epoch} "
                      f"(best val macro-F1 = {best_val_f1:.4f})")
                break

    # ── Load best checkpoint & evaluate on the held-out test set ─────────────
    print(f"\n Loading best checkpoint from {output_dir} …")
    best_tokenizer = AutoTokenizer.from_pretrained(str(output_dir))
    best_model     = AutoModelForSequenceClassification.from_pretrained(str(output_dir))
    best_model.to(device)

    test_ds     = DepressionDataset(X_test, y_test, best_tokenizer, max_length=args.max_length)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    y_test_true, y_test_pred = run_inference(best_model, test_loader, device)

    # Use the short model name (last path component) as the display label
    model_label = args.model_name.split("/")[-1]

    print_metrics(model_label, y_test_true, y_test_pred)
    save_confusion_matrix_plot(
        model_label, y_test_true, y_test_pred,
        output_dir=str(figures_dir),
    )

    metrics = compute_metrics(y_test_true, y_test_pred)
    save_results(model_label, metrics, results_dir)

    print(f"\n Done. Best val macro-F1 = {best_val_f1:.4f}  |  "
          f"Test macro-F1 = {metrics['macro_f1']:.4f}")

    return {"best_val_f1": round(best_val_f1, 4), **metrics}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a multilingual transformer for DAIC-WOZ depression detection.\n\n"
            "Examples:\n"
            "  python scripts/train_sentence_transformer.py "
            "--model_name bert-base-multilingual-cased --output_dir models/mbert\n"
            "  python scripts/train_sentence_transformer.py "
            "--model_name xlm-roberta-base --output_dir models/xlm_roberta"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_name", default="bert-base-multilingual-cased",
        help="HuggingFace model name or local path (default: bert-base-multilingual-cased)",
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Tokeniser max sequence length; inputs are truncated/padded to this (default: 512)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Maximum number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="AdamW learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1,
        help="Fraction of total training steps used for LR warm-up (default: 0.1)",
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience in epochs based on val macro-F1 (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--depressed_weight", type=float, default=None,
        help="Override the loss weight for the depressed class (label=1). "
             "If omitted, the weight is computed automatically via class_weight='balanced'. "
             "Use a higher value (e.g. 4.0, 6.0) when the model collapses to predicting "
             "only the majority class.",
    )
    parser.add_argument(
        "--output_dir", default="models/",
        help="Directory to save the best model checkpoint (default: models/)",
    )
    parser.add_argument(
        "--results_dir", default="results/sentence_transformers/",
        help="Directory to save evaluation CSV and confusion matrix PNG (default: results/sentence_transformers/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
