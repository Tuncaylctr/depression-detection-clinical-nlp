#!/usr/bin/env python3
"""
evaluate_sentence_transformer.py — Standalone evaluation of a fine-tuned
transformer checkpoint on the DAIC-WOZ test set.

Loads a checkpoint saved by train_sentence_transformer.py
(model.save_pretrained + tokenizer.save_pretrained) and runs inference on the
held-out test set without re-training.  Produces the same metrics and
confusion matrix PNG as the training script.

Usage:
  python scripts/evaluate_sentence_transformer.py \\
      --checkpoint_dir models/mbert

  python scripts/evaluate_sentence_transformer.py \\
      --checkpoint_dir models/xlm_roberta \\
      --results_dir results/
"""

import sys
import argparse
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local modules
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).parent.parent / "classical"))
from preprocess   import load_participant_transcripts
from clean_labels import clean_labels
from evaluate     import print_metrics, save_confusion_matrix_plot
from dataset_st   import DepressionDataset

# Reuse all shared helpers and path constants from the training script
from train_sentence_transformer import (
    BASE_DIR,
    LABEL_DIR,
    TRANS_DIR,
    TEST_SPLIT_FILE,
    load_split_ids,
    compute_metrics,
    save_results,
    run_inference,
)

warnings.filterwarnings("ignore")


# Evaluation function

def evaluate(args) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice : {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir    = Path(args.results_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = BASE_DIR / checkpoint_dir
    if not results_dir.is_absolute():
        results_dir = BASE_DIR / results_dir
    figures_dir = results_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Labels & transcripts
    print("\n[1/3] Loading data…")
    labels_df = clean_labels(
        label_dir=str(LABEL_DIR),
        output_dir=str(BASE_DIR / "data" / "processed"),
        output_filename="labels.csv",
        verbose=False,
    )
    dataset = load_participant_transcripts(TRANS_DIR, labels_df, verbose=True)

    if dataset.empty:
        raise RuntimeError("Dataset is empty — check transcript and label paths.")

    # 2. Test split
    test_ids = load_split_ids(TEST_SPLIT_FILE)
    test_df  = dataset[dataset["participant_id"].isin(test_ids)].reset_index(drop=True)

    X_test = test_df["text"].tolist()
    y_test = test_df["PHQ_Binary"].tolist()
    print(f"   Test participants : {len(test_df)}  "
          f"(dep={sum(y_test)}, non-dep={len(y_test) - sum(y_test)})")

    # 3. Load checkpoint
    print(f"\n[2/3] Loading checkpoint from {checkpoint_dir} …")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    model     = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
    model.to(device)

    # Use --model_label if provided, otherwise fall back to the directory name
    model_label = args.model_label if args.model_label else checkpoint_dir.name

    # 4. Inference & evaluation
    print(f"\n[3/3] Running inference on test set…")
    test_ds     = DepressionDataset(X_test, y_test, tokenizer, max_length=args.max_length)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    y_true, y_pred = run_inference(model, test_loader, device)

    print_metrics(model_label, y_true, y_pred)
    save_confusion_matrix_plot(
        model_label, y_true, y_pred,
        output_dir=str(figures_dir),
    )

    metrics = compute_metrics(y_true, y_pred)
    save_results(model_label, metrics, results_dir)

    print(f"\n Done. Test macro-F1 = {metrics['macro_f1']:.4f}")


# CLI entry-point

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fine-tuned transformer checkpoint on the DAIC-WOZ test set.\n\n"
            "Example:\n"
            "  python scripts/evaluate_sentence_transformer.py --checkpoint_dir models/mbert"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_dir", required=True,
        help="Path to the saved model/tokenizer directory (output of train_sentence_transformer.py)",
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Tokeniser max sequence length (must match the value used during training; default: 512)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Inference batch size (default: 16)",
    )
    parser.add_argument(
        "--results_dir", default="results/sentence_transformers/",
        help="Directory to save evaluation CSV and confusion matrix PNG (default: results/sentence_transformers/)",
    )
    parser.add_argument(
        "--model_label", default=None,
        help="Override the display label / PNG filename (default: checkpoint directory name)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
