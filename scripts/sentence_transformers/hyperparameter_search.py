#!/usr/bin/env python3
"""
hyperparameter_search.py — Grid search over key hyperparameters for
sentence transformer fine-tuning on DAIC-WOZ.

Searches over learning rate, depressed class weight, and warmup ratio for
both mBERT and XLM-RoBERTa.  Each combination is trained and evaluated on
the test set.  Results are ranked by validation macro-F1 and saved to
results/hyperparameter_search_results.csv.

Usage
-----
  # Full grid search (both models, all combinations)
  python scripts/hyperparameter_search.py

  # Search only XLM-RoBERTa (the model that collapsed)
  python scripts/hyperparameter_search.py --models xlm-roberta-base

  # Dry run — print all combinations without training
  python scripts/hyperparameter_search.py --dry_run
"""

import sys
import argparse
import itertools
from argparse import Namespace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from train_sentence_transformer import train, BASE_DIR


# ---------------------------------------------------------------------------
# Search grid — edit these lists to add or remove values
# ---------------------------------------------------------------------------

MODELS = [
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
]

GRID = {
    "lr":               [5e-6, 1e-5, 2e-5],
    "depressed_weight": [None, 4.0],
    "warmup_ratio":     [0.0],
}

# Fixed hyperparameters (not searched)
FIXED = {
    "max_length": 512,
    "batch_size": 8,
    "epochs":     10,
    "patience":   5,
    "seed":       42,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_run_name(model_name: str, combo: dict) -> str:
    """Short identifier for one (model, hyperparams) combination."""
    short = model_name.split("-")[0]           # 'bert' or 'xlm'
    dw    = combo["depressed_weight"]
    dw_s  = f"dw{dw}" if dw is not None else "dwAuto"
    return f"{short}_lr{combo['lr']}_wr{combo['warmup_ratio']}_{dw_s}"


def all_combinations(models, grid):
    """Yield (model_name, combo_dict) for every point in the grid."""
    keys   = list(grid.keys())
    values = list(grid.values())
    for model in models:
        for combo_values in itertools.product(*values):
            yield model, dict(zip(keys, combo_values))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_search(args_cli):
    models     = args_cli.models
    results_dir = BASE_DIR / "results" / "sentence_transformers"
    search_csv  = results_dir / "hyperparameter_search_results.csv"
    results_dir.mkdir(parents=True, exist_ok=True)

    combos = list(all_combinations(models, GRID))
    total  = len(combos)

    print("=" * 70)
    print("  Hyperparameter Grid Search — DAIC-WOZ Sentence Transformers")
    print("=" * 70)
    print(f"  Models              : {models}")
    print(f"  lr values           : {GRID['lr']}")
    print(f"  depressed_weight    : {GRID['depressed_weight']}")
    print(f"  warmup_ratio        : {GRID['warmup_ratio']}")
    print(f"  Total combinations  : {total}")
    print(f"  Fixed — epochs={FIXED['epochs']}  patience={FIXED['patience']}  "
          f"batch_size={FIXED['batch_size']}  seed={FIXED['seed']}")
    print("=" * 70)

    if args_cli.dry_run:
        print("\n[DRY RUN] Combinations that would be trained:\n")
        for i, (model, combo) in enumerate(combos, 1):
            name = build_run_name(model, combo)
            print(f"  {i:>2d}/{total}  {name}  |  {combo}")
        return

    rows = []

    # Load any prior results so a crashed search can be resumed
    if search_csv.exists():
        rows = pd.read_csv(search_csv).to_dict("records")
        done_names = {r["run_name"] for r in rows}
        print(f"\n  Resuming — {len(done_names)} run(s) already completed.\n")
    else:
        done_names = set()

    for i, (model_name, combo) in enumerate(combos, 1):
        run_name = build_run_name(model_name, combo)

        if run_name in done_names:
            print(f"\n[{i:>2d}/{total}] SKIP (already done): {run_name}")
            continue

        print(f"\n{'=' * 70}")
        print(f"[{i:>2d}/{total}]  {run_name}")
        print(f"         model={model_name}  lr={combo['lr']}  "
              f"depressed_weight={combo['depressed_weight']}  "
              f"warmup_ratio={combo['warmup_ratio']}")
        print(f"{'=' * 70}")

        # Each combination saves its checkpoint to its own subdirectory
        output_dir = BASE_DIR / "models" / "search" / run_name

        train_args = Namespace(
            model_name       = model_name,
            max_length       = FIXED["max_length"],
            batch_size       = FIXED["batch_size"],
            epochs           = FIXED["epochs"],
            lr               = combo["lr"],
            warmup_ratio     = combo["warmup_ratio"],
            depressed_weight = combo["depressed_weight"],
            patience         = FIXED["patience"],
            seed             = FIXED["seed"],
            output_dir       = str(output_dir),
            results_dir      = str(results_dir),
        )

        try:
            metrics = train(train_args)
        except Exception as exc:
            print(f"\n  [ERROR] Run failed: {exc}")
            metrics = {
                "best_val_f1": None, "accuracy": None,
                "macro_precision": None, "macro_recall": None,
                "macro_f1": None, "f1_depressed": None,
                "f1_non_depressed": None,
            }

        row = {
            "run_name":        run_name,
            "model":           model_name,
            "lr":              combo["lr"],
            "depressed_weight": combo["depressed_weight"],
            "warmup_ratio":    combo["warmup_ratio"],
            **metrics,
        }
        rows.append(row)
        done_names.add(run_name)

        # Save after every run so a crash doesn't lose completed results
        pd.DataFrame(rows).to_csv(search_csv, index=False)
        print(f"\n  Intermediate results saved → {search_csv}")

    # ── Final ranked summary ──────────────────────────────────────────────────
    df = pd.DataFrame(rows).sort_values("best_val_f1", ascending=False)

    print("\n" + "=" * 70)
    print("  GRID SEARCH RESULTS — ranked by validation macro-F1")
    print("=" * 70)

    display_cols = [
        "run_name", "lr", "depressed_weight", "warmup_ratio",
        "best_val_f1", "macro_f1", "f1_depressed", "f1_non_depressed",
    ]
    print(df[display_cols].to_string(index=False))

    print("\n  Best configuration per model:")
    for model in models:
        sub = df[df["model"] == model]
        if sub.empty or sub["best_val_f1"].isna().all():
            continue
        best = sub.loc[sub["best_val_f1"].idxmax()]
        print(f"\n  {model}")
        print(f"    lr={best['lr']}  depressed_weight={best['depressed_weight']}  "
              f"warmup_ratio={best['warmup_ratio']}")
        print(f"    val macro-F1={best['best_val_f1']}  "
              f"test macro-F1={best['macro_f1']}  "
              f"f1_depressed={best['f1_depressed']}")

    df.to_csv(search_csv, index=False)
    print(f"\n  Full results saved → {search_csv.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid search over hyperparameters for sentence transformer fine-tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+",
        default=MODELS,
        help="Which models to search (default: both mBERT and XLM-RoBERTa)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print all combinations that would be run without actually training",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_search(parse_args())
