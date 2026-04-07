#!/usr/bin/env python3
"""
train_classical.py — Classical ML pipeline for DAIC-WOZ depression detection.

Pipeline overview
-----------------
1.  Load and preprocess transcripts via preprocess.py
2.  Partition data using the *official* DAIC-WOZ split IDs:
      Train  = train_split_Depression_AVEC2017.csv  (+ dev set combined)
      Test   = full_test_split.csv
3.  For each of three classifiers (Logistic Regression, LinearSVC, Random Forest):
      a. Build a scikit-learn Pipeline:  TF-IDF → Classifier
      b. Run GridSearchCV (5-fold, macro-F1 scoring) over:
           TF-IDF: max_features ∈ {5000, 10000, 20000}
                   ngram_range  ∈ {(1,1), (1,2), (1,3)}
           LR:     C ∈ {0.01, 0.1, 1, 10}
           SVM:    C ∈ {0.01, 0.1, 1, 10}
           RF:     n_estimators ∈ {100, 300}
                   max_depth    ∈ {None, 10, 20}
      c. Refit on the full training set with best hyperparameters
      d. Evaluate on the held-out test set
4.  Print a summary comparison table
5.  Save results to results/results_classical.csv
6.  Persist each trained model + vectorizer via joblib

Class-imbalance handling
------------------------
The DAIC-WOZ dataset contains 57 depressed and 132 non-depressed participants
(ratio ≈ 1:2.3).  All classifiers use class_weight='balanced', which scales
class weights inversely proportional to class frequencies, giving the minority
(depressed) class higher importance during optimisation.

Primary evaluation metric
--------------------------
Macro-F1 is used as the primary metric throughout GridSearchCV and the final
comparison table because it weighs both classes equally, making it robust to
class imbalance; a model that simply predicts the majority class can achieve
high accuracy but yields a low macro-F1.

Reference
---------
Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
Journal of Machine Learning Research, 12, 2825–2830.
"""

import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline        import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import LinearSVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics         import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Local modules (run from project root or add scripts/ to path)
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import load_participant_transcripts
from clean_labels import clean_labels
from evaluate import print_metrics, save_confusion_matrix_plot

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).parent.parent          # project root
LABEL_DIR   = BASE_DIR / "data" / "labels"
TRANS_DIR   = BASE_DIR / "data" / "transcripts"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"

TRAIN_SPLIT_FILE = LABEL_DIR / "train_split_Depression_AVEC2017.csv"
DEV_SPLIT_FILE   = LABEL_DIR / "dev_split_Depression_AVEC2017.csv"
TEST_SPLIT_FILE  = LABEL_DIR / "full_test_split.csv"


# ---------------------------------------------------------------------------
# Helper: load official split IDs
# ---------------------------------------------------------------------------

def _load_split_ids(csv_path: Path, id_col: str = "Participant_ID") -> set:
    """Return a set of integer participant IDs from a split CSV."""
    df = pd.read_csv(csv_path)
    # Handle 'participant_ID' (lowercase) variant in test file
    if id_col not in df.columns:
        id_col = df.columns[0]
    return set(df[id_col].astype(int).tolist())


# ---------------------------------------------------------------------------
# Helper: macro metric summary
# ---------------------------------------------------------------------------

def _summarise(model_name: str, y_true, y_pred) -> dict:
    return {
        "model":      model_name,
        "accuracy":   round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro":        round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_depressed":    round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "f1_nondepressed": round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_and_evaluate(verbose: bool = True) -> pd.DataFrame:
    """
    Run the full classical ML pipeline.

    Returns
    -------
    pd.DataFrame with one row per model and key performance metrics.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Labels & transcripts ──────────────────────────────────────────────
    print("=" * 65)
    print("  DAIC-WOZ Classical ML Pipeline")
    print("=" * 65)

    print("\n[1/4] Cleaning labels…")
    labels_df = clean_labels(
        label_dir=str(LABEL_DIR),
        output_dir=str(BASE_DIR / "data" / "processed"),
        output_filename="labels.csv",
        verbose=verbose,
    )

    print("\n[2/4] Loading transcripts…")
    dataset = load_participant_transcripts(TRANS_DIR, labels_df, verbose=verbose)

    if dataset.empty:
        raise RuntimeError("Dataset is empty — check transcript and label paths.")

    # ── 2. Official train / test split ───────────────────────────────────────
    print("\n[3/4] Partitioning data using official DAIC-WOZ splits…")

    train_ids = _load_split_ids(TRAIN_SPLIT_FILE)
    dev_ids   = _load_split_ids(DEV_SPLIT_FILE)
    test_ids  = _load_split_ids(TEST_SPLIT_FILE)

    # Combine train + dev for training (common practice; dev is the tuning set
    # but since we tune via CV, it is safe to absorb into the train pool)
    train_pool_ids = train_ids | dev_ids

    train_mask = dataset["participant_id"].isin(train_pool_ids)
    test_mask  = dataset["participant_id"].isin(test_ids)

    train_df = dataset[train_mask].reset_index(drop=True)
    test_df  = dataset[test_mask].reset_index(drop=True)

    X_train, y_train = train_df["text"].tolist(), train_df["PHQ_Binary"].tolist()
    X_test,  y_test  = test_df["text"].tolist(),  test_df["PHQ_Binary"].tolist()

    if verbose:
        print(f"   Train participants : {len(train_df)}"
              f"  (dep={sum(y_train)}, non-dep={len(y_train)-sum(y_train)})")
        print(f"   Test  participants : {len(test_df)}"
              f"  (dep={sum(y_test)}, non-dep={len(y_test)-sum(y_test)})")

    # ── 3. Define classifiers + hyperparameter grids ─────────────────────────
    print("\n[4/4] Training and evaluating models (GridSearchCV)…\n")

    # Shared TF-IDF parameter grid
    tfidf_grid = {
        "tfidf__max_features": [5000, 10000, 20000],
        "tfidf__ngram_range":  [(1, 1), (1, 2), (1, 3)],
    }

    model_configs = [
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                random_state=42,
            ),
            "param_grid": {
                **tfidf_grid,
                "clf__C": [0.01, 0.1, 1.0, 10.0],
            },
        },
        {
            "name": "LinearSVC (SVM)",
            "estimator": LinearSVC(
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            ),
            "param_grid": {
                **tfidf_grid,
                "clf__C": [0.01, 0.1, 1.0, 10.0],
            },
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "param_grid": {
                **tfidf_grid,
                "clf__n_estimators": [100, 300],
                "clf__max_depth":    [None, 10, 20],
            },
        },
    ]

    summary_rows = []

    for cfg in model_configs:
        mname = cfg["name"]
        print(f"  ── {mname} ──")

        # Build Pipeline: TF-IDF → classifier
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(
                sublinear_tf=True,       # log(1 + tf) scaling
                strip_accents="unicode",
                analyzer="word",
                stop_words="english",
                min_df=2,                # ignore very rare terms
            )),
            ("clf", cfg["estimator"]),
        ])

        # GridSearchCV — 5-fold stratified CV, macro-F1 scoring
        gs = GridSearchCV(
            pipe,
            param_grid=cfg["param_grid"],
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=0,
            refit=True,
        )

        gs.fit(X_train, y_train)

        best_params = gs.best_params_
        best_cv_f1  = round(gs.best_score_, 4)

        print(f"     Best CV macro-F1 : {best_cv_f1}")
        print(f"     Best params      : {best_params}")

        # Evaluate on held-out test set
        y_pred = gs.predict(X_test)

        print_metrics(mname, y_test, y_pred)
        save_confusion_matrix_plot(mname, y_test, y_pred, output_dir=str(FIGURES_DIR))

        # Collect summary
        row = _summarise(mname, y_test, y_pred)
        row["best_cv_f1_macro"] = best_cv_f1
        row["best_params"]      = str(best_params)
        summary_rows.append(row)

        # ── Persist model ──────────────────────────────────────────────────
        safe = mname.lower().replace(" ", "_").replace("(", "").replace(")", "")
        model_path = MODELS_DIR / f"{safe}_pipeline.joblib"
        joblib.dump(gs.best_estimator_, model_path)
        print(f"    Model saved → {model_path}")
        print()

    # ── 4. Summary table ─────────────────────────────────────────────────────
    results_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    display_cols = [
        "model", "accuracy",
        "precision_macro", "recall_macro", "f1_macro",
        "f1_depressed", "f1_nondepressed",
        "best_cv_f1_macro",
    ]
    print(results_df[display_cols].to_string(index=False))
    print("=" * 65)

    # Find best model
    best_idx   = results_df["f1_macro"].idxmax()
    best_model = results_df.loc[best_idx, "model"]
    best_f1    = results_df.loc[best_idx, "f1_macro"]
    print(f"\n    Best model: {best_model}  (macro-F1 = {best_f1})\n")

    # ── 5. Save results CSV ──────────────────────────────────────────────────
    out_csv = RESULTS_DIR / "results_classical.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"   Results saved → {out_csv.resolve()}")

    return results_df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = train_and_evaluate(verbose=True)
