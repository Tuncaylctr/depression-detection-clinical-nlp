#!/usr/bin/env python3
"""
clean_labels.py — Label preparation for DAIC-WOZ depression detection.

Reads the three original DAIC-WOZ split label files:
  • train_split_Depression_AVEC2017.csv  (Participant_ID, PHQ8_Score, …)
  • dev_split_Depression_AVEC2017.csv    (Participant_ID, PHQ8_Score, …)
  • full_test_split.csv                  (Participant_ID, PHQ_Score,  …)

Combines them into a single DataFrame, removes duplicate IDs (keeps first
occurrence), renames PHQ8_Score → PHQ_Score for consistency, and derives a
binary class label:

    PHQ_Binary = 1  if  PHQ_Score >= 10  (moderate–severe depression)
    PHQ_Binary = 0  otherwise

Output: data/processed/labels.csv  (columns: Participant_ID, PHQ_Score, PHQ_Binary)

Clinical threshold rationale:
The PHQ-8 cutpoint of 10 is widely used in clinical practice and is the
official threshold adopted by the DAIC-WOZ challenge organisers (Gratch et al.,
2014).  Scores of 10–14 indicate moderate depression, 15–19 moderately severe,
and ≥ 20 severe depression.
"""

import sys
import pandas as pd
from pathlib import Path


# Constants — actual filenames in data/labels/

TRAIN_FILE = "train_split_Depression_AVEC2017.csv"
DEV_FILE   = "dev_split_Depression_AVEC2017.csv"
TEST_FILE  = "full_test_split.csv"          # the version that includes PHQ scores


# Core function

def clean_labels(
    label_dir: str = "data/labels",
    output_dir: str = "data/processed",
    output_filename: str = "labels.csv",
    threshold: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load, merge, and clean DAIC-WOZ label files.

    Parameters:
    label_dir       : Directory that contains the three split CSVs.
    output_dir      : Directory where the cleaned CSV will be saved.
    output_filename : Name of the output file (default: labels.csv).
    threshold       : PHQ score at or above which PHQ_Binary = 1 (default: 10).
    verbose         : Print processing information.

    Returns:
    pd.DataFrame
        Columns: Participant_ID (int), PHQ_Score (int), PHQ_Binary (int).
    """
    label_path  = Path(label_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define the three source files
    sources = [
        (label_path / TRAIN_FILE, "PHQ8_Score"),
        (label_path / DEV_FILE,   "PHQ8_Score"),
        (label_path / TEST_FILE,  "PHQ_Score"),
    ]

    dfs = []
    for file_path, phq_col_name in sources:
        if not file_path.exists():
            # Try to detect alternate PHQ column names gracefully
            if verbose:
                print(f"  [WARN] File not found, skipping: {file_path.name}")
            continue

        df = pd.read_csv(file_path)

        # Normalise column name: handle both PHQ_Score / PHQ8_Score
        detected_col = None
        for candidate in [phq_col_name, "PHQ_Score", "PHQ8_Score"]:
            if candidate in df.columns:
                detected_col = candidate
                break

        if detected_col is None:
            if verbose:
                print(f"  [WARN] No PHQ score column found in {file_path.name}, skipping.")
            continue

        # Normalise Participant_ID column name (test file uses 'participant_ID')
        if "participant_ID" in df.columns and "Participant_ID" not in df.columns:
            df = df.rename(columns={"participant_ID": "Participant_ID"})

        df_clean = df[["Participant_ID", detected_col]].copy()
        df_clean = df_clean.rename(columns={detected_col: "PHQ_Score"})
        df_clean["Participant_ID"] = df_clean["Participant_ID"].astype(int)
        df_clean["PHQ_Score"]      = pd.to_numeric(df_clean["PHQ_Score"], errors="coerce")
        df_clean = df_clean.dropna(subset=["PHQ_Score"])
        df_clean["PHQ_Score"] = df_clean["PHQ_Score"].astype(int)

        if verbose:
            print(f"    {file_path.name}: {len(df_clean)} participants, "
                  f"PHQ range [{df_clean['PHQ_Score'].min()}, {df_clean['PHQ_Score'].max()}]")

        dfs.append(df_clean)

    if not dfs:
        raise RuntimeError(
            f"No valid label files found in '{label_dir}'. "
            f"Expected: {TRAIN_FILE}, {DEV_FILE}, {TEST_FILE}"
        )

    # Combine all splits
    merged = pd.concat(dfs, ignore_index=True)

    n_before = len(merged)
    merged = merged.drop_duplicates(subset=["Participant_ID"], keep="first")
    n_dupes = n_before - len(merged)

    # Binary classification label
    merged["PHQ_Binary"] = (merged["PHQ_Score"] >= threshold).astype(int)

    # Sort and finalise
    result = (
        merged[["Participant_ID", "PHQ_Score", "PHQ_Binary"]]
        .sort_values("Participant_ID")
        .reset_index(drop=True)
    )

    # Save
    out_file = output_path / output_filename
    result.to_csv(out_file, index=False)

    if verbose:
        n_dep    = (result["PHQ_Binary"] == 1).sum()
        n_nondep = (result["PHQ_Binary"] == 0).sum()
        print(f"\n   Merged total rows  : {n_before}")
        print(f"   Duplicates removed : {n_dupes}")
        print(f"   Unique participants: {len(result)}")
        print(f"\n   PHQ_Binary=0 (non-depressed) : {n_nondep}")
        print(f"   PHQ_Binary=1 (depressed)     : {n_dep}")
        print(f"   Class ratio (0:1)            : {n_nondep / max(n_dep, 1):.2f}:1")
        print(f"\n   Threshold : PHQ_Score >= {threshold}")
        print(f"   Saved to  : {out_file.resolve()}")

    return result


# CLI entry-point
if __name__ == "__main__":
    threshold = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    label_dir = sys.argv[2] if len(sys.argv) > 2 else "data/labels"

    print(f" Cleaning DAIC-WOZ depression labels (PHQ threshold = {threshold})…\n")
    result = clean_labels(label_dir=label_dir, threshold=threshold, verbose=True)

    print(f"\n Final labels shape: {result.shape}")
    print(result.head(10).to_string(index=False))
