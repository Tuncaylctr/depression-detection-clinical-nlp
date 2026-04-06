#!/usr/bin/env python3
"""
Clean and merge depression label data from multiple label CSV files.

This script reads all label CSVs from data/labels/ and produces a single CSV
with all unique participant IDs and their PHQ binary classification.

Features:
  - Handles both PHQ_Score (full_test_split) and PHQ8_Score (train/dev)
  - Merges multiple label files
  - Handles duplicate participant IDs (keeps first occurrence)
  - Computes binary classification: 1 if score >= threshold, else 0
  - Outputs: all_participants_phq_binary.csv
"""

import pandas as pd
import os
from pathlib import Path


def clean_labels(
    label_dir: str = "data/labels",
    output_dir: str = "data/labels/processed",
    threshold: int = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load, merge, and clean label files.
    
    Args:
        label_dir: Directory containing label CSVs
        output_dir: Output directory for cleaned CSV
        threshold: PHQ score threshold for binary classification (default: 10)
        verbose: Print processing info
        
    Returns:
        DataFrame with columns: Participant_ID, PHQ_Score, PHQ_Binary
    """
    
    label_path = Path(label_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # List all CSV files in label directory
    csv_files = sorted(label_path.glob("*_split*.csv")) + sorted(label_path.glob("full_test*.csv"))
    
    if verbose:
        print(f" Found {len(csv_files)} label files:")
        for f in csv_files:
            print(f"   - {f.name}")
    
    dfs = []
    
    for csv_file in csv_files:
        if verbose:
            print(f"\n Processing {csv_file.name}...")
        
        df = pd.read_csv(csv_file)
        
        # Detect which PHQ score column exists
        if "PHQ_Score" in df.columns:
            phq_col = "PHQ_Score"
        elif "PHQ8_Score" in df.columns:
            phq_col = "PHQ8_Score"
        else:
            print(f"    No PHQ score column found. Skipping {csv_file.name}")
            continue
        
        # Select only Participant_ID and PHQ score
        df_clean = df[["Participant_ID", phq_col]].copy()
        df_clean.rename(columns={phq_col: "PHQ_Score"}, inplace=True)
        
        if verbose:
            print(f"   ✓ Loaded {len(df_clean)} participants, PHQ score range: [{df_clean['PHQ_Score'].min()}, {df_clean['PHQ_Score'].max()}]")
        
        dfs.append(df_clean)
    
    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates, keeping first occurrence
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=["Participant_ID"], keep="first")
    duplicates_removed = initial_count - len(merged_df)
    
    if verbose:
        print(f"\n Merged data:")
        print(f"   Total rows before dedup: {initial_count}")
        print(f"   Duplicates removed: {duplicates_removed}")
        print(f"   Unique participants: {len(merged_df)}")
    
    # Compute binary classification
    merged_df["PHQ_Binary"] = (merged_df["PHQ_Score"] >= threshold).astype(int)
    
    # Final output columns
    result_df = merged_df[["Participant_ID", "PHQ_Score", "PHQ_Binary"]].sort_values("Participant_ID").reset_index(drop=True)
    
    # Save to CSV
    output_file = output_path / "all_participants_phq_binary.csv"
    result_df.to_csv(output_file, index=False)
    
    if verbose:
        print(f"\n Results (threshold={threshold}):")
        print(f"   PHQ_Binary=0 (no/mild depression): {(result_df['PHQ_Binary'] == 0).sum()}")
        print(f"   PHQ_Binary=1 (moderate+ depression): {(result_df['PHQ_Binary'] == 1).sum()}")
        print(f"\n Saved to: {output_file}")
    
    return result_df


if __name__ == "__main__":
    import sys
    
    # Optional: accept threshold as command-line argument
    threshold = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    
    print(f"🧹 Cleaning depression labels (PHQ threshold={threshold})...\n")
    result = clean_labels(threshold=threshold, verbose=True)
    print(f"\n Final dataset shape: {result.shape}")
    print(result.head(10))
