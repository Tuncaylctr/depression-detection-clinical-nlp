#!/usr/bin/env python3
"""
preprocess.py — Transcript preprocessing for DAIC-WOZ depression detection.

Pipeline:
  1. Scan transcript directory for files matching {participant_id}_TRANSCRIPT.{tsv,csv}
  2. For each transcript, keep ONLY rows where speaker == 'Participant'
  3. Concatenate all Participant 'value' cells into one space-separated string
  4. Merge with labels (Participant_ID, PHQ_Score, PHQ_Binary)
  5. Return a clean DataFrame: participant_id | text | PHQ_Binary

Why Ellie's turns are excluded :
Ellie (the virtual interviewer) delivers a fixed, standardised set of
questions that are practically identical across every session.  Including her
turns would inject interview-structure signals — correlated with the protocol,
not with the participant's mental state — and would bias any text-based model
toward session ordering rather than genuine linguistic depression markers such
as reduced lexical diversity, negative-affect vocabulary, or hedging language.
"""

import os
import re
import sys
import pandas as pd
from pathlib import Path
from typing import Union

# Ensure project root is on sys.path so `scripts.data.clean_labels` resolves
# when this file is executed directly (python scripts/data/preprocess.py).
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Public API
def load_participant_transcripts(
    transcript_dir: Union[str, Path],
    labels_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load DAIC-WOZ transcript files and merge with PHQ label data.

    Parameters:
    transcript_dir : str | Path
        Directory containing ``*_TRANSCRIPT.tsv`` or ``*_TRANSCRIPT.csv`` files.
    labels_df : pd.DataFrame
        Label table with columns ``Participant_ID`` (int), ``PHQ_Score`` (int),
        and ``PHQ_Binary`` (int, 0/1).
    verbose : bool
        Print a progress summary when True.

    Returns:
    pd.DataFrame
        Columns: ``participant_id`` (int), ``text`` (str), ``PHQ_Binary`` (int).
        One row per participant with both a valid transcript and a label.
    """
    transcript_dir = Path(transcript_dir)

    if not transcript_dir.is_dir():
        raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")

    # Build fast lookup: Participant_ID (int) → label row
    labels_df = labels_df.copy()
    labels_df["Participant_ID"] = labels_df["Participant_ID"].astype(int)
    label_index = labels_df.set_index("Participant_ID")

    # Accept both .tsv and .csv extensions (DAIC-WOZ files are tab-separated
    # but may be distributed with a .csv extension)
    tsv_files = sorted(transcript_dir.glob("*_TRANSCRIPT.tsv")) + \
                sorted(transcript_dir.glob("*_TRANSCRIPT.csv"))

    if not tsv_files:
        raise FileNotFoundError(
            f"No transcript files matching '*_TRANSCRIPT.{{tsv,csv}}' found in: {transcript_dir}"
        )

    records          = []
    skipped_no_label = []
    skipped_empty    = []

    for file_path in tsv_files:
        # 1. Extract participant ID 
        match = re.match(r"^(\d+)_TRANSCRIPT\.(tsv|csv)$", file_path.name)
        if not match:
            if verbose:
                print(f"  [SKIP] Unexpected filename pattern: {file_path.name}")
            continue

        participant_id = int(match.group(1))

        #  2. Skip early if no label exists 
        if participant_id not in label_index.index:
            skipped_no_label.append(participant_id)
            continue

        #  3. Read tab-separated file 
        try:
            df = pd.read_csv(
                file_path,
                sep="\t",
                header=0,
                dtype=str,
                on_bad_lines="warn",
            )
        except Exception as exc:
            if verbose:
                print(f"  [ERROR] Could not read {file_path.name}: {exc}")
            continue

        #  4. Validate required columns 
        if "speaker" not in df.columns or "value" not in df.columns:
            if verbose:
                print(f"  [SKIP] Missing 'speaker'/'value' columns in {file_path.name}")
            continue

        #  5. Filter to Participant rows only 
        participant_rows = df[df["speaker"] == "Participant"]["value"].dropna()

        #  6. Skip if empty after filtering 
        if participant_rows.empty:
            skipped_empty.append(participant_id)
            continue

        #  7. Concatenate all turns into one document 
        combined_text = " ".join(participant_rows.str.strip())

        #  8. Retrieve label 
        row = label_index.loc[participant_id]
        phq_binary = int(row["PHQ_Binary"])

        records.append({
            "participant_id": participant_id,
            "text": combined_text,
            "PHQ_Binary": phq_binary,
        })

    #  Assemble output DataFrame 
    result_df = pd.DataFrame(records, columns=["participant_id", "text", "PHQ_Binary"])
    result_df = result_df.sort_values("participant_id").reset_index(drop=True)

    if verbose:
        n_dep   = (result_df["PHQ_Binary"] == 1).sum()
        n_nondep = (result_df["PHQ_Binary"] == 0).sum()
        print(f"\n Transcript directory : {transcript_dir}")
        print(f"   Files found          : {len(tsv_files)}")
        print(f"   Loaded successfully  : {len(result_df)}")
        print(f"   Skipped (no label)   : {len(skipped_no_label)}"
              + (f"  → IDs: {skipped_no_label}" if skipped_no_label else ""))
        print(f"   Skipped (empty text) : {len(skipped_empty)}"
              + (f"  → IDs: {skipped_empty}" if skipped_empty else ""))
        if not result_df.empty:
            print(f"\n   PHQ_Binary=0 (non-depressed) : {n_nondep}")
            print(f"   PHQ_Binary=1 (depressed)     : {n_dep}")
            ratio = n_nondep / max(n_dep, 1)
            print(f"   Class ratio (0:1)            : {ratio:.2f}:1")

    return result_df


def preprocess(
    transcript_dir: Union[str, Path] = "data/transcripts",
    label_dir: Union[str, Path] = "data/labels",
    output_path: Union[str, Path] = "data/processed/dataset.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    End-to-end convenience wrapper: clean labels → load transcripts → save.

    Parameters:
    transcript_dir : path to directory holding transcript files
    label_dir      : path to directory holding split CSVs
    output_path    : where to write the final dataset CSV
    verbose        : verbosity
    Returns
    pd.DataFrame with columns participant_id, text, PHQ_Binary
    """
    # Import here to keep the module usable independently
    from scripts.data.clean_labels import clean_labels  # noqa: PLC0415

    if verbose:
        print(" Step 1 — Cleaning labels…")
    labels = clean_labels(label_dir=str(label_dir), verbose=verbose)

    if verbose:
        print("\n Step 2 — Loading transcripts…")
    dataset = load_participant_transcripts(transcript_dir, labels, verbose=verbose)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out, index=False)
    if verbose:
        print(f"\n Dataset saved → {out.resolve()}  (shape: {dataset.shape})")

    return dataset

# CLI entry-point
if __name__ == "__main__":
    import sys

    transcript_dir = sys.argv[1] if len(sys.argv) > 1 else "data/transcripts"
    label_dir      = sys.argv[2] if len(sys.argv) > 2 else "data/labels"
    output_path    = sys.argv[3] if len(sys.argv) > 3 else "data/processed/dataset.csv"

    dataset = preprocess(
        transcript_dir=transcript_dir,
        label_dir=label_dir,
        output_path=output_path,
        verbose=True,
    )

    print(f"\n Final dataset: {dataset.shape[0]} participants")
    print(dataset.head())
