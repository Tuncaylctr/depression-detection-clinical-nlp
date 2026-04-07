#!/usr/bin/env python3
"""
Load and preprocess DAIC-WOZ clinical interview transcripts.

Each interview is stored as a TSV file named {participant_id}_TRANSCRIPT.tsv
with four columns: start_time, stop_time, speaker, value.

The speaker column contains either 'Ellie' (the interviewer) or 'Participant'.

Why Ellie's turns are excluded
-------------------------------
Ellie's speech consists of standardised interview questions that are identical
or near-identical across all sessions (e.g. "How are you feeling today?",
"Can you tell me more about that?"). If her turns were included, the model
could exploit systematic patterns in the question sequence — a signal that
is correlated with the *structure of the interview protocol*, not with the
participant's mental state. Retaining only the Participant turns ensures the
model learns from the respondent's own language, which is where depression-
related linguistic markers (e.g. reduced lexical diversity, negative affect
words, slower speech cadence) actually manifest.
"""

import os
import re
import pandas as pd
from pathlib import Path
from typing import Union


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_participant_transcripts(
    transcript_dir: Union[str, Path],
    labels_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load DAIC-WOZ transcript TSV files and merge with PHQ label data.

    For each transcript file the function:
      1. Parses the participant ID from the filename.
      2. Reads the tab-separated file (with header).
      3. Keeps only rows where ``speaker == 'Participant'``.
      4. Concatenates all ``value`` cells into one space-separated string.
      5. Looks the participant up in *labels_df*; skips if not found.
      6. Skips participants whose transcript is empty after filtering.

    Parameters
    ----------
    transcript_dir : str | Path
        Directory that contains the ``*_TRANSCRIPT.tsv`` files.
    labels_df : pd.DataFrame
        Label table with columns ``Participant_ID`` (int), ``PHQ_Score`` (int),
        and ``PHQ_Binary`` (int, 0/1).
    verbose : bool
        Print a progress summary when True.

    Returns
    -------
    pd.DataFrame
        Columns: ``participant_id`` (int), ``text`` (str), ``PHQ_Binary`` (int).
        One row per participant that has both a valid transcript and a label.
    """
    transcript_dir = Path(transcript_dir)

    if not transcript_dir.is_dir():
        raise FileNotFoundError(f"Transcript directory not found: {transcript_dir}")

    # Build a fast lookup: Participant_ID (int) → row index
    labels_df = labels_df.copy()
    labels_df["Participant_ID"] = labels_df["Participant_ID"].astype(int)
    label_index = labels_df.set_index("Participant_ID")

    tsv_files = sorted(transcript_dir.glob("*_TRANSCRIPT.csv"))

    if not tsv_files:
        raise FileNotFoundError(
            f"No transcript CSV files matching '*_TRANSCRIPT.csv' found in: {transcript_dir}"
        )

    records = []
    skipped_no_label = []
    skipped_empty = []

    for tsv_path in tsv_files:
        # ── 1. Extract participant ID from filename ──────────────────────────
        match = re.match(r"^(\d+)_TRANSCRIPT\.csv$", tsv_path.name)  # e.g. 300_TRANSCRIPT.csv
        if not match:
            if verbose:
                print(f"  [SKIP] Filename does not match expected pattern: {tsv_path.name}")
            continue

        participant_id = int(match.group(1))

        # ── 2. Check label exists before reading the (potentially large) file ─
        if participant_id not in label_index.index:
            skipped_no_label.append(participant_id)
            continue

        # ── 3. Read TSV ──────────────────────────────────────────────────────
        try:
            df = pd.read_csv(
                tsv_path,
                sep="\t",
                header=0,
                dtype=str,           # read everything as str to avoid type issues
                on_bad_lines="warn", # tolerate malformed rows without crashing
            )
        except Exception as exc:
            if verbose:
                print(f"  [ERROR] Could not read {tsv_path.name}: {exc}")
            continue

        # Ensure required columns exist
        if "speaker" not in df.columns or "value" not in df.columns:
            if verbose:
                print(f"  [SKIP] Missing required columns in {tsv_path.name}")
            continue

        # ── 4. Filter to Participant rows only ───────────────────────────────
        participant_rows = df[df["speaker"] == "Participant"]["value"].dropna()

        # ── 5. Skip if transcript is empty after filtering ───────────────────
        if participant_rows.empty:
            skipped_empty.append(participant_id)
            continue

        # ── 6. Concatenate text ──────────────────────────────────────────────
        combined_text = " ".join(participant_rows.str.strip())

        # ── 7. Retrieve label ────────────────────────────────────────────────
        phq_binary = int(label_index.loc[participant_id, "PHQ_Binary"])

        records.append(
            {
                "participant_id": participant_id,
                "text": combined_text,
                "PHQ_Binary": phq_binary,
            }
        )

    # ── Build output DataFrame ───────────────────────────────────────────────
    result_df = pd.DataFrame(records, columns=["participant_id", "text", "PHQ_Binary"])
    result_df = result_df.sort_values("participant_id").reset_index(drop=True)

    if verbose:
        print(f"\n📂 Transcript directory : {transcript_dir}")
        print(f"   TSV files found       : {len(tsv_files)}")
        print(f"   Loaded successfully   : {len(result_df)}")
        print(f"   Skipped (no label)    : {len(skipped_no_label)}"
              + (f"  → IDs: {skipped_no_label}" if skipped_no_label else ""))
        print(f"   Skipped (empty text)  : {len(skipped_empty)}"
              + (f"  → IDs: {skipped_empty}" if skipped_empty else ""))
        if not result_df.empty:
            print(f"\n   PHQ_Binary=0 (no/mild): {(result_df['PHQ_Binary'] == 0).sum()}")
            print(f"   PHQ_Binary=1 (mod+)   : {(result_df['PHQ_Binary'] == 1).sum()}")

    return result_df


# ---------------------------------------------------------------------------
# Quick smoke-test / CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from clean_labels import clean_labels  # sibling script in scripts/

    transcript_dir = sys.argv[1] if len(sys.argv) > 1 else "data/transcripts"
    label_dir      = sys.argv[2] if len(sys.argv) > 2 else "data/labels"
    output_path    = sys.argv[3] if len(sys.argv) > 3 else "data/processed/transcripts_dataset.csv"

    print("🧹 Loading labels…")
    labels = clean_labels(label_dir=label_dir, verbose=False)

    print("📜 Loading transcripts…\n")
    dataset = load_participant_transcripts(transcript_dir, labels, verbose=True)

    print(f"\n✅ Final dataset shape: {dataset.shape}")
    print(dataset.head())

    # ── Save to disk ─────────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out, index=False)
    print(f"\n💾 Dataset saved to: {out.resolve()}")
