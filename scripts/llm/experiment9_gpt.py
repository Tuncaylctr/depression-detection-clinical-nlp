#!/usr/bin/env python3
"""
experiment9_gpt.py — Zero-shot and few-shot GPT evaluation for DAIC-WOZ
depression detection (Experiment 9 of the thesis).

This script evaluates a closed-weights GPT model (default: gpt-4o) on the
official 47-participant DAIC-WOZ test set in two prompting conditions:

  Zero-shot : task description only, no labeled examples in the prompt
  Few-shot  : task description + 4 labeled examples from the training pool
              (2 depressed, 2 non-depressed), chosen by shortest transcript

Both conditions use temperature=0 and seed=42 for full reproducibility.
Transcripts are truncated to 3 000 words before being sent to the API.

Usage
  # Full run (requires OPENAI_API_KEY env var)
  python scripts/experiment9_gpt.py

Output
  results/gpt/gpt_results.csv          — one row per condition, same columns
                                         as results/classical/results_classical.csv
  results/gpt/figures/cm_*.png         — confusion matrix PNGs
"""

import os
import sys
import re
import time
import logging
import warnings
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

#  Import evaluate utilities from scripts/classical/evaluate.py 
# We insert the classical sub-package onto sys.path so the import mirrors how
# train_classical.py consumes it (bare module name, no package prefix).
sys.path.insert(0, str(Path(__file__).parent.parent / "classical"))
from evaluate import print_metrics, save_confusion_matrix_plot  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


#  Paths 

BASE_DIR  = Path(__file__).parent.parent.parent   # project root
LABEL_DIR = BASE_DIR / "data" / "labels"
DATA_FILE = BASE_DIR / "data" / "processed" / "dataset.csv"

TRAIN_SPLIT_FILE = LABEL_DIR / "train_split_Depression_AVEC2017.csv"
DEV_SPLIT_FILE   = LABEL_DIR / "dev_split_Depression_AVEC2017.csv"
TEST_SPLIT_FILE  = LABEL_DIR / "full_test_split.csv"


#  Constants 

MAX_WORDS   = 3_000   # hard word-count cap before sending to the API
RANDOM_SEED = 42      # used for OpenAI seed param and pandas sampling


#  System prompt (shared by both conditions) 
#
# The prompt explains:
#   • the task (binary depression classification)
#   • the PHQ-8 threshold of 10 used to define "depressed"
#   • the interview context (DAIC-WOZ clinical session)
#   • the required output format (ONLY "0" or "1")

SYSTEM_PROMPT = """\
You are a clinical-NLP assistant helping to detect depression from interview transcripts.

Task: Given a transcript of a structured clinical interview (DAIC-WOZ dataset),
classify whether the participant is depressed.

Criterion:
  PHQ-8 total score ≥ 10  →  depressed     (output: 1)
  PHQ-8 total score < 10  →  non-depressed  (output: 0)

Context: The transcript contains the participant's spoken responses during a
semi-structured interview covering mood, daily activities, sleep, and life events.
The interviewer turns are not included.

Output format: Respond with EXACTLY one character — either 0 or 1. No explanation,
no punctuation, no additional text — just the digit.\
"""


#  Helpers 

def _load_split_ids(csv_path: Path) -> set:
    """Return a set of integer participant IDs from a DAIC-WOZ split CSV."""
    df = pd.read_csv(csv_path)
    # The ID column is always the first column but its name varies across files.
    id_col = df.columns[0]
    return set(df[id_col].astype(int).tolist())


def _truncate_text(text: str, max_words: int = MAX_WORDS) -> str:
    """Return text truncated to at most max_words whitespace-separated tokens."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _select_few_shot_examples(train_df: pd.DataFrame, n_per_class: int = 2) -> list:
    """
    Deterministically pick n_per_class examples per class from the training pool.

    Selection strategy: shortest transcript first (fewest words), ties broken by
    ascending participant_id. This keeps the few-shot block as compact as possible
    while still covering both classes.

    Returns:
        List of dicts: {participant_id, text, label}
        Order: non-depressed examples first, then depressed — matches the
        interleaved user/assistant pattern in _build_few_shot_messages().
    """
    examples = []
    for label in [0, 1]:  # non-depressed first so the prompt ends on a depressed example
        subset = train_df[train_df["PHQ_Binary"] == label].copy()
        subset["word_count"] = subset["text"].str.split().str.len()
        subset = subset.sort_values(
            ["word_count", "participant_id"], ascending=True
        ).reset_index(drop=True)
        for _, row in subset.head(n_per_class).iterrows():
            examples.append(
                {
                    "participant_id": int(row["participant_id"]),
                    "text": row["text"],
                    "label": int(label),
                }
            )
    return examples


def _build_zero_shot_messages(transcript: str) -> list:
    """Return the messages list for a zero-shot chat completion call."""
    user_content = (
        "Transcript:\n"
        f"{transcript}\n\n"
        "Classification (0 or 1):"
    )
    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
    ]


def _build_few_shot_messages(transcript: str, examples: list) -> list:
    """
    Return the messages list for a few-shot chat completion call.

    The few-shot demonstrations are embedded as alternating user/assistant turns
    before the actual query, which is the idiomatic way to do few-shot prompting
    with the OpenAI chat format.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i, ex in enumerate(examples, start=1):
        truncated = _truncate_text(ex["text"])
        label_hint = "Depressed (1)" if ex["label"] == 1 else "Non-depressed (0)"
        user_turn = (
            f"[Example {i} — {label_hint}]\n"
            "Transcript:\n"
            f"{truncated}\n\n"
            "Classification (0 or 1):"
        )
        messages.append({"role": "user",      "content": user_turn})
        messages.append({"role": "assistant", "content": str(ex["label"])})

    # Actual query
    query = (
        "Transcript:\n"
        f"{transcript}\n\n"
        "Classification (0 or 1):"
    )
    messages.append({"role": "user", "content": query})
    return messages


def _parse_prediction(response_text: str, participant_id: int) -> int:
    """
    Extract the binary prediction (0 or 1) from a GPT response string.

    Tries word-boundary match first; falls back to any digit in the string.
    Defaults to 0 with a warning when nothing matches.
    """
    text = response_text.strip()

    # Prefer a bare digit at the start of the response
    match = re.search(r"^([01])", text)
    if match:
        return int(match.group(1))

    # Word-boundary match anywhere in the response
    match = re.search(r"\b([01])\b", text)
    if match:
        return int(match.group(1))

    # Last resort: any 0 or 1 character
    match = re.search(r"([01])", text)
    if match:
        return int(match.group(1))

    logger.warning(
        "Participant %d: unparseable response %r — defaulting to 0",
        participant_id,
        text[:120],
    )
    return 0


def _summarise(model_name: str, y_true: list, y_pred: list) -> dict:
    """
    Compute evaluation metrics in the exact format used by all other experiments
    (train_classical.py, train_sentence_transformer.py).
    """
    return {
        "model":            model_name,
        "accuracy":         round(accuracy_score(y_true, y_pred), 4),
        "precision_macro":  round(precision_score(y_true, y_pred, average="macro",   zero_division=0), 4),
        "recall_macro":     round(recall_score(   y_true, y_pred, average="macro",   zero_division=0), 4),
        "f1_macro":         round(f1_score(        y_true, y_pred, average="macro",   zero_division=0), 4),
        "f1_depressed":     round(f1_score(        y_true, y_pred, pos_label=1,       zero_division=0), 4),
        "f1_nondepressed":  round(f1_score(        y_true, y_pred, pos_label=0,       zero_division=0), 4),
    }


def _call_api(
    client,
    messages: list,
    model: str,
    max_tokens: int,
    dry_run: bool,
    participant_id: int,
    condition: str,
) -> int:
    """
    Issue one chat completion request and return the parsed binary prediction.

    In --dry_run mode the prompt is printed to stdout and 0 is returned.
    """
    if dry_run:
        print(f"\n{'='*70}")
        print(f"DRY RUN — {condition} — Participant {participant_id}")
        print(f"{'='*70}")
        for msg in messages:
            role    = msg["role"].upper()
            preview = msg["content"][:400]
            suffix  = "…" if len(msg["content"]) > 400 else ""
            print(f"[{role}]\n{preview}{suffix}\n")
        return 0

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_tokens,
        seed=RANDOM_SEED,    # best-effort reproducibility
    )
    raw_text = response.choices[0].message.content or ""
    return _parse_prediction(raw_text, participant_id)


#  Main 

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT zero-shot and few-shot depression classification — Experiment 9"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model name (default: gpt-4o)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
        help="Maximum tokens in the GPT response (default: 1000)",
    )
    parser.add_argument(
        "--results_dir",
        default="results/gpt",
        help="Directory for saving results CSV and figures (default: results/gpt)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print prompts to stdout without calling the OpenAI API",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Initialise OpenAI client 
    client = None
    if not args.dry_run:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY environment variable is not set.")
            sys.exit(1)
        try:
            from openai import OpenAI
        except ImportError:
            print(
                "ERROR: openai package not found.\n"
                "Install with:  pip install openai>=1.0.0"
            )
            sys.exit(1)
        client = OpenAI(api_key=api_key)

    print("=" * 65)
    print("  DAIC-WOZ GPT Inference — Experiment 9")
    print("=" * 65)
    print(f"  Model        : {args.model}")
    print(f"  Max tokens   : {args.max_tokens}")
    print(f"  Results dir  : {results_dir.resolve()}")
    print(f"  Dry run      : {args.dry_run}")
    print()

    #  Step 1: Load preprocessed dataset 
    print("[1/5] Loading dataset…")
    if not DATA_FILE.exists():
        print(f"ERROR: dataset not found at {DATA_FILE}")
        sys.exit(1)
    dataset = pd.read_csv(DATA_FILE)
    print(f"   Total rows in dataset.csv: {len(dataset)}")

    #  Step 2: Partition using official DAIC-WOZ splits 
    print("\n[2/5] Partitioning data using official split IDs…")
    train_ids      = _load_split_ids(TRAIN_SPLIT_FILE)
    dev_ids        = _load_split_ids(DEV_SPLIT_FILE)
    test_ids       = _load_split_ids(TEST_SPLIT_FILE)
    train_pool_ids = train_ids | dev_ids  # combined train + dev for few-shot sourcing

    train_df = dataset[dataset["participant_id"].isin(train_pool_ids)].reset_index(drop=True)
    test_df  = dataset[dataset["participant_id"].isin(test_ids)].reset_index(drop=True)

    n_dep_test     = int(test_df["PHQ_Binary"].sum())
    n_nondep_test  = len(test_df) - n_dep_test
    print(f"   Train+dev pool : {len(train_df)} participants")
    print(f"   Test set       : {len(test_df)} participants "
          f"(dep={n_dep_test}, non-dep={n_nondep_test})")

    if test_df.empty:
        raise RuntimeError("Test set is empty — verify split CSV paths.")

    #  Step 3: Select few-shot examples 
    print("\n[3/5] Selecting few-shot examples from training pool…")
    few_shot_examples = _select_few_shot_examples(train_df, n_per_class=2)
    print("   Selected examples (shortest transcripts per class):")
    for ex in few_shot_examples:
        wc = len(ex["text"].split())
        tag = "depressed" if ex["label"] == 1 else "non-depressed"
        print(f"     Participant {ex['participant_id']:>4}  label={ex['label']} ({tag})  {wc} words")

    #  Step 4: Run GPT inference 
    n_test = len(test_df)
    print(f"\n[4/5] Running inference ({n_test} participants × 2 conditions)…")
    if args.dry_run:
        print("   *** DRY RUN — no API calls ***")

    y_true   = test_df["PHQ_Binary"].tolist()
    zs_preds = []   # zero-shot predictions
    fs_preds = []   # few-shot predictions

    for i, (_, row) in enumerate(test_df.iterrows(), start=1):
        pid       = int(row["participant_id"])
        truncated = _truncate_text(row["text"])

        #  Zero-shot call 
        zs_messages = _build_zero_shot_messages(truncated)
        zs_pred = _call_api(
            client, zs_messages, args.model, args.max_tokens,
            args.dry_run, pid, "Zero-Shot",
        )
        zs_preds.append(zs_pred)
        if not args.dry_run:
            time.sleep(1)   # rate-limit guard between API calls

        #  Few-shot call 
        fs_messages = _build_few_shot_messages(truncated, few_shot_examples)
        fs_pred = _call_api(
            client, fs_messages, args.model, args.max_tokens,
            args.dry_run, pid, "Few-Shot",
        )
        fs_preds.append(fs_pred)
        if not args.dry_run:
            time.sleep(1)

        if not args.dry_run:
            true_lbl = int(row["PHQ_Binary"])
            print(
                f"   [{i:>3}/{n_test}] pid={pid:>4}  "
                f"true={true_lbl}  zs={zs_pred}  fs={fs_pred}"
            )

    if args.dry_run:
        print("\nDry run complete. No results saved.")
        return

    #  Step 5: Evaluate, print, and save 
    print("\n[5/5] Evaluating and saving results…")

    name_zs = "GPT Zero-Shot"
    name_fs = "GPT Few-Shot"

    print_metrics(name_zs, y_true, zs_preds)
    save_confusion_matrix_plot(name_zs, y_true, zs_preds, output_dir=str(figures_dir))

    print_metrics(name_fs, y_true, fs_preds)
    save_confusion_matrix_plot(name_fs, y_true, fs_preds, output_dir=str(figures_dir))

    # Build per-condition result rows
    rows = [
        {**_summarise(name_zs, y_true, zs_preds), "gpt_model": args.model},
        {**_summarise(name_fs, y_true, fs_preds), "gpt_model": args.model},
    ]
    new_df = pd.DataFrame(rows)

    # Append to existing CSV (overwrite rows for the same condition names on re-run)
    out_csv = results_dir / "gpt_results.csv"
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        existing = existing[~existing["model"].isin([name_zs, name_fs])]
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(out_csv, index=False)

    print(f"\n   Results appended → {out_csv.resolve()}")

    # Final summary table
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    display_cols = ["model", "accuracy", "f1_macro", "f1_depressed", "f1_nondepressed"]
    print(new_df[display_cols].to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    main()
