#!/usr/bin/env python3
"""
train_llm.py — LoRA fine-tuning of open-weights LLMs for DAIC-WOZ depression
detection (Experiments 7 & 8 of the thesis).

Supported models (pass via --model_name):
  Experiment 7 : google/gemma-3-12b-it    (~12 B parameters)
  Experiment 8 : Qwen/Qwen3.5-9B         (~9.65 B parameters, vision-language)

What is LoRA?
LoRA (Low-Rank Adaptation) adds small trainable "adapter" matrices to a
frozen pretrained model. Instead of updating all ~7-12 billion weights we
only train ~0.1% of them, which:
  • drastically reduces GPU memory requirements
  • is much faster to train
  • produces results close to full fine-tuning on small datasets

What is 4-bit quantisation?
Quantisation stores model weights in 4-bit integers instead of 16-bit floats,
cutting memory use by ~4×. Combined with LoRA this is called QLoRA.
NOTE: bitsandbytes 4-bit quantisation requires CUDA (NVIDIA GPU). On Apple
Silicon (MPS) the model is loaded in bfloat16 WITHOUT quantisation, which
uses ~2× more memory. 12 B models will likely OOM on 16 GB; the script will
catch this and print a clear message.

Pipeline overview
1.  Clean labels via clean_labels.py (same as other experiments)
2.  Load participant transcripts via preprocess.py
3.  Partition using official DAIC-WOZ splits (train+dev pool / test)
4.  Hold out 15 % of pool as a validation set (stratified, same seed)
5.  Load base model with 4-bit quantisation (CUDA) or bfloat16 (MPS)
6.  Wrap with LoRA adapters via peft
7.  Fine-tune with AdamW; class-weighted loss for the ~1:2.3 imbalance
8.  Early stopping on validation macro-F1 (patience = --patience)
9.  Inference: generate one token and parse "0" or "1"
10. Evaluate best checkpoint on test set; save metrics + confusion matrix
11. Outer loop: repeat steps 5-10 for each (learning rate, LoRA rank) combination

Usage
-----
  # Experiment 7 — Gemma 3 12B  (requires CUDA / Colab)
  python scripts/llm/train_llm.py --model_name google/gemma-3-12b-it

  # Experiment 8 — Qwen3.5 9B  (requires CUDA / Colab)
  python scripts/llm/train_llm.py --model_name Qwen/Qwen3.5-9B

  # Dry run (prints grid without training)
  python scripts/llm/train_llm.py --model_name Qwen/Qwen3.5-9B --dry_run

Output
------
  results/llm/llm_grid_search_results.csv  — one row per (model, lr) run
  results/llm/figures/                     — confusion matrix PNGs
  models/llm/search/<run_name>/            — LoRA adapter weights per run
"""

import os
import sys
import random
import argparse
import warnings
from pathlib import Path
from argparse import Namespace

# Read HF token from environment (set via Colab Secrets or HF_TOKEN env var).
# Falls back to True so huggingface_hub uses its own cached token if available.
_HF_TOKEN = os.environ.get("HF_TOKEN") or True

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

import re

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

#  Local module imports 
# We resolve paths relative to this file so the script works when called from
# any working directory (project root, scripts/llm/, etc.).

_HERE    = Path(__file__).parent                  # scripts/llm/
_SCRIPTS = _HERE.parent                           # scripts/
sys.path.insert(0, str(_HERE))                    # dataset_llm.py
sys.path.insert(0, str(_SCRIPTS / "data"))        # clean_labels.py, preprocess.py
sys.path.insert(0, str(_SCRIPTS / "classical"))   # evaluate.py

from clean_labels import clean_labels
from preprocess   import load_participant_transcripts
from evaluate     import print_metrics, save_confusion_matrix_plot
from dataset_llm  import DepressionPromptDataset, USER_TEMPLATE

warnings.filterwarnings("ignore")

#  Project-level paths 
BASE_DIR  = _HERE.parent.parent          # project root
LABEL_DIR = BASE_DIR / "data" / "labels"
TRANS_DIR = BASE_DIR / "data" / "transcripts"

TRAIN_SPLIT_FILE = LABEL_DIR / "train_split_Depression_AVEC2017.csv"
DEV_SPLIT_FILE   = LABEL_DIR / "dev_split_Depression_AVEC2017.csv"
TEST_SPLIT_FILE  = LABEL_DIR / "full_test_split.csv"

#  Fixed LoRA configuration (not part of the grid search) 
LORA_ALPHA_RATIO = 2     # lora_alpha = LORA_ALPHA_RATIO × lora_r (keeps alpha/r ratio constant)
LORA_DROPOUT     = 0.05
LORA_TARGET_MODS = ["q_proj", "v_proj"]  # attention projection layers

#  Hyperparameter grid 
LR_GRID     = [1e-4, 2e-4, 5e-5]
LORA_R_GRID = [4, 8, 16]   # LoRA rank — controls adapter capacity



# Helper functions shared with the rest of the thesis pipeline
# (identical to train_sentence_transformer.py)


def load_split_ids(csv_path: Path, id_col: str = "Participant_ID") -> set:
    """Return a set of integer participant IDs from a DAIC-WOZ split CSV."""
    df = pd.read_csv(csv_path)
    if id_col not in df.columns:
        id_col = df.columns[0]   # full_test_split.csv uses lowercase
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



# Results persistence


def save_grid_result(run_name: str, model_name: str, lr: float, lora_r: int,
                     best_val_f1: float, metrics: dict,
                     results_dir: Path) -> None:
    """
    Append (or overwrite) one row in llm_grid_search_results.csv.

    Saves immediately after each run so a crash mid-search does not lose
    completed results. If the run_name already exists (resumed run), its
    row is replaced with the new values.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "llm_grid_search_results.csv"

    row = {
        "run_name":     run_name,
        "model":        model_name,
        "lr":           lr,
        "lora_r":       lora_r,
        "best_val_f1":  round(best_val_f1, 4) if best_val_f1 is not None else None,
        **metrics,
    }
    df_new = pd.DataFrame([row])

    if out_csv.exists():
        df_old = pd.read_csv(out_csv)
        df_old = df_old[df_old["run_name"] != run_name]   # remove prior run
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(out_csv, index=False)
    print(f"   Grid results saved → {out_csv.resolve()}")



# Device & model loading


def select_device() -> tuple:
    """
    Choose the best available compute device and whether to use 4-bit quant.

    Returns
    -------
    device         : torch.device
    use_quant      : bool — True only on CUDA (bitsandbytes requires CUDA)
    """
    if torch.cuda.is_available():
        device    = torch.device("cuda")
        use_quant = True
        print(f"   Device: CUDA ({torch.cuda.get_device_name(0)})  →  4-bit quantisation ON")
    elif torch.backends.mps.is_available():
        device    = torch.device("mps")
        use_quant = False
        print("   Device: Apple Silicon MPS  →  4-bit quantisation NOT supported on MPS.")
        print("   Loading in bfloat16 (no quantisation). Large models (≥12 B) may OOM.")
        print("   If you hit OOM, run this script on a CUDA GPU (e.g. Google Colab).")
    else:
        device    = torch.device("cpu")
        use_quant = False
        print("   Device: CPU  →  training will be VERY slow. Consider using Colab.")

    return device, use_quant


def load_base_model_and_tokenizer(model_name: str, device: torch.device,
                                  use_quant: bool):
    """
    Load the pretrained base model and its processor.

    Both Gemma-3-12B and Qwen3.5-9B are vision-language models, so both use
    AutoModelForImageTextToText + AutoProcessor.  For our text-only task the
    vision encoder is never called — we simply pass text-only messages and the
    model works as a normal causal LM.

    On CUDA  : loads with bitsandbytes 4-bit NF4 quantisation (QLoRA).
    On MPS   : loads in bfloat16 without quantisation (may OOM for large models).
    On CPU   : loads in float32 (slow, for debugging only).

    Returns
    -------
    model, processor   (processor.apply_chat_template works like a tokenizer)
    """
    # AutoProcessor wraps both the tokenizer and the image processor.
    # For text-only inputs apply_chat_template behaves identically to a plain tokenizer.
    print(f"   Loading processor: {model_name} …")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, token=_HF_TOKEN)

    # Ensure a pad token exists (Gemma-3 uses EOS as pad by default)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    # Expose pad_token_id and apply_chat_template at the top level so the rest
    # of the code can treat the processor exactly like a tokenizer.
    if not hasattr(processor, "pad_token_id") or processor.pad_token_id is None:
        processor.pad_token_id = processor.tokenizer.pad_token_id
    if not hasattr(processor, "apply_chat_template"):
        processor.apply_chat_template = processor.tokenizer.apply_chat_template

    print(f"   Loading model (AutoModelForImageTextToText): {model_name} …")

    try:
        if use_quant:
            #  QLoRA path (CUDA) 
            # NF4 (NormalFloat4) is the quantisation type recommended by the
            # QLoRA paper (Dettmers et al., 2023).  Double quantisation further
            # reduces memory by quantising the quantisation constants themselves.
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,    # compute in bf16 for speed
                bnb_4bit_use_double_quant=True,           # saves ~0.4 bits/param extra
            )
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",       # distributes across available CUDA devices
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                token=_HF_TOKEN,
            )
        else:
            # Plain bfloat16 path (MPS / CPU) 
            dtype = torch.bfloat16 if device.type in ("mps", "cuda") else torch.float32
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=None,         # we place it manually below
                trust_remote_code=True,
                token=_HF_TOKEN,
            )
            model = model.to(device)

    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg or "oom" in msg:
            print("\n" + "=" * 65)
            print("  ERROR: Out of memory while loading the model.")
            print(f"  Model : {model_name}")
            print(f"  Device: {device}")
            print()
            print("  On Apple Silicon (MPS) bitsandbytes 4-bit quantisation is")
            print("  not available, so the full bfloat16 model must fit in RAM.")
            print("  Qwen3.5-9B requires ~19 GB; Gemma-3-12B requires ~24 GB.")
            print()
            print("  Solutions:")
            print("  1. Run this script on Google Colab (free T4 GPU, 15 GB VRAM)")
            print("     where 4-bit quantisation IS supported.")
            print("  2. Use a smaller model (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
            print("     which fits comfortably in 16 GB unified memory.")
            print("=" * 65)
            sys.exit(1)
        raise   # re-raise unexpected errors

    return model, processor


def apply_lora(model, use_quant: bool, lora_r: int):
    """
    Wrap the base model with LoRA adapters.

    LoRA adds trainable low-rank matrices (rank=lora_r) to the query (q_proj)
    and value (v_proj) projection layers of every self-attention block.
    All other model weights remain frozen.

    Returns the PEFT-wrapped model and prints the number of trainable params.
    """
    # prepare_model_for_kbit_training casts LayerNorm weights to float32
    # (numerical stability) and sets up gradient checkpointing for quantised models.
    if use_quant:
        try:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
            )
        except Exception:
            pass   # newer peft versions handle this automatically

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=LORA_ALPHA_RATIO * lora_r,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODS,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()   # prints e.g. "trainable params: 3,407,872"

    # Gradient checkpointing saves activation memory at the cost of ~20% slower
    # backward pass by recomputing them on demand. Strongly recommended for LLMs.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    return model


# Inference  (generation-based, not classification-head-based)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _apply_chat_template(processor, messages, **kwargs):
    """
    Wrapper that adds enable_thinking=False for Qwen3.5 (suppresses the
    <think>...</think> chain-of-thought block so the model outputs just
    "0" or "1"). Falls back silently for models that don't support the param.
    """
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def run_llm_inference(model, processor, texts: list, device: torch.device,
                      use_quant: bool, max_prompt_len: int = 2000) -> list:
    """
    Generate label predictions ("0" or "1") for a list of transcripts.

    Strategy:
      1. Format each text as a prompt-only message (no assistant label).
      2. Disable thinking mode (enable_thinking=False) for Qwen3.5 so the
         model does not emit a long <think>...</think> block before the digit.
      3. Call model.generate() for up to 64 new tokens with greedy decoding.
         (64 instead of 5 as a safety margin — Qwen3.5 may emit a short
          thinking block even with thinking disabled during the first epochs.)
      4. Strip any residual <think>...</think> from the decoded output.
      5. Extract the first "0" or "1" from the cleaned text.

    Returns
    -------
    list of int (0 or 1), one per input text.
    """
    model.eval()
    predictions = []

    # Determine where model parameters actually live (handles device_map="auto")
    compute_device = next(model.parameters()).device

    with torch.no_grad():
        for text in tqdm(texts, desc="Inference", leave=False):
            messages = [{"role": "user", "content": USER_TEMPLATE.format(transcript=text)}]

            prompt_ids = _apply_chat_template(
                processor,
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )  # shape: (1, prompt_len)

            # Truncate from the START (keep the end / instruction) if too long
            if prompt_ids.shape[1] > max_prompt_len:
                prompt_ids = prompt_ids[:, -max_prompt_len:]

            prompt_ids     = prompt_ids.to(compute_device)
            attention_mask = torch.ones_like(prompt_ids)

            output_ids = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,    # enough room even if model emits a short preamble
                do_sample=False,      # greedy decoding — deterministic
                pad_token_id=processor.pad_token_id,
            )

            # Decode only the newly generated tokens (after the prompt)
            generated_ids  = output_ids[0][prompt_ids.shape[1]:]
            generated_text = processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()

            # Remove any <think>...</think> block (Qwen3.5 thinking mode)
            generated_text = _THINK_RE.sub("", generated_text).strip()

            # Extract first "0" or "1" from the cleaned text
            pred = 0   # safe default (non-depressed)
            found = False
            for char in generated_text:
                if char in ("0", "1"):
                    pred = int(char)
                    found = True
                    break
            if not found:
                print(f"\n    [WARN] Unexpected model output: '{generated_text}' → defaulting to 0")

            predictions.append(pred)

    return predictions



# Single training run

def train_single(model_name: str, lr: float, lora_r: int, run_name: str,
                 X_train, y_train, X_val, y_val, X_test, y_test,
                 args: Namespace, device: torch.device,
                 use_quant: bool, adapter_dir: Path,
                 results_dir: Path, figures_dir: Path) -> dict:
    """
    Fine-tune one (model, learning_rate) combination and evaluate on test.

    Steps:
      1. Load base model + tokenizer
      2. Apply LoRA adapters
      3. Train for up to `args.epochs` epochs with early stopping
      4. Load best checkpoint, run test inference, save metrics

    Returns dict with keys: best_val_f1, accuracy, macro_f1, f1_depressed, …
    """
    #  Set random seeds for reproducibility 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    #  Load model 
    model, processor = load_base_model_and_tokenizer(model_name, device, use_quant)
    model = apply_lora(model, use_quant, lora_r)

    #  Dataset & DataLoader 
    print(f"\n   Building datasets (max_length={args.max_length}) …")
    print("   This step tokenises all transcripts once — may take a few minutes.")

    train_ds = DepressionPromptDataset(X_train, y_train, processor, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    #  Optimiser & LR scheduler 
    # AdamW is standard for transformer fine-tuning (decoupled weight decay).
    # The scheduler linearly warms up for 10% of steps then linearly decays,
    # which helps prevent large early gradient steps from destabilising the adapters.
    optimizer   = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    #  Determine compute device for moving tensors 
    # With device_map="auto" (quantised CUDA) the model may be spread across
    # multiple GPUs. next(model.parameters()).device gives us the primary device.
    compute_device = next(model.parameters()).device

    #  Training loop 
    print(f"\n   Training  lr={lr}  epochs={args.epochs}  patience={args.patience}")

    best_val_f1       = -1.0
    epochs_no_improve = 0
    epoch_log         = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_steps    = 0

        bar = tqdm(train_loader, desc=f"Epoch {epoch:>2d}/{args.epochs}", leave=True)

        for batch in bar:
            input_ids      = batch["input_ids"].to(compute_device)
            attention_mask = batch["attention_mask"].to(compute_device)
            labels         = batch["labels"].to(compute_device)
            ground_truth   = int(batch["ground_truth"].item())   # 0 or 1

            optimizer.zero_grad()

            # Forward pass: HuggingFace CausalLM computes cross-entropy loss
            # internally, but only at positions where labels != -100.
            # Because we masked the prompt, loss is computed at the label
            # token position only (i.e. predicting "0" or "1").
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss   # scalar — mean NLL at unmasked positions

            #  Class-weighted loss 
            # With batch_size=1 we can scale the entire loss by the class weight
            # of this sample. This gives the minority (depressed) class a
            # proportionally higher training signal, countering the ~1:2.3 skew.
            weight = args.depressed_weight if ground_truth == 1 else 1.0
            loss   = loss * weight

            # Guard against NaN (can occur on MPS with bfloat16 edge cases)
            if torch.isnan(loss):
                print(f"\n    [WARN] NaN loss at epoch {epoch} — skipping batch")
                optimizer.zero_grad()
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_steps    += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_steps, 1)

        #  Validation: generate predictions and compute macro-F1 
        y_val_pred = run_llm_inference(model, processor, X_val, device, use_quant,
                                       max_prompt_len=args.max_length - 10)
        val_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

        print(f"  Epoch {epoch:>2d}/{args.epochs}  "
              f"train_loss={avg_loss:.4f}  val_macro_f1={val_f1:.4f}")

        epoch_log.append({
            "epoch":        epoch,
            "train_loss":   round(avg_loss, 6),
            "val_macro_f1": round(val_f1, 6),
        })

        #  Early stopping & checkpoint 
        if val_f1 > best_val_f1:
            best_val_f1       = val_f1
            epochs_no_improve = 0
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(adapter_dir))       # saves LoRA weights only
            processor.save_pretrained(str(adapter_dir))  # saves tokenizer/processor config
            print(f"    Checkpoint saved → {adapter_dir}  (val macro-F1={best_val_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"    No improvement for {epochs_no_improve}/{args.patience} epoch(s)")
            if epochs_no_improve >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val macro-F1={best_val_f1:.4f})")
                break

    # Save per-epoch metrics for learning-curve plots
    epoch_csv = results_dir / f"epoch_metrics_{run_name}.csv"
    pd.DataFrame(epoch_log).to_csv(epoch_csv, index=False)
    print(f"   Epoch metrics saved → {epoch_csv}")

    #  Test evaluation using the best checkpoint 
    print(f"\n   Loading best LoRA adapter from {adapter_dir} …")

    # We must reload the base model before applying the saved adapter,
    # because the in-memory model now has weights from the last epoch.
    del model   # free GPU/MPS memory before loading again
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_base, best_proc = load_base_model_and_tokenizer(model_name, device, use_quant)
    best_model = PeftModel.from_pretrained(best_base, str(adapter_dir))
    best_model.eval()

    y_test_pred = run_llm_inference(best_model, best_proc, X_test, device, use_quant,
                                    max_prompt_len=args.max_length - 10)

    print_metrics(run_name, y_test, y_test_pred)
    save_confusion_matrix_plot(run_name, y_test, y_test_pred, output_dir=str(figures_dir))

    metrics = compute_metrics(y_test, y_test_pred)
    return {"best_val_f1": round(best_val_f1, 4), **metrics}



# Grid search (outer loop over learning rates)

def run_grid_search(args: Namespace) -> None:
    """
    Run the full learning-rate grid search for the given model.

    For each learning rate in LR_GRID:
      • Skip if already completed (allows safe resume after a crash)
      • Train and evaluate
      • Append results to results/llm/llm_grid_search_results.csv immediately

    After all runs, print a ranked summary table.
    """
    device, use_quant = select_device()

    # Resolve output directories
    results_dir = BASE_DIR / args.results_dir
    models_dir  = BASE_DIR / args.output_dir
    figures_dir = results_dir / "figures"
    search_csv  = results_dir / "llm_grid_search_results.csv"

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Short model name used in run IDs and filenames (no slashes / spaces)
    model_short = args.model_name.split("/")[-1].lower().replace("-", "_")

    #  Load data once (all runs share the same train/val/test split) 
    print("\n[1/2] Preparing dataset …")

    labels_df = clean_labels(
        label_dir=str(LABEL_DIR),
        output_dir=str(BASE_DIR / "data" / "processed"),
        output_filename="labels.csv",
        verbose=True,
    )
    dataset = load_participant_transcripts(TRANS_DIR, labels_df, verbose=True)

    if dataset.empty:
        raise RuntimeError("Dataset is empty — check transcript and label paths.")

    print("\n[2/2] Partitioning using official DAIC-WOZ splits …")
    train_ids = load_split_ids(TRAIN_SPLIT_FILE)
    dev_ids   = load_split_ids(DEV_SPLIT_FILE)
    test_ids  = load_split_ids(TEST_SPLIT_FILE)

    # Combine train + dev as the training pool (consistent with other experiments)
    pool_ids = train_ids | dev_ids

    pool_df = dataset[dataset["participant_id"].isin(pool_ids)].reset_index(drop=True)
    test_df = dataset[dataset["participant_id"].isin(test_ids)].reset_index(drop=True)

    X_pool = pool_df["text"].tolist()
    y_pool = pool_df["PHQ_Binary"].tolist()
    X_test = test_df["text"].tolist()
    y_test = test_df["PHQ_Binary"].tolist()

    # Stratified 15 % validation hold-out (same split logic as other experiments)
    X_train, X_val, y_train, y_val = train_test_split(
        X_pool, y_pool,
        test_size=0.15,
        stratify=y_pool,
        random_state=args.seed,
    )

    print(f"   Train : {len(X_train):3d}  (dep={sum(y_train)}, non-dep={len(y_train)-sum(y_train)})")
    print(f"   Val   : {len(X_val):3d}  (dep={sum(y_val)}, non-dep={len(y_val)-sum(y_val)})")
    print(f"   Test  : {len(X_test):3d}  (dep={sum(y_test)}, non-dep={len(y_test)-sum(y_test)})")

    #  Load any prior results so a crashed search can be resumed 
    done_run_names: set = set()
    if search_csv.exists():
        prior = pd.read_csv(search_csv)
        done_run_names = set(prior["run_name"].tolist())
        print(f"\n   Resuming — {len(done_run_names)} run(s) already completed: "
              f"{done_run_names}")

    total = len(LR_GRID) * len(LORA_R_GRID)

    print("\n" + "=" * 65)
    print(f"  LLM LoRA Grid Search — {args.model_name}")
    print("=" * 65)
    print(f"  Learning rates  : {LR_GRID}")
    print(f"  LoRA ranks      : {LORA_R_GRID}  (alpha = {LORA_ALPHA_RATIO}×r)")
    print(f"  Fixed           : epochs={args.epochs}  batch_size=1  "
          f"depressed_weight={args.depressed_weight}  patience={args.patience}")
    print(f"  LoRA            : dropout={LORA_DROPOUT}  target={LORA_TARGET_MODS}")
    print("=" * 65)

    if args.dry_run:
        print("\n[DRY RUN] Runs that would be executed:")
        i = 0
        for lr in LR_GRID:
            for lora_r in LORA_R_GRID:
                i += 1
                rn = f"{model_short}_lr{lr}_r{lora_r}"
                status = "SKIP" if rn in done_run_names else "RUN"
                print(f"  {i:>2d}/{total}  [{status}]  {rn}  lr={lr}  r={lora_r}")
        return

    rows = []   # accumulates results for the final summary

    i = 0
    for lr in LR_GRID:
        for lora_r in LORA_R_GRID:
            i += 1
            run_name    = f"{model_short}_lr{lr}_r{lora_r}"
            adapter_dir = models_dir / "search" / run_name

            print(f"\n{'=' * 65}")
            print(f"[{i:>2d}/{total}]  {run_name}  |  lr={lr}  r={lora_r}")
            print("=" * 65)

            if run_name in done_run_names:
                print(f"  SKIP — already completed in a prior run.")
                continue

            try:
                result = train_single(
                    model_name   = args.model_name,
                    lr           = lr,
                    lora_r       = lora_r,
                    run_name     = run_name,
                    X_train      = X_train,
                    y_train      = y_train,
                    X_val        = X_val,
                    y_val        = y_val,
                    X_test       = X_test,
                    y_test       = y_test,
                    args         = args,
                    device       = device,
                    use_quant    = use_quant,
                    adapter_dir  = adapter_dir,
                    results_dir  = results_dir,
                    figures_dir  = figures_dir,
                )
            except Exception as exc:
                print(f"\n  [ERROR] Run {run_name} failed: {exc}")
                result = {
                    "best_val_f1":      None,
                    "accuracy":         None,
                    "macro_precision":  None,
                    "macro_recall":     None,
                    "macro_f1":         None,
                    "f1_depressed":     None,
                    "f1_non_depressed": None,
                }

            # Save immediately so a crash on the next run doesn't lose this result
            save_grid_result(run_name, args.model_name, lr, lora_r,
                             result["best_val_f1"], result, results_dir)
            rows.append({"run_name": run_name, "lr": lr, "lora_r": lora_r, **result})
            done_run_names.add(run_name)

    #  Final summary table 
    if not rows:
        print("\n  All runs were already completed. See:", search_csv)
        return

    df_summary = pd.DataFrame(rows).sort_values("best_val_f1", ascending=False)

    print("\n" + "=" * 65)
    print("  GRID SEARCH RESULTS — ranked by validation macro-F1")
    print("=" * 65)
    cols = ["run_name", "lr", "lora_r", "best_val_f1", "macro_f1", "f1_depressed", "f1_non_depressed"]
    print(df_summary[cols].to_string(index=False))

    best = df_summary.iloc[0]
    print(f"\n  Best configuration:")
    print(f"    lr={best['lr']}  lora_r={int(best['lora_r'])}  "
          f"val_macro_f1={best['best_val_f1']}  "
          f"test_macro_f1={best['macro_f1']}  f1_depressed={best['f1_depressed']}")
    print(f"\n  Full results saved → {search_csv.resolve()}")



# CLI

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LoRA fine-tuning of open-weights LLMs for DAIC-WOZ depression detection.\n\n"
            "Examples:\n"
            "  # Experiment 7 — Gemma 3 12B (requires CUDA):\n"
            "  python scripts/llm/train_llm.py --model_name google/gemma-3-12b-it\n\n"
            "  # Experiment 8 — Qwen3.5 9B (requires CUDA):\n"
            "  python scripts/llm/train_llm.py --model_name Qwen/Qwen3.5-9B"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_name", required=True,
        help=(
            "HuggingFace model ID.  "
            "Exp 7: google/gemma-3-12b-it  |  Exp 8: Qwen/Qwen3.5-9B"
        ),
    )
    parser.add_argument(
        "--max_length", type=int, default=2048,
        help=(
            "Max token length for the [prompt + label] sequence. "
            "Transcripts that produce longer sequences are truncated. "
            "Reduce to 1024 if you hit OOM during dataset construction. (default: 2048)"
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Max training epochs per run. (default: 3)",
    )
    parser.add_argument(
        "--depressed_weight", type=float, default=2.3,
        help=(
            "Loss weight for depressed samples (label=1). "
            "Set to the class ratio ~2.3 to counter the dataset imbalance. (default: 2.3)"
        ),
    )
    parser.add_argument(
        "--patience", type=int, default=2,
        help="Early stopping patience in epochs based on val macro-F1. (default: 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for data splitting and weight initialisation. (default: 42)",
    )
    parser.add_argument(
        "--output_dir", default="models/llm",
        help="Directory under project root to save LoRA adapters. (default: models/llm)",
    )
    parser.add_argument(
        "--results_dir", default="results/llm",
        help="Directory under project root to save CSVs and plots. (default: results/llm)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print the grid of runs without actually training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_grid_search(parse_args())
