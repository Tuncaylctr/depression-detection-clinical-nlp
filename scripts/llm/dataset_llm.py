#!/usr/bin/env python3
"""
dataset_llm.py — PyTorch Dataset for LLM instruction-tuning on DAIC-WOZ.

Formats each participant transcript as an instruction-style prompt and pairs
it with the expected label ("0" or "1") as the generation target.

How this differs from the encoder-only (BERT/XLM-R) approach:
  - Encoder models classify a summary embedding of the whole input.
  - Here we fine-tune a DECODER (causal LM) to *generate* the label token.
  - The full sequence passed to the model is [prompt] + ["0" or "1"] + [eos].
  - We mask out the prompt tokens from the loss (-100) so the model only
    learns to predict the label, not to reproduce its own input.

Usage:
    from dataset_llm import DepressionPromptDataset
    dataset = DepressionPromptDataset(texts, labels, tokenizer, max_length=2048)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in loader:
        input_ids      = batch["input_ids"]      # (1, max_length)
        attention_mask = batch["attention_mask"] # (1, max_length)
        labels         = batch["labels"]         # (1, max_length) — -100 at prompt/pad
        ground_truth   = batch["ground_truth"]   # (1,)  — 0 or 1 for loss weighting
"""

import torch
from torch.utils.data import Dataset


#  Prompt template 
# {transcript} is replaced per-sample. The model is expected to output "0" or
# "1" as its first (and only) token.
#
# Qwen3.5-9B note: this model has a "thinking" mode that outputs a long
# chain-of-thought block <think>...</think> before the answer. We disable
# thinking (enable_thinking=False) so training targets just "0" or "1".

USER_TEMPLATE = """\
Below is a transcript of a patient's spoken responses during a clinical \
interview conducted by a virtual agent. Only the patient's words are shown \
(the interviewer's questions have been removed).

Patient transcript:
{transcript}

Based on the patient's language, emotional tone, and content, is this patient \
clinically depressed (PHQ-8 score ≥ 10)?

Answer with a single digit only — 1 for depressed, 0 for not depressed."""


def _apply_chat_template(processor_or_tokenizer, messages, **kwargs):
    """
    Wrapper around apply_chat_template that adds enable_thinking=False for
    models that support it (Qwen3.5). Falls back silently for models that
    don't recognise the parameter (Gemma-3, older Qwen).
    """
    try:
        return processor_or_tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        return processor_or_tokenizer.apply_chat_template(messages, **kwargs)


class DepressionPromptDataset(Dataset):
    """
    Map-style Dataset that converts participant transcripts into tokenised
    instruction prompts for causal language model (LLM) fine-tuning.

    Each item returned by __getitem__ is a dict with four keys:

      input_ids      : (max_length,) long tensor
          Full tokenised sequence: [prompt tokens] + [label token(s)] + [padding]

      attention_mask : (max_length,) long tensor
          1 for every real token (prompt + label), 0 for padding.
          We track padding via `real_len` rather than comparing to pad_token_id
          because Gemma reuses eos_token_id as pad_token_id, which would
          incorrectly mask natural EOS tokens inside the chat template.

      labels         : (max_length,) long tensor
          Copy of input_ids, but -100 at every prompt position and at every
          padding position. HuggingFace CausalLM ignores positions where
          labels == -100, so loss is computed ONLY at the label token(s).

      ground_truth   : int  (0 or 1)
          The original PHQ_Binary label, returned so the training loop can
          apply a class-specific weight to the loss for this sample.

    Parameters
    
    texts      : list[str]   — raw transcript strings (one per participant)
    labels     : list[int]   — binary depression labels (0 / 1)
    tokenizer  : transformers.PreTrainedTokenizer or AutoProcessor
                 Must support apply_chat_template (Gemma-3 and Qwen3.5 do).
    max_length : int
                 Max token sequence length. Long transcripts are truncated from
                 the END of the transcript, preserving the label suffix.
    """

    def __init__(self, texts, labels, tokenizer, max_length: int = 2048):
        if len(texts) != len(labels):
            raise ValueError(
                f"texts and labels must have the same length "
                f"(got {len(texts)} vs {len(labels)})"
            )

        # Some models (Gemma-3) have no dedicated pad token — they reuse EOS.
        # Setting it here prevents DataLoader collation errors.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.examples      = []
        self.ground_truths = []

        for text, label in zip(texts, labels):
            label_str = str(int(label))   # "0" or "1"

            #  Build tokenised sequences via the model's own chat template 
            # apply_chat_template adds model-specific special tokens automatically.
            # enable_thinking=False tells Qwen3.5 NOT to output a <think>...</think>
            # block before the answer — we want just "0" or "1" as the target.

            full_messages = [
                {"role": "user",      "content": USER_TEMPLATE.format(transcript=text)},
                {"role": "assistant", "content": label_str},
            ]
            # Full sequence including the label token
            full_ids = _apply_chat_template(
                tokenizer,
                full_messages,
                tokenize=True,
                add_generation_prompt=False,  # label is already appended
                return_tensors="pt",
            ).squeeze(0)   # shape (full_seq_len,)

            # Prompt-only sequence (used to locate where the label starts)
            # add_generation_prompt=True appends the assistant-turn opener so the
            # model knows it should produce a response, matching inference behaviour.
            prompt_only = [full_messages[0]]
            prompt_ids = _apply_chat_template(
                tokenizer,
                prompt_only,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).squeeze(0)   # shape (prompt_len,)

            prompt_len = len(prompt_ids)
            real_len   = len(full_ids)     # actual sequence length BEFORE padding

            #  Truncate if the sequence exceeds max_length 
            # Always preserve the label suffix (full_ids[prompt_len:]).
            # We shorten only the prompt (i.e. the end of the transcript text).
            if real_len > max_length:
                label_suffix = full_ids[prompt_len:]          # tokens to predict
                available    = max_length - len(label_suffix)

                if available < 32:
                    raise ValueError(
                        f"max_length={max_length} is too small to fit the prompt "
                        f"template (need at least {len(label_suffix) + 32} tokens). "
                        f"Increase --max_length."
                    )

                full_ids   = torch.cat([prompt_ids[:available], label_suffix])
                prompt_len = available
                real_len   = len(full_ids)

            #  Pad to exactly max_length 
            pad_id  = tokenizer.pad_token_id
            pad_len = max_length - real_len
            if pad_len > 0:
                padding  = torch.full((pad_len,), pad_id, dtype=torch.long)
                full_ids = torch.cat([full_ids, padding])

            #  Attention mask 
            # Constructed from real_len rather than (full_ids != pad_id) to
            # avoid masking natural EOS tokens when pad_id == eos_id (Gemma).
            attention_mask = torch.zeros(max_length, dtype=torch.long)
            attention_mask[:real_len] = 1

            #  Label tensor for loss computation 
            # Clone input_ids, then zero-out positions we don't want to train on.
            label_seq = full_ids.clone()
            label_seq[:prompt_len] = -100   # ignore the user prompt
            label_seq[real_len:]   = -100   # ignore padding tokens

            self.examples.append({
                "input_ids":      full_ids,        # (max_length,)
                "attention_mask": attention_mask,  # (max_length,)
                "labels":         label_seq,       # (max_length,)
            })
            self.ground_truths.append(int(label))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return {**self.examples[idx], "ground_truth": self.ground_truths[idx]}
