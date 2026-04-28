#!/usr/bin/env python3
"""
dataset_st.py — PyTorch Dataset wrapper for sentence-transformer fine-tuning.

Provides DepressionDataset, a map-style Dataset that tokenises raw transcript
strings into fixed-length input tensors for use with HuggingFace
AutoModelForSequenceClassification.

Usage:
    from dataset_st import DepressionDataset
    from torch.utils.data import DataLoader

    dataset = DepressionDataset(texts, labels, tokenizer, max_length=512)
    loader  = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch in loader:
        input_ids      = batch["input_ids"]       # (B, max_length)
        attention_mask = batch["attention_mask"]  # (B, max_length)
        labels         = batch["label"]           # (B,)
"""

import torch
from torch.utils.data import Dataset


class DepressionDataset(Dataset):
    """
    Map-style PyTorch Dataset for binary depression classification.

    Each item is a dict containing the tokenised representation of one
    participant transcript along with its binary PHQ-8 label.

    Parameters:
    texts      : List[str]
        Raw participant transcript strings (one string per participant).
    labels     : List[int]
        Binary labels — 0 for non-depressed, 1 for depressed.
    tokenizer  : transformers.PreTrainedTokenizer
        HuggingFace tokenizer compatible with the target model.
    max_length : int
        Maximum token sequence length; inputs longer than this are
        truncated; shorter inputs are padded to this length.

    Notes:
    ``return_tensors='pt'`` adds a leading batch dimension of size 1.
    We call ``.squeeze(0)`` on ``input_ids`` and ``attention_mask`` so
    that DataLoader can collate individual items into a proper batch of
    shape (batch_size, max_length) without a spurious extra dimension.
    """

    def __init__(self, texts, labels, tokenizer, max_length: int = 512):
        if len(texts) != len(labels):
            raise ValueError(
                f"texts and labels must have the same length "
                f"(got {len(texts)} vs {len(labels)})"
            )
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        # squeeze the batch dimension introduced by return_tensors='pt'
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),       # (max_length,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (max_length,)
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }
