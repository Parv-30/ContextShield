import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset

from contextshield.ml.labels import LABELS


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class MultiLabelDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        encoding = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(example["labels"], dtype=torch.float),
        }


def compute_metrics(eval_pred) -> dict:
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=1)),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=1)),
    }
    per_label = f1_score(labels, preds, average=None, zero_division=1)
    for label, score in zip(LABELS, per_label):
        metrics[f"f1_{label}"] = float(score)
    return metrics
