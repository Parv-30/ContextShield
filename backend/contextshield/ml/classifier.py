from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from contextshield.ml.labels import LABELS, toxicity_score as compute_toxicity_score


class Classifier:
    def __init__(self, model_dir: str | Path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        encodings = self.tokenizer(
            texts, truncation=True, max_length=128, padding=True, return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**encodings).logits
        probs = torch.sigmoid(logits).numpy()

        results = []
        for row in probs:
            label_probs = {label: float(p) for label, p in zip(LABELS, row)}
            results.append(
                {
                    "probabilities": label_probs,
                    "toxicity_score": compute_toxicity_score(label_probs),
                }
            )
        return results
