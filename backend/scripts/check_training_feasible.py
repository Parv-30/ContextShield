# backend/scripts/check_training_feasible.py
import time
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from contextshield.ml.labels import LABELS

MODEL_NAME = "xlm-roberta-base"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), problem_type="multi_label_classification",
    )
    model.to(device)
    model.eval()

    sample_texts = ["this is a sample sentence for timing"] * 16
    encodings = tokenizer(sample_texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
    encodings = {k: v.to(device) for k, v in encodings.items()}

    start = time.time()
    with torch.no_grad():
        model(**encodings)
    elapsed = time.time() - start

    print(f"Device: {device}")
    print(f"One forward pass of batch=16 took {elapsed:.2f}s on this machine.")
    print("Estimated seconds per training step (fwd+bwd, ~3x forward time):", elapsed * 3)


if __name__ == "__main__":
    main()
