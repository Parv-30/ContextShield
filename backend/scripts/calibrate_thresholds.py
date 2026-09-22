import json
from pathlib import Path

from contextshield.ml.classifier import Classifier
from contextshield.ml.calibration import percentile_thresholds
from contextshield.ml.train import load_jsonl

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "classifier"


def main():
    classifier = Classifier(MODEL_DIR)
    val_examples = load_jsonl(DATA_DIR / "val.jsonl")
    texts = [ex["text"] for ex in val_examples]

    scores = [r["toxicity_score"] for r in classifier.predict_batch(texts)]
    tau_low, tau_high = percentile_thresholds(scores)

    output = {"tau_low": tau_low, "tau_high": tau_high, "low_pct": 60.0, "high_pct": 85.0}
    with open(MODEL_DIR / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(output)


if __name__ == "__main__":
    main()
