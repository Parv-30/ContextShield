import json
import numpy as np
from pathlib import Path

from contextshield.ml.train import load_jsonl, compute_metrics


def test_load_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"text": "a", "language": "en", "labels": [0,0,0,0,1]}\n'
        '{"text": "b", "language": "hi", "labels": [1,0,0,0,0]}\n',
        encoding="utf-8",
    )
    rows = load_jsonl(path)
    assert len(rows) == 2
    assert rows[0]["text"] == "a"
    assert rows[1]["labels"] == [1, 0, 0, 0, 0]


class FakeEvalPred:
    def __init__(self, predictions, label_ids):
        self.predictions = predictions
        self.label_ids = label_ids


def test_compute_metrics_perfect_predictions():
    logits = np.array([[10, -10, -10, -10, -10], [-10, -10, -10, -10, 10]])
    labels = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])
    metrics = compute_metrics(FakeEvalPred(logits, labels))
    assert metrics["macro_f1"] == 1.0
    assert metrics["micro_f1"] == 1.0
