import argparse
import json
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from contextshield.ml.labels import LABELS
from contextshield.ml.train import load_jsonl, MultiLabelDataset, compute_metrics

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_OUT = Path(__file__).resolve().parent.parent / "models" / "classifier"
MODEL_NAME = "xlm-roberta-base"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-examples", type=int, default=None)
    parsed_args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    )

    train_examples = load_jsonl(DATA_DIR / "train.jsonl")
    if parsed_args.max_train_examples is not None:
        train_examples = train_examples[: parsed_args.max_train_examples]
    val_examples = load_jsonl(DATA_DIR / "val.jsonl")
    test_examples = load_jsonl(DATA_DIR / "test.jsonl")

    train_dataset = MultiLabelDataset(train_examples, tokenizer)
    val_dataset = MultiLabelDataset(val_examples, tokenizer)
    test_dataset = MultiLabelDataset(test_examples, tokenizer)

    args = TrainingArguments(
        output_dir=str(MODEL_OUT / "checkpoints"),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_metrics = trainer.evaluate(test_dataset)

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)
    with open(MODEL_OUT / "eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
