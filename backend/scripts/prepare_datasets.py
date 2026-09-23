import csv
import json
import random
from pathlib import Path

from contextshield.ml.dataset import build_examples
from contextshield.ml.labels import (
    map_hasoc_label,
    map_dravidian_label,
    map_hasoc_hindi_label,
    map_davidson_label,
)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
FALLBACK_PATH = Path(__file__).resolve().parent.parent / "data" / "seed_fallback.jsonl"


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def hasoc_label_fn(row: dict) -> list[str]:
    task2 = row.get("task2") or None
    return map_hasoc_label(row["task1"], task2)


def dravidian_label_fn(row: dict) -> list[str]:
    return map_dravidian_label(row["label"])


def hasoc_hindi_label_fn(row: dict) -> list[str]:
    return map_hasoc_hindi_label(row["task1"], row.get("task2"))


def davidson_label_fn(row: dict) -> list[str]:
    return map_davidson_label(int(row["label"]))


def load_all_examples() -> list[dict]:
    examples = []
    hasoc_en = RAW_DIR / "hasoc_en.csv"
    hasoc_hi = RAW_DIR / "hasoc_hi.csv"
    dravidian_ta = RAW_DIR / "dravidian_ta.csv"
    davidson_en_real = RAW_DIR / "davidson_en.csv"
    hasoc_hindi_real = RAW_DIR / "hasoc_hindi_real.csv"
    dravidian_ta_real = RAW_DIR / "dravidian_ta_real.csv"

    found_any = False
    if hasoc_en.exists():
        examples += build_examples(load_csv(hasoc_en), "text", hasoc_label_fn)
        found_any = True
    if hasoc_hi.exists():
        examples += build_examples(load_csv(hasoc_hi), "text", hasoc_label_fn)
        found_any = True
    if dravidian_ta.exists():
        examples += build_examples(load_csv(dravidian_ta), "text", dravidian_label_fn)
        found_any = True
    if davidson_en_real.exists():
        examples += build_examples(load_csv(davidson_en_real), "text", davidson_label_fn)
        found_any = True
    if hasoc_hindi_real.exists():
        examples += build_examples(load_csv(hasoc_hindi_real), "text", hasoc_hindi_label_fn)
        found_any = True
    if dravidian_ta_real.exists():
        examples += build_examples(load_csv(dravidian_ta_real), "text", dravidian_label_fn)
        found_any = True

    if not found_any:
        print(f"No raw datasets found in {RAW_DIR}; using fallback seed set.")
        with open(FALLBACK_PATH, encoding="utf-8") as f:
            examples = [json.loads(line) for line in f if line.strip()]

    return examples


def stratified_split(examples: list[dict], seed: int = 42):
    random.Random(seed).shuffle(examples)

    def has_signal(ex):
        return ex["labels"][:-1] != [0, 0, 0, 0]

    positives = [e for e in examples if has_signal(e)]
    negatives = [e for e in examples if not has_signal(e)]

    def split_group(group):
        n = len(group)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        return group[:n_train], group[n_train : n_train + n_val], group[n_train + n_val :]

    pos_train, pos_val, pos_test = split_group(positives)
    neg_train, neg_val, neg_test = split_group(negatives)

    train = pos_train + neg_train
    val = pos_val + neg_val
    test = pos_test + neg_test
    random.Random(seed).shuffle(train)
    random.Random(seed).shuffle(val)
    random.Random(seed).shuffle(test)
    return train, val, test


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    examples = load_all_examples()
    train, val, test = stratified_split(examples)
    write_jsonl(PROCESSED_DIR / "train.jsonl", train)
    write_jsonl(PROCESSED_DIR / "val.jsonl", val)
    write_jsonl(PROCESSED_DIR / "test.jsonl", test)
    print(f"train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
