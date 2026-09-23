import csv
from pathlib import Path

from datasets import load_dataset

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

DAVIDSON_LIMIT = 6000
HASOC_HI_LIMIT = None
DRAVIDIAN_TA_LIMIT = 6000


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch_davidson_en():
    ds = load_dataset("krishan-CSE/Davidson_Hate_Speech", split="train")
    if DAVIDSON_LIMIT is not None:
        ds = ds.select(range(min(DAVIDSON_LIMIT, len(ds))))
    rows = [
        {"text": row["text"], "language": "en", "label": row["labels"]}
        for row in ds
        if row["text"] and row["text"].strip()
    ]
    write_csv(RAW_DIR / "davidson_en.csv", rows, ["text", "language", "label"])
    print(f"davidson_en: {len(rows)} rows")


def fetch_hasoc_hindi():
    ds = load_dataset("nikitadesai/hasoc", split="train")
    if HASOC_HI_LIMIT is not None:
        ds = ds.select(range(min(HASOC_HI_LIMIT, len(ds))))
    rows = [
        {
            "text": row["cleaned Tweets"] or row["rawtweets"],
            "language": "hi",
            "task1": row["binary class"],
            "task2": row["Context class"],
        }
        for row in ds
        if (row["cleaned Tweets"] or row["rawtweets"])
    ]
    write_csv(RAW_DIR / "hasoc_hindi_real.csv", rows, ["text", "language", "task1", "task2"])
    print(f"hasoc_hindi_real: {len(rows)} rows")


def fetch_dravidian_tamil():
    ds = load_dataset("community-datasets/offenseval_dravidian", "tamil", split="train")
    if DRAVIDIAN_TA_LIMIT is not None:
        ds = ds.select(range(min(DRAVIDIAN_TA_LIMIT, len(ds))))
    label_names = ds.features["label"].names
    rows = [
        {"text": row["text"], "language": "ta", "label": label_names[row["label"]]}
        for row in ds
        if row["text"] and row["text"].strip()
    ]
    write_csv(RAW_DIR / "dravidian_ta_real.csv", rows, ["text", "language", "label"])
    print(f"dravidian_ta_real: {len(rows)} rows")


def main():
    fetch_davidson_en()
    fetch_hasoc_hindi()
    fetch_dravidian_tamil()


if __name__ == "__main__":
    main()
