from contextshield.ml.labels import LABELS


def to_multi_hot(labels_list: list[str]) -> list[int]:
    return [1 if label in labels_list else 0 for label in LABELS]


def build_examples(raw_rows: list[dict], text_key: str, label_fn) -> list[dict]:
    examples = []
    for row in raw_rows:
        applicable = label_fn(row)
        examples.append(
            {
                "text": row[text_key],
                "language": row["language"],
                "labels": to_multi_hot(applicable),
            }
        )
    return examples
