from contextshield.ml.dataset import to_multi_hot, build_examples


def test_to_multi_hot_single_label():
    assert to_multi_hot(["offensive"]) == [0, 1, 0, 0, 0]


def test_to_multi_hot_none():
    assert to_multi_hot(["none"]) == [0, 0, 0, 0, 1]


def test_to_multi_hot_multiple_labels():
    result = to_multi_hot(["hate_speech", "threat"])
    assert result == [1, 0, 1, 0, 0]


def test_build_examples_basic():
    raw_rows = [
        {"text": "hello world", "language": "en", "task1": "NOT", "task2": None},
        {"text": "bad text", "language": "hi", "task1": "HOF", "task2": "HATE"},
    ]

    def label_fn(row):
        from contextshield.ml.labels import map_hasoc_label
        return map_hasoc_label(row["task1"], row["task2"])

    examples = build_examples(raw_rows, text_key="text", label_fn=label_fn)
    assert examples == [
        {"text": "hello world", "language": "en", "labels": [0, 0, 0, 0, 1]},
        {"text": "bad text", "language": "hi", "labels": [1, 0, 0, 0, 0]},
    ]
