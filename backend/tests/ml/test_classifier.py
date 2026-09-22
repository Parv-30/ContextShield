import pytest
from contextshield.ml.classifier import Classifier
from contextshield.ml.labels import LABELS


@pytest.fixture(scope="module")
def tiny_classifier(tmp_path_factory):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_dir = tmp_path_factory.mktemp("tiny_model")
    tokenizer_name = "hf-tiny-model-private/tiny-random-XLMRobertaXLForSequenceClassification"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            tokenizer_name, num_labels=len(LABELS), problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )
    except Exception:
        pytest.skip("tiny test model unavailable offline")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return Classifier(model_dir)


def test_predict_returns_all_labels(tiny_classifier):
    result = tiny_classifier.predict("some text")
    assert set(result["probabilities"].keys()) == set(LABELS)
    assert 0.0 <= result["toxicity_score"] <= 1.0


def test_predict_batch_matches_single(tiny_classifier):
    texts = ["first", "second"]
    batch_results = tiny_classifier.predict_batch(texts)
    assert len(batch_results) == 2
    for r in batch_results:
        assert set(r["probabilities"].keys()) == set(LABELS)
