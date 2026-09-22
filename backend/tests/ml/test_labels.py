import pytest
from contextshield.ml.labels import (
    LABELS,
    toxicity_score,
    map_hasoc_label,
    map_dravidian_label,
)


def test_labels_fixed_schema():
    assert LABELS == ["hate_speech", "offensive", "threat", "harassment", "none"]


def test_toxicity_score_picks_max_non_none():
    probs = {"hate_speech": 0.2, "offensive": 0.8, "threat": 0.1, "harassment": 0.3, "none": 0.05}
    assert toxicity_score(probs) == pytest.approx(0.8)


def test_toxicity_score_all_none():
    probs = {"hate_speech": 0.0, "offensive": 0.0, "threat": 0.0, "harassment": 0.0, "none": 0.99}
    assert toxicity_score(probs) == pytest.approx(0.0)


def test_map_hasoc_not_offensive():
    assert map_hasoc_label("NOT", None) == ["none"]


def test_map_hasoc_hate():
    assert map_hasoc_label("HOF", "HATE") == ["hate_speech"]


def test_map_hasoc_offensive():
    assert map_hasoc_label("HOF", "OFFN") == ["offensive"]


def test_map_hasoc_profane_maps_to_offensive():
    assert map_hasoc_label("HOF", "PRFN") == ["offensive"]


def test_map_dravidian_not_offensive():
    assert map_dravidian_label("Not_offensive") == ["none"]


def test_map_dravidian_untargeted():
    assert map_dravidian_label("Offensive_Untargeted") == ["offensive"]


def test_map_dravidian_targeted_individual_is_harassment():
    assert map_dravidian_label("Offensive_Targeted_Insult_Individual") == ["harassment"]


def test_map_dravidian_targeted_group_is_hate_speech():
    assert map_dravidian_label("Offensive_Targeted_Insult_Group") == ["hate_speech"]


def test_map_dravidian_targeted_other_is_offensive():
    assert map_dravidian_label("Offensive_Targeted_Insult_Other") == ["offensive"]


def test_map_dravidian_not_tamil():
    assert map_dravidian_label("not-Tamil") == ["none"]
