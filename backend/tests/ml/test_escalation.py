import pytest
from contextshield.ml.escalation import (
    windowed_mean,
    escalation_slope,
    escalation_score,
    classify_status,
    compute_timeline,
)


def test_windowed_mean_basic():
    scores = [0.1, 0.2, 0.3, 0.4]
    assert windowed_mean(scores, k=2, w=3) == pytest.approx((0.1 + 0.2 + 0.3) / 3)


def test_windowed_mean_last_window():
    scores = [0.1, 0.2, 0.3, 0.4]
    assert windowed_mean(scores, k=3, w=3) == pytest.approx((0.2 + 0.3 + 0.4) / 3)


def test_windowed_mean_raises_when_k_too_small():
    with pytest.raises(ValueError):
        windowed_mean([0.1, 0.2], k=0, w=3)


def test_escalation_slope_increasing():
    scores = [0.1, 0.2, 0.3]
    slope = escalation_slope(scores, k=2, w=3)
    assert slope == pytest.approx(0.1, abs=1e-9)


def test_escalation_slope_flat():
    scores = [0.5, 0.5, 0.5]
    slope = escalation_slope(scores, k=2, w=3)
    assert slope == pytest.approx(0.0, abs=1e-9)


def test_escalation_slope_decreasing():
    scores = [0.9, 0.5, 0.1]
    slope = escalation_slope(scores, k=2, w=3)
    assert slope == pytest.approx(-0.4, abs=1e-9)


def test_escalation_score_weights_mean_and_positive_slope():
    score = escalation_score(mean=0.5, slope=0.2, alpha=0.6)
    assert score == pytest.approx(0.6 * 0.5 + 0.4 * 0.2)


def test_escalation_score_clamps_negative_slope():
    score = escalation_score(mean=0.5, slope=-0.3, alpha=0.6)
    assert score == pytest.approx(0.6 * 0.5)


def test_classify_status_bands():
    assert classify_status(0.1, tau_low=0.3, tau_high=0.7) == "stable"
    assert classify_status(0.3, tau_low=0.3, tau_high=0.7) == "elevated"
    assert classify_status(0.5, tau_low=0.3, tau_high=0.7) == "elevated"
    assert classify_status(0.7, tau_low=0.3, tau_high=0.7) == "escalating"
    assert classify_status(0.9, tau_low=0.3, tau_high=0.7) == "escalating"


def test_compute_timeline_shape_and_monotonic_case():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    timeline = compute_timeline(scores, w=3, alpha=0.6, tau_low=0.2, tau_high=0.4)
    assert len(timeline) == len(scores) - (3 - 1)
    assert timeline[0]["seq_index"] == 2
    assert timeline[-1]["seq_index"] == 4
    for entry in timeline:
        assert set(entry.keys()) == {
            "seq_index", "windowed_mean", "slope", "escalation_score", "status",
        }
    assert timeline[-1]["status"] == "elevated"


def test_compute_timeline_empty_when_too_short():
    assert compute_timeline([0.1, 0.2], w=3) == []
