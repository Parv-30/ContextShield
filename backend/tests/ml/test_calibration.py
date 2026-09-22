import pytest
from contextshield.ml.calibration import percentile_thresholds


def test_percentile_thresholds_basic():
    scores = list(range(1, 101))  # 1..100
    tau_low, tau_high = percentile_thresholds(scores, low_pct=60.0, high_pct=85.0)
    assert tau_low == pytest.approx(60.4, abs=0.5)
    assert tau_high == pytest.approx(85.15, abs=0.5)
    assert tau_low < tau_high


def test_percentile_thresholds_constant_scores():
    scores = [0.5] * 10
    tau_low, tau_high = percentile_thresholds(scores)
    assert tau_low == pytest.approx(0.5)
    assert tau_high == pytest.approx(0.5)
