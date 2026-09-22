import numpy as np


def windowed_mean(scores: list[float], k: int, w: int = 3) -> float:
    if k < w - 1 or k >= len(scores):
        raise ValueError(f"k={k} out of valid range for w={w}, len={len(scores)}")
    window = scores[k - w + 1 : k + 1]
    return sum(window) / w


def escalation_slope(scores: list[float], k: int, w: int = 3) -> float:
    if k < w - 1 or k >= len(scores):
        raise ValueError(f"k={k} out of valid range for w={w}, len={len(scores)}")
    window = np.array(scores[k - w + 1 : k + 1], dtype=float)
    x = np.arange(w, dtype=float)
    x_mean = x.mean()
    y_mean = window.mean()
    numerator = ((x - x_mean) * (window - y_mean)).sum()
    denominator = ((x - x_mean) ** 2).sum()
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def escalation_score(mean: float, slope: float, alpha: float = 0.6) -> float:
    return alpha * mean + (1 - alpha) * max(slope, 0.0)


def classify_status(score: float, tau_low: float, tau_high: float) -> str:
    if score >= tau_high:
        return "escalating"
    if score >= tau_low:
        return "elevated"
    return "stable"


def compute_timeline(
    scores: list[float],
    w: int = 3,
    alpha: float = 0.6,
    tau_low: float = 0.0,
    tau_high: float = 1.0,
) -> list[dict]:
    if len(scores) < w:
        return []
    timeline = []
    for k in range(w - 1, len(scores)):
        mean = windowed_mean(scores, k, w)
        slope = escalation_slope(scores, k, w)
        score = escalation_score(mean, slope, alpha)
        status = classify_status(score, tau_low, tau_high)
        timeline.append(
            {
                "seq_index": k,
                "windowed_mean": mean,
                "slope": slope,
                "escalation_score": score,
                "status": status,
            }
        )
    return timeline
