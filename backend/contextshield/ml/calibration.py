import numpy as np


def percentile_thresholds(
    scores: list[float], low_pct: float = 60.0, high_pct: float = 85.0
) -> tuple[float, float]:
    arr = np.array(scores, dtype=float)
    tau_low = float(np.percentile(arr, low_pct))
    tau_high = float(np.percentile(arr, high_pct))
    return tau_low, tau_high
