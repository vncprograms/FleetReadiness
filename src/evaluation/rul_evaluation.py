"""RUL regression metrics.

The asymmetric score is Saxena et al. 2008, equation 11. The paper sets
a1 = 10 and a2 = 13, defines d as estimated RUL minus true RUL, and says
late predictions are penalized more than early ones. That pair of statements
holds when a positive d (estimated life too high) is divided by 10 and a
negative d is divided by 13. The score is a sum over units under test, so
this project computes it on one row per engine, not on every cycle.
"""

import numpy as np
import pandas as pd

LATE_DIVISOR = 10.0
EARLY_DIVISOR = 13.0


def nasa_asymmetric_score(y_true, y_pred) -> float:
    """Sum of asymmetric penalties. Zero when every prediction matches the label."""
    truth = np.asarray(y_true, dtype=float)
    estimate = np.asarray(y_pred, dtype=float)
    if truth.shape != estimate.shape:
        raise ValueError(f"shape mismatch: labels {truth.shape}, predictions {estimate.shape}")
    if truth.size == 0:
        raise ValueError("nasa_asymmetric_score received no rows")
    error = estimate - truth
    late = error >= 0
    penalty = np.empty(error.shape, dtype=float)
    penalty[late] = np.exp(error[late] / LATE_DIVISOR) - 1.0
    penalty[~late] = np.exp(-error[~late] / EARLY_DIVISOR) - 1.0
    return float(penalty.sum())


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, signed error, and the NASA sum for one set of predictions."""
    truth = np.asarray(y_true, dtype=float)
    estimate = np.asarray(y_pred, dtype=float)
    if truth.shape != estimate.shape:
        raise ValueError(f"shape mismatch: labels {truth.shape}, predictions {estimate.shape}")
    if truth.size == 0:
        raise ValueError("regression_metrics received no rows")
    error = estimate - truth
    return {
        "n": int(truth.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_error": float(np.mean(error)),
        "n_late": int(np.sum(error > 0)),
        "n_early": int(np.sum(error < 0)),
        "n_negative_predictions": int(np.sum(estimate < 0)),
        "nasa_score": nasa_asymmetric_score(truth, estimate),
        "nasa_score_per_engine": float(nasa_asymmetric_score(truth, estimate) / truth.size),
    }


def rows_at_rul(frame: pd.DataFrame, rul_value: int) -> pd.DataFrame:
    """Rows whose label equals one remaining-life value. One row per full run."""
    matched = frame.loc[frame["rul"] == rul_value]
    counts = matched.groupby(["source_file", "unit_number"]).size()
    if (counts != 1).any():
        raise ValueError(f"expected one row per engine at rul {rul_value}")
    return matched
