"""RUL metrics and the training-row guard."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.evaluation.rul_evaluation import nasa_asymmetric_score, regression_metrics
from src.models.rul_model import fit_regressor, predict_rul, target_values


def test_perfect_predictions_have_zero_nasa_score():
    assert nasa_asymmetric_score([40, 10], [40, 10]) == 0.0


def test_late_prediction_is_penalized_more_than_the_same_early_miss():
    late = nasa_asymmetric_score([50], [60])
    early = nasa_asymmetric_score([50], [40])
    assert late == pytest.approx(np.exp(1) - 1)
    assert early == pytest.approx(np.exp(10 / 13) - 1)
    assert late > early


def test_regression_metrics_match_a_hand_calculation():
    metrics = regression_metrics([10, 20], [12, 16])
    assert metrics["n"] == 2
    assert metrics["mae"] == pytest.approx(3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt((4 + 16) / 2))
    assert metrics["mean_error"] == pytest.approx(-1.0)
    assert metrics["n_late"] == 1
    assert metrics["n_early"] == 1


def test_rul_cap_leaves_short_labels_unchanged():
    frame = pd.DataFrame({"rul": [10, 125, 200]})
    assert target_values(frame, None).tolist() == [10, 125, 200]
    assert target_values(frame, 125).tolist() == [10, 125, 125]


def test_fit_rejects_test_engines():
    frame = pd.DataFrame(
        {
            "split": ["test"],
            "source_file": ["test_FD001"],
            "rul": [10],
            "sensor_2": [1.0],
        }
    )
    scaler = {"columns": ["sensor_2"], "mean": {"sensor_2": 0.0}, "std": {"sensor_2": 1.0}}
    with pytest.raises(ValueError, match="train"):
        fit_regressor(Ridge(alpha=1.0), frame, ["sensor_2"], scaler, rul_cap=None)


def test_prediction_does_not_read_the_rul_column():
    rows = []
    for cycle in range(12):
        rows.append(
            {
                "split": "train",
                "source_file": "train_FD001",
                "rul": 30 - cycle,
                "sensor_2": float(cycle),
            }
        )
    frame = pd.DataFrame(rows)
    scaler = {"columns": ["sensor_2"], "mean": {"sensor_2": 0.0}, "std": {"sensor_2": 1.0}}
    model = fit_regressor(Ridge(alpha=1.0), frame, ["sensor_2"], scaler, rul_cap=None)
    altered = frame.copy()
    altered["rul"] = 999
    assert np.allclose(
        predict_rul(model, frame, ["sensor_2"], scaler),
        predict_rul(model, altered, ["sensor_2"], scaler),
    )
