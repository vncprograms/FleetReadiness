"""Anomaly scores and synthetic copies leave the source rows unchanged."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from src.data.preprocessing import included_sensors
from src.evaluation.anomaly_evaluation import inject_measurement
from src.models.anomaly_model import score_cycles


def test_extreme_point_scores_as_more_unusual_than_the_training_cloud():
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(80, 2))
    model = IsolationForest(n_estimators=50, contamination=0.05, random_state=0)
    model.fit(cloud)
    frame = pd.DataFrame({"sensor_2": [0.0, 12.0], "sensor_3": [0.0, 12.0]})
    scaler = {
        "columns": ["sensor_2", "sensor_3"],
        "mean": {"sensor_2": 0.0, "sensor_3": 0.0},
        "std": {"sensor_2": 1.0, "sensor_3": 1.0},
    }
    scored = score_cycles(model, frame, ["sensor_2", "sensor_3"], scaler)
    assert scored["anomaly_score"].iloc[1] > scored["anomaly_score"].iloc[0]
    assert scored["anomaly_label"].iloc[1] == "anomalous"


def test_injection_does_not_edit_the_source_engine():
    engine = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 1],
            "time_cycles": [1, 2, 3, 4],
            "op_setting_1": 0.0,
            "op_setting_2": 0.0,
        }
    )
    for sensor in included_sensors():
        engine[sensor] = np.linspace(1.0, 2.0, 4)
    engine["sensor_4"] = [10.0, 10.0, 10.0, 10.0]
    original = engine["sensor_4"].tolist()
    injected = inject_measurement(engine, "sensor_4", start_cycle=3, kind="step", magnitude=25.0)
    assert engine["sensor_4"].tolist() == original
    assert injected.loc[injected["time_cycles"] == 4, "sensor_4"].iloc[0] == pytest.approx(35.0)
