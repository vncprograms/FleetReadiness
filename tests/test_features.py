"""Trailing window features stay inside the current engine and the current cycle."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import CYCLE, UNIT, load_fd001
from src.features.feature_engineering import add_sensor_features, feature_column_names

ROOT = Path(__file__).resolve().parents[1]


def _series(unit: int, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIT: unit,
            CYCLE: np.arange(1, len(values) + 1),
            "sensor_2": values,
        }
    )


def test_difference_and_short_windows_use_only_rows_seen_so_far():
    frame = _series(1, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    features = add_sensor_features(frame, ["sensor_2"])

    assert features["sensor_2_delta"].tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert features.loc[features[CYCLE] == 1, "sensor_2_roll_mean_5"].iloc[0] == 1.0
    assert features.loc[features[CYCLE] == 1, "sensor_2_roll_std_5"].iloc[0] == 0.0
    assert features.loc[features[CYCLE] == 3, "sensor_2_roll_mean_5"].iloc[0] == pytest.approx(2.0)
    assert features.loc[features[CYCLE] == 6, "sensor_2_roll_mean_5"].iloc[0] == pytest.approx(4.0)
    assert features.loc[features[CYCLE] == 3, "sensor_2_roll_std_5"].iloc[0] == pytest.approx(
        float(np.std([1.0, 2.0, 3.0], ddof=0))
    )
    assert features.loc[features[CYCLE] == 6, "sensor_2_roll_mean_10"].iloc[0] == pytest.approx(3.5)


def test_deleting_future_cycles_does_not_change_the_current_row():
    frame = _series(1, [1.0, 3.0, 2.0, 8.0, 5.0, 9.0, 4.0, 7.0])
    full = add_sensor_features(frame, ["sensor_2"])
    columns = [column for column in full.columns if column.startswith("sensor_2")]
    for cycle in (1, 2, 5, 8):
        prefix = frame.loc[frame[CYCLE] <= cycle].copy()
        partial = add_sensor_features(prefix, ["sensor_2"])
        pd.testing.assert_frame_equal(
            full.loc[full[CYCLE] == cycle, columns].reset_index(drop=True),
            partial.loc[partial[CYCLE] == cycle, columns].reset_index(drop=True),
        )


def test_one_engine_does_not_borrow_another_engines_history():
    combined = pd.concat(
        [_series(1, [100.0] * 6), _series(2, [0.0, 1.0, 2.0])],
        ignore_index=True,
    )
    features = add_sensor_features(combined, ["sensor_2"])
    engine_two = features.loc[features[UNIT] == 2]
    assert engine_two.loc[engine_two[CYCLE] == 1, "sensor_2_roll_mean_10"].iloc[0] == 0.0
    assert engine_two.loc[engine_two[CYCLE] == 1, "sensor_2_delta"].iloc[0] == 0.0
    assert engine_two.loc[engine_two[CYCLE] == 3, "sensor_2_roll_mean_5"].iloc[0] == pytest.approx(1.0)


def test_row_order_does_not_change_feature_values():
    frame = _series(4, [2.0, 4.0, 6.0, 8.0, 10.0])
    forward = add_sensor_features(frame, ["sensor_2"])
    backward = add_sensor_features(frame.iloc[::-1].copy(), ["sensor_2"])
    pd.testing.assert_series_equal(
        forward["sensor_2_roll_mean_5"].sort_index(),
        backward["sensor_2_roll_mean_5"].sort_index(),
    )
    pd.testing.assert_series_equal(
        forward["sensor_2_delta"].sort_index(),
        backward["sensor_2_delta"].sort_index(),
    )


def test_feature_names_omit_the_label():
    names = feature_column_names(["sensor_4", "sensor_11"])
    assert names[:4] == ["time_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
    assert "sensor_4_roll_mean_5" in names
    assert "sensor_4_roll_std_10" in names
    assert "sensor_11_delta" in names
    assert "rul" not in names
    assert "sensor_1" not in names


def test_fd001_unit_features_ignore_later_cycles():
    train, _, _ = load_fd001(ROOT / "data" / "raw")
    engine = train.loc[train[UNIT] == 1].copy()
    sensors = ["sensor_4", "sensor_11"]
    full = add_sensor_features(engine, sensors)
    columns = [column for column in full.columns if column.startswith("sensor_")]
    for cycle in (1, 5, 15, int(engine[CYCLE].max())):
        prefix = engine.loc[engine[CYCLE] <= cycle]
        partial = add_sensor_features(prefix, sensors)
        pd.testing.assert_frame_equal(
            full.loc[full[CYCLE] == cycle, columns].reset_index(drop=True),
            partial.loc[partial[CYCLE] == cycle, columns].reset_index(drop=True),
        )
