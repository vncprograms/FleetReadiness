"""Trailing sensor features for one engine cycle.

A feature at cycle t is computed from that engine's rows with cycle <= t.
Windows are shorter at the start of a run. They are not padded with later
cycles. The first cycle has no previous reading, so its difference is 0.
Rolling standard deviation uses divisor n (ddof=0); one observation has
standard deviation 0.
"""

import pandas as pd

from src.data.ingestion import CYCLE, SETTINGS, UNIT

ROLLING_WINDOWS = (5, 10)


def feature_column_names(
    sensor_columns: list[str],
    setting_columns: list[str] | None = None,
    windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> list[str]:
    """Model inputs, in a stable order. The label and the engine id are omitted."""
    if setting_columns is None:
        setting_columns = list(SETTINGS)
    names = [CYCLE, *setting_columns]
    for sensor in sensor_columns:
        names.append(sensor)
        names.append(f"{sensor}_delta")
        for window in windows:
            names.append(f"{sensor}_roll_mean_{window}")
            names.append(f"{sensor}_roll_std_{window}")
    return names


def add_sensor_features(
    frame: pd.DataFrame,
    sensor_columns: list[str],
    windows: tuple[int, ...] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Return a copy of frame with trailing rolling features and cycle differences."""
    if not sensor_columns:
        raise ValueError("sensor_columns is empty")
    missing = [column for column in [UNIT, CYCLE, *sensor_columns] if column not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if frame[sensor_columns].isna().any().any():
        raise ValueError("sensor columns contain missing values")
    if any(window < 1 for window in windows):
        raise ValueError("rolling windows must be positive")

    ordered = frame.sort_values([UNIT, CYCLE], kind="mergesort")
    pieces: list[pd.DataFrame] = []
    for _, group in ordered.groupby(UNIT, sort=False):
        block = group.copy()
        for sensor in sensor_columns:
            values = block[sensor]
            block[f"{sensor}_delta"] = values.diff().fillna(0.0)
            for window in windows:
                rolling = values.rolling(window=window, min_periods=1)
                block[f"{sensor}_roll_mean_{window}"] = rolling.mean()
                block[f"{sensor}_roll_std_{window}"] = rolling.std(ddof=0)
        pieces.append(block)

    features = pd.concat(pieces, axis=0)
    return features.loc[frame.index]
