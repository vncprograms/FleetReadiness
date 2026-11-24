"""Labels, engine splits, and the FD001 build."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.ingestion import CYCLE, SENSORS, UNIT, load_fd001
from src.data.preprocessing import (
    FD001_EXCLUDED_SENSORS,
    add_test_rul,
    add_training_rul,
    apply_standardizer,
    assert_fd001_exclusions,
    build_fd001_dataset,
    fit_standardizer,
    included_sensors,
    split_training_engines,
    validate_cycles,
)
from src.features.feature_engineering import add_sensor_features

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def _engine(unit: int, values: list[float]) -> pd.DataFrame:
    rows = []
    for cycle, value in enumerate(values, start=1):
        row = {UNIT: unit, CYCLE: cycle}
        for setting_index in range(1, 4):
            row[f"op_setting_{setting_index}"] = 0.0
        for sensor in SENSORS:
            if sensor in FD001_EXCLUDED_SENSORS:
                row[sensor] = 1.0
            else:
                row[sensor] = value + float(sensor.split("_")[1])
        rows.append(row)
    return pd.DataFrame(rows)


def test_validate_cycles_rejects_a_gap():
    frame = _engine(1, [1.0, 2.0, 3.0])
    frame.loc[1, CYCLE] = 4
    with pytest.raises(ValueError, match="not 1..N"):
        validate_cycles(frame, "synthetic")


def test_training_rul_is_cycles_remaining_until_the_final_row():
    frame = _engine(7, [1.0, 2.0, 4.0])
    labeled = add_training_rul(frame)
    assert labeled["rul"].tolist() == [2, 1, 0]


def test_test_rul_uses_the_file_value_only_after_the_last_observed_cycle():
    frame = _engine(1, [5.0, 6.0, 7.0])
    labeled = add_test_rul(frame, pd.Series([10]))
    assert labeled["rul"].tolist() == [12, 11, 10]


def test_engine_split_is_reproducible_and_does_not_share_engines():
    units = list(range(1, 101))
    first_train, first_val = split_training_engines(units, validation_fraction=0.2, seed=42)
    second_train, second_val = split_training_engines(units, validation_fraction=0.2, seed=42)
    assert first_train == second_train
    assert first_val == second_val
    assert first_train.isdisjoint(first_val)
    assert first_train | first_val == set(units)
    assert len(first_val) == 20
    assert len(first_train) == 80
    other_train, _ = split_training_engines(units, validation_fraction=0.2, seed=7)
    assert other_train != first_train


def test_exclusion_list_must_match_the_constant_channels():
    train = _engine(1, [1.0, 2.0])
    test = _engine(1, [3.0, 4.0])
    assert_fd001_exclusions(train, test)

    varying = train.copy()
    varying[FD001_EXCLUDED_SENSORS[0]] = [0.0, 1.0]
    with pytest.raises(ValueError, match="differ"):
        assert_fd001_exclusions(varying, test)


def test_an_unlisted_constant_sensor_fails_the_exclusion_check():
    train = _engine(1, [1.0, 2.0])
    test = _engine(1, [3.0, 4.0])
    train["sensor_2"] = 5.0
    test["sensor_2"] = 5.0
    with pytest.raises(ValueError, match="unexpected constants"):
        assert_fd001_exclusions(train, test)


def test_standardizer_ignores_validation_rows_and_centers_training_rows():
    frame = pd.DataFrame(
        {
            "sensor_2": [0.0, 2.0, 4.0, 100.0],
            "split": ["train", "train", "train", "validation"],
        }
    )
    scaler = fit_standardizer(frame, ["sensor_2"], frame["split"].eq("train"))
    assert scaler["mean"]["sensor_2"] == pytest.approx(2.0)
    training = frame.loc[frame["split"].eq("train")].copy()
    scaled = apply_standardizer(training, scaler)
    assert scaled["sensor_2"].mean() == pytest.approx(0.0, abs=1e-8)
    assert scaled["sensor_2"].std(ddof=0) == pytest.approx(1.0, abs=1e-8)


def test_build_writes_processed_files_and_leaves_raw_bytes_unchanged(tmp_path):
    raw_file = RAW / "train_FD001.txt"
    before = hashlib.md5(raw_file.read_bytes()).hexdigest()
    dataset_path = build_fd001_dataset(RAW, tmp_path)
    after = hashlib.md5(raw_file.read_bytes()).hexdigest()
    assert before == after

    dataset = pd.read_csv(dataset_path)
    manifest = json.loads((tmp_path / "fd001_split.json").read_text())
    features = json.loads((tmp_path / "fd001_feature_columns.json").read_text())
    scaler = json.loads((tmp_path / "fd001_scaler.json").read_text())

    assert set(dataset["split"]) == {"train", "validation", "test"}
    assert len(manifest["train_units"]) == 80
    assert len(manifest["validation_units"]) == 20
    assert set(manifest["train_units"]).isdisjoint(manifest["validation_units"])
    train_units = set(manifest["train_units"])
    validation_units = set(manifest["validation_units"])
    for (source_file, unit), group in dataset.groupby(["source_file", "unit_number"]):
        labels = set(group["split"])
        assert labels <= {"train", "validation", "test"}
        assert len(labels) == 1
        label = next(iter(labels))
        if source_file == "train_FD001" and label == "train":
            assert unit in train_units
        elif source_file == "train_FD001" and label == "validation":
            assert unit in validation_units
        else:
            assert source_file == "test_FD001"
            assert label == "test"

    assert dataset["rul"].ge(0).all()
    training_rows = dataset["split"].isin(["train", "validation"])
    endings = dataset.loc[training_rows].groupby("unit_number")["rul"].min()
    assert (endings == 0).all()
    assert not any(sensor in features for sensor in FD001_EXCLUDED_SENSORS)
    assert "op_setting_3" not in features
    assert "op_setting_3" in dataset.columns
    assert "rul" not in features
    assert scaler["columns"] == features
    assert int((dataset["split"] == "train").sum()) > 0
    source = load_fd001(RAW)
    assert len(dataset) == len(source[0]) + len(source[1])
    assert not dataset[features].isna().any().any()

    raw_train, raw_test, rul = source
    last_test = (
        dataset.loc[dataset["split"].eq("test")]
        .sort_values([UNIT, CYCLE])
        .groupby(UNIT, sort=True)
        .tail(1)
    )
    assert last_test["rul"].to_numpy().tolist() == rul.tolist()

    unit = manifest["train_units"][0]
    prefix = raw_train.loc[(raw_train[UNIT] == unit) & (raw_train[CYCLE] <= 10)]
    expected = add_sensor_features(prefix, included_sensors())
    saved = dataset.loc[
        (dataset["source_file"] == "train_FD001") & (dataset[UNIT] == unit) & (dataset[CYCLE] == 10)
    ].iloc[0]
    expected_row = expected.loc[expected[CYCLE] == 10].iloc[0]
    for column in ("sensor_11", "sensor_11_delta", "sensor_11_roll_mean_5", "sensor_11_roll_std_10"):
        assert saved[column] == pytest.approx(expected_row[column])
    assert len(raw_test) > 0
