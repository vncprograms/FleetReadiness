"""FD001 labels, engine-level splits, and training-set normalization.

The official test engines stay in the test split. Engines from the training
file are divided into train and validation by engine id. Cycles from one
engine are never placed on both sides of that split.

Training-file RUL at cycle t is final_cycle - t. The final cycle has label 0.
Test-file RUL at cycle t is (last observed cycle + RUL file value) - t. At the
last observed test cycle the label equals the RUL file.

FD001 sensors with zero variance are excluded from the feature matrix. The
raw files are not edited, and those columns remain available in the loaded
measurements. The exclusion list is checked against the data on every build.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ingestion import CYCLE, SETTINGS, SENSORS, UNIT, load_fd001
from src.features.feature_engineering import ROLLING_WINDOWS, add_sensor_features, feature_column_names

# Zero variance on every FD001 training row and every FD001 test row.
# sensor_6 is not in this list: it takes two distinct values and stays in the features.
FD001_EXCLUDED_SENSORS = (
    "sensor_1",
    "sensor_5",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
)
EXCLUSION_REASON = (
    "These channels are constant on every FD001 training and test row, so they "
    "add no information under FD001's single operating condition. They are omitted "
    "from the feature matrix. data/raw is unchanged."
)

# op_setting_3 is 100 on every FD001 row. It stays in the processed table and is
# left out of the model inputs, because a constant cannot be standardized and
# adds no information under the single sea-level condition.
FD001_EXCLUDED_SETTINGS = ("op_setting_3",)
SPLIT_SEED = 42
VALIDATION_FRACTION = 0.2


def validate_cycles(frame: pd.DataFrame, name: str) -> None:
    """Require a complete 1..N cycle index for every engine and no missing values."""
    if frame.isna().any().any():
        raise ValueError(f"{name} contains missing values")
    if frame.duplicated([UNIT, CYCLE]).any():
        raise ValueError(f"{name} contains duplicate unit-cycle rows")
    for unit, group in frame.groupby(UNIT, sort=False):
        cycles = group[CYCLE].to_numpy()
        expected = np.arange(1, len(cycles) + 1)
        if not np.array_equal(cycles, expected):
            raise ValueError(f"{name} unit {unit} cycles are not 1..N in order")


def validate_rul_alignment(test: pd.DataFrame, rul: pd.Series) -> None:
    """Require the RUL file to line up with test units stored as 1..N."""
    units = test[UNIT].drop_duplicates().to_numpy()
    if len(units) != len(rul):
        raise ValueError(f"RUL file has {len(rul)} values for {len(units)} test engines")
    expected = np.arange(1, len(units) + 1)
    if not np.array_equal(units, expected):
        raise ValueError("test engines are not stored in order 1..N; refusing to align the RUL file")


def add_training_rul(frame: pd.DataFrame) -> pd.DataFrame:
    """Label each training cycle with cycles remaining until the end of that run."""
    labeled = frame.copy()
    failure_cycle = labeled.groupby(UNIT)[CYCLE].transform("max")
    labeled["rul"] = (failure_cycle - labeled[CYCLE]).astype(int)
    return labeled


def add_test_rul(frame: pd.DataFrame, rul: pd.Series) -> pd.DataFrame:
    """Label each test cycle. The RUL file is the life remaining after the last row."""
    validate_rul_alignment(frame, rul)
    labeled = frame.copy()
    units = labeled[UNIT].drop_duplicates().to_numpy()
    rul_after_last = pd.Series(rul.to_numpy(), index=units)
    last_cycle = labeled.groupby(UNIT)[CYCLE].transform("max")
    labeled["rul"] = (labeled[UNIT].map(rul_after_last) + (last_cycle - labeled[CYCLE])).astype(int)
    return labeled


def zero_variance_sensors(frame: pd.DataFrame, sensors: list[str] | tuple[str, ...] = SENSORS) -> list[str]:
    return [sensor for sensor in sensors if frame[sensor].nunique(dropna=False) <= 1]


def assert_fd001_exclusions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    expected: tuple[str, ...] = FD001_EXCLUDED_SENSORS,
) -> None:
    """Fail the build if FD001's constant channels are not exactly the recorded list."""
    expected_set = set(expected)
    for name, frame in (("train", train), ("test", test)):
        found = set(zero_variance_sensors(frame))
        if found != expected_set:
            extra = sorted(found - expected_set)
            missing = sorted(expected_set - found)
            raise ValueError(
                f"{name} zero-variance sensors differ from the FD001 exclusion list. "
                f"unexpected constants: {extra or 'none'}; "
                f"listed sensors that vary: {missing or 'none'}"
            )


def assert_fd001_settings(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Require the recorded constant setting, and require the other two to vary."""
    for name, frame in (("train", train), ("test", test)):
        for setting in FD001_EXCLUDED_SETTINGS:
            if frame[setting].nunique(dropna=False) != 1:
                raise ValueError(f"{name} {setting} is not constant; the FD001 exclusion is stale")
        for setting in SETTINGS:
            if setting in FD001_EXCLUDED_SETTINGS:
                continue
            if frame[setting].nunique(dropna=False) <= 1:
                raise ValueError(f"{name} {setting} is constant and is not on the exclusion list")


def included_sensors(excluded: tuple[str, ...] = FD001_EXCLUDED_SENSORS) -> list[str]:
    excluded_set = set(excluded)
    return [sensor for sensor in SENSORS if sensor not in excluded_set]


def split_training_engines(
    unit_ids,
    validation_fraction: float = VALIDATION_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[set[int], set[int]]:
    """Hold out a fraction of engines for validation.

    The count is floor(n * validation_fraction), and at least one engine when
    that product is positive and more than one engine is available. The same
    seed returns the same membership.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    units = np.array(sorted({int(unit) for unit in unit_ids}))
    if len(units) < 2:
        raise ValueError("need at least two engines to split")
    n_validation = int(len(units) * validation_fraction)
    if n_validation < 1:
        n_validation = 1
    if n_validation >= len(units):
        raise ValueError("validation split would leave no training engines")
    shuffled = np.random.default_rng(seed).permutation(units)
    validation_units = {int(unit) for unit in shuffled[:n_validation]}
    train_units = {int(unit) for unit in shuffled[n_validation:]}
    return train_units, validation_units


def assign_split(
    frame: pd.DataFrame,
    train_units: set[int],
    validation_units: set[int],
    default: str | None = None,
) -> pd.DataFrame:
    """Attach a split label. Every cycle of an engine receives the same label."""
    overlap = train_units & validation_units
    if overlap:
        raise ValueError(f"engines assigned to both train and validation: {sorted(overlap)[:5]}")
    assigned = frame.copy()

    def label_for(unit: int) -> str:
        if unit in train_units:
            return "train"
        if unit in validation_units:
            return "validation"
        if default is None:
            raise ValueError(f"unit {unit} has no split")
        return default

    assigned["split"] = assigned[UNIT].map(lambda unit: label_for(int(unit)))
    return assigned


def fit_standardizer(frame: pd.DataFrame, feature_columns: list[str], train_mask: pd.Series) -> dict:
    """Fit a standard score on training rows only. Variance uses divisor n."""
    if int(train_mask.sum()) == 0:
        raise ValueError("no training rows to fit the standardizer")
    subset = frame.loc[train_mask, feature_columns]
    if subset.isna().any().any():
        raise ValueError("training features contain missing values")
    mean = subset.mean(axis=0)
    std = subset.std(axis=0, ddof=0)
    flat = [column for column in feature_columns if std[column] == 0]
    if flat:
        raise ValueError(f"training features have zero variance: {flat}")
    return {
        "method": "standard_score",
        "ddof": 0,
        "fit_on": "split == train",
        "columns": list(feature_columns),
        "mean": {column: float(mean[column]) for column in feature_columns},
        "std": {column: float(std[column]) for column in feature_columns},
    }


def apply_standardizer(frame: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    """Apply a frozen training-set standardizer. Other columns are copied through."""
    scaled = frame.copy()
    columns = scaler["columns"]
    mean = pd.Series(scaler["mean"])
    std = pd.Series(scaler["std"])
    scaled[columns] = (scaled[columns] - mean[columns]) / std[columns]
    return scaled


def build_fd001_dataset(
    raw_dir: Path,
    processed_dir: Path,
    seed: int = SPLIT_SEED,
    validation_fraction: float = VALIDATION_FRACTION,
) -> Path:
    """Write the FD001 feature table, split manifest, and training-set scaler.

    Raw files are only read. Features are computed per engine before the split
    statistics are used. The saved table stays in original measurement units.
    The scaler JSON is the normalization fit on training engines.
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    train, test, rul = load_fd001(raw_dir)
    validate_cycles(train, "train_FD001")
    validate_cycles(test, "test_FD001")
    validate_rul_alignment(test, rul)
    assert_fd001_exclusions(train, test)
    assert_fd001_settings(train, test)

    sensors = included_sensors()
    settings = [setting for setting in SETTINGS if setting not in FD001_EXCLUDED_SETTINGS]
    train = add_training_rul(train)
    test = add_test_rul(test, rul)
    train_units, validation_units = split_training_engines(
        train[UNIT].unique(),
        validation_fraction=validation_fraction,
        seed=seed,
    )
    train = assign_split(train, train_units, validation_units)
    test = assign_split(test, set(), set(), default="test")
    # Train and test files both number engines from 1. They are different engines.
    train["source_file"] = "train_FD001"
    test["source_file"] = "test_FD001"

    train = add_sensor_features(train, sensors)
    test = add_sensor_features(test, sensors)
    dataset = pd.concat([train, test], ignore_index=True)

    features = feature_column_names(sensors, setting_columns=settings)
    if dataset[features].isna().any().any():
        raise ValueError("feature matrix contains missing values")
    leaked = [column for column in features if column == "rul" or column.startswith("failure")]
    if leaked:
        raise ValueError(f"label columns entered the feature list: {leaked}")

    scaler = fit_standardizer(dataset, features, dataset["split"].eq("train"))
    scaler["seed"] = seed
    scaler["validation_fraction"] = validation_fraction
    scaler["subset"] = "FD001"

    id_columns = [
        "source_file",
        UNIT,
        CYCLE,
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
        "split",
        "rul",
    ]
    extra = [column for column in features if column not in id_columns]
    output = dataset[id_columns + extra].copy()
    output[UNIT] = output[UNIT].astype(int)
    output[CYCLE] = output[CYCLE].astype(int)
    output["rul"] = output["rul"].astype(int)

    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = processed_dir / "fd001_features.csv"
    output.to_csv(dataset_path, index=False)
    (processed_dir / "fd001_feature_columns.json").write_text(json.dumps(features, indent=2) + "\n")
    (processed_dir / "fd001_scaler.json").write_text(json.dumps(scaler, indent=2) + "\n")

    counts = output.groupby("split").size().to_dict()
    manifest = {
        "subset": "FD001",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "split_rule": (
            "Official test engines are split 'test'. Engines from train_FD001.txt are "
            "divided by engine id. Validation count is floor(n * validation_fraction). "
            "Unit numbers restart at 1 in the test file, so an engine is "
            "(source_file, unit_number)."
        ),
        "train_units": sorted(train_units),
        "validation_units": sorted(validation_units),
        "test_units": sorted(int(unit) for unit in test[UNIT].unique()),
        "excluded_sensors": list(FD001_EXCLUDED_SENSORS),
        "exclusion_reason": EXCLUSION_REASON,
        "excluded_settings": list(FD001_EXCLUDED_SETTINGS),
        "setting_exclusion_reason": (
            "op_setting_3 is 100 on every FD001 row. The column is kept in fd001_features.csv "
            "and omitted from model inputs."
        ),
        "retained_near_constant_sensor": "sensor_6",
        "label": (
            "Training engines: rul = final cycle - current cycle. "
            "Test engines: rul = (last observed cycle + RUL file value) - current cycle."
        ),
        "rolling_windows": list(ROLLING_WINDOWS),
        "rolling_std_ddof": 0,
        "first_cycle_delta": 0,
        "normalization": "standard score fit on split == train, stored in fd001_scaler.json; the CSV is unscaled",
        "row_counts": {key: int(value) for key, value in counts.items()},
        "feature_count": len(features),
    }
    (processed_dir / "fd001_split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dataset_path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    path = build_fd001_dataset(root / "data" / "raw", root / "data" / "processed")
    print(path)


if __name__ == "__main__":
    main()
