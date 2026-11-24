"""Read original C-MAPSS text files.

The files have no header. Column names follow the 2008 readme's positions:
unit number, time in cycles, three operational settings, then 21 sensor
channels. Sensor symbols from Saxena et al. 2008, Table 2, are documentation
only. They are not stored in the files, and this loader does not rename the
numbered sensor columns to those symbols.
"""

from pathlib import Path

import pandas as pd

UNIT = "unit_number"
CYCLE = "time_cycles"
SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSORS = [f"sensor_{index}" for index in range(1, 22)]
COLUMNS = [UNIT, CYCLE, *SETTINGS, *SENSORS]

# Saxena et al. 2008, Table 2, in listed order. Index i corresponds to sensor_i.
SENSOR_TABLE2 = [
    (1, "T2", "Total temperature at fan inlet", "°R"),
    (2, "T24", "Total temperature at LPC outlet", "°R"),
    (3, "T30", "Total temperature at HPC outlet", "°R"),
    (4, "T50", "Total temperature at LPT outlet", "°R"),
    (5, "P2", "Pressure at fan inlet", "psia"),
    (6, "P15", "Total pressure in bypass-duct", "psia"),
    (7, "P30", "Total pressure at HPC outlet", "psia"),
    (8, "Nf", "Physical fan speed", "rpm"),
    (9, "Nc", "Physical core speed", "rpm"),
    (10, "epr", "Engine pressure ratio (P50/P2)", "--"),
    (11, "Ps30", "Static pressure at HPC outlet", "psia"),
    (12, "phi", "Ratio of fuel flow to Ps30", "pps/psi"),
    (13, "NRf", "Corrected fan speed", "rpm"),
    (14, "NRc", "Corrected core speed", "rpm"),
    (15, "BPR", "Bypass ratio", "--"),
    (16, "farB", "Burner fuel-air ratio", "--"),
    (17, "htBleed", "Bleed enthalpy", "--"),
    (18, "Nf_dmd", "Demanded fan speed", "rpm"),
    (19, "PCNfR_dmd", "Demanded corrected fan speed", "rpm"),
    (20, "W31", "HPT coolant bleed", "lbm/s"),
    (21, "W32", "LPT coolant bleed", "lbm/s"),
]


def load_cycle_file(path: Path) -> pd.DataFrame:
    """Load one train or test cycle file. The source file is only read."""
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name} has {frame.shape[1]} columns; expected {len(COLUMNS)}")
    if frame.empty:
        raise ValueError(f"{path.name} has no rows")
    frame[UNIT] = frame[UNIT].astype(int)
    frame[CYCLE] = frame[CYCLE].astype(int)
    return frame


def load_rul_file(path: Path) -> pd.Series:
    """Load the test-set RUL vector. One integer per test engine, file order."""
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=["rul"], engine="python")
    if frame.empty:
        raise ValueError(f"{path.name} has no rows")
    return frame["rul"].astype(int)


def load_fd001(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load FD001 train cycles, test cycles, and the test RUL vector."""
    raw_dir = Path(raw_dir)
    train = load_cycle_file(raw_dir / "train_FD001.txt")
    test = load_cycle_file(raw_dir / "test_FD001.txt")
    rul = load_rul_file(raw_dir / "RUL_FD001.txt")
    return train, test, rul
