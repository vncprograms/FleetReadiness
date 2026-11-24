"""Replay one engine's recorded cycles without revealing later cycles.

Speed is the number of cycles released on each advance call: 1, 5, or 10.
The class does not sleep. A dashboard can wait between advances.
"""

from __future__ import annotations

import pandas as pd

SPEEDS = (1, 5, 10)


class EngineSimulator:
    def __init__(self, engine_rows: pd.DataFrame):
        rows = engine_rows.sort_values("time_cycles").reset_index(drop=True)
        if rows.empty:
            raise ValueError("engine has no cycles")
        if rows["time_cycles"].duplicated().any():
            raise ValueError("engine has duplicate cycles")
        if not rows["time_cycles"].is_monotonic_increasing:
            raise ValueError("engine cycles are not increasing")
        self._rows = rows
        self.speed = 1
        self._cursor = 0
        self.playing = False

    @property
    def n_cycles(self) -> int:
        return int(len(self._rows))

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def finished(self) -> bool:
        return self._cursor >= len(self._rows)

    def set_speed(self, speed: int) -> None:
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}")
        self.speed = int(speed)

    def pause(self) -> None:
        self.playing = False

    def resume(self) -> None:
        self.playing = True

    def reset(self) -> None:
        self._cursor = 0
        self.playing = False

    def start(self) -> pd.DataFrame:
        self.playing = True
        if self._cursor == 0:
            return self.advance()
        return self.seen()

    def advance(self) -> pd.DataFrame:
        """Release the next speed cycles. Returns only those new rows."""
        if not self.playing or self.finished:
            return self._rows.iloc[0:0].copy()
        start = self._cursor
        self._cursor = min(self._cursor + self.speed, len(self._rows))
        return self._rows.iloc[start : self._cursor].copy()

    def seen(self) -> pd.DataFrame:
        """Cycles released so far, in recorded order."""
        return self._rows.iloc[: self._cursor].copy()


def simulator_for(catalog: pd.DataFrame, source_file: str, unit_number: int) -> EngineSimulator:
    rows = catalog.loc[
        catalog["source_file"].eq(source_file) & catalog["unit_number"].eq(unit_number)
    ]
    if rows.empty:
        raise ValueError(f"no engine {source_file} unit {unit_number}")
    return EngineSimulator(rows)
