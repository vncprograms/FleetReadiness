"""Historical replay releases cycles in order and hides the rest."""

import pandas as pd
import pytest

from src.simulation.engine_simulator import EngineSimulator, simulator_for


def _engine(unit: int = 3, cycles: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_file": "test_FD001",
            "unit_number": unit,
            "time_cycles": list(range(1, cycles + 1)),
            "sensor_2": [float(cycle) for cycle in range(1, cycles + 1)],
        }
    )


def test_advance_is_chronological_and_does_not_reveal_later_cycles():
    simulator = EngineSimulator(_engine())
    simulator.start()
    assert simulator.seen()["time_cycles"].tolist() == [1]
    simulator.set_speed(5)
    released = simulator.advance()
    assert released["time_cycles"].tolist() == [2, 3, 4, 5, 6]
    assert simulator.seen()["time_cycles"].max() == 6
    assert 7 not in simulator.seen()["time_cycles"].tolist()


def test_pause_resume_and_reset():
    simulator = EngineSimulator(_engine())
    simulator.start()
    simulator.pause()
    assert simulator.advance().empty
    assert simulator.cursor == 1
    simulator.resume()
    simulator.set_speed(10)
    simulator.advance()
    assert simulator.cursor == 11
    simulator.reset()
    assert simulator.cursor == 0
    assert simulator.seen().empty
    assert simulator.playing is False


def test_speed_must_be_a_supported_playback_rate():
    simulator = EngineSimulator(_engine())
    with pytest.raises(ValueError):
        simulator.set_speed(2)


def test_catalog_lookup_uses_source_and_unit():
    catalog = pd.concat([_engine(1), _engine(2)], ignore_index=True)
    simulator = simulator_for(catalog, "test_FD001", 2)
    simulator.start()
    assert simulator.seen()["unit_number"].iloc[0] == 2
    assert simulator.seen()["time_cycles"].iloc[0] == 1
