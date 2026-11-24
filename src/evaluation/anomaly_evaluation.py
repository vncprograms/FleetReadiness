"""Three anomaly checks. Scenario C edits a copy, never the source table.

Scenario A replays unmodified validation cycles from early in each run.
Scenario B replays one validation engine through its recorded end.
Scenario C injects synthetic measurement changes into a copy of one engine.
None of these cases has a ground-truth fault label from NASA.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.preprocessing import included_sensors
from src.features.feature_engineering import add_sensor_features
from src.models.anomaly_model import score_cycles

INJECTION_CASES = (
    ("temperature_increase", "sensor_4", "step"),
    ("pressure_deviation", "sensor_11", "step"),
    ("measurement_spike", "sensor_2", "spike"),
    ("measurement_fluctuation", "sensor_7", "fluctuation"),
)


def inject_measurement(
    engine: pd.DataFrame,
    sensor: str,
    start_cycle: int,
    kind: str,
    magnitude: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a copy with one controlled measurement change."""
    modified = engine.copy()
    if kind == "step":
        modified.loc[modified["time_cycles"] >= start_cycle, sensor] = (
            modified.loc[modified["time_cycles"] >= start_cycle, sensor] + magnitude
        )
    elif kind == "spike":
        modified.loc[modified["time_cycles"] == start_cycle, sensor] = (
            modified.loc[modified["time_cycles"] == start_cycle, sensor] + magnitude
        )
    elif kind == "fluctuation":
        mask = modified["time_cycles"] >= start_cycle
        noise = np.random.default_rng(seed).normal(0.0, magnitude, size=int(mask.sum()))
        modified.loc[mask, sensor] = modified.loc[mask, sensor].to_numpy() + noise
    else:
        raise ValueError(f"unknown injection kind: {kind}")
    rebuilt = add_sensor_features(modified, included_sensors())
    return rebuilt


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#d0d5dd",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def evaluate_anomaly_model(frame: pd.DataFrame, bundle: dict, figure_dir: Path | None = None) -> dict:
    model = bundle["model"]
    features = bundle["feature_columns"]
    scaler = bundle["scaler"]
    validation = frame.loc[frame["split"].eq("validation")].copy()
    early = validation.loc[validation["rul"] > 100]
    early_scored = score_cycles(model, early, features, scaler)
    scenario_a = {
        "description": "Unmodified validation cycles with recorded RUL above 100.",
        "n": int(len(early_scored)),
        "fraction_anomalous": float((early_scored["anomaly_label"] == "anomalous").mean()),
        "mean_anomaly_score": float(early_scored["anomaly_score"].mean()),
    }

    unit = int(validation["unit_number"].iloc[0])
    engine = validation.loc[validation["unit_number"] == unit].sort_values("time_cycles")
    life = score_cycles(model, engine, features, scaler)
    scenario_b = {
        "description": "One unmodified validation engine, replayed from the first cycle through the recorded end.",
        "source_file": "train_FD001",
        "unit_number": unit,
        "correlation_anomaly_score_vs_cycle": float(life["anomaly_score"].corr(life["time_cycles"])),
        "mean_score_first_20_cycles": float(life["anomaly_score"].head(20).mean()),
        "mean_score_last_20_cycles": float(life["anomaly_score"].tail(20).mean()),
        "fraction_anomalous": float((life["anomaly_label"] == "anomalous").mean()),
        "note": "This is an observed score path. The dataset has no anomaly labels for these cycles.",
    }

    start = int(engine["time_cycles"].quantile(0.7))
    original_row = engine.loc[engine["time_cycles"] == start]
    original_score = float(score_cycles(model, original_row, features, scaler)["anomaly_score"].iloc[0])
    cases = []
    for name, sensor, kind in INJECTION_CASES:
        magnitude = float(engine[sensor].std(ddof=0) * 10.0)
        if magnitude == 0:
            magnitude = 10.0
        injected = inject_measurement(engine, sensor, start, kind, magnitude)
        injected_row = injected.loc[injected["time_cycles"] == start]
        scored = score_cycles(model, injected_row, features, scaler)
        after = float(scored["anomaly_score"].iloc[0])
        label = str(scored["anomaly_label"].iloc[0])
        cases.append(
            {
                "name": name,
                "sensor": sensor,
                "kind": kind,
                "magnitude": magnitude,
                "score_before": original_score,
                "score_after": after,
                "label_after": label,
                "identified": bool(label == "anomalous" and after > original_score),
            }
        )
    scenario_c = {
        "label": "Synthetic injections on a copy of one validation engine. Not real aircraft faults.",
        "source_file": "train_FD001",
        "unit_number": unit,
        "cycle": start,
        "cases": cases,
        "n_identified": int(sum(case["identified"] for case in cases)),
        "n_cases": len(cases),
    }

    if figure_dir is not None:
        figure_dir.mkdir(parents=True, exist_ok=True)
        _style()
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        axes[0].plot(life["time_cycles"], life["anomaly_score"], color="#1f4e79", linewidth=1.2)
        axes[0].set_title(f"Validation unit {unit}")
        axes[0].set_xlabel("Operating cycle")
        axes[0].set_ylabel("Anomaly score")
        names = ["temperature", "pressure", "spike", "fluctuation"]
        axes[1].bar(np.arange(len(cases)) - 0.18, [case["score_before"] for case in cases], width=0.36, color="#98a2b3", label="Unmodified")
        axes[1].bar(np.arange(len(cases)) + 0.18, [case["score_after"] for case in cases], width=0.36, color="#c45911", label="Synthetic copy")
        axes[1].set_xticks(np.arange(len(cases)), names)
        axes[1].set_title("Synthetic injections")
        axes[1].set_ylabel("Anomaly score")
        axes[1].legend(frameon=False, loc="lower right")
        figure.tight_layout()
        figure.savefig(figure_dir / "anomaly_scenarios.png", dpi=150, bbox_inches="tight")
        plt.close(figure)

    return {
        "model": "isolation_forest",
        "contamination": bundle.get("contamination", 0.02),
        "trained_on": "split == train",
        "score_definition": bundle.get("score_definition"),
        "limitation": "No NASA fault labels are available for these checks. Scenario C is synthetic.",
        "scenario_a_normal": scenario_a,
        "scenario_b_degrading": scenario_b,
        "scenario_c_synthetic": scenario_c,
    }


def main() -> None:
    import joblib

    from src.models.rul_model import load_fd001_features

    root = Path(__file__).resolve().parents[2]
    frame, _, _ = load_fd001_features(root / "data" / "processed")
    bundle = joblib.load(root / "models" / "anomaly_model.pkl")
    report = evaluate_anomaly_model(frame, bundle, root / "notebooks" / "figures")
    path = root / "models" / "anomaly_evaluation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)
    print(
        "scenario C identified",
        report["scenario_c_synthetic"]["n_identified"],
        "of",
        report["scenario_c_synthetic"]["n_cases"],
    )


if __name__ == "__main__":
    main()
