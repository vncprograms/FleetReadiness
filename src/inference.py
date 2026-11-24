"""Score released cycles with the saved RUL and anomaly models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.data.ingestion import SENSOR_TABLE2
from src.decision.maintenance_engine import Thresholds, recommend
from src.models.anomaly_model import score_cycles
from src.models.rul_model import load_fd001_features, predict_rul

SENSOR_SYMBOL = {f"sensor_{index}": symbol for index, symbol, _, _ in SENSOR_TABLE2}
TREND_SENSORS = ("sensor_11", "sensor_4", "sensor_12", "sensor_7")


@dataclass
class Assets:
    frame: pd.DataFrame
    features: list[str]
    scaler: dict
    rul_model: object
    anomaly_model: object
    anomaly_bundle: dict
    residual_rmse: float
    comparison: dict
    anomaly_report: dict | None


def load_assets(root: Path) -> Assets:
    root = Path(root)
    frame, features, scaler = load_fd001_features(root / "data" / "processed")
    rul_bundle = joblib.load(root / "models" / "rul_model.pkl")
    anomaly_bundle = joblib.load(root / "models" / "anomaly_model.pkl")
    comparison = json.loads((root / "models" / "rul_comparison.json").read_text())
    report_path = root / "models" / "anomaly_evaluation.json"
    anomaly_report = json.loads(report_path.read_text()) if report_path.exists() else None
    selected = comparison["selected_model"]
    residual = float(comparison["validation"][selected]["at_rul_30"]["rmse"])
    return Assets(
        frame=frame,
        features=features,
        scaler=scaler,
        rul_model=rul_bundle["model"],
        anomaly_model=anomaly_bundle["model"],
        anomaly_bundle=anomaly_bundle,
        residual_rmse=residual,
        comparison=comparison,
        anomaly_report=anomaly_report,
    )


def trend_notes(history: pd.DataFrame) -> list[str]:
    """Describe the last cycle-to-cycle change for a few sensors."""
    if len(history) < 2:
        return []
    notes = []
    current = history.iloc[-1]
    previous = history.iloc[-2]
    for sensor in TREND_SENSORS:
        if sensor not in history.columns:
            continue
        change = float(current[sensor] - previous[sensor])
        symbol = SENSOR_SYMBOL.get(sensor, sensor)
        notes.append(f"{symbol} ({sensor}) changed by {change:+.3f} from the previous released cycle.")
    return notes


def score_history(assets: Assets, history: pd.DataFrame, thresholds: Thresholds | None = None) -> dict:
    """Score the latest released cycle. Earlier rows are used only for the trend note."""
    if history.empty:
        raise ValueError("no cycles have been released")
    current = history.iloc[[-1]]
    predicted = float(predict_rul(assets.rul_model, current, assets.features, assets.scaler)[0])
    scored = score_cycles(assets.anomaly_model, current, assets.features, assets.scaler)
    label = str(scored["anomaly_label"].iloc[0])
    engine_label = f"{current['source_file'].iloc[0]} unit {int(current['unit_number'].iloc[0])}"
    decision = recommend(
        predicted_rul=predicted,
        anomaly_label=label,
        residual_rmse=assets.residual_rmse,
        thresholds=thresholds,
        trend_notes=trend_notes(history),
        engine_label=engine_label,
    )
    decision["anomaly_score"] = float(scored["anomaly_score"].iloc[0])
    decision["time_cycles"] = int(current["time_cycles"].iloc[0])
    decision["recorded_rul"] = float(current["rul"].iloc[0])
    decision["source_file"] = str(current["source_file"].iloc[0])
    decision["unit_number"] = int(current["unit_number"].iloc[0])
    return decision


def fleet_snapshot(assets: Assets, thresholds: Thresholds | None = None) -> pd.DataFrame:
    """Score each official test engine at its last recorded cycle."""
    test = assets.frame.loc[assets.frame["split"].eq("test")]
    last = (
        test.sort_values(["unit_number", "time_cycles"])
        .groupby("unit_number", sort=True)
        .tail(1)
        .reset_index(drop=True)
    )
    predicted = predict_rul(assets.rul_model, last, assets.features, assets.scaler)
    scored = score_cycles(assets.anomaly_model, last, assets.features, assets.scaler)
    rows = []
    for position, row in last.iterrows():
        history = last.iloc[[position]]
        decision = recommend(
            predicted_rul=float(predicted[len(rows)]),
            anomaly_label=str(scored["anomaly_label"].iloc[len(rows)]),
            residual_rmse=assets.residual_rmse,
            thresholds=thresholds,
            engine_label=f"test_FD001 unit {int(row['unit_number'])}",
        )
        rows.append(
            {
                "unit_number": int(row["unit_number"]),
                "time_cycles": int(row["time_cycles"]),
                "recorded_rul": float(row["rul"]),
                "predicted_rul": decision["predicted_rul"],
                "anomaly_label": decision["anomaly_label"],
                "anomaly_score": float(scored["anomaly_score"].iloc[len(rows)]),
                "priority": decision["priority"],
                "recommendation": decision["recommendation"],
            }
        )
    order = {"High": 0, "Medium": 1, "Low": 2}
    snapshot = pd.DataFrame(rows)
    snapshot["_order"] = snapshot["priority"].map(order)
    return snapshot.sort_values(["_order", "predicted_rul"]).drop(columns="_order").reset_index(drop=True)
