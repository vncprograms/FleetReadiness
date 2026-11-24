"""Isolation Forest for unusual FD001 sensor patterns.

The forest is fit on the training engines only, in the same standardized
feature space as the RUL model. It has no fault labels. A high anomaly score
means the cycle is unusual relative to that training distribution. It does
not identify a failed component.

sklearn's decision_function is higher for points that look more like the
training data. This module stores anomaly_score = -decision_function, so a
higher anomaly score means more unusual.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.models.rul_model import feature_matrix, load_fd001_features, training_rows

CONTAMINATION = 0.02
RANDOM_STATE = 42


def fit_isolation_forest(frame: pd.DataFrame, feature_columns: list[str], scaler: dict) -> IsolationForest:
    rows = training_rows(frame)
    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(feature_matrix(rows, feature_columns, scaler))
    return model


def score_cycles(model, frame: pd.DataFrame, feature_columns: list[str], scaler: dict) -> pd.DataFrame:
    """Score cycles. The input frame is not modified."""
    matrix = feature_matrix(frame, feature_columns, scaler)
    normality = np.asarray(model.decision_function(matrix), dtype=float)
    labels = np.where(model.predict(matrix) == -1, "anomalous", "normal")
    scored = frame.copy()
    scored["anomaly_score"] = -normality
    scored["anomaly_label"] = labels
    return scored


def train_fd001_anomaly(processed_dir: Path, model_dir: Path) -> Path:
    frame, features, scaler = load_fd001_features(processed_dir)
    train = frame.loc[frame["split"].eq("train")].copy()
    model = fit_isolation_forest(train, features, scaler)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "anomaly_model.pkl"
    joblib.dump(
        {
            "model": model,
            "feature_columns": features,
            "scaler": scaler,
            "contamination": CONTAMINATION,
            "random_state": RANDOM_STATE,
            "score_definition": "anomaly_score = -decision_function; higher is more unusual",
        },
        path,
    )
    return path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    path = train_fd001_anomaly(root / "data" / "processed", root / "models")
    print(path)


if __name__ == "__main__":
    main()
