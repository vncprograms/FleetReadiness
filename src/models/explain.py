"""SHAP values for the saved XGBoost model.

SHAP attributes a prediction to model inputs. It does not identify the
physical cause of engine degradation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.models.rul_model import feature_matrix, load_fd001_features


def _explainer(model):
    import shap

    return shap.TreeExplainer(model)


def explain_matrix(model, matrix: np.ndarray) -> np.ndarray:
    values = _explainer(model).shap_values(matrix)
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[0]
    return values


def summarize_training_influences(root: Path, sample_size: int = 200) -> Path:
    """Save mean absolute SHAP values for a sample of training cycles."""
    import joblib

    root = Path(root)
    frame, features, scaler = load_fd001_features(root / "data" / "processed")
    bundle = joblib.load(root / "models" / "rul_model.pkl")
    train = frame.loc[frame["split"].eq("train")]
    sample = train.sample(n=min(sample_size, len(train)), random_state=42)
    values = explain_matrix(bundle["model"], feature_matrix(sample, features, scaler))
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    summary = {
        "note": "Mean absolute SHAP value on a sample of training cycles. Model influence, not a physical cause.",
        "sample_size": int(len(sample)),
        "features": [
            {"feature": features[int(index)], "mean_abs_shap": float(mean_abs[int(index)])}
            for index in order[:15]
        ],
    }
    path = root / "models" / "shap_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return path


def explain_frame(model, frame, features, scaler) -> list[dict]:
    matrix = feature_matrix(frame, features, scaler)
    values = explain_matrix(model, matrix)[0]
    order = np.argsort(np.abs(values))[::-1]
    return [
        {"feature": features[int(index)], "shap": float(values[int(index)])}
        for index in order[:12]
    ]


def main() -> None:
    path = summarize_training_influences(Path(__file__).resolve().parents[2])
    print(path)


if __name__ == "__main__":
    main()
