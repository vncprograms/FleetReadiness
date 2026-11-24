"""Train and compare FD001 remaining-useful-life models.

Three models see the same scaled training engines: ridge regression is the
linear baseline, then a random forest, then XGBoost. Hyperparameters are
fixed in this file. They are not searched against the official test engines.

Selection uses validation engines only. Each of those engines ran to the
recorded end, so the cycle where 30 cycles remain is known. That single
cycle per engine is the selection slice. All-cycle error is reported as
well. The official test set is scored once, after the model is chosen, and
only the chosen model is scored there.

A second fit clips training labels at 125 cycles. That cap is a modeling
choice used in later C-MAPSS papers. It is not in the dataset readme. The
capped fits are an experiment and are not eligible for the saved model.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from src.data.preprocessing import apply_standardizer
from src.evaluation.rul_evaluation import regression_metrics, rows_at_rul

SELECTION_RUL = 30
RUL_CAP = 125
RANDOM_STATE = 42


def load_fd001_features(processed_dir: Path) -> tuple[pd.DataFrame, list[str], dict]:
    processed_dir = Path(processed_dir)
    frame = pd.read_csv(processed_dir / "fd001_features.csv")
    features = json.loads((processed_dir / "fd001_feature_columns.json").read_text())
    scaler = json.loads((processed_dir / "fd001_scaler.json").read_text())
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"processed table is missing feature columns: {missing}")
    return frame, features, scaler


def training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep rows from the training split of train_FD001 only."""
    if not frame["split"].eq("train").all():
        raise ValueError("training rows include a split other than train")
    if not frame["source_file"].eq("train_FD001").all():
        raise ValueError("training rows include a source file other than train_FD001")
    return frame


def feature_matrix(frame: pd.DataFrame, feature_columns: list[str], scaler: dict) -> np.ndarray:
    scaled = apply_standardizer(frame, scaler)
    return scaled[feature_columns].to_numpy(dtype=float)


def target_values(frame: pd.DataFrame, rul_cap: int | None) -> np.ndarray:
    target = frame["rul"].to_numpy(dtype=float)
    if rul_cap is None:
        return target
    if rul_cap < 1:
        raise ValueError("rul_cap must be positive")
    return np.minimum(target, rul_cap)


def build_estimators() -> dict:
    """Fixed model definitions. Changing these changes the comparison."""
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def fit_regressor(estimator, frame: pd.DataFrame, feature_columns: list[str], scaler: dict, rul_cap: int | None):
    rows = training_rows(frame)
    model = estimator
    model.fit(feature_matrix(rows, feature_columns, scaler), target_values(rows, rul_cap))
    return model


def predict_rul(model, frame: pd.DataFrame, feature_columns: list[str], scaler: dict) -> np.ndarray:
    """Predict from sensor features. The rul column is not read."""
    return np.asarray(model.predict(feature_matrix(frame, feature_columns, scaler)), dtype=float)


def evaluate_frame(frame: pd.DataFrame, predictions: np.ndarray) -> dict:
    return regression_metrics(frame["rul"].to_numpy(dtype=float), predictions)


def slice_metrics(frame: pd.DataFrame, predictions: np.ndarray, rul_value: int) -> dict:
    """Score the one cycle per engine whose true label equals rul_value."""
    scored = frame.copy()
    scored["_prediction"] = predictions
    chosen = rows_at_rul(scored, rul_value)
    return regression_metrics(chosen["rul"].to_numpy(dtype=float), chosen["_prediction"].to_numpy(dtype=float))


def last_cycle_metrics(frame: pd.DataFrame, predictions: np.ndarray) -> dict:
    scored = frame.copy()
    scored["_prediction"] = predictions
    chosen = (
        scored.sort_values(["source_file", "unit_number", "time_cycles"])
        .groupby(["source_file", "unit_number"], sort=False)
        .tail(1)
    )
    return regression_metrics(chosen["rul"].to_numpy(dtype=float), chosen["_prediction"].to_numpy(dtype=float))


def _top_influences(name: str, model, feature_columns: list[str], limit: int = 10) -> list[dict]:
    if name == "ridge":
        values = np.abs(model.coef_)
        kind = "absolute_coefficient"
    else:
        values = np.asarray(model.feature_importances_, dtype=float)
        kind = "impurity_importance"
    order = np.argsort(values)[::-1][:limit]
    return [
        {"feature": feature_columns[int(index)], "value": float(values[int(index)]), "kind": kind}
        for index in order
    ]


def _style_axes() -> None:
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


def save_evaluation_plots(
    validation: pd.DataFrame,
    prediction_by_model: dict[str, np.ndarray],
    selected: str,
    figure_dir: Path,
) -> None:
    """Write the validation comparison, prediction scatter, and error histogram."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    _style_axes()
    names = list(prediction_by_model)
    at_horizon = []
    all_cycles = []
    for name in names:
        at_horizon.append(slice_metrics(validation, prediction_by_model[name], SELECTION_RUL)["rmse"])
        all_cycles.append(evaluate_frame(validation, prediction_by_model[name])["rmse"])

    positions = np.arange(len(names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    axis.bar(positions - width / 2, at_horizon, width, color="#1f4e79", label="30 cycles remaining")
    axis.bar(positions + width / 2, all_cycles, width, color="#8fa8c4", label="Every validation cycle")
    axis.set_xticks(positions, names)
    axis.set_ylabel("RMSE (cycles)")
    axis.set_title("Validation error by model")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "rul_validation_rmse.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    selected_pred = prediction_by_model[selected]
    truth = validation["rul"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(6.2, 6.0))
    axis.scatter(truth, selected_pred, s=8, alpha=0.25, color="#1f4e79", linewidths=0)
    horizon = validation["rul"].to_numpy() == SELECTION_RUL
    axis.scatter(
        truth[horizon],
        selected_pred[horizon],
        s=36,
        color="#c45911",
        label="30 cycles remaining",
        zorder=3,
    )
    limit = float(max(truth.max(), selected_pred.max(), 1))
    axis.plot([0, limit], [0, limit], color="#667085", linewidth=1)
    axis.set_xlabel("True RUL (cycles)")
    axis.set_ylabel("Predicted RUL (cycles)")
    axis.set_title(f"Validation predictions, {selected}")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "rul_validation_pred_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    error = selected_pred - truth
    figure, axis = plt.subplots(figsize=(8.0, 4.4))
    axis.hist(error, bins=40, color="#1f4e79", edgecolor="white")
    axis.axvline(0, color="#c45911", linewidth=1.4)
    axis.set_xlabel("Predicted RUL minus true RUL (cycles)")
    axis.set_ylabel("Validation cycles")
    axis.set_title(f"Validation error distribution, {selected}")
    figure.tight_layout()
    figure.savefig(figure_dir / "rul_validation_error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def train_fd001(processed_dir: Path, model_dir: Path, figure_dir: Path) -> dict:
    frame, features, scaler = load_fd001_features(processed_dir)
    train = frame.loc[frame["split"].eq("train")].copy()
    validation = frame.loc[frame["split"].eq("validation")].copy()
    test = frame.loc[frame["split"].eq("test")].copy()
    training_rows(train)
    if validation.empty or test.empty:
        raise ValueError("validation or test split is empty")

    primary_predictions = {}
    primary_metrics = {}
    influences = {}
    fitted = {}
    for name, estimator in build_estimators().items():
        model = fit_regressor(estimator, train, features, scaler, rul_cap=None)
        predictions = predict_rul(model, validation, features, scaler)
        primary_predictions[name] = predictions
        primary_metrics[name] = {
            "at_rul_30": slice_metrics(validation, predictions, SELECTION_RUL),
            "all_cycles": evaluate_frame(validation, predictions),
        }
        influences[name] = _top_influences(name, model, features)
        fitted[name] = model

    selected = min(primary_metrics, key=lambda name: primary_metrics[name]["at_rul_30"]["rmse"])
    test_predictions = predict_rul(fitted[selected], test, features, scaler)
    test_metrics = last_cycle_metrics(test, test_predictions)

    capped_metrics = {}
    for name in ("random_forest", "xgboost"):
        model = fit_regressor(build_estimators()[name], train, features, scaler, rul_cap=RUL_CAP)
        predictions = predict_rul(model, validation, features, scaler)
        capped_metrics[name] = {
            "at_rul_30": slice_metrics(validation, predictions, SELECTION_RUL),
            "all_cycles": evaluate_frame(validation, predictions),
        }

    model_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": fitted[selected],
        "model_name": selected,
        "feature_columns": features,
        "scaler": scaler,
        "rul_cap": None,
        "selection_rul": SELECTION_RUL,
        "random_state": RANDOM_STATE,
    }
    model_path = model_dir / "rul_model.pkl"
    joblib.dump(bundle, model_path)

    report = {
        "subset": "FD001",
        "selection_rule": (
            "Lowest RMSE on validation engines at the cycle where true RUL is 30. "
            "Official test engines were not used to choose the model."
        ),
        "selected_model": selected,
        "saved_model": str(model_path.name),
        "hyperparameters": {
            "ridge": {"alpha": 1.0},
            "random_forest": {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 2},
            "xgboost": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },
        "validation": primary_metrics,
        "test_last_cycle": {"model": selected, **test_metrics},
        "capped_rul_experiment": {
            "cap": RUL_CAP,
            "note": (
                "Training labels were clipped at 125 cycles. Metrics below compare "
                "those models with the uncapped true RUL on the validation engines. "
                "This experiment was not eligible for model selection."
            ),
            "validation": capped_metrics,
        },
        "top_influences": influences,
        "influence_note": (
            "Ridge values are absolute coefficients on standardized features. "
            "Tree values are impurity importances. Neither is a physical cause."
        ),
    }
    (model_dir / "rul_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    save_evaluation_plots(validation, primary_predictions, selected, figure_dir)
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    report = train_fd001(root / "data" / "processed", root / "models", root / "notebooks" / "figures")
    selected = report["selected_model"]
    horizon = report["validation"][selected]["at_rul_30"]
    test = report["test_last_cycle"]
    print(f"selected: {selected}")
    print(
        f"validation at 30 cycles remaining: MAE {horizon['mae']:.2f}, "
        f"RMSE {horizon['rmse']:.2f}, NASA {horizon['nasa_score']:.2f}"
    )
    print(
        f"official test last cycle: MAE {test['mae']:.2f}, "
        f"RMSE {test['rmse']:.2f}, NASA {test['nasa_score']:.2f}"
    )


if __name__ == "__main__":
    main()
