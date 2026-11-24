"""Turn a prediction into a project recommendation.

The cycle thresholds are demonstration settings. They are not aviation
maintenance limits. An anomaly flag raises attention by one level and is
described as an unusual pattern, not as a failed part.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    medium_below: float = 50.0
    high_below: float = 20.0

    def __post_init__(self) -> None:
        if self.high_below <= 0 or self.medium_below <= self.high_below:
            raise ValueError("require 0 < high_below < medium_below")


def _band(predicted_rul: float, thresholds: Thresholds) -> str:
    if predicted_rul < thresholds.high_below:
        return "High"
    if predicted_rul <= thresholds.medium_below:
        return "Medium"
    return "Low"


def _raise_one(priority: str) -> str:
    return {"Low": "Medium", "Medium": "High", "High": "High"}[priority]


def recommend(
    predicted_rul: float,
    anomaly_label: str,
    residual_rmse: float,
    thresholds: Thresholds | None = None,
    trend_notes: list[str] | None = None,
    engine_label: str | None = None,
) -> dict:
    """Explain a demonstration priority from the RUL prediction and context."""
    if anomaly_label not in {"normal", "anomalous"}:
        raise ValueError("anomaly_label must be normal or anomalous")
    if residual_rmse < 0:
        raise ValueError("residual_rmse must be non-negative")
    thresholds = thresholds or Thresholds()
    priority = _band(predicted_rul, thresholds)
    reasons = []
    if engine_label:
        reasons.append(f"Engine {engine_label}.")
    reasons.append(
        f"Predicted remaining useful life is {predicted_rul:.1f} cycles. "
        f"Under the project settings, above {thresholds.medium_below:.0f} is low, "
        f"{thresholds.high_below:.0f} to {thresholds.medium_below:.0f} is medium, "
        f"and below {thresholds.high_below:.0f} is high."
    )
    reasons.append(
        f"The validation RMSE at 30 cycles remaining is {residual_rmse:.1f} cycles. "
        "That figure is a residual band from the held-out engines, not a certified interval."
    )
    low_end = predicted_rul - residual_rmse
    if priority != "High" and low_end < thresholds.high_below:
        priority = "High"
        reasons.append(
            "The low end of that residual band falls below the high-priority setting, so the priority is raised."
        )
    if anomaly_label == "anomalous":
        if priority != "High":
            priority = _raise_one(priority)
        reasons.append(
            "This cycle is marked anomalous relative to the training distribution. "
            "That raises attention by one level. It is not evidence that a component has failed."
        )
    else:
        reasons.append("This cycle is inside the isolation forest's normal range for the training engines.")
    for note in trend_notes or []:
        reasons.append(note)
    action = {
        "Low": "Continue monitoring.",
        "Medium": "Schedule an inspection.",
        "High": "Prioritize inspection and review the contributing sensor measurements.",
    }[priority]
    return {
        "priority": priority,
        "recommendation": action,
        "reasons": reasons,
        "predicted_rul": float(predicted_rul),
        "residual_rmse": float(residual_rmse),
        "anomaly_label": anomaly_label,
    }
