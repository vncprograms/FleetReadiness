"""Demonstration maintenance thresholds and anomaly wording."""

from src.decision.maintenance_engine import Thresholds, recommend


def test_rul_bands_follow_the_project_settings():
    assert recommend(80, "normal", residual_rmse=0)["priority"] == "Low"
    assert recommend(50, "normal", residual_rmse=0)["priority"] == "Medium"
    assert recommend(20, "normal", residual_rmse=0)["priority"] == "Medium"
    assert recommend(19, "normal", residual_rmse=0)["priority"] == "High"


def test_anomaly_raises_one_level_and_does_not_claim_a_failed_part():
    result = recommend(80, "anomalous", residual_rmse=0)
    assert result["priority"] == "Medium"
    text = " ".join(result["reasons"])
    assert "not evidence that a component has failed" in text
    assert "failed part" not in text.lower() or "not evidence" in text


def test_residual_band_can_raise_priority_when_it_crosses_the_high_setting():
    result = recommend(25, "normal", residual_rmse=10)
    assert result["priority"] == "High"
    assert any("residual band" in reason for reason in result["reasons"])


def test_thresholds_are_configurable():
    custom = Thresholds(medium_below=30, high_below=10)
    assert recommend(25, "normal", residual_rmse=0, thresholds=custom)["priority"] == "Medium"
    assert recommend(9, "normal", residual_rmse=0, thresholds=custom)["priority"] == "High"
