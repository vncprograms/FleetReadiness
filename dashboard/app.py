"""FleetReadiness dashboard.

Research prototype. The cycles on screen are historical C-MAPSS rows released
by the replay simulator. They are not a live aircraft feed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ingestion import SENSOR_TABLE2
from src.decision.maintenance_engine import Thresholds
from src.inference import fleet_snapshot, load_assets, score_history
from src.models.explain import explain_frame
from src.simulation.engine_simulator import SPEEDS, simulator_for

SENSOR_SYMBOL = {f"sensor_{index}": symbol for index, symbol, _description, _units in SENSOR_TABLE2}
TREND_SENSORS = ("sensor_11", "sensor_4", "sensor_12", "sensor_7")
PRIORITY_COLORS = {"High": "#b42318", "Medium": "#b54708", "Low": "#027a48"}


st.set_page_config(page_title="FleetReadiness", layout="wide")


@st.cache_resource
def assets():
    return load_assets(ROOT)


def thresholds_from_sidebar() -> Thresholds:
    st.sidebar.header("Project thresholds")
    medium = st.sidebar.slider("Medium priority at or below (cycles)", 21, 120, 50)
    high = st.sidebar.slider("High priority below (cycles)", 1, medium - 1, min(20, medium - 1))
    st.sidebar.caption("Demonstration settings, not certified maintenance limits.")
    return Thresholds(medium_below=float(medium), high_below=float(high))


def priority_badge(priority: str) -> str:
    return f"<span style='color:{PRIORITY_COLORS[priority]};font-weight:600'>{priority}</span>"


def sensor_figure(history: pd.DataFrame) -> go.Figure:
    long = history.melt(
        id_vars=["time_cycles"],
        value_vars=list(TREND_SENSORS),
        var_name="sensor",
        value_name="measurement",
    )
    long["sensor"] = long["sensor"].map(lambda name: f"{SENSOR_SYMBOL[name]} ({name})")
    figure = px.line(long, x="time_cycles", y="measurement", color="sensor")
    figure.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        legend_title_text="",
        xaxis_title="Released operating cycle",
        yaxis_title="Recorded measurement",
    )
    return figure


def current_simulator(catalog: pd.DataFrame):
    sources = sorted(catalog["source_file"].unique())
    source = st.selectbox("Source file", sources, index=sources.index("test_FD001") if "test_FD001" in sources else 0)
    units = sorted(catalog.loc[catalog["source_file"].eq(source), "unit_number"].unique())
    unit = st.selectbox("Engine", units)
    key = f"{source}:{int(unit)}"
    if st.session_state.get("sim_key") != key:
        st.session_state["sim"] = simulator_for(catalog, source, int(unit))
        st.session_state["sim_key"] = key
        st.session_state["running"] = False
    return st.session_state["sim"]


def render_controls(simulator) -> None:
    speed = st.radio("Playback", SPEEDS, horizontal=True, format_func=lambda value: f"{value}x")
    simulator.set_speed(int(speed))
    start, advance, pause, resume, reset = st.columns(5)
    if start.button("Start", width="stretch"):
        simulator.start()
        st.session_state["running"] = False
    if advance.button("Advance", width="stretch"):
        simulator.resume()
        simulator.advance()
        simulator.pause()
    if pause.button("Pause", width="stretch"):
        simulator.pause()
        st.session_state["running"] = False
    if resume.button("Resume", width="stretch"):
        simulator.resume()
        st.session_state["running"] = True
    if reset.button("Reset", width="stretch"):
        simulator.reset()
        st.session_state["running"] = False
    if st.session_state.get("running") and simulator.playing and not simulator.finished:
        time.sleep(0.35)
        simulator.advance()
        st.rerun()


def render_cycle_status(scored: dict) -> None:
    columns = st.columns(4)
    columns[0].metric("Released cycle", scored["time_cycles"])
    columns[1].metric("Predicted RUL", f"{scored['predicted_rul']:.1f}")
    columns[2].metric("Anomaly", scored["anomaly_label"])
    columns[3].markdown(f"Priority<br>{priority_badge(scored['priority'])}", unsafe_allow_html=True)
    st.caption(
        f"Recorded RUL label for this historical cycle: {scored['recorded_rul']:.0f}. "
        "A live engine would not provide that label."
    )


def page_fleet(data, thresholds: Thresholds) -> None:
    st.header("Fleet overview")
    st.write(
        "The snapshot below scores each official FD001 test engine at its last recorded cycle. "
        "Training-file engines are available for replay, and their unit numbers start again at 1."
    )
    snapshot = fleet_snapshot(data, thresholds)
    simulated = 1 if st.session_state.get("sim") is not None and st.session_state["sim"].cursor else 0
    available = data.frame.groupby(["source_file", "unit_number"]).ngroups
    counts = snapshot["priority"].value_counts()
    anomalies = int((snapshot["anomaly_label"] == "anomalous").sum())
    metrics = st.columns(5)
    metrics[0].metric("Engines available", available)
    metrics[1].metric("Engines in this replay", simulated)
    metrics[2].metric("Average predicted RUL", f"{snapshot['predicted_rul'].mean():.1f}")
    metrics[3].metric("Need attention", int(counts.get("High", 0) + counts.get("Medium", 0)))
    metrics[4].metric("Anomalous at last cycle", anomalies)
    chart = px.bar(
        x=["High", "Medium", "Low"],
        y=[int(counts.get(name, 0)) for name in ("High", "Medium", "Low")],
        color=["High", "Medium", "Low"],
        color_discrete_map=PRIORITY_COLORS,
        labels={"x": "Priority", "y": "Test engines"},
    )
    chart.update_layout(template="plotly_white", showlegend=False, height=320, margin=dict(l=40, r=20, t=20, b=40))
    left, right = st.columns([1, 1.4])
    left.plotly_chart(chart, width="stretch")
    attention = snapshot.loc[snapshot["priority"].isin(["High", "Medium"])]
    right.dataframe(
        attention[["unit_number", "time_cycles", "predicted_rul", "anomaly_label", "priority", "recommendation"]],
        width="stretch",
        hide_index=True,
    )
    st.subheader("Recent anomaly flags")
    flagged = snapshot.loc[snapshot["anomaly_label"].eq("anomalous")]
    if flagged.empty:
        st.write("No test engine is flagged at its last recorded cycle.")
    else:
        st.dataframe(
            flagged[["unit_number", "time_cycles", "predicted_rul", "anomaly_score", "priority"]],
            width="stretch",
            hide_index=True,
        )


def page_monitor(data, thresholds: Thresholds) -> None:
    st.header("Engine monitoring")
    simulator = current_simulator(data.frame)
    render_controls(simulator)
    seen = simulator.seen()
    if seen.empty:
        st.info("Start the replay to release the first recorded cycle.")
        return
    scored = score_history(data, seen, thresholds)
    render_cycle_status(scored)
    st.plotly_chart(sensor_figure(seen), width="stretch")
    st.caption(f"Released {simulator.cursor} of {simulator.n_cycles} recorded cycles.")


def page_maintenance(data, thresholds: Thresholds) -> None:
    st.header("Predictive maintenance")
    simulator = current_simulator(data.frame)
    render_controls(simulator)
    seen = simulator.seen()
    if seen.empty:
        st.info("Release a cycle to generate a recommendation.")
        return
    scored = score_history(data, seen, thresholds)
    render_cycle_status(scored)
    st.subheader(scored["recommendation"])
    for reason in scored["reasons"]:
        st.write(reason)
    st.plotly_chart(sensor_figure(seen), width="stretch")


def page_explain(data) -> None:
    st.header("Model explainability")
    st.write(
        "SHAP attributes the XGBoost prediction to input features. "
        "A large value means the feature moved this model's output. "
        "It does not identify the physical cause of degradation."
    )
    summary_path = ROOT / "models" / "shap_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        table = pd.DataFrame(summary["features"])
        figure = px.bar(table.iloc[::-1], x="mean_abs_shap", y="feature", orientation="h")
        figure.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title="Mean absolute SHAP value",
            yaxis_title="",
            title=f"Training-cycle sample, n={summary['sample_size']}",
        )
        st.plotly_chart(figure, width="stretch")
    else:
        st.info("Run python -m src.models.explain to save the training-sample summary.")

    simulator = st.session_state.get("sim")
    if simulator is None or simulator.seen().empty:
        st.write("Start a replay on Engine monitoring to explain the released cycle.")
        return
    current = simulator.seen().iloc[[-1]]
    contributions = explain_frame(data.rul_model, current, data.features, data.scaler)
    detail = pd.DataFrame(contributions)
    detail["direction"] = detail["shap"].map(lambda value: "increases RUL" if value >= 0 else "decreases RUL")
    figure = px.bar(detail.iloc[::-1], x="shap", y="feature", orientation="h", color="direction")
    figure.update_layout(template="plotly_white", height=420, margin=dict(l=40, r=20, t=30, b=40), title="This released cycle")
    st.plotly_chart(figure, width="stretch")


def page_performance(data) -> None:
    st.header("Model performance")
    comparison = data.comparison
    rows = []
    for name, block in comparison["validation"].items():
        rows.append(
            {
                "model": name,
                "RMSE at 30 cycles remaining": block["at_rul_30"]["rmse"],
                "MAE at 30 cycles remaining": block["at_rul_30"]["mae"],
                "RMSE, every validation cycle": block["all_cycles"]["rmse"],
            }
        )
    table = pd.DataFrame(rows)
    figure = px.bar(
        table,
        x="model",
        y=["RMSE at 30 cycles remaining", "RMSE, every validation cycle"],
        barmode="group",
    )
    figure.update_layout(template="plotly_white", height=360, yaxis_title="RMSE (cycles)", legend_title_text="")
    st.plotly_chart(figure, width="stretch")
    st.dataframe(table, width="stretch", hide_index=True)
    test = comparison["test_last_cycle"]
    st.subheader("Official test engines, scored after model selection")
    st.write(
        f"Selected model: {test['model']}. Last observed cycle only. "
        f"MAE {test['mae']:.2f}, RMSE {test['rmse']:.2f}, NASA sum {test['nasa_score']:.0f}. "
        "The NASA sum is exponential, so a few late predictions dominate it."
    )
    st.subheader("Anomaly checks")
    report = data.anomaly_report
    if report is None:
        st.info("Run python -m src.evaluation.anomaly_evaluation after training the isolation forest.")
        return
    st.write(report["limitation"])
    normal = report["scenario_a_normal"]
    degrading = report["scenario_b_degrading"]
    synthetic = report["scenario_c_synthetic"]
    st.write(
        f"Scenario A, unmodified early validation cycles: {normal['fraction_anomalous']:.1%} "
        f"of {normal['n']} cycles were flagged."
    )
    st.write(
        f"Scenario B, validation unit {degrading['unit_number']}: anomaly-score correlation with cycle is "
        f"{degrading['correlation_anomaly_score_vs_cycle']:.2f}. "
        f"Mean score in the first 20 cycles is {degrading['mean_score_first_20_cycles']:.3f}; "
        f"in the last 20 cycles it is {degrading['mean_score_last_20_cycles']:.3f}."
    )
    st.write(synthetic["label"])
    st.write(
        f"{synthetic['n_identified']} of {synthetic['n_cases']} copies crossed the flag. "
        "The scores move toward unusual and stay below the threshold."
    )
    st.dataframe(pd.DataFrame(synthetic["cases"]), width="stretch", hide_index=True)
    image = ROOT / "notebooks" / "figures" / "anomaly_scenarios.png"
    if image.exists():
        st.image(str(image), caption="Left: one validation engine. Right: synthetic copies.")


def main() -> None:
    data = assets()
    st.sidebar.title("FleetReadiness")
    st.sidebar.caption("Research prototype. Historical C-MAPSS replay, not a live aircraft and not a certified maintenance system.")
    thresholds = thresholds_from_sidebar()
    page = st.sidebar.radio(
        "Page",
        ["Fleet overview", "Engine monitoring", "Predictive maintenance", "Model explainability", "Model performance"],
    )
    pages = {
        "Fleet overview": page_fleet,
        "Engine monitoring": page_monitor,
        "Predictive maintenance": page_maintenance,
        "Model explainability": page_explain,
        "Model performance": page_performance,
    }
    if page in {"Fleet overview", "Engine monitoring", "Predictive maintenance"}:
        pages[page](data, thresholds)
    elif page == "Model explainability":
        page_explain(data)
    else:
        page_performance(data)


if __name__ == "__main__":
    main()
