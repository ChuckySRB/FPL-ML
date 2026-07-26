"""Local Streamlit dashboard for the corrected FPL prediction pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app_data import load_current_predictions
from src.fpl_assistant.streamlit_ui import (
    inject_theme,
    render_weekly_assistant,
)
from src.optimization import aggregate_player_projections, optimize_fpl_squad


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"
PREDICTION_FILE = RESULTS / "predictions_2024_25_fixed.csv"

MODEL_COLUMNS = {
    "XGBoost": "pred_xgboost",
    "Random Forest": "pred_random_forest",
    "Linear Regression": "pred_linear_regression",
    "Last-5 baseline": "pred_last-5_baseline",
    "Tier-0 Form+FDR": "pred_tier-0_formplusfdr",
    "Position ensemble": "pred_position_rfplusxgb_ensemble",
}


@st.cache_data
def load_results() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load all dashboard data from local pipeline outputs."""
    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            "Run `python scripts/run_academic_pipeline.py` before the app."
        )
    predictions = pd.read_csv(PREDICTION_FILE, low_memory=False)
    metrics = pd.read_csv(RESULTS / "model_metrics_fixed.csv")
    summary = json.loads(
        (RESULTS / "model_summary_fixed.json").read_text(encoding="utf-8")
    )
    return predictions, metrics, summary


def metric_table(metrics: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Return the overall model leaderboard for a chosen scope."""
    return (
        metrics[
            (metrics["scope"] == scope)
            & (metrics["group_type"] == "overall")
        ][["model", "n", "mae", "rmse", "r2"]]
        .sort_values("rmse")
        .reset_index(drop=True)
    )


def render_overview(
    metrics: pd.DataFrame,
    summary: dict,
    scope: str,
) -> None:
    """Render model-level academic results."""
    table = metric_table(metrics, scope)
    best = table.iloc[0]
    columns = st.columns(4)
    columns[0].metric("Best model", best["model"])
    columns[1].metric("RMSE", f"{best['rmse']:.3f}")
    columns[2].metric("MAE", f"{best['mae']:.3f}")
    columns[3].metric("Rows", f"{int(best['n']):,}")
    st.image(
        str(FIGURES / "model_comparison_fixed.png"),
        caption="Prospective GW32-38 comparison",
        width="stretch",
    )
    st.dataframe(
        table.style.format(
            {"mae": "{:.3f}", "rmse": "{:.3f}", "r2": "{:.3f}"}
        ),
        width="stretch",
        hide_index=True,
    )
    st.info(summary["methodological_scope"])


def render_player_explorer(
    filtered: pd.DataFrame,
    prediction_column: str,
) -> None:
    """Render player projections and selectable filters."""
    positions = st.multiselect(
        "Positions",
        ["GK", "DEF", "MID", "FWD"],
        default=["GK", "DEF", "MID", "FWD"],
    )
    teams = sorted(filtered["team_name"].dropna().unique())
    selected_teams = st.multiselect("Teams", teams)
    pool = filtered[filtered["position_label"].isin(positions)]
    if selected_teams:
        pool = pool[pool["team_name"].isin(selected_teams)]

    players = aggregate_player_projections(pool, prediction_column)
    actual = (
        pool.groupby("element", as_index=False)["actual_points"]
        .sum()
    )
    players = players.merge(actual, on="element", how="left")
    players["value_score"] = players["projected_points"] / players["price"]
    display = players.sort_values("projected_points", ascending=False)

    st.dataframe(
        display[
            [
                "name",
                "position_label",
                "team_name",
                "fixtures",
                "price",
                "projected_points",
                "actual_points",
                "value_score",
            ]
        ].style.format(
            {
                "price": "£{:.1f}m",
                "projected_points": "{:.2f}",
                "actual_points": "{:.1f}",
                "value_score": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
        height=520,
    )
    chart = px.scatter(
        players,
        x="price",
        y="projected_points",
        color="position_label",
        hover_name="name",
        hover_data=["team_name", "fixtures"],
        labels={"price": "Price (£m)", "projected_points": "Projected points"},
        title="Projection versus price",
    )
    st.plotly_chart(chart, width="stretch")


def render_errors(filtered: pd.DataFrame, prediction_column: str) -> None:
    """Render the largest historical misses for the selected model."""
    if "actual_points" not in filtered or filtered["actual_points"].notna().sum() == 0:
        st.info("Actual outcomes are unavailable for current-season projections.")
        return
    errors = filtered.copy()
    errors["prediction"] = errors[prediction_column]
    errors["error"] = errors["prediction"] - errors["actual_points"]
    errors["absolute_error"] = errors["error"].abs()
    errors = errors.sort_values("absolute_error", ascending=False)
    st.caption(
        "Negative error means the model underestimated the player's return."
    )
    st.dataframe(
        errors[
            [
                "name",
                "position_label",
                "team_name",
                "opponent_team",
                "round",
                "actual_points",
                "prediction",
                "error",
            ]
        ]
        .head(50)
        .style.format({"prediction": "{:.2f}", "error": "{:+.2f}"}),
        width="stretch",
        hide_index=True,
    )
    figure = px.scatter(
        errors,
        x="actual_points",
        y="prediction",
        color="position_label",
        hover_name="name",
        labels={"actual_points": "Actual points", "prediction": "Prediction"},
        title="Actual versus predicted points",
    )
    minimum = errors["actual_points"].min()
    maximum = errors["actual_points"].max()
    figure.add_shape(
        type="line",
        x0=minimum,
        y0=minimum,
        x1=maximum,
        y1=maximum,
        line={"dash": "dash", "color": "#64748b"},
    )
    st.plotly_chart(figure, width="stretch")


def render_optimizer(filtered: pd.DataFrame, prediction_column: str) -> None:
    """Render a legal-squad optimizer for the selected gameweek horizon."""
    st.write(
        "Selects 2 GK, 5 DEF, 5 MID and 3 FWD, with at most three players "
        "per club. Multiple fixtures in the selected horizon are summed."
    )
    budget = st.slider("Budget (£m)", 75.0, 120.0, 100.0, 0.5)
    candidates = aggregate_player_projections(filtered, prediction_column)
    if st.button("Optimize 15-player squad", type="primary"):
        try:
            squad = optimize_fpl_squad(candidates, budget=budget)
        except ValueError as error:
            st.error(str(error))
            return
        columns = st.columns(3)
        columns[0].metric(
            "Projected points", f"{squad['projected_points'].sum():.1f}"
        )
        columns[1].metric("Cost", f"£{squad['price'].sum():.1f}m")
        columns[2].metric("Players", len(squad))
        st.dataframe(
            squad[
                [
                    "name",
                    "position_label",
                    "team_name",
                    "price",
                    "fixtures",
                    "projected_points",
                    "value_score",
                ]
            ].style.format(
                {
                    "price": "£{:.1f}m",
                    "projected_points": "{:.2f}",
                    "value_score": "{:.2f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )


def render_research_lab() -> None:
    """Render the existing academic evaluation workspace."""
    st.title("⚽ FPL Machine Learning Assistant")
    st.caption(
        "Leakage-safe historical evaluation and squad planning from local data"
    )
    try:
        academic_predictions, metrics, summary = load_results()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    live_files = sorted((ROOT / "outputs" / "predictions").glob("predictions_*.csv"))
    source_files = {
        f"Current projection: {path.stem}": path for path in live_files
    }
    with st.sidebar:
        st.header("Analysis scope")
        source_name = st.selectbox(
            "Data source",
            ["Academic 2024/25 test", *source_files],
        )
        if source_name == "Academic 2024/25 test":
            predictions = academic_predictions
            model_name = st.selectbox("Prediction model", list(MODEL_COLUMNS))
            is_live = False
        else:
            raw_seasons = sorted(
                path for path in (ROOT / "data" / "raw").glob("20??-??")
                if (path / "players_raw.csv").exists()
            )
            if not raw_seasons:
                st.error("No current-season player metadata is available.")
                st.stop()
            try:
                predictions = load_current_predictions(
                    source_files[source_name], raw_seasons[-1]
                )
            except (FileNotFoundError, ValueError) as error:
                st.error(str(error))
                st.stop()
            model_name = "XGBoost"
            is_live = True
            st.caption(f"Metadata: {raw_seasons[-1].name}")

        eligible = predictions[predictions["evaluation_eligible"]].copy()
        available_rounds = sorted(
            int(value) for value in eligible["round"].unique()
        )
        default_rounds = (
            available_rounds
            if is_live
            else [value for value in available_rounds if value >= 32]
        )
        selected_rounds = st.multiselect(
            "Gameweeks",
            available_rounds,
            default=default_rounds,
        )
        scope = st.radio(
            "Historical model leaderboard",
            ["Prospective GW32-38", "Full 2024/25 test"],
        )
        st.caption("All calculations stay on this computer.")

    if not selected_rounds:
        st.warning("Select at least one gameweek.")
        st.stop()
    filtered = eligible[eligible["round"].isin(selected_rounds)].copy()
    prediction_column = MODEL_COLUMNS[model_name]
    metric_scope = (
        "prospective_gw32_38"
        if scope == "Prospective GW32-38"
        else "full_test"
    )

    tabs = st.tabs(
        [
            "Model results",
            "Player explorer",
            "Error analysis",
            "Squad optimizer",
            "Methodology",
        ]
    )
    with tabs[0]:
        render_overview(metrics, summary, metric_scope)
    with tabs[1]:
        render_player_explorer(filtered, prediction_column)
    with tabs[2]:
        render_errors(filtered, prediction_column)
    with tabs[3]:
        render_optimizer(filtered, prediction_column)
    with tabs[4]:
        st.subheader("Reproducible workflow")
        st.markdown(
            """
            1. Build the corrected temporal dataset.
            2. Train baselines and ML models on 2020/21-2023/24.
            3. Evaluate only on held-out 2024/25 fixtures.
            4. Use GW32-38 as the primary prospective comparison.

            The current result is a **partial OpenFPL reproduction** because
            the public inputs do not include every Understat and availability
            feature used in the paper.
            """
        )
        st.image(
            str(FIGURES / "feature_importance_fixed.png"),
            width="stretch",
        )


def main() -> None:
    """Run the production assistant with the research view preserved."""
    st.set_page_config(
        page_title="FPL Command Center",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    with st.sidebar:
        st.markdown("## ⚽ FPL Assistant")
        workspace = st.radio(
            "Радни простор",
            ["Недељни асистент", "Истраживање модела"],
            label_visibility="collapsed",
        )
        st.divider()
    if workspace == "Недељни асистент":
        render_weekly_assistant()
    else:
        render_research_lab()


if __name__ == "__main__":
    main()
