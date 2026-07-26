"""Adapters that normalize current-season prediction exports for the app."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_current_predictions(
    prediction_file: Path,
    raw_season_dir: Path,
) -> pd.DataFrame:
    """Convert ``predict_next_5_gameweeks.py`` output to dashboard columns."""
    predictions = pd.read_csv(prediction_file)
    required = {
        "element",
        "name",
        "position_label",
        "team",
        "gw",
        "predicted_points",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"Unsupported current prediction file; missing {sorted(missing)}"
        )

    if "has_fixture" in predictions:
        predictions = predictions[predictions["has_fixture"]].copy()

    teams_file = raw_season_dir / "teams.csv"
    players_file = raw_season_dir / "players_raw.csv"
    if not teams_file.exists() or not players_file.exists():
        raise FileNotFoundError(
            f"Current player/team metadata is missing in {raw_season_dir}"
        )
    teams = (
        pd.read_csv(teams_file, usecols=["id", "name"])
        .rename(columns={"id": "team", "name": "team_name"})
    )
    players = (
        pd.read_csv(players_file, usecols=["id", "now_cost", "status", "news"])
        .rename(columns={"id": "element"})
    )
    normalized = (
        predictions.merge(teams, on="team", how="left")
        .merge(players, on="element", how="left")
        .rename(
            columns={
                "gw": "round",
                "predicted_points": "pred_xgboost",
            }
        )
    )
    normalized["price"] = normalized["now_cost"] / 10
    normalized["actual_points"] = np.nan
    normalized["opponent_team"] = "Upcoming fixture"
    normalized["evaluation_eligible"] = normalized["position_label"].isin(
        ["GK", "DEF", "MID", "FWD"]
    )
    normalized["season"] = raw_season_dir.name
    return normalized
