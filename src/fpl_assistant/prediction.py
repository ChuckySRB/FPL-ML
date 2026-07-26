"""Selectable-gameweek predictions for the dual-horizon FPL assistant."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.fpl_assistant.data import build_player_state, current_team_strength_map
from src.fpl_assistant.models import load_dual_models, predict_bundle
from src.fpl_assistant.targets import FIVE_GW_SCHEDULE_FEATURES
from src.preprocessing.data_loader import FPLDataLoader


def early_season_current_weight(gameweek: int) -> float:
    """Increase current-season evidence from 0 at GW1 to 1 from GW6."""
    return float(np.clip((gameweek - 1) / 5.0, 0.0, 1.0))


def available_prediction_gameweeks(season: str) -> list[int]:
    """Return gameweeks that have at least one scheduled fixture."""
    fixtures = FPLDataLoader().load_fixtures(season)
    event_column = "event" if "event" in fixtures else "round"
    return sorted(
        pd.to_numeric(fixtures[event_column], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )


def _fixture_lookup(
    fixtures: pd.DataFrame,
    gameweeks: list[int],
) -> dict[int, dict[int, list[dict[str, Any]]]]:
    """Build a team -> GW -> fixture list lookup."""
    lookup: dict[int, dict[int, list[dict[str, Any]]]] = {}
    if fixtures.empty:
        return lookup
    event_column = "event" if "event" in fixtures else "round"
    for _, fixture in fixtures.iterrows():
        gameweek = fixture.get(event_column)
        if pd.isna(gameweek) or int(gameweek) not in gameweeks:
            continue
        gameweek = int(gameweek)
        home_team = int(fixture["team_h"])
        away_team = int(fixture["team_a"])
        fixture_id = int(fixture.get("id", 0))
        home_difficulty = float(fixture.get("team_h_difficulty", 3) or 3)
        away_difficulty = float(fixture.get("team_a_difficulty", 3) or 3)
        lookup.setdefault(home_team, {}).setdefault(gameweek, []).append(
            {
                "fixture": fixture_id,
                "opponent_team": away_team,
                "opponent_difficulty": home_difficulty,
                "was_home": 1.0,
            }
        )
        lookup.setdefault(away_team, {}).setdefault(gameweek, []).append(
            {
                "fixture": fixture_id,
                "opponent_team": home_team,
                "opponent_difficulty": away_difficulty,
                "was_home": 0.0,
            }
        )
    return lookup


def build_future_fixture_rows(
    player_state: pd.DataFrame,
    season: str,
    start_gameweek: int,
    horizon: int = 5,
) -> tuple[pd.DataFrame, list[int]]:
    """Create fixture-level feature rows for the selected five-GW window."""
    gameweeks = list(range(start_gameweek, start_gameweek + horizon))
    loader = FPLDataLoader()
    fixtures = loader.load_fixtures(season)
    lookup = _fixture_lookup(fixtures, gameweeks)
    teams = loader.load_teams(season)
    strength_map = current_team_strength_map(season, loader)
    team_name_map = (
        teams.set_index("id")["name"].to_dict()
        if {"id", "name"}.issubset(teams.columns)
        else {}
    )
    rows = []
    for _, player in player_state.iterrows():
        if pd.isna(player.get("element")):
            continue
        element = int(player["element"])
        team = int(player["team"]) if pd.notna(player.get("team")) else 0
        base = player.to_dict()
        base["element"] = element
        base["team"] = team
        base["team_name"] = team_name_map.get(team, base.get("team_name", ""))
        for gameweek in gameweeks:
            scheduled = lookup.get(team, {}).get(gameweek, [])
            if not scheduled:
                rows.append(
                    {
                        **base,
                        "gw": gameweek,
                        "has_fixture": False,
                        "fixture": np.nan,
                        "opponent_team": np.nan,
                        "opponent_name": "Blank",
                    }
                )
                continue
            for fixture in scheduled:
                opponent = fixture["opponent_team"]
                rows.append(
                    {
                        **base,
                        **fixture,
                        "gw": gameweek,
                        "has_fixture": True,
                        "opponent_name": team_name_map.get(opponent, str(opponent)),
                        "opponent_strength": float(
                            strength_map.get(opponent, np.nan)
                        ),
                    }
                )
    return pd.DataFrame(rows), gameweeks


def build_five_gw_prediction_rows(
    player_state: pd.DataFrame,
    fixture_rows: pd.DataFrame,
    start_gameweek: int,
    gameweeks: list[int],
) -> pd.DataFrame:
    """Create one schedule-aware row per player for the direct five-GW model."""
    rows = []
    for _, player in player_state.iterrows():
        if pd.isna(player.get("element")):
            continue
        element = int(player["element"])
        player_fixtures = fixture_rows[fixture_rows["element"].eq(element)]
        actual_fixtures = player_fixtures[
            player_fixtures["has_fixture"].astype(bool)
        ]
        current = actual_fixtures[actual_fixtures["gw"].eq(start_gameweek)]
        counts = actual_fixtures.groupby("gw").size()
        schedule = {
            "fixtures_next_5_gws": float(len(actual_fixtures)),
            "blank_gws_next_5": float(
                sum(gameweek not in counts.index for gameweek in gameweeks)
            ),
            "double_gws_next_5": float((counts > 1).sum()),
            "fdr_mean_next_5": actual_fixtures[
                "opponent_difficulty"
            ].mean(),
            "fdr_min_next_5": actual_fixtures[
                "opponent_difficulty"
            ].min(),
            "fdr_max_next_5": actual_fixtures[
                "opponent_difficulty"
            ].max(),
            "home_fixture_rate_next_5": actual_fixtures["was_home"].mean(),
            "opponent_strength_mean_next_5": actual_fixtures[
                "opponent_strength"
            ].mean(),
        }
        row = player.to_dict()
        row.update(schedule)
        row["was_home"] = current["was_home"].mean()
        row["opponent_difficulty"] = current[
            "opponent_difficulty"
        ].mean()
        row["opponent_strength"] = current["opponent_strength"].mean()
        rows.append(row)
    result = pd.DataFrame(rows)
    for feature in FIVE_GW_SCHEDULE_FEATURES:
        if feature not in result:
            result[feature] = np.nan
    return result


def generate_dual_predictions(
    season: str,
    gameweek: int,
    model_directory: Path,
    horizon: int = 5,
) -> dict[str, Any]:
    """Generate independent current-GW and direct next-five-GW forecasts."""
    if horizon != 5:
        raise ValueError("The direct long-horizon model requires horizon=5")
    bundles = load_dual_models(model_directory)
    has_preseason = {
        "preseason_single",
        "preseason_five_gw",
    }.issubset(bundles)
    forecast_mode = "production_gw6_plus"
    current_history_weight = 1.0

    if gameweek == 1 and has_preseason:
        from src.fpl_assistant.preseason import build_preseason_prediction_rows

        prepared = build_preseason_prediction_rows(
            season,
            start_gameweek=gameweek,
            base_features=bundles["preseason_single"]["features"],
        )
        player_state = prepared["state"]
        fixture_rows = prepared["fixture_rows"]
        five_rows = prepared["five_rows"]
        gameweeks = prepared["gameweeks"]
        single_bundle = bundles["preseason_single"]
        five_bundle = bundles["preseason_five_gw"]
        forecast_mode = "preseason_gw1"
        current_history_weight = 0.0
    else:
        base_features = bundles["single"]["features"]
        player_state = build_player_state(
            season,
            gameweek=gameweek,
            base_features=base_features,
        )
        fixture_rows, gameweeks = build_future_fixture_rows(
            player_state,
            season=season,
            start_gameweek=gameweek,
            horizon=horizon,
        )
        five_rows = build_five_gw_prediction_rows(
            player_state,
            fixture_rows,
            start_gameweek=gameweek,
            gameweeks=gameweeks,
        )
        single_bundle = bundles["single"]
        five_bundle = bundles["five_gw"]

    if fixture_rows.empty or not fixture_rows["has_fixture"].any():
        raise ValueError(
            f"No scheduled fixtures found for {season} from GW{gameweek}"
        )

    fixture_rows["predicted_points_one_fixture"] = predict_bundle(
        single_bundle,
        fixture_rows,
    )
    five_rows["predicted_average_next_5_gws"] = predict_bundle(
        five_bundle,
        five_rows,
    )

    if 2 <= gameweek <= 5 and has_preseason:
        from src.fpl_assistant.preseason import build_preseason_prediction_rows

        preseason = build_preseason_prediction_rows(
            season,
            start_gameweek=gameweek,
            base_features=bundles["preseason_single"]["features"],
        )
        preseason_fixtures = preseason["fixture_rows"].copy()
        preseason_fixtures["preseason_prediction"] = predict_bundle(
            bundles["preseason_single"],
            preseason_fixtures,
        )
        fixture_keys = ["element", "gw", "fixture"]
        preseason_fixture_map = preseason_fixtures[
            fixture_keys + ["preseason_prediction"]
        ].drop_duplicates(fixture_keys)
        fixture_rows = fixture_rows.merge(
            preseason_fixture_map,
            on=fixture_keys,
            how="left",
        )
        current_history_weight = early_season_current_weight(gameweek)
        fixture_rows["predicted_points_one_fixture"] = (
            current_history_weight
            * fixture_rows["predicted_points_one_fixture"]
            + (1.0 - current_history_weight)
            * fixture_rows["preseason_prediction"].fillna(
                fixture_rows["predicted_points_one_fixture"]
            )
        )
        preseason_five = preseason["five_rows"].copy()
        preseason_five["preseason_five_prediction"] = predict_bundle(
            bundles["preseason_five_gw"],
            preseason_five,
        )
        five_rows = five_rows.merge(
            preseason_five[["element", "preseason_five_prediction"]],
            on="element",
            how="left",
        )
        five_rows["predicted_average_next_5_gws"] = (
            current_history_weight
            * five_rows["predicted_average_next_5_gws"]
            + (1.0 - current_history_weight)
            * five_rows["preseason_five_prediction"].fillna(
                five_rows["predicted_average_next_5_gws"]
            )
        )
        forecast_mode = "early_season_blend"

    fixture_rows.loc[
        ~fixture_rows["has_fixture"].astype(bool),
        "predicted_points_one_fixture",
    ] = 0.0
    current_rows = fixture_rows[fixture_rows["gw"].eq(gameweek)].copy()
    current_summary = (
        current_rows.groupby("element", as_index=False)
        .agg(
            predicted_points_current_gw=(
                "predicted_points_one_fixture",
                "sum",
            ),
            current_gw_fixtures=("has_fixture", "sum"),
        )
    )

    metadata_columns = [
        column
        for column in [
            "element",
            "name",
            "position_label",
            "team",
            "team_name",
            "current_price",
            "status",
            "news",
            "chance_of_playing_next_round",
            "history_source_season",
            "ep_next",
            "points_per_game",
            "selected_by_percent",
        ]
        if column in five_rows
    ]
    player_predictions = (
        five_rows[
            metadata_columns
            + FIVE_GW_SCHEDULE_FEATURES
            + ["predicted_average_next_5_gws"]
        ]
        .merge(current_summary, on="element", how="left")
    )
    player_predictions["predicted_points_current_gw"] = player_predictions[
        "predicted_points_current_gw"
    ].fillna(0.0)
    player_predictions["current_gw_fixtures"] = player_predictions[
        "current_gw_fixtures"
    ].fillna(0).astype(int)
    return {
        "season": season,
        "gameweek": gameweek,
        "gameweeks": gameweeks,
        "fixture_predictions": fixture_rows,
        "player_predictions": player_predictions,
        "model_metadata": {
            "forecast_mode": forecast_mode,
            "current_history_weight": current_history_weight,
            "one_fixture": {
                "target": single_bundle["target"],
                "validation": single_bundle["validation"],
            },
            "five_gw": {
                "target": five_bundle["target"],
                "validation": five_bundle["validation"],
            },
            "player_state": {
                "players": int(len(player_state)),
                "previous_season_history_players": int(
                    player_state.get(
                        "history_source_season",
                        pd.Series(pd.NA, index=player_state.index),
                    ).notna().sum()
                ),
                "imputed_history_players": int(
                    player_state.get(
                        "history_source_season",
                        pd.Series(pd.NA, index=player_state.index),
                    ).isna().sum()
                ),
            },
        },
    }
