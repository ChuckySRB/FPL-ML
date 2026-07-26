"""Data preparation for the dual-horizon production FPL models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from configs.config import RAW_DATA_DIR
from src.fpl_assistant.defensive_points import (
    add_defensive_contribution_target,
    train_defensive_contribution_estimator,
)
from src.fpl_assistant.targets import build_five_gw_training_rows
from src.preprocessing.data_loader import FPLDataLoader
from src.preprocessing.feature_engineering import (
    FPLFeatureEngineer,
    TIER2_FEATURES,
)


POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD", 5: "AM"}
ELIGIBLE_POSITIONS = {"GK", "DEF", "MID", "FWD"}
PERFORMANCE_COLUMNS = [
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "clearances_blocks_interceptions",
    "recoveries",
    "tackles",
    "defensive_contribution",
    "xP",
]

CROSS_SEASON_FORM_FEATURES = [
    "form_last_3",
    "form_last_5",
    "minutes_last_3",
    "ict_index_last_3",
    "goals_last_5",
    "assists_last_5",
    "clean_sheets_last_5",
    "bps_last_5",
    "influence_last_5",
    "creativity_last_5",
    "threat_last_5",
    "xG_last_5",
    "xA_last_5",
    "xGC_last_5",
    "saves_last_5",
    "yellow_cards_last_5",
    "bonus_last_5",
    "minutes_last_5",
]


def available_seasons(
    raw_data_dir: Path = RAW_DATA_DIR,
    first_season: str = "2020-21",
    last_season: str | None = None,
) -> list[str]:
    """Discover locally available seasons with gameweek data."""
    seasons = []
    for path in raw_data_dir.iterdir():
        if not path.is_dir() or not path.name[:4].isdigit():
            continue
        gameweek_dir = path / "gws"
        if not gameweek_dir.exists():
            continue
        if path.name < first_season:
            continue
        if last_season is not None and path.name > last_season:
            continue
        seasons.append(path.name)
    return sorted(seasons)


def load_player_metadata(
    season: str,
    raw_data_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Load current player identity, position, price, status, and team."""
    path = raw_data_dir / season / "players_raw.csv"
    if not path.exists():
        return pd.DataFrame()
    players = pd.read_csv(path, low_memory=False)
    required = {"id", "element_type", "team"}
    if not required.issubset(players.columns):
        return pd.DataFrame()
    keep = [
        column
        for column in [
            "id",
            "code",
            "web_name",
            "first_name",
            "second_name",
            "element_type",
            "team",
            "now_cost",
            "selected_by_percent",
            "status",
            "news",
            "chance_of_playing_next_round",
            "ep_next",
            "points_per_game",
            "total_points",
            "minutes",
            "starts",
        ]
        if column in players
    ]
    metadata = players[keep].copy().rename(
        columns={"id": "element", "web_name": "name"}
    )
    if "name" not in metadata:
        first_name = metadata.get(
            "first_name",
            pd.Series("", index=metadata.index, dtype="string"),
        )
        second_name = metadata.get(
            "second_name",
            pd.Series("", index=metadata.index, dtype="string"),
        )
        metadata["name"] = (
            first_name.fillna("").astype(str)
            + " "
            + second_name.fillna("").astype(str)
        ).str.strip()
    metadata["position_label"] = metadata["element_type"].map(POSITION_MAP)
    if "now_cost" in metadata:
        metadata["current_price"] = pd.to_numeric(
            metadata["now_cost"], errors="coerce"
        ) / 10.0
    if "selected_by_percent" in metadata:
        bootstrap_path = raw_data_dir / season / "bootstrap_static.json"
        total_players = np.nan
        if bootstrap_path.exists():
            try:
                total_players = float(
                    json.loads(
                        bootstrap_path.read_text(encoding="utf-8")
                    ).get("total_players", np.nan)
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                total_players = np.nan
        selected_percent = pd.to_numeric(
            metadata["selected_by_percent"], errors="coerce"
        )
        if pd.notna(total_players) and total_players > 0:
            estimated_selected = selected_percent.mul(total_players / 100.0)
            metadata["selected_pct"] = np.log1p(estimated_selected)
    return metadata


def current_team_strength_map(
    season: str,
    loader: FPLDataLoader,
) -> dict[int, float]:
    """Load team strength, falling back to difficulty imposed on opponents."""
    try:
        teams = loader.load_teams(season)
    except FileNotFoundError:
        return {}
    if {"id", "strength"}.issubset(teams.columns):
        strength = pd.to_numeric(teams["strength"], errors="coerce")
        if strength.notna().any() and strength.abs().sum() > 0:
            return dict(zip(teams["id"].astype(int), strength.astype(float)))
    try:
        fixtures = loader.load_fixtures(season)
    except FileNotFoundError:
        return {}
    imposed: dict[int, list[float]] = {}
    for _, fixture in fixtures.iterrows():
        home = fixture.get("team_h")
        away = fixture.get("team_a")
        home_imposed = fixture.get("team_a_difficulty")
        away_imposed = fixture.get("team_h_difficulty")
        if pd.notna(home) and pd.notna(home_imposed):
            imposed.setdefault(int(home), []).append(float(home_imposed))
        if pd.notna(away) and pd.notna(away_imposed):
            imposed.setdefault(int(away), []).append(float(away_imposed))
    return {
        team: float(np.mean(difficulties))
        for team, difficulties in imposed.items()
        if difficulties
    }


def enrich_player_metadata(
    frame: pd.DataFrame,
    season: str,
    loader: FPLDataLoader,
) -> pd.DataFrame:
    """Fill identity, position, team name, and position one-hot columns."""
    result = frame.copy()
    metadata = load_player_metadata(season, raw_data_dir=loader.data_dir)
    if not metadata.empty:
        merge_columns = [
            column
            for column in [
                "element",
                "code",
                "name",
                "position_label",
                "element_type",
                "team",
            ]
            if column in metadata
        ]
        result = result.merge(
            metadata[merge_columns],
            on="element",
            how="left",
            suffixes=("", "_metadata"),
        )
        for column in ["code", "name", "position_label", "element_type", "team"]:
            metadata_column = f"{column}_metadata"
            if metadata_column not in result:
                continue
            if column not in result:
                result[column] = result[metadata_column]
            else:
                result[column] = result[column].fillna(result[metadata_column])
            result = result.drop(columns=metadata_column)
    if "position_label" not in result and "element_type" in result:
        result["position_label"] = result["element_type"].map(POSITION_MAP)
    result["position_label"] = result["position_label"].replace({"GKP": "GK"})

    try:
        teams = loader.load_teams(season)
    except FileNotFoundError:
        teams = pd.DataFrame()
    if not teams.empty and {"id", "name"}.issubset(teams.columns):
        team_names = teams[["id", "name"]].rename(
            columns={"id": "team", "name": "team_name"}
        )
        result = result.merge(team_names, on="team", how="left")
    return result


def load_raw_season(
    season: str,
    loader: FPLDataLoader | None = None,
) -> pd.DataFrame:
    """Load and enrich one season's fixture-level rows."""
    loader = loader or FPLDataLoader()
    frame = loader.load_gameweeks(season)
    frame = enrich_player_metadata(frame, season, loader)
    frame["season"] = season
    return frame


def train_dc_estimator_from_season(
    scoring_season: str,
    loader: FPLDataLoader | None = None,
) -> dict[str, Any]:
    """Train the historical DC reconstruction model on an exact rules season."""
    exact = load_raw_season(scoring_season, loader=loader)
    return train_defensive_contribution_estimator(exact)


def engineer_adjusted_season(
    season: str,
    dc_estimator: dict[str, Any],
    loader: FPLDataLoader | None = None,
    engineer: FPLFeatureEngineer | None = None,
    cutoff_before_gw: int | None = None,
) -> pd.DataFrame:
    """Apply current scoring to raw history, then create leakage-safe features."""
    loader = loader or FPLDataLoader()
    engineer = engineer or FPLFeatureEngineer()
    raw = load_raw_season(season, loader=loader)
    if cutoff_before_gw is not None:
        raw = raw[raw["round"].lt(cutoff_before_gw)].copy()
    adjusted = add_defensive_contribution_target(
        raw,
        season=season,
        estimator_bundle=dc_estimator,
    )
    adjusted["official_total_points"] = adjusted["base_total_points"]
    adjusted["total_points"] = adjusted["adjusted_total_points"]
    try:
        teams = loader.load_teams(season)
    except FileNotFoundError:
        teams = None
    try:
        fixtures = loader.load_fixtures(season)
    except FileNotFoundError:
        fixtures = None
    return engineer.create_all_features(
        adjusted,
        teams_df=teams,
        fixtures_df=fixtures,
        tier=2,
    )


def build_dual_training_frames(
    seasons: list[str],
    scoring_season: str = "2025-26",
    min_gameweek: int = 6,
) -> dict[str, Any]:
    """Build single-fixture and direct-five-GW production training frames."""
    if scoring_season not in seasons:
        raise ValueError(
            f"{scoring_season} is required to estimate historical DC bonuses"
        )
    loader = FPLDataLoader()
    engineer = FPLFeatureEngineer()
    dc_estimator = train_dc_estimator_from_season(
        scoring_season,
        loader=loader,
    )
    single_frames = []
    five_frames = []
    adjustment_summary = []

    for season in seasons:
        engineered = engineer_adjusted_season(
            season,
            dc_estimator=dc_estimator,
            loader=loader,
            engineer=engineer,
        )
        engineered = engineered[
            engineered["position_label"].isin(ELIGIBLE_POSITIONS)
        ].copy()
        eligible_single = engineered[
            engineered["round"].ge(min_gameweek)
            & engineered["total_points"].notna()
        ].copy()
        single_frames.append(eligible_single)
        five_frames.append(
            build_five_gw_training_rows(
                engineered,
                base_features=TIER2_FEATURES,
                points_column="total_points",
                horizon=5,
                min_gameweek=min_gameweek,
            )
        )
        adjustment_summary.append(
            {
                "season": season,
                "rows": int(len(engineered)),
                "adjustment_source": str(
                    engineered["dc_adjustment_source"].mode().iloc[0]
                ),
                "estimated_or_added_dc_points": float(
                    engineered["dc_target_adjustment"].sum()
                ),
            }
        )

    single = pd.concat(single_frames, ignore_index=True, sort=False)
    five = pd.concat(five_frames, ignore_index=True, sort=False)
    features = [column for column in TIER2_FEATURES if column in single]
    return {
        "single": single,
        "five": five,
        "base_features": features,
        "dc_estimator": dc_estimator,
        "adjustment_summary": adjustment_summary,
    }


def _previous_season_with_history(
    season: str,
    loader: FPLDataLoader,
) -> str | None:
    """Return the latest earlier local season containing gameweek data."""
    candidates = []
    for path in loader.data_dir.iterdir():
        if not path.is_dir() or path.name >= season:
            continue
        gameweek_dir = path / "gws"
        if not gameweek_dir.is_dir():
            continue
        if (gameweek_dir / "merged_gw.csv").exists() or any(
            gameweek_dir.glob("gw*.csv")
        ):
            candidates.append(path.name)
    return max(candidates, default=None)


def _metadata_only_state(
    metadata: pd.DataFrame,
    base_features: list[str],
    season: str,
    loader: FPLDataLoader,
) -> pd.DataFrame:
    """Create a pre-GW1 state from current metadata and neutral season totals."""
    if metadata.empty:
        raise FileNotFoundError(
            f"Player metadata not found for season {season}; "
            "collect players_raw.csv before generating predictions"
        )
    state = metadata.copy()
    for feature in base_features:
        if feature not in state:
            state[feature] = np.nan
    if "current_price" in state:
        state["price"] = state["current_price"]
    for position in ["DEF", "MID", "FWD"]:
        state[f"pos_{position}"] = (
            state["position_label"].eq(position).astype(int)
        )
    state["cumulative_points_season"] = 0.0
    state["games_played_season"] = 0.0
    try:
        teams = loader.load_teams(season)
    except FileNotFoundError:
        teams = pd.DataFrame()
    strength = current_team_strength_map(season, loader)
    if strength:
        state["team_strength"] = state["team"].map(strength)
    if {"id", "name"}.issubset(teams.columns):
        team_names = teams.set_index("id")["name"]
        state["team_name"] = state["team"].map(team_names)
    state["history_source_season"] = pd.NA
    return state


def _seed_previous_season_form(
    state: pd.DataFrame,
    season: str,
    base_features: list[str],
    loader: FPLDataLoader,
) -> pd.DataFrame:
    """Carry recent form for returning players into a new season's GW1."""
    if "code" not in state or state["code"].isna().all():
        return state
    previous_season = _previous_season_with_history(season, loader)
    if previous_season is None:
        return state
    previous_raw = load_raw_season(previous_season, loader=loader)
    previous_round = pd.to_numeric(
        previous_raw["round"], errors="coerce"
    ).max()
    if pd.isna(previous_round):
        return state
    previous_state = build_player_state(
        previous_season,
        gameweek=int(previous_round) + 1,
        base_features=base_features,
        loader=loader,
        seed_previous_season=False,
    )
    transferable = [
        feature
        for feature in CROSS_SEASON_FORM_FEATURES
        if feature in base_features and feature in previous_state
    ]
    if not transferable or "code" not in previous_state:
        return state
    previous_values = (
        previous_state[["code", *transferable]]
        .dropna(subset=["code"])
        .drop_duplicates("code", keep="last")
        .rename(
            columns={
                feature: f"{feature}_previous"
                for feature in transferable
            }
        )
    )
    result = state.merge(previous_values, on="code", how="left")
    matched = pd.Series(False, index=result.index)
    for feature in transferable:
        previous_column = f"{feature}_previous"
        available = result[previous_column].notna()
        result.loc[available, feature] = result.loc[available, previous_column]
        matched |= available
        result = result.drop(columns=previous_column)
    result.loc[matched, "history_source_season"] = previous_season
    return result


def build_player_state(
    season: str,
    gameweek: int,
    base_features: list[str],
    loader: FPLDataLoader | None = None,
    seed_previous_season: bool = True,
) -> pd.DataFrame:
    """Build one pre-deadline state row per current player for a selected GW."""
    loader = loader or FPLDataLoader()
    engineer = FPLFeatureEngineer()
    metadata = load_player_metadata(season, raw_data_dir=loader.data_dir)
    try:
        raw = load_raw_season(season, loader=loader)
    except FileNotFoundError as error:
        if gameweek != 1:
            raise FileNotFoundError(
                f"Completed gameweek data is required to predict "
                f"{season} GW{gameweek}. Refresh the current-season data first."
            ) from error
        state = _metadata_only_state(
            metadata,
            base_features=base_features,
            season=season,
            loader=loader,
        )
        if seed_previous_season:
            state = _seed_previous_season_form(
                state,
                season=season,
                base_features=base_features,
                loader=loader,
            )
        return state.reset_index(drop=True)

    history = raw[raw["round"].lt(gameweek)].copy()

    if history.empty:
        state = _metadata_only_state(
            metadata,
            base_features=base_features,
            season=season,
            loader=loader,
        )
        if gameweek == 1 and seed_previous_season:
            state = _seed_previous_season_form(
                state,
                season=season,
                base_features=base_features,
                loader=loader,
            )
        return state.reset_index(drop=True)

    sort_columns = ["element", "round"]
    if "kickoff_time" in history:
        sort_columns.append("kickoff_time")
    history = history.sort_values(sort_columns, na_position="last")
    snapshot = history.groupby("element", as_index=False).tail(1).copy()
    snapshot["round"] = gameweek
    snapshot["_is_snapshot"] = True
    for column in PERFORMANCE_COLUMNS:
        if column in snapshot:
            snapshot[column] = np.nan
    for column in ["fixture", "opponent_team", "kickoff_time"]:
        if column in snapshot:
            snapshot[column] = np.nan
    history["_is_snapshot"] = False
    combined = pd.concat([history, snapshot], ignore_index=True, sort=False)
    try:
        teams = loader.load_teams(season)
    except FileNotFoundError:
        teams = None
    try:
        fixtures = loader.load_fixtures(season)
    except FileNotFoundError:
        fixtures = None
    engineered = engineer.create_all_features(
        combined,
        teams_df=teams,
        fixtures_df=fixtures,
        tier=2,
    )
    state = engineered[engineered["_is_snapshot"]].copy()
    state = state.drop(columns="_is_snapshot", errors="ignore")
    state["history_source_season"] = season

    if not metadata.empty:
        metadata_columns = [
            column
            for column in [
                "element",
                "code",
                "name",
                "position_label",
                "element_type",
                "team",
                "current_price",
                "selected_by_percent",
                "selected_pct",
                "status",
                "news",
                "chance_of_playing_next_round",
                "ep_next",
                "points_per_game",
            ]
            if column in metadata
        ]
        state = state.merge(
            metadata[metadata_columns],
            on="element",
            how="outer",
            suffixes=("", "_current"),
        )
        for column in metadata_columns:
            if column == "element":
                continue
            current_column = f"{column}_current"
            if current_column not in state:
                continue
            if column not in state:
                state[column] = state[current_column]
            else:
                state[column] = state[current_column].where(
                    state[current_column].notna(), state[column]
                )
            state = state.drop(columns=current_column)
    if "current_price" in state:
        state["price"] = state["current_price"].combine_first(
            state.get("price", pd.Series(index=state.index, dtype=float))
        )
    for feature in base_features:
        if feature not in state:
            state[feature] = np.nan
    return state.reset_index(drop=True)
