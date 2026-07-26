"""Leakage-safe cross-season training and prediction for preseason/GW1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.fpl_assistant.data import (
    ELIGIBLE_POSITIONS,
    build_player_state,
    current_team_strength_map,
    load_raw_season,
    train_dc_estimator_from_season,
)
from src.fpl_assistant.defensive_points import add_defensive_contribution_target
from src.fpl_assistant.models import fit_production_bundle
from src.fpl_assistant.prediction import (
    build_five_gw_prediction_rows,
    build_future_fixture_rows,
)
from src.fpl_assistant.targets import FIVE_GW_SCHEDULE_FEATURES, FIVE_GW_TARGET
from src.preprocessing.data_loader import FPLDataLoader
from src.preprocessing.feature_engineering import TIER2_FEATURES


PRESEASON_EXCLUDED_FEATURES = {
    "cumulative_points_season",
    "games_played_season",
}
PRESEASON_BASE_FEATURES = [
    feature
    for feature in TIER2_FEATURES
    if feature not in PRESEASON_EXCLUDED_FEATURES
]
PRESEASON_SINGLE_TARGET = "total_points"


def _apply_observed_gw1_context(
    state: pd.DataFrame,
    raw: pd.DataFrame,
    season: str,
    loader: FPLDataLoader,
) -> pd.DataFrame:
    """Use price, ownership, club, and position observed in historical GW1."""
    gw1 = raw[pd.to_numeric(raw["round"], errors="coerce").eq(1)].copy()
    if gw1.empty:
        return state
    context_columns = [
        column
        for column in [
            "element",
            "team",
            "position_label",
            "element_type",
            "value",
            "selected",
        ]
        if column in gw1
    ]
    context = (
        gw1[context_columns]
        .sort_values("element")
        .groupby("element", as_index=False)
        .first()
    )
    context = context.rename(
        columns={
            column: f"{column}_gw1"
            for column in context.columns
            if column != "element"
        }
    )
    result = state.merge(context, on="element", how="inner")
    for column in ["team", "position_label", "element_type"]:
        observed = f"{column}_gw1"
        if observed in result:
            result[column] = (
                result[observed].where(result[observed].notna(), result[column])
                if column in result
                else result[observed]
            )
            result = result.drop(columns=observed)
    if "value_gw1" in result:
        result["price"] = pd.to_numeric(
            result["value_gw1"], errors="coerce"
        ) / 10.0
        result = result.drop(columns="value_gw1")
    if "selected_gw1" in result:
        result["selected_pct"] = np.log1p(
            pd.to_numeric(result["selected_gw1"], errors="coerce")
        )
        result = result.drop(columns="selected_gw1")
    for position in ["DEF", "MID", "FWD"]:
        result[f"pos_{position}"] = (
            result["position_label"].eq(position).astype(int)
        )
    strength = current_team_strength_map(season, loader)
    if strength:
        result["team_strength"] = result["team"].map(strength)
    return result


def build_preseason_prediction_rows(
    season: str,
    start_gameweek: int,
    base_features: list[str] | None = None,
    use_observed_gw1_context: bool = False,
) -> dict[str, Any]:
    """Build previous-season state plus the selected current schedule."""
    features = list(base_features or PRESEASON_BASE_FEATURES)
    loader = FPLDataLoader()
    state = build_player_state(
        season,
        gameweek=1,
        base_features=features,
        loader=loader,
        seed_previous_season=True,
    )
    raw = None
    if use_observed_gw1_context:
        raw = load_raw_season(season, loader=loader)
        state = _apply_observed_gw1_context(state, raw, season, loader)
    fixture_rows, gameweeks = build_future_fixture_rows(
        state,
        season=season,
        start_gameweek=start_gameweek,
        horizon=5,
    )
    five_rows = build_five_gw_prediction_rows(
        state,
        fixture_rows,
        start_gameweek=start_gameweek,
        gameweeks=gameweeks,
    )
    return {
        "state": state,
        "fixture_rows": fixture_rows,
        "five_rows": five_rows,
        "gameweeks": gameweeks,
        "raw": raw,
    }


def _adjusted_targets(
    raw: pd.DataFrame,
    season: str,
    dc_estimator: dict[str, Any],
) -> pd.DataFrame:
    """Apply current defensive-contribution scoring before target aggregation."""
    adjusted = add_defensive_contribution_target(
        raw,
        season=season,
        estimator_bundle=dc_estimator,
    )
    adjusted["target_points"] = adjusted["adjusted_total_points"]
    return adjusted


def build_preseason_transition_rows(
    season: str,
    dc_estimator: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create GW1 fixture rows and direct GW1-5 targets for one season."""
    prepared = build_preseason_prediction_rows(
        season,
        start_gameweek=1,
        use_observed_gw1_context=True,
    )
    raw = prepared["raw"]
    adjusted = _adjusted_targets(raw, season, dc_estimator)
    gw1_targets = adjusted[
        pd.to_numeric(adjusted["round"], errors="coerce").eq(1)
    ][["element", "fixture", "target_points"]].copy()
    single = prepared["fixture_rows"]
    single = single[
        single["gw"].eq(1) & single["has_fixture"].astype(bool)
    ].merge(gw1_targets, on=["element", "fixture"], how="inner")
    single[PRESEASON_SINGLE_TARGET] = single["target_points"]
    single["season"] = season

    first_five = adjusted[
        pd.to_numeric(adjusted["round"], errors="coerce").between(1, 5)
    ]
    per_gameweek = (
        first_five.groupby(["element", "round"])["target_points"]
        .sum()
        .unstack("round")
        .reindex(columns=range(1, 6), fill_value=0.0)
        .fillna(0.0)
    )
    five_target = per_gameweek.mean(axis=1).rename(FIVE_GW_TARGET)
    five = prepared["five_rows"].merge(
        five_target,
        left_on="element",
        right_index=True,
        how="inner",
    )
    five["season"] = season
    return single, five


def build_preseason_training_frames(
    seasons: list[str],
    scoring_season: str = "2025-26",
    dc_estimator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build all available cross-season transitions for preseason models."""
    dc_estimator = dc_estimator or train_dc_estimator_from_season(scoring_season)
    single_frames = []
    five_frames = []
    for season in seasons[1:]:
        print(f"Building preseason transition into {season}...")
        single, five = build_preseason_transition_rows(season, dc_estimator)
        single_frames.append(single)
        five_frames.append(five)
    return {
        "single": pd.concat(single_frames, ignore_index=True, sort=False),
        "five": pd.concat(five_frames, ignore_index=True, sort=False),
        "base_features": list(PRESEASON_BASE_FEATURES),
        "dc_estimator": dc_estimator,
    }


def train_preseason_models(
    training_data: dict[str, Any],
    model_directory: Path,
) -> dict[str, Any]:
    """Validate on the newest transition, refit all rows, and save bundles."""
    model_directory.mkdir(parents=True, exist_ok=True)
    single = training_data["single"]
    five = training_data["five"]
    base_features = list(training_data["base_features"])
    validation_season = max(single["season"].dropna().unique())
    single_bundle = fit_production_bundle(
        single,
        features=base_features,
        target=PRESEASON_SINGLE_TARGET,
        horizon_name="preseason_one_fixture",
        validation_season=validation_season,
    )
    five_features = base_features + FIVE_GW_SCHEDULE_FEATURES
    five_bundle = fit_production_bundle(
        five,
        features=five_features,
        target=FIVE_GW_TARGET,
        horizon_name="preseason_direct_average_next_5_gameweeks",
        validation_season=validation_season,
    )
    joblib.dump(
        single_bundle,
        model_directory / "preseason_one_fixture_model.joblib",
    )
    joblib.dump(
        five_bundle,
        model_directory / "preseason_five_gw_average_model.joblib",
    )
    result = {
        "single": single_bundle,
        "five_gw": five_bundle,
        "validation_season": validation_season,
    }
    manifest = {
        "single": {
            key: value
            for key, value in single_bundle.items()
            if key not in {"model", "imputer"}
        },
        "five_gw": {
            key: value
            for key, value in five_bundle.items()
            if key not in {"model", "imputer"}
        },
        "validation_season": validation_season,
    }
    joblib.dump(manifest, model_directory / "preseason_manifest.joblib")
    return result
