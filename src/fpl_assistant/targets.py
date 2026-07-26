"""Direct multi-horizon targets for the production FPL assistant."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


FIVE_GW_TARGET = "target_average_points_next_5_gws"
FIVE_GW_SCHEDULE_FEATURES = [
    "fixtures_next_5_gws",
    "blank_gws_next_5",
    "double_gws_next_5",
    "fdr_mean_next_5",
    "fdr_min_next_5",
    "fdr_max_next_5",
    "home_fixture_rate_next_5",
    "opponent_strength_mean_next_5",
]


def _forward_rolling(
    values: pd.Series,
    window: int,
    operation: str,
    min_periods: int = 1,
) -> pd.Series:
    """Apply a rolling operation to the current and following rows."""
    reversed_values = values.iloc[::-1]
    rolling = reversed_values.rolling(window, min_periods=min_periods)
    result = getattr(rolling, operation)()
    return result.iloc[::-1]


def add_direct_five_gw_target(
    frame: pd.DataFrame,
    points_column: str = "adjusted_total_points",
    horizon: int = 5,
    target_column: str = FIVE_GW_TARGET,
) -> pd.DataFrame:
    """Add the directly observed average FPL return over the next five GWs.

    Returns within a double gameweek are summed before the five-GW average is
    calculated. Blank gameweeks contribute zero. A target is emitted only when
    the complete future horizon lies inside that season.
    """
    required = {"season", "element", "round", points_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing horizon-target columns: {sorted(missing)}")
    if horizon < 1:
        raise ValueError("horizon must be positive")

    result = frame.copy()
    result[target_column] = np.nan
    season_last_gw = (
        result.groupby("season")["round"].max().dropna().astype(int).to_dict()
    )
    aggregated = (
        result.groupby(["season", "element", "round"], as_index=False)[
            points_column
        ]
        .sum()
        .sort_values(["season", "element", "round"])
    )
    target_lookup: dict[tuple[str, int, int], float] = {}
    for (season, element), player in aggregated.groupby(
        ["season", "element"], sort=False
    ):
        first_gw = int(player["round"].min())
        last_gw = season_last_gw[season]
        gameweeks = pd.RangeIndex(first_gw, last_gw + 1)
        points = (
            player.set_index("round")[points_column]
            .reindex(gameweeks, fill_value=0.0)
            .astype(float)
        )
        average = _forward_rolling(
            points,
            window=horizon,
            operation="mean",
            min_periods=horizon,
        )
        target_lookup.update(
            {
                (str(season), int(element), int(gameweek)): float(value)
                for gameweek, value in average.dropna().items()
            }
        )

    keys = zip(
        result["season"].astype(str),
        result["element"].astype(int),
        result["round"].astype(int),
    )
    result[target_column] = [target_lookup.get(key, np.nan) for key in keys]
    return result


def add_future_schedule_features(
    frame: pd.DataFrame,
    horizon: int = 5,
) -> pd.DataFrame:
    """Add known fixture-run descriptors for a direct multi-GW model."""
    required = {"season", "element", "round"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing schedule columns: {sorted(missing)}")
    result = frame.copy()
    for column in FIVE_GW_SCHEDULE_FEATURES:
        result[column] = np.nan

    working = result.copy()
    for column in ["opponent_difficulty", "was_home", "opponent_strength"]:
        if column not in working:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce")
    season_last_gw = (
        working.groupby("season")["round"].max().dropna().astype(int).to_dict()
    )
    lookup: dict[tuple[str, int, int], dict[str, float]] = {}

    for (season, element), player in working.groupby(
        ["season", "element"], sort=False
    ):
        first_gw = int(player["round"].min())
        last_gw = season_last_gw[season]
        gameweeks = pd.RangeIndex(first_gw, last_gw + 1)
        per_gw = player.groupby("round").agg(
            fixture_count=("element", "size"),
            fdr_sum=("opponent_difficulty", "sum"),
            fdr_count=("opponent_difficulty", "count"),
            fdr_min=("opponent_difficulty", "min"),
            fdr_max=("opponent_difficulty", "max"),
            home_sum=("was_home", "sum"),
            home_count=("was_home", "count"),
            strength_sum=("opponent_strength", "sum"),
            strength_count=("opponent_strength", "count"),
        )
        per_gw = per_gw.reindex(gameweeks)
        per_gw["fixture_count"] = per_gw["fixture_count"].fillna(0.0)
        for column in [
            "fdr_sum",
            "fdr_count",
            "home_sum",
            "home_count",
            "strength_sum",
            "strength_count",
        ]:
            per_gw[column] = per_gw[column].fillna(0.0)

        fixture_count = _forward_rolling(
            per_gw["fixture_count"], horizon, "sum"
        )
        blank_count = _forward_rolling(
            per_gw["fixture_count"].eq(0).astype(float), horizon, "sum"
        )
        double_count = _forward_rolling(
            per_gw["fixture_count"].gt(1).astype(float), horizon, "sum"
        )
        fdr_sum = _forward_rolling(per_gw["fdr_sum"], horizon, "sum")
        fdr_count = _forward_rolling(per_gw["fdr_count"], horizon, "sum")
        home_sum = _forward_rolling(per_gw["home_sum"], horizon, "sum")
        home_count = _forward_rolling(per_gw["home_count"], horizon, "sum")
        strength_sum = _forward_rolling(
            per_gw["strength_sum"], horizon, "sum"
        )
        strength_count = _forward_rolling(
            per_gw["strength_count"], horizon, "sum"
        )
        fdr_min = _forward_rolling(per_gw["fdr_min"], horizon, "min")
        fdr_max = _forward_rolling(per_gw["fdr_max"], horizon, "max")

        for gameweek in gameweeks:
            lookup[(str(season), int(element), int(gameweek))] = {
                "fixtures_next_5_gws": float(fixture_count.loc[gameweek]),
                "blank_gws_next_5": float(blank_count.loc[gameweek]),
                "double_gws_next_5": float(double_count.loc[gameweek]),
                "fdr_mean_next_5": float(
                    fdr_sum.loc[gameweek] / fdr_count.loc[gameweek]
                )
                if fdr_count.loc[gameweek] > 0
                else np.nan,
                "fdr_min_next_5": float(fdr_min.loc[gameweek])
                if pd.notna(fdr_min.loc[gameweek])
                else np.nan,
                "fdr_max_next_5": float(fdr_max.loc[gameweek])
                if pd.notna(fdr_max.loc[gameweek])
                else np.nan,
                "home_fixture_rate_next_5": float(
                    home_sum.loc[gameweek] / home_count.loc[gameweek]
                )
                if home_count.loc[gameweek] > 0
                else np.nan,
                "opponent_strength_mean_next_5": float(
                    strength_sum.loc[gameweek] / strength_count.loc[gameweek]
                )
                if strength_count.loc[gameweek] > 0
                else np.nan,
            }

    keys = list(
        zip(
            result["season"].astype(str),
            result["element"].astype(int),
            result["round"].astype(int),
        )
    )
    for column in FIVE_GW_SCHEDULE_FEATURES:
        result[column] = [
            lookup.get(key, {}).get(column, np.nan) for key in keys
        ]
    return result


def build_five_gw_training_rows(
    frame: pd.DataFrame,
    base_features: Iterable[str],
    points_column: str = "adjusted_total_points",
    horizon: int = 5,
    min_gameweek: int = 6,
) -> pd.DataFrame:
    """Create one direct-average training row per player and starting GW."""
    enriched = add_direct_five_gw_target(
        frame,
        points_column=points_column,
        horizon=horizon,
    )
    enriched = add_future_schedule_features(enriched, horizon=horizon)
    base_features = list(base_features)
    metadata = [
        column
        for column in [
            "name",
            "position_label",
            "team",
            "team_name",
        ]
        if column in enriched
    ]
    context_averages = [
        column
        for column in [
            "was_home",
            "opponent_difficulty",
            "opponent_strength",
        ]
        if column in enriched
    ]
    invariant_features = [
        column
        for column in base_features
        if column not in context_averages and column in enriched
    ]
    aggregations: dict[str, str] = {
        column: "first" for column in metadata + invariant_features
    }
    aggregations.update({column: "mean" for column in context_averages})
    aggregations.update(
        {column: "first" for column in FIVE_GW_SCHEDULE_FEATURES}
    )
    aggregations[FIVE_GW_TARGET] = "first"

    rows = (
        enriched.sort_values(["season", "element", "round"])
        .groupby(["season", "element", "round"], as_index=False)
        .agg(aggregations)
    )
    rows = rows[
        rows["round"].ge(min_gameweek) & rows[FIVE_GW_TARGET].notna()
    ].copy()
    return rows.reset_index(drop=True)
