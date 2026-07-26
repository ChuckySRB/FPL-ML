"""Fantasy Premier League squad optimization helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


POSITION_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


def aggregate_player_projections(
    predictions: pd.DataFrame,
    prediction_column: str = "pred_xgboost",
) -> pd.DataFrame:
    """Aggregate fixture predictions to one row per player.

    Multiple fixtures in a selected horizon are summed. Price and team are
    taken from the last selected row for the player.
    """
    required = {
        "element",
        "name",
        "position_label",
        "team_name",
        "price",
        prediction_column,
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing projection columns: {sorted(missing)}")

    eligible = predictions[
        predictions["position_label"].isin(POSITION_LIMITS)
    ].copy()
    eligible[prediction_column] = pd.to_numeric(
        eligible[prediction_column], errors="coerce"
    ).fillna(0.0)
    eligible["price"] = pd.to_numeric(eligible["price"], errors="coerce")
    sort_columns = ["element", "round"] if "round" in eligible else ["element"]
    aggregated = (
        eligible.sort_values(sort_columns)
        .groupby(
            ["element", "name", "position_label", "team_name"],
            as_index=False,
            dropna=False,
        )
        .agg(
            price=("price", "last"),
            projected_points=(prediction_column, "sum"),
            fixtures=(prediction_column, "size"),
        )
    )
    return aggregated.dropna(subset=["price"]).reset_index(drop=True)


def optimize_fpl_squad(
    candidates: pd.DataFrame,
    budget: float = 100.0,
    max_per_team: int = 3,
    locked_elements: Iterable[int] = (),
) -> pd.DataFrame:
    """Select a legal 15-player squad with maximum projected points."""
    required = {
        "element",
        "name",
        "position_label",
        "team_name",
        "price",
        "projected_points",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Missing optimizer columns: {sorted(missing)}")
    if budget <= 0:
        raise ValueError("Budget must be positive")

    pool = candidates[
        candidates["position_label"].isin(POSITION_LIMITS)
    ].copy().reset_index(drop=True)
    if pool["element"].duplicated().any():
        raise ValueError("Candidates must contain one row per element")
    if len(pool) < sum(POSITION_LIMITS.values()):
        raise ValueError("Not enough eligible players to create a squad")

    n_players = len(pool)
    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    rows.append(np.ones(n_players))
    lower.append(15)
    upper.append(15)

    for position, count in POSITION_LIMITS.items():
        rows.append((pool["position_label"] == position).to_numpy(dtype=float))
        lower.append(count)
        upper.append(count)

    rows.append(pool["price"].to_numpy(dtype=float))
    lower.append(0)
    upper.append(float(budget))

    for team in sorted(pool["team_name"].dropna().unique()):
        rows.append((pool["team_name"] == team).to_numpy(dtype=float))
        lower.append(0)
        upper.append(max_per_team)

    for element in set(locked_elements):
        mask = (pool["element"] == element).to_numpy(dtype=float)
        if not mask.any():
            raise ValueError(f"Locked element {element} is not in the candidate pool")
        rows.append(mask)
        lower.append(1)
        upper.append(1)

    result = milp(
        c=-pool["projected_points"].to_numpy(dtype=float),
        integrality=np.ones(n_players),
        bounds=Bounds(np.zeros(n_players), np.ones(n_players)),
        constraints=LinearConstraint(
            np.vstack(rows),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        ),
        options={"time_limit": 20},
    )
    if not result.success or result.x is None:
        raise ValueError(
            "No legal squad found. Increase budget or remove locked players."
        )

    selected = pool.loc[result.x > 0.5].copy()
    selected["value_score"] = np.where(
        selected["price"] > 0,
        selected["projected_points"] / selected["price"],
        0,
    )
    position_order = pd.Categorical(
        selected["position_label"],
        categories=list(POSITION_LIMITS),
        ordered=True,
    )
    return (
        selected.assign(_position_order=position_order)
        .sort_values(
            ["_position_order", "projected_points"],
            ascending=[True, False],
        )
        .drop(columns="_position_order")
        .reset_index(drop=True)
    )
