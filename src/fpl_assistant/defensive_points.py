"""Apply the 2025/26 defensive-contribution scoring rules to model targets.

The current FPL season already includes defensive-contribution points in
``total_points``. Earlier public FPL gameweek files usually do not contain
CBI, tackles, and recoveries, so their bonus must be estimated rather than
silently presented as an observed value.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DC_COMPONENT_COLUMNS = (
    "clearances_blocks_interceptions",
    "tackles",
    "recoveries",
)
DC_PROXY_FEATURES = [
    "minutes",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "yellow_cards",
    "position_label",
]
OUTFIELD_POSITIONS = {"DEF", "MID", "FWD", "AM"}


def _position_series(frame: pd.DataFrame) -> pd.Series:
    """Return normalized position labels from either supported representation."""
    if "position_label" in frame:
        return frame["position_label"].replace({"GKP": "GK"}).astype("string")
    if "element_type" in frame:
        return frame["element_type"].map(
            {1: "GK", 2: "DEF", 3: "MID", 4: "FWD", 5: "AM"}
        ).astype("string")
    raise ValueError("A position_label or element_type column is required")


def has_exact_dc_components(frame: pd.DataFrame) -> bool:
    """Return whether every component needed for the official rule is present."""
    return set(DC_COMPONENT_COLUMNS).issubset(frame.columns)


def calculate_defensive_contributions(frame: pd.DataFrame) -> pd.Series:
    """Calculate official per-match CBIT/CBIRT totals.

    Defenders use CBI plus tackles. Midfielders and forwards additionally use
    recoveries. Goalkeepers cannot earn these points.
    """
    if not has_exact_dc_components(frame):
        missing = sorted(set(DC_COMPONENT_COLUMNS).difference(frame.columns))
        raise ValueError(f"Missing defensive-contribution components: {missing}")
    position = _position_series(frame)
    cbi = pd.to_numeric(
        frame["clearances_blocks_interceptions"], errors="coerce"
    ).fillna(0)
    tackles = pd.to_numeric(frame["tackles"], errors="coerce").fillna(0)
    recoveries = pd.to_numeric(frame["recoveries"], errors="coerce").fillna(0)
    total = cbi + tackles
    total = total.where(position.eq("DEF"), total + recoveries)
    return total.where(position.isin(OUTFIELD_POSITIONS), 0.0)


def defensive_bonus_from_total(
    dc_total: pd.Series,
    position: pd.Series,
) -> pd.Series:
    """Convert a defensive-contribution count to the capped FPL bonus."""
    normalized = position.replace({"GKP": "GK"}).astype("string")
    qualifies = (
        (normalized.eq("DEF") & dc_total.ge(10))
        | (normalized.isin(["MID", "FWD", "AM"]) & dc_total.ge(12))
    )
    return qualifies.astype(float) * 2.0


def _ensure_proxy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return estimator inputs with stable columns and normalized positions."""
    proxy = frame.copy()
    proxy["position_label"] = _position_series(proxy)
    for column in DC_PROXY_FEATURES:
        if column not in proxy:
            proxy[column] = np.nan
    return proxy[DC_PROXY_FEATURES]


def train_defensive_contribution_estimator(
    exact_frame: pd.DataFrame,
    validation_gameweeks: int = 8,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a probability model for seasons without defensive-action columns.

    The estimator uses same-match proxy statistics only to reconstruct an
    historical target. These proxy columns must never be used as predictors
    for the match being forecast.
    """
    if not has_exact_dc_components(exact_frame):
        raise ValueError("Exact component data is required to train the estimator")
    data = exact_frame.copy()
    data["position_label"] = _position_series(data)
    data = data[data["position_label"].isin(OUTFIELD_POSITIONS)].copy()
    dc_total = calculate_defensive_contributions(data)
    target = (
        defensive_bonus_from_total(dc_total, data["position_label"]) > 0
    ).astype(int)
    features = _ensure_proxy_columns(data)

    numeric = [column for column in DC_PROXY_FEATURES if column != "position_label"]
    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "position",
                OneHotEncoder(handle_unknown="ignore"),
                ["position_label"],
            ),
        ]
    )

    def make_pipeline() -> Pipeline:
        return Pipeline(
            [
                ("preprocessing", preprocessing),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=220,
                        max_leaf_nodes=15,
                        learning_rate=0.07,
                        l2_regularization=2.0,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    metrics: dict[str, float | int] = {
        "rows": int(len(data)),
        "positive_rows": int(target.sum()),
        "positive_rate": float(target.mean()),
    }
    if "round" in data and data["round"].nunique() > validation_gameweeks:
        last_train_gw = int(data["round"].max()) - validation_gameweeks
        train_mask = data["round"].le(last_train_gw)
        validation_mask = ~train_mask
        validation_model = make_pipeline()
        validation_model.fit(features.loc[train_mask], target.loc[train_mask])
        probability = validation_model.predict_proba(
            features.loc[validation_mask]
        )[:, 1]
        validation_target = target.loc[validation_mask]
        if validation_target.nunique() == 2:
            metrics.update(
                {
                    "validation_start_gw": last_train_gw + 1,
                    "validation_rows": int(validation_mask.sum()),
                    "roc_auc": float(
                        roc_auc_score(validation_target, probability)
                    ),
                    "average_precision": float(
                        average_precision_score(validation_target, probability)
                    ),
                    "brier_score": float(
                        brier_score_loss(validation_target, probability)
                    ),
                }
            )

    final_model = make_pipeline()
    final_model.fit(features, target)
    return {
        "model": final_model,
        "metrics": metrics,
        "features": DC_PROXY_FEATURES.copy(),
        "label": "earned_two_defensive_contribution_points",
    }


def add_defensive_contribution_target(
    frame: pd.DataFrame,
    season: str,
    estimator_bundle: dict[str, Any] | None = None,
    scoring_started_season: str = "2025-26",
) -> pd.DataFrame:
    """Add a comparable new-rules target and provenance columns.

    For 2025/26 and later, ``total_points`` already contains the bonus.
    Earlier seasons receive either an exact +2 adjustment from component
    columns or an expected adjustment ``2 * P(qualifies)`` from the estimator.
    """
    if "total_points" not in frame:
        raise ValueError("total_points is required")
    result = frame.copy()
    result["position_label"] = _position_series(result)
    result["base_total_points"] = pd.to_numeric(
        result["total_points"], errors="coerce"
    )
    result["dc_bonus_probability"] = 0.0
    result["dc_bonus_points_under_current_rules"] = 0.0

    if has_exact_dc_components(result):
        result["defensive_contribution_reconstructed"] = (
            calculate_defensive_contributions(result)
        )
        exact_bonus = defensive_bonus_from_total(
            result["defensive_contribution_reconstructed"],
            result["position_label"],
        )
        result["dc_bonus_probability"] = exact_bonus / 2.0
        result["dc_bonus_points_under_current_rules"] = exact_bonus
    else:
        result["defensive_contribution_reconstructed"] = np.nan

    if season >= scoring_started_season:
        result["dc_target_adjustment"] = 0.0
        result["dc_adjustment_source"] = "already_in_official_total_points"
    elif has_exact_dc_components(result):
        result["dc_target_adjustment"] = (
            result["dc_bonus_points_under_current_rules"]
        )
        result["dc_adjustment_source"] = "exact_from_match_components"
    elif estimator_bundle is not None:
        probability = estimator_bundle["model"].predict_proba(
            _ensure_proxy_columns(result)
        )[:, 1]
        outfield = result["position_label"].isin(OUTFIELD_POSITIONS).to_numpy()
        probability = np.where(outfield, probability, 0.0)
        result["dc_bonus_probability"] = probability
        result["dc_bonus_points_under_current_rules"] = 2.0 * probability
        result["dc_target_adjustment"] = 2.0 * probability
        result["dc_adjustment_source"] = "estimated_probability_from_2025_26"
    else:
        raise ValueError(
            f"{season} has no DC components; provide an estimator bundle"
        )

    result["adjusted_total_points"] = (
        result["base_total_points"] + result["dc_target_adjustment"]
    )
    return result
