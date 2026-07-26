"""Training and persistence for the production dual-horizon models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.fpl_assistant.targets import (
    FIVE_GW_SCHEDULE_FEATURES,
    FIVE_GW_TARGET,
)


SINGLE_TARGET = "total_points"


def _new_regressor(random_state: int = 42) -> XGBRegressor:
    """Return the production XGBoost configuration."""
    return XGBRegressor(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.2,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float | int]:
    """Calculate regression metrics for a temporal validation season."""
    actual_series = pd.Series(actual).reset_index(drop=True)
    predicted_series = pd.Series(predicted).reset_index(drop=True)
    return {
        "n": int(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "spearman": float(actual_series.corr(predicted_series, method="spearman")),
    }


def fit_production_bundle(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    horizon_name: str,
    validation_season: str | None = None,
) -> dict[str, Any]:
    """Validate chronologically, then refit a model on every available row."""
    usable = frame[frame[target].notna()].copy()
    for feature in features:
        if feature not in usable:
            usable[feature] = np.nan
    validation_metrics: dict[str, Any] | None = None

    if (
        validation_season is not None
        and "season" in usable
        and usable["season"].nunique() > 1
    ):
        train_mask = usable["season"].ne(validation_season)
        validation_mask = usable["season"].eq(validation_season)
        if train_mask.any() and validation_mask.any():
            validation_imputer = SimpleImputer(strategy="median")
            train_x = validation_imputer.fit_transform(
                usable.loc[train_mask, features]
            )
            validation_x = validation_imputer.transform(
                usable.loc[validation_mask, features]
            )
            validation_model = _new_regressor()
            validation_model.fit(train_x, usable.loc[train_mask, target])
            validation_prediction = np.clip(
                validation_model.predict(validation_x),
                0.0,
                None,
            )
            validation_metrics = {
                "season": validation_season,
                **_metrics(
                    usable.loc[validation_mask, target],
                    validation_prediction,
                ),
            }

    imputer = SimpleImputer(strategy="median")
    full_x = imputer.fit_transform(usable[features])
    model = _new_regressor()
    model.fit(full_x, usable[target])
    return {
        "model": model,
        "imputer": imputer,
        "features": features,
        "target": target,
        "horizon": horizon_name,
        "training_rows": int(len(usable)),
        "training_seasons": sorted(usable["season"].dropna().unique().tolist()),
        "validation": validation_metrics,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }


def train_dual_production_models(
    training_data: dict[str, Any],
    model_directory: Path,
) -> dict[str, Any]:
    """Train and save independent one-fixture and direct-five-GW models."""
    model_directory.mkdir(parents=True, exist_ok=True)
    base_features = list(training_data["base_features"])
    single_frame = training_data["single"]
    five_frame = training_data["five"]
    validation_season = max(single_frame["season"].dropna().unique())

    single_bundle = fit_production_bundle(
        single_frame,
        features=base_features,
        target=SINGLE_TARGET,
        horizon_name="one_fixture",
        validation_season=validation_season,
    )
    five_features = base_features + FIVE_GW_SCHEDULE_FEATURES
    five_bundle = fit_production_bundle(
        five_frame,
        features=five_features,
        target=FIVE_GW_TARGET,
        horizon_name="direct_average_next_5_gameweeks",
        validation_season=validation_season,
    )
    dc_bundle = training_data["dc_estimator"]

    joblib.dump(single_bundle, model_directory / "one_fixture_model.joblib")
    joblib.dump(five_bundle, model_directory / "five_gw_average_model.joblib")
    joblib.dump(
        dc_bundle,
        model_directory / "defensive_contribution_estimator.joblib",
    )
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
        "dc_estimator_metrics": dc_bundle["metrics"],
        "dc_adjustments": training_data["adjustment_summary"],
        "important_definition": (
            "The five-GW forecast is a separately trained direct average "
            "target. It is not the sum or mean of five one-fixture forecasts."
        ),
    }
    joblib.dump(manifest, model_directory / "training_manifest.joblib")
    return {
        "single": single_bundle,
        "five_gw": five_bundle,
        "dc_estimator": dc_bundle,
        "manifest": manifest,
    }


def load_dual_models(model_directory: Path) -> dict[str, Any]:
    """Load both production prediction bundles."""
    bundles = {
        "single": joblib.load(model_directory / "one_fixture_model.joblib"),
        "five_gw": joblib.load(
            model_directory / "five_gw_average_model.joblib"
        ),
    }
    preseason_single = model_directory / "preseason_one_fixture_model.joblib"
    preseason_five = model_directory / "preseason_five_gw_average_model.joblib"
    if preseason_single.exists() and preseason_five.exists():
        bundles["preseason_single"] = joblib.load(preseason_single)
        bundles["preseason_five_gw"] = joblib.load(preseason_five)
    return bundles


def predict_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    """Predict with a saved bundle while preserving its feature order."""
    matrix = frame.copy()
    for feature in bundle["features"]:
        if feature not in matrix:
            matrix[feature] = np.nan
    transformed = bundle["imputer"].transform(matrix[bundle["features"]])
    return np.clip(bundle["model"].predict(transformed), 0.0, None)
