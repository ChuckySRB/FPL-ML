"""Production helpers for the weekly Fantasy Premier League assistant."""

from .defensive_points import (
    DC_PROXY_FEATURES,
    add_defensive_contribution_target,
    calculate_defensive_contributions,
    train_defensive_contribution_estimator,
)
from .targets import (
    FIVE_GW_SCHEDULE_FEATURES,
    add_direct_five_gw_target,
    build_five_gw_training_rows,
)

__all__ = [
    "DC_PROXY_FEATURES",
    "FIVE_GW_SCHEDULE_FEATURES",
    "add_defensive_contribution_target",
    "add_direct_five_gw_target",
    "build_five_gw_training_rows",
    "calculate_defensive_contributions",
    "train_defensive_contribution_estimator",
]
