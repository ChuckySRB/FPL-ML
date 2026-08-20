"""Shared orchestration for CLI and Streamlit weekly prediction runs."""

from __future__ import annotations

from pathlib import Path

from configs.config import MODELS_DIR, OUTPUTS_DIR, RAW_DATA_DIR
from src.data_collection.current_season_collector import CurrentSeasonCollector
from src.fpl_assistant.prediction import generate_dual_predictions
from src.fpl_assistant.reporting import create_weekly_report


def ensure_current_prediction_inputs(
    season: str,
    offline: bool = False,
) -> None:
    """Refresh lightweight official inputs, or verify an offline cache."""
    season_directory = RAW_DATA_DIR / season
    required = [
        season_directory / "players_raw.csv",
        season_directory / "teams.csv",
        season_directory / "fixtures.csv",
    ]
    missing_before = [path for path in required if not path.exists()]
    if offline:
        if missing_before:
            missing_names = ", ".join(path.name for path in missing_before)
            raise FileNotFoundError(
                f"Offline mode is missing {missing_names} for {season}. "
                "Run with an official-data refresh first."
            )
        return

    collector = CurrentSeasonCollector(season=season)
    bootstrap = collector.collect_bootstrap_data()
    fixtures = collector.collect_fixtures() if bootstrap is not None else None
    missing_after = [path for path in required if not path.exists()]
    if bootstrap is None or fixtures is None:
        if missing_after:
            missing_names = ", ".join(path.name for path in missing_after)
            raise RuntimeError(
                f"Could not download {missing_names} for {season}. "
                "The official FPL API may not have opened that season yet."
            )
        raise RuntimeError(
            "Official FPL refresh failed; the existing cache was preserved "
            "but was not treated as fresh. Retry the refresh, or explicitly "
            "use offline mode if the cached snapshot is acceptable."
        )


def generate_weekly_package(
    season: str,
    gameweek: int,
    refresh_official_data: bool = True,
    model_directory: Path | None = None,
    output_directory: Path | None = None,
    prompt_template: Path | None = None,
) -> dict[str, Path]:
    """Refresh/verify inputs, predict both horizons, and save all artifacts."""
    ensure_current_prediction_inputs(
        season,
        offline=not refresh_official_data,
    )
    prediction_result = generate_dual_predictions(
        season=season,
        gameweek=int(gameweek),
        model_directory=model_directory or MODELS_DIR / "fpl_assistant",
    )
    return create_weekly_report(
        prediction_result,
        output_root=output_directory or OUTPUTS_DIR / "assistant",
        prompt_template=prompt_template,
    )
