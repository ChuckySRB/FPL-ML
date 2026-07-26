"""Train the production one-fixture and direct-five-GW FPL models."""

from __future__ import annotations

import argparse
import joblib
import json
import os
import sys
from pathlib import Path


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import MODELS_DIR
from src.fpl_assistant.data import (
    available_seasons,
    build_dual_training_frames,
)
from src.fpl_assistant.models import train_dual_production_models
from src.fpl_assistant.preseason import (
    build_preseason_training_frames,
    train_preseason_models,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Train both production horizons on all local seasons."
    )
    parser.add_argument("--first-season", default="2020-21")
    parser.add_argument("--last-season", default="2025-26")
    parser.add_argument(
        "--scoring-season",
        default="2025-26",
        help="Season containing exact defensive-contribution components.",
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=MODELS_DIR / "fpl_assistant",
    )
    return parser.parse_args()


def main() -> None:
    """Build adjusted targets, train both horizons, and persist artifacts."""
    args = parse_args()
    seasons = available_seasons(
        first_season=args.first_season,
        last_season=args.last_season,
    )
    if args.scoring_season not in seasons:
        raise RuntimeError(
            f"Exact scoring season {args.scoring_season} is not available"
        )
    print("Training seasons:", ", ".join(seasons))
    training_data = build_dual_training_frames(
        seasons,
        scoring_season=args.scoring_season,
    )
    result = train_dual_production_models(
        training_data,
        model_directory=args.model_directory,
    )
    preseason_data = build_preseason_training_frames(
        seasons,
        scoring_season=args.scoring_season,
        dc_estimator=training_data["dc_estimator"],
    )
    preseason_result = train_preseason_models(
        preseason_data,
        model_directory=args.model_directory,
    )
    result["manifest"]["preseason"] = {
        "single": {
            key: value
            for key, value in preseason_result["single"].items()
            if key not in {"model", "imputer"}
        },
        "five_gw": {
            key: value
            for key, value in preseason_result["five_gw"].items()
            if key not in {"model", "imputer"}
        },
    }
    joblib.dump(
        result["manifest"],
        args.model_directory / "training_manifest.joblib",
    )
    manifest_path = args.model_directory / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "one_fixture_rows": result["single"]["training_rows"],
                "five_gw_rows": result["five_gw"]["training_rows"],
                "one_fixture_validation": result["single"]["validation"],
                "five_gw_validation": result["five_gw"]["validation"],
                "dc_estimator": result["dc_estimator"]["metrics"],
                "preseason_one_fixture_rows": preseason_result["single"][
                    "training_rows"
                ],
                "preseason_five_gw_rows": preseason_result["five_gw"][
                    "training_rows"
                ],
                "preseason_one_fixture_validation": preseason_result[
                    "single"
                ]["validation"],
                "preseason_five_gw_validation": preseason_result["five_gw"][
                    "validation"
                ],
                "output": "models/fpl_assistant",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
