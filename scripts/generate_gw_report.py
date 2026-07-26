"""Generate dual-horizon predictions and the weekly AI handoff package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import MODELS_DIR, OUTPUTS_DIR
from src.fpl_assistant.service import generate_weekly_package


def parse_args() -> argparse.Namespace:
    """Parse the selected season and gameweek."""
    parser = argparse.ArgumentParser(
        description="Create one-GW and direct-five-GW FPL predictions."
    )
    parser.add_argument("--season", required=True, help="Example: 2025-26")
    parser.add_argument("--gw", required=True, type=int)
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=MODELS_DIR / "fpl_assistant",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=OUTPUTS_DIR / "assistant",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached season files without calling the official FPL API.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate predictions and save report, CSV, JSON, and prompt files."""
    args = parse_args()
    print(
        "Using cached official inputs..."
        if args.offline
        else "Refreshing official FPL inputs..."
    )
    paths = generate_weekly_package(
        season=args.season,
        gameweek=args.gw,
        model_directory=args.model_directory,
        output_directory=args.output_directory,
        refresh_official_data=not args.offline,
        prompt_template=ROOT / "prompts" / "weekly_analysis_template.md",
    )
    print(f"Created GW{args.gw} assistant package")
    for name, path in paths.items():
        if name == "directory":
            continue
        print(f"  {name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
