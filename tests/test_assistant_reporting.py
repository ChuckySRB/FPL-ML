"""Tests for the weekly report and prompt handoff package."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.fpl_assistant.reporting import create_weekly_report


class AssistantReportingTests(unittest.TestCase):
    """Ensure both horizons remain distinct in generated artifacts."""

    def test_report_contains_rankings_captains_and_direct_target_definition(self):
        players = pd.DataFrame(
            {
                "element": [1, 2, 3],
                "name": ["Short Pick", "Long Pick", "Injured Outlier"],
                "position_label": ["MID", "DEF", "FWD"],
                "team_name": ["A", "B", "C"],
                "current_price": [8.0, 6.0, 9.0],
                "status": ["a", "a", "i"],
                "news": ["", "", "Injured"],
                "current_gw_fixtures": [1, 1, 1],
                "predicted_points_current_gw": [8.0, 5.0, 12.0],
                "predicted_average_next_5_gws": [4.0, 6.0, 10.0],
                "ep_next": [7.0, 5.0, 0.0],
                "fixtures_next_5_gws": [5.0, 5.0, 5.0],
                "blank_gws_next_5": [0.0, 0.0, 0.0],
                "double_gws_next_5": [0.0, 0.0, 0.0],
                "fdr_mean_next_5": [3.0, 2.0, 3.0],
                "fdr_min_next_5": [2.0, 1.0, 2.0],
                "fdr_max_next_5": [4.0, 3.0, 4.0],
                "home_fixture_rate_next_5": [0.6, 0.6, 0.6],
                "opponent_strength_mean_next_5": [3.0, 2.0, 3.0],
            }
        )
        fixtures = pd.DataFrame(
            {
                "element": [1, 2, 3],
                "gw": [10, 10, 10],
                "fixture": [100, 101, 102],
                "has_fixture": [True, True, True],
                "opponent_name": ["B", "A", "A"],
                "was_home": [1, 0, 1],
                "opponent_difficulty": [3, 2, 3],
            }
        )
        prediction_result = {
            "season": "2099-00",
            "gameweek": 10,
            "gameweeks": [10, 11, 12, 13, 14],
            "player_predictions": players,
            "fixture_predictions": fixtures,
            "model_metadata": {},
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = create_weekly_report(
                prediction_result,
                Path(temporary_directory),
            )
            report = paths["report"].read_text(encoding="utf-8")
            structured = json.loads(paths["json"].read_text(encoding="utf-8"))

        self.assertIn("Short Pick", report)
        self.assertIn("Long Pick", report)
        self.assertIn("није", report)
        self.assertEqual(
            structured["captain_current_gw"][0]["name"],
            "Short Pick",
        )
        self.assertEqual(
            structured["captain_next_5_gws"][0]["name"],
            "Long Pick",
        )
        self.assertNotIn(
            "Injured Outlier",
            [row["name"] for row in structured["top_25_current_gw"]],
        )
        self.assertEqual(
            structured["excluded_unavailable"][0]["name"],
            "Injured Outlier",
        )


if __name__ == "__main__":
    unittest.main()
