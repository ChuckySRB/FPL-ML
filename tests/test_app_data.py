"""Tests for adapting current-season predictions to the dashboard."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.app_data import load_current_predictions


class AppDataTests(unittest.TestCase):
    """Verify the live-projection CSV adapter."""

    def test_current_prediction_export_is_normalized(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            predictions = root / "predictions.csv"
            raw = root / "2025-26"
            raw.mkdir()
            pd.DataFrame(
                {
                    "element": [10],
                    "name": ["Example"],
                    "position_label": ["MID"],
                    "team": [2],
                    "gw": [35],
                    "predicted_points": [5.25],
                }
            ).to_csv(predictions, index=False)
            pd.DataFrame({"id": [2], "name": ["Example FC"]}).to_csv(
                raw / "teams.csv", index=False
            )
            pd.DataFrame(
                {
                    "id": [10],
                    "now_cost": [75],
                    "status": ["a"],
                    "news": [""],
                }
            ).to_csv(raw / "players_raw.csv", index=False)

            result = load_current_predictions(predictions, raw)

            self.assertEqual(result.loc[0, "round"], 35)
            self.assertEqual(result.loc[0, "team_name"], "Example FC")
            self.assertAlmostEqual(result.loc[0, "price"], 7.5)
            self.assertAlmostEqual(result.loc[0, "pred_xgboost"], 5.25)
            self.assertTrue(result.loc[0, "evaluation_eligible"])


if __name__ == "__main__":
    unittest.main()
