"""Tests for the direct five-gameweek average target."""

import unittest

import pandas as pd

from src.fpl_assistant.targets import (
    FIVE_GW_TARGET,
    add_direct_five_gw_target,
    add_future_schedule_features,
)


class AssistantTargetTests(unittest.TestCase):
    """Verify blanks, doubles, and season boundaries."""

    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "season": ["A"] * 4 + ["B"],
                "element": [1, 1, 1, 1, 1],
                "round": [1, 3, 3, 5, 1],
                "adjusted_total_points": [2.0, 3.0, 4.0, 6.0, 20.0],
                "opponent_difficulty": [2, 3, 4, 5, 1],
                "opponent_strength": [2, 3, 4, 5, 1],
                "was_home": [1, 0, 1, 0, 1],
            }
        )

    def test_direct_target_sums_double_and_counts_blanks_as_zero(self):
        result = add_direct_five_gw_target(self.frame)
        first = result[(result["season"] == "A") & (result["round"] == 1)]
        self.assertAlmostEqual(first.iloc[0][FIVE_GW_TARGET], 3.0)
        self.assertTrue(
            result[result["season"] == "B"][FIVE_GW_TARGET].isna().all()
        )

    def test_schedule_features_describe_five_gameweek_run(self):
        result = add_future_schedule_features(self.frame)
        first = result[(result["season"] == "A") & (result["round"] == 1)].iloc[0]
        self.assertEqual(first["fixtures_next_5_gws"], 4)
        self.assertEqual(first["blank_gws_next_5"], 2)
        self.assertEqual(first["double_gws_next_5"], 1)
        self.assertAlmostEqual(first["fdr_mean_next_5"], 3.5)
        self.assertAlmostEqual(first["home_fixture_rate_next_5"], 0.5)


if __name__ == "__main__":
    unittest.main()
