"""Tests for retrospective defensive-contribution point adjustment."""

import unittest

import pandas as pd

from src.fpl_assistant.defensive_points import (
    add_defensive_contribution_target,
    calculate_defensive_contributions,
)


class DefensiveContributionTests(unittest.TestCase):
    """Check official thresholds, position formulas, and double-count safety."""

    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "position_label": ["DEF", "MID", "FWD", "GK"],
                "clearances_blocks_interceptions": [8, 5, 2, 20],
                "tackles": [2, 2, 1, 5],
                "recoveries": [50, 5, 9, 10],
                "total_points": [4, 3, 2, 6],
            }
        )

    def test_position_specific_contribution_formula(self):
        result = calculate_defensive_contributions(self.frame)
        self.assertEqual(result.tolist(), [10, 12, 12, 0])

    def test_historical_points_receive_exact_capped_bonus(self):
        result = add_defensive_contribution_target(
            self.frame,
            season="2016-17",
        )
        self.assertEqual(
            result["dc_bonus_points_under_current_rules"].tolist(),
            [2, 2, 2, 0],
        )
        self.assertEqual(result["adjusted_total_points"].tolist(), [6, 5, 4, 6])

    def test_current_rules_are_not_added_twice(self):
        result = add_defensive_contribution_target(
            self.frame,
            season="2025-26",
        )
        self.assertEqual(result["dc_target_adjustment"].sum(), 0)
        self.assertEqual(result["adjusted_total_points"].tolist(), [4, 3, 2, 6])


if __name__ == "__main__":
    unittest.main()
