"""Regression tests for the local FPL squad optimizer."""

import unittest

import pandas as pd

from src.optimization import (
    POSITION_LIMITS,
    aggregate_player_projections,
    optimize_fpl_squad,
)


class OptimizationTests(unittest.TestCase):
    """Check aggregation and official squad constraints."""

    def test_fixture_predictions_are_summed_per_player(self):
        predictions = pd.DataFrame(
            {
                "element": [1, 1, 2],
                "name": ["Player A", "Player A", "Player B"],
                "position_label": ["MID", "MID", "FWD"],
                "team_name": ["Club A", "Club A", "Club B"],
                "price": [7.0, 7.1, 8.0],
                "round": [32, 33, 32],
                "pred_xgboost": [3.0, 4.0, 5.0],
            }
        )

        result = aggregate_player_projections(predictions)
        player = result[result["element"] == 1].iloc[0]

        self.assertEqual(player["fixtures"], 2)
        self.assertAlmostEqual(player["projected_points"], 7.0)
        self.assertAlmostEqual(player["price"], 7.1)

    def test_optimizer_returns_legal_fifteen_player_squad(self):
        rows = []
        element = 1
        price_by_position = {"GK": 4.5, "DEF": 5.0, "MID": 6.0, "FWD": 6.5}
        for team_index in range(8):
            for position_index, position in enumerate(POSITION_LIMITS):
                rows.append(
                    {
                        "element": element,
                        "name": f"Player {element}",
                        "position_label": position,
                        "team_name": f"Club {team_index}",
                        "price": price_by_position[position],
                        "projected_points": 3 + team_index + position_index / 10,
                        "fixtures": 1,
                    }
                )
                element += 1
        candidates = pd.DataFrame(rows)

        squad = optimize_fpl_squad(candidates, budget=100.0)

        self.assertEqual(len(squad), 15)
        self.assertLessEqual(squad["price"].sum(), 100.0)
        self.assertTrue((squad.groupby("team_name").size() <= 3).all())
        self.assertEqual(
            squad["position_label"].value_counts().to_dict(),
            POSITION_LIMITS,
        )


if __name__ == "__main__":
    unittest.main()
