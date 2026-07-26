"""Tests for new-season GW1 player-state construction."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.fpl_assistant.data import (
    _seed_previous_season_form,
    build_player_state,
    current_team_strength_map,
    load_player_metadata,
)
from src.fpl_assistant.prediction import early_season_current_weight
from src.preprocessing.data_loader import FPLDataLoader


class AssistantGameweekOneTests(unittest.TestCase):
    """GW1 must work before any current-season gameweek file exists."""

    def _write_current_metadata(self, root: Path) -> FPLDataLoader:
        season_directory = root / "2026-27"
        season_directory.mkdir(parents=True)
        pd.DataFrame(
            {
                "id": [1, 2],
                "code": [101, 202],
                "web_name": ["Returner", "New signing"],
                "element_type": [3, 2],
                "team": [1, 2],
                "now_cost": [75, 50],
            }
        ).to_csv(season_directory / "players_raw.csv", index=False)
        pd.DataFrame(
            {
                "id": [1, 2],
                "name": ["Alpha", "Beta"],
                "strength": [4, 3],
            }
        ).to_csv(season_directory / "teams.csv", index=False)
        return FPLDataLoader(data_dir=root)

    def test_gw1_uses_metadata_when_gameweek_directory_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            loader = self._write_current_metadata(Path(temporary_directory))
            state = build_player_state(
                "2026-27",
                gameweek=1,
                base_features=[
                    "form_last_5",
                    "price",
                    "pos_DEF",
                    "pos_MID",
                    "team_strength",
                    "cumulative_points_season",
                ],
                loader=loader,
                seed_previous_season=False,
            )

        self.assertEqual(len(state), 2)
        self.assertEqual(state["price"].tolist(), [7.5, 5.0])
        self.assertEqual(state["team_strength"].tolist(), [4, 3])
        self.assertEqual(state["team_name"].tolist(), ["Alpha", "Beta"])
        self.assertEqual(state["cumulative_points_season"].sum(), 0)
        self.assertTrue(state["form_last_5"].isna().all())

    def test_missing_history_after_gw1_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            loader = self._write_current_metadata(Path(temporary_directory))
            with self.assertRaisesRegex(
                FileNotFoundError,
                "Completed gameweek data is required",
            ):
                build_player_state(
                    "2026-27",
                    gameweek=2,
                    base_features=["form_last_5"],
                    loader=loader,
                )

    @patch("src.fpl_assistant.data.build_player_state")
    @patch("src.fpl_assistant.data.load_raw_season")
    @patch("src.fpl_assistant.data._previous_season_with_history")
    def test_returning_players_receive_previous_season_form(
        self,
        previous_season_mock,
        raw_mock,
        previous_state_mock,
    ):
        previous_season_mock.return_value = "2025-26"
        raw_mock.return_value = pd.DataFrame({"round": [38]})
        previous_state_mock.return_value = pd.DataFrame(
            {
                "code": [101],
                "form_last_5": [6.4],
                "minutes_last_5": [88.0],
            }
        )
        current = pd.DataFrame(
            {
                "code": [101, 202],
                "form_last_5": [np.nan, np.nan],
                "minutes_last_5": [np.nan, np.nan],
                "history_source_season": [pd.NA, pd.NA],
            }
        )
        loader = FPLDataLoader(data_dir=Path("unused"))

        seeded = _seed_previous_season_form(
            current,
            season="2026-27",
            base_features=["form_last_5", "minutes_last_5"],
            loader=loader,
        )

        self.assertEqual(seeded.loc[0, "form_last_5"], 6.4)
        self.assertEqual(seeded.loc[0, "history_source_season"], "2025-26")
        self.assertTrue(pd.isna(seeded.loc[1, "form_last_5"]))

    def test_bootstrap_ownership_is_converted_to_training_scale(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            season_directory = root / "2026-27"
            season_directory.mkdir()
            pd.DataFrame(
                {
                    "id": [1],
                    "element_type": [3],
                    "team": [1],
                    "selected_by_percent": [25.0],
                }
            ).to_csv(season_directory / "players_raw.csv", index=False)
            (season_directory / "bootstrap_static.json").write_text(
                json.dumps({"total_players": 1000}),
                encoding="utf-8",
            )
            metadata = load_player_metadata("2026-27", raw_data_dir=root)

        self.assertAlmostEqual(metadata.iloc[0]["selected_pct"], np.log1p(250))

    def test_team_strength_falls_back_to_fixture_difficulty(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            season_directory = root / "2026-27"
            season_directory.mkdir()
            pd.DataFrame(
                {"id": [1, 2], "name": ["A", "B"], "strength": [np.nan, np.nan]}
            ).to_csv(season_directory / "teams.csv", index=False)
            pd.DataFrame(
                {
                    "team_h": [1],
                    "team_a": [2],
                    "team_h_difficulty": [2],
                    "team_a_difficulty": [4],
                }
            ).to_csv(season_directory / "fixtures.csv", index=False)
            strength = current_team_strength_map(
                "2026-27",
                FPLDataLoader(data_dir=root),
            )

        self.assertEqual(strength, {1: 4.0, 2: 2.0})

    def test_early_season_weight_reaches_full_current_model_at_gw6(self):
        self.assertEqual(early_season_current_weight(1), 0.0)
        self.assertEqual(early_season_current_weight(2), 0.2)
        self.assertEqual(early_season_current_weight(5), 0.8)
        self.assertEqual(early_season_current_weight(6), 1.0)


if __name__ == "__main__":
    unittest.main()
