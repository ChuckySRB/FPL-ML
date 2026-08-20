"""Tests for the weekly assistant's dashboard data helpers."""

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import pandas as pd

from src.fpl_assistant.dashboard import (
    availability_mask,
    available_gameweeks,
    build_ai_workbook,
    build_strategy_prompt,
    discover_seasons,
    infer_default_gameweek,
    load_weekly_package,
    package_paths,
    rank_players,
    load_user_profile,
    save_user_profile,
)


class AssistantDashboardTests(unittest.TestCase):
    """Keep UI-facing filtering and package discovery deterministic."""

    def test_season_gameweek_discovery_and_official_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_root = Path(temporary_directory)
            season = raw_root / "2026-27"
            season.mkdir()
            pd.DataFrame({"id": [1]}).to_csv(
                season / "players_raw.csv",
                index=False,
            )
            pd.DataFrame({"id": [1]}).to_csv(
                season / "teams.csv",
                index=False,
            )
            pd.DataFrame({"event": [1, 2, 2, 3]}).to_csv(
                season / "fixtures.csv",
                index=False,
            )
            pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "is_next": [False, True, False],
                    "is_current": [False, False, False],
                    "finished": [False, False, False],
                }
            ).to_csv(season / "events.csv", index=False)

            self.assertEqual(discover_seasons(raw_root), ["2026-27"])
            gameweeks = available_gameweeks(raw_root, "2026-27")
            self.assertEqual(gameweeks, [1, 2, 3])
            self.assertEqual(
                infer_default_gameweek(raw_root, "2026-27", gameweeks),
                2,
            )

    def test_rankings_apply_availability_price_and_position(self):
        players = pd.DataFrame(
            {
                "element": [1, 2, 3],
                "name": ["Fit Mid", "Injured Mid", "Fit Def"],
                "position_label": ["MID", "MID", "DEF"],
                "team_name": ["A", "A", "B"],
                "current_price": [8.0, 7.0, 5.0],
                "status": ["a", "i", "a"],
                "chance_of_playing_next_round": [None, 0, None],
                "predicted_points_current_gw": [6.0, 9.0, 5.0],
            }
        )

        mask = availability_mask(players)
        self.assertEqual(mask.tolist(), [True, False, True])
        ranked = rank_players(
            players,
            "predicted_points_current_gw",
            positions=["MID"],
            maximum_price=8.0,
        )

        self.assertEqual(ranked["name"].tolist(), ["Fit Mid"])
        self.assertAlmostEqual(ranked.loc[0, "value_score"], 0.75)

    def test_complete_weekly_package_and_strategy_prompt(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            paths = package_paths(output_root, "2026-27", 1)
            paths["directory"].mkdir(parents=True)
            players = pd.DataFrame(
                {
                    "element": [1],
                    "name": ["Example"],
                    "position_label": ["MID"],
                    "team_name": ["Example FC"],
                }
            )
            players.to_csv(paths["players"], index=False)
            pd.DataFrame({"element": [1]}).to_csv(
                paths["fixtures"],
                index=False,
            )
            paths["report"].write_text("# Report", encoding="utf-8")
            paths["prompt"].write_text("# Prompt", encoding="utf-8")
            paths["json"].write_text(
                json.dumps({"season": "2026-27"}),
                encoding="utf-8",
            )

            package = load_weekly_package(output_root, "2026-27", 1)
            prompt = build_strategy_prompt(
                package["prompt"],
                players,
                watchlist=players,
                bank=1.5,
                free_transfers=2,
                chips="Wildcard",
                risk_profile="Умерен",
                external_notes="Проверити конференцију.",
                attachment_filename="assistant.xlsx",
                data_timestamp="2026-08-20T19:00",
            )

        self.assertEqual(package["structured"]["season"], "2026-27")
        self.assertIn("MID: Example", prompt)
        self.assertIn("Watchlist", prompt)
        self.assertIn("assistant.xlsx", prompt)
        self.assertIn("2026-08-20T19:00", prompt)
        self.assertIn("£1.5m", prompt)
        self.assertIn("Проверити конференцију.", prompt)

    def test_workbook_and_profile_include_draft_and_watchlist(self):
        players = pd.DataFrame(
            {
                "element": [1, 2],
                "name": ["Draft Pick", "Watch Pick"],
                "position_label": ["MID", "FWD"],
                "team_name": ["A", "B"],
                "current_price": [8.0, 7.0],
                "status": ["a", "a"],
                "predicted_points_current_gw": [6.0, 5.0],
                "predicted_average_next_5_gws": [4.0, 4.5],
                "current_gw_fixtures": [1, 1],
            }
        )
        fixtures = pd.DataFrame(
            {"element": [1, 2], "gw": [1, 1], "has_fixture": [True, True]}
        )
        workbook = build_ai_workbook(
            "2026-27",
            1,
            players,
            fixtures,
            {"excluded_unavailable": [], "large_model_ep_next_disagreements": []},
            players.iloc[[0]],
            players.iloc[[1]],
            {"bank": 1.0, "free_transfers": 0},
        )
        sheets = pd.ExcelFile(BytesIO(workbook)).sheet_names
        self.assertIn("DRAFT_TIM", sheets)
        self.assertIn("WATCHLIST", sheets)
        self.assertIn("TOP50_GW", sheets)

        with tempfile.TemporaryDirectory() as temporary_directory:
            profile_path = Path(temporary_directory) / "profile.json"
            save_user_profile(
                profile_path,
                {"squad_elements": [1], "watchlist_elements": [2]},
            )
            profile = load_user_profile(profile_path)
        self.assertEqual(profile["squad_elements"], [1])
        self.assertEqual(profile["watchlist_elements"], [2])


if __name__ == "__main__":
    unittest.main()
