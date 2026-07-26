"""Tests that newer individual gameweeks override a stale merged file."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.preprocessing.data_loader import FPLDataLoader


class DataLoaderFreshnessTests(unittest.TestCase):
    """Protect production training from silently missing recent gameweeks."""

    def test_newer_individual_gameweek_is_loaded(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            gameweeks = root / "2099-00" / "gws"
            gameweeks.mkdir(parents=True)
            base = {
                "element": [1],
                "position": ["MID"],
                "team": [1],
                "total_points": [2],
                "minutes": [90],
            }
            pd.DataFrame({**base, "round": [1]}).to_csv(
                gameweeks / "merged_gw.csv", index=False
            )
            pd.DataFrame({**base, "round": [1]}).to_csv(
                gameweeks / "gw1.csv", index=False
            )
            pd.DataFrame({**base, "round": [2]}).to_csv(
                gameweeks / "gw2.csv", index=False
            )

            result = FPLDataLoader(data_dir=root).load_gameweeks("2099-00")

            self.assertEqual(sorted(result["round"].tolist()), [1, 2])


if __name__ == "__main__":
    unittest.main()
