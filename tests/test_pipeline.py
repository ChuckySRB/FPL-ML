'''Regression tests for leakage-safe preprocessing and evaluation.'''
import unittest

import numpy as np
import pandas as pd

from src.evaluation.cross_validation import (
    TeamStratifiedKFold,
    balanced_return_weights,
)
from src.evaluation.tracker import _get_category
from src.preprocessing.feature_engineering import FPLFeatureEngineer
from src.preprocessing.schemas import validate_gameweek_data


class FeatureEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.engineer = FPLFeatureEngineer()

    def test_history_does_not_cross_season_boundary(self):
        frame = pd.DataFrame({
            'season': ['2020-21', '2020-21', '2021-22', '2021-22'],
            'element': [1, 1, 1, 1],
            'round': [1, 2, 1, 2],
            'total_points': [9, 3, 1, 2],
            'minutes': [90, 90, 90, 90],
            'was_home': [1, 0, 1, 0],
        })
        result = self.engineer.create_tier1_features(frame)
        first_rows = result.groupby('season').head(1)
        self.assertTrue(first_rows['form_last_3'].isna().all())
        second_new_season = result.query(
            "season == '2021-22' and round == 2")
        self.assertEqual(second_new_season['form_last_3'].iloc[0], 1)

    def test_double_gameweek_has_no_intra_round_leakage(self):
        frame = pd.DataFrame({
            'season': ['s'] * 4,
            'element': [1] * 4,
            'round': [1, 2, 2, 3],
            'fixture': [10, 20, 21, 30],
            'total_points': [5, 1, 10, 2],
            'minutes': [90, 45, 90, 90],
            'was_home': [1, 1, 0, 1],
        })
        result = self.engineer.create_tier1_features(frame)
        round_two = result.loc[result['round'] == 2, 'form_last_3']
        self.assertEqual(round_two.nunique(), 1)
        self.assertEqual(round_two.iloc[0], 5)
        self.assertEqual(
            result.loc[result['round'] == 3, 'form_last_3'].iloc[0], 8)

    def test_fixture_specific_difficulty_is_preserved(self):
        frame = pd.DataFrame({
            'team': [1, 1],
            'round': [2, 2],
            'fixture': [20, 21],
            'opponent_team': [2, 3],
        })
        teams = pd.DataFrame({
            'id': [1, 2, 3],
            'strength': [100, 90, 110],
        })
        fixtures = pd.DataFrame({
            'id': [20, 21],
            'event': [2, 2],
            'team_h': [1, 3],
            'team_a': [2, 1],
            'team_h_difficulty': [2, 4],
            'team_a_difficulty': [3, 5],
        })
        result = self.engineer.add_opponent_features(
            frame, teams, fixtures)
        self.assertEqual(result['opponent_difficulty'].tolist(), [2, 5])


class ValidationAndEvaluationTests(unittest.TestCase):
    def test_schema_rejects_missing_columns(self):
        valid, issues = validate_gameweek_data(pd.DataFrame({'element': [1]}))
        self.assertFalse(valid)
        self.assertTrue(any('Missing required columns' in item
                            for item in issues))

    def test_schema_allows_double_gameweek_but_not_duplicate_fixture(self):
        frame = pd.DataFrame({
            'element': [1, 1],
            'round': [2, 2],
            'fixture': [20, 21],
            'total_points': [1, 2],
            'minutes': [45, 90],
        })
        self.assertTrue(validate_gameweek_data(frame)[0])
        frame.loc[1, 'fixture'] = 20
        self.assertFalse(validate_gameweek_data(frame)[0])

    def test_openfpl_return_categories_use_minutes(self):
        self.assertEqual(_get_category(0, 0), 'Zeros')
        self.assertEqual(_get_category(0, 30), 'Blanks')
        self.assertEqual(_get_category(-2, 90), 'Blanks')
        self.assertEqual(_get_category(4, 90), 'Tickers')
        self.assertEqual(_get_category(5, 90), 'Haulers')

    def test_team_split_uses_stable_club_names(self):
        frame = pd.DataFrame({
            'team_name': np.repeat(['A', 'B', 'C', 'D', 'E'], 4),
            'team': np.tile([1, 2], 10),
            'total_points': np.arange(20),
        })
        cv = TeamStratifiedKFold(n_splits=5, random_state=42)
        for train_idx, val_idx in cv.split(frame):
            train_teams = set(frame.iloc[train_idx]['team_name'])
            val_teams = set(frame.iloc[val_idx]['team_name'])
            self.assertTrue(val_teams)
            self.assertTrue(train_teams.isdisjoint(val_teams))

    def test_balanced_weights_are_positive_for_negative_scores(self):
        weights = balanced_return_weights(np.array([-7, 0, 2, 10]))
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights > 0).all())
        self.assertAlmostEqual(weights.mean(), 1.0)


if __name__ == '__main__':
    unittest.main()
