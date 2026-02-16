"""
Feature engineering for FPL player performance prediction
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import ROLLING_WINDOWS, LAG_FEATURES


class FPLFeatureEngineer:
    """Feature engineering for FPL data"""

    def __init__(self):
        self.rolling_windows = ROLLING_WINDOWS
        self.lag_features = LAG_FEATURES

    def create_rolling_features(self, df: pd.DataFrame,
                               columns: List[str],
                               windows: Optional[List[int]] = None,
                               group_by: str = 'element') -> pd.DataFrame:
        """Create rolling average features

        Args:
            df (pd.DataFrame): Gameweek data sorted by element and round
            columns (list): Columns to create rolling features for
            windows (list): Window sizes for rolling averages
            group_by (str): Column to group by (usually 'element' for player)

        Returns:
            pd.DataFrame: DataFrame with rolling features added
        """
        if windows is None:
            windows = self.rolling_windows

        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            for window in windows:
                # Rolling mean
                df[f'{col}_rolling_{window}'] = df.groupby(group_by)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )

                # Rolling std (for volatility)
                df[f'{col}_rolling_std_{window}'] = df.groupby(group_by)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )

        return df

    def create_lag_features(self, df: pd.DataFrame,
                           columns: List[str],
                           lags: Optional[List[int]] = None,
                           group_by: str = 'element') -> pd.DataFrame:
        """Create lagged features (previous gameweek values)

        Args:
            df (pd.DataFrame): Gameweek data sorted by element and round
            columns (list): Columns to create lag features for
            lags (list): Number of lags to create
            group_by (str): Column to group by

        Returns:
            pd.DataFrame: DataFrame with lag features added
        """
        if lags is None:
            lags = self.lag_features

        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            for lag in lags:
                df[f'{col}_lag_{lag}'] = df.groupby(group_by)[col].shift(lag)

        return df

    def create_form_features(self, df: pd.DataFrame,
                            group_by: str = 'element') -> pd.DataFrame:
        """Create form-based features

        Args:
            df (pd.DataFrame): Gameweek data
            group_by (str): Column to group by

        Returns:
            pd.DataFrame: DataFrame with form features
        """
        df = df.copy()

        if 'total_points' in df.columns:
            # Recent form (weighted average of last 5 games)
            weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])  # More weight on recent games

            def weighted_form(x):
                if len(x) < 5:
                    return x.mean()
                recent = x.tail(5).values
                return np.average(recent, weights=weights)

            df['form_weighted'] = df.groupby(group_by)['total_points'].transform(
                lambda x: x.rolling(window=5, min_periods=1).apply(weighted_form, raw=False)
            )

            # Consistency (inverse of std deviation)
            df['consistency_5'] = 1 / (1 + df.groupby(group_by)['total_points'].transform(
                lambda x: x.rolling(window=5, min_periods=1).std()
            ).fillna(0))

            # Trend (is form improving or declining?)
            df['form_trend'] = df.groupby(group_by)['total_points'].transform(
                lambda x: x.rolling(window=5, min_periods=2).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0,
                    raw=False
                )
            )

        # Games played count
        df['games_played'] = df.groupby(group_by).cumcount() + 1

        # Minutes-based form (points per 90 minutes)
        if 'total_points' in df.columns and 'minutes' in df.columns:
            df['points_per_90'] = df.groupby(group_by).apply(
                lambda x: (x['total_points'] / (x['minutes'] + 1) * 90)
            ).reset_index(level=0, drop=True)

        return df

    def create_opponent_features(self, df: pd.DataFrame,
                                 teams_df: pd.DataFrame) -> pd.DataFrame:
        """Create opponent difficulty features

        Args:
            df (pd.DataFrame): Gameweek data with opponent_team column
            teams_df (pd.DataFrame): Team data with strength metrics

        Returns:
            pd.DataFrame: DataFrame with opponent features
        """
        df = df.copy()

        if 'opponent_team' not in df.columns:
            return df

        # Merge opponent strength
        opponent_cols = []
        for col in ['strength', 'strength_attack_home', 'strength_defence_home',
                   'strength_attack_away', 'strength_defence_away']:
            if col in teams_df.columns:
                df = df.merge(
                    teams_df[['id', col]].rename(columns={col: f'opponent_{col}'}),
                    left_on='opponent_team',
                    right_on='id',
                    how='left',
                    suffixes=('', '_opp')
                )
                opponent_cols.append(f'opponent_{col}')

        # Create difficulty rating (inverse of opponent strength)
        if 'opponent_strength' in df.columns:
            df['opponent_difficulty'] = 100 - df['opponent_strength']

        return df

    def create_home_away_features(self, df: pd.DataFrame,
                                  group_by: str = 'element') -> pd.DataFrame:
        """Create home/away specific features

        Args:
            df (pd.DataFrame): Gameweek data with was_home column
            group_by (str): Column to group by

        Returns:
            pd.DataFrame: DataFrame with home/away features
        """
        df = df.copy()

        if 'was_home' not in df.columns:
            return df

        # Convert to boolean if needed
        df['was_home'] = df['was_home'].astype(bool)

        if 'total_points' in df.columns:
            # Home and away averages
            df['avg_points_home'] = df[df['was_home']].groupby(group_by)['total_points'].transform('mean')
            df['avg_points_away'] = df[~df['was_home']].groupby(group_by)['total_points'].transform('mean')

            # Fill NaN with overall average
            overall_avg = df.groupby(group_by)['total_points'].transform('mean')
            df['avg_points_home'] = df['avg_points_home'].fillna(overall_avg)
            df['avg_points_away'] = df['avg_points_away'].fillna(overall_avg)

        return df

    def create_attacking_features(self, df: pd.DataFrame,
                                  group_by: str = 'element') -> pd.DataFrame:
        """Create attacking performance features

        Args:
            df (pd.DataFrame): Gameweek data
            group_by (str): Column to group by

        Returns:
            pd.DataFrame: DataFrame with attacking features
        """
        df = df.copy()

        # Goal involvement (goals + assists)
        if 'goals_scored' in df.columns and 'assists' in df.columns:
            df['goal_involvement'] = df['goals_scored'] + df['assists']

            # Rolling goal involvement
            for window in [3, 5]:
                df[f'goal_involvement_rolling_{window}'] = df.groupby(group_by)['goal_involvement'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )

        # Expected goals features
        if 'expected_goals' in df.columns:
            df['xg_rolling_3'] = df.groupby(group_by)['expected_goals'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )

            # Overperformance/underperformance vs xG
            if 'goals_scored' in df.columns:
                df['goals_vs_xg'] = df['goals_scored'] - df['expected_goals']
                df['goals_vs_xg_cumsum'] = df.groupby(group_by)['goals_vs_xg'].cumsum()

        # Expected assists features
        if 'expected_assists' in df.columns:
            df['xa_rolling_3'] = df.groupby(group_by)['expected_assists'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )

        # Creativity metrics
        if 'creativity' in df.columns:
            df['creativity_rolling_5'] = df.groupby(group_by)['creativity'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )

        # Threat metrics
        if 'threat' in df.columns:
            df['threat_rolling_5'] = df.groupby(group_by)['threat'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )

        return df

    def create_defensive_features(self, df: pd.DataFrame,
                                  group_by: str = 'element') -> pd.DataFrame:
        """Create defensive performance features

        Args:
            df (pd.DataFrame): Gameweek data
            group_by (str): Column to group by

        Returns:
            pd.DataFrame: DataFrame with defensive features
        """
        df = df.copy()

        # Clean sheet rolling average
        if 'clean_sheets' in df.columns:
            for window in [3, 5]:
                df[f'clean_sheets_rolling_{window}'] = df.groupby(group_by)['clean_sheets'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )

        # Goals conceded rolling average
        if 'goals_conceded' in df.columns:
            for window in [3, 5]:
                df[f'goals_conceded_rolling_{window}'] = df.groupby(group_by)['goals_conceded'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )

        # Expected goals conceded
        if 'expected_goals_conceded' in df.columns:
            df['xgc_rolling_5'] = df.groupby(group_by)['expected_goals_conceded'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )

        # Saves (for goalkeepers)
        if 'saves' in df.columns:
            df['saves_rolling_5'] = df.groupby(group_by)['saves'].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )

        return df

    def create_value_features(self, df: pd.DataFrame,
                             players_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create value and selection features

        Args:
            df (pd.DataFrame): Gameweek data
            players_df (pd.DataFrame): Player data with cost information

        Returns:
            pd.DataFrame: DataFrame with value features
        """
        df = df.copy()

        # Points per cost
        if 'now_cost' in df.columns and 'total_points' in df.columns:
            df['points_per_cost'] = df['total_points'] / (df['now_cost'] / 10 + 0.1)

        # Value score (points per million)
        if players_df is not None and 'element' in df.columns:
            if 'now_cost' in players_df.columns:
                cost_map = players_df.set_index('id')['now_cost'].to_dict()
                df['player_cost'] = df['element'].map(cost_map)
                df['value_score'] = df['total_points'] / (df['player_cost'] / 10 + 0.1)

        # Selection trends
        if 'selected' in df.columns:
            df['selection_change'] = df.groupby('element')['selected'].diff()

        if 'transfers_in' in df.columns and 'transfers_out' in df.columns:
            df['net_transfers'] = df['transfers_in'] - df['transfers_out']
            df['transfer_momentum'] = df.groupby('element')['net_transfers'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )

        return df

    def create_all_features(self, df: pd.DataFrame,
                           teams_df: Optional[pd.DataFrame] = None,
                           players_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create all features at once

        Args:
            df (pd.DataFrame): Gameweek data (must be sorted by element and round)
            teams_df (pd.DataFrame): Team data
            players_df (pd.DataFrame): Player data

        Returns:
            pd.DataFrame: DataFrame with all engineered features
        """
        print("Creating features...")

        # Ensure data is sorted
        if 'element' in df.columns and 'round' in df.columns:
            df = df.sort_values(['element', 'round'])

        # Core performance features
        core_features = ['total_points', 'minutes', 'goals_scored', 'assists', 'bonus']

        print("  → Rolling features...")
        df = self.create_rolling_features(df, core_features)

        print("  → Lag features...")
        df = self.create_lag_features(df, core_features)

        print("  → Form features...")
        df = self.create_form_features(df)

        print("  → Attacking features...")
        df = self.create_attacking_features(df)

        print("  → Defensive features...")
        df = self.create_defensive_features(df)

        print("  → Home/away features...")
        df = self.create_home_away_features(df)

        print("  → Value features...")
        df = self.create_value_features(df, players_df)

        if teams_df is not None:
            print("  → Opponent features...")
            df = self.create_opponent_features(df, teams_df)

        print("✓ Feature engineering complete!")
        print(f"  Total features: {len(df.columns)}")

        return df

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get feature groups for analysis

        Returns:
            dict: Feature groups
        """
        return {
            'rolling': [col for col in [] if 'rolling' in str(col)],
            'lag': [col for col in [] if 'lag' in str(col)],
            'form': ['form_weighted', 'consistency_5', 'form_trend', 'points_per_90'],
            'attacking': ['goal_involvement', 'goals_vs_xg', 'creativity_rolling_5', 'threat_rolling_5'],
            'defensive': ['clean_sheets_rolling_3', 'goals_conceded_rolling_3', 'saves_rolling_5'],
            'opponent': ['opponent_difficulty', 'opponent_strength'],
            'value': ['points_per_cost', 'value_score', 'transfer_momentum']
        }


def prepare_training_data(df: pd.DataFrame,
                         target_col: str = 'total_points',
                         drop_first_n_games: int = 5) -> pd.DataFrame:
    """Prepare data for training by removing incomplete records

    Args:
        df (pd.DataFrame): Feature-engineered data
        target_col (str): Target variable column
        drop_first_n_games (int): Drop first N games per player (insufficient history)

    Returns:
        pd.DataFrame: Training-ready data
    """
    df = df.copy()

    # Drop first N games per player (incomplete rolling features)
    if 'games_played' in df.columns:
        df = df[df['games_played'] > drop_first_n_games]

    # Drop rows where target is missing
    if target_col in df.columns:
        df = df[df[target_col].notna()]

    # Drop rows with too many missing values
    threshold = len(df.columns) * 0.5  # Allow 50% missing
    df = df.dropna(thresh=threshold)

    return df
