"""
Feature engineering for FPL player performance prediction.

CRITICAL: All rolling/lag features are shifted by 1 gameweek to prevent
data leakage. When predicting GW N, we only use data from GW 1..N-1.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import ROLLING_WINDOWS
from src.preprocessing.schemas import POSITIONS


def _shifted_rolling_mean(group: pd.Series, window: int) -> pd.Series:
    """Compute rolling mean SHIFTED by 1 to avoid data leakage.

    For GW N, this returns the mean of GW (N-window)...(N-1).
    """
    return group.shift(1).rolling(window=window, min_periods=1).mean()


def _shifted_rolling_sum(group: pd.Series, window: int) -> pd.Series:
    """Compute rolling sum SHIFTED by 1 to avoid data leakage."""
    return group.shift(1).rolling(window=window, min_periods=1).sum()


def _shifted_rolling_std(group: pd.Series, window: int) -> pd.Series:
    """Compute rolling std SHIFTED by 1 to avoid data leakage."""
    return group.shift(1).rolling(window=window, min_periods=1).std()


class FPLFeatureEngineer:
    """Feature engineering aligned with MODEL_SPEC.md Tier 1 + Tier 2 features.

    All rolling features use shift(1) so that for predicting GW N,
    only data from GW 1..N-1 is used (no data leakage).
    """

    def __init__(self, rolling_windows: Optional[List[int]] = None):
        self.rolling_windows = rolling_windows or [3, 5]

    def _group_key(self, df: pd.DataFrame):
        """Return groupby key: ['element'] if multi-season, else 'element'.

        """
        return ['season', 'element'] if 'season' in df.columns else ['element']

    def _round_feature(self, df: pd.DataFrame, source: str, window: int,
                       statistic: str = 'mean',
                       round_aggregation: str = 'sum') -> pd.Series:
        '''Calculate a lagged rolling feature without leaking within a DGW.

        All fixtures in the same gameweek receive the same value calculated
        solely from completed earlier gameweeks.
        '''
        player_keys = self._group_key(df)
        round_keys = player_keys + ['round']
        per_round = (
            df.groupby(round_keys, sort=False, dropna=False)[source]
            .agg(round_aggregation)
            .reset_index()
            .sort_values(round_keys)
        )
        grouped = per_round.groupby(player_keys, sort=False)[source]
        calculators = {
            'mean': _shifted_rolling_mean,
            'sum': _shifted_rolling_sum,
            'std': _shifted_rolling_std,
        }
        if statistic not in calculators:
            raise ValueError(f'Unsupported statistic: {statistic}')
        per_round['_feature_value'] = grouped.transform(
            lambda x: calculators[statistic](x, window))

        original = df[round_keys].copy()
        original['_row_order'] = np.arange(len(df))
        return (
            original.merge(
                per_round[round_keys + ['_feature_value']],
                on=round_keys,
                how='left',
            )
            .sort_values('_row_order')['_feature_value']
            .set_axis(df.index)
        )

    def _expanding_round_feature(
            self, df: pd.DataFrame, source: str, operation: str,
            round_aggregation: str = 'sum') -> pd.Series:
        '''Calculate a lagged expanding feature at gameweek boundaries.'''
        player_keys = self._group_key(df)
        round_keys = player_keys + ['round']
        per_round = (
            df.groupby(round_keys, sort=False, dropna=False)[source]
            .agg(round_aggregation)
            .reset_index()
            .sort_values(round_keys)
        )

        def calculate(series):
            if operation == 'cumsum':
                return series.shift(1).cumsum()
            if operation == 'count_positive':
                return (series > 0).shift(1).cumsum()
            raise ValueError(f'Unsupported operation: {operation}')

        per_round['_feature_value'] = (
            per_round.groupby(player_keys, sort=False)[source]
            .transform(calculate)
        )
        original = df[round_keys].copy()
        original['_row_order'] = np.arange(len(df))
        return (
            original.merge(
                per_round[round_keys + ['_feature_value']],
                on=round_keys,
                how='left',
            )
            .sort_values('_row_order')['_feature_value']
            .set_axis(df.index)
        )

    def _custom_round_feature(self, df: pd.DataFrame, source: str,
                              calculator,
                              round_aggregation: str = 'sum') -> pd.Series:
        '''Map a custom lagged calculation from player-rounds to fixtures.'''
        player_keys = self._group_key(df)
        round_keys = player_keys + ['round']
        per_round = (
            df.groupby(round_keys, sort=False, dropna=False)[source]
            .agg(round_aggregation)
            .reset_index()
            .sort_values(round_keys)
        )
        per_round['_feature_value'] = (
            per_round.groupby(player_keys, sort=False)[source]
            .transform(calculator)
        )
        original = df[round_keys].copy()
        original['_row_order'] = np.arange(len(df))
        return (
            original.merge(
                per_round[round_keys + ['_feature_value']],
                on=round_keys,
                how='left',
            )
            .sort_values('_row_order')['_feature_value']
            .set_axis(df.index)
        )

    def create_tier1_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Tier 1 (baseline) features for Linear Regression.

        Features (~10):
          - form_last_3, form_last_5: avg total_points over last 3/5 GWs
          - minutes_last_3: avg minutes over last 3 GWs
          - was_home: already in data (1/0)
          - opponent_difficulty: added separately from fixtures
          - ict_index_last_3: avg ICT index over last 3 GWs
          - position: one-hot encoded (GK, DEF, MID, FWD)
          - value: player price / 10
        """
        df = df.copy()
        # Form: average points in last 3 and 5 games (shifted!)
        df['form_last_3'] = self._round_feature(df, 'total_points', 3)
        df['form_last_5'] = self._round_feature(df, 'total_points', 5)

        # Minutes: average over last 3 games
        df['minutes_last_3'] = self._round_feature(df, 'minutes', 3)

        # ICT index: average over last 3 games
        if 'ict_index' in df.columns:
            df['ict_index_last_3'] = self._round_feature(df, 'ict_index', 3)

        # Value as price in millions
        if 'value' in df.columns:
            df['price'] = df['value'] / 10.0

        # Position one-hot encoding
        if 'position_label' in df.columns:
            for pos in ['DEF', 'MID', 'FWD']:  # GK is the reference category
                df[f'pos_{pos}'] = (df['position_label'] == pos).astype(int)

        return df

    def create_tier2_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Tier 2 (extended) features

        All Tier 1 features PLUS additional rolling stats.
        """
        df = df.copy()
        # --- Rolling last-5 stats ---
        rolling5_cols = {
            'goals_scored': 'goals_last_5',
            'assists': 'assists_last_5',
            'clean_sheets': 'clean_sheets_last_5',
            'bps': 'bps_last_5',
            'influence': 'influence_last_5',
            'creativity': 'creativity_last_5',
            'threat': 'threat_last_5',
            'saves': 'saves_last_5',
            'yellow_cards': 'yellow_cards_last_5',
            'bonus': 'bonus_last_5',
        }

        for src_col, feat_name in rolling5_cols.items():
            if src_col in df.columns:
                df[feat_name] = self._round_feature(df, src_col, 5)

        # --- xG/xA features (available in 2022-23+) ---
        xg_cols = {
            'expected_goals': 'xG_last_5',
            'expected_assists': 'xA_last_5',
            'expected_goals_conceded': 'xGC_last_5',
        }
        for src_col, feat_name in xg_cols.items():
            if src_col in df.columns:
                df[feat_name] = self._round_feature(df, src_col, 5)

        # --- Season cumulative features ---
        # Cumulative points so far this season (shifted - excludes current GW)
        df['cumulative_points_season'] = self._expanding_round_feature(
            df, 'total_points', 'cumsum')

        # Games played so far (minutes > 0), shifted
        df['games_played_season'] = self._expanding_round_feature(
            df, 'minutes', 'count_positive')

        # --- Selection percentage (log-scaled) ---
        if 'selected' in df.columns:
            df['selected_pct'] = np.log1p(df['selected'])

        # --- Minutes last 5 (useful for XGBoost too) ---
        df['minutes_last_5'] = self._round_feature(df, 'minutes', 5)

        return df

    def create_tier3_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Tier 3 (advanced) features — ideas from FEATURE_IDEAS.md.

        Requires Tier 1 + Tier 2 features to already exist on df.

        Features added:
          best_3_streak_season  — season-best consecutive 3-game points sum (shifted)
          best_5_streak_season  — season-best consecutive 5-game points sum (shifted)
          red_card_last_game    — 1 if player received a red card last GW (suspended)
          hauler_rate_season    — % of games this season where player scored 5+ pts
          points_std_last_10    — rolling std of points over last 10 GWs (shifted)
          form_momentum         — form_last_3 minus form_last_5 (positive = improving)
        """
        df = df.copy()
        # ── 1. Season-best form streaks ───────────────────────────────────────
        # For each GW N, what was the best 3-game consecutive sum up to GW N-1?
        # Uses shift(1) → no leakage; min_periods=window ensures full window.
        df['best_3_streak_season'] = self._custom_round_feature(
            df, 'total_points',
            lambda x: (x.shift(1)
                        .rolling(3, min_periods=3)
                        .sum()
                        .expanding()
                        .max())
        )
        df['best_5_streak_season'] = self._custom_round_feature(
            df, 'total_points',
            lambda x: (x.shift(1)
                        .rolling(5, min_periods=5)
                        .sum()
                        .expanding()
                        .max())
        )

        # ── 2. Red card / suspension flag ─────────────────────────────────────
        # A red card in GW N-1 almost always means suspension in GW N.
        # clip(upper=1) turns multi-red-card edge cases into a clean 0/1 flag.
        if 'red_cards' in df.columns:
            df['red_card_last_game'] = self._custom_round_feature(
                df, 'red_cards',
                lambda x: x.shift(1).clip(upper=1).astype(float),
            )

        # ── 3. Return consistency: hauler rate ────────────────────────────────
        # Expanding proportion of games this season where player scored 5+ pts.
        # min_periods=3 so the ratio is only trusted after at least 3 games.
        df['hauler_rate_season'] = self._custom_round_feature(
            df, 'total_points',
            lambda x: (x >= 5).astype(float).shift(1)
            .expanding(min_periods=3).mean(),
        )

        # ── 4. Return consistency: points volatility ──────────────────────────
        # Rolling std over last 10 GWs. High std = streaky/volatile player.
        # Combine with hauler_rate to distinguish consistent haulers vs flukes.
        df['points_std_last_10'] = self._custom_round_feature(
            df, 'total_points',
            lambda x: x.shift(1).rolling(10, min_periods=3).std(),
        )

        # ── 5. Form momentum (derived from Tier 1 features) ───────────────────
        # Positive  → player is on form (recent avg > medium-term avg)
        # Negative  → player is cooling off or in a slump
        if 'form_last_3' in df.columns and 'form_last_5' in df.columns:
            df['form_momentum'] = df['form_last_3'] - df['form_last_5']

        return df

    def add_opponent_features(self, df: pd.DataFrame,
                              teams_df: pd.DataFrame,
                              fixtures_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Add opponent difficulty and team strength features.

        Uses fixtures.csv for FDR (team_h_difficulty, team_a_difficulty)
        and teams.csv for team strength ratings.
        """
        df = df.copy()

        # --- Team strength (from teams.csv) ---
        if 'id' in teams_df.columns and 'strength' in teams_df.columns:
            team_strength = teams_df.set_index('id')['strength'].to_dict()

            # Player's team strength
            if 'team' in df.columns:
                df['team_strength'] = df['team'].map(team_strength)

            # Opponent team strength
            if 'opponent_team' in df.columns:
                df['opponent_strength'] = df['opponent_team'].map(team_strength)

        # --- Opponent difficulty (FDR from fixtures.csv) ---
        if fixtures_df is not None and 'event' in fixtures_df.columns:
            # Build a lookup: for each (team, round) -> FDR they face
            fdr_records = []
            for _, row in fixtures_df.iterrows():
                if pd.notna(row.get('event')):
                    gw = int(row['event'])
                    # Home team faces away-team difficulty
                    if 'team_h' in row and 'team_h_difficulty' in row:
                        fdr_records.append({
                            'fixture': row.get('id'),
                            'team': int(row['team_h']),
                            'round': gw,
                            'opponent_difficulty': row['team_h_difficulty']
                        })
                    # Away team faces home-team difficulty
                    if 'team_a' in row and 'team_a_difficulty' in row:
                        fdr_records.append({
                            'fixture': row.get('id'),
                            'team': int(row['team_a']),
                            'round': gw,
                            'opponent_difficulty': row['team_a_difficulty']
                        })

            if fdr_records:
                fdr_df = pd.DataFrame(fdr_records)
                df = df.drop(columns=['opponent_difficulty'], errors='ignore')

                # Fixture IDs preserve distinct FDR values in double gameweeks.
                if 'fixture' in df.columns and fdr_df['fixture'].notna().any():
                    fixture_fdr = (
                        fdr_df.dropna(subset=['fixture'])
                        .drop_duplicates(['fixture', 'team'])
                    )
                    df = df.merge(
                        fixture_fdr[
                            ['fixture', 'team', 'opponent_difficulty']],
                        on=['fixture', 'team'],
                        how='left',
                    )

                # Historical files without fixture IDs use a conservative
                # player-team/gameweek fallback.
                if ('opponent_difficulty' not in df.columns
                        or df['opponent_difficulty'].isna().any()):
                    round_fdr = (
                        fdr_df.groupby(['team', 'round'], as_index=False)
                        ['opponent_difficulty'].max()
                        .rename(columns={
                            'opponent_difficulty': '_round_difficulty'})
                    )
                    df = df.merge(round_fdr, on=['team', 'round'], how='left')
                    if 'opponent_difficulty' not in df.columns:
                        df['opponent_difficulty'] = df['_round_difficulty']
                    else:
                        df['opponent_difficulty'] = (
                            df['opponent_difficulty']
                            .fillna(df['_round_difficulty'])
                        )
                    df = df.drop(columns=['_round_difficulty'])
        elif 'opponent_team' in df.columns and 'opponent_strength' in df.columns:
            # Fallback: use opponent strength as difficulty proxy
            if 'opponent_difficulty' not in df.columns:
                df['opponent_difficulty'] = df['opponent_strength']

        return df

    def create_all_features(self, df: pd.DataFrame,
                            teams_df: Optional[pd.DataFrame] = None,
                            fixtures_df: Optional[pd.DataFrame] = None,
                            tier: int = 2) -> pd.DataFrame:
        """Create all features for the specified tier.

        Args:
            df: Gameweek data for a SINGLE season only. For multiple seasons,
                call this function once per season and concatenate afterward.
                Passing combined multi-season data will break rolling windows
                because rounds reset 1→38 each season.
            teams_df: Team data from teams.csv
            fixtures_df: Fixture data from fixtures.csv
            tier: 1 for baseline features, 2 for all features

        Returns:
            DataFrame with engineered features added
        """
        # Sort by season (if present) then element then round.
        # Season must come before round to prevent cross-season interleaving.
        sort_cols = (['season', 'element', 'round']
                     if 'season' in df.columns else ['element', 'round'])
        df = df.sort_values(sort_cols).reset_index(drop=True)

        print("  Creating Tier 1 (baseline) features...")
        df = self.create_tier1_features(df)

        if tier >= 2:
            print("  Creating Tier 2 (extended) features...")
            df = self.create_tier2_features(df)

        if tier >= 3:
            print("  Creating Tier 3 (advanced consistency) features...")
            df = self.create_tier3_features(df)

        if teams_df is not None:
            print("  Adding opponent/team features...")
            df = self.add_opponent_features(df, teams_df, fixtures_df)

        # Count new features
        base_cols = {'element', 'round', 'total_points', 'minutes', 'name',
                     'position', 'position_label', 'element_type', 'team',
                     'season', 'fixture', 'kickoff_time', 'opponent_team',
                     'was_home', 'value', 'selected', 'transfers_in',
                     'transfers_out', 'transfers_balance', 'starts',
                     'team_a_score', 'team_h_score', 'xP'}
        new_features = [c for c in df.columns if c not in base_cols]
        print(f"  Total engineered features: {len(new_features)}")

        return df


# --- Feature name lists for model building ---

TIER0_FEATURES = [
    # Transparent baseline: recent form plus a fixed fixture adjustment.
    'form_last_5',
    'opponent_difficulty',
]

TIER1_FEATURES = [
    'form_last_3', 'form_last_5', 'minutes_last_3',
    'was_home', 'opponent_difficulty',
    'ict_index_last_3', 'price',
    'pos_DEF', 'pos_MID', 'pos_FWD',
]

TIER2_FEATURES = TIER1_FEATURES + [
    'goals_last_5', 'assists_last_5', 'clean_sheets_last_5',
    'bps_last_5', 'influence_last_5', 'creativity_last_5',
    'threat_last_5', 'xG_last_5', 'xA_last_5', 'xGC_last_5',
    'saves_last_5', 'yellow_cards_last_5', 'bonus_last_5',
    'selected_pct', 'team_strength', 'opponent_strength',
    'cumulative_points_season', 'games_played_season',
    'minutes_last_5',
]

TIER3_FEATURES = TIER2_FEATURES + [
    # Best seasonal streaks (idea 1)
    'best_3_streak_season', 'best_5_streak_season',
    # Suspension flag (idea 3)
    'red_card_last_game',
    # Return consistency metrics (idea 4)
    'hauler_rate_season', 'points_std_last_10',
    # Form momentum (derived)
    'form_momentum',
]


def prepare_training_data(df: pd.DataFrame,
                          target_col: str = 'total_points',
                          min_gw: int = 6,
                          feature_list: Optional[List[str]] = None) -> pd.DataFrame:
    """Prepare data for training by removing rows with insufficient history.

    Args:
        df: Feature-engineered data
        target_col: Target variable column
        min_gw: Minimum gameweek to include (first N GWs have incomplete features)
        feature_list: If provided, only keep these feature columns + target

    Returns:
        Training-ready DataFrame
    """
    df = df.copy()

    # Drop early gameweeks where rolling features are unreliable
    if 'round' in df.columns:
        df = df[df['round'] >= min_gw]

    # Drop rows where target is missing
    df = df[df[target_col].notna()]

    # Drop rows where key features are all NaN
    if feature_list:
        available = [c for c in feature_list if c in df.columns]
        df = df.dropna(subset=available, how='all')

    return df
