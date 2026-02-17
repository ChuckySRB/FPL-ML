"""
Data loading utilities for FPL datasets
"""
import sys
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SEASONS
from src.preprocessing.schemas import DataSchema, POSITIONS


def read_csv_safe(file_path: Path) -> pd.DataFrame:
    """Read CSV with automatic encoding detection"""
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(file_path, encoding='utf-8', errors='ignore')


class FPLDataLoader:
    """Loader for FPL datasets"""

    def __init__(self, data_dir: Path = RAW_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.schema = DataSchema()

    def load_gameweeks(self, season: str, standardize: bool = True,
                       gameweeks: Optional[List[int]] = None) -> pd.DataFrame:
        """Load gameweek data for a season.

        Handles the duplicate round/GW column issue and position text column.
        """
        merged_path = self.data_dir / season / 'gws' / 'merged_gw.csv'

        if merged_path.exists():
            df = read_csv_safe(merged_path)
        else:
            # Load individual GW files
            gws_dir = self.data_dir / season / 'gws'
            if not gws_dir.exists():
                raise FileNotFoundError(f"Gameweek directory not found: {gws_dir}")

            gw_files = sorted(gws_dir.glob('gw*.csv'))
            if gameweeks is not None:
                gw_files = [gws_dir / f'gw{gw}.csv' for gw in gameweeks
                           if (gws_dir / f'gw{gw}.csv').exists()]
            if not gw_files:
                raise FileNotFoundError(f"No gameweek files found for season {season}")

            dfs = []
            for gw_file in gw_files:
                try:
                    gw_df = pd.read_csv(gw_file)
                    if 'round' not in gw_df.columns and 'GW' not in gw_df.columns:
                        gw_num = int(gw_file.stem.replace('gw', ''))
                        gw_df['round'] = gw_num
                    dfs.append(gw_df)
                except Exception as e:
                    warnings.warn(f"Failed to load {gw_file}: {str(e)}")
            df = pd.concat(dfs, ignore_index=True)

        # --- Fix duplicate round/GW column ---
        # merged_gw.csv has both 'round' and 'GW' columns with same data
        if 'GW' in df.columns and 'round' in df.columns:
            # Keep 'round', drop 'GW'
            df = df.drop(columns=['GW'])
        elif 'GW' in df.columns:
            df = df.rename(columns={'GW': 'round'})

        # --- Fix position column ---
        # merged_gw.csv has 'position' as text (GK, DEF, MID, FWD)
        # We need 'element_type' as numeric ID for consistency
        position_to_id = {'GK': 1, 'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
        if 'position' in df.columns and df['position'].dtype == object:
            df['element_type'] = df['position'].map(position_to_id)
            df['position_label'] = df['position']
        elif 'element_type' in df.columns:
            df['position_label'] = df['element_type'].map(POSITIONS)

        # --- Ensure numeric types for key columns ---
        numeric_cols = ['element', 'round', 'total_points', 'minutes', 'goals_scored',
                       'assists', 'clean_sheets', 'goals_conceded', 'bonus', 'bps',
                       'saves', 'yellow_cards', 'red_cards', 'own_goals',
                       'penalties_saved', 'penalties_missed', 'value', 'selected',
                       'transfers_in', 'transfers_out']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        float_cols = ['influence', 'creativity', 'threat', 'ict_index',
                     'expected_goals', 'expected_assists',
                     'expected_goal_involvements', 'expected_goals_conceded', 'xP']
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- Handle was_home ---
        if 'was_home' in df.columns:
            df['was_home'] = df['was_home'].map(
                {True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0}
            ).fillna(0).astype(int)

        # --- Map team name to numeric ID ---
        # merged_gw.csv has 'team' as text (e.g., "Southampton") but
        # opponent_team is numeric. We need team as numeric for joins.
        if 'team' in df.columns and df['team'].dtype == object:
            df['team_name'] = df['team']  # Keep original name
            # Load teams.csv to get name->id mapping
            try:
                teams_df = pd.read_csv(self.data_dir / season / 'teams.csv')
                name_to_id = dict(zip(teams_df['name'], teams_df['id']))
                # Also try short_name for partial matches
                short_to_id = dict(zip(teams_df['short_name'], teams_df['id']))
                df['team'] = df['team_name'].map(name_to_id)
                # Fill any unmapped using short_name
                unmapped = df['team'].isna()
                if unmapped.any():
                    df.loc[unmapped, 'team'] = df.loc[unmapped, 'team_name'].map(short_to_id)
                df['team'] = df['team'].astype('Int64')
            except FileNotFoundError:
                pass

        # Filter to specific gameweeks if requested
        if gameweeks is not None and 'round' in df.columns:
            df = df[df['round'].isin(gameweeks)]

        return df

    def load_teams(self, season: str) -> pd.DataFrame:
        """Load team data for a season"""
        file_path = self.data_dir / season / 'teams.csv'
        if not file_path.exists():
            raise FileNotFoundError(f"Team data not found for season {season}")
        return pd.read_csv(file_path)

    def load_fixtures(self, season: str) -> pd.DataFrame:
        """Load fixture data for a season"""
        file_path = self.data_dir / season / 'fixtures.csv'
        if not file_path.exists():
            raise FileNotFoundError(f"Fixture data not found for season {season}")
        df = pd.read_csv(file_path)
        if 'kickoff_time' in df.columns:
            df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], errors='coerce')
        return df

    def load_players(self, season: str) -> pd.DataFrame:
        """Load player data for a season"""
        for fname in ['players_raw.csv', 'cleaned_players.csv']:
            file_path = self.data_dir / season / fname
            if file_path.exists():
                return read_csv_safe(file_path)
        raise FileNotFoundError(f"Player data not found for season {season}")

    def load_season(self, season: str) -> dict:
        """Load all data for a season, returned as a dict of DataFrames"""
        result = {'season': season}
        result['gameweeks'] = self.load_gameweeks(season)
        result['teams'] = self.load_teams(season)
        try:
            result['fixtures'] = self.load_fixtures(season)
        except FileNotFoundError:
            result['fixtures'] = None
        try:
            result['players'] = self.load_players(season)
        except FileNotFoundError:
            result['players'] = None
        return result

    def load_multi_season(self, seasons: List[str]) -> pd.DataFrame:
        """Load and combine gameweek data across multiple seasons.

        Handles double gameweeks by aggregating duplicate player-round entries.
        """
        all_data = []
        for season in seasons:
            try:
                df = self.load_gameweeks(season)
                df['season'] = season
                all_data.append(df)
                print(f"  Loaded {season}: {len(df):,} records, "
                      f"{df['element'].nunique()} players, "
                      f"{int(df['round'].max())} GWs")
            except FileNotFoundError:
                print(f"  Skipping {season} (not found)")
                continue

        if not all_data:
            raise ValueError("No gameweek data loaded")

        combined = pd.concat(all_data, ignore_index=True)

        # Handle double gameweeks: aggregate stats for players with multiple
        # entries in the same round (e.g., GW7 double fixtures)
        group_cols = ['element', 'round', 'season']
        duplicates = combined.duplicated(subset=group_cols, keep=False)
        n_dups = duplicates.sum()

        if n_dups > 0:
            print(f"\n  Found {n_dups} double-GW records, aggregating...")

            # Columns to sum (match stats)
            sum_cols = ['total_points', 'minutes', 'goals_scored', 'assists',
                       'clean_sheets', 'goals_conceded', 'own_goals',
                       'penalties_saved', 'penalties_missed', 'yellow_cards',
                       'red_cards', 'saves', 'bonus', 'bps']
            sum_cols = [c for c in sum_cols if c in combined.columns]

            # Columns to average (rates/indices)
            mean_cols = ['influence', 'creativity', 'threat', 'ict_index',
                        'expected_goals', 'expected_assists',
                        'expected_goal_involvements', 'expected_goals_conceded', 'xP']
            mean_cols = [c for c in mean_cols if c in combined.columns]

            # Columns to take first (metadata)
            first_cols = ['name', 'position', 'position_label', 'element_type',
                         'team', 'value', 'was_home', 'opponent_team',
                         'selected', 'transfers_in', 'transfers_out',
                         'transfers_balance', 'kickoff_time', 'fixture', 'starts']
            first_cols = [c for c in first_cols if c in combined.columns]

            agg_dict = {}
            for c in sum_cols:
                agg_dict[c] = 'sum'
            for c in mean_cols:
                agg_dict[c] = 'mean'
            for c in first_cols:
                agg_dict[c] = 'first'

            combined = combined.groupby(group_cols, as_index=False).agg(agg_dict)
            print(f"  After aggregation: {len(combined):,} records")

        print(f"\nTotal: {len(combined):,} player-gameweek records")
        return combined

    def get_available_seasons(self) -> List[str]:
        """Get list of seasons with available data"""
        if not self.data_dir.exists():
            return []
        return sorted([d.name for d in self.data_dir.iterdir()
                       if d.is_dir() and '-' in d.name])


def quick_load(season: str = '2023-24', data_type: str = 'gameweeks') -> pd.DataFrame:
    """Quick load function for common use cases"""
    loader = FPLDataLoader()
    loaders = {
        'gameweeks': loader.load_gameweeks,
        'players': loader.load_players,
        'teams': loader.load_teams,
        'fixtures': loader.load_fixtures,
    }
    if data_type not in loaders:
        raise ValueError(f"Unknown data_type: {data_type}")
    return loaders[data_type](season)
