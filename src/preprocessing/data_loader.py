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
from src.preprocessing.schemas import DataSchema


def read_csv_safe(file_path: Path) -> pd.DataFrame:
    """Read CSV with automatic encoding detection

    Args:
        file_path (Path): Path to CSV file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue

    # If all encodings fail, try with error handling
    return pd.read_csv(file_path, encoding='utf-8', errors='ignore')


class FPLDataLoader:
    """Loader for FPL datasets with automatic schema application"""

    def __init__(self, data_dir: Path = RAW_DATA_DIR):
        """Initialize data loader

        Args:
            data_dir (Path): Base directory for data
        """
        self.data_dir = Path(data_dir)
        self.schema = DataSchema()

    def load_players(self, season: str, standardize: bool = True) -> pd.DataFrame:
        """Load player data for a season

        Args:
            season (str): Season identifier (e.g., '2023-24')
            standardize (bool): Apply schema standardization

        Returns:
            pd.DataFrame: Player data
        """
        file_path = self.data_dir / season / 'players_raw.csv'

        if not file_path.exists():
            # Try alternative name
            file_path = self.data_dir / season / 'cleaned_players.csv'

        if not file_path.exists():
            raise FileNotFoundError(f"Player data not found for season {season} at {file_path}")

        # Try different encodings
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='latin1')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='cp1252')

        if standardize:
            df = self.schema.prepare_player_data(df)

        return df

    def load_gameweeks(self, season: str, standardize: bool = True,
                      gameweeks: Optional[List[int]] = None) -> pd.DataFrame:
        """Load gameweek data for a season

        Args:
            season (str): Season identifier
            standardize (bool): Apply schema standardization
            gameweeks (list): Specific gameweeks to load. If None, load all

        Returns:
            pd.DataFrame: Gameweek data
        """
        # Try merged file first
        merged_path = self.data_dir / season / 'gws' / 'merged_gw.csv'

        if merged_path.exists():
            # Try different encodings
            try:
                df = pd.read_csv(merged_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(merged_path, encoding='latin1')
                except UnicodeDecodeError:
                    df = pd.read_csv(merged_path, encoding='cp1252')

            # Filter to specific gameweeks if requested
            if gameweeks is not None and 'round' in df.columns:
                df = df[df['round'].isin(gameweeks)]

            if standardize:
                df = self.schema.prepare_gameweek_data(df)

            return df

        # Otherwise, load and merge individual gameweek files
        gws_dir = self.data_dir / season / 'gws'
        if not gws_dir.exists():
            raise FileNotFoundError(f"Gameweek directory not found: {gws_dir}")

        if gameweeks is None:
            # Load all available gameweek files
            gw_files = sorted(gws_dir.glob('gw*.csv'))
        else:
            gw_files = [gws_dir / f'gw{gw}.csv' for gw in gameweeks
                       if (gws_dir / f'gw{gw}.csv').exists()]

        if not gw_files:
            raise FileNotFoundError(f"No gameweek files found for season {season}")

        dfs = []
        for gw_file in gw_files:
            try:
                df = pd.read_csv(gw_file)
                # Add gameweek number if not present
                if 'round' not in df.columns and 'GW' not in df.columns:
                    gw_num = int(gw_file.stem.replace('gw', ''))
                    df['round'] = gw_num
                dfs.append(df)
            except Exception as e:
                warnings.warn(f"Failed to load {gw_file}: {str(e)}")

        if not dfs:
            raise ValueError(f"No gameweek data could be loaded for season {season}")

        df = pd.concat(dfs, ignore_index=True)

        if standardize:
            df = self.schema.prepare_gameweek_data(df)

        return df

    def load_teams(self, season: str, standardize: bool = True) -> pd.DataFrame:
        """Load team data for a season

        Args:
            season (str): Season identifier
            standardize (bool): Apply schema standardization

        Returns:
            pd.DataFrame: Team data
        """
        file_path = self.data_dir / season / 'teams.csv'

        if not file_path.exists():
            raise FileNotFoundError(f"Team data not found for season {season}")

        df = pd.read_csv(file_path)

        if standardize:
            df = self.schema.prepare_team_data(df)

        return df

    def load_fixtures(self, season: str, standardize: bool = True) -> pd.DataFrame:
        """Load fixture data for a season

        Args:
            season (str): Season identifier
            standardize (bool): Apply schema standardization

        Returns:
            pd.DataFrame: Fixture data
        """
        file_path = self.data_dir / season / 'fixtures.csv'

        if not file_path.exists():
            raise FileNotFoundError(f"Fixture data not found for season {season}")

        df = pd.read_csv(file_path)

        if standardize:
            df = self.schema.prepare_fixture_data(df)

        return df

    def load_player_gameweeks(self, season: str, player_id: int) -> pd.DataFrame:
        """Load gameweek history for a specific player

        Args:
            season (str): Season identifier
            player_id (int): Player ID

        Returns:
            pd.DataFrame: Player's gameweek history
        """
        player_dir = self.data_dir / season / 'players' / f'player_{player_id}'

        # Try different file names
        possible_files = ['gw_history.csv', 'gw.csv', 'gws.csv']

        for filename in possible_files:
            file_path = player_dir / filename
            if file_path.exists():
                return pd.read_csv(file_path)

        raise FileNotFoundError(
            f"Player gameweek data not found for player {player_id} in season {season}"
        )

    def load_player_history(self, season: str, player_id: int) -> pd.DataFrame:
        """Load season history for a specific player

        Args:
            season (str): Season identifier
            player_id (int): Player ID

        Returns:
            pd.DataFrame: Player's season history
        """
        player_dir = self.data_dir / season / 'players' / f'player_{player_id}'

        # Try different file names
        possible_files = ['season_history.csv', 'history.csv']

        for filename in possible_files:
            file_path = player_dir / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                return df if not df.empty else None

        return None  # No history available (e.g., rookie player)

    def load_multi_season_gameweeks(self, seasons: Optional[List[str]] = None,
                                   standardize: bool = True) -> pd.DataFrame:
        """Load gameweek data across multiple seasons

        Args:
            seasons (list): List of seasons to load. If None, load all available
            standardize (bool): Apply schema standardization

        Returns:
            pd.DataFrame: Combined gameweek data with season column
        """
        if seasons is None:
            seasons = self.get_available_seasons()

        all_data = []

        for season in seasons:
            try:
                df = self.load_gameweeks(season, standardize=standardize)
                df['season'] = season
                all_data.append(df)
                print(f"  ✓ Loaded {len(df)} records from {season}")
            except FileNotFoundError:
                print(f"  ⊙ Skipping {season} (data not available)")
                continue

        if not all_data:
            raise ValueError("No gameweek data could be loaded from any season")

        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal records: {len(combined_df)}")

        return combined_df

    def load_multi_season_players(self, seasons: Optional[List[str]] = None,
                                 standardize: bool = True) -> pd.DataFrame:
        """Load player data across multiple seasons

        Args:
            seasons (list): List of seasons to load. If None, load all available
            standardize (bool): Apply schema standardization

        Returns:
            pd.DataFrame: Combined player data with season column
        """
        if seasons is None:
            seasons = self.get_available_seasons()

        all_data = []

        for season in seasons:
            try:
                df = self.load_players(season, standardize=standardize)
                df['season'] = season
                all_data.append(df)
                print(f"  ✓ Loaded {len(df)} players from {season}")
            except FileNotFoundError:
                print(f"  ⊙ Skipping {season} (data not available)")
                continue

        if not all_data:
            raise ValueError("No player data could be loaded from any season")

        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal records: {len(combined_df)}")

        return combined_df

    def get_available_seasons(self) -> List[str]:
        """Get list of seasons with available data

        Returns:
            list: Available season identifiers
        """
        if not self.data_dir.exists():
            return []

        # Look for directories that match season pattern
        season_dirs = [d.name for d in self.data_dir.iterdir()
                      if d.is_dir() and '-' in d.name]

        return sorted(season_dirs)

    def get_data_summary(self, season: str) -> dict:
        """Get summary of available data for a season

        Args:
            season (str): Season identifier

        Returns:
            dict: Summary of available data
        """
        season_dir = self.data_dir / season

        if not season_dir.exists():
            return {'exists': False}

        summary = {
            'exists': True,
            'season': season,
            'files': {},
            'gameweeks': 0,
            'players': 0,
            'teams': 0,
        }

        # Check for main files
        file_checks = {
            'players': ['players_raw.csv', 'cleaned_players.csv'],
            'teams': ['teams.csv'],
            'fixtures': ['fixtures.csv'],
            'merged_gw': ['gws/merged_gw.csv'],
        }

        for key, filenames in file_checks.items():
            for filename in filenames:
                file_path = season_dir / filename
                if file_path.exists():
                    summary['files'][key] = filename
                    break

        # Count gameweek files
        gws_dir = season_dir / 'gws'
        if gws_dir.exists():
            gw_files = list(gws_dir.glob('gw*.csv'))
            summary['gameweeks'] = len(gw_files)

        # Count player directories
        players_dir = season_dir / 'players'
        if players_dir.exists():
            player_dirs = list(players_dir.glob('player_*'))
            summary['players'] = len(player_dirs)

        # Count teams
        if 'teams' in summary['files']:
            try:
                teams_df = pd.read_csv(season_dir / summary['files']['teams'])
                summary['teams'] = len(teams_df)
            except Exception:
                pass

        return summary


def quick_load(season: str = '2024-25', data_type: str = 'gameweeks') -> pd.DataFrame:
    """Quick load function for common use cases

    Args:
        season (str): Season to load
        data_type (str): Type of data ('gameweeks', 'players', 'teams', 'fixtures')

    Returns:
        pd.DataFrame: Requested data
    """
    loader = FPLDataLoader()

    if data_type == 'gameweeks':
        return loader.load_gameweeks(season)
    elif data_type == 'players':
        return loader.load_players(season)
    elif data_type == 'teams':
        return loader.load_teams(season)
    elif data_type == 'fixtures':
        return loader.load_fixtures(season)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


if __name__ == '__main__':
    # Example usage
    loader = FPLDataLoader()

    print("Available seasons:")
    seasons = loader.get_available_seasons()
    for season in seasons:
        summary = loader.get_data_summary(season)
        if summary['exists']:
            print(f"  {season}: {summary['gameweeks']} GWs, "
                  f"{summary['players']} players, {summary['teams']} teams")
