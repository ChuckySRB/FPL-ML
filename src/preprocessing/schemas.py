"""
Data schemas and column definitions for FPL data
"""
from typing import Dict, List
import pandas as pd


# Player position mapping
POSITIONS = {
    1: 'GK',
    2: 'DEF',
    3: 'MID',
    4: 'FWD'
}

POSITION_IDS = {v: k for k, v in POSITIONS.items()}


# Core columns that should be present in player data
PLAYER_CORE_COLUMNS = [
    'id',
    'web_name',
    'first_name',
    'second_name',
    'team',
    'element_type',
    'now_cost',
    'selected_by_percent',
    'form',
    'points_per_game',
    'total_points',
]


# Gameweek performance columns
GW_PERFORMANCE_COLUMNS = [
    'element',  # player_id
    'fixture',
    'opponent_team',
    'total_points',
    'was_home',
    'kickoff_time',
    'team_h_score',
    'team_a_score',
    'round',  # gameweek number
    # Performance metrics
    'minutes',
    'goals_scored',
    'assists',
    'clean_sheets',
    'goals_conceded',
    'own_goals',
    'penalties_saved',
    'penalties_missed',
    'yellow_cards',
    'red_cards',
    'saves',
    'bonus',
    'bps',  # bonus points system
    'influence',
    'creativity',
    'threat',
    'ict_index',
    'value',
    'transfers_balance',
    'selected',
    'transfers_in',
    'transfers_out',
]


# Statistical columns for analysis
STATISTICAL_COLUMNS = [
    'expected_goals',
    'expected_assists',
    'expected_goal_involvements',
    'expected_goals_conceded',
]


# Team columns
TEAM_COLUMNS = [
    'id',
    'name',
    'short_name',
    'strength',
    'strength_overall_home',
    'strength_overall_away',
    'strength_attack_home',
    'strength_attack_away',
    'strength_defence_home',
    'strength_defence_away',
]


# Fixture columns
FIXTURE_COLUMNS = [
    'id',
    'event',  # gameweek
    'team_h',  # home team id
    'team_a',  # away team id
    'team_h_difficulty',
    'team_a_difficulty',
    'kickoff_time',
    'finished',
    'started',
    'team_h_score',
    'team_a_score',
]


def get_dtype_mapping() -> Dict[str, str]:
    """Get pandas dtype mapping for common columns

    Returns:
        dict: Column name to dtype mapping
    """
    return {
        # Identifiers
        'id': 'int64',
        'element': 'int64',
        'team': 'int64',
        'element_type': 'int64',
        'fixture': 'int64',
        'round': 'int64',
        'event': 'int64',

        # Numeric performance
        'minutes': 'int64',
        'goals_scored': 'int64',
        'assists': 'int64',
        'clean_sheets': 'int64',
        'goals_conceded': 'int64',
        'own_goals': 'int64',
        'penalties_saved': 'int64',
        'penalties_missed': 'int64',
        'yellow_cards': 'int64',
        'red_cards': 'int64',
        'saves': 'int64',
        'bonus': 'int64',
        'bps': 'int64',
        'total_points': 'int64',

        # Floats
        'now_cost': 'float64',
        'selected_by_percent': 'float64',
        'form': 'float64',
        'points_per_game': 'float64',
        'influence': 'float64',
        'creativity': 'float64',
        'threat': 'float64',
        'ict_index': 'float64',
        'value': 'float64',
        'expected_goals': 'float64',
        'expected_assists': 'float64',
        'expected_goal_involvements': 'float64',
        'expected_goals_conceded': 'float64',

        # Booleans
        'was_home': 'bool',
        'finished': 'bool',
        'started': 'bool',

        # Strings
        'web_name': 'str',
        'first_name': 'str',
        'second_name': 'str',
        'name': 'str',
        'short_name': 'str',
    }


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names across different data sources

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: DataFrame with standardized columns
    """
    # Common column name mappings
    column_mapping = {
        'GW': 'round',
        'gameweek': 'round',
        'player_id': 'element',
        'position': 'element_type',
        'team_id': 'team',
        'match_id': 'fixture',
    }

    # Rename columns that exist in the mapping
    existing_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
    if existing_mappings:
        df = df.rename(columns=existing_mappings)

    return df


def add_position_labels(df: pd.DataFrame, position_col='element_type') -> pd.DataFrame:
    """Add position labels to dataframe

    Args:
        df (pd.DataFrame): Input dataframe with position IDs
        position_col (str): Name of the position column

    Returns:
        pd.DataFrame: DataFrame with position_label column added
    """
    if position_col in df.columns:
        df['position_label'] = df[position_col].map(POSITIONS)
    return df


def convert_cost_to_millions(df: pd.DataFrame, cost_col='now_cost') -> pd.DataFrame:
    """Convert cost from FPL format (e.g., 115) to millions (e.g., 11.5)

    Args:
        df (pd.DataFrame): Input dataframe
        cost_col (str): Name of the cost column

    Returns:
        pd.DataFrame: DataFrame with cost converted
    """
    if cost_col in df.columns:
        df[f'{cost_col}_millions'] = df[cost_col] / 10.0
    return df


def validate_gameweek_data(df: pd.DataFrame) -> tuple:
    """Validate gameweek data structure

    Args:
        df (pd.DataFrame): Gameweek dataframe to validate

    Returns:
        tuple: (is_valid, list of issues)
    """
    issues = []

    # Check required columns
    required = ['element', 'round', 'total_points', 'minutes']
    missing = [col for col in required if col not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    # Check for nulls in critical columns
    if 'element' in df.columns and df['element'].isnull().any():
        issues.append("Null values found in 'element' column")

    if 'round' in df.columns and df['round'].isnull().any():
        issues.append("Null values found in 'round' column")

    # Check data types
    if 'total_points' in df.columns and not pd.api.types.is_numeric_dtype(df['total_points']):
        issues.append("'total_points' should be numeric")

    # Check for duplicate player-gameweek combinations
    if 'element' in df.columns and 'round' in df.columns:
        duplicates = df.duplicated(subset=['element', 'round'], keep=False)
        if duplicates.any():
            issues.append(f"Found {duplicates.sum()} duplicate player-gameweek combinations")

    is_valid = len(issues) == 0
    return is_valid, issues


def get_feature_groups() -> Dict[str, List[str]]:
    """Get grouped features for different modeling purposes

    Returns:
        dict: Feature groups
    """
    return {
        'attacking': [
            'goals_scored',
            'assists',
            'expected_goals',
            'expected_assists',
            'shots',
            'shots_on_target',
            'big_chances_created',
            'creativity',
            'threat',
        ],
        'defensive': [
            'clean_sheets',
            'goals_conceded',
            'expected_goals_conceded',
            'saves',
            'penalties_saved',
            'tackles',
            'clearances_blocks_interceptions',
        ],
        'disciplinary': [
            'yellow_cards',
            'red_cards',
            'fouls',
            'penalties_missed',
        ],
        'value': [
            'now_cost',
            'cost_change_start',
            'value',
            'points_per_game',
            'selected_by_percent',
            'transfers_in',
            'transfers_out',
        ],
        'performance': [
            'minutes',
            'total_points',
            'bonus',
            'bps',
            'influence',
            'ict_index',
            'form',
        ],
        'metadata': [
            'element',
            'team',
            'element_type',
            'opponent_team',
            'was_home',
            'round',
            'fixture',
        ]
    }


class DataSchema:
    """Data schema handler for FPL data"""

    @staticmethod
    def prepare_player_data(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and standardize player data

        Args:
            df (pd.DataFrame): Raw player data

        Returns:
            pd.DataFrame: Standardized player data
        """
        df = standardize_column_names(df)
        df = add_position_labels(df)
        df = convert_cost_to_millions(df)

        # Apply dtype mapping where applicable
        dtype_map = get_dtype_mapping()
        for col, dtype in dtype_map.items():
            if col in df.columns:
                try:
                    if dtype == 'int64':
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                    elif dtype == 'float64':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif dtype == 'bool':
                        df[col] = df[col].astype('bool')
                except Exception:
                    pass  # Keep original dtype if conversion fails

        return df

    @staticmethod
    def prepare_gameweek_data(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and standardize gameweek data

        Args:
            df (pd.DataFrame): Raw gameweek data

        Returns:
            pd.DataFrame: Standardized gameweek data
        """
        df = standardize_column_names(df)

        # Ensure element and round are present
        if 'element' not in df.columns and 'id' in df.columns:
            df['element'] = df['id']

        # Apply dtype mapping
        dtype_map = get_dtype_mapping()
        for col, dtype in dtype_map.items():
            if col in df.columns:
                try:
                    if dtype == 'int64':
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                    elif dtype == 'float64':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif dtype == 'bool':
                        df[col] = df[col].astype('bool')
                except Exception:
                    pass

        # Validate
        is_valid, issues = validate_gameweek_data(df)
        if not is_valid:
            print(f"Warning: Gameweek data validation issues: {issues}")

        return df

    @staticmethod
    def prepare_team_data(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and standardize team data

        Args:
            df (pd.DataFrame): Raw team data

        Returns:
            pd.DataFrame: Standardized team data
        """
        df = standardize_column_names(df)
        return df

    @staticmethod
    def prepare_fixture_data(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and standardize fixture data

        Args:
            df (pd.DataFrame): Raw fixture data

        Returns:
            pd.DataFrame: Standardized fixture data
        """
        df = standardize_column_names(df)

        # Convert kickoff_time to datetime
        if 'kickoff_time' in df.columns:
            df['kickoff_time'] = pd.to_datetime(df['kickoff_time'], errors='coerce')

        return df
