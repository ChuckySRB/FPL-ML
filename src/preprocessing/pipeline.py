"""
Preprocessing pipeline for FPL ML models
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE, MODELS_DIR


class FPLPreprocessor:
    """Preprocessing pipeline for FPL data"""

    def __init__(self, scaler_type: str = 'standard'):
        """Initialize preprocessor

        Args:
            scaler_type (str): Type of scaler ('standard', 'robust', 'minmax')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.categorical_mappings = {}

        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

    def handle_missing_values(self, df: pd.DataFrame,
                             strategy: str = 'median',
                             fill_value: float = 0) -> pd.DataFrame:
        """Handle missing values

        Args:
            df (pd.DataFrame): Input data
            strategy (str): Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
            fill_value (float): Fill value for 'constant' strategy

        Returns:
            pd.DataFrame: Data with imputed values
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if strategy == 'constant':
            self.imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        else:
            self.imputer = SimpleImputer(strategy=strategy)

        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])

        return df

    def encode_categorical(self, df: pd.DataFrame,
                          categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Encode categorical variables

        Args:
            df (pd.DataFrame): Input data
            categorical_cols (list): List of categorical columns

        Returns:
            pd.DataFrame: Data with encoded categories
        """
        df = df.copy()

        if categorical_cols is None:
            # Auto-detect categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in categorical_cols:
            if col not in df.columns:
                continue

            # Label encoding for ordinal or low-cardinality features
            if df[col].nunique() < 50:
                unique_vals = df[col].unique()
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                self.categorical_mappings[col] = mapping
                df[col] = df[col].map(mapping)
            else:
                # One-hot encoding for high-cardinality features
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        return df

    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features

        Args:
            X (pd.DataFrame): Features to scale
            fit (bool): Whether to fit the scaler (True for train, False for test)

        Returns:
            pd.DataFrame: Scaled features
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = X.columns.tolist()
        else:
            X_scaled = self.scaler.transform(X)

        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def select_features(self, df: pd.DataFrame,
                       exclude_cols: Optional[List[str]] = None,
                       target_col: str = 'total_points') -> Tuple[pd.DataFrame, pd.Series]:
        """Select features and target

        Args:
            df (pd.DataFrame): Input data
            exclude_cols (list): Columns to exclude from features
            target_col (str): Target column name

        Returns:
            tuple: (X, y) features and target
        """
        if exclude_cols is None:
            # Default columns to exclude
            exclude_cols = [
                'element', 'round', 'fixture', 'kickoff_time',
                'opponent_team', 'was_home', 'season',
                'web_name', 'first_name', 'second_name', 'name'
            ]

        # Add target to exclusion list
        exclude_cols = list(set(exclude_cols + [target_col]))

        # Select feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[target_col] if target_col in df.columns else None

        return X, y

    def create_train_test_split(self, X: pd.DataFrame, y: pd.Series,
                                test_size: float = TEST_SIZE,
                                random_state: int = RANDOM_STATE,
                                stratify_col: Optional[pd.Series] = None) -> Tuple:
        """Create train/test split

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Test set size
            random_state (int): Random seed
            stratify_col (pd.Series): Column for stratification

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col,
            shuffle=True
        )

    def create_time_series_splits(self, df: pd.DataFrame,
                                  n_splits: int = 5) -> TimeSeriesSplit:
        """Create time series cross-validation splits

        Args:
            df (pd.DataFrame): Data sorted by time
            n_splits (int): Number of splits

        Returns:
            TimeSeriesSplit: Time series splitter
        """
        return TimeSeriesSplit(n_splits=n_splits)

    def create_temporal_split(self, df: pd.DataFrame,
                             date_col: str = 'round',
                             test_rounds: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data temporally (use recent gameweeks for testing)

        Args:
            df (pd.DataFrame): Data with temporal column
            date_col (str): Column indicating time (gameweek)
            test_rounds (int): Number of recent rounds for test set

        Returns:
            tuple: (train_df, test_df)
        """
        if date_col not in df.columns:
            raise ValueError(f"Column {date_col} not found in dataframe")

        # Get unique sorted rounds
        rounds = sorted(df[date_col].unique())

        # Split point
        split_round = rounds[-test_rounds] if len(rounds) > test_rounds else rounds[0]

        train_df = df[df[date_col] < split_round].copy()
        test_df = df[df[date_col] >= split_round].copy()

        return train_df, test_df

    def prepare_for_training(self, df: pd.DataFrame,
                            target_col: str = 'total_points',
                            temporal_split: bool = True,
                            test_rounds: int = 5,
                            scale: bool = True,
                            handle_missing: bool = True) -> Dict:
        """Complete preprocessing pipeline

        Args:
            df (pd.DataFrame): Feature-engineered data
            target_col (str): Target column
            temporal_split (bool): Use temporal split instead of random
            test_rounds (int): Rounds for test set (if temporal_split=True)
            scale (bool): Whether to scale features
            handle_missing (bool): Whether to impute missing values

        Returns:
            dict: Dictionary with train/test data and metadata
        """
        print("Preprocessing pipeline:")
        df = df.copy()

        # Handle missing values
        if handle_missing:
            print("  → Handling missing values...")
            df = self.handle_missing_values(df)

        # Encode categorical variables
        print("  → Encoding categorical variables...")
        df = self.encode_categorical(df)

        # Create train/test split
        if temporal_split and 'round' in df.columns:
            print(f"  → Creating temporal split ({test_rounds} test rounds)...")
            train_df, test_df = self.create_temporal_split(df, test_rounds=test_rounds)
        else:
            print("  → Creating random train/test split...")
            train_df, test_df = train_test_split(
                df, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

        # Select features
        print("  → Selecting features...")
        X_train, y_train = self.select_features(train_df, target_col=target_col)
        X_test, y_test = self.select_features(test_df, target_col=target_col)

        # Scale features
        if scale:
            print(f"  → Scaling features ({self.scaler_type})...")
            X_train = self.scale_features(X_train, fit=True)
            X_test = self.scale_features(X_test, fit=False)

        print("✓ Preprocessing complete!")
        print(f"  Train size: {len(X_train):,}")
        print(f"  Test size: {len(X_test):,}")
        print(f"  Features: {len(X_train.columns)}")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'train_df': train_df,
            'test_df': test_df,
            'feature_names': X_train.columns.tolist(),
            'scaler': self.scaler,
            'imputer': self.imputer,
        }

    def save_pipeline(self, filepath: str):
        """Save preprocessing pipeline

        Args:
            filepath (str): Path to save pipeline
        """
        pipeline_data = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'categorical_mappings': self.categorical_mappings,
            'scaler_type': self.scaler_type
        }

        joblib.dump(pipeline_data, filepath)
        print(f"Pipeline saved to {filepath}")

    def load_pipeline(self, filepath: str):
        """Load preprocessing pipeline

        Args:
            filepath (str): Path to pipeline file
        """
        pipeline_data = joblib.load(filepath)

        self.scaler = pipeline_data['scaler']
        self.imputer = pipeline_data['imputer']
        self.feature_names = pipeline_data['feature_names']
        self.categorical_mappings = pipeline_data['categorical_mappings']
        self.scaler_type = pipeline_data['scaler_type']

        print(f"Pipeline loaded from {filepath}")


def create_position_specific_data(df: pd.DataFrame,
                                  position: str) -> pd.DataFrame:
    """Filter data for position-specific modeling

    Args:
        df (pd.DataFrame): Full dataset with position_label
        position (str): Position to filter ('GK', 'DEF', 'MID', 'FWD')

    Returns:
        pd.DataFrame: Position-filtered data
    """
    if 'position_label' not in df.columns:
        raise ValueError("position_label column not found")

    return df[df['position_label'] == position].copy()


def get_feature_columns_by_type(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Categorize features by type for analysis

    Args:
        df (pd.DataFrame): Feature dataframe

    Returns:
        dict: Feature categories
    """
    features = {
        'rolling': [col for col in df.columns if 'rolling' in col],
        'lag': [col for col in df.columns if 'lag' in col],
        'form': [col for col in df.columns if 'form' in col or 'consistency' in col],
        'opponent': [col for col in df.columns if 'opponent' in col],
        'home_away': [col for col in df.columns if 'home' in col or 'away' in col],
        'value': [col for col in df.columns if 'cost' in col or 'value' in col or 'transfer' in col],
        'attacking': [col for col in df.columns if any(x in col for x in ['goal', 'assist', 'xg', 'xa', 'creativity', 'threat'])],
        'defensive': [col for col in df.columns if any(x in col for x in ['clean_sheet', 'save', 'conceded', 'xgc'])],
        'base': ['minutes', 'total_points', 'bonus', 'bps', 'influence', 'ict_index']
    }

    return {k: [col for col in v if col in df.columns] for k, v in features.items()}
